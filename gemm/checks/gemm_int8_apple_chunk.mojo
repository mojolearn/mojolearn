# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`mojolearn.identical.gemm.int8i32.v1` on Apple's FLOAT matrix unit, in
EXACT CHUNKS. A PROBE, not a plan of the profile.

Lane lane/lowbit-units, 2026-09-29. The
flat realization is `gemm/checks/gemm_lowbit.mojo`; the answer is
`gemm/host/gemm_lowbit_oracle.mojo::gemm_int8_oracle`; the gate is
`gemm/checks/gemm_int8_apple_chunk_check.mojo`.

WHAT IT IS. Contract clause L-9 keeps Apple on the flat int8 kernel because
Metal's simdgroup matrix takes half and float operands only. This file holds
each int8 code as the fp32 value it names (an integer in [-127, 127], exact)
and runs the product on the fp32 unit `PLAN_APPLE_MMA` already uses
(`air.simdgroup_matrix_8x8_multiply_accumulate`), with `k` cut into chunks
short enough that no float in the unit can round.

WHY NO FLOAT CAN ROUND (the construction argument)
--------------------------------------------------
A product of two codes is an integer of magnitude at most 127 * 127 = 16129.
Inside one chunk of `S` steps every value the unit can hold, whatever order
it adds in and whether or not it fuses the multiply into the add, is a sum of
at most `S` such products (the accumulator enters the chunk at zero), so its
magnitude is at most `16129 * S`. With `S <= INT8_APPLE_CHUNK_MAX_STEPS =
1040`, `16129 * 1040 = 16774160 < 2^24 = 16777216`, and every integer below
2^24 in magnitude is a float32. So every product, every partial sum and the
chunk sum are exact; the flush to zero has nothing to act on (a nonzero
integer is never subnormal); and the unit's internal order is SCHEDULING, as
the integer units' is under L-9. At a chunk end each accumulator converts to
Int32 (exact: it is an integer below 2^24) and is added to an Int32 running
sum, which L-7 bounds (`k <= INT8_MAX_K`), and the accumulator restarts at
zero. The Int32 that reaches the epilogue is the Int32 `int8_dot_cell`
computes, and the epilogue is `dequant_int8_pinned`, character for character
the flat kernel's.

THE PADDING RULE is L-9's: a row beyond `m`, a column beyond `n` and a step
beyond `k` are staged as the ZERO CODE, which contributes exactly nothing.
Rows and columns beyond `m`, `n` are masked at the store.

WHAT THE ARGUMENT DOES NOT COVER. It assumes the unit computes in IEEE
float32 with a significand of 24 bits at every internal step. That is the
reading `gemm/checks/apple_simdgroup_probe.mojo` measured on the M4 and the
M2 Pro steward's fixed matrix plan agrees with; it is a measurement per Apple
generation, not a documented property, which is why the gate plants the worst
cases (every product +16129, every product -16129, and halves that cancel)
and why the probe is owed a run on each Apple generation before anything
ships on it.

NOT IN THE DISPATCHER. `identical_gemm_int8_into` does not call this file,
`lib_int8_matrix_unit_for` still answers False on Apple, and nothing a caller
gets by default moves. The timing harness and the gate call
`identical_gemm_int8_apple_chunk_into` by name.

THE SABOTAGE ARMS.
  `-D MOJOLEARN_INT8_APPLE_CHUNK_SABOTAGE=1` removes the chunk boundary: one
      float accumulator runs the whole of `k`. The planted worst cases with
      `k > 1040` then pass 2^24, where odd integers are not floats, and the
      gate must FAIL on them. It is the arm that shows the chunk is what
      keeps the sum exact; random codes cannot show it (their sums stay far
      below 2^24).
  `-D MOJOLEARN_LOWBIT_SABOTAGE=1` flips the value of every cell stored, as
      in the flat kernel and the integer-unit kernel (DEVIATION 2908).
"""

from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from std.sys import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN, column_name
from checks.numerics import dequant_int8_pinned
from gemm.checks.gemm_identical import _AMMA_M64, _amma_load_t, _amma_mma
from gemm.host.gemm_lowbit_oracle import INT8_MAX_K
from gemm.host.gemm_oracle import gemm_oracle_sabotage_value_flip

#: The largest magnitude of a product of two int8 codes (contract L-4 clamps
#: a code to [-127, 127]).
comptime INT8_PRODUCT_MAX = 127 * 127

#: Every integer of magnitude below this is a float32.
comptime F32_EXACT_INTEGER_BOUND = 16777216

#: The longest chunk whose every partial sum is a float32:
#: `16129 * 1040 = 16774160 < 2^24`, and 1041 steps would reach 16790289.
comptime INT8_APPLE_CHUNK_MAX_STEPS = (F32_EXACT_INTEGER_BOUND - 1) // INT8_PRODUCT_MAX

#: The window: steps staged into threadgroup memory at once. SCHEDULING.
comptime INT8_APPLE_CHUNK_KB = 32

#: Whole windows per chunk, so a chunk boundary is a window boundary:
#: 32 windows of 32 steps, 1024 steps, under the 1040 bound.
comptime INT8_APPLE_CHUNK_WINDOWS = INT8_APPLE_CHUNK_MAX_STEPS // INT8_APPLE_CHUNK_KB

#: The arm that removes the chunk boundary. Off in every build that does not
#: name it.
comptime INT8_APPLE_CHUNK_SABOTAGE = is_defined["MOJOLEARN_INT8_APPLE_CHUNK_SABOTAGE"]()

#: DEVIATION 2908, the value arm, read from the define the other int8 plans
#: read.
comptime INT8_APPLE_CHUNK_VALUE_SABOTAGE = is_defined["MOJOLEARN_LOWBIT_SABOTAGE"]()

#: Whether this build can run the probe at all: the Metal column only.
comptime INT8_APPLE_CHUNK_AVAILABLE = TARGET_COLUMN == COLUMN_APPLE

#: The two tile geometries (simdgroups M x N, 8x8 fragments M x N per
#: simdgroup). WIDE owns a 64 x 64 output tile; ROW owns 8 x 128, for the
#: decode rows, where a 64-row tile would multiply 63 rows of zero codes for
#: one row of output. SCHEDULING: the gate runs both on every shape.
comptime INT8_APPLE_CHUNK_GEOMETRY_WIDE = 0
comptime INT8_APPLE_CHUNK_GEOMETRY_ROW = 1

#: Outputs of at most this many rows take the ROW geometry.
comptime INT8_APPLE_CHUNK_ROW_MAX_M = 8


def int8_apple_chunk_sabotage_name() -> String:
    comptime if INT8_APPLE_CHUNK_SABOTAGE:
        return String("CHUNK_BOUNDARY_REMOVED")
    elif INT8_APPLE_CHUNK_VALUE_SABOTAGE:
        return String("LOWBIT_VALUE_FLIP")
    else:
        return String("none")


def int8_apple_chunk_geometry(m: Int) -> Int:
    """The geometry the launcher picks. Reads `m` and may: the tile is a
    schedule and every cell's Int32 is the same on either."""
    if m <= INT8_APPLE_CHUNK_ROW_MAX_M:
        return INT8_APPLE_CHUNK_GEOMETRY_ROW
    return INT8_APPLE_CHUNK_GEOMETRY_WIDE


def int8_apple_chunk_geometry_name(geometry: Int) -> String:
    if geometry == INT8_APPLE_CHUNK_GEOMETRY_ROW:
        return String("ROW 8x128 (1x4 simdgroups, 1x4 fragments)")
    return String("WIDE 64x64 (2x2 simdgroups, 4x4 fragments)")


@always_inline
def _stage_codes[
    ROWS: Int, KB: Int, NT: Int, PMAJOR: Bool, ST: Int
](
    dst: UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    q: MutPointer[Int8, MutAnyOrigin],
    row0: Int,
    rows: Int,
    k0: Int,
    k: Int,
    tid: Int,
):
    """One window of one operand into threadgroup memory, each code as the
    float32 it names. A slot is four consecutive steps of one row; a thread
    owns slots `tid, tid + NT, ...` below the window's `ROWS * KB / 4` (the
    ROW geometry's left operand has fewer slots than the block has threads,
    and the threads past them stage nothing). `PMAJOR`: element (row r, step p) at
    `dst[p * ST + r]` (the left operand); else at `dst[r * ST + p]` (the
    right). A row at or beyond `rows` and a step at or beyond `k` are the
    ZERO CODE. The vector load is taken only when it is aligned and entirely
    inside the row (`gemm_int8_mma.mojo::_pack4`'s discipline)."""
    comptime SLOTS = (ROWS * KB) // 4
    comptime SL = (SLOTS + NT - 1) // NT
    comptime assert KB % 4 == 0, "_stage_codes: a window is whole slots"
    comptime for sl in range(SL):
        var s = sl * NT + tid
        if s < SLOTS:
            var r = s // (KB // 4)
            var p4 = (s % (KB // 4)) * 4
            var v = SIMD[DType.float32, 4](0.0)
            var gr = row0 + r
            if gr < rows:
                var base = gr * k + k0 + p4
                if k0 + p4 + 4 <= k and (k & 3) == 0:
                    v = q.unsafe_load[width=4](base).cast[DType.float32]()
                else:
                    comptime for e in range(4):
                        if k0 + p4 + e < k:
                            v[e] = q.unsafe_load(base + e).cast[DType.float32]()
            comptime if PMAJOR:
                comptime for e in range(4):
                    dst[(p4 + e) * ST + r] = v[e]
            else:
                (dst + r * ST + p4).store[alignment=16](v)


def identical_gemm_int8_apple_chunk_kernel[
    SGM: Int, SGN: Int, FM: Int, FN: Int, KB: Int
](
    c: MutPointer[Float32, MutAnyOrigin],
    qa: MutPointer[Int8, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    qb: MutPointer[Int8, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
):
    """OP_NT, `C[m x n] = Qa[m x k] . Qb[n x k]^T`. One block owns a
    `BM x BN` output tile; each simdgroup owns `FM x FN` 8x8 fragments, each
    lane two cells of each (`identical_gemm_apple_mma_kernel`'s layout).

    Per window of `KB` steps the block stages the codes of both operands as
    float32 (`_stage_codes`), then every simdgroup multiplies on the matrix
    unit. Every `INT8_APPLE_CHUNK_WINDOWS` windows, and at the end of `k`,
    each lane converts its accumulators to Int32, adds them to its running
    sums and restarts the accumulators at zero. The epilogue is the flat
    kernel's."""
    comptime NSG = SGM * SGN
    comptime NT = NSG * 32
    comptime BM = 8 * FM * SGM
    comptime BN = 8 * FN * SGN
    comptime AST = BM + 4
    comptime BST = KB + 4
    comptime NF = FM * FN
    comptime NC = 2 * NF
    comptime assert KB % 8 == 0, "the window is whole 8-step fragments"
    comptime assert (
        INT8_PRODUCT_MAX * INT8_APPLE_CHUNK_WINDOWS * KB < F32_EXACT_INTEGER_BOUND
    ), "a chunk's largest partial sum must be a float32"
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var tid = Int(thread_idx.x)
    var sg = tid // 32
    var lane = tid % 32
    var sgm = sg // SGN
    var sgn = sg % SGN
    var nbn = (n + BN - 1) // BN
    var bid = Int(block_idx.x)
    var m0 = (bid // nbn) * BM
    var n0 = (bid % nbn) * BN
    var qd = lane // 4
    var frow = (qd & 4) + ((lane // 2) % 4)
    var fcol = (qd & 2) * 2 + (lane % 2) * 2
    var at = stack_allocation[KB * AST, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var bt = stack_allocation[BN * BST, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var acc = InlineArray[_AMMA_M64, NF](fill=_AMMA_M64(0))
    var total = SIMD[DType.int32, NC](0)
    var windows = (k + KB - 1) // KB
    for w in range(windows):
        var k0 = w * KB
        _stage_codes[BM, KB, NT, True, AST](at, qa, m0, m, k0, k, tid)
        _stage_codes[BN, KB, NT, False, BST](bt, qb, n0, n, k0, k, tid)
        barrier()
        comptime for p8 in range(KB // 8):
            var af = InlineArray[_AMMA_M64, FM](fill=_AMMA_M64(0))
            var bf = InlineArray[_AMMA_M64, FN](fill=_AMMA_M64(0))
            comptime for fm in range(FM):
                af[fm] = _amma_load_t(at + (8 * p8) * AST + (sgm * FM + fm) * 8, AST)
            comptime for fq in range(FN):
                bf[fq] = _amma_load_t(bt + ((sgn * FN + fq) * 8) * BST + 8 * p8, BST)
            comptime for fm in range(FM):
                comptime for fq in range(FN):
                    acc[fm * FN + fq] = _amma_mma(af[fm], bf[fq], acc[fm * FN + fq])
        barrier()
        var chunk_end = w + 1 == windows
        comptime if not INT8_APPLE_CHUNK_SABOTAGE:
            chunk_end = chunk_end or (w + 1) % INT8_APPLE_CHUNK_WINDOWS == 0
        if chunk_end:
            comptime for f in range(NF):
                comptime for e in range(2):
                    total[2 * f + e] += acc[f][e].cast[DType.int32]()
                acc[f] = _AMMA_M64(0)
    comptime for fm in range(FM):
        comptime for fq in range(FN):
            comptime for e in range(2):
                var gi = m0 + (sgm * FM + fm) * 8 + frow
                var gj = n0 + (sgn * FN + fq) * 8 + fcol + e
                if gi < m and gj < n:
                    var out = dequant_int8_pinned(
                        total[2 * (fm * FN + fq) + e],
                        Int(ea.unsafe_load(gi)) + Int(eb.unsafe_load(gj)),
                    )
                    comptime if INT8_APPLE_CHUNK_VALUE_SABOTAGE:
                        out = gemm_oracle_sabotage_value_flip(out)
                    c.unsafe_store(gi * n + gj, out)


def identical_gemm_int8_apple_chunk_with_geometry(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut qa: DeviceBuffer[DType.int8],
    mut ea: DeviceBuffer[DType.int32],
    mut qb: DeviceBuffer[DType.int8],
    mut eb: DeviceBuffer[DType.int32],
    m: Int,
    n: Int,
    k: Int,
    geometry: Int,
) raises:
    """The probe on a NAMED geometry. The gate calls this to compare the two;
    the harness calls `identical_gemm_int8_apple_chunk_into`. Refuses by name
    on a column that is not Apple. Asynchronous."""
    comptime if not INT8_APPLE_CHUNK_AVAILABLE:
        raise Error(
            "identical_gemm_int8_apple_chunk: column " + column_name(TARGET_COLUMN)
            + " is not Apple; the probe runs on Metal's float matrix unit only"
        )
    else:
        if m <= 0 or n <= 0 or k <= 0 or k > INT8_MAX_K:
            raise Error(
                "identical_gemm_int8_apple_chunk: m, n and k must be positive and"
                " k at most " + String(INT8_MAX_K) + " (contract L-7), got m="
                + String(m) + " n=" + String(n) + " k=" + String(k)
            )
        if geometry == INT8_APPLE_CHUNK_GEOMETRY_ROW:
            comptime kern_row = identical_gemm_int8_apple_chunk_kernel[
                1, 4, 1, 4, INT8_APPLE_CHUNK_KB
            ]
            ctx.enqueue_function[kern_row](
                c.unsafe_ptr(),
                qa.unsafe_ptr(),
                ea.unsafe_ptr(),
                qb.unsafe_ptr(),
                eb.unsafe_ptr(),
                Int32(m),
                Int32(n),
                Int32(k),
                grid_dim=(((m + 7) // 8) * ((n + 127) // 128), 1, 1),
                block_dim=(128, 1, 1),
            )
            return
        if geometry != INT8_APPLE_CHUNK_GEOMETRY_WIDE:
            raise Error(
                "identical_gemm_int8_apple_chunk: geometry must be 0 (WIDE) or 1"
                " (ROW), got " + String(geometry)
            )
        comptime kern_wide = identical_gemm_int8_apple_chunk_kernel[
            2, 2, 4, 4, INT8_APPLE_CHUNK_KB
        ]
        ctx.enqueue_function[kern_wide](
            c.unsafe_ptr(),
            qa.unsafe_ptr(),
            ea.unsafe_ptr(),
            qb.unsafe_ptr(),
            eb.unsafe_ptr(),
            Int32(m),
            Int32(n),
            Int32(k),
            grid_dim=(((m + 63) // 64) * ((n + 63) // 64), 1, 1),
            block_dim=(128, 1, 1),
        )


def identical_gemm_int8_apple_chunk_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut qa: DeviceBuffer[DType.int8],
    mut ea: DeviceBuffer[DType.int32],
    mut qb: DeviceBuffer[DType.int8],
    mut eb: DeviceBuffer[DType.int32],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """The probe on the geometry `int8_apple_chunk_geometry` picks.
    Asynchronous."""
    identical_gemm_int8_apple_chunk_with_geometry(
        ctx, c, qa, ea, qb, eb, m, n, k, int8_apple_chunk_geometry(m)
    )
