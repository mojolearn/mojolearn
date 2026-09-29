# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`mojolearn.identical.gemm.int8i32.v1` on the integer matrix units, TUNED:
the schedule of `gemm/checks/gemm_int8_mma.mojo` changed and its bits kept.

Lane lane/lowbit-mma-speed, 2026-09-29. Contract clause L-9 of
`gemm/IDENTICAL_LOWBIT_CONTRACT.md`. The reference unit plan is
`gemm/checks/gemm_int8_mma.mojo` and is not edited; the flat plan and the
dispatcher are `gemm/checks/gemm_lowbit.mojo`; the answer is
`gemm/host/gemm_lowbit_oracle.mojo::gemm_int8_oracle`; the gate is
`gemm/checks/gemm_int8_mma_tuned_check.mojo`. NOTHING HERE IS DISPATCHED:
`identical_gemm_int8_into` still takes the reference plan.

WHY IT EXISTS. lane/lowbit-units timed the reference unit plan on an H100:
1.38 to 1.66 times fp32.v1's time at the 512-token rows (qkv.t512: 6.0
T MAC/s against 8.3). In that plan one warp owns one 16 x 16 output tile
and reads every fragment of both operands from device memory at every k
step: eight four-byte loads for two unit steps.

WHY NO PLAN HERE CAN MOVE A BIT. Clause L-9's argument, unchanged: a product
of two int8 codes is an exact integer, an Int32 sum of them cannot overflow
for `k <= INT8_MAX_K`, and a sum of exact integers is the same integer under
every order, grouping and tile. What this file changes is which thread holds
which cell, where a code waits before the unit reads it (device memory or
threadgroup memory), how many bytes one load carries, and how many unit
steps run between two barriers. None of it reaches the Int32 a cell ends
with, and the only floating steps stay `dequant_int8_pinned` (L-5, L-6),
called with the reference's arguments.

THE PADDING RULE is the reference's: a step at or beyond `k`, and a row at
or beyond `m` or `n`, are the ZERO CODE, and cells beyond `m`, `n` are
masked at the store. A staged window is written WHOLE, zero codes included,
because threadgroup memory still holds the window before it.

THE LEVERS, each a comptime parameter, so every plan below is one
instantiation of one kernel and the timing harness runs them as arms of one
binary:

    FM, FN   16 x 16 unit tiles per warp, along m and along n: a warp owns
             a `16 FM x 16 FN` output tile. On NVIDIA a unit tile is two
             m16n8k32 steps that share their A fragments; on AMD it is one
             16x16x32 MFMA.
    WM, WN   warps per block: a block owns `BM x BN = 16 FM WM x 16 FN WN`
             and stages `BM + BN` operand rows per window.
    KB       k steps staged per window, a multiple of the unit's 32: `KB /
             32` unit steps between two barriers, unrolled.
    LW       bytes per staging load and store, 4 or 16.

THE STAGED ROW is `KB + 16` bytes. The sixteen bytes of padding are never
read; they move each row's bank. On NVIDIA one fragment load is eight rows
of sixteen bytes read by one warp, and with `KB` a multiple of 32 the eight
rows' first words then fall in eight different groups of four banks.

THE DIRECT KERNEL (`identical_gemm_int8_mma_direct_kernel`) is the
reference's schedule, one warp one tile, nothing staged, with ONE thing
changed per instantiation. It exists to test what is NVIDIA-specific in the
reference before anything is redesigned around a guess (the brief,
2026-09-29: the same schedule takes 0.28 of fp32.v1's time on the AMD unit
and 1.39 on the NVIDIA one). Its PROBE instantiations compute a WRONG
product on purpose, to time a part of the kernel alone; they are named
`probe`, no digest of theirs is compared with anything, and no dispatcher
may ever name one.

THE SABOTAGE ARMS.
  `-D MOJOLEARN_INT8_TUNED_SABOTAGE=1` breaks the padding rule of the
      staging: a load that lies wholly beyond `k` or beyond the operand's
      rows stores NOTHING, so the ragged tail of the last window reads the
      codes the window before it left. A SCHEDULING defect, the kind this
      file can have. It reaches a shape whose last window holds a load
      wholly beyond `k` where an earlier window wrote, and it cannot reach
      a shape whose `k` is a whole number of windows, which must still pass.
  `-D MOJOLEARN_LOWBIT_SABOTAGE=1` flips the value of every cell stored, as
      in the flat kernel and the reference unit kernel (DEVIATION 2908).
"""

from std.gpu import WARP_SIZE, block_idx, lane_id, thread_idx
from std.memory import bitcast, stack_allocation
from std.sys import is_defined, llvm_intrinsic
from std.sys.info import is_amd_gpu, is_nvidia_gpu
from std.sys.intrinsics import _RegisterPackType
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.kernel_matrix import (
    COLUMN_NVIDIA,
    TARGET_COLUMN,
    column_name,
    column_shared_limit,
    lib_int8_matrix_unit_for,
)
from checks.numerics import dequant_int8_pinned
from gemm.checks.gemm_int8_mma import _imma_m16n8k32, _pack4, int8_mma_admits
from gemm.host.gemm_lowbit_oracle import INT8_MAX_K
from gemm.host.gemm_oracle import gemm_oracle_sabotage_value_flip

#: The scheduling arm: a staging load wholly outside the operand stores
#: nothing. Off in every build that does not name it.
comptime INT8_TUNED_SABOTAGE = is_defined["MOJOLEARN_INT8_TUNED_SABOTAGE"]()

#: DEVIATION 2908, the value arm, read from the define the other int8 plans
#: read.
comptime INT8_TUNED_VALUE_SABOTAGE = is_defined["MOJOLEARN_LOWBIT_SABOTAGE"]()

#: The unit's k step on both vendors. SCHEDULING.
comptime INT8_TUNED_K_TILE = 32

#: The side of one unit tile: two m16n8 halves on NVIDIA, one MFMA on AMD.
comptime INT8_TUNED_TILE = 16

#: Bytes of padding after a staged row (see THE STAGED ROW).
comptime INT8_TUNED_ROW_PAD = 16

#: Whether this build has the unit at all.
comptime INT8_TUNED_AVAILABLE = lib_int8_matrix_unit_for[TARGET_COLUMN]()

#: Whether this build has the direct kernel's probes: NVIDIA only, since
#: what they test is NVIDIA's fragment layout.
comptime INT8_DIRECT_AVAILABLE = TARGET_COLUMN == COLUMN_NVIDIA

#: The staged plans. SCHEDULING, every one; the gate runs every one on
#: every shape. The name says what changed against the plan before it.
comptime INT8_TUNED_PLAN_STAGED = 0  #: 16x16 per warp, 2x2 warps, KB 32, LW 4
comptime INT8_TUNED_PLAN_FRAG2 = 1  #: 32x32 per warp
comptime INT8_TUNED_PLAN_FRAG4 = 2  #: 64x64 per warp
comptime INT8_TUNED_PLAN_WIDE = 3  #: 64x64 per warp, 2x4 warps
comptime INT8_TUNED_PLAN_PACK = 4  #: FRAG4 with 16-byte staging loads
comptime INT8_TUNED_PLAN_K64 = 5  #: PACK with 64 k steps per window
comptime INT8_TUNED_PLAN_K128 = 6  #: PACK with 128 k steps per window
comptime INT8_TUNED_PLAN_WIDE_K64 = 7  #: WIDE with 16-byte loads and KB 64
comptime INT8_TUNED_PLAN_ROW = 8  #: 16x64 per warp, 1x4 warps: the decode rows
comptime INT8_TUNED_PLAN_COUNT = 9

#: Outputs of at most this many rows take the ROW plan: a 128-row block
#: would multiply 112 rows of zero codes for them.
comptime INT8_TUNED_ROW_MAX_M = 16

#: The direct kernel's instantiations (NVIDIA only).
comptime INT8_DIRECT_REFERENCE_LOADS = 0  #: the reference, respelled here: the control
comptime INT8_DIRECT_ALIGNED_LOADS = 1  #: fragment loads with their alignment stated
comptime INT8_DIRECT_SCALAR_ACC = 2  #: ALIGNED, accumulators as four scalars
comptime INT8_DIRECT_PROBE_HOISTED = 3  #: PROBE: fragments loaded once, not per k step
comptime INT8_DIRECT_PROBE_ONE_HALF = 4  #: PROBE: one m16n8 half per k step
comptime INT8_DIRECT_PROBE_RAW_STORE = 5  #: PROBE: the epilogue without its seam
comptime INT8_DIRECT_COUNT = 6


def int8_tuned_sabotage_name() -> String:
    comptime if INT8_TUNED_SABOTAGE:
        return String("STAGING_PAD_NOT_WRITTEN")
    elif INT8_TUNED_VALUE_SABOTAGE:
        return String("LOWBIT_VALUE_FLIP")
    else:
        return String("none")


def int8_tuned_plan_name(plan: Int) -> String:
    """No spaces: the timing harness prints it as one field."""
    if plan == INT8_TUNED_PLAN_STAGED:
        return String("staged.w16x16.b32x32.k32.l4")
    if plan == INT8_TUNED_PLAN_FRAG2:
        return String("staged.w32x32.b64x64.k32.l4")
    if plan == INT8_TUNED_PLAN_FRAG4:
        return String("staged.w64x64.b128x128.k32.l4")
    if plan == INT8_TUNED_PLAN_WIDE:
        return String("staged.w64x64.b128x256.k32.l4")
    if plan == INT8_TUNED_PLAN_PACK:
        return String("staged.w64x64.b128x128.k32.l16")
    if plan == INT8_TUNED_PLAN_K64:
        return String("staged.w64x64.b128x128.k64.l16")
    if plan == INT8_TUNED_PLAN_K128:
        return String("staged.w64x64.b128x128.k128.l16")
    if plan == INT8_TUNED_PLAN_WIDE_K64:
        return String("staged.w64x64.b128x256.k64.l16")
    return String("staged.w16x64.b16x256.k64.l16")


def int8_direct_name(which: Int) -> String:
    """No spaces. A PROBE's product is wrong on purpose."""
    if which == INT8_DIRECT_REFERENCE_LOADS:
        return String("direct.reference-loads")
    if which == INT8_DIRECT_ALIGNED_LOADS:
        return String("direct.aligned-loads")
    if which == INT8_DIRECT_SCALAR_ACC:
        return String("direct.aligned-loads.scalar-acc")
    if which == INT8_DIRECT_PROBE_HOISTED:
        return String("probe.loads-hoisted")
    if which == INT8_DIRECT_PROBE_ONE_HALF:
        return String("probe.one-half")
    return String("probe.raw-store")


def int8_direct_is_probe(which: Int) -> Bool:
    """Whether the instantiation computes a wrong product on purpose."""
    return which >= INT8_DIRECT_PROBE_HOISTED


def int8_tuned_dispatch(m: Int, n: Int, k: Int) -> Int:
    """The plan `identical_gemm_int8_mma_tuned_into` takes. Reads the shape
    and may: every plan is the profile. The threshold and the choice are
    the measurements of `docs/lanes/progress/lowbit-mma-speed.md`."""
    if m <= INT8_TUNED_ROW_MAX_M:
        return INT8_TUNED_PLAN_ROW
    return INT8_TUNED_PLAN_K64


# ===========================================================================
# the epilogue, shared by every kernel of this file
# ===========================================================================


@always_inline
def _store_cell_tuned(
    c: MutPointer[Float32, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    acc: Int32,
    i: Int,
    j: Int,
    m: Int,
    n: Int,
):
    """`gemm_int8_mma.mojo::_store_cell`: the dequantization seam (L-5,
    L-6) and the store, masked to the output."""
    if i >= m or j >= n:
        return
    var out = dequant_int8_pinned(acc, Int(ea.unsafe_load(i)) + Int(eb.unsafe_load(j)))
    comptime if INT8_TUNED_VALUE_SABOTAGE:
        out = gemm_oracle_sabotage_value_flip(out)
    c.unsafe_store(i * n + j, out)


# ===========================================================================
# THE DIRECT KERNEL: the reference's schedule, one thing changed at a time
# ===========================================================================


@always_inline
def _pack4_aligned(
    p: MutPointer[Int8, MutAnyOrigin], row: Int, k0: Int, rows: Int, k: Int
) -> Int32:
    """`_pack4` with the load's alignment STATED. The reference takes its
    vector load only where the address is a multiple of four and does not
    say so to the compiler, which may then assemble the word from four
    byte loads. Same bytes, same zero codes."""
    if row >= rows:
        return Int32(0)
    var base = row * k
    if k0 + 4 <= k and (k & 3) == 0:
        return bitcast[DType.int32, 1]((p + base + k0).load[width=4, alignment=4]())[0]
    var v = SIMD[DType.int8, 4](0)
    comptime for i in range(4):
        if k0 + i < k:
            v[i] = p.unsafe_load(base + k0 + i)
    return bitcast[DType.int32, 1](v)[0]


@always_inline
def _frag_word[ALIGNED: Bool](
    p: MutPointer[Int8, MutAnyOrigin], row: Int, k0: Int, rows: Int, k: Int
) -> Int32:
    comptime if ALIGNED:
        return _pack4_aligned(p, row, k0, rows, k)
    else:
        return _pack4(p, row, k0, rows, k)


@always_inline
def _direct_warp_tile[WHICH: Int](
    c: MutPointer[Float32, MutAnyOrigin],
    qa: MutPointer[Int8, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    qb: MutPointer[Int8, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    lane: Int,
    row0: Int,
    col0: Int,
    m: Int,
    n: Int,
    k: Int,
):
    """`gemm_int8_mma.mojo::_nvidia_warp_tile`, its fragment layout and its
    order, with the one change `WHICH` names."""
    comptime ALIGNED = WHICH != INT8_DIRECT_REFERENCE_LOADS
    var g = lane >> 2
    var t = lane & 3
    var ra0 = row0 + g
    var ra1 = row0 + g + 8
    var cb0 = col0 + g
    var cb1 = col0 + 8 + g
    var acc0 = SIMD[DType.int32, 4](0)
    var acc1 = SIMD[DType.int32, 4](0)
    # SCALAR_ACC: the eight accumulators as eight scalars, so nothing
    # re-spells the intrinsic's register pack as a vector.
    var s0 = Int32(0)
    var s1 = Int32(0)
    var s2 = Int32(0)
    var s3 = Int32(0)
    var s4 = Int32(0)
    var s5 = Int32(0)
    var s6 = Int32(0)
    var s7 = Int32(0)
    # PROBE_HOISTED: the first k step's fragments, read once.
    var h0 = _frag_word[ALIGNED](qa, ra0, t * 4, m, k)
    var h1 = _frag_word[ALIGNED](qa, ra1, t * 4, m, k)
    var h2 = _frag_word[ALIGNED](qa, ra0, t * 4 + 16, m, k)
    var h3 = _frag_word[ALIGNED](qa, ra1, t * 4 + 16, m, k)
    var h4 = _frag_word[ALIGNED](qb, cb0, t * 4, n, k)
    var h5 = _frag_word[ALIGNED](qb, cb0, t * 4 + 16, n, k)
    var h6 = _frag_word[ALIGNED](qb, cb1, t * 4, n, k)
    var h7 = _frag_word[ALIGNED](qb, cb1, t * 4 + 16, n, k)
    for kt in range(0, k, INT8_TUNED_K_TILE):
        var ka = kt + t * 4
        var a0 = h0
        var a1 = h1
        var a2 = h2
        var a3 = h3
        var b0 = h4
        var b1 = h5
        var b2 = h6
        var b3 = h7
        comptime if WHICH != INT8_DIRECT_PROBE_HOISTED:
            a0 = _frag_word[ALIGNED](qa, ra0, ka, m, k)
            a1 = _frag_word[ALIGNED](qa, ra1, ka, m, k)
            a2 = _frag_word[ALIGNED](qa, ra0, ka + 16, m, k)
            a3 = _frag_word[ALIGNED](qa, ra1, ka + 16, m, k)
            b0 = _frag_word[ALIGNED](qb, cb0, ka, n, k)
            b1 = _frag_word[ALIGNED](qb, cb0, ka + 16, n, k)
        comptime if WHICH == INT8_DIRECT_SCALAR_ACC:
            b2 = _frag_word[ALIGNED](qb, cb1, ka, n, k)
            b3 = _frag_word[ALIGNED](qb, cb1, ka + 16, n, k)
            var r0 = llvm_intrinsic[
                "llvm.nvvm.mma.m16n8k32.row.col.s8",
                _RegisterPackType[Int32, Int32, Int32, Int32],
                has_side_effect=False,
            ](a0, a1, a2, a3, b0, b1, s0, s1, s2, s3)
            s0 = r0[0]
            s1 = r0[1]
            s2 = r0[2]
            s3 = r0[3]
            var r1 = llvm_intrinsic[
                "llvm.nvvm.mma.m16n8k32.row.col.s8",
                _RegisterPackType[Int32, Int32, Int32, Int32],
                has_side_effect=False,
            ](a0, a1, a2, a3, b2, b3, s4, s5, s6, s7)
            s4 = r1[0]
            s5 = r1[1]
            s6 = r1[2]
            s7 = r1[3]
        elif WHICH == INT8_DIRECT_PROBE_ONE_HALF:
            acc0 = _imma_m16n8k32(a0, a1, a2, a3, b0, b1, acc0)
        else:
            acc0 = _imma_m16n8k32(a0, a1, a2, a3, b0, b1, acc0)
            comptime if WHICH != INT8_DIRECT_PROBE_HOISTED:
                b2 = _frag_word[ALIGNED](qb, cb1, ka, n, k)
                b3 = _frag_word[ALIGNED](qb, cb1, ka + 16, n, k)
            acc1 = _imma_m16n8k32(a0, a1, a2, a3, b2, b3, acc1)
    comptime if WHICH == INT8_DIRECT_SCALAR_ACC:
        acc0 = SIMD[DType.int32, 4](s0, s1, s2, s3)
        acc1 = SIMD[DType.int32, 4](s4, s5, s6, s7)
    comptime if WHICH == INT8_DIRECT_PROBE_ONE_HALF:
        # The half that did not run stores the other's cells: written, and
        # wrong on purpose.
        acc1 = acc0
    var jc = col0 + t * 2
    comptime for h in range(2):
        comptime for e in range(4):
            var acc = acc0[e]
            comptime if h == 1:
                acc = acc1[e]
            var gi = ra0 if e < 2 else ra1
            var gj = jc + h * 8 + (e & 1)
            comptime if WHICH == INT8_DIRECT_PROBE_RAW_STORE:
                # PROBE: no seam and no exponent loads; the backend's own
                # conversion of the Int32.
                if gi < m and gj < n:
                    c.unsafe_store(gi * n + gj, Float32(acc))
            else:
                _store_cell_tuned(c, ea, eb, acc, gi, gj, m, n)


def identical_gemm_int8_mma_direct_kernel[WHICH: Int](
    c: MutPointer[Float32, MutAnyOrigin],
    qa: MutPointer[Int8, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    qb: MutPointer[Int8, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
):
    """`identical_gemm_int8_mma_kernel`'s grid, block and tile ownership
    (2 x 2 warps, a 32 x 32 tile per block), on NVIDIA. On any other target
    the body is dead and the launcher refuses."""
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var warp = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(lane_id())
    var wm = warp // 2
    var wn = warp - wm * 2
    var row0 = Int(block_idx.y) * 32 + wm * INT8_TUNED_TILE
    var col0 = Int(block_idx.x) * 32 + wn * INT8_TUNED_TILE
    # Uniform across the warp, as in the reference.
    if row0 >= m or col0 >= n:
        return
    comptime if is_nvidia_gpu():
        _direct_warp_tile[WHICH](c, qa, ea, qb, eb, lane, row0, col0, m, n, k)
    else:
        return


def _refuse_tuned_shape(m: Int, n: Int, k: Int) raises:
    if not int8_mma_admits(m, n, k):
        raise Error(
            "identical_gemm_int8_mma_tuned: m, n and k must be positive and k"
            " at most " + String(INT8_MAX_K) + " (contract L-7), got m="
            + String(m) + " n=" + String(n) + " k=" + String(k)
        )


def _launch_direct[WHICH: Int](
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
    comptime kern = identical_gemm_int8_mma_direct_kernel[WHICH]
    ctx.enqueue_function[kern](
        c.unsafe_ptr(),
        qa.unsafe_ptr(),
        ea.unsafe_ptr(),
        qb.unsafe_ptr(),
        eb.unsafe_ptr(),
        Int32(m),
        Int32(n),
        Int32(k),
        grid_dim=((n + 31) // 32, (m + 31) // 32, 1),
        block_dim=(4 * WARP_SIZE, 1, 1),
    )


def identical_gemm_int8_mma_direct_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut qa: DeviceBuffer[DType.int8],
    mut ea: DeviceBuffer[DType.int32],
    mut qb: DeviceBuffer[DType.int8],
    mut eb: DeviceBuffer[DType.int32],
    m: Int,
    n: Int,
    k: Int,
    which: Int,
) raises:
    """The direct kernel on a NAMED instantiation, for the gate (the three
    that are plans) and the timing harness (all six). Refuses by name on a
    column that is not NVIDIA. Asynchronous."""
    comptime if not INT8_DIRECT_AVAILABLE:
        raise Error(
            "identical_gemm_int8_mma_direct: column " + column_name(TARGET_COLUMN)
            + " is not NVIDIA; the direct kernel tests NVIDIA's fragment layout"
        )
    else:
        _refuse_tuned_shape(m, n, k)
        if which == INT8_DIRECT_REFERENCE_LOADS:
            _launch_direct[INT8_DIRECT_REFERENCE_LOADS](ctx, c, qa, ea, qb, eb, m, n, k)
        elif which == INT8_DIRECT_ALIGNED_LOADS:
            _launch_direct[INT8_DIRECT_ALIGNED_LOADS](ctx, c, qa, ea, qb, eb, m, n, k)
        elif which == INT8_DIRECT_SCALAR_ACC:
            _launch_direct[INT8_DIRECT_SCALAR_ACC](ctx, c, qa, ea, qb, eb, m, n, k)
        elif which == INT8_DIRECT_PROBE_HOISTED:
            _launch_direct[INT8_DIRECT_PROBE_HOISTED](ctx, c, qa, ea, qb, eb, m, n, k)
        elif which == INT8_DIRECT_PROBE_ONE_HALF:
            _launch_direct[INT8_DIRECT_PROBE_ONE_HALF](ctx, c, qa, ea, qb, eb, m, n, k)
        elif which == INT8_DIRECT_PROBE_RAW_STORE:
            _launch_direct[INT8_DIRECT_PROBE_RAW_STORE](ctx, c, qa, ea, qb, eb, m, n, k)
        else:
            raise Error(
                "identical_gemm_int8_mma_direct: no instantiation " + String(which)
            )


# ===========================================================================
# THE STAGED KERNEL
# ===========================================================================


@always_inline
def _stage_window[
    ROWS: Int, KB: Int, NT: Int, LW: Int, SW: Int
](
    dst: UnsafePointer[Int32, MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    q: MutPointer[Int8, MutAnyOrigin],
    row0: Int,
    rows: Int,
    k0: Int,
    k: Int,
    tid: Int,
):
    """One window of one operand, device memory to threadgroup memory.

    A LOAD is `LW` consecutive codes of one row; the window has `ROWS * KB
    / LW` of them and thread `tid` owns loads `tid, tid + NT, ...`, so
    consecutive threads read consecutive bytes of a row. Code `(row r, step
    p)` lands at byte `r * 4 SW + p` of `dst`. A row at or beyond `rows` and
    a step at or beyond `k` are the ZERO CODE, and the zero codes are
    STORED: the page still holds the window before. The vector load is
    taken only when it is aligned (`k` a multiple of `LW` makes every row
    start one) and entirely inside the row; its alignment is stated."""
    comptime LPR = KB // LW
    comptime LOADS = ROWS * LPR
    comptime SL = (LOADS + NT - 1) // NT
    comptime for sl in range(SL):
        var s = sl * NT + tid
        if s < LOADS:
            var r = s // LPR
            var kq = (s % LPR) * LW
            var v = SIMD[DType.int8, LW](0)
            var gr = row0 + r
            var inside = gr < rows and k0 + kq < k
            if inside:
                var base = gr * k + k0 + kq
                if k0 + kq + LW <= k and (k & (LW - 1)) == 0:
                    v = (q + base).load[width=LW, alignment=LW]()
                else:
                    comptime for i in range(LW):
                        if k0 + kq + i < k:
                            v[i] = q.unsafe_load(base + i)
            comptime if INT8_TUNED_SABOTAGE:
                # SABOTAGE: the padding rule broken. What lies wholly
                # outside the operand is not written.
                if inside:
                    (dst + r * SW + kq // 4).store[alignment=LW](
                        bitcast[DType.int32, LW // 4](v)
                    )
            else:
                (dst + r * SW + kq // 4).store[alignment=LW](
                    bitcast[DType.int32, LW // 4](v)
                )


def identical_gemm_int8_mma_tuned_kernel[
    FM: Int, FN: Int, WM: Int, WN: Int, KB: Int, LW: Int
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
    """OP_NT, `C[m x n] = Qa[m x k] . Qb[n x k]^T`. One block owns a `BM x
    BN` output tile and all of its `k`; per window of `KB` steps the block
    stages `BM` rows of `Qa` and `BN` rows of `Qb`, then every warp runs
    `KB / 32` unit steps over its `16 FM x 16 FN` tile out of threadgroup
    memory. Int32 accumulation, then the reference's epilogue. Grid
    `(ceil(n / BN), ceil(m / BM), 1)`, block `WM * WN * WARP_SIZE`.

    Every thread of the block reaches every `barrier()`: the window count is
    a function of `k` and the comptime `KB`; a warp whose tile lies wholly
    outside the output still stages and still waits, and only its unit
    steps and its stores are skipped (`live`, uniform across the warp, so
    no lane leaves a unit step another lane enters).

    On a target with no integer matrix unit the unit steps are dead code
    and the launcher refuses (`lib_int8_matrix_unit_for`)."""
    comptime TM = INT8_TUNED_TILE * FM
    comptime TN = INT8_TUNED_TILE * FN
    comptime BM = WM * TM
    comptime BN = WN * TN
    comptime NT = WM * WN * WARP_SIZE
    comptime SS = KB + INT8_TUNED_ROW_PAD
    comptime SW = SS // 4
    comptime KSTEPS = KB // INT8_TUNED_K_TILE
    #: Accumulator registers of four Int32: two per unit tile on NVIDIA
    #: (the two m16n8 halves), one on AMD.
    comptime HALVES = 2 if is_nvidia_gpu() else 1
    comptime NACC = FM * FN * HALVES

    comptime assert KB % INT8_TUNED_K_TILE == 0 and KB >= INT8_TUNED_K_TILE, (
        "identical_gemm_int8_mma_tuned_kernel: a window is whole unit steps"
    )
    comptime assert LW == 4 or LW == 8 or LW == 16, (
        "identical_gemm_int8_mma_tuned_kernel: a staging load is 4, 8 or 16 bytes"
    )
    comptime assert NT <= 1024, (
        "identical_gemm_int8_mma_tuned_kernel: a block is at most 1024 threads"
    )
    comptime assert (BM + BN) * SS <= column_shared_limit(TARGET_COLUMN), (
        "identical_gemm_int8_mma_tuned_kernel: the staged window does not fit"
        " the column's threadgroup memory"
    )

    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var tid = Int(thread_idx.x)
    var warp = tid // WARP_SIZE
    var lane = Int(lane_id())
    var wm = warp // WN
    var wn = warp - wm * WN
    var i0 = Int(block_idx.y) * BM
    var j0 = Int(block_idx.x) * BN
    # This warp's tile, inside the block and in the output.
    var lrow = wm * TM
    var lcol = wn * TN
    var live = i0 + lrow < m and j0 + lcol < n

    var as_ = stack_allocation[
        BM * SW,
        Scalar[DType.int32],
        alignment=16,
        address_space = AddressSpace.SHARED,
    ]()
    var bs_ = stack_allocation[
        BN * SW,
        Scalar[DType.int32],
        alignment=16,
        address_space = AddressSpace.SHARED,
    ]()

    var acc = InlineArray[SIMD[DType.int32, 4], NACC](fill=SIMD[DType.int32, 4](0))

    var windows = (k + KB - 1) // KB
    for w in range(windows):
        var k0 = w * KB
        _stage_window[BM, KB, NT, LW, SW](as_, qa, i0, m, k0, k, tid)
        _stage_window[BN, KB, NT, LW, SW](bs_, qb, j0, n, k0, k, tid)
        barrier()
        if live:
            comptime for ks in range(KSTEPS):
                # A unit step wholly beyond `k` adds zero codes only.
                # Uniform across the block.
                if k0 + ks * INT8_TUNED_K_TILE < k:
                    comptime if is_nvidia_gpu():
                        # PTX ISA, Matrix Fragments for mma.m16n8k32 with
                        # .s8 operands (the reference's note): group `g`,
                        # thread in group `t`. A word index is a byte index
                        # over four.
                        var g = lane >> 2
                        var t = lane & 3
                        var kw = ks * 8 + t
                        var bfr = InlineArray[Int32, 4 * FN](fill=Int32(0))
                        comptime for fq in range(FN):
                            comptime for h in range(2):
                                var br = (lcol + fq * 16 + h * 8 + g) * SW + kw
                                bfr[(fq * 2 + h) * 2] = bs_.unsafe_load(br)
                                bfr[(fq * 2 + h) * 2 + 1] = bs_.unsafe_load(br + 4)
                        comptime for fm in range(FM):
                            var ar = (lrow + fm * 16 + g) * SW + kw
                            var a0 = as_.unsafe_load(ar)
                            var a1 = as_.unsafe_load(ar + 8 * SW)
                            var a2 = as_.unsafe_load(ar + 4)
                            var a3 = as_.unsafe_load(ar + 8 * SW + 4)
                            comptime for fq in range(FN):
                                comptime for h in range(2):
                                    acc[(fm * FN + fq) * 2 + h] = _imma_m16n8k32(
                                        a0,
                                        a1,
                                        a2,
                                        a3,
                                        bfr[(fq * 2 + h) * 2],
                                        bfr[(fq * 2 + h) * 2 + 1],
                                        acc[(fm * FN + fq) * 2 + h],
                                    )
                    elif is_amd_gpu():
                        # CDNA3 `v_mfma_i32_16x16x32_i8` (the reference's
                        # note): lane `l` holds row or column `l & 15`,
                        # steps `8 (l >> 4) .. + 7` of the 32.
                        var i16 = lane & 15
                        var kw = ks * 8 + (lane >> 4) * 2
                        var bfr = InlineArray[Int64, FN](fill=Int64(0))
                        comptime for fq in range(FN):
                            bfr[fq] = bitcast[DType.int64, 1](
                                (bs_ + (lcol + fq * 16 + i16) * SW + kw).load[
                                    width=2, alignment=8
                                ]()
                            )[0]
                        comptime for fm in range(FM):
                            var a = bitcast[DType.int64, 1](
                                (as_ + (lrow + fm * 16 + i16) * SW + kw).load[
                                    width=2, alignment=8
                                ]()
                            )[0]
                            comptime for fq in range(FN):
                                acc[fm * FN + fq] = llvm_intrinsic[
                                    "llvm.amdgcn.mfma.i32.16x16x32.i8",
                                    SIMD[DType.int32, 4],
                                    has_side_effect=False,
                                ](a, bfr[fq], acc[fm * FN + fq], Int32(0), Int32(0), Int32(0))
        # The page is free once every thread has left the window's steps.
        barrier()

    if not live:
        return

    # ---- THE EPILOGUE. The accumulators leave their registers once, then
    # one loop stores the cells, so the seam is spelled once per kernel and
    # not once per cell.
    var outs = stack_allocation[NACC * 4, Scalar[DType.int32]]()
    comptime for a in range(NACC):
        comptime for e in range(4):
            outs.unsafe_store(a * 4 + e, acc[a][e])
    comptime if is_nvidia_gpu():
        # C/D (16 x 8, s32): c0, c1 row `g`, columns `2 t`, `2 t + 1`;
        # c2, c3 row `g + 8`.
        var g = lane >> 2
        var t = lane & 3
        for a in range(NACC):
            var tile = a >> 1
            var h = a & 1
            var fm = tile // FN
            var fq = tile - fm * FN
            var gi = i0 + lrow + fm * 16 + g
            var gj = j0 + lcol + fq * 16 + h * 8 + t * 2
            _store_cell_tuned(c, ea, eb, outs.unsafe_load(a * 4), gi, gj, m, n)
            _store_cell_tuned(c, ea, eb, outs.unsafe_load(a * 4 + 1), gi, gj + 1, m, n)
            _store_cell_tuned(c, ea, eb, outs.unsafe_load(a * 4 + 2), gi + 8, gj, m, n)
            _store_cell_tuned(c, ea, eb, outs.unsafe_load(a * 4 + 3), gi + 8, gj + 1, m, n)
    elif is_amd_gpu():
        # D[i][j]: lane `j + 16 (i // 4)`, register `i % 4`.
        var i16 = lane & 15
        var i4 = (lane >> 4) * 4
        for a in range(NACC):
            var fm = a // FN
            var fq = a - fm * FN
            var gi = i0 + lrow + fm * 16 + i4
            var gj = j0 + lcol + fq * 16 + i16
            comptime for e in range(4):
                _store_cell_tuned(c, ea, eb, outs.unsafe_load(a * 4 + e), gi + e, gj, m, n)


def _launch_tuned[
    FM: Int, FN: Int, WM: Int, WN: Int, KB: Int, LW: Int
](
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
    comptime BM = WM * FM * INT8_TUNED_TILE
    comptime BN = WN * FN * INT8_TUNED_TILE
    comptime kern = identical_gemm_int8_mma_tuned_kernel[FM, FN, WM, WN, KB, LW]
    ctx.enqueue_function[kern](
        c.unsafe_ptr(),
        qa.unsafe_ptr(),
        ea.unsafe_ptr(),
        qb.unsafe_ptr(),
        eb.unsafe_ptr(),
        Int32(m),
        Int32(n),
        Int32(k),
        grid_dim=((n + BN - 1) // BN, (m + BM - 1) // BM, 1),
        block_dim=(WM * WN * WARP_SIZE, 1, 1),
    )


def identical_gemm_int8_mma_tuned_with_plan(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut qa: DeviceBuffer[DType.int8],
    mut ea: DeviceBuffer[DType.int32],
    mut qb: DeviceBuffer[DType.int8],
    mut eb: DeviceBuffer[DType.int32],
    m: Int,
    n: Int,
    k: Int,
    plan: Int,
) raises:
    """The staged kernel on a NAMED plan. The gate and the timing harness
    call this; a caller that wants the product calls
    `identical_gemm_int8_mma_tuned_into`. Refuses by name on a column whose
    kernel-matrix row says it has no integer matrix unit. Asynchronous."""
    comptime if not INT8_TUNED_AVAILABLE:
        raise Error(
            "identical_gemm_int8_mma_tuned: column " + column_name(TARGET_COLUMN)
            + " has no int8 matrix unit (kernel_matrix row"
            " lib_int8_matrix_unit_for); the flat kernel serves it"
        )
    else:
        _refuse_tuned_shape(m, n, k)
        if plan == INT8_TUNED_PLAN_STAGED:
            _launch_tuned[1, 1, 2, 2, 32, 4](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_TUNED_PLAN_FRAG2:
            _launch_tuned[2, 2, 2, 2, 32, 4](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_TUNED_PLAN_FRAG4:
            _launch_tuned[4, 4, 2, 2, 32, 4](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_TUNED_PLAN_WIDE:
            _launch_tuned[4, 4, 2, 4, 32, 4](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_TUNED_PLAN_PACK:
            _launch_tuned[4, 4, 2, 2, 32, 16](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_TUNED_PLAN_K64:
            _launch_tuned[4, 4, 2, 2, 64, 16](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_TUNED_PLAN_K128:
            _launch_tuned[4, 4, 2, 2, 128, 16](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_TUNED_PLAN_WIDE_K64:
            _launch_tuned[4, 4, 2, 4, 64, 16](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_TUNED_PLAN_ROW:
            _launch_tuned[1, 4, 1, 4, 64, 16](ctx, c, qa, ea, qb, eb, m, n, k)
        else:
            raise Error("identical_gemm_int8_mma_tuned: no plan " + String(plan))


def identical_gemm_int8_mma_tuned_into(
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
    """The tuned unit plan on the plan `int8_tuned_dispatch` names.
    `identical_gemm_int8_mma_into`'s signature and its bits. Asynchronous."""
    identical_gemm_int8_mma_tuned_with_plan(
        ctx, c, qa, ea, qb, eb, m, n, k, int8_tuned_dispatch(m, n, k)
    )
