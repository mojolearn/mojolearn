# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`mojolearn.identical.gemm.int8i32.v1` on AMD's integer matrix unit, the
schedules written FOR A WAVEFRONT OF 64: the plans of
`gemm/checks/gemm_int8_mma_tuned.mojo` in AMD's geometry, and a kernel that
stages nothing.

Lane lane/lowbit-amd-tuned, 2026-09-29 (Andrew: "why don't we port the tuned
kernel to amd?"). Contract clause L-9 of `gemm/IDENTICAL_LOWBIT_CONTRACT.md`.
The reference unit plan is `gemm/checks/gemm_int8_mma.mojo`; the tuned plans
and the four-product kernel are lane/lowbit-mma-speed's
`gemm/checks/gemm_int8_mma_tuned.mojo`; NEITHER IS EDITED HERE. This file
instantiates that file's staged kernels at geometries it does not name and
adds kernels of its own, every one AMD only. The gate is
`gemm/checks/gemm_int8_mma_amd_check.mojo`. NOTHING HERE IS DISPATCHED:
`identical_gemm_int8_into` still takes the reference plan.

WHY IT EXISTS. On the MI325X the reference unit plan, which stages nothing,
already takes 0.28 to 0.37 of fp32.v1's time at the 512-token rows
(lane/lowbit-units, qkv.t512: 0.268 ms against 0.953), where on the H100 it
took 1.4. So what the H100 taught (threadgroup staging, 32 x 32 per warp,
sixteen warps a block) is a hypothesis here and not a result. A wavefront is
64 lanes and one `v_mfma_i32_16x16x32_i8` is a whole 16 x 16 tile over 32
steps: a lane holds EIGHT consecutive codes of one row of each operand, so
one eight-byte load is one operand of one unit step, and a wave that owns
`FM x FN` unit tiles runs `FM FN` unit steps on `FM + FN` loads.

WHY NO PLAN HERE CAN MOVE A BIT. Clause L-9's argument, unchanged: a product
of two int8 codes is an exact integer, an Int32 sum of them cannot overflow
under the profile's bound on `k`, and a sum of exact integers is the same
integer under every order, grouping and tile. This file changes which lane
holds which cell, how many unit tiles a wave owns, how many waves a block
holds and whether a code waits in threadgroup memory. The only floating
steps stay `dequant_int8_pinned` (L-5, L-6) through the tuned file's
`_store_cell_tuned`, called with the reference's arguments.

THE PADDING RULE is the reference's: a step at or beyond `k`, and a row at
or beyond `m` or `n`, are the ZERO CODE, and cells beyond `m`, `n` are
masked at the store. A unit tile that lies WHOLLY beyond `m` or `n` is
skipped, which the whole wave decides alike: its every product is a product
with a zero code and its every cell is masked.

THE DIRECT KERNEL (`identical_gemm_int8_mma_amd_direct_kernel`). The
reference's schedule with two things freed: a wave owns `16 FM x 16 FN`
cells, and a block holds `WM x WN` waves. Nothing is staged and there is no
barrier: a lane reads its eight codes of each operand row from device
memory at every unit step.

    FM, FN   16 x 16 unit tiles per wave, along m and along n
    WM, WN   waves per block
    STATED   whether a load states its alignment. False is the reference's
             load (`_pack8`, respelled): the CONTROL of the first lever,
             the one that bought a factor of 3.6 on the H100.
    KL       bytes per load, 8 or 16. At 16 one load is a lane's operand
             of TWO unit steps, and the k loop takes 64 steps a turn: lane
             group `g = l >> 4` reads steps `16 g .. 16 g + 15` of the 64,
             the first unit step takes the low eight and the second the
             high eight. Both operands are cut the same way, so every
             product is still `a[p] b[p]` of one `p`, and which unit step
             adds it is scheduling.

THE STAGED PLANS are lane/lowbit-mma-speed's kernel
(`identical_gemm_int8_mma_tuned_kernel`, its MFMA branch) at geometries that
file does not name: fewer and larger waves, since a wave here is twice an
H100's warp.

FOUR PRODUCTS (`identical_gemm_int8_pieces_amd_direct_kernel`). The
fifteen-bit profile's four piece products with nothing staged: a lane reads
eight codes of each of the two planes of each operand row, and a unit tile
runs four unit steps into three accumulators, HH, HL + LH and LL, as the
tuned file's four-product kernel does. It stores the three Int32 sums and
nothing else; the bound on `k` is that kernel's `INT8_PIECES_MAX_K`, for
the operands it is stated for.

THE ALIGNMENT OF A LOAD IS STATED ONLY WHERE IT IS TRUE, the tuned file's
rule: `k` a multiple of `KL` (every row then starts on one), the load
inside the row, and the base of the buffer a multiple of `KL` bytes, which
the launch reads from the pointers it is about to pass. Where any of the
three fails the codes are read one byte at a time.
`-D MOJOLEARN_INT8_TUNED_UNSTATED=1` answers "not aligned" at every launch.

THE SABOTAGE ARMS.
  `-D MOJOLEARN_INT8_AMD_SABOTAGE=1` breaks the padding rule of the direct
      loads: a step at or beyond `k` reads the row's LAST code where the
      zero code belongs. A SCHEDULING defect, the kind this file can have.
      It reaches every shape whose `k` is not a whole number of the
      plan's turns (32 steps at `KL` 8, 64 at 16) and cannot reach one
      whose `k` is, which must still pass. The staged plans do
      not read it; their own arm is the tuned file's
      `MOJOLEARN_INT8_TUNED_SABOTAGE`.
  `-D MOJOLEARN_INT8_PIECES_SABOTAGE=1` pairs the wrong fragments in the
      four-product kernels: the middle accumulator takes HL twice and never
      LH (the tuned file's arm, read here too).
  `-D MOJOLEARN_LOWBIT_SABOTAGE=1` flips the value of every cell stored
      (DEVIATION 2908), through the tuned file's epilogue.
"""

from std.gpu import WARP_SIZE, block_idx, lane_id, thread_idx
from std.memory import bitcast
from std.sys import is_defined
from std.sys.info import is_amd_gpu
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.kernel_matrix import COLUMN_AMD, TARGET_COLUMN, column_name
from gemm.checks.gemm_int8_mma import _pack8
from gemm.checks.gemm_int8_mma_tuned import (
    INT8_PIECES_SABOTAGE,
    INT8_TUNED_MAX_TPB,
    INT8_TUNED_ROW_MAX_M,
    INT8_TUNED_SABOTAGE,
    INT8_TUNED_TILE,
    INT8_TUNED_VALUE_SABOTAGE,
    _bases_aligned,
    _launch_pieces,
    _launch_tuned,
    _mfma_i8,
    _refuse_pieces_shape,
    _refuse_tuned_shape,
    _store_cell_tuned,
    _store_sums,
)

#: The scheduling arm of the direct loads: a step beyond `k` is the row's
#: last code. Off in every build that does not name it.
comptime INT8_AMD_SABOTAGE = is_defined["MOJOLEARN_INT8_AMD_SABOTAGE"]()

#: Whether this build has the plans of this file: AMD CDNA only. What they
#: are tuned for is the wavefront of 64 and the MFMA's fragment layout.
comptime INT8_AMD_AVAILABLE = TARGET_COLUMN == COLUMN_AMD

#: The unit's k step. SCHEDULING.
comptime INT8_AMD_K_TILE = 32

#: Bytes of one operand of one unit step in one lane.
comptime INT8_AMD_LANE_BYTES = 8

#: The one-product plans. SCHEDULING, every one; the gate runs every one on
#: every shape. `direct` stages nothing; `staged` is the tuned file's kernel.
comptime INT8_AMD_PLAN_DIRECT_UNSTATED = 0  #: the reference's schedule and load: the control
comptime INT8_AMD_PLAN_DIRECT = 1  #: the same, alignment stated
comptime INT8_AMD_PLAN_DIRECT_L16 = 2  #: the same, 16-byte loads
comptime INT8_AMD_PLAN_DIRECT_ONE_WAVE = 3  #: 16x16 per wave, ONE wave a block
comptime INT8_AMD_PLAN_DIRECT_FRAG2_L8 = 4  #: 32x32 per wave, 2x2 waves, 8-byte loads
comptime INT8_AMD_PLAN_DIRECT_FRAG2 = 5  #: 32x32 per wave, 2x2 waves
comptime INT8_AMD_PLAN_DIRECT_FRAG4 = 6  #: 64x64 per wave, 2x2 waves
comptime INT8_AMD_PLAN_DIRECT_FRAG4_ONE_WAVE = 7  #: 64x64 per wave, one wave a block
comptime INT8_AMD_PLAN_DIRECT_FRAG2X4 = 8  #: 32x64 per wave, 2x2 waves
comptime INT8_AMD_PLAN_DIRECT_ROW = 9  #: 16x64 per wave, 1x4 waves: the decode rows
comptime INT8_AMD_PLAN_DIRECT_ROW_ONE_WAVE = 10  #: 16x64 per wave, one wave a block
comptime INT8_AMD_PLAN_STAGED_WAVES8 = 11  #: 32x32 per wave, 2x4 waves, KB 64, LW 16
comptime INT8_AMD_PLAN_STAGED_FRAG2X4 = 12  #: 32x64 per wave, 2x2 waves, KB 64, LW 16
comptime INT8_AMD_PLAN_STAGED_WAVES2 = 13  #: 64x64 per wave, 1x2 waves, KB 64, LW 16
comptime INT8_AMD_PLAN_STAGED_FRAG2_K128 = 14  #: 32x32 per wave, 2x2 waves, KB 128, LW 16
comptime INT8_AMD_PLAN_COUNT = 15

#: The four-product plans. SCHEDULING, every one.
comptime INT8_AMD_PIECES_DIRECT_L8 = 0  #: 16x16 per wave, 2x2 waves, 8-byte loads
comptime INT8_AMD_PIECES_DIRECT = 1  #: the same, 16-byte loads
comptime INT8_AMD_PIECES_DIRECT_ONE_WAVE = 2  #: 16x16 per wave, one wave a block
comptime INT8_AMD_PIECES_DIRECT_FRAG2 = 3  #: 32x32 per wave, 2x2 waves
comptime INT8_AMD_PIECES_DIRECT_FRAG2X4 = 4  #: 32x64 per wave, 2x2 waves
comptime INT8_AMD_PIECES_DIRECT_ROW = 5  #: 16x64 per wave, 1x4 waves: the decode rows
comptime INT8_AMD_PIECES_STAGED_FRAG2 = 6  #: 32x32 per wave, 2x2 waves, KB 64, LW 16
comptime INT8_AMD_PIECES_STAGED_FRAG2X4 = 7  #: 32x64 per wave, 2x2 waves, KB 64, LW 16
comptime INT8_AMD_PIECES_STAGED_WAVES8 = 8  #: 16x32 per wave, 2x4 waves, KB 64, LW 16
comptime INT8_AMD_PIECES_PLAN_COUNT = 9


def int8_amd_sabotage_name() -> String:
    comptime if INT8_AMD_SABOTAGE:
        return String("DIRECT_TAIL_NOT_THE_ZERO_CODE")
    elif INT8_PIECES_SABOTAGE:
        return String("MIDDLE_TAKES_HL_TWICE")
    elif INT8_TUNED_SABOTAGE:
        return String("STAGING_PAD_NOT_WRITTEN")
    elif INT8_TUNED_VALUE_SABOTAGE:
        return String("LOWBIT_VALUE_FLIP")
    else:
        return String("none")


def int8_amd_plan_name(plan: Int) -> String:
    """No spaces: the timing harness prints it as one field. The form is the
    tuned file's: the wave's tile, the block's, then for a direct plan the
    bytes per load and for a staged plan the k steps per window and the
    bytes per staging load."""
    if plan == INT8_AMD_PLAN_DIRECT_UNSTATED:
        return String("direct.w16x16.b32x32.l8.unstated")
    if plan == INT8_AMD_PLAN_DIRECT:
        return String("direct.w16x16.b32x32.l8")
    if plan == INT8_AMD_PLAN_DIRECT_L16:
        return String("direct.w16x16.b32x32.l16")
    if plan == INT8_AMD_PLAN_DIRECT_ONE_WAVE:
        return String("direct.w16x16.b16x16.l16")
    if plan == INT8_AMD_PLAN_DIRECT_FRAG2_L8:
        return String("direct.w32x32.b64x64.l8")
    if plan == INT8_AMD_PLAN_DIRECT_FRAG2:
        return String("direct.w32x32.b64x64.l16")
    if plan == INT8_AMD_PLAN_DIRECT_FRAG4:
        return String("direct.w64x64.b128x128.l16")
    if plan == INT8_AMD_PLAN_DIRECT_FRAG4_ONE_WAVE:
        return String("direct.w64x64.b64x64.l16")
    if plan == INT8_AMD_PLAN_DIRECT_FRAG2X4:
        return String("direct.w32x64.b64x128.l16")
    if plan == INT8_AMD_PLAN_DIRECT_ROW:
        return String("direct.w16x64.b16x256.l16")
    if plan == INT8_AMD_PLAN_DIRECT_ROW_ONE_WAVE:
        return String("direct.w16x64.b16x64.l16")
    if plan == INT8_AMD_PLAN_STAGED_WAVES8:
        return String("staged.w32x32.b64x128.k64.l16")
    if plan == INT8_AMD_PLAN_STAGED_FRAG2X4:
        return String("staged.w32x64.b64x128.k64.l16")
    if plan == INT8_AMD_PLAN_STAGED_WAVES2:
        return String("staged.w64x64.b64x128.k64.l16")
    return String("staged.w32x32.b64x64.k128.l16")


def int8_amd_plan_is_direct(plan: Int) -> Bool:
    """Whether the plan stages nothing, and so reads this file's scheduling
    arm and not the tuned file's."""
    return plan < INT8_AMD_PLAN_STAGED_WAVES8


def int8_amd_pieces_plan_name(plan: Int) -> String:
    """No spaces."""
    if plan == INT8_AMD_PIECES_DIRECT_L8:
        return String("direct.w16x16.b32x32.l8")
    if plan == INT8_AMD_PIECES_DIRECT:
        return String("direct.w16x16.b32x32.l16")
    if plan == INT8_AMD_PIECES_DIRECT_ONE_WAVE:
        return String("direct.w16x16.b16x16.l16")
    if plan == INT8_AMD_PIECES_DIRECT_FRAG2:
        return String("direct.w32x32.b64x64.l16")
    if plan == INT8_AMD_PIECES_DIRECT_FRAG2X4:
        return String("direct.w32x64.b64x128.l16")
    if plan == INT8_AMD_PIECES_DIRECT_ROW:
        return String("direct.w16x64.b16x256.l16")
    if plan == INT8_AMD_PIECES_STAGED_FRAG2:
        return String("staged.w32x32.b64x64.k64.l16")
    if plan == INT8_AMD_PIECES_STAGED_FRAG2X4:
        return String("staged.w32x64.b64x128.k64.l16")
    return String("staged.w16x32.b32x128.k64.l16")


def int8_amd_pieces_plan_is_direct(plan: Int) -> Bool:
    return plan < INT8_AMD_PIECES_STAGED_FRAG2


def int8_amd_dispatch(m: Int, n: Int, k: Int) -> Int:
    """The plan `identical_gemm_int8_mma_amd_into` takes. Reads the shape
    and may: every plan is the profile. NOT YET A MEASUREMENT: until the
    MI325X has timed the plans this is the first guess (few rows: the
    reference's tile with its alignment stated; many: 32 x 32 per wave),
    and the progress file says when it became one."""
    if m <= INT8_TUNED_ROW_MAX_M:
        return INT8_AMD_PLAN_DIRECT
    return INT8_AMD_PLAN_DIRECT_FRAG2


def int8_amd_pieces_dispatch(m: Int, n: Int, k: Int) -> Int:
    """The plan `identical_gemm_int8_pieces_amd_into` takes. NOT YET A
    MEASUREMENT, as `int8_amd_dispatch`."""
    if m <= INT8_TUNED_ROW_MAX_M:
        return INT8_AMD_PIECES_DIRECT
    return INT8_AMD_PIECES_DIRECT_FRAG2


# ===========================================================================
# the direct load: a lane's codes of one row, zero codes beyond k and the rows
# ===========================================================================


@always_inline
def _lane_words[STATED: Bool, KL: Int](
    p: MutPointer[Int8, MutAnyOrigin],
    row: Int,
    k0: Int,
    rows: Int,
    k: Int,
    aligned: Bool,
) -> SIMD[DType.int64, KL // INT8_AMD_LANE_BYTES]:
    """`KL` consecutive codes of `row` from `k0` as `KL / 8` 64-bit
    registers, code `i` in byte `i % 8` of register `i // 8`: the order the
    MFMA's i8 operand expects (`gemm_int8_mma.mojo::_pack8`, which this is
    at `KL` 8 with `STATED` False). A row at or beyond `rows` and a step at
    or beyond `k` are the ZERO CODE. The vector load is taken only where it
    is aligned (`k` a multiple of `KL` makes every row start a multiple of
    `KL` bytes, and `k0` is always one) and entirely inside the row. With
    `STATED` its alignment is stated, and then only where `aligned` says
    the launch found the buffer's base a multiple of `KL` bytes."""
    comptime KU = KL // INT8_AMD_LANE_BYTES
    if row >= rows:
        return SIMD[DType.int64, KU](0)
    var base = row * k
    if k0 + KL <= k and (k & (KL - 1)) == 0:
        comptime if STATED:
            if aligned:
                return bitcast[DType.int64, KU](
                    p.unsafe_load[width=KL, alignment=KL](base + k0)
                )
        else:
            return bitcast[DType.int64, KU](p.unsafe_load[width=KL](base + k0))
    var v = SIMD[DType.int8, KL](0)
    comptime for i in range(KL):
        if k0 + i < k:
            v[i] = p.unsafe_load(base + k0 + i)
        else:
            comptime if INT8_AMD_SABOTAGE:
                # SABOTAGE: the padding rule broken. A step beyond `k` is
                # the row's last code.
                v[i] = p.unsafe_load(base + k - 1)
    return bitcast[DType.int64, KU](v)


# ===========================================================================
# THE DIRECT KERNEL, one product
# ===========================================================================


def identical_gemm_int8_mma_amd_direct_kernel[
    FM: Int, FN: Int, WM: Int, WN: Int, STATED: Bool
](
    c: MutPointer[Float32, MutAnyOrigin],
    qa: MutPointer[Int8, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    qb: MutPointer[Int8, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    aligned_in: Int32,
):
    """OP_NT, `C[m x n] = Qa[m x k] . Qb[n x k]^T`. One wave owns a `16 FM x
    16 FN` output tile and all of its `k`; at every unit step a lane reads
    eight codes of `FM` rows of `Qa` and of `FN` rows of `Qb` from device
    memory and the wave runs `FM FN` unit steps. Int32 accumulation, then
    the reference's epilogue. Grid `(ceil(n / BN), ceil(m / BM), 1)`, block
    `WM * WN * WARP_SIZE`. No threadgroup memory and no barrier.

    CDNA3 `v_mfma_i32_16x16x32_i8` (the reference's note): lane `l` holds
    row or column `l & 15`, steps `8 (l >> 4) .. + 7` of the 32, and of the
    result column `l & 15`, rows `4 (l >> 4) .. + 3`.

    On any target but AMD the body is dead and the launcher refuses."""
    comptime TM = INT8_TUNED_TILE * FM
    comptime TN = INT8_TUNED_TILE * FN
    comptime BM = WM * TM
    comptime BN = WN * TN
    comptime NACC = FM * FN

    comptime assert WM * WN * WARP_SIZE <= INT8_TUNED_MAX_TPB, (
        "identical_gemm_int8_mma_amd_direct_kernel: a block is at most 1024 threads"
    )

    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var aligned = aligned_in != Int32(0)
    var warp = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(lane_id())
    var wm = warp // WN
    var wn = warp - wm * WN
    var row0 = Int(block_idx.y) * BM + wm * TM
    var col0 = Int(block_idx.x) * BN + wn * TN
    # Uniform across the wave, as in the reference: every lane of a wave
    # shares row0 and col0, so no lane leaves an MFMA another lane enters.
    if row0 >= m or col0 >= n:
        return
    comptime if is_amd_gpu():
        var i16 = lane & 15
        var kq = (lane >> 4) * INT8_AMD_LANE_BYTES
        var i4 = (lane >> 4) * 4
        var acc = InlineArray[SIMD[DType.int32, 4], NACC](fill=SIMD[DType.int32, 4](0))
        for kt in range(0, k, INT8_AMD_K_TILE):
            var bfr = InlineArray[Int64, FN](fill=Int64(0))
            comptime for fq in range(FN):
                # A unit tile wholly beyond `n` is skipped below; its
                # fragment stays the zero code and is not read.
                if col0 + fq * 16 < n:
                    bfr[fq] = _word8[STATED](qb, col0 + fq * 16 + i16, kt + kq, n, k, aligned)
            comptime for fm in range(FM):
                # Uniform across the wave.
                if row0 + fm * 16 < m:
                    var a = _word8[STATED](qa, row0 + fm * 16 + i16, kt + kq, m, k, aligned)
                    comptime for fq in range(FN):
                        if col0 + fq * 16 < n:
                            acc[fm * FN + fq] = _mfma_i8(a, bfr[fq], acc[fm * FN + fq])
        # ---- THE EPILOGUE. D[i][j]: lane `j + 16 (i // 4)`, register
        # `i % 4`. A tile that was skipped holds zeros and every cell of it
        # is masked.
        comptime for fm in range(FM):
            comptime for fq in range(FN):
                var gi = row0 + fm * 16 + i4
                var gj = col0 + fq * 16 + i16
                comptime for e in range(4):
                    _store_cell_tuned(c, ea, eb, acc[fm * FN + fq][e], gi + e, gj, m, n)
    else:
        return


def _refuse_not_amd(who: String) raises:
    comptime if not INT8_AMD_AVAILABLE:
        raise Error(
            who + ": column " + column_name(TARGET_COLUMN)
            + " is not AMD; the plans of gemm_int8_mma_amd.mojo are written for"
            " the MFMA unit and a wavefront of 64"
        )


def _launch_amd_direct[
    FM: Int, FN: Int, WM: Int, WN: Int, STATED: Bool
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
    comptime kern = identical_gemm_int8_mma_amd_direct_kernel[FM, FN, WM, WN, STATED]
    var aligned = Int32(0)
    if _bases_aligned(Int(qa.unsafe_ptr()), Int(qb.unsafe_ptr()), INT8_AMD_LANE_BYTES):
        aligned = Int32(1)
    ctx.enqueue_function[kern](
        c.unsafe_ptr(),
        qa.unsafe_ptr(),
        ea.unsafe_ptr(),
        qb.unsafe_ptr(),
        eb.unsafe_ptr(),
        Int32(m),
        Int32(n),
        Int32(k),
        aligned,
        grid_dim=((n + BN - 1) // BN, (m + BM - 1) // BM, 1),
        block_dim=(WM * WN * WARP_SIZE, 1, 1),
    )


def identical_gemm_int8_mma_amd_with_plan(
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
    """One product on a NAMED plan of this file, for the gate and the timing
    harness. Refuses by name on a column that is not AMD. Asynchronous."""
    _refuse_not_amd(String("identical_gemm_int8_mma_amd"))
    comptime if INT8_AMD_AVAILABLE:
        _refuse_tuned_shape(m, n, k)
        if plan == INT8_AMD_PLAN_DIRECT_UNSTATED:
            _launch_amd_direct[1, 1, 2, 2, False](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_AMD_PLAN_DIRECT:
            _launch_amd_direct[1, 1, 2, 2, True](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_AMD_PLAN_DIRECT_ONE_WAVE:
            _launch_amd_direct[1, 1, 1, 1, True](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_AMD_PLAN_DIRECT_FRAG2:
            _launch_amd_direct[2, 2, 2, 2, True](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_AMD_PLAN_DIRECT_FRAG4:
            _launch_amd_direct[4, 4, 2, 2, True](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_AMD_PLAN_DIRECT_FRAG4_ONE_WAVE:
            _launch_amd_direct[4, 4, 1, 1, True](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_AMD_PLAN_DIRECT_FRAG2X4:
            _launch_amd_direct[2, 4, 2, 2, True](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_AMD_PLAN_DIRECT_ROW:
            _launch_amd_direct[1, 4, 1, 4, True](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_AMD_PLAN_DIRECT_ROW_ONE_WAVE:
            _launch_amd_direct[1, 4, 1, 1, True](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_AMD_PLAN_STAGED_WAVES8:
            _launch_tuned[2, 2, 2, 4, 64, 16](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_AMD_PLAN_STAGED_FRAG2X4:
            _launch_tuned[2, 4, 2, 2, 64, 16](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_AMD_PLAN_STAGED_WAVES2:
            _launch_tuned[4, 4, 1, 2, 64, 16](ctx, c, qa, ea, qb, eb, m, n, k)
        elif plan == INT8_AMD_PLAN_STAGED_FRAG2_K128:
            _launch_tuned[2, 2, 2, 2, 128, 16](ctx, c, qa, ea, qb, eb, m, n, k)
        else:
            raise Error("identical_gemm_int8_mma_amd: no plan " + String(plan))


def identical_gemm_int8_mma_amd_into(
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
    """One product on the plan `int8_amd_dispatch` names.
    `identical_gemm_int8_mma_into`'s signature and its bits. Asynchronous."""
    identical_gemm_int8_mma_amd_with_plan(
        ctx, c, qa, ea, qb, eb, m, n, k, int8_amd_dispatch(m, n, k)
    )


# ===========================================================================
# FOUR PRODUCTS, nothing staged
# ===========================================================================


def identical_gemm_int8_pieces_amd_direct_kernel[
    FM: Int, FN: Int, WM: Int, WN: Int
](
    s: MutPointer[Int32, MutAnyOrigin],
    ah: MutPointer[Int8, MutAnyOrigin],
    al: MutPointer[Int8, MutAnyOrigin],
    bh: MutPointer[Int8, MutAnyOrigin],
    bl: MutPointer[Int8, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    aligned_in: Int32,
):
    """OP_NT, the four products of two-plane operands:
    `S[i, j] = (Ah . Bh^T, Ah . Bl^T + Al . Bh^T, Al . Bl^T)[i, j]`, three
    Int32 per cell. `identical_gemm_int8_mma_amd_direct_kernel`'s block,
    waves and tiles; at every unit step a lane reads eight codes of each
    plane of each operand row, and a unit tile runs four unit steps into
    three accumulators. The same grid and block. No barrier."""
    comptime TM = INT8_TUNED_TILE * FM
    comptime TN = INT8_TUNED_TILE * FN
    comptime BM = WM * TM
    comptime BN = WN * TN
    comptime NACC = FM * FN

    comptime assert WM * WN * WARP_SIZE <= INT8_TUNED_MAX_TPB, (
        "identical_gemm_int8_pieces_amd_direct_kernel: a block is at most 1024 threads"
    )

    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var aligned = aligned_in != Int32(0)
    var warp = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(lane_id())
    var wm = warp // WN
    var wn = warp - wm * WN
    var row0 = Int(block_idx.y) * BM + wm * TM
    var col0 = Int(block_idx.x) * BN + wn * TN
    # Uniform across the wave.
    if row0 >= m or col0 >= n:
        return
    comptime if is_amd_gpu():
        var i16 = lane & 15
        var kq = (lane >> 4) * INT8_AMD_LANE_BYTES
        var i4 = (lane >> 4) * 4
        # Three accumulators per unit tile: HH, HL + LH, LL.
        var acc = InlineArray[SIMD[DType.int32, 4], 3 * NACC](
            fill=SIMD[DType.int32, 4](0)
        )
        for kt in range(0, k, INT8_AMD_K_TILE):
            # Per unit tile of B: the high plane's word, then the low's.
            var bfr = InlineArray[Int64, 2 * FN](fill=Int64(0))
            comptime for fq in range(FN):
                if col0 + fq * 16 < n:
                    bfr[fq * 2] = _word8_stated(
                        bh, col0 + fq * 16 + i16, kt + kq, n, k, aligned
                    )
                    bfr[fq * 2 + 1] = _word8_stated(
                        bl, col0 + fq * 16 + i16, kt + kq, n, k, aligned
                    )
            comptime for fm in range(FM):
                if row0 + fm * 16 < m:
                    var a_hi = _word8_stated(
                        ah, row0 + fm * 16 + i16, kt + kq, m, k, aligned
                    )
                    var a_lo = _word8_stated(
                        al, row0 + fm * 16 + i16, kt + kq, m, k, aligned
                    )
                    comptime for fq in range(FN):
                        if col0 + fq * 16 < n:
                            acc[(fm * FN + fq) * 3] = _mfma_i8(
                                a_hi, bfr[fq * 2], acc[(fm * FN + fq) * 3]
                            )
                            acc[(fm * FN + fq) * 3 + 1] = _mfma_i8(
                                a_hi, bfr[fq * 2 + 1], acc[(fm * FN + fq) * 3 + 1]
                            )
                            comptime if INT8_PIECES_SABOTAGE:
                                # SABOTAGE: HL again where LH belongs.
                                acc[(fm * FN + fq) * 3 + 1] = _mfma_i8(
                                    a_hi, bfr[fq * 2 + 1], acc[(fm * FN + fq) * 3 + 1]
                                )
                            else:
                                acc[(fm * FN + fq) * 3 + 1] = _mfma_i8(
                                    a_lo, bfr[fq * 2], acc[(fm * FN + fq) * 3 + 1]
                                )
                            acc[(fm * FN + fq) * 3 + 2] = _mfma_i8(
                                a_lo, bfr[fq * 2 + 1], acc[(fm * FN + fq) * 3 + 2]
                            )
        comptime for fm in range(FM):
            comptime for fq in range(FN):
                var gi = row0 + fm * 16 + i4
                var gj = col0 + fq * 16 + i16
                comptime for e in range(4):
                    _store_sums(
                        s,
                        acc[(fm * FN + fq) * 3][e],
                        acc[(fm * FN + fq) * 3 + 1][e],
                        acc[(fm * FN + fq) * 3 + 2][e],
                        gi + e,
                        gj,
                        m,
                        n,
                    )
    else:
        return


def _launch_amd_pieces_direct[
    FM: Int, FN: Int, WM: Int, WN: Int
](
    ctx: DeviceContext,
    mut s: DeviceBuffer[DType.int32],
    mut ah: DeviceBuffer[DType.int8],
    mut al: DeviceBuffer[DType.int8],
    mut bh: DeviceBuffer[DType.int8],
    mut bl: DeviceBuffer[DType.int8],
    m: Int,
    n: Int,
    k: Int,
) raises:
    comptime BM = WM * FM * INT8_TUNED_TILE
    comptime BN = WN * FN * INT8_TUNED_TILE
    comptime kern = identical_gemm_int8_pieces_amd_direct_kernel[FM, FN, WM, WN]
    var aligned = Int32(0)
    if _bases_aligned(
        Int(ah.unsafe_ptr()), Int(al.unsafe_ptr()), INT8_AMD_LANE_BYTES
    ) and _bases_aligned(Int(bh.unsafe_ptr()), Int(bl.unsafe_ptr()), INT8_AMD_LANE_BYTES):
        aligned = Int32(1)
    ctx.enqueue_function[kern](
        s.unsafe_ptr(),
        ah.unsafe_ptr(),
        al.unsafe_ptr(),
        bh.unsafe_ptr(),
        bl.unsafe_ptr(),
        Int32(m),
        Int32(n),
        Int32(k),
        aligned,
        grid_dim=((n + BN - 1) // BN, (m + BM - 1) // BM, 1),
        block_dim=(WM * WN * WARP_SIZE, 1, 1),
    )


def identical_gemm_int8_pieces_amd_with_plan(
    ctx: DeviceContext,
    mut s: DeviceBuffer[DType.int32],
    mut ah: DeviceBuffer[DType.int8],
    mut al: DeviceBuffer[DType.int8],
    mut bh: DeviceBuffer[DType.int8],
    mut bl: DeviceBuffer[DType.int8],
    m: Int,
    n: Int,
    k: Int,
    plan: Int,
) raises:
    """The four products on a NAMED plan of this file, for the gate and the
    timing harness. `s` holds `3 m n` Int32: cell `(i, j)`'s HH, HL + LH and
    LL at `3 (i n + j)`. Refuses by name on a column that is not AMD.
    Asynchronous."""
    _refuse_not_amd(String("identical_gemm_int8_pieces_amd"))
    comptime if INT8_AMD_AVAILABLE:
        _refuse_pieces_shape(m, n, k)
        if plan == INT8_AMD_PIECES_DIRECT:
            _launch_amd_pieces_direct[1, 1, 2, 2](ctx, s, ah, al, bh, bl, m, n, k)
        elif plan == INT8_AMD_PIECES_DIRECT_ONE_WAVE:
            _launch_amd_pieces_direct[1, 1, 1, 1](ctx, s, ah, al, bh, bl, m, n, k)
        elif plan == INT8_AMD_PIECES_DIRECT_FRAG2:
            _launch_amd_pieces_direct[2, 2, 2, 2](ctx, s, ah, al, bh, bl, m, n, k)
        elif plan == INT8_AMD_PIECES_DIRECT_FRAG2X4:
            _launch_amd_pieces_direct[2, 4, 2, 2](ctx, s, ah, al, bh, bl, m, n, k)
        elif plan == INT8_AMD_PIECES_DIRECT_ROW:
            _launch_amd_pieces_direct[1, 4, 1, 4](ctx, s, ah, al, bh, bl, m, n, k)
        elif plan == INT8_AMD_PIECES_STAGED_FRAG2:
            _launch_pieces[2, 2, 2, 2, 64, 16](ctx, s, ah, al, bh, bl, m, n, k)
        elif plan == INT8_AMD_PIECES_STAGED_FRAG2X4:
            _launch_pieces[2, 4, 2, 2, 64, 16](ctx, s, ah, al, bh, bl, m, n, k)
        elif plan == INT8_AMD_PIECES_STAGED_WAVES8:
            _launch_pieces[1, 2, 2, 4, 64, 16](ctx, s, ah, al, bh, bl, m, n, k)
        else:
            raise Error("identical_gemm_int8_pieces_amd: no plan " + String(plan))


def identical_gemm_int8_pieces_amd_into(
    ctx: DeviceContext,
    mut s: DeviceBuffer[DType.int32],
    mut ah: DeviceBuffer[DType.int8],
    mut al: DeviceBuffer[DType.int8],
    mut bh: DeviceBuffer[DType.int8],
    mut bl: DeviceBuffer[DType.int8],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """The four products on the plan `int8_amd_pieces_dispatch` names: two
    int8 planes of each operand in, three exact Int32 sums per cell out.
    `identical_gemm_int8_pieces_tuned_into`'s signature and its sums.
    Asynchronous."""
    identical_gemm_int8_pieces_amd_with_plan(
        ctx, s, ah, al, bh, bl, m, n, k, int8_amd_pieces_dispatch(m, n, k)
    )
