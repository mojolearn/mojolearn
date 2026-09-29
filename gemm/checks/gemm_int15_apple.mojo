# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`mojolearn.identical.gemm.int15i64.v1` on Apple's FLOAT matrix unit: the
APPLE-UNIT plans, in two forms, every sum exact.

    TWO      two products per step of the unit, the left operand WHOLE,
             the accumulators carried into integers after EVERY step
    FOUR     four products per step of the unit, both operands in pieces
             (the integer units' construction, clauses W-3 to W-5), the
             accumulators carried into integers every 512 steps of `k`
Both are the profile. Which costs less time is a measurement and the
timing harness runs both. What follows describes TWO first, then FOUR.

Lane lane/lowbit-int15, 2026-09-29, DEVIATION 2977. Contract clause W-12 of
`gemm/IDENTICAL_LOWBIT_CONTRACT.md`. The other plans are
`gemm/checks/gemm_int15.mojo`; the answer is
`gemm/host/gemm_int15_oracle.mojo::gemm_int15_oracle`; the gates are
`gemm/checks/gemm_int15_check.mojo`. It stands on lane/lowbit-units'
exact-chunk probe (`gemm_int8_apple_chunk.mojo`), whose gate passed on the
M2 Pro: an integer held as the float32 it names, on the unit
`PLAN_APPLE_MMA` already uses, with the sums cut before they can round.

WHY IT EXISTS. Metal has no integer matrix unit, and the one-thread-per-cell
kernels measured 12 to 14 times `fp32.v1`'s time at the training rows on an
M3 Ultra (run 1). `fp32.v1` runs the float unit there; this is the
fifteen-bit product on the same unit.

WHAT IT COMPUTES. With `a` a code of the left operand and `b = bh * 128 +
bl` a code of the right one,

    sum(a * b) = sum(a * bh) * 2^7 + sum(a * bl)

TWO products per step of the unit, not four: the LEFT operand enters whole.
That is possible here and not on the integer units because the float unit
takes a fifteen-bit integer as an operand, where an int8 unit cannot.

WHY NO FLOAT CAN ROUND (the construction argument, clause W-12)
----------------------------------------------------------------
  the operands     `|a| <= 16383`, `|bh| <= 128`, `0 <= bl <= 127`: integers
                   below 2^24, so each IS a float32. The left operand is
                   staged from its two planes as `ah * 128 + al`, a product
                   by a power of two and an addition of integers below
                   2^15: exact.
  one product      `|a * bh| <= 16383 * 128 = 2097024 < 2^21`, and
                   `|a * bl| <= 16383 * 127`: integers below 2^24, exact.
  one step         the unit multiplies 8 by 8 by 8: every cell receives 8
                   products and the accumulator ENTERS EVERY STEP AT ZERO.
                   Whatever order the unit adds in, and whether or not it
                   fuses a multiply into an add, every value it can hold is
                   a sum of at most 8 such products: at most
                   `8 * 2097024 = 16776192 < 2^24 = 16777216`. NINE would
                   reach 18873216. So the chunk is ONE step of the unit and
                   cannot be longer.
  the carry        after every step each accumulator converts to Int32
                   (exact: an integer below 2^24) and is added to an Int64
                   running sum: `|sum(a * bh)| <= 2097024 * 65536 < 2^38`.
  the result       `S = HI * 2^7 + LO` in Int64, a shift and an addition,
                   and then the epilogue every plan shares,
                   `dequant_int15_pinned`.
The flush to zero has nothing to act on (a nonzero integer is never
subnormal). The Int64 that reaches the epilogue is the Int64
`int15_dot_cell` computes.

WHAT THE ARGUMENT DOES NOT COVER. It assumes the unit computes in IEEE
float32 with a 24-bit significand at every internal step, as the int8 probe
does. That is a measurement per Apple generation, which is why the gate
plants the cases that separate it: every product ODD and at its largest
(`a = 16383`, `bh = 127`), where the ninth product would pass 2^24 on an odd
integer, the largest magnitude (`bh = -128`), and halves that cancel.

THE PADDING RULE is L-9's: a row beyond `m`, a column beyond `n` and a step
beyond `k` are staged as the ZERO CODE.

THE FORM FOUR. Both operands staged as their two planes, the three
accumulators of clause W-4 (`HH`, `HL + LH`, `LL`) as floats. A piece
product is at most `128 * 128 = 16384`, the cross term at most 32512 per
step of `k`, so inside a chunk of `S` steps every value the unit can hold is
at most `32512 * S`, and `S = 512` gives `16646144 < 2^24` (517 steps would
pass it). At a chunk end each accumulator converts to Int32 and is added to
an Int32 running sum, which clause W-4 bounds for `k <= 65536`, and
restarts at zero. Then clause W-5's recombination and the epilogue.

THE SABOTAGE ARMS.
  `-D MOJOLEARN_INT15_APPLE_CHUNK_SABOTAGE=1` removes the chunk boundary
      that keeps the sums exact: form TWO carries its accumulators across
      TWO steps of the unit, form FOUR across the whole of `k`. The planted
      odd sums then pass 2^24 and the gate must FAIL on them.
  `-D MOJOLEARN_LOWBIT_SABOTAGE=1` flips the value of every cell stored.
"""

from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from std.sys import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN, column_name
from checks.numerics_int15 import dequant_int15_pinned, int15_recombine
from gemm.checks.gemm_identical import _AMMA_M64, _amma_load_t, _amma_mma
from gemm.checks.gemm_int15_epilogue import INT15_EXPONENT_SABOTAGE
from gemm.host.gemm_int15_oracle import INT15_MAX_K
from gemm.host.gemm_oracle import gemm_oracle_sabotage_value_flip

#: The largest magnitude of one product of this plan: a whole code times a
#: high piece.
comptime INT15_APPLE_PRODUCT_MAX = 16383 * 128

#: Every integer of magnitude below this is a float32.
comptime INT15_APPLE_EXACT_BOUND = 16777216

#: Steps of `k` in one step of the unit (an 8 by 8 by 8 multiply).
comptime INT15_APPLE_UNIT_STEPS = 8

#: The window: steps staged into threadgroup memory at once. SCHEDULING.
#: 16 keeps the three staged tiles of the WIDE geometry under 15 KB of the
#: 32 KB a threadgroup has.
comptime INT15_APPLE_KB = 16

#: The two forms.
comptime INT15_APPLE_FORM_TWO = 2
comptime INT15_APPLE_FORM_FOUR = 4

#: Form FOUR: the largest magnitude one step of `k` adds to an accumulator
#: (the cross term, `HL + LH`), and the steps of `k` in a chunk.
comptime INT15_APPLE4_STEP_MAX = 2 * 128 * 127
comptime INT15_APPLE4_CHUNK_STEPS = 512

#: The arm that carries the accumulators across two steps of the unit.
comptime INT15_APPLE_CHUNK_SABOTAGE = is_defined["MOJOLEARN_INT15_APPLE_CHUNK_SABOTAGE"]()

#: DEVIATION 2973, the value arm, the define every fifteen-bit plan reads.
comptime INT15_APPLE_VALUE_SABOTAGE = is_defined["MOJOLEARN_LOWBIT_SABOTAGE"]()

#: Whether this build can run the plan at all: the Metal column only.
comptime INT15_APPLE_AVAILABLE = TARGET_COLUMN == COLUMN_APPLE

#: The two tile geometries, lane/lowbit-units': WIDE owns a 64 x 64 output
#: tile (2 x 2 simdgroups of 4 x 4 fragments); ROW owns 8 x 128 (1 x 4
#: simdgroups of 1 x 4 fragments), for the decode rows. SCHEDULING: the gate
#: runs both on every shape.
comptime INT15_APPLE_GEOMETRY_WIDE = 0
comptime INT15_APPLE_GEOMETRY_ROW = 1

#: Outputs of at most this many rows take the ROW geometry.
comptime INT15_APPLE_ROW_MAX_M = 8


def int15_apple_sabotage_name() -> String:
    comptime if INT15_APPLE_CHUNK_SABOTAGE:
        return String("INT15_APPLE_CHUNK_BOUNDARY_REMOVED")
    else:
        return String("none")


def int15_apple_geometry(m: Int) -> Int:
    """The geometry the launcher picks. Reads `m` and may: the tile is a
    schedule and every cell's Int64 is the same on either."""
    if m <= INT15_APPLE_ROW_MAX_M:
        return INT15_APPLE_GEOMETRY_ROW
    return INT15_APPLE_GEOMETRY_WIDE


def int15_apple_geometry_name(geometry: Int) -> String:
    if geometry == INT15_APPLE_GEOMETRY_ROW:
        return String("ROW 8x128 (1x4 simdgroups, 1x4 fragments)")
    return String("WIDE 64x64 (2x2 simdgroups, 4x4 fragments)")


@always_inline
def _stage_left[
    ROWS: Int, KB: Int, NT: Int, ST: Int
](
    dst: UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    hi: MutPointer[Int8, MutAnyOrigin],
    lo: MutPointer[Int8, MutAnyOrigin],
    row0: Int,
    rows: Int,
    k0: Int,
    k: Int,
    tid: Int,
):
    """One window of the LEFT operand into threadgroup memory, each code
    whole, as the float32 it names: `hi * 128 + lo`, exact. Element (row r,
    step p) at `dst[p * ST + r]`. A slot is four consecutive steps of one
    row; a thread owns slots `tid, tid + NT, ...`. A row at or beyond
    `rows` and a step at or beyond `k` are the ZERO CODE."""
    comptime SLOTS = (ROWS * KB) // 4
    comptime SL = (SLOTS + NT - 1) // NT
    comptime assert KB % 4 == 0, "_stage_left: a window is whole slots"
    comptime for sl in range(SL):
        var s = sl * NT + tid
        if s < SLOTS:
            var r = s // (KB // 4)
            var p4 = (s % (KB // 4)) * 4
            var gr = row0 + r
            comptime for e in range(4):
                var v = Float32(0.0)
                if gr < rows and k0 + p4 + e < k:
                    var at_ = gr * k + k0 + p4 + e
                    v = Float32(Int32(hi.unsafe_load(at_))) * Float32(128.0) + Float32(
                        Int32(lo.unsafe_load(at_))
                    )
                dst[(p4 + e) * ST + r] = v


@always_inline
def _stage_piece_left[
    ROWS: Int, KB: Int, NT: Int, ST: Int
](
    dst: UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    q: MutPointer[Int8, MutAnyOrigin],
    row0: Int,
    rows: Int,
    k0: Int,
    k: Int,
    tid: Int,
):
    """One window of ONE PLANE of the left operand (form FOUR), each piece
    as the float32 it names, in `_stage_left`'s layout: element (row r,
    step p) at `dst[p * ST + r]`."""
    comptime SLOTS = (ROWS * KB) // 4
    comptime SL = (SLOTS + NT - 1) // NT
    comptime assert KB % 4 == 0, "_stage_piece_left: a window is whole slots"
    comptime for sl in range(SL):
        var s = sl * NT + tid
        if s < SLOTS:
            var r = s // (KB // 4)
            var p4 = (s % (KB // 4)) * 4
            var gr = row0 + r
            comptime for e in range(4):
                var v = Float32(0.0)
                if gr < rows and k0 + p4 + e < k:
                    v = Float32(Int32(q.unsafe_load(gr * k + k0 + p4 + e)))
                dst[(p4 + e) * ST + r] = v


@always_inline
def _stage_right[
    ROWS: Int, KB: Int, NT: Int, ST: Int
](
    dst: UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    q: MutPointer[Int8, MutAnyOrigin],
    row0: Int,
    rows: Int,
    k0: Int,
    k: Int,
    tid: Int,
):
    """One window of ONE PLANE of the right operand, each piece as the
    float32 it names. Element (row r, step p) at `dst[r * ST + p]`. The
    padding rule and the slots of `_stage_left`."""
    comptime SLOTS = (ROWS * KB) // 4
    comptime SL = (SLOTS + NT - 1) // NT
    comptime assert KB % 4 == 0, "_stage_right: a window is whole slots"
    comptime for sl in range(SL):
        var s = sl * NT + tid
        if s < SLOTS:
            var r = s // (KB // 4)
            var p4 = (s % (KB // 4)) * 4
            var gr = row0 + r
            comptime for e in range(4):
                var v = Float32(0.0)
                if gr < rows and k0 + p4 + e < k:
                    v = Float32(Int32(q.unsafe_load(gr * k + k0 + p4 + e)))
                dst[r * ST + p4 + e] = v


def identical_gemm_int15_apple_kernel[
    SGM: Int, SGN: Int, FM: Int, FN: Int, KB: Int
](
    c: MutPointer[Float32, MutAnyOrigin],
    ah: MutPointer[Int8, MutAnyOrigin],
    al: MutPointer[Int8, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    bh: MutPointer[Int8, MutAnyOrigin],
    bl: MutPointer[Int8, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
):
    """OP_NT, `C[m x n] = Qa[m x k] . Qb[n x k]^T`. One block owns a
    `BM x BN` output tile; each simdgroup owns `FM x FN` 8x8 fragments, each
    lane two cells of each (`identical_gemm_apple_mma_kernel`'s layout).

    Per window of `KB` steps the block stages the left operand whole and
    the two planes of the right one as float32, then every simdgroup takes
    the window one step of the unit at a time: two multiplies per fragment
    from a ZERO accumulator, each converted to Int32 and added to its Int64
    running sum before the next step. The epilogue is every plan's."""
    comptime NSG = SGM * SGN
    comptime NT = NSG * 32
    comptime BM = 8 * FM * SGM
    comptime BN = 8 * FN * SGN
    comptime AST = BM + 4
    comptime BST = KB + 4
    comptime NF = FM * FN
    comptime NC = 2 * NF
    comptime assert KB % INT15_APPLE_UNIT_STEPS == 0, "the window is whole steps of the unit"
    comptime assert (
        INT15_APPLE_PRODUCT_MAX * INT15_APPLE_UNIT_STEPS < INT15_APPLE_EXACT_BOUND
    ), "a step's largest partial sum must be a float32"
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
    var bth = stack_allocation[BN * BST, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var btl = stack_allocation[BN * BST, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var hi_total = InlineArray[Int64, NC](fill=Int64(0))
    var lo_total = InlineArray[Int64, NC](fill=Int64(0))
    var hi_acc = InlineArray[_AMMA_M64, NF](fill=_AMMA_M64(0))
    var lo_acc = InlineArray[_AMMA_M64, NF](fill=_AMMA_M64(0))
    var windows = (k + KB - 1) // KB
    var step = 0
    for w in range(windows):
        var k0 = w * KB
        _stage_left[BM, KB, NT, AST](at, ah, al, m0, m, k0, k, tid)
        _stage_right[BN, KB, NT, BST](bth, bh, n0, n, k0, k, tid)
        _stage_right[BN, KB, NT, BST](btl, bl, n0, n, k0, k, tid)
        barrier()
        comptime for p8 in range(KB // 8):
            var af = InlineArray[_AMMA_M64, FM](fill=_AMMA_M64(0))
            var bhf = InlineArray[_AMMA_M64, FN](fill=_AMMA_M64(0))
            var blf = InlineArray[_AMMA_M64, FN](fill=_AMMA_M64(0))
            comptime for fm in range(FM):
                af[fm] = _amma_load_t(at + (8 * p8) * AST + (sgm * FM + fm) * 8, AST)
            comptime for fq in range(FN):
                bhf[fq] = _amma_load_t(bth + ((sgn * FN + fq) * 8) * BST + 8 * p8, BST)
                blf[fq] = _amma_load_t(btl + ((sgn * FN + fq) * 8) * BST + 8 * p8, BST)
            comptime for fm in range(FM):
                comptime for fq in range(FN):
                    hi_acc[fm * FN + fq] = _amma_mma(af[fm], bhf[fq], hi_acc[fm * FN + fq])
                    lo_acc[fm * FN + fq] = _amma_mma(af[fm], blf[fq], lo_acc[fm * FN + fq])
            step += 1
            # THE CHUNK IS ONE STEP OF THE UNIT: convert, carry, restart.
            var chunk_end = True
            comptime if INT15_APPLE_CHUNK_SABOTAGE:
                # THE DEFECT ARM: the accumulators live across two steps.
                chunk_end = (step % 2) == 0 or (w + 1 == windows and p8 == KB // 8 - 1)
            if chunk_end:
                comptime for f in range(NF):
                    comptime for e in range(2):
                        hi_total[2 * f + e] += Int64(hi_acc[f][e].cast[DType.int32]())
                        lo_total[2 * f + e] += Int64(lo_acc[f][e].cast[DType.int32]())
                    hi_acc[f] = _AMMA_M64(0)
                    lo_acc[f] = _AMMA_M64(0)
        barrier()
    comptime for fm in range(FM):
        comptime for fq in range(FN):
            comptime for e in range(2):
                var gi = m0 + (sgm * FM + fm) * 8 + frow
                var gj = n0 + (sgn * FN + fq) * 8 + fcol + e
                if gi < m and gj < n:
                    var at_cell = 2 * (fm * FN + fq) + e
                    var s = (hi_total[at_cell] << Int64(7)) + lo_total[at_cell]
                    var gjb = gj
                    comptime if INT15_EXPONENT_SABOTAGE:
                        gjb = gi % n
                    var out = dequant_int15_pinned(
                        s, Int(ea.unsafe_load(gi)) + Int(eb.unsafe_load(gjb))
                    )
                    comptime if INT15_APPLE_VALUE_SABOTAGE:
                        out = gemm_oracle_sabotage_value_flip(out)
                    c.unsafe_store(gi * n + gj, out)


def identical_gemm_int15_apple4_kernel[
    SGM: Int, SGN: Int, FM: Int, FN: Int, KB: Int
](
    c: MutPointer[Float32, MutAnyOrigin],
    ah: MutPointer[Int8, MutAnyOrigin],
    al: MutPointer[Int8, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    bh: MutPointer[Int8, MutAnyOrigin],
    bl: MutPointer[Int8, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
):
    """THE FORM FOUR. The layout of `identical_gemm_int15_apple_kernel`;
    four staged planes, four multiplies per fragment and step of the unit
    into three float accumulators, carried into Int32 every
    `INT15_APPLE4_CHUNK_STEPS` steps of `k` and at the end of `k`."""
    comptime NSG = SGM * SGN
    comptime NT = NSG * 32
    comptime BM = 8 * FM * SGM
    comptime BN = 8 * FN * SGN
    comptime AST = BM + 4
    comptime BST = KB + 4
    comptime NF = FM * FN
    comptime NC = 2 * NF
    comptime CHUNK_WINDOWS = INT15_APPLE4_CHUNK_STEPS // KB
    comptime assert KB % INT15_APPLE_UNIT_STEPS == 0, "the window is whole steps of the unit"
    comptime assert CHUNK_WINDOWS * KB == INT15_APPLE4_CHUNK_STEPS, "a chunk is whole windows"
    comptime assert (
        INT15_APPLE4_STEP_MAX * INT15_APPLE4_CHUNK_STEPS < INT15_APPLE_EXACT_BOUND
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
    var ath = stack_allocation[KB * AST, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var atl = stack_allocation[KB * AST, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var bth = stack_allocation[BN * BST, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var btl = stack_allocation[BN * BST, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var hh_total = SIMD[DType.int32, NC](0)
    var mid_total = SIMD[DType.int32, NC](0)
    var ll_total = SIMD[DType.int32, NC](0)
    var hh_acc = InlineArray[_AMMA_M64, NF](fill=_AMMA_M64(0))
    var mid_acc = InlineArray[_AMMA_M64, NF](fill=_AMMA_M64(0))
    var ll_acc = InlineArray[_AMMA_M64, NF](fill=_AMMA_M64(0))
    var windows = (k + KB - 1) // KB
    for w in range(windows):
        var k0 = w * KB
        _stage_piece_left[BM, KB, NT, AST](ath, ah, m0, m, k0, k, tid)
        _stage_piece_left[BM, KB, NT, AST](atl, al, m0, m, k0, k, tid)
        _stage_right[BN, KB, NT, BST](bth, bh, n0, n, k0, k, tid)
        _stage_right[BN, KB, NT, BST](btl, bl, n0, n, k0, k, tid)
        barrier()
        comptime for p8 in range(KB // 8):
            var ahf = InlineArray[_AMMA_M64, FM](fill=_AMMA_M64(0))
            var alf = InlineArray[_AMMA_M64, FM](fill=_AMMA_M64(0))
            var bhf = InlineArray[_AMMA_M64, FN](fill=_AMMA_M64(0))
            var blf = InlineArray[_AMMA_M64, FN](fill=_AMMA_M64(0))
            comptime for fm in range(FM):
                ahf[fm] = _amma_load_t(ath + (8 * p8) * AST + (sgm * FM + fm) * 8, AST)
                alf[fm] = _amma_load_t(atl + (8 * p8) * AST + (sgm * FM + fm) * 8, AST)
            comptime for fq in range(FN):
                bhf[fq] = _amma_load_t(bth + ((sgn * FN + fq) * 8) * BST + 8 * p8, BST)
                blf[fq] = _amma_load_t(btl + ((sgn * FN + fq) * 8) * BST + 8 * p8, BST)
            comptime for fm in range(FM):
                comptime for fq in range(FN):
                    hh_acc[fm * FN + fq] = _amma_mma(ahf[fm], bhf[fq], hh_acc[fm * FN + fq])
                    mid_acc[fm * FN + fq] = _amma_mma(ahf[fm], blf[fq], mid_acc[fm * FN + fq])
                    mid_acc[fm * FN + fq] = _amma_mma(alf[fm], bhf[fq], mid_acc[fm * FN + fq])
                    ll_acc[fm * FN + fq] = _amma_mma(alf[fm], blf[fq], ll_acc[fm * FN + fq])
        barrier()
        var chunk_end = w + 1 == windows
        comptime if not INT15_APPLE_CHUNK_SABOTAGE:
            chunk_end = chunk_end or (w + 1) % CHUNK_WINDOWS == 0
        if chunk_end:
            comptime for f in range(NF):
                comptime for e in range(2):
                    hh_total[2 * f + e] += hh_acc[f][e].cast[DType.int32]()
                    mid_total[2 * f + e] += mid_acc[f][e].cast[DType.int32]()
                    ll_total[2 * f + e] += ll_acc[f][e].cast[DType.int32]()
                hh_acc[f] = _AMMA_M64(0)
                mid_acc[f] = _AMMA_M64(0)
                ll_acc[f] = _AMMA_M64(0)
    comptime for fm in range(FM):
        comptime for fq in range(FN):
            comptime for e in range(2):
                var gi = m0 + (sgm * FM + fm) * 8 + frow
                var gj = n0 + (sgn * FN + fq) * 8 + fcol + e
                if gi < m and gj < n:
                    var at_cell = 2 * (fm * FN + fq) + e
                    var gjb = gj
                    comptime if INT15_EXPONENT_SABOTAGE:
                        gjb = gi % n
                    var out = dequant_int15_pinned(
                        int15_recombine(hh_total[at_cell], mid_total[at_cell], ll_total[at_cell]),
                        Int(ea.unsafe_load(gi)) + Int(eb.unsafe_load(gjb)),
                    )
                    comptime if INT15_APPLE_VALUE_SABOTAGE:
                        out = gemm_oracle_sabotage_value_flip(out)
                    c.unsafe_store(gi * n + gj, out)


def identical_gemm_int15_apple_with_geometry(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut ah: DeviceBuffer[DType.int8],
    mut al: DeviceBuffer[DType.int8],
    mut ea: DeviceBuffer[DType.int32],
    mut bh: DeviceBuffer[DType.int8],
    mut bl: DeviceBuffer[DType.int8],
    mut eb: DeviceBuffer[DType.int32],
    m: Int,
    n: Int,
    k: Int,
    geometry: Int,
    form: Int = INT15_APPLE_FORM_TWO,
) raises:
    """The plan on a NAMED geometry and form. The gate calls this to
    compare them. Refuses by name on a column that is not Apple.
    Asynchronous."""
    comptime if not INT15_APPLE_AVAILABLE:
        raise Error(
            "identical_gemm_int15_apple: column " + column_name(TARGET_COLUMN)
            + " is not Apple; this plan runs on Metal's float matrix unit only"
        )
    else:
        if m <= 0 or n <= 0 or k <= 0 or k > INT15_MAX_K:
            raise Error(
                "identical_gemm_int15_apple: m, n and k must be positive and"
                " k at most " + String(INT15_MAX_K) + " (contract W-4), got m="
                + String(m) + " n=" + String(n) + " k=" + String(k)
            )
        if form != INT15_APPLE_FORM_TWO and form != INT15_APPLE_FORM_FOUR:
            raise Error("identical_gemm_int15_apple: form must be 2 or 4, got " + String(form))
        if geometry != INT15_APPLE_GEOMETRY_WIDE and geometry != INT15_APPLE_GEOMETRY_ROW:
            raise Error(
                "identical_gemm_int15_apple: geometry must be 0 (WIDE) or 1"
                " (ROW), got " + String(geometry)
            )
        if form == INT15_APPLE_FORM_FOUR:
            if geometry == INT15_APPLE_GEOMETRY_ROW:
                comptime kern4_row = identical_gemm_int15_apple4_kernel[1, 4, 1, 4, INT15_APPLE_KB]
                ctx.enqueue_function[kern4_row](
                    c.unsafe_ptr(),
                    ah.unsafe_ptr(),
                    al.unsafe_ptr(),
                    ea.unsafe_ptr(),
                    bh.unsafe_ptr(),
                    bl.unsafe_ptr(),
                    eb.unsafe_ptr(),
                    Int32(m),
                    Int32(n),
                    Int32(k),
                    grid_dim=(((m + 7) // 8) * ((n + 127) // 128), 1, 1),
                    block_dim=(128, 1, 1),
                )
                return
            comptime kern4_wide = identical_gemm_int15_apple4_kernel[2, 2, 4, 4, INT15_APPLE_KB]
            ctx.enqueue_function[kern4_wide](
                c.unsafe_ptr(),
                ah.unsafe_ptr(),
                al.unsafe_ptr(),
                ea.unsafe_ptr(),
                bh.unsafe_ptr(),
                bl.unsafe_ptr(),
                eb.unsafe_ptr(),
                Int32(m),
                Int32(n),
                Int32(k),
                grid_dim=(((m + 63) // 64) * ((n + 63) // 64), 1, 1),
                block_dim=(128, 1, 1),
            )
            return
        if geometry == INT15_APPLE_GEOMETRY_ROW:
            comptime kern_row = identical_gemm_int15_apple_kernel[1, 4, 1, 4, INT15_APPLE_KB]
            ctx.enqueue_function[kern_row](
                c.unsafe_ptr(),
                ah.unsafe_ptr(),
                al.unsafe_ptr(),
                ea.unsafe_ptr(),
                bh.unsafe_ptr(),
                bl.unsafe_ptr(),
                eb.unsafe_ptr(),
                Int32(m),
                Int32(n),
                Int32(k),
                grid_dim=(((m + 7) // 8) * ((n + 127) // 128), 1, 1),
                block_dim=(128, 1, 1),
            )
            return
        comptime kern_wide = identical_gemm_int15_apple_kernel[2, 2, 4, 4, INT15_APPLE_KB]
        ctx.enqueue_function[kern_wide](
            c.unsafe_ptr(),
            ah.unsafe_ptr(),
            al.unsafe_ptr(),
            ea.unsafe_ptr(),
            bh.unsafe_ptr(),
            bl.unsafe_ptr(),
            eb.unsafe_ptr(),
            Int32(m),
            Int32(n),
            Int32(k),
            grid_dim=(((m + 63) // 64) * ((n + 63) // 64), 1, 1),
            block_dim=(128, 1, 1),
        )


def identical_gemm_int15_apple_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut ah: DeviceBuffer[DType.int8],
    mut al: DeviceBuffer[DType.int8],
    mut ea: DeviceBuffer[DType.int32],
    mut bh: DeviceBuffer[DType.int8],
    mut bl: DeviceBuffer[DType.int8],
    mut eb: DeviceBuffer[DType.int32],
    m: Int,
    n: Int,
    k: Int,
    form: Int = INT15_APPLE_FORM_TWO,
) raises:
    """The plan on the geometry `int15_apple_geometry` picks, in the form
    named. Asynchronous."""
    identical_gemm_int15_apple_with_geometry(
        ctx, c, ah, al, ea, bh, bl, eb, m, n, k, int15_apple_geometry(m), form
    )
