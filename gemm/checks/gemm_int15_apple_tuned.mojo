# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`mojolearn.identical.gemm.int15i64.v1` on Apple's FLOAT matrix unit, the
TUNED plans: one kernel, its schedule and its form named at compile time,
every sum exact.

Lane lane/lowbit-apple-tuned, 2026-09-29. It stands on
`gemm/checks/gemm_int15_apple.mojo` (lane/lowbit-int15, clause W-12: the
forms TWO and FOUR) and on `gemm/checks/gemm_int8_apple_chunk.mojo`
(lane/lowbit-units: the staging that takes four codes in one load). The
answer is `gemm/host/gemm_int15_oracle.mojo::gemm_int15_oracle`; the gate is
`gemm/checks/gemm_int15_apple_tuned_check.mojo`; the clock is
`bench/gemm_int15_apple_tuned_price_main.mojo`. NOT DISPATCHED: nothing a
caller gets by default moves.

WHAT IS A LEVER HERE AND WHAT IS NOT. The tile (`SGM x SGN` simdgroups of
`FM x FN` fragments), the window `KB`, the staging load and the slices of a
launch are SCHEDULING: an integer sum is the same integer under every
order, tile and schedule (contract clause L-9). The FORM is arithmetic, and
each form carries its own proof that no float can round.

THE FORMS
---------
With `a = ah * 128 + al` and `b = bh * 128 + bl` (clause W-3: `ah`, `bh` in
`[-128, 127]`, `al`, `bl` in `[0, 127]`), clause W-5 wants

    S = HH * 2^14 + MID * 2^7 + LL,   MID = HL + LH

  FOUR    lane/lowbit-int15's: four products per step of the unit into the
          three float accumulators `HH`, `MID`, `LL`.
  THREE   three products per step of the unit: `HH`, `LL` and
          `PP = sum((ah + al) * (bh + bl))`, and then, in integers,
          `MID = PP - HH - LL`. An int8 unit cannot take `ah + al` (it
          reaches 254); a float unit takes any integer below 2^24. The
          brief held this identity in reserve for the integer units at
          twelve-bit codes; on a float unit it carries all fifteen bits.
          With `REGSUM` the two sums are formed in registers from the
          fragments of the pieces (one float addition a cell, exact)
          and only two planes an operand are staged.
  TWO     lane/lowbit-int15's arithmetic (clause W-12: the left operand
          WHOLE, `sum(a * b) = sum(a * bh) * 2^7 + sum(a * bl)`), with
          another carry: no Int64 inside the `k` loop.

WHY NO FLOAT CAN ROUND (the construction argument of each form)
---------------------------------------------------------------
Every operand is an integer of magnitude at most 254 and so IS a float32.
Inside one chunk of `S` steps of `k` every accumulator enters at zero, so
every value the unit can hold, whatever order it adds in and whether or not
it fuses a multiply into an add, is a sum of at most `S` terms of one
accumulator, and its magnitude is at most `S` times the largest term.

  FOUR    the largest term is the cross term's, `|ah * bl + al * bh| <=
          2 * 128 * 127 = 32512`. `S = 512`: `32512 * 512 = 16646144 <
          2^24 = 16777216`. 516 steps is the last that holds
          (`32512 * 516 = 16776192`), 517 passes it.
  THREE   `ah + al` lies in `[-127, 254]`: the largest is the code 16383,
          `(127, 127)`; the smallest is -127, at `(-128, 1)` and at
          `(-127, 0)`. So `|(ah + al) * (bh + bl)| <= 254 * 254 = 64516`,
          and that is the largest term of the three accumulators (`HH` has
          16384, `LL` 16129). `S = 256`: `64516 * 256 = 16516096 < 2^24`.
          260 steps is the last that holds (`64516 * 260 = 16774160`), 261
          passes it.

  TWO     `|a * bh| <= 16383 * 128 = 2097024`, the largest term. `S = 8`,
          one step of the unit: `2097024 * 8 = 16776192 < 2^24`; nine
          would reach 18873216. THE CARRY OF TWO, after every step of the
          unit: `th` and `tl`, the two accumulators as Int32 (exact), then
          `t = th * 2^7 + tl`, which is the sum of the step's 8 code
          products and so at most `8 * 16383^2 = 2147221512 < 2^31`
          (`th * 2^7` alone is at most `16776192 * 128 = 2147352576 <
          2^31`); then `t` is cut in two, `t = (t >> 16) * 2^16 + (t &
          65535)`, an arithmetic shift and a mask, and the halves are
          added to two Int32 running sums. Over the 8192 steps of the
          unit that `k = 65536` has, the low sum is at most `65535 * 8192
          < 2^29` and the high one at most `2^15 * 8192 = 2^28` in
          magnitude. The epilogue forms `S = HIGH * 2^16 + LOW` in Int64,
          once a cell.
          THE DEFERRED CARRY (f2d): after every step of the unit only
          the two conversions and two Int32 additions, `RH += th` and `RL
          += tl`; every 64 steps of the unit (512 of `k`) and at the end
          of `k` the runs are cut in 16-bit halves and flushed, `HIGH +=
          (RH >> 16) * 2^7 + (RL >> 16)`, `LOW += (RH & 65535) * 2^7 + (RL
          & 65535)`, and restarted. `|RH| <= 64 * 16776192 = 1073676288 <
          2^31` (and `RL` less). A flush adds at most `2^14 * 2^7 + 2^14`
          to `HIGH` and `65535 * 129` to `LOW`; `k = 65536` has 128
          flushes and the last one, so `|HIGH| < 2^29` and `LOW < 2^31`.
          The same epilogue: `S = HIGH * 2^16 + LOW`.

In THREE and FOUR, at a chunk end each accumulator converts to Int32
(exact: an integer below 2^24). THREE forms `MID`'s share of the chunk there, `PP - HH - LL`, three
integers below 2^24, no overflow. Each is added to an Int32 running sum,
which clause W-4 bounds for `k <= 65536`: the running sums are `HH`, `MID`
and `LL` themselves in both forms, never `PP`, whose sum over 65536 steps
would pass 2^31. Then clause W-5's recombination and the epilogue every
plan shares, `dequant_int15_pinned`.

WHAT THE ARGUMENT DOES NOT COVER. It assumes the unit computes in IEEE
float32 with a 24-bit significand at every internal step. That is a
measurement per Apple generation, which is why the gate plants, at every
chunk boundary, the cases that separate it.

THE PADDING RULE is L-9's: a row beyond `m`, a column beyond `n` and a step
beyond `k` are staged as the ZERO CODE in every plane, the sum plane
included.

THE SLICES OF A LAUNCH. macOS aborts a command buffer that holds the GPU
for seconds and leaves the output partly written (the brief, "macOS ABORTS
A LONG METAL LAUNCH SILENTLY"). A launch here covers whole rows of tiles
and at most `slice_macs` multiply-accumulates of the PROFILE's product
(`m * n * k`, not counted per unit product), with a wait between two
slices. The caller waits after the last.

THE SABOTAGE ARMS.
  `-D MOJOLEARN_INT15_APPLE_TUNED_CHUNK_SABOTAGE=1` removes the chunk
      boundary: the float accumulators of THREE and FOUR run the whole of
      `k`, those of TWO run two steps of the unit. The planted sums then
      pass 2^24 on odd integers and the gate must FAIL on them.
  `-D MOJOLEARN_LOWBIT_SABOTAGE=1` flips the value of every cell stored
      (DEVIATION 2973, the define every fifteen-bit plan reads).
"""

from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from std.sys import is_defined
from std.gpu import MAX_THREADS_PER_BLOCK_METADATA
from std.utils import StaticTuple
from max.gpu.host import Attribute, DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN, column_name
from checks.numerics_int15 import dequant_int15_pinned, int15_recombine
from gemm.checks.gemm_identical import _AMMA_M64, _amma_load_t, _amma_mma
from gemm.host.gemm_int15_oracle import INT15_MAX_K
from gemm.host.gemm_oracle import gemm_oracle_sabotage_value_flip

#: Every integer of magnitude below this is a float32.
comptime INT15_TUNED_EXACT_BOUND = 16777216

#: The three forms.
comptime INT15_TUNED_FORM_TWO = 2
comptime INT15_TUNED_FORM_THREE = 3
comptime INT15_TUNED_FORM_FOUR = 4

#: Form TWO: the largest magnitude of one product (a whole code times a
#: high piece), and the steps of `k` in a chunk: one step of the unit.
comptime INT15_TUNED2_STEP_MAX = 16383 * 128
comptime INT15_TUNED2_CHUNK_STEPS = 8

#: Form TWO, the DEFERRED carry: the steps of the unit between two flushes
#: of the Int32 runs into the halves (the header has the bounds).
comptime INT15_TUNED2D_FLUSH_STEPS = 64

#: What `_stage_planes` writes: the two planes, the two planes and their
#: sums, or the whole code alone.
comptime STAGE_PAIR = 0
comptime STAGE_PAIR_SUM = 1
comptime STAGE_WHOLE = 2

#: Form FOUR: the largest magnitude one step of `k` adds to an accumulator
#: (the cross term), and the steps of `k` in a chunk.
comptime INT15_TUNED4_STEP_MAX = 2 * 128 * 127
comptime INT15_TUNED4_CHUNK_STEPS = 512

#: Form THREE: the largest magnitude one step of `k` adds to an accumulator
#: (`(ah + al) * (bh + bl)` at the code 16383 on both sides), and the steps
#: of `k` in a chunk.
comptime INT15_TUNED3_STEP_MAX = 254 * 254
comptime INT15_TUNED3_CHUNK_STEPS = 256

#: Bytes of threadgroup memory one block may stage into.
comptime INT15_TUNED_SHARED_BYTES = 32768

#: The default slice of a launch, in multiply-accumulates of the profile's
#: product. SCHEDULING.
comptime INT15_TUNED_SLICE_MACS = 4_294_967_296

#: The arm that removes the chunk boundary.
comptime INT15_TUNED_CHUNK_SABOTAGE = is_defined["MOJOLEARN_INT15_APPLE_TUNED_CHUNK_SABOTAGE"]()

#: DEVIATION 2973, the value arm.
comptime INT15_TUNED_VALUE_SABOTAGE = is_defined["MOJOLEARN_LOWBIT_SABOTAGE"]()

#: Whether this build can run the plans at all: the Metal column only.
comptime INT15_TUNED_AVAILABLE = TARGET_COLUMN == COLUMN_APPLE

#: THE VARIANTS: a form and a schedule each, named. The gate runs every one
#: on every case; the clock times every one at every row. Job 1 (19ad540d3)
#: measured the first thirteen on the M2 Pro: the 64 x 64 tiles ran 2.3 to
#: 3.0 times fp32.v1 and the 32 x 32 tiles 1.5 to 1.7, and the 512-thread
#: tiles never ran there (every poison survived), so this list keeps the
#: 32 x 32 tile as the start line and tries the smaller ones.
comptime TUNED_F4_T32_KB32 = 0
comptime TUNED_F4_T32_KB16 = 1
comptime TUNED_F3_T32_KB16 = 2
comptime TUNED_F3R_T32_KB16 = 3
comptime TUNED_F3R_T32_KB32 = 4
comptime TUNED_F2_T32_KB16 = 5
comptime TUNED_F2_T32_KB32 = 6
comptime TUNED_F2D_T32_KB16 = 7
comptime TUNED_F2D_T32_KB32 = 8
comptime TUNED_F2_W64_KB16 = 9
comptime TUNED_F4_T16_KB32 = 10
comptime TUNED_F3R_T16_KB32 = 11
comptime TUNED_F2D_T16_KB32 = 12
comptime TUNED_F4_T32X16_KB32 = 13
comptime TUNED_F2D_T32X16_KB32 = 14
comptime TUNED_F4_T32_KB32_B128 = 15
comptime TUNED_F2D_T32_KB32_B128 = 16
comptime TUNED_F4_W64_KB16_B128 = 17
comptime TUNED_F2D_W64_KB16_B128 = 18
comptime TUNED_F4_ROW_KB16 = 19
comptime TUNED_F3_ROW64_KB16 = 20
comptime TUNED_F2D_ROW64_KB16 = 21
comptime TUNED_F2_T32_KB8 = 22
comptime TUNED_F2_T32X64_KB16 = 23
comptime TUNED_F2_T64X32_KB16 = 24
comptime TUNED_F2_SG8_KB16 = 25
comptime TUNED_F2_T16_KB16 = 26
comptime TUNED_F2_T32_KB16_B128 = 27
comptime TUNED_F2_ROW64_KB16 = 28
comptime TUNED_VARIANT_COUNT = 29


def int15_apple_tuned_sabotage_name() -> String:
    comptime if INT15_TUNED_CHUNK_SABOTAGE:
        return String("INT15_APPLE_TUNED_CHUNK_BOUNDARY_REMOVED")
    elif INT15_TUNED_VALUE_SABOTAGE:
        return String("LOWBIT_VALUE_FLIP")
    else:
        return String("none")


def int15_apple_tuned_variant_name(variant: Int) -> String:
    """`<form>.<tile>.<window>`: f4, f3 and f2 are the forms FOUR, THREE and
    TWO; f3r is THREE with the sums formed in registers; f2d is TWO with the
    DEFERRED carry (the header). t32 a 32 x 32 tile of 2 x 2 simdgroups with
    2 x 2 fragments; t16 a 16 x 16 tile of 2 x 2 simdgroups with one
    fragment each; t32x16 a 32 x 16 tile of 2 x 2 simdgroups with 2 x 1
    fragments; w64 a 64 x 64 tile of 2 x 2 simdgroups with 4 x 4 fragments;
    row an 8 x 128 tile and row64 an 8 x 64 tile of 1 x 4 simdgroups, for
    the decode rows; t32x64 32 x 64 (2 x 4 fragments), t64x32 64 x 32
    (4 x 2); sg8 4 x 2 simdgroups of 2 x 2 fragments (64 x 32, 256
    threads); kb8 a window of one step of the unit. Every other tile has
    128 threads. `.b128` declares that
    launch bound to the compiler (`identical_gemm_int15_apple_tuned_kernel_b128`)."""
    if variant == TUNED_F4_T32_KB32:
        return String("f4.t32.kb32")
    if variant == TUNED_F4_T32_KB16:
        return String("f4.t32.kb16")
    if variant == TUNED_F3_T32_KB16:
        return String("f3.t32.kb16")
    if variant == TUNED_F3R_T32_KB16:
        return String("f3r.t32.kb16")
    if variant == TUNED_F3R_T32_KB32:
        return String("f3r.t32.kb32")
    if variant == TUNED_F2_T32_KB16:
        return String("f2.t32.kb16")
    if variant == TUNED_F2_T32_KB32:
        return String("f2.t32.kb32")
    if variant == TUNED_F2D_T32_KB16:
        return String("f2d.t32.kb16")
    if variant == TUNED_F2D_T32_KB32:
        return String("f2d.t32.kb32")
    if variant == TUNED_F2_W64_KB16:
        return String("f2.w64.kb16")
    if variant == TUNED_F4_T16_KB32:
        return String("f4.t16.kb32")
    if variant == TUNED_F3R_T16_KB32:
        return String("f3r.t16.kb32")
    if variant == TUNED_F2D_T16_KB32:
        return String("f2d.t16.kb32")
    if variant == TUNED_F4_T32X16_KB32:
        return String("f4.t32x16.kb32")
    if variant == TUNED_F2D_T32X16_KB32:
        return String("f2d.t32x16.kb32")
    if variant == TUNED_F4_T32_KB32_B128:
        return String("f4.t32.kb32.b128")
    if variant == TUNED_F2D_T32_KB32_B128:
        return String("f2d.t32.kb32.b128")
    if variant == TUNED_F4_W64_KB16_B128:
        return String("f4.w64.kb16.b128")
    if variant == TUNED_F2D_W64_KB16_B128:
        return String("f2d.w64.kb16.b128")
    if variant == TUNED_F4_ROW_KB16:
        return String("f4.row.kb16")
    if variant == TUNED_F3_ROW64_KB16:
        return String("f3.row64.kb16")
    if variant == TUNED_F2D_ROW64_KB16:
        return String("f2d.row64.kb16")
    if variant == TUNED_F2_T32_KB8:
        return String("f2.t32.kb8")
    if variant == TUNED_F2_T32X64_KB16:
        return String("f2.t32x64.kb16")
    if variant == TUNED_F2_T64X32_KB16:
        return String("f2.t64x32.kb16")
    if variant == TUNED_F2_SG8_KB16:
        return String("f2.sg8.kb16")
    if variant == TUNED_F2_T16_KB16:
        return String("f2.t16.kb16")
    if variant == TUNED_F2_T32_KB16_B128:
        return String("f2.t32.kb16.b128")
    if variant == TUNED_F2_ROW64_KB16:
        return String("f2.row64.kb16")
    return String("unknown")


def int15_apple_tuned_chunk_steps(variant: Int) -> Int:
    """The steps of `k` in one chunk of the variant's form: the steps the
    float accumulators run before they are carried."""
    if variant == TUNED_F2_T32_KB16 or variant == TUNED_F2_T32_KB32 or variant == TUNED_F2D_T32_KB16 or variant == TUNED_F2D_T32_KB32 or variant == TUNED_F2_W64_KB16 or variant == TUNED_F2D_T16_KB32 or variant == TUNED_F2D_T32X16_KB32 or variant == TUNED_F2D_T32_KB32_B128 or variant == TUNED_F2D_W64_KB16_B128 or variant == TUNED_F2D_ROW64_KB16 or variant == TUNED_F2_T32_KB8 or variant == TUNED_F2_T32X64_KB16 or variant == TUNED_F2_T64X32_KB16 or variant == TUNED_F2_SG8_KB16 or variant == TUNED_F2_T16_KB16 or variant == TUNED_F2_T32_KB16_B128 or variant == TUNED_F2_ROW64_KB16:
        return INT15_TUNED2_CHUNK_STEPS
    if variant == TUNED_F3_T32_KB16 or variant == TUNED_F3R_T32_KB16 or variant == TUNED_F3R_T32_KB32 or variant == TUNED_F3R_T16_KB32 or variant == TUNED_F3_ROW64_KB16:
        return INT15_TUNED3_CHUNK_STEPS
    return INT15_TUNED4_CHUNK_STEPS


@always_inline
def _stage_planes[
    ROWS: Int, KB: Int, NT: Int, PMAJOR: Bool, ST: Int, MODE: Int, SCALAR: Bool
](
    dh: UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    dl: UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    ds: UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    qh: MutPointer[Int8, MutAnyOrigin],
    ql: MutPointer[Int8, MutAnyOrigin],
    row0: Int,
    rows: Int,
    k0: Int,
    k: Int,
    tid: Int,
    aligned: Bool,
):
    """One window of BOTH planes of one operand into threadgroup memory,
    each piece as the float32 it names, and under `STAGE_PAIR_SUM` the
    plane of their sums `hi + lo` (an integer in `[-127, 254]`, exact).
    Under `STAGE_WHOLE` ONE plane, into `dh`: the code whole, `hi * 128 +
    lo`, a product by a power of two and an addition of integers below
    2^15, exact. A slot is
    four consecutive steps of one row; a thread owns slots `tid, tid + NT,
    ...` below the window's `ROWS * KB / 4`. `PMAJOR`: element (row r, step
    p) at `d[p * ST + r]` (the left operand); else at `d[r * ST + p]` (the
    right). A row at or beyond `rows` and a step at or beyond `k` are the
    ZERO CODE in every plane.

    FOUR CODES IN ONE LOAD where the load is aligned: inside the row, `k` a
    multiple of four (every row then starts on one), and the bases of the
    planes aligned, which the launch reads off the pointers and passes as
    `aligned` (clause W-11's discipline). One code per load elsewhere, and
    everywhere under `SCALAR`, the staging lane/lowbit-int15's run 1 had."""
    comptime SLOTS = (ROWS * KB) // 4
    comptime SL = (SLOTS + NT - 1) // NT
    comptime assert KB % 4 == 0, "_stage_planes: a window is whole slots"
    comptime for sl in range(SL):
        var s = sl * NT + tid
        if s < SLOTS:
            var r = s // (KB // 4)
            var p4 = (s % (KB // 4)) * 4
            var vh = SIMD[DType.float32, 4](0.0)
            var vl = SIMD[DType.float32, 4](0.0)
            var gr = row0 + r
            if gr < rows:
                var base = gr * k + k0 + p4
                var wide = aligned and k0 + p4 + 4 <= k and (k & 3) == 0
                comptime if SCALAR:
                    wide = False
                if wide:
                    vh = qh.unsafe_load[width=4, alignment=4](base).cast[DType.float32]()
                    vl = ql.unsafe_load[width=4, alignment=4](base).cast[DType.float32]()
                else:
                    comptime for e in range(4):
                        if k0 + p4 + e < k:
                            vh[e] = qh.unsafe_load(base + e).cast[DType.float32]()
                            vl[e] = ql.unsafe_load(base + e).cast[DType.float32]()
            comptime if MODE == STAGE_WHOLE:
                vh = vh * SIMD[DType.float32, 4](128.0) + vl
            comptime if PMAJOR:
                comptime for e in range(4):
                    dh[(p4 + e) * ST + r] = vh[e]
                comptime if MODE != STAGE_WHOLE:
                    comptime for e in range(4):
                        dl[(p4 + e) * ST + r] = vl[e]
                comptime if MODE == STAGE_PAIR_SUM:
                    var vs = vh + vl
                    comptime for e in range(4):
                        ds[(p4 + e) * ST + r] = vs[e]
            else:
                (dh + r * ST + p4).store[alignment=16](vh)
                comptime if MODE != STAGE_WHOLE:
                    (dl + r * ST + p4).store[alignment=16](vl)
                comptime if MODE == STAGE_PAIR_SUM:
                    (ds + r * ST + p4).store[alignment=16](vh + vl)


@always_inline
def _flush_deferred[
    NC: Int
](
    mut hi: InlineArray[Int32, NC],
    mut lo: InlineArray[Int32, NC],
    mut run_h: InlineArray[Int32, NC],
    mut run_l: InlineArray[Int32, NC],
):
    """TWO's deferred carry: each run cut in 16-bit halves (an arithmetic
    shift and a mask), `HI += (run_h >> 16) * 2^7 + (run_l >> 16)` and
    `LO += (run_h & 65535) * 2^7 + (run_l & 65535)`, the runs restarted at
    zero. Then `S = HI * 2^16 + LO`."""
    comptime for c in range(NC):
        var h = run_h[c]
        var l = run_l[c]
        hi[c] += ((h >> Int32(16)) << Int32(7)) + (l >> Int32(16))
        lo[c] += ((h & Int32(65535)) << Int32(7)) + (l & Int32(65535))
        run_h[c] = Int32(0)
        run_l[c] = Int32(0)


@always_inline
def _tuned_body[
    SGM: Int, SGN: Int, FM: Int, FN: Int, KB: Int, FORM: Int, SCALAR: Bool, REGSUM: Bool, DEFER: Bool
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
    tile_row0_in: Int32,
    aligned_in: Int32,
):
    """OP_NT, `C[m x n] = Qa[m x k] . Qb[n x k]^T`, the rows of tiles from
    `tile_row0_in` on that the grid covers. One block owns a `BM x BN`
    output tile; each simdgroup owns `FM x FN` 8x8 fragments, each lane two
    cells of each (`identical_gemm_apple_mma_kernel`'s layout).

    Per window of `KB` steps the block stages the planes of both operands
    as float32 (`_stage_planes`), then every simdgroup multiplies on the
    matrix unit: four products per fragment and step of the unit (FOUR),
    three (THREE) or two (TWO). THREE and FOUR keep three float
    accumulators and, every chunk and at the end of `k`, each lane converts
    them to Int32, adds them to its running sums `HH`, `MID`, `LL` and
    restarts them at zero. TWO keeps two and carries after every step of
    the unit (the header has the carry). The epilogue is every plan's."""
    comptime NSG = SGM * SGN
    comptime NT = NSG * 32
    comptime BM = 8 * FM * SGM
    comptime BN = 8 * FN * SGN
    comptime AST = BM + 4
    comptime BST = KB + 4
    comptime NF = FM * FN
    comptime NC = 2 * NF
    comptime TWO = FORM == INT15_TUNED_FORM_TWO
    comptime THREE = FORM == INT15_TUNED_FORM_THREE
    comptime STAGED_SUM = THREE and not REGSUM
    comptime NPLB = 3 if STAGED_SUM else 2
    comptime NPLA = 1 if TWO else NPLB
    comptime MODE_B = STAGE_PAIR_SUM if STAGED_SUM else STAGE_PAIR
    comptime MODE_A = STAGE_WHOLE if TWO else MODE_B
    comptime CHUNK = INT15_TUNED2_CHUNK_STEPS if TWO else (INT15_TUNED3_CHUNK_STEPS if THREE else INT15_TUNED4_CHUNK_STEPS)
    comptime STEP_MAX = INT15_TUNED2_STEP_MAX if TWO else (INT15_TUNED3_STEP_MAX if THREE else INT15_TUNED4_STEP_MAX)
    comptime CHUNK_WINDOWS = 1 if TWO else CHUNK // KB
    comptime ASZ = KB * AST
    comptime BSZ = BN * BST
    comptime assert FORM >= INT15_TUNED_FORM_TWO and FORM <= INT15_TUNED_FORM_FOUR, "the form is TWO, THREE or FOUR"
    comptime assert TWO or not DEFER, "the deferred carry is TWO's"
    comptime assert INT15_TUNED2_STEP_MAX * INT15_TUNED2_CHUNK_STEPS * INT15_TUNED2D_FLUSH_STEPS < 2147483648, "a deferred run must be an Int32"
    comptime assert KB % 8 == 0, "the window is whole steps of the unit"
    comptime assert TWO or CHUNK_WINDOWS * KB == CHUNK, "a chunk is whole windows"
    comptime assert STEP_MAX * CHUNK < INT15_TUNED_EXACT_BOUND, "a chunk's largest partial sum must be a float32"
    comptime assert (NPLA * ASZ + NPLB * BSZ) * 4 <= INT15_TUNED_SHARED_BYTES, "the staged planes must fit a threadgroup's memory"
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var aligned = aligned_in != Int32(0)
    var tid = Int(thread_idx.x)
    var sg = tid // 32
    var lane = tid % 32
    var sgm = sg // SGN
    var sgn = sg % SGN
    var nbn = (n + BN - 1) // BN
    var bid = Int(block_idx.x)
    var m0 = (Int(tile_row0_in) + bid // nbn) * BM
    var n0 = (bid % nbn) * BN
    var qd = lane // 4
    var frow = (qd & 4) + ((lane // 2) % 4)
    var fcol = (qd & 2) * 2 + (lane % 2) * 2
    # The planes of an operand lie one after another: high (under TWO the
    # left operand's one plane, the code whole), low, and the sums where
    # they are staged.
    var at = stack_allocation[NPLA * ASZ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var bt = stack_allocation[NPLB * BSZ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    # The running sums. THREE and FOUR: `HH`, `MID`, `LL`. TWO: `hh_total`
    # is the HIGH half of the carry and `ll_total` the LOW one.
    var hh_total = InlineArray[Int32, NC](fill=Int32(0))
    var mid_total = InlineArray[Int32, NC](fill=Int32(0))
    var ll_total = InlineArray[Int32, NC](fill=Int32(0))
    # TWO with DEFER: `mid_total` is the run of `a * bh` and `run_l` the
    # run of `a * bl`, both since the last flush.
    var run_l = InlineArray[Int32, NC](fill=Int32(0))
    # `mid_acc` is `MID` under FOUR and `PP` under THREE. Under TWO
    # `hh_acc` is `a * bh` and `ll_acc` is `a * bl`.
    var hh_acc = InlineArray[_AMMA_M64, NF](fill=_AMMA_M64(0))
    var mid_acc = InlineArray[_AMMA_M64, NF](fill=_AMMA_M64(0))
    var ll_acc = InlineArray[_AMMA_M64, NF](fill=_AMMA_M64(0))
    var windows = (k + KB - 1) // KB
    var step = 0
    for w in range(windows):
        var k0 = w * KB
        # The left operand's planes: high (or the code whole under TWO) at
        # `at`, low at `at + ASZ`, the sums last. Job 2 (8fcd8be81) passed
        # the LAST plane as the low one, which under THREE with staged sums
        # wrote the low plane over the sums: its gate failed f3.t32.kb16 and
        # f3.row64.kb16 on every case, as it had to.
        _stage_planes[BM, KB, NT, True, AST, MODE_A, SCALAR](
            at, at + ASZ, at + (NPLA - 1) * ASZ, ah, al, m0, m, k0, k, tid, aligned
        )
        _stage_planes[BN, KB, NT, False, BST, MODE_B, SCALAR](
            bt, bt + BSZ, bt + (NPLB - 1) * BSZ, bh, bl, n0, n, k0, k, tid, aligned
        )
        barrier()
        comptime for p8 in range(KB // 8):
            var ahf = InlineArray[_AMMA_M64, FM](fill=_AMMA_M64(0))
            var alf = InlineArray[_AMMA_M64, FM](fill=_AMMA_M64(0))
            var asf = InlineArray[_AMMA_M64, FM](fill=_AMMA_M64(0))
            var bhf = InlineArray[_AMMA_M64, FN](fill=_AMMA_M64(0))
            var blf = InlineArray[_AMMA_M64, FN](fill=_AMMA_M64(0))
            var bsf = InlineArray[_AMMA_M64, FN](fill=_AMMA_M64(0))
            comptime for fm in range(FM):
                ahf[fm] = _amma_load_t(at + (8 * p8) * AST + (sgm * FM + fm) * 8, AST)
                comptime if not TWO:
                    alf[fm] = _amma_load_t(at + ASZ + (8 * p8) * AST + (sgm * FM + fm) * 8, AST)
                comptime if STAGED_SUM:
                    asf[fm] = _amma_load_t(at + 2 * ASZ + (8 * p8) * AST + (sgm * FM + fm) * 8, AST)
                elif THREE:
                    asf[fm] = ahf[fm] + alf[fm]
            comptime for fq in range(FN):
                bhf[fq] = _amma_load_t(bt + ((sgn * FN + fq) * 8) * BST + 8 * p8, BST)
                blf[fq] = _amma_load_t(bt + BSZ + ((sgn * FN + fq) * 8) * BST + 8 * p8, BST)
                comptime if STAGED_SUM:
                    bsf[fq] = _amma_load_t(bt + 2 * BSZ + ((sgn * FN + fq) * 8) * BST + 8 * p8, BST)
                elif THREE:
                    bsf[fq] = bhf[fq] + blf[fq]
            comptime for fm in range(FM):
                comptime for fq in range(FN):
                    hh_acc[fm * FN + fq] = _amma_mma(ahf[fm], bhf[fq], hh_acc[fm * FN + fq])
                    comptime if TWO:
                        ll_acc[fm * FN + fq] = _amma_mma(ahf[fm], blf[fq], ll_acc[fm * FN + fq])
                    elif THREE:
                        mid_acc[fm * FN + fq] = _amma_mma(asf[fm], bsf[fq], mid_acc[fm * FN + fq])
                        ll_acc[fm * FN + fq] = _amma_mma(alf[fm], blf[fq], ll_acc[fm * FN + fq])
                    else:
                        mid_acc[fm * FN + fq] = _amma_mma(ahf[fm], blf[fq], mid_acc[fm * FN + fq])
                        mid_acc[fm * FN + fq] = _amma_mma(alf[fm], bhf[fq], mid_acc[fm * FN + fq])
                        ll_acc[fm * FN + fq] = _amma_mma(alf[fm], blf[fq], ll_acc[fm * FN + fq])
            comptime if TWO:
                step += 1
                # THE CHUNK IS ONE STEP OF THE UNIT: convert, join, cut in
                # two, carry, restart. No Int64.
                var step_end = True
                comptime if INT15_TUNED_CHUNK_SABOTAGE:
                    # THE DEFECT ARM: the accumulators live across two steps.
                    step_end = (step % 2) == 0 or (w + 1 == windows and p8 == KB // 8 - 1)
                if step_end:
                    comptime for f in range(NF):
                        comptime for e in range(2):
                            var th = hh_acc[f][e].cast[DType.int32]()
                            var tl = ll_acc[f][e].cast[DType.int32]()
                            comptime if DEFER:
                                mid_total[2 * f + e] += th
                                run_l[2 * f + e] += tl
                            else:
                                var t = (th << Int32(7)) + tl
                                hh_total[2 * f + e] += t >> Int32(16)
                                ll_total[2 * f + e] += t & Int32(65535)
                        hh_acc[f] = _AMMA_M64(0)
                        ll_acc[f] = _AMMA_M64(0)
                comptime if DEFER:
                    if step % INT15_TUNED2D_FLUSH_STEPS == 0:
                        _flush_deferred[NC](hh_total, ll_total, mid_total, run_l)
        barrier()
        comptime if not TWO:
            var chunk_end = w + 1 == windows
            comptime if not INT15_TUNED_CHUNK_SABOTAGE:
                chunk_end = chunk_end or (w + 1) % CHUNK_WINDOWS == 0
            if chunk_end:
                # THE CARRY: three exact conversions per cell, and under
                # THREE the cross term's share of the chunk, in integers.
                comptime for f in range(NF):
                    comptime for e in range(2):
                        var vh = hh_acc[f][e].cast[DType.int32]()
                        var vm = mid_acc[f][e].cast[DType.int32]()
                        var vl = ll_acc[f][e].cast[DType.int32]()
                        comptime if THREE:
                            vm = vm - vh - vl
                        hh_total[2 * f + e] += vh
                        mid_total[2 * f + e] += vm
                        ll_total[2 * f + e] += vl
                    hh_acc[f] = _AMMA_M64(0)
                    mid_acc[f] = _AMMA_M64(0)
                    ll_acc[f] = _AMMA_M64(0)
    comptime if DEFER:
        _flush_deferred[NC](hh_total, ll_total, mid_total, run_l)
    comptime for fm in range(FM):
        comptime for fq in range(FN):
            comptime for e in range(2):
                var gi = m0 + (sgm * FM + fm) * 8 + frow
                var gj = n0 + (sgn * FN + fq) * 8 + fcol + e
                if gi < m and gj < n:
                    var at_cell = 2 * (fm * FN + fq) + e
                    var total = Int64(0)
                    comptime if TWO:
                        total = (Int64(hh_total[at_cell]) << Int64(16)) + Int64(ll_total[at_cell])
                    else:
                        total = int15_recombine(hh_total[at_cell], mid_total[at_cell], ll_total[at_cell])
                    var cell = dequant_int15_pinned(
                        total, Int(ea.unsafe_load(gi)) + Int(eb.unsafe_load(gj))
                    )
                    comptime if INT15_TUNED_VALUE_SABOTAGE:
                        cell = gemm_oracle_sabotage_value_flip(cell)
                    c.unsafe_store(gi * n + gj, cell)


def identical_gemm_int15_apple_tuned_kernel[
    SGM: Int, SGN: Int, FM: Int, FN: Int, KB: Int, FORM: Int, SCALAR: Bool, REGSUM: Bool, DEFER: Bool
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
    tile_row0_in: Int32,
    aligned_in: Int32,
):
    """The tuned kernel (`_tuned_body`), no launch bound declared."""
    _tuned_body[SGM, SGN, FM, FN, KB, FORM, SCALAR, REGSUM, DEFER](c, ah, al, ea, bh, bl, eb, m_in, n_in, k_in, tile_row0_in, aligned_in)


#: The launch bound the `b128` variants declare: their real block size.
comptime INT15_TUNED_LAUNCH_BOUND = 128


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(INT15_TUNED_LAUNCH_BOUND)))
def identical_gemm_int15_apple_tuned_kernel_b128[
    SGM: Int, SGN: Int, FM: Int, FN: Int, KB: Int, FORM: Int, SCALAR: Bool, REGSUM: Bool, DEFER: Bool
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
    tile_row0_in: Int32,
    aligned_in: Int32,
):
    """The same kernel DECLARING its launch bound, 128 threads (what
    lane/amd-step-time found on gfx942: a compiler that assumes 1,024
    threads a block budgets registers for them and spills a big register
    tile). SCHEDULING: the arithmetic is `_tuned_body`'s."""
    _tuned_body[SGM, SGN, FM, FN, KB, FORM, SCALAR, REGSUM, DEFER](c, ah, al, ea, bh, bl, eb, m_in, n_in, k_in, tile_row0_in, aligned_in)


#: The block every Apple plan here stays under when the pipeline's limit
#: cannot be read: 256 threads ran on the M2 Pro at 2 x 2 fragments of form
#: FOUR (job 2's probe), 512 did not.
comptime INT15_APPLE_SAFE_THREADS = 256


def int15_apple_refuse_block(threads: Int, admits: Int, who: String) raises:
    """Raise, naming both numbers, when a block of `threads` passes what
    the pipeline admits (`admits`, -1 when it could not be read: then the
    bound is `INT15_APPLE_SAFE_THREADS`)."""
    if admits >= 0 and threads > admits:
        raise Error(
            who + ": REFUSED, the block has " + String(threads)
            + " threads and the pipeline admits " + String(admits)
            + " (maxTotalThreadsPerThreadgroup); Metal would launch nothing and report nothing"
        )
    if admits < 0 and threads > INT15_APPLE_SAFE_THREADS:
        raise Error(
            who + ": REFUSED, the block has " + String(threads)
            + " threads, the pipeline's limit could not be read and the bound is "
            + String(INT15_APPLE_SAFE_THREADS)
        )


def int15_apple_tuned_pipeline_admits[
    SGM: Int, SGN: Int, FM: Int, FN: Int, KB: Int, FORM: Int, SCALAR: Bool, REGSUM: Bool, DEFER: Bool, BOUND: Bool
](ctx: DeviceContext) -> Int:
    """What the pipeline of one schedule admits, -1 when it cannot be read
    (for the probe)."""
    try:
        comptime if BOUND:
            return Int(ctx.compile_function[
                identical_gemm_int15_apple_tuned_kernel_b128[SGM, SGN, FM, FN, KB, FORM, SCALAR, REGSUM, DEFER]
            ]().get_attribute(Attribute.MAX_THREADS_PER_BLOCK))
        else:
            return Int(ctx.compile_function[
                identical_gemm_int15_apple_tuned_kernel[SGM, SGN, FM, FN, KB, FORM, SCALAR, REGSUM, DEFER]
            ]().get_attribute(Attribute.MAX_THREADS_PER_BLOCK))
    except:
        return -1


def _launch_tuned[
    SGM: Int, SGN: Int, FM: Int, FN: Int, KB: Int, FORM: Int, SCALAR: Bool, REGSUM: Bool, DEFER: Bool, BOUND: Bool
](
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
    slice_macs: Int,
) raises:
    """One variant, in slices of whole rows of tiles, a wait between two
    slices and none after the last."""
    comptime kern = identical_gemm_int15_apple_tuned_kernel[SGM, SGN, FM, FN, KB, FORM, SCALAR, REGSUM, DEFER]
    comptime kern_b = identical_gemm_int15_apple_tuned_kernel_b128[SGM, SGN, FM, FN, KB, FORM, SCALAR, REGSUM, DEFER]
    comptime assert not BOUND or SGM * SGN * 32 <= INT15_TUNED_LAUNCH_BOUND, "the block passes its declared bound"
    # A LAUNCH METAL WOULD DROP IS REFUSED HERE (job 2's probe): a pipeline
    # admits at most `maxTotalThreadsPerThreadgroup` threads, which falls
    # with its registers (on the M2 Pro, below 512 for 4 x 4 simdgroups of
    # 2 x 2 fragments); a larger block launches nothing and says nothing.
    var admits = -1
    try:
        comptime if BOUND:
            admits = Int(ctx.compile_function[kern_b]().get_attribute(Attribute.MAX_THREADS_PER_BLOCK))
        else:
            admits = Int(ctx.compile_function[kern]().get_attribute(Attribute.MAX_THREADS_PER_BLOCK))
    except:
        admits = -1
    int15_apple_refuse_block(SGM * SGN * 32, admits, "identical_gemm_int15_apple_tuned")
    comptime BM = 8 * FM * SGM
    comptime BN = 8 * FN * SGN
    var nbm = (m + BM - 1) // BM
    var nbn = (n + BN - 1) // BN
    # The staging takes four codes in one load only where the bases of all
    # four planes are aligned, read here (clause W-11).
    var aligned = Int32(0)
    if (
        (Int(ah.unsafe_ptr()) | Int(al.unsafe_ptr()) | Int(bh.unsafe_ptr()) | Int(bl.unsafe_ptr())) & 7
    ) == 0:
        aligned = Int32(1)
    var per_tile_row = BM * n * k
    var rows_per = slice_macs // per_tile_row
    if rows_per < 1:
        rows_per = 1
    var r0 = 0
    while r0 < nbm:
        var rows = nbm - r0
        if rows > rows_per:
            rows = rows_per
        comptime if BOUND:
            ctx.enqueue_function[kern_b](
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
                Int32(r0),
                aligned,
                grid_dim=(rows * nbn, 1, 1),
                block_dim=(SGM * SGN * 32, 1, 1),
            )
        else:
            ctx.enqueue_function[kern](
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
                Int32(r0),
                aligned,
                grid_dim=(rows * nbn, 1, 1),
                block_dim=(SGM * SGN * 32, 1, 1),
            )
        r0 += rows
        if r0 < nbm:
            ctx.synchronize()


def identical_gemm_int15_apple_tuned_into(
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
    variant: Int,
    slice_macs: Int = INT15_TUNED_SLICE_MACS,
) raises:
    """The NAMED variant. Refuses by name on a column that is not Apple and
    above the profile's `k`. It waits between the slices of a launch; the
    caller waits after the last."""
    comptime if not INT15_TUNED_AVAILABLE:
        raise Error(
            "identical_gemm_int15_apple_tuned: column " + column_name(TARGET_COLUMN)
            + " is not Apple; this plan runs on Metal's float matrix unit only"
        )
    else:
        if m <= 0 or n <= 0 or k <= 0 or k > INT15_MAX_K:
            raise Error(
                "identical_gemm_int15_apple_tuned: m, n and k must be positive and"
                " k at most " + String(INT15_MAX_K) + " (contract W-4), got m="
                + String(m) + " n=" + String(n) + " k=" + String(k)
            )
        if slice_macs <= 0:
            raise Error("identical_gemm_int15_apple_tuned: the slice must be positive")
        if variant == TUNED_F4_T32_KB32:
            _launch_tuned[2, 2, 2, 2, 32, INT15_TUNED_FORM_FOUR, False, False, False, False](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        elif variant == TUNED_F4_T32_KB16:
            _launch_tuned[2, 2, 2, 2, 16, INT15_TUNED_FORM_FOUR, False, False, False, False](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        elif variant == TUNED_F3_T32_KB16:
            _launch_tuned[2, 2, 2, 2, 16, INT15_TUNED_FORM_THREE, False, False, False, False](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        elif variant == TUNED_F3R_T32_KB16:
            _launch_tuned[2, 2, 2, 2, 16, INT15_TUNED_FORM_THREE, False, True, False, False](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        elif variant == TUNED_F3R_T32_KB32:
            _launch_tuned[2, 2, 2, 2, 32, INT15_TUNED_FORM_THREE, False, True, False, False](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        elif variant == TUNED_F2_T32_KB16:
            _launch_tuned[2, 2, 2, 2, 16, INT15_TUNED_FORM_TWO, False, False, False, False](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        elif variant == TUNED_F2_T32_KB32:
            _launch_tuned[2, 2, 2, 2, 32, INT15_TUNED_FORM_TWO, False, False, False, False](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        elif variant == TUNED_F2D_T32_KB16:
            _launch_tuned[2, 2, 2, 2, 16, INT15_TUNED_FORM_TWO, False, False, True, False](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        elif variant == TUNED_F2D_T32_KB32:
            _launch_tuned[2, 2, 2, 2, 32, INT15_TUNED_FORM_TWO, False, False, True, False](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        elif variant == TUNED_F2_W64_KB16:
            _launch_tuned[2, 2, 4, 4, 16, INT15_TUNED_FORM_TWO, False, False, False, False](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        elif variant == TUNED_F4_T16_KB32:
            _launch_tuned[2, 2, 1, 1, 32, INT15_TUNED_FORM_FOUR, False, False, False, False](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        elif variant == TUNED_F3R_T16_KB32:
            _launch_tuned[2, 2, 1, 1, 32, INT15_TUNED_FORM_THREE, False, True, False, False](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        elif variant == TUNED_F2D_T16_KB32:
            _launch_tuned[2, 2, 1, 1, 32, INT15_TUNED_FORM_TWO, False, False, True, False](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        elif variant == TUNED_F4_T32X16_KB32:
            _launch_tuned[2, 2, 2, 1, 32, INT15_TUNED_FORM_FOUR, False, False, False, False](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        elif variant == TUNED_F2D_T32X16_KB32:
            _launch_tuned[2, 2, 2, 1, 32, INT15_TUNED_FORM_TWO, False, False, True, False](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        elif variant == TUNED_F4_T32_KB32_B128:
            _launch_tuned[2, 2, 2, 2, 32, INT15_TUNED_FORM_FOUR, False, False, False, True](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        elif variant == TUNED_F2D_T32_KB32_B128:
            _launch_tuned[2, 2, 2, 2, 32, INT15_TUNED_FORM_TWO, False, False, True, True](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        elif variant == TUNED_F4_W64_KB16_B128:
            _launch_tuned[2, 2, 4, 4, 16, INT15_TUNED_FORM_FOUR, False, False, False, True](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        elif variant == TUNED_F2D_W64_KB16_B128:
            _launch_tuned[2, 2, 4, 4, 16, INT15_TUNED_FORM_TWO, False, False, True, True](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        elif variant == TUNED_F4_ROW_KB16:
            _launch_tuned[1, 4, 1, 4, 16, INT15_TUNED_FORM_FOUR, False, False, False, False](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        elif variant == TUNED_F3_ROW64_KB16:
            _launch_tuned[1, 4, 1, 2, 16, INT15_TUNED_FORM_THREE, False, False, False, False](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        elif variant == TUNED_F2D_ROW64_KB16:
            _launch_tuned[1, 4, 1, 2, 16, INT15_TUNED_FORM_TWO, False, False, True, False](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        elif variant == TUNED_F2_T32_KB8:
            _launch_tuned[2, 2, 2, 2, 8, INT15_TUNED_FORM_TWO, False, False, False, False](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        elif variant == TUNED_F2_T32X64_KB16:
            _launch_tuned[2, 2, 2, 4, 16, INT15_TUNED_FORM_TWO, False, False, False, False](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        elif variant == TUNED_F2_T64X32_KB16:
            _launch_tuned[2, 2, 4, 2, 16, INT15_TUNED_FORM_TWO, False, False, False, False](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        elif variant == TUNED_F2_SG8_KB16:
            _launch_tuned[4, 2, 2, 2, 16, INT15_TUNED_FORM_TWO, False, False, False, False](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        elif variant == TUNED_F2_T16_KB16:
            _launch_tuned[2, 2, 1, 1, 16, INT15_TUNED_FORM_TWO, False, False, False, False](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        elif variant == TUNED_F2_T32_KB16_B128:
            _launch_tuned[2, 2, 2, 2, 16, INT15_TUNED_FORM_TWO, False, False, False, True](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        elif variant == TUNED_F2_ROW64_KB16:
            _launch_tuned[1, 4, 1, 2, 16, INT15_TUNED_FORM_TWO, False, False, False, False](
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, slice_macs
            )
        else:
            raise Error(
                "identical_gemm_int15_apple_tuned: variant must be below "
                + String(TUNED_VARIANT_COUNT) + ", got " + String(variant)
            )
