"""Gate for `gbdt/methods/pointwise_kernels.mojo`, the POINTWISE host layer.

THIS IS THE FIRST TIME THE WHOLE FAMILY RUNS TOGETHER. Nine checks already
gate the six accumulators and the three drivers, each one launched by hand at
a grid the check chose. Nothing until now computed a grid, chose a
multiplier, fanned out over the four one-byte bit widths, scanned the result,
or subtracted a sibling. That is what `compute_hist2` does and it is what
this file gates.

THE FIXTURE HAS EVERY BIN WIDTH IN IT AT ONCE, which is the point:

    BINARY policy     64 features, 1 fold each, TWO packed UInt32 columns
    HALF-BYTE policy  24 features of at most 16 folds, THREE columns
    ONE-BYTE policy   16 features in four groups of four, whose WIDEST
                      member sends each group to a different kernel:

        group 0   folds  20  25  30  32    max  32  ->  5-bit
        group 1   folds  40  50  60  64    max  64  ->  6-bit
        group 2   folds  90 100 120 128    max 128  ->  7-bit
        group 3   folds 200 150 256 180    max 256  ->  8-bit

EVERY POLICY SPANS MORE THAN ONE BLOCK, and that is deliberate. Each
launcher's `numBlocks.x` is `ceil(featureCount / featuresPerBlock)` with a
different divisor -- 32 binary, 8 half-byte, 4 one-byte -- and at ONE block
per policy the divisor is unobservable: `ceil(8 / 8)`, `ceil(8 / 4)` and
`ceil(8 / 16)` are all 1. The fixture began with 8 half-byte and 32 binary
features and the divisors were checked by sabotage; NOTHING MOVED. At three
half-byte blocks and two binary blocks, under-provisioning moves 158 and 64
cells respectively.

OVER-provisioning still moves nothing, at any block count, and that is not a
hole in the gate -- it is a property of their kernels. A block whose feature
slice starts past the end computes `fCount = min(fCount - base, N) <= 0` and
returns, so a grid larger than the features need is harmless by construction.
Only a grid too SMALL loses data, and that is the direction the gate covers.

All three policies read THE SAME rows, the same permutation, the same target
and weight planes, so a disagreement between them is a disagreement about the
dispatch and not about the data.

EVERY PLANTED VALUE IS DISTINCT AND HASHED. `target[r] = (r * 37) % 100 + 1`
and `weight[r] = (r * 53) % 64 + 1`, and the bin of feature `f` at row `r` is
`(r * (7 + 3f) + 5k) % folds[f]` -- so no two features see the same bin
sequence and no two cells of a feature hold the same total. Uniform data
would verify the totals and nothing about placement
([[uniform-test-data-hides-permutation]]).

EVERY MAGNITUDE STAYS UNDER 2^24. The largest cell is one feature's whole
column, 23,800 rows x 100 = 2.38e6, so every partial sum and every prefix is
an exactly representable integer in Float32 and EXACT equality is the
comparison at every gate. That is also what makes the M > 1 arm comparable to
the M == 1 arm cell for cell: integer partial sums add exactly in any order.

GATES
-----
  F1  END TO END, ALL THREE POLICIES, per cell against a host tally.
      `compute_hist2` is called once per policy on a full pass; the one-byte
      call fans out to all four bit widths and every group must come back
      filled exactly once. The comparison is made AFTER the scan, so the
      expected value is the host tally's running PREFIX within each feature
      -- except for the one feature marked one-hot, whose cells must stay
      RAW, and except for the binary policy, which their dispatcher does not
      scan at all (`pointwise_kernels.cpp:70`).

  F2  THE MULTIPLIER, both arms, same rows, compared to each other and to
      F1's host tally. Two calls differing ONLY in how many leaves the level
      has, which is what `EstimateBlockPerFeatureMultiplier` reads:

          part_count  1    ->  1 block on the feature axis   ->  M = 4
          part_count 64    -> 64 blocks (63 leaves empty)    ->  M = 1

      and part 0 holds the same rows in both, so its histogram must be
      identical. The M = 4 arm is the one that takes the atomicAdd
      writeback: `comptime if m > 1` in all three drivers. The two
      multipliers are also asserted DIFFERENT on the host, because two runs
      that both took M = 1 would agree perfectly and gate nothing.

  F2c THE MULTIPLIER'S VALUE, which is the only thing that can see WHICH
      feature count sized the estimate. Sizing the estimate wrong is silent
      under every other gate -- both choices give a self-consistent grid and
      a correct histogram -- but it changes `M`, and `M` changes the
      WRITEBACK from a store to an atomic add. Seeded with a non-zero
      pattern, a store and an add are different programs.

  F3  THE PARTIAL PASS: the scan offset, the smaller-sibling choice, and
      `UpdatePointwiseHistograms`. Four parts, two pairs, with pair 0's
      smaller child on the LEFT and pair 1's on the RIGHT so a launcher that
      always picks one side gets exactly half of it right. The left slots
      are PRE-FILLED with each pair's parent histogram, as they would be at
      the previous level; after the call every one of the four slots must
      hold its own part's histogram.

  F4  REACH, per bit width. Each one-byte kernel is run ALONE and the set of
      groups it wrote is recorded; it must be exactly one group and the
      right one. F1 cannot see a kernel that never claims anything -- a
      group claimed zero times and a group claimed twice both show up as
      wrong cells, but so does every other defect. This separates them, and
      it is the guard against "launched over every block and returned
      immediately every time".

  F5  `IsGridEmpty`, both ways in. A partial pass at `part_count == 1` makes
      every `numBlocks.y` zero; a policy with no features makes
      `numBlocks.x` zero. Neither may launch anything, and the buffer must
      come back bit-identical to a sentinel. NOTE ITS TWO FAILURE MODES: a
      launcher that mis-sizes `numBlocks.y` moves cells and F5 reports them,
      but a launcher that drops the guard entirely does not get that far --
      MAX raises `Dim value grid_dim.y must be a positive number` and the
      check dies at the launch. Both are red; only one prints F5.

  F6  the folds-to-bits binning, which is what sizes each bit width's
      estimate. Pinned at the 16-fold boundary in both directions.

  F7  the multiplier ladder's DOMAIN: every value
      `EstimateBlockPerFeatureMultiplier` can return, over a sweep of grids,
      dataset sizes and core counts, must be one of the seven the ladder
      instantiates -- and 128 must occur somewhere in the sweep, or their
      `min(..., 64)` clamp would be dead code and the raise would be
      reachable.

SABOTAGE, MEASURED 2026-08-21. Every one applied to the LIBRARY, run, and
reverted; the check itself was not touched.

    what was broken                            gate      cells
    ---------------------------------------------------------------------
    launch grid sized on featureCountForBits   F1        2796 of 3686
      (the two `numBlocks.x` collapsed to one) F2        2796
                                               F2c       1890 of 3090
                                               F3       11184 of 12360
                                               F4     3 of 4 widths claim
                                                      the wrong group or
                                                      nothing at all
    ESTIMATE sized on nbCount instead          F2c       2004 of 3090
      (the same two lines, the other way)      everything else GREEN
    `scanOffset` forced to 0                   F3       11600 of 12360
    `isLeftCalculated` flipped                 F3       11920 of 12360
    UpdatePointwiseHistograms numBlocks.y      F3       11920
      = partCount instead of partCount / 2     F5         108
    scan histPartCount = partCount on a        F3       11920
      partial pass                             F5         420
    half-byte divisor 8 -> 16                  F1         158 of 3686
    binary divisor 32 -> 64                    F1          64 of 3686
    folds-to-bits off by one at the powers     F6           5 bins
      of two
    `IsGridEmpty` always False                 F5      the launch RAISES
                                                       (`grid_dim.y must be
                                                       a positive number`)
      -- so the guard is not defensive tidiness: MAX refuses a zero-extent
      grid outright, where CUDA tolerates one.

    half-byte divisor 8 -> 4                   NOTHING MOVED, by
      (over-provision)                         construction; see above
    the `policy != BinaryFeatures` guard       NOTHING MOVED
      on the scan removed

THE LAST ONE IS WORTH KEEPING. Their dispatcher skips the scan for binary
features (`pointwise_kernels.cpp:70`) and removing that guard changes no
cell here -- because a binary feature has ONE fold and the ported scan
kernel returns early on `folds <= 1`, so the extra launch is a no-op at
every cell. The guard is a launch saved, not a value protected, and no
output check can distinguish the two. It is transcribed because it is
theirs and because the kernel's `folds <= 1` early-out is OURS (the
greedy-subsets scan's, adopted here) rather than CatBoost's -- their warp
scan would run 32 lanes over a 1-fold feature and write the same value
back. Leaning on our early-out to justify dropping their guard would be
depending on a deviation to excuse skipping a port.

RUN IT

    pixi run mojo run -I . mojo_only/pointwise_dispatch_check.mojo
"""

from max.gpu.host import DeviceContext

from mojo_only.kernel_matrix import (
    TARGET_COLUMN,
    pointwise_one_byte_fixed_for,
)
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

from gbdt.gpu_data.grid_policy import (
    POLICY_BINARY,
    POLICY_HALF_BYTE,
    POLICY_ONE_BYTE,
)
from gbdt.methods.kernel.split_properties_helpers import (
    estimate_block_per_feature_multiplier,
)
from gbdt.methods.pointwise_kernels import (
    FoldsHistogram,
    PW_MAX_MULTIPLIER,
    compute_hist2,
    compute_hist2_non_binary,
    folds_histogram_from_folds,
)

# 24,000 rows is not decoration. `EstimateBlockPerFeatureMultiplier` refuses
# to split the document axis below ~10,000 rows a piece
# (`(dsSize / multiplier) > 10000`), so a 3,000-row fixture -- which is what
# every earlier check in this family uses -- can only ever produce M = 1 and
# could not gate the atomic writeback at all. At 24,000 the ladder reaches
# exactly 4: 24000, 12000, then 6000 which fails the test.
comptime N_ROWS = 24000

#: Pinned, NOT queried. The library threads `sm_count` in for the reason
#: `estimate_block_per_feature_multiplier`'s docstring gives, and pinning it
#: here is what makes F2's two multipliers the same two numbers on every
#: machine. A device-dependent multiplier would make this gate report a
#: different thing on a laptop than in CI.
comptime SM_COUNT = 16

comptime N_OB = 16   # one-byte features, 4 blocks of 4
comptime N_HB = 24   # half-byte features, 3 blocks of 8
comptime N_B = 64    # binary features,   2 blocks of 32

#: The compressed index: 4 one-byte columns, 3 half-byte, 2 binary.
comptime N_HB_BLOCKS = N_HB // 8
comptime N_B_BLOCKS = N_B // 32
comptime N_COLUMNS = 4 + N_HB_BLOCKS + N_B_BLOCKS
comptime HB_COL0 = 4
comptime B_COL0 = 4 + N_HB_BLOCKS

#: The one-byte feature that is marked one-hot. Its cells must survive the
#: scan UNCHANGED: a one-hot bin is an equality test, so a running prefix
#: across its bins is not a quantity that means anything
#: (`split_properties_helpers.cuh:126`).
comptime ONE_HOT_OB = 5

#: `M > 1` switches the drivers to a fixed-point accumulator on the 8-bit
#: path (DEVIATION 93). Scale 4 rather than 1 because at 1 the division on
#: the way out is a no-op and deleting it leaves every gate green -- measured
#: in `pointwise_driver_check.mojo` D5. Four is a power of two over integer
#: stats, so the round trip is exact and every comparison here stays exact.
comptime FIXED_SCALE = Float32(4.0)


# ---------------------------------------------------------------------------
# the fixture, built once on the host
# ---------------------------------------------------------------------------


struct Fixture(Movable):
    var indices: List[UInt32]
    var target: List[Float32]
    var weight: List[Float32]
    var cindex: List[UInt32]          # 6 columns of N_ROWS

    var ob_folds: List[UInt32]
    var ob_offset: List[UInt32]
    var ob_first: List[UInt32]
    var ob_one_hot: List[UInt8]
    var ob_line: Int

    var hb_folds: List[UInt32]
    var hb_offset: List[UInt32]
    var hb_first: List[UInt32]
    var hb_one_hot: List[UInt8]
    var hb_line: Int

    var b_folds: List[UInt32]
    var b_offset: List[UInt32]
    var b_first: List[UInt32]
    var b_one_hot: List[UInt8]
    var b_line: Int

    def __init__(out self):
        self.indices = List[UInt32]()
        self.target = List[Float32]()
        self.weight = List[Float32]()
        for r in range(N_ROWS):
            # a permutation, so the kernels gather rather than stream
            self.indices.append(UInt32((r * 2654435761) % N_ROWS))
            self.target.append(Float32((r * 37) % 100 + 1))
            self.weight.append(Float32((r * 53) % 64 + 1))

        self.ob_folds = [
            20, 25, 30, 32,        # group 0 -> 5-bit
            40, 50, 60, 64,        # group 1 -> 6-bit
            90, 100, 120, 128,     # group 2 -> 7-bit
            200, 150, 256, 180,    # group 3 -> 8-bit
        ]
        self.hb_folds = [
            16, 5, 12, 3, 9, 16, 7, 11,       # half-byte block 0
            4, 16, 13, 6, 2, 15, 8, 10,       # half-byte block 1
            16, 9, 3, 14, 7, 5, 11, 16,       # half-byte block 2
        ]
        self.b_folds = List[UInt32]()
        for _ in range(N_B):
            self.b_folds.append(UInt32(1))

        self.ob_offset = List[UInt32]()
        for f in range(N_OB):
            self.ob_offset.append(UInt32((f // 4) * N_ROWS))
        # every feature of a block must name the SAME column: the kernels
        # read `feature->Offset` off the FIRST feature of the block only
        self.hb_offset = List[UInt32]()
        for j in range(N_HB):
            self.hb_offset.append(UInt32((HB_COL0 + j // 8) * N_ROWS))
        self.b_offset = List[UInt32]()
        for j in range(N_B):
            self.b_offset.append(UInt32((B_COL0 + j // 32) * N_ROWS))

        self.ob_first = List[UInt32]()
        var c = UInt32(0)
        for f in range(N_OB):
            self.ob_first.append(c)
            c += self.ob_folds[f]
        self.ob_line = Int(c)

        self.hb_first = List[UInt32]()
        c = UInt32(0)
        for f in range(N_HB):
            self.hb_first.append(c)
            c += self.hb_folds[f]
        self.hb_line = Int(c)

        self.b_first = List[UInt32]()
        for f in range(N_B):
            self.b_first.append(UInt32(f))
        self.b_line = N_B

        self.ob_one_hot = List[UInt8]()
        for f in range(N_OB):
            self.ob_one_hot.append(UInt8(1) if f == ONE_HOT_OB else UInt8(0))
        self.hb_one_hot = List[UInt8]()
        for _ in range(N_HB):
            self.hb_one_hot.append(UInt8(0))
        self.b_one_hot = List[UInt8]()
        for _ in range(N_B):
            self.b_one_hot.append(UInt8(0))

        # ---- the compressed index: six columns, one packing each --------
        self.cindex = List[UInt32]()
        for _ in range(N_COLUMNS * N_ROWS):
            self.cindex.append(UInt32(0))

        for g in range(4):
            for r in range(N_ROWS):
                var word = UInt32(0)
                for k in range(4):
                    var f = 4 * g + k
                    var nf = Int(self.ob_folds[f])
                    var b = (r * (7 + 3 * f) + 5 * k) % nf
                    # `Shift(f) = 32 - (1 + localId) * 8`, high bits first
                    word |= UInt32(b) << UInt32(24 - 8 * k)
                self.cindex[g * N_ROWS + r] = word

        for hbg in range(N_HB_BLOCKS):
            for r in range(N_ROWS):
                var word = UInt32(0)
                for k in range(8):
                    var j = 8 * hbg + k
                    var nf = Int(self.hb_folds[j])
                    var b = (r * (11 + 5 * j) + 3 * j) % nf
                    word |= UInt32(b) << UInt32(28 - 4 * k)
                self.cindex[(HB_COL0 + hbg) * N_ROWS + r] = word

        for bg in range(N_B_BLOCKS):
            for r in range(N_ROWS):
                var word = UInt32(0)
                for k in range(32):
                    var j = 32 * bg + k
                    var bit = (r * (13 + 2 * j) + j) % 2
                    word |= UInt32(bit) << UInt32(31 - k)
                self.cindex[(B_COL0 + bg) * N_ROWS + r] = word


def zeros(n: Int) -> List[Float32]:
    var v = List[Float32]()
    for _ in range(n):
        v.append(Float32(0.0))
    return v^


def tally_one_byte(
    fx: Fixture, offs: List[Int], lens: List[Int]
) -> List[Float32]:
    """Host histogram of the one-byte policy over the given row ranges.

    Stat 0 is WEIGHT and stat 1 is TARGET, which is the order the loop pushes
    them in (`compute_point_hist2_loop.mojo`), not alphabetical luck.
    """
    var want = zeros(fx.ob_line * 2)
    for i in range(len(offs)):
        for r in range(offs[i], offs[i] + lens[i]):
            var row = Int(fx.indices[r])
            for g in range(4):
                var word = fx.cindex[g * N_ROWS + row]
                for k in range(4):
                    var f = 4 * g + k
                    var b = Int((word >> UInt32(24 - 8 * k)) & 255)
                    var at = (Int(fx.ob_first[f]) + b) * 2
                    want[at + 0] += fx.weight[r]
                    want[at + 1] += fx.target[r]
    return want^


def tally_half_byte(
    fx: Fixture, offs: List[Int], lens: List[Int]
) -> List[Float32]:
    var want = zeros(fx.hb_line * 2)
    for i in range(len(offs)):
        for r in range(offs[i], offs[i] + lens[i]):
            var row = Int(fx.indices[r])
            for hbg in range(N_HB_BLOCKS):
                var word = fx.cindex[(HB_COL0 + hbg) * N_ROWS + row]
                for k in range(8):
                    var j = 8 * hbg + k
                    var b = Int((word >> UInt32(28 - 4 * k)) & 15)
                    var at = (Int(fx.hb_first[j]) + b) * 2
                    want[at + 0] += fx.weight[r]
                    want[at + 1] += fx.target[r]
    return want^


def tally_binary(
    fx: Fixture, offs: List[Int], lens: List[Int]
) -> List[Float32]:
    """Host histogram of the binary policy.

    A BINARY FEATURE'S ONE CELL IS ITS `bit == 0` SIDE, not its count. The
    accumulator builds a 16-bin histogram over each NIBBLE's value and the
    driver recovers the feature by summing the nibble values whose bit is
    clear (`pointwise_hist2_binary.mojo`), so the cell at
    `FirstFoldIndex * 2 + w` is the total over rows where the bit is 0.
    Tallying the rows where it is 1 instead would be a perfectly plausible
    reading and would be wrong at every cell.
    """
    var want = zeros(fx.b_line * 2)
    for i in range(len(offs)):
        for r in range(offs[i], offs[i] + lens[i]):
            var row = Int(fx.indices[r])
            for bg in range(N_B_BLOCKS):
                var word = fx.cindex[(B_COL0 + bg) * N_ROWS + row]
                for k in range(32):
                    var j = 32 * bg + k
                    var bit = Int((word >> UInt32(31 - k)) & 1)
                    if bit == 0:
                        var at = Int(fx.b_first[j]) * 2
                        want[at + 0] += fx.weight[r]
                        want[at + 1] += fx.target[r]
    return want^


def apply_scan(
    var raw: List[Float32],
    folds: List[UInt32],
    first: List[UInt32],
    one_hot: List[UInt8],
) -> List[Float32]:
    """What `ScanPointwiseHistograms` does, on the host.

    A running prefix per (feature, stat) over the feature's own folds,
    SKIPPING one-hot features and features with a single fold -- theirs at
    `split_properties_helpers.cuh:126` and the ported kernel's own guard.
    """
    for f in range(len(folds)):
        var nf = Int(folds[f])
        if one_hot[f] != UInt8(0) or nf <= 1:
            continue
        var base = Int(first[f]) * 2
        for s in range(2):
            var run = Float32(0.0)
            for b in range(nf):
                run += raw[base + b * 2 + s]
                raw[base + b * 2 + s] = run
    return raw^


def compare(
    name: String,
    got: List[Float32],
    want: List[Float32],
    slot: Int,
    line2: Int,
) -> Int:
    """Per cell, exact. Returns the number of wrong cells and prints the
    first few with enough context to locate them."""
    var bad = 0
    for k in range(line2):
        var g = got[slot * line2 + k]
        if g != want[k]:
            if bad < 4:
                print(
                    "     ", name, "cell", k, "(bin-feature", k // 2,
                    "stat", k % 2, "): got", g, "want", want[k],
                )
            bad += 1
    return bad


def main() raises:
    var ctx = DeviceContext()
    var failures = 0
    var fx = Fixture()

    var ob_hist = folds_histogram_from_folds(fx.ob_folds)

    # ---- device copies, shared by every gate ------------------------
    var d_idx = ctx.enqueue_create_buffer[DType.uint32](N_ROWS)
    var d_tgt = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var d_wt = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var d_ci = ctx.enqueue_create_buffer[DType.uint32](N_COLUMNS * N_ROWS)
    ctx.enqueue_copy(dst_buf=d_idx, src_ptr=fx.indices.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_tgt, src_ptr=fx.target.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_wt, src_ptr=fx.weight.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_ci, src_ptr=fx.cindex.unsafe_ptr())

    var d_ob_off = ctx.enqueue_create_buffer[DType.uint32](N_OB)
    var d_ob_first = ctx.enqueue_create_buffer[DType.uint32](N_OB)
    var d_ob_folds = ctx.enqueue_create_buffer[DType.uint32](N_OB)
    var d_ob_oh = ctx.enqueue_create_buffer[DType.uint8](N_OB)
    ctx.enqueue_copy(dst_buf=d_ob_off, src_ptr=fx.ob_offset.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_ob_first, src_ptr=fx.ob_first.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_ob_folds, src_ptr=fx.ob_folds.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_ob_oh, src_ptr=fx.ob_one_hot.unsafe_ptr())

    var d_hb_off = ctx.enqueue_create_buffer[DType.uint32](N_HB)
    var d_hb_first = ctx.enqueue_create_buffer[DType.uint32](N_HB)
    var d_hb_folds = ctx.enqueue_create_buffer[DType.uint32](N_HB)
    var d_hb_oh = ctx.enqueue_create_buffer[DType.uint8](N_HB)
    ctx.enqueue_copy(dst_buf=d_hb_off, src_ptr=fx.hb_offset.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_hb_first, src_ptr=fx.hb_first.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_hb_folds, src_ptr=fx.hb_folds.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_hb_oh, src_ptr=fx.hb_one_hot.unsafe_ptr())

    var d_b_off = ctx.enqueue_create_buffer[DType.uint32](N_B)
    var d_b_first = ctx.enqueue_create_buffer[DType.uint32](N_B)
    var d_b_folds = ctx.enqueue_create_buffer[DType.uint32](N_B)
    var d_b_oh = ctx.enqueue_create_buffer[DType.uint8](N_B)
    ctx.enqueue_copy(dst_buf=d_b_off, src_ptr=fx.b_offset.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_b_first, src_ptr=fx.b_first.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_b_folds, src_ptr=fx.b_folds.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_b_oh, src_ptr=fx.b_one_hot.unsafe_ptr())

    # =============================================================== F1
    # one part, offset 8, 23,800 rows: not the whole pool, so a launcher
    # that ignores `partition->Offset` reads a different set of rows.
    var p_off = 8
    var p_n = 23800
    var parts1: List[UInt32] = [UInt32(p_off), UInt32(p_n)]
    var d_parts1 = ctx.enqueue_create_buffer[DType.uint32](2)
    ctx.enqueue_copy(dst_buf=d_parts1, src_ptr=parts1.unsafe_ptr())

    var one_off: List[Int] = [p_off]
    var one_len: List[Int] = [p_n]
    var want_ob = apply_scan(
        tally_one_byte(fx, one_off, one_len),
        fx.ob_folds, fx.ob_first, fx.ob_one_hot,
    )
    var want_hb = apply_scan(
        tally_half_byte(fx, one_off, one_len),
        fx.hb_folds, fx.hb_first, fx.hb_one_hot,
    )
    var want_b = tally_binary(fx, one_off, one_len)  # binary is NOT scanned

    var d_ob = ctx.enqueue_create_buffer[DType.float32](fx.ob_line * 2)
    var d_hb = ctx.enqueue_create_buffer[DType.float32](fx.hb_line * 2)
    var d_b = ctx.enqueue_create_buffer[DType.float32](fx.b_line * 2)
    var h_ob = ctx.enqueue_create_host_buffer[DType.float32](fx.ob_line * 2)
    var h_hb = ctx.enqueue_create_host_buffer[DType.float32](fx.hb_line * 2)
    var h_b = ctx.enqueue_create_host_buffer[DType.float32](fx.b_line * 2)
    ctx.enqueue_memset(d_ob, Float32(0.0))
    ctx.enqueue_memset(d_hb, Float32(0.0))
    ctx.enqueue_memset(d_b, Float32(0.0))

    compute_hist2(
        ctx, POLICY_ONE_BYTE,
        d_ob_off.unsafe_ptr(), d_ob_first.unsafe_ptr(),
        d_ob_folds.unsafe_ptr(), d_ob_oh.unsafe_ptr(), N_OB,
        0, fx.ob_line,
        d_ci.unsafe_ptr(), d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
        d_idx.unsafe_ptr(), N_ROWS, d_parts1.unsafe_ptr(), 1, 1,
        d_ob.unsafe_ptr(), fx.ob_line, True, ob_hist, SM_COUNT, FIXED_SCALE,
    )
    compute_hist2(
        ctx, POLICY_HALF_BYTE,
        d_hb_off.unsafe_ptr(), d_hb_first.unsafe_ptr(),
        d_hb_folds.unsafe_ptr(), d_hb_oh.unsafe_ptr(), N_HB,
        0, fx.hb_line,
        d_ci.unsafe_ptr(), d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
        d_idx.unsafe_ptr(), N_ROWS, d_parts1.unsafe_ptr(), 1, 1,
        d_hb.unsafe_ptr(), fx.hb_line, True, FoldsHistogram(), SM_COUNT,
        FIXED_SCALE,
    )
    compute_hist2(
        ctx, POLICY_BINARY,
        d_b_off.unsafe_ptr(), d_b_first.unsafe_ptr(),
        d_b_folds.unsafe_ptr(), d_b_oh.unsafe_ptr(), N_B,
        0, fx.b_line,
        d_ci.unsafe_ptr(), d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
        d_idx.unsafe_ptr(), N_ROWS, d_parts1.unsafe_ptr(), 1, 1,
        d_b.unsafe_ptr(), fx.b_line, True, FoldsHistogram(), SM_COUNT,
        FIXED_SCALE,
    )
    ctx.enqueue_copy(dst_buf=h_ob, src_buf=d_ob)
    ctx.enqueue_copy(dst_buf=h_hb, src_buf=d_hb)
    ctx.enqueue_copy(dst_buf=h_b, src_buf=d_b)
    ctx.synchronize()

    var got_ob = List[Float32]()
    for k in range(fx.ob_line * 2):
        got_ob.append(h_ob[k])
    var got_hb = List[Float32]()
    for k in range(fx.hb_line * 2):
        got_hb.append(h_hb[k])
    var got_b = List[Float32]()
    for k in range(fx.b_line * 2):
        got_b.append(h_b[k])

    var bad = compare("F1 one-byte", got_ob, want_ob, 0, fx.ob_line * 2)
    bad += compare("F1 half-byte", got_hb, want_hb, 0, fx.hb_line * 2)
    bad += compare("F1 binary", got_b, want_b, 0, fx.b_line * 2)
    var n_cells = (fx.ob_line + fx.hb_line + fx.b_line) * 2
    if bad != 0:
        print(
            "FAIL F1: --", bad, "of", n_cells,
            "cells wrong across the three policies.",
        )
        failures += 1
    else:
        print(
            "  ok   F1 --", n_cells,
            "cells exact: 5/6/7/8-bit + half-byte + binary, one fixture,"
            " scanned",
        )

    # =============================================================== F2
    # M = 4 (one leaf) against M = 1 (64 leaves, 63 of them empty).
    var line2 = fx.ob_line * 2
    var m_small = estimate_block_per_feature_multiplier(
        1, 1, 1, N_ROWS, SM_COUNT
    )
    var m_big = estimate_block_per_feature_multiplier(
        1, 64, 1, N_ROWS, SM_COUNT
    )
    if m_small <= 1 or m_big != 1:
        print(
            "FAIL F2: the fixture does not separate the arms -- one leaf"
            " gives M =", m_small, "and 64 leaves give M =", m_big,
            ". Both arms must run for this gate to mean anything.",
        )
        failures += 1
    else:
        print(
            "  ok   F2 setup -- one leaf gives M =", m_small,
            "(atomicAdd writeback), 64 leaves give M =", m_big,
            "(plain store)",
        )

    var n_parts_wide = 64
    var parts64 = List[UInt32]()
    parts64.append(UInt32(p_off))
    parts64.append(UInt32(p_n))
    for _ in range(1, n_parts_wide):
        parts64.append(UInt32(0))
        parts64.append(UInt32(0))
    var d_parts64 = ctx.enqueue_create_buffer[DType.uint32](
        2 * n_parts_wide
    )
    ctx.enqueue_copy(dst_buf=d_parts64, src_ptr=parts64.unsafe_ptr())

    var wide = fx.ob_line * 2 * n_parts_wide
    var d_ob_wide = ctx.enqueue_create_buffer[DType.float32](wide)
    var h_ob_wide = ctx.enqueue_create_host_buffer[DType.float32](wide)
    ctx.enqueue_memset(d_ob_wide, Float32(0.0))
    compute_hist2(
        ctx, POLICY_ONE_BYTE,
        d_ob_off.unsafe_ptr(), d_ob_first.unsafe_ptr(),
        d_ob_folds.unsafe_ptr(), d_ob_oh.unsafe_ptr(), N_OB,
        0, fx.ob_line,
        d_ci.unsafe_ptr(), d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
        d_idx.unsafe_ptr(), N_ROWS, d_parts64.unsafe_ptr(),
        n_parts_wide, 1,
        d_ob_wide.unsafe_ptr(), fx.ob_line, True, ob_hist, SM_COUNT,
        FIXED_SCALE,
    )
    ctx.enqueue_copy(dst_buf=h_ob_wide, src_buf=d_ob_wide)
    ctx.synchronize()

    var got_wide = List[Float32]()
    for k in range(fx.ob_line * 2):
        got_wide.append(h_ob_wide[k])
    var bad2 = compare("F2 M=1", got_wide, want_ob, 0, fx.ob_line * 2)

    # and the two arms against EACH OTHER, which is the comparison the
    # multiplier is actually about
    var bad2b = 0
    for k in range(fx.ob_line * 2):
        if got_wide[k] != got_ob[k]:
            bad2b += 1
    if bad2 != 0 or bad2b != 0:
        print(
            "FAIL F2: --", bad2, "cells wrong at M = 1 against the host,",
            bad2b, "cells where M = 1 and M =", m_small, "disagree."
            " A short M > 1 result means the document blocks are not"
            " partitioning; a multiple means the atomic path was not"
            " taken and the blocks overwrote instead of adding.",
        )
        failures += 1
    else:
        print(
            "  ok   F2 --", fx.ob_line * 2,
            "cells exact at M = 1 and identical to the M =", m_small,
            "run cell for cell",
        )

    # ============================================================== F2c
    # THE MULTIPLIER'S VALUE, not just its consequences -- and this is the
    # only gate in the file that can see WHICH feature count sized the
    # estimate.
    #
    # `ComputeHist2NonBinary` computes `numBlocks.x` twice: once from
    # `featureCountForBits` to size the estimate, once from `nbCount` to
    # size the launch. Getting the LAUNCH wrong is loud -- F1 and F4 both go
    # red, because groups past the count are never reached. Getting the
    # ESTIMATE wrong is SILENT under every gate above: both choices produce
    # a self-consistent grid, so the histogram is correct either way and
    # only the block COUNT differs.
    #
    # The block count is observable through the writeback, which is the one
    # thing `M` changes about the answer: at `M == 1` the driver STORES and
    # at `M > 1` it ADDS. Seed the buffer with a non-zero pattern and the
    # two are different programs -- a store erases the seed, an atomic adds
    # to it.
    #
    # At `sm_count = 1` the two feature counts separate cleanly:
    #     estimate on ceil(4 / 4)  = 1 block   ->  M = 4   (adds)
    #     estimate on ceil(16 / 4) = 4 blocks  ->  M = 1   (stores)
    # so a launcher that sizes its estimate on `nbCount` fails here and
    # nowhere else.
    # ROUTE-AWARE (the fixed one-byte route, `pointwise_one_byte_fixed_for`):
    # under that route the shipped dispatch is ONE launch whose estimate
    # count EQUALS its launch count -- every one-byte feature -- so the
    # estimate-vs-launch discrimination this gate was built on does not
    # exist on such a build: sized either way, the grids agree. What stays
    # observable is the M-path itself: at M > 1 the writeback is an
    # atomicAdd and the seed survives, and a driver that lost the
    # multiplier would store and erase it. The discrimination arm still
    # runs as written on columns under their dispatch.
    comptime pw_routed = pointwise_one_byte_fixed_for[
        TARGET_COLUMN, GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    ]()
    var sm_2c: Int
    var m_2c: Int
    comptime if pw_routed:
        sm_2c = SM_COUNT
        m_2c = estimate_block_per_feature_multiplier(
            (ob_hist.feature_count_for_bits(4, 8) + 3) // 4, 1, 1,
            N_ROWS, sm_2c,
        )
        if m_2c <= 1:
            print(
                "FAIL F2c: the routed shape cannot force M > 1 -- the"
                " atomic writeback is then indistinguishable from a store"
                " and this gate holds nothing. M =", m_2c,
            )
            failures += 1
    else:
        sm_2c = 1
        m_2c = estimate_block_per_feature_multiplier(
            1, 1, 1, N_ROWS, sm_2c
        )
        var m_if_wrong = estimate_block_per_feature_multiplier(
            4, 1, 1, N_ROWS, sm_2c
        )
        if m_2c <= 1 or m_if_wrong != 1:
            print(
                "FAIL F2c: the shape does not separate the two feature"
                " counts -- sizing on featureCountForBits gives M =", m_2c,
                "and sizing on nbCount gives M =", m_if_wrong,
            )
            failures += 1

    var seed2 = zeros(line2)
    for k in range(line2):
        seed2[k] = Float32(k % 89 + 5)
    var raw_ob = tally_one_byte(fx, one_off, one_len)
    var want2c = zeros(line2)
    for k in range(line2):
        want2c[k] = seed2[k] + raw_ob[k]

    ctx.enqueue_copy(dst_buf=d_ob, src_ptr=seed2.unsafe_ptr())
    comptime if pw_routed:
        # The shipped dispatch under the route: one launch, every one-byte
        # feature, and the widened `pw_bounds[8]` claims all four groups --
        # so every cell still gets tallied exactly once and `want2c` is
        # the same value the four-launch arm builds.
        compute_hist2_non_binary[8](
            ctx, d_ob_off.unsafe_ptr(), d_ob_first.unsafe_ptr(),
            d_ob_folds.unsafe_ptr(), N_OB, d_ci.unsafe_ptr(),
            d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(), d_idx.unsafe_ptr(),
            N_ROWS, d_parts1.unsafe_ptr(), 1, 1, True, fx.ob_line,
            d_ob.unsafe_ptr(), ob_hist.feature_count_for_bits(4, 8),
            sm_2c, FIXED_SCALE,
        )
    else:
        compute_hist2_non_binary[5](
            ctx, d_ob_off.unsafe_ptr(), d_ob_first.unsafe_ptr(),
            d_ob_folds.unsafe_ptr(), N_OB, d_ci.unsafe_ptr(),
            d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(), d_idx.unsafe_ptr(),
            N_ROWS, d_parts1.unsafe_ptr(), 1, 1, True, fx.ob_line,
            d_ob.unsafe_ptr(), ob_hist.feature_count_for_bits(4, 5), sm_2c,
            FIXED_SCALE,
        )
        compute_hist2_non_binary[6](
            ctx, d_ob_off.unsafe_ptr(), d_ob_first.unsafe_ptr(),
            d_ob_folds.unsafe_ptr(), N_OB, d_ci.unsafe_ptr(),
            d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(), d_idx.unsafe_ptr(),
            N_ROWS, d_parts1.unsafe_ptr(), 1, 1, True, fx.ob_line,
            d_ob.unsafe_ptr(), ob_hist.feature_count_for_bits(6, 6), sm_2c,
            FIXED_SCALE,
        )
        compute_hist2_non_binary[7](
            ctx, d_ob_off.unsafe_ptr(), d_ob_first.unsafe_ptr(),
            d_ob_folds.unsafe_ptr(), N_OB, d_ci.unsafe_ptr(),
            d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(), d_idx.unsafe_ptr(),
            N_ROWS, d_parts1.unsafe_ptr(), 1, 1, True, fx.ob_line,
            d_ob.unsafe_ptr(), ob_hist.feature_count_for_bits(7, 7), sm_2c,
            FIXED_SCALE,
        )
        compute_hist2_non_binary[8](
            ctx, d_ob_off.unsafe_ptr(), d_ob_first.unsafe_ptr(),
            d_ob_folds.unsafe_ptr(), N_OB, d_ci.unsafe_ptr(),
            d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(), d_idx.unsafe_ptr(),
            N_ROWS, d_parts1.unsafe_ptr(), 1, 1, True, fx.ob_line,
            d_ob.unsafe_ptr(), ob_hist.feature_count_for_bits(8, 8), sm_2c,
            FIXED_SCALE,
        )
    ctx.enqueue_copy(dst_buf=h_ob, src_buf=d_ob)
    ctx.synchronize()

    var got2c = List[Float32]()
    for k in range(line2):
        got2c.append(h_ob[k])
    var bad2c = compare("F2c", got2c, want2c, 0, line2)
    if bad2c != 0:
        print(
            "FAIL F2c: --", bad2c, "of", line2,
            "cells wrong. A cell holding the raw tally WITHOUT the seed"
            " means the driver took the M == 1 store path, i.e. the"
            " multiplier estimate was sized on nbCount and not on"
            " featureCountForBits.",
        )
        failures += 1
    else:
        print(
            "  ok   F2c --", line2,
            "cells exact: the estimate is sized on featureCountForBits, so"
            " M =", m_2c, "here and the seed survives under the atomic",
        )

    # =============================================================== F3
    # four parts, two pairs, the smaller child on a different side in each
    var f3_off: List[Int] = [0, 6000, 15000, 22000]
    var f3_len: List[Int] = [6000, 9000, 7000, 2000]
    var parts4 = List[UInt32]()
    for p in range(4):
        parts4.append(UInt32(f3_off[p]))
        parts4.append(UInt32(f3_len[p]))
    var d_parts4 = ctx.enqueue_create_buffer[DType.uint32](8)
    ctx.enqueue_copy(dst_buf=d_parts4, src_ptr=parts4.unsafe_ptr())

    # each part's own scanned histogram, which is what must come back
    var per_part = List[List[Float32]]()
    for p in range(4):
        var o: List[Int] = [f3_off[p]]
        var n: List[Int] = [f3_len[p]]
        per_part.append(
            apply_scan(
                tally_one_byte(fx, o, n),
                fx.ob_folds, fx.ob_first, fx.ob_one_hot,
            )
        )

    # the level's INPUT: parents in the left slots, zero on the right
    var seed = zeros(line2 * 4)
    for pair in range(2):
        var lo = pair
        var hi = pair + 2
        for k in range(line2):
            seed[lo * line2 + k] = per_part[lo][k] + per_part[hi][k]

    var d_ob4 = ctx.enqueue_create_buffer[DType.float32](line2 * 4)
    var h_ob4 = ctx.enqueue_create_host_buffer[DType.float32](line2 * 4)
    ctx.enqueue_copy(dst_buf=d_ob4, src_ptr=seed.unsafe_ptr())
    compute_hist2(
        ctx, POLICY_ONE_BYTE,
        d_ob_off.unsafe_ptr(), d_ob_first.unsafe_ptr(),
        d_ob_folds.unsafe_ptr(), d_ob_oh.unsafe_ptr(), N_OB,
        0, fx.ob_line,
        d_ci.unsafe_ptr(), d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
        d_idx.unsafe_ptr(), N_ROWS, d_parts4.unsafe_ptr(), 4, 1,
        d_ob4.unsafe_ptr(), fx.ob_line, False, ob_hist, SM_COUNT,
        FIXED_SCALE,
    )
    ctx.enqueue_copy(dst_buf=h_ob4, src_buf=d_ob4)
    ctx.synchronize()

    var got4 = List[Float32]()
    for k in range(line2 * 4):
        got4.append(h_ob4[k])
    var bad3 = 0
    for p in range(4):
        bad3 += compare(
            "F3 part " + String(p), got4, per_part[p], p, line2
        )
    if bad3 != 0:
        print(
            "FAIL F3: --", bad3, "of", line2 * 4,
            "cells wrong on the partial pass. Pair 0's smaller child is on"
            " the LEFT and pair 1's is on the RIGHT, so a launcher that"
            " always takes one side gets exactly half of this right; a"
            " wrong `scanOffset` moves every cell of both pairs.",
        )
        failures += 1
    else:
        print(
            "  ok   F3 --", line2 * 4,
            "cells exact: smaller sibling, scan at the right-half offset,"
            " subtraction back into both slots",
        )

    # =============================================================== F4
    # each bit width alone; which groups did it write? Under the fixed
    # one-byte route the 8-bit kernel's bounds widen to (15, 256] and it
    # claims EVERY group -- that is the routing row working, not a leak.
    # The 5/6/7 kernels keep their own bounds in any build, so their
    # claims do not move.
    var expect_8: String
    comptime if pw_routed:
        expect_8 = String("0,1,2,3")
    else:
        expect_8 = String("3")
    var expect_claims: List[String] = [
        String("0"), String("1"), String("2"), expect_8,
    ]
    var widths: List[Int] = [5, 6, 7, 8]
    for wi in range(len(widths)):
        ctx.enqueue_memset(d_ob, Float32(0.0))
        var w = widths[wi]
        var fcb = ob_hist.feature_count_for_bits(4, 5) if w == 5 else (
            ob_hist.feature_count_for_bits(w, w)
        )
        if w == 5:
            compute_hist2_non_binary[5](
                ctx, d_ob_off.unsafe_ptr(), d_ob_first.unsafe_ptr(),
                d_ob_folds.unsafe_ptr(), N_OB, d_ci.unsafe_ptr(),
                d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(), d_idx.unsafe_ptr(),
                N_ROWS, d_parts1.unsafe_ptr(), 1, 1, True, fx.ob_line,
                d_ob.unsafe_ptr(), fcb, SM_COUNT, FIXED_SCALE,
            )
        elif w == 6:
            compute_hist2_non_binary[6](
                ctx, d_ob_off.unsafe_ptr(), d_ob_first.unsafe_ptr(),
                d_ob_folds.unsafe_ptr(), N_OB, d_ci.unsafe_ptr(),
                d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(), d_idx.unsafe_ptr(),
                N_ROWS, d_parts1.unsafe_ptr(), 1, 1, True, fx.ob_line,
                d_ob.unsafe_ptr(), fcb, SM_COUNT, FIXED_SCALE,
            )
        elif w == 7:
            compute_hist2_non_binary[7](
                ctx, d_ob_off.unsafe_ptr(), d_ob_first.unsafe_ptr(),
                d_ob_folds.unsafe_ptr(), N_OB, d_ci.unsafe_ptr(),
                d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(), d_idx.unsafe_ptr(),
                N_ROWS, d_parts1.unsafe_ptr(), 1, 1, True, fx.ob_line,
                d_ob.unsafe_ptr(), fcb, SM_COUNT, FIXED_SCALE,
            )
        else:
            compute_hist2_non_binary[8](
                ctx, d_ob_off.unsafe_ptr(), d_ob_first.unsafe_ptr(),
                d_ob_folds.unsafe_ptr(), N_OB, d_ci.unsafe_ptr(),
                d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(), d_idx.unsafe_ptr(),
                N_ROWS, d_parts1.unsafe_ptr(), 1, 1, True, fx.ob_line,
                d_ob.unsafe_ptr(), fcb, SM_COUNT, FIXED_SCALE,
            )
        ctx.enqueue_copy(dst_buf=h_ob, src_buf=d_ob)
        ctx.synchronize()
        var claimed = String("")
        for g in range(4):
            var touched = False
            for k in range(4):
                var f = 4 * g + k
                var b0 = Int(fx.ob_first[f]) * 2
                for j in range(Int(fx.ob_folds[f]) * 2):
                    if h_ob[b0 + j] != 0.0:
                        touched = True
            if touched:
                if claimed.byte_length() > 0:
                    claimed += ","
                claimed += String(g)
        if claimed != expect_claims[wi]:
            print(
                "FAIL F4: the", w, "-bit launch claimed groups [", claimed,
                "], expected [", expect_claims[wi],
                "]. An empty claim means the kernel ran over every block and"
                " returned from all of them.",
            )
            failures += 1
        else:
            print(
                "  ok   F4", w, "-bit writes group [", claimed,
                "] and nothing else (featureCountForBits =", fcb, ")",
            )

    # =============================================================== F5
    # IsGridEmpty, both ways in
    var sentinel = zeros(fx.hb_line * 2)
    for k in range(fx.hb_line * 2):
        sentinel[k] = Float32(k % 97 + 3)
    var d_sent = ctx.enqueue_create_buffer[DType.float32](fx.hb_line * 2)
    var h_sent = ctx.enqueue_create_host_buffer[DType.float32](
        fx.hb_line * 2
    )

    # (a) partial pass at one part: every `numBlocks.y` is `1 / 2 == 0`
    ctx.enqueue_copy(dst_buf=d_sent, src_ptr=sentinel.unsafe_ptr())
    compute_hist2(
        ctx, POLICY_HALF_BYTE,
        d_hb_off.unsafe_ptr(), d_hb_first.unsafe_ptr(),
        d_hb_folds.unsafe_ptr(), d_hb_oh.unsafe_ptr(), N_HB,
        0, fx.hb_line,
        d_ci.unsafe_ptr(), d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
        d_idx.unsafe_ptr(), N_ROWS, d_parts1.unsafe_ptr(), 1, 1,
        d_sent.unsafe_ptr(), fx.hb_line, False, FoldsHistogram(), SM_COUNT,
        FIXED_SCALE,
    )
    ctx.enqueue_copy(dst_buf=h_sent, src_buf=d_sent)
    ctx.synchronize()
    var bad5 = 0
    for k in range(fx.hb_line * 2):
        if h_sent[k] != sentinel[k]:
            bad5 += 1

    # (b) a policy with no features: `numBlocks.x` is zero
    ctx.enqueue_copy(dst_buf=d_sent, src_ptr=sentinel.unsafe_ptr())
    compute_hist2(
        ctx, POLICY_BINARY,
        d_b_off.unsafe_ptr(), d_b_first.unsafe_ptr(),
        d_b_folds.unsafe_ptr(), d_b_oh.unsafe_ptr(), 0,
        0, 0,
        d_ci.unsafe_ptr(), d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
        d_idx.unsafe_ptr(), N_ROWS, d_parts1.unsafe_ptr(), 1, 1,
        d_sent.unsafe_ptr(), fx.hb_line, True, FoldsHistogram(), SM_COUNT,
        FIXED_SCALE,
    )
    ctx.enqueue_copy(dst_buf=h_sent, src_buf=d_sent)
    ctx.synchronize()
    var bad5b = 0
    for k in range(fx.hb_line * 2):
        if h_sent[k] != sentinel[k]:
            bad5b += 1

    if bad5 != 0 or bad5b != 0:
        print(
            "FAIL F5: an empty grid launched anyway --", bad5,
            "cells moved with numBlocks.y == 0 and", bad5b,
            "with numBlocks.x == 0.",
        )
        failures += 1
    else:
        print(
            "  ok   F5 -- an empty grid launches nothing, on both axes"
            " (buffer bit-identical to the sentinel)",
        )

    # =============================================================== F6
    var probe: List[UInt32] = [1, 2, 3, 15, 16, 17, 32, 33, 64, 128, 255, 256]
    var ph = folds_histogram_from_folds(probe)
    #  1->0  2->1  3->2  15->4  16->4  17->5  32->5  33->6
    # 64->6  128->7  255->8  256->8
    var want_counts: List[UInt32] = [1, 1, 1, 0, 2, 2, 2, 1, 2]
    var bad6 = 0
    for b in range(9):
        if ph.counts[b] != want_counts[b]:
            print(
                "     F6 bit", b, ": got", Int(ph.counts[b]), "want",
                Int(want_counts[b]),
            )
            bad6 += 1
    # the 5-bit kernel's range is 4..5, NOT 5..5
    if ph.feature_count_for_bits(4, 5) != 4:
        print(
            "     F6 FeatureCountForBits(4, 5) =",
            ph.feature_count_for_bits(4, 5),
            "want 4 -- the 5-bit kernel claims bit FOUR as well, because"
            " `lowerBound = BITS > 5 ? upperBound / 2 : 15` sends 16-fold"
            " features to it",
        )
        bad6 += 1
    if bad6 != 0:
        print("FAIL F6: --", bad6, "folds-to-bits bins wrong")
        failures += 1
    else:
        print(
            "  ok   F6 -- folds bin to bits at the 16/17 boundary and the"
            " 5-bit range covers bits 4 and 5",
        )

    # =============================================================== F7
    var allowed: List[Int] = [1, 2, 4, 8, 16, 32, 64]
    var saw_128 = False
    var bad7 = 0
    var seen_over_64 = 0
    var grids: List[Int] = [1, 2, 3, 7, 16, 64, 300]
    var sizes: List[Int] = [0, 1, 9999, 10001, 40000, 2000000, 100000000]
    var sms: List[Int] = [1, 8, 16, 80, 132]
    for gi in range(len(grids)):
        for si in range(len(sizes)):
            for mi in range(len(sms)):
                var m = estimate_block_per_feature_multiplier(
                    grids[gi], 1, 1, sizes[si], sms[mi]
                )
                if m == 128:
                    saw_128 = True
                if m > PW_MAX_MULTIPLIER:
                    seen_over_64 += 1
                    m = PW_MAX_MULTIPLIER
                var ok = False
                for ai in range(len(allowed)):
                    if m == allowed[ai]:
                        ok = True
                if not ok:
                    print(
                        "     F7 grid", grids[gi], "size", sizes[si], "sm",
                        sms[mi], "-> clamped multiplier", m,
                        "which has no kernel instantiation",
                    )
                    bad7 += 1
    if not saw_128:
        print(
            "     F7 the sweep never produced 128, so the `min(..., 64)`"
            " clamp is untested and their `exit(1)` arm may be reachable"
            " after all",
        )
        bad7 += 1
    if bad7 != 0:
        print("FAIL F7: --", bad7, "problems in the multiplier domain")
        failures += 1
    else:
        print(
            "  ok   F7 -- every multiplier over", len(grids) * len(sizes)
            * len(sms), "configurations lands on one of the seven"
            " instantiations;", seen_over_64,
            "of them needed the 64 clamp to get there",
        )

    _ = d_idx^
    _ = d_tgt^
    _ = d_wt^
    _ = d_ci^
    _ = d_ob_off^
    _ = d_ob_first^
    _ = d_ob_folds^
    _ = d_ob_oh^
    _ = d_hb_off^
    _ = d_hb_first^
    _ = d_hb_folds^
    _ = d_hb_oh^
    _ = d_b_off^
    _ = d_b_first^
    _ = d_b_folds^
    _ = d_b_oh^
    _ = d_parts1^
    _ = d_parts4^
    _ = d_parts64^
    _ = d_ob^
    _ = d_hb^
    _ = d_b^
    _ = d_ob_wide^
    _ = d_ob4^
    _ = d_sent^

    if failures != 0:
        raise Error(String(failures) + " gate(s) failed")
    print("pointwise dispatch: F1-F7 pass")
