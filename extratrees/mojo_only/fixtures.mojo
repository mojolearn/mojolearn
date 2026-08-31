# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Fixtures for the histogram-free ExtraTrees lane: hashed adversarial data and
analytic datasets whose correct split is hand-computable for EVERY admissible
threshold.

    Verified by:  extratrees/mojo_only/fixtures_check.mojo

WHY THIS FILE EXISTS AT ALL
---------------------------
ExtraTrees draws ONE UNIFORM RANDOM THRESHOLD per candidate feature inside that
feature's observed range (sklearn `RandomSplitter`, `_splitter.pyx:507-720`).
A random threshold means an expected value cannot be written down for "the"
split -- there is no "the" split. So a fixture is analytic ONLY IF the answer
is IDENTICAL for every threshold in a known interval. Every analytic fixture
below is built by putting an EMPTY GAP in the data and declaring the interval
that gap spans: inside it the partition cannot move, so counts, sums and the
criterion score are constants that can be written in this docstring.

That is also the property `fixtures_check.mojo` BRUTE-FORCES rather than
asserts: it sweeps the interval, and -- as a negative control -- sweeps outside
it and requires the answer to CHANGE. A property that holds everywhere is not
a property.

THE REFERENCE, read at tag 1.9.0 (77def0e), not paraphrased from memory
-----------------------------------------------------------------------
* `_partitioner.pxd:13`  `const float32_t FEATURE_THRESHOLD = 1e-7`
* `_partitioner.pyx:129-165`  `find_min_max`: min/max over the node's rows for
  one feature, NaNs counted into `n_missing` and excluded from min/max.
* `_splitter.pyx:617-618`  the CONSTANT test, verbatim:

      max_feature_value <= min_feature_value + FEATURE_THRESHOLD and n_missing == 0

  Both operands are `float32_t`, so the addition rounds in FLOAT32. That
  rounding is not cosmetic: at min == 0.5 the float32 sum `0.5 + 1e-7f` rounds
  up to `0.5 + 2 ulp`, which silently widens the constant band. Every
  near-constant column here is therefore anchored at min == 0.0, where
  `0.0 + FEATURE_THRESHOLD` is EXACT and the boundary is the boundary.
* `_splitter.pyx:633-637`  threshold ~ `rand_uniform(min, max)`, which is
  `[low, high)` (`_utils.pyx:57-61`), then `if threshold == max: threshold = min`.
* `_partitioner.pyx:218-238`  `partition_samples`: `feature_values[p] <= threshold`
  goes LEFT. `threshold` is float64, the values float32, so the comparison is
  exact.
* `_criterion.pyx:147-163`  the CLASSIFICATION proxy is the BASE one (Gini does
  not override it):  `proxy = -n_L * gini_L - n_R * gini_R`.
* `_criterion.pyx:944-973`  the REGRESSION proxy:  `proxy = sum_L^2/n_L + sum_R^2/n_R`.
* `_criterion.pyx` Gini `children_impurity`: `gini = 1 - sum_k c_k^2 / n^2`.
* MSE `children_impurity`: `impurity = sum_sq/n - (sum/n)^2`.

WHAT IS IN HERE
---------------
1. `hashed_classification` / `hashed_regression` -- n_rows x n_cols of values
   derived from a splitmix64 hash of `(seed, row, col, salt)`. SCATTERED, and
   essentially all distinct. This shape is not decoration: a UNIFORM fixture in
   this repository once reported 0 wrong of 512 on a kernel a HASHED fixture
   showed to be 490 wrong of 512, with two separate checks having certified
   that kernel correct at exactly the failing parameters. No `i % k`, no
   constants, no value repeated across cells.
2. `shaped_dataset` -- one column per requested SHAPE, the shapes that break a
   range/threshold implementation: exactly constant; near-constant just below,
   exactly at, and just above FEATURE_THRESHOLD; two-valued; a single far
   outlier over a dense cluster; all-negative; spanning zero; all-equal-but-one.
3. Four ANALYTIC fixtures, each with its closed form stated below and its
   expected quantities returned as DATA (`AnalyticFixture`), not as prose, so a
   later check can compare per cell.
4. `partition` / `feature_min_max` / `is_constant_feature` -- the reference
   predicates transcribed from the files cited above, so every check in this
   lane computes "what sklearn would say" the same way.

THE ANALYTIC FIXTURES, IN CLOSED FORM
-------------------------------------

**`analytic_separable_gap(seed)`** -- classification, 256 rows, 2 features,
2 classes. Rows `0..127` are class 0 with `x0 in [0, 1)`; rows `128..255` are
class 1 with `x0 in (9, 10]`. NOTHING lies in `[1, 9]`. Feature 1 is pure
hashed noise, uncorrelated with the label.

    For EVERY threshold t in the CLOSED interval [1.0, 9.0]:
        left  = {x0 <= t} = exactly the 128 class-0 rows
        right = {x0 >  t} = exactly the 128 class-1 rows
        n_left = 128, n_right = 128
        left_class_counts  = [128, 0]
        right_class_counts = [0, 128]
        gini_left = 1 - 128^2/128^2 = 0.0     gini_right = 0.0
        proxy = -128*0 - 128*0 = 0.0          (the maximum a Gini proxy can be)

    Outside it the answer MOVES: t = 0.5 splits the class-0 band, t = 9.5
    splits the class-1 band. Both are checked as negative controls.
    The CHOSEN FEATURE is also determined: feature 1 is verified by exhaustive
    sweep over its own distinct values to score strictly below 0.0 everywhere,
    so feature 0 wins for every draw, not just usually.

**`analytic_regression_step(seed)`** -- regression, 256 rows, 2 features.
`y = a = 2.0` for the 128 rows with `x0 in [0, 4)`, and `y = b = -3.0` for the
128 rows with `x0 in (6, 10]`. The band `[4, 6]` is empty. Feature 1 is noise.

    For EVERY threshold t in the CLOSED interval [4.0, 6.0]:
        n_left = 128, n_right = 128
        left_sum    = 128 *  2.0 =  256.0     left_sum_sq  = 128 * 4.0 =  512.0
        right_sum   = 128 * -3.0 = -384.0     right_sum_sq = 128 * 9.0 = 1152.0
        impurity_left  = 512/128  - (256/128)^2  = 4 - 4 = 0.0
        impurity_right = 1152/128 - (-384/128)^2 = 9 - 9 = 0.0
        proxy = 256^2/128 + (-384)^2/128 = 512 + 1152 = 1664.0

**`analytic_tie_pair(seed)`** -- classification, 256 rows, 3 features. Feature 0
is the separable-gap feature; feature 1 is a BIT-FOR-BIT COPY of feature 0;
feature 2 is noise. Two features that are exact copies score EQUAL at every
corresponding threshold, so the TIE-BREAK path is the only thing that can
decide, and it must decide the same way every run. Expected quantities are
those of `analytic_separable_gap`.

**`analytic_all_constant()`** -- 128 rows, 4 features, every column exactly
constant (0.0, -1.0, 3.25, 1e6). `splittable` is False: sklearn marks all four
features constant and `node_split_random` leaves `best_split.pos == end`, i.e.
NO SPLIT. This is the degenerate case a builder silently gets wrong -- it is
the one fixture whose correct answer is "refuse".

DETERMINISM CONTRACT
--------------------
Everything here is a pure function of an integer seed and integer indices, in
host-side Mojo, with no file I/O, no Python, and no RNG object. A device check
and a host check can generate the identical bytes independently. The check
proves this by comparing FLOAT BIT PATTERNS, never decimal strings: in this
toolchain `String(Float32)` does not round-trip for 0.46% of values.

DEVIATION BLOCK 149 -- fixtures are counter-based, not stream-based
-------------------------------------------------------------------
**Theirs.** sklearn's own tests build data with `numpy.random.RandomState`, a
SEQUENTIAL MT19937 stream: value `k` depends on every value drawn before it.

**Ours.** Every value is `splitmix64` of `(seed, row, col, salt)` -- a
COUNTER-BASED hash. Cell `(r, c)` depends on nothing but its own coordinates.

**Reason.** A sequential stream cannot be reproduced by a parallel generator
without materializing the whole stream in traversal order, so a device-side
fixture and a host-side fixture would have to agree on an ordering that the
device does not have. Counter-based keying makes "the same bytes" a property of
the coordinates instead of the schedule. It is the same discipline `PLAN.md`
records for the threshold draws themselves, applied to the data.

**Price.** These fixtures are NOT byte-comparable with any fixture a
`RandomState` seed would produce, so no test here can be cross-checked against
a stored numpy array. Nothing in this lane needs that; the analytic fixtures
are checked against a closed form and the hashed ones against their own claimed
statistical properties.

DEVIATION BLOCK 150 -- no fixture contains a missing value
-----------------------------------------------------------
**Theirs.** `find_min_max` counts NaNs into `n_missing` (`_partitioner.pyx:152-155`),
the constant test is `... and n_missing == 0` (`_splitter.pyx:617`), and missing
values are sent left or right by a per-candidate coin flip (`_splitter.pyx:649`).

**Ours.** No fixture in this file contains a NaN, because DEVIATION 136 refuses
missing values by name at the API boundary rather than randomizing them.

**Reason.** A fixture for a path that is refused would be a fixture for code
that does not exist. Manufacturing one now would freeze a guess at how the coin
flip gets keyed.

**Price.** The `n_missing == 0` conjunct of the constant test HAS NO FIXTURE
COVERAGE HERE, and neither does `missing_go_to_left`. If DEVIATION 136 is ever
reversed, this file must grow a NaN shape and this block must be rewritten,
not annotated. Tracked with 136 in `UNPORTED.tsv`.
"""

from std.memory import bitcast


# ----------------------------------------------------------------------------
# The reference constant. `_partitioner.pxd:13`, and it is a float32 there, so
# it is a Float32 here: the whole point of the near-constant shapes is that the
# comparison happens in float32 and the width of the band is a float32 fact.
# ----------------------------------------------------------------------------
comptime FEATURE_THRESHOLD: Float32 = 1e-7

# Column shapes for `shaped_dataset`.
comptime SHAPE_HASHED = 0
comptime SHAPE_CONSTANT = 1
comptime SHAPE_NEAR_CONST_BELOW = 2
comptime SHAPE_NEAR_CONST_EQUAL = 3
comptime SHAPE_NEAR_CONST_ABOVE = 4
comptime SHAPE_TWO_VALUED = 5
comptime SHAPE_OUTLIER = 6
comptime SHAPE_NEGATIVE = 7
comptime SHAPE_SPANS_ZERO = 8
comptime SHAPE_ONE_ODD_ROW = 9
comptime N_SHAPES = 10

# Salts. Distinct per generated PLANE so that x, y and label of the same cell
# are independent draws rather than three views of one hash.
comptime SALT_X: UInt64 = 0x00000000_00000001
comptime SALT_Y: UInt64 = 0x00000000_00000002
comptime SALT_LABEL: UInt64 = 0x00000000_00000003


# ----------------------------------------------------------------------------
# Hashing. splitmix64, chosen because it is short enough to audit right here
# and is already the generator this repository uses elsewhere
# (`cluster/ported/cluster/detail/kmeans.mojo:132`).
# ----------------------------------------------------------------------------
def splitmix64(x: UInt64) -> UInt64:
    """splitmix64.

    NOTE, and it cost a real bug in this file before the first run: `&+` and
    `&*` are NOT wrapping-arithmetic operators in Mojo 1.0. `&*` is a parse
    error ("can't use starred expression here") and `&+` SILENTLY PARSES AS
    BITWISE AND applied to a unary plus, so `x &+ K` computes `x & K`. Written
    that way, this hash masked `row` and `col` instead of mixing them: rows 0
    and 2 produced identical values and every column was identical to every
    other column whose index differed only in a bit `K` happened to have clear.
    Plain `+` and `*` on `UInt64` already wrap (`UInt64.MAX + 1 == 0`, measured),
    so plain operators are the correct and only spelling.
    """
    var z = x + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def cell_hash(seed: UInt64, row: Int, col: Int, salt: UInt64) -> UInt64:
    """A hash of the CELL, not of a position in a stream.

    Mixed SEQUENTIALLY rather than by xor-ing `row * c1 ^ col * c2`, because an
    xor of two multiplied indices has structure: it is linear over GF(2) in the
    low bits and collides in patterns. Four splitmix rounds cost nothing at
    fixture sizes and leave nothing to argue about.
    """
    var h = splitmix64(seed + 0x9E3779B97F4A7C15)
    h = splitmix64(h ^ (UInt64(row) + 0x165667B19E3779F9))
    h = splitmix64(h ^ (UInt64(col) + 0x27D4EB2F165667C5))
    return splitmix64(h ^ salt)


def unit_float(h: UInt64) -> Float32:
    """`[0, 1)` on the 2^24 EXACTLY REPRESENTABLE grid `k / 2^24`.

    24 bits because that is float32's significand: every distinct `k` gives a
    distinct float32, so distinctness of the values is exactly distinctness of
    the hash and the scatter check measures the hash rather than the rounding.
    """
    return Float32(Float64(h >> 40) / 16777216.0)


def unit_float_open(h: UInt64) -> Float32:
    """`(0, 1]` on the same grid: `(k + 1) / 2^24`. Used for the UPPER band of
    a gap fixture, so that the band's minimum is STRICTLY above the gap's top
    endpoint and the invariant interval can be stated CLOSED."""
    return Float32(Float64((h >> 40) + 1) / 16777216.0)


def unit_float20_open(h: UInt64) -> Float32:
    """`(0, 1]` on the COARSER 2^20 grid: `(k + 1) / 2^20`.

    This exists because of a real trap. The upper band of a gap fixture is
    `base + u` with `base` around 9 or 6, where float32's ulp is 2^-20 or
    2^-21 -- far coarser than the 2^-24 grid `unit_float` lives on. `9.0f +
    2^-24` ROUNDS BACK TO 9.0f, which would put a row exactly ON the gap's
    upper endpoint and quietly destroy the closed-interval claim. A 2^-20 grid
    is coarser than the ulp everywhere in `[6, 16)`, so `base + u` is EXACT and
    strictly greater than `base` by construction rather than by luck.
    """
    return Float32(Float64((h >> 44) + 1) / 1048576.0)


def signed_unit(h: UInt64) -> Float32:
    """`[-1, 1)` on the grid `k / 2^23 - 1`.

    `2 * u - 1` is EXACT in float32 for `u` on the `k / 2^24` grid (the
    subtraction is Sterbenz-exact for `u >= 0.5`, and for `u < 0.5` the result
    lands in `(-1, -0.5]` whose ulp is 2^-24, which `u` already respects). So
    distinct hashes still give distinct values, and the column spans zero.
    """
    return 2.0 * unit_float(h) - 1.0


def float_from_bits(b: UInt32) -> Float32:
    return bitcast[DType.float32](b)


def next_up(x: Float32) -> Float32:
    """Next float32 above `x`. Valid for POSITIVE finite `x`, which is all this
    file needs; incrementing the bit pattern of a positive float steps to the
    adjacent representable value by construction of IEEE-754's ordering."""
    return float_from_bits(x.to_bits[DType.uint32]() + 1)


def next_down(x: Float32) -> Float32:
    """Next float32 below `x`. Valid for POSITIVE finite `x` strictly above the
    smallest subnormal."""
    return float_from_bits(x.to_bits[DType.uint32]() - 1)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@fieldwise_init
struct Dataset(Copyable, Movable):
    """Row-major features plus BOTH targets. A fixture carries a class label and
    a regression target at once so a classification check and a regression check
    can be run over the SAME feature bytes; a check uses whichever it needs."""

    var n_rows: Int
    var n_cols: Int
    var n_classes: Int
    var x: List[Float32]
    var y: List[Float32]
    var label: List[Int32]

    def value(self, row: Int, col: Int) -> Float32:
        return self.x[row * self.n_cols + col]

    def bits(self, row: Int, col: Int) -> UInt32:
        return self.x[row * self.n_cols + col].to_bits[DType.uint32]()


@fieldwise_init
struct MinMax(Copyable, Movable):
    var min: Float32
    var max: Float32


@fieldwise_init
struct PartitionStats(Copyable, Movable):
    """Everything both criteria need from one partition, computed the way
    `_criterion.pyx` computes it: counts and sums in float64."""

    var n_left: Int
    var n_right: Int
    var left_class_counts: List[Int]
    var right_class_counts: List[Int]
    var left_sum: Float64
    var right_sum: Float64
    var left_sum_sq: Float64
    var right_sum_sq: Float64

    def gini_left(self) -> Float64:
        if self.n_left == 0:
            return 0.0
        var sq = Float64(0.0)
        for i in range(len(self.left_class_counts)):
            var c = Float64(self.left_class_counts[i])
            sq += c * c
        var n = Float64(self.n_left)
        return 1.0 - sq / (n * n)

    def gini_right(self) -> Float64:
        if self.n_right == 0:
            return 0.0
        var sq = Float64(0.0)
        for i in range(len(self.right_class_counts)):
            var c = Float64(self.right_class_counts[i])
            sq += c * c
        var n = Float64(self.n_right)
        return 1.0 - sq / (n * n)

    def gini_proxy(self) -> Float64:
        """`_criterion.pyx:147-163`. Gini does NOT override the base proxy, so
        this is `-n_R * gini_R - n_L * gini_L`, not the sum-of-squares form."""
        return (
            -Float64(self.n_right) * self.gini_right()
            - Float64(self.n_left) * self.gini_left()
        )

    def mse_left(self) -> Float64:
        if self.n_left == 0:
            return 0.0
        var n = Float64(self.n_left)
        var m = self.left_sum / n
        return self.left_sum_sq / n - m * m

    def mse_right(self) -> Float64:
        if self.n_right == 0:
            return 0.0
        var n = Float64(self.n_right)
        var m = self.right_sum / n
        return self.right_sum_sq / n - m * m

    def mse_proxy(self) -> Float64:
        """`_criterion.pyx:944-973`: `sum_L^2/n_L + sum_R^2/n_R`."""
        var out = Float64(0.0)
        if self.n_left > 0:
            out += self.left_sum * self.left_sum / Float64(self.n_left)
        if self.n_right > 0:
            out += self.right_sum * self.right_sum / Float64(self.n_right)
        return out


# ----------------------------------------------------------------------------
# The reference predicates, transcribed from the cited lines
# ----------------------------------------------------------------------------
def feature_min_max(data: Dataset, col: Int) -> MinMax:
    """`_partitioner.pyx:129-165`, over ALL rows (a fixture is one node).

    No NaN branch: see DEVIATION BLOCK 150.
    """
    var lo = data.value(0, col)
    var hi = lo
    for r in range(1, data.n_rows):
        var v = data.value(r, col)
        if v < lo:
            lo = v
        elif v > hi:
            hi = v
    return MinMax(lo, hi)


def is_constant_feature(data: Dataset, col: Int) -> Bool:
    """`_splitter.pyx:617` verbatim, minus the `n_missing == 0` conjunct that
    DEVIATION BLOCK 150 records as uncovered. The addition is deliberately left
    in FLOAT32 -- that is what the reference does and the whole near-constant
    family exists to pin it."""
    var mm = feature_min_max(data, col)
    return mm.max <= mm.min + FEATURE_THRESHOLD


def partition(data: Dataset, col: Int, threshold: Float64) -> PartitionStats:
    """`_partitioner.pyx:218-238`: `value <= threshold` goes LEFT.

    `threshold` is Float64 as it is in the reference (`current_split.threshold`
    is `float64_t`); float32 values widen exactly, so the comparison is exact.
    """
    var lc = List[Int]()
    var rc = List[Int]()
    for _ in range(data.n_classes):
        lc.append(0)
        rc.append(0)
    var n_left = 0
    var n_right = 0
    var ls = Float64(0.0)
    var rs = Float64(0.0)
    var lss = Float64(0.0)
    var rss = Float64(0.0)
    for r in range(data.n_rows):
        var v = Float64(data.value(r, col))
        var yv = Float64(data.y[r])
        var k = Int(data.label[r])
        if v <= threshold:
            n_left += 1
            ls += yv
            lss += yv * yv
            if k >= 0 and k < data.n_classes:
                lc[k] += 1
        else:
            n_right += 1
            rs += yv
            rss += yv * yv
            if k >= 0 and k < data.n_classes:
                rc[k] += 1
    return PartitionStats(n_left, n_right, lc^, rc^, ls, rs, lss, rss)


# ----------------------------------------------------------------------------
# 1. Hashed adversarial fixtures
# ----------------------------------------------------------------------------
def hashed_classification(
    seed: UInt64, n_rows: Int, n_cols: Int, n_classes: Int
) -> Dataset:
    """`n_rows x n_cols` of hashed float32 in `[-1, 1)` plus hashed class labels.

    The labels are drawn from their OWN salt, so they are not a function of any
    feature and -- the property the check enforces -- not a function of the row
    index either. `y` is filled with the label cast to float so a regression
    reader of this same fixture still sees a defined target.
    """
    var x = List[Float32]()
    var y = List[Float32]()
    var lab = List[Int32]()
    for r in range(n_rows):
        for c in range(n_cols):
            x.append(signed_unit(cell_hash(seed, r, c, SALT_X)))
        var k = Int((cell_hash(seed, r, 0, SALT_LABEL) >> 40) % UInt64(n_classes))
        lab.append(Int32(k))
        y.append(Float32(k))
    return Dataset(n_rows, n_cols, n_classes, x^, y^, lab^)


def hashed_regression(seed: UInt64, n_rows: Int, n_cols: Int) -> Dataset:
    """Same feature plane, but the target is a hashed float32 in `[-1, 1)` from
    its own salt. `n_classes` is 1 and every label is 0, so a classification
    reader sees a single-class node rather than garbage."""
    var x = List[Float32]()
    var y = List[Float32]()
    var lab = List[Int32]()
    for r in range(n_rows):
        for c in range(n_cols):
            x.append(signed_unit(cell_hash(seed, r, c, SALT_X)))
        y.append(signed_unit(cell_hash(seed, r, 0, SALT_Y)))
        lab.append(Int32(0))
    return Dataset(n_rows, n_cols, 1, x^, y^, lab^)


# ----------------------------------------------------------------------------
# 2. Adversarial column shapes
# ----------------------------------------------------------------------------
def shape_name(shape: Int) -> String:
    if shape == SHAPE_HASHED:
        return "hashed"
    if shape == SHAPE_CONSTANT:
        return "constant"
    if shape == SHAPE_NEAR_CONST_BELOW:
        return "near_const_below"
    if shape == SHAPE_NEAR_CONST_EQUAL:
        return "near_const_equal"
    if shape == SHAPE_NEAR_CONST_ABOVE:
        return "near_const_above"
    if shape == SHAPE_TWO_VALUED:
        return "two_valued"
    if shape == SHAPE_OUTLIER:
        return "outlier"
    if shape == SHAPE_NEGATIVE:
        return "negative"
    if shape == SHAPE_SPANS_ZERO:
        return "spans_zero"
    if shape == SHAPE_ONE_ODD_ROW:
        return "one_odd_row"
    return "unknown"


def all_shapes() -> List[Int]:
    """The canonical shape list, one column each, in shape order."""
    var s = List[Int]()
    for i in range(N_SHAPES):
        s.append(i)
    return s^


def shape_expects_constant(shape: Int) -> Bool:
    """What `is_constant_feature` MUST answer for this shape. Stated here so the
    check compares against a declaration rather than re-deriving the same
    reasoning it is supposed to be testing."""
    return (
        shape == SHAPE_CONSTANT
        or shape == SHAPE_NEAR_CONST_BELOW
        or shape == SHAPE_NEAR_CONST_EQUAL
    )


def shaped_value(seed: UInt64, shape: Int, row: Int, col: Int, n_rows: Int) -> Float32:
    """One cell of a shaped column.

    The two-valued shapes pick their value from a HASH BIT, so the two values
    are SCATTERED through the column rather than blocked -- a blocked layout
    passes a wrong partition that a scattered one catches. Rows 0 and 1 are then
    PINNED to the two values so that both are guaranteed present and the
    column's min and max are a closed form rather than a probability.
    """
    var h = cell_hash(seed, row, col, SALT_X)

    if shape == SHAPE_HASHED:
        return signed_unit(h)

    if shape == SHAPE_CONSTANT:
        # 3.25 = 13/4, exact in float32, and not 0 or 1 so a zeroed or
        # unwritten buffer does not impersonate it.
        return 3.25

    if (
        shape == SHAPE_NEAR_CONST_BELOW
        or shape == SHAPE_NEAR_CONST_EQUAL
        or shape == SHAPE_NEAR_CONST_ABOVE
    ):
        # Anchored at min == 0.0 so that `min + FEATURE_THRESHOLD` is EXACT in
        # float32 (see the docstring: at min == 0.5 it rounds up 2 ulp and the
        # band silently widens). The three highs then straddle the boundary as
        # tightly as float32 permits -- one ulp apart at 1e-7.
        var hi: Float32
        if shape == SHAPE_NEAR_CONST_BELOW:
            hi = next_down(FEATURE_THRESHOLD)
        elif shape == SHAPE_NEAR_CONST_EQUAL:
            hi = FEATURE_THRESHOLD
        else:
            hi = next_up(FEATURE_THRESHOLD)
        if row == 0:
            return 0.0
        if row == 1:
            return hi
        return 0.0 if (h & 1) == 0 else hi

    if shape == SHAPE_TWO_VALUED:
        # Only min and max are present, nothing between: a range-based
        # implementation that assumes interior values has nowhere to hide.
        if row == 0:
            return -2.0
        if row == 1:
            return 5.0
        return -2.0 if (h & 1) == 0 else 5.0

    if shape == SHAPE_OUTLIER:
        # Dense cluster in [0, 1/1024), one row at 1024. A uniform draw in
        # [min, max] lands in the EMPTY gap with probability > 1 - 1e-6, so
        # essentially every draw yields n_left = n_rows - 1, n_right = 1.
        if row == n_rows // 2:
            return 1024.0
        return unit_float(h) / 1024.0

    if shape == SHAPE_NEGATIVE:
        # [-9, -5): entirely negative, so a sign-blind min/max or an unsigned
        # bit comparison inverts the order.
        return -9.0 + 4.0 * unit_float(h)

    if shape == SHAPE_SPANS_ZERO:
        # [-1, 1). Float sign-magnitude ordering is NOT integer ordering across
        # zero; this is the column that says so.
        return signed_unit(h)

    if shape == SHAPE_ONE_ODD_ROW:
        # All equal except exactly one row. min/max are 7.5 and 11.5; every
        # threshold in [7.5, 11.5) gives n_left = n_rows - 1, n_right = 1.
        if row == n_rows // 3:
            return 11.5
        return 7.5

    return 0.0


def shaped_dataset(seed: UInt64, n_rows: Int, shapes: List[Int]) -> Dataset:
    """One column per entry of `shapes`, in order. Labels are hashed 2-class and
    the regression target is hashed, both independent of the features: these
    columns exist to exercise the RANGE and CONSTANT machinery, not scoring."""
    var n_cols = len(shapes)
    var x = List[Float32]()
    var y = List[Float32]()
    var lab = List[Int32]()
    for r in range(n_rows):
        for c in range(n_cols):
            x.append(shaped_value(seed, shapes[c], r, c, n_rows))
        lab.append(Int32(Int((cell_hash(seed, r, 0, SALT_LABEL) >> 40) % 2)))
        y.append(signed_unit(cell_hash(seed, r, 0, SALT_Y)))
    return Dataset(n_rows, n_cols, 2, x^, y^, lab^)


# ----------------------------------------------------------------------------
# 3. Analytic fixtures
# ----------------------------------------------------------------------------
@fieldwise_init
struct AnalyticFixture(Copyable, Movable):
    """A dataset plus the HAND-COMPUTED answer, as data.

    `t_lo`/`t_hi` bound the CLOSED interval over which the answer is invariant.
    `t_below`/`t_above` are the negative controls: thresholds outside the
    interval at which the answer MUST differ. `feature` is the feature the
    expected quantities describe, and `noise_feature` is the companion that must
    never beat it (-1 if there is none).
    """

    var name: String
    var data: Dataset
    var feature: Int
    var noise_feature: Int
    var is_classification: Bool
    var splittable: Bool
    var t_lo: Float64
    var t_hi: Float64
    var t_below: Float64
    var t_above: Float64
    var n_left: Int
    var n_right: Int
    var left_class_counts: List[Int]
    var right_class_counts: List[Int]
    var left_sum: Float64
    var right_sum: Float64
    var left_sum_sq: Float64
    var right_sum_sq: Float64
    var impurity_left: Float64
    var impurity_right: Float64
    var proxy: Float64


def analytic_separable_gap(seed: UInt64) -> AnalyticFixture:
    """See the module docstring: 128 class-0 rows in `[0,1)`, 128 class-1 rows
    in `(9,10]`, gap `[1,9]` empty, invariant over the CLOSED interval
    `[1.0, 9.0]`, proxy exactly 0.0."""
    var n = 256
    var half = 128
    var x = List[Float32]()
    var y = List[Float32]()
    var lab = List[Int32]()
    for r in range(n):
        var h0 = cell_hash(seed, r, 0, SALT_X)
        var h1 = cell_hash(seed, r, 1, SALT_X)
        if r < half:
            x.append(unit_float(h0))  # [0, 1)
            lab.append(Int32(0))
            y.append(0.0)
        else:
            x.append(9.0 + unit_float20_open(h0))  # (9, 10], exact on the 2^-20 grid
            lab.append(Int32(1))
            y.append(1.0)
        x.append(unit_float(h1))  # noise, [0, 1)
    var data = Dataset(n, 2, 2, x^, y^, lab^)

    var lc = List[Int]()
    lc.append(half)
    lc.append(0)
    var rc = List[Int]()
    rc.append(0)
    rc.append(half)
    return AnalyticFixture(
        "separable_gap",
        data^,
        0,
        1,
        True,
        True,
        1.0,
        9.0,
        0.5,
        9.5,
        half,
        half,
        lc^,
        rc^,
        0.0,  # left_sum: 128 rows of y = 0.0
        Float64(half),  # right_sum: 128 rows of y = 1.0
        0.0,
        Float64(half),
        0.0,  # gini_left
        0.0,  # gini_right
        0.0,  # proxy = -128*0 - 128*0
    )


def analytic_regression_step(seed: UInt64) -> AnalyticFixture:
    """See the module docstring: `y = 2.0` on `x0 in [0,4)`, `y = -3.0` on
    `x0 in (6,10]`, gap `[4,6]` empty, invariant over `[4.0, 6.0]`,
    proxy exactly 1664.0, both child impurities exactly 0.0."""
    var n = 256
    var half = 128
    var a = Float32(2.0)
    var b = Float32(-3.0)
    var x = List[Float32]()
    var y = List[Float32]()
    var lab = List[Int32]()
    for r in range(n):
        var h0 = cell_hash(seed, r, 0, SALT_X)
        var h1 = cell_hash(seed, r, 1, SALT_X)
        if r < half:
            x.append(4.0 * unit_float(h0))  # [0, 4)
            y.append(a)
        else:
            x.append(6.0 + 4.0 * unit_float20_open(h0))  # (6, 10], exact on the 2^-18 grid
            y.append(b)
        x.append(unit_float(h1))  # noise
        lab.append(Int32(0))
    var data = Dataset(n, 2, 1, x^, y^, lab^)

    var lc = List[Int]()
    lc.append(half)
    var rc = List[Int]()
    rc.append(half)
    var ls = Float64(half) * 2.0
    var rs = Float64(half) * -3.0
    var lss = Float64(half) * 4.0
    var rss = Float64(half) * 9.0
    return AnalyticFixture(
        "regression_step",
        data^,
        0,
        1,
        False,
        True,
        4.0,
        6.0,
        2.0,
        8.0,
        half,
        half,
        lc^,
        rc^,
        ls,
        rs,
        lss,
        rss,
        0.0,  # impurity_left  = 512/128 - 2^2
        0.0,  # impurity_right = 1152/128 - 3^2
        ls * ls / Float64(half) + rs * rs / Float64(half),  # 1664.0
    )


def analytic_tie_pair(seed: UInt64) -> AnalyticFixture:
    """Feature 1 is a BIT-FOR-BIT copy of feature 0 (the separable-gap feature),
    feature 2 is noise. Two exact copies score EQUAL at every corresponding
    threshold, so only the tie-break can decide -- and it must decide the same
    way every run. Expected quantities are those of `analytic_separable_gap`."""
    var n = 256
    var half = 128
    var x = List[Float32]()
    var y = List[Float32]()
    var lab = List[Int32]()
    for r in range(n):
        var h0 = cell_hash(seed, r, 0, SALT_X)
        var h2 = cell_hash(seed, r, 2, SALT_X)
        var v: Float32
        if r < half:
            v = unit_float(h0)
            lab.append(Int32(0))
            y.append(0.0)
        else:
            v = 9.0 + unit_float20_open(h0)  # (9, 10], exact on the 2^-20 grid
            lab.append(Int32(1))
            y.append(1.0)
        x.append(v)
        x.append(v)  # exact copy, same bits
        x.append(unit_float(h2))  # noise
    var data = Dataset(n, 3, 2, x^, y^, lab^)

    var lc = List[Int]()
    lc.append(half)
    lc.append(0)
    var rc = List[Int]()
    rc.append(0)
    rc.append(half)
    return AnalyticFixture(
        "tie_pair",
        data^,
        0,
        2,
        True,
        True,
        1.0,
        9.0,
        0.5,
        9.5,
        half,
        half,
        lc^,
        rc^,
        0.0,
        Float64(half),
        0.0,
        Float64(half),
        0.0,
        0.0,
        0.0,
    )


def analytic_all_constant() -> AnalyticFixture:
    """Every column exactly constant. `splittable` is False: all four features
    fail the constant test, `node_split_random` never enters the scoring body,
    and the correct answer is NO SPLIT. Labels are still two-class and mixed, so
    a builder cannot get the right answer by noticing a pure node instead."""
    var n = 128
    var x = List[Float32]()
    var y = List[Float32]()
    var lab = List[Int32]()
    var consts = List[Float32]()
    consts.append(0.0)
    consts.append(-1.0)
    consts.append(3.25)
    consts.append(1000000.0)
    for r in range(n):
        for c in range(len(consts)):
            x.append(consts[c])
        lab.append(Int32(r % 2))
        y.append(Float32(r % 2))
    var data = Dataset(n, len(consts), 2, x^, y^, lab^)

    var lc = List[Int]()
    lc.append(0)
    lc.append(0)
    var rc = List[Int]()
    rc.append(0)
    rc.append(0)
    return AnalyticFixture(
        "all_constant",
        data^,
        -1,
        -1,
        True,
        False,  # NOT splittable
        0.0,
        0.0,
        0.0,
        0.0,
        0,
        0,
        lc^,
        rc^,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def analytic_fixtures(seed: UInt64) -> List[AnalyticFixture]:
    """All four, in one list, for a check that wants to sweep them."""
    var out = List[AnalyticFixture]()
    out.append(analytic_separable_gap(seed))
    out.append(analytic_regression_step(seed))
    out.append(analytic_tie_pair(seed))
    out.append(analytic_all_constant())
    return out^
