# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Verifies the INSTRUMENT: proves `extratrees/mojo_only/fixtures.mojo` has the
properties its docstring claims.

    cd <repo root>
    pixi run mojo run -I . extratrees/mojo_only/fixtures_check.mojo

Nothing here asserts a property. Every property is either BRUTE-FORCED over a
sweep or counted cell by cell, and every invariance claim is paired with a
NEGATIVE CONTROL that requires the answer to CHANGE outside the claimed
interval -- a property that holds everywhere is not a property, and a check
that has never been seen to fail is not evidence.

WHAT IS PROVED, AND HOW
-----------------------
1. DETERMINISM. Each generator is run twice from the same seed and compared as
   FLOAT BIT PATTERNS, never as decimal text: `String(Float32)` in this
   toolchain does not round-trip for 0.46% of values, so a string comparison
   would be a weaker check that looks like a stronger one. Negative control: a
   DIFFERENT seed must produce different bytes, otherwise a generator that
   ignored its seed entirely would pass the determinism test perfectly.

2. SCATTER. Per-column distinct counts over 4096 rows; no column constant when
   it was not meant to be; no two columns bit-equal; and the two-column union's
   distinct count near the sum of the two, which is what refutes "the values
   are a permutation of a small set" rather than merely "the values differ".
   Labels are checked against every small-modulus and blocked function of the
   row index, and against being sorted.

3. THE NEAR-CONSTANT STRADDLE. `max - min` in FLOAT32 for the three
   near-constant columns, against `FEATURE_THRESHOLD` -- the exact expression
   `_splitter.pyx:617` uses. The three are one ulp apart at 1e-7, which is as
   tight as float32 allows, and the check reads their BIT PATTERNS to prove the
   adjacency rather than trusting the decimal print.

4. THE ANALYTIC PROPERTIES. For each analytic fixture: the claimed gap is
   verified EMPTY by counting rows inside it; then 1001 thresholds are swept
   across the claimed CLOSED interval and every one must reproduce the
   hand-computed counts, class counts, sums, child impurities and criterion
   proxy EXACTLY (these quantities are all exactly representable, so exact
   equality is the right comparison and a tolerance would be a weaker test).
   Then 200 thresholds are swept on each side OUTSIDE the interval and the
   partition must actually move there.

5. THE CHOSEN FEATURE. The companion noise feature is swept EXHAUSTIVELY at
   every one of its own distinct values -- which covers every partition that
   feature can induce, because a partition only changes at a data value -- and
   must score strictly worse than the winner at every one. So "feature 0 wins"
   is a proved statement about this fixture, not a probabilistic one.

6. THE TIE. Features 0 and 1 of `tie_pair` are compared bit for bit, and their
   proxies are required EQUAL at every threshold of a 1001-point sweep across
   the whole range. Negative control: feature 2 must differ somewhere,
   otherwise "everything ties" would pass.

7. ALL-CONSTANT. Every column of `analytic_all_constant` has min bitwise equal
   to max and is reported constant. Negative control: a known non-constant
   column is reported non-constant.

SABOTAGE (rule 8, one per MECHANISM, recorded in the sub-lane report)
--------------------------------------------------------------------
Run against a temporarily broken `fixtures.mojo`, then restored:
  * hash ignores `col`            -> SCATTER goes red (columns become equal)
  * analytic gap shrunk to [1,1.5]-> INTERVAL INVARIANCE goes red
  * near-constant "above" built
    with `next_down` instead of
    `next_up`                     -> STRADDLE goes red
  * one all-constant column made
    non-constant                  -> ALL-CONSTANT goes red
"""

from extratrees.mojo_only.fixtures import (
    FEATURE_THRESHOLD,
    N_SHAPES,
    SHAPE_NEAR_CONST_ABOVE,
    SHAPE_NEAR_CONST_BELOW,
    SHAPE_NEAR_CONST_EQUAL,
    AnalyticFixture,
    Dataset,
    PartitionStats,
    all_shapes,
    analytic_all_constant,
    analytic_regression_step,
    analytic_separable_gap,
    analytic_tie_pair,
    feature_min_max,
    hashed_classification,
    hashed_regression,
    is_constant_feature,
    next_down,
    next_up,
    partition,
    shape_expects_constant,
    shape_name,
    shaped_dataset,
)

comptime SEED: UInt64 = 20260821
comptime OTHER_SEED: UInt64 = 20260822
comptime BIG_ROWS = 4096
comptime BIG_COLS = 8
comptime N_CLASSES = 3
comptime SWEEP = 1001
comptime OUTSIDE_SWEEP = 200


# ----------------------------------------------------------------------------
# Reporting. Failures accumulate so that a sabotage run shows EVERY mechanism
# it broke rather than only the first.
# ----------------------------------------------------------------------------
@fieldwise_init
struct Tally(Copyable, Movable):
    var checks: Int
    var failures: Int

    def expect(mut self, ok: Bool, msg: String):
        self.checks += 1
        if not ok:
            self.failures += 1
            print("  FAIL:", msg)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def same_bits(a: Dataset, b: Dataset) raises -> Bool:
    """Bit-for-bit equality of features, regression target and labels."""
    if a.n_rows != b.n_rows or a.n_cols != b.n_cols or a.n_classes != b.n_classes:
        return False
    for r in range(a.n_rows):
        for c in range(a.n_cols):
            if a.bits(r, c) != b.bits(r, c):
                return False
        if a.y[r].to_bits[DType.uint32]() != b.y[r].to_bits[DType.uint32]():
            return False
        if a.label[r] != b.label[r]:
            return False
    return True


def distinct_bits(data: Dataset, col: Int) raises -> Int:
    var d = Dict[UInt32, Int]()
    for r in range(data.n_rows):
        d[data.bits(r, col)] = 1
    return len(d)


def distinct_union(data: Dataset, c1: Int, c2: Int) raises -> Int:
    var d = Dict[UInt32, Int]()
    for r in range(data.n_rows):
        d[data.bits(r, c1)] = 1
        d[data.bits(r, c2)] = 1
    return len(d)


def columns_equal(data: Dataset, c1: Int, c2: Int) raises -> Bool:
    for r in range(data.n_rows):
        if data.bits(r, c1) != data.bits(r, c2):
            return False
    return True


def distinct_values(data: Dataset, col: Int) raises -> List[Float32]:
    """Every distinct value of a column, as thresholds for an exhaustive sweep:
    a partition can only change AT a data value, so sweeping the distinct values
    covers every partition the feature can induce."""
    var d = Dict[UInt32, Int]()
    var out = List[Float32]()
    for r in range(data.n_rows):
        var b = data.bits(r, col)
        if b not in d:
            d[b] = 1
            out.append(data.value(r, col))
    return out^


def counts_equal(ps: PartitionStats, f: AnalyticFixture) raises -> Bool:
    if ps.n_left != f.n_left or ps.n_right != f.n_right:
        return False
    if len(ps.left_class_counts) != len(f.left_class_counts):
        return False
    for k in range(len(f.left_class_counts)):
        if ps.left_class_counts[k] != f.left_class_counts[k]:
            return False
        if ps.right_class_counts[k] != f.right_class_counts[k]:
            return False
    return True


def stats_equal(ps: PartitionStats, f: AnalyticFixture) raises -> Bool:
    """Exact equality. Every expected quantity here is an integer or a small
    dyadic rational, so it is exactly representable in float64 and a tolerance
    would only hide a real disagreement."""
    if not counts_equal(ps, f):
        return False
    if ps.left_sum != f.left_sum or ps.right_sum != f.right_sum:
        return False
    if ps.left_sum_sq != f.left_sum_sq or ps.right_sum_sq != f.right_sum_sq:
        return False
    if f.is_classification:
        if ps.gini_left() != f.impurity_left or ps.gini_right() != f.impurity_right:
            return False
        if ps.gini_proxy() != f.proxy:
            return False
    else:
        if ps.mse_left() != f.impurity_left or ps.mse_right() != f.impurity_right:
            return False
        if ps.mse_proxy() != f.proxy:
            return False
    return True


def score_of(ps: PartitionStats, is_classification: Bool) raises -> Float64:
    return ps.gini_proxy() if is_classification else ps.mse_proxy()


# ----------------------------------------------------------------------------
# 1. Determinism
# ----------------------------------------------------------------------------
def check_determinism(mut t: Tally) raises:
    print("1. DETERMINISM (float BIT PATTERNS, not decimal text)")

    var a = hashed_classification(SEED, 512, 6, N_CLASSES)
    var b = hashed_classification(SEED, 512, 6, N_CLASSES)
    t.expect(same_bits(a, b), "hashed_classification not reproducible from its seed")

    var ra = hashed_regression(SEED, 512, 6)
    var rb = hashed_regression(SEED, 512, 6)
    t.expect(same_bits(ra, rb), "hashed_regression not reproducible from its seed")

    var sa = shaped_dataset(SEED, 512, all_shapes())
    var sb = shaped_dataset(SEED, 512, all_shapes())
    t.expect(same_bits(sa, sb), "shaped_dataset not reproducible from its seed")

    var ga = analytic_separable_gap(SEED)
    var gb = analytic_separable_gap(SEED)
    t.expect(same_bits(ga.data, gb.data), "analytic_separable_gap not reproducible")
    var qa = analytic_regression_step(SEED)
    var qb = analytic_regression_step(SEED)
    t.expect(same_bits(qa.data, qb.data), "analytic_regression_step not reproducible")
    var ta = analytic_tie_pair(SEED)
    var tb = analytic_tie_pair(SEED)
    t.expect(same_bits(ta.data, tb.data), "analytic_tie_pair not reproducible")
    var ca = analytic_all_constant()
    var cb = analytic_all_constant()
    t.expect(same_bits(ca.data, cb.data), "analytic_all_constant not reproducible")

    # NEGATIVE CONTROL. Without this, a generator that ignored its seed and
    # returned zeros would score a perfect determinism result.
    var c = hashed_classification(OTHER_SEED, 512, 6, N_CLASSES)
    t.expect(
        not same_bits(a, c),
        "a DIFFERENT seed produced identical bytes -- the seed is being ignored",
    )
    var sc = shaped_dataset(OTHER_SEED, 512, all_shapes())
    t.expect(
        not same_bits(sa, sc),
        "shaped_dataset ignores its seed (different seed, identical bytes)",
    )
    print("   generators reproduced:", 7, " seed-sensitivity controls:", 2)


# ----------------------------------------------------------------------------
# 2. Scatter
# ----------------------------------------------------------------------------
def check_scatter(mut t: Tally) raises:
    print("2. SCATTER of the hashed adversarial fixture (", BIG_ROWS, "x", BIG_COLS, ")")
    var d = hashed_classification(SEED, BIG_ROWS, BIG_COLS, N_CLASSES)

    var worst = BIG_ROWS + 1
    for c in range(BIG_COLS):
        var n = distinct_bits(d, c)
        if n < worst:
            worst = n
        t.expect(
            n * 100 >= BIG_ROWS * 99,
            String("column ") + String(c) + " has only " + String(n)
            + " distinct values of " + String(BIG_ROWS) + " rows",
        )
        var mm = feature_min_max(d, c)
        t.expect(
            not is_constant_feature(d, c),
            String("column ") + String(c) + " is CONSTANT but was not meant to be",
        )
        t.expect(
            mm.min < mm.max,
            String("column ") + String(c) + " has min >= max",
        )
    print("   min distinct values in any column:", worst, "of", BIG_ROWS)

    # No two columns equal, and no two columns drawing from the same small set.
    var pairs = 0
    var worst_union_ratio = Float64(10.0)
    for i in range(BIG_COLS):
        for j in range(i + 1, BIG_COLS):
            pairs += 1
            t.expect(
                not columns_equal(d, i, j),
                String("columns ") + String(i) + " and " + String(j) + " are IDENTICAL",
            )
            var u = distinct_union(d, i, j)
            var di = distinct_bits(d, i)
            var dj = distinct_bits(d, j)
            var ratio = Float64(u) / Float64(di + dj)
            if ratio < worst_union_ratio:
                worst_union_ratio = ratio
            # If two columns were permutations of one small set the union would
            # collapse toward the size of ONE column, i.e. ratio -> 0.5.
            t.expect(
                ratio > 0.95,
                String("columns ") + String(i) + "," + String(j)
                + " share too many values (union/sum = " + String(ratio) + ")",
            )
    print("   column pairs compared:", pairs, " worst union/sum ratio:", worst_union_ratio)

    # Labels: present, balanced, and not a function of the row index.
    var counts = List[Int]()
    for _ in range(N_CLASSES):
        counts.append(0)
    for r in range(BIG_ROWS):
        counts[Int(d.label[r])] += 1
    for k in range(N_CLASSES):
        t.expect(
            counts[k] * N_CLASSES * 2 >= BIG_ROWS and counts[k] * N_CLASSES <= BIG_ROWS * 2,
            String("class ") + String(k) + " count " + String(counts[k]) + " is not balanced",
        )
    print("   label counts:", counts[0], counts[1], counts[2])

    var worst_mod = Float64(0.0)
    for k in range(2, 33):
        var agree = 0
        for r in range(BIG_ROWS):
            if Int(d.label[r]) == r % k:
                agree += 1
        var frac = Float64(agree) / Float64(BIG_ROWS)
        if frac > worst_mod:
            worst_mod = frac
        t.expect(
            frac < 0.5,
            String("labels agree with row % ") + String(k) + " on "
            + String(frac) + " of rows -- label is a function of the row index",
        )
    print("   highest agreement with any row % k, k in 2..32:", worst_mod)

    var blocked = 0
    for r in range(BIG_ROWS):
        if Int(d.label[r]) == (r * N_CLASSES) // BIG_ROWS:
            blocked += 1
    t.expect(
        Float64(blocked) / Float64(BIG_ROWS) < 0.5,
        "labels are BLOCKED by row index (sorted into contiguous class runs)",
    )

    var descents = 0
    for r in range(BIG_ROWS - 1):
        if d.label[r] > d.label[r + 1]:
            descents += 1
    t.expect(descents > BIG_ROWS // 8, "labels are (nearly) sorted by row index")
    print("   blocked-agreement:", Float64(blocked) / Float64(BIG_ROWS), " label descents:", descents)

    # Every residue class mod 16 must contain more than one label: a label that
    # depended on a small window of the row index would be constant inside one.
    for m in range(16):
        var seen = Dict[Int32, Int]()
        var r = m
        while r < BIG_ROWS:
            seen[d.label[r]] = 1
            r += 16
        t.expect(
            len(seen) >= 2,
            String("all rows with row % 16 == ") + String(m) + " share ONE label",
        )


# ----------------------------------------------------------------------------
# 3. Shapes, and the near-constant straddle
# ----------------------------------------------------------------------------
def check_shapes(mut t: Tally) raises:
    print("3. COLUMN SHAPES and the FEATURE_THRESHOLD straddle")
    var shapes = all_shapes()
    var d = shaped_dataset(SEED, 1024, shapes)
    t.expect(d.n_cols == N_SHAPES, "shaped_dataset did not make one column per shape")

    for c in range(d.n_cols):
        var mm = feature_min_max(d, c)
        var got = is_constant_feature(d, c)
        var want = shape_expects_constant(shapes[c])
        t.expect(
            got == want,
            String("shape ") + shape_name(shapes[c]) + ": constant test returned "
            + String(got) + ", expected " + String(want),
        )
        print(
            "   ", shape_name(shapes[c]),
            " min=", mm.min, " max=", mm.max,
            " spread=", mm.max - mm.min,
            " constant=", got,
            " distinct=", distinct_bits(d, c),
        )

    # The straddle itself, read as `max - min` in FLOAT32 against
    # FEATURE_THRESHOLD -- the expression at _splitter.pyx:617.
    var below = -1
    var equal = -1
    var above = -1
    for c in range(d.n_cols):
        if shapes[c] == SHAPE_NEAR_CONST_BELOW:
            below = c
        elif shapes[c] == SHAPE_NEAR_CONST_EQUAL:
            equal = c
        elif shapes[c] == SHAPE_NEAR_CONST_ABOVE:
            above = c
    t.expect(below >= 0 and equal >= 0 and above >= 0, "near-constant shapes missing")

    var mb = feature_min_max(d, below)
    var me = feature_min_max(d, equal)
    var ma = feature_min_max(d, above)
    var sb = mb.max - mb.min
    var se = me.max - me.min
    var sa = ma.max - ma.min

    t.expect(sb > 0.0, "near_const_below has ZERO spread -- it is not a near-constant")
    t.expect(sb <= FEATURE_THRESHOLD, "near_const_below spread is NOT <= FEATURE_THRESHOLD")
    t.expect(sb < FEATURE_THRESHOLD, "near_const_below spread is not STRICTLY below")
    t.expect(se == FEATURE_THRESHOLD, "near_const_equal spread does not EQUAL FEATURE_THRESHOLD")
    t.expect(se <= FEATURE_THRESHOLD, "near_const_equal fails the `<=` form of the test")
    t.expect(sa > FEATURE_THRESHOLD, "near_const_above spread is NOT > FEATURE_THRESHOLD")

    # And they are adjacent floats, i.e. the pair is as tight as float32 allows.
    t.expect(
        sb.to_bits[DType.uint32]() == next_down(FEATURE_THRESHOLD).to_bits[DType.uint32](),
        "near_const_below is not exactly one ulp below FEATURE_THRESHOLD",
    )
    t.expect(
        sa.to_bits[DType.uint32]() == next_up(FEATURE_THRESHOLD).to_bits[DType.uint32](),
        "near_const_above is not exactly one ulp above FEATURE_THRESHOLD",
    )
    # The anchor that makes the boundary exact: min == 0.0, so
    # `min + FEATURE_THRESHOLD` does not round.
    t.expect(
        mb.min == 0.0 and me.min == 0.0 and ma.min == 0.0,
        "a near-constant column is not anchored at min == 0.0, so the float32 "
        "addition in the constant test rounds and the boundary moves",
    )
    print(
        "    straddle bits: below=", sb.to_bits[DType.uint32](),
        " FT=", FEATURE_THRESHOLD.to_bits[DType.uint32](),
        " above=", sa.to_bits[DType.uint32](),
    )


# ----------------------------------------------------------------------------
# 4/5. The analytic fixtures: invariance, negative controls, chosen feature
# ----------------------------------------------------------------------------
def check_analytic(mut t: Tally, f: AnalyticFixture) raises:
    print("4.", f.name, "-- invariance over the claimed interval")

    if not f.splittable:
        # `all_constant` has its own section.
        return

    # (a) The gap is EMPTY. This is the structural fact the invariance rests on,
    #     and it is checked independently of any sweep.
    var inside = 0
    for r in range(f.data.n_rows):
        var v = Float64(f.data.value(r, f.feature))
        if v >= f.t_lo and v <= f.t_hi:
            inside += 1
    t.expect(
        inside == 0,
        String(f.name) + ": " + String(inside) + " rows lie INSIDE the claimed gap ["
        + String(f.t_lo) + ", " + String(f.t_hi) + "]",
    )

    # (b) Every threshold in the CLOSED interval gives the hand-computed answer.
    var bad = 0
    for i in range(SWEEP):
        var t_ = f.t_lo + (f.t_hi - f.t_lo) * Float64(i) / Float64(SWEEP - 1)
        var ps = partition(f.data, f.feature, t_)
        if not stats_equal(ps, f):
            bad += 1
    t.expect(
        bad == 0,
        String(f.name) + ": " + String(bad) + " of " + String(SWEEP)
        + " in-interval thresholds disagreed with the hand-computed answer",
    )
    print("   in-interval thresholds swept:", SWEEP, " disagreements:", bad)

    # (c) NEGATIVE CONTROLS. The two named outside thresholds must move the
    #     partition, and the partition must genuinely VARY on each side.
    var pb = partition(f.data, f.feature, f.t_below)
    var pa = partition(f.data, f.feature, f.t_above)
    t.expect(
        not counts_equal(pb, f),
        String(f.name) + ": threshold " + String(f.t_below)
        + " BELOW the interval still gives the in-interval answer",
    )
    t.expect(
        not counts_equal(pa, f),
        String(f.name) + ": threshold " + String(f.t_above)
        + " ABOVE the interval still gives the in-interval answer",
    )

    var mm = feature_min_max(f.data, f.feature)
    var moved_lo = 0
    var seen_lo = Dict[Int, Int]()
    for i in range(OUTSIDE_SWEEP):
        var t_ = Float64(mm.min) + (f.t_lo - Float64(mm.min)) * Float64(i) / Float64(OUTSIDE_SWEEP)
        var ps = partition(f.data, f.feature, t_)
        seen_lo[ps.n_left] = 1
        if not counts_equal(ps, f):
            moved_lo += 1
    var moved_hi = 0
    var seen_hi = Dict[Int, Int]()
    for i in range(1, OUTSIDE_SWEEP + 1):
        var t_ = f.t_hi + (Float64(mm.max) - f.t_hi) * Float64(i) / Float64(OUTSIDE_SWEEP)
        var ps = partition(f.data, f.feature, t_)
        seen_hi[ps.n_left] = 1
        if not counts_equal(ps, f):
            moved_hi += 1
    t.expect(
        moved_lo * 10 >= OUTSIDE_SWEEP * 9,
        String(f.name) + ": only " + String(moved_lo) + " of " + String(OUTSIDE_SWEEP)
        + " thresholds BELOW the interval changed the answer",
    )
    t.expect(
        moved_hi * 10 >= OUTSIDE_SWEEP * 9,
        String(f.name) + ": only " + String(moved_hi) + " of " + String(OUTSIDE_SWEEP)
        + " thresholds ABOVE the interval changed the answer",
    )
    t.expect(
        len(seen_lo) >= 10 and len(seen_hi) >= 10,
        String(f.name) + ": the partition barely varies outside the interval ("
        + String(len(seen_lo)) + "/" + String(len(seen_hi)) + " distinct n_left)",
    )
    print(
        "   outside-interval thresholds:", 2 * OUTSIDE_SWEEP,
        " changed:", moved_lo + moved_hi,
        " distinct partitions seen:", len(seen_lo) + len(seen_hi),
    )

    # (d) THE CHOSEN FEATURE. Exhaustive over the noise feature's own distinct
    #     values, which covers every partition it can induce.
    if f.noise_feature >= 0:
        var vals = distinct_values(f.data, f.noise_feature)
        var beat = 0
        var best = Float64(-1.0e300)
        for i in range(len(vals)):
            var ps = partition(f.data, f.noise_feature, Float64(vals[i]))
            var s = score_of(ps, f.is_classification)
            if s > best:
                best = s
            if s >= f.proxy:
                beat += 1
        t.expect(
            beat == 0,
            String(f.name) + ": the NOISE feature matched or beat the winner at "
            + String(beat) + " of " + String(len(vals)) + " of its own values",
        )
        print(
            "   noise-feature thresholds swept:", len(vals),
            " best noise score:", best, " winner score:", f.proxy,
        )


def check_tie(mut t: Tally) raises:
    print("5. TIE-BREAK fixture: two features that are exact copies")
    var f = analytic_tie_pair(SEED)
    t.expect(columns_equal(f.data, 0, 1), "tie_pair features 0 and 1 are NOT bit-identical")
    t.expect(
        not columns_equal(f.data, 0, 2),
        "tie_pair feature 2 is identical to feature 0 -- the control is dead",
    )

    var mm = feature_min_max(f.data, 0)
    var ties = 0
    var noise_differs = 0
    for i in range(SWEEP):
        var t_ = Float64(mm.min) + (Float64(mm.max) - Float64(mm.min)) * Float64(i) / Float64(SWEEP - 1)
        var p0 = partition(f.data, 0, t_)
        var p1 = partition(f.data, 1, t_)
        var p2 = partition(f.data, 2, t_)
        if p0.n_left == p1.n_left and p0.gini_proxy() == p1.gini_proxy():
            ties += 1
        if p0.n_left != p2.n_left or p0.gini_proxy() != p2.gini_proxy():
            noise_differs += 1
    t.expect(
        ties == SWEEP,
        String("tie_pair: features 0 and 1 tied at only ") + String(ties)
        + " of " + String(SWEEP) + " thresholds",
    )
    t.expect(
        noise_differs > SWEEP // 2,
        "tie_pair: feature 2 ties feature 0 nearly everywhere -- 'they tie' is vacuous",
    )
    print("   thresholds where f0 and f1 tie:", ties, "of", SWEEP,
          " where f2 differs:", noise_differs)


def check_all_constant(mut t: Tally) raises:
    print("6. ALL-CONSTANT fixture")
    var f = analytic_all_constant()
    t.expect(not f.splittable, "all_constant is marked splittable")
    for c in range(f.data.n_cols):
        var mm = feature_min_max(f.data, c)
        t.expect(
            mm.min.to_bits[DType.uint32]() == mm.max.to_bits[DType.uint32](),
            String("all_constant column ") + String(c) + " has min != max",
        )
        t.expect(
            is_constant_feature(f.data, c),
            String("all_constant column ") + String(c) + " is not reported constant",
        )
        t.expect(
            distinct_bits(f.data, c) == 1,
            String("all_constant column ") + String(c) + " has more than one value",
        )
    # NEGATIVE CONTROL: the constant test is not simply returning True.
    var d = hashed_classification(SEED, 256, 2, 2)
    t.expect(
        not is_constant_feature(d, 0),
        "the constant test reports a hashed column constant -- it always says True",
    )
    # The node is not trivially pure either, so a builder cannot get "no split"
    # right for the wrong reason.
    var mixed = 0
    for r in range(f.data.n_rows):
        if f.data.label[r] != f.data.label[0]:
            mixed += 1
    t.expect(mixed > 0, "all_constant node is PURE -- 'no split' would be right for the wrong reason")
    print("   columns:", f.data.n_cols, " all constant, node impure with", mixed, "differing labels")


def main() raises:
    print("fixtures_check -- verifying the ExtraTrees fixture generators")
    print("seed:", SEED, " reference: sklearn 1.9.0 (77def0e)")
    print()
    var t = Tally(0, 0)

    check_determinism(t)
    print()
    check_scatter(t)
    print()
    check_shapes(t)
    print()
    check_analytic(t, analytic_separable_gap(SEED))
    print()
    check_analytic(t, analytic_regression_step(SEED))
    print()
    check_analytic(t, analytic_tie_pair(SEED))
    print()
    check_tie(t)
    print()
    check_all_constant(t)
    print()

    print("checks:", t.checks, " failures:", t.failures)
    if t.failures != 0:
        raise Error(String(t.failures) + " fixture property checks FAILED")
    print("OK")
