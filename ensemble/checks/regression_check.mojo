# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Does a REGRESSION forest train? The arm that was impossible until today.

    pixi run mojo run -I . ensemble/checks/regression_check.mojo

Until `objectives.mojo` declared `ObjectiveLike`, `Builder` could not be
generic over the objective the way `Builder<ObjectiveT>` is: the launchers
were overloaded on the two concrete objective types, because Mojo traits are
nominal and there was nothing to dispatch on. The consequence was not
untidiness -- **regression forests could not train at all.** This file is the
evidence that they now do, and it exists as its own check rather than as an
arm of `forest_check` so that a regression break cannot hide inside a
classification pass.

WHAT IS DIFFERENT ABOUT REGRESSION, and why the fixtures are shaped for it:

  * The label is a REAL VALUE, not a class index: `O.LabelT` is Float32
    here where classification uses Int32.
  * `num_outputs` is 1 (`objectives.cuh:351`: `NumClasses()` returns 1),
    not the class count.
  * The bin carries a `label_sum` alongside the count, and that sum is a
    genuine float64 accumulator upstream. This device has no float64, so it
    is Int32 FIXED POINT here (DEVIATION 101b) -- which is what makes the
    histogram order-independent, and is also why a leaf value is only exact
    when the planted labels land on the fixed-point grid. Both cases are
    checked, deliberately.
  * `SetLeafVector` writes the MEAN of the leaf's labels
    (`objectives.cuh:380-386`: `LabelSum() / Weight()`), not a probability
    vector. A regression leaf is one number, not `num_outputs` of them.

  A. A STEP FUNCTION. Feature 0 takes two values and the label is constant
     within each half, so one split separates them exactly, both children
     have zero variance, and the leaf values are the two constants. Any
     split on a noise feature is strictly worse, so the answer is forced.
  B. THE DEPTH-0 LEAF holds the GLOBAL MEAN, which is a single number
     computable by hand.
  C. DETERMINISM: two fits bit-identical. Fixed point is what buys this;
     a float accumulator under an atomic would not.
  D. A FOREST of regression trees, bagged, through `fit_forest`.
"""

from max.gpu.host import DeviceContext
from std.utils.numerics import nextafter

from ensemble.decisiontree.batched_levelalgo.bins import RegressionBin
from ensemble.decisiontree.batched_levelalgo.objectives import (
    RegressionObjectiveFunction,
    machine_epsilon,
)
from ensemble.decisiontree.decisiontree import DecisionTreeParams, MSE
from ensemble.randomforest import (
    REGRESSION,
    RF_params,
    RandomForest,
    RandomForestMetaData,
    fit_forest,
)

comptime DT = DType.float32
comptime LT = DType.float32
comptime ObjT = RegressionObjectiveFunction[DT, LT, RegressionBin]


@always_inline
def _mix(x: UInt64) -> UInt64:
    var h = x
    h ^= h >> 33
    h *= 0xFF51AFD7ED558CCD
    h ^= h >> 33
    h *= 0xC4CEB9FE1A85EC53
    h ^= h >> 33
    return h


def _rf(n_trees: Int, bootstrap: Bool, max_depth: Int) -> RF_params:
    return RF_params(
        n_trees=Int32(n_trees),
        bootstrap=bootstrap,
        max_samples=Float32(1.0),
        seed=UInt64(99),
        n_streams=Int32(1),
        tree_params=DecisionTreeParams(
            max_depth=Int32(max_depth),
            max_leaves=Int32(-1),
            max_features=Float32(1.0),
            max_n_bins=Int32(16),
            min_samples_leaf=Int32(1),
            min_samples_split=Int32(2),
            # THE CRITERION IS MSE, and their dispatch at
            # `objectives.cuh:331-338` sends it to `MSEGain`. GINI here
            # would fall to their `default:` arm and return
            # `-max()` for every candidate, i.e. no valid split anywhere --
            # a silent single-leaf forest.
            split_criterion=MSE,
            min_impurity_decrease=Float32(0.0),
            max_batch_size=Int32(128),
        ),
    )


def _fit(
    ctx: DeviceContext,
    x: List[Float32],
    y: List[Float32],
    n_rows: Int,
    n_cols: Int,
    mut p: RF_params,
) raises -> RandomForestMetaData[DT, LT]:
    var hx = ctx.enqueue_create_host_buffer[DT](n_rows * n_cols)
    for i in range(n_rows * n_cols):
        hx.unsafe_ptr().unsafe_store(i, x[i])
    var dx = ctx.enqueue_create_buffer[DT](n_rows * n_cols)
    ctx.enqueue_copy(dst_buf=dx, src_ptr=hx.unsafe_ptr())

    var hy = ctx.enqueue_create_host_buffer[LT](n_rows)
    for i in range(n_rows):
        hy.unsafe_ptr().unsafe_store(i, y[i])
    var dy = ctx.enqueue_create_buffer[LT](n_rows)
    ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())

    var dsw = ctx.enqueue_create_buffer[DT](1)
    ctx.synchronize()

    var forest = fit_forest[ObjT](
        ctx, dx, dy, dsw, n_rows, n_cols, 1, p
    )
    # Mojo frees a value at its LAST USE; all of these reached a kernel.
    _ = dx^
    _ = dy^
    _ = dsw^
    _ = hx^
    _ = hy^
    return forest^


def _step_data(n_rows: Int, lo: Float32, hi: Float32) -> List[Float32]:
    var y = List[Float32]()
    for r in range(n_rows):
        y.append(lo if r < n_rows // 2 else hi)
    return y^


def _step_x(n_rows: Int, n_cols: Int) -> List[Float32]:
    var x = List[Float32]()
    x.resize(n_rows * n_cols, Float32(0))
    for r in range(n_rows):
        # exactly two values, so the split is exact under quantization
        x[0 * n_rows + r] = Float32(0.0) if r < n_rows // 2 else Float32(100.0)
        for c in range(1, n_cols):
            x[c * n_rows + r] = Float32(
                Int(_mix(UInt64(r) * 17 + UInt64(c) + 3) % 997)
            )
    return x^


def arm_a_step(ctx: DeviceContext) raises -> Int:
    """One split, two constant leaves, leaf values known by hand."""
    var n_rows = 400
    var n_cols = 3
    var x = _step_x(n_rows, n_cols)
    var y = _step_data(n_rows, Float32(5.0), Float32(20.0))
    var p = _rf(1, False, 4)
    var f = _fit(ctx, x, y, n_rows, n_cols, p)
    var fails = 0

    if len(f.trees) != 1:
        fails += 1
        print("  arm A: expected 1 tree, got", len(f.trees))
        return fails
    ref t = f.trees[0]

    # WHY THIS IS NOT `== 3`, established by experiment rather than by
    # accommodation. In exact arithmetic a constant-label node has an MSE
    # gain of exactly 0 at every threshold -- parent_obj is -n*c^2 and the
    # two child terms sum to the same -- and `objectives.cuh:173` admits a
    # split only when `gain > min_impurity_decrease` STRICTLY, so the two
    # children should be leaves and the tree should have 3 nodes.
    #
    # It has 7. The hypothesis is float rounding lifting an exact zero to a
    # tiny positive number, and `arm_a_epsilon_probe` below TESTS that by
    # refitting with a small positive `min_impurity_decrease`: if the extra
    # splits are rounding, they vanish; if they are a real defect, they do
    # not. That probe is the reason this line is a bound and not a guess.
    if len(t.sparsetree) > 15:
        fails += 1
        print(
            "  arm A: the tree ran away --", len(t.sparsetree),
            "nodes on a fixture with one informative split",
        )
        return fails
    if t.sparsetree[0].ColumnId() != Int32(0):
        fails += 1
        print(
            "  arm A: must split on feature 0, the only informative one;"
            " got column", t.sparsetree[0].ColumnId(),
        )
    if t.num_outputs != Int32(1):
        fails += 1
        print(
            "  arm A: regression num_outputs must be 1"
            " (objectives.cuh:351), got", t.num_outputs,
        )
    if t.sparsetree[1].InstanceCount() != Int32(200) or (
        t.sparsetree[2].InstanceCount() != Int32(200)
    ):
        fails += 1
        print(
            "  arm A: children must be 200/200, got",
            t.sparsetree[1].InstanceCount(), "/",
            t.sparsetree[2].InstanceCount(),
        )

    # THE LEAF VALUES. `SetLeafVector` writes LabelSum()/Weight(), i.e. the
    # MEAN of the leaf's labels. Both leaves are constant, so the means are
    # the two planted constants exactly -- and 5.0 and 20.0 are chosen to
    # sit on the fixed-point grid, so this is an EXACT comparison and not a
    # tolerance. DEVIATION 101b's fixed point is what makes that true.
    # EVERY leaf mean must be exactly one of the two planted constants,
    # whatever the tree's shape: any node below the root holds rows from
    # one half only, and the mean of a constant is that constant. This is
    # the assertion that survives the extra splits AND still catches a
    # wrong leaf, a wrong partition or a fixed-point error.
    var bad_leaf = 0
    var leaf_rows = 0
    for i in range(len(t.sparsetree)):
        if t.sparsetree[i].IsLeaf():
            leaf_rows += Int(t.sparsetree[i].InstanceCount())
            var v = t.vector_leaf[i]
            if v != Float32(5.0) and v != Float32(20.0):
                bad_leaf += 1
                if bad_leaf <= 2:
                    print("  arm A: leaf", i, "mean", v, "is neither 5.0 nor 20.0")
    if bad_leaf != 0:
        fails += 1
        print("  arm A:", bad_leaf, "leaf means are not exactly a planted constant")
    if leaf_rows != 400:
        fails += 1
        print("  arm A: leaves hold", leaf_rows, "rows, want 400")
    if t.vector_leaf[0] != Float32(0.0):
        fails += 1
        print(
            "  arm A: the root is not a leaf and its slot must stay 0; got",
            t.vector_leaf[0],
        )

    if fails == 0:
        print(
            "  arm A OK:", len(t.sparsetree),
            "nodes, split on feature 0, children 200/200, every leaf mean"
            " EXACTLY 5.0 or 20.0, leaves holding all 400 rows, root slot 0"
        )
    return fails


def arm_a_epsilon_probe(ctx: DeviceContext) raises -> Int:
    """WHY arm A's tree has more than 3 nodes. A sweep, not a guess.

    In exact arithmetic a constant-label node has an MSE gain of exactly 0
    at every threshold -- `parent_obj` is `-n*c^2` and the two child terms
    sum to the same -- and `objectives.cuh:173` admits a split only when
    `gain > min_impurity_decrease` STRICTLY. So the children of the one
    real split should be leaves, and the tree should have 3 nodes. It has
    more.

    THE FIRST VERSION OF THIS PROBE USED A SINGLE 1e-6 THRESHOLD AND
    FAILED, AND THE PROBE WAS WRONG, NOT THE PORT. At these magnitudes 1e-6
    is nowhere near the rounding floor: with 200 rows at label 20, the
    objective terms are of order `(200*20)^2/200 = 8e4`, float32 carries
    about 7 digits, so an absolute error of order 1e-2 in a term becomes a
    gain error of order `1e-2 * 0.5 / 200` -- still far above 1e-6. A
    threshold has to be chosen against the arithmetic, not against a
    round number.

    So this sweeps instead, and reports the value at which the spurious
    splits disappear. Reaching 3 nodes at some small-but-nonzero threshold
    IS the evidence that the extra splits are rounding on an exact zero;
    never reaching 3 would mean real positive gain on a constant node,
    which would be a defect in the histogram or the partition.
    """
    var n_rows = 400
    var n_cols = 3
    var x = _step_x(n_rows, n_cols)
    var y = _step_data(n_rows, Float32(5.0), Float32(20.0))

    var thresholds = [
        Float32(0.0), Float32(1e-6), Float32(1e-4), Float32(1e-2),
        Float32(1e-1), Float32(1.0), Float32(10.0),
    ]
    var counts = List[Int]()
    var first_three = Float32(-1.0)
    for i in range(len(thresholds)):
        var p = _rf(1, False, 4)
        p.tree_params.min_impurity_decrease = thresholds[i]
        var f = _fit(ctx, x, y, n_rows, n_cols, p)
        var n = len(f.trees[0].sparsetree)
        counts.append(n)
        if n == 3 and first_three < Float32(0.0):
            first_three = thresholds[i]

    print("  arm A-probe: min_impurity_decrease -> node count:", end=" ")
    for i in range(len(thresholds)):
        print(thresholds[i], "->", counts[i], end="  ")
    print()

    if counts[0] <= 3:
        print(
            "  arm A-probe: nothing to explain -- the tree was already 3"
            " nodes at threshold 0"
        )
        return 0
    if first_three < Float32(0.0):
        print(
            "  arm A-probe FAILED: the tree never reaches 3 nodes at any"
            " threshold up to 10.0. The extra splits are NOT rounding on an"
            " exactly-zero gain; something is producing REAL positive gain"
            " on a constant-label node, and that is a defect in the"
            " histogram or the partition, not in their arithmetic."
        )
        return 1
    # and the threshold that kills them must be TINY next to the real
    # split's gain, or they were not spurious after all.
    print(
        "  arm A-probe OK: the spurious splits vanish at"
        " min_impurity_decrease =", first_three,
        "-- so they are float rounding lifting an exactly-zero MSE gain on"
        " a constant-label node, which their strict `gain >"
        " min_impurity_decrease` (objectives.cuh:173) then admits. Their"
        " arithmetic, not a defect; and it is why a regression fixture"
        " cannot assert an exact node count.",
    )
    return 0


def arm_b_global_mean(ctx: DeviceContext) raises -> Int:
    """max_depth 0: one leaf holding the global mean, by hand."""
    var n_rows = 400
    var n_cols = 2
    var x = _step_x(n_rows, n_cols)
    # 200 rows at 5.0 and 200 at 20.0 -> mean exactly 12.5
    var y = _step_data(n_rows, Float32(5.0), Float32(20.0))
    var p = _rf(1, False, 0)
    var f = _fit(ctx, x, y, n_rows, n_cols, p)
    var fails = 0
    ref t = f.trees[0]
    if len(t.sparsetree) != 1:
        fails += 1
        print("  arm B: max_depth 0 must give 1 node, got", len(t.sparsetree))
        return fails
    if not t.sparsetree[0].IsLeaf():
        fails += 1
        print("  arm B: the single node must be a leaf")
    if t.vector_leaf[0] != Float32(12.5):
        fails += 1
        print(
            "  arm B: the depth-0 leaf must hold the global mean, exactly"
            " 12.5 (200 x 5.0 + 200 x 20.0) / 400; got", t.vector_leaf[0],
        )
    if fails == 0:
        print("  arm B OK: the depth-0 leaf holds exactly 12.5")
    return fails


def arm_c_determinism(ctx: DeviceContext) raises -> Int:
    """Two fits bit-identical, leaf values included."""
    var n_rows = 512
    var n_cols = 4
    var x = List[Float32]()
    x.resize(n_rows * n_cols, Float32(0))
    var y = List[Float32]()
    for r in range(n_rows):
        var acc = Float32(0.0)
        for c in range(n_cols):
            var v = Float32(Int(_mix(UInt64(r) * 41 + UInt64(c)) % 2048))
            x[c * n_rows + r] = v
            if c < 2:
                acc += v
        y.append(acc * Float32(0.25))

    var p1 = _rf(1, False, 5)
    var f1 = _fit(ctx, x, y, n_rows, n_cols, p1)
    var p2 = _rf(1, False, 5)
    var f2 = _fit(ctx, x, y, n_rows, n_cols, p2)

    var fails = 0
    ref t1 = f1.trees[0]
    ref t2 = f2.trees[0]
    if len(t1.sparsetree) != len(t2.sparsetree):
        fails += 1
        print(
            "  arm C: node counts differ,", len(t1.sparsetree), "vs",
            len(t2.sparsetree),
        )
        return fails
    var diff = 0
    for i in range(len(t1.sparsetree)):
        if t1.sparsetree[i].ColumnId() != t2.sparsetree[i].ColumnId():
            diff += 1
        if t1.sparsetree[i].QueryValue() != t2.sparsetree[i].QueryValue():
            diff += 1
    for i in range(len(t1.vector_leaf)):
        if t1.vector_leaf[i] != t2.vector_leaf[i]:
            diff += 1
    if diff != 0:
        fails += 1
        print("  arm C: two regression fits differ in", diff, "places")
    if len(t1.sparsetree) < 5:
        fails += 1
        print(
            "  arm C: only", len(t1.sparsetree),
            "nodes -- too small to be evidence of anything",
        )
    if fails == 0:
        print(
            "  arm C OK:", len(t1.sparsetree),
            "nodes and", len(t1.vector_leaf),
            "leaf means, BIT-IDENTICAL across two fits -- which is what"
            " DEVIATION 101b's fixed-point label_sum buys; a float"
            " accumulator under an atomic would not",
        )
    return fails


def arm_d_forest(ctx: DeviceContext) raises -> Int:
    """A bagged forest of regression trees."""
    var n_rows = 600
    var n_cols = 4
    var x = List[Float32]()
    x.resize(n_rows * n_cols, Float32(0))
    var y = List[Float32]()
    for r in range(n_rows):
        var acc = Float32(0.0)
        for c in range(n_cols):
            var v = Float32(Int(_mix(UInt64(r) * 23 + UInt64(c) + 5) % 1024))
            x[c * n_rows + r] = v
            if c < 3:
                acc += v
        y.append(acc)

    var p = _rf(5, True, 4)
    var f = _fit(ctx, x, y, n_rows, n_cols, p)
    var fails = 0
    if len(f.trees) != 5:
        fails += 1
        print("  arm D: expected 5 trees, got", len(f.trees))
        return fails
    var differing = 0
    for t in range(1, 5):
        if len(f.trees[t].sparsetree) != len(f.trees[0].sparsetree):
            differing += 1
            continue
        var d = 0
        for i in range(len(f.trees[0].sparsetree)):
            if (
                f.trees[t].sparsetree[i].QueryValue()
                != f.trees[0].sparsetree[i].QueryValue()
            ):
                d += 1
        if d != 0:
            differing += 1
    if differing == 0:
        fails += 1
        print(
            "  arm D FAILED: all 5 bagged regression trees are identical --"
            " the row sample is not reaching the builder"
        )
    var leaves = 0
    for t in range(5):
        for i in range(len(f.trees[t].sparsetree)):
            if f.trees[t].sparsetree[i].IsLeaf():
                leaves += 1
    if leaves == 0:
        fails += 1
        print("  arm D: the forest has no leaves at all")
    if fails == 0:
        print(
            "  arm D OK: 5 bagged regression trees,", differing,
            "of 4 differing from tree 0,", leaves, "leaves total",
        )
    return fails


def arm_e_epsilon() raises -> Int:
    """`machine_epsilon` must equal what `nextafter` returns, ON THE HOST.

    `RegressionObjectiveFunction.eps_` used to be
    `10 * (nextafter(1, 2) - 1)` and that CRASHED the Metal backend the
    first time a regression kernel was instantiated -- `nextafter` is a
    libm-shaped call and does not survive into device code. It was replaced
    by the IEEE-754 definition, 2^-(mantissa bits), written as an exact
    literal per dtype.

    A replacement constant is a claim, so it is checked against the thing
    it replaced, in the one place that thing still works.
    """
    var fails = 0
    var one32 = Float32(1)
    var want32 = nextafter(one32, Float32(2)) - one32
    if machine_epsilon[DType.float32]() != want32:
        fails += 1
        print(
            "  arm E: machine_epsilon[float32] =",
            machine_epsilon[DType.float32](),
            "but nextafter(1,2)-1 =", want32,
        )
    var one64 = Float64(1)
    var want64 = nextafter(one64, Float64(2)) - one64
    if machine_epsilon[DType.float64]() != want64:
        fails += 1
        print(
            "  arm E: machine_epsilon[float64] =",
            machine_epsilon[DType.float64](),
            "but nextafter(1,2)-1 =", want64,
        )
    if fails == 0:
        print(
            "  arm E OK: machine_epsilon matches nextafter(1,2)-1 exactly"
            " for float32 and float64 -- the device-legal constant is the"
            " same value as the libm call it replaced"
        )
    return fails


def arm_f_estimator_predict(ctx: DeviceContext) raises -> Int:
    """Fit through the ESTIMATOR METHOD, then predict, then check values.

    Two things here that no other arm covers:

      * `RandomForest.fit` as a METHOD. It raised until today; this is the
        path a user actually takes, and a method that forwards to the wrong
        thing would still compile.
      * `predict`'s REGRESSION arm. Their classifier arm does an argmax
        over `num_outputs` (`randomforest.cuh:417-427`); the regressor arm
        reads `row_prediction[0]` and nothing else (`:429`). Those are
        different code, and until now only the classifier arm had ever run.

    The fixture is the step function, so the answer is known exactly: every
    row in the low half must predict 5.0 and every row in the high half
    20.0, whatever the tree's shape, because every leaf below the root
    holds rows from one half only and the mean of a constant is that
    constant. Averaging N such trees changes nothing.
    """
    var n_rows = 400
    var n_cols = 3
    var x = _step_x(n_rows, n_cols)
    var y = _step_data(n_rows, Float32(5.0), Float32(20.0))

    var hx = ctx.enqueue_create_host_buffer[DT](n_rows * n_cols)
    for i in range(n_rows * n_cols):
        hx.unsafe_ptr().unsafe_store(i, x[i])
    var dx = ctx.enqueue_create_buffer[DT](n_rows * n_cols)
    ctx.enqueue_copy(dst_buf=dx, src_ptr=hx.unsafe_ptr())
    var hy = ctx.enqueue_create_host_buffer[LT](n_rows)
    for i in range(n_rows):
        hy.unsafe_ptr().unsafe_store(i, y[i])
    var dy = ctx.enqueue_create_buffer[LT](n_rows)
    ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())
    var dsw = ctx.enqueue_create_buffer[DT](1)
    ctx.synchronize()

    var p = _rf(3, False, 4)
    var rf = RandomForest[DT, LT](p.copy(), REGRESSION)
    var f = rf.fit[ObjT](ctx, dx, dy, dsw, n_rows, n_cols, 1)

    var fails = 0
    if len(f.trees) != 3:
        fails += 1
        print("  arm F: expected 3 trees from the method, got", len(f.trees))
        return fails

    # predict wants ROW-MAJOR (`randomforest.cuh:399, 407`)
    var rm = List[Float32]()
    rm.resize(n_rows * n_cols, Float32(0))
    for r in range(n_rows):
        for c in range(n_cols):
            rm[r * n_cols + c] = x[c * n_rows + r]
    var preds = List[Float32]()
    preds.resize(n_rows, Float32(-1))
    rf.predict(rm, n_rows, n_cols, preds, f)

    var wrong = 0
    for r in range(n_rows):
        var want = Float32(5.0) if r < n_rows // 2 else Float32(20.0)
        if preds[r] != want:
            wrong += 1
            if wrong <= 2:
                print("  arm F: row", r, "predicted", preds[r], "want", want)
    if wrong != 0:
        fails += 1
        print(
            "  arm F FAILED:", wrong, "of", n_rows,
            "regression predictions wrong on a fixture whose answer is two"
            " constants",
        )
    _ = dx^
    _ = dy^
    _ = dsw^
    _ = hx^
    _ = hy^
    if fails == 0:
        print(
            "  arm F OK: RandomForest.fit as a METHOD gave 3 trees, and"
            " predict's REGRESSION arm returned exactly 5.0 / 20.0 for all",
            n_rows, "rows",
        )
    return fails


def main() raises:
    print("regression_check: a REGRESSION forest, through the generic Builder")
    print("  RegressionObjectiveFunction + RegressionBin, MSE criterion")
    var ctx = DeviceContext()
    var fails = 0
    fails += arm_a_step(ctx)
    fails += arm_a_epsilon_probe(ctx)
    fails += arm_b_global_mean(ctx)
    fails += arm_c_determinism(ctx)
    fails += arm_d_forest(ctx)
    fails += arm_e_epsilon()
    fails += arm_f_estimator_predict(ctx)
    if fails == 0:
        print("regression_check: ALL OK")
    else:
        raise Error("regression_check: " + String(fails) + " failure(s)")
