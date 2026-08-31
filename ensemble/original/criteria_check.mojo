# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Every split criterion cuML accepts, actually run.

    pixi run mojo run -I . ensemble/original/criteria_check.mojo

WHY THIS FILE EXISTS. `DecisionTreeParams.check()` refuses only MAE (which
cuML refuses too) and out-of-range values. So a caller can set ENTROPY,
POISSON, GAMMA or INVERSE_GAUSSIAN today and it will be accepted -- and
until this file, not one of them had ever been RUN. They compiled, they were
reachable, and nothing exercised them.

That is precisely this repository's rule 8: **a non-default path is an
unchecked path.** The scar behind it is `ball_cover`, which shipped opt-in
behind a flag, passed set-equality at five configurations with two
sabotages, and was still passing the whole dataset as the query on every
batch -- 412 of 612 labels wrong. The check that would have caught it
existed and was green, because with the flag off it exercised the other
branch. **Flipping the default is what ran the check, and the check failed
on the first try.**

WHAT THIS FILE ASSERTS, and what it deliberately does not. It does NOT claim
the four rarer criteria compute cuML's numbers -- that needs their per-cell
output on the NVIDIA column and this box cannot run cuML. It asserts three
things that are checkable here and that catch the failure modes that
actually occur:

  1. EACH ONE TRAINS. A criterion that raises, hangs, crashes the Metal
     backend, or silently returns a single leaf is caught. (The Metal crash
     is not hypothetical: three of these four crashed the backend until
     `nextafter` came out of `eps_` earlier today, and nothing here ran
     them, so nothing noticed.)
  2. EACH ONE IS ACTUALLY REACHED. Two criteria that produce identical
     trees on data with real structure are indistinguishable from one
     criterion and a dead branch. Their `GainPerSplit` is a switch
     (`objectives.cuh:132-136`, `:331-338`) whose `default:` arm returns
     `-max()` for every candidate, i.e. NO valid split anywhere -- so a
     mis-wired criterion constant does not error, it silently produces a
     stump. Both failure directions are asserted.
  3. THE DOMAIN GUARDS FIRE. Poisson, Gamma and InverseGaussian all refuse
     non-positive label sums (`objectives.cuh:251-253`, `:279-281`,
     `:306-308`), returning `-max()`. Fed negative labels they must produce
     a stump rather than a tree of garbage, and that is checked separately
     from the positive-label case.
"""

from max.gpu.host import DeviceContext

from ensemble.decisiontree.batched_levelalgo.bins import (
    ClassificationBin,
    RegressionBin,
)
from ensemble.decisiontree.batched_levelalgo.objectives import (
    ClassificationObjectiveFunction,
    RegressionObjectiveFunction,
)
from ensemble.decisiontree.decisiontree import (
    CRITERION_END,
    DecisionTreeParams,
    ENTROPY,
    GAMMA,
    GINI,
    INVERSE_GAUSSIAN,
    MAE,
    MSE,
    POISSON,
    criterion_name,
)
from ensemble.randomforest import RF_params, RandomForestMetaData, fit_forest

comptime DT = DType.float32
comptime CLS_LT = DType.int32
comptime REG_LT = DType.float32
comptime ClsObj = ClassificationObjectiveFunction[DT, CLS_LT, ClassificationBin]
comptime RegObj = RegressionObjectiveFunction[DT, REG_LT, RegressionBin]


@always_inline
def _mix(x: UInt64) -> UInt64:
    var h = x
    h ^= h >> 33
    h *= 0xFF51AFD7ED558CCD
    h ^= h >> 33
    h *= 0xC4CEB9FE1A85EC53
    h ^= h >> 33
    return h


def _params(criterion: Int, min_samples_leaf: Int = 1) -> RF_params:
    return RF_params(
        n_trees=Int32(1),
        bootstrap=False,
        max_samples=Float32(1.0),
        seed=UInt64(2024),
        n_streams=Int32(1),
        tree_params=DecisionTreeParams(
            max_depth=Int32(4),
            max_leaves=Int32(-1),
            max_features=Float32(1.0),
            max_n_bins=Int32(16),
            min_samples_leaf=Int32(min_samples_leaf),
            min_samples_split=Int32(2),
            split_criterion=criterion,
            min_impurity_decrease=Float32(0.0),
            max_batch_size=Int32(128),
        ),
    )


struct Shape(Copyable, Movable):
    """A tree's shape, reduced to what a criterion can move."""

    var n_nodes: Int
    var cols: List[Int32]
    var quesvals: List[Float32]

    def __init__(out self):
        self.n_nodes = 0
        self.cols = List[Int32]()
        self.quesvals = List[Float32]()

    def differs_from(self, other: Self) -> Bool:
        if self.n_nodes != other.n_nodes:
            return True
        for i in range(len(self.cols)):
            if self.cols[i] != other.cols[i]:
                return True
            if self.quesvals[i] != other.quesvals[i]:
                return True
        return False


def _shape_of[
    lt: DType
](f: RandomForestMetaData[DT, lt]) -> Shape:
    var s = Shape()
    ref t = f.trees[0]
    s.n_nodes = len(t.sparsetree)
    for i in range(len(t.sparsetree)):
        s.cols.append(t.sparsetree[i].ColumnId())
        s.quesvals.append(t.sparsetree[i].QueryValue())
    return s^


def _fit_cls(
    ctx: DeviceContext,
    x: List[Float32],
    y: List[Int32],
    n_rows: Int,
    n_cols: Int,
    n_classes: Int,
    criterion: Int,
    min_samples_leaf: Int = 1,
) raises -> Shape:
    var hx = ctx.enqueue_create_host_buffer[DT](n_rows * n_cols)
    for i in range(n_rows * n_cols):
        hx.unsafe_ptr().unsafe_store(i, x[i])
    var dx = ctx.enqueue_create_buffer[DT](n_rows * n_cols)
    ctx.enqueue_copy(dst_buf=dx, src_ptr=hx.unsafe_ptr())
    var hy = ctx.enqueue_create_host_buffer[CLS_LT](n_rows)
    for i in range(n_rows):
        hy.unsafe_ptr().unsafe_store(i, y[i])
    var dy = ctx.enqueue_create_buffer[CLS_LT](n_rows)
    ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())
    var dsw = ctx.enqueue_create_buffer[DT](1)
    ctx.synchronize()
    var p = _params(criterion, min_samples_leaf)
    var f = fit_forest[ClsObj](
        ctx, dx, dy, dsw, n_rows, n_cols, n_classes, p
    )
    var s = _shape_of(f)
    _ = dx^
    _ = dy^
    _ = dsw^
    _ = hx^
    _ = hy^
    return s^


def _fit_reg(
    ctx: DeviceContext,
    x: List[Float32],
    y: List[Float32],
    n_rows: Int,
    n_cols: Int,
    criterion: Int,
    min_samples_leaf: Int = 1,
) raises -> Shape:
    var hx = ctx.enqueue_create_host_buffer[DT](n_rows * n_cols)
    for i in range(n_rows * n_cols):
        hx.unsafe_ptr().unsafe_store(i, x[i])
    var dx = ctx.enqueue_create_buffer[DT](n_rows * n_cols)
    ctx.enqueue_copy(dst_buf=dx, src_ptr=hx.unsafe_ptr())
    var hy = ctx.enqueue_create_host_buffer[REG_LT](n_rows)
    for i in range(n_rows):
        hy.unsafe_ptr().unsafe_store(i, y[i])
    var dy = ctx.enqueue_create_buffer[REG_LT](n_rows)
    ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())
    var dsw = ctx.enqueue_create_buffer[DT](1)
    ctx.synchronize()
    var p = _params(criterion, min_samples_leaf)
    var f = fit_forest[RegObj](ctx, dx, dy, dsw, n_rows, n_cols, 1, p)
    var s = _shape_of(f)
    _ = dx^
    _ = dy^
    _ = dsw^
    _ = hx^
    _ = hy^
    return s^


def _cls_data(n_rows: Int, n_cols: Int, n_classes: Int) -> Tuple[
    List[Float32], List[Int32]
]:
    var x = List[Float32]()
    x.resize(n_rows * n_cols, Float32(0))
    var y = List[Int32]()
    for r in range(n_rows):
        var acc = UInt64(0)
        for c in range(n_cols):
            var v = Int(_mix(UInt64(r) * 61 + UInt64(c) + 13) % 500)
            x[c * n_rows + r] = Float32(v)
            if c < 2:
                acc += UInt64(v)
        y.append(Int32(Int(acc % UInt64(n_classes))))
    return (x^, y^)


def _reg_data(n_rows: Int, n_cols: Int, positive: Bool) -> Tuple[
    List[Float32], List[Float32]
]:
    """Labels STRICTLY POSITIVE when asked, because Poisson, Gamma and
    InverseGaussian all return `-max()` for a non-positive label sum
    (`objectives.cuh:251-253`, `:279-281`, `:306-308`)."""
    var x = List[Float32]()
    x.resize(n_rows * n_cols, Float32(0))
    var y = List[Float32]()
    for r in range(n_rows):
        var acc = Float32(0.0)
        for c in range(n_cols):
            var v = Float32(Int(_mix(UInt64(r) * 71 + UInt64(c) + 5) % 400))
            x[c * n_rows + r] = v
            if c < 2:
                acc += v
        if positive:
            y.append(acc * Float32(0.01) + Float32(1.0))
        else:
            y.append(-(acc * Float32(0.01) + Float32(1.0)))
    return (x^, y^)


def arm_a_classification(ctx: DeviceContext) raises -> Int:
    """GINI and ENTROPY both run, and are distinguishable."""
    var n_rows = 600
    var n_cols = 4
    var d = _cls_data(n_rows, n_cols, 3)
    var fails = 0

    var gini = _fit_cls(ctx, d[0], d[1], n_rows, n_cols, 3, GINI)
    var entropy = _fit_cls(ctx, d[0], d[1], n_rows, n_cols, 3, ENTROPY)

    if gini.n_nodes < 5:
        fails += 1
        print(
            "  arm A: GINI produced only", gini.n_nodes,
            "nodes -- a stump, so the criterion is not reaching a gain",
        )
    if entropy.n_nodes < 5:
        fails += 1
        print(
            "  arm A: ENTROPY produced only", entropy.n_nodes,
            "nodes -- a stump. Their `default:` arm returns -max() for"
            " every candidate, so a mis-wired criterion looks exactly like"
            " this rather than erroring.",
        )
    if not gini.differs_from(entropy):
        fails += 1
        print(
            "  arm A: GINI and ENTROPY produced the IDENTICAL tree on data"
            " with real structure. One of them is not being reached --"
            " indistinguishable from one criterion and a dead branch."
        )
    if fails == 0:
        print(
            "  arm A OK:", criterion_name(GINI), gini.n_nodes, "nodes,",
            criterion_name(ENTROPY), entropy.n_nodes,
            "nodes, and the two trees DIFFER -- both are reached",
        )
    return fails


def arm_b_regression(ctx: DeviceContext) raises -> Int:
    """All four regression criteria run on positive labels."""
    var n_rows = 600
    var n_cols = 4
    var d = _reg_data(n_rows, n_cols, True)
    var fails = 0

    var crits = [MSE, POISSON, GAMMA, INVERSE_GAUSSIAN]
    var shapes = List[Shape]()
    for i in range(len(crits)):
        var s = _fit_reg(ctx, d[0], d[1], n_rows, n_cols, crits[i])
        if s.n_nodes < 5:
            fails += 1
            print(
                "  arm B:", criterion_name(crits[i]), "produced only",
                s.n_nodes, "nodes -- a stump. Note their `default:` arm"
                " returns -max() for every candidate, so an unreached"
                " criterion looks like this and does not error.",
            )
        shapes.append(s^)

    # Every pair must differ from at least one other, or they are not all
    # being reached. MSE against each of the other three is the sharp test:
    # the three share a domain guard MSE does not have, and all three
    # compute a different objective.
    var same_as_mse = 0
    for i in range(1, len(crits)):
        if not shapes[0].differs_from(shapes[i]):
            same_as_mse += 1
            print(
                "  arm B:", criterion_name(crits[i]),
                "produced the IDENTICAL tree to MSE -- it is not being"
                " reached",
            )
    if same_as_mse != 0:
        fails += 1

    if fails == 0:
        print(
            "  arm B OK: MSE", shapes[0].n_nodes, "nodes, POISSON",
            shapes[1].n_nodes, "nodes, GAMMA", shapes[2].n_nodes,
            "nodes, INVERSE_GAUSSIAN", shapes[3].n_nodes,
            "nodes -- all four train and all three differ from MSE",
        )
    return fails


def arm_c_domain_guards(ctx: DeviceContext) raises -> Int:
    """Negative labels must STUMP Poisson/Gamma/IG, not train garbage."""
    var n_rows = 600
    var n_cols = 4
    var neg = _reg_data(n_rows, n_cols, False)
    var pos = _reg_data(n_rows, n_cols, True)
    var fails = 0

    # MSE is the control: it has NO domain guard, so it must still train on
    # negative labels. If it stumps too, the fixture is broken rather than
    # the guards being proven.
    var mse_neg = _fit_reg(ctx, neg[0], neg[1], n_rows, n_cols, MSE)
    if mse_neg.n_nodes < 5:
        fails += 1
        print(
            "  arm C: MSE stumped on negative labels, but MSE has no domain"
            " guard (objectives.cuh:213-234). The fixture is wrong, not the"
            " guards."
        )

    for c in [POISSON, GAMMA, INVERSE_GAUSSIAN]:
        var s_neg = _fit_reg(ctx, neg[0], neg[1], n_rows, n_cols, c)
        var s_pos = _fit_reg(ctx, pos[0], pos[1], n_rows, n_cols, c)
        if s_neg.n_nodes != 1:
            fails += 1
            print(
                "  arm C:", criterion_name(c), "on NEGATIVE labels gave",
                s_neg.n_nodes,
                "nodes; their guard returns -max() for a non-positive label"
                " sum, so no split can be valid and the answer must be a"
                " single leaf",
            )
        if s_pos.n_nodes < 5:
            fails += 1
            print(
                "  arm C:", criterion_name(c),
                "stumped on POSITIVE labels too, so the previous line"
                " proves nothing about the guard",
            )

    if fails == 0:
        print(
            "  arm C OK: MSE still trains on negative labels (no guard),"
            " while POISSON, GAMMA and INVERSE_GAUSSIAN each collapse to"
            " exactly 1 node on negative labels and train on positive ones"
            " -- the domain guards fire, and fire only where they exist"
        )
    return fails


def arm_d_mae_refused() raises -> Int:
    """MAE is refused by name, as cuML refuses it."""
    var p = _params(MAE)
    var raised = False
    try:
        p.tree_params.check()
    except e:
        raised = True
        if String(e).find("MAE") < 0:
            print("  arm D: MAE raised but not by name:", e)
            return 1
    if not raised:
        print(
            "  arm D FAILED: MAE was accepted. cuML refuses it at"
            " decisiontree.cu:28 and randomforest_common.pyx:147; there is"
            " no upstream MAE path to port, so accepting it would train"
            " something that is not MAE."
        )
        return 1
    print("  arm D OK: MAE refused by name, as cuML refuses it")
    return 0


def arm_e_criterion_end(ctx: DeviceContext) raises -> Int:
    """`decisiontree.cuh:251-256`. CRITERION_END is the sentinel their
    header defaults `split_criterion` to (`decisiontree.hpp:89`), and they
    resolve it to GINI for an integer LabelT and MSE for a float one
    before any objective is built.

    This port carried the sentinel and resolved it NOWHERE: a params struct
    left at the C++ default reached training with `split_criterion == 7`
    and passed every validator on the way. So the resolution is checked by
    the only thing that can see it -- the TREE. Fitting at CRITERION_END
    must give bit-identically the tree that GINI gives (classification) or
    MSE gives (regression), and the comparison is only worth anything
    because a DIFFERENT criterion gives a different tree, which arms A and
    B already establish and this arm re-establishes locally."""
    print()
    print("ARM E -- CRITERION_END resolves to GINI / MSE")
    var wrong = 0

    var n_rows = 400
    var n_cols = 3
    var xs = List[Float32]()
    var ys = List[Int32]()
    for j in range(n_cols):
        for i in range(n_rows):
            var h = _mix(UInt64(i * 31 + j * 7 + 1))
            xs.append(Float32(Int(h % UInt64(1000))) / Float32(1000.0))
    for i in range(n_rows):
        ys.append(Int32(i % 3))

    var end_cls = _fit_cls(ctx, xs, ys, n_rows, n_cols, 3, CRITERION_END)
    var gini = _fit_cls(ctx, xs, ys, n_rows, n_cols, 3, GINI)
    var entropy = _fit_cls(ctx, xs, ys, n_rows, n_cols, 3, ENTROPY)
    print(
        "    classification: CRITERION_END ->",
        end_cls.n_nodes,
        "nodes, GINI ->",
        gini.n_nodes,
        "nodes, ENTROPY ->",
        entropy.n_nodes,
        "nodes",
    )
    if end_cls.differs_from(gini):
        print("      FAIL: CRITERION_END did not resolve to GINI")
        wrong += 1
    if not gini.differs_from(entropy):
        print(
            "      FAIL: GINI and ENTROPY agree on this fixture, so the"
            " equality above proves nothing"
        )
        wrong += 1

    var yr = List[Float32]()
    for i in range(n_rows):
        yr.append(Float32(i % 5) * Float32(1.5))
    var end_reg = _fit_reg(ctx, xs, yr, n_rows, n_cols, CRITERION_END)
    var mse = _fit_reg(ctx, xs, yr, n_rows, n_cols, MSE)
    var poisson = _fit_reg(ctx, xs, yr, n_rows, n_cols, POISSON)
    print(
        "    regression:     CRITERION_END ->",
        end_reg.n_nodes,
        "nodes, MSE ->",
        mse.n_nodes,
        "nodes, POISSON ->",
        poisson.n_nodes,
        "nodes",
    )
    if end_reg.differs_from(mse):
        print("      FAIL: CRITERION_END did not resolve to MSE")
        wrong += 1
    if not mse.differs_from(poisson):
        print(
            "      FAIL: MSE and POISSON agree here, so the equality above"
            " proves nothing"
        )
        wrong += 1

    if wrong == 0:
        print(
            "  arm E OK: CRITERION_END gives GINI's tree for an integer"
            " label and MSE's for a float one, and both differ from a"
            " neighbouring criterion"
        )
    return wrong


def arm_f_params_reach_the_device(ctx: DeviceContext) raises -> Int:
    """`builder.cuh:592-596` builds the objective from `params` INSIDE
    `computeSplit`, which is what makes `params` the single source of
    truth for `min_samples_leaf`, `split_criterion` and
    `min_impurity_decrease`.

    Ours used to take a pre-built objective from the caller and never read
    those three from `params` at all, so setting them there was silently
    ignored. Nothing caught it because every caller happened to build its
    objective out of the same params.

    So this arm passes NO objective -- there is nowhere left to pass one --
    and requires `params.min_samples_leaf` to move the tree on its own."""
    print()
    print("ARM F -- params reach the device with no objective in sight")
    var wrong = 0
    var n_rows = 400
    var n_cols = 3
    var xs = List[Float32]()
    var ys = List[Int32]()
    for j in range(n_cols):
        for i in range(n_rows):
            var h = _mix(UInt64(i * 17 + j * 101 + 5))
            xs.append(Float32(Int(h % UInt64(1000))) / Float32(1000.0))
    for i in range(n_rows):
        ys.append(Int32(i % 3))

    var loose = _fit_cls(ctx, xs, ys, n_rows, n_cols, 3, GINI, 1)
    var tight = _fit_cls(ctx, xs, ys, n_rows, n_cols, 3, GINI, 150)
    print(
        "    min_samples_leaf 1 ->",
        loose.n_nodes,
        "nodes; min_samples_leaf 150 ->",
        tight.n_nodes,
        "nodes",
    )
    if tight.n_nodes >= loose.n_nodes:
        print(
            "      FAIL: raising params.min_samples_leaf did not shrink the"
            " tree, so params is not reaching GainPerSplit"
        )
        wrong += 1
    if wrong == 0:
        print(
            "  arm F OK: params.min_samples_leaf alone moved the tree,"
            " so the objective is built from params"
        )
    return wrong


def main() raises:
    print("criteria_check: every split criterion cuML accepts, actually run")
    var ctx = DeviceContext()
    var fails = 0
    fails += arm_a_classification(ctx)
    fails += arm_b_regression(ctx)
    fails += arm_c_domain_guards(ctx)
    fails += arm_d_mae_refused()
    fails += arm_e_criterion_end(ctx)
    fails += arm_f_params_reach_the_device(ctx)
    if fails == 0:
        print("criteria_check: ALL OK")
    else:
        raise Error("criteria_check: " + String(fails) + " failure(s)")
