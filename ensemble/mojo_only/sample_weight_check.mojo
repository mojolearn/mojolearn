"""`sample_weight`: the zero-weight drop, the validation, and their
double-counting rule.

    pixi run mojo run -I . ensemble/mojo_only/sample_weight_check.mojo

Three of cuML's four `RowSampler` arms and both WEIGHTED bin types depended
on `sample_weight`, which this port did not accept. This file covers what is
now wired, and each arm rests on an ANALYTIC IDENTITY rather than on "it
trained and looked plausible".

  A. THE ZERO-WEIGHT DROP, per cell. `selected_rows` must hold exactly the
     nonzero-weight indices IN ORDER (`thrust::copy_if`,
     `randomforest.cuh:146-152`), the builder's root must hold exactly that
     many rows, and the forest must differ from the unweighted one. The
     tempting stronger identity -- "same as fitting the survivors
     directly" -- is FALSE, and why is written at the arm: their quantiles
     are computed on the FULL input before the sampler exists, so a
     zero-weight row still shapes the bin edges.

  B. THE VALIDATION REFUSES, BY VALUE. cuML rejects any non-finite or
     negative weight (`randomforest.cuh:202-208`) and asserts the total is
     strictly positive (`:93-95`). A silently-accepted bad weight trains a
     forest on a row set nobody asked for -- the same failure class as a
     wrong bootstrap, and just as invisible.

  C. THEIR DOUBLE-COUNTING RULE, which is the subtle one.
     `tree_sample_weight()` (`:166-167`) is

         return bootstrap_ ? nullptr : sample_weight_;

     with their comment: "Use sample weights in impurity / objective
     calculation only when bootstrapping is not enabled." When bootstrapping
     the weights are ALREADY expressed by drawing rows in proportion to
     them, so passing them to the objective as well would apply them twice.
     A port that always passed them down would double-count on the DEFAULT
     path and merely look "differently regularised". Checked in both
     directions, both now reachable: weighted bootstrap is ported, so
     bootstrap+weights runs their `:125-138` arm instead of raising.

  D. THE WEIGHTED BINS ARE REACHABLE AND ARE ACTUALLY READ. They have been
     ported since this morning and no caller could construct one.

A NOTE ON SCALES, because it decides what a weighted fixture may contain.
`WeightedClassificationBin.weight` is Int32 FIXED POINT (DEVIATION 101b),
quantized by `BinScales.weight_scale`. At the default scale of 1.0 a weight
of 0.5 truncates to 0 -- so the fixtures below use INTEGER weights, where
scale 1.0 is exact, and the choice is deliberate rather than incidental.
Fractional weights need a scale chosen from the data, which is the caller's
job and is not what this file is checking.
"""

from max.gpu.host import DeviceContext

from ensemble.decisiontree.batched_levelalgo.bins import (
    BinScales,
    ClassificationBin,
    WeightedClassificationBin,
)
from ensemble.decisiontree.batched_levelalgo.objectives import (
    ClassificationObjectiveFunction,
    ObjectiveLike,
)
from ensemble.decisiontree.decisiontree import DecisionTreeParams, GINI
from ensemble.randomforest import (
    RF_params,
    RandomForestMetaData,
    RowSampler,
    fit_forest,
)

comptime DT = DType.float32
comptime LT = DType.int32
comptime PlainObj = ClassificationObjectiveFunction[DT, LT, ClassificationBin]
comptime WtObj = ClassificationObjectiveFunction[
    DT, LT, WeightedClassificationBin
]


@always_inline
def _mix(x: UInt64) -> UInt64:
    var h = x
    h ^= h >> 33
    h *= 0xFF51AFD7ED558CCD
    h ^= h >> 33
    h *= 0xC4CEB9FE1A85EC53
    h ^= h >> 33
    return h


def _params(bootstrap: Bool) -> RF_params:
    return RF_params(
        n_trees=Int32(2),
        bootstrap=bootstrap,
        max_samples=Float32(1.0),
        seed=UInt64(4242),
        n_streams=Int32(1),
        tree_params=DecisionTreeParams(
            max_depth=Int32(4),
            max_leaves=Int32(-1),
            max_features=Float32(1.0),
            max_n_bins=Int32(16),
            min_samples_leaf=Int32(1),
            min_samples_split=Int32(2),
            split_criterion=GINI,
            min_impurity_decrease=Float32(0.0),
            max_batch_size=Int32(128),
        ),
    )


def _data(n_rows: Int, n_cols: Int, n_classes: Int) -> Tuple[
    List[Float32], List[Int32]
]:
    var x = List[Float32]()
    x.resize(n_rows * n_cols, Float32(0))
    var y = List[Int32]()
    for r in range(n_rows):
        var acc = UInt64(0)
        for c in range(n_cols):
            var v = Int(_mix(UInt64(r) * 91 + UInt64(c) + 17) % 400)
            x[c * n_rows + r] = Float32(v)
            if c < 2:
                acc += UInt64(v)
        y.append(Int32(Int(acc % UInt64(n_classes))))
    return (x^, y^)


def _fit[
    O: ObjectiveLike
](
    ctx: DeviceContext,
    x: List[Float32],
    y: List[Int32],
    n_rows: Int,
    n_cols: Int,
    n_classes: Int,
    bootstrap: Bool,
    objective: O,
    weights: List[Float32],
) raises -> RandomForestMetaData[O.DataT, O.LabelT] where O.DataT == DT:
    var hx = ctx.enqueue_create_host_buffer[DT](n_rows * n_cols)
    for i in range(n_rows * n_cols):
        hx.unsafe_ptr().unsafe_store(i, x[i])
    var dx = ctx.enqueue_create_buffer[DT](n_rows * n_cols)
    ctx.enqueue_copy(dst_buf=dx, src_ptr=hx.unsafe_ptr())
    var hy = ctx.enqueue_create_host_buffer[O.LabelT](n_rows)
    for i in range(n_rows):
        hy.unsafe_ptr().unsafe_store(i, rebind[Scalar[O.LabelT]](y[i]))
    var dy = ctx.enqueue_create_buffer[O.LabelT](n_rows)
    ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())

    var nw = len(weights) if len(weights) > 0 else 1
    var hw = ctx.enqueue_create_host_buffer[DT](nw)
    for i in range(len(weights)):
        hw.unsafe_ptr().unsafe_store(i, weights[i])
    var dw = ctx.enqueue_create_buffer[DT](nw)
    ctx.enqueue_copy(dst_buf=dw, src_ptr=hw.unsafe_ptr())
    ctx.synchronize()

    var p = _params(bootstrap)
    var f = fit_forest[O](
        ctx, dx, dy, dw, n_rows, n_cols, n_classes, p, objective, weights
    )
    _ = dx^
    _ = dy^
    _ = dw^
    _ = hx^
    _ = hy^
    _ = hw^
    return f^


def _same[
    lt: DType
](
    a: RandomForestMetaData[DT, lt], b: RandomForestMetaData[DT, lt]
) -> Int:
    """Differing FIELDS across the whole forest."""
    if len(a.trees) != len(b.trees):
        return 1000000
    var diff = 0
    for t in range(len(a.trees)):
        ref ta = a.trees[t]
        ref tb = b.trees[t]
        if len(ta.sparsetree) != len(tb.sparsetree):
            diff += 1000
            continue
        for i in range(len(ta.sparsetree)):
            if ta.sparsetree[i].ColumnId() != tb.sparsetree[i].ColumnId():
                diff += 1
            if ta.sparsetree[i].QueryValue() != tb.sparsetree[i].QueryValue():
                diff += 1
            if (
                ta.sparsetree[i].InstanceCount()
                != tb.sparsetree[i].InstanceCount()
            ):
                diff += 1
        for i in range(len(ta.vector_leaf)):
            if ta.vector_leaf[i] != tb.vector_leaf[i]:
                diff += 1
    return diff


def arm_a_zero_weight_drop(ctx: DeviceContext) raises -> Int:
    """The drop selects exactly the nonzero rows, in order, and the tree
    is built on exactly those.

    THE OBVIOUS IDENTITY DOES NOT HOLD, AND THAT IS A FINDING ABOUT THEIR
    DESIGN RATHER THAN A LIMITATION OF THE CHECK. The first version of this
    arm asserted that fitting `n` rows with the zero-weight ones dropped
    gives the BIT-IDENTICAL forest to fitting the survivors directly. It
    measured 184 field differences, and it was wrong:
    `DT::computeQuantiles` is called on the FULL input with the FULL
    `n_rows` (`randomforest.cuh:318-325`), before the row sampler exists
    and with no knowledge of `sample_weight`. So the bin edges are drawn
    from ALL rows including the zero-weight ones, while fitting the subset
    directly quantizes only the subset. Different edges, different splits,
    and both are correct.

    That is worth knowing independently of this check: a zero-weight row in
    cuML still influences the model, through the quantiles, even though it
    contributes to no histogram.

    So this asserts what IS exactly true: `selected_rows` holds precisely
    the nonzero-weight indices, in increasing order, and the tree the
    builder grows holds exactly that many rows at its root.
    """
    var n_rows = 600
    var n_cols = 4
    var d = _data(n_rows, n_cols, 3)

    var w = List[Float32]()
    var keep = List[Int]()
    for r in range(n_rows):
        if r % 3 == 0:
            w.append(Float32(0.0))
        else:
            w.append(Float32(1.0))
            keep.append(r)
    var n_keep = len(keep)
    var fails = 0

    # --- the sampler, per cell -------------------------------------------
    var s = RowSampler(ctx, False, UInt64(4242), n_rows, n_rows, True)
    s.prepare_weights(ctx, w)
    s.sample(ctx, Int32(0))
    if s.n_selected != n_keep:
        fails += 1
        print(
            "  arm A: n_selected", s.n_selected, "want", n_keep,
        )
    var hb = ctx.enqueue_create_host_buffer[DType.int32](n_rows)
    ctx.enqueue_copy(dst_buf=hb, src_buf=s.selected_rows)
    ctx.synchronize()
    var wrong = 0
    for j in range(min(s.n_selected, n_keep)):
        if Int(hb.unsafe_ptr().unsafe_load(j)) != keep[j]:
            wrong += 1
            if wrong <= 2:
                print(
                    "  arm A: selected_rows[", j, "] =",
                    hb.unsafe_ptr().unsafe_load(j), "want", keep[j],
                )
    if wrong != 0:
        fails += 1
        print(
            "  arm A FAILED:", wrong,
            "of", n_keep, "selected rows wrong. `thrust::copy_if`"
            " (randomforest.cuh:146-152) keeps the survivors IN ORDER.",
        )
    _ = s^
    _ = hb^

    # --- and the builder sees exactly that many rows ----------------------
    var obj = PlainObj(Int32(3), Int32(1), Int32(GINI), Float32(0.0))
    var f = _fit[PlainObj](
        ctx, d[0], d[1], n_rows, n_cols, 3, False, obj, w
    )
    if Int(f.trees[0].sparsetree[0].InstanceCount()) != n_keep:
        fails += 1
        print(
            "  arm A: the root holds",
            f.trees[0].sparsetree[0].InstanceCount(), "rows, want", n_keep,
            "-- n_sampled_rows was left stale after the drop",
        )
    # and it must DIFFER from the unweighted fit, or the drop never reached
    # the builder at all
    var f_all = _fit[PlainObj](
        ctx, d[0], d[1], n_rows, n_cols, 3, False, obj, List[Float32]()
    )
    if Int(f_all.trees[0].sparsetree[0].InstanceCount()) != n_rows:
        fails += 1
        print("  arm A: the unweighted control did not use all rows")
    if _same(f, f_all) == 0:
        fails += 1
        print(
            "  arm A FAILED: the zero-weight fit is identical to the"
            " unweighted fit -- the drop is not reaching the builder"
        )

    if fails == 0:
        print(
            "  arm A OK: the drop selected exactly", n_keep,
            "of", n_rows, "rows, in order, per cell; the root holds",
            n_keep, "rows; and the forest differs from the unweighted one",
        )
    return fails


def arm_b_validation(ctx: DeviceContext) raises -> Int:
    """NaN, negative and all-zero must each raise, by value."""
    var n_rows = 64
    var fails = 0
    var s = RowSampler(ctx, False, UInt64(1), n_rows, n_rows, True)

    var nan_w = List[Float32]()
    for i in range(n_rows):
        nan_w.append(Float32(1.0))
    nan_w[7] = Float32(0.0) / Float32(0.0)
    var r1 = False
    try:
        s.prepare_weights(ctx, nan_w)
    except e:
        r1 = True
        if String(e).find("NaN") < 0:
            print("  arm B: NaN raised but not by name:", e)
    if not r1:
        fails += 1
        print("  arm B FAILED: a NaN weight was accepted")

    var neg_w = List[Float32]()
    for i in range(n_rows):
        neg_w.append(Float32(1.0))
    neg_w[3] = Float32(-2.0)
    var r2 = False
    try:
        s.prepare_weights(ctx, neg_w)
    except e:
        r2 = True
    if not r2:
        fails += 1
        print("  arm B FAILED: a negative weight was accepted")

    var zero_w = List[Float32]()
    for i in range(n_rows):
        zero_w.append(Float32(0.0))
    var r3 = False
    try:
        s.prepare_weights(ctx, zero_w)
    except e:
        r3 = True
    if not r3:
        fails += 1
        print(
            "  arm B FAILED: an all-zero sample_weight was accepted;"
            " randomforest.cuh:93-95 asserts the total is positive"
        )

    # and a VALID one must be accepted and count correctly
    var ok_w = List[Float32]()
    for i in range(n_rows):
        ok_w.append(Float32(0.0) if i % 4 == 0 else Float32(1.0))
    s.prepare_weights(ctx, ok_w)
    if s.n_selected != n_rows - (n_rows // 4):
        fails += 1
        print(
            "  arm B: n_selected is", s.n_selected, "want",
            n_rows - (n_rows // 4),
        )
    _ = s^
    if fails == 0:
        print(
            "  arm B OK: NaN, negative and all-zero each refused; a valid"
            " weight vector selects exactly the nonzero rows"
        )
    return fails


def arm_c_double_counting(ctx: DeviceContext) raises -> Int:
    """Their `bootstrap ? nullptr : sample_weight` rule -- ONE direction,
    and the other one stated as a gap rather than skipped quietly.

    `tree_sample_weight()` (`randomforest.cuh:166-167`) returns `nullptr`
    when bootstrapping, with their comment: "Use sample weights in impurity
    / objective calculation only when bootstrapping is not enabled." When
    bootstrapping, the weights are already expressed by DRAWING rows in
    proportion to them, so applying them in the objective too would count
    them twice.

    WHAT IS CHECKABLE HERE: the OFF direction. With `bootstrap=False` the
    objective is the only place the weights can act, so varying them MUST
    change the forest. If it does not, the weighted bin is not reading
    `dataset.sample_weight` and the whole weighted path is inert.

    WHAT IS NOT, AND WHY -- this is a REACH GAP, recorded rather than
    hidden. The ON direction needs `bootstrap=True` WITH weights, and their
    own dispatch sends exactly that combination to the WEIGHTED BOOTSTRAP
    arm (`use_weighted_bootstrap() = bootstrap_ && sample_weight_ !=
    nullptr`, `:214`), which is not ported -- it needs
    `raft::random::uniform<double>`, declined under DEVIATION 187c. So
    there is no configuration in this port today that reaches
    `tree_sample_weight() == nullptr` with weights present. The rule IS
    implemented (`objective_sees_weights = has_sw and not bootstrap` in
    `fit_forest`), and it is UNCHECKED until weighted bootstrap lands.
    Whoever lands it should add the other direction here first.
    """
    var n_rows = 600
    var n_cols = 4
    var d = _data(n_rows, n_cols, 3)

    # INTEGER weights, so the default weight_scale of 1.0 is exact.
    var w = List[Float32]()
    var ones = List[Float32]()
    for r in range(n_rows):
        w.append(Float32(1 + Int(_mix(UInt64(r) + 3) % 3)))
        ones.append(Float32(1.0))

    var obj = WtObj(
        Int32(3), Int32(1), Int32(GINI), Float32(0.0), BinScales(1.0, 1.0)
    )
    var fails = 0

    var nb_w = _fit[WtObj](ctx, d[0], d[1], n_rows, n_cols, 3, False, obj, w)
    var nb_1 = _fit[WtObj](
        ctx, d[0], d[1], n_rows, n_cols, 3, False, obj, ones
    )
    if _same(nb_w, nb_1) == 0:
        fails += 1
        print(
            "  arm C FAILED: with bootstrap OFF, varied weights made NO"
            " difference -- the weighted bin is not reading"
            " dataset.sample_weight, so the whole weighted path is inert."
        )

    # THE OTHER DIRECTION, reachable now that weighted bootstrap is ported.
    # With bootstrap ON the objective must NOT see the weights
    # (tree_sample_weight() returns nullptr, randomforest.cuh:167) -- the
    # weights are already in the DRAW. So a fit with varied weights and a
    # fit with all-ones weights must differ ONLY through the row sample,
    # never through the objective. The way to see that: the weighted
    # bootstrap draws rows in proportion to the weights, so the two forests
    # DIFFER -- but the objective contribution is identical, which shows up
    # as both being reproducible and neither carrying a weighted histogram.
    #
    # The sharp assertion available here: run the SAME weighted bootstrap
    # twice with the weighted bin and with the PLAIN bin. If the objective
    # were seeing the weights, those two would differ; since it is not, the
    # row sample is the only input and both must produce the same tree.
    var plain = PlainObj(Int32(3), Int32(1), Int32(GINI), Float32(0.0))
    var boot_wt = _fit[WtObj](ctx, d[0], d[1], n_rows, n_cols, 3, True, obj, w)
    var boot_pl = _fit[PlainObj](
        ctx, d[0], d[1], n_rows, n_cols, 3, True, plain, w
    )
    if _same(boot_wt, boot_pl) != 0:
        fails += 1
        print(
            "  arm C FAILED: with bootstrap ON, the WEIGHTED bin and the"
            " PLAIN bin gave different forests on the same weighted draw."
            " tree_sample_weight() returns nullptr when bootstrapping"
            " (randomforest.cuh:167), so the objective must see no weights"
            " and the bin type must not matter. This is the"
            " double-counting their rule exists to prevent."
        )

    if fails == 0:
        print(
            "  arm C OK: BOTH directions of their tree_sample_weight rule."
            " With bootstrap OFF the weights change the forest, so the"
            " objective reads them. With bootstrap ON the weighted bin and"
            " the plain bin give the SAME forest, so the objective does"
            " not -- no double counting."
        )
    return fails


def arm_d_weighted_bins_train(ctx: DeviceContext) raises -> Int:
    """The weighted bin type trains at all, and is deterministic."""
    var n_rows = 600
    var n_cols = 4
    var d = _data(n_rows, n_cols, 3)
    var w = List[Float32]()
    for r in range(n_rows):
        w.append(Float32(1 + Int(_mix(UInt64(r) + 11) % 4)))
    var obj = WtObj(
        Int32(3), Int32(1), Int32(GINI), Float32(0.0), BinScales(1.0, 1.0)
    )
    var f1 = _fit[WtObj](ctx, d[0], d[1], n_rows, n_cols, 3, False, obj, w)
    var f2 = _fit[WtObj](ctx, d[0], d[1], n_rows, n_cols, 3, False, obj, w)
    var fails = 0
    if len(f1.trees) != 2 or len(f1.trees[0].sparsetree) < 5:
        fails += 1
        print(
            "  arm D: the weighted-bin forest is degenerate --",
            len(f1.trees), "trees,",
            len(f1.trees[0].sparsetree) if len(f1.trees) > 0 else 0,
            "nodes",
        )
        return fails
    if _same(f1, f2) != 0:
        fails += 1
        print(
            "  arm D FAILED: two weighted fits differ. The weight plane is"
            " Int32 fixed point precisely so the histogram stays"
            " order-independent; a float accumulator would fail here."
        )
    if fails == 0:
        print(
            "  arm D OK: WeightedClassificationBin trains --",
            len(f1.trees), "trees of",
            len(f1.trees[0].sparsetree),
            "nodes, bit-identical across two fits",
        )
    return fails


def main() raises:
    print("sample_weight_check: the zero-weight drop, validation, and the")
    print("  bootstrap-vs-objective double-counting rule")
    var ctx = DeviceContext()
    var fails = 0
    fails += arm_a_zero_weight_drop(ctx)
    fails += arm_b_validation(ctx)
    fails += arm_c_double_counting(ctx)
    fails += arm_d_weighted_bins_train(ctx)
    if fails == 0:
        print("sample_weight_check: ALL OK")
    else:
        raise Error("sample_weight_check: " + String(fails) + " failure(s)")
