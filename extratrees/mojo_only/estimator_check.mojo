"""Every sklearn parameter: honoured or refused, one named case per side.

`estimator.mojo` exists because DEVIATION 154 recorded a debt — a caller
arriving from sklearn with `min_weight_fraction_leaf` or `monotonic_cst` got no
error from this port, because neither had a field to refuse. This file is what
holds that promise: it enumerates every parameter in sklearn's constructor and
requires each to be either honoured with a checked effect, or refused with a
message that names it.

`max_features` gets the most attention, because it is the one whose resolution
round-trips through an integer, a ratio and a truncation
(`builder.cuh:222`), and a value that lands one short is invisible in a fitted
model.
"""

from std.testing import assert_equal, assert_true

from extratrees.estimator import (
    DEPTH_SLACK,
    ExtraTreesConfig,
    MAX_FEATURES_ALL,
    MAX_FEATURES_LOG2,
    MAX_FEATURES_SQRT,
    count_to_ratio,
    fit_extra_trees_classifier,
    fit_extra_trees_regressor,
    refuse_unported,
    resolve,
    resolve_max_features,
)
from extratrees.mojo_only.fixtures import (
    analytic_separable_gap,
    hashed_classification,
)
from extratrees.ported.decisiontree.decisiontree import (
    CRITERION_ENTROPY,
    CRITERION_GINI,
    CRITERION_MSE,
)
from extratrees.ported.decisiontree.batched_levelalgo.builder import (
    n_sampled_cols_for,
)
from extratrees.ported.randomforest.randomforest import predict_class_forest


def refused(config: ExtraTreesConfig) -> Bool:
    try:
        refuse_unported(config)
        return False
    except:
        return True


def main() raises:
    var cells = 0

    # ---- max_features: sklearn's rule, in integers, and the round trip ----
    # sklearn: 'sqrt' -> max(1, int(sqrt(n))), 'log2' -> max(1, int(log2(n))),
    # None -> n, float f -> max(1, int(f*n)), int k -> k. `int()` TRUNCATES.
    print("[max_features] sklearn's resolution, and the ratio round trip")
    var sqrt_cases = [
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 2),
        (8, 2),
        (9, 3),
        (15, 3),
        (16, 4),
        (48, 6),
        (100, 10),
        (101, 10),
    ]
    for c in sqrt_cases:
        assert_equal(
            resolve_max_features(MAX_FEATURES_SQRT, 0.0, c[0]),
            c[1],
            "sqrt of " + String(c[0]),
        )
        cells += 1
    var log2_cases = [(1, 1), (2, 1), (3, 1), (4, 2), (7, 2), (8, 3), (1024, 10)]
    for c in log2_cases:
        assert_equal(
            resolve_max_features(MAX_FEATURES_LOG2, 0.0, c[0]),
            c[1],
            "log2 of " + String(c[0]),
        )
        cells += 1
    for n in [1, 2, 7, 48, 1000]:
        assert_equal(
            resolve_max_features(MAX_FEATURES_ALL, 0.0, n), n, "None means all"
        )
        cells += 1
    # the fraction form, including the floor of one
    assert_equal(resolve_max_features(0, 0.5, 8), 4)
    assert_equal(resolve_max_features(0, 0.3, 10), 3)
    assert_equal(resolve_max_features(0, 0.05, 8), 1, "the floor of one")
    assert_equal(resolve_max_features(0, 1.0, 8), 8)
    cells += 4
    # explicit counts, and the two refusals
    assert_equal(resolve_max_features(3, 0.0, 8), 3)
    var bad = 0
    try:
        _ = resolve_max_features(9, 0.0, 8)
    except:
        bad += 1
    try:
        _ = resolve_max_features(0, 0.0, 8)
    except:
        bad += 1
    try:
        _ = resolve_max_features(0, 1.5, 8)
    except:
        bad += 1
    assert_equal(bad, 3, "count > n, fraction 0 and fraction > 1 must refuse")
    cells += 4

    # THE ROUND TRIP, which is the part that can land one short: the ratio
    # must truncate back to exactly the count sklearn resolved.
    var round_trips = 0
    for n in range(1, 200):
        for spec in [MAX_FEATURES_SQRT, MAX_FEATURES_LOG2, MAX_FEATURES_ALL]:
            var count = resolve_max_features(spec, 0.0, n)
            var params_ratio = count_to_ratio(count, n)
            var back = Int(n_sampled_cols_for_ratio(params_ratio, Int32(n)))
            assert_equal(
                back,
                count,
                "max_features round trip lost a column at n="
                + String(n)
                + " count="
                + String(count),
            )
            round_trips += 1
            cells += 1
    print("    ", round_trips, "resolutions round-tripped through the ratio")

    # ---- every refusal, by name ------------------------------------------
    print("[refusals] every unported sklearn parameter, one case each")
    var base = ExtraTreesConfig()
    assert_true(not refused(base), "the shipped default must be accepted")
    cells += 1

    var n_refused = 0
    var c1 = base.copy()
    c1.min_weight_fraction_leaf = 0.1
    assert_true(refused(c1), "min_weight_fraction_leaf")
    n_refused += 1
    var c2 = base.copy()
    c2.has_monotonic_cst = True
    assert_true(refused(c2), "monotonic_cst -- the DEVIATION 154 debt")
    n_refused += 1
    var c3 = base.copy()
    c3.has_class_weight = True
    assert_true(refused(c3), "class_weight")
    n_refused += 1
    var c4 = base.copy()
    c4.bootstrap = True
    assert_true(refused(c4), "bootstrap=True")
    n_refused += 1
    var c5 = base.copy()
    c5.oob_score = True
    assert_true(refused(c5), "oob_score")
    n_refused += 1
    var c6 = base.copy()
    c6.max_samples_set = True
    assert_true(refused(c6), "max_samples")
    n_refused += 1
    var c7 = base.copy()
    c7.warm_start = True
    assert_true(refused(c7), "warm_start")
    n_refused += 1
    var c8 = base.copy()
    c8.ccp_alpha = 0.01
    assert_true(refused(c8), "ccp_alpha")
    n_refused += 1
    var c9 = base.copy()
    c9.max_leaf_nodes = 8
    assert_true(
        refused(c9),
        "max_leaf_nodes -- sklearn's is a BEST-FIRST growth limit and cuML's"
        " max_leaves is a breadth-first soft cap; accepting the name would"
        " accept a different algorithm",
    )
    n_refused += 1
    var c10 = base.copy()
    c10.n_estimators = 0
    assert_true(refused(c10), "n_estimators=0")
    n_refused += 1
    cells += n_refused
    print("    ", n_refused, "parameters refused by name; the default accepted")

    # A criterion the tree layer refuses must also not slip through here.
    var ce = base.copy()
    ce.criterion = CRITERION_ENTROPY
    var entropy_refused = False
    try:
        _ = resolve(ce, 100, 8)
    except:
        entropy_refused = True
    assert_true(
        entropy_refused, "entropy must be refused, not downgraded to gini"
    )
    cells += 1

    # ---- max_depth=None: mapped, and the substitution REPORTED -----------
    print("[max_depth] sklearn's None, mapped and reported")
    var plan_unlimited = resolve(base, 1024, 8)
    assert_true(
        plan_unlimited.depth_was_unlimited,
        "max_depth=None must be reported as a substitution",
    )
    assert_equal(
        plan_unlimited.params.max_depth,
        Int32(10) + DEPTH_SLACK,
        "ceil_log2(1024) is 10",
    )
    var explicit = base.copy()
    explicit.max_depth = 5
    var plan_explicit = resolve(explicit, 1024, 8)
    assert_true(
        not plan_explicit.depth_was_unlimited,
        "an explicit depth is not a substitution",
    )
    assert_equal(plan_explicit.params.max_depth, 5)
    cells += 4

    # ---- and the fit reports whether the cap actually BOUND ---------------
    # Both sides: a shallow explicit cap must bind, an unlimited request on an
    # easy fixture must not. A flag that is always false is not a report.
    var gap = analytic_separable_gap(0x5EED)
    var gx = List[Float32](
        length=gap.data.n_rows * gap.data.n_cols, fill=Float32(0.0)
    )
    for r in range(gap.data.n_rows):
        for c in range(gap.data.n_cols):
            gx[c * gap.data.n_rows + r] = gap.data.value(r, c)
    var glab = List[Float32]()
    for r in range(gap.data.n_rows):
        glab.append(Float32(Int(gap.data.label[r])))

    var small = base.copy()
    small.n_estimators = 8
    small.max_depth = 1
    var bound_fit = fit_extra_trees_classifier(
        gx,
        glab,
        Int32(gap.data.n_rows),
        Int32(gap.data.n_cols),
        Int32(gap.data.n_classes),
        small,
    )
    assert_true(
        bound_fit.depth_cap_bound,
        "max_depth=1 on a fixture that wants deeper MUST report the cap bound",
    )
    var loose = base.copy()
    loose.n_estimators = 8
    var free_fit = fit_extra_trees_classifier(
        gx,
        glab,
        Int32(gap.data.n_rows),
        Int32(gap.data.n_cols),
        Int32(gap.data.n_classes),
        loose,
    )
    assert_true(
        not free_fit.depth_cap_bound,
        "an unlimited request on a separable fixture must NOT hit the"
        " substituted cap -- if it does, DEPTH_SLACK is too small and the"
        " substitution is changing models silently",
    )
    cells += 2
    print(
        "    max_depth=1 bound the cap; max_depth=None did not (depth cap",
        free_fit.plan.params.max_depth,
        ")",
    )

    # The unlimited fit must also be CORRECT, not merely uncapped.
    var wrong = 0
    for r in range(gap.data.n_rows):
        var row = List[Float32]()
        for c in range(gap.data.n_cols):
            row.append(gap.data.value(r, c))
        if predict_class_forest(free_fit.forest, row, 0) != Int(
            gap.data.label[r]
        ):
            wrong += 1
    assert_equal(wrong, 0, "the estimator surface must fit as well as the raw forest")
    cells += 1

    # ---- the regression defaults differ, and that is sklearn's -----------
    var reg = ExtraTreesConfig().for_regression()
    assert_equal(reg.criterion, CRITERION_MSE, "squared_error")
    assert_equal(
        reg.max_features_spec,
        MAX_FEATURES_ALL,
        "sklearn's ExtraTreesRegressor defaults max_features to 1.0, NOT sqrt",
    )
    assert_equal(
        ExtraTreesConfig().max_features_spec,
        MAX_FEATURES_SQRT,
        "and the classifier defaults to sqrt",
    )
    assert_equal(ExtraTreesConfig().criterion, CRITERION_GINI)
    assert_true(not reg.bootstrap, "both default bootstrap=False")
    cells += 5

    var hashed = hashed_classification(0xE571, 256, 5, 2)
    var hx = List[Float32](
        length=hashed.n_rows * hashed.n_cols, fill=Float32(0.0)
    )
    for r in range(hashed.n_rows):
        for c in range(hashed.n_cols):
            hx[c * hashed.n_rows + r] = hashed.value(r, c)
    var hy = List[Float32]()
    for r in range(hashed.n_rows):
        hy.append(hashed.y[r])
    var rcfg = ExtraTreesConfig().for_regression()
    rcfg.n_estimators = 6
    var rfit = fit_extra_trees_regressor(
        hx, hy, Int32(hashed.n_rows), Int32(hashed.n_cols), rcfg
    )
    assert_equal(
        rfit.plan.max_features_count,
        hashed.n_cols,
        "the regressor must sample every column by default",
    )
    assert_equal(len(rfit.forest.trees), 6)
    cells += 2

    _ = gx.unsafe_ptr()
    _ = glab.unsafe_ptr()
    _ = hx.unsafe_ptr()
    _ = hy.unsafe_ptr()

    print("estimator: ", cells, "cells")
    print("estimator_check: PASS")


def n_sampled_cols_for_ratio(ratio: Float32, n_cols: Int32) -> Int32:
    """`n_sampled_cols_for` with the ratio supplied directly, so the round-trip
    test does not have to build a whole `DecisionTreeParams` per case. Same
    arithmetic: `builder.cuh:222`'s `max(1, IdxT(max_features * n_cols))`."""
    var k = Int32(ratio * Float32(n_cols))
    return 1 if k < 1 else k
