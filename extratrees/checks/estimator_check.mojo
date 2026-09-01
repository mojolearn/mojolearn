# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
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
from extratrees.checks.fixtures import (
    analytic_separable_gap,
    hashed_classification,
)
from extratrees.impl.decisiontree.decisiontree import (
    CRITERION_ENTROPY,
    CRITERION_GINI,
    CRITERION_MSE,
    CRITERION_POISSON,
)
from extratrees.impl.decisiontree.batched_levelalgo.builder import (
    n_sampled_cols_for,
)
from extratrees.impl.randomforest.randomforest import (
    Forest,
    predict_class_forest,
)


def forests_equal(a: Forest, b: Forest) -> Bool:
    if len(a.trees) != len(b.trees):
        return False
    for t in range(len(a.trees)):
        if a.trees[t].num_nodes() != b.trees[t].num_nodes():
            return False
        for i in range(a.trees[t].num_nodes()):
            if not (a.trees[t].sparsetree[i] == b.trees[t].sparsetree[i]):
                return False
        if len(a.trees[t].vector_leaf) != len(b.trees[t].vector_leaf):
            return False
        for i in range(len(a.trees[t].vector_leaf)):
            if (
                a.trees[t].vector_leaf[i].to_bits()
                != b.trees[t].vector_leaf[i].to_bits()
            ):
                return False
    return True


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
    # DEVIATION 460: bootstrap=True is HONOURED -- both sides of the switch.
    var c4 = base.copy()
    c4.bootstrap = True
    assert_true(not refused(c4), "bootstrap=True is accepted (DEVIATION 460)")
    var c4b = base.copy()
    c4b.bootstrap = True
    c4b.max_samples = 100
    assert_true(
        not refused(c4b), "bootstrap=True with max_samples is accepted"
    )
    cells += 2
    var c5 = base.copy()
    c5.oob_score = True
    assert_true(refused(c5), "oob_score")
    n_refused += 1
    var c6 = base.copy()
    c6.max_samples = 100
    assert_true(refused(c6), "max_samples WITHOUT bootstrap (sklearn raises)")
    n_refused += 1
    var c7 = base.copy()
    c7.warm_start = True
    assert_true(refused(c7), "warm_start")
    n_refused += 1
    var c8 = base.copy()
    c8.ccp_alpha = 0.01
    assert_true(refused(c8), "ccp_alpha")
    n_refused += 1
    # `max_leaf_nodes` WAS a refusal cell here until 2026-09-01. It is now
    # HONOURED (DEVIATION 466: best-first growth is a second growth mode in
    # `builder.mojo`), so the cell is the reach test that replaces it --
    # the value must ride into `DecisionTreeParams`, which is what selects
    # the mode, and the default must stay sklearn's `None`.
    var c9 = base.copy()
    c9.max_leaf_nodes = 8
    assert_true(
        not refused(c9), "max_leaf_nodes=8 is accepted, not refused"
    )
    var plan_mln = resolve(c9, 1000, 8)
    assert_equal(
        Int(plan_mln.params.max_leaf_nodes), 8,
        "REACH: max_leaf_nodes must ride into DecisionTreeParams; that"
        " field is the growth-mode selector and a value that stopped at"
        " the config would be a knob that did nothing",
    )
    assert_equal(
        Int(resolve(base, 1000, 8).params.max_leaf_nodes), -1,
        "the default stays sklearn's None, which is depth-wise growth",
    )
    # SABOTAGE ARM for that reach, and it is the bound rather than the
    # plumbing: sklearn constrains max_leaf_nodes to >= 2
    # (`_classes.py`'s Interval(Integral, 2, None, closed="left")), so 1
    # must be refused. Like `max_leaves=0` below, the guard is
    # `validity_check` inside `resolve`, not `refuse_unported`, so this
    # arm cannot use `refused()`.
    var c9a = base.copy()
    c9a.max_leaf_nodes = 1
    var one_leaf_refused = False
    try:
        _ = resolve(c9a, 1000, 8)
    except:
        one_leaf_refused = True
    assert_true(
        one_leaf_refused,
        "max_leaf_nodes=1 must be refused: a one-leaf tree is a tree that"
        " was never split and the best-first builder cannot express it",
    )
    cells += 4
    # cuML's OWN budget, under cuML's own name, is HONOURED and REACHES the
    # params (2026-09-01). `resolve` pinned `params.max_leaves = -1` until
    # then, so `builder.mojo:292-296` and `:341-345` were transcribed,
    # checked by `builder_check`, and unreachable from any fit.
    var c9b = base.copy()
    c9b.max_leaves = 8
    assert_true(not refused(c9b), "max_leaves=8 is accepted, not refused")
    var plan_ml = resolve(c9b, 1000, 8)
    assert_equal(
        Int(plan_ml.params.max_leaves), 8,
        "REACH: max_leaves must ride into DecisionTreeParams, not be pinned"
        " at -1 the way it was until 2026-09-01",
    )
    var plan_ml_default = resolve(base, 1000, 8)
    assert_equal(
        Int(plan_ml_default.params.max_leaves), -1,
        "the default stays cuML's unlimited",
    )
    # SABOTAGE ARM for that reach: if `resolve` goes back to pinning -1, the
    # assert above goes red. The complement is that a value cuML itself
    # rejects must still be rejected HERE, and not smuggled past because the
    # field is new. max_leaves=0 is neither -1 nor positive.
    #
    # NOTE WHICH GUARD CATCHES IT: not `refuse_unported`, which is why this
    # arm cannot use `refused()` like every other one above. `max_leaves` is
    # cuML's parameter, so it is cuML's `validity_check` that rejects it
    # (`decisiontree.mojo:138-139`, their `decisiontree.cu:30-32`), and that
    # runs inside `resolve` AFTER the sklearn refusals. Restating the bound
    # in `refuse_unported` would be a second copy of a constant, which is how
    # a copy drifts, so the test goes through `resolve` instead.
    var c9c = base.copy()
    c9c.max_leaves = 0
    var zero_leaves_refused = False
    try:
        _ = resolve(c9c, 1000, 8)
    except:
        zero_leaves_refused = True
    assert_true(
        zero_leaves_refused,
        "max_leaves=0 must be refused by cuML's validity_check, not accepted"
        " as a silent unlimited",
    )
    # ... and the two fields must stay INDEPENDENT. They are different
    # guarantees -- one selects best-first growth, the other caps a
    # breadth-first frontier -- so setting either must not move the other,
    # and both may be set at once with the tighter one binding.
    var c9d = base.copy()
    c9d.max_leaves = 8
    c9d.max_leaf_nodes = 6
    var plan_both = resolve(c9d, 1000, 8)
    assert_equal(
        Int(plan_both.params.max_leaves), 8, "max_leaves rides untouched"
    )
    assert_equal(
        Int(plan_both.params.max_leaf_nodes), 6,
        "max_leaf_nodes rides untouched; neither field is derived from the"
        " other and neither is an alias for the other",
    )
    assert_equal(
        Int(plan_ml.params.max_leaf_nodes), -1,
        "setting cuML's max_leaves alone must NOT switch on best-first"
        " growth: that would be the silent aliasing this lane refused for",
    )
    cells += 6
    var c10 = base.copy()
    c10.n_estimators = 0
    assert_true(refused(c10), "n_estimators=0")
    n_refused += 1
    cells += n_refused
    print("    ", n_refused, "parameters refused by name; the default accepted")

    # DEVIATION 459: entropy is ACCEPTED and rides to the tree params as
    # itself; a criterion the tree layer refuses (POISSON here, the sabotage
    # arm of the same switch) must still not slip through.
    var ce = base.copy()
    ce.criterion = CRITERION_ENTROPY
    var plan_e = resolve(ce, 100, 8)
    assert_equal(
        plan_e.params.split_criterion,
        Int32(CRITERION_ENTROPY),
        "entropy must reach DecisionTreeParams as ENTROPY, not as gini",
    )
    var cp = base.copy()
    cp.criterion = CRITERION_POISSON
    var poisson_refused = False
    try:
        _ = resolve(cp, 100, 8)
    except:
        poisson_refused = True
    assert_true(
        poisson_refused, "poisson must be refused, not downgraded to gini"
    )
    # and the bootstrap plan reports the count it will use
    var plan_b = resolve(c4b, 1000, 8)
    assert_true(plan_b.bootstrap, "the plan carries bootstrap")
    assert_equal(Int(plan_b.n_sampled_rows), 100, "max_samples=100 -> 100")
    var plan_b0 = resolve(c4, 1000, 8)
    assert_equal(Int(plan_b0.n_sampled_rows), 1000, "max_samples=None -> n")
    assert_equal(
        Int(resolve(base, 1000, 8).n_sampled_rows), 0,
        "no bootstrap -> 0 reported",
    )
    cells += 6

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

    # ---- DEVIATION 459 / 460 REACH on the host arm -----------------------
    # entropy vs gini must build DIFFERENT forests on a fixture where the
    # criteria are distinguishable (a 4-class hashed target, where the two
    # impurities rank candidates differently -- a separable binary target
    # cannot tell them apart, measured); bootstrap vs not must differ;
    # bootstrap at one seed twice must be identical.
    print("[reach] entropy and bootstrap move the host forest")
    var h4 = hashed_classification(0xE459, 2048, 12, 4)
    var x4 = List[Float32](length=h4.n_rows * h4.n_cols, fill=Float32(0.0))
    for r in range(h4.n_rows):
        for c in range(h4.n_cols):
            x4[c * h4.n_rows + r] = h4.value(r, c)
    var y4 = List[Float32]()
    for r in range(h4.n_rows):
        y4.append(Float32(Int(h4.label[r])))
    var cg = ExtraTreesConfig()
    cg.n_estimators = 4
    cg.max_depth = 6
    cg.random_state = 7
    var ce4 = cg.copy()
    ce4.criterion = CRITERION_ENTROPY
    var cb4 = cg.copy()
    cb4.bootstrap = True
    var fg = fit_extra_trees_classifier(
        x4, y4, Int32(h4.n_rows), Int32(h4.n_cols), 4, cg
    )
    var fe = fit_extra_trees_classifier(
        x4, y4, Int32(h4.n_rows), Int32(h4.n_cols), 4, ce4
    )
    var fb = fit_extra_trees_classifier(
        x4, y4, Int32(h4.n_rows), Int32(h4.n_cols), 4, cb4
    )
    var fb2 = fit_extra_trees_classifier(
        x4, y4, Int32(h4.n_rows), Int32(h4.n_cols), 4, cb4
    )
    var fg2 = fit_extra_trees_classifier(
        x4, y4, Int32(h4.n_rows), Int32(h4.n_cols), 4, cg
    )
    assert_true(
        forests_equal(fg.forest, fg2.forest), "gini twice is one forest"
    )
    assert_true(
        not forests_equal(fg.forest, fe.forest),
        "REACH: criterion=entropy must build a different forest than gini",
    )
    assert_true(
        not forests_equal(fg.forest, fb.forest),
        "REACH: bootstrap=True must build a different forest than False",
    )
    assert_true(
        forests_equal(fb.forest, fb2.forest),
        "bootstrap=True at random_state 7 twice is one forest",
    )
    assert_equal(Int(fb.plan.n_sampled_rows), h4.n_rows)
    cells += 5
    print("     gini != entropy, bootstrap != none, bootstrap repeats")

    _ = gx.unsafe_ptr()
    _ = glab.unsafe_ptr()
    _ = hx.unsafe_ptr()
    _ = hy.unsafe_ptr()
    _ = x4.unsafe_ptr()
    _ = y4.unsafe_ptr()

    print("estimator: ", cells, "cells")
    print("estimator_check: PASS")


def n_sampled_cols_for_ratio(ratio: Float32, n_cols: Int32) -> Int32:
    """`n_sampled_cols_for` with the ratio supplied directly, so the round-trip
    test does not have to build a whole `DecisionTreeParams` per case. Same
    arithmetic: `builder.cuh:222`'s `max(1, IdxT(max_features * n_cols))`."""
    var k = Int32(ratio * Float32(n_cols))
    return 1 if k < 1 else k
