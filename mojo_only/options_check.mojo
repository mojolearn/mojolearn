# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Do the options refuse what is not honored?

The point of the file under test is that an option present and ignored is
worse than one absent. This checks the refusals actually fire, because a
`check()` nobody exercises is exactly the kind of machinery this repository
keeps finding unwired.
"""

from gbdt.options.catboost_options import (
    LEAF_ESTIMATION_EXACT,
    LEAF_ESTIMATION_GRADIENT,
    LEAF_ESTIMATION_NEWTON,
    LEAF_ESTIMATION_SIMPLE,
    CatBoostOptions,
    set_leaves_estimation_default,
    DETERMINISM_CROSS_DEVICE,
    DETERMINISM_DEVICE,
    DETERMINISM_OFF,
    GROW_DEPTHWISE,
    GROW_LOSSGUIDE,
    GROW_SYMMETRIC,
    SCORE_FUNCTION_NEWTON_L2,
    determinism_name,
    grow_policy_name,
)
from gbdt.options.loss_description import make_loss_description


def _newton_override(
    name: String,
    q: Float32 = Float32(-1.0),
    delta: Float32 = Float32(-1.0),
) raises -> Bool:
    """True when `loss_estimation_method=Newton` is REFUSED for `name`."""
    var loss = make_loss_description(name, q=q, delta=delta)
    try:
        _ = set_leaves_estimation_default(
            loss, method_override=LEAF_ESTIMATION_NEWTON
        )
    except e:
        var msg = String(e)
        if not (
            "Newton leaves estimation method is not supported" in msg
        ):
            raise Error(
                "Newton on " + name + " refused with the WRONG message: "
                + msg
            )
        return True
    return False


def check_newton_availability() raises:
    """DEVIATION 257: `EnsureNewtonIsAvailable` (`catboost_options.cpp:
    588-601`, via `Validate` `:740-742`) on the RESOLVED method.

    Before 2026-08-23 every arm below was accepted, and the E2 matrix's
    `Quantile+Newton` / `MAE+Newton` cells ran a zero-Hessian Newton
    step that CatBoost itself refuses to configure. Both directions are
    asserted: the five refusals AND the acceptances, so the guard can
    neither vanish nor over-reach.
    """
    var refused = List[String]()
    refused.append(String("Quantile"))
    refused.append(String("MAE"))
    refused.append(String("LogLinQuantile"))
    refused.append(String("MAPE"))
    for i in range(len(refused)):
        if not _newton_override(refused[i]):
            raise Error(
                "leaf_estimation_method=Newton on " + refused[i]
                + " should have been refused (catboost_options.cpp:588)"
            )
        print("    refused: leaf_estimation_method=Newton on", refused[i])
    if not _newton_override(String("Lq"), q=Float32(1.5)):
        raise Error("Newton on Lq q=1.5 should have been refused (:599)")
    print("    refused: leaf_estimation_method=Newton on Lq q=1.5")

    # the other direction: Newton IS their default or legal here
    if _newton_override(String("Lq"), q=Float32(2.5)):
        raise Error("Newton on Lq q=2.5 is legal (:599) and was refused")
    if _newton_override(String("Huber"), delta=Float32(1.0)):
        raise Error("Newton on Huber is THEIR default (:187-192), refused")
    if _newton_override(String("RMSE")):
        raise Error("Newton on RMSE is their default (:59-64), refused")
    # the default path for the refused losses must still resolve (Exact /
    # Gradient), i.e. the guard is on the OVERRIDE, not the loss
    var q_default = set_leaves_estimation_default(
        make_loss_description(String("Quantile"))
    )
    if q_default.method == LEAF_ESTIMATION_NEWTON:
        raise Error("Quantile's default resolved to Newton")
    print(
        "    accepted: Newton on Lq q=2.5, Huber, RMSE; Quantile default"
        " resolves without Newton"
    )


def expect_refusal(mut o: CatBoostOptions, what: String) raises:
    var refused = False
    try:
        o.check()
    except:
        refused = True
    if not refused:
        raise Error(what + " should have been refused and was not")
    print("    refused:", what)


def check_options() raises:
    var d = CatBoostOptions.default()
    d.check()
    print("  defaults pass:")
    print(
        "    depth",
        d.depth,
        " grow_policy",
        grow_policy_name(d.grow_policy),
        " l2_leaf_reg",
        d.l2_leaf_reg,
        " border_count",
        d.border_count,
    )
    print("    determinism", determinism_name(d.determinism))

    # `grow_policy` WAS A REFUSAL until 2026-08-23 (DEVIATION 259) and is a
    # feature: Depthwise and Lossguide are grown from `train()`. This check
    # asserted the refusal and went red when it was deleted, which is what
    # a refusal check is for. What CatBoost refuses BESIDE the policy is
    # what stays refused, by their rule: `max_leaves` off `1 << depth` on
    # any policy but Lossguide (`catboost_options.cpp:993-1001`), a
    # Lossguide budget past 65536 (`oblivious_tree_options.cpp:130-133`),
    # and `min_data_in_leaf` under SymmetricTree (ours, stricter: theirs
    # discards it, `greedy_search_helper.cpp:685`).
    var a = CatBoostOptions.default()
    a.grow_policy = GROW_LOSSGUIDE
    a.max_leaves = 31
    a.check()
    var a2 = CatBoostOptions.default()
    a2.grow_policy = GROW_LOSSGUIDE
    a2.max_leaves = 1 << 17
    expect_refusal(a2, String("grow_policy=Lossguide max_leaves=131072"))

    var b = CatBoostOptions.default()
    b.grow_policy = GROW_DEPTHWISE
    b.check()
    var b2 = CatBoostOptions.default()
    b2.grow_policy = GROW_DEPTHWISE
    b2.max_leaves = 31
    expect_refusal(b2, String("grow_policy=Depthwise max_leaves=31"))
    var b3 = CatBoostOptions.default()
    b3.grow_policy = GROW_DEPTHWISE
    b3.min_data_in_leaf = 20
    b3.check()
    print(
        "    accepted: grow_policy Depthwise / Lossguide; Lossguide"
        " max_leaves=31; Depthwise min_data_in_leaf=20"
    )

    # `leaf_estimation_iterations` and `leaf_estimation_method` were
    # REFUSALS until 2026-08-21 and are now features: the descent walker,
    # the Gradient arm and the Exact weighted-quantile estimator all
    # landed. This check asserted the refusal and correctly went red when
    # it was deleted, which is what a refusal check is for -- it fails
    # both when a refusal wrongly disappears AND when a capability
    # wrongly arrives.
    var c = CatBoostOptions.default()
    c.leaf_estimation_iterations = 10
    c.check()  # must NOT raise: ten Newton steps is their Logloss default
    var c2 = CatBoostOptions.default()
    c2.leaf_estimation_method = LEAF_ESTIMATION_GRADIENT
    c2.check()
    var c3 = CatBoostOptions.default()
    c3.leaf_estimation_method = LEAF_ESTIMATION_EXACT
    c3.check()
    print("    accepted: leaf_estimation Newton x10, Gradient, Exact")

    # `Simple` IS still refused, and for a reason that is not "unported":
    # it turns the estimator OFF (`greedy_subsets_searcher.h:67-69`) and
    # that branch has no check of its own.
    var c4 = CatBoostOptions.default()
    c4.leaf_estimation_method = LEAF_ESTIMATION_SIMPLE
    expect_refusal(c4, String("leaf_estimation_method=Simple"))

    var c5 = CatBoostOptions.default()
    c5.leaf_estimation_iterations = -5
    expect_refusal(c5, String("leaf_estimation_iterations=-5"))

    # `random_strength` IS PORTED NOW, so 1.0 at the default score
    # function must be ACCEPTED. What stays refused is the pairing where
    # no calcer of CatBoost's carries a noise term at all.
    var e_ok = CatBoostOptions.default()
    e_ok.random_strength = 1.0
    e_ok.check()
    print("    accepted: random_strength=1.0 at score_function=Cosine")

    var e = CatBoostOptions.default()
    e.random_strength = 1.0
    e.score_function = SCORE_FUNCTION_NEWTON_L2
    expect_refusal(e, String("random_strength=1.0 under NewtonL2"))

    var e_neg = CatBoostOptions.default()
    e_neg.random_strength = -1.0
    expect_refusal(e_neg, String("random_strength=-1.0"))

    var f = CatBoostOptions.default()
    f.rsm = 0.8
    expect_refusal(f, String("rsm=0.8"))

    var g = CatBoostOptions.default()
    g.min_data_in_leaf = 20
    expect_refusal(g, String("min_data_in_leaf=20"))

    var h = CatBoostOptions.default()
    h.depth = 0
    expect_refusal(h, String("depth=0"))

    # The three determinism levels must all be accepted.
    var levels = List[Int]()
    levels.append(DETERMINISM_OFF)
    levels.append(DETERMINISM_DEVICE)
    levels.append(DETERMINISM_CROSS_DEVICE)
    for i in range(len(levels)):
        var k = CatBoostOptions.default()
        k.determinism = levels[i]
        k.check()
    print("    all three determinism levels accepted")
    print("  every unported option refuses by name")
    check_newton_availability()
    print("  Newton availability matches their Validate (DEVIATION 257)")


def main() raises:
    # STANDALONE DRIVER, the same call `probe_main.mojo` makes first.
    print("options:")
    check_options()
