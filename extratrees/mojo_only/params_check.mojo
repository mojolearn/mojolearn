# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Every branch of `validity_check`, accepted and refused, one case each.

`PORTING_RULES.md` rule 8: a switch is exercised on BOTH sides by a named case
per side, with the value set explicitly inside the case. "The suite covers it"
is not coverage. A validator is nothing but switches, so this file is a table:
one accepted configuration per parameter at its boundary, and one refused
configuration per rejection branch, each asserting that the refusal actually
happened rather than that nothing crashed.

The refusals matter more than the acceptances. An option silently accepted and
silently ignored is indistinguishable, from the caller's side, from an option
that works -- and cuML's struct carries four criteria whose kernels this
directory does not have.
"""

from std.testing import assert_true

from extratrees.ported.decisiontree.decisiontree import (
    CRITERION_END,
    CRITERION_ENTROPY,
    CRITERION_GAMMA,
    CRITERION_GINI,
    CRITERION_INVERSE_GAUSSIAN,
    CRITERION_MAE,
    CRITERION_MSE,
    CRITERION_POISSON,
    DecisionTreeParams,
    validity_check,
)


def accepts(params: DecisionTreeParams) -> Bool:
    try:
        validity_check(params)
        return True
    except:
        return False


def main() raises:
    var accepted = 0
    var refused = 0

    # --- the default must validate ---------------------------------------
    # Their documented default (max_depth = -1) does NOT, by their own
    # assertion `decisiontree.cu:29`. This port takes the branch, not the
    # comment, so its default is a concrete depth and must pass.
    var d = DecisionTreeParams()
    assert_true(accepts(d), "the shipped default must pass its own validator")
    accepted += 1

    # ... and their documented default must still be refused, so that the
    # discrepancy in their header is pinned by a case rather than a comment.
    var neg_depth = DecisionTreeParams()
    neg_depth.max_depth = -1
    assert_true(
        not accepts(neg_depth),
        "max_depth = -1 is cuML's documented default and their own"
        " validity_check rejects it",
    )
    refused += 1

    # --- max_depth boundary ----------------------------------------------
    var zero_depth = DecisionTreeParams()
    zero_depth.max_depth = 0
    assert_true(accepts(zero_depth), "max_depth = 0 is >= 0 and must pass")
    accepted += 1

    # --- max_leaves: -1 or > 0, nothing else ------------------------------
    var leaves_unlimited = DecisionTreeParams()
    leaves_unlimited.max_leaves = -1
    assert_true(accepts(leaves_unlimited), "max_leaves = -1 means unlimited")
    accepted += 1
    var leaves_one = DecisionTreeParams()
    leaves_one.max_leaves = 1
    assert_true(accepts(leaves_one), "max_leaves = 1 must pass")
    accepted += 1
    for bad in [Int32(0), Int32(-2), Int32(-100)]:
        var p = DecisionTreeParams()
        p.max_leaves = bad
        assert_true(not accepts(p), "max_leaves must be -1 or > 0")
        refused += 1

    # --- max_features: the open-closed interval (0, 1] --------------------
    # Both ends, and both sides of both ends. An interval check written with
    # the wrong strictness passes every value except exactly these.
    var f_one = DecisionTreeParams()
    f_one.max_features = 1.0
    assert_true(accepts(f_one), "max_features = 1.0 is INSIDE (0, 1]")
    accepted += 1
    var f_tiny = DecisionTreeParams()
    f_tiny.max_features = 1e-30
    assert_true(accepts(f_tiny), "max_features just above 0 is inside")
    accepted += 1
    for bad in [Float32(0.0), Float32(-0.5), Float32(1.0000001), Float32(2.0)]:
        var p = DecisionTreeParams()
        p.max_features = bad
        assert_true(not accepts(p), "max_features outside (0, 1] must refuse")
        refused += 1

    # --- min_samples_leaf >= 1, min_samples_split >= 2 --------------------
    var msl = DecisionTreeParams()
    msl.min_samples_leaf = 1
    assert_true(accepts(msl), "min_samples_leaf = 1 is their lower bound")
    accepted += 1
    for bad in [Int32(0), Int32(-1)]:
        var p = DecisionTreeParams()
        p.min_samples_leaf = bad
        assert_true(not accepts(p), "min_samples_leaf < 1 must refuse")
        refused += 1

    var mss = DecisionTreeParams()
    mss.min_samples_split = 2
    assert_true(accepts(mss), "min_samples_split = 2 is their lower bound")
    accepted += 1
    for bad in [Int32(1), Int32(0), Int32(-3)]:
        var p = DecisionTreeParams()
        p.min_samples_split = bad
        assert_true(not accepts(p), "min_samples_split < 2 must refuse")
        refused += 1

    # --- the criteria: three accepted, four refused, BY NAME ---------------
    # ENTROPY moved from the refused list to the accepted one with DEVIATION
    # 459 (2026-08-23).
    for ok in [CRITERION_GINI, CRITERION_ENTROPY, CRITERION_MSE, CRITERION_END]:
        var p = DecisionTreeParams()
        p.split_criterion = Int32(ok)
        assert_true(accepts(p), "a ported criterion must pass")
        accepted += 1
    for bad in [
        CRITERION_MAE,
        CRITERION_POISSON,
        CRITERION_GAMMA,
        CRITERION_INVERSE_GAUSSIAN,
    ]:
        var p = DecisionTreeParams()
        p.split_criterion = Int32(bad)
        assert_true(
            not accepts(p),
            "an unported criterion must be REFUSED, never downgraded to MSE",
        )
        refused += 1
    for bad in [Int32(-1), Int32(8), Int32(9999)]:
        var p = DecisionTreeParams()
        p.split_criterion = Int32(bad)
        assert_true(not accepts(p), "an unknown criterion id must refuse")
        refused += 1

    # --- max_batch_size ---------------------------------------------------
    var b1 = DecisionTreeParams()
    b1.max_batch_size = 1
    assert_true(accepts(b1), "a frontier of one node is legal, just narrow")
    accepted += 1
    for bad in [Int32(0), Int32(-1)]:
        var p = DecisionTreeParams()
        p.max_batch_size = bad
        assert_true(not accepts(p), "max_batch_size < 1 must refuse")
        refused += 1

    print("params: ", accepted, "accepted cases,", refused, "refused cases")
    print("params_check: PASS")
