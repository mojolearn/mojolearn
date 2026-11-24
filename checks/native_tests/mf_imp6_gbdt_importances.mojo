# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""`feature_importances_` of a gradient-boosting model, THE HOST COLUMN (lane mf-imp6, 2026-11-23), in a process
with no Python: core/gbdt_importance.mojo's plan checks, bins, exact leaf weights, per-tree decreases, per-column
folds and normalisation over hand-built models whose answers are worked out below (exact oracles). The device
column runs the same `@always_inline` bodies (gbdt/gpu_importance.mojo); its bits are compared with this column
on the GPU box (tools/identity_lanes, lane gbdt_importances).

The four rows x two input columns used throughout (column-major): x0 = [0, 0, 1, 1], x1 = [0, 1, 0, 1]; every
model column has one border 0.5, so a value's bin is 1 when it is above 0.5.

Cases (one row each, run across the host cores by `check_cases`):
  0 the weight words: exponent of 1.0 is 127, of a subnormal 1, of 0 / -1 / NaN / inf 0; the word of 1.0 under
    e_max 127 is 2^23, of 0.5 is 2^22, of 3.0 under e_max 128 is 3 * 2^22, a weight 2^25 times smaller is 0.
    Fail-proof: drop the implicit bit (`mant | 0x800000`) in `gbdt_imp_weight_word` (1.0 gives 0).
  1 two oblivious depth-1 trees, unweighted: tree 0 splits column 0 (leaves 1, 3), tree 1 splits column 1 (leaves
    0, 1). Decreases (2/4)(2/4)(1-3)^2 = 1 and (2/4)(2/4)(0-1)^2 = 0.25, so [0.8, 0.2] exactly as float32
    (correctly rounded 1/1.25 and 0.25/1.25). Fail-proof: build with -D MOJOLEARN_GBDT_IMPORTANCE_SABOTAGE (every
    split credited to the next column: [0.2, 0.8]), or compute `b` in `gbdt_imp_node_gain` as W_R / W_root.
  2 one oblivious depth-2 tree (level 0 column 0, level 1 column 1), leaves [0, 2, 0, 2]: only column 0 separates
    its leaves, so [1, 0]. Fail-proof: pair the leaves of level j as `p + m` instead of `p + m * 2^j` (column 1
    gets a nonzero share).
  3 a non-symmetric tree: root on column 0 (left 1 leaf, right 2), its right child on column 1; leaves [0, 1, 3].
    Root decrease (2/4)(2/4)(0-2)^2 = 1, child (1/4)(1/2)(1-3)^2 = 0.5, so [2/3, 1/3] (within 1e-7). Fail-proof:
    skip `b0 += lj` in the descent of `gbdt_imp_tree_body` (the child reads leaves 0 and 1: 0 and 1 apart).
  4 weights: case 1 under [1, 1, 1, 3] gives 8/9 and 2/9, [0.8, 0.2] within 1e-6; leaves [1, 3] / [0, 4] give
    [0.2, 0.8] unweighted and [0.25, 0.75] under [3, 1, 3, 1] (tree 1's factor 1/4 becomes 3/16); a constant
    weight 2.0 gives the unweighted bits exactly (the integer leaf weights scale by 2^23). Fail-proof: read
    `q = 1` for weighted rows in `gbdt_imp_leaf_weights_host` (the weighted answer stays [0.2, 0.8]).
  5 a combination column: three model columns over two inputs, column 2 credited half to each input; one tree on
    column 2 gives [0.5, 0.5]; equal leaves give zeros. Fail-proof: credit `gain` without `map_share` (the sum
    stays 1 after normalising, so the test reads the unnormalised fold: total [0.5, 0.5] for a gain of 1).
  6 the plan's refusals: a split column past the model columns, a tree whose leaf values are not 2^depth * dim,
    a map entry past the input columns. Fail-proof: drop the split-column test in `gbdt_importance_finish_plan`.
"""
from std.testing import TestSuite
from std.memory import bitcast
from checks.native_tests.testsuite_cases import check_cases, require
from core.gbdt_importance import (
    GbdtImportancePlan, GbdtImpF32, gbdt_importance_finish_plan, gbdt_imp_bins_host, gbdt_imp_leaf_weights_host,
    gbdt_imp_from_leaf_weights_host, gbdt_imp_exponent, gbdt_imp_weight_word, gbdt_imp_fold_body,
    GBDT_IMP_SABOTAGE,
)

comptime CASES = 7


def _plan(n_cols: Int, n_input: Int, non_symmetric: Bool, tree_lo: List[Int32], split_feature: List[Int32],
          node_left: List[Int32], node_right: List[Int32], leaves: List[Float32], leaf_lo: List[Int32],
          map_lo: List[Int32], map_col: List[Int32], map_share: List[Float32]) raises -> GbdtImportancePlan:
    """A plan of `n_cols` model columns with one border 0.5 each (TakeGreater splits at bin 0, dim 1)."""
    var plan = GbdtImportancePlan()
    plan.n_cols = n_cols
    plan.n_input = n_input
    plan.n_trees = len(tree_lo) - 1
    plan.dim = 1
    plan.non_symmetric = non_symmetric
    for c in range(n_cols + 1):  # small-loop(the model columns)
        plan.border_lo.append(Int32(c))
    for _ in range(n_cols):  # small-loop(the model columns)
        plan.borders.append(Float32(0.5))
        plan.treat.append(Int32(0))
    plan.tree_lo = tree_lo.copy()
    plan.leaf_lo = leaf_lo.copy()
    plan.split_feature = split_feature.copy()
    plan.split_bin = List[Int32](length=len(split_feature), fill=Int32(0))
    plan.split_take = List[Int32](length=len(split_feature), fill=Int32(0))
    plan.node_left = node_left.copy()
    plan.node_right = node_right.copy()
    plan.leaves = leaves.copy()
    plan.map_lo = map_lo.copy()
    plan.map_col = map_col.copy()
    plan.map_share = map_share.copy()
    gbdt_importance_finish_plan(plan)
    return plan^


def _identity_map(n: Int, mut map_lo: List[Int32], mut map_col: List[Int32], mut map_share: List[Float32]):
    map_lo.append(Int32(0))
    for c in range(n):  # small-loop(the model columns)
        map_col.append(Int32(c))
        map_share.append(Float32(1.0))
        map_lo.append(Int32(c + 1))


def _x() -> List[Float32]:
    var x: List[Float32] = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    return x^


def _run(plan: GbdtImportancePlan, x: List[Float32], n_rows: Int, w: List[Float32], n_weights: Int) raises -> List[Float32]:
    var bins = gbdt_imp_bins_host(plan, Int(x.unsafe_ptr()), n_rows, False)
    var pairs = gbdt_imp_leaf_weights_host(plan, bins, n_rows, Int(w.unsafe_ptr()), n_weights)
    var result = gbdt_imp_from_leaf_weights_host(plan, pairs)
    _ = len(x)  # read through its address above
    _ = len(w)
    return result^


def _two_stumps(leaves: List[Float32]) raises -> GbdtImportancePlan:
    var map_lo = List[Int32]()
    var map_col = List[Int32]()
    var map_share = List[Float32]()
    _identity_map(2, map_lo, map_col, map_share)
    var tree_lo: List[Int32] = [0, 1, 2]
    var split: List[Int32] = [0, 1]
    var zeros: List[Int32] = [0, 0]
    var leaf_lo: List[Int32] = [0, 2, 4]
    return _plan(2, 2, False, tree_lo, split, zeros, zeros, leaves, leaf_lo, map_lo, map_col, map_share)


def _near(a: Float32, b: Float64, tol: Float64) -> Bool:
    var d = Float64(a) - b
    return d <= tol and d >= -tol


def check_weight_words() raises:
    require(gbdt_imp_exponent(Float32(1.0)) == 127, "exponent of 1.0")
    require(gbdt_imp_exponent(bitcast[DType.float32](UInt32(1))) == 1, "exponent of the least subnormal")
    require(gbdt_imp_exponent(Float32(0.0)) == 0 and gbdt_imp_exponent(Float32(-1.0)) == 0, "zero / negative")
    require(gbdt_imp_exponent(bitcast[DType.float32](UInt32(0x7FC00000))) == 0, "NaN")
    require(gbdt_imp_exponent(bitcast[DType.float32](UInt32(0x7F800000))) == 0, "inf")
    require(gbdt_imp_weight_word(Float32(1.0), 127) == UInt32(0x800000), "word of 1.0")
    require(gbdt_imp_weight_word(Float32(0.5), 127) == UInt32(0x400000), "word of 0.5")
    require(gbdt_imp_weight_word(Float32(3.0), 128) == UInt32(0xC00000), "word of 3.0")
    require(gbdt_imp_weight_word(Float32(1.0), 128) == UInt32(0x400000), "word of 1.0 under 3.0")
    require(gbdt_imp_weight_word(Float32(1.0), 152) == UInt32(0), "a weight 2^25 times smaller")


def check_two_stumps() raises:
    var leaves: List[Float32] = [1.0, 3.0, 0.0, 1.0]
    var plan = _two_stumps(leaves)
    var w = List[Float32](length=1, fill=Float32(0.0))
    var got = _run(plan, _x(), 4, w, 0)
    require(len(got) == 2, "two input columns")
    comptime if GBDT_IMP_SABOTAGE:
        # the sabotage build credits the next column: the lane's digest must move
        require(got[0] == Float32(0.2) and got[1] == Float32(0.8), "sabotage: [0.2, 0.8]")
    else:
        require(got[0] == Float32(0.8) and got[1] == Float32(0.2),
                "two stumps: [" + String(got[0]) + ", " + String(got[1]) + "], want [0.8, 0.2]")


def check_oblivious_depth2() raises:
    var map_lo = List[Int32]()
    var map_col = List[Int32]()
    var map_share = List[Float32]()
    _identity_map(2, map_lo, map_col, map_share)
    var tree_lo: List[Int32] = [0, 2]
    var split: List[Int32] = [0, 1]
    var zeros: List[Int32] = [0, 0]
    var leaves: List[Float32] = [0.0, 2.0, 0.0, 2.0]
    var leaf_lo: List[Int32] = [0, 4]
    var plan = _plan(2, 2, False, tree_lo, split, zeros, zeros, leaves, leaf_lo, map_lo, map_col, map_share)
    var w = List[Float32](length=1, fill=Float32(0.0))
    var got = _run(plan, _x(), 4, w, 0)
    comptime if not GBDT_IMP_SABOTAGE:
        require(got[0] == Float32(1.0) and got[1] == Float32(0.0),
                "depth 2: [" + String(got[0]) + ", " + String(got[1]) + "], want [1, 0]")


def check_non_symmetric() raises:
    var map_lo = List[Int32]()
    var map_col = List[Int32]()
    var map_share = List[Float32]()
    _identity_map(2, map_lo, map_col, map_share)
    var tree_lo: List[Int32] = [0, 2]
    var split: List[Int32] = [0, 1]
    var left: List[Int32] = [1, 1]
    var right: List[Int32] = [2, 1]
    var leaves: List[Float32] = [0.0, 1.0, 3.0]
    var leaf_lo: List[Int32] = [0, 3]
    var plan = _plan(2, 2, True, tree_lo, split, left, right, leaves, leaf_lo, map_lo, map_col, map_share)
    var w = List[Float32](length=1, fill=Float32(0.0))
    var got = _run(plan, _x(), 4, w, 0)
    comptime if not GBDT_IMP_SABOTAGE:
        require(_near(got[0], 2.0 / 3.0, 1e-7) and _near(got[1], 1.0 / 3.0, 1e-7),
                "non-symmetric: [" + String(got[0]) + ", " + String(got[1]) + "], want [2/3, 1/3]")


def check_weights() raises:
    var leaves: List[Float32] = [1.0, 3.0, 0.0, 1.0]
    var plan = _two_stumps(leaves)
    # [1, 1, 1, 3]: tree 0 (column 0) W_L 2, W_R 4, root 6: (2/6)(4/6)(1-3)^2 = 8/9; tree 1 (column 1) W_L 2
    # (rows 0, 2), W_R 4 (rows 1, 3): (2/6)(4/6)(0-1)^2 = 2/9; so [0.8, 0.2] up to float32 rounding.
    var w: List[Float32] = [1.0, 1.0, 1.0, 3.0]
    var got = _run(plan, _x(), 4, w, 4)
    comptime if not GBDT_IMP_SABOTAGE:
        require(_near(got[0], 0.8, 1e-6) and _near(got[1], 0.2, 1e-6),
                "weighted: [" + String(got[0]) + ", " + String(got[1]) + "], want [0.8, 0.2]")
    # weights that move the answer: [3, 1, 3, 1] balance tree 0 (4 | 4: factor 1/4) and unbalance tree 1
    # (rows 0, 2 = 6 | rows 1, 3 = 2: factor 3/16), leaves [1, 3] / [0, 4] (squared gaps 4, 16).
    var moved: List[Float32] = [1.0, 3.0, 0.0, 4.0]
    var plan2 = _two_stumps(moved)
    var w2: List[Float32] = [3.0, 1.0, 3.0, 1.0]
    var plain = _run(plan2, _x(), 4, List[Float32](length=1, fill=Float32(0.0)), 0)
    var heavy = _run(plan2, _x(), 4, w2, 4)
    # unweighted: 1 and 4 -> [0.2, 0.8]; weighted: 1 and 3 -> [0.25, 0.75]
    comptime if not GBDT_IMP_SABOTAGE:
        require(_near(plain[0], 0.2, 1e-6) and _near(heavy[0], 0.25, 1e-6),
                "weights move the share: " + String(plain[0]) + " / " + String(heavy[0]))
    var two = List[Float32](length=4, fill=Float32(2.0))
    var scaled = _run(plan2, _x(), 4, two, 4)
    require(scaled[0] == plain[0] and scaled[1] == plain[1], "a constant weight 2.0 gives the unweighted bits")


def check_combination() raises:
    var map_lo: List[Int32] = [0, 1, 2, 4]
    var map_col: List[Int32] = [0, 1, 0, 1]
    var map_share: List[Float32] = [1.0, 1.0, 0.5, 0.5]
    var tree_lo: List[Int32] = [0, 1]
    var split: List[Int32] = [2]
    var zeros: List[Int32] = [0]
    var leaves: List[Float32] = [1.0, 3.0]
    var leaf_lo: List[Int32] = [0, 2]
    var plan = _plan(3, 2, False, tree_lo, split, zeros, zeros, leaves, leaf_lo, map_lo, map_col, map_share)
    # model column 2 holds x0 (its bins split rows 0, 1 | 2, 3): the X has three model columns
    var x: List[Float32] = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    var w = List[Float32](length=1, fill=Float32(0.0))
    var got = _run(plan, x, 4, w, 0)
    require(len(got) == 2, "two input columns")
    comptime if not GBDT_IMP_SABOTAGE:
        require(got[0] == Float32(0.5) and got[1] == Float32(0.5), "combination: [0.5, 0.5]")
    # the unnormalised fold of one tree's contributions: [0.5, 0.5] for a decrease of 1
    var contrib: List[Float32] = [0.5, 0.5]
    var total = List[Float32](length=2, fill=Float32(0.0))
    gbdt_imp_fold_body(0, 1, 2, GbdtImpF32(unsafe_from_address=Int(contrib.unsafe_ptr())),
                       GbdtImpF32(unsafe_from_address=Int(total.unsafe_ptr())))
    require(total[0] == Float32(0.5), "fold of one tree")
    _ = len(contrib)
    var flat: List[Float32] = [2.0, 2.0]
    var plan_flat = _plan(3, 2, False, tree_lo, split, zeros, zeros, flat, leaf_lo, map_lo, map_col, map_share)
    var none = _run(plan_flat, x, 4, w, 0)
    require(none[0] == Float32(0.0) and none[1] == Float32(0.0), "equal leaves give zeros")


def check_plan_refusals() raises:
    var map_lo = List[Int32]()
    var map_col = List[Int32]()
    var map_share = List[Float32]()
    _identity_map(2, map_lo, map_col, map_share)
    var tree_lo: List[Int32] = [0, 1]
    var zeros: List[Int32] = [0]
    var leaves: List[Float32] = [1.0, 3.0]
    var leaf_lo: List[Int32] = [0, 2]
    var past: List[Int32] = [2]
    var refused = False
    try:
        _ = _plan(2, 2, False, tree_lo, past, zeros, zeros, leaves, leaf_lo, map_lo, map_col, map_share)
    except e:
        refused = "splits on column 2 of 2" in String(e)
    require(refused, "a split column past the model columns")
    var ok_split: List[Int32] = [0]
    var short_lo: List[Int32] = [0, 1]
    refused = False
    try:
        _ = _plan(2, 2, False, tree_lo, ok_split, zeros, zeros, leaves, short_lo, map_lo, map_col, map_share)
    except e:
        refused = "leaf values, not 2" in String(e)
    require(refused, "a tree with the wrong leaf count")
    var bad_col: List[Int32] = [0, 5]
    refused = False
    try:
        _ = _plan(2, 2, False, tree_lo, ok_split, zeros, zeros, leaves, leaf_lo, map_lo, bad_col, map_share)
    except e:
        refused = "maps past the input columns" in String(e)
    require(refused, "a map entry past the input columns")


def test_mf_imp6_gbdt_importances() raises:
    def run_case(which: Int) raises:
        if which == 0:
            check_weight_words()
        elif which == 1:
            check_two_stumps()
        elif which == 2:
            check_oblivious_depth2()
        elif which == 3:
            check_non_symmetric()
        elif which == 4:
            check_weights()
        elif which == 5:
            check_combination()
        else:
            check_plan_refusals()
    check_cases("mf_imp6_gbdt_importances", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
