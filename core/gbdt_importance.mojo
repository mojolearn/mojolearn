# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`feature_importances_` of a fitted gradient-boosting model (lane mf-imp6,
2026-11-23): the plan, the ONE set of `@always_inline` bodies and the host
column. The device column is `gbdt/gpu_importance.mojo`, whose kernels call
the SAME bodies in the same fold order, so the host and every GPU vendor
return the same bits.

WHAT IS SUMMED. scikit-learn's `GradientBoosting*.feature_importances_`
(sklearn/ensemble/_gb.py): per tree `tree_.compute_feature_importances(
normalize=False)`, the weighted impurity decrease of every split credited to
its feature (`w_N * imp_N - w_L * imp_L - w_R * imp_R`) divided by the root
weight, averaged over the trees with a nonzero decrease, then divided by the
sum (all zero: zeros). Our trees keep no per-node impurity, so the impurity
of a node is the weighted squared spread of the TREE'S OWN LEAF VALUES over
the fit's rows that reach it:

  * W(leaf) is the sum of the fit's sample weights (1 per row without
    sample_weight) of the training rows the fitted tree sends to the leaf
    (the predict walk on the fit's own X, so bootstrap and subsample play no
    part);
  * a split node N with children L and R decreases that spread by
        W_L * W_R / W_N * sum_d (m_L,d - m_R,d)^2
    where m is the W-weighted mean leaf value of a child's leaves and d runs
    over the model's output dimensions (MultiClass and MultiRMSE sum their
    dimensions, as scikit-learn sums its per-class trees). This IS
    scikit-learn's squared-error decrease whenever every leaf holds the mean
    of its rows' targets (scaled by the learning rate); for Newton or
    regularised leaves it is the same formula on the leaves the model holds;
  * per tree the decreases are divided by the root weight W_root and
    credited to the split's model column; a simple CTR column credits its
    input column; a tensor or combination CTR column credits each of its
    distinct input columns an equal share;
  * the input columns are summed over the trees in tree order and divided by
    their sum; a model whose sum is zero (no split with an effect) gives
    zeros. scikit-learn's mean over the splitting trees is a common factor
    the normalisation removes, so it is not taken.

BITS, ONE ARITHMETIC ON EVERY COLUMN. The leaf weights are EXACT integers: a
sample weight becomes its 24-bit mantissa shifted right by its exponent's
distance below the largest weight's exponent (`gbdt_imp_weight_word`: integer
operations only), and the words are added as 64-bit integers (the device as
two 32-bit atomics with an exact carry), so the sum does not depend on the
order. Everything after is float32 with the identity row's construction:
pinned products (`identical_mul`), the flushed correctly rounded division
(`identical_div`), every result flushed (`ftz`), in source order: one fixed
fold per tree, one per input column over the trees, one for the normalising
sum. Float32 because the Apple device has no float64; the door widens the
result exactly to float64.

PARALLEL PLAN (host). The bins split over model columns; the leaf weights
over row ranges, each task with its own exact accumulators (integer sums,
any order), joined over leaf slots; the per-tree decreases over trees; the
per-column sums over input columns. The normalisation is one fold over the
input columns, serial by nature (its order is the bits). Device: one thread
per (row, column) for the bins, per (row, tree) for the weights, per tree,
per input column, and one for the normalisation.
"""
from std.memory import bitcast
from std.sys.compile import is_defined

from checks.numerics import ftz, identical_div, identical_mul
from core.forest_host_predict import host_task_count, host_worker_count
from core.gbdt_host_ctr import gbdt_host_expand_ctr
from core.host_archive_gbdt import HostArchiveGBDT, host_archive_gbdt_text
from core.host_parallel import host_parallelize
from gbdt.data.quantization import NAN_TREATMENT_AS_FALSE, NAN_TREATMENT_AS_TRUE
from gbdt.models.model_text import load_model_text
from gbdt.models.ctr_value_table import column_plan
from gbdt.models.non_symmetric_walk_body import ns_node_goes_right, ns_walk_step

comptime GbdtImpI32 = MutPointer[Int32, MutAnyOrigin]
comptime GbdtImpU32 = MutPointer[UInt32, MutAnyOrigin]
comptime GbdtImpU8 = MutPointer[UInt8, MutAnyOrigin]
comptime GbdtImpF32 = MutPointer[Float32, MutAnyOrigin]

#: The sabotage arm of the importance bodies (tools/identity_lanes
#: coverage_sabotage.tsv, row gbdt-importances): a build with this define
#: credits every split to the NEXT model column, on the host and the device,
#: so the importance vector of any model splitting on a column moves and the
#: verify lane must say so.
comptime GBDT_IMP_SABOTAGE = is_defined["MOJOLEARN_GBDT_IMPORTANCE_SABOTAGE"]()

#: The host accumulators of one leaf-weight task are 8 bytes a leaf slot;
#: the task count is capped so their sum stays under this many bytes.
comptime GBDT_IMP_HOST_ACC_BYTES = 67108864


# ===========================================================================
# THE BODIES (host and device: typed values, no launch, no allocation)
# ===========================================================================


@always_inline
def gbdt_imp_exponent(w: Float32) -> Int:
    """The scale exponent of a sample weight: the biased exponent (1 for a
    subnormal), 0 for a weight that counts as zero (zero, negative, NaN or
    infinite: the fit refuses such weights; a row with one adds nothing)."""
    var bits = bitcast[DType.uint32](w)
    if (bits & UInt32(0x80000000)) != UInt32(0):
        return 0
    var e = Int((bits >> UInt32(23)) & UInt32(0xFF))
    if e == 255:
        return 0
    if e == 0:
        if (bits & UInt32(0x7FFFFF)) == UInt32(0):
            return 0
        return 1
    return e


@always_inline
def gbdt_imp_weight_word(w: Float32, e_max: Int) -> UInt32:
    """The exact integer word of a sample weight: its 24-bit mantissa (the
    implicit bit included) shifted right by `e_max - exponent`, `e_max` the
    largest weight's exponent. The largest weight maps into [2^23, 2^24); a
    weight more than 2^24 times smaller maps to 0. Integer operations only."""
    var e = gbdt_imp_exponent(w)
    if e == 0:
        return UInt32(0)
    var bits = bitcast[DType.uint32](w)
    var mant = bits & UInt32(0x7FFFFF)
    if ((bits >> UInt32(23)) & UInt32(0xFF)) != UInt32(0):
        mant = mant | UInt32(0x800000)
    var shift = e_max - e
    if shift >= 24:
        return UInt32(0)
    if shift <= 0:
        return mant
    return mant >> UInt32(shift)


@always_inline
def gbdt_imp_bin_body(
    value: Float32, lo: Int, n_b: Int, is_sorted: Bool, treat: Int, borders: GbdtImpF32,
) -> UInt8:
    """The quantized bin of one value of a model column: the count of the
    column's borders the value EXCEEDS (the predict quantizer's rule,
    `core/gbdt_host_predict.mojo::_binarize_block`: bisection for
    non-decreasing borders, else the linear count), a NaN replaced by the
    column's substitution first (`nan_substitution`: +inf AsTrue, -inf
    AsFalse; AsIs keeps it and it exceeds no border). A column carries at
    most 255 borders, so the bin fits a byte."""
    var v = value
    if v != v:
        if treat == NAN_TREATMENT_AS_TRUE:
            v = bitcast[DType.float32](UInt32(0x7F800000))
        elif treat == NAN_TREATMENT_AS_FALSE:
            v = bitcast[DType.float32](UInt32(0xFF800000))
    var index = 0
    if is_sorted:
        var a = 0
        var b = n_b
        while a < b:
            var mid = (a + b) // 2
            if v > borders[lo + mid]:
                a = mid + 1
            else:
                b = mid
        index = a
    else:
        for k in range(n_b):
            if v > borders[lo + k]:
                index += 1
    return UInt8(index)


@always_inline
def gbdt_imp_row_leaf(
    t: Int, r: Int, bins: GbdtImpU8, n_rows: Int, tree_lo: GbdtImpI32, split_feature: GbdtImpI32,
    split_bin: GbdtImpI32, split_take: GbdtImpI32, node_left: GbdtImpI32, node_right: GbdtImpI32,
    non_symmetric: Bool,
) -> Int:
    """The leaf tree `t` sends row `r` to (oblivious: `sum bit_l << l`;
    non-symmetric: the bin, the leaves to the row's left), with predict's
    predicates (`==` for TakeBin, `>` for TakeGreater) and predict's
    non-symmetric step (`ns_walk_step`), over the column-major bins. -1 for a
    walk that leaves its tree or its bins (a malformed model)."""
    var lo = Int(tree_lo[t])
    var n = Int(tree_lo[t + 1]) - lo
    if not non_symmetric:
        var leaf = 0
        for level in range(n):
            var rec = lo + level
            var b = UInt32(bins[Int(split_feature[rec]) * n_rows + r])
            var go: Bool
            if split_take[rec] != 0:
                go = b == UInt32(split_bin[rec])
            else:
                go = b > UInt32(split_bin[rec])
            if go:
                leaf += 1 << level
        return leaf
    var bin = 0
    var node = 0
    var stop = n == 0
    while not stop:
        if node >= n:
            return -1
        var rec = lo + node
        var b = UInt32(bins[Int(split_feature[rec]) * n_rows + r])
        stop = ns_walk_step(ns_node_goes_right(b, UInt32(split_bin[rec]), split_take[rec] != 0),
                            Int(node_left[rec]), Int(node_right[rec]), bin, node)
    if bin > n:
        return -1
    return bin


@always_inline
def gbdt_imp_f32(w: UInt64) -> Float32:
    """A leaf weight as float32: the integer's correctly rounded value."""
    return w.cast[DType.float32]()


@always_inline
def gbdt_imp_w(wpair: GbdtImpU32, slot: Int) -> UInt64:
    """A leaf's exact weight from its (low, high) 32-bit words."""
    return UInt64(wpair[2 * slot]) | (UInt64(wpair[2 * slot + 1]) << UInt64(32))


@always_inline
def gbdt_imp_credit(f: Int, gain: Float32, n_cols: Int, map_lo: GbdtImpI32, map_col: GbdtImpI32,
                    map_share: GbdtImpF32, contrib: GbdtImpF32, base: Int):
    """`gain` credited to model column `f`'s input columns, in map order."""
    var col = f
    comptime if GBDT_IMP_SABOTAGE:
        col = (f + 1) % n_cols
    for k in range(Int(map_lo[col]), Int(map_lo[col + 1])):
        var at = base + Int(map_col[k])
        contrib[at] = ftz(contrib[at] + ftz(identical_mul(gain, map_share[k])))


@always_inline
def gbdt_imp_node_gain(wl: UInt64, wr: UInt64, froot: Float32, acc: Float32) -> Float32:
    """`W_L * W_R / (W_N * W_root) * acc`, as (W_L / W_root) * (W_R / W_N) * acc."""
    var a = identical_div(gbdt_imp_f32(wl), froot)
    var b = identical_div(gbdt_imp_f32(wr), gbdt_imp_f32(wl + wr))
    return ftz(identical_mul(ftz(identical_mul(a, b)), acc))


@always_inline
def gbdt_imp_tree_body(
    t: Int, tree_lo: GbdtImpI32, leaf_lo: GbdtImpI32, dim: Int, non_symmetric: Bool,
    split_feature: GbdtImpI32, node_left: GbdtImpI32, node_right: GbdtImpI32, leaves: GbdtImpF32,
    wpair: GbdtImpU32, n_cols: Int, map_lo: GbdtImpI32, map_col: GbdtImpI32, map_share: GbdtImpF32,
    n_input: Int, contrib: GbdtImpF32,
):
    """Tree `t`'s decreases, divided by its root weight, into
    `contrib[t * n_input ..]` (zeroed here first). Oblivious: the node of
    level j holding prefix p covers the leaves `p + m * 2^j`, its left child
    the even m. Non-symmetric (pre-order nodes whose left / right words are
    LEAF counts): node i's first leaf is found by descending from the root
    (left subtree: nodes j+1 .. j+left-1; right: from j+left, its leaves
    after the left's), its children own the next `left` and `right` leaves."""
    var base = t * n_input
    for c in range(n_input):
        contrib[base + c] = Float32(0.0)
    var lo = Int(tree_lo[t])
    var n = Int(tree_lo[t + 1]) - lo
    if n <= 0:
        return
    var vlo = Int(leaf_lo[t])
    var wlo = vlo // dim
    var n_leaves = n + 1 if non_symmetric else 1 << n
    var root = UInt64(0)
    for l in range(n_leaves):
        root += gbdt_imp_w(wpair, wlo + l)
    if root == UInt64(0):
        return
    var froot = gbdt_imp_f32(root)
    if not non_symmetric:
        for j in range(n):
            var span = 1 << j
            var count = n_leaves // span
            var gain = Float32(0.0)
            for p in range(span):
                var wl = UInt64(0)
                var wr = UInt64(0)
                for m in range(count):
                    if (m & 1) == 0:
                        wl += gbdt_imp_w(wpair, wlo + p + m * span)
                    else:
                        wr += gbdt_imp_w(wpair, wlo + p + m * span)
                if wl == UInt64(0) or wr == UInt64(0):
                    continue
                var acc = Float32(0.0)
                for d in range(dim):
                    var sl = Float32(0.0)
                    var sr = Float32(0.0)
                    for m in range(count):
                        var l = p + m * span
                        var wv = ftz(identical_mul(gbdt_imp_f32(gbdt_imp_w(wpair, wlo + l)), leaves[vlo + l * dim + d]))
                        if (m & 1) == 0:
                            sl = ftz(sl + wv)
                        else:
                            sr = ftz(sr + wv)
                    var diff = ftz(identical_div(sl, gbdt_imp_f32(wl)) - identical_div(sr, gbdt_imp_f32(wr)))
                    acc = ftz(acc + ftz(identical_mul(diff, diff)))
                gain = ftz(gain + gbdt_imp_node_gain(wl, wr, froot, acc))
            gbdt_imp_credit(Int(split_feature[lo + j]), gain, n_cols, map_lo, map_col, map_share, contrib, base)
        return
    for i in range(n):
        var b0 = 0
        var j = 0
        var steps = 0
        while j != i and steps < n:
            var lj = Int(node_left[lo + j])
            if lj < 1:
                break
            if i < j + lj:
                j += 1
            else:
                b0 += lj
                j += lj
            steps += 1
        if j != i:
            continue
        var left = Int(node_left[lo + i])
        var right = Int(node_right[lo + i])
        if left < 1 or right < 1 or b0 + left + right > n_leaves:
            continue
        var wl = UInt64(0)
        var wr = UInt64(0)
        for k in range(left):
            wl += gbdt_imp_w(wpair, wlo + b0 + k)
        for k in range(right):
            wr += gbdt_imp_w(wpair, wlo + b0 + left + k)
        if wl == UInt64(0) or wr == UInt64(0):
            continue
        var acc = Float32(0.0)
        for d in range(dim):
            var sl = Float32(0.0)
            var sr = Float32(0.0)
            for k in range(left):
                var l = b0 + k
                sl = ftz(sl + ftz(identical_mul(gbdt_imp_f32(gbdt_imp_w(wpair, wlo + l)), leaves[vlo + l * dim + d])))
            for k in range(right):
                var l = b0 + left + k
                sr = ftz(sr + ftz(identical_mul(gbdt_imp_f32(gbdt_imp_w(wpair, wlo + l)), leaves[vlo + l * dim + d])))
            var diff = ftz(identical_div(sl, gbdt_imp_f32(wl)) - identical_div(sr, gbdt_imp_f32(wr)))
            acc = ftz(acc + ftz(identical_mul(diff, diff)))
        gbdt_imp_credit(Int(split_feature[lo + i]), gbdt_imp_node_gain(wl, wr, froot, acc), n_cols, map_lo,
                        map_col, map_share, contrib, base)


@always_inline
def gbdt_imp_fold_body(c: Int, n_trees: Int, n_input: Int, contrib: GbdtImpF32, total: GbdtImpF32):
    """Input column `c`'s sum over the trees, in tree order."""
    var s = Float32(0.0)
    for t in range(n_trees):
        s = ftz(s + contrib[t * n_input + c])
    total[c] = s


@always_inline
def gbdt_imp_finish_body(n_input: Int, total: GbdtImpF32, dst: GbdtImpF32):
    """The normalisation: the columns' sum in column order, each column over
    it; zeros when the sum is zero."""
    var s = Float32(0.0)
    for c in range(n_input):
        s = ftz(s + total[c])
    for c in range(n_input):
        if s > Float32(0.0):
            dst[c] = identical_div(total[c], s)
        else:
            dst[c] = Float32(0.0)


# ===========================================================================
# THE PLAN (host, once per call: model data only)
# ===========================================================================


struct GbdtImportancePlan(Movable):
    """The flat model the bodies read, and the model-column -> input-column
    map (CSR: `map_lo` per model column, `map_col` / `map_share`)."""

    var n_cols: Int
    var n_input: Int
    var n_trees: Int
    var dim: Int
    var non_symmetric: Bool
    var has_ctr: Bool
    var border_lo: List[Int32]
    var borders: List[Float32]
    var treat: List[Int32]
    var sorted: List[Int32]
    var tree_lo: List[Int32]
    var leaf_lo: List[Int32]
    var split_feature: List[Int32]
    var split_bin: List[Int32]
    var split_take: List[Int32]
    var node_left: List[Int32]
    var node_right: List[Int32]
    var leaves: List[Float32]
    var map_lo: List[Int32]
    var map_col: List[Int32]
    var map_share: List[Float32]
    var n_leaf_slots: Int

    def __init__(out self):
        self.n_cols = 0
        self.n_input = 0
        self.n_trees = 0
        self.dim = 1
        self.non_symmetric = False
        self.has_ctr = False
        self.border_lo = List[Int32]()
        self.borders = List[Float32]()
        self.treat = List[Int32]()
        self.sorted = List[Int32]()
        self.tree_lo = List[Int32]()
        self.leaf_lo = List[Int32]()
        self.split_feature = List[Int32]()
        self.split_bin = List[Int32]()
        self.split_take = List[Int32]()
        self.node_left = List[Int32]()
        self.node_right = List[Int32]()
        self.leaves = List[Float32]()
        self.map_lo = List[Int32]()
        self.map_col = List[Int32]()
        self.map_share = List[Float32]()
        self.n_leaf_slots = 0


def gbdt_importance_finish_plan(mut plan: GbdtImportancePlan) raises:
    """The plan's derived words and checks, once the flat arrays are in:
    the sorted flags, the leaf slot count and the structural checks the
    bodies rely on (offsets are prefix scans, split columns exist, a tree's
    leaf values are its leaves times `dim`, the column map stays inside the
    input columns). Model data only."""
    if plan.dim < 1 or plan.n_cols < 1 or plan.n_input < 1:
        raise Error("gbdt importances: the model has no columns or no output dimension")
    if len(plan.border_lo) < plan.n_cols + 1 or len(plan.treat) < plan.n_cols:
        raise Error("gbdt importances: the border table does not cover the model columns")
    plan.sorted = List[Int32](length=plan.n_cols, fill=Int32(1))
    for c in range(plan.n_cols):  # driver-loop: over the model's columns (model data)
        var lo = Int(plan.border_lo[c])
        var hi = Int(plan.border_lo[c + 1])
        if lo < 0 or hi < lo or hi > len(plan.borders) or hi - lo > 255:
            raise Error("gbdt importances: border offsets are not a prefix scan of at most 255 per column")
        for b in range(lo + 1, hi):  # small-loop(the borders of one column, at most 255)
            if not (plan.borders[b - 1] <= plan.borders[b]):
                plan.sorted[c] = Int32(0)
    if len(plan.tree_lo) < plan.n_trees + 1 or len(plan.leaf_lo) < plan.n_trees + 1:
        raise Error("gbdt importances: tree offsets do not cover the trees")
    var slots = 0
    for t in range(plan.n_trees):  # driver-loop: over the model's trees (model data)
        var lo = Int(plan.tree_lo[t])
        var n = Int(plan.tree_lo[t + 1]) - lo
        if lo < 0 or n < 0 or lo + n > len(plan.split_feature):
            raise Error("gbdt importances: tree " + String(t) + " points past the split records")
        if not plan.non_symmetric and n > 30:
            raise Error("gbdt importances: tree " + String(t) + " depth " + String(n) + " is not sane")
        var n_leaves = n + 1 if plan.non_symmetric else 1 << n
        var have = Int(plan.leaf_lo[t + 1]) - Int(plan.leaf_lo[t])
        if Int(plan.leaf_lo[t]) != slots * plan.dim or have != n_leaves * plan.dim:
            raise Error("gbdt importances: tree " + String(t) + " has " + String(have) + " leaf values, not "
                        + String(n_leaves * plan.dim))
        for k in range(n):  # small-loop(the splits of one tree)
            var f = Int(plan.split_feature[lo + k])
            if f < 0 or f >= plan.n_cols:
                raise Error("gbdt importances: tree " + String(t) + " splits on column " + String(f) + " of "
                            + String(plan.n_cols))
            if plan.non_symmetric and (Int(plan.node_left[lo + k]) < 1 or Int(plan.node_right[lo + k]) < 1):
                raise Error("gbdt importances: non-symmetric tree " + String(t) + " node " + String(k)
                            + " has an empty subtree")
        slots += n_leaves
    plan.n_leaf_slots = slots
    if len(plan.map_lo) != plan.n_cols + 1:
        raise Error("gbdt importances: the column map does not cover the model columns")
    for k in range(len(plan.map_col)):  # driver-loop: over the column map (model data)
        if Int(plan.map_col[k]) < 0 or Int(plan.map_col[k]) >= plan.n_input:
            raise Error("gbdt importances: a model column maps past the input columns")
    # the bodies and the device upload read index 0 of every array: an empty one holds one word
    if len(plan.split_feature) == 0:
        plan.split_feature.append(Int32(0))
        plan.split_bin.append(Int32(0))
        plan.split_take.append(Int32(0))
    if len(plan.node_left) < len(plan.split_feature):
        plan.node_left.resize(len(plan.split_feature), Int32(0))
        plan.node_right.resize(len(plan.split_feature), Int32(0))
    if len(plan.borders) == 0:
        plan.borders.append(Float32(0.0))
    if len(plan.leaves) == 0:
        plan.leaves.append(Float32(0.0))
    if len(plan.map_col) == 0:
        plan.map_col.append(Int32(0))
        plan.map_share.append(Float32(0.0))


def _distinct(values: List[Int]) -> List[Int]:
    var found = List[Int]()
    for v in values:  # small-loop(the members of one combination)
        var seen = False
        for u in found:  # small-loop(the members of one combination)
            if u == v:
                seen = True
        if not seen:
            found.append(v)
    return found^


def gbdt_importance_plan(text: String, arch: HostArchiveGBDT) raises -> GbdtImportancePlan:
    """The plan of a model text (`arch` its flat host layout,
    `host_archive_gbdt_text(text)`): the flat arrays and the column map
    (identity for raw columns; a simple CTR column its input through
    `column_plan`; a tensor CTR column its source columns; a combination
    column the distinct input features of its map's members)."""
    var plan = GbdtImportancePlan()
    plan.n_cols = arch.metadata["n_features"]
    plan.n_input = arch.metadata["n_input_features"]
    plan.n_trees = arch.metadata["n_trees"]
    plan.dim = arch.metadata["dim"]
    plan.non_symmetric = arch.non_symmetric
    plan.has_ctr = arch.metadata["n_ctr_tables"] != 0 or arch.metadata["n_tensor_tables"] != 0
    var n_splits = arch.metadata["n_splits"]
    var n_values = arch.metadata["n_leaf_values"]
    plan.border_lo = arch.integers["border_offsets"].copy()
    plan.borders = arch.floats["borders"].copy()
    plan.borders.resize(arch.metadata["n_borders"], Float32(0.0))
    plan.treat = arch.integers["nan_treatment"].copy()
    plan.tree_lo = arch.integers["tree_offsets"].copy()
    plan.leaf_lo = arch.integers["leaf_offsets"].copy()
    plan.split_feature = arch.integers["split_feature"].copy()
    plan.split_bin = arch.integers["split_bin"].copy()
    plan.split_take = arch.integers["split_take_bin"].copy()
    plan.node_left = arch.integers["node_left"].copy()
    plan.node_right = arch.integers["node_right"].copy()
    plan.split_feature.resize(n_splits, Int32(0))
    plan.split_bin.resize(n_splits, Int32(0))
    plan.split_take.resize(n_splits, Int32(0))
    plan.node_left.resize(n_splits, Int32(0))
    plan.node_right.resize(n_splits, Int32(0))
    plan.leaves = arch.floats["leaves"].copy()
    plan.leaves.resize(n_values, Float32(0.0))
    # THE COLUMN MAP, from the product reader's model (model data only)
    var saved = load_model_text(text)
    var n_tensors = len(saved.tensor_ctr_registry.features)
    var n_comb = len(saved.combination_ctrs.tables)
    var n_base = plan.n_cols - n_comb
    var source = List[Int]()
    if n_tensors == 0 and len(saved.ctr_tables) != 0:
        source = column_plan(saved.ctr_tables, n_base).source_of_column.copy()
    plan.map_lo.append(Int32(0))
    for c in range(plan.n_cols):  # driver-loop: over the model's columns (model data)
        var cols = List[Int]()
        if n_tensors != 0 and c >= saved.tensor_ctr_registry.first_model_column:
            var at = c - saved.tensor_ctr_registry.first_model_column
            cols = _distinct(saved.tensor_ctr_registry.features[at].table.source_features)
        elif n_tensors == 0 and c >= n_base:
            var map_at = saved.combination_ctrs.tables[c - n_base].source_feature
            var feats = List[Int]()
            for j in range(len(saved.combination_ctrs.maps[map_at].members)):  # small-loop(the members of one combination)
                feats.append(saved.combination_ctrs.maps[map_at].members[j].feature)
            cols = _distinct(feats)
        elif len(source) != 0:
            cols.append(source[c])
        else:
            cols.append(c)
        var share = identical_div(Float32(1.0), Float32(len(cols))) if len(cols) > 0 else Float32(0.0)
        for k in range(len(cols)):  # small-loop(the input columns of one model column)
            plan.map_col.append(Int32(cols[k]))
            plan.map_share.append(share)
        plan.map_lo.append(Int32(len(plan.map_col)))
    gbdt_importance_finish_plan(plan)
    return plan^


def gbdt_importance_model_x(
    arch: HostArchiveGBDT, x_addr: Int, n_rows: Int, n_input: Int, row_major: Bool,
) raises -> List[Float32]:
    """A CTR model's MODEL columns (column-major) from the fit's input (its
    categorical columns as the fit's dense codes), through the predict path's
    own expansion (`gbdt_host_expand_ctr`, the host statement the GPU predict
    runs as well, `gbdt/train.mojo::_expand_model_columns`), which reads the
    raw columns column-major: a C-order X is reordered first, a pure move
    split over columns."""
    var count = n_rows * n_input
    var x_raw = List[Float32](length=count, fill=Float32(0.0))
    var src = GbdtImpF32(unsafe_from_address=x_addr)
    var dst = GbdtImpF32(unsafe_from_address=Int(x_raw.unsafe_ptr()))

    def _column(c: Int) {imm src, imm dst, imm n_rows, imm n_input, imm row_major}:
        for r in range(n_rows):  # native data loop (one column's rows)
            if row_major:
                dst[c * n_rows + r] = src[r * n_input + c]
            else:
                dst[c * n_rows + r] = src[c * n_rows + r]

    host_parallelize(_column, n_input)
    var ctr_ints = arch.integers["ctr_ints"].copy()
    var ctr_floats = arch.floats["ctr_floats"].copy()
    var tensor_ints = arch.integers["tensor_ints"].copy()
    var tensor_floats = arch.floats["tensor_floats"].copy()
    var counts = arch.integers["ctr_counts"].copy()
    ctr_ints.resize(arch.metadata["n_ctr_ints"], Int32(0))
    ctr_floats.resize(arch.metadata["n_ctr_floats"], Float32(0))
    tensor_ints.resize(arch.metadata["n_tensor_ints"], Int32(0))
    tensor_floats.resize(arch.metadata["n_tensor_floats"], Float32(0))
    counts.resize(arch.metadata["n_ctr_counts"], Int32(0))
    var expanded = gbdt_host_expand_ctr(
        x_raw, n_rows, arch.metadata["n_features"], ctr_ints, ctr_floats, arch.metadata["n_ctr_tables"],
        tensor_ints, tensor_floats, arch.metadata["n_tensor_tables"], counts,
        arch.integers["border_offsets"], arch.floats["borders"], arch.integers["one_hot"],
    )
    _ = len(x_raw)  # the reorder tasks wrote it through `dst`; the expansion read it
    return expanded^


struct GbdtImportanceInput(Movable):
    """The plan and the float32 MODEL columns the walk reads: the caller's X
    itself (`x_addr`, `row_major` as given) or, for a CTR model, the expanded
    column-major copy this struct owns (`model_x`)."""

    var plan: GbdtImportancePlan
    var model_x: List[Float32]
    var x_addr: Int
    var row_major: Bool

    def __init__(out self, var plan: GbdtImportancePlan, var model_x: List[Float32], x_addr: Int, row_major: Bool):
        self.plan = plan^
        self.model_x = model_x^
        self.x_addr = x_addr
        self.row_major = row_major


def gbdt_importance_input(text: String, x_addr: Int, n_rows: Int, n_input: Int, row_major: Bool,
                          n_weights: Int) raises -> GbdtImportanceInput:
    """The checks of a request and its plan (both columns): the rows the
    model was fitted on (`n_rows` x `n_input` float32 at `x_addr`, C order
    when `row_major`), `n_weights` 0 or `n_rows`."""
    if n_rows <= 0 or n_input <= 0:
        raise Error("gbdt importances: the fit has no rows or no columns")
    if n_weights != 0 and n_weights != n_rows:
        raise Error("gbdt importances: sample weights must be one per row")
    var arch = host_archive_gbdt_text(text)
    var plan = gbdt_importance_plan(text, arch)
    if plan.n_input != n_input:
        raise Error("gbdt importances: the model reads " + String(plan.n_input) + " input columns, the fit "
                    + String(n_input))
    if plan.has_ctr or plan.n_cols != n_input:
        var mx = gbdt_importance_model_x(arch, x_addr, n_rows, n_input, row_major)
        var addr = Int(mx.unsafe_ptr())
        return GbdtImportanceInput(plan^, mx^, addr, False)
    return GbdtImportanceInput(plan^, List[Float32](), x_addr, row_major)


# ===========================================================================
# THE HOST COLUMN
# ===========================================================================


def _i32(values: List[Int32]) -> GbdtImpI32:
    return GbdtImpI32(unsafe_from_address=Int(values.unsafe_ptr()))


def _f32(values: List[Float32]) -> GbdtImpF32:
    return GbdtImpF32(unsafe_from_address=Int(values.unsafe_ptr()))


def gbdt_imp_bins_host(plan: GbdtImportancePlan, x_addr: Int, n_rows: Int, row_major: Bool,
                       workers: Int = 0) -> List[UInt8]:
    """The bins of every model column (column-major bytes), split over the
    columns: `gbdt_imp_bin_body` per cell."""
    var n_cols = plan.n_cols
    var bins = List[UInt8](length=max(n_rows * n_cols, 1), fill=UInt8(0))
    var bp = GbdtImpU8(unsafe_from_address=Int(bins.unsafe_ptr()))
    var xp = GbdtImpF32(unsafe_from_address=x_addr)
    var border_lo = _i32(plan.border_lo)
    var borders = _f32(plan.borders)
    var treat = _i32(plan.treat)
    var sorted = _i32(plan.sorted)

    def _column(c: Int) {imm bp, imm xp, imm border_lo, imm borders, imm treat, imm sorted, imm n_rows,
                         imm n_cols, imm row_major}:
        var lo = Int(border_lo[c])
        var n_b = Int(border_lo[c + 1]) - lo
        var is_sorted = sorted[c] != 0
        var tr = Int(treat[c])
        for r in range(n_rows):  # native data loop (one column's rows)
            var v: Float32
            if row_major:
                v = xp[r * n_cols + c]
            else:
                v = xp[c * n_rows + r]
            bp[c * n_rows + r] = gbdt_imp_bin_body(v, lo, n_b, is_sorted, tr, borders)

    host_parallelize(_column, n_cols)
    _ = workers
    return bins^


def gbdt_imp_max_exponent_host(w_addr: Int, n_rows: Int, workers: Int = 0) -> Int:
    """The largest weight exponent (1 when every weight counts as zero),
    split over row ranges; an integer maximum, order free."""
    var tasks = host_task_count(n_rows, host_worker_count(workers))
    var chunk = (n_rows + tasks - 1) // tasks
    var best = List[Int](length=tasks, fill=1)
    var bp = MutPointer[Int, MutAnyOrigin](unsafe_from_address=Int(best.unsafe_ptr()))
    var wp = GbdtImpF32(unsafe_from_address=w_addr)

    def _task(k: Int) {imm bp, imm wp, imm chunk, imm n_rows}:
        var m = 1
        for r in range(k * chunk, min(n_rows, (k + 1) * chunk)):  # native data loop (one task's rows)
            var e = gbdt_imp_exponent(wp[r])
            if e > m:
                m = e
        bp[k] = m

    host_parallelize(_task, tasks)
    var m = 1
    for k in range(tasks):  # small-loop(one word per task)
        if best[k] > m:
            m = best[k]
    return m


def gbdt_imp_leaf_weights_host(
    plan: GbdtImportancePlan, bins: List[UInt8], n_rows: Int, w_addr: Int, n_weights: Int, workers: Int = 0,
) raises -> List[UInt32]:
    """Every leaf's exact weight as (low, high) 32-bit words. The rows split
    into ranges, each task walking its rows through every tree into its OWN
    64-bit accumulators (integer sums: any order, the same words); then the
    tasks' accumulators are added per leaf slot, split over the slots."""
    var e_max = 1
    if n_weights > 0:
        e_max = gbdt_imp_max_exponent_host(w_addr, n_rows, workers)
    var slots = max(plan.n_leaf_slots, 1)
    var tasks = host_task_count(n_rows, host_worker_count(workers))
    var cap = GBDT_IMP_HOST_ACC_BYTES // (8 * slots)
    if cap < 1:
        cap = 1
    if tasks > cap:
        tasks = cap
    var chunk = (n_rows + tasks - 1) // tasks
    var acc = List[UInt64](length=tasks * slots, fill=UInt64(0))
    var failed = List[Int](length=tasks, fill=-1)
    var ap = MutPointer[UInt64, MutAnyOrigin](unsafe_from_address=Int(acc.unsafe_ptr()))
    var fp = MutPointer[Int, MutAnyOrigin](unsafe_from_address=Int(failed.unsafe_ptr()))
    var bp = GbdtImpU8(unsafe_from_address=Int(bins.unsafe_ptr()))
    var wp = GbdtImpF32(unsafe_from_address=w_addr)
    var tree_lo = _i32(plan.tree_lo)
    var leaf_lo = _i32(plan.leaf_lo)
    var split_feature = _i32(plan.split_feature)
    var split_bin = _i32(plan.split_bin)
    var split_take = _i32(plan.split_take)
    var node_left = _i32(plan.node_left)
    var node_right = _i32(plan.node_right)
    var n_trees = plan.n_trees
    var dim = plan.dim
    var non_symmetric = plan.non_symmetric

    def _rows(k: Int) {imm ap, imm fp, imm bp, imm wp, imm tree_lo, imm leaf_lo, imm split_feature, imm split_bin,
                       imm split_take, imm node_left, imm node_right, imm n_trees, imm dim, imm non_symmetric,
                       imm chunk, imm n_rows, imm n_weights, imm e_max, imm slots}:
        var own = k * slots
        for r in range(k * chunk, min(n_rows, (k + 1) * chunk)):  # native data loop (one task's rows)
            var q = UInt64(1)
            if n_weights > 0:
                q = UInt64(gbdt_imp_weight_word(wp[r], e_max))
            for t in range(n_trees):  # driver-loop: over the model's trees (one row's walk)
                var leaf = gbdt_imp_row_leaf(t, r, bp, n_rows, tree_lo, split_feature, split_bin, split_take,
                                             node_left, node_right, non_symmetric)
                if leaf < 0:
                    fp[k] = t
                    return
                var at = own + Int(leaf_lo[t]) // dim + leaf
                ap[at] = ap[at] + q

    host_parallelize(_rows, tasks)
    for k in range(tasks):  # small-loop(one refusal word per task)
        if failed[k] >= 0:
            raise Error("gbdt importances: non-symmetric tree " + String(failed[k]) + " walks a row out of its tree")
    var pairs = List[UInt32](length=2 * slots, fill=UInt32(0))
    var pp = GbdtImpU32(unsafe_from_address=Int(pairs.unsafe_ptr()))
    var joins = host_task_count(slots, host_worker_count(workers))
    var span = (slots + joins - 1) // joins

    def _join(j: Int) {imm ap, imm pp, imm tasks, imm slots, imm span}:
        for s in range(j * span, min(slots, (j + 1) * span)):  # native data loop (one task's leaf slots)
            var total = UInt64(0)
            for k in range(tasks):  # small-loop(one accumulator per row task)
                total += ap[k * slots + s]
            pp[2 * s] = UInt32(total & UInt64(0xFFFFFFFF))
            pp[2 * s + 1] = UInt32(total >> UInt64(32))

    host_parallelize(_join, joins)
    _ = len(acc)  # the row tasks and the join read it through `ap`
    _ = len(bins)
    return pairs^


def gbdt_imp_from_leaf_weights_host(plan: GbdtImportancePlan, wpairs: List[UInt32]) -> List[Float32]:
    """The per-tree decreases (split over trees), the per-column sums (split
    over input columns) and the normalisation, from the exact leaf weights."""
    var n_trees = plan.n_trees
    var n_input = plan.n_input
    var contrib = List[Float32](length=max(n_trees * n_input, 1), fill=Float32(0.0))
    var total = List[Float32](length=n_input, fill=Float32(0.0))
    var result = List[Float32](length=n_input, fill=Float32(0.0))
    var cp = _f32(contrib)
    var tp = _f32(total)
    var tree_lo = _i32(plan.tree_lo)
    var leaf_lo = _i32(plan.leaf_lo)
    var split_feature = _i32(plan.split_feature)
    var node_left = _i32(plan.node_left)
    var node_right = _i32(plan.node_right)
    var leaves = _f32(plan.leaves)
    var wp = GbdtImpU32(unsafe_from_address=Int(wpairs.unsafe_ptr()))
    var map_lo = _i32(plan.map_lo)
    var map_col = _i32(plan.map_col)
    var map_share = _f32(plan.map_share)
    var dim = plan.dim
    var non_symmetric = plan.non_symmetric
    var n_cols = plan.n_cols

    def _tree(t: Int) {imm tree_lo, imm leaf_lo, imm dim, imm non_symmetric, imm split_feature, imm node_left,
                       imm node_right, imm leaves, imm wp, imm n_cols, imm map_lo, imm map_col, imm map_share,
                       imm n_input, imm cp}:
        gbdt_imp_tree_body(t, tree_lo, leaf_lo, dim, non_symmetric, split_feature, node_left, node_right, leaves,
                           wp, n_cols, map_lo, map_col, map_share, n_input, cp)

    def _column(c: Int) {imm n_trees, imm n_input, imm cp, imm tp}:
        gbdt_imp_fold_body(c, n_trees, n_input, cp, tp)

    if n_trees > 0:
        host_parallelize(_tree, n_trees)
    host_parallelize(_column, n_input)
    gbdt_imp_finish_body(n_input, tp, _f32(result))
    _ = len(contrib)  # the tasks wrote it through `cp`
    _ = len(total)
    _ = len(wpairs)
    return result^


def gbdt_feature_importances_host(
    text: String, x_addr: Int, n_rows: Int, n_input: Int, row_major: Bool, w_addr: Int, n_weights: Int,
    workers: Int = 0,
) raises -> List[Float32]:
    """`feature_importances_` (THE HOST COLUMN) of the model `text` fitted on
    the `n_rows` x `n_input` float32 rows at `x_addr` (C order when
    `row_major`, else column-major; a CTR model's categorical columns as the
    fit's dense codes) with `n_weights` (0 or `n_rows`) sample weights at
    `w_addr`. Float32, one per input column."""
    var request = gbdt_importance_input(text, x_addr, n_rows, n_input, row_major, n_weights)
    var bins = gbdt_imp_bins_host(request.plan, request.x_addr, n_rows, request.row_major, workers)
    _ = len(request.model_x)  # the bins read the expanded model columns through `x_addr`
    var pairs = gbdt_imp_leaf_weights_host(request.plan, bins, n_rows, w_addr, n_weights, workers)
    return gbdt_imp_from_leaf_weights_host(request.plan, pairs)
