# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Gradient-boosted tree prediction on the host, for a box with no GPU.

HOST ONLY. Nothing here imports `max.gpu`, `max.gpu` or a `DeviceContext`,
and the GPU bindings do not import this file. It exists because every step
of a GBDT prediction on the GPU path is a kernel or a device buffer:
`predict_floats` (`gbdt/train.mojo:2322-2415`) quantizes on the device
(`_build_cindex_from_floats`, `gbdt/train.mojo:261-370`, launching
`binarize_float_feature_kernel`), then `predict`
(`gbdt/methods/doc_parallel_boosting.mojo:2319-2495`) fills a device cursor
with the bias and launches one apply kernel per tree. The model text parser
(`gbdt/models/model_text.mojo:706`) returns a `TrainedModel` that lives in
`gbdt/train.mojo`, which imports `max.gpu.host` at line 21, and the tree
structures (`gbdt/models/oblivious_model.mojo`) reach
`gbdt/methods/helpers.mojo` and `gbdt/data/leaf_path.mojo`, which import
the searcher and the tensor builder. So neither the parser nor the
structures can be imported into a build with no accelerator target, and
the Python loader (`python/mojolearn/_gbdt_host.py`) parses the text into
flat arrays instead, the way `GradientBoosting._tree_metadata` already
reads it for `get_leaf_values`.

WHAT IS REUSED, NOT RESTATED. `build_layout`
(`gbdt/gpu_data/compressed_index_builder.mojo:64`), which decides every
feature's column, mask and shift, and `nan_substitution` with the
`NAN_TREATMENT_*` codes (`gbdt/data/quantization.mojo:73-123`). Both come
from files whose module-level imports are GPU-free (`grid_policy.mojo` and
`gpu_structures.mojo` import nothing; `quantization.mojo` imports
`std.math`, `gbdt/grid_creator/binarization.mojo` and
`gbdt/options/data_processing_options.mojo`, neither of which imports a
device module). The packed compressed index is built with the SAME layout
the GPU path builds, so the shifted-and-masked comparisons below are the
kernels' comparisons on the kernels' words, not an argument that an
unpacked comparison is equivalent.

WHAT IS RESTATED, AND WHERE THE ORIGINAL IS. Each restatement is the
kernel's loop body with the grid index replaced by a row loop.

| here | MIRRORS | file:lines |
| --- | --- | --- |
| `_plan_features` and `_binarize_block` | the NaN staging of `_build_cindex_from_floats` and `binarize_float_feature_kernel` | `gbdt/train.mojo:325-360`, `gbdt/gpu_data/kernel/binarize.mojo:83-160` |
| `_plan_oblivious` and `_apply_oblivious_block` | the packing loop of `predict` and `compute_bins_and_add_kernel` | `gbdt/methods/doc_parallel_boosting.mojo:2392-2437`, `gbdt/models/kernel/add_bin_values.mojo:33-124` |
| `_plan_non_symmetric` and `_apply_non_symmetric_block` | `compute_non_symmetric_bins_for_model`, `compute_non_symmetric_decision_tree_bins_kernel`, `add_bin_model_value_kernel` | `gbdt/models/add_non_symmetric_tree_doc_parallel.mojo:63-188`, `gbdt/models/kernel/add_bin_values.mojo:222-320`, `gbdt/methods/kernel_add_model_value.mojo:114-184` |
| the cursor seed and the row-major transpose | `predict`'s `enqueue_fill(ctx, cursor, Float32(model.bias))` and `predict_multi_floats`'s read-back | `gbdt/methods/doc_parallel_boosting.mojo:2347`, `gbdt/train.mojo:2470-2474` |

The arithmetic a prediction's bits depend on is three things. The
`value > border` float32 comparisons of the quantizer, the integer bin
walk, and one float32 add per tree into a cursor seeded with
`Float32(bias)`, in tree order. The link functions (the Logloss sigmoid)
are the binding's, not this module's; `gbdt_sigmoid` already runs on the
host in the GPU binding (`bindings/_mojolearn_gbdt.mojo:133-149`) and the
host binding carries the same body.

The restatement is a prediction until measured. tools/forest_host_gate.py
is the measurement.

DEVIATION 2901 (lane/infer-speed-trees, 2026-09-17): THE SAME ARITHMETIC,
LAID OUT FOR A HOST. The GPU path quantizes every row of a feature in one
kernel and walks every row of a tree in one kernel; a serial host restating
that shape swept the whole row set once per feature and once per tree. Now
one serial pass validates the model and packs the per-feature and
per-record tables (`_plan_*`, every check and message of the old loops),
and the rows fan out to host threads (`host_worker_count`, the forest
walk's `MOJOLEARN_CPU_THREADS` reading) in contiguous ranges walked in
blocks of `GBDT_HOST_BLOCK_ROWS`. Inside a block every cell still gets the
kernels' comparisons on the kernels' words and one float32 add per tree in
tree order into a cursor seeded with `Float32(bias)`. Bit invariants, per
change:
  threads and blocks: a cell's adds are its own row's, in tree order, so
    no thread count and no block size reaches a bit;
  the sorted-border bisection: `borders` are visited in file order and the
    bin is the count of borders the value EXCEEDS; for a non-decreasing
    border list that count is the length of the true prefix of
    `value > border`, which a bisection finds without evaluating every
    compare. A feature whose borders are not non-decreasing (a NaN border
    included) keeps the linear count. The compare is the same `>` on the
    same float32 values either way, and the bin is an integer.
What moved: a model that is malformed AND fed a NaN on an `AsIs` column is
refused for the model before the NaN, where the old loops refused the NaN
first; both are refusals.
"""
from max.algorithm import sync_parallelize
from std.sys.compile import is_defined

from core.forest_host_predict import host_task_count, host_worker_count
from gbdt.data.quantization import NAN_TREATMENT_AS_IS, nan_substitution
from gbdt.gpu_data.compressed_index_builder import (
    CompressedIndexLayout,
    build_layout,
)


#: The gate's negative control, shared with the forest walk
#: (`core/forest_host_predict.mojo`). A build with this define seeds the
#: cursor at `bias + 1` instead of `bias`, so every raw prediction of every
#: fixture is wrong by construction, and the gate must say so.
comptime GBDT_HOST_SABOTAGE = is_defined["MOJOLEARN_FOREST_HOST_SABOTAGE"]()

#: `borders[0]` is the count and the kernel's shared buffer has 256 slots
#: (`binarize.mojo:121-125`), so a feature carries at most 255 borders.
comptime GBDT_HOST_MAX_BORDERS = 255
#: `load_model_text` refuses a deeper oblivious tree ("tree depth is not
#: sane"), and `1 << depth` leaf values must fit an Int.
comptime GBDT_HOST_MAX_DEPTH = 31
#: DEVIATION 2901: rows quantized and walked per block inside a task. At
#: 220 one-byte features a block's compressed index is 55 words a row, so
#: 2048 rows keep the index and the cursor in a core's second-level cache
#: while every tree of the model passes over them.
comptime GBDT_HOST_BLOCK_ROWS = 2048
#: The per-task failure record: kind (0 none, 1 NaN on an AsIs feature, 2 a
#: non-symmetric walk left its tree, 3 a non-symmetric walk left its bins),
#: then the three numbers the message names.
comptime GBDT_HOST_FAIL_WORDS = 4
comptime GBDT_HOST_FAIL_NAN = 1
comptime GBDT_HOST_FAIL_NODE = 2
comptime GBDT_HOST_FAIL_BIN = 3


def _seed(bias: Float64) -> Float32:
    """`Float32(model.bias)`, the cursor's starting value
    (`doc_parallel_boosting.mojo:2347`)."""
    comptime if GBDT_HOST_SABOTAGE:
        return Float32(bias) + Float32(1.0)
    return Float32(bias)


struct _GbdtHostPlan(Movable):
    """The tables a row task reads (DEVIATION 2901): one entry per feature,
    one per record (an oblivious level or a non-symmetric node) and one per
    tree, packed by the serial `_plan_*` pass that also makes every
    structural check the old per-tree loops made."""

    var columns: Int
    var f_lo: List[Int]
    var f_n: List[Int]
    var f_sorted: List[UInt8]
    var f_treat: List[Int]
    var f_sub: List[Float32]
    var f_col: List[Int]
    var f_mask: List[UInt32]
    var f_shift: List[UInt32]
    var r_col: List[Int]
    var r_mask: List[UInt32]
    var r_shift: List[UInt32]
    var r_val: List[UInt32]
    var r_eq: List[UInt8]
    var r_left: List[Int]
    var r_right: List[Int]
    var t_lo: List[Int]
    var t_n: List[Int]
    var t_leaf: List[Int]

    def __init__(out self, columns: Int):
        self.columns = columns
        self.f_lo = List[Int]()
        self.f_n = List[Int]()
        self.f_sorted = List[UInt8]()
        self.f_treat = List[Int]()
        self.f_sub = List[Float32]()
        self.f_col = List[Int]()
        self.f_mask = List[UInt32]()
        self.f_shift = List[UInt32]()
        self.r_col = List[Int]()
        self.r_mask = List[UInt32]()
        self.r_shift = List[UInt32]()
        self.r_val = List[UInt32]()
        self.r_eq = List[UInt8]()
        self.r_left = List[Int]()
        self.r_right = List[Int]()
        self.t_lo = List[Int]()
        self.t_n = List[Int]()
        self.t_leaf = List[Int]()


def _plan_features(
    mut plan: _GbdtHostPlan,
    n_features: Int,
    border_offsets_p: MutPointer[Int32, MutUntrackedOrigin],
    borders_p: MutPointer[Float32, MutUntrackedOrigin],
    nan_p: MutPointer[Int32, MutUntrackedOrigin],
    layout: CompressedIndexLayout,
) raises:
    """The per-feature half of the old `_binarize`: its offset and count
    checks, in feature order, plus whether the feature's borders are
    non-decreasing (the bisection's precondition, DEVIATION 2901)."""
    for f in range(n_features):
        var lo = Int(border_offsets_p[f])
        var hi = Int(border_offsets_p[f + 1])
        if hi < lo or lo < 0:
            raise Error("gbdt host: border offsets are not a prefix scan")
        var n_b = hi - lo
        if n_b > GBDT_HOST_MAX_BORDERS:
            raise Error(
                "gbdt host: feature " + String(f) + " carries " + String(n_b)
                + " borders, more than the quantizer's 255"
            )
        var sorted = True
        for b in range(1, n_b):
            if not (borders_p[lo + b - 1] <= borders_p[lo + b]):
                sorted = False
        ref cf = layout.features[f]
        var treat = Int(nan_p[f])
        plan.f_lo.append(lo)
        plan.f_n.append(n_b)
        plan.f_sorted.append(UInt8(1) if sorted else UInt8(0))
        plan.f_treat.append(treat)
        plan.f_sub.append(nan_substitution(treat))
        plan.f_col.append(Int(cf.offset))
        plan.f_mask.append(cf.mask)
        plan.f_shift.append(cf.shift)


def _binarize_block(
    plan: _GbdtHostPlan,
    x_p: MutPointer[Float32, MutUntrackedOrigin],
    borders_p: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    row0: Int,
    nb: Int,
    cindex_p: MutPointer[UInt32, MutUntrackedOrigin],
    fail_p: MutPointer[Int, MutUntrackedOrigin],
    row_major: Bool,
) -> Bool:
    """The compressed index of rows `[row0, row0 + nb)`, `nb * columns`
    words at `cindex_p`, row `r` of column `c` at `c * nb + r`, zeroed by
    the caller.

    MIRRORS `_build_cindex_from_floats` (`gbdt/train.mojo:325-360`): a
    feature with no borders is skipped and its bits stay zero; under
    `AsIs` a NaN is refused with their message (recorded in `fail_p`, the
    task cannot raise); under `AsFalse` and `AsTrue` it is replaced by the
    substitution value before the compare. Then
    `binarize_float_feature_kernel` (`binarize.mojo:139-160`): the bin is
    the number of borders the value EXCEEDS, borders visited in file order,
    and `(bin & mask) << shift` is OR-ed into the word. For non-decreasing
    borders the count is found by bisection (DEVIATION 2901).
    Returns False when the block was refused."""
    var f_lo = plan.f_lo.unsafe_ptr()
    var f_n = plan.f_n.unsafe_ptr()
    var f_sorted = plan.f_sorted.unsafe_ptr()
    var f_treat = plan.f_treat.unsafe_ptr()
    var f_sub = plan.f_sub.unsafe_ptr()
    var f_col = plan.f_col.unsafe_ptr()
    var f_mask = plan.f_mask.unsafe_ptr()
    var f_shift = plan.f_shift.unsafe_ptr()
    for f in range(n_features):
        var n_b = f_n.unsafe_load(f)
        if n_b == 0:
            continue
        var lo = f_lo.unsafe_load(f)
        var sorted = f_sorted.unsafe_load(f) != 0
        var treat = f_treat.unsafe_load(f)
        var sub = f_sub.unsafe_load(f)
        var base = f_col.unsafe_load(f) * nb
        var mask = f_mask.unsafe_load(f)
        var shift = f_shift.unsafe_load(f)
        var src = f * n_rows + row0
        var bp = borders_p + lo
        for r in range(nb):
            var v: Float32
            if row_major:
                v = x_p.unsafe_load((row0 + r) * n_features + f)
            else:
                v = x_p.unsafe_load(src + r)
            if v != v:
                if treat == NAN_TREATMENT_AS_IS:
                    fail_p.unsafe_store(0, GBDT_HOST_FAIL_NAN)
                    fail_p.unsafe_store(1, f)
                    fail_p.unsafe_store(2, row0 + r)
                    fail_p.unsafe_store(3, 0)
                    return False
                v = sub
            var index = UInt32(0)
            if sorted:
                # the first border the value does not exceed; every border
                # before it is exceeded, none after it is
                var a = 0
                var b = n_b
                while a < b:
                    var mid = (a + b) // 2
                    if v > bp.unsafe_load(mid):
                        a = mid + 1
                    else:
                        b = mid
                index = UInt32(a)
            else:
                for b in range(n_b):
                    if v > bp.unsafe_load(b):
                        index += 1
            var at = base + r
            cindex_p.unsafe_store(at, cindex_p.unsafe_load(at) | ((index & mask) << shift))
    return True


def _check_split_feature(
    layout: CompressedIndexLayout,
    feature: Int32,
    take_bin: Int32,
    n_features: Int,
    what: String,
) raises -> Int:
    """The feature id must exist and the predicate must agree with the
    layout, which is `predict`'s check (`doc_parallel_boosting.mojo:2409-2427`)
    and `compute_non_symmetric_bins_for_model`'s (`:113-131`)."""
    var fid = Int(feature)
    if fid < 0 or fid >= n_features:
        raise Error(
            "gbdt host: " + what + " splits on feature " + String(fid)
            + " of " + String(n_features)
        )
    var is_take_bin = take_bin != 0
    if is_take_bin != layout.features[fid].one_hot_feature:
        raise Error(
            "gbdt host: " + what + " is a "
            + String("TakeBin" if is_take_bin else "TakeGreater")
            + " split on feature " + String(fid)
            + ", which the layout says is "
            + String("one-hot" if layout.features[fid].one_hot_feature else "ordered")
        )
    return fid


def _plan_oblivious(
    mut plan: _GbdtHostPlan,
    layout: CompressedIndexLayout,
    n_features: Int,
    dim: Int,
    n_trees: Int,
    tree_offsets_p: MutPointer[Int32, MutUntrackedOrigin],
    split_feature_p: MutPointer[Int32, MutUntrackedOrigin],
    split_bin_p: MutPointer[Int32, MutUntrackedOrigin],
    split_take_bin_p: MutPointer[Int32, MutUntrackedOrigin],
    leaf_offsets_p: MutPointer[Int32, MutUntrackedOrigin],
    n_splits: Int,
    n_leaf_values: Int,
) raises:
    """The per-tree records of every tree, packed as `predict` packs them
    (`doc_parallel_boosting.mojo:2392-2437`): the feature's column,
    `mask << shift`, `bin_idx << shift`, and the predicate off the model;
    every check of the old per-tree loop, in tree order."""
    for t in range(n_trees):
        var lo = Int(tree_offsets_p[t])
        var hi = Int(tree_offsets_p[t + 1])
        if lo < 0 or hi < lo or hi > n_splits:
            raise Error("gbdt host: tree offsets are not a prefix scan inside the split arrays")
        var depth = hi - lo
        if depth > GBDT_HOST_MAX_DEPTH:
            raise Error("gbdt host: tree " + String(t) + " depth " + String(depth) + " is not sane")
        var leaf_lo = Int(leaf_offsets_p[t])
        var leaf_hi = Int(leaf_offsets_p[t + 1])
        var n_values = (1 << depth) * dim
        if leaf_lo < 0 or leaf_hi - leaf_lo != n_values or leaf_hi > n_leaf_values:
            raise Error(
                "gbdt host: tree " + String(t) + " has depth " + String(depth)
                + ", dim " + String(dim) + " and " + String(leaf_hi - leaf_lo)
                + " leaf values, not " + String(n_values)
            )
        plan.t_lo.append(len(plan.r_col))
        plan.t_n.append(depth)
        plan.t_leaf.append(leaf_lo)
        for level in range(depth):
            var fid = _check_split_feature(
                layout, split_feature_p[lo + level], split_take_bin_p[lo + level],
                n_features, "tree " + String(t) + " level " + String(level),
            )
            ref cf = layout.features[fid]
            var bin_idx = Int(split_bin_p[lo + level])
            if bin_idx < 0:
                raise Error("gbdt host: tree " + String(t) + " level " + String(level) + " has a negative bin")
            plan.r_col.append(Int(cf.offset))
            plan.r_mask.append(cf.mask << cf.shift)
            plan.r_shift.append(cf.shift)
            plan.r_val.append(UInt32(bin_idx) << cf.shift)
            plan.r_eq.append(UInt8(1) if split_take_bin_p[lo + level] != 0 else UInt8(0))
            plan.r_left.append(0)
            plan.r_right.append(0)


def _apply_oblivious_block(
    plan: _GbdtHostPlan,
    cindex_p: MutPointer[UInt32, MutUntrackedOrigin],
    nb: Int,
    dim: Int,
    n_trees: Int,
    leaves_p: MutPointer[Float32, MutUntrackedOrigin],
    cursor_p: MutPointer[Float32, MutUntrackedOrigin],
):
    """MIRRORS `compute_bins_and_add_kernel` per tree in tree order
    (`add_bin_values.mojo:98-124`) over the `nb` rows of one block. Leaf
    values are BIN-major `[leaf * dim + d]`; the block's cursor is
    PLANE-major `[d * nb + r]`, so cell `(d, r)` receives one add per tree
    in tree order, exactly the whole-row-set cursor's sequence for its row."""
    var r_col = plan.r_col.unsafe_ptr()
    var r_mask = plan.r_mask.unsafe_ptr()
    var r_val = plan.r_val.unsafe_ptr()
    var r_eq = plan.r_eq.unsafe_ptr()
    var t_lo = plan.t_lo.unsafe_ptr()
    var t_n = plan.t_n.unsafe_ptr()
    var t_leaf = plan.t_leaf.unsafe_ptr()
    for t in range(n_trees):
        var rlo = t_lo.unsafe_load(t)
        var depth = t_n.unsafe_load(t)
        var leaf_lo = t_leaf.unsafe_load(t)
        for r in range(nb):
            var leaf = 0
            for level in range(depth):
                var rec = rlo + level
                var feature_val = (
                    cindex_p.unsafe_load(r_col.unsafe_load(rec) * nb + r)
                    & r_mask.unsafe_load(rec)
                )
                var split: Bool
                if r_eq.unsafe_load(rec) != 0:
                    split = feature_val == r_val.unsafe_load(rec)
                else:
                    split = feature_val > r_val.unsafe_load(rec)
                if split:
                    leaf += 1 << level
            for d in range(dim):
                var at = d * nb + r
                cursor_p.unsafe_store(
                    at, cursor_p.unsafe_load(at) + leaves_p.unsafe_load(leaf_lo + leaf * dim + d)
                )


def _plan_non_symmetric(
    mut plan: _GbdtHostPlan,
    layout: CompressedIndexLayout,
    n_features: Int,
    dim: Int,
    n_trees: Int,
    tree_offsets_p: MutPointer[Int32, MutUntrackedOrigin],
    node_feature_p: MutPointer[Int32, MutUntrackedOrigin],
    node_bin_p: MutPointer[Int32, MutUntrackedOrigin],
    node_take_bin_p: MutPointer[Int32, MutUntrackedOrigin],
    node_left_p: MutPointer[Int32, MutUntrackedOrigin],
    node_right_p: MutPointer[Int32, MutUntrackedOrigin],
    leaf_offsets_p: MutPointer[Int32, MutUntrackedOrigin],
    n_nodes_total: Int,
    n_leaf_values: Int,
) raises:
    """The node table of every tree, `compute_non_symmetric_bins_for_model`'s
    (`add_non_symmetric_tree_doc_parallel.mojo:104-141`), with every check
    of the old per-tree loop in tree order. A tree with no nodes is the
    constant tree: every row lands in bin 0."""
    for t in range(n_trees):
        var lo = Int(tree_offsets_p[t])
        var hi = Int(tree_offsets_p[t + 1])
        if lo < 0 or hi < lo or hi > n_nodes_total:
            raise Error("gbdt host: tree offsets are not a prefix scan inside the node arrays")
        var n_nodes = hi - lo
        var n_bins = n_nodes + 1
        var leaf_lo = Int(leaf_offsets_p[t])
        var leaf_hi = Int(leaf_offsets_p[t + 1])
        if leaf_lo < 0 or leaf_hi - leaf_lo != n_bins * dim or leaf_hi > n_leaf_values:
            raise Error(
                "gbdt host: non-symmetric tree " + String(t) + " has " + String(n_bins)
                + " bins, dim " + String(dim) + " and " + String(leaf_hi - leaf_lo)
                + " leaf values"
            )
        plan.t_lo.append(len(plan.r_col))
        plan.t_n.append(n_nodes)
        plan.t_leaf.append(leaf_lo)
        for i in range(n_nodes):
            var fid = _check_split_feature(
                layout, node_feature_p[lo + i], node_take_bin_p[lo + i],
                n_features, "non-symmetric node " + String(i) + " of tree " + String(t),
            )
            ref cf = layout.features[fid]
            var bin = Int(node_bin_p[lo + i])
            var left = Int(node_left_p[lo + i])
            var right = Int(node_right_p[lo + i])
            # their `CB_ENSURE(LeftSubtree >= 1 && RightSubtree >= 1)`
            # (`non_symmetric_tree.mojo:151-153`)
            if bin < 0 or left < 1 or right < 1:
                raise Error(
                    "gbdt host: non-symmetric tree " + String(t) + " node "
                    + String(i) + " has bin " + String(bin) + ", left subtree "
                    + String(left) + ", right subtree " + String(right)
                )
            plan.r_col.append(Int(cf.offset))
            plan.r_mask.append(cf.mask)
            plan.r_shift.append(cf.shift)
            plan.r_val.append(UInt32(bin))
            plan.r_eq.append(UInt8(1) if node_take_bin_p[lo + i] != 0 else UInt8(0))
            plan.r_left.append(left)
            plan.r_right.append(right)


def _apply_non_symmetric_block(
    plan: _GbdtHostPlan,
    cindex_p: MutPointer[UInt32, MutUntrackedOrigin],
    nb: Int,
    row0: Int,
    dim: Int,
    n_trees: Int,
    leaves_p: MutPointer[Float32, MutUntrackedOrigin],
    cursor_p: MutPointer[Float32, MutUntrackedOrigin],
    fail_p: MutPointer[Int, MutUntrackedOrigin],
) -> Bool:
    """MIRRORS, per tree in tree order over one block, the walk of
    `compute_non_symmetric_decision_tree_bins_kernel`
    (`add_bin_values.mojo:277-320`, shift THEN mask, as written there) and
    the add of `add_bin_model_value_kernel`
    (`kernel_add_model_value.mojo:171-184`, `bin_values[bin * dim + d]`
    into cell `(d, r)`). The walk is bounds-checked, which the kernel is
    not; a file whose subtree sizes point past the tree is refused
    (recorded in `fail_p`) instead of read past. Returns False when the
    block was refused."""
    var r_col = plan.r_col.unsafe_ptr()
    var r_mask = plan.r_mask.unsafe_ptr()
    var r_shift = plan.r_shift.unsafe_ptr()
    var r_val = plan.r_val.unsafe_ptr()
    var r_eq = plan.r_eq.unsafe_ptr()
    var r_left = plan.r_left.unsafe_ptr()
    var r_right = plan.r_right.unsafe_ptr()
    var t_lo = plan.t_lo.unsafe_ptr()
    var t_n = plan.t_n.unsafe_ptr()
    var t_leaf = plan.t_leaf.unsafe_ptr()
    for t in range(n_trees):
        var rlo = t_lo.unsafe_load(t)
        var n_nodes = t_n.unsafe_load(t)
        var n_bins = n_nodes + 1
        var leaf_lo = t_leaf.unsafe_load(t)
        for r in range(nb):
            var bin = 0
            var node = 0
            var stop = n_nodes == 0
            while not stop:
                if node >= n_nodes:
                    fail_p.unsafe_store(0, GBDT_HOST_FAIL_NODE)
                    fail_p.unsafe_store(1, t)
                    fail_p.unsafe_store(2, node)
                    fail_p.unsafe_store(3, n_nodes)
                    return False
                var rec = rlo + node
                var feature_val = (
                    cindex_p.unsafe_load(r_col.unsafe_load(rec) * nb + r)
                    >> r_shift.unsafe_load(rec)
                ) & r_mask.unsafe_load(rec)
                var split: Bool
                if r_eq.unsafe_load(rec) != 0:
                    split = feature_val == r_val.unsafe_load(rec)
                else:
                    split = feature_val > r_val.unsafe_load(rec)
                if split:
                    var left = r_left.unsafe_load(rec)
                    bin += left
                    stop = r_right.unsafe_load(rec) == 1
                    if not stop:
                        node += left
                else:
                    stop = r_left.unsafe_load(rec) == 1
                    if not stop:
                        node += 1
            if bin >= n_bins:
                fail_p.unsafe_store(0, GBDT_HOST_FAIL_BIN)
                fail_p.unsafe_store(1, t)
                fail_p.unsafe_store(2, bin)
                fail_p.unsafe_store(3, n_bins)
                return False
            for d in range(dim):
                var at = d * nb + r
                cursor_p.unsafe_store(
                    at, cursor_p.unsafe_load(at) + leaves_p.unsafe_load(leaf_lo + bin * dim + d)
                )
    _ = row0
    return True


def _walk_rows(
    plan: _GbdtHostPlan,
    x_p: MutPointer[Float32, MutUntrackedOrigin],
    borders_p: MutPointer[Float32, MutUntrackedOrigin],
    leaves_p: MutPointer[Float32, MutUntrackedOrigin],
    out_p: MutPointer[Float32, MutUntrackedOrigin],
    fail_p: MutPointer[Int, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    dim: Int,
    n_trees: Int,
    non_symmetric: Bool,
    seed: Float32,
    lo: Int,
    hi: Int,
    row_major: Bool,
):
    """One task's rows `[lo, hi)`, a block at a time: quantize the block,
    seed its cursor, add every tree, then write the block's rows to `out_p`
    ROW-major (`predict_multi_floats`'s read-back, `[row * approx_dim + dim]`,
    `gbdt/train.mojo:2470-2474`; for `dim == 1` it is `predict_floats`'s).
    Stops at the first refusal, which is in `fail_p`."""
    var columns = plan.columns
    var cindex = List[UInt32](length=GBDT_HOST_BLOCK_ROWS * columns, fill=UInt32(0))
    var cursor = List[Float32](length=GBDT_HOST_BLOCK_ROWS * dim, fill=seed)
    var cp = rebind[MutPointer[UInt32, MutUntrackedOrigin]](cindex.unsafe_ptr())
    var up = rebind[MutPointer[Float32, MutUntrackedOrigin]](cursor.unsafe_ptr())
    var row0 = lo
    while row0 < hi:
        var nb = hi - row0
        if nb > GBDT_HOST_BLOCK_ROWS:
            nb = GBDT_HOST_BLOCK_ROWS
        for i in range(nb * columns):
            cp.unsafe_store(i, UInt32(0))
        for i in range(nb * dim):
            up.unsafe_store(i, seed)
        if not _binarize_block(
            plan, x_p, borders_p, n_rows, n_features, row0, nb, cp, fail_p,
            row_major,
        ):
            return
        if non_symmetric:
            if not _apply_non_symmetric_block(plan, cp, nb, row0, dim, n_trees, leaves_p, up, fail_p):
                return
        else:
            _apply_oblivious_block(plan, cp, nb, dim, n_trees, leaves_p, up)
        for r in range(nb):
            for d in range(dim):
                out_p.unsafe_store((row0 + r) * dim + d, up.unsafe_load(d * nb + r))
        row0 += nb
    _ = cindex^
    _ = cursor^


def _raise_first_failure(failed: List[Int], tasks: Int) raises:
    """The refusal the old serial loops would have raised first: among the
    tasks that refused, the smallest feature (a NaN) or the smallest tree
    (a walk), ties to the lower rows."""
    var best = -1
    for c in range(tasks):
        var kind = failed[c * GBDT_HOST_FAIL_WORDS]
        if kind == 0:
            continue
        if best < 0:
            best = c
            continue
        var bk = failed[best * GBDT_HOST_FAIL_WORDS]
        var ba = failed[best * GBDT_HOST_FAIL_WORDS + 1]
        var a = failed[c * GBDT_HOST_FAIL_WORDS + 1]
        # a NaN refusal precedes a walk refusal, as the quantizer ran first
        if (kind == GBDT_HOST_FAIL_NAN and bk != GBDT_HOST_FAIL_NAN) or (
                (kind == GBDT_HOST_FAIL_NAN) == (bk == GBDT_HOST_FAIL_NAN) and a < ba):
            best = c
    if best < 0:
        return
    var kind = failed[best * GBDT_HOST_FAIL_WORDS]
    var a = failed[best * GBDT_HOST_FAIL_WORDS + 1]
    var b = failed[best * GBDT_HOST_FAIL_WORDS + 2]
    var d = failed[best * GBDT_HOST_FAIL_WORDS + 3]
    if kind == GBDT_HOST_FAIL_NAN:
        raise Error(
            "There are NaNs in feature number " + String(a)
            + " but there were no NaNs in the learn dataset"
        )
    if kind == GBDT_HOST_FAIL_NODE:
        raise Error(
            "gbdt host: non-symmetric tree " + String(a)
            + " walks to node " + String(b) + " of " + String(d)
        )
    raise Error(
        "gbdt host: non-symmetric tree " + String(a) + " sends a row to bin "
        + String(b) + " of " + String(d)
    )


def gbdt_host_predict(
    x_colmajor: List[Float32],
    n_rows: Int,
    n_features: Int,
    border_offsets_p: MutPointer[Int32, MutUntrackedOrigin],
    borders_p: MutPointer[Float32, MutUntrackedOrigin],
    fold_counts_p: MutPointer[Int32, MutUntrackedOrigin],
    one_hot_p: MutPointer[Int32, MutUntrackedOrigin],
    nan_p: MutPointer[Int32, MutUntrackedOrigin],
    tree_offsets_p: MutPointer[Int32, MutUntrackedOrigin],
    split_feature_p: MutPointer[Int32, MutUntrackedOrigin],
    split_bin_p: MutPointer[Int32, MutUntrackedOrigin],
    split_take_bin_p: MutPointer[Int32, MutUntrackedOrigin],
    node_left_p: MutPointer[Int32, MutUntrackedOrigin],
    node_right_p: MutPointer[Int32, MutUntrackedOrigin],
    leaf_offsets_p: MutPointer[Int32, MutUntrackedOrigin],
    leaves_p: MutPointer[Float32, MutUntrackedOrigin],
    n_trees: Int,
    dim: Int,
    non_symmetric: Bool,
    n_splits: Int,
    n_leaf_values: Int,
    bias: Float64,
    mut out: List[Float32],
    workers: Int = 0,
    row_major: Bool = False,
) raises:
    """`predict_floats` / `predict_multi_floats` on the host: RAW approxes,
    `out[r * dim + d]`, ROW-major, `n_rows * dim` values.

    `x_colmajor` is COLUMN-major, `n_rows * n_features`, the layout the GPU
    binding takes (`gbdt_predict`, `bindings/_mojolearn_gbdt.mojo:410`).
    The split arrays hold `n_splits` records, one per oblivious level or
    one per non-symmetric node, `tree_offsets` their prefix scan;
    `leaf_offsets` is the prefix scan of leaf VALUE counts (leaves times
    `dim`). `fold_counts`, `one_hot` and `nan` are one per feature.
    `workers` is the thread count, `host_worker_count`'s reading of zero
    (DEVIATION 2901).
    """
    if n_rows <= 0 or n_features <= 0 or dim <= 0 or n_trees < 0:
        raise Error("gbdt host: n_rows, n_features and dim must be positive")
    if len(x_colmajor) < n_rows * n_features:
        raise Error("gbdt host: x holds fewer than n_rows * n_features values")
    if len(out) < n_rows * dim:
        raise Error("gbdt host: out holds fewer than n_rows * dim values")
    if Int(tree_offsets_p[0]) != 0 or Int(tree_offsets_p[n_trees]) != n_splits:
        raise Error("gbdt host: tree_offsets must start at 0 and end at n_splits")
    if Int(leaf_offsets_p[0]) != 0 or Int(leaf_offsets_p[n_trees]) != n_leaf_values:
        raise Error("gbdt host: leaf_offsets must start at 0 and end at n_leaf_values")
    var fold_counts = List[Int](capacity=n_features)
    var one_hot = List[Bool](capacity=n_features)
    for f in range(n_features):
        fold_counts.append(Int(fold_counts_p[f]))
        one_hot.append(one_hot_p[f] != 0)
    var layout = build_layout(fold_counts, one_hot)
    var plan = _GbdtHostPlan(layout.columns)
    _plan_features(plan, n_features, border_offsets_p, borders_p, nan_p, layout)
    if non_symmetric:
        _plan_non_symmetric(
            plan, layout, n_features, dim, n_trees,
            tree_offsets_p, split_feature_p, split_bin_p, split_take_bin_p,
            node_left_p, node_right_p, leaf_offsets_p, n_splits, n_leaf_values,
        )
    else:
        _plan_oblivious(
            plan, layout, n_features, dim, n_trees,
            tree_offsets_p, split_feature_p, split_bin_p, split_take_bin_p,
            leaf_offsets_p, n_splits, n_leaf_values,
        )
    var seed = _seed(bias)
    var tasks = host_task_count(n_rows, host_worker_count(workers))
    var chunk = (n_rows + tasks - 1) // tasks
    var failed = List[Int](length=tasks * GBDT_HOST_FAIL_WORDS, fill=0)
    var pp = Pointer(to=plan)
    var xp = rebind[MutPointer[Float32, MutUntrackedOrigin]](x_colmajor.unsafe_ptr())
    var op = rebind[MutPointer[Float32, MutUntrackedOrigin]](out.unsafe_ptr())
    var fp = rebind[MutPointer[Int, MutUntrackedOrigin]](failed.unsafe_ptr())

    def _rows_task(c: Int) {imm pp, imm xp, imm borders_p, imm leaves_p, imm op, imm fp,
                            imm chunk, imm n_rows, imm n_features, imm dim, imm n_trees,
                            imm non_symmetric, imm seed, imm row_major}:
        var lo = c * chunk
        var hi = lo + chunk
        if hi > n_rows:
            hi = n_rows
        _walk_rows(
            pp[], xp, borders_p, leaves_p, op, fp + c * GBDT_HOST_FAIL_WORDS,
            n_rows, n_features, dim, n_trees, non_symmetric, seed, lo, hi,
            row_major,
        )

    if tasks == 1:
        _rows_task(0)
    else:
        sync_parallelize(_rows_task, tasks)
    _raise_first_failure(failed, tasks)
    # the tasks read `plan`, `x_colmajor` and wrote `out` through pointers;
    # a use after the join keeps every owner alive past it
    _ = plan^
    _ = failed^
