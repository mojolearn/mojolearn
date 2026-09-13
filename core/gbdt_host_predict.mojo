# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Gradient-boosted tree prediction on the host, for a box with no GPU.

HOST ONLY. Nothing here imports `max.gpu`, `std.gpu` or a `DeviceContext`,
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
| `_binarize` | the NaN staging of `_build_cindex_from_floats` and `binarize_float_feature_kernel` | `gbdt/train.mojo:325-360`, `gbdt/gpu_data/kernel/binarize.mojo:83-160` |
| `_apply_oblivious` | the packing loop of `predict` and `compute_bins_and_add_kernel` | `gbdt/methods/doc_parallel_boosting.mojo:2392-2437`, `gbdt/models/kernel/add_bin_values.mojo:33-124` |
| `_apply_non_symmetric` | `compute_non_symmetric_bins_for_model`, `compute_non_symmetric_decision_tree_bins_kernel`, `add_bin_model_value_kernel` | `gbdt/models/add_non_symmetric_tree_doc_parallel.mojo:63-188`, `gbdt/models/kernel/add_bin_values.mojo:222-320`, `gbdt/methods/kernel_add_model_value.mojo:114-184` |
| the cursor seed and the row-major transpose | `predict`'s `enqueue_fill(ctx, cursor, Float32(model.bias))` and `predict_multi_floats`'s read-back | `gbdt/methods/doc_parallel_boosting.mojo:2347`, `gbdt/train.mojo:2470-2474` |

The arithmetic a prediction's bits depend on is three things. The
`value > border` float32 comparisons of the quantizer, the integer bin
walk, and one float32 add per tree into a cursor seeded with
`Float32(bias)`, in tree order. The link functions (the Logloss sigmoid)
are the binding's, not this module's; `gbdt_sigmoid` already runs on the
host in the GPU binding (`bindings/_mojolearn_gbdt.mojo:133-149`) and the
host binding carries the same body.

The restatement is a prediction until measured. tools/forest_host_gate.py
is the measurement, and the brief in docs/lanes/ records what it has shown.
"""
from std.sys.compile import is_defined

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


def _seed(bias: Float64) -> Float32:
    """`Float32(model.bias)`, the cursor's starting value
    (`doc_parallel_boosting.mojo:2347`)."""
    comptime if GBDT_HOST_SABOTAGE:
        return Float32(bias) + Float32(1.0)
    return Float32(bias)


def _binarize(
    x_colmajor: List[Float32],
    n_rows: Int,
    n_features: Int,
    border_offsets_p: MutPointer[Int32, MutUntrackedOrigin],
    borders_p: MutPointer[Float32, MutUntrackedOrigin],
    nan_p: MutPointer[Int32, MutUntrackedOrigin],
    layout: CompressedIndexLayout,
) raises -> List[UInt32]:
    """The compressed index of the rows, `n_rows * layout.columns` words,
    row `r` of column `c` at `c * n_rows + r`.

    MIRRORS `_build_cindex_from_floats` (`gbdt/train.mojo:325-360`): a
    feature with no borders is skipped and its bits stay zero; under
    `AsIs` a NaN is refused with their message; under `AsFalse` and
    `AsTrue` it is replaced by the substitution value before the compare.
    Then `binarize_float_feature_kernel` (`binarize.mojo:139-160`): the
    bin is the number of borders the value EXCEEDS, borders visited in
    file order, and `(bin & mask) << shift` is OR-ed into the word.
    """
    var cindex = List[UInt32](length=n_rows * layout.columns, fill=UInt32(0))
    for f in range(n_features):
        var lo = Int(border_offsets_p[f])
        var hi = Int(border_offsets_p[f + 1])
        if hi < lo or lo < 0:
            raise Error("gbdt host: border offsets are not a prefix scan")
        var n_b = hi - lo
        if n_b == 0:
            continue
        if n_b > GBDT_HOST_MAX_BORDERS:
            raise Error(
                "gbdt host: feature " + String(f) + " carries " + String(n_b)
                + " borders, more than the quantizer's 255"
            )
        ref cf = layout.features[f]
        var treat = Int(nan_p[f])
        var sub = nan_substitution(treat)
        var base = Int(cf.offset) * n_rows
        for r in range(n_rows):
            var v = x_colmajor[f * n_rows + r]
            if v != v:
                if treat == NAN_TREATMENT_AS_IS:
                    raise Error(
                        "There are NaNs in feature number " + String(f)
                        + " but there were no NaNs in the learn dataset"
                    )
                v = sub
            var index = UInt32(0)
            for b in range(n_b):
                if v > borders_p[lo + b]:
                    index += 1
            var word = cindex[base + r]
            word |= (index & cf.mask) << cf.shift
            cindex[base + r] = word
    return cindex^


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


def _apply_oblivious(
    layout: CompressedIndexLayout,
    cindex: List[UInt32],
    n_rows: Int,
    n_features: Int,
    dim: Int,
    n_trees: Int,
    tree_offsets_p: MutPointer[Int32, MutUntrackedOrigin],
    split_feature_p: MutPointer[Int32, MutUntrackedOrigin],
    split_bin_p: MutPointer[Int32, MutUntrackedOrigin],
    split_take_bin_p: MutPointer[Int32, MutUntrackedOrigin],
    leaf_offsets_p: MutPointer[Int32, MutUntrackedOrigin],
    leaves_p: MutPointer[Float32, MutUntrackedOrigin],
    n_splits: Int,
    n_leaf_values: Int,
    mut cursor: List[Float32],
) raises:
    """MIRRORS `compute_bins_and_add_kernel` per tree in tree order
    (`add_bin_values.mojo:98-124`), with the per-level records packed as
    `predict` packs them (`doc_parallel_boosting.mojo:2392-2437`): offset
    `column * n_rows`, `mask << shift`, `bin_idx << shift`, and the
    predicate off the model. Leaf values are BIN-major
    `[leaf * dim + d]`; the cursor is PLANE-major `[d * n_rows + r]`."""
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
        # the per-level records, `h_off`/`h_shift`/`h_mask`/`h_bin`/`h_eq`
        var offs = List[Int](capacity=depth)
        var masks = List[UInt32](capacity=depth)
        var values = List[UInt32](capacity=depth)
        var eqs = List[Bool](capacity=depth)
        for level in range(depth):
            var fid = _check_split_feature(
                layout, split_feature_p[lo + level], split_take_bin_p[lo + level],
                n_features, "tree " + String(t) + " level " + String(level),
            )
            ref cf = layout.features[fid]
            var bin_idx = Int(split_bin_p[lo + level])
            if bin_idx < 0:
                raise Error("gbdt host: tree " + String(t) + " level " + String(level) + " has a negative bin")
            offs.append(Int(cf.offset) * n_rows)
            masks.append(cf.mask << cf.shift)
            values.append(UInt32(bin_idx) << cf.shift)
            eqs.append(split_take_bin_p[lo + level] != 0)
        for r in range(n_rows):
            var leaf = 0
            for level in range(depth):
                var feature_val = cindex[offs[level] + r] & masks[level]
                var split: Bool
                if eqs[level]:
                    split = feature_val == values[level]
                else:
                    split = feature_val > values[level]
                if split:
                    leaf += 1 << level
            for d in range(dim):
                var at = d * n_rows + r
                cursor[at] = cursor[at] + leaves_p[leaf_lo + leaf * dim + d]


def _apply_non_symmetric(
    layout: CompressedIndexLayout,
    cindex: List[UInt32],
    n_rows: Int,
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
    leaves_p: MutPointer[Float32, MutUntrackedOrigin],
    n_nodes_total: Int,
    n_leaf_values: Int,
    mut cursor: List[Float32],
) raises:
    """MIRRORS, per tree in tree order, the node table of
    `compute_non_symmetric_bins_for_model`
    (`add_non_symmetric_tree_doc_parallel.mojo:104-141`), the walk of
    `compute_non_symmetric_decision_tree_bins_kernel`
    (`add_bin_values.mojo:277-320`, shift THEN mask, as written there) and
    the add of `add_bin_model_value_kernel`
    (`kernel_add_model_value.mojo:171-184`, `bin_values[bin * dim + d]`
    into `cursor[d * n_rows + r]`). A tree with no nodes is the constant
    tree: every row lands in bin 0. The walk is bounds-checked, which the
    kernel is not; a file whose subtree sizes point past the tree is
    refused instead of read past."""
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
        var offs = List[Int](capacity=n_nodes)
        var masks = List[UInt32](capacity=n_nodes)
        var shifts = List[UInt32](capacity=n_nodes)
        var bins = List[UInt32](capacity=n_nodes)
        var eqs = List[Bool](capacity=n_nodes)
        var lefts = List[Int](capacity=n_nodes)
        var rights = List[Int](capacity=n_nodes)
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
            offs.append(Int(cf.offset) * n_rows)
            masks.append(cf.mask)
            shifts.append(cf.shift)
            bins.append(UInt32(bin))
            eqs.append(node_take_bin_p[lo + i] != 0)
            lefts.append(left)
            rights.append(right)
        for r in range(n_rows):
            var bin = 0
            var node = 0
            var stop = n_nodes == 0
            while not stop:
                if node >= n_nodes:
                    raise Error(
                        "gbdt host: non-symmetric tree " + String(t)
                        + " walks to node " + String(node) + " of " + String(n_nodes)
                    )
                var feature_val = (cindex[offs[node] + r] >> shifts[node]) & masks[node]
                var split: Bool
                if eqs[node]:
                    split = feature_val == bins[node]
                else:
                    split = feature_val > bins[node]
                if split:
                    bin += lefts[node]
                    stop = rights[node] == 1
                    if not stop:
                        node += lefts[node]
                else:
                    stop = lefts[node] == 1
                    if not stop:
                        node += 1
            if bin >= n_bins:
                raise Error(
                    "gbdt host: non-symmetric tree " + String(t) + " sends a row to bin "
                    + String(bin) + " of " + String(n_bins)
                )
            for d in range(dim):
                var at = d * n_rows + r
                cursor[at] = cursor[at] + leaves_p[leaf_lo + bin * dim + d]


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
) raises:
    """`predict_floats` / `predict_multi_floats` on the host: RAW approxes,
    `out[r * dim + d]`, ROW-major, `n_rows * dim` values.

    `x_colmajor` is COLUMN-major, `n_rows * n_features`, the layout the GPU
    binding takes (`gbdt_predict`, `bindings/_mojolearn_gbdt.mojo:410`).
    The split arrays hold `n_splits` records, one per oblivious level or
    one per non-symmetric node, `tree_offsets` their prefix scan;
    `leaf_offsets` is the prefix scan of leaf VALUE counts (leaves times
    `dim`). `fold_counts`, `one_hot` and `nan` are one per feature.
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
    var cindex = _binarize(
        x_colmajor, n_rows, n_features, border_offsets_p, borders_p, nan_p, layout
    )
    var cursor = List[Float32](length=dim * n_rows, fill=_seed(bias))
    if non_symmetric:
        _apply_non_symmetric(
            layout, cindex, n_rows, n_features, dim, n_trees,
            tree_offsets_p, split_feature_p, split_bin_p, split_take_bin_p,
            node_left_p, node_right_p, leaf_offsets_p, leaves_p,
            n_splits, n_leaf_values, cursor,
        )
    else:
        _apply_oblivious(
            layout, cindex, n_rows, n_features, dim, n_trees,
            tree_offsets_p, split_feature_p, split_bin_p, split_take_bin_p,
            leaf_offsets_p, leaves_p, n_splits, n_leaf_values, cursor,
        )
    # `predict_multi_floats`'s read-back, `[row * approx_dim + dim]`
    # (`gbdt/train.mojo:2470-2474`); for `dim == 1` it is `predict_floats`'s.
    for r in range(n_rows):
        for d in range(dim):
            out[r * dim + d] = cursor[d * n_rows + r]
