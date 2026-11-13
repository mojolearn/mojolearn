# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The fitted forest out of and back into a resident entry (lane
fix-iforest-save-device-forest, 2026-11-13).

Why: since restore-iforest-kde a fit on a resident binary borrows a float32
C-order X (no 1.8 GB copy), so the estimator's `_x` is the caller's buffer.
`save()` and pickle wrote `_x`, and a loaded or unpickled model refit its
forest from it, so a caller who edited X in place after fit got a saved
model of the EDITED data. They now write the forest the scores already use
(this module's export, from the device) and a loaded model scores that
forest (this module's import), with no training matrix at all.

The host form is compact: `offsets` (n_trees + 1 int32, offsets[t] is tree
t's first node), then `feature` (int32, < 0 marks a leaf), `threshold`
(float32; a leaf's stored path length), `left` and `right` (int32, tree-local
node ids), `offsets[n_trees]` nodes each. It is exactly the four node arrays
`compute_path_lengths_global_kernel` walks, with each tree's USED nodes
(`tree_n_nodes_host`) and none of the unused capacity, so the import scores
with the same kernels in the same tree order: the same bits. The host oracle
forest (`isolation_forest/checks/if_oracle.mojo`, the CPU column) has the
same four arrays per tree (`svm/host/iforest_forest_io_host.mojo`), so a file
saved on one column loads on every other.

Typed Mojo only, no Python import (RUN)."""

from std.memory import unsafe_memcpy
from max.gpu.host import DeviceBuffer, DeviceContext
from core.host_parallel import host_parallelize
from isolation_forest.impl.isolation_forest import (
    IF_C50_PACKED,
    IF_params,
    IsolationForestModel,
    compute_c_normalization,
    if_pack_nodes_kernel,
    read_f32,
    read_i32,
)
from isolation_forest.estimator import IsolationForestEstimator, IFResidentKey

#: `IFResidentKey.max_samples_mode` of an imported forest: no fitted key
#: carries it (fitted modes are 0, 1 and 2), so a token minted for an
#: imported forest never matches a fitted forest's key or the reverse.
comptime IF_IMPORTED_KEY_MODE = -7


@fieldwise_init
struct IFForestWords(Copyable, Movable):
    """The five host arrays of a compact forest, by address, with their
    lengths: `offsets` holds `n_trees + 1` int32, the four node arrays
    `total` words each."""

    var offsets_addr: Int
    var feature_addr: Int
    var threshold_addr: Int
    var left_addr: Int
    var right_addr: Int
    var n_trees: Int
    var total: Int


def if_imported_key(words: IFForestWords, n_features: Int, max_samples_: Int, offset_: Float64) -> IFResidentKey:
    """The resident-store key of an imported forest: its node storage
    address and size, the width, the resolved subsample size and the
    offset, under `IF_IMPORTED_KEY_MODE`."""
    return IFResidentKey(
        words.feature_addr, words.total, n_features, words.n_trees, IF_IMPORTED_KEY_MODE,
        max_samples_, offset_, -1, 0, 0, 0.0, False, 0, True, 0.0,
    )


def if_forest_sizes(model: IsolationForestModel) -> Tuple[Int, Int]:
    """(trees, used nodes over all trees) of a fitted model."""
    var total = 0
    for t in range(len(model.tree_n_nodes_host)):  # driver-loop: over trees, one word each
        total += Int(model.tree_n_nodes_host[t])
    return (len(model.tree_n_nodes_host), total)


def _export_part(
    ctx: DeviceContext,
    feature: DeviceBuffer[DType.int32],
    threshold: DeviceBuffer[DType.float32],
    left: DeviceBuffer[DType.int32],
    right: DeviceBuffer[DType.int32],
    tree_offsets: DeviceBuffer[DType.int32],
    first: Int,
    count: Int,
    prefix: List[Int],
    words: IFForestWords,
) raises:
    """Trees `first .. first + count` of one owner (the model, or one tree
    shard on its own device): the node arrays come down once each, then
    each tree's used nodes are copied to their compact place, one task per
    tree across the host cores."""
    if count <= 0:
        return
    var dev_off = read_i32(ctx, tree_offsets, count)
    var h_feat = read_i32(ctx, feature, len(feature))
    var h_thr = read_f32(ctx, threshold, len(threshold))
    var h_left = read_i32(ctx, left, len(left))
    var h_right = read_i32(ctx, right, len(right))
    var capacity = len(h_feat)
    for t in range(count):  # driver-loop: over trees, bounds of each tree's stored nodes
        var n = prefix[first + t + 1] - prefix[first + t]
        var src = Int(dev_off[t])
        if src < 0 or src + n > capacity or src + n > len(h_thr) or src + n > len(h_left) or src + n > len(h_right):
            raise Error("IsolationForest export: tree " + String(first + t) + " lies outside its node storage")
    var op = rebind[MutPointer[Int32, MutUntrackedOrigin]](dev_off.unsafe_ptr())
    var pp = rebind[MutPointer[Int, MutUntrackedOrigin]](prefix.unsafe_ptr())
    var fs = rebind[MutPointer[Int32, MutUntrackedOrigin]](h_feat.unsafe_ptr())
    var ts = rebind[MutPointer[Float32, MutUntrackedOrigin]](h_thr.unsafe_ptr())
    var ls = rebind[MutPointer[Int32, MutUntrackedOrigin]](h_left.unsafe_ptr())
    var rs = rebind[MutPointer[Int32, MutUntrackedOrigin]](h_right.unsafe_ptr())
    var fd = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=words.feature_addr)
    var td = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=words.threshold_addr)
    var ld = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=words.left_addr)
    var rd = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=words.right_addr)

    def task(t: Int) {imm op, imm pp, imm fs, imm ts, imm ls, imm rs, imm fd, imm td, imm ld, imm rd, imm first}:
        var dst = pp[first + t]
        var n = pp[first + t + 1] - dst
        var src = Int(op[t])
        unsafe_memcpy(dest=fd + dst, src=fs + src, count=n)
        unsafe_memcpy(dest=td + dst, src=ts + src, count=n)
        unsafe_memcpy(dest=ld + dst, src=ls + src, count=n)
        unsafe_memcpy(dest=rd + dst, src=rs + src, count=n)

    if count == 1:
        task(0)
    else:
        host_parallelize(task, count)
    _ = dev_off^  # keep the host copies alive past the parallel copies above
    _ = h_feat^
    _ = h_thr^
    _ = h_left^
    _ = h_right^


def if_forest_export(ctx: DeviceContext, model: IsolationForestModel, words: IFForestWords) raises:
    """Write `model`'s compact forest at `words` (sized by `if_forest_sizes`):
    the offsets, then every tree's used nodes, read from the device (each
    tree shard from its own device)."""
    if not model.fitted:
        raise Error("IsolationForest export: the forest is not fitted")
    var sizes = if_forest_sizes(model)
    var n_trees = sizes[0]
    if words.n_trees != n_trees or words.total != sizes[1]:
        raise Error("IsolationForest export: the destination holds " + String(words.n_trees) + " trees and "
                    + String(words.total) + " nodes, the forest " + String(n_trees) + " and " + String(sizes[1]))
    if sizes[1] > 2147483647:
        raise Error("IsolationForest export: node offsets exceed int range")
    var prefix = List[Int](length=n_trees + 1, fill=0)
    for t in range(n_trees):  # driver-loop: over trees, one prefix word each
        prefix[t + 1] = prefix[t] + Int(model.tree_n_nodes_host[t])
    var od = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=words.offsets_addr)
    for t in range(n_trees + 1):  # driver-loop: over trees, one offset word each
        od[t] = Int32(prefix[t])
    if len(model.shards) == 0:
        _export_part(ctx, model.node_feature, model.node_threshold, model.node_left, model.node_right,
                     model.global_tree_offsets, 0, n_trees, prefix, words)
    else:
        for s in range(len(model.shards)):  # driver-loop: over tree-shard owners (devices)
            ref shard = model.shards[s]
            _export_part(shard.ctx, shard.node_feature, shard.node_threshold, shard.node_left, shard.node_right,
                         shard.global_tree_offsets, shard.first, shard.count, prefix, words)
    _ = prefix^


def if_forest_check(words: IFForestWords, n_features: Int) raises -> List[Int32]:
    """Refuse a compact forest the traversal could not walk safely: offsets
    that do not start at 0, cover every tree with at least one node and end
    at `total`; a split feature at or past `n_features`; a child that is not
    a later node of the same tree (the builders number children after their
    parent, so every walk ends at a leaf). Returns each tree's node count.
    The node checks run one task per tree across the host cores."""
    var n_trees = words.n_trees
    if n_trees < 1:
        raise Error("IsolationForest forest: it holds no tree")
    if words.total < n_trees or words.total > 2147483647:
        raise Error("IsolationForest forest: " + String(words.total) + " nodes for " + String(n_trees) + " trees")
    var offsets = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=words.offsets_addr)
    if Int(offsets[0]) != 0 or Int(offsets[n_trees]) != words.total:
        raise Error("IsolationForest forest: the tree offsets do not span the nodes")
    var counts = List[Int32](length=n_trees, fill=0)
    for t in range(n_trees):  # driver-loop: over trees, one offset pair each
        var n = Int(offsets[t + 1]) - Int(offsets[t])
        if n < 1:
            raise Error("IsolationForest forest: tree " + String(t) + " has no node")
        counts[t] = Int32(n)
    var failures = List[Int](length=n_trees, fill=0)
    var fp = rebind[MutPointer[Int, MutUntrackedOrigin]](failures.unsafe_ptr())
    var fe = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=words.feature_addr)
    var le = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=words.left_addr)
    var ri = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=words.right_addr)

    def task(t: Int) {imm fp, imm fe, imm le, imm ri, imm offsets, imm n_features}:
        var base = Int(offsets[t])
        var n = Int(offsets[t + 1]) - base
        for i in range(n):
            var f = Int(fe[base + i])
            if f < 0:
                continue
            var l = Int(le[base + i])
            var r = Int(ri[base + i])
            if f >= n_features or l <= i or l >= n or r <= i or r >= n:
                fp[t] = 1
                return

    if n_trees == 1:
        task(0)
    else:
        host_parallelize(task, n_trees)
    for t in range(n_trees):  # driver-loop: over trees, one failure word each
        if failures[t] != 0:
            raise Error("IsolationForest forest: tree " + String(t) + " has a split outside its features or nodes")
    _ = failures^
    return counts^


def _upload_i32_words(ctx: DeviceContext, addr: Int, n: Int) raises -> DeviceBuffer[DType.int32]:
    var buf = ctx.enqueue_create_buffer[DType.int32](n)
    var host = ctx.enqueue_create_host_buffer[DType.int32](n)
    unsafe_memcpy(dest=host.unsafe_ptr(), src=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr), count=n)
    ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return buf^


def _upload_f32_words(ctx: DeviceContext, addr: Int, n: Int) raises -> DeviceBuffer[DType.float32]:
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    unsafe_memcpy(dest=host.unsafe_ptr(), src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr), count=n)
    ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return buf^


def if_forest_import(
    ctx: DeviceContext, words: IFForestWords, n_features: Int, max_samples_: Int, offset_: Float64,
) raises -> IsolationForestEstimator:
    """A fitted estimator whose model is the compact forest at `words`,
    uploaded to `ctx` (one owner, no shards: the sharded scorer continues
    the same serial tree fold, so one owner scores the same bits). The
    scalars are the saved fit's: `c_normalization` is the fit's
    `compute_c_normalization(max_samples_)`."""
    if n_features < 1:
        raise Error("IsolationForest forest: n_features must be at least one")
    if max_samples_ < 1:
        raise Error("IsolationForest forest: max_samples_ must be at least one")
    var counts = if_forest_check(words, n_features)
    var n_trees = words.n_trees
    var total = words.total
    var est = IsolationForestEstimator(ctx)
    var model = IsolationForestModel(ctx)
    var max_nodes = 0
    for t in range(n_trees):  # driver-loop: over trees, one word each
        if Int(counts[t]) > max_nodes:
            max_nodes = Int(counts[t])
    model.params = IF_params(n_trees, max_samples_, -1, n_features, False, 0)
    model.n_features = n_features
    model.n_features_per_tree = n_features
    model.n_samples_per_tree = max_samples_
    model.c_normalization = compute_c_normalization(max_samples_)
    model.max_nodes_per_tree = max_nodes
    model.has_feature_indices = False
    model.node_feature = _upload_i32_words(ctx, words.feature_addr, total)
    model.node_threshold = _upload_f32_words(ctx, words.threshold_addr, total)
    model.node_left = _upload_i32_words(ctx, words.left_addr, total)
    model.node_right = _upload_i32_words(ctx, words.right_addr, total)
    # The first n_trees offsets are each tree's start (the kernel reads
    # `tree_offsets[t]`); the node counts are the offsets' differences.
    model.global_tree_offsets = _upload_i32_words(ctx, words.offsets_addr, n_trees)
    model.global_tree_n_nodes = _upload_i32_words(ctx, Int(counts.unsafe_ptr()), n_trees)
    var depths = List[Int32](length=n_trees, fill=0)
    model.global_tree_max_depth = _upload_i32_words(ctx, Int(depths.unsafe_ptr()), n_trees)
    comptime if IF_C50_PACKED:
        model.packed_nodes = ctx.enqueue_create_buffer[DType.int32](4 * total)
        ctx.enqueue_function[if_pack_nodes_kernel](model.node_feature.unsafe_ptr(), model.node_threshold.unsafe_ptr(),
            model.node_left.unsafe_ptr(), model.node_right.unsafe_ptr(), model.packed_nodes.value().unsafe_ptr(),
            Int32(total), grid_dim=(total + 255) // 256, block_dim=256)
        ctx.synchronize()
    model.tree_n_nodes_host = counts^
    model.tree_max_depth_host = depths^
    model.fitted = True
    est.n_estimators = n_trees
    est.offset_ = offset_
    est.max_samples_ = max_samples_
    est.n_features_in_ = n_features
    est.model = model^
    est.fitted = True
    return est^
