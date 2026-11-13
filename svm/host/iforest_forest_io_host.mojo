# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host twin of `isolation_forest/forest_io.mojo` (lane
fix-iforest-save-device-forest, 2026-11-13): the CPU column's resident
oracle forest out to the compact host form `save()` and pickle write, and a
saved compact forest back into an oracle forest the host scorer walks.

The oracle tree stores the same four node arrays the device walks
(`feat`, `thr`, `left`, `right`, tree-local ids, `feat < 0` a leaf whose
`thr` is its path length) and `oracle_path_lengths` folds the trees in the
same ascending order with the same `ftz` seams, so a forest saved on any
column scores the same bits on every column. Typed Mojo only (RUN)."""

from std.memory import unsafe_memcpy
from core.host_parallel import host_parallelize
from isolation_forest.checks.if_oracle import OracleForest, OracleTree
from isolation_forest.impl.isolation_forest import IF_params, compute_c_normalization
from isolation_forest.forest_io import IFForestWords, if_forest_check
from svm.host.svm_host_impl import IForestHostFitted


def if_host_forest_sizes(fitted: IForestHostFitted) -> Tuple[Int, Int]:
    """(trees, used nodes over all trees) of a fitted host forest."""
    var total = 0
    for t in range(len(fitted.forest.trees)):  # driver-loop: over trees, one word each
        total += fitted.forest.trees[t].n_nodes
    return (len(fitted.forest.trees), total)


def if_host_forest_export(fitted: IForestHostFitted, words: IFForestWords) raises:
    """Write the compact forest at `words` (sized by `if_host_forest_sizes`):
    the offsets, then each tree's used nodes, one task per tree."""
    var sizes = if_host_forest_sizes(fitted)
    var n_trees = sizes[0]
    if words.n_trees != n_trees or words.total != sizes[1]:
        raise Error("IsolationForest export: the destination holds " + String(words.n_trees) + " trees and "
                    + String(words.total) + " nodes, the forest " + String(n_trees) + " and " + String(sizes[1]))
    if sizes[1] > 2147483647:
        raise Error("IsolationForest export: node offsets exceed int range")
    var prefix = List[Int](length=n_trees + 1, fill=0)
    for t in range(n_trees):  # driver-loop: over trees, one prefix word each
        var n = fitted.forest.trees[t].n_nodes
        if (n < 1 or n > len(fitted.forest.trees[t].feat) or n > len(fitted.forest.trees[t].thr)
                or n > len(fitted.forest.trees[t].left) or n > len(fitted.forest.trees[t].right)):
            raise Error("IsolationForest export: tree " + String(t) + " lies outside its node storage")
        prefix[t + 1] = prefix[t] + n
    var od = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=words.offsets_addr)
    for t in range(n_trees + 1):  # driver-loop: over trees, one offset word each
        od[t] = Int32(prefix[t])
    var tp = rebind[MutPointer[OracleTree, MutUntrackedOrigin]](fitted.forest.trees.unsafe_ptr())
    var pp = rebind[MutPointer[Int, MutUntrackedOrigin]](prefix.unsafe_ptr())
    var fd = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=words.feature_addr)
    var td = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=words.threshold_addr)
    var ld = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=words.left_addr)
    var rd = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=words.right_addr)

    def task(t: Int) {imm tp, imm pp, imm fd, imm td, imm ld, imm rd}:
        var dst = pp[t]
        var n = pp[t + 1] - dst
        unsafe_memcpy(dest=fd + dst, src=tp[t].feat.unsafe_ptr(), count=n)
        unsafe_memcpy(dest=td + dst, src=tp[t].thr.unsafe_ptr(), count=n)
        unsafe_memcpy(dest=ld + dst, src=tp[t].left.unsafe_ptr(), count=n)
        unsafe_memcpy(dest=rd + dst, src=tp[t].right.unsafe_ptr(), count=n)

    if n_trees == 1:
        task(0)
    else:
        host_parallelize(task, n_trees)
    _ = prefix^  # keep the offsets alive past the parallel copies above


def if_host_forest_import(
    words: IFForestWords, n_features: Int, max_samples_: Int, offset_: Float64,
) raises -> IForestHostFitted:
    """A fitted host forest whose trees are the compact forest at `words`
    (`if_forest_check`'s refusals first). `c_normalization` is the fit's
    `compute_c_normalization(max_samples_)`; the per-tree sample and
    leaf-statistics lists stay empty (the scorer reads none of them)."""
    if n_features < 1:
        raise Error("IsolationForest forest: n_features must be at least one")
    if max_samples_ < 1:
        raise Error("IsolationForest forest: max_samples_ must be at least one")
    var counts = if_forest_check(words, n_features)
    var n_trees = words.n_trees
    var forest = OracleForest(IF_params(n_trees, max_samples_, -1, n_features, False, 0))
    forest.n_features = n_features
    forest.n_features_per_tree = n_features
    forest.n_samples_per_tree = max_samples_
    forest.c_normalization = compute_c_normalization(max_samples_)
    var max_nodes = 0
    for t in range(n_trees):  # driver-loop: over trees, storage of each tree (filled in parallel below)
        var n = Int(counts[t])
        var tree = OracleTree()
        tree.feat = List[Int32](length=n, fill=0)
        tree.thr = List[Float32](length=n, fill=0)
        tree.left = List[Int32](length=n, fill=0)
        tree.right = List[Int32](length=n, fill=0)
        tree.n_nodes = n
        forest.trees.append(tree^)
        if n > max_nodes:
            max_nodes = n
    forest.max_nodes_per_tree = max_nodes
    var tp = rebind[MutPointer[OracleTree, MutUntrackedOrigin]](forest.trees.unsafe_ptr())
    var os = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=words.offsets_addr)
    var fs = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=words.feature_addr)
    var ts = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=words.threshold_addr)
    var ls = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=words.left_addr)
    var rs = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=words.right_addr)

    def task(t: Int) {imm tp, imm os, imm fs, imm ts, imm ls, imm rs}:
        var src = Int(os[t])
        var n = Int(os[t + 1]) - src
        unsafe_memcpy(dest=tp[t].feat.unsafe_ptr(), src=fs + src, count=n)
        unsafe_memcpy(dest=tp[t].thr.unsafe_ptr(), src=ts + src, count=n)
        unsafe_memcpy(dest=tp[t].left.unsafe_ptr(), src=ls + src, count=n)
        unsafe_memcpy(dest=tp[t].right.unsafe_ptr(), src=rs + src, count=n)

    if n_trees == 1:
        task(0)
    else:
        host_parallelize(task, n_trees)
    _ = counts^
    return IForestHostFitted(forest^, offset_, max_samples_)
