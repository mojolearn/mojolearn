# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host oracle: a SECOND, independent transcription of cuML's
isolation tree builder and scorer, serial Float32 through the same
`identical_*` helpers and the same XORWOW port, plus a Float64 score
reference. NOT A PORT (cuML has no host arm; its CPU story is refusing).

Why a second transcription rather than calling the device function on
the host: the device builder's pointer arithmetic (tree offsets, the
four node arrays, the flat stack) is exactly what a mis-port hides in,
and a gate that runs the same function twice cannot see it. This file
is written from `isolation_tree_builder.cuh` again, recursively, over
`List`s, with the ONE thing that must be shared shared: the RNG
(`derived/curand/curand_kernel.mojo`) and its consumption order (rows,
then features, then per node `sample_bounded(n_cols)` + `curand_uniform`
in pre-order, left child first -- which is what their explicit stack,
pushing right then left, walks).

The partition reproduces their IN-PLACE SWAP ORDER (`:200-208`), because
the only order-dependent thing in the whole build -- which of two zeros
a node's strict `<` fold calls its min (ADDENDUM 11) -- depends on the
order rows reach the next level. A structurally equal tree with rows in
another order is not bit-identical to theirs in that branch, so the
oracle must not take the easy stable partition.

The Float64 reference (`oracle_scores_f64`) walks the SAME trees (the
structure is integer + one float per node and is the certified object)
and recomputes c(n) and 2^(-E[h]/c) in double, so a float32 score that
drifts from its own double is caught before a device is asked anything.
"""

from std.math import log, exp2
from std.memory import bitcast

from isolation_forest.derived.curand.curand_kernel import (
    XorwowTables,
    curand,
    curand_init,
    curand_uniform,
    curandStateXORWOW,
)
from isolation_forest.derived.isolation_forest.isolation_forest import (
    IF_params,
    ceil_log2_int,
    compute_c_normalization,
    compute_global_max_nodes_per_tree,
)
from isolation_forest.derived.isolation_forest.isolation_tree_builder import (
    EULER_MASCHERONI_F32,
)
from original.numerics import ftz, identical_log, identical_mul_add, identical_pow


struct OracleTree(Movable):
    var rows: List[Int64]
    """The subsample's source rows (their `tree_sample_indices`)."""
    var features: List[Int32]
    """Sampled feature ids when `max_features < n_cols`, else empty."""
    var local: List[Float32]
    """The gathered subsample, row-major `max_samples x max_features`."""
    var feat: List[Int32]
    var thr: List[Float32]
    var left: List[Int32]
    var right: List[Int32]
    var n_nodes: Int
    var max_depth: Int
    var leaf_depth: List[Int32]
    """Per node: the depth of a leaf (for the Float64 reference), -1 for
    an internal node."""
    var leaf_count: List[Int32]
    """Per node: `n_node_samples` at a leaf, -1 for an internal node."""

    def __init__(out self):
        self.rows = List[Int64]()
        self.features = List[Int32]()
        self.local = List[Float32]()
        self.feat = List[Int32]()
        self.thr = List[Float32]()
        self.left = List[Int32]()
        self.right = List[Int32]()
        self.n_nodes = 0
        self.max_depth = 0
        self.leaf_depth = List[Int32]()
        self.leaf_count = List[Int32]()


struct OracleForest(Movable):
    var params: IF_params
    var n_features: Int
    var n_features_per_tree: Int
    var n_samples_per_tree: Int
    var max_depth: Int
    var max_nodes_per_tree: Int
    var c_normalization: Float64
    var trees: List[OracleTree]

    def __init__(out self, params: IF_params):
        self.params = params.copy()
        self.n_features = 0
        self.n_features_per_tree = 0
        self.n_samples_per_tree = 0
        self.max_depth = 0
        self.max_nodes_per_tree = 0
        self.c_normalization = 0.0
        self.trees = List[OracleTree]()


def oracle_c_n(n_samples: Int) -> Float32:
    """`compute_c_n<float>` again, written out."""
    if n_samples <= 1:
        return Float32(0.0)
    if n_samples == 2:
        return Float32(1.0)
    var n = Float32(n_samples)
    var h = ftz(identical_log(n - Float32(1.0)) + EULER_MASCHERONI_F32)
    var tail = ftz(Float32(2.0) * (n - Float32(1.0)) / n)
    return ftz(Float32(2.0) * h - tail)


def _u64(mut st: curandStateXORWOW) -> UInt64:
    var hi = UInt64(curand(st))
    var lo = UInt64(curand(st))
    return (hi << 32) | lo


def _bounded(mut st: curandStateXORWOW, bound: UInt64) -> UInt64:
    if bound <= 1:
        return 0
    var m: UInt64 = 0xFFFFFFFFFFFFFFFF
    var limit = m - (m % bound)
    var v = _u64(st)
    while v >= limit:
        v = _u64(st)
    return v % bound


def _set_node(mut t: OracleTree, idx: Int, f: Int32, thr: Float32, l: Int32, r: Int32, ld: Int32, lc: Int32):
    while len(t.feat) <= idx:
        t.feat.append(Int32(-2))
        t.thr.append(Float32(0.0))
        t.left.append(Int32(-2))
        t.right.append(Int32(-2))
        t.leaf_depth.append(Int32(-1))
        t.leaf_count.append(Int32(-1))
    t.feat[idx] = f
    t.thr[idx] = thr
    t.left[idx] = l
    t.right[idx] = r
    t.leaf_depth[idx] = ld
    t.leaf_count[idx] = lc


def _build_node(
    mut t: OracleTree,
    mut st: curandStateXORWOW,
    mut idx: List[Int],
    start: Int,
    end: Int,
    node_idx: Int,
    depth: Int,
    n_cols: Int,
    max_depth: Int,
    max_nodes_per_tree: Int,
    mut n_nodes: Int,
    mut observed_max_depth: Int,
):
    """One node of `build_tree_iterative_global` (`:144-239`), recursive
    pre-order left first. `idx` is the tree's work_indices."""
    var n_node_samples = end - start
    if depth > observed_max_depth:
        observed_max_depth = depth
    if node_idx >= max_nodes_per_tree:
        return
    if depth >= max_depth or n_node_samples <= 1 or n_nodes + 2 > max_nodes_per_tree:
        var pl = ftz(Float32(depth) + oracle_c_n(n_node_samples))
        _set_node(t, node_idx, -1, pl, -1, -1, Int32(depth), Int32(n_node_samples))
        return

    var local_feature = -1
    var min_val = Float32(0.0)
    var max_val = Float32(0.0)
    var feature_start = Int(_bounded(st, UInt64(n_cols)))
    for attempt in range(n_cols):
        var candidate = (feature_start + attempt) % n_cols
        var cmin = t.local[idx[start] * n_cols + candidate]
        var cmax = cmin
        for r in range(start + 1, end):
            var v = t.local[idx[r] * n_cols + candidate]
            if v < cmin:
                cmin = v
            if v > cmax:
                cmax = v
        if cmin < cmax:
            local_feature = candidate
            min_val = cmin
            max_val = cmax
            break

    if local_feature < 0:
        var pl = ftz(Float32(depth) + oracle_c_n(n_node_samples))
        _set_node(t, node_idx, -1, pl, -1, -1, Int32(depth), Int32(n_node_samples))
        return

    var original_feature = Int32(local_feature)
    if len(t.features) > 0:
        original_feature = t.features[local_feature]
    var frac = curand_uniform(st)
    var threshold = ftz(identical_mul_add(frac, ftz(max_val - min_val), min_val))

    var left_end = start
    for r in range(start, end):
        var v = t.local[idx[r] * n_cols + local_feature]
        if v < threshold:
            var tmp = idx[left_end]
            idx[left_end] = idx[r]
            idx[r] = tmp
            left_end += 1
    if left_end == start or left_end == end:
        threshold = max_val
        left_end = start
        for r in range(start, end):
            var v = t.local[idx[r] * n_cols + local_feature]
            if v < threshold:
                var tmp = idx[left_end]
                idx[left_end] = idx[r]
                idx[r] = tmp
                left_end += 1

    var left_child = n_nodes
    var right_child = n_nodes + 1
    n_nodes += 2
    _set_node(t, node_idx, original_feature, threshold, Int32(left_child), Int32(right_child), -1, -1)
    # Their stack pushes right then left and pops left first: pre-order,
    # left subtree completely before the right one.
    _build_node(t, st, idx, start, left_end, left_child, depth + 1, n_cols, max_depth, max_nodes_per_tree, n_nodes, observed_max_depth)
    _build_node(t, st, idx, left_end, end, right_child, depth + 1, n_cols, max_depth, max_nodes_per_tree, n_nodes, observed_max_depth)


def oracle_fit(
    x_rowmajor: List[Float32],
    n_rows: Int,
    n_cols: Int,
    params: IF_params,
    tables: XorwowTables,
) raises -> OracleForest:
    """`IsolationForest::fit` + `build_isolation_trees_global_kernel`, on
    the host, tree by tree in `tree_id` order. `x_rowmajor` is the
    natural host layout; the gather indexes it as `data[row*n_cols+col]`
    where the device reads column-major `data[row + col*n_rows]` -- the
    same cell."""
    var f = OracleForest(params)
    if n_rows <= 0 or n_cols <= 0 or params.n_estimators <= 0:
        raise Error("oracle_fit: bad shape or n_estimators")
    var n_sampled_rows = params.max_samples if params.max_samples < n_rows else n_rows
    var max_depth = params.max_depth
    if max_depth <= 0:
        max_depth = ceil_log2_int(n_sampled_rows)
        if max_depth < 1:
            max_depth = 1
    var n_sampled_features = params.max_features
    if n_sampled_features <= 0 or n_sampled_features > n_cols:
        n_sampled_features = n_cols
    f.n_features = n_cols
    f.n_features_per_tree = n_sampled_features
    f.n_samples_per_tree = n_sampled_rows
    f.max_depth = max_depth
    f.max_nodes_per_tree = compute_global_max_nodes_per_tree(max_depth, n_sampled_rows)
    f.c_normalization = compute_c_normalization(n_sampled_rows)
    var seqp = tables.sequence.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()
    var offp = tables.offset.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()

    for tree_id in range(params.n_estimators):
        var t = OracleTree()
        var st = curandStateXORWOW.zero()
        curand_init(params.seed, UInt64(tree_id), UInt64(0), st, seqp, offp)
        # rows
        if params.bootstrap:
            for _ in range(n_sampled_rows):
                t.rows.append(Int64(_bounded(st, UInt64(n_rows))))
        else:
            var start = n_rows - n_sampled_rows
            for i in range(n_sampled_rows):
                var j = start + i
                var cand = Int64(_bounded(st, UInt64(j + 1)))
                var seen = False
                for q in range(i):
                    if t.rows[q] == cand:
                        seen = True
                t.rows.append(Int64(j) if seen else cand)
        # features
        if n_sampled_features < n_cols:
            var start = n_cols - n_sampled_features
            for i in range(n_sampled_features):
                var j = start + i
                var cand = Int32(_bounded(st, UInt64(j + 1)))
                var seen = False
                for q in range(i):
                    if t.features[q] == cand:
                        seen = True
                t.features.append(Int32(j) if seen else cand)
        # gather
        for s in range(n_sampled_rows):
            var src_row = Int(t.rows[s])
            for c in range(n_sampled_features):
                var src_col = c
                if len(t.features) > 0:
                    src_col = Int(t.features[c])
                t.local.append(x_rowmajor[src_row * n_cols + src_col])
        # build
        var idx = List[Int]()
        for i in range(n_sampled_rows):
            idx.append(i)
        var n_nodes = 1
        var observed = 0
        _build_node(t, st, idx, 0, n_sampled_rows, 0, 0, n_sampled_features, max_depth, f.max_nodes_per_tree, n_nodes, observed)
        t.n_nodes = n_nodes
        t.max_depth = observed
        f.trees.append(t^)
    return f^


def _traverse(t: OracleTree, x: List[Float32], row: Int, n_cols: Int) -> Int:
    """Returns the leaf's node index."""
    var node = 0
    while True:
        var fe = Int(t.feat[node])
        if fe < 0:
            return node
        var v = x[row * n_cols + fe]
        if v < t.thr[node]:
            node = Int(t.left[node])
        else:
            node = Int(t.right[node])
    return node


def oracle_path_lengths(
    f: OracleForest, x_rowmajor: List[Float32], n_rows: Int, n_cols: Int
) -> List[Float32]:
    """`compute_path_lengths_global_kernel`: ascending tree order, `ftz`
    at the stored seams, `/ n_trees`."""
    var out = List[Float32]()
    var n_trees = len(f.trees)
    for i in range(n_rows):
        var total = Float32(0.0)
        for t in range(n_trees):
            var leaf = _traverse(f.trees[t], x_rowmajor, i, n_cols)
            total = ftz(total + f.trees[t].thr[leaf])
        var pl = Float32(0.0)
        if n_trees > 0:
            pl = ftz(total / Float32(n_trees))
        out.append(pl)
    return out^


def oracle_scores(f: OracleForest, path_lengths: List[Float32]) -> List[Float32]:
    """`compute_anomaly_scores`: `c_n = float(c_normalization)`; `<= 0` ->
    0.5; else `identical_pow(2, -pl / c_n)` (DEVIATION 681)."""
    var out = List[Float32]()
    var c_n = Float32(f.c_normalization)
    for i in range(len(path_lengths)):
        if c_n <= Float32(0.0):
            out.append(Float32(0.5))
        else:
            var y = ftz(-path_lengths[i] / c_n)
            out.append(ftz(identical_pow(Float32(2.0), y)))
    return out^


def oracle_scores_f64(
    f: OracleForest, x_rowmajor: List[Float32], n_rows: Int, n_cols: Int
) -> List[Float64]:
    """The Float64 reference over the same trees: leaf path length =
    depth + c(n_leaf) in double (libm log), mean over trees, 2^(-mean /
    c(n)) with `c(n)` the model's double."""
    var out = List[Float64]()
    var n_trees = len(f.trees)
    var gamma = 0.5772156649015329
    for i in range(n_rows):
        var total = 0.0
        for t in range(n_trees):
            var leaf = _traverse(f.trees[t], x_rowmajor, i, n_cols)
            var d = Float64(Int(f.trees[t].leaf_depth[leaf]))
            var m = Int(f.trees[t].leaf_count[leaf])
            var c = 0.0
            if m == 2:
                c = 1.0
            elif m > 2:
                var mm = Float64(m)
                c = 2.0 * (log(mm - 1.0) + gamma) - 2.0 * (mm - 1.0) / mm
            total += d + c
        var mean = total / Float64(n_trees)
        if f.c_normalization <= 0.0:
            out.append(0.5)
        else:
            out.append(exp2(-mean / f.c_normalization))
    return out^
