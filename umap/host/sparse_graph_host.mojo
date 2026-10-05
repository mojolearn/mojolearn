# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into the CPU host binding through umap_oracle.mojo; product, not only a check.
"""The CSR fuzzy graph on the HOST: the CPU column's builder.

A GPU fit builds the graph on the device
(`umap/sparse_graph.mojo::sparse_fuzzy_simplicial_graph_device`). Both call
the same per-row statements (`ug_row_rho_kern`, `ug_row_sigma`, `ug_member`,
`ug_merge_weight` in `umap/sparse_graph.mojo`): binary64 arithmetic through
`checks/soft_f64.mojo` (integer instructions, correctly rounded), float32
through the pinned `identical_*` seams. So the two columns return the same
words by construction. Rows are independent, so the host threads split rows
and change no statement or order within a row.

Storage is O(n*k). Row-local insertion sorting costs O(n*k*k); transpose
construction and row merges are linear in stored entries. Explicit zero
entries from kNN candidates are retained; consumers must apply their
existing weight policy.
"""

from checks.numerics import ftz
from checks.soft_f64 import sf64_is_nan, sf64_to_f32
from core.host_predict_threads import (
    host_list_ptr,
    host_predict_chunk,
    host_predict_task_count,
)
from core.host_parallel import host_parallelize
from umap.graph import _finite
from umap.sparse_graph import (
    SparseFuzzySimplicialGraph,
    UG_F32P,
    ug_categorical_constants,
    ug_categorical_weight,
    ug_constants,
    ug_intersect_cell,
    ug_intersect_expo,
    ug_intersect_floor,
    ug_member,
    ug_merge_weight,
    ug_reset_scale,
    ug_reset_weight,
    ug_row_rho_kern,
    ug_row_sigma,
)


def sparse_fuzzy_simplicial_graph(
    knn_indices: List[UInt32], knn_distances: List[Float32],
    n_samples: Int, n_neighbors: Int,
    set_op_mix_ratio: Float32 = Float32(1.0),
    local_connectivity: Float32 = Float32(1.0),
) raises -> SparseFuzzySimplicialGraph:
    if n_samples < 2 or n_neighbors < 2 or n_neighbors > n_samples:
        raise Error("invalid UMAP k-NN graph shape")
    # Division avoids overflowing n*k in a malformed shape request.
    if len(knn_indices) != len(knn_distances) or (
        len(knn_indices) // n_samples != n_neighbors
        or len(knn_indices) % n_samples != 0
    ):
        raise Error("UMAP k-NN arrays do not match their shape")
    if not _finite(set_op_mix_ratio):
        raise Error("UMAP set operation mix ratio must be finite")
    if set_op_mix_ratio < Float32(0.0) or set_op_mix_ratio > Float32(1.0):
        raise Error("UMAP set operation mix ratio must be in [0, 1]")
    var rhos = List[Float32]()
    var sigmas = List[Float32]()
    rhos.resize(n_samples, Float32(0.0))
    sigmas.resize(n_samples, Float32(0.0))
    var c = ug_constants(n_neighbors)
    var target = c[0]
    var tol = c[1]
    var big = c[2]
    var dp = rebind[UG_F32P](knn_distances.unsafe_ptr())
    # Validate in caller order so malformed input keeps the same first error.
    for i in range(n_samples):
        if Int(knn_indices[i * n_neighbors]) != i:
            raise Error("UMAP expects self in k-NN slot zero")
        var previous = Float32(-1.0)
        for j in range(n_neighbors):
            var d = knn_distances[i * n_neighbors + j]
            if not _finite(d) or d < Float32(0.0) or (
                j > 0 and d < previous
            ):
                raise Error("UMAP k-NN distances must be finite and sorted")
            previous = d
        rhos[i] = ug_row_rho_kern(dp, i, n_neighbors, local_connectivity, tol)
    var tasks = host_predict_task_count(n_samples)
    # The thread-pool join is not worthwhile for small graph builds.
    if n_samples < 256:
        tasks = 1
    var chunk = host_predict_chunk(n_samples, tasks)
    var rp = host_list_ptr(rhos)
    var sp = host_list_ptr(sigmas)
    var failed = List[Int](length=tasks, fill=0)
    var fp = failed.unsafe_ptr()

    def _sigma_rows(task: Int) {imm dp, imm target, imm tol, imm big, imm n_neighbors, imm n_samples, imm chunk, imm rp, imm sp, imm fp}:
        var lo = task * chunk
        var hi = min(lo + chunk, n_samples)
        for i in range(lo, hi):
            var sigma = ug_row_sigma(
                dp, i, n_neighbors, rp.unsafe_load(i), target, tol, big
            )
            if sf64_is_nan(sigma):
                fp.unsafe_store(task, 1)
                return
            sp.unsafe_store(i, sf64_to_f32(sigma))

    if tasks == 1:
        _sigma_rows(0)
    else:
        host_parallelize(_sigma_rows, tasks)
    for task in range(tasks):
        if failed[task] != 0:
            raise Error("UMAP sigma search did not bracket its target")

    var doff = List[Int]()
    var dcol = List[UInt32]()
    var dval = List[Float32]()
    doff.append(0)
    for i in range(n_samples):
        var row_start = len(dcol)
        # Membership evaluation and duplicate max update stay in original
        # distance-rank order; only the finished row is column sorted.
        for j in range(1, n_neighbors):
            var dst = Int(knn_indices[i * n_neighbors + j])
            if dst < 0 or dst >= n_samples or dst == i:
                raise Error("UMAP k-NN index is invalid or repeats self")
            var value = ug_member(
                knn_distances[i * n_neighbors + j] - rhos[i], sigmas[i]
            )
            var existing = -1
            for at in range(row_start, len(dcol)):
                if Int(dcol[at]) == dst:
                    existing = at
                    break
            if existing >= 0:
                if value > dval[existing]:
                    dval[existing] = value
            else:
                dcol.append(UInt32(dst))
                dval.append(value)
        for at in range(row_start + 1, len(dcol)):
            var col = dcol[at]
            var val = dval[at]
            var pos = at
            while pos > row_start:
                if dcol[pos - 1] < col:
                    break
                dcol[pos] = dcol[pos - 1]
                dval[pos] = dval[pos - 1]
                pos -= 1
            dcol[pos] = col
            dval[pos] = val
        doff.append(len(dcol))

    # Transpose CSR: count, prefix-sum, scatter rows in ascending order.
    # That scatter order makes each transpose row column sorted already.
    var toff = List[Int]()
    toff.resize(n_samples + 1, 0)
    for col in dcol:
        toff[Int(col) + 1] += 1
    for i in range(n_samples):
        toff[i + 1] += toff[i]
    var cursor = toff.copy()
    var tcol = List[UInt32]()
    var tval = List[Float32]()
    tcol.resize(len(dcol), UInt32(0))
    tval.resize(len(dcol), Float32(0.0))
    for i in range(n_samples):
        for at in range(doff[i], doff[i + 1]):
            var col = Int(dcol[at])
            var target_at = cursor[col]
            tcol[target_at] = UInt32(i)
            tval[target_at] = dval[at]
            cursor[col] += 1

    var offsets = List[Int]()
    var indices = List[UInt32]()
    var values = List[Float32]()
    offsets.append(0)
    for i in range(n_samples):
        var left = doff[i]
        var right = toff[i]
        while left < doff[i + 1] or right < toff[i + 1]:
            var lc = n_samples
            var rc = n_samples
            if left < doff[i + 1]:
                lc = Int(dcol[left])
            if right < toff[i + 1]:
                rc = Int(tcol[right])
            var col = min(lc, rc)
            var a = Float32(0.0)
            var b = Float32(0.0)
            if lc == col:
                a = dval[left]
                left += 1
            if rc == col:
                b = tval[right]
                right += 1
            indices.append(UInt32(col))
            values.append(ug_merge_weight(a, b, set_op_mix_ratio))
        offsets.append(len(indices))
    return SparseFuzzySimplicialGraph(
        n_samples, n_neighbors, rhos^, sigmas^, doff^, dcol^, dval^,
        offsets^, indices^, values^
    )


# ---------------------------------------------------------------------------
# Supervised UMAP's set operations, the HOST column (lane cpu4-umap,
# 2026-10-04). The GPU fit runs them on the device
# (`umap/sparse_graph.mojo::categorical_intersection`, `general_intersection`,
# `reset_local_connectivity`); both call the same per-cell statements
# (`ug_reset_scale`, `ug_reset_weight`, `ug_categorical_weight`,
# `ug_intersect_cell`) and host scalars (`ug_categorical_constants`,
# `ug_intersect_floor`, `ug_intersect_expo`), in the same row order.
# ---------------------------------------------------------------------------


def _host_with_csr(
    graph: SparseFuzzySimplicialGraph, var offsets: List[Int], var indices: List[UInt32], var values: List[Float32]
) -> SparseFuzzySimplicialGraph:
    return SparseFuzzySimplicialGraph(
        graph.n_samples, graph.n_neighbors, graph.rhos.copy(), graph.sigmas.copy(),
        graph.directed_offsets.copy(), graph.directed_indices.copy(), graph.directed_values.copy(),
        offsets^, indices^, values^,
    )


def host_reset_local_connectivity(graph: SparseFuzzySimplicialGraph) raises -> SparseFuzzySimplicialGraph:
    """umap-learn `reset_local_connectivity`: every row scaled by its largest
    stored value (`ug_reset_scale`), then `ug_reset_weight` per cell of
    S + S^T in ascending column order, exact zeros eliminated."""
    var n = graph.n_samples
    var nv = List[Float32](capacity=len(graph.values))
    for row in range(n):
        var mx = Float32(0.0)
        for e in range(graph.offsets[row], graph.offsets[row + 1]):
            var v = ftz(graph.values[e])
            if v > mx:
                mx = v
        for e in range(graph.offsets[row], graph.offsets[row + 1]):
            nv.append(ug_reset_scale(graph.values[e], mx))
    # the transpose: rows scattered in ascending order, so each row of it is
    # column sorted
    var toff = List[Int](length=n + 1, fill=0)
    for col in graph.indices:
        toff[Int(col) + 1] += 1
    for i in range(n):
        toff[i + 1] += toff[i]
    var cursor = toff.copy()
    var tcol = List[UInt32](length=len(graph.indices), fill=UInt32(0))
    var tval = List[Float32](length=len(graph.indices), fill=Float32(0.0))
    for i in range(n):
        for at in range(graph.offsets[i], graph.offsets[i + 1]):
            var col = Int(graph.indices[at])
            tcol[cursor[col]] = UInt32(i)
            tval[cursor[col]] = nv[at]
            cursor[col] += 1
    var offsets = List[Int]()
    var indices = List[UInt32]()
    var values = List[Float32]()
    offsets.append(0)
    for i in range(n):
        var left = graph.offsets[i]
        var right = toff[i]
        while left < graph.offsets[i + 1] or right < toff[i + 1]:
            var lc = n
            var rc = n
            if left < graph.offsets[i + 1]:
                lc = Int(graph.indices[left])
            if right < toff[i + 1]:
                rc = Int(tcol[right])
            var col = min(lc, rc)
            var a = Float32(0.0)
            var b = Float32(0.0)
            if lc == col:
                a = nv[left]
                left += 1
            if rc == col:
                b = tval[right]
                right += 1
            var w = ug_reset_weight(a, b)
            if w != Float32(0.0):
                indices.append(UInt32(col))
                values.append(w)
        offsets.append(len(indices))
    return _host_with_csr(graph, offsets^, indices^, values^)


def host_categorical_intersection(
    graph: SparseFuzzySimplicialGraph, target: List[Float32], far_dist: Float64,
    unknown_dist: Float64 = Float64(1.0),
) raises -> SparseFuzzySimplicialGraph:
    """umap-learn `discrete_metric_simplicial_set_intersection` (no target
    metric): `ug_categorical_weight` per edge, exact zeros eliminated, then
    `host_reset_local_connectivity`."""
    var n = graph.n_samples
    if len(target) != n:
        raise Error("UMAP supervised target length differs from n_samples")
    var consts = ug_categorical_constants(far_dist, unknown_dist)
    var offsets = List[Int]()
    var indices = List[UInt32]()
    var values = List[Float32]()
    offsets.append(0)
    for i in range(n):
        for e in range(graph.offsets[i], graph.offsets[i + 1]):
            var j = Int(graph.indices[e])
            var w = ug_categorical_weight(graph.values[e], target[i], target[j], consts[0], consts[1])
            if w != Float32(0.0):
                indices.append(UInt32(j))
                values.append(w)
        offsets.append(len(indices))
    return host_reset_local_connectivity(_host_with_csr(graph, offsets^, indices^, values^))


def _host_min_stored(values: List[Float32]) -> Float32:
    """The smallest nonzero stored value, flushed (FLT_MAX when none)."""
    var m = Float32(3.4028234663852886e38)
    for value in values:
        var v = ftz(value)
        if v != Float32(0.0) and v < m:
            m = v
    return m


def host_general_intersection(
    left: SparseFuzzySimplicialGraph, right: SparseFuzzySimplicialGraph, weight: Float32,
) raises -> SparseFuzzySimplicialGraph:
    """umap-learn `general_simplicial_set_intersection` +
    `sparse.general_sset_intersection` (right_complement False): the union
    pattern of the two graphs, `ug_intersect_cell` per cell, then
    `host_reset_local_connectivity`."""
    var n = left.n_samples
    if right.n_samples != n:
        raise Error("UMAP supervised target graph size differs")
    var left_min = ug_intersect_floor(_host_min_stored(left.values))
    var right_min = ug_intersect_floor(_host_min_stored(right.values))
    var ex = ug_intersect_expo(weight)
    var offsets = List[Int]()
    var indices = List[UInt32]()
    var values = List[Float32]()
    offsets.append(0)
    for i in range(n):
        var a = left.offsets[i]
        var b = right.offsets[i]
        while a < left.offsets[i + 1] or b < right.offsets[i + 1]:
            var ac = n
            var bc = n
            if a < left.offsets[i + 1]:
                ac = Int(left.indices[a])
            if b < right.offsets[i + 1]:
                bc = Int(right.indices[b])
            var col = min(ac, bc)
            var lv = Float32(0.0)
            var rv = Float32(0.0)
            if ac == col:
                lv = left.values[a]
                a += 1
            if bc == col:
                rv = right.values[b]
                b += 1
            indices.append(UInt32(col))
            values.append(ug_intersect_cell(lv, rv, left_min, right_min, ex[0], ex[1]))
        offsets.append(len(indices))
    return host_reset_local_connectivity(_host_with_csr(left, offsets^, indices^, values^))
