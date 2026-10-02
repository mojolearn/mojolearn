# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into the CPU host binding through umap_oracle.mojo; product, not only a check.
"""The CSR fuzzy graph on the HOST: the CPU column's builder.

A GPU fit builds the graph on the device
(`umap/sparse_graph.mojo::sparse_fuzzy_simplicial_graph_device`). Both call
the same per-row statements (`ug_row_rho`, `ug_row_sigma`, `ug_member`,
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
    ug_constants,
    ug_member,
    ug_merge_weight,
    ug_row_rho,
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
        rhos[i] = ug_row_rho(dp, i, n_neighbors, local_connectivity)
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
