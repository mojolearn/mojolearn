# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CSR fuzzy graph groundwork; not connected to the dense estimator yet.

Storage is O(n*k). Row-local insertion sorting costs O(n*k*k); transpose
construction and row merges are linear in stored entries. Arithmetic and
accepted input semantics follow graph.mojo. Explicit zero entries from kNN
candidates are retained; consumers must apply their existing weight policy.
"""
from checks.numerics import identical_exp64, identical_log2_64, identical_mul, identical_mul_add, identical_pow64
from std.math import fma

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_predict_threads import (
    host_list_ptr,
    host_predict_chunk,
    host_predict_task_count,
)
from core.host_parallel import host_parallelize
from umap.graph import _finite, _sigma_fast, _sigma_identical


struct SparseFuzzySimplicialGraph(Copyable, Movable):
    var n_samples: Int
    var n_neighbors: Int
    var rhos: List[Float32]
    var sigmas: List[Float32]
    var directed_offsets: List[Int]
    var directed_indices: List[UInt32]
    var directed_values: List[Float32]
    var offsets: List[Int]
    var indices: List[UInt32]
    var values: List[Float32]

    def __init__(
        out self, n_samples: Int, n_neighbors: Int,
        var rhos: List[Float32], var sigmas: List[Float32],
        var directed_offsets: List[Int], var directed_indices: List[UInt32],
        var directed_values: List[Float32], var offsets: List[Int],
        var indices: List[UInt32], var values: List[Float32],
    ):
        self.n_samples = n_samples
        self.n_neighbors = n_neighbors
        self.rhos = rhos^
        self.sigmas = sigmas^
        self.directed_offsets = directed_offsets^
        self.directed_indices = directed_indices^
        self.directed_values = directed_values^
        self.offsets = offsets^
        self.indices = indices^
        self.values = values^

    def logical_payload_bytes(self) -> Int:
        """Occupied scalar bytes on the supported 64-bit hosts.

        Excludes caller inputs, temporary transpose/cursors, List spare
        capacity and allocator overhead; not a resident-memory measurement.
        """
        return (
            4 * (len(self.rhos) + len(self.sigmas))
            + 8 * (len(self.directed_offsets) + len(self.offsets))
            + 4 * (len(self.directed_indices) + len(self.directed_values))
            + 4 * (len(self.indices) + len(self.values))
        )


# DEVIATION 5323 (PIN; lane/algos-decomp, 2026-09-27): umap-learn's
# `smooth_knn_dist` rho at a local_connectivity other than 1, host code both
# columns run: the row's positive distances in rank order, index =
# floor(lc), rho = nz[index - 1] + interp (nz[index] - nz[index - 1]) as ONE
# Float64 fma rounded once to Float32 when interp > 1e-5 (SMOOTH_K_TOLERANCE),
# interp * nz[0] when index is 0, the largest positive distance when the row
# has fewer than lc of them, 0 when it has none. lc = 1 keeps the first
# positive distance (the path above, unchanged).
comptime UMAP_SMOOTH_K_TOLERANCE = Float64(1.0e-5)


def _local_rho(distances: List[Float32], row: Int, k: Int, lc: Float32) -> Float32:
    var nz = List[Float32]()
    for j in range(k):
        var d = distances[row * k + j]
        if d > Float32(0.0):
            nz.append(d)
    var lc64 = Float64(lc)
    if Float64(len(nz)) >= lc64:
        var index = Int(lc64)          # floor: lc >= 0
        var interp = lc64 - Float64(index)
        if index > 0:
            var rho = nz[index - 1]
            if interp > UMAP_SMOOTH_K_TOLERANCE:
                var diff = nz[index] - nz[index - 1]
                rho = Float32(fma(interp, Float64(diff), Float64(rho)))
            return rho
        return Float32(interp * Float64(nz[0])) if len(nz) > 0 else Float32(0.0)
    if len(nz) > 0:
        return nz[len(nz) - 1]
    return Float32(0.0)


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
    var target = identical_log2_64(Float64(n_neighbors))
    # Validate in caller order so malformed input keeps the same first error,
    # and compute rho while the row is already hot.  Sigma searches are then
    # independent by row: moving whole searches to worker threads changes no
    # statement or reduction order within a row.
    for i in range(n_samples):
        if Int(knn_indices[i * n_neighbors]) != i:
            raise Error("UMAP expects self in k-NN slot zero")
        var previous = Float32(-1.0)
        var rho = Float64(0.0)
        for j in range(n_neighbors):
            var d = knn_distances[i * n_neighbors + j]
            if not _finite(d) or d < Float32(0.0) or (
                j > 0 and d < previous
            ):
                raise Error("UMAP k-NN distances must be finite and sorted")
            previous = d
            if rho == 0.0 and d > Float32(0.0):
                rho = Float64(d)
        if local_connectivity != Float32(1.0):
            rho = Float64(_local_rho(knn_distances, i, n_neighbors, local_connectivity))
        rhos[i] = Float32(rho)
    var tasks = host_predict_task_count(n_samples)
    # The thread-pool join is not worthwhile for small graph builds.
    if n_samples < 256:
        tasks = 1
    var chunk = host_predict_chunk(n_samples, tasks)
    var rp = host_list_ptr(rhos)
    var sp = host_list_ptr(sigmas)
    var failed = List[Int](length=tasks, fill=0)
    var fp = failed.unsafe_ptr()

    def _sigma_rows(task: Int) {imm knn_distances, imm target, imm n_neighbors, imm n_samples, imm chunk, imm rp, imm sp, imm fp}:
        try:
            var lo = task * chunk
            var hi = min(lo + chunk, n_samples)
            for i in range(lo, hi):
                var rho = Float64(rp.unsafe_load(i))
                var sigma: Float64
                comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
                    sigma = _sigma_identical(
                        knn_distances, i, n_neighbors, rho, target
                    )
                else:
                    sigma = _sigma_fast(
                        knn_distances, i, n_neighbors, rho, target
                    )
                sp.unsafe_store(i, Float32(sigma))
        except:
            fp.unsafe_store(task, 1)

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
            var delta = knn_distances[i * n_neighbors + j] - rhos[i]
            var value = Float32(1.0)
            if delta > Float32(0.0):
                value = Float32(identical_exp64(-Float64(delta) / Float64(sigmas[i])))
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
            # Preserve dense expression and operand orientation per cell.
            # Do not calculate once and mirror to the opposite row.
            var union = a + b - a * b
            var intersection = a * b
            # ONE rounding on the intersection's product: the default
            # (contract=fast) build fused `(1 - mix) * intersection` into the
            # add (lane/explicit-fma-contract-proof, 2026-09-26)
            var weight = identical_mul_add(
                Float32(1.0) - set_op_mix_ratio, intersection,
                set_op_mix_ratio * union,
            )
            indices.append(UInt32(col))
            values.append(weight)
        offsets.append(len(indices))
    return SparseFuzzySimplicialGraph(
        n_samples, n_neighbors, rhos^, sigmas^, doff^, dcol^, dval^,
        offsets^, indices^, values^
    )


# ---------------------------------------------------------------------------
# Supervised UMAP (lane/algos-decomp, 2026-09-27; DEVIATION 5324, PIN): the
# target's graph set operations of umap-learn's `umap_.py`, host code both
# columns run, every CSR row in ascending column order.
# ---------------------------------------------------------------------------


def _with_csr(
    graph: SparseFuzzySimplicialGraph, var offsets: List[Int], var indices: List[UInt32], var values: List[Float32]
) -> SparseFuzzySimplicialGraph:
    return SparseFuzzySimplicialGraph(
        graph.n_samples, graph.n_neighbors, graph.rhos.copy(), graph.sigmas.copy(),
        graph.directed_offsets.copy(), graph.directed_indices.copy(), graph.directed_values.copy(),
        offsets^, indices^, values^,
    )


def reset_local_connectivity(graph: SparseFuzzySimplicialGraph) raises -> SparseFuzzySimplicialGraph:
    """umap-learn `reset_local_connectivity`: every row divided by its
    largest stored value (sklearn `normalize(norm='max')`, one float32
    division), then S + S^T - S o S^T per cell as (a + b) - a*b with the
    product pinned (no contraction), and exact zeros eliminated."""
    var n = graph.n_samples
    var nv = List[Float32](capacity=len(graph.values))
    for row in range(n):
        var mx = Float32(0.0)
        for e in range(graph.offsets[row], graph.offsets[row + 1]):
            if graph.values[e] > mx:
                mx = graph.values[e]
        for e in range(graph.offsets[row], graph.offsets[row + 1]):
            nv.append(graph.values[e] / mx if mx > Float32(0.0) else graph.values[e])
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
            var w = (a + b) - identical_mul(a, b)
            if w != Float32(0.0):
                indices.append(UInt32(col))
                values.append(w)
        offsets.append(len(indices))
    return _with_csr(graph, offsets^, indices^, values^)


def categorical_intersection(
    graph: SparseFuzzySimplicialGraph, target: List[Float32], far_dist: Float64,
    unknown_dist: Float64 = Float64(1.0),
) raises -> SparseFuzzySimplicialGraph:
    """umap-learn `discrete_metric_simplicial_set_intersection` (no target
    metric): an edge whose ends carry different labels is scaled by
    exp(-far_dist), one with an unknown label (-1) by exp(-unknown_dist),
    each as Float32(Float64(w) * exp) with the portable exp; exact zeros are
    eliminated, then `reset_local_connectivity`."""
    var n = graph.n_samples
    if len(target) != n:
        raise Error("UMAP supervised target length differs from n_samples")
    var far = identical_exp64(-far_dist)
    var unknown = identical_exp64(-unknown_dist)
    var offsets = List[Int]()
    var indices = List[UInt32]()
    var values = List[Float32]()
    offsets.append(0)
    for i in range(n):
        for e in range(graph.offsets[i], graph.offsets[i + 1]):
            var j = Int(graph.indices[e])
            var w = graph.values[e]
            if target[i] == Float32(-1.0) or target[j] == Float32(-1.0):
                w = Float32(Float64(w) * unknown)
            elif target[i] != target[j]:
                w = Float32(Float64(w) * far)
            if w != Float32(0.0):
                indices.append(UInt32(j))
                values.append(w)
        offsets.append(len(indices))
    return reset_local_connectivity(_with_csr(graph, offsets^, indices^, values^))


def _csr_at(offsets: List[Int], indices: List[UInt32], values: List[Float32], row: Int, col: Int) -> Tuple[Bool, Float32]:
    for e in range(offsets[row], offsets[row + 1]):
        if Int(indices[e]) == col:
            return (True, values[e])
    return (False, Float32(0.0))


def general_intersection(
    left: SparseFuzzySimplicialGraph, right: SparseFuzzySimplicialGraph, weight: Float32,
) raises -> SparseFuzzySimplicialGraph:
    """umap-learn `general_simplicial_set_intersection` +
    `sparse.general_sset_intersection` (right_complement False): the union
    pattern of the two graphs holding left + right; a cell where either side
    beats its floor (half its graph's smallest stored value, at least 1e-8,
    in Float64) becomes left * right^(w / (1 - w)) (w < 0.5) or
    left^((1 - w) / w) * right, Float64 with the portable pow, rounded once;
    then `reset_local_connectivity`."""
    var n = left.n_samples
    if right.n_samples != n:
        raise Error("UMAP supervised target graph size differs")
    # the smallest STORED value: umap-learn's graphs have had their explicit
    # zeros eliminated, so a stored zero here counts as absent
    var lmin_v = Float32(3.4028234663852886e38)
    for v in left.values:
        if v != Float32(0.0) and v < lmin_v:
            lmin_v = v
    var rmin_v = Float32(3.4028234663852886e38)
    for v in right.values:
        if v != Float32(0.0) and v < rmin_v:
            rmin_v = v
    var left_min = max(Float64(lmin_v) / 2.0, Float64(1.0e-8))
    var right_min = max(Float64(rmin_v) / 2.0, Float64(1.0e-8))
    var w64 = Float64(weight)
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
            var has_l = False
            var has_r = False
            if ac == col:
                lv = left.values[a]
                has_l = lv != Float32(0.0)
                a += 1
            if bc == col:
                rv = right.values[b]
                has_r = rv != Float32(0.0)
                b += 1
            var out = lv + rv
            var left_val = Float64(lv) if has_l else left_min
            var right_val = Float64(rv) if has_r else right_min
            if left_val > left_min or right_val > right_min:
                if w64 < 0.5:
                    out = Float32(left_val * identical_pow64(right_val, w64 / (1.0 - w64)))
                else:
                    out = Float32(identical_pow64(left_val, (1.0 - w64) / w64) * right_val)
            indices.append(UInt32(col))
            values.append(out)
        offsets.append(len(indices))
    return reset_local_connectivity(_with_csr(left, offsets^, indices^, values^))
