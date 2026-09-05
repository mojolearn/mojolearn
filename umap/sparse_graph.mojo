# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CSR fuzzy graph groundwork; not connected to the dense estimator yet.

Storage is O(n*k). Row-local insertion sorting costs O(n*k*k); transpose
construction and row merges are linear in stored entries. Arithmetic and
accepted input semantics follow graph.mojo. Explicit zero entries from kNN
candidates are retained; consumers must apply their existing weight policy.
"""
from std.math import exp, log2
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
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


def sparse_fuzzy_simplicial_graph(
    knn_indices: List[UInt32], knn_distances: List[Float32],
    n_samples: Int, n_neighbors: Int,
    set_op_mix_ratio: Float32 = Float32(1.0),
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
    var target = log2(Float64(n_neighbors))
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
        rhos[i] = Float32(rho)
        var sigma: Float64
        comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
            sigma = _sigma_identical(knn_distances, i, n_neighbors, rho, target)
        else:
            sigma = _sigma_fast(knn_distances, i, n_neighbors, rho, target)
        sigmas[i] = Float32(sigma)

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
                value = Float32(exp(-Float64(delta) / Float64(sigmas[i])))
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
            var weight = (
                set_op_mix_ratio * union
                + (Float32(1.0) - set_op_mix_ratio) * intersection
            )
            indices.append(UInt32(col))
            values.append(weight)
        offsets.append(len(indices))
    return SparseFuzzySimplicialGraph(
        n_samples, n_neighbors, rhos^, sigmas^, doff^, dcol^, dval^,
        offsets^, indices^, values^
    )
