# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Main-lane only: exact dense/CSR comparison and linear storage accounting.

The only dense allocation is in these SMALL comparison fixtures. Production
sparse_graph.mojo never densifies. Set MOJOLEARN_UMAP_SPARSE_LARGE=1 for a
65,536-row k=8 ring; MOJOLEARN_UMAP_SPARSE_ROWS overrides its size. Printed
logical bytes exclude List spare capacity, temporaries and allocator costs.
"""
from std.memory import bitcast
from std.os import getenv
from checks.numerics import numeric_mode_name
from umap.graph import fuzzy_simplicial_graph
from umap.sparse_graph import SparseFuzzySimplicialGraph, sparse_fuzzy_simplicial_graph


def equal_bits(a: Float32, b: Float32, what: String) raises:
    if bitcast[DType.uint32](a) != bitcast[DType.uint32](b):
        raise Error("sparse UMAP bytes differ: " + what)


def structure(g: SparseFuzzySimplicialGraph) raises:
    var n = g.n_samples
    if len(g.offsets) != n + 1 or len(g.directed_offsets) != n + 1:
        raise Error("CSR offset shape")
    if g.offsets[0] != 0 or g.directed_offsets[0] != 0:
        raise Error("CSR must start at zero")
    if g.offsets[n] != len(g.indices) or len(g.indices) != len(g.values):
        raise Error("CSR union terminal offset")
    if g.directed_offsets[n] != len(g.directed_indices) or len(
        g.directed_indices
    ) != len(g.directed_values):
        raise Error("CSR directed terminal offset")
    for i in range(n):
        if g.offsets[i] > g.offsets[i + 1] or (
            g.directed_offsets[i] > g.directed_offsets[i + 1]
        ):
            raise Error("CSR offset monotonicity")
        var previous = -1
        for at in range(g.offsets[i], g.offsets[i + 1]):
            var col = Int(g.indices[at])
            if col <= previous or col >= n or col == i:
                raise Error("CSR union columns must be unique sorted nonself")
            previous = col
        previous = -1
        for at in range(g.directed_offsets[i], g.directed_offsets[i + 1]):
            var col = Int(g.directed_indices[at])
            if col <= previous or col >= n or col == i:
                raise Error("CSR directed columns must be unique sorted nonself")
            previous = col
    if len(g.directed_indices) > n * (g.n_neighbors - 1):
        raise Error("directed storage exceeds kNN edge bound")
    if len(g.indices) > 2 * len(g.directed_indices):
        raise Error("union storage exceeds doubled directed edge bound")


def compare(
    idx: List[UInt32], dist: List[Float32], n: Int, k: Int,
    mix: Float32, tag: String,
) raises:
    var dense = fuzzy_simplicial_graph(idx, dist, n, k, mix)
    var sparse = sparse_fuzzy_simplicial_graph(idx, dist, n, k, mix)
    structure(sparse)
    var directed = List[Float32]()
    var weights = List[Float32]()
    directed.resize(n * n, Float32(0.0))
    weights.resize(n * n, Float32(0.0))
    for i in range(n):
        equal_bits(dense.rhos[i], sparse.rhos[i], tag + " rho")
        equal_bits(dense.sigmas[i], sparse.sigmas[i], tag + " sigma")
        print("UMAP_SPARSE_BITS", tag, "rho", i, bitcast[DType.uint32](sparse.rhos[i]))
        print("UMAP_SPARSE_BITS", tag, "sigma", i, bitcast[DType.uint32](sparse.sigmas[i]))
        for at in range(sparse.directed_offsets[i], sparse.directed_offsets[i + 1]):
            directed[i * n + Int(sparse.directed_indices[at])] = sparse.directed_values[at]
        for at in range(sparse.offsets[i], sparse.offsets[i + 1]):
            weights[i * n + Int(sparse.indices[at])] = sparse.values[at]
    for i in range(n * n):
        equal_bits(dense.directed[i], directed[i], tag + " directed")
        equal_bits(dense.weights[i], weights[i], tag + " union")
        print("UMAP_SPARSE_BITS", tag, "directed", i, bitcast[DType.uint32](directed[i]))
        print("UMAP_SPARSE_BITS", tag, "weights", i, bitcast[DType.uint32](weights[i]))
    print("UMAP_SPARSE_CASE_PASS", tag)


def refusal(
    idx: List[UInt32], dist: List[Float32], n: Int, k: Int, mix: Float32,
) raises:
    var dense_refused = False
    var sparse_refused = False
    try:
        _ = fuzzy_simplicial_graph(idx, dist, n, k, mix)
    except:
        dense_refused = True
    try:
        _ = sparse_fuzzy_simplicial_graph(idx, dist, n, k, mix)
    except:
        sparse_refused = True
    if not dense_refused or not sparse_refused:
        raise Error("dense/sparse invalid-input refusal mismatch")


def scaling(n: Int) raises -> Int:
    var k = 8
    if n < 16:
        raise Error("sparse ring needs at least 16 rows")
    var idx = List[UInt32]()
    var dist = List[Float32]()
    for i in range(n):
        for j in range(k):
            idx.append(UInt32((i + j) % n))
            dist.append(Float32(j))
    var graph = sparse_fuzzy_simplicial_graph(idx, dist, n, k)
    structure(graph)
    if len(graph.directed_indices) != n * 7 or len(graph.indices) != n * 14:
        raise Error("ring CSR edge count")
    var payload = graph.logical_payload_bytes()
    # rhos/sigmas 8n; two offset arrays 16(n+1); directed7n pairs56n;
    # symmetric14n pairs112n. 192n+16: no n*n buffer is present.
    if payload != 192 * n + 16:
        raise Error("CSR occupied scalar byte accounting")
    print("UMAP_SPARSE_STORAGE", n, k, "directed", len(graph.directed_indices),
          "union", len(graph.indices), "logical_bytes", payload,
          "dense_graph_logical_bytes", 8 * n * n + 8 * n)
    return payload


def main() raises:
    print("UMAP sparse graph mode", numeric_mode_name())
    var idx: List[UInt32] = [0, 1, 2, 1, 0, 2, 2, 1, 0]
    var dist: List[Float32] = [0, 1, 3, 0, 1, 2, 0, 2, 3]
    compare(idx, dist, 3, 3, Float32(1.0), "triangle")
    # Duplicate destinations, unsorted destination IDs, unequal ranks,
    # one-way edges and distance ties are valid dense input semantics.
    var dup_idx: List[UInt32] = [
        0, 3, 1, 3, 1, 4, 2, 0, 2, 0, 4, 0,
        3, 2, 1, 2, 4, 1, 0, 3,
    ]
    var dup_dist: List[Float32] = [
        0, 0.5, 1, 2, 0, 0.5, 0.5, 3, 0, 1, 2, 3,
        0, 1, 1, 4, 0, 0.5, 1, 2,
    ]
    var mixes: List[Float32] = [0, 0.25, 0.5, 1]
    for i in range(len(mixes)):
        compare(dup_idx, dup_dist, 5, 4, mixes[i], "duplicates-mix" + String(i))
    # Dense graph currently accepts a nonzero distance for the self slot.
    var nonzero_self = dup_dist.copy()
    nonzero_self[0] = Float32(0.125)
    compare(dup_idx, nonzero_self, 5, 4, Float32(0.5), "nonzero-self-distance")
    var tied_idx: List[UInt32] = [0, 2, 1, 1, 2, 0, 2, 1, 0]
    var zero_dist: List[Float32] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    compare(tied_idx, zero_dist, 3, 3, Float32(1.0), "coincident-tied-points")
    var signed_zero = zero_dist.copy()
    signed_zero[0] = bitcast[DType.float32](UInt32(0x80000000))
    compare(tied_idx, signed_zero, 3, 3, Float32(0.0), "signed-zero")
    var pair_idx: List[UInt32] = [0, 1, 1, 0]
    var pair_dist: List[Float32] = [0, 1, 0, 1]
    compare(pair_idx, pair_dist, 2, 2, Float32(1.0), "minimum-k2")

    refusal(idx, dist, 1, 3, Float32(1.0))
    refusal(idx, dist, 3, 1, Float32(1.0))
    refusal(idx, dist, 3, 4, Float32(1.0))
    refusal(pair_idx, dist, 3, 3, Float32(1.0))
    refusal(idx, pair_dist, 3, 3, Float32(1.0))
    refusal(idx, dist, 3, 3, Float32(-0.1))
    refusal(idx, dist, 3, 3, Float32(1.1))
    for at in range(3):
        var bad_idx = idx.copy()
        if at == 0:
            bad_idx[0] = UInt32(1)
        elif at == 1:
            bad_idx[1] = UInt32(0)
        else:
            bad_idx[1] = UInt32(3)
        refusal(bad_idx, dist, 3, 3, Float32(1.0))
    var patterns: List[UInt32] = [0x7FC00001, 0x7F800000, 0xFF800000]
    for bits in patterns:
        var bad = bitcast[DType.float32](bits)
        refusal(idx, dist, 3, 3, bad)
        var bad_dist = dist.copy()
        bad_dist[1] = bad
        refusal(idx, bad_dist, 3, 3, Float32(1.0))
    var negative = dist.copy()
    negative[1] = Float32(-1.0)
    refusal(idx, negative, 3, 3, Float32(1.0))
    var descending = dist.copy()
    descending[2] = Float32(0.5)
    refusal(idx, descending, 3, 3, Float32(1.0))
    print("UMAP sparse graph refusals PASS: 18 cases")
    var small = scaling(128)
    var doubled = scaling(256)
    if doubled != 2 * small - 16:
        raise Error("CSR payload does not scale linearly")
    if String(getenv("MOJOLEARN_UMAP_SPARSE_LARGE")) == "1":
        var n = 65536
        var text = String(getenv("MOJOLEARN_UMAP_SPARSE_ROWS"))
        if text != "":
            n = Int(atol(text))
        _ = scaling(n)
    print("UMAP sparse graph PASS")
