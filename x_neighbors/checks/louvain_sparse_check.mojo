# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`louvain_item_sparse` (x_neighbors/louvain_sparse.mojo) against the dense
reference `o_louvain` (x_neighbors/checks/oracles.mojo) and the dense item
`louvain_item` (x_neighbors/items.mojo), bit for bit, on the host (lane
neural-pass14): labels, modularity and level count, over a ring and over
random symmetric kNN-like graphs (k = 2 and k = 10, unit and random weights,
n up to 1,200), then the walls of the dense and the sparse item at n = 1,200
and the sparse item alone at n = 6,000.

    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . x_neighbors/checks/louvain_sparse_check.mojo

Raises (nonzero exit) on any difference.
"""
from std.memory import bitcast
from std.time import perf_counter_ns

from x_neighbors.checks.oracles import o_louvain
from x_neighbors.items import FP, IP, louvain_item
from x_neighbors.louvain_sparse import louvain_item_sparse


def _next(mut state: UInt64) -> UInt64:
    state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
    return state


def _knn_graph(n: Int, k: Int, random_weights: Bool, seed: Int) -> List[Float32]:
    """A symmetric graph: node u links to k pseudo-random others (a kNN-like
    sparsity), each edge once in both cells, no self-loops."""
    var a = List[Float32](length=n * n, fill=Float32(0))
    var st = UInt64(seed)
    for u in range(n):
        for _ in range(k):
            var v = Int(_next(st) >> 33) % n
            if v == u:
                continue
            var w = Float32(1.0)
            if random_weights:
                w = Float32(Int((_next(st) >> 40) & UInt64(1023)) + 1) / Float32(256.0)
            a[u * n + v] = w
            a[v * n + u] = w
    return a^


def _ring(n: Int) -> List[Float32]:
    var a = List[Float32](length=n * n, fill=Float32(0))
    for u in range(n):
        var v = (u + 1) % n
        a[u * n + v] = Float32(1.0)
        a[v * n + u] = Float32(1.0)
    return a^


def _fp(xs: List[Float32]) -> FP:
    return FP(unsafe_from_address=Int(xs.unsafe_ptr()))


def _ip(xs: List[Int32]) -> IP:
    return IP(unsafe_from_address=Int(xs.unsafe_ptr()))


def _run_sparse(a: List[Float32], n: Int) raises -> Tuple[List[Int32], Float32, Int, Float64]:
    var labels = List[Int32](length=n, fill=Int32(0))
    var info = List[Float32](length=2, fill=Float32(0))
    var t0 = perf_counter_ns()
    louvain_item_sparse(_fp(a), _ip(labels), _fp(info), n, 0, Float32(1), Float32(1e-7))
    var ms = Float64(perf_counter_ns() - t0) / 1e6
    return (labels^, info[0], Int(info[1]), ms)


def _run_dense(a: List[Float32], n: Int) raises -> Tuple[List[Int32], Float32, Int, Float64]:
    var labels = List[Int32](length=n, fill=Int32(0))
    var info = List[Float32](length=2, fill=Float32(0))
    var w = List[Float32](length=n * n, fill=Float32(0))
    var w2 = List[Float32](length=n * n, fill=Float32(0))
    var comm = List[Int32](length=n, fill=Int32(0))
    var node_of = List[Int32](length=n, fill=Int32(0))
    var deg = List[Float32](length=n, fill=Float32(0))
    var stot = List[Float32](length=n, fill=Float32(0))
    var k2c = List[Float32](length=n, fill=Float32(0))
    var tmp = List[Float32](length=n, fill=Float32(0))
    var t0 = perf_counter_ns()
    louvain_item(0, _fp(a), _ip(labels), _fp(info), _fp(w), _fp(w2), _ip(comm), _ip(node_of),
                 _fp(deg), _fp(stot), _fp(k2c), _fp(tmp), n, 0, Float32(1), Float32(1e-7))
    var ms = Float64(perf_counter_ns() - t0) / 1e6
    _ = w^
    _ = w2^
    _ = comm^
    _ = node_of^
    _ = deg^
    _ = stot^
    _ = k2c^
    _ = tmp^
    return (labels^, info[0], Int(info[1]), ms)


def _compare(name: String, a: List[Float32], n: Int, with_dense: Bool) raises:
    var oracle = o_louvain(a, n, 0, Float32(1), Float32(1e-7))
    var sp = _run_sparse(a, n)
    var bad = 0
    for u in range(n):
        if oracle[0][u] != sp[0][u]:
            bad += 1
    var mod_same = bitcast[DType.uint32](oracle[1]) == bitcast[DType.uint32](sp[1])
    var lev_same = oracle[2] == sp[2]
    var dense_note = String("")
    if with_dense:
        var de = _run_dense(a, n)
        var dbad = 0
        for u in range(n):
            if de[0][u] != sp[0][u]:
                dbad += 1
        if dbad != 0 or bitcast[DType.uint32](de[1]) != bitcast[DType.uint32](sp[1]) or de[2] != sp[2]:
            raise Error(name + ": the sparse item differs from the dense item (" + String(dbad) + " labels)")
        dense_note = " dense " + String(de[3]) + " ms"
    print(name, "n", n, "labels differ", bad, "modularity", sp[1], "same" if mod_same else "DIFFERS",
          "levels", sp[2], "same" if lev_same else "DIFFERS", "sparse", sp[3], "ms" + dense_note)
    if bad != 0 or not mod_same or not lev_same:
        raise Error(name + ": the sparse item differs from o_louvain")


def main() raises:
    _compare("ring 64", _ring(64), 64, True)
    _compare("ring 7", _ring(7), 7, True)
    var seed = 1
    for n in [40, 300, 1200]:
        for k in [2, 10]:
            _compare("knn n=" + String(n) + " k=" + String(k) + " unit", _knn_graph(n, k, False, seed), n, n <= 1200)
            seed += 1
            _compare("knn n=" + String(n) + " k=" + String(k) + " random", _knn_graph(n, k, True, seed), n, n <= 1200)
            seed += 1
    # the sparse item alone at a size the dense item would take a minute on
    var big = _knn_graph(6000, 10, True, 99)
    var sp = _run_sparse(big, 6000)
    print("knn n=6000 k=10 random: sparse", sp[3], "ms, modularity", sp[1], "levels", sp[2])
    print("PASS")
