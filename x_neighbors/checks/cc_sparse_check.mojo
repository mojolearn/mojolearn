# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`cc_iterate_sparse` (x_neighbors/cc_sparse.mojo) against the dense host
loop `cc_iterate_dense` (x_neighbors/iter_host.mojo, the rounds of
`cc_step_item` over every cell), labels and round count, on rings (one
component, many rounds), directed random graphs with isolated nodes and
several components, and a symmetric kNN-like graph (lane neural-pass22);
then the walls of both at n = 3,000 and the sparse walk alone at n = 12,000.

    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . x_neighbors/checks/cc_sparse_check.mojo

Raises (nonzero exit) on any difference.
"""
from std.time import perf_counter_ns

from x_neighbors.cc_sparse import cc_iterate_sparse
from x_neighbors.items import FP, IP
from x_neighbors.iter_host import cc_iterate_dense


def _next(mut state: UInt64) -> UInt64:
    state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
    return state


def _ring(n: Int) -> List[Float32]:
    var a = List[Float32](length=n * n, fill=Float32(0))
    for u in range(n):
        a[u * n + (u + 1) % n] = Float32(1.0)  # directed ring: weak connectivity joins it
    return a^


def _random_directed(n: Int, k: Int, pieces: Int, seed: Int) -> List[Float32]:
    """`pieces` blocks of nodes; inside a block node u gets k random out-edges
    to its own block (some to itself, which the dense step reads as a
    self-loop); every 7th node of the last block is left isolated."""
    var a = List[Float32](length=n * n, fill=Float32(0))
    var st = UInt64(seed)
    var size = (n + pieces - 1) // pieces
    for u in range(n):
        var blk = u // size
        var lo = blk * size
        var hi = min(lo + size, n)
        if blk == pieces - 1 and u % 7 == 0:
            continue
        for _ in range(k):
            var v = lo + Int(_next(st) >> 33) % (hi - lo)
            var w = Float32(Int((_next(st) >> 40) & UInt64(1023)) + 1) / Float32(256.0)
            a[u * n + v] = w
    return a^


def _knn_symmetric(n: Int, k: Int, seed: Int) -> List[Float32]:
    var a = List[Float32](length=n * n, fill=Float32(0))
    var st = UInt64(seed)
    for u in range(n):
        for _ in range(k):
            var v = Int(_next(st) >> 33) % n
            if v == u:
                continue
            a[u * n + v] = Float32(1.0)
            a[v * n + u] = Float32(1.0)
    return a^


def _fp(xs: List[Float32]) -> FP:
    return FP(unsafe_from_address=Int(xs.unsafe_ptr()))


def _ip(xs: List[Int32]) -> IP:
    return IP(unsafe_from_address=Int(xs.unsafe_ptr()))


def _run(a: List[Float32], n: Int, sparse: Bool) raises -> Tuple[List[Int32], Int, Float64]:
    var lab = List[Int32](length=n, fill=Int32(0))
    for i in range(n):
        lab[i] = Int32(i)
    var info = List[Int32](length=1, fill=Int32(0))
    var t0 = perf_counter_ns()
    if sparse:
        cc_iterate_sparse(_fp(a), _ip(lab), _ip(info), n)
    else:
        cc_iterate_dense(Int(a.unsafe_ptr()), Int(lab.unsafe_ptr()), Int(info.unsafe_ptr()), n)
    var ms = Float64(perf_counter_ns() - t0) / 1e6
    return (lab^, Int(info[0]), ms)


def _compare(name: String, a: List[Float32], n: Int) raises:
    var d = _run(a, n, False)
    var s = _run(a, n, True)
    var bad = 0
    for u in range(n):
        if d[0][u] != s[0][u]:
            bad += 1
    var comps = 0
    for u in range(n):
        if Int(d[0][u]) == u:
            comps += 1
    print(name, "n", n, "labels differ", bad, "rounds dense", d[1], "sparse", s[1], "components", comps,
          "dense", d[2], "ms sparse", s[2], "ms")
    if bad != 0 or d[1] != s[1]:
        raise Error(name + ": the sparse walk differs from the dense loop")


def main() raises:
    _compare("ring 64", _ring(64), 64)
    _compare("ring 7", _ring(7), 7)
    _compare("ring 1", _ring(1), 1)
    var seed = 1
    for n in [40, 300, 1200]:
        for k in [1, 3]:
            _compare("directed n=" + String(n) + " k=" + String(k) + " pieces=3", _random_directed(n, k, 3, seed), n)
            seed += 1
    _compare("knn n=1200 k=5", _knn_symmetric(1200, 5, 77), 1200)
    _compare("knn n=3000 k=10", _knn_symmetric(3000, 10, 78), 3000)
    var big = _knn_symmetric(12000, 10, 99)
    var s = _run(big, 12000, True)
    print("knn n=12000 k=10: sparse", s[2], "ms, rounds", s[1])
    print("PASS")
