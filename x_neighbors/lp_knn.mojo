# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Compact k-NN connectivity graph, with the dense graph's exact cell order.

Rows contain k slots, sorted by column. Duplicates and missing neighbors
become -1 padding. No n-by-m storage or scan, including graph normalization.
"""
from std.memory import bitcast
from checks.numerics import ftz, identical_div, identical_sqrt, identical_mul_add
from x_neighbors.items import FP, IP, _add


def op_lp_knn_graph(idx: Int, cols: Int, vals: Int, n: Int, m: Int, k: Int, variant: Int) raises:
    """variant: 0 row-normalized propagation, 1 spreading, 2 prediction."""
    if variant < 0 or variant > 2 or (variant != 2 and n != m):
        raise Error("lp_knn_graph: invalid variant or non-square fit graph")
    var pi = IP(unsafe_from_address=idx)
    var pc = IP(unsafe_from_address=cols)
    var pv = FP(unsafe_from_address=vals)
    var deg = List[Float32](length=m if variant == 1 else 0, fill=Float32(0))
    var counts = List[Int](length=n, fill=0)
    for i in range(n):
        var count = 0
        for e in range(k):
            pc.unsafe_store(i * k + e, Int32(-1))
            pv.unsafe_store(i * k + e, Float32(0))
        for e in range(k):
            var j = Int(pi.unsafe_load(i * k + e))
            if j < 0:
                continue
            if j >= m:
                raise Error("lp_knn_graph: neighbor outside reference rows")
            var at = 0
            while at < count and Int(pc.unsafe_load(i * k + at)) < j:
                at += 1
            if at < count and Int(pc.unsafe_load(i * k + at)) == j:
                continue
            var end = count
            while end > at:
                pc.unsafe_store(i * k + end, pc.unsafe_load(i * k + end - 1))
                end -= 1
            pc.unsafe_store(i * k + at, Int32(j))
            count += 1
            if variant == 1 and i != j:
                # i ascending: same float32 fold as col_degree_item.
                deg[j] = _add(deg[j], Float32(1))
        counts[i] = count
    for i in range(n):
        var denom = Float32(0)
        for e in range(counts[i]):
            denom = _add(denom, Float32(1))
        if denom == Float32(0):
            denom = Float32(1)
        for e in range(counts[i]):
            var j = Int(pc.unsafe_load(i * k + e))
            var value = Float32(1)
            if variant == 0:
                value = ftz(identical_div(value, denom))
            elif variant == 1:
                if i == j:
                    value = Float32(0)
                else:
                    var wi = ftz(identical_sqrt(deg[i])) if deg[i] != Float32(0) else Float32(1)
                    var wj = ftz(identical_sqrt(deg[j])) if deg[j] != Float32(0) else Float32(1)
                    value = ftz(identical_div(value, wj))
                    value = ftz(identical_div(value, wi))
            pv.unsafe_store(i * k + e, value)


def lp_knn_finite(x: FP, count: Int) -> Bool:
    for t in range(count):
        if (bitcast[DType.uint32](x.unsafe_load(t)) & UInt32(0x7F800000)) == UInt32(0x7F800000):
            return False
    return True


def lp_knn_product_item(t: Int, cols: IP, vals: FP, x: FP, out: FP,
                        n: Int, m: Int, k: Int, c: Int, finite: Bool):
    var i = t // c
    var j = t % c
    var acc = Float32(0)
    if finite:
        for e in range(k):
            var p = Int(cols.unsafe_load(i * k + e))
            if p < 0:
                break
            var a = ftz(vals.unsafe_load(i * k + e))
            if a != Float32(0):
                acc = ftz(identical_mul_add(a, ftz(x.unsafe_load(p * c + j)), acc))
    else:
        # Preserve dense IEEE 0*NaN/Inf behavior without allocating dense G.
        var e = 0
        for p in range(m):
            var a = Float32(0)
            if e < k and Int(cols.unsafe_load(i * k + e)) == p:
                a = ftz(vals.unsafe_load(i * k + e))
                e += 1
            acc = ftz(identical_mul_add(a, ftz(x.unsafe_load(p * c + j)), acc))
    out.unsafe_store(t, acc)
