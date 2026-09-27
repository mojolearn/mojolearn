# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IVF-PQ's seams restated as plain host code, written apart from
`x_ann/ivf_pq_core.mojo` so a sabotage of a shared cell moves the device and
not this oracle (DEVIATIONS 5800-5804).

  5800  the fused square fold of a (sub)space distance: ascending coordinates
  5801  the encoding tie: the LOWER code wins an exact tie
  5803  the search top-k: the total order (distance, row id)
  5804  the probe selection: the total order (coarse distance, list id)

The coarse quantizer and the codebooks are cluster/'s k-means; their host
oracles (`host_ivf_build`, `host_kmeans_fit`) are cluster/'s and ivf/'s, and
the device side runs the device k-means, so they are compared here too.

`descending` / `high_tie` are the UNPINNED spellings, computed only to show
a fixture separates them from the pinned ones."""

from std.memory import bitcast
from cluster.host.kmeans_oracle import host_kmeans_fit
from cluster.impl.kmeans_params import INIT_KMEANS_PLUS_PLUS, METRIC_L2_EXPANDED
from ivf.host.ivf_host import host_ivf_build
from checks.numerics import ftz, identical_div, identical_mul_add


def or_dist(a: List[Float32], ao: Int, b: List[Float32], bo: Int, length: Int, descending: Bool = False) -> Float32:
    """DEVIATION 5800."""
    var acc = Float32(0.0)
    for s in range(length):
        var t = length - 1 - s if descending else s
        var diff = ftz(ftz(a[ao + t]) - ftz(b[bo + t]))
        acc = ftz(identical_mul_add(diff, diff, acc))
    return acc


def or_argmin(r: List[Float32], ro: Int, cb: List[Float32], cbo: Int, n_codes: Int, length: Int,
              high_tie: Bool = False) -> Int:
    """DEVIATION 5801: strict < keeps the lower code."""
    var best = 0
    var bd = or_dist(r, ro, cb, cbo, length)
    for c in range(1, n_codes):
        var d = or_dist(r, ro, cb, cbo + c * length, length)
        if d < bd or (high_tie and d == bd):
            bd = d
            best = c
    return best


@fieldwise_init
struct OrIndex(Movable):
    var centers: List[Float32]
    var labels: List[Int]
    var offsets: List[Int32]
    var list_indices: List[Int32]
    var codebooks: List[Float32]
    var codes: List[Int32]


def or_build(x: List[Float32], n: Int, dim: Int, n_lists: Int, iters: Int, seed: Int, pq_dim: Int,
             pq_bits: Int, pq_iters: Int) raises -> OrIndex:
    var flat = host_ivf_build(x, n, dim, n_lists, iters, METRIC_L2_EXPANDED, UInt64(seed))
    var centers = flat.centers.copy()
    var offsets = flat.offsets.copy()
    var list_indices = List[Int32]()
    for s in range(n):
        list_indices.append(Int32(Int(flat.list_indices[s])))
    var labels = List[Int](length=n, fill=0)
    for l in range(n_lists):
        for s in range(Int(offsets[l]), Int(offsets[l + 1])):
            labels[Int(list_indices[s])] = l
    var pq_len = (dim + pq_dim - 1) // pq_dim
    var rot = pq_len * pq_dim
    var r = List[Float32](length=n * rot, fill=Float32(0.0))
    for i in range(n):
        for c in range(dim):
            r[i * rot + c] = ftz(ftz(x[i * dim + c]) - ftz(centers[labels[i] * dim + c]))
    var n_codes = 1 << pq_bits
    var codebooks = List[Float32]()
    for j in range(pq_dim):
        var sub = List[Float32]()
        for i in range(n):
            for t in range(pq_len):
                sub.append(r[i * rot + j * pq_len + t])
        var cb = List[Float32](length=n_codes * pq_len, fill=Float32(0.0))
        var lab = List[UInt32](length=n, fill=UInt32(0))
        _ = host_kmeans_fit(sub, n, pq_len, n_codes, cb, lab, List[Float32](), 0, pq_iters, Float64(1e-4),
                            UInt64(seed), 1, INIT_KMEANS_PLUS_PLUS, METRIC_L2_EXPANDED, Float64(2.0))
        for e in range(len(cb)):
            codebooks.append(cb[e])
    var codes = List[Int32](length=n * pq_dim, fill=Int32(0))
    for i in range(n):
        for j in range(pq_dim):
            codes[i * pq_dim + j] = Int32(or_argmin(r, i * rot + j * pq_len, codebooks, j * n_codes * pq_len, n_codes, pq_len))
    return OrIndex(centers^, labels^, offsets^, list_indices^, codebooks^, codes^)


def or_before(d: Float32, id: Int, sd: Float32, sid: Int, high_tie: Bool = False) -> Bool:
    """DEVIATIONS 5803/5804: the (value, id) total order; sid < 0 is empty."""
    if sid < 0:
        return True
    if high_tie:
        return d < sd or (d == sd and id > sid)
    return d < sd or (d == sd and id < sid)


def or_search(
    centers: List[Float32], offsets: List[Int32], list_indices: List[Int32], codebooks: List[Float32],
    codes: List[Int32], mask: List[Int32], n_lists: Int, dim: Int, pq_dim: Int, pq_bits: Int,
    queries: List[Float32], m: Int, k: Int, n_probes: Int,
    mut out_d: List[Float32], mut out_i: List[Int32], mut out_n: List[Int32],
):
    var pq_len = (dim + pq_dim - 1) // pq_dim
    var n_codes = 1 << pq_bits
    out_d = List[Float32](length=m * k, fill=bitcast[DType.float32](UInt32(0x7F800000)))
    out_i = List[Int32](length=m * k, fill=Int32(-1))
    out_n = List[Int32](length=m, fill=Int32(0))
    for q in range(m):
        # the probes: sort list ids by (coarse distance, id)
        var cd = List[Float32]()
        for l in range(n_lists):
            cd.append(or_dist(queries, q * dim, centers, l * dim, dim))
        var order = List[Int]()
        for _ in range(n_probes):
            var best = -1
            for l in range(n_lists):
                var taken = False
                for u in range(len(order)):
                    if order[u] == l:
                        taken = True
                if taken:
                    continue
                if best < 0 or or_before(cd[l], l, cd[best], best):
                    best = l
            order.append(best)
        var bd = List[Float32]()
        var bi = List[Int]()
        for p in range(n_probes):
            var l = order[p]
            var qr = List[Float32](length=pq_len * pq_dim, fill=Float32(0.0))
            for c in range(dim):
                qr[c] = ftz(ftz(queries[q * dim + c]) - ftz(centers[l * dim + c]))
            for s in range(Int(offsets[l]), Int(offsets[l + 1])):
                var row = Int(list_indices[s])
                if mask[row] == 0:
                    continue
                var total = Float32(0.0)
                for j in range(pq_dim):
                    var code = Int(codes[row * pq_dim + j])
                    total = ftz(total + or_dist(qr, j * pq_len, codebooks, (j * n_codes + code) * pq_len, pq_len))
                bd.append(total)
                bi.append(row)
                out_n[q] += 1
        # top-k by repeated selection under (distance, id)
        var used = List[Bool](length=len(bd), fill=False)
        for s in range(k):
            var best = -1
            for t in range(len(bd)):
                if used[t]:
                    continue
                if best < 0 or or_before(bd[t], bi[t], bd[best], bi[best]):
                    best = t
            if best < 0:
                continue
            used[best] = True
            out_d[q * k + s] = bd[best]
            out_i[q * k + s] = Int32(bi[best])
