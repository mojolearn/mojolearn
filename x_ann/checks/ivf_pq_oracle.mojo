# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IVF-PQ's seams restated as plain host code, written apart from
`x_ann/ivf_pq_core.mojo` so a sabotage of a shared cell moves the device and
not this oracle (DEVIATIONS 5800-5804).

  5800  the fused square fold of a (sub)space distance: ascending coordinates
  5801  the encoding tie: the LOWER code wins an exact tie
  5803  the search top-k: the total order (distance, row id)
  5804  the probe selection: the total order (coarse distance, list id)

The coarse quantizer is cluster/'s k-means; its host oracle
(`host_ivf_build`) is ivf/'s, and the device side runs the device k-means, so
it is compared here too. The codebooks are the batched Lloyd loop of
`x_ann/pq_kmeans_device.mojo` (the IDENTICAL default since 2026-10-09,
IDN_PQ_DEVICE_CODEBOOKS; strided training rows since 2026-10-11,
IDN_PQ_TRAINSET_STRIDED), restated here as `or_codebooks_batched`; a subspace
wider than 16 or more than 256 codes takes cluster/'s k-means per subspace
(`host_kmeans_fit`), as the build does. Until 2026-11-19 this oracle trained
every codebook with `host_kmeans_fit`, which the device had not run since
those two promotions: the first GPU run of ivf_pq_check (build 0c68d6cd27)
showed it.

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


def or_codebooks_batched(
    r: List[Float32], n: Int, rot: Int, pq_dim: Int, pq_len: Int, n_codes: Int, pq_iters: Int, seed: Int
) -> List[Float32]:
    """The batched codebooks, plainly. Training rows: every row, or the
    256 * n_codes rows s -> (s n) // n_train when n is larger. Seeds: code c
    of subspace j is sample row c * stride + (h mod stride), stride =
    n_train // n_codes (at least 1), h the splitmix64 finalizer of
    seed (as Int32) ^ ((j + 1) * 0x9E3779B97F4A7C15), clamped to the last
    sample row. Then pq_iters times, per subspace: every sample row takes its
    nearest code (5800's fold, 5801's lower code on a tie); the rows of each
    code are summed in blocks of 256 sample rows (row order, from 0, each add
    flushed), the block sums added in block order (from 0, flushed), and the
    code becomes sum / count (one quotient); a code with no rows keeps its
    words."""
    var n_train = n
    if n > 256 * n_codes:
        n_train = 256 * n_codes
    var per = n_codes * pq_len
    var cb = List[Float32](length=pq_dim * per, fill=Float32(0.0))
    var stride = n_train // n_codes
    if stride < 1:
        stride = 1
    for j in range(pq_dim):
        var h = UInt64(Int(Int32(seed))) ^ (UInt64(j + 1) * UInt64(0x9E3779B97F4A7C15))
        h = (h ^ (h >> 30)) * UInt64(0xBF58476D1CE4E5B9)
        h = (h ^ (h >> 27)) * UInt64(0x94D049BB133111EB)
        h = h ^ (h >> 31)
        for c in range(n_codes):
            var s = c * stride + Int(h % UInt64(stride))
            if s >= n_train:
                s = n_train - 1
            var row = (s * n) // n_train
            for u in range(pq_len):
                cb[j * per + c * pq_len + u] = ftz(r[row * rot + j * pq_len + u])
    var n_blocks = (n_train + 255) // 256
    for _ in range(pq_iters):
        for j in range(pq_dim):
            var lab = List[Int](length=n_train, fill=0)
            for s in range(n_train):
                lab[s] = or_argmin(r, ((s * n) // n_train) * rot + j * pq_len, cb, j * per, n_codes, pq_len)
            var tot = List[Float32](length=per, fill=Float32(0.0))
            var cnt = List[Int](length=n_codes, fill=0)
            for b in range(n_blocks):
                var part = List[Float32](length=per, fill=Float32(0.0))
                var s1 = min(b * 256 + 256, n_train)
                for s in range(b * 256, s1):
                    var off = ((s * n) // n_train) * rot + j * pq_len
                    for u in range(pq_len):
                        part[lab[s] * pq_len + u] = ftz(part[lab[s] * pq_len + u] + ftz(r[off + u]))
                    cnt[lab[s]] += 1
                for e in range(per):
                    tot[e] = ftz(tot[e] + part[e])
            for c in range(n_codes):
                if cnt[c] > 0:
                    for u in range(pq_len):
                        cb[j * per + c * pq_len + u] = identical_div(tot[c * pq_len + u], Float32(cnt[c]))
    return cb^


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
    var batched = pq_len <= 16 and n_codes <= 256
    if batched:
        codebooks = or_codebooks_batched(r, n, rot, pq_dim, pq_len, n_codes, pq_iters, seed)
    for j in range(pq_dim if not batched else 0):
        var sub = List[Float32]()
        for i in range(n):
            for t in range(pq_len):
                sub.append(r[i * rot + j * pq_len + t])
        var cb = List[Float32](length=n_codes * pq_len, fill=Float32(0.0))
        var lab = List[UInt32](length=n, fill=UInt32(0))
        _ = host_kmeans_fit(sub, n, pq_len, n_codes, cb, lab, List[Float32](), 0, pq_iters, Float64(1e-4),
                            UInt64(seed), 1, INIT_KMEANS_PLUS_PLUS, METRIC_L2_EXPANDED, Float64(2.0),
                            lazy_shift=True)  # the per-subspace route's lazy_shift (x_ann/ivf_pq_device.mojo `_codebooks`)
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
