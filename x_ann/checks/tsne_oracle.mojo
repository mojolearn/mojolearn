# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""t-SNE's seams restated as plain host code, apart from
`x_ann/tsne_core.mojo` (DEVIATIONS 5810-5815):

  5810  k-NN membership and order: (squared distance, index)
  5811  the perplexity bisection: 100 float32 steps, sums ascending
  5812  P's normalization: the total summed over CSR edges ascending
  5813  the repulsion: sum over j ascending; Z over rows ascending
  5814  the attraction: over each row's CSR edges ascending
  5815  the gains update: `update * grad < 0` STRICT (0 at the first step)

`rev` flags compute the unpinned spelling only to show a fixture separates."""

from core.device_fold import host_sum_f32_fixed
from checks.numerics import ftz, identical_div, identical_exp, identical_log, identical_mul, identical_mul_add


def to_sqdist(x: List[Float32], i: Int, j: Int, d: Int) -> Float32:
    var acc = Float32(0.0)
    for c in range(d):
        var diff = ftz(ftz(x[i * d + c]) - ftz(x[j * d + c]))
        acc = ftz(identical_mul_add(diff, diff, acc))
    return acc


def to_knn(x: List[Float32], n: Int, d: Int, nn: Int, mut nd: List[Float32], mut ni: List[Int]):
    """5810 by full selection: repeatedly the smallest (distance, index)."""
    nd = List[Float32](length=n * nn, fill=Float32(0.0))
    ni = List[Int](length=n * nn, fill=0)
    for i in range(n):
        var dist = List[Float32](length=n, fill=Float32(0.0))
        for j in range(n):
            dist[j] = to_sqdist(x, i, j, d)
        var used = List[Bool](length=n, fill=False)
        used[i] = True
        for s in range(nn):
            var best = -1
            for j in range(n):
                if used[j]:
                    continue
                if best < 0 or dist[j] < dist[best] or (dist[j] == dist[best] and j < best):
                    best = j
            used[best] = True
            nd[i * nn + s] = dist[best]
            ni[i * nn + s] = best


def to_sum(v: List[Float32], lo: Int, hi: Int, rev: Bool = False) -> Float32:
    var acc = Float32(0.0)
    for s in range(lo, hi):
        var t = hi - 1 - (s - lo) if rev else s
        acc = ftz(acc + v[t])
    return acc


def to_perplexity(nd: List[Float32], n: Int, nn: Int, log_perp: Float32) -> List[Float32]:
    """5811."""
    var p = List[Float32](length=n * nn, fill=Float32(0.0))
    for i in range(n):
        var beta = Float32(1.0)
        var has_min = False
        var has_max = False
        var bmin = Float32(0.0)
        var bmax = Float32(0.0)
        for _ in range(100):
            for j in range(nn):
                p[i * nn + j] = ftz(identical_exp(ftz(-identical_mul(nd[i * nn + j], beta))))
            var sp = to_sum(p, i * nn, i * nn + nn)
            if sp == Float32(0.0):
                sp = Float32(1e-8)
            var sdp = Float32(0.0)
            for j in range(nn):
                p[i * nn + j] = ftz(identical_div(p[i * nn + j], sp))
                sdp = ftz(sdp + ftz(identical_mul(nd[i * nn + j], p[i * nn + j])))
            var ent = ftz(identical_log(sp) + ftz(identical_mul(beta, sdp)))
            var diff = ftz(ent - log_perp)
            if abs(diff) <= Float32(1e-5):
                break
            if diff > Float32(0.0):
                bmin = beta
                has_min = True
                beta = ftz(identical_mul(ftz(beta + bmax), Float32(0.5))) if has_max else ftz(identical_mul(beta, Float32(2.0)))
            else:
                bmax = beta
                has_max = True
                beta = ftz(identical_mul(ftz(beta + bmin), Float32(0.5))) if has_min else ftz(identical_mul(beta, Float32(0.5)))
    return p^


def to_symmetrize(ni: List[Int], p: List[Float32], n: Int, nn: Int, mut indptr: List[Int], mut indices: List[Int],
                  mut values: List[Float32], rev_total: Bool = False):
    """Dense restatement: P_ij = P_cond(i->j) + P_cond(j->i) over the union
    graph, columns ascending; 5812's total over edges in CSR order."""
    var dense = List[Float32](length=n * n, fill=Float32(0.0))
    var mark = List[Bool](length=n * n, fill=False)
    for i in range(n):
        for s in range(nn):
            var j = ni[i * nn + s]
            mark[i * n + j] = True
            mark[j * n + i] = True
    for i in range(n):
        for j in range(n):
            if not mark[i * n + j]:
                continue
            var a = Float32(0.0)
            var b = Float32(0.0)
            for s in range(nn):
                if ni[i * nn + s] == j:
                    a = p[i * nn + s]
                if ni[j * nn + s] == i:
                    b = p[j * nn + s]
            dense[i * n + j] = ftz(a + b)
    indptr = List[Int](capacity=n + 1)
    indices = List[Int]()
    values = List[Float32]()
    indptr.append(0)
    for i in range(n):
        for j in range(n):
            if mark[i * n + j]:
                indices.append(j)
                values.append(dense[i * n + j])
        indptr.append(len(indices))
    # the product's fixed fold order (core/device_fold.mojo); the reversed
    # sum stays the sabotage arm
    var total = to_sum(values, 0, len(values), True) if rev_total else host_sum_f32_fixed(values, len(values))
    if total < Float32(1.1920929e-07):
        total = Float32(1.1920929e-07)
    for e in range(len(values)):
        values[e] = ftz(identical_div(values[e], total))


def to_q(y: List[Float32], i: Int, j: Int) -> Float32:
    var d0 = ftz(ftz(y[2 * i]) - ftz(y[2 * j]))
    var d1 = ftz(ftz(y[2 * i + 1]) - ftz(y[2 * j + 1]))
    var acc = ftz(identical_mul_add(d0, d0, Float32(0.0)))
    acc = ftz(identical_mul_add(d1, d1, acc))
    return ftz(identical_div(Float32(1.0), ftz(Float32(1.0) + acc)))


def to_fit(x: List[Float32], n: Int, d: Int, y0: List[Float32], perplexity: Float32, exaggeration: Float32,
           lr: Float32, max_iter: Int, exploration: Int, mut kl_out: Float32) -> List[Float32]:
    var k = Int(identical_mul(Float32(3.0), perplexity)) + 1
    var nn = k if k < n - 1 else n - 1
    var nd = List[Float32]()
    var ni = List[Int]()
    to_knn(x, n, d, nn, nd, ni)
    var p = to_perplexity(nd, n, nn, identical_log(perplexity))
    var indptr = List[Int]()
    var indices = List[Int]()
    var values = List[Float32]()
    to_symmetrize(ni, p, n, nn, indptr, indices, values)
    var y = y0.copy()
    var upd = List[Float32](length=2 * n, fill=Float32(0.0))
    var gains = List[Float32](length=2 * n, fill=Float32(1.0))
    var z = Float32(0.0)
    var rep = List[Float32](length=2 * n, fill=Float32(0.0))
    for it in range(max_iter + 1):
        # 5813: row partial sums, then Z over rows
        var rowz = List[Float32](length=n, fill=Float32(0.0))
        for i in range(n):
            var r0 = Float32(0.0)
            var r1 = Float32(0.0)
            var zz = Float32(0.0)
            for j in range(n):
                if j == i:
                    continue
                var q = to_q(y, i, j)
                zz = ftz(zz + q)
                var qq = ftz(identical_mul(q, q))
                r0 = ftz(r0 + ftz(identical_mul(qq, ftz(ftz(y[2 * i]) - ftz(y[2 * j])))))
                r1 = ftz(r1 + ftz(identical_mul(qq, ftz(ftz(y[2 * i + 1]) - ftz(y[2 * j + 1])))))
            rowz[i] = zz
            rep[2 * i] = r0
            rep[2 * i + 1] = r1
        z = to_sum(rowz, 0, n)
        if it == max_iter:
            break
        var ex = exaggeration if it < exploration else Float32(1.0)
        var mom = Float32(0.5) if it < exploration else Float32(0.8)
        var ynew = y.copy()
        for e in range(2 * n):
            var i = e // 2
            var c = e % 2
            var attr = Float32(0.0)
            for s in range(indptr[i], indptr[i + 1]):
                var j = indices[s]
                var pq = ftz(identical_mul(ftz(values[s]), to_q(y, i, j)))
                attr = ftz(attr + ftz(identical_mul(pq, ftz(ftz(y[e]) - ftz(y[2 * j + c])))))
            var neg = ftz(identical_div(rep[e], z))
            var grad = ftz(identical_mul(Float32(4.0), ftz(ftz(identical_mul(ex, attr)) - neg)))
            var gain = gains[e]
            if ftz(identical_mul(upd[e], grad)) < Float32(0.0):
                gain = ftz(gain + Float32(0.2))
            else:
                gain = ftz(identical_mul(gain, Float32(0.8)))
            if gain < Float32(0.01):
                gain = Float32(0.01)
            grad = ftz(identical_mul(grad, gain))
            upd[e] = ftz(ftz(identical_mul(mom, upd[e])) - ftz(identical_mul(lr, grad)))
            gains[e] = gain
            ynew[e] = ftz(ftz(y[e]) + upd[e])
        y = ynew^
    var kl = List[Float32](length=n, fill=Float32(0.0))
    for i in range(n):
        var acc = Float32(0.0)
        for s in range(indptr[i], indptr[i + 1]):
            var pp = ftz(values[s])
            var q = ftz(identical_div(to_q(y, i, indices[s]), z))
            var a = pp if pp > Float32(1.1920929e-07) else Float32(1.1920929e-07)
            var b = q if q > Float32(1.1920929e-07) else Float32(1.1920929e-07)
            acc = ftz(acc + ftz(identical_mul(pp, ftz(identical_log(ftz(identical_div(a, b)))))))
        kl[i] = acc
    # the device's fixed fold order (core/device_fold.mojo)
    kl_out = host_sum_f32_fixed(kl, n)
    return y^
