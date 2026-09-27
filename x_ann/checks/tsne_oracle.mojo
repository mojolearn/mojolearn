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

from checks.numerics import (
    ftz, identical_div, identical_exp, identical_log, identical_mul, identical_mul_add, identical_pow, identical_sqrt,
)


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
    var total = to_sum(values, 0, len(values), rev_total)
    if total < Float32(1.1920929e-07):
        total = Float32(1.1920929e-07)
    for e in range(len(values)):
        values[e] = ftz(identical_div(values[e], total))


def to_q(y: List[Float32], i: Int, j: Int, nc: Int, dof: Int) -> Float32:
    var acc = Float32(0.0)
    for c in range(nc):
        var dd = ftz(ftz(y[i * nc + c]) - ftz(y[j * nc + c]))
        acc = ftz(identical_mul_add(dd, dd, acc))
    var fd = Float32(dof)
    var q = ftz(identical_div(fd, ftz(fd + acc)))
    if dof == 1:
        return q
    if dof == 2:
        return ftz(identical_mul(q, ftz(identical_sqrt(q))))
    return ftz(identical_pow(q, identical_div(Float32(dof + 1), Float32(2.0))))


def to_fit(x: List[Float32], n: Int, d: Int, nc: Int, y0: List[Float32], perplexity: Float32, exaggeration: Float32,
           lr: Float32, max_iter: Int, exploration: Int, exact: Bool, patience_main: Int, min_grad_norm: Float32,
           mut kl_out: Float32, mut n_iter_out: Int) -> List[Float32]:
    """sklearn's schedule restated: two phases, update and gains reset at the
    change, error / grad norm every 50 steps and at a phase's last step."""
    var k = Int(identical_mul(Float32(3.0), perplexity)) + 1
    var nn = n - 1 if exact else (k if k < n - 1 else n - 1)
    var dof = nc - 1 if nc > 1 else 1
    var nd = List[Float32]()
    var ni = List[Int]()
    to_knn(x, n, d, nn, nd, ni)
    var p = to_perplexity(nd, n, nn, identical_log(perplexity))
    var indptr = List[Int]()
    var indices = List[Int]()
    var values = List[Float32]()
    to_symmetrize(ni, p, n, nn, indptr, indices, values)
    var y = y0.copy()
    var tiny = Float32(1.1754944e-38)
    var cfac = ftz(identical_div(Float32(2 * (dof + 1)), Float32(dof)))
    var it = 0
    var last = -1
    var kl_last = Float32(0.0)
    for phase in range(2):
        var end = exploration if phase == 0 else max_iter
        if it >= end:
            continue
        var upd = List[Float32](length=nc * n, fill=Float32(0.0))
        var gains = List[Float32](length=nc * n, fill=Float32(1.0))
        var ex = exaggeration if phase == 0 else Float32(1.0)
        var mom = Float32(0.5) if phase == 0 else Float32(0.8)
        var patience = 250 if phase == 0 else patience_main
        var best_err = Float32(3.4028235e38)
        var best_iter = it
        for i in range(it, end):
            var check = (i + 1) % 50 == 0 or i == end - 1
            var rowz = List[Float32](length=n, fill=Float32(0.0))
            var rep = List[Float32](length=nc * n, fill=Float32(0.0))
            for ii in range(n):
                for c in range(nc):
                    var zz = Float32(0.0)
                    var r = Float32(0.0)
                    for j in range(n):
                        if j == ii:
                            continue
                        var q = to_q(y, ii, j, nc, dof)
                        zz = ftz(zz + q)
                        var qq = ftz(identical_mul(q, q))
                        r = ftz(r + ftz(identical_mul(qq, ftz(ftz(y[ii * nc + c]) - ftz(y[j * nc + c])))))
                    rep[ii * nc + c] = r
                    if c == 0:
                        rowz[ii] = zz
            var z = to_sum(rowz, 0, n)
            var err = Float32(0.0)
            if check:
                for ii in range(n):
                    var acc = Float32(0.0)
                    for s in range(indptr[ii], indptr[ii + 1]):
                        var pp = ftz(identical_mul(ftz(values[s]), ex))
                        var q = ftz(identical_div(to_q(y, ii, indices[s], nc, dof), z))
                        var a = pp if pp > tiny else tiny
                        var b = q if q > tiny else tiny
                        acc = ftz(acc + ftz(identical_mul(pp, ftz(identical_log(ftz(identical_div(a, b)))))))
                    err = ftz(err + acc)
            var ynew = y.copy()
            var g2 = Float32(0.0)
            for e in range(nc * n):
                var ii = e // nc
                var c = e % nc
                var attr = Float32(0.0)
                for s in range(indptr[ii], indptr[ii + 1]):
                    var j = indices[s]
                    var pq = ftz(identical_mul(ftz(values[s]), to_q(y, ii, j, nc, dof)))
                    attr = ftz(attr + ftz(identical_mul(pq, ftz(ftz(y[e]) - ftz(y[j * nc + c])))))
                var neg = ftz(identical_div(rep[e], z))
                var grad = ftz(identical_mul(cfac, ftz(ftz(identical_mul(ex, attr)) - neg)))
                var gain = gains[e]
                if ftz(identical_mul(upd[e], grad)) < Float32(0.0):
                    gain = ftz(gain + Float32(0.2))
                else:
                    gain = ftz(identical_mul(gain, Float32(0.8)))
                if gain < Float32(0.01):
                    gain = Float32(0.01)
                grad = ftz(identical_mul(grad, gain))
                g2 = ftz(identical_mul_add(grad, grad, g2))
                upd[e] = ftz(ftz(identical_mul(mom, upd[e])) - ftz(identical_mul(lr, grad)))
                gains[e] = gain
                ynew[e] = ftz(ftz(y[e]) + upd[e])
            y = ynew^
            last = i
            if check:
                kl_last = err
            if (i + 1) % 50 == 0:
                var gnorm = ftz(identical_sqrt(g2))
                if err < best_err:
                    best_err = err
                    best_iter = i
                elif i - best_iter > patience:
                    break
                if gnorm <= min_grad_norm:
                    break
        it = last + 1
    kl_out = kl_last
    n_iter_out = last
    return y^
