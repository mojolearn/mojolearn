# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into the CPU host binding through decomposition/host/pca_oracle.mojo.
"""The host column of `core/blocked_moments.mojo` (lane classical-decomp,
2026-10-07): the same leaves, the same per-leaf chains and sub-chain order,
the same binary-counter levels and the same scalar operations
(`core/blocked_moments_ops.mojo`), run in loops. GPU free."""
from core.blocked_moments_ops import (
    BM_GRAM_MEAN_CHAINS,
    BM_MAX_LEVELS,
    BM_TPB,
    bm_add,
    bm_chan_m2,
    bm_chan_mean,
    bm_div,
    bm_fma,
    bm_leaf_count,
    bm_mean_leaf_rows,
    bm_node_rows,
    bm_onepass_leaf_rows,
    bm_sub,
    bm_sub_chains,
)
from checks.numerics import ftz


def host_bm_sum_fold(var part: List[Float32], leaves: Int, cells: Int, divisor: Int) -> List[Float32]:
    """`bm_sum_fold`: the binary-counter levels, then the remainders right
    to left, then `/ divisor` when `divisor > 0`."""
    var rem = List[Float32](length=BM_MAX_LEVELS * max(cells, 1), fill=Float32(0.0))
    var count = leaves
    var level = 0
    var mask = 0
    while count > 0:
        var half = count // 2
        var nxt = List[Float32](length=max(1, half) * max(cells, 1), fill=Float32(0.0))
        for p in range((count + 1) // 2):
            for c in range(cells):
                if 2 * p + 1 < count:
                    nxt[p * cells + c] = bm_add(part[2 * p * cells + c], part[(2 * p + 1) * cells + c])
                else:
                    rem[level * cells + c] = part[2 * p * cells + c]
        if count % 2 == 1:
            mask |= 1 << level
        part = nxt^
        count = half
        level += 1
    var out = List[Float32](length=cells, fill=Float32(0.0))
    for c in range(cells):
        var acc = Float32(0.0)
        var have = False
        for lv in range(BM_MAX_LEVELS):
            if (mask >> lv) & 1 != 0:
                var r = rem[lv * cells + c]
                acc = bm_add(r, acc) if have else r
                have = True
        if divisor > 0:
            acc = bm_div(acc, Float32(divisor))
        out[c] = acc
    return out^


def host_bm_chan_fold(
    var m2: List[Float32], var mean: List[Float32], leaves: Int, m: Int, diag: Bool,
    n: Int, leaf: Int, divisor: Int, mut out_mean: List[Float32],
) -> List[Float32]:
    """`bm_chan_fold`: returns M2 / divisor per cell; `out_mean` gets the
    merged means (length m)."""
    var cells = m if diag else m * m
    var rem2 = List[Float32](length=BM_MAX_LEVELS * cells, fill=Float32(0.0))
    var remm = List[Float32](length=BM_MAX_LEVELS * m, fill=Float32(0.0))
    var count = leaves
    var level = 0
    var mask = 0
    while count > 0:
        var half = count // 2
        var n2 = List[Float32](length=max(1, half) * cells, fill=Float32(0.0))
        var nm = List[Float32](length=max(1, half) * m, fill=Float32(0.0))
        for p in range((count + 1) // 2):
            var lf = 2 * p
            var rt = lf + 1
            if rt < count:
                var na = bm_node_rows(n, leaf, level, lf)
                var nb = bm_node_rows(n, leaf, level, rt)
                for e in range(cells):
                    var i = e if diag else e // m
                    var j = e if diag else e - (e // m) * m
                    n2[p * cells + e] = bm_chan_m2(
                        m2[lf * cells + e], m2[rt * cells + e],
                        mean[lf * m + i], mean[rt * m + i], mean[lf * m + j], mean[rt * m + j],
                        na, nb,
                    )
                for col in range(m):
                    nm[p * m + col] = bm_chan_mean(mean[lf * m + col], mean[rt * m + col], na, nb)
            else:
                for e in range(cells):
                    rem2[level * cells + e] = m2[lf * cells + e]
                for col in range(m):
                    remm[level * m + col] = mean[lf * m + col]
        if count % 2 == 1:
            mask |= 1 << level
        m2 = n2^
        mean = nm^
        count = half
        level += 1
    var out = List[Float32](length=cells, fill=Float32(0.0))
    out_mean = List[Float32](length=m, fill=Float32(0.0))
    for c in range(cells):
        var i = c if diag else c // m
        var j = c if diag else c - (c // m) * m
        var acc = Float32(0.0)
        var mi = Float32(0.0)
        var mj = Float32(0.0)
        var rows = 0
        var have = False
        for lv in range(BM_MAX_LEVELS):
            if (mask >> lv) & 1 != 0:
                var nr = bm_node_rows(n, leaf, lv, (leaves >> lv) - 1)
                var r2 = rem2[lv * cells + c]
                var ri = remm[lv * m + i]
                var rj = remm[lv * m + j]
                if have:
                    acc = bm_chan_m2(r2, acc, ri, mi, rj, mj, nr, rows)
                    mi = bm_chan_mean(ri, mi, nr, rows)
                    mj = bm_chan_mean(rj, mj, nr, rows)
                    rows += nr
                else:
                    acc = r2
                    mi = ri
                    mj = rj
                    rows = nr
                    have = True
        out[c] = bm_div(acc, Float32(divisor))
        if i == j:
            out_mean[i] = mi
    return out^


def host_bm_column_mean(x: List[Float32], n: Int, d: Int) -> List[Float32]:
    """`bm_column_mean`."""
    var leaf = bm_mean_leaf_rows(n)
    var leaves = bm_leaf_count(n, leaf)
    var part = List[Float32](length=max(1, leaves * d), fill=Float32(0.0))
    for k in range(leaves):
        var r0 = k * leaf
        var r1 = min(n, r0 + leaf)
        if d <= BM_TPB:
            var chains = bm_sub_chains(d)
            for j in range(d):
                var tot = Float32(0.0)
                for s in range(chains):
                    var acc = Float32(0.0)
                    var r = r0 + s
                    while r < r1:
                        acc = bm_add(acc, x[r * d + j])
                        r += chains
                    tot = acc if s == 0 else bm_add(tot, acc)
                part[k * d + j] = tot
        else:
            for j in range(d):
                var acc = Float32(0.0)
                for r in range(r0, r1):
                    acc = bm_add(acc, x[r * d + j])
                part[k * d + j] = acc
    return host_bm_sum_fold(part^, leaves, d, n)


def host_bm_onepass_covariance(
    x: List[Float32], n: Int, d: Int, mut mu: List[Float32],
) -> List[Float32]:
    """`bm_onepass_covariance`: returns the (n - 1)-scaled covariance and
    sets `mu`."""
    var cells = d * d
    var leaf = bm_onepass_leaf_rows(n, cells)
    var leaves = bm_leaf_count(n, leaf)
    var part = List[Float32](length=max(1, leaves * cells), fill=Float32(0.0))
    var means = List[Float32](length=max(1, leaves * d), fill=Float32(0.0))
    var c = List[Float32](length=max(1, d), fill=Float32(0.0))
    for k in range(leaves):
        var r0 = k * leaf
        var r1 = min(n, r0 + leaf)
        for col in range(d):
            var tot = Float32(0.0)
            for s in range(BM_GRAM_MEAN_CHAINS):
                var acc = Float32(0.0)
                var r = r0 + s
                while r < r1:
                    acc = bm_add(acc, x[r * d + col])
                    r += BM_GRAM_MEAN_CHAINS
                tot = acc if s == 0 else bm_add(tot, acc)
            c[col] = bm_div(tot, Float32(r1 - r0))
            means[k * d + col] = c[col]
        for i in range(d):
            for j in range(i, d):
                var acc = Float32(0.0)
                for r in range(r0, r1):
                    acc = bm_fma(bm_sub(x[r * d + i], c[i]), bm_sub(x[r * d + j], c[j]), acc)
                part[k * cells + i * d + j] = acc
                part[k * cells + j * d + i] = acc
    return host_bm_chan_fold(part^, means^, leaves, d, False, n, leaf, n - 1, mu)


def _host_tsvd_value(x: List[Float32], v: List[Float32], r: Int, d: Int, j: Int) -> Float32:
    if j < d:
        return ftz(x[r * d + j])
    var cc = j - d
    var acc = Float32(0.0)
    for f in range(d):
        acc = bm_fma(ftz(x[r * d + f]), ftz(v[cc * d + f]), acc)
    return acc


def host_bm_tsvd_variances(x: List[Float32], v: List[Float32], n: Int, d: Int, nc: Int) -> List[Float32]:
    """`bm_tsvd_variances`: [var(X) | var(X V^T)], ddof 0."""
    var m = d + nc
    var leaf = bm_onepass_leaf_rows(n, m)
    var leaves = bm_leaf_count(n, leaf)
    var lmean = List[Float32](length=max(1, leaves * m), fill=Float32(0.0))
    var lm2 = List[Float32](length=max(1, leaves * m), fill=Float32(0.0))
    var chains = bm_sub_chains(m)
    for k in range(leaves):
        var r0 = k * leaf
        var r1 = min(n, r0 + leaf)
        var cnt = Float32(r1 - r0)
        for j in range(m):
            var mj: Float32
            var tot2 = Float32(0.0)
            if m <= BM_TPB:
                var tot = Float32(0.0)
                for s in range(chains):
                    var acc = Float32(0.0)
                    var r = r0 + s
                    while r < r1:
                        acc = bm_add(acc, _host_tsvd_value(x, v, r, d, j))
                        r += chains
                    tot = acc if s == 0 else bm_add(tot, acc)
                mj = bm_div(tot, cnt)
                for s in range(chains):
                    var acc2 = Float32(0.0)
                    var r = r0 + s
                    while r < r1:
                        var cv = bm_sub(_host_tsvd_value(x, v, r, d, j), mj)
                        acc2 = bm_fma(cv, cv, acc2)
                        r += chains
                    tot2 = acc2 if s == 0 else bm_add(tot2, acc2)
            else:
                var acc = Float32(0.0)
                for r in range(r0, r1):
                    acc = bm_add(acc, _host_tsvd_value(x, v, r, d, j))
                mj = bm_div(acc, cnt)
                for r in range(r0, r1):
                    var cv = bm_sub(_host_tsvd_value(x, v, r, d, j), mj)
                    tot2 = bm_fma(cv, cv, tot2)
            lmean[k * m + j] = mj
            lm2[k * m + j] = tot2
    var unused = List[Float32]()
    return host_bm_chan_fold(lm2^, lmean^, leaves, m, True, n, leaf, n, unused)
