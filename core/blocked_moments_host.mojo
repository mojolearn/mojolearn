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


# Tried 2026-10-08 (MOJOLEARN_CLASSICAL_PCA_COV=23, the C23 one-pass Chan covariance arm, run ge123e6f9): NV/AMD pca
# istella 2.28x/1.27x SLOWER, taxi 0.90x/0.78x faster (dimension-dependent; combined 1.195x SLOWER) -> deleted
# (c04 deleted 2026-10-10, lane/grid-act-6). Recoverable at main 42d1e42c6; row in docs/apple-fast/EXPERIMENTS.md.


# TOMBSTONE: MOJOLEARN_CLASSICAL_TSVD_FUSED_STATS (slower) deleted 2026-10-10 by lane/grid-act-6; code recoverable at 328b0ae58.
# Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_TSVD_FUSED_STATS.patch; record in docs/TOMBSTONES.md.
