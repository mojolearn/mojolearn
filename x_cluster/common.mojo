"""Helpers every cluster-lane driver shares (lane/algos-cluster), compiled
from this one source into both bindings; the n-sized work goes through
`ClusterOps`."""
from experiments.classical_identical_ideas.graph_controls import XCLUSTER_KPP_DISTINCT
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632

from checks.numerics import ftz, identical_mul
from x_cluster.bodies import SplitMix64
from x_cluster.ops import ClusterOps
from x_cluster.post_bodies import FM_PROD, FM_VAL


def gather_rows(x: List[Float32], d: Int, idx: List[Int]) -> List[Float32]:
    var out = List[Float32](capacity=len(idx) * d)
    for r in idx:
        for f in range(d):
            out.append(x[r * d + f])
    return out^


def sum_f64(v: List[Float32], n: Int) -> Float64:
    """One ascending Float64 chain."""
    var acc = Float64(0)
    for t in range(n):
        acc = acc + Float64(v[t])
    return acc


def nearest_all[O: ClusterOps](
    mut ops: O, xs: Int, n: Int, c: List[Float32], k: Int, d: Int,
    mut labels: List[Int32], mut dist: List[Float32],
) raises:
    """labels/dist of every row of slot `xs` against the centers `c`."""
    var cs = ops.put(c)
    var ls = ops.zeros_i(n)
    var ds = ops.zeros(n)
    ops.nearest(xs, n, cs, k, d, ls, ds)
    ops.get_if(ls, n, ds, n, labels, dist)


def distances_to[O: ClusterOps](
    mut ops: O, xs: Int, n: Int, c: List[Float32], k: Int, d: Int
) raises -> List[Float32]:
    """Euclidean (rooted) distances n x k of slot `xs` to the centers `c`."""
    var cs = ops.put(c)
    var out = ops.zeros(n * k)
    ops.sqdist(xs, n, cs, k, d, out)
    ops.sqrt(out, n * k)
    return ops.get(out, n * k)


def greedy_kmeans_pp[O: ClusterOps](
    mut ops: O, xs_host: List[Float32], m: Int, d: Int, k: Int, mut rng: SplitMix64,
    w: List[Float32] = List[Float32](),
) raises -> List[Float32]:
    """The centers of `greedy_kmeans_pp_indices`."""
    return gather_rows(xs_host, d, greedy_kmeans_pp_indices(ops, xs_host, m, d, k, rng, w))


def greedy_kmeans_pp_indices[O: ClusterOps](
    mut ops: O, xs_host: List[Float32], m: Int, d: Int, k: Int, mut rng: SplitMix64,
    w: List[Float32] = List[Float32](),
) raises -> List[Int]:
    """sklearn `_kmeans_plusplus` (cluster/_kmeans.py:200-280), unit weights:
    the first center uniform, then `2 + int(log(k))` candidates per pick drawn
    by `searchsorted(cumsum(closest), u * pot)` (side left, clipped), each
    candidate's potential `sum(min(closest, d_cand))`, the lowest potential
    under a strict `<`. Lane cgr2-cluster: everything n-sized is the
    device's: the distances, the potentials (the float-float fold of
    `post_bodies`), the search of the cumulative table (`kpp_search_cell`:
    the chunk totals ascending, then the chunk's values), the closest
    distances; the host draws the uniforms and keeps the k-sized picks. With
    sample weights `w` (empty: unit) every potential and the cumulative sum
    weigh each row, as sklearn's `closest_dist_sq @ sample_weight`."""
    var weighted = len(w) > 0
    from std.math import log

    var n_trials = 2 + Int(log(Float64(k)))
    var xs = ops.put(xs_host)
    var ws = ops.put(w) if weighted else -1
    var picks = List[Int](capacity=k)
    var first = rng.below(m)
    picks.append(first)
    var ids = ops.zeros_i(n_trials)
    ops.set_i(ids, [Int32(first)])
    var cslot = ops.alloc(n_trials * d)
    ops.gather_rows(xs, d, ids, 1, cslot)
    var closest = ops.alloc(m)
    ops.sqdist(cslot, 1, xs, m, d, closest)
    var pot = ops.sum_ff(closest, ws, -1, m, FM_PROD if weighted else FM_VAL)
    var dc_s = ops.alloc(n_trials * m)
    for _c in range(1, k):
        var vs = List[Float64](capacity=n_trials)
        for _t in range(n_trials):
            vs.append(rng.unit() * pot)
        ops.kpp_search(closest, ws, m, vs, ids)
        ops.gather_rows(xs, d, ids, n_trials, cslot)
        comptime if XCLUSTER_KPP_DISTINCT:
            ops.kpp_distinct(cslot,ids,n_trials,xs,m,d,dc_s)
        else:
            ops.sqdist(cslot, n_trials, xs, m, d, dc_s)
        var pots = ops.kpp_pots(dc_s, closest, ws, n_trials, m)
        var best = 0
        for t in range(1, n_trials):
            if pots[t] < pots[best]:
                best = t
        ops.kpp_take(dc_s, closest, best, m)
        pot = pots[best]
        picks.append(Int(ops.get_i(ids, n_trials)[best]))
    return picks^


def weighted_draw(cum: List[Float64], mut rng: SplitMix64) -> Int:
    """An index drawn with probability proportional to the weights whose
    ascending Float64 cumulative sum is `cum`: `u * total` searched with side
    left, clipped (numpy's `choice(p=...)` rule on a cumulative table)."""
    var n = len(cum)
    var v = rng.unit() * cum[n - 1]
    var lo = 0
    var hi = n
    while lo < hi:
        var mid = (lo + hi) // 2
        if cum[mid] <= v:
            lo = mid + 1
        else:
            hi = mid
    return lo if lo < n else n - 1
