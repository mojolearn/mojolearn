# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host-side helpers every cluster-lane driver shares (lane/algos-cluster).
Host code, compiled from this one source into both bindings; nothing here
reaches a device."""
from checks.numerics import ftz, identical_mul
from x_cluster.bodies import SplitMix64
from x_cluster.ops import ClusterOps


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
    labels = ops.get_i(ls, n)
    dist = ops.get(ds, n)


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
    mut ops: O, xs_host: List[Float32], m: Int, d: Int, k: Int, mut rng: SplitMix64
) raises -> List[Float32]:
    """sklearn `_kmeans_plusplus` (cluster/_kmeans.py:200-280), unit weights:
    the first center uniform, then `2 + int(log(k))` candidates per pick drawn
    by `searchsorted(cumsum(closest), u * pot)` (side left, clipped), each
    candidate's potential `sum(min(closest, d_cand))`, the lowest potential
    under a strict `<`. Potentials and the cumulative sum are ascending
    Float64 chains on the host; the distances are the device's."""
    from std.math import log

    var n_trials = 2 + Int(log(Float64(k)))
    var xs = ops.put(xs_host)
    var centers = List[Float32](capacity=k * d)
    var first = rng.below(m)
    for f in range(d):
        centers.append(xs_host[first * d + f])
    var cslot = ops.put(gather_rows(xs_host, d, [first]))
    var closest_s = ops.zeros(m)
    ops.sqdist(cslot, 1, xs, m, d, closest_s)
    var closest = ops.get(closest_s, m)
    var pot = sum_f64(closest, m)
    var cand_s = ops.zeros(n_trials * d)
    var dc_s = ops.zeros(n_trials * m)
    for _c in range(1, k):
        var cum = List[Float64](capacity=m)
        var acc = Float64(0)
        for t in range(m):
            acc = acc + Float64(closest[t])
            cum.append(acc)
        var ids = List[Int](capacity=n_trials)
        for _t in range(n_trials):
            var v = rng.unit() * pot
            var lo = 0
            var hi = m
            while lo < hi:
                var mid = (lo + hi) // 2
                if cum[mid] < v:
                    lo = mid + 1
                else:
                    hi = mid
            ids.append(lo if lo < m else m - 1)
        ops.set(cand_s, gather_rows(xs_host, d, ids))
        ops.sqdist(cand_s, n_trials, xs, m, d, dc_s)
        var dc = ops.get(dc_s, n_trials * m)
        var best = 0
        var best_pot = Float64(0)
        for t in range(n_trials):
            var p = Float64(0)
            for j in range(m):
                var v = dc[t * m + j]
                p = p + Float64(v if v < closest[j] else closest[j])
            if t == 0 or p < best_pot:
                best_pot = p
                best = t
        for j in range(m):
            var v = dc[best * m + j]
            if v < closest[j]:
                closest[j] = v
        pot = best_pot
        for f in range(d):
            centers.append(xs_host[ids[best] * d + f])
    return centers^
