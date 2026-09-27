# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MeanShift (lane/algos-cluster). Reference: scikit-learn
`sklearn/cluster/_mean_shift.py` (`estimate_bandwidth` :35-95,
`_mean_shift_single_seed` :97-120, `MeanShift.fit` :450-540).

The bandwidth, when not given, is the mean over the rows of the distance to
the `int(n * quantile)`-th nearest row (itself included): the n x n squared
distances and the row order statistic are the device's
(`bodies.kth_smallest_row`, an exact bisection on the float bits), the roots
and the mean one ascending Float64 chain on the host. Every seed's shift loop
is ONE device thread (`bodies.meanshift_seed`), the flat kernel over all rows
in ascending order. The rest is sklearn's host logic from one source: the
centers with a nonzero intensity kept per distinct center (the last seed's
intensity, a dict), sorted by (intensity, coordinates) descending, a center
within the bandwidth of a kept one dropped, then the nearest center per row
(the lowest index on a tie), -1 beyond the bandwidth when not `cluster_all`."""
from checks.numerics import ftz, identical_mul, identical_sqrt
from x_cluster.ops import ClusterOps


def estimate_bandwidth_ops[O: ClusterOps](
    mut ops: O, x: List[Float32], n: Int, d: Int, quantile: Float64
) raises -> Float32:
    var k = Int(Float64(n) * quantile)
    if k < 1:
        k = 1
    var xs = ops.put(x)
    var dm = ops.zeros(n * n)
    ops.sqdist(xs, n, xs, n, d, dm)
    var kt = ops.zeros(n)
    ops.kth(dm, n, n, k, kt)
    var kth = ops.get(kt, n)
    var acc = Float64(0)
    for r in range(n):
        acc = acc + Float64(identical_sqrt(kth[r]))
    return Float32(acc / Float64(n))


def _key_greater(a: List[Float32], ia: Int, b: List[Float32], ib: Int, d: Int, inten: List[Int]) -> Bool:
    """(intensity, coordinates) of center ia > that of ib, lexicographic."""
    if inten[ia] != inten[ib]:
        return inten[ia] > inten[ib]
    for f in range(d):
        var u = a[ia * d + f]
        var v = b[ib * d + f]
        if u != v:
            return u > v
    return False


def meanshift_fit[O: ClusterOps](
    mut ops: O, x: List[Float32], n: Int, d: Int, bandwidth_in: Float32, seeds_in: List[Float32],
    n_seeds_in: Int, cluster_all: Bool, max_iter: Int,
    mut centers_out: List[Float32], mut labels: List[Int32], mut bw_out: Float32, mut n_iter: Int,
) raises:
    var bw = bandwidth_in
    if bw <= Float32(0):
        bw = estimate_bandwidth_ops(ops, x, n, d, 0.3)
    if not (bw > Float32(0)):
        raise Error("bandwidth needs to be greater than zero or None, got " + String(bw))
    bw_out = bw
    var ns = n_seeds_in
    var seeds = seeds_in.copy()
    if ns == 0:
        ns = n
        seeds = x.copy()
    var xs = ops.put(x)
    var cs = ops.put(seeds)
    var sc = ops.zeros(ns * d)
    var it_s = ops.zeros_i(ns)
    var ic_s = ops.zeros_i(ns)
    var stop = ftz(identical_mul(Float32(1e-3), bw))
    ops.meanshift(xs, n, d, bw, stop, max_iter, cs, ns, sc, ic_s, it_s)
    var cen = ops.get(cs, ns * d)
    var inten = ops.get_i(ic_s, ns)
    var iters = ops.get_i(it_s, ns)
    n_iter = 0
    for s in range(ns):
        if Int(iters[s]) > n_iter:
            n_iter = Int(iters[s])
    # the dict: distinct centers in first-insertion order, the last intensity
    var uc = List[Float32]()
    var ui = List[Int]()
    var m = 0
    for s in range(ns):
        if inten[s] == 0:
            continue
        var found = -1
        for u in range(m):
            var same = True
            for f in range(d):
                if uc[u * d + f] != cen[s * d + f]:
                    same = False
                    break
            if same:
                found = u
                break
        if found >= 0:
            ui[found] = Int(inten[s])
        else:
            for f in range(d):
                uc.append(cen[s * d + f])
            ui.append(Int(inten[s]))
            m += 1
    if m == 0:
        raise Error("No point was within bandwidth=" + String(bw) + " of any seed. Try a different seeding strategy or increase the bandwidth.")
    # sorted by (intensity, coordinates), descending: an insertion sort of indices
    var order = List[Int](capacity=m)
    for u in range(m):
        order.append(u)
    for a in range(1, m):
        var b = a
        while b > 0 and _key_greater(uc, order[b], uc, order[b - 1], d, ui):
            var t = order[b - 1]
            order[b - 1] = order[b]
            order[b] = t
            b -= 1
    var sorted = List[Float32](capacity=m * d)
    for q in range(m):
        for f in range(d):
            sorted.append(uc[order[q] * d + f])
    # radius suppression over the sorted centers
    var ss = ops.put(sorted)
    var dd_s = ops.zeros(m * m)
    ops.sqdist(ss, m, ss, m, d, dd_s)
    ops.sqrt(dd_s, m * m)
    var dd = ops.get(dd_s, m * m)
    var unique = List[Bool](length=m, fill=True)
    for i in range(m):
        if unique[i]:
            for j in range(m):
                if dd[i * m + j] <= bw:
                    unique[j] = False
            unique[i] = True
    centers_out = List[Float32]()
    var kc = 0
    for i in range(m):
        if unique[i]:
            for f in range(d):
                centers_out.append(sorted[i * d + f])
            kc += 1
    var cs2 = ops.put(centers_out)
    var ls = ops.zeros_i(n)
    var ds = ops.zeros(n)
    ops.nearest(xs, n, cs2, kc, d, ls, ds)
    ops.sqrt(ds, n)
    labels = ops.get_i(ls, n)
    if not cluster_all:
        var dist = ops.get(ds, n)
        for r in range(n):
            if not (dist[r] <= bw):
                labels[r] = Int32(-1)
