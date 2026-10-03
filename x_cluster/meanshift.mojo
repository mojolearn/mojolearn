# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MeanShift (lane/algos-cluster). Reference: scikit-learn
`sklearn/cluster/_mean_shift.py` (`estimate_bandwidth` :35-95,
`_mean_shift_single_seed` :97-120, `MeanShift.fit` :450-540).

The bandwidth, when not given, is the mean over the rows of the distance to
the `int(n * quantile)`-th nearest row (itself included): the n x n squared
distances and the row order statistic are the device's
(`bodies.kth_smallest_row`, an exact bisection on the float bits), the roots
and their sum the device's float-float fold (`post_bodies`), the mean one
host division. Every seed's shift loop
is ONE device thread (`bodies.meanshift_seed`), the flat kernel over all rows
in ascending order. The bin seeds and the post-processing are device
primitives (lane cgr2-cluster): the centers with a nonzero intensity kept
per distinct center (the last seed's intensity, a dict), sorted by
(intensity, coordinates) descending, a center within the bandwidth of a kept
one dropped, then the nearest center per row (the lowest index on a tie), -1
beyond the bandwidth when not `cluster_all`."""
from checks.numerics import ftz, identical_mul
from x_cluster.ops import ClusterOps
from x_cluster.post_bodies import FM_SQRT
from x_cluster.optics import dist_slot


def estimate_bandwidth_ops[O: ClusterOps](
    mut ops: O, x: List[Float32], n: Int, d: Int, quantile: Float64
) raises -> Float32:
    var k = Int(Float64(n) * quantile)
    if k < 1:
        k = 1
    var xs = ops.put(x)
    var dm = dist_slot(ops, n * n)
    ops.sqdist(xs, n, xs, n, d, dm)
    var kt = ops.zeros(n)
    ops.kth(dm, n, n, k, kt)
    # the mean of the roots: the float-float fold on the device
    var acc = ops.sum_ff(kt, -1, -1, n, FM_SQRT)
    return Float32(acc / Float64(n))


def meanshift_fit[O: ClusterOps](
    mut ops: O, x: List[Float32], n: Int, d: Int, bandwidth_in: Float32, seeds_in: List[Float32],
    n_seeds_in: Int, cluster_all: Bool, max_iter: Int, bin_seeding: Bool, min_bin_freq: Int,
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
    var xs = ops.put(x)
    var cs: Int
    if ns == 0 and bin_seeding and bw != Float32(0):
        # sklearn `get_bin_seeds` on the device: every row's bin, the first
        # row of each distinct bin, the bins with at least min_bin_freq rows
        # in first-seen order (`ops.bin_seeds`); the rows themselves when
        # every row is its own bin
        var bs = ops.alloc(n * d)
        ns = ops.bin_seeds(xs, n, d, bw, min_bin_freq, bs)
        if ns == 0:
            raise Error("No point was within bandwidth=" + String(bw) + " of any seed. Try a different seeding strategy or increase the bandwidth.")
        if ns == n:
            cs = ops.put(x)
        else:
            cs = bs
    elif ns == 0:
        ns = n
        cs = ops.put(x)
    else:
        cs = ops.put(seeds)
    var sc = ops.zeros(ns * d)
    var it_s = ops.zeros_i(ns)
    var ic_s = ops.zeros_i(ns)
    var stop = ftz(identical_mul(Float32(1e-3), bw))
    ops.meanshift(xs, n, d, bw, stop, max_iter, cs, ns, sc, ic_s, it_s)
    # THE POST-PROCESSING ON THE DEVICE (lane cgr2-cluster): the distinct
    # centers (the dict: first insertion, last intensity), their sort by
    # (intensity, coordinates) descending, the radius suppression by rounds
    # (the lowest undecided center is kept and drops the undecided ones
    # within the bandwidth: the reference's greedy loop), the labels.
    var us = ops.alloc(ns * d)
    var m = ops.ms_unique(cs, ic_s, it_s, ns, d, us, n_iter)
    if m == 0:
        raise Error("No point was within bandwidth=" + String(bw) + " of any seed. Try a different seeding strategy or increase the bandwidth.")
    var dd_s = ops.alloc(m * m)
    ops.sqdist(us, m, us, m, d, dd_s)
    ops.sqrt(dd_s, m * m)
    var cs2 = ops.alloc(m * d)
    var kc = ops.ms_suppress(us, dd_s, m, d, bw, cs2)
    var ls = ops.zeros_i(n)
    var ds = ops.zeros(n)
    ops.nearest(xs, n, cs2, kc, d, ls, ds)
    ops.sqrt(ds, n)
    if not cluster_all:
        ops.ms_noise(ls, ds, n, bw)
    centers_out = ops.get(cs2, kc * d)
    labels = ops.get_i(ls, n)
