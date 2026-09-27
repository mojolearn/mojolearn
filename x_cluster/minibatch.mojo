# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MiniBatchKMeans (lane/algos-cluster). Reference: scikit-learn
`sklearn/cluster/_kmeans.py` (MiniBatchKMeans.fit :2056-2210,
`_mini_batch_step` :1566-1684, `_mini_batch_convergence`) and
`_k_means_minibatch.pyx::update_center_dense`.

One generic driver over `ClusterOps`: the batch assignment (the n x k
distance work) is the device's; the batch draws, the center update in
sklearn's order (`c * w`, `+ x` in batch order, `w += wsum`, `* (1 / w)`),
the random reassignment and the early stop are host code from this one
source in both bindings.

THE BATCH ORDER COMES FROM THE SEED: `SplitMix64(seed)` draws the validation
rows, then per init the init rows and the k-means++ picks, then per step the
batch rows (uniform with replacement, unit weights) and any reassignment.
Not scikit-learn's Mersenne Twister stream, so a fit agrees with sklearn's
at a tolerance, never bit for bit (NOT_IMPLEMENTED.tsv)."""
from checks.numerics import ftz, identical_div, identical_mul, identical_mul64
from x_cluster.bodies import SplitMix64
from x_cluster.common import gather_rows, greedy_kmeans_pp, nearest_all, sum_f64
from x_cluster.ops import ClusterOps


@fieldwise_init
struct MiniBatchParams(Copyable, Movable):
    var k: Int
    var max_iter: Int
    var batch_size: Int
    var tol: Float64
    var max_no_improvement: Int  # < 0 means None
    var init_size: Int  # <= 0 means the default
    var n_init: Int
    var reassignment_ratio: Float64
    var seed: UInt64
    var has_init: Bool  # centers passed in (init array): n_init is 1


@fieldwise_init
struct MiniBatchResult(Copyable, Movable):
    var inertia: Float64
    var n_steps: Int
    var n_iter: Int


def minibatch_fit[O: ClusterOps](
    mut ops: O, x: List[Float32], n: Int, d: Int, p: MiniBatchParams,
    mut centers: List[Float32], mut labels: List[Int32], mut counts: List[Float32],
) raises -> MiniBatchResult:
    """Fit; `centers` holds the init array when `p.has_init`, and is
    overwritten with the fitted centers."""
    var k = p.k
    if k < 1 or k > n:
        raise Error("MiniBatchKMeans: n_samples=" + String(n) + " should be >= n_clusters=" + String(k))
    if p.batch_size < 1:
        raise Error("MiniBatchKMeans: batch_size must be >= 1")
    if p.reassignment_ratio < 0:
        raise Error("reassignment_ratio should be >= 0")
    var batch = p.batch_size if p.batch_size < n else n
    var init_size = p.init_size
    if init_size <= 0:
        init_size = 3 * batch
        if init_size < k:
            init_size = 3 * k
    elif init_size < k:
        init_size = 3 * k
    if init_size > n:
        init_size = n
    var rng = SplitMix64(p.seed)
    var xs = ops.put(x)

    # validation rows, then n_init inits scored on them
    var vidx = List[Int](capacity=init_size)
    for _t in range(init_size):
        vidx.append(rng.below(n))
    var vslot = ops.put(gather_rows(x, d, vidx))
    var best = List[Float32]()
    var best_inertia = Float64(0)
    var n_init = 1 if p.has_init else p.n_init
    for it in range(n_init):
        var cand: List[Float32]
        if p.has_init:
            cand = centers.copy()
        else:
            var iidx = List[Int](capacity=init_size)
            for _t in range(init_size):
                iidx.append(rng.below(n))
            cand = greedy_kmeans_pp(ops, gather_rows(x, d, iidx), init_size, d, k, rng)
        var vl = List[Int32]()
        var vd = List[Float32]()
        nearest_all(ops, vslot, init_size, cand, k, d, vl, vd)
        var inertia = sum_f64(vd, init_size)
        if it == 0 or inertia < best_inertia:
            best_inertia = inertia
            best = cand^

    var c = best^
    var w = List[Float32](length=k, fill=Float32(0))
    var ewa = Float64(0)
    var have_ewa = False
    var ewa_min = Float64(0)
    var have_min = False
    var no_improvement = 0
    var since_reassign = 0
    var n_steps = (p.max_iter * n) // batch
    var bslot = ops.zeros(batch * d)
    var cslot = ops.zeros(k * d)
    var lslot = ops.zeros_i(batch)
    var dslot = ops.zeros(batch)
    var steps_done = 0
    for step in range(n_steps):
        var bidx = List[Int](capacity=batch)
        for _t in range(batch):
            bidx.append(rng.below(n))
        var bx = gather_rows(x, d, bidx)
        # _random_reassign(): counted BEFORE the step, as sklearn evaluates the argument
        since_reassign += batch
        var any_empty = False
        for j in range(k):
            if w[j] == Float32(0):
                any_empty = True
        var reassign = False
        if any_empty or since_reassign >= 10 * k:
            since_reassign = 0
            reassign = True
        ops.set(bslot, bx)
        ops.set(cslot, c)
        ops.nearest(bslot, batch, cslot, k, d, lslot, dslot)
        var bl = ops.get_i(lslot, batch)
        var bd = ops.get(dslot, batch)
        var batch_inertia = sum_f64(bd, batch)
        # update_center_dense, per center
        var c_new = c.copy()
        for j in range(k):
            var wsum = Float32(0)
            for t in range(batch):
                if Int(bl[t]) == j:
                    wsum = ftz(wsum + Float32(1))
            if wsum > Float32(0):
                for f in range(d):
                    c_new[j * d + f] = ftz(identical_mul(c[j * d + f], w[j]))
                for t in range(batch):
                    if Int(bl[t]) == j:
                        for f in range(d):
                            c_new[j * d + f] = ftz(c_new[j * d + f] + ftz(bx[t * d + f]))
                w[j] = ftz(w[j] + wsum)
                var alpha = ftz(identical_div(Float32(1), w[j]))
                for f in range(d):
                    c_new[j * d + f] = ftz(identical_mul(c_new[j * d + f], alpha))
        if reassign and p.reassignment_ratio > 0:
            var wmax = w[0]
            for j in range(1, k):
                if w[j] > wmax:
                    wmax = w[j]
            var thr = identical_mul64(Float64(p.reassignment_ratio), Float64(wmax))
            var to = List[Bool](length=k, fill=False)
            var nre = 0
            for j in range(k):
                if Float64(w[j]) < thr:
                    to[j] = True
                    nre += 1
            var half = batch // 2
            if 2 * nre > batch:
                # np.argsort(weight_sums)[half:] stay: a stable ascending sort by weight
                var order = List[Int](capacity=k)
                for j in range(k):
                    order.append(j)
                for a in range(1, k):
                    var b = a
                    while b > 0 and w[order[b - 1]] > w[order[b]]:
                        var tmp = order[b - 1]
                        order[b - 1] = order[b]
                        order[b] = tmp
                        b -= 1
                for q in range(half, k):
                    to[order[q]] = False
                nre = 0
                for j in range(k):
                    if to[j]:
                        nre += 1
            if nre > 0:
                # choice(batch, nre, replace=False): a partial Fisher-Yates
                var pool = List[Int](capacity=batch)
                for t in range(batch):
                    pool.append(t)
                var picked = List[Int](capacity=nre)
                for q in range(nre):
                    var r = q + rng.below(batch - q)
                    var tmp = pool[q]
                    pool[q] = pool[r]
                    pool[r] = tmp
                    picked.append(pool[q])
                var q = 0
                var wmin = Float32(0)
                var have = False
                for j in range(k):
                    if not to[j] and (not have or w[j] < wmin):
                        wmin = w[j]
                        have = True
                for j in range(k):
                    if to[j]:
                        for f in range(d):
                            c_new[j * d + f] = bx[picked[q] * d + f]
                        q += 1
                        w[j] = wmin
        var diff = Float64(0)
        if p.tol > 0:
            for t in range(k * d):
                var e = Float64(c_new[t]) - Float64(c[t])
                diff = diff + identical_mul64(e, e)
        c = c_new^
        steps_done = step + 1
        # _mini_batch_convergence
        var bi = batch_inertia / Float64(batch)
        if step == 0:
            continue
        if not have_ewa:
            ewa = bi
            have_ewa = True
        else:
            var alpha = identical_mul64(Float64(batch), 2.0) / Float64(n + 1)
            if alpha > 1:
                alpha = 1
            ewa = identical_mul64(ewa, 1 - alpha) + identical_mul64(bi, alpha)
        if p.tol > 0 and diff <= p.tol:
            break
        if not have_min or ewa < ewa_min:
            no_improvement = 0
            ewa_min = ewa
            have_min = True
        else:
            no_improvement += 1
        if p.max_no_improvement >= 0 and no_improvement >= p.max_no_improvement:
            break

    var dist = List[Float32]()
    nearest_all(ops, xs, n, c, k, d, labels, dist)
    centers = c^
    counts = w^
    var n_iter = (steps_done * batch + n - 1) // n
    return MiniBatchResult(sum_f64(dist, n), steps_done, n_iter)
