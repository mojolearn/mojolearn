# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CLUSTER LANE'S ENTRY POINTS, generic over the column (lane/algos-cluster).

Both bindings (`bindings/_mojolearn_x_cluster.mojo`, `..._host.mojo`) read
their Python arguments, call ONE of these with their own `ClusterOps`, and
hand the `ClusterOut` back. The integer and float parameter lists are
documented per entry and mirrored in `python/mojolearn/_x_cluster_impl.py`."""
from x_cluster.affinity import affinity_fit
from x_cluster.bgmm import BgmmPriors, BgmmState, bgmm_constants, bgmm_fit, bgmm_score, bgmm_weights
from x_cluster.bisect import BisectTree, bisect_fit, bisect_predict
from checks.numerics import identical_mul64
from x_cluster.common import distances_to, nearest_all
from x_cluster.meanshift import meanshift_fit
from x_cluster.minibatch import MiniBatchParams, minibatch_fit
from x_cluster.ops import ClusterOps
from x_cluster.optics import optics_dbscan_labels, optics_graph, optics_xi_clusters, optics_xi_labels
from x_cluster.out import ClusterOut


def nearest_entry[O: ClusterOps](mut ops: O, x: List[Float32], c: List[Float32], ip: List[Int]) raises -> ClusterOut:
    """ip = [n, k, d]. i = [labels], f = [squared distance to the nearest]."""
    var n = ip[0]
    var k = ip[1]
    var d = ip[2]
    var xs = ops.put(x)
    var labels = List[Int32]()
    var dist = List[Float32]()
    nearest_all(ops, xs, n, c, k, d, labels, dist)
    var out = ClusterOut()
    out.i.append(labels^)
    out.f.append(dist^)
    return out^


def distances_entry[O: ClusterOps](mut ops: O, x: List[Float32], c: List[Float32], ip: List[Int]) raises -> ClusterOut:
    """ip = [n, k, d]. f = [n x k euclidean distances]."""
    var n = ip[0]
    var k = ip[1]
    var d = ip[2]
    var xs = ops.put(x)
    var out = ClusterOut()
    out.f.append(distances_to(ops, xs, n, c, k, d))
    return out^


def minibatch_entry[O: ClusterOps](
    mut ops: O, x: List[Float32], init: List[Float32], ip: List[Int], fp: List[Float64]
) raises -> ClusterOut:
    """ip = [n, d, k, max_iter, batch_size, max_no_improvement (-1 None),
    init_size (0 default), n_init, has_init, seed]; fp = [tol,
    reassignment_ratio]. f = [centers, counts], i = [labels],
    s = [inertia, n_steps, n_iter]."""
    var n = ip[0]
    var d = ip[1]
    var p = MiniBatchParams(
        k=ip[2], max_iter=ip[3], batch_size=ip[4], tol=fp[0], max_no_improvement=ip[5],
        init_size=ip[6], n_init=ip[7], reassignment_ratio=fp[1], seed=UInt64(ip[9]),
        has_init=ip[8] != 0,
    )
    var centers = init.copy()
    var labels = List[Int32]()
    var counts = List[Float32]()
    var r = minibatch_fit(ops, x, n, d, p, centers, labels, counts)
    var out = ClusterOut()
    out.f.append(centers^)
    out.f.append(counts^)
    out.i.append(labels^)
    out.s.append(r.inertia)
    out.s.append(Float64(r.n_steps))
    out.s.append(Float64(r.n_iter))
    return out^


def bisect_entry[O: ClusterOps](mut ops: O, x: List[Float32], ip: List[Int], fp: List[Float64]) raises -> ClusterOut:
    """ip = [n, d, k, n_init, init (0 k-means++, 1 random), max_iter, seed,
    largest_cluster]; fp = [tol]. f = [centers, tree_centers], i = [labels,
    nodes (left, right, label) per node], s = [inertia]."""
    var n = ip[0]
    var d = ip[1]
    var tree = BisectTree()
    var labels = List[Int32]()
    var centers = List[Float32]()
    var inertia = bisect_fit(
        ops, x, n, d, ip[2], ip[3], ip[4], ip[5], fp[0], UInt64(ip[6]), ip[7] != 0, tree, labels, centers
    )
    var nodes = List[Int32]()
    for t in range(len(tree.left)):
        nodes.append(Int32(tree.left[t]))
        nodes.append(Int32(tree.right[t]))
        nodes.append(Int32(tree.label[t]))
    var out = ClusterOut()
    out.f.append(centers^)
    out.f.append(tree.centers.copy())
    out.i.append(labels^)
    out.i.append(nodes^)
    out.s.append(inertia)
    return out^


def bisect_predict_entry[O: ClusterOps](
    mut ops: O, x: List[Float32], tree_centers: List[Float32], ip: List[Int]
) raises -> ClusterOut:
    """ip = [n, d, then the nodes (left, right, label) flattened]. i = [labels]."""
    var nodes = List[Int32]()
    for t in range(2, len(ip)):
        nodes.append(Int32(ip[t]))
    var out = ClusterOut()
    out.i.append(bisect_predict(ops, x, ip[0], ip[1], tree_centers, nodes))
    return out^


def meanshift_entry[O: ClusterOps](
    mut ops: O, x: List[Float32], seeds: List[Float32], ip: List[Int], fp: List[Float64]
) raises -> ClusterOut:
    """ip = [n, d, n_seeds (0: the rows), cluster_all, max_iter]; fp =
    [bandwidth (<= 0: estimated)]. f = [centers], i = [labels],
    s = [bandwidth, n_iter, n_centers]."""
    var n = ip[0]
    var d = ip[1]
    var centers = List[Float32]()
    var labels = List[Int32]()
    var bw = Float32(0)
    var n_iter = 0
    meanshift_fit(ops, x, n, d, Float32(fp[0]), seeds, ip[2], ip[3] != 0, ip[4], centers, labels, bw, n_iter)
    var out = ClusterOut()
    var kc = len(centers) // d
    out.f.append(centers^)
    out.i.append(labels^)
    out.s.append(Float64(bw))
    out.s.append(Float64(n_iter))
    out.s.append(Float64(kc))
    return out^


def _i32(v: List[Int]) -> List[Int32]:
    var out = List[Int32](capacity=len(v))
    for t in v:
        out.append(Int32(t))
    return out^


def optics_entry[O: ClusterOps](mut ops: O, x: List[Float32], ip: List[Int], fp: List[Float64]) raises -> ClusterOut:
    """ip = [n, d, min_samples, min_cluster_size, method (0 xi, 1 dbscan),
    predecessor_correction]; fp = [max_eps, xi, eps]. f = [core_distances,
    reachability], i = [ordering, predecessor, labels, clusters (start, end)
    flattened]."""
    var n = ip[0]
    var d = ip[1]
    var ordering = List[Int]()
    var core = List[Float32]()
    var reach = List[Float32]()
    var pred = List[Int]()
    optics_graph(ops, x, n, d, ip[2], Float32(fp[0]), ordering, core, reach, pred)
    var labels: List[Int32]
    var clusters = List[Int]()
    if ip[4] == 0:
        clusters = optics_xi_clusters(reach, pred, ordering, fp[1], ip[2], ip[3], ip[5] != 0)
        labels = optics_xi_labels(ordering, clusters)
    else:
        labels = optics_dbscan_labels(reach, core, ordering, Float32(fp[2]))
    var out = ClusterOut()
    out.f.append(core^)
    out.f.append(reach^)
    out.i.append(_i32(ordering))
    out.i.append(_i32(pred))
    out.i.append(labels^)
    out.i.append(_i32(clusters))
    return out^


def affinity_entry[O: ClusterOps](
    mut ops: O, x: List[Float32], pref: List[Float32], ip: List[Int], fp: List[Float64]
) raises -> ClusterOut:
    """ip = [n, d, precomputed, pref_mode (0 median, 1 scalar, 2 array),
    max_iter, convergence_iter, seed]; fp = [damping, preference scalar].
    i = [cluster_centers_indices, labels], f = [affinity_matrix],
    s = [n_iter]."""
    var centers = List[Int32]()
    var labels = List[Int32]()
    var n_iter = 0
    var aff = List[Float32]()
    affinity_fit(
        ops, x, ip[0], ip[1], ip[2] != 0, ip[3], Float32(fp[1]), pref, Float32(fp[0]), ip[4], ip[5],
        UInt64(ip[6]), centers, labels, n_iter, aff,
    )
    var out = ClusterOut()
    out.i.append(centers^)
    out.i.append(labels^)
    out.f.append(aff^)
    out.s.append(Float64(n_iter))
    return out^


def _f32_of(v: List[Float64]) -> List[Float32]:
    var out = List[Float32](capacity=len(v))
    for t in v:
        out.append(Float32(t))
    return out^


def bgmm_entry[O: ClusterOps](
    mut ops: O, x: List[Float32], a: List[Float32], ip: List[Int], fp: List[Float64]
) raises -> ClusterOut:
    """ip = [n, d, n_components, dirichlet_process, max_iter, n_init,
    init_random, seed, has_mean_prior, has_covariance_prior]; fp =
    [weight_concentration_prior (< 0 None), mean_precision_prior (< 0 None),
    degrees_of_freedom_prior (< 0 None), reg_covar, tol]; a = the mean prior
    (d) then the covariance prior (d x d), each when given.
    f = [weights, means, covariances, precisions_cholesky, wc0, wc1,
    mean_precision, degrees_of_freedom, constants, mean_prior,
    covariance_prior], i = [labels], s = [lower_bound, n_iter, converged,
    wcp, mpp, dofp]."""
    var n = ip[0]
    var d = ip[1]
    var kc = ip[2]
    var off = 0
    var mean_prior = List[Float64](capacity=d)
    if ip[8] != 0:
        for f in range(d):
            mean_prior.append(Float64(a[f]))
        off = d
    else:
        for f in range(d):
            var acc = Float64(0)
            for r in range(n):
                acc = acc + Float64(x[r * d + f])
            mean_prior.append(acc / Float64(n))
    var cov_prior = List[Float64](length=d * d, fill=0)
    if ip[9] != 0:
        for t in range(d * d):
            cov_prior[t] = Float64(a[off + t])
    else:
        # np.cov(X.T): ddof 1, one ascending Float64 chain per cell
        var mean = List[Float64](capacity=d)
        for f in range(d):
            var acc = Float64(0)
            for r in range(n):
                acc = acc + Float64(x[r * d + f])
            mean.append(acc / Float64(n))
        for p in range(d):
            for q in range(d):
                var acc = Float64(0)
                for r in range(n):
                    acc = acc + identical_mul64(Float64(x[r * d + p]) - mean[p], Float64(x[r * d + q]) - mean[q])
                cov_prior[p * d + q] = acc / Float64(n - 1 if n > 1 else 1)
    var wcp = fp[0] if fp[0] >= 0 else 1.0 / Float64(kc)
    var mpp = fp[1] if fp[1] >= 0 else 1.0
    var dofp = fp[2] if fp[2] >= 0 else Float64(d)
    if dofp <= Float64(d) - 1.0:
        raise Error("The parameter 'degrees_of_freedom_prior' should be greater than " + String(d - 1) + ", but got " + String(dofp) + ".")
    var pr = BgmmPriors(kc, d, ip[3] != 0, wcp, mpp, mean_prior.copy(), dofp, cov_prior.copy())
    var best = BgmmState()
    var labels = List[Int32]()
    var r = bgmm_fit(ops, x, n, d, pr, Float32(fp[3]), fp[4], ip[4], ip[5], ip[6] != 0, UInt64(ip[7]), best, labels)
    var out = ClusterOut()
    out.f.append(_f32_of(bgmm_weights(pr, best)))
    out.f.append(_f32_of(best.means))
    out.f.append(_f32_of(best.cov))
    out.f.append(_f32_of(best.pchol))
    out.f.append(_f32_of(best.wc0))
    out.f.append(_f32_of(best.wc1))
    out.f.append(_f32_of(best.mean_prec))
    out.f.append(_f32_of(best.dof))
    out.f.append(bgmm_constants(pr, best))
    out.f.append(_f32_of(mean_prior))
    out.f.append(_f32_of(cov_prior))
    out.i.append(labels^)
    out.s.append(r.lower_bound)
    out.s.append(Float64(r.n_iter))
    out.s.append(Float64(1) if r.converged else Float64(0))
    out.s.append(wcp)
    out.s.append(mpp)
    out.s.append(dofp)
    return out^


def bgmm_score_entry[O: ClusterOps](
    mut ops: O, x: List[Float32], a: List[Float32], ip: List[Int]
) raises -> ClusterOut:
    """ip = [n, d, n_components]; a = means (k x d), precisions_cholesky
    (k x d x d), constants (k). f = [log_resp (n x k), log_prob_norm (n)]."""
    var n = ip[0]
    var d = ip[1]
    var kc = ip[2]
    var means = List[Float32](capacity=kc * d)
    var pchol = List[Float32](capacity=kc * d * d)
    var c = List[Float32](capacity=kc)
    for t in range(kc * d):
        means.append(a[t])
    for t in range(kc * d * d):
        pchol.append(a[kc * d + t])
    for t in range(kc):
        c.append(a[kc * d + kc * d * d + t])
    var lr = List[Float32]()
    var lpn = List[Float32]()
    bgmm_score(ops, x, n, d, kc, means, pchol, c, lr, lpn)
    var out = ClusterOut()
    out.f.append(lr^)
    out.f.append(lpn^)
    return out^


# ---------------------------------------------------------------- dispatcher
comptime ENTRY_NEAREST = 0
comptime ENTRY_DISTANCES = 1
comptime ENTRY_MINIBATCH = 2
comptime ENTRY_BISECT = 3
comptime ENTRY_BISECT_PREDICT = 4
comptime ENTRY_MEANSHIFT = 5
comptime ENTRY_OPTICS = 6
comptime ENTRY_AFFINITY = 7
comptime ENTRY_BGMM = 8
comptime ENTRY_BGMM_SCORE = 9


def run_entry[O: ClusterOps](
    mut ops: O, which: Int, x: List[Float32], a: List[Float32], ip: List[Int], fp: List[Float64]
) raises -> ClusterOut:
    """THE ONE CALL both bindings export as `x_cluster_call(which, x_addr,
    x_len, a_addr, a_len, ip, fp)`; `which` names the entry above."""
    if which == ENTRY_NEAREST:
        return nearest_entry(ops, x, a, ip)
    if which == ENTRY_DISTANCES:
        return distances_entry(ops, x, a, ip)
    if which == ENTRY_MINIBATCH:
        return minibatch_entry(ops, x, a, ip, fp)
    if which == ENTRY_BISECT:
        return bisect_entry(ops, x, ip, fp)
    if which == ENTRY_BISECT_PREDICT:
        return bisect_predict_entry(ops, x, a, ip)
    if which == ENTRY_MEANSHIFT:
        return meanshift_entry(ops, x, a, ip, fp)
    if which == ENTRY_OPTICS:
        return optics_entry(ops, x, ip, fp)
    if which == ENTRY_AFFINITY:
        return affinity_entry(ops, x, a, ip, fp)
    if which == ENTRY_BGMM:
        return bgmm_entry(ops, x, a, ip, fp)
    if which == ENTRY_BGMM_SCORE:
        return bgmm_score_entry(ops, x, a, ip)
    raise Error("x_cluster: unknown entry " + String(which))
