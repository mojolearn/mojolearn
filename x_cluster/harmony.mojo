# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Harmony batch integration (lane sc-stats-1, 2026-11-28): ONE generic driver over `ClusterOps`, run on the device
column (DeviceOps) and the host column (HostOps) from this one source.

Reference: Korsunsky et al. 2019, "Fast, sensitive and accurate integration of single-cell data with Harmony";
harmonypy 0.0.10 (`run_harmony`, `Harmony.harmonize`). The steps and their order are harmonypy's:

  Zc = Z with unit rows; Y = k-means centroids of Zc (the repository k-means, `ClusterOps.kmeans_slot`, k-means++
  init, n_init 10, max_iter 25, seeded from random_state), unit rows; dist = ||Zc_i - Y_j||^2 (= 2 (1 - Y_j . Zc_i)
  for unit rows, `ClusterOps.sqdist`); R = softmax(-dist / sigma) per cell; E, O; objective. Then up to
  max_iter_harmony rounds of: up to max_iter_kmeans clustering iterations (Y = R^T Zc with unit rows; dist; R
  updated block by block with the diversity penalty sum over the cell's levels of ((E + 1) / (O + 1))^theta;
  the objective, converged when its 3-iteration window moves less than epsilon_cluster), then the mixture of
  experts ridge correction (per cluster j: W_j = (Phi diag(R_j) Phi^T + diag(0, lamb))^-1 Phi diag(R_j) Z,
  W_j's intercept row zeroed, Z_corr = Z - sum_j W_j^T Phi diag(R_j)), Zc = Z_corr with unit rows; converged when
  the objective's relative decrease is below epsilon_harmony.

What differs from harmonypy (each a determinism or parallel form, stated in the public docstring too):
  - The update blocks follow an affine order s -> (a s + c) mod n drawn per update from random_state (a coprime
    to n), in place of numpy's shuffle; the blocks are harmonypy's `array_split` sizes.
  - After each update's blocks, E and O are recomputed from R in one pass (harmonypy keeps the incremental
    updates); this removes the float32 drift of the repeated subtract / add.
  - Zc is Z over its row norms (harmonypy first divides by the row maximum, which only flips the sign of a row
    whose maximum is negative); the distance is the squared distance of unit rows.
  - float32 throughout with pinned fold orders; the objective is a float-float fold (`sum_ff`) read as float64.

Host control (the round loops, the convergence tests on the objective, the per-level probabilities) is this file,
compiled into both bindings; every pass over cells is one primitive of the ops.
"""
from x_cluster.ops import ClusterOps
from x_cluster.bodies import SplitMix64
from x_cluster.post_bodies import FM_VAL
from cluster.impl.kmeans_params import INIT_KMEANS_PLUS_PLUS
from checks.numerics import portable_exp64
from core.harmony_api import HarmonyFitResult

comptime HM_WINDOW = 3
comptime HM_KMEANS_INIT_ITER = 25
comptime HM_KMEANS_INIT_TOL = 1.0e-4
comptime HM_KMEANS_N_INIT = 10


@fieldwise_init
struct HarmonyParams(Copyable, Movable):
    """n cells, d dimensions, k clusters, c covariates over nb levels in all (each cell's c codes are global level
    indices in [0, nb), every level held by at least one cell). theta and lamb: one value per level."""
    var n: Int
    var d: Int
    var k: Int
    var c: Int
    var nb: Int
    var max_iter_harmony: Int
    var max_iter_kmeans: Int
    var seed: UInt64
    var sigma: Float64
    var tau: Float64
    var block_size: Float64
    var epsilon_cluster: Float64
    var epsilon_harmony: Float64
    var theta: List[Float64]
    var lamb: List[Float64]


def _gcd(a_in: Int, b_in: Int) -> Int:
    var a = a_in
    var b = b_in
    while b != 0:
        var t = a % b
        a = b
        b = t
    return a


def _block_order(mut rng: SplitMix64, n: Int) -> Tuple[Int, Int]:
    """(a, c) of one update's cell order s -> (a s + c) mod n, a coprime to n (a = 1 for n = 1)."""
    if n <= 1:
        return (1, 0)
    var a = 1 + rng.below(n - 1)
    while _gcd(a, n) != 1:  # small-loop(candidates: a few draws until one is coprime to n)
        a = 1 + rng.below(n - 1)
    var c0 = rng.below(n)
    return (a, c0)


def harmony_blocks(n: Int, block_size: Float64) -> Int:
    """harmonypy's n_blocks = ceil(1 / block_size), at most n."""
    var q = 1.0 / block_size
    var nb = Int(q)
    if Float64(nb) < q:
        nb += 1
    return max(1, min(nb, n))


def _objective[O: ClusterOps](
    mut ops: O, r: Int, dist: Int, e: Int, o: Int, codes: Int, theta: Int, n: Int, k: Int, c: Int, nb: Int,
    sigma: Float32, cells: Int,
) raises -> Float64:
    ops.hm_objective_cells(r, dist, e, o, codes, theta, n, k, c, nb, sigma, cells)
    return ops.sum_ff(cells, -1, -1, n, FM_VAL)


def harmony_fit[O: ClusterOps](
    mut ops: O, z: List[Float32], codes: List[Int32], offsets: List[Int32], p: HarmonyParams
) raises -> HarmonyFitResult:
    """`codes`: c x n per-covariate level codes, `offsets`: each covariate's first global level."""
    var n = p.n
    var d = p.d
    var k = p.k
    var c = p.c
    var nb = p.nb
    var b1 = nb + 1
    var sigma = Float32(p.sigma)
    var zs = ops.put(z)
    var raw = ops.put_i(codes)
    var offs = ops.put_i(offsets)
    var cs = ops.zeros_i(n * c)
    ops.hm_codes(raw, offs, n, c, cs)
    # the level sizes N_b: pr = N_b / n; theta decays with tau (harmonypy: theta (1 - exp(-(N_b / (k tau))^2)))
    var cnt_s = ops.zeros(nb)
    ops.hm_counts(cs, n, c, nb, cnt_s)
    var cnt = ops.get(cnt_s, nb)
    var pr = List[Float32](length=nb, fill=Float32(0))
    var th = List[Float32](length=nb, fill=Float32(0))
    var lm = List[Float32](length=nb, fill=Float32(0))
    for b in range(nb):  # driver-loop: over the batch levels (k-sized parameters)
        pr[b] = Float32(Float64(cnt[b]) / Float64(n))
        var theta_b = p.theta[b]
        if p.tau > 0.0:
            var x = Float64(cnt[b]) / (Float64(k) * p.tau)
            theta_b = theta_b * (1.0 - portable_exp64(-(x * x)))
        th[b] = Float32(theta_b)
        lm[b] = Float32(p.lamb[b])
    var prs = ops.put(pr)
    var ths = ops.put(th)
    var lms = ops.put(lm)
    # init_cluster
    var zc = ops.zeros(n * d)
    ops.hm_rownorm(zs, n, d, zc)
    var centers = List[Float32]()
    var labels = List[Int32]()
    _ = ops.kmeans_slot(zc, n, d, k, HM_KMEANS_INIT_ITER, HM_KMEANS_INIT_TOL, p.seed, HM_KMEANS_N_INIT,
                        INIT_KMEANS_PLUS_PLUS, centers, labels)
    var ys = ops.put(centers)
    ops.hm_rownorm(ys, k, d, ys)
    var dist = ops.zeros(n * k)
    ops.sqdist(zc, n, ys, k, d, dist)
    var rs = ops.zeros(n * k)
    ops.hm_scale(dist, n, k, sigma, rs, True)
    var es = ops.zeros(k * nb)
    var osl = ops.zeros(k * nb)
    ops.hm_eo(rs, cs, prs, n, k, c, nb, es, osl)
    var cells = ops.zeros(n)
    var res = HarmonyFitResult()
    var obj_kmeans = List[Float64]()
    obj_kmeans.append(_objective(ops, rs, dist, es, osl, cs, ths, n, k, c, nb, sigma, cells))
    res.objective.append(obj_kmeans[0])
    var sd = ops.zeros(n * k)
    var zcorr = ops.zeros(n * d)
    var am = ops.zeros(k * b1 * b1)
    var rhs = ops.zeros(k * b1 * d)
    var w = ops.zeros(k * b1 * d)
    var rng = SplitMix64(p.seed ^ UInt64(0x48524D59))
    var nblocks = harmony_blocks(n, p.block_size)
    var base = n // nblocks
    var extra = n - base * nblocks
    for rnd in range(1, p.max_iter_harmony + 1):  # driver-loop: over Harmony rounds
        # cluster()
        for it in range(p.max_iter_kmeans):  # driver-loop: over the clustering iterations
            ops.hm_centroids(rs, zc, n, k, d, ys)
            ops.hm_rownorm(ys, k, d, ys)
            ops.sqdist(zc, n, ys, k, d, dist)
            # update_R
            ops.hm_scale(dist, n, k, sigma, sd, False)
            var order = _block_order(rng, n)
            var lo = 0
            for bl in range(nblocks):  # driver-loop: over the update blocks (sequential by the algorithm)
                var hi = lo + base + (1 if bl < extra else 0)
                if hi > lo:
                    ops.hm_block_eo(rs, cs, prs, n, k, c, nb, order[0], order[1], lo, hi, Float32(-1), es, osl)
                    ops.hm_block_r(sd, es, osl, cs, ths, n, k, c, nb, order[0], order[1], lo, hi, rs)
                    ops.hm_block_eo(rs, cs, prs, n, k, c, nb, order[0], order[1], lo, hi, Float32(1), es, osl)
                lo = hi
            ops.hm_eo(rs, cs, prs, n, k, c, nb, es, osl)
            obj_kmeans.append(_objective(ops, rs, dist, es, osl, cs, ths, n, k, c, nb, sigma, cells))
            if it > HM_WINDOW:
                var last = len(obj_kmeans)
                var old = 0.0
                var fresh = 0.0
                for wi in range(HM_WINDOW):  # small-loop(window: three objective values)
                    old += obj_kmeans[last - 2 - wi]
                    fresh += obj_kmeans[last - 1 - wi]
                if old != 0.0 and abs(old - fresh) / abs(old) < p.epsilon_cluster:
                    break
        res.objective.append(obj_kmeans[len(obj_kmeans) - 1])
        # moe_correct_ridge
        ops.hm_ridge(rs, zs, cs, lms, n, k, d, c, b1, am, rhs)
        ops.hm_solve(am, rhs, k, b1, d, w)
        ops.hm_correct(zs, rs, w, cs, n, k, d, c, b1, zcorr)
        ops.hm_rownorm(zcorr, n, d, zc)
        res.n_iter = rnd
        var h = len(res.objective)
        var o_old = res.objective[h - 2]
        var o_new = res.objective[h - 1]
        if o_old != 0.0 and (o_old - o_new) / abs(o_old) < p.epsilon_harmony:
            res.converged = True
            break
    res.z_corr = ops.get(zcorr, n * d)
    res.r = ops.get(rs, n * k)
    res.y = ops.get(ys, k * d)
    _ = labels^
    return res^
