# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Harmony's per-cell bodies (lane sc-stats-1, 2026-11-28), ONE spelling each: the device kernels
(x_cluster/harmony_kernels.mojo, one thread per cell) and the host column (x_cluster/host/host_ops.mojo, the same
body over `host_cells`) both call them, so CPU == GPU bit for bit by construction.

Reference: Korsunsky et al. 2019 (Nature Methods 16, 1289); harmonypy 0.0.10 `harmony.py` (`Harmony.init_cluster`,
`cluster`, `update_R`, `compute_objective`, `moe_correct_ridge`).

Layouts (row-major float32 slots, int32 codes): Z, Zc n x d; R, dist, S n x K; codes n x C (each cell's level of
every covariate, as one global level index in [0, B)); pr, theta, lamb B; E, O K x B; A K x B1 x B1 and RHS, W
K x B1 x d with B1 = B + 1 (the intercept first). Every fold runs its index ascending; every product, quotient and
transcendental goes through the IDENTICAL seams (checks/numerics.mojo) and every word through `ftz`.

An update block is the positions [lo, hi) of the cell order s -> (a * s + c) mod n (`hm_cell`), an affine
permutation with a coprime to n drawn per round from the seed (x_cluster/harmony.mojo `_block_order`): the blocks
are known on both columns without a host pass over the cells.
"""
from checks.numerics import ftz, identical_div, identical_exp, identical_log, identical_mul, identical_pow, identical_sqrt
from x_cluster.bodies import FPtr, IPtr


@always_inline
def _m(a: Float32, b: Float32) -> Float32:
    return ftz(identical_mul(ftz(a), ftz(b)))


@always_inline
def _q(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(ftz(a), ftz(b)))


@always_inline
def _a(a: Float32, b: Float32) -> Float32:
    return ftz(ftz(a) + ftz(b))


@always_inline
def _s(a: Float32, b: Float32) -> Float32:
    return ftz(ftz(a) - ftz(b))


@always_inline
def hm_cell(s: Int, a: Int, c0: Int, n: Int) -> Int:
    """The cell at position s of the round's order."""
    return (a * s + c0) % n


@always_inline
def hm_phi(codes: IPtr, i: Int, c: Int, u: Int) -> Bool:
    """Row u of Harmony's design for cell i: u = 0 the intercept (always 1), u = b + 1 level b's indicator."""
    if u == 0:
        return True
    for cc in range(c):
        if Int(codes[i * c + cc]) == u - 1:
            return True
    return False


@always_inline
def hm_rownorm_row(src: FPtr, dst: FPtr, d: Int, i: Int):
    """dst row i = src row i / its L2 norm (the squares folded ascending); a zero row stays zero. In place is
    allowed (each word is read before it is written)."""
    var s = Float32(0)
    for f in range(d):
        var v = src[i * d + f]
        s = _a(s, _m(v, v))
    var nrm = ftz(identical_sqrt(s))
    for f in range(d):
        if nrm > Float32(0):
            dst[i * d + f] = _q(src[i * d + f], nrm)
        else:
            dst[i * d + f] = Float32(0)


@always_inline
def hm_count_cell(codes: IPtr, n: Int, c: Int, dst: FPtr, b: Int):
    """dst[b] = the cells at level b (over every covariate's column; the levels are disjoint)."""
    var cnt = 0
    for i in range(n):
        for cc in range(c):
            if Int(codes[i * c + cc]) == b:
                cnt += 1
    dst[b] = Float32(cnt)


@always_inline
def hm_scale_row(dist: FPtr, k: Int, sigma: Float32, dst: FPtr, normalize: Bool, i: Int):
    """v_j = -dist_ij / sigma, minus the row's max (the first on a tie), exp; with `normalize` divided by the row
    sum (harmonypy init_cluster's R; without it update_R's `_scale_dist`)."""
    var mx = Float32(0)
    for j in range(k):
        var v = _q(Float32(0) - dist[i * k + j], sigma)
        if j == 0 or v > mx:
            mx = v
    var tot = Float32(0)
    for j in range(k):
        var e = ftz(identical_exp(_s(_q(Float32(0) - dist[i * k + j], sigma), mx)))
        dst[i * k + j] = e
        tot = _a(tot, e)
    if normalize:
        for j in range(k):
            dst[i * k + j] = _q(dst[i * k + j], tot)


@always_inline
def hm_eo_cell(r: FPtr, codes: IPtr, pr: FPtr, n: Int, k: Int, c: Int, nb: Int, e: FPtr, o: FPtr, t: Int):
    """t = j * B + b: E[j, b] = (sum_i R[i, j]) * pr[b], O[j, b] = sum over the cells at level b of R[i, j]; the
    cells ascending."""
    var j = t // nb
    var b = t - j * nb
    var rs = Float32(0)
    var ob = Float32(0)
    for i in range(n):
        var v = r[i * k + j]
        rs = _a(rs, v)
        for cc in range(c):
            if Int(codes[i * c + cc]) == b:
                ob = _a(ob, v)
    e[t] = _m(rs, pr[b])
    o[t] = ob


@always_inline
def hm_block_eo_cell(
    r: FPtr, codes: IPtr, pr: FPtr, n: Int, k: Int, c: Int, nb: Int, a: Int, c0: Int, lo: Int, hi: Int,
    sign: Float32, e: FPtr, o: FPtr, t: Int,
):
    """update_R's E -= / += outer(sum of R over the block, pr) and O -= / += R_block Phi_block^T for t = j * B + b
    (`sign` -1 or +1); the block's positions ascending."""
    var j = t // nb
    var b = t - j * nb
    var rs = Float32(0)
    var ob = Float32(0)
    for s in range(lo, hi):
        var i = hm_cell(s, a, c0, n)
        var v = r[i * k + j]
        rs = _a(rs, v)
        for cc in range(c):
            if Int(codes[i * c + cc]) == b:
                ob = _a(ob, v)
    e[t] = _a(e[t], _m(sign, _m(rs, pr[b])))
    o[t] = _a(o[t], _m(sign, ob))


@always_inline
def hm_block_r_cell(
    sd: FPtr, e: FPtr, o: FPtr, codes: IPtr, theta: FPtr, n: Int, k: Int, c: Int, nb: Int, a: Int, c0: Int,
    lo: Int, r: FPtr, t: Int,
):
    """The block's cell at position lo + t: R[i, j] = S[i, j] * sum over its levels b of ((E[j, b] + 1) /
    (O[j, b] + 1))^theta[b] (harmonypy's `dot(power((E + 1) / (O + 1), theta), Phi)`), then the row over its L1
    norm (j ascending)."""
    var i = hm_cell(lo + t, a, c0, n)
    var tot = Float32(0)
    for j in range(k):
        var pen = Float32(0)
        for cc in range(c):
            var b = Int(codes[i * c + cc])
            var ratio = _q(_a(e[j * nb + b], Float32(1)), _a(o[j * nb + b], Float32(1)))
            pen = _a(pen, ftz(identical_pow(ratio, theta[b])))
        var v = _m(sd[i * k + j], pen)
        r[i * k + j] = v
        tot = _a(tot, v)
    for j in range(k):
        r[i * k + j] = _q(r[i * k + j], tot)


@always_inline
def hm_objective_row(
    r: FPtr, dist: FPtr, e: FPtr, o: FPtr, codes: IPtr, theta: FPtr, k: Int, c: Int, nb: Int, sigma: Float32,
    dst: FPtr, i: Int,
):
    """Cell i's share of harmonypy's objective: sum_j R dist + sigma sum_j R log R (0 log 0 = 0) + sigma sum_j R
    sum over its levels b of theta[b] log((O[j, b] + 1) / (E[j, b] + 1)). The driver folds the n shares
    (`sum_ff`)."""
    var km = Float32(0)
    var ent = Float32(0)
    var cross = Float32(0)
    for j in range(k):
        var v = r[i * k + j]
        km = _a(km, _m(v, dist[i * k + j]))
        if v > Float32(0):
            ent = _a(ent, _m(v, ftz(identical_log(v))))
        var w = Float32(0)
        for cc in range(c):
            var b = Int(codes[i * c + cc])
            var lr = ftz(identical_log(_q(_a(o[j * nb + b], Float32(1)), _a(e[j * nb + b], Float32(1)))))
            w = _a(w, _m(theta[b], lr))
        cross = _a(cross, _m(v, w))
    dst[i] = _a(km, _m(sigma, _a(ent, cross)))


@always_inline
def hm_centroid_cell(r: FPtr, z: FPtr, n: Int, k: Int, d: Int, y: FPtr, t: Int):
    """t = j * d + f: Y[j, f] = sum_i R[i, j] Zc[i, f] (cells ascending; the driver normalizes the rows)."""
    var j = t // d
    var f = t - j * d
    var s = Float32(0)
    for i in range(n):
        s = _a(s, _m(r[i * k + j], z[i * d + f]))
    y[t] = s


@always_inline
def hm_ridge_a_cell(r: FPtr, codes: IPtr, lamb: FPtr, n: Int, k: Int, c: Int, b1: Int, am: FPtr, t: Int):
    """t = (j * B1 + u) * B1 + v: A_j[u, v] = sum_i R[i, j] phi_u(i) phi_v(i), plus lamb[u - 1] on the diagonal
    past the intercept (harmonypy's `Phi_Rk Phi_moe^T + lamb`, lamb[0] = 0)."""
    var j = t // (b1 * b1)
    var rem = t - j * b1 * b1
    var u = rem // b1
    var v = rem - u * b1
    var s = Float32(0)
    for i in range(n):
        if hm_phi(codes, i, c, u) and hm_phi(codes, i, c, v):
            s = _a(s, r[i * k + j])
    if u == v and u > 0:
        s = _a(s, lamb[u - 1])
    am[t] = s


@always_inline
def hm_ridge_rhs_cell(r: FPtr, z: FPtr, codes: IPtr, n: Int, k: Int, d: Int, c: Int, b1: Int, rhs: FPtr, t: Int):
    """t = (j * B1 + u) * d + f: RHS_j[u, f] = sum_i R[i, j] phi_u(i) Z[i, f] (the uncorrected embedding)."""
    var j = t // (b1 * d)
    var rem = t - j * b1 * d
    var u = rem // d
    var f = rem - u * d
    var s = Float32(0)
    for i in range(n):
        if hm_phi(codes, i, c, u):
            s = _a(s, _m(r[i * k + j], z[i * d + f]))
    rhs[t] = s


@always_inline
def hm_chol_block(am: FPtr, b1: Int, j: Int):
    """A_j <- its lower Cholesky factor in place (columns ascending, each sum ascending). A_j is positive definite
    (lamb > 0 past the intercept, the intercept's weight sum_i R[i, j] > 0); a pivot that rounds to zero or below
    is clamped to the smallest normal float so the solve stays finite."""
    var base = j * b1 * b1
    for col in range(b1):
        var s = am[base + col * b1 + col]
        for l in range(col):
            var x = am[base + col * b1 + l]
            s = _s(s, _m(x, x))
        if s <= Float32(0):
            s = Float32(1.1754944e-38)
        var piv = ftz(identical_sqrt(s))
        am[base + col * b1 + col] = piv
        for row in range(col + 1, b1):
            var t = am[base + row * b1 + col]
            for l in range(col):
                t = _s(t, _m(am[base + row * b1 + l], am[base + col * b1 + l]))
            am[base + row * b1 + col] = _q(t, piv)


@always_inline
def hm_solve_cell(am: FPtr, rhs: FPtr, b1: Int, d: Int, w: FPtr, t: Int):
    """t = j * d + f: W_j[:, f] = A_j^-1 RHS_j[:, f] by the factor (forward, then back substitution), then
    W_j[0, f] = 0 (harmonypy keeps the intercept out of the correction)."""
    var j = t // d
    var f = t - j * d
    var base = j * b1 * b1
    var wb = j * b1 * d
    for u in range(b1):
        var s = rhs[wb + u * d + f]
        for l in range(u):
            s = _s(s, _m(am[base + u * b1 + l], w[wb + l * d + f]))
        w[wb + u * d + f] = _q(s, am[base + u * b1 + u])
    var bu = b1 - 1
    while bu >= 0:
        var s = w[wb + bu * d + f]
        for l in range(bu + 1, b1):
            s = _s(s, _m(am[base + l * b1 + bu], w[wb + l * d + f]))
        w[wb + bu * d + f] = _q(s, am[base + bu * b1 + bu])
        bu -= 1
    w[wb + f] = Float32(0)


@always_inline
def hm_correct_cell(z: FPtr, r: FPtr, w: FPtr, codes: IPtr, k: Int, d: Int, c: Int, b1: Int, dst: FPtr, t: Int):
    """t = i * d + f: Z_corr[i, f] = Z[i, f] - sum over clusters j (ascending, one subtraction each, harmonypy's
    `Z_corr -= W^T Phi_Rk` per cluster) of R[i, j] * sum over the cell's levels b of W_j[b + 1, f]."""
    var i = t // d
    var f = t - i * d
    var v = z[t]
    for j in range(k):
        var s = Float32(0)
        for cc in range(c):
            var b = Int(codes[i * c + cc])
            s = _a(s, w[(j * b1 + b + 1) * d + f])
        v = _s(v, _m(r[i * k + j], s))
    dst[t] = v
