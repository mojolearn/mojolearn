# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Lars and LassoLars (lane/algos-linear, 2026-09-27).

Reference: scikit-learn `sklearn/linear_model/_least_angle.py`,
`_lars_path_solver` with a precomputed Gram (the `while True:` loop at its
line ~637): the most correlated inactive feature joins with the sign of its
correlation, the equiangular direction is `L L' ls = sign_A` scaled by
AA = 1/sqrt(sum ls*sign), the step is min(g1, g2, C/AA) over the inactive
correlations, the lasso drop at z_pos = min_pos(-coef_A / ls), the early stop at
alpha <= alpha_min with linear interpolation, the degenerate-regressor skip
(pivot < 1e-7) and the lasso stop when alpha grows. Named differences:
  * the correlations are recomputed as X'y - G coef every step instead of
    their incremental `Cov -= gamma * corr_eq_dir` (the same quantity), and
    their `np.around(corr_eq_dir, cov_precision)` is not applied;
  * the Cholesky of G_AA is kept across steps and grown by one row per
    addition (theirs too); after a LassoLars drop the rows from the first
    dropped position on are refactored instead of downdated (`_t_chol_rows`);
    ties in argmax|Cov| go to the lowest feature index
    (theirs: the lowest position in their permuted Cov array);
  * float32 throughout; equality tolerance float32 eps, tiny32 as theirs;
  * DEVIATION 5010: the lar method lets an active coefficient cross zero
    (plain LAR); theirs flips its sign and adds nothing next step, which
    diverges (see the comment at the step).
"""
from x_linear.ops import (
    FP, IP, fa, fs, fm, fd, fmad, fsqrt, fabs, fmin, fsign, ld, st, ldi, sti, i2f,
    fill, copy,
)
from std.sys.info import is_gpu
from x_linear.team import Team
from x_linear.tops import t_col_means, t_mean, t_centered_gram, t_centered_xty
from core.classical_centered import centered_gram_v1_cell, centered_cross_v1_cell, bm_column_mean_cell

# lane/lars-grid-gram (2026-10-11), PROMOTED to the default (lane
# i17-decide-nbr-lin-prep, 2026-10-11); the old path (main's moments grid,
# `-D MOJOLEARN_IDN_LARS_GRID_GRAM_OFF`) is deleted, see the TOMBSTONE below. A/B one run per arm,
# main@b2657cc89 (runs v0045/v0046, a1628/a1629), NV L40S / AMD MI325X ms:
#   lars taxi          59.1 -> 3.1 (0.05)    / 126.1 -> 3.3 (0.03)
#   lasso-lars taxi    58.9 -> 3.0 (0.05)    / 125.7 -> 3.3 (0.03)
#   lasso-lars istella 112.8 -> 83.4 (0.74)  / 191.6 -> 74.6 (0.39)
#   lars istella       165.6 -> 340.7 (2.06) / 390.2 -> 1155.2 (2.96)  SLOWER
# Digests NV == AMD on both arms; quality equal or better (lars istella r2
# 0.3090 -> 0.3287, the rest equal to 1e-4).
# The lars istella regression (lane/fix-lars-grid-gram, 2026-11-15): the
# route changes only the moments, and lasso-lars istella, on the same moments,
# got faster; the time went into the PATH. With the new Gram's bits, Lars on
# istella's 220 features runs further along its path (r2 0.3090 -> 0.3287),
# and each step cost O(k^3) with k barriers: the path re-copied G_AA and
# refactored it twice a step (the (k+1) factor of the addition, then the k
# factor of the direction, `t_cholesky`: k barriers each, every thread
# re-summing every pivot), O(d^4) over a full path, all of it on one block.
# The pivots of the last steps dominate, so a path that goes ~20% further
# costs ~2x (k^4). Fix, for every shape: the factor is kept across steps
# (stride d in fw[ll]); an addition factors row k only (a lead forward
# substitution, O(k^2), no barrier but its broadcast), the direction reuses
# it (no second factorization), a LassoLars drop refactors only the rows
# from the first dropped position; no switch is kept for the old path. The
# row, the refactor and both solves then went onto the team with a chunked
# fold (lane/fix-lars-chol-parallel, `LARS_CH` below; bits change, every
# column and the host together). Both arms of `_OFF` took the kept factor,
# so that A/B isolated the moments again: A/B nvc-ab-20261124T095727Z-92fd9f
# (NVIDIA L40S, wheel 4519097d380a, job nvc1-0002), `_OFF` / this route, fit:
# lars taxi 2.99 -> 63.01 ms (21.09), lasso-lars taxi 2.97 -> 63.01 (21.23),
# lasso-lars istella 80.11 -> 114 (1.42), lars istella 118 -> 127 (1.08,
# noise); 8 cells, median 1.04. The old path is deleted.
# The route takes LARS / LassoLars' moments from the
# row-parallel blocked moments (core/blocked_moments.mojo: leaf column sums +
# binary-counter fold for the means of X and y, `bm_centered_gram_panels` and
# `bm_centered_cross_panels` at leaves of `contract_leaf_size(n)` rows) instead
# of main's moments grid (x_linear/moments_grid.mojo: one block per 16-column
# tile pair, one serial chain over every row per cell). Structural reason: at
# taxi's 16 features main's grid is 2 tiles -> 3 blocks, each cell one
# 1M-row dependent chain; the blocked form puts one block on every leaf of
# rows (hundreds of blocks) and folds the leaves in log2 levels. Expected
# lars taxi 60.8 -> ~10 ms NV (ideas-dense-linalg-20261011 F5). Bits change
# (the fold order), on NVIDIA, AMD and this host column together: the host
# below runs the device's reference cells (`centered_gram_v1_cell`,
# `centered_cross_v1_cell`, `bm_column_mean_cell`). Apple runs the same
# blocked fold (lane/apple-lars, 2026-10-12; x_linear/device.mojo
# the `lars_pre` setup), so its column matches NVIDIA == AMD == host.
# TOMBSTONE: MOJOLEARN_IDN_ALL_OFF (old-path umbrella, EXP:2438 DELETED; non-owner sites) deleted 2026-11-15 by lane/expdel2-21-linear; code recoverable at 9ff89ae635.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_ALL_OFF.expdel2-21-linear.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_IDN_LARS_GRID_GRAM_OFF (old path: main's moments grid, 21x slower lars / lasso-lars taxi, 1.42 lasso-lars istella; A/B nvc-ab-20261124T095727Z-92fd9f NVIDIA L40S) deleted 2026-11-24 by lane/ab-outcome-2; code recoverable at ba83d4655f.
# Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_IDN_LARS_GRID_GRAM_OFF.ab-outcome-2.patch; record in docs/TOMBSTONES.md.


def _lars_host_blocked_moments(x: FP, y: FP, n: Int, d: Int, fi: Bool, fw: FP, xm: Int, gg: Int, xty: Int) -> Float32:
    """The host column of the blocked moments: the means (zeros
    without an intercept), the centered Gram (upper cells, mirrored: the
    product a*b is commutative, so the mirror is the same word) and the
    centered X'y, each by the blocked fold the device runs. Returns y's mean."""
    var ym = Float32(0)
    fill(fw, xm, d, Float32(0))
    if fi and n > 0:
        for j in range(d):
            st(fw, xm + j, bm_column_mean_cell(x, n, d, j))
        ym = bm_column_mean_cell(y, n, 1, 0)
    var mp = fw + xm
    for i in range(d):
        for j in range(i, d):
            var v = centered_gram_v1_cell(x, mp, n, d, i, j)
            st(fw, gg + i * d + j, v)
            st(fw, gg + j * d + i, v)
    for j in range(d):
        st(fw, xty + j, centered_cross_v1_cell(x, mp, y, ym, n, d, j))
    return ym

comptime BIG = Float32(3.0e38)
comptime TINY32 = Float32(1.1754944e-38)
comptime EQ_TOL = Float32(1.1920929e-07)


# lane/fix-lars-chol-parallel (2026-11-15): the Cholesky of G_AA is kept
# across steps at row stride d in fw[ll] (lane/fix-lars-grid-gram), and every
# triangular pass over it (the new row of an addition, the rows refactored
# after a LassoLars drop, the two solves of the direction) runs on the whole
# team. Each value has ONE fold order, fixed by the index grid alone (chunks
# of LARS_CH, never the thread count), so the host (a team of one) and every
# device write the same words: an inner product over p is the fmad chain of
# each LARS_CH-chunk from 0 (p ascending), the chunks subtracted from the
# right side one after another (ascending for the lower solves and the pivot,
# descending for the upper solve), the chunk of the value's own block last.
# The chunk partials of the finished blocks are dealt across the team (one
# (value, chunk) a thread), folded one value a thread, and the in-block
# triangle (at most LARS_CH^2/2 fmads) runs on the lead. Per LARS step with k
# active and nb = ceil(k / LARS_CH): about 9 nb + 6 barriers and a lead chain
# of about 3 nb LARS_CH^2/2 fmads, where lane/fix-lars-grid-gram's lead ran
# k^2/2 (new row) + k^2 (two solves) dependent fmads behind 2 barriers. A
# factor row depends only on G and the rows above it, so a row refactored
# after a drop is the word an addition would have written. Bits change from
# the serial statements: lars and lasso-lars, every vendor column and the
# host column together.
# LARS_CH: a chunk is one thread's dependent chain and bounds the lead's
# in-block triangle; it is fixed, not keyed on d, n or a dataset (cost: 16
# keeps both short for every k from 1 to thousands, and the partials of one
# block, one word per (value, chunk), never exceed k <= d words of scratch).
comptime LARS_CH = 16


def _t_lower_solve(t: Team, a: FP, ll: Int, s: Int, m: Int, xo: Int, pw: Int):
    """In place: a[xo + j] (j < m) holds the right side g on entry and the
    solution of L x = g on return, L the lower factor at a[ll] (row stride s).
    x_j = (g_j - P_0 - ... - P_{b-1} - P_in) / L[j, j], b = j // LARS_CH, P_c
    the fmad chain over chunk c of L[j, p] x_p from 0 (p ascending), P_in the
    same over [b * LARS_CH, j). pw: scratch of at most m words (the chunk
    partials of one block). Ends on a barrier when m > 0."""
    var nb = (m + LARS_CH - 1) // LARS_CH
    for b in range(nb):  # driver-loop: over column blocks of the factor
        var j0 = b * LARS_CH
        var nj = min(LARS_CH, m - j0)
        if b > 0:
            # the chunk partials of the finished blocks, one (j, chunk) a thread
            for task in range(t.tid, nj * b, t.nt):
                var jl = task // b
                var c = task - jl * b
                var row = ll + (j0 + jl) * s
                var acc = Float32(0)
                for p in range(c * LARS_CH, c * LARS_CH + LARS_CH):
                    acc = fmad(ld(a, row + p), ld(a, xo + p), acc)
                st(a, pw + task, acc)
            t.sync()
            # folded into the right side in chunk order, one j a thread
            for jl in range(t.tid, nj, t.nt):
                var tv = ld(a, xo + j0 + jl)
                for c in range(b):
                    tv = fs(tv, ld(a, pw + jl * b + c))
                st(a, xo + j0 + jl, tv)
            t.sync()
        if t.lead():
            # the in-block triangle (at most LARS_CH rows)
            for j in range(j0, j0 + nj):
                var acc = Float32(0)
                for p in range(j0, j):
                    acc = fmad(ld(a, ll + j * s + p), ld(a, xo + p), acc)
                st(a, xo + j, fd(fs(ld(a, xo + j), acc), ld(a, ll + j * s + j)))
        t.sync()


def _t_upper_solve(t: Team, a: FP, ll: Int, s: Int, m: Int, xo: Int, pw: Int):
    """In place: the solution of L' x = y (y in a[xo + i], i < m, on entry).
    x_i = (y_i - P_{nb-1} - ... - P_{b+1} - P_in) / L[i, i], b = i // LARS_CH,
    P_c the fmad chain over chunk c of L[p, i] x_p from 0 (p ascending), P_in
    the same over (i, end of i's block); blocks run descending. pw: scratch of
    at most m words. Ends on a barrier when m > 0."""
    var nb = (m + LARS_CH - 1) // LARS_CH
    var b = nb - 1
    while b >= 0:  # driver-loop: over column blocks of the factor, descending
        var i0 = b * LARS_CH
        var ni = min(LARS_CH, m - i0)
        var na = nb - 1 - b  # finished chunks above this block
        if na > 0:
            # one (i, chunk) a thread; ca = 0 is the last chunk (descending)
            for task in range(t.tid, ni * na, t.nt):
                var il = task // na
                var c = nb - 1 - (task - il * na)
                var p1 = min(c * LARS_CH + LARS_CH, m)
                var acc = Float32(0)
                for p in range(c * LARS_CH, p1):
                    acc = fmad(ld(a, ll + p * s + i0 + il), ld(a, xo + p), acc)
                st(a, pw + task, acc)
            t.sync()
            for il in range(t.tid, ni, t.nt):
                var tv = ld(a, xo + i0 + il)
                for ca in range(na):
                    tv = fs(tv, ld(a, pw + il * na + ca))
                st(a, xo + i0 + il, tv)
            t.sync()
        if t.lead():
            var i = i0 + ni - 1
            while i >= i0:
                var acc = Float32(0)
                for p in range(i + 1, i0 + ni):
                    acc = fmad(ld(a, ll + p * s + i), ld(a, xo + p), acc)
                st(a, xo + i, fd(fs(ld(a, xo + i), acc), ld(a, ll + i * s + i)))
                i -= 1
        t.sync()
        b -= 1


def _t_chol_row(t: Team, a: FP, ll: Int, s: Int, gg: Int, d: Int, iw: IP, act: Int, r: Int, pw: Int) -> Bool:
    """Row r of the lower Cholesky of G_AA (A = the active list at iw[act])
    into a[ll] at row stride s; rows < r already hold it. The off-diagonal
    entries solve L[:r, :r] x = G[A_r, A_:r] (`_t_lower_solve`); the pivot is
    sqrt(G[A_r, A_r] - Q_0 - Q_1 - ...), Q_c the fmad chain of x_p^2 over
    chunk c. False (uniform) on a non-positive pivot, the pivot unwritten."""
    var gr = ldi(iw, act + r)
    var xo = ll + r * s
    for j in range(t.tid, r, t.nt):
        st(a, xo + j, ld(a, gg + gr * d + ldi(iw, act + j)))
    t.sync()
    _t_lower_solve(t, a, ll, s, r, xo, pw)
    var nc = (r + LARS_CH - 1) // LARS_CH
    for c in range(t.tid, nc, t.nt):
        var acc = Float32(0)
        for p in range(c * LARS_CH, min(c * LARS_CH + LARS_CH, r)):
            var l = ld(a, xo + p)
            acc = fmad(l, l, acc)
        st(a, pw + c, acc)
    t.sync()
    var ok = 0
    if t.lead():
        var sv = ld(a, gg + gr * d + gr)
        for c in range(nc):
            sv = fs(sv, ld(a, pw + c))
        if sv > 0:
            st(a, xo + r, fsqrt(sv))
            ok = 1
    return t.bcast_int(ok, 6) == 1


def _t_chol_rows(t: Team, a: FP, ll: Int, s: Int, gg: Int, d: Int, iw: IP, act: Int, lo: Int, m: Int, pw: Int) -> Bool:
    """Rows [lo, m) of the kept factor, one `_t_chol_row` each (an addition is
    one row; a LassoLars drop refactors the rows from its first dropped
    position). False (uniform) where a pivot is not positive."""
    for r in range(lo, m):  # driver-loop: over factor rows
        if not _t_chol_row(t, a, ll, s, gg, d, iw, act, r, pw):
            return False
    return True


def lars_fit(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """ip: [max_iter, fit_intercept, lasso, positive]; fp: [alpha_min].
    res: coef d, intercept, n_iter, alpha, n_active, active d.
    fw: xm d | G d*d | xty d | prev d | cov d | L d*d | ls d | sgn d | corr d.
    iw: state d (0 inactive, 1 active, 2 degenerate) | active list d."""
    var max_iter = ldi(ip, 0)
    var fi = ldi(ip, 1) != 0
    var lasso = ldi(ip, 2) != 0
    var alpha_min = ld(fp, 0)
    var positive = ldi(ip, 3) != 0
    var xm = 0
    var gg = d
    var xty = gg + d * d
    var prev = xty + d
    var cov = prev + d
    var ll = cov + d
    var ls = ll + d * d
    var sgn = ls + d
    var corr = sgn + d
    var state = 0
    var act = d
    # the moments come from the device kernels or the host's blocked fold;
    # the path runs on the team (`_lars_path_team`)
    var ym = Float32(0)
    comptime if is_gpu():
        # the device (x_linear/device.mojo `lars_path_kernel`): the blocked
        # moments already wrote the means, the Gram and X'y into fw, and y's
        # mean waits in fw[prev] (read here, then the slot zeroed as the
        # path expects); no row pass runs on the team
        ym = ld(fw, prev)
        t.sync()
        if t.lead():
            st(fw, prev, Float32(0))
        t.sync()
    else:
        # the host: the device's blocked fold, cell by cell
        ym = _lars_host_blocked_moments(x, y, n, d, fi, fw, xm, gg, xty)
    # lane/fix-lars-chol-parallel (2026-11-15): one path on every column. The
    # host is a team of one (sync is nothing, bcast returns its argument) and
    # runs the device statements; the lead-only copy of the path that sat here
    # (the host column and a device team of one) is deleted.
    _lars_path_team(t, n, d, max_iter, lasso, positive, alpha_min, fi, ym, fw, iw, res,
                    xm, gg, xty, prev, cov, ll, ls, sgn, corr, state, act)


def _lars_path_team(
    t: Team, n: Int, d: Int, max_iter: Int, lasso: Bool, positive: Bool, alpha_min: Float32, fi: Bool, ym: Float32,
    fw: FP, iw: IP, res: FP, xm: Int, gg: Int, xty: Int, prev: Int, cov: Int, ll: Int, ls: Int, sgn: Int, corr: Int,
    state: Int, act: Int,
):
    """lars_fit's path, the only one: the whole block on a device, a team of
    one on the host (lane/neural-pass90, 2026-10-01; lane/fix-lars-chol-parallel,
    2026-11-15). Every thread runs the loop and computes its scalar decisions
    from the same words (read after a barrier), so the control flow is
    uniform; the lead alone writes the shared state. The O(d^2) and O(k^2)
    parts are split across the team: the correlations (one j a thread, its
    chain over l ascending), every pass over the kept Cholesky (the new row
    of an addition, the refactor after a LassoLars drop, the two solves of
    the direction: `_t_chol_rows`, `_t_lower_solve`, `_t_upper_solve`, whose
    fold order is fixed by the index grid, not the thread count) and the
    d x k products of the step search (cj, one j a thread); the minimum over
    j stays the lead's. The device ran the whole path on the lead thread:
    Lars on istella 100k x 220 took 9.4 s on the M4's GPU."""
    if t.lead():
        fill(res, 0, d, Float32(0))
        fill(fw, prev, d, Float32(0))
        for j in range(d):
            sti(iw, state + j, 0)
    t.sync()
    var k = 0
    var lv = 0  # rows of the kept Cholesky factor (stride d at fw[ll]); uniform
    var n_iter = 0
    var drop = False
    var alpha = Float32(0)
    var prev_alpha = Float32(0)
    var guard = 0
    while guard < 4 * d + 4 * max_iter + 8:
        guard += 1
        for j in range(t.tid, d, t.nt):
            var acc = ld(fw, xty + j)
            for l in range(d):
                acc = fs(acc, fm(ld(fw, gg + j * d + l), ld(res, l)))
            st(fw, cov + j, acc)
        t.sync()
        var c_idx = -1
        var cbig = Float32(0)
        for j in range(d):
            if ldi(iw, state + j) == 0:
                # their positive=True takes argmax(Cov), not argmax |Cov|
                var a = ld(fw, cov + j) if positive else fabs(ld(fw, cov + j))
                # DEVIATION 5005 (IDENTITY_PATHS row 105): strict >, the
                # lowest index wins an exact tie
                if c_idx < 0 or a > cbig:
                    c_idx = j
                    cbig = a
        alpha = fd(cbig, i2f(n))
        if alpha <= fa(alpha_min, EQ_TOL):
            if fabs(fs(alpha, alpha_min)) > EQ_TOL:
                if n_iter > 0 and t.lead():
                    var ss = fd(fs(prev_alpha, alpha_min), fs(prev_alpha, alpha))
                    for j in range(d):
                        var pj = ld(fw, prev + j)
                        st(res, j, fmad(ss, fs(ld(res, j), pj), pj))
                alpha = alpha_min
            break
        if n_iter >= max_iter or k >= d:
            break
        if not drop:
            if c_idx < 0:
                break
            if t.lead():
                sti(iw, act + k, c_idx)
                st(fw, sgn + k, Float32(1) if positive else fsign(ld(fw, cov + c_idx)))
            t.sync()
            # row k of the kept factor on the team (rows < lv kept; lv == k
            # unless a refactor after a drop failed); corr is free scratch
            # here (written next by the step search below)
            var okc = _t_chol_rows(t, fw, ll, d, gg, d, iw, act, lv, k + 1, corr)
            if not okc or ld(fw, ll + k * d + k) < Float32(1e-7):
                if t.lead():
                    sti(iw, state + c_idx, 2)
                t.sync()
                continue
            if t.lead():
                sti(iw, state + c_idx, 1)
            k += 1
            lv = k
            t.sync()
        if lasso and n_iter > 0 and prev_alpha < alpha:
            break
        # no second factorization: the addition finished G_AA's factor; a
        # LassoLars drop refactors rows [lv, k) on the team
        if lv < k:
            if _t_chol_rows(t, fw, ll, d, gg, d, iw, act, lv, k, corr):
                lv = k
        # L L' ls = sgn on the team (`_t_lower_solve`, `_t_upper_solve`)
        for a in range(t.tid, k, t.nt):
            st(fw, ls + a, ld(fw, sgn + a))
        t.sync()
        _t_lower_solve(t, fw, ll, d, k, ls, corr)
        _t_upper_solve(t, fw, ll, d, k, ls, corr)
        var aa = Float32(0)
        if t.lead():
            if k == 1 and ld(fw, ls) == 0:
                st(fw, ls, Float32(1))
                aa = Float32(1)
            else:
                var sm = Float32(0)
                for a in range(k):
                    sm = fmad(ld(fw, ls + a), ld(fw, sgn + a), sm)
                aa = fd(Float32(1), fsqrt(sm))
                for a in range(k):
                    st(fw, ls + a, fm(ld(fw, ls + a), aa))
        aa = t.bcast(aa, 2)
        # cj for every inactive j, one j a thread (its chain over a ascending)
        for j in range(t.tid, d, t.nt):
            if ldi(iw, state + j) != 0:
                continue
            var cj = Float32(0)
            for a in range(k):
                cj = fmad(ld(fw, gg + j * d + ldi(iw, act + a)), ld(fw, ls + a), cj)
            st(fw, corr + j, cj)
        t.sync()
        var gamma = fd(cbig, aa)
        var z_pos = BIG
        drop = False
        var kk = k
        var lvl = lv
        if t.lead():
            for j in range(d):
                if ldi(iw, state + j) != 0:
                    continue
                var cj = ld(fw, corr + j)
                var cv = ld(fw, cov + j)
                var g1 = fd(fs(cbig, cv), fa(fs(aa, cj), TINY32))
                if g1 > 0 and g1 < gamma:
                    gamma = g1
                if not positive:
                    var g2 = fd(fa(cbig, cv), fa(fa(aa, cj), TINY32))
                    if g2 > 0 and g2 < gamma:
                        gamma = g2
            # DEVIATION 5010 (x_linear/README.md): the zero crossing is the
            # lasso modification only. Their lar method also flips
            # sign_active and skips the next addition when an active
            # coefficient crosses zero, which breaks the equiangular
            # invariant (an active correlation keeps its sign in LAR; only
            # the coefficient changes sign) and diverges: their own Lars,
            # float64, ends far from least squares even with every feature
            # active (python -m mojolearn.tests.test_x_linear_sanity
            # lars-crossing). Plain LAR (Efron et al. 2004) lets it cross.
            if lasso:
                for a in range(k):
                    var z = fd(-ld(res, ldi(iw, act + a)), fa(ld(fw, ls + a), TINY32))
                    if z > 0 and z < z_pos:
                        z_pos = z
                if z_pos < gamma:
                    for a in range(k):
                        var z = fd(-ld(res, ldi(iw, act + a)), fa(ld(fw, ls + a), TINY32))
                        if z == z_pos:
                            st(fw, sgn + a, -ld(fw, sgn + a))
                    gamma = z_pos
                    drop = True
            copy(fw, prev, res, 0, d)
            for a in range(k):
                var j = ldi(iw, act + a)
                st(res, j, fmad(gamma, ld(fw, ls + a), ld(res, j)))
            if drop and lasso:
                var w = 0
                for a in range(k):
                    var j = ldi(iw, act + a)
                    var z = fd(-ld(fw, prev + j), fa(ld(fw, ls + a), TINY32))
                    if z == z_pos:
                        sti(iw, state + j, 0)
                        st(res, j, Float32(0))
                        if a < lvl:
                            lvl = a
                    else:
                        sti(iw, act + w, j)
                        st(fw, sgn + w, ld(fw, sgn + a))
                        w += 1
                kk = w
        drop = t.bcast_int(1 if drop else 0, 1) == 1
        k = t.bcast_int(kk, 3)
        lv = t.bcast_int(lvl, 5)
        n_iter += 1
        prev_alpha = alpha
    if not t.lead():
        return
    var intercept = Float32(0)
    if fi:
        var acc = Float32(0)
        for j in range(d):
            acc = fmad(ld(fw, xm + j), ld(res, j), acc)
        intercept = fs(ym, acc)
    st(res, d, intercept)
    st(res, d + 1, i2f(n_iter))
    st(res, d + 2, alpha)
    st(res, d + 3, i2f(k))
    for a in range(d):
        st(res, d + 4 + a, i2f(ldi(iw, act + a)) if a < k else Float32(-1))
