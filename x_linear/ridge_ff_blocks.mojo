# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Ridge's float-float statistics and solve in the blocked order
(lane/classical-cv-folds, 2026-10-07; IDENTICAL control
`-D MOJOLEARN_CLASSICAL_RIDGECV_FF_BLOCKED`, default off, NOT MEASURED).

The incumbent float-float fallback (x_linear/ridge.mojo `ridge_ff_unit`,
`ridge_ff_solve`) runs one thread per statistic serially over every training
row, and the float-float Cholesky of an alpha on one thread. Here:

  1. the first pass is one partial per (column of [X | Y] plus the weight
     sum, FOLD_BLOCK-row block): `ff_col_mean`'s chain (ff_add_f of the
     word, two_prod of weight and word with sample weights) from zero over
     the block's training rows ascending; a thread per (column, block);
  2. a thread per column folds its partials blocks ascending (ff_add from
     zero) and divides by the weight sum folded the same way: the mean;
  3. the second pass is one partial per (upper Gram cell or X'Y entry,
     block): `ff_cross`'s chain (centered words, ff_mul, ff_mul_f by the
     weight, ff_add) from zero over the block's training rows ascending;
  4. a thread per cell folds its partials blocks ascending into the
     statistics (both Gram triangles);
  5. the solve runs on a block team: G + alpha I a cell a thread, the
     float-float Cholesky split as `t_cholesky` (every thread computes the
     pivot from the same words; the entries below it, each its own chain
     over k ascending, across the team), one target a thread.

Every value is one cell function called by one device thread or by the host
loop (the host column), so the host and both GPU vendors produce the same
words. The block fold order changes the bits of the float-float sums
against the incumbent (allowed: IDENTICAL is same bits across NVIDIA and
AMD within one version; the host column changes here together).

Cost reasoning (no shape rule): the work, about n * (d + 1)^2 / 2
float-float adds, is unchanged; the incumbent keeps (d + 1)^2 / 2 threads
busy, each a serial chain over the training rows with stride-d loads; the
blocked form keeps n / FOLD_BLOCK times as many in flight, adjacent threads
on adjacent cells of one row block, and holds for any n well above
FOLD_BLOCK and any d whose partial table fits device memory (the
`rw_gram_parts_kernel` shape of x_linear/ridge_grid.mojo, already the
weighted Ridge's). The one-thread Cholesky was latency bound at d^3 / 6
dependent float-float operations; the team cuts that to d^2 steps.
"""
from std.gpu import block_idx, block_dim, thread_idx
from x_linear.ops import FP, IP, ld, st, ldi, sti, i2f
from x_linear.ff import (
    ff_of, ff_add, ff_add_f, ff_sub, ff_mul, ff_mul_f, ff_div, ff_sqrt, ff_f32, ff_ld, ff_st,
    ff_cholesky, ff_chol_solve, ff_centered, two_prod,
)
from x_linear.tops import upper_cell, FOLD_BLOCK, fold_blocks
from x_linear.team import Team, TEAM_SLOTS, team_at
from x_linear.witness import witness_end

comptime RFF_TPB = 256


@always_inline
def rff_blocks(count: Int) -> Int:
    return max((count + RFF_TPB - 1) // RFF_TPB, 1)


@always_inline
def rff_stats(d: Int, t_n: Int) -> Int:
    """Cells of the second pass: the upper Gram cells, then the d * T X'Y entries."""
    return d * (d + 1) // 2 + d * t_n


@always_inline
def rff_mcols(d: Int, t_n: Int) -> Int:
    """Columns of the first pass: d + T means, then the weight sum (the
    training row count without sample weights)."""
    return d + t_n + 1


@always_inline
def rff_part_words(n: Int, d: Int, t_n: Int) -> Int:
    """hi (or lo) words of the partial table, one per (cell, block) of the
    larger pass (the two passes reuse it, the means fold between them)."""
    return max(rff_stats(d, t_n), rff_mcols(d, t_n)) * fold_blocks(n)


# ------------------------------------------------------------ the cells


def rff_mean_part(x: FP, y: FP, n: Int, d: Int, t_n: Int, sw: Bool, wo: Int, s: Int, e: Int, u: Int, b: Int,
                  ph: FP, pl: FP):
    """Partial u (column u of [X | Y]; u == d + T: the weight sum) over block
    b's rows outside [s, e) (s == e: every row), rows ascending in
    float-float from zero. Weights at y + wo when sw. A block inside the
    held-out fold stores zero."""
    var nb = fold_blocks(n)
    var lo = b * FOLD_BLOCK
    var hi = min(lo + FOLD_BLOCK, n)
    var acc = ff_of(Float32(0))
    var cnt = 0
    if not (lo >= s and hi <= e):
        for i in range(lo, hi):
            if i >= s and i < e:
                continue
            if u == d + t_n:
                if sw:
                    acc = ff_add_f(acc, ld(y, wo + i))
                else:
                    cnt += 1
            else:
                var v: Float32
                if u < d:
                    v = ld(x, i * d + u)
                else:
                    v = ld(y, i * t_n + (u - d))
                if sw:
                    acc = ff_add(acc, two_prod(ld(y, wo + i), v))
                else:
                    acc = ff_add_f(acc, v)
    if u == d + t_n and not sw:
        acc = ff_of(i2f(cnt))
    ff_st(ph, pl, u * nb + b, acc)


def rff_mean_fold(u: Int, n: Int, d: Int, t_n: Int, fi: Bool, ph: FP, pl: FP, st_h: FP, st_l: FP):
    """Mean u: the column's partials folded blocks ascending (ff_add from
    zero) over the weight sum's partials folded the same way; zero without
    an intercept. Into the statistics' [xm d | ym T] words."""
    var nb = fold_blocks(n)
    var m = ff_of(Float32(0))
    if fi:
        var num = ff_of(Float32(0))
        var den = ff_of(Float32(0))
        for b in range(nb):
            num = ff_add(num, ff_ld(ph, pl, u * nb + b))
            den = ff_add(den, ff_ld(ph, pl, (d + t_n) * nb + b))
        m = ff_div(num, den)
    ff_st(st_h, st_l, u, m)


def rff_cell_part(x: FP, y: FP, n: Int, d: Int, t_n: Int, sw: Bool, wo: Int, s: Int, e: Int, c: Int, b: Int,
                  st_h: FP, st_l: FP, ph: FP, pl: FP):
    """Partial of cell c (an upper Gram cell, then an X'Y entry) over block
    b's training rows: `ff_cross`'s chain (the words centered at the means
    of step 2, ff_mul, ff_mul_f by the weight, ff_add) from zero, rows
    ascending."""
    var nb = fold_blocks(n)
    var cells = d * (d + 1) // 2
    var j = 0
    var k = 0
    var ma = ff_of(Float32(0))
    var mb = ff_of(Float32(0))
    if c < cells:
        var jk = upper_cell(c, d)
        j = jk[0]
        k = jk[1]
        ma = ff_ld(st_h, st_l, j)
        mb = ff_ld(st_h, st_l, k)
    else:
        var q = c - cells
        j = q // t_n
        k = q - j * t_n
        ma = ff_ld(st_h, st_l, j)
        mb = ff_ld(st_h, st_l, d + k)
    var lo = b * FOLD_BLOCK
    var hi = min(lo + FOLD_BLOCK, n)
    var acc = ff_of(Float32(0))
    if not (lo >= s and hi <= e):
        for i in range(lo, hi):
            if i >= s and i < e:
                continue
            var bv: Float32
            if c < cells:
                bv = ld(x, i * d + k)
            else:
                bv = ld(y, i * t_n + k)
            var p = ff_mul(ff_centered(ld(x, i * d + j), ma), ff_centered(bv, mb))
            if sw:
                p = ff_mul_f(p, ld(y, wo + i))
            acc = ff_add(acc, p)
    ff_st(ph, pl, c * nb + b, acc)


def rff_cell_fold(c: Int, n: Int, d: Int, t_n: Int, ph: FP, pl: FP, st_h: FP, st_l: FP):
    """Cell c: its partials folded blocks ascending (ff_add from zero) into
    the statistics' Gram (both triangles) or X'Y words, `ridge_ff_unit`'s
    layout [xm d | ym T] [G d*d] [X'Y d*T]."""
    var nb = fold_blocks(n)
    var acc = ff_of(Float32(0))
    for b in range(nb):
        acc = ff_add(acc, ff_ld(ph, pl, c * nb + b))
    var gofs = d + t_n
    var xofs = gofs + d * d
    var cells = d * (d + 1) // 2
    if c < cells:
        var jk = upper_cell(c, d)
        ff_st(st_h, st_l, gofs + jk[0] * d + jk[1], acc)
        ff_st(st_h, st_l, gofs + jk[1] * d + jk[0], acc)
    else:
        var q = c - cells
        var j = q // t_n
        var tt = q - j * t_n
        ff_st(st_h, st_l, xofs + j * t_n + tt, acc)


# ------------------------------------------------------------ the solve


def t_ff_cholesky(t: Team, ah: FP, al: FP, m: Int) -> Bool:
    """`ff_cholesky` (x_linear/ff.mojo) on the team, split as `t_cholesky`
    (x_linear/tops.mojo): column j ascending; every thread computes the
    pivot from the same words (so the stop at a non-positive pivot is
    uniform), and the entries below it, each its own chain over k
    ascending reading only finished columns, are split across the team.
    The same statements per entry, so the same words as the serial loop.
    A team of one runs `ff_cholesky` itself."""
    if t.nt <= 1:
        return ff_cholesky(ah, al, m)
    for j in range(m):
        var s = ff_ld(ah, al, j * m + j)
        for k in range(j):
            var l = ff_ld(ah, al, j * m + k)
            s = ff_sub(s, ff_mul(l, l))
        if not (s.hi > 0):
            t.sync()
            return False
        var r = ff_sqrt(s)
        t.sync()
        if t.lead():
            ff_st(ah, al, j * m + j, r)
        for i in range(j + 1 + t.tid, m, t.nt):
            var tv = ff_ld(ah, al, i * m + j)
            for k in range(j):
                tv = ff_sub(tv, ff_mul(ff_ld(ah, al, i * m + k), ff_ld(ah, al, j * m + k)))
            ff_st(ah, al, i * m + j, ff_div(tv, r))
        t.sync()
    return True


def t_ridge_ff_solve(t: Team, d: Int, t_n: Int, fi: Bool, alpha: Float32, st_h: FP, st_l: FP, bh: FP, bl: FP,
                     res: FP, fh: FP, fl: FP) -> Bool:
    """`ridge_ff_solve` on a team: G + alpha I into fh / fl (a cell a
    thread), the factor by `t_ff_cholesky`, then one target a thread (its
    own d words of bh / bl: T*d words each); coef and intercept words into
    res (T*d | T). The same statements per value. Every thread returns the
    same verdict (False when float-float cannot factor it either)."""
    var gofs = d + t_n
    var xofs = gofs + d * d
    for q in range(t.tid, d * d, t.nt):
        var v = ff_ld(st_h, st_l, gofs + q)
        if q // d == q % d:
            v = ff_add_f(v, alpha)
        ff_st(fh, fl, q, v)
    t.sync()
    if not t_ff_cholesky(t, fh, fl, d):
        return False
    for tt in range(t.tid, t_n, t.nt):
        var o = tt * d
        for j in range(d):
            ff_st(bh, bl, o + j, ff_ld(st_h, st_l, xofs + j * t_n + tt))
        ff_chol_solve(fh, fl, d, bh + o, bl + o)
        var acc = ff_of(Float32(0))
        for j in range(d):
            var bj = ff_ld(bh, bl, o + j)
            st(res, tt * d + j, ff_f32(bj))
            acc = ff_add(acc, ff_mul(ff_ld(st_h, st_l, j), bj))
        st(res, t_n * d + tt, ff_f32(ff_sub(ff_ld(st_h, st_l, d + tt), acc)) if fi else Float32(0))
    t.sync()
    return True


# ------------------------------------------------------------ the host column


def rff_stats_host(x: FP, y: FP, n: Int, d: Int, t_n: Int, fi: Bool, sw: Bool, wo: Int, s: Int, e: Int,
                   st_h: FP, st_l: FP):
    """The host column's float-float statistics into st_h / st_l
    (`ridge_ff_unit`'s layout): the device kernels' cells looped (each
    cell is independent; the folds take the blocks ascending inside)."""
    var pw = rff_part_words(n, d, t_n)
    var parts = List[Float32](length=max(2 * pw, 1), fill=Float32(0))
    var ph = FP(unsafe_from_address=Int(parts.unsafe_ptr()))
    var pl = ph + pw
    var nb = fold_blocks(n)
    var mc = rff_mcols(d, t_n)
    for b in range(nb):
        for u in range(mc):
            rff_mean_part(x, y, n, d, t_n, sw, wo, s, e, u, b, ph, pl)
    for u in range(d + t_n):
        rff_mean_fold(u, n, d, t_n, fi, ph, pl, st_h, st_l)
    var stats = rff_stats(d, t_n)
    for b in range(nb):
        for c in range(stats):
            rff_cell_part(x, y, n, d, t_n, sw, wo, s, e, c, b, st_h, st_l, ph, pl)
    for c in range(stats):
        rff_cell_fold(c, n, d, t_n, ph, pl, st_h, st_l)
    _ = parts^


# ------------------------------------------------------------ the device kernels
# The statistics kernels take a gate word: they do nothing while it is zero.
# The k-fold grid sets it on the device from the alphas' trust flags
# (`rff_untrusted_kernel`), so no host step decides whether a fold needs the
# float-float pass; the refit passes a word set to one.


def rff_untrusted_kernel(trust: FP, na: Int32, gate: IP, wf: IP, woff: Int32, nonce: Int32):
    """Thread per alpha: an untrusted float32 factor (`kf_solve_kernel`'s
    trust word != 1) sets the gate (every writer stores the same one)."""
    var a = Int(block_idx.x) * RFF_TPB + Int(thread_idx.x)
    if a < Int(na) and ld(trust, a) != Float32(1):
        sti(gate, 0, 1)
    witness_end(wf, woff, nonce)


def rff_mean_part_kernel(x: FP, y: FP, n: Int32, d: Int32, t_n: Int32, sw: Int32, wo: Int32, s: Int32, e: Int32,
                         ph: FP, pl: FP, gate: IP, wf: IP, woff: Int32, nonce: Int32):
    """Thread (column, block): `rff_mean_part`; adjacent threads take
    adjacent columns of one row block."""
    var nn = Int(n)
    var dd = Int(d)
    var tn = Int(t_n)
    var mc = rff_mcols(dd, tn)
    var t = Int(block_idx.x) * RFF_TPB + Int(thread_idx.x)
    if ldi(gate, 0) != 0 and t < mc * fold_blocks(nn):
        rff_mean_part(x, y, nn, dd, tn, sw != 0, Int(wo), Int(s), Int(e), t % mc, t // mc, ph, pl)
    witness_end(wf, woff, nonce)


def rff_mean_fold_kernel(n: Int32, d: Int32, t_n: Int32, fi: Int32, ph: FP, pl: FP, sh: FP, sl: FP, gate: IP,
                         wf: IP, woff: Int32, nonce: Int32):
    """Thread per mean: `rff_mean_fold`."""
    var dd = Int(d)
    var tn = Int(t_n)
    var u = Int(block_idx.x) * RFF_TPB + Int(thread_idx.x)
    if ldi(gate, 0) != 0 and u < dd + tn:
        rff_mean_fold(u, Int(n), dd, tn, fi != 0, ph, pl, sh, sl)
    witness_end(wf, woff, nonce)


def rff_cell_part_kernel(x: FP, y: FP, n: Int32, d: Int32, t_n: Int32, sw: Int32, wo: Int32, s: Int32, e: Int32,
                         sh: FP, sl: FP, ph: FP, pl: FP, gate: IP, wf: IP, woff: Int32, nonce: Int32):
    """Thread (cell, block): `rff_cell_part`; adjacent threads take adjacent
    cells of one row block (the `rw_gram_parts_kernel` mapping)."""
    var nn = Int(n)
    var dd = Int(d)
    var tn = Int(t_n)
    var stats = rff_stats(dd, tn)
    var t = Int(block_idx.x) * RFF_TPB + Int(thread_idx.x)
    if ldi(gate, 0) != 0 and t < stats * fold_blocks(nn):
        rff_cell_part(x, y, nn, dd, tn, sw != 0, Int(wo), Int(s), Int(e), t % stats, t // stats, sh, sl, ph, pl)
    witness_end(wf, woff, nonce)


def rff_cell_fold_kernel(n: Int32, d: Int32, t_n: Int32, ph: FP, pl: FP, sh: FP, sl: FP, gate: IP,
                         wf: IP, woff: Int32, nonce: Int32):
    """Thread per cell: `rff_cell_fold`."""
    var dd = Int(d)
    var tn = Int(t_n)
    var c = Int(block_idx.x) * RFF_TPB + Int(thread_idx.x)
    if ldi(gate, 0) != 0 and c < rff_stats(dd, tn):
        rff_cell_fold(c, Int(n), dd, tn, ph, pl, sh, sl)
    witness_end(wf, woff, nonce)


def rff_kf_solve_kernel(d: Int32, fi: Int32, alphas: FP, na: Int32, trust: FP, sh: FP, sl: FP, bh: FP, bl: FP,
                        fh: FP, fl: FP, tmp: FP, tw: FP, w: FP, b: FP, wf: IP, woff: Int32, nonce: Int32):
    """Block a (one launch for every alpha of the fold): alpha a's
    float-float solve on the block team when its float32 factor was not
    trusted (`kf_solve_kernel`'s trust word != 1): `kf_ff_solve`'s words
    (w, the intercept; NaN words when float-float cannot factor it either).
    A trusted alpha's block does nothing. Per alpha: bh / bl d words, fh /
    fl d*d, tmp d + 1, tw TEAM_SLOTS."""
    var a = Int(block_idx.x)
    var dd = Int(d)
    if a < Int(na) and ld(trust, a) != Float32(1):
        var t = team_at(Int(thread_idx.x), Int(block_dim.x), tw + a * TEAM_SLOTS, 0, 0, 0)
        var out = tmp + a * (dd + 1)
        var ok = t_ridge_ff_solve(t, dd, 1, fi != 0, ld(alphas, a), sh, sl, bh + a * dd, bl + a * dd, out,
                                  fh + a * dd * dd, fl + a * dd * dd)
        if t.lead():
            var nan = Float32(0) / Float32(0)
            for j in range(dd):
                st(w, a * dd + j, ld(out, j) if ok else nan)
            st(b, a, ld(out, dd) if ok else nan)
    witness_end(wf, woff, nonce)


def rff_solve_kernel(d: Int32, t_n: Int32, fi: Int32, alpha: Float32, sh: FP, sl: FP, bh: FP, bl: FP, dst: FP,
                     fh: FP, fl: FP, tw: FP, wf: IP, woff: Int32, nonce: Int32):
    """`ridge_ff_solve_kernel` on one block team (the refit): dst coef T*d |
    intercept T | ok (1 / 0). bh, bl: T*d words; tw: TEAM_SLOTS."""
    var dd = Int(d)
    var tn = Int(t_n)
    var t = team_at(Int(thread_idx.x), Int(block_dim.x), tw, 0, 0, 0)
    var ok = t_ridge_ff_solve(t, dd, tn, fi != 0, alpha, sh, sl, bh, bl, dst, fh, fl)
    if t.lead():
        st(dst, tn * dd + tn, Float32(1) if ok else Float32(0))
    witness_end(wf, woff, nonce)
