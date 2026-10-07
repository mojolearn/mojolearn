# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LassoCV / ElasticNetCV fold statistics from fold-aligned compensated block
partials (lane/classical-cv, 2026-10-07; IDENTICAL control
`-D MOJOLEARN_CLASSICAL_ENETCV_FOLD_BLOCKS=<chunks per fold>`, arms 16|32|64).

The incumbent (x_linear/cd_grid.mojo) folds every prep (F training sets and
the full data) over ALL n rows: F + 1 reads of every row per pass, one plain
f32 chain per value. Here every row is read once per pass:

  1. each fold's row span [lo, lo + span) (KFold: exactly the fold's rows;
     rows of another fold inside a span are skipped by their id) is cut into
     C = ENETCV_FB_CHUNKS chunks; a chunk never straddles a fold;
  2. per (fold, chunk, column) a TwoSum-compensated sum and the row count;
     the chunks merge ascending (TwoSum) into each fold's sum and mean;
  3. per (fold, chunk, cell) of [X | y] centered at the fold's own mean, a
     Dot2-compensated cross product (exact product error by fmad, TwoSum on
     the running sum); the chunks merge ascending (TwoSum) into the fold's
     centered Gram;
  4. a training set (all folds but p; all folds for p == F) combines its
     folds ascending by the parallel-axis rule
     G_S = sum_g [C_g + n_g (mu_g - m_S)(mu_g - m_S)'], which never
     subtracts two large uncentered sums; m_S = (sum_g S_g) / n_S
     (zero without an intercept, which leaves G_S uncentered).

No plain f32 running sum over many rows: within a chunk the sums carry a
compensation word, across chunks and folds the merges are TwoSum too. The
chunk count is a fixed constant (not a data shape), so the fold order and
the bits are the same on every vendor. Every value is one cell function
called by one device thread or by the host loop (the host column), so the
host and both GPU vendors produce the same words. Rows whose fold id is
outside [0, F) belong to no fold (Python's KFold ids never are).
"""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from x_linear.ops import FP, IP, fa, fs, fm, fd, fmad, ld, st, ldi, sti, i2f
from x_linear.tops import upper_cell
from x_linear.witness import witness_end
from experiments.classical_identical_ideas.linear_controls import ENETCV_FB_CHUNKS

comptime FB_C = ENETCV_FB_CHUNKS
"""Chunks per fold."""
comptime FB_TPB = 64
comptime FB_TC = 16
"""Gram tile edge (a block owns a pair of tiles)."""
comptime FB_TR = 128
"""Rows a Gram block stages per step."""
comptime FB_NT = 256
comptime FB_LG = FB_TR * 2 * FB_TC // FB_NT


@always_inline
def fb_two_sum(mut hi: Float32, mut lo: Float32, v: Float32):
    """hi + lo += v: TwoSum of (hi, v), its error into lo."""
    var s = fa(hi, v)
    var bp = fs(s, hi)
    var ap = fs(s, bp)
    lo = fa(lo, fa(fs(hi, ap), fs(v, bp)))
    hi = s


@always_inline
def fb_dot2(mut hi: Float32, mut lo: Float32, a: Float32, b: Float32):
    """hi + lo += a * b (Dot2): the product's exact error by one fmad, the
    sum's by TwoSum, both into lo."""
    var p = fm(a, b)
    var pe = fmad(a, b, -p)
    var s = fa(hi, p)
    var bp = fs(s, hi)
    var ap = fs(s, bp)
    lo = fa(lo, fa(fa(fs(hi, ap), fs(p, bp)), pe))
    hi = s


@always_inline
def fb_words(m: Int) -> Int:
    """A fold's statistics: sums m | means m | centered Gram m*m (m = d + 1)."""
    return 2 * m + m * m


@always_inline
def fb_chunk_lo(lo: Int, span: Int, c: Int) -> Int:
    return lo + span * c // FB_C


@always_inline
def fb_aug(x: FP, y: FP, d: Int, row: Int, col: Int) -> Float32:
    """[X | y] at (row, col)."""
    if col == d:
        return ld(y, row)
    return ld(x, row * d + col)


@always_inline
def fb_in_fold(y: FP, n: Int, row: Int, f: Int) -> Bool:
    return Int(ld(y, n + row)) == f


# ------------------------------------------------------------ the cells


def fb_mean_cell(x: FP, y: FP, n: Int, d: Int, spans: IP, f: Int, c: Int, col: Int, part_s: FP, part_n: IP):
    """Chunk c of fold f: column col's compensated sum (and, col 0, the
    chunk's row count), rows ascending."""
    var m = d + 1
    var lo = ldi(spans, 2 * f)
    var span = ldi(spans, 2 * f + 1)
    var hi = Float32(0)
    var cc = Float32(0)
    var cnt = 0
    for i in range(fb_chunk_lo(lo, span, c), fb_chunk_lo(lo, span, c + 1)):
        if fb_in_fold(y, n, i, f):
            fb_two_sum(hi, cc, fb_aug(x, y, d, i, col))
            cnt += 1
    st(part_s, (f * FB_C + c) * m + col, fa(hi, cc))
    if col == 0:
        sti(part_n, f * FB_C + c, cnt)


def fb_fold_mean_cell(part_s: FP, part_n: IP, m: Int, f: Int, col: Int, fst: FP, fcnt: IP):
    """Fold f's column col: its chunks' sums merged ascending, the mean;
    col 0 also stores the fold's row count."""
    var hi = Float32(0)
    var cc = Float32(0)
    var cnt = 0
    for c in range(FB_C):
        fb_two_sum(hi, cc, ld(part_s, (f * FB_C + c) * m + col))
        cnt += ldi(part_n, f * FB_C + c)
    var s = fa(hi, cc)
    var base = f * fb_words(m)
    st(fst, base + col, s)
    st(fst, base + m + col, fd(s, i2f(cnt)) if cnt > 0 else Float32(0))
    if col == 0:
        sti(fcnt, f, cnt)


def fb_gram_cell_host(x: FP, y: FP, n: Int, d: Int, spans: IP, f: Int, c: Int, j: Int, k: Int, fst: FP, pg: FP):
    """Chunk c of fold f: the cell (j, k) of [X | y] centered at the fold's
    means, Dot2 over the chunk's rows ascending (the Gram kernel's thread)."""
    var m = d + 1
    var base = f * fb_words(m)
    var mj = ld(fst, base + m + j)
    var mk = ld(fst, base + m + k)
    var lo = ldi(spans, 2 * f)
    var span = ldi(spans, 2 * f + 1)
    var hi = Float32(0)
    var cc = Float32(0)
    for i in range(fb_chunk_lo(lo, span, c), fb_chunk_lo(lo, span, c + 1)):
        if fb_in_fold(y, n, i, f):
            fb_dot2(hi, cc, fs(fb_aug(x, y, d, i, j), mj), fs(fb_aug(x, y, d, i, k), mk))
    st(pg, (f * FB_C + c) * m * m + j * m + k, fa(hi, cc))


def fb_fold_gram_cell(pg: FP, m: Int, f: Int, j: Int, k: Int, fst: FP):
    """Fold f's centered Gram cell (j <= k): the chunks merged ascending;
    both triangles."""
    var hi = Float32(0)
    var cc = Float32(0)
    for c in range(FB_C):
        fb_two_sum(hi, cc, ld(pg, (f * FB_C + c) * m * m + j * m + k))
    var v = fa(hi, cc)
    var base = f * fb_words(m) + 2 * m
    st(fst, base + j * m + k, v)
    st(fst, base + k * m + j, v)


@always_inline
def fb_set_count(fcnt: IP, f_n: Int, p: Int) -> Int:
    var n = 0
    for g in range(f_n):
        if g != p:
            n += ldi(fcnt, g)
    return n


def fb_set_mean(fst: FP, fcnt: IP, m: Int, f_n: Int, p: Int, j: Int, fi: Bool) -> Float32:
    """Training set p's mean of column j (folds ascending); 0 without an
    intercept or rows."""
    if not fi:
        return Float32(0)
    var ns = fb_set_count(fcnt, f_n, p)
    if ns == 0:
        return Float32(0)
    var hi = Float32(0)
    var cc = Float32(0)
    for g in range(f_n):
        if g != p:
            fb_two_sum(hi, cc, ld(fst, g * fb_words(m) + j))
    return fd(fa(hi, cc), i2f(ns))


def fb_set_cell(fst: FP, fcnt: IP, m: Int, f_n: Int, p: Int, j: Int, k: Int, mj: Float32, mk: Float32) -> Float32:
    """Training set p's cell (j, k) at its means (mj, mk): the parallel-axis
    sum over its folds ascending, TwoSum throughout."""
    var hi = Float32(0)
    var cc = Float32(0)
    for g in range(f_n):
        if g != p:
            var base = g * fb_words(m)
            fb_two_sum(hi, cc, ld(fst, base + 2 * m + j * m + k))
            var ng = ldi(fcnt, g)
            if ng > 0:
                var dj = fs(ld(fst, base + m + j), mj)
                var dk = fs(ld(fst, base + m + k), mk)
                fb_two_sum(hi, cc, fm(fm(i2f(ng), dj), dk))
    return fa(hi, cc)


def fb_prep_unit(u: Int, fst: FP, fcnt: IP, d: Int, f_n: Int, p: Int, fi: Bool,
                 dst: FP, xm: Int, gg: Int, q: Int, sc: Int):
    """Unit u = j * m + k (j <= k) of training set p into the cd prep words:
    xm d | G d*d | q d | (y mean, |yc|^2, rows), as `_prep` writes them."""
    var m = d + 1
    var j = u // m
    var k = u % m
    if j > k:
        return
    var mj = fb_set_mean(fst, fcnt, m, f_n, p, j, fi)
    var mk = fb_set_mean(fst, fcnt, m, f_n, p, k, fi)
    var v = fb_set_cell(fst, fcnt, m, f_n, p, j, k, mj, mk)
    if k < d:
        st(dst, gg + j * d + k, v)
        st(dst, gg + k * d + j, v)
        if j == k:
            st(dst, xm + j, mj)
    elif j < d:
        st(dst, q + j, v)
    else:
        st(dst, sc, mj)
        st(dst, sc + 1, v)
        st(dst, sc + 2, i2f(fb_set_count(fcnt, f_n, p)))


# ------------------------------------------------------------ the host column


def fb_host_words(d: Int, f_n: Int) -> Int:
    """Host work words past enetcv_fit's own: fold stats | counts | spans."""
    return f_n * fb_words(d + 1) + 3 * f_n


def fb_host_stats(x: FP, y: FP, n: Int, d: Int, f_n: Int, wk: FP):
    """The host column's fold statistics into wk (`fb_host_words`): the
    device kernels' cells looped in any order (each cell is independent)."""
    var m = d + 1
    var fst = wk
    var fcnt = (wk + f_n * fb_words(m)).bitcast[Int32]()
    var sp = (wk + f_n * fb_words(m) + f_n).bitcast[Int32]()
    # the spans: the device's span kernels' (lo, hi - lo), (0, 0) if empty
    var lohi = List[Int](length=max(2 * f_n, 1), fill=0)
    for f in range(f_n):
        lohi[2 * f] = n
    for i in range(n):
        var f = Int(ld(y, n + i))
        if f >= 0 and f < f_n:
            lohi[2 * f] = min(lohi[2 * f], i)
            lohi[2 * f + 1] = max(lohi[2 * f + 1], i + 1)
    for f in range(f_n):
        var empty = lohi[2 * f + 1] == 0
        sti(sp, 2 * f, 0 if empty else lohi[2 * f])
        sti(sp, 2 * f + 1, 0 if empty else lohi[2 * f + 1] - lohi[2 * f])
    var part_s = List[Float32](length=max(f_n * FB_C * m, 1), fill=Float32(0))
    var part_n = List[Int32](length=max(f_n * FB_C, 1), fill=Int32(0))
    var ps = FP(unsafe_from_address=Int(part_s.unsafe_ptr()))
    var pn = IP(unsafe_from_address=Int(part_n.unsafe_ptr()))
    for f in range(f_n):
        for c in range(FB_C):
            for col in range(m):
                fb_mean_cell(x, y, n, d, sp, f, c, col, ps, pn)
    for f in range(f_n):
        for col in range(m):
            fb_fold_mean_cell(ps, pn, m, f, col, fst, fcnt)
    var part_g = List[Float32](length=max(f_n * FB_C * m * m, 1), fill=Float32(0))
    var pg = FP(unsafe_from_address=Int(part_g.unsafe_ptr()))
    for f in range(f_n):
        for c in range(FB_C):
            for j in range(m):
                for k in range(j, m):
                    fb_gram_cell_host(x, y, n, d, sp, f, c, j, k, fst, pg)
    for f in range(f_n):
        for j in range(m):
            for k in range(j, m):
                fb_fold_gram_cell(pg, m, f, j, k, fst)
    _ = part_s^
    _ = part_n^
    _ = part_g^
    _ = lohi^


def fb_host_prep(wk: FP, d: Int, f_n: Int, fold: Int, fi: Bool, dst: FP, xm: Int, gg: Int, q: Int, sc: Int):
    """`_prep`'s words for the rows not in `fold` (all rows: fold < 0)."""
    var m = d + 1
    var p = f_n if fold < 0 else fold
    var fcnt = (wk + f_n * fb_words(m)).bitcast[Int32]()
    for u in range(m * m):
        fb_prep_unit(u, wk, fcnt, d, f_n, p, fi, dst, xm, gg, q, sc)


# ------------------------------------------------------------ the device kernels


def fb_means_kernel(x: FP, y: FP, n: Int32, d: Int32, f_n: Int32, spans: IP, part_s: FP, part_n: IP,
                    wf: IP, woff: Int32, nonce: Int32):
    """Thread (f, c, col): `fb_mean_cell`."""
    var m = Int(d) + 1
    var g = Int(block_idx.x) * FB_TPB + Int(thread_idx.x)
    if g < Int(f_n) * FB_C * m:
        var col = g % m
        var fc = g // m
        fb_mean_cell(x, y, Int(n), Int(d), spans, fc // FB_C, fc % FB_C, col, part_s, part_n)
    witness_end(wf, woff, nonce)


def fb_fold_means_kernel(part_s: FP, part_n: IP, d: Int32, f_n: Int32, fst: FP, fcnt: IP,
                         wf: IP, woff: Int32, nonce: Int32):
    """Thread (f, col): `fb_fold_mean_cell`."""
    var m = Int(d) + 1
    var g = Int(block_idx.x) * FB_TPB + Int(thread_idx.x)
    if g < Int(f_n) * m:
        fb_fold_mean_cell(part_s, part_n, m, g // m, g % m, fst, fcnt)
    witness_end(wf, woff, nonce)


@always_inline
def _fb_stage(x: FP, y: FP, n: Int, d: Int, row: Int, col: Int) -> Float32:
    """[X | y] at (row, col), 0 past the edges; loads from clamped in-range
    addresses (the word itself when in range)."""
    var rr = min(row, n - 1)
    var xv = ld(x, rr * d + min(col, d - 1))
    var yv = ld(y, rr)
    var v = yv if col == d else xv
    return v if (row < n and col <= d) else Float32(0)


def fb_gram_kernel(x: FP, y: FP, n: Int32, d: Int32, f_n: Int32, spans: IP, fst: FP, pg: FP,
                   wf: IP, woff: Int32, nonce: Int32):
    """Block b = (f * C + c) * pairs + pair: chunk c of fold f, the tile pair
    upper_cell(pair, tiles) of [X | y]; thread (a, b) runs
    `fb_gram_cell_host`'s statements for the cell (16J + a, 16K + b), j <= k,
    on rows staged in threadgroup memory (the same rows, the same order)."""
    var dd = Int(d)
    var nn = Int(n)
    var m = dd + 1
    var tiles = (m + FB_TC - 1) // FB_TC
    var pairs = tiles * (tiles + 1) // 2
    var b = Int(block_idx.x)
    var pair = b % pairs
    var fc = b // pairs
    var f = fc // FB_C
    var c = fc % FB_C
    var jk = upper_cell(pair, tiles)
    var tid = Int(thread_idx.x)
    var cj = jk[0] * FB_TC
    var ck = jk[1] * FB_TC
    var j = cj + tid // FB_TC
    var k = ck + tid % FB_TC
    var live = j <= k and k < m
    var base = f * fb_words(m)
    var mj = Float32(0)
    var mk = Float32(0)
    if live:
        mj = ld(fst, base + m + j)
        mk = ld(fst, base + m + k)
    var lo = ldi(spans, 2 * f)
    var span = ldi(spans, 2 * f + 1)
    var r_lo = fb_chunk_lo(lo, span, c)
    var r_hi = fb_chunk_lo(lo, span, c + 1)
    var xs = stack_allocation[FB_TR * 2 * FB_TC, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var fs_ = stack_allocation[FB_TR, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var a = tid // FB_TC
    var bb = FB_TC + tid % FB_TC
    var hi = Float32(0)
    var cc = Float32(0)
    var r0 = r_lo
    while r0 < r_hi:
        var cnt = min(FB_TR, r_hi - r0)
        barrier()
        comptime for v in range(FB_LG):
            var u = v * FB_NT + tid
            var col = u % (2 * FB_TC)
            xs[u] = _fb_stage(x, y, nn, dd, r0 + u // (2 * FB_TC), cj + col if col < FB_TC else ck + col - FB_TC)
        if tid < FB_TR:
            var row = r0 + tid
            fs_[tid] = Int32(Int(ld(y, nn + min(row, nn - 1)))) if row < r_hi else Int32(-1)
        barrier()
        if live:
            for r in range(cnt):
                if Int(fs_[r]) == f:
                    fb_dot2(hi, cc, fs(xs[r * 2 * FB_TC + a], mj), fs(xs[r * 2 * FB_TC + bb], mk))
        r0 += cnt
    if live:
        st(pg, fc * m * m + j * m + k, fa(hi, cc))
    witness_end(wf, woff, nonce)


def fb_fold_gram_kernel(pg: FP, d: Int32, f_n: Int32, fst: FP, wf: IP, woff: Int32, nonce: Int32):
    """Thread (f, j, k), j <= k: `fb_fold_gram_cell`."""
    var m = Int(d) + 1
    var g = Int(block_idx.x) * FB_TPB + Int(thread_idx.x)
    if g < Int(f_n) * m * m:
        var u = g % (m * m)
        var j = u // m
        var k = u % m
        if j <= k:
            fb_fold_gram_cell(pg, m, g // (m * m), j, k, fst)
    witness_end(wf, woff, nonce)


def fb_combine_kernel(fst: FP, fcnt: IP, d: Int32, f_n: Int32, fi: Int32, ps: Int32, ew: FP,
                      wf: IP, woff: Int32, nonce: Int32):
    """Thread (p, u): `fb_prep_unit` of training set p (p == F: all rows)
    into prep p of the cd_grid work words (ps words a prep)."""
    var dd = Int(d)
    var m = dd + 1
    var g = Int(block_idx.x) * FB_TPB + Int(thread_idx.x)
    if g < (Int(f_n) + 1) * m * m:
        var p = g // (m * m)
        var base = p * Int(ps)
        fb_prep_unit(g % (m * m), fst, fcnt, dd, Int(f_n), p, fi != 0, ew, base, base + dd,
                     base + dd + dd * dd, base + 2 * dd + dd * dd)
    witness_end(wf, woff, nonce)
