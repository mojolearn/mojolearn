# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LassoCV / ElasticNetCV on the whole device (lane/neural-pass109, 2026-10-02).

The team form (x_linear/cd.mojo `enetcv_fit`) ran the whole fit on ONE
block: every fold's Gram on 256 threads, every path's coordinate descent on
the lead thread, every held-out sum on the lead. bench_board 0.8.34: taxi
3.2 s on the M3 Ultra against sklearn's 0.23, istella 76 s against 5.5.
Here the fit is launches:
  1. the means (and the y mean, the training row count) of every fold and of
     the full data, one thread per value;
  2. their Gram, X'y and |yc|^2, one thread per value;
  3. the alpha grids from the full data's X'y, one thread;
  4. one block per (fold, l1_ratio) path, the coordinate descent on the
     block (`t_enet_gram_cd`), warm started along the alphas;
  5. one thread per (fold, l1_ratio, alpha) held-out squared-error sum;
  6. one block: the choice, then the refit on the full data from zero.
Every value is the host schedule's statements over the same rows in the
same order (cd.mojo `ecv_*`), so every word is the host's.
`MOJOLEARN_X_LINEAR_ENETCV_GRID=0` restores the one-block team fit.
"""
from std.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext
from x_linear.ops import FP, IP, fm, fs, fd, ld, st, ldi, i2f, fill
from x_linear.team import Team, TEAM_SLOTS, LINEAR_TPB, team_at
from x_linear.tops import upper_cell
from x_linear.cd import (
    ecv_alphas, ecv_choose, ecv_finish, ecv_rows, ecv_fold_fa, ecv_cfmad, ecv_held_sse,
    t_enet_gram_cd,
)

comptime ECV_TPB = 64


struct _EcvLayout(ImplicitlyCopyable, Movable):
    """Words of the fit's work buffer: P = F + 1 preps (prep F the full
    data), each xm d | G d*d | q d | (y mean, |yc|^2, rows); then F*L + 1
    paths (the last the refit), each Qw d | w d | path A*(d + 1)."""
    var d: Int
    var a_n: Int
    var f_n: Int
    var l_n: Int
    var ps: Int
    var rs: Int

    def __init__(out self, d: Int, a_n: Int, f_n: Int, l_n: Int):
        self.d = d
        self.a_n = a_n
        self.f_n = f_n
        self.l_n = l_n
        self.ps = 2 * d + d * d + 3
        self.rs = 2 * d + a_n * (d + 1)

    @always_inline
    def prep(self, p: Int) -> Int:
        return p * self.ps

    @always_inline
    def path(self, r: Int) -> Int:
        return (self.f_n + 1) * self.ps + r * self.rs

    def words(self) -> Int:
        return self.path(self.f_n * self.l_n + 1)


@always_inline
def _fold_of(p: Int, f_n: Int) -> Int:
    return -1 if p == f_n else p


def ecv_means_kernel(x: FP, y: FP, n: Int32, d: Int32, ip: IP, ew: FP):
    """Thread (p, j): column j's mean of prep p (j == d: the y mean and the
    training row count)."""
    var dd = Int(d)
    var nn = Int(n)
    var lay = _EcvLayout(dd, ldi(ip, 2), ldi(ip, 3), ldi(ip, 4))
    var g = Int(block_idx.x) * ECV_TPB + Int(thread_idx.x)
    if g >= (lay.f_n + 1) * (dd + 1):
        return
    var p = g // (dd + 1)
    var j = g % (dd + 1)
    var fold = _fold_of(p, lay.f_n)
    var fi = ldi(ip, 1) != 0
    var fid = y + nn
    var base = lay.prep(p)
    var rows = ecv_rows(fid, nn, fold)
    if j < dd:
        st(ew, base + j, fd(ecv_fold_fa(x, j, dd, fid, nn, fold), i2f(rows)) if fi else Float32(0))
    else:
        var sc = base + 2 * dd + dd * dd
        st(ew, sc, fd(ecv_fold_fa(y, 0, 1, fid, nn, fold), i2f(rows)) if fi else Float32(0))
        st(ew, sc + 2, i2f(rows))


def ecv_gram_kernel(x: FP, y: FP, n: Int32, d: Int32, ip: IP, ew: FP):
    """Thread (p, c): Gram cell c (upper triangle, mirrored), X'y entry or
    |yc|^2 of prep p, from the means of `ecv_means_kernel`."""
    var dd = Int(d)
    var nn = Int(n)
    var lay = _EcvLayout(dd, ldi(ip, 2), ldi(ip, 3), ldi(ip, 4))
    var cells = dd * (dd + 1) // 2
    var per = cells + dd + 1
    var g = Int(block_idx.x) * ECV_TPB + Int(thread_idx.x)
    if g >= (lay.f_n + 1) * per:
        return
    var p = g // per
    var c = g % per
    var fold = _fold_of(p, lay.f_n)
    var fid = y + nn
    var base = lay.prep(p)
    var gg = base + dd
    var q = gg + dd * dd
    var sc = q + dd
    var ym = ld(ew, sc)
    if c < cells:
        var jk = upper_cell(c, dd)
        var j = jk[0]
        var k = jk[1]
        var acc = ecv_cfmad(x, j, dd, ld(ew, base + j), x, k, dd, ld(ew, base + k), fid, nn, fold)
        st(ew, gg + j * dd + k, acc)
        st(ew, gg + k * dd + j, acc)
    elif c < cells + dd:
        var j = c - cells
        st(ew, q + j, ecv_cfmad(x, j, dd, ld(ew, base + j), y, 0, 1, ym, fid, nn, fold))
    else:
        st(ew, sc + 1, ecv_cfmad(y, 0, 1, ym, y, 0, 1, ym, fid, nn, fold))


def ecv_grid_kernel(n: Int32, d: Int32, ip: IP, fp: FP, res: FP, ew: FP):
    """One thread: the alpha grids from the full data's X'y."""
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    var dd = Int(d)
    var lay = _EcvLayout(dd, ldi(ip, 2), ldi(ip, 3), ldi(ip, 4))
    var q = lay.prep(lay.f_n) + dd + dd * dd
    ecv_alphas(res, dd + 4, fp, lay.l_n, lay.a_n, ldi(ip, 5) != 0, ld(fp, 0), ew, q, dd, Int(n))


def ecv_path_kernel(d: Int32, ip: IP, fp: FP, res: FP, ew: FP, tw: FP):
    """Block r = f * L + l: the warm-started path of fold f at l1_ratio l,
    each point's (coef, intercept) into the path's words."""
    var dd = Int(d)
    var lay = _EcvLayout(dd, ldi(ip, 2), ldi(ip, 3), ldi(ip, 4))
    var r = Int(block_idx.x)
    var f = r // lay.l_n
    var l = r % lay.l_n
    var t = team_at(Int(thread_idx.x), Int(block_dim.x), tw + r * TEAM_SLOTS, 0, 0, 0)
    var base = lay.prep(f)
    var gg = base + dd
    var q = gg + dd * dd
    var sc = q + dd
    var qw = lay.path(r)
    var w = qw + dd
    var pb = w + dd
    var ym = ld(ew, sc)
    var yn = ld(ew, sc + 1)
    var rows = ld(ew, sc + 2)
    var l1r = ld(fp, 2 + l)
    for j in range(t.tid, dd, t.nt):
        st(ew, w + j, Float32(0))
    t.sync()
    for k in range(lay.a_n):
        var alpha = ld(res, dd + 4 + l * lay.a_n + k)
        var l1 = fm(fm(alpha, l1r), rows)
        var l2 = fm(fm(alpha, fs(Float32(1), l1r)), rows)
        _ = t_enet_gram_cd(t, ew, gg, q, qw, w, dd, yn, l1, l2, ldi(ip, 0), ld(fp, 1), ldi(ip, 6) != 0)
        if t.lead():
            var b = ym
            for j in range(dd):
                b = fs(b, fm(ld(ew, base + j), ld(ew, w + j)))
            for j in range(dd):
                st(ew, pb + k * (dd + 1) + j, ld(ew, w + j))
            st(ew, pb + k * (dd + 1) + dd, b)
        t.sync()


def ecv_score_kernel(x: FP, y: FP, n: Int32, d: Int32, ip: IP, res: FP, ew: FP):
    """Thread (f, l, k): the held-out mean squared error of that path point
    into res's mse words."""
    var dd = Int(d)
    var nn = Int(n)
    var lay = _EcvLayout(dd, ldi(ip, 2), ldi(ip, 3), ldi(ip, 4))
    var g = Int(block_idx.x) * ECV_TPB + Int(thread_idx.x)
    if g >= lay.f_n * lay.l_n * lay.a_n:
        return
    var k = g % lay.a_n
    var r = g // lay.a_n
    var f = r // lay.l_n
    var l = r % lay.l_n
    var n_te = nn - Int(ld(ew, lay.prep(f) + 2 * dd + dd * dd + 2))
    var o = lay.path(r) + 2 * dd + k * (dd + 1)
    var acc = ecv_held_sse(x, y, y + nn, nn, dd, f, ew, o)
    var mse = dd + 4 + lay.l_n * lay.a_n
    st(res, mse + (l * lay.a_n + k) * lay.f_n + f, fd(acc, i2f(n_te)) if n_te > 0 else Float32(0))


def ecv_refit_kernel(n: Int32, d: Int32, ip: IP, fp: FP, res: FP, ew: FP, tw: FP):
    """One block: the choice, then the refit on the full data from zero."""
    var dd = Int(d)
    var lay = _EcvLayout(dd, ldi(ip, 2), ldi(ip, 3), ldi(ip, 4))
    var r = lay.f_n * lay.l_n
    var t = team_at(Int(thread_idx.x), Int(block_dim.x), tw + r * TEAM_SLOTS, 0, 0, 0)
    if t.lead():
        ecv_choose(res, fp, dd, lay.l_n, lay.a_n, lay.f_n)
    var base = lay.prep(lay.f_n)
    var gg = base + dd
    var q = gg + dd * dd
    var sc = q + dd
    var qw = lay.path(r)
    var w = qw + dd
    for j in range(t.tid, dd, t.nt):
        st(ew, w + j, Float32(0))
    t.sync()
    var alpha = ld(res, dd + 1)
    var l1r = ld(res, dd + 2)
    var nf = i2f(Int(n))
    var iters = t_enet_gram_cd(t, ew, gg, q, qw, w, dd, ld(ew, sc + 1), fm(fm(alpha, l1r), nf),
                               fm(fm(alpha, fs(Float32(1), l1r)), nf), ldi(ip, 0), ld(fp, 1), ldi(ip, 6) != 0)
    if t.lead():
        ecv_finish(res, ew, base, w, sc, dd, ldi(ip, 1) != 0, alpha, l1r, iters)


def _blocks(count: Int) -> Int:
    return max((count + ECV_TPB - 1) // ECV_TPB, 1)


def enetcv_fit_grid(
    ctx: DeviceContext, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], n_out: Int, res: FP,
) raises:
    """`enetcv_fit` as the launches above (ip, fp, y, res as enetcv_fit's)."""
    var a_n = Int(ip[2])
    var f_n = Int(ip[3])
    var l_n = Int(ip[4])
    var lay = _EcvLayout(d, a_n, f_n, l_n)
    var paths = f_n * l_n
    var nt = min(LINEAR_TPB, max(32, (d + 31) // 32 * 32))
    var hip = ip.copy()
    var hfp = fp.copy()
    var dx = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
    var dy = ctx.enqueue_create_buffer[DType.float32](max(n_y, 1))
    var dip = ctx.enqueue_create_buffer[DType.int32](max(len(hip), 1))
    var dfp = ctx.enqueue_create_buffer[DType.float32](max(len(hfp), 1))
    var dout = ctx.enqueue_create_buffer[DType.float32](max(n_out, 1))
    var dew = ctx.enqueue_create_buffer[DType.float32](max(lay.words(), 1))
    var dtw = ctx.enqueue_create_buffer[DType.float32]((paths + 1) * TEAM_SLOTS)
    if n_x > 0:
        ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    if n_y > 0:
        ctx.enqueue_copy(dst_buf=dy, src_ptr=y)
    ctx.enqueue_copy(dst_buf=dip, src_ptr=hip.unsafe_ptr())
    if len(hfp) > 0:
        ctx.enqueue_copy(dst_buf=dfp, src_ptr=hfp.unsafe_ptr())
    dout.enqueue_fill(Float32(0))
    dew.enqueue_fill(Float32(0))
    dtw.enqueue_fill(Float32(0))
    ctx.enqueue_function[ecv_means_kernel](
        dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), dip.unsafe_ptr(), dew.unsafe_ptr(),
        grid_dim=_blocks((f_n + 1) * (d + 1)), block_dim=ECV_TPB,
    )
    ctx.enqueue_function[ecv_gram_kernel](
        dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), dip.unsafe_ptr(), dew.unsafe_ptr(),
        grid_dim=_blocks((f_n + 1) * (d * (d + 1) // 2 + d + 1)), block_dim=ECV_TPB,
    )
    ctx.enqueue_function[ecv_grid_kernel](
        Int32(n), Int32(d), dip.unsafe_ptr(), dfp.unsafe_ptr(), dout.unsafe_ptr(), dew.unsafe_ptr(),
        grid_dim=1, block_dim=1,
    )
    if paths > 0:
        ctx.enqueue_function[ecv_path_kernel](
            Int32(d), dip.unsafe_ptr(), dfp.unsafe_ptr(), dout.unsafe_ptr(), dew.unsafe_ptr(), dtw.unsafe_ptr(),
            grid_dim=paths, block_dim=nt,
        )
        ctx.enqueue_function[ecv_score_kernel](
            dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), dip.unsafe_ptr(), dout.unsafe_ptr(),
            dew.unsafe_ptr(), grid_dim=_blocks(paths * a_n), block_dim=ECV_TPB,
        )
    ctx.enqueue_function[ecv_refit_kernel](
        Int32(n), Int32(d), dip.unsafe_ptr(), dfp.unsafe_ptr(), dout.unsafe_ptr(), dew.unsafe_ptr(),
        dtw.unsafe_ptr(), grid_dim=1, block_dim=nt,
    )
    if n_out > 0:
        ctx.enqueue_copy(dst_ptr=res, src_buf=dout)
    ctx.synchronize()
    _ = hip^
    _ = hfp^
    _ = dx^
    _ = dy^
    _ = dip^
    _ = dfp^
    _ = dout^
    _ = dew^
    _ = dtw^
