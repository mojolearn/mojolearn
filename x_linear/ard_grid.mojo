# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ARDRegression on the grid (cgr-linear, 2026-10-03).

The fit kernel ran ARD on ONE block: the means and X'y (one thread a column
over every row), the target variance (the lead over every row) and every
iteration's sse (each row's residual across 256 threads). Here:
  * the moments of [X | y] (means, centered Gram, X'y, y's mean) come from
    x_linear/moments_grid.mojo, the host column's chains;
  * the target variance is the blocked fold `bayes_yvar` (block partials,
    a thread each, then folded ascending), as the host column now folds it;
  * per iteration: sigma and the coefficients on one block team
    (`_t_ard_sigma`, `_t_ard_coef`: d-sized), each row's residual a thread
    (`_resid_rows`' statements), each FOLD_BLOCK partial a thread
    (`_sse_part`), then the lead's `ard_update` on one thread (O(d)).
FAST on Apple keeps the Gram sse (`_t_sse_delta`) on the team kernel; the
row kernels skip an iteration whose delta was trusted. A stop sets a state
word every later launch reads first, so iterations queue in batches.
"""
from std.gpu import block_idx, thread_idx
from std.os import getenv
from max.gpu.host import DeviceContext
from x_linear.ops import FP, IP, fs, fd, fa, fmad, ld, st, ldi, sti, i2f, fill, copy
from x_linear.tops import fold_parts, fold_blocks, fold_fa, FOLD_BLOCK
from x_linear.team import device_team, team_work, LINEAR_TPB
from x_linear.witness import Witness, witness_end, WITNESS_TRIES
from x_linear.moments_grid import mg_means_kernel, mg_cross_kernel, mg_tiles, MG_NT
from x_linear.bayes import (
    bayes_yvar_part, _sse_part, _t_ard_sigma, _t_ard_coef, _t_sse_delta, ard_update, ard_layout, ard_finish,
    X_LINEAR_GRAM_SSE,
)

comptime AG_TPB = 256
comptime AG_BATCH = 8

comptime AS_ALPHA = 0
comptime AS_YM = 1
comptime AS_DONE = 2
comptime AS_ITERS = 3
comptime AS_DK = 4
comptime AS_FRESH = 5
comptime AS_SDELTA = 6
comptime AS_S0 = 7
comptime AS_REF = 8
comptime AS_KEPT = 9
comptime AS_WORDS = 16


@always_inline
def _ab(count: Int) -> Int:
    return max((count + AG_TPB - 1) // AG_TPB, 1)


@always_inline
def _gid() -> Int:
    return Int(block_idx.x) * AG_TPB + Int(thread_idx.x)


def ard_yparts_kernel(y: FP, n: Int32, yp: FP, wf: IP, woff: Int32, nonce: Int32):
    """Thread b: row block b's target sum from zero."""
    var nn = Int(n)
    var b = _gid()
    if b < fold_blocks(nn):
        var lo = b * FOLD_BLOCK
        st(yp, b, fold_fa(y, lo, 1, min(FOLD_BLOCK, nn - lo)))
    witness_end(wf, woff, nonce)


def ard_vparts_kernel(y: FP, n: Int32, yp: FP, vp: FP, wf: IP, woff: Int32, nonce: Int32):
    """Thread b: row block b's squared deviations from the blocked mean."""
    var nn = Int(n)
    var nb = fold_blocks(nn)
    var b = _gid()
    if b < nb:
        var m = fd(fold_parts(yp, 0, nb), i2f(nn))
        var lo = b * FOLD_BLOCK
        st(vp, b, bayes_yvar_part(y, m, lo, min(FOLD_BLOCK, nn - lo)))
    witness_end(wf, woff, nonce)


def ard_init_kernel(d: Int32, nb: Int32, nf: Float32, fw: FP, res: FP, iw: IP, vp: FP, state: FP,
                    wf: IP, woff: Int32, nonce: Int32):
    """One thread: the starting alpha (the blocked variance over nb
    partials), y's mean out of its parking slot, lambdas 1, every feature kept."""
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        var dd = Int(d)
        var o = ard_layout(dd)
        for k in range(AS_WORDS):
            st(state, k, Float32(0))
        st(state, AS_YM, ld(fw, o[3]))
        var yvar = fd(fold_parts(vp, 0, Int(nb)), nf)
        st(state, AS_ALPHA, fd(Float32(1), fa(yvar, Float32(1.1920929e-07))))
        st(state, AS_KEPT, Float32(1))
        fill(fw, o[5], dd, Float32(1))
        fill(res, 0, dd, Float32(0))
        for j in range(dd):
            sti(iw, j, 1)
    witness_end(wf, woff, nonce)


@always_inline
def _ard_sc_body(d: Int32, fw: FP, res: FP, iw: IP, state: FP, tw: FP, gram: Int32, final: Int32):
    if final != 0:
        if ld(state, AS_KEPT) == Float32(0):
            return
    elif ld(state, AS_DONE) != Float32(0):
        return
    var dd = Int(d)
    var o = ard_layout(dd)
    var t = device_team(tw, dd, 3, 0)
    var alpha = ld(state, AS_ALPHA)
    var have_ref = ld(state, AS_REF) != Float32(0)
    t.sync()
    var dk = _t_ard_sigma(t, dd, fw, o[1], o[3], o[4], o[5], alpha, iw, 0)
    _t_ard_coef(t, dd, dk, fw, o[4], o[2], alpha, iw, 0, res)
    if final != 0:
        return
    var fresh = 1
    var s = Float32(-1)
    if gram != 0 and have_ref:
        s = _t_sse_delta(t, fw, o[1], o[2], res, o[7], dd, ld(state, AS_S0))
        if s >= 0:
            fresh = 0
    if t.lead():
        st(state, AS_DK, i2f(dk))
        st(state, AS_FRESH, i2f(fresh))
        st(state, AS_SDELTA, s)


def ard_sc_kernel(d: Int32, fw: FP, res: FP, iw: IP, state: FP, tw: FP, gram: Int32, final: Int32,
                  wf: IP, woff: Int32, nonce: Int32):
    """One block team: sigma and the coefficients (d-sized); FAST on Apple
    also the Gram sse's delta and its trust."""
    _ard_sc_body(d, fw, res, iw, state, tw, gram, final)
    witness_end(wf, woff, nonce)


@always_inline
def _ard_rows_live(state: FP) -> Bool:
    return ld(state, AS_DONE) == Float32(0) and ld(state, AS_FRESH) != Float32(0)


def ard_resid_kernel(x: FP, y: FP, n: Int32, d: Int32, fw: FP, res: FP, state: FP, rows: FP,
                     wf: IP, woff: Int32, nonce: Int32):
    """Thread per row: (y_i - ym) - sum_j (x_ij - xm_j) coef_j, j ascending."""
    var i = _gid()
    if i < Int(n) and _ard_rows_live(state):
        var dd = Int(d)
        var p = Float32(0)
        for j in range(dd):
            p = fmad(fs(ld(x, i * dd + j), ld(fw, j)), ld(res, j), p)
        st(rows, i, fs(fs(ld(y, i), ld(state, AS_YM)), p))
    witness_end(wf, woff, nonce)


def ard_part_kernel(rows: FP, y: FP, n: Int32, state: FP, parts: FP, wf: IP, woff: Int32, nonce: Int32):
    """Thread per FOLD_BLOCK rows: the block's squared residuals from zero."""
    var nn = Int(n)
    var b = _gid()
    if b < fold_blocks(nn) and _ard_rows_live(state):
        var lo = b * FOLD_BLOCK
        st(parts, b, _sse_part(rows, y, nn, False, lo, min(FOLD_BLOCK, nn - lo)))
    witness_end(wf, woff, nonce)


def ard_step_kernel(d: Int32, nb: Int32, nf: Float32, fw: FP, res: FP, iw: IP, fp: FP, parts: FP, state: FP,
                    gram: Int32, it: Int32, wf: IP, woff: Int32, nonce: Int32):
    """One thread: the sse (the nb partials folded, or the trusted delta),
    the FAST reference, then `ard_update`."""
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0 and ld(state, AS_DONE) == Float32(0):
        var dd = Int(d)
        var o = ard_layout(dd)
        var fresh = ld(state, AS_FRESH) != Float32(0)
        var sse = fold_parts(parts, 0, Int(nb)) if fresh else ld(state, AS_SDELTA)
        if gram != 0 and fresh:
            st(state, AS_S0, sse)
            st(state, AS_REF, Float32(1))
            copy(fw, o[7], res, 0, dd)
        var r = ard_update(fw, res, iw, dd, nf, Int(ld(state, AS_DK)), sse, ld(state, AS_ALPHA), fp, Int(it))
        st(state, AS_ALPHA, r[0])
        st(state, AS_KEPT, Float32(1) if r[2] else Float32(0))
        st(state, AS_ITERS, i2f(Int(it) + 1))
        if r[1] == 1:
            st(state, AS_DONE, Float32(1))
    witness_end(wf, woff, nonce)


def ard_finish_kernel(d: Int32, fi: Int32, fw: FP, res: FP, state: FP, wf: IP, woff: Int32, nonce: Int32):
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        ard_finish(fw, res, Int(d), fi != 0, ld(state, AS_YM), ld(state, AS_ALPHA),
                   ld(state, AS_KEPT) != Float32(0), Int(ld(state, AS_ITERS)))
    witness_end(wf, woff, nonce)


def ard_fit_grid(
    var ctx: DeviceContext, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], n_out: Int, n_fw: Int, n_iw: Int, res: FP,
) raises:
    """ip: [max_iter, fit_intercept]; fp: [tol, alpha_1, alpha_2, lambda_1,
    lambda_2, threshold_lambda] (x_linear/bayes.mojo `ard_fit`)."""
    var max_iter = Int(ip[0])
    var fi = Int(ip[1])
    var o = ard_layout(d)
    var nb = fold_blocks(n)
    var gram = 0
    comptime if X_LINEAR_GRAM_SSE:
        # FAST on Apple (lane/apple-fast-classical); `=0` is that lane's A/B arm
        if n >= d and String(getenv("MOJOLEARN_X_LINEAR_GRAM_SSE")) != "0":
            gram = 1
    var hfp = fp.copy()
    var dx = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
    var dy = ctx.enqueue_create_buffer[DType.float32](max(n_y, 1))
    var dfp = ctx.enqueue_create_buffer[DType.float32](max(len(hfp), 1))
    var dfw = ctx.enqueue_create_buffer[DType.float32](max(n_fw, o[7] + d))
    var diw = ctx.enqueue_create_buffer[DType.int32](max(n_iw, 2 * d))
    var dout = ctx.enqueue_create_buffer[DType.float32](max(n_out, 2 * d + 3))
    var drows = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
    var dparts = ctx.enqueue_create_buffer[DType.float32](max(nb, 1))
    var dyp = ctx.enqueue_create_buffer[DType.float32](max(nb, 1))
    var dvp = ctx.enqueue_create_buffer[DType.float32](max(nb, 1))
    var dstate = ctx.enqueue_create_buffer[DType.float32](AS_WORDS)
    var dtw = ctx.enqueue_create_buffer[DType.float32](team_work(d, 3, 0))
    if n_x > 0:
        ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    if n_y > 0:
        ctx.enqueue_copy(dst_buf=dy, src_ptr=y)
    ctx.enqueue_copy(dst_buf=dfp, src_ptr=hfp.unsafe_ptr())
    var xp = dx.unsafe_ptr()
    var yp = dy.unsafe_ptr()
    var fwp = dfw.unsafe_ptr()
    var stp = dstate.unsafe_ptr()
    var g_nb = _ab(nb)
    var g_rows = _ab(n)
    var wit = Witness(ctx, max(2 * g_nb + 1, 1 + g_rows + g_nb + 1) + 1)
    var nf = i2f(n)
    var tl = mg_tiles(d, 1)
    var tries = 0
    while True:
        var nonce = wit.begin()
        var wo = 0
        dfw.enqueue_fill(Float32(0))
        diw.enqueue_fill(Int32(0))
        dout.enqueue_fill(Float32(0))
        # the moments of [X | y]: xm at 0, G at d, X'y at d + d*d, y's mean parked in A
        ctx.enqueue_function[mg_means_kernel](
            xp, yp, Int32(n), Int32(d), Int32(1), Int32(fi), fwp, Int32(0), Int32(o[3]),
            grid_dim=tl, block_dim=MG_NT,
        )
        ctx.enqueue_function[mg_cross_kernel](
            xp, yp, Int32(n), Int32(d), Int32(1), fwp, Int32(0), Int32(o[3]), Int32(o[1]), Int32(o[2]),
            grid_dim=tl * (tl + 1) // 2, block_dim=MG_NT,
        )
        ctx.enqueue_function[ard_yparts_kernel](yp, Int32(n), dyp.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                                grid_dim=g_nb, block_dim=AG_TPB)
        wo += g_nb
        ctx.enqueue_function[ard_vparts_kernel](yp, Int32(n), dyp.unsafe_ptr(), dvp.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                                grid_dim=g_nb, block_dim=AG_TPB)
        wo += g_nb
        ctx.enqueue_function[ard_init_kernel](Int32(d), Int32(nb), nf, fwp, dout.unsafe_ptr(), diw.unsafe_ptr(),
                                              dvp.unsafe_ptr(), stp, wit.p(), Int32(wo), nonce, grid_dim=1, block_dim=1)
        wo += 1
        if wit.ok(ctx, wo, "ARD setup"):
            break
        tries += 1
        if tries >= WITNESS_TRIES:
            wit.fail()
    var hst = List[Float32](length=AS_WORDS, fill=Float32(0))
    for it in range(max_iter):
        # in place: a cut raises
        var nonce = wit.begin()
        var wo = 0
        ctx.enqueue_function[ard_sc_kernel](Int32(d), fwp, dout.unsafe_ptr(), diw.unsafe_ptr(), stp, dtw.unsafe_ptr(),
                                            Int32(gram), Int32(0), wit.p(), Int32(wo), nonce, grid_dim=1, block_dim=LINEAR_TPB)
        wo += 1
        ctx.enqueue_function[ard_resid_kernel](xp, yp, Int32(n), Int32(d), fwp, dout.unsafe_ptr(), stp, drows.unsafe_ptr(),
                                               wit.p(), Int32(wo), nonce, grid_dim=g_rows, block_dim=AG_TPB)
        wo += g_rows
        ctx.enqueue_function[ard_part_kernel](drows.unsafe_ptr(), yp, Int32(n), stp, dparts.unsafe_ptr(),
                                              wit.p(), Int32(wo), nonce, grid_dim=g_nb, block_dim=AG_TPB)
        wo += g_nb
        ctx.enqueue_function[ard_step_kernel](Int32(d), Int32(nb), nf, fwp, dout.unsafe_ptr(), diw.unsafe_ptr(),
                                              dfp.unsafe_ptr(), dparts.unsafe_ptr(), stp, Int32(gram), Int32(it),
                                              wit.p(), Int32(wo), nonce, grid_dim=1, block_dim=1)
        wo += 1
        if not wit.ok(ctx, wo, "ARD iteration"):
            wit.fail()
        if (it + 1) % AG_BATCH == 0 and it + 1 < max_iter:
            ctx.enqueue_copy(dst_ptr=hst.unsafe_ptr(), src_buf=dstate)
            ctx.synchronize()
            if hst[AS_DONE] != Float32(0):
                break
    var nf2 = wit.begin()
    ctx.enqueue_function[ard_sc_kernel](Int32(d), fwp, dout.unsafe_ptr(), diw.unsafe_ptr(), stp, dtw.unsafe_ptr(),
                                        Int32(0), Int32(1), wit.p(), Int32(0), nf2, grid_dim=1, block_dim=LINEAR_TPB)
    ctx.enqueue_function[ard_finish_kernel](Int32(d), Int32(fi), fwp, dout.unsafe_ptr(), stp, wit.p(), Int32(1), nf2,
                                            grid_dim=1, block_dim=1)
    if not wit.ok(ctx, 2, "ARD finish"):
        wit.fail()
    if n_out > 0:
        ctx.enqueue_copy(dst_ptr=res, src_buf=dout.create_sub_buffer[DType.float32](0, n_out))
    ctx.synchronize()
    _ = hfp^
    _ = hst^
    _ = dx^
    _ = dy^
    _ = dfp^
    _ = dfw^
    _ = diw^
    _ = dout^
    _ = drows^
    _ = dparts^
    _ = dyp^
    _ = dvp^
    _ = dstate^
    _ = dtw^
    _ = wit^
