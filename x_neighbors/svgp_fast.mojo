# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SVGP's m x m solve on the device (lane/apple-fast-neighbors2, 2026-10-02;
FAST + Apple only, under `-D MOJOLEARN_SVGP_FAST_GPU=1` from the generated
`op_svgp` driver (x_neighbors/gen.py FAST_ALT), default off).

Cause: `svgp` is a HOST_RUN op (x_neighbors/gen.py): `svgp_item` runs on the
host, three serial m x m Cholesky factorizations (m = 512 on the board) and
column solves over the host pool, a host step inside the fit between two
device ops (`svgp_stats`, `svgp_predict`). Here the same quantities come
from the cholesky lane's device factorization and solve (`potrf_lower`,
`cho_solve`, `chol_logdet` in cholesky/checks/) and elementwise / matmul
kernels of this lane, with only five scalars read back for the collapsed
bound. Same algebra as `svgp_item`: Sigma = Kuu + jitter I + B / noise,
alpha = Sigma^-1 b / noise, C = Kuu^-1 - Sigma^-1, q_mu = Kuu alpha,
q_sqrt = chol(Kuu Sigma^-1 Kuu), elbo as the item's formula; FAST promises
quality, not the item's bits (the blocked factorization and the solves fold
in their own order).
"""
from std.gpu import block_idx, thread_idx
from std.math import log
from max.gpu.host import DeviceBuffer, DeviceContext
from core.identity_trace import IdentityTrace
from core.pinned_reduce import pinned_block_sum
from cholesky.checks.potrf import (
    potrf_lower,
    chol_workspace_floats,
    chol_nb_for,
    chol_default_nb_hint,
    chol_logdet,
    CHOL_PANEL_TPB,
    CHOL_ELEM_TPB,
)
from cholesky.checks.trsm import cho_solve, CHOL_SOLVE_TPB
from x_neighbors.items import FP
from x_neighbors.device_ops import xn_ctx, _grid, _tid, BLOCK, matmul_kernel

comptime SVF_TPB = 256


def svf_ew_kernel(a: FP, b: FP, dst: FP, count_: Int64, m_: Int64, mode_: Int64, s_: Float32):
    """Elementwise over `count` cells of an m x m (or m x 1) buffer:
    0 dst = a; 1 dst = a + s on the diagonal (jitter); 2 dst = a + b / s;
    3 dst = a - b; 4 dst = I; 5 dst = a / s."""
    var t = _tid()
    if t >= Int(count_):
        return
    var m = Int(m_)
    var i = t // m
    var j = t - i * m
    var v = Float32(0)
    if mode_ == 0:
        v = a.unsafe_load(t)
    elif mode_ == 1:
        v = a.unsafe_load(t)
        if i == j:
            v = v + s_
    elif mode_ == 2:
        v = a.unsafe_load(t) + b.unsafe_load(t) / s_
    elif mode_ == 3:
        v = a.unsafe_load(t) - b.unsafe_load(t)
    elif mode_ == 4:
        v = Float32(1) if i == j else Float32(0)
    else:
        v = a.unsafe_load(t) / s_
    dst.unsafe_store(t, v)


#: the folds: SVF_FOLD_BLOCKS blocks of partials over any count, then one
#: block over that fixed number of partials (no one-block launch over a
#: runtime size; the same order every run)
comptime SVF_FOLD_BLOCKS = 64


def svf_fold_partial_kernel(a: FP, b: FP, part: FP, count_: Int64, m_: Int64, mode_: Int64):
    """Block g of SVF_FOLD_BLOCKS: a strided fold of its share of the
    `count` terms, then the block sum into part[g]. 0 sum a[i] * b[i]; 1 sum
    of the diagonal of the m x m a."""
    var tid = Int(thread_idx.x)
    var g = Int(block_idx.x)
    var acc = Float32(0)
    var i = g * SVF_TPB + tid
    while i < Int(count_):
        if mode_ == 0:
            acc += a.unsafe_load(i) * b.unsafe_load(i)
        else:
            acc += a.unsafe_load(i * Int(m_) + i)
        i += SVF_FOLD_BLOCKS * SVF_TPB
    var s = pinned_block_sum[SVF_TPB](acc)
    if tid == 0:
        part.unsafe_store(g, s)


def svf_fold_final_kernel(part: FP, dst: FP):
    """ONE block over the SVF_FOLD_BLOCKS partials (a fixed count): their
    sum to dst[0]."""
    var tid = Int(thread_idx.x)
    var v = part.unsafe_load(tid) if tid < SVF_FOLD_BLOCKS else Float32(0)
    var s = pinned_block_sum[SVF_TPB](v)
    if tid == 0:
        dst.unsafe_store(0, s)


def _ew(ctx: DeviceContext, a: FP, b: FP, dst: FP, count: Int, m: Int, mode: Int, s: Float32) raises:
    ctx.enqueue_function[svf_ew_kernel](
        a, b, dst, Int64(count), Int64(m), Int64(mode), s,
        grid_dim=_grid(count), block_dim=(BLOCK if count > 1 else 1),
    )


def _fold(ctx: DeviceContext, a: FP, b: FP, part: FP, dst: FP, count: Int, m: Int, mode: Int) raises:
    ctx.enqueue_function[svf_fold_partial_kernel](
        a, b, part, Int64(count), Int64(m), Int64(mode), grid_dim=SVF_FOLD_BLOCKS, block_dim=SVF_TPB,
    )
    ctx.enqueue_function[svf_fold_final_kernel](part, dst, grid_dim=1, block_dim=SVF_TPB)


def _ptr(mut buf: DeviceBuffer[DType.float32]) -> FP:
    return buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


def svgp_solve_device(
    kuu: Int, bmat: Int, b: Int, y: Int, alpha: Int, cmat: Int, qmu: Int, qsqrt: Int, info: Int,
    m: Int, n: Int, noise: Float32, jitter: Float32, kdiag: Float32,
) raises:
    """`svgp_item` on the device (module note). info = [elbo, ok]; on a
    failed factorization info = [0, 0] and the other outputs are not
    written, as the item leaves them."""
    var ctx = xn_ctx()
    var mm = m * m
    var d_kuu = ctx.enqueue_create_buffer[DType.float32](mm)
    ctx.enqueue_copy(dst_buf=d_kuu, src_ptr=FP(unsafe_from_address=kuu))
    var d_bm = ctx.enqueue_create_buffer[DType.float32](mm)
    ctx.enqueue_copy(dst_buf=d_bm, src_ptr=FP(unsafe_from_address=bmat))
    var d_b = ctx.enqueue_create_buffer[DType.float32](m)
    ctx.enqueue_copy(dst_buf=d_b, src_ptr=FP(unsafe_from_address=b))
    var d_y = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_buf=d_y, src_ptr=FP(unsafe_from_address=y))
    var d_kj = ctx.enqueue_create_buffer[DType.float32](mm)
    var d_luu = ctx.enqueue_create_buffer[DType.float32](mm)
    var d_ls = ctx.enqueue_create_buffer[DType.float32](mm)
    var d_t = ctx.enqueue_create_buffer[DType.float32](mm)
    var d_e2 = ctx.enqueue_create_buffer[DType.float32](mm)
    var d_c = ctx.enqueue_create_buffer[DType.float32](mm)
    var d_s = ctx.enqueue_create_buffer[DType.float32](mm)
    var d_al = ctx.enqueue_create_buffer[DType.float32](m)
    var d_col = ctx.enqueue_create_buffer[DType.float32](m)
    var d_qmu = ctx.enqueue_create_buffer[DType.float32](m)
    var d_part = ctx.enqueue_create_buffer[DType.float32](SVF_FOLD_BLOCKS)
    var d_sc = ctx.enqueue_create_buffer[DType.float32](4)
    var nb = chol_nb_for(m, chol_default_nb_hint())
    var ws = ctx.enqueue_create_buffer[DType.float32](chol_workspace_floats(m, nb))
    var dwork = ctx.enqueue_create_buffer[DType.float32](m + 1)
    var trace = IdentityTrace()
    trace.header("svgp_fast: m=" + String(m) + " n=" + String(n))
    var pkj = _ptr(d_kj)
    var pluu = _ptr(d_luu)
    var pls = _ptr(d_ls)
    var pt = _ptr(d_t)
    var pe2 = _ptr(d_e2)
    var pc = _ptr(d_c)
    var ps = _ptr(d_s)
    var pal = _ptr(d_al)
    var pcol = _ptr(d_col)
    var psc = _ptr(d_sc)
    var pb = _ptr(d_b)
    var pbm = _ptr(d_bm)
    var ppart = _ptr(d_part)
    # Kuu + jitter I, its copy to factor, Sigma = Kuu_j + B / noise
    _ew(ctx, _ptr(d_kuu), pkj, pkj, mm, m, 1, jitter)
    _ew(ctx, pkj, pkj, pluu, mm, m, 0, Float32(0))
    _ew(ctx, pkj, pbm, pls, mm, m, 2, noise)
    var run1 = potrf_lower(ctx, d_luu, ws, m, trace, chol_default_nb_hint(), CHOL_PANEL_TPB, CHOL_ELEM_TPB)
    var run2 = potrf_lower(ctx, d_ls, ws, m, trace, chol_default_nb_hint(), CHOL_PANEL_TPB, CHOL_ELEM_TPB)
    var inf = FP(unsafe_from_address=info)
    if run1.info != 0 or run2.info != 0:
        ctx.synchronize()
        inf.unsafe_store(0, Float32(0))
        inf.unsafe_store(1, Float32(0))
    else:
        # alpha = Sigma^-1 b / noise; bsb = b . Sigma^-1 b
        _ew(ctx, pb, pb, pcol, m, 1, 0, Float32(0))
        cho_solve(ctx, d_ls, d_col, m, 1, trace, CHOL_SOLVE_TPB)
        _ew(ctx, pcol, pcol, pal, m, 1, 5, noise)
        _fold(ctx, pb, pcol, ppart, psc, m, m, 0)
        # C = Kuu_j^-1 - Sigma^-1
        _ew(ctx, pt, pt, pt, mm, m, 4, Float32(0))
        _ew(ctx, pe2, pe2, pe2, mm, m, 4, Float32(0))
        cho_solve(ctx, d_luu, d_t, m, m, trace, CHOL_SOLVE_TPB)
        cho_solve(ctx, d_ls, d_e2, m, m, trace, CHOL_SOLVE_TPB)
        _ew(ctx, pt, pe2, pc, mm, m, 3, Float32(0))
        # q_mu = Kuu_j alpha
        ctx.enqueue_function[matmul_kernel](
            pkj, pal, _ptr(d_qmu), Int64(m), Int64(m), Int64(1),
            grid_dim=_grid(m), block_dim=(BLOCK if m > 1 else 1),
        )
        # S = Kuu_j Sigma^-1 Kuu_j, then its Cholesky
        _ew(ctx, pkj, pkj, pt, mm, m, 0, Float32(0))
        cho_solve(ctx, d_ls, d_t, m, m, trace, CHOL_SOLVE_TPB)
        ctx.enqueue_function[matmul_kernel](
            pkj, pt, ps, Int64(m), Int64(m), Int64(m),
            grid_dim=_grid(mm), block_dim=(BLOCK if mm > 1 else 1),
        )
        var run3 = potrf_lower(ctx, d_s, ws, m, trace, chol_default_nb_hint(), CHOL_PANEL_TPB, CHOL_ELEM_TPB)
        # tr(Kuu_j^-1 B): the diagonal of Luu^-T Luu^-1 B
        _ew(ctx, pbm, pbm, pt, mm, m, 0, Float32(0))
        cho_solve(ctx, d_luu, d_t, m, m, trace, CHOL_SOLVE_TPB)
        _fold(ctx, pt, pt, ppart, psc + 1, m, m, 1)
        # y . y
        _fold(ctx, _ptr(d_y), _ptr(d_y), ppart, psc + 2, n, m, 0)
        ctx.synchronize()
        var ld_ls = chol_logdet(ctx, d_ls, dwork, m, trace, CHOL_ELEM_TPB)
        var ld_luu = chol_logdet(ctx, d_luu, dwork, m, trace, CHOL_ELEM_TPB)
        var hs = List[Float32](length=4, fill=Float32(0))
        ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=d_sc)
        ctx.enqueue_copy(dst_ptr=FP(unsafe_from_address=alpha), src_buf=d_al)
        ctx.enqueue_copy(dst_ptr=FP(unsafe_from_address=cmat), src_buf=d_c)
        ctx.enqueue_copy(dst_ptr=FP(unsafe_from_address=qmu), src_buf=d_qmu)
        ctx.enqueue_copy(dst_ptr=FP(unsafe_from_address=qsqrt), src_buf=d_s)
        ctx.synchronize()
        var bsb = hs[0]
        var trq = hs[1]
        var yty = hs[2]
        var nf = Float32(n)
        var quad = yty / noise - bsb / (noise * noise)
        var logdet = (ld_ls - ld_luu) + nf * log(noise)
        var trace_term = (nf * kdiag - trq) / noise
        var log2pi = Float32(1.8378770664093453)
        var elbo = -(Float32(0.5) * ((nf * log2pi + logdet) + (quad + trace_term)))
        inf.unsafe_store(0, elbo)
        inf.unsafe_store(1, Float32(1) if run3.info == 0 else Float32(0))
        _ = hs^
    _ = d_kuu^
    _ = d_bm^
    _ = d_b^
    _ = d_y^
    _ = d_kj^
    _ = d_luu^
    _ = d_ls^
    _ = d_t^
    _ = d_e2^
    _ = d_c^
    _ = d_s^
    _ = d_al^
    _ = d_col^
    _ = d_qmu^
    _ = d_part^
    _ = d_sc^
    _ = ws^
    _ = dwork^
    _ = ctx^
