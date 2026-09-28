# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""t-SNE on the device: one thread per cell of `x_ann/tsne_core.mojo`; the
symmetrization is the shared host function (`tsne_symmetrize`)."""

from std.gpu import block_idx, block_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext
from x_ann.device_ctx import x_ann_ctx
from x_ann.stage_timer import AnnStages
from x_ann.knn_device import knn_enqueue

from metrics.checks.device_io import upload_f32, upload_i32, download_f32, download_i32
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz, identical_log
from std.sys.info import has_apple_gpu_accelerator
from x_ann.switches import ANN3_TSNE_RB32, ANN3_TSNE_RB64
from x_ann.tsne_core import (
    F32P, I32P, ts_kl_cell, ts_perplexity_cell, ts_repulse_fold, ts_repulse_pair, ts_repulse_terms, ts_step_cell,
    ts_sum_cell, tsne_nn, tsne_symmetrize, tsne_validate,
)

comptime TPB = 128


def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def _grid(count: Int) -> Int:
    return (count + TPB - 1) // TPB


def perplexity_kernel(n: Int32, nn_d: F32P, nn: Int32, log_perp: Float32, p: F32P):
    var i = _tid()
    if i < Int(n):
        ts_perplexity_cell(i, nn_d, Int(nn), log_perp, p)


comptime _FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
#: rows per threadgroup and staged candidate rows per tile of the repulsion
#: (lane ann-apple3, OPT-IN trials under FAST on Apple: 32 or 64 rows per
#: threadgroup, x_ann/switches.mojo)
comptime RTB = 32 if (_FAST_APPLE and ANN3_TSNE_RB32) else (64 if (_FAST_APPLE and ANN3_TSNE_RB64) else 128)
comptime RTJ = 256


def repulse_tiled_kernel(n: Int32, y: F32P, row_z: F32P, rep: F32P):
    """`ts_repulse_cell` with the candidate rows staged in threadgroup memory
    (lane ann-apple): tiles and rows ascending, so row i folds j = 0, 1, ...,
    n - 1 in the cell's order, each j through the cell's own
    `ts_repulse_pair` on ftz(y) (the staged values are flushed once; ftz is
    idempotent). The same bits."""
    var t = Int(thread_idx.x)
    var nr = Int(n)
    var i = Int(block_idx.x) * RTB + t
    var tile = stack_allocation[2 * RTJ, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var live = i < nr
    var y0 = Float32(0.0)
    var y1 = Float32(0.0)
    if live:
        y0 = ftz(y.unsafe_load(2 * i))
        y1 = ftz(y.unsafe_load(2 * i + 1))
    var z = Float32(0.0)
    var r0 = Float32(0.0)
    var r1 = Float32(0.0)
    var j0 = 0
    while j0 < nr:
        for e in range(t, 2 * RTJ, RTB):
            var v = Float32(0.0)
            if 2 * j0 + e < 2 * nr:
                v = ftz(y.unsafe_load(2 * j0 + e))
            tile[e] = v
        barrier()
        if live:
            var jn = RTJ if nr - j0 > RTJ else nr - j0
            # four j at a time: their terms are independent, so they are
            # formed first and folded after in ascending j (the same
            # statements in the same fold order; only the independent work
            # overlaps). The group holding row i itself takes the plain loop.
            var r = 0
            while r + 4 <= jn:
                if i >= j0 + r and i < j0 + r + 4:
                    for u in range(4):
                        if j0 + r + u != i:
                            ts_repulse_pair(y0, y1, tile[2 * (r + u)], tile[2 * (r + u) + 1], z, r0, r1)
                else:
                    var ta = ts_repulse_terms(y0, y1, tile[2 * r], tile[2 * r + 1])
                    var tb = ts_repulse_terms(y0, y1, tile[2 * r + 2], tile[2 * r + 3])
                    var tc = ts_repulse_terms(y0, y1, tile[2 * r + 4], tile[2 * r + 5])
                    var td = ts_repulse_terms(y0, y1, tile[2 * r + 6], tile[2 * r + 7])
                    ts_repulse_fold(ta, z, r0, r1)
                    ts_repulse_fold(tb, z, r0, r1)
                    ts_repulse_fold(tc, z, r0, r1)
                    ts_repulse_fold(td, z, r0, r1)
                r += 4
            while r < jn:
                if j0 + r != i:
                    ts_repulse_pair(y0, y1, tile[2 * r], tile[2 * r + 1], z, r0, r1)
                r += 1
        barrier()
        j0 += RTJ
    if live:
        row_z.unsafe_store(i, z)
        rep.unsafe_store(2 * i, r0)
        rep.unsafe_store(2 * i + 1, r1)


def sum_kernel(n: Int32, row_z: F32P, z: F32P):
    if _tid() == 0:
        ts_sum_cell(row_z, Int(n), z)


def step_kernel(
    count: Int32, y: F32P, y_new: F32P, indptr: I32P, indices: I32P, values: F32P, rep: F32P,
    z: F32P, update: F32P, gains: F32P, exaggeration: Float32, momentum: Float32, lr: Float32,
):
    var e = _tid()
    if e < Int(count):
        ts_step_cell(e, y, y_new, indptr, indices, values, rep, z, update, gains, exaggeration, momentum, lr)


def kl_kernel(n: Int32, y: F32P, indptr: I32P, indices: I32P, values: F32P, z: F32P, kl: F32P):
    var i = _tid()
    if i < Int(n):
        ts_kl_cell(i, y, indptr, indices, values, z, kl)


def _ts_iter(
    ctx: DeviceContext, mut ycur: DeviceBuffer[DType.float32], mut ynext: DeviceBuffer[DType.float32], n: Int,
    mut dptr: DeviceBuffer[DType.int32], mut dind: DeviceBuffer[DType.int32], mut dval: DeviceBuffer[DType.float32],
    mut drz: DeviceBuffer[DType.float32], mut drep: DeviceBuffer[DType.float32], mut dz: DeviceBuffer[DType.float32],
    mut dupd: DeviceBuffer[DType.float32], mut dgain: DeviceBuffer[DType.float32], ex: Float32, mom: Float32,
    lr: Float32,
) raises:
    ctx.enqueue_function[repulse_tiled_kernel](Int32(n), ycur.unsafe_ptr(), drz.unsafe_ptr(), drep.unsafe_ptr(),
                                               grid_dim=(n + RTB - 1) // RTB, block_dim=RTB)
    ctx.enqueue_function[sum_kernel](Int32(n), drz.unsafe_ptr(), dz.unsafe_ptr(), grid_dim=1, block_dim=1)
    ctx.enqueue_function[step_kernel](
        Int32(2 * n), ycur.unsafe_ptr(), ynext.unsafe_ptr(), dptr.unsafe_ptr(), dind.unsafe_ptr(),
        dval.unsafe_ptr(), drep.unsafe_ptr(), dz.unsafe_ptr(), dupd.unsafe_ptr(), dgain.unsafe_ptr(), ex,
        mom, lr, grid_dim=_grid(2 * n), block_dim=TPB,
    )


def _ts_kl(
    ctx: DeviceContext, mut y: DeviceBuffer[DType.float32], n: Int, mut dptr: DeviceBuffer[DType.int32],
    mut dind: DeviceBuffer[DType.int32], mut dval: DeviceBuffer[DType.float32], mut drz: DeviceBuffer[DType.float32],
    mut drep: DeviceBuffer[DType.float32], mut dz: DeviceBuffer[DType.float32], mut dkl: DeviceBuffer[DType.float32],
) raises:
    ctx.enqueue_function[repulse_tiled_kernel](Int32(n), y.unsafe_ptr(), drz.unsafe_ptr(), drep.unsafe_ptr(),
                                               grid_dim=(n + RTB - 1) // RTB, block_dim=RTB)
    ctx.enqueue_function[sum_kernel](Int32(n), drz.unsafe_ptr(), dz.unsafe_ptr(), grid_dim=1, block_dim=1)
    ctx.enqueue_function[kl_kernel](Int32(n), y.unsafe_ptr(), dptr.unsafe_ptr(), dind.unsafe_ptr(),
                                    dval.unsafe_ptr(), dz.unsafe_ptr(), dkl.unsafe_ptr(), grid_dim=_grid(n),
                                    block_dim=TPB)


def tsne_fit_device(
    x: List[Float32], n: Int, d: Int, y0: List[Float32], perplexity: Float32, exaggeration: Float32,
    learning_rate: Float32, max_iter: Int, exploration: Int, mut y_out: List[Float32], mut kl_out: Float32,
) raises:
    tsne_validate(n, d, perplexity, max_iter, exploration)
    var nn = tsne_nn(n, perplexity)
    var st = AnnStages("tsne_fit")
    var ctx = x_ann_ctx()
    var dx = upload_f32(ctx, x)
    st.mark(ctx, "upload")
    var dnd = ctx.enqueue_create_buffer[DType.float32](n * nn)
    var dni = ctx.enqueue_create_buffer[DType.int32](n * nn)
    var dp = ctx.enqueue_create_buffer[DType.float32](n * nn)
    knn_enqueue(ctx, dx, n, d, nn, dnd, dni)
    ctx.enqueue_function[perplexity_kernel](Int32(n), dnd.unsafe_ptr(), Int32(nn), identical_log(perplexity),
                                            dp.unsafe_ptr(), grid_dim=_grid(n), block_dim=TPB)
    ctx.synchronize()
    st.host("knn_perplexity")
    var nn_i = download_i32(ctx, dni, n * nn)
    var p_cond = download_f32(ctx, dp, n * nn)
    var indptr = List[Int32]()
    var indices = List[Int32]()
    var values = List[Float32]()
    tsne_symmetrize(n, nn, nn_i, p_cond, indptr, indices, values)
    st.host("symmetrize")

    var dptr = upload_i32(ctx, indptr)
    var dind = upload_i32(ctx, indices)
    var dval = upload_f32(ctx, values)
    var dy = upload_f32(ctx, y0)
    var dy2 = upload_f32(ctx, y0)
    var dupd = upload_f32(ctx, List[Float32](length=2 * n, fill=Float32(0.0)))
    var dgain = upload_f32(ctx, List[Float32](length=2 * n, fill=Float32(1.0)))
    var drz = ctx.enqueue_create_buffer[DType.float32](n)
    var drep = ctx.enqueue_create_buffer[DType.float32](2 * n)
    var dz = ctx.enqueue_create_buffer[DType.float32](1)
    var dkl = ctx.enqueue_create_buffer[DType.float32](n)
    st.mark(ctx, "upload_graph")
    for it in range(max_iter):
        var ex = exaggeration if it < exploration else Float32(1.0)
        var mom = Float32(0.5) if it < exploration else Float32(0.8)
        if it % 2 == 0:
            _ts_iter(ctx, dy, dy2, n, dptr, dind, dval, drz, drep, dz, dupd, dgain, ex, mom, learning_rate)
        else:
            _ts_iter(ctx, dy2, dy, n, dptr, dind, dval, drz, drep, dz, dupd, dgain, ex, mom, learning_rate)
    st.mark(ctx, "iterations")
    if max_iter % 2 == 0:
        _ts_kl(ctx, dy, n, dptr, dind, dval, drz, drep, dz, dkl)
    else:
        _ts_kl(ctx, dy2, n, dptr, dind, dval, drz, drep, dz, dkl)
    ctx.synchronize()
    if max_iter % 2 == 0:
        y_out = download_f32(ctx, dy, 2 * n)
    else:
        y_out = download_f32(ctx, dy2, 2 * n)
    var kl = download_f32(ctx, dkl, n)
    var total = Float32(0.0)
    for i in range(n):
        total = total + kl[i]
    kl_out = total
    st.host("kl_download")
    _ = dkl^
    _ = dz^
    _ = drep^
    _ = drz^
    _ = dgain^
    _ = dupd^
    _ = dy2^
    _ = dy^
    _ = dval^
    _ = dind^
    _ = dptr^
    _ = dp^
    _ = dni^
    _ = dnd^
    _ = dx^
    _ = ctx^
