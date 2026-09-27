# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""t-SNE on the device: one thread per cell of `x_ann/tsne_core.mojo`; the
symmetrization and the stop decisions are shared host code
(`tsne_symmetrize`, `ts_fold`, `ts_sq_fold`), so the host driver
(`x_ann/host/tsne_host.mojo`) takes the same branch at every check.

The schedule is sklearn's `TSNE._tsne` + `_gradient_descent`: an
exploration phase (exaggerated P, momentum 0.5, patience 250) then the main
phase (momentum 0.8, patience n_iter_without_progress), update and gains
reset at the phase change, the error and grad norm every 50 steps and at a
phase's last step, a stop on no progress or on grad norm <= min_grad_norm."""

from std.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from metrics.checks.device_io import upload_f32, upload_i32, download_f32, download_i32
from checks.numerics import ftz, identical_log, identical_sqrt
from x_ann.tsne_core import (
    F32P, I32P, ts_fold, ts_kl_cell, ts_knn_cell, ts_perplexity_cell, ts_repulse_cell, ts_sq_fold,
    ts_step_cell, ts_sum_cell, tsne_nn, tsne_symmetrize, tsne_validate,
)

comptime TPB = 128
comptime TS_CHECK_EVERY = 50
comptime TS_EXPLORATION_PATIENCE = 250


def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def _grid(count: Int) -> Int:
    return (count + TPB - 1) // TPB


def knn_kernel(n: Int32, x: F32P, d: Int32, nn: Int32, nn_d: F32P, nn_i: I32P):
    var i = _tid()
    if i < Int(n):
        ts_knn_cell(i, x, Int(n), Int(d), Int(nn), nn_d, nn_i)


def perplexity_kernel(n: Int32, nn_d: F32P, nn: Int32, log_perp: Float32, p: F32P):
    var i = _tid()
    if i < Int(n):
        ts_perplexity_cell(i, nn_d, Int(nn), log_perp, p)


def repulse_kernel(count: Int32, y: F32P, n: Int32, nc: Int32, dof: Int32, row_z: F32P, rep: F32P):
    var e = _tid()
    if e < Int(count):
        ts_repulse_cell(e, y, Int(n), Int(nc), Int(dof), row_z, rep)


def sum_kernel(n: Int32, row_z: F32P, z: F32P):
    if _tid() == 0:
        ts_sum_cell(row_z, Int(n), z)


def step_kernel(
    count: Int32, y: F32P, y_new: F32P, nc: Int32, dof: Int32, indptr: I32P, indices: I32P, values: F32P,
    rep: F32P, z: F32P, update: F32P, gains: F32P, gbuf: F32P, exaggeration: Float32, momentum: Float32,
    lr: Float32,
):
    var e = _tid()
    if e < Int(count):
        ts_step_cell(e, y, y_new, Int(nc), Int(dof), indptr, indices, values, rep, z, update, gains, gbuf,
                     exaggeration, momentum, lr)


def kl_kernel(n: Int32, y: F32P, nc: Int32, dof: Int32, indptr: I32P, indices: I32P, values: F32P, z: F32P,
              ex: Float32, kl: F32P):
    var i = _tid()
    if i < Int(n):
        ts_kl_cell(i, y, Int(nc), Int(dof), indptr, indices, values, z, ex, kl)


def _ts_iter(
    ctx: DeviceContext, mut ycur: DeviceBuffer[DType.float32], mut ynext: DeviceBuffer[DType.float32], n: Int,
    nc: Int, dof: Int, mut dptr: DeviceBuffer[DType.int32], mut dind: DeviceBuffer[DType.int32],
    mut dval: DeviceBuffer[DType.float32], mut drz: DeviceBuffer[DType.float32],
    mut drep: DeviceBuffer[DType.float32], mut dz: DeviceBuffer[DType.float32],
    mut dupd: DeviceBuffer[DType.float32], mut dgain: DeviceBuffer[DType.float32],
    mut dg: DeviceBuffer[DType.float32], mut dkl: DeviceBuffer[DType.float32], ex: Float32, mom: Float32,
    lr: Float32, check: Bool, mut err: Float32, mut gnorm: Float32,
) raises:
    ctx.enqueue_function[repulse_kernel](Int32(n * nc), ycur.unsafe_ptr(), Int32(n), Int32(nc), Int32(dof),
                                         drz.unsafe_ptr(), drep.unsafe_ptr(), grid_dim=_grid(n * nc), block_dim=TPB)
    ctx.enqueue_function[sum_kernel](Int32(n), drz.unsafe_ptr(), dz.unsafe_ptr(), grid_dim=1, block_dim=1)
    if check:
        ctx.enqueue_function[kl_kernel](Int32(n), ycur.unsafe_ptr(), Int32(nc), Int32(dof), dptr.unsafe_ptr(),
                                        dind.unsafe_ptr(), dval.unsafe_ptr(), dz.unsafe_ptr(), ex, dkl.unsafe_ptr(),
                                        grid_dim=_grid(n), block_dim=TPB)
    ctx.enqueue_function[step_kernel](
        Int32(n * nc), ycur.unsafe_ptr(), ynext.unsafe_ptr(), Int32(nc), Int32(dof), dptr.unsafe_ptr(),
        dind.unsafe_ptr(), dval.unsafe_ptr(), drep.unsafe_ptr(), dz.unsafe_ptr(), dupd.unsafe_ptr(),
        dgain.unsafe_ptr(), dg.unsafe_ptr(), ex, mom, lr, grid_dim=_grid(n * nc), block_dim=TPB,
    )
    if check:
        ctx.synchronize()
        err = ts_fold(download_f32(ctx, dkl, n))
        gnorm = ftz(identical_sqrt(ts_sq_fold(download_f32(ctx, dg, n * nc))))


def tsne_fit_device(
    x: List[Float32], n: Int, d: Int, nc: Int, y0: List[Float32], perplexity: Float32, exaggeration: Float32,
    learning_rate: Float32, max_iter: Int, exploration: Int, exact: Bool, n_iter_without_progress: Int,
    min_grad_norm: Float32, mut y_out: List[Float32], mut kl_out: Float32, mut n_iter_out: Int,
) raises:
    tsne_validate(n, d, nc, perplexity, max_iter, exploration)
    var nn = tsne_nn(n, perplexity, exact)
    var dof = nc - 1 if nc > 1 else 1
    var ctx = DeviceContext()
    var dx = upload_f32(ctx, x)
    var dnd = ctx.enqueue_create_buffer[DType.float32](n * nn)
    var dni = ctx.enqueue_create_buffer[DType.int32](n * nn)
    var dp = ctx.enqueue_create_buffer[DType.float32](n * nn)
    ctx.enqueue_function[knn_kernel](Int32(n), dx.unsafe_ptr(), Int32(d), Int32(nn), dnd.unsafe_ptr(),
                                     dni.unsafe_ptr(), grid_dim=_grid(n), block_dim=TPB)
    ctx.enqueue_function[perplexity_kernel](Int32(n), dnd.unsafe_ptr(), Int32(nn), identical_log(perplexity),
                                            dp.unsafe_ptr(), grid_dim=_grid(n), block_dim=TPB)
    ctx.synchronize()
    var nn_i = download_i32(ctx, dni, n * nn)
    var p_cond = download_f32(ctx, dp, n * nn)
    var indptr = List[Int32]()
    var indices = List[Int32]()
    var values = List[Float32]()
    tsne_symmetrize(n, nn, nn_i, p_cond, indptr, indices, values)

    var dptr = upload_i32(ctx, indptr)
    var dind = upload_i32(ctx, indices)
    var dval = upload_f32(ctx, values)
    var dy = upload_f32(ctx, y0)
    var dy2 = upload_f32(ctx, y0)
    var dupd = upload_f32(ctx, List[Float32](length=nc * n, fill=Float32(0.0)))
    var dgain = upload_f32(ctx, List[Float32](length=nc * n, fill=Float32(1.0)))
    var dg = ctx.enqueue_create_buffer[DType.float32](nc * n)
    var drz = ctx.enqueue_create_buffer[DType.float32](n)
    var drep = ctx.enqueue_create_buffer[DType.float32](nc * n)
    var dz = ctx.enqueue_create_buffer[DType.float32](1)
    var dkl = ctx.enqueue_create_buffer[DType.float32](n)
    var it = 0
    var last = -1
    var in_a = True
    var kl_last = Float32(0.0)
    for phase in range(2):
        var end = exploration if phase == 0 else max_iter
        if it >= end:
            continue
        if phase == 1:
            dupd = upload_f32(ctx, List[Float32](length=nc * n, fill=Float32(0.0)))
            dgain = upload_f32(ctx, List[Float32](length=nc * n, fill=Float32(1.0)))
        var ex = exaggeration if phase == 0 else Float32(1.0)
        var mom = Float32(0.5) if phase == 0 else Float32(0.8)
        var patience = TS_EXPLORATION_PATIENCE if phase == 0 else n_iter_without_progress
        var best_err = Float32(3.4028235e38)
        var best_iter = it
        for i in range(it, end):
            var check = (i + 1) % TS_CHECK_EVERY == 0 or i == end - 1
            var err = Float32(0.0)
            var gnorm = Float32(0.0)
            if in_a:
                _ts_iter(ctx, dy, dy2, n, nc, dof, dptr, dind, dval, drz, drep, dz, dupd, dgain, dg, dkl, ex, mom,
                         learning_rate, check, err, gnorm)
            else:
                _ts_iter(ctx, dy2, dy, n, nc, dof, dptr, dind, dval, drz, drep, dz, dupd, dgain, dg, dkl, ex, mom,
                         learning_rate, check, err, gnorm)
            in_a = not in_a
            last = i
            if check:
                kl_last = err
            if (i + 1) % TS_CHECK_EVERY == 0:
                if err < best_err:
                    best_err = err
                    best_iter = i
                elif i - best_iter > patience:
                    break
                if gnorm <= min_grad_norm:
                    break
        it = last + 1
    ctx.synchronize()
    if in_a:
        y_out = download_f32(ctx, dy, nc * n)
    else:
        y_out = download_f32(ctx, dy2, nc * n)
    kl_out = kl_last
    n_iter_out = last
    _ = dkl^
    _ = dz^
    _ = drep^
    _ = drz^
    _ = dg^
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
