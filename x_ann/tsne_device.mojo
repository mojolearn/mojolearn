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
from x_ann.knn_device import knn_graph_device

from metrics.checks.device_io import upload_f32, upload_i32, download_f32, download_i32
from checks.numerics import identical_log
from x_ann.tsne_core import (
    F32P, I32P, ts_kl_cell, ts_knn_cell, ts_perplexity_cell, ts_repulse_step, ts_repulse_visit, ts_step_cell,
    ts_sum_cell, tsne_nn, tsne_symmetrize, tsne_validate,
)

comptime TPB = 128


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


comptime REP_TILE = 1024


def repulse_kernel(n: Int32, y: F32P, row_z: F32P, rep: F32P):
    """`ts_repulse_cell` for row i with the points staged through shared
    memory in `ts_repulse_visit` order: each thread still folds its row over
    t ascending through `ts_repulse_step`, so the same instruction sequence
    on the same operands (the O(n^2) per-iteration cost, now reading each
    point once per block instead of once per row)."""
    var tid = Int(thread_idx.x)
    var i = _tid()
    var nr = Int(n)
    var tile = stack_allocation[2 * REP_TILE, Float32, address_space=AddressSpace.SHARED]()
    var tj = stack_allocation[REP_TILE, Int32, address_space=AddressSpace.SHARED]()
    var active = i < nr
    var yi0 = Float32(0.0)
    var yi1 = Float32(0.0)
    if active:
        yi0 = y.unsafe_load(2 * i)
        yi1 = y.unsafe_load(2 * i + 1)
    var z = Float32(0.0)
    var r0 = Float32(0.0)
    var r1 = Float32(0.0)
    var t0 = 0
    while t0 < nr:
        var cnt = REP_TILE if t0 + REP_TILE <= nr else nr - t0
        barrier()
        var e = tid
        while e < cnt:
            var j = ts_repulse_visit(t0 + e, nr)
            tj[e] = Int32(j)
            tile[2 * e] = y.unsafe_load(2 * j)
            tile[2 * e + 1] = y.unsafe_load(2 * j + 1)
            e += TPB
        barrier()
        if active:
            for t in range(cnt):
                if Int(tj[t]) == i:
                    continue
                ts_repulse_step(yi0, yi1, tile[2 * t], tile[2 * t + 1], z, r0, r1)
        t0 += cnt
    if active:
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
    ctx.enqueue_function[repulse_kernel](Int32(n), ycur.unsafe_ptr(), drz.unsafe_ptr(), drep.unsafe_ptr(),
                                         grid_dim=_grid(n), block_dim=TPB)
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
    ctx.enqueue_function[repulse_kernel](Int32(n), y.unsafe_ptr(), drz.unsafe_ptr(), drep.unsafe_ptr(),
                                         grid_dim=_grid(n), block_dim=TPB)
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
    var ctx = x_ann_ctx()
    var dx = upload_f32(ctx, x)
    var dnd = ctx.enqueue_create_buffer[DType.float32](n * nn)
    var dni = ctx.enqueue_create_buffer[DType.int32](n * nn)
    var dp = ctx.enqueue_create_buffer[DType.float32](n * nn)
    knn_graph_device(ctx, dx, n, d, nn, dnd, dni)
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
    var dupd = upload_f32(ctx, List[Float32](length=2 * n, fill=Float32(0.0)))
    var dgain = upload_f32(ctx, List[Float32](length=2 * n, fill=Float32(1.0)))
    var drz = ctx.enqueue_create_buffer[DType.float32](n)
    var drep = ctx.enqueue_create_buffer[DType.float32](2 * n)
    var dz = ctx.enqueue_create_buffer[DType.float32](1)
    var dkl = ctx.enqueue_create_buffer[DType.float32](n)
    for it in range(max_iter):
        var ex = exaggeration if it < exploration else Float32(1.0)
        var mom = Float32(0.5) if it < exploration else Float32(0.8)
        if it % 2 == 0:
            _ts_iter(ctx, dy, dy2, n, dptr, dind, dval, drz, drep, dz, dupd, dgain, ex, mom, learning_rate)
        else:
            _ts_iter(ctx, dy2, dy, n, dptr, dind, dval, drz, drep, dz, dupd, dgain, ex, mom, learning_rate)
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
