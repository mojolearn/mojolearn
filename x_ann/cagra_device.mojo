# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CAGRA on the device: the exact k-NN graph (tiled, `x_ann/knn_device.mojo`),
the prune (`cagra_prune_cell`, one node per thread) and the search, one
thread per cell of `x_ann/cagra_core.mojo` / `x_ann/tsne_core.mojo`; the
reverse-edge merge is the shared host function."""

from std.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext
from x_ann.device_ctx import x_ann_ctx

from metrics.checks.device_io import upload_f32, upload_i32, download_f32, download_i32
from x_ann.tsne_core import ts_knn_cell
from x_ann.cagra_core import F32P, I32P, cagra_reverse_merge, cg_search_cell
from x_ann.knn_device import cagra_prune_device, knn_graph_device

comptime TPB = 64


def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def _grid(count: Int) -> Int:
    return (count + TPB - 1) // TPB


def cg_knn_kernel(n: Int32, x: F32P, d: Int32, nn: Int32, nn_d: F32P, nn_i: I32P):
    var i = _tid()
    if i < Int(n):
        ts_knn_cell(i, x, Int(n), Int(d), Int(nn), nn_d, nn_i)


def cg_search_kernel(
    m: Int32, queries: F32P, x: F32P, n: Int32, d: Int32, graph: I32P, deg: Int32, k: Int32,
    L: Int32, width: Int32, max_iter: Int32, n_seeds: Int32, bd: F32P, bi: I32P, bx: I32P,
    visited: I32P, words: Int32, out_d: F32P, out_i: I32P,
):
    var q = _tid()
    if q < Int(m):
        cg_search_cell(q, queries, x, Int(n), Int(d), graph, Int(deg), Int(k), Int(L), Int(width),
                       Int(max_iter), Int(n_seeds), bd, bi, bx, visited, Int(words), out_d, out_i)


def cagra_build_device(x: List[Float32], n: Int, d: Int, kdeg: Int, deg: Int) raises -> List[Int32]:
    var ctx = x_ann_ctx()
    var dx = upload_f32(ctx, x)
    var dnd = ctx.enqueue_create_buffer[DType.float32](n * kdeg)
    var dni = ctx.enqueue_create_buffer[DType.int32](n * kdeg)
    knn_graph_device(ctx, dx, n, d, kdeg, dnd, dni)
    var pruned = cagra_prune_device(ctx, dni, n, kdeg, deg)
    _ = dni^
    _ = dnd^
    _ = dx^
    _ = ctx^
    return cagra_reverse_merge(n, deg, pruned)


def cagra_search_device(
    x: List[Float32], n: Int, d: Int, graph: List[Int32], deg: Int, queries: List[Float32], m: Int,
    k: Int, L: Int, width: Int, max_iter: Int, n_seeds: Int,
    mut out_d: List[Float32], mut out_i: List[Int32],
) raises:
    var words = (n + 31) // 32
    var ctx = x_ann_ctx()
    var dx = upload_f32(ctx, x)
    var dg = upload_i32(ctx, graph)
    var dq = upload_f32(ctx, queries)
    var bd = ctx.enqueue_create_buffer[DType.float32](m * L)
    var bi = ctx.enqueue_create_buffer[DType.int32](m * L)
    var bx = ctx.enqueue_create_buffer[DType.int32](m * L)
    var vis = ctx.enqueue_create_buffer[DType.int32](m * words)
    var od = ctx.enqueue_create_buffer[DType.float32](m * k)
    var oi = ctx.enqueue_create_buffer[DType.int32](m * k)
    ctx.enqueue_function[cg_search_kernel](
        Int32(m), dq.unsafe_ptr(), dx.unsafe_ptr(), Int32(n), Int32(d), dg.unsafe_ptr(), Int32(deg), Int32(k),
        Int32(L), Int32(width), Int32(max_iter), Int32(n_seeds), bd.unsafe_ptr(), bi.unsafe_ptr(),
        bx.unsafe_ptr(), vis.unsafe_ptr(), Int32(words), od.unsafe_ptr(), oi.unsafe_ptr(),
        grid_dim=_grid(m), block_dim=TPB,
    )
    ctx.synchronize()
    out_d = download_f32(ctx, od, m * k)
    out_i = download_i32(ctx, oi, m * k)
    _ = oi^
    _ = od^
    _ = vis^
    _ = bx^
    _ = bi^
    _ = bd^
    _ = dq^
    _ = dg^
    _ = dx^
    _ = ctx^
