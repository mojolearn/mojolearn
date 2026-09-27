# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IVF-PQ on the device: one thread per cell of `x_ann/ivf_pq_core.mojo`.

The coarse quantizer is the same Lloyd cells over whole rows."""

from std.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext

from metrics.checks.device_io import upload_f32, upload_i32, download_f32, download_i32
from x_ann.ivf_pq_core import (
    F32P, I32P, IvfPqIndex, pq_assign_cell, pq_init_cell, pq_lists_from_labels,
    pq_len_of, pq_residual_cell, pq_search_cell, pq_update_cell, pq_validate,
)

comptime TPB = 128


def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def residual_kernel(count: Int32, x: F32P, centers: F32P, labels: I32P, dim: Int32, rot_dim: Int32, dst: F32P):
    var e = _tid()
    if e < Int(count):
        pq_residual_cell(e, x, centers, labels, Int(dim), Int(rot_dim), dst)


def init_kernel(count: Int32, r: F32P, n: Int32, rot_dim: Int32, pq_len: Int32, n_codes: Int32, seed: Int32, cb: F32P):
    var e = _tid()
    if e < Int(count):
        pq_init_cell(e, r, Int(n), Int(rot_dim), Int(pq_len), Int(n_codes), Int(seed), cb)


def assign_kernel(count: Int32, r: F32P, cb: F32P, pq_dim: Int32, rot_dim: Int32, pq_len: Int32, n_codes: Int32, codes: I32P):
    var e = _tid()
    if e < Int(count):
        pq_assign_cell(e, r, cb, Int(pq_dim), Int(rot_dim), Int(pq_len), Int(n_codes), codes)


def update_kernel(count: Int32, r: F32P, codes: I32P, n: Int32, pq_dim: Int32, rot_dim: Int32, pq_len: Int32, n_codes: Int32, cb: F32P):
    var e = _tid()
    if e < Int(count):
        pq_update_cell(e, r, codes, Int(n), Int(pq_dim), Int(rot_dim), Int(pq_len), Int(n_codes), cb)


def search_kernel(
    count: Int32, queries: F32P, dim: Int32, centers: F32P, n_lists: Int32, offsets: I32P,
    list_indices: I32P, codes: I32P, cb: F32P, pq_dim: Int32, pq_len: Int32, n_codes: Int32,
    k: Int32, n_probes: Int32, out_d: F32P, out_i: I32P, out_n: I32P,
):
    var e = _tid()
    if e < Int(count):
        pq_search_cell(
            e, queries, Int(dim), centers, Int(n_lists), offsets, list_indices, codes, cb,
            Int(pq_dim), Int(pq_len), Int(n_codes), Int(k), Int(n_probes), out_d, out_i, out_n,
        )


def _grid(count: Int) -> Int:
    return (count + TPB - 1) // TPB


def ivf_pq_build_device(
    x: List[Float32], n: Int, dim: Int, n_lists: Int, kmeans_n_iters: Int, seed: Int,
    pq_dim: Int, pq_bits: Int, pq_iters: Int,
) raises -> IvfPqIndex:
    pq_validate(n, dim, n_lists, pq_dim, pq_bits, pq_iters)
    var pq_len = pq_len_of(dim, pq_dim)
    var rot_dim = pq_len * pq_dim
    var n_codes = 1 << pq_bits
    var ctx = DeviceContext()
    var dx = upload_f32(ctx, x)
    var dc = ctx.enqueue_create_buffer[DType.float32](n_lists * dim)
    var dl = ctx.enqueue_create_buffer[DType.int32](n)
    # the coarse Lloyd: the PQ cells with one subspace of width dim
    ctx.enqueue_function[init_kernel](
        Int32(n_lists * dim), dx.unsafe_ptr(), Int32(n), Int32(dim), Int32(dim), Int32(n_lists),
        Int32(seed), dc.unsafe_ptr(), grid_dim=_grid(n_lists * dim), block_dim=TPB,
    )
    for _ in range(kmeans_n_iters):
        ctx.enqueue_function[assign_kernel](
            Int32(n), dx.unsafe_ptr(), dc.unsafe_ptr(), Int32(1), Int32(dim), Int32(dim), Int32(n_lists),
            dl.unsafe_ptr(), grid_dim=_grid(n), block_dim=TPB,
        )
        ctx.enqueue_function[update_kernel](
            Int32(n_lists * dim), dx.unsafe_ptr(), dl.unsafe_ptr(), Int32(n), Int32(1), Int32(dim),
            Int32(dim), Int32(n_lists), dc.unsafe_ptr(), grid_dim=_grid(n_lists * dim), block_dim=TPB,
        )
    ctx.enqueue_function[assign_kernel](
        Int32(n), dx.unsafe_ptr(), dc.unsafe_ptr(), Int32(1), Int32(dim), Int32(dim), Int32(n_lists),
        dl.unsafe_ptr(), grid_dim=_grid(n), block_dim=TPB,
    )
    ctx.synchronize()
    var centers = download_f32(ctx, dc, n_lists * dim)
    var labels = download_i32(ctx, dl, n)
    var offsets = List[Int32]()
    var list_indices = List[Int32]()
    pq_lists_from_labels(labels, n, n_lists, offsets, list_indices)

    var dr = ctx.enqueue_create_buffer[DType.float32](n * rot_dim)
    var cb_count = pq_dim * n_codes * pq_len
    var dcb = ctx.enqueue_create_buffer[DType.float32](cb_count)
    var dcodes = ctx.enqueue_create_buffer[DType.int32](n * pq_dim)
    ctx.enqueue_function[residual_kernel](
        Int32(n * rot_dim), dx.unsafe_ptr(), dc.unsafe_ptr(), dl.unsafe_ptr(), Int32(dim),
        Int32(rot_dim), dr.unsafe_ptr(), grid_dim=_grid(n * rot_dim), block_dim=TPB,
    )
    ctx.enqueue_function[init_kernel](
        Int32(cb_count), dr.unsafe_ptr(), Int32(n), Int32(rot_dim), Int32(pq_len), Int32(n_codes),
        Int32(seed), dcb.unsafe_ptr(), grid_dim=_grid(cb_count), block_dim=TPB,
    )
    for _ in range(pq_iters):
        ctx.enqueue_function[assign_kernel](
            Int32(n * pq_dim), dr.unsafe_ptr(), dcb.unsafe_ptr(), Int32(pq_dim), Int32(rot_dim),
            Int32(pq_len), Int32(n_codes), dcodes.unsafe_ptr(), grid_dim=_grid(n * pq_dim), block_dim=TPB,
        )
        ctx.enqueue_function[update_kernel](
            Int32(cb_count), dr.unsafe_ptr(), dcodes.unsafe_ptr(), Int32(n), Int32(pq_dim),
            Int32(rot_dim), Int32(pq_len), Int32(n_codes), dcb.unsafe_ptr(),
            grid_dim=_grid(cb_count), block_dim=TPB,
        )
    ctx.enqueue_function[assign_kernel](
        Int32(n * pq_dim), dr.unsafe_ptr(), dcb.unsafe_ptr(), Int32(pq_dim), Int32(rot_dim),
        Int32(pq_len), Int32(n_codes), dcodes.unsafe_ptr(), grid_dim=_grid(n * pq_dim), block_dim=TPB,
    )
    ctx.synchronize()
    var codebooks = download_f32(ctx, dcb, cb_count)
    var codes = download_i32(ctx, dcodes, n * pq_dim)
    _ = dcodes^
    _ = dcb^
    _ = dr^
    _ = dl^
    _ = dc^
    _ = dx^
    _ = ctx^
    return IvfPqIndex(n_lists, dim, n, pq_dim, pq_len, n_codes, centers^, offsets^, list_indices^, codebooks^, codes^)


def ivf_pq_search_device(
    centers: List[Float32], offsets: List[Int32], list_indices: List[Int32], codebooks: List[Float32],
    codes: List[Int32], n_lists: Int, dim: Int, pq_dim: Int, pq_bits: Int,
    queries: List[Float32], m: Int, k: Int, n_probes: Int,
    mut out_d: List[Float32], mut out_i: List[Int32], mut out_n: List[Int32],
) raises:
    var pq_len = pq_len_of(dim, pq_dim)
    var n_codes = 1 << pq_bits
    var ctx = DeviceContext()
    var dq = upload_f32(ctx, queries)
    var dc = upload_f32(ctx, centers)
    var doff = upload_i32(ctx, offsets)
    var dli = upload_i32(ctx, list_indices)
    var dcodes = upload_i32(ctx, codes)
    var dcb = upload_f32(ctx, codebooks)
    var dd = ctx.enqueue_create_buffer[DType.float32](m * k)
    var di = ctx.enqueue_create_buffer[DType.int32](m * k)
    var dn = ctx.enqueue_create_buffer[DType.int32](m)
    ctx.enqueue_function[search_kernel](
        Int32(m), dq.unsafe_ptr(), Int32(dim), dc.unsafe_ptr(), Int32(n_lists), doff.unsafe_ptr(),
        dli.unsafe_ptr(), dcodes.unsafe_ptr(), dcb.unsafe_ptr(), Int32(pq_dim), Int32(pq_len),
        Int32(n_codes), Int32(k), Int32(n_probes), dd.unsafe_ptr(), di.unsafe_ptr(), dn.unsafe_ptr(),
        grid_dim=_grid(m), block_dim=TPB,
    )
    ctx.synchronize()
    out_d = download_f32(ctx, dd, m * k)
    out_i = download_i32(ctx, di, m * k)
    out_n = download_i32(ctx, dn, m)
    _ = dn^
    _ = di^
    _ = dd^
    _ = dcb^
    _ = dcodes^
    _ = dli^
    _ = doff^
    _ = dc^
    _ = dq^
    _ = ctx^
