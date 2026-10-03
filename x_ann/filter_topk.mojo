# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CAGRA's filtered search compaction in Mojo (lane apple-fast-py2mojo-cluster,
2026-10-03). `CagraIndex._search_filtered` walked every query's itopk list in
Python and kept the first k entries whose row passes the filter (an id < 0 is
padding). Here one thread per query on the device, the same cell in a loop on
the host column. Output slots past the kept entries are +inf / -1."""
from std.gpu import block_dim, block_idx, thread_idx

from x_ann.device_ctx import x_ann_ctx
from x_ann.io import download_f32, download_i32, upload_f32, upload_i32
from x_ann.ivf_pq_core import F32P, I32P, pq_inf

comptime FILTER_TPB = 128


@always_inline
def filter_topk_cell(q: Int, bd: F32P, bi: I32P, keep: I32P, n: Int, L: Int, k: Int, od: F32P, oi: I32P):
    var o = 0
    for s in range(L):
        if o == k:
            break
        var v = Int(bi.unsafe_load(q * L + s))
        if v >= 0 and v < n and keep.unsafe_load(v) != 0:
            od.unsafe_store(q * k + o, bd.unsafe_load(q * L + s))
            oi.unsafe_store(q * k + o, Int32(v))
            o += 1
    while o < k:
        od.unsafe_store(q * k + o, pq_inf())
        oi.unsafe_store(q * k + o, Int32(-1))
        o += 1


def filter_topk_kernel(m: Int32, bd: F32P, bi: I32P, keep: I32P, n: Int32, L: Int32, k: Int32, od: F32P, oi: I32P):
    var q = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if q < Int(m):
        filter_topk_cell(q, bd, bi, keep, Int(n), Int(L), Int(k), od, oi)


def filter_topk_device(
    bd: List[Float32], bi: List[Int32], keep: List[Int32], m: Int, n: Int, L: Int, k: Int,
    mut out_d: List[Float32], mut out_i: List[Int32],
) raises:
    var ctx = x_ann_ctx()
    var dbd = upload_f32(ctx, bd)
    var dbi = upload_i32(ctx, bi)
    var dkeep = upload_i32(ctx, keep)
    var dd = ctx.enqueue_create_buffer[DType.float32](m * k)
    var di = ctx.enqueue_create_buffer[DType.int32](m * k)
    ctx.enqueue_function[filter_topk_kernel](
        Int32(m), dbd.unsafe_ptr(), dbi.unsafe_ptr(), dkeep.unsafe_ptr(), Int32(n), Int32(L), Int32(k),
        dd.unsafe_ptr(), di.unsafe_ptr(), grid_dim=(m + FILTER_TPB - 1) // FILTER_TPB, block_dim=FILTER_TPB,
    )
    ctx.synchronize()
    out_d = download_f32(ctx, dd, m * k)
    out_i = download_i32(ctx, di, m * k)
    _ = di^
    _ = dd^
    _ = dkeep^
    _ = dbi^
    _ = dbd^
    _ = ctx^


def filter_topk_host(bd: F32P, bi: I32P, keep: I32P, m: Int, n: Int, L: Int, k: Int, od: F32P, oi: I32P):
    for q in range(m):
        filter_topk_cell(q, bd, bi, keep, n, L, k, od, oi)
