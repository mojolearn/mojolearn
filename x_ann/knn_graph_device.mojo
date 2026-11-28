# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device column of the public all-rows k-NN graph (`mojolearn.knn_graph`,
lane sc-ann-1, 2026-11-28): the rows go up once, the planned graph is built
on the device (`ivf_graph_enqueue`, or `knn_enqueue` for the exact graph and
for a refused IVF plan), the optional euclidean roots are one element-wise
pass, and the two n x k arrays come down into the caller's arrays. The CPU
column is x_ann/host/knn_graph_host.mojo, the same words under IDENTICAL."""

from max.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import identical_sqrt
from x_ann.device_ctx import x_ann_ctx
from x_ann.io import upload_f32_ptr, download_f32_into, download_i32_into
from x_ann.knn_device import knn_enqueue
from x_ann.cagra_fast_knn import ivf_graph_enqueue
from x_ann.tsne_core import F32P
from x_ann.knn_graph_plan import KnnGraphPlan, KG_METHOD_EXACT, KG_METHOD_IVF

comptime KG_TPB = 256


def kg_root_kernel(dist: F32P, total: Int32):
    """dist[e] = identical_sqrt(dist[e]): the euclidean root of a kept
    squared distance (the CPU column's root pass)."""
    var e = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if e < Int(total):
        dist.unsafe_store(e, identical_sqrt(dist.unsafe_load(e)))


def knn_graph_device_ptr(
    xp: MutPointer[Float32, MutUntrackedOrigin], n: Int, d: Int, k: Int, plan: KnnGraphPlan,
    dist_p: MutPointer[Float32, MutUntrackedOrigin], idx_p: MutPointer[Int32, MutUntrackedOrigin], root: Bool,
) raises -> Int:
    """The planned graph of the caller's n x d rows into the caller's n x k
    arrays (kept alive by the caller through the call); returns the method
    run (KG_METHOD_IVF, or KG_METHOD_EXACT when asked, planned or refused
    by a short probe pool)."""
    var ctx = x_ann_ctx()
    var dx = upload_f32_ptr(ctx, xp, n * d)
    var total = n * k
    var dnd = ctx.enqueue_create_buffer[DType.float32](total)
    var dni = ctx.enqueue_create_buffer[DType.int32](total)
    var used = KG_METHOD_EXACT
    if plan.method == KG_METHOD_IVF:
        if ivf_graph_enqueue(ctx, dx, n, d, k, plan.n_lists, plan.n_probes, plan.seed_off, dnd, dni):
            used = KG_METHOD_IVF
    if used == KG_METHOD_EXACT:
        knn_enqueue(ctx, dx, n, d, k, dnd, dni)
    if root:
        ctx.enqueue_function[kg_root_kernel](
            dnd.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(total),
            grid_dim=(total + KG_TPB - 1) // KG_TPB, block_dim=KG_TPB,
        )
    download_f32_into(ctx, dnd, total, dist_p)
    download_i32_into(ctx, dni, total, idx_p)
    _ = dx^  # the rows outlive every launch above (the downloads drain the queue)
    _ = dnd^
    _ = dni^
    return used
