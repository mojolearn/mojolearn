# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""UMAP's graph through the IVF k-NN graph (`UMAP(neighbors='ivf')`,
`mojolearn.neighbors_graph(method='ivf')`; lane sc-ann-1-umap, 2026-11-28).

UMAP's graph takes n_neighbors neighbors per row INCLUDING self (self first,
distance 0) under euclidean distances. The IVF graph
(x_ann/cagra_fast_knn.mojo `ivf_graph_enqueue`, planned by
x_ann/knn_graph_plan.mojo `knn_graph_plan`) gives the k = n_neighbors - 1
nearest OTHER rows under squared distances, ascending by (distance, index).
One launch (`umap_self_first_kernel`, one thread per row) writes self at slot
0 and the `identical_sqrt` roots of the k distances after it; the graph build
(`sparse_fuzzy_simplicial_graph_device`) then runs as for the exact search.
The CPU column is umap/host/umap_oracle.mojo `host_umap_ivf_self_knn`, the
same words (`ivf_graph_host`, the same root).

When the plan is the exact graph (inputs under 12,288 rows with the default
lists and probes) or a probe pool is short (`ivf_graph_enqueue` returns
False), this returns False and the caller runs UMAP's own exact search: then
`neighbors='ivf'` is `neighbors='exact'`, the same bits.
"""
from max.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import identical_sqrt
from x_ann.cagra_fast_knn import ivf_graph_enqueue
from x_ann.knn_graph_plan import knn_graph_plan, KG_METHOD_IVF
from umap.params import UMAPParams, umap_ivf_plan_seed

comptime _SF_TPB = 256
comptime _SF_F32P = MutPointer[Float32, MutAnyOrigin]
comptime _SF_I32P = MutPointer[Int32, MutAnyOrigin]
comptime _SF_U32P = MutPointer[UInt32, MutAnyOrigin]


def umap_self_first_kernel(src_d: _SF_F32P, src_i: _SF_I32P, dst_d: _SF_F32P, dst_i: _SF_U32P,
                           n_in: Int32, kk_in: Int32):
    """Row r of the n x (kk + 1) self-first graph: self at slot 0 (distance
    0), then the kk IVF neighbors' indices and the roots of their squared
    distances, in their (distance, index) order."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(n_in)
    var kk = Int(kk_in)
    if r < n:
        var k = kk + 1
        dst_i.unsafe_store(r * k, UInt32(r))
        dst_d.unsafe_store(r * k, Float32(0.0))
        for j in range(kk):  # one row's kk slots (this thread's row)
            dst_i.unsafe_store(r * k + 1 + j, UInt32(src_i.unsafe_load(r * kk + j)))
            dst_d.unsafe_store(r * k + 1 + j, identical_sqrt(src_d.unsafe_load(r * kk + j)))


def umap_ivf_self_knn_device(
    ctx: DeviceContext, mut head: DeviceBuffer[DType.float32], n: Int, d: Int, params: UMAPParams,
    mut d_dist: DeviceBuffer[DType.float32], mut d_idx: DeviceBuffer[DType.uint32],
) raises -> Bool:
    """The n x n_neighbors self-first euclidean graph of the rows in `head`
    into `d_dist` / `d_idx` through the IVF graph (module docstring). False
    (nothing written) when the plan is exact or a probe pool is short."""
    var kk = params.n_neighbors - 1
    var plan = knn_graph_plan(n, d, kk, KG_METHOD_IVF, params.ivf_lists, params.ivf_probes,
                              umap_ivf_plan_seed(params))
    if plan.method != KG_METHOD_IVF:
        return False
    var total = n * kk
    var dnd = ctx.enqueue_create_buffer[DType.float32](max(total, 1))
    var dni = ctx.enqueue_create_buffer[DType.int32](max(total, 1))
    if not ivf_graph_enqueue(ctx, head, n, d, kk, plan.n_lists, plan.n_probes, plan.seed_off, dnd, dni):
        _ = dnd^
        _ = dni^
        return False
    ctx.enqueue_function[umap_self_first_kernel](
        dnd.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dni.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        d_dist.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), d_idx.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        Int32(n), Int32(kk),
        grid_dim=(n + _SF_TPB - 1) // _SF_TPB, block_dim=_SF_TPB,
    )
    ctx.synchronize()  # the IVF buffers are read by the launch above
    _ = dnd^
    _ = dni^
    _ = head.unsafe_ptr()
    return True


def umap_knn_strip_kernel(knn_d: _SF_F32P, knn_i: _SF_U32P, indptr: MutPointer[Int64, MutAnyOrigin],
                          dst_i: _SF_I32P, dst_d: _SF_F32P, n_in: Int32, k_in: Int32):
    """`mojolearn.neighbors_graph`'s distances CSR from the self-first
    n x k graph: row r keeps slots 1..k-1 (self, slot 0, is dropped, as
    Scanpy's `distances` holds no self edge) in their ascending order, and
    indptr[r] = r (k - 1) (thread n writes indptr[n])."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(n_in)
    var k = Int(k_in)
    var kk = k - 1
    if r <= n:
        indptr.unsafe_store(r, Int64(r * kk))
    if r < n:
        for j in range(kk):  # one row's k - 1 kept slots (this thread's row)
            dst_i.unsafe_store(r * kk + j, Int32(knn_i.unsafe_load(r * k + 1 + j)))
            dst_d.unsafe_store(r * kk + j, knn_d.unsafe_load(r * k + 1 + j))
