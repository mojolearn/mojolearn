# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CAGRA on the device: the exact k-NN graph and the search, one thread per
cell of `x_ann/cagra_core.mojo` / `x_ann/tsne_core.mojo`; pruning and the
reverse-edge merge are the shared host functions."""

from std.gpu import block_idx, block_dim, thread_idx
from std.memory import stack_allocation
from std.sys.compile import is_defined
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from x_ann.device_ctx import x_ann_ctx
from x_ann.stage_timer import AnnStages
from x_ann.knn_device import knn_enqueue

from metrics.checks.device_io import upload_f32, upload_i32, download_f32, download_i32
from x_ann.cagra_core import F32P, I32P, cagra_prune, cagra_reverse_merge, cg_search_cell

comptime TPB = 64


def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def _grid(count: Int) -> Int:
    return (count + TPB - 1) // TPB


def cg_search_kernel(
    m: Int32, queries: F32P, x: F32P, n: Int32, d: Int32, graph: I32P, deg: Int32, k: Int32,
    L: Int32, width: Int32, max_iter: Int32, n_seeds: Int32, bd: F32P, bi: I32P, bx: I32P,
    visited: I32P, words: Int32, out_d: F32P, out_i: I32P,
):
    var q = _tid()
    if q < Int(m):
        cg_search_cell(q, queries, x, Int(n), Int(d), graph, Int(deg), Int(k), Int(L), Int(width),
                       Int(max_iter), Int(n_seeds), bd, bi, bx, visited, Int(words), out_d, out_i)


#: the device prune's widest k-NN row (one thread per entry)
comptime PRUNE_KMAX = 64


def prune_kernel(n: Int32, kdeg: Int32, deg: Int32, knn: I32P, pruned: I32P, bad: I32P):
    """`cagra_prune` for node a = block_idx.x, one thread per k-NN rank
    (lane ann-apple2). For a row of kdeg DISTINCT ids in [0, n) the host's
    detour count of rank kab is the number of (kad < kab, kdb) with
    knn[knn[a, kad], kdb] == knn[a, kab] (plus kdeg when that id is a
    itself), and its selection takes the deg least (count, rank) in order,
    so rank kab lands at output slot #{k : (cnt[k], k) < (cnt[kab], kab)}.
    Integer counts: the same integers, the same graph. A row that is not
    distinct is flagged (bad[a] = 1) and the host prunes the graph itself."""
    var a = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var k = Int(kdeg)
    var nr = Int(n)
    var ids = stack_allocation[PRUNE_KMAX, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var cnt = stack_allocation[PRUNE_KMAX, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var flag = stack_allocation[1, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    if t == 0:
        flag[0] = 0
    if t < k:
        ids[t] = knn.unsafe_load(a * k + t)
    barrier()
    if t < k:
        var v = Int(ids[t])
        var dup = v < 0 or v >= nr
        for u in range(t):
            if Int(ids[u]) == v:
                dup = True
        if dup:
            flag[0] = 1
    barrier()
    var is_bad = flag[0] != 0
    if is_bad:
        if t == 0:
            bad.unsafe_store(a, Int32(1))
        return
    if t < k:
        var target = ids[t]
        var c = k if Int(target) == a else 0
        for kad in range(t):
            var d = Int(ids[kad])
            for kdb in range(k):
                if knn.unsafe_load(d * k + kdb) == target:
                    c += 1
        cnt[t] = Int32(c)
    barrier()
    if t < k:
        var mine = cnt[t]
        var rank = 0
        for u in range(k):
            var o = cnt[u]
            if o < mine or (o == mine and u < t):
                rank += 1
        if rank < Int(deg):
            pruned.unsafe_store(a * Int(deg) + rank, ids[t])
    if t == 0:
        bad.unsafe_store(a, Int32(0))


def cagra_build_device(x: List[Float32], n: Int, d: Int, kdeg: Int, deg: Int) raises -> List[Int32]:
    var st = AnnStages("cagra_build")
    var ctx = x_ann_ctx()
    var dx = upload_f32(ctx, x)
    st.mark(ctx, "upload")
    var dnd = ctx.enqueue_create_buffer[DType.float32](n * kdeg)
    var dni = ctx.enqueue_create_buffer[DType.int32](n * kdeg)
    knn_enqueue(ctx, dx, n, d, kdeg, dnd, dni)
    var pruned = List[Int32]()
    var on_device = kdeg <= PRUNE_KMAX and deg <= kdeg and not is_defined["MOJOLEARN_CAGRA_HOST_PRUNE"]()
    if on_device:
        var dout = ctx.enqueue_create_buffer[DType.int32](n * deg)
        var dbad = ctx.enqueue_create_buffer[DType.int32](n)
        ctx.enqueue_function[prune_kernel](Int32(n), Int32(kdeg), Int32(deg), dni.unsafe_ptr(), dout.unsafe_ptr(),
                                           dbad.unsafe_ptr(), grid_dim=n, block_dim=PRUNE_KMAX)
        ctx.synchronize()
        st.host("knn_prune")
        var bad = download_i32(ctx, dbad, n)
        for a in range(n):
            if bad[a] != 0:
                on_device = False
                break
        if on_device:
            pruned = download_i32(ctx, dout, n * deg)
        _ = dbad^
        _ = dout^
    if not on_device:
        ctx.synchronize()
        st.host("knn")
        var knn = download_i32(ctx, dni, n * kdeg)
        st.host("download")
        pruned = cagra_prune(n, kdeg, knn, deg)
        st.host("prune")
    _ = dni^
    _ = dnd^
    _ = dx^
    _ = ctx^
    var merged = cagra_reverse_merge(n, deg, pruned)
    st.host("reverse_merge")
    return merged^


def cagra_search_device(
    x: List[Float32], n: Int, d: Int, graph: List[Int32], deg: Int, queries: List[Float32], m: Int,
    k: Int, L: Int, width: Int, max_iter: Int, n_seeds: Int,
    mut out_d: List[Float32], mut out_i: List[Int32],
) raises:
    var words = (n + 31) // 32
    var st = AnnStages("cagra_search")
    var ctx = x_ann_ctx()
    var dx = upload_f32(ctx, x)
    var dg = upload_i32(ctx, graph)
    var dq = upload_f32(ctx, queries)
    st.mark(ctx, "upload")
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
    st.host("search")
    out_d = download_f32(ctx, od, m * k)
    out_i = download_i32(ctx, oi, m * k)
    st.host("download")
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
