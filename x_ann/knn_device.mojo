# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE EXACT k-NN GRAPH AND THE CAGRA PRUNE ON THE DEVICE, SAME BITS
(lane/algos-ann-speed, phase C, 2026-09-28).

`knn_tiled_kernel` is `ts_knn_cell` (t-SNE's affinities, CAGRA's
intermediate graph) with the candidate rows staged through shared memory: a
block of rows reads each tile of candidates once instead of every thread
reading every candidate from global memory. Each thread still owns one row
i, visits the candidates j in ascending order, forms each squared distance
through `ts_sq_step` over c ascending (the step `ts_sqdist` takes) and
offers it through `ts_knn_offer` (the insertion `ts_knn_cell` makes), so
row i's list is the same instruction sequence on the same operands.

`prune_kernel` runs `cagra_prune_cell` one node per thread; the host ran
the same function in a loop (O(n * kdeg^3) integer work, the CAGRA build's
largest host cost)."""

from std.gpu import block_idx, block_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext

from x_ann.tsne_core import F32P, I32P, ts_knn_cell, ts_knn_offer, ts_sq_step
from x_ann.cagra_core import cagra_prune_cell

comptime KNN_TPB = 128
comptime KNN_TILE = 4096


def knn_tiled_kernel(n: Int32, x: F32P, d: Int32, nn: Int32, nn_d: F32P, nn_i: I32P):
    var tid = Int(thread_idx.x)
    var i = Int(block_idx.x) * KNN_TPB + tid
    var nr = Int(n)
    var dd = Int(d)
    var k = Int(nn)
    var t_rows = KNN_TILE // dd
    var tile = stack_allocation[KNN_TILE, Float32, address_space=AddressSpace.SHARED]()
    var active = i < nr
    var base = i * k
    var filled = 0
    var j0 = 0
    while j0 < nr:
        var rows = t_rows if j0 + t_rows <= nr else nr - j0
        barrier()
        var e = tid
        while e < rows * dd:
            tile[e] = x.unsafe_load(j0 * dd + e)
            e += KNN_TPB
        barrier()
        if active:
            for jj in range(rows):
                var j = j0 + jj
                if j == i:
                    continue
                var acc = Float32(0.0)
                for c in range(dd):
                    acc = ts_sq_step(x.unsafe_load(i * dd + c), tile[jj * dd + c], acc)
                filled = ts_knn_offer(base, filled, k, acc, j, nn_d, nn_i)
        j0 += rows


def knn_cell_kernel(n: Int32, x: F32P, d: Int32, nn: Int32, nn_d: F32P, nn_i: I32P):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        ts_knn_cell(i, x, Int(n), Int(d), Int(nn), nn_d, nn_i)


def knn_graph_device(
    ctx: DeviceContext, dx: DeviceBuffer[DType.float32], n: Int, d: Int, nn: Int,
    mut dnd: DeviceBuffer[DType.float32], mut dni: DeviceBuffer[DType.int32],
) raises:
    """Every row's nn nearest other rows into dnd/dni (enqueued, not synced)."""
    var grid = (n + KNN_TPB - 1) // KNN_TPB
    if d <= KNN_TILE:
        ctx.enqueue_function[knn_tiled_kernel](Int32(n), dx.unsafe_ptr(), Int32(d), Int32(nn), dnd.unsafe_ptr(),
                                               dni.unsafe_ptr(), grid_dim=grid, block_dim=KNN_TPB)
    else:
        ctx.enqueue_function[knn_cell_kernel](Int32(n), dx.unsafe_ptr(), Int32(d), Int32(nn), dnd.unsafe_ptr(),
                                              dni.unsafe_ptr(), grid_dim=grid, block_dim=KNN_TPB)


def prune_kernel(count: Int32, a0: Int32, kdeg: Int32, knn: I32P, deg: Int32, out: I32P, cnt: I32P, bad: I32P):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(count):
        if not cagra_prune_cell(Int(a0) + t, Int(kdeg), knn, Int(deg), out, cnt, t * Int(kdeg)):
            bad.unsafe_store(0, Int32(1))


def cagra_prune_device(
    ctx: DeviceContext, dknn: DeviceBuffer[DType.int32], n: Int, kdeg: Int, deg: Int,
) raises -> List[Int32]:
    """`cagra_prune` with one node per thread; raises the same error."""
    var chunk = (1 << 24) // kdeg
    if chunk > n:
        chunk = n
    if chunk < 1:
        chunk = 1
    var dout = ctx.enqueue_create_buffer[DType.int32](n * deg)
    var dcnt = ctx.enqueue_create_buffer[DType.int32](chunk * kdeg)
    var dbad = ctx.enqueue_create_buffer[DType.int32](1)
    dbad.enqueue_fill(Int32(0))
    var a0 = 0
    while a0 < n:
        var c = chunk if a0 + chunk <= n else n - a0
        ctx.enqueue_function[prune_kernel](Int32(c), Int32(a0), Int32(kdeg), dknn.unsafe_ptr(), Int32(deg),
                                           dout.unsafe_ptr(), dcnt.unsafe_ptr(), dbad.unsafe_ptr(),
                                           grid_dim=(c + KNN_TPB - 1) // KNN_TPB, block_dim=KNN_TPB)
        a0 += c
    var out = List[Int32](length=n * deg, fill=Int32(0))
    var bad = List[Int32](length=1, fill=Int32(0))
    ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=dout)
    ctx.enqueue_copy(dst_ptr=bad.unsafe_ptr(), src_buf=dbad)
    ctx.synchronize()
    _ = dcnt^
    _ = dout^
    _ = dbad^
    if bad[0] != 0:
        raise Error("CAGRA: the k-NN graph has too few distinct neighbors for graph_degree")
    return out^
