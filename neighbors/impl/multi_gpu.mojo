# SPDX-License-Identifier: Apache-2.0
"""Native KNN query rows for graph estimators, retaining original norm bytes.

THE NEGATIVE CONTROL. `-D MOJOLEARN_NEIGHBORS_PARALLEL_SABOTAGE=1` makes every
owner above rank 0 read its query rows, and their norms, one row early. `rows`
and `qt` are still derived from the true `first`, every allocation keeps its
size, and the write-back keeps the true `shard.first`, so no length and no
validation moves; only which query each shard answers does. It is a
`comptime if`, so no production bit can move, and it is INERT AT ONE DEVICE:
the shift is guarded by `rank > 0`, and `knn_device_count` caps the count at
`rows`, so `first >= 1` whenever `rank >= 1`. That guard is what makes a moved
`par-graph-agglomerative`, `par-graph-spectral` or `par-graph-umap` cell
attributable to the define rather than to the second device. Owed a two-device
column (`MOJOLEARN_PAR_DEVICES=0,1`); no host binding restates this driver.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from max.algorithm import sync_parallelize
from std.os import getenv
from std.sys.compile import is_defined
from core.multi_gpu import peer_clone
from core.step_phase import STEP_PHASE_TIMERS
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from neighbors.impl.detail.knn_brute_force import (
    brute_force_knn_impl, tiled_distance_tile_cells,
    KNN_METHOD_AUTO, KNN_METHOD_TILED,
)


def knn_device_count(rows: Int, method: Int) raises -> Int:
    var count = Int(getenv("MOJOLEARN_NEIGHBORS_DEVICE_COUNT", "1"))
    if count < 1 or count > 64:
        raise Error("neighbors device count must be in 1..64")
    if count > 1:
        if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
            raise Error("parallel native neighbors require IDENTICAL mode")
        if method != KNN_METHOD_AUTO and method != KNN_METHOD_TILED:
            raise Error("parallel native neighbors require the pinned tiled arm")
        comptime if STEP_PHASE_TIMERS:
            raise Error("parallel neighbors cannot use process-global GEMM phase counters")
    return min(rows, count)


@fieldwise_init
struct KNNRowShard(Movable):
    var ctx: DeviceContext
    var queries: DeviceBuffer[DType.float32]
    var query_norm: DeviceBuffer[DType.float32]
    var index: DeviceBuffer[DType.float32]
    var index_norm: DeviceBuffer[DType.float32]
    var tile: DeviceBuffer[DType.float32]
    var values: DeviceBuffer[DType.float32]
    var indices: DeviceBuffer[DType.uint32]
    var out_dist: DeviceBuffer[DType.float32]
    var out_idx: DeviceBuffer[DType.uint32]
    var out_i32: DeviceBuffer[DType.int32]
    var first: Int
    var rows: Int
    var query_tile: Int

    def __deinit__(deinit self):
        _ = self.queries^
        _ = self.query_norm^
        _ = self.index^
        _ = self.index_norm^
        _ = self.tile^
        _ = self.values^
        _ = self.indices^
        _ = self.out_dist^
        _ = self.out_idx^
        _ = self.out_i32^
        try:
            self.ctx.synchronize()
        except:
            pass
        _ = self.ctx^


def parallel_knn_rows(
    ctx: DeviceContext,
    mut queries: DeviceBuffer[DType.float32], mut query_norm: DeviceBuffer[DType.float32],
    mut index: DeviceBuffer[DType.float32], mut index_norm: DeviceBuffer[DType.float32],
    mut out_dist: DeviceBuffer[DType.float32], mut out_idx: DeviceBuffer[DType.uint32],
    nq: Int, ni: Int, d: Int, k: Int, query_tile: Int, buf_len: Int,
    return_sqrt: Bool, method: Int, metric: Int, metric_arg: Float32, count: Int,
) raises -> Int:
    ctx.synchronize()
    var shards = List[KNNRowShard]()
    for rank in range(count):
        var first = nq * rank // count
        var rows = nq * (rank + 1) // count - first
        var qt = min(query_tile, rows)
        var source = first
        comptime if is_defined["MOJOLEARN_NEIGHBORS_PARALLEL_SABOTAGE"]():
            # Check-only arm: later owners read their query rows one row early.
            if rank > 0:
                source = first - 1
        var device = DeviceContext(device_id=rank)
        var qv = queries.create_sub_buffer[DType.float32](source*d, rows*d)
        var nv = query_norm.create_sub_buffer[DType.float32](source, rows)
        var q = peer_clone(ctx, device, qv)
        var qn = peer_clone(ctx, device, nv)
        var x = peer_clone(ctx, device, index)
        var xn = peer_clone(ctx, device, index_norm)
        var tile = device.enqueue_create_buffer[DType.float32](tiled_distance_tile_cells(qt, ni, d, k, metric))
        var values = device.enqueue_create_buffer[DType.float32](qt * 2 * buf_len)
        var indices = device.enqueue_create_buffer[DType.uint32](qt * 2 * buf_len)
        var od = device.enqueue_create_buffer[DType.float32](rows*k)
        var oi = device.enqueue_create_buffer[DType.uint32](rows*k)
        var oi32 = device.enqueue_create_buffer[DType.int32](rows*k)
        device.synchronize()
        shards.append(KNNRowShard(device^, q^, qn^, x^, xn^, tile^, values^, indices^,
                                  od^, oi^, oi32^, first, rows, qt))
    var failures = List[Int](length=count, fill=0)
    var sp = rebind[MutPointer[KNNRowShard, MutUntrackedOrigin]](shards.unsafe_ptr())
    var fp = rebind[MutPointer[Int, MutUntrackedOrigin]](failures.unsafe_ptr())
    def task(rank: Int) {imm sp, imm fp, imm ni, imm d, imm k, imm buf_len,
                         imm return_sqrt, imm method, imm metric, imm metric_arg}:
        try:
            ref s = sp[rank]
            brute_force_knn_impl(s.ctx, s.queries, s.query_norm, s.index, s.index_norm,
                s.tile, s.values, s.indices, s.out_dist, s.out_idx, s.out_i32,
                s.rows, ni, d, k, s.query_tile, buf_len, return_sqrt, False,
                True, True, method, metric, metric_arg)
            s.ctx.synchronize()
        except:
            fp[rank] = 1
    sync_parallelize(task, count)
    for rank in range(count):
        if failures[rank] != 0:
            raise Error("native neighbor query shard failed: " + String(rank))
    var largest_tile = 0
    for rank in range(count):
        ref s = shards[rank]
        var dv = out_dist.create_sub_buffer[DType.float32](s.first*k, s.rows*k)
        var iv = out_idx.create_sub_buffer[DType.uint32](s.first*k, s.rows*k)
        s.out_dist.enqueue_copy_to(dv)
        s.out_idx.enqueue_copy_to(iv)
        s.ctx.synchronize()
        largest_tile = max(largest_tile, s.query_tile)
    _ = shards^
    ctx.synchronize()
    return largest_tile
