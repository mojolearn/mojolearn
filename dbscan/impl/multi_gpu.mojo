# SPDX-License-Identifier: Apache-2.0
"""Disjoint DBSCAN neighborhood rows; original root core/label order.

THE NEGATIVE CONTROL. `-D MOJOLEARN_DBSCAN_PARALLEL_SABOTAGE=1` makes every
owner above rank 0 read its query rows one row early, in both partitions here:
the ball-cover rows of `_rbc_rows` and the dense vertex-degree rows of
`vertex_deg_dispatch`. Row counts, CSR offsets and every allocation are still
computed from the true `first`, so no length and no validation moves; only the
coordinates each shard measures against do. It is a `comptime if`, so no
production bit can move, and it is INERT AT ONE DEVICE: the shift is guarded by
`rank > 0`, and `_devices` caps the count at `rows` so `first >= 1` whenever
`rank >= 1`. That guard is what makes a moved `par-dbscan` cell attributable to
the define rather than to the second device. Owed a two-device column
(`MOJOLEARN_PAR_DEVICES=0,1`); no host binding restates this driver.
"""
from max.gpu.host import DeviceContext, DeviceBuffer
from max.algorithm import sync_parallelize
from std.os import getenv
from std.sys.compile import is_defined
from core.multi_gpu import peer_clone
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from neighbors.impl.ball_cover.ball_cover import (
    rbc_eps_nn_query_count as native_count,
    rbc_eps_nn_query_fill as native_fill,
    rbc_eps_nn_query_max_k as native_max_k,
)
from dbscan.impl.vertexdeg.algo import vertex_deg_dispatch as native_dense
from neighbors.impl.ball_cover.common import RBC_METRIC_DEFAULT


def _devices(rows: Int) raises -> Int:
    var count = Int(getenv("MOJOLEARN_DBSCAN_DEVICE_COUNT", "1"))
    if count < 1 or count > 64:
        raise Error("DBSCAN device count must be in 1..64")
    if count > 1 and GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("parallel DBSCAN requires IDENTICAL mode")
    return min(count, rows)


@fieldwise_init
struct RBCShard(Movable):
    var ctx: DeviceContext
    var x: DeviceBuffer[DType.float32]
    var q: DeviceBuffer[DType.float32]
    var r: DeviceBuffer[DType.float32]
    var ip: DeviceBuffer[DType.int32]
    var cols: DeviceBuffer[DType.int32]
    var dist: DeviceBuffer[DType.float32]
    var radius: DeviceBuffer[DType.float32]
    var ia: DeviceBuffer[DType.int32]
    var ja: DeviceBuffer[DType.int32]
    var vd: DeviceBuffer[DType.int32]
    var tmp: DeviceBuffer[DType.int32]
    var scratch: DeviceBuffer[DType.int32]
    var first: Int
    var rows: Int
    var edges: Int
    var longest: Int

    def __deinit__(deinit self):
        _ = self.x^
        _ = self.q^
        _ = self.r^
        _ = self.ip^
        _ = self.cols^
        _ = self.dist^
        _ = self.radius^
        _ = self.ia^
        _ = self.ja^
        _ = self.vd^
        _ = self.tmp^
        _ = self.scratch^
        try:
            self.ctx.synchronize()
        except:
            pass
        _ = self.ctx^


def _rbc_rows(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32], mut q: DeviceBuffer[DType.float32],
    mut r: DeviceBuffer[DType.float32], mut ip: DeviceBuffer[DType.int32],
    mut cols: DeviceBuffer[DType.int32], mut dist: DeviceBuffer[DType.float32],
    mut radius: DeviceBuffer[DType.float32], mut ia: DeviceBuffer[DType.int32],
    mut ja: DeviceBuffer[DType.int32], mut vd: DeviceBuffer[DType.int32],
    rows: Int, features: Int, landmarks: Int, eps: Float32,
    mode: Int, max_k: Int, metric: Int, metric_arg: Float32, count: Int,
) raises -> Int:
    # mode 0=count, 1=fill, 2=bounded one-pass. Integer offsets alone merge.
    ctx.synchronize()
    var root_ia = ctx.enqueue_create_host_buffer[DType.int32](rows + 1)
    if mode == 1:
        var iv = ia.create_sub_buffer[DType.int32](0, rows + 1)
        ctx.enqueue_copy(dst_ptr=root_ia.unsafe_ptr(), src_buf=iv)
        ctx.synchronize()
    var shards = List[RBCShard]()
    for rank in range(count):
        var first = rows * rank // count
        var nr = rows * (rank + 1) // count - first
        var source = first
        comptime if is_defined["MOJOLEARN_DBSCAN_PARALLEL_SABOTAGE"]():
            # Check-only arm: later owners read their query rows one row early.
            if rank > 0:
                source = first - 1
        var device = DeviceContext(device_id=rank)
        var qv = q.create_sub_buffer[DType.float32](source * features, nr * features)
        var local_x = peer_clone(ctx, device, x)
        var local_q = peer_clone(ctx, device, qv)
        var local_r = peer_clone(ctx, device, r)
        var local_ip = peer_clone(ctx, device, ip)
        var local_cols = peer_clone(ctx, device, cols)
        var local_dist = peer_clone(ctx, device, dist)
        var local_radius = peer_clone(ctx, device, radius)
        var edges = 0
        if mode == 1:
            edges = Int(root_ia.unsafe_ptr()[first + nr]) - Int(root_ia.unsafe_ptr()[first])
        var local_ia = device.enqueue_create_buffer[DType.int32](nr + 1)
        var local_ja = device.enqueue_create_buffer[DType.int32](max(1, nr * max_k if mode == 2 else edges))
        var local_vd = device.enqueue_create_buffer[DType.int32](nr + 1)
        var local_tmp = device.enqueue_create_buffer[DType.int32](max(1, nr * max_k if mode == 2 else 1))
        var local_scratch = device.enqueue_create_buffer[DType.int32](1)
        device.synchronize()
        if mode == 1:
            var local_ptr = device.enqueue_create_host_buffer[DType.int32](nr + 1)
            for i in range(nr + 1):
                local_ptr.unsafe_ptr()[i] = root_ia.unsafe_ptr()[first + i] - root_ia.unsafe_ptr()[first]
            device.enqueue_copy(dst_buf=local_ia, src_ptr=local_ptr.unsafe_ptr())
            device.synchronize()
        shards.append(RBCShard(device^, local_x^, local_q^, local_r^, local_ip^, local_cols^, local_dist^, local_radius^, local_ia^, local_ja^, local_vd^, local_tmp^, local_scratch^, first, nr, edges, 0))
    var failures = List[Int](length=count, fill=0)
    var sp = rebind[MutPointer[RBCShard, MutUntrackedOrigin]](shards.unsafe_ptr())
    var fp = rebind[MutPointer[Int, MutUntrackedOrigin]](failures.unsafe_ptr())
    def task(rank: Int) {imm sp, imm fp, imm features, imm landmarks, imm eps,
                         imm mode, imm max_k, imm metric, imm metric_arg}:
        try:
            ref s = sp[rank]
            if mode == 0:
                s.edges = native_count(s.ctx, s.x, s.q, s.r, s.ip, s.cols, s.dist,
                    s.radius, s.ia, s.vd, s.rows, features, landmarks, eps, metric, metric_arg)
            elif mode == 1:
                native_fill(s.ctx, s.x, s.q, s.r, s.ip, s.cols, s.dist,
                    s.radius, s.ia, s.ja, s.rows, features, landmarks, eps, metric, metric_arg)
            else:
                s.longest = native_max_k(s.ctx, s.x, s.q, s.r, s.ip, s.cols, s.dist,
                    s.radius, s.ia, s.ja, s.vd, s.tmp, s.scratch, s.rows, features,
                    landmarks, eps, max_k, metric, metric_arg)
                var tail = s.ia.create_sub_buffer[DType.int32](s.rows, 1)
                var host = s.ctx.enqueue_create_host_buffer[DType.int32](1)
                s.ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=tail)
                s.ctx.synchronize()
                s.edges = Int(host.unsafe_ptr()[0])
            s.ctx.synchronize()
        except:
            fp[rank] = 1
    sync_parallelize(task, count)
    for rank in range(count):
        if failures[rank] != 0:
            raise Error("DBSCAN neighborhood shard failed: " + String(rank))
    var total = 0
    var longest = 0
    for rank in range(count):
        ref s = shards[rank]
        if s.edges < 0 or total + s.edges > 2147483647:
            raise Error("DBSCAN shard edge count exceeds int32 CSR capacity")
        longest = max(longest, s.longest)
        if mode != 1:
            var ptr = s.ctx.enqueue_create_host_buffer[DType.int32](s.rows + 1)
            s.ctx.enqueue_copy(dst_ptr=ptr.unsafe_ptr(), src_buf=s.ia)
            s.ctx.synchronize()
            for i in range(s.rows + 1):
                root_ia.unsafe_ptr()[s.first + i] = ptr.unsafe_ptr()[i] + Int32(total)
            var source = s.vd.create_sub_buffer[DType.int32](0, s.rows)
            var target = vd.create_sub_buffer[DType.int32](s.first, s.rows)
            source.enqueue_copy_to(target)
            s.ctx.synchronize()
        if mode != 0 and s.edges > 0:
            var source = s.ja.create_sub_buffer[DType.int32](0, s.edges)
            var target = ja.create_sub_buffer[DType.int32](total, s.edges)
            source.enqueue_copy_to(target)
            s.ctx.synchronize()
        total += s.edges
    if mode != 1:
        var iv = ia.create_sub_buffer[DType.int32](0, rows + 1)
        ctx.enqueue_copy(dst_buf=iv, src_ptr=root_ia.unsafe_ptr())
        var tail = vd.create_sub_buffer[DType.int32](rows, 1)
        ctx.enqueue_copy(dst_buf=tail, src_ptr=root_ia.unsafe_ptr() + rows)
        ctx.synchronize()
    _ = shards^
    ctx.synchronize()
    return longest if mode == 2 else total


def rbc_eps_nn_query_count(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32], mut q: DeviceBuffer[DType.float32],
    mut r: DeviceBuffer[DType.float32], mut ip: DeviceBuffer[DType.int32],
    mut cols: DeviceBuffer[DType.int32], mut dist: DeviceBuffer[DType.float32],
    mut radius: DeviceBuffer[DType.float32], mut ia: DeviceBuffer[DType.int32],
    mut vd: DeviceBuffer[DType.int32],
    rows: Int, features: Int, landmarks: Int, eps: Float32,
    metric: Int = RBC_METRIC_DEFAULT, metric_arg: Float32 = Float32(2.0),
) raises -> Int:
    var count = _devices(rows)
    if count <= 1:
        return native_count(ctx, x, q, r, ip, cols, dist, radius, ia, vd, rows, features, landmarks, eps, metric, metric_arg)
    var ja = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.synchronize()
    return _rbc_rows(ctx, x, q, r, ip, cols, dist, radius, ia, ja, vd,
        rows, features, landmarks, eps, 0, 0, metric, metric_arg, count)


def rbc_eps_nn_query_fill(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32], mut q: DeviceBuffer[DType.float32],
    mut r: DeviceBuffer[DType.float32], mut ip: DeviceBuffer[DType.int32],
    mut cols: DeviceBuffer[DType.int32], mut dist: DeviceBuffer[DType.float32],
    mut radius: DeviceBuffer[DType.float32], mut ia: DeviceBuffer[DType.int32],
    mut ja: DeviceBuffer[DType.int32],
    rows: Int, features: Int, landmarks: Int, eps: Float32,
    metric: Int = RBC_METRIC_DEFAULT, metric_arg: Float32 = Float32(2.0),
) raises:
    var count = _devices(rows)
    if count <= 1:
        native_fill(ctx, x, q, r, ip, cols, dist, radius, ia, ja, rows, features, landmarks, eps, metric, metric_arg)
        return
    var vd = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.synchronize()
    _ = _rbc_rows(ctx, x, q, r, ip, cols, dist, radius, ia, ja, vd,
        rows, features, landmarks, eps, 1, 0, metric, metric_arg, count)


def rbc_eps_nn_query_max_k(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32], mut q: DeviceBuffer[DType.float32],
    mut r: DeviceBuffer[DType.float32], mut ip: DeviceBuffer[DType.int32],
    mut cols: DeviceBuffer[DType.int32], mut dist: DeviceBuffer[DType.float32],
    mut radius: DeviceBuffer[DType.float32], mut ia: DeviceBuffer[DType.int32],
    mut ja: DeviceBuffer[DType.int32],
    mut vd: DeviceBuffer[DType.int32],
    mut tmp: DeviceBuffer[DType.int32], mut scratch: DeviceBuffer[DType.int32],
    rows: Int, features: Int, landmarks: Int, eps: Float32,
    max_k: Int,
    metric: Int = RBC_METRIC_DEFAULT, metric_arg: Float32 = Float32(2.0),
) raises -> Int:
    var count = _devices(rows)
    if count <= 1:
        return native_max_k(ctx, x, q, r, ip, cols, dist, radius, ia, ja, vd, tmp, scratch, rows, features, landmarks, eps, max_k, metric, metric_arg)
    return _rbc_rows(ctx, x, q, r, ip, cols, dist, radius, ia, ja, vd,
        rows, features, landmarks, eps, 2, max_k, metric, metric_arg, count)


@fieldwise_init
struct DenseShard(Movable):
    var ctx: DeviceContext
    var x: DeviceBuffer[DType.float32]
    var adj: DeviceBuffer[DType.uint8]
    var vd: DeviceBuffer[DType.int32]
    var first: Int
    var rows: Int

    def __deinit__(deinit self):
        _ = self.x^
        _ = self.adj^
        _ = self.vd^
        try:
            self.ctx.synchronize()
        except:
            pass
        _ = self.ctx^


def vertex_deg_dispatch(
    ctx: DeviceContext, mut adj: DeviceBuffer[DType.uint8],
    mut vd: DeviceBuffer[DType.int32], mut x: DeviceBuffer[DType.float32],
    start: Int, rows: Int, references: Int, features: Int, eps: Float64, metric: Int,
) raises:
    var count = _devices(rows)
    if count <= 1:
        native_dense(ctx, adj, vd, x, start, rows, references, features, eps, metric)
        return
    ctx.synchronize()
    var shards = List[DenseShard]()
    for rank in range(count):
        var first = rows * rank // count
        var nr = rows * (rank + 1) // count - first
        var source = first
        comptime if is_defined["MOJOLEARN_DBSCAN_PARALLEL_SABOTAGE"]():
            # Check-only arm: later owners read their query rows one row early.
            if rank > 0:
                source = first - 1
        var device = DeviceContext(device_id=rank)
        var data = peer_clone(ctx, device, x)
        var local_adj = device.enqueue_create_buffer[DType.uint8](nr * references)
        var local_vd = device.enqueue_create_buffer[DType.int32](nr + 1)
        device.synchronize()
        native_dense(device, local_adj, local_vd, data, start + source,
                     nr, references, features, eps, metric)
        shards.append(DenseShard(device^, data^, local_adj^, local_vd^, first, nr))
    var total = 0
    for rank in range(count):
        ref s = shards[rank]
        s.ctx.synchronize()
        var av = adj.create_sub_buffer[DType.uint8](s.first * references, s.rows * references)
        s.adj.enqueue_copy_to(av)
        var dv = vd.create_sub_buffer[DType.int32](s.first, s.rows)
        var source = s.vd.create_sub_buffer[DType.int32](0, s.rows)
        source.enqueue_copy_to(dv)
        var tail = s.vd.create_sub_buffer[DType.int32](s.rows, 1)
        var host = s.ctx.enqueue_create_host_buffer[DType.int32](1)
        s.ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=tail)
        s.ctx.synchronize()
        var edges = Int(host.unsafe_ptr()[0])
        if edges < 0 or total + edges > 2147483647:
            raise Error("DBSCAN neighborhood exceeds int32 edge count")
        total += edges
    var host = ctx.enqueue_create_host_buffer[DType.int32](1)
    host.unsafe_ptr()[0] = Int32(total)
    var tail = vd.create_sub_buffer[DType.int32](rows, 1)
    ctx.enqueue_copy(dst_buf=tail, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = shards^
