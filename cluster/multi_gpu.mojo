# SPDX-License-Identifier: Apache-2.0
"""Cooperative KMeans assignment. Split whole row tiles, never feature sums.

The original policy and centroid order run on every shard. Full-data centroid
updates, inertia, initialization and convergence remain in the existing driver.
This correctness path stages buffers per call; it makes no throughput claim.

THE NEGATIVE CONTROL. `-D MOJOLEARN_KMEANS_PARALLEL_SABOTAGE=1` makes every
owner above rank 0 read its row tile one row early. It is a `comptime if`, so
no production bit can move, and it is INERT AT ONE DEVICE by construction: the
shift is guarded by `rank > 0` and a one-device column has only rank 0. That
guard is what makes a moved `par-kmeans` cell attributable to the define
rather than to the second device. Owed a two-device column
(`MOJOLEARN_PAR_DEVICES=0,1`); no CPU route restates this driver, so no Mac and
no CPU-only box can watch it fire.
"""
from max.gpu.host import DeviceContext, DeviceBuffer
from std.os import getenv
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from cluster.impl.distance.fused_distance_nn.simt_kernel import fused_distance_nn_kernel
from neighbors.impl.distance.detail.pairwise_distance_base import launch_config_generator


@fieldwise_init
struct AssignmentShard(Movable):
    var ctx: DeviceContext
    var x: DeviceBuffer[DType.float32]
    var xn: DeviceBuffer[DType.float32]
    var c: DeviceBuffer[DType.float32]
    var cn: DeviceBuffer[DType.float32]
    var key: DeviceBuffer[DType.uint32]
    var value: DeviceBuffer[DType.float32]
    var begin: Int
    var rows: Int

    def __deinit__(deinit self):
        _ = self.x^
        _ = self.xn^
        _ = self.c^
        _ = self.cn^
        _ = self.key^
        _ = self.value^
        try:
            self.ctx.synchronize()
        except:
            pass
        _ = self.ctx^


def assignment_device_count() raises -> Int:
    var value = String(getenv("MOJOLEARN_KMEANS_DEVICE_COUNT"))
    if value == "":
        return 1
    var count = Int(value)
    if count < 1 or count > 64:
        raise Error("KMeans device count must be in [1, 64]")
    if count > 1 and GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("multi-GPU KMeans requires IDENTICAL numeric mode")
    return count


def assignment_parallel[veclen: Int, kblk: Int, tr: Int, tc: Int](
    ctx: DeviceContext, mut out_key: DeviceBuffer[DType.uint32],
    mut out_value: DeviceBuffer[DType.float32], mut x: DeviceBuffer[DType.float32],
    mut centroids: DeviceBuffer[DType.float32], mut x_norm: DeviceBuffer[DType.float32],
    mut centroid_norm: DeviceBuffer[DType.float32], n_samples: Int,
    n_clusters: Int, n_features: Int, is_sqrt: Int32, count: Int,
) raises:
    comptime mblk = 4 * tr
    comptime nblk = 4 * tc
    comptime threads = tr * tc
    comptime smem = (mblk + nblk) * (kblk + veclen) * 4 + (mblk + nblk) * 4
    var tiles = (n_samples + mblk - 1) // mblk
    var active = min(count, tiles)
    var shards = List[AssignmentShard]()
    ctx.synchronize()
    for rank in range(active):
        var begin = (tiles * rank // active) * mblk
        var end = min(n_samples, (tiles * (rank + 1) // active) * mblk)
        var rows = end - begin
        var source = begin
        comptime if is_defined["MOJOLEARN_KMEANS_PARALLEL_SABOTAGE"]():
            # Check-only arm: later owners read their row tile one row early.
            # INERT AT ONE DEVICE. `active` is 1 there, the loop runs rank 0
            # only and `source` stays `begin`, so a cell that moves under this
            # build moved because of the define and not because of the second
            # device. The write-back below still uses `shard.begin`, so no
            # buffer length, allocation or validation changes; only the bytes
            # the assignment kernel reads do.
            if rank > 0:
                source = begin - 1
        var device = DeviceContext(device_id=rank)
        var sx = device.enqueue_create_buffer[DType.float32](rows * n_features)
        var sxn = device.enqueue_create_buffer[DType.float32](rows)
        var sc = device.enqueue_create_buffer[DType.float32](n_clusters * n_features)
        var scn = device.enqueue_create_buffer[DType.float32](n_clusters)
        var key = device.enqueue_create_buffer[DType.uint32](rows)
        var value = device.enqueue_create_buffer[DType.float32](rows)
        device.synchronize()
        var xv = x.create_sub_buffer[DType.float32](source * n_features, rows * n_features)
        var nv = x_norm.create_sub_buffer[DType.float32](source, rows)
        xv.enqueue_copy_to(sx)
        nv.enqueue_copy_to(sxn)
        centroids.enqueue_copy_to(sc)
        centroid_norm.enqueue_copy_to(scn)
        ctx.synchronize()
        shards.append(AssignmentShard(device^, sx^, sxn^, sc^, scn^, key^, value^, begin, rows))
    for rank in range(active):
        ref shard = shards[rank]
        var cfg = launch_config_generator(shard.rows, n_clusters, mblk, nblk, threads, smem)
        shard.ctx.enqueue_function[fused_distance_nn_kernel[veclen, kblk, tr, tc]](
            shard.key.unsafe_ptr(), shard.value.unsafe_ptr(), shard.x.unsafe_ptr(),
            shard.c.unsafe_ptr(), shard.xn.unsafe_ptr(), shard.cn.unsafe_ptr(),
            Int32(shard.rows), Int32(n_clusters), Int32(n_features), is_sqrt,
            grid_dim=(1, cfg[1], 1), block_dim=(threads, 1, 1))
    # Kernels are in flight together. Gather exact result bytes into original
    # row positions; no per-device centroid sums ever enter the arithmetic.
    for rank in range(active):
        ref shard = shards[rank]
        var keys = out_key.create_sub_buffer[DType.uint32](shard.begin, shard.rows)
        var values = out_value.create_sub_buffer[DType.float32](shard.begin, shard.rows)
        shard.key.enqueue_copy_to(keys)
        shard.value.enqueue_copy_to(values)
        shard.ctx.synchronize()
    _ = shards^
    ctx.synchronize()
