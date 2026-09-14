# SPDX-License-Identifier: Apache-2.0
"""Distribute whole FP32-v1 dot leaves, then use the original balanced fold.

Only automatic plan selection is intercepted. Explicit plan probes continue
through their requested implementation. Root solver state remains resident.
"""
from std.os import getenv
from max.gpu.host import DeviceContext, DeviceBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.multi_gpu import peer_clone
from gemm.checks.gemm_oracle import contract_leaf_size
from gemm.checks.gemm_identical import (
    identical_gemm_leaf_kernel, identical_gemm_fold_kernel,
    SPLITK_LEAF_TPB, SPLITK_FOLD_TPB,
)


@fieldwise_init
struct DotShard(Movable):
    var ctx: DeviceContext
    var a: DeviceBuffer[DType.float32]
    var b: DeviceBuffer[DType.float32]
    var partials: DeviceBuffer[DType.float32]
    var first: Int
    var count: Int
    var rows: Int

    def __deinit__(deinit self):
        _ = self.a^
        _ = self.b^
        _ = self.partials^
        try:
            self.ctx.synchronize()
        except:
            pass
        _ = self.ctx^


def profile_dot_parallel(
    ctx: DeviceContext, mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32], mut b: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32], k: Int,
) raises -> Bool:
    var value = String(getenv("MOJOLEARN_SOLVER_DEVICE_COUNT"))
    if value == "" or value == "1":
        return False
    var count = Int(value)
    if count < 1 or count > 64 or GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("parallel solver requires IDENTICAL and 1..64 devices")
    if k <= 128:
        return False
    var leaf = contract_leaf_size(k)
    var leaves = (k + leaf - 1) // leaf
    var active = min(count, leaves)
    ctx.synchronize()
    var shards = List[DotShard]()
    for rank in range(active):
        var first = leaves * rank // active
        var width = leaves * (rank + 1) // active - first
        var start = first * leaf
        var rows = min(k - start, width * leaf)
        var source_a = a.create_sub_buffer[DType.float32](start, rows)
        var source_b = b.create_sub_buffer[DType.float32](start, rows)
        var device = DeviceContext(device_id=rank)
        var local_a = peer_clone(ctx, device, source_a)
        var local_b = peer_clone(ctx, device, source_b)
        var partials = device.enqueue_create_buffer[DType.float32](width)
        device.synchronize()
        shards.append(DotShard(device^, local_a^, local_b^, partials^, first, width, rows))
    for rank in range(active):
        ref shard = shards[rank]
        shard.ctx.enqueue_function[identical_gemm_leaf_kernel](
            shard.partials.unsafe_ptr(), shard.a.unsafe_ptr(), shard.b.unsafe_ptr(),
            Int32(1), Int32(1), Int32(shard.rows), Int32(leaf), Int32(shard.count),
            Int32(shard.rows), Int32(1), Int32(1), Int32(shard.rows), Int32(shard.count),
            grid_dim=((shard.count + SPLITK_LEAF_TPB - 1) // SPLITK_LEAF_TPB, 1, 1),
            block_dim=(SPLITK_LEAF_TPB, 1, 1))
    for rank in range(active):
        ref shard = shards[rank]
        var destination = ws.create_sub_buffer[DType.float32](shard.first, shard.count)
        shard.partials.enqueue_copy_to(destination)
        shard.ctx.synchronize()
    _ = shards^
    ctx.enqueue_function[identical_gemm_fold_kernel[False]](
        c.unsafe_ptr(), ws.unsafe_ptr(), Int32(1), Int32(leaves), Int32(leaves),
        grid_dim=(1, 1, 1), block_dim=(SPLITK_FOLD_TPB, 1, 1))
    return True
