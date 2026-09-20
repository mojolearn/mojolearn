# SPDX-License-Identifier: Apache-2.0
"""Feature partitions for the QN objective's X-transpose times dZ.

Each output cell retains the original STATS_TPB row partition and pinned sum.
The loss, bias gradient, regularization and optimizer stay on the root.

THE NEGATIVE CONTROL. `-D MOJOLEARN_GLM_PARALLEL_SABOTAGE=1` packs every owner
above rank 0 from one feature column early. The packed buffer keeps its
`cells` length and the write-back keeps the true `shard.first`, so no
allocation, length or validation changes; only which column of `X` each shard
contracts against `dZ` does. It is a `comptime if`, so no production bit can
move, and it is INERT AT ONE DEVICE: the shift is guarded by `rank > 0` and a
one-device column has only rank 0. That is what makes a moved `par-logistic`
cell attributable to the define rather than to the second device. Owed a
two-device column (`MOJOLEARN_PAR_DEVICES=0,1`); no host binding restates this
driver, so no CPU column can watch it fire.
"""
from std.os import getenv
from std.sys.compile import is_defined
from max.gpu.host import DeviceContext, DeviceBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.column_stats import STATS_TPB, xty_kernel
from core.multi_gpu import peer_clone, copy_columns_kernel
from glm.impl.qn.glm_softmax import xtdz_multi_kernel


@fieldwise_init
struct GradientColumnShard(Movable):
    var ctx: DeviceContext
    var x: DeviceBuffer[DType.float32]
    var dz: DeviceBuffer[DType.float32]
    var output: DeviceBuffer[DType.float32]
    var first: Int
    var width: Int

    def __deinit__(deinit self):
        _ = self.x^
        _ = self.dz^
        _ = self.output^
        try:
            self.ctx.synchronize()
        except:
            pass
        _ = self.ctx^


def gradient_columns(
    ctx: DeviceContext, mut output: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32], mut dz: DeviceBuffer[DType.float32],
    rows: Int, columns: Int, classes: Int,
) raises -> Bool:
    var value = String(getenv("MOJOLEARN_GLM_DEVICE_COUNT"))
    if value == "" or value == "1":
        return False
    var count = Int(value)
    if count < 1 or count > 64 or GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("parallel GLM requires IDENTICAL and 1..64 devices")
    var active = min(count, columns)
    if active < 1:
        raise Error("parallel GLM needs features")
    ctx.synchronize()
    var shards = List[GradientColumnShard]()
    for rank in range(active):
        var first = columns * rank // active
        var width = columns * (rank + 1) // active - first
        var cells = rows * width
        var source = first
        comptime if is_defined["MOJOLEARN_GLM_PARALLEL_SABOTAGE"]():
            # Check-only arm: later owners pack from one feature column early.
            if rank > 0:
                source = first - 1
        var packed = ctx.enqueue_create_buffer[DType.float32](cells)
        ctx.enqueue_function[copy_columns_kernel[False]](x.unsafe_ptr(), packed.unsafe_ptr(),
            Int32(columns), Int32(source), Int32(width), Int32(cells),
            grid_dim=((cells + 255) // 256, 1, 1), block_dim=(256, 1, 1))
        ctx.synchronize()
        var device = DeviceContext(device_id=rank)
        var local_x = peer_clone(ctx, device, packed)
        _ = packed^
        var local_dz = peer_clone(ctx, device, dz)
        var out = device.enqueue_create_buffer[DType.float32](width * classes)
        device.synchronize()
        shards.append(GradientColumnShard(device^, local_x^, local_dz^, out^, first, width))
    for rank in range(active):
        ref shard = shards[rank]
        if classes == 1:
            shard.ctx.enqueue_function[xty_kernel](shard.output.unsafe_ptr(),
                shard.x.unsafe_ptr(), shard.dz.unsafe_ptr(), Int32(rows), Int32(shard.width),
                grid_dim=(shard.width, 1, 1), block_dim=(STATS_TPB, 1, 1))
        else:
            shard.ctx.enqueue_function[xtdz_multi_kernel](shard.output.unsafe_ptr(),
                shard.x.unsafe_ptr(), shard.dz.unsafe_ptr(), Int32(rows), Int32(shard.width), Int32(classes),
                grid_dim=(classes * shard.width, 1, 1), block_dim=(STATS_TPB, 1, 1))
    for rank in range(active):
        ref shard = shards[rank]
        var destination = output.create_sub_buffer[DType.float32](shard.first * classes, shard.width * classes)
        shard.output.enqueue_copy_to(destination)
        shard.ctx.synchronize()
    _ = shards^
    ctx.synchronize()
    return True
