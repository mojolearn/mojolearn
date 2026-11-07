# SPDX-License-Identifier: Apache-2.0
"""Feature partitions for the QN objective's X-transpose times dZ.

Each output cell keeps the one-device chain: the tile order of
`glm/impl/qn/qn_tiled_multi.mojo` `qntm_tiled_into` (QNTM_ROWS row tiles,
then the pinned STATS_TPB fold of the tile partials), the order IDENTICAL's
one-device gradient takes at every C (`linear_bwd`, `enqueue_tiled`). A cell
reads only its own column of X, so packing a column range onto a device and
running the same call there stores the same words: a plan of any size gives
the bits of one device. The forward, the loss, the bias gradient, the
regularization and the optimizer stay on the root.

Lane ne2-i-j (2026-11-07, plan row N453): the shards used the STATS_TPB
256-chain kernels (`xty_kernel`, `xtdz_multi_kernel`), an order no
IDENTICAL one-device gradient takes any more, and `linear_bwd` skipped this
dispatcher whenever the C > 1 tile order applied while binary fits never
reached `linear_bwd` at all (`enqueue_tiled`), so `devices=` ran every GLM on
one device. Now both objectives call it and the shards run the tile order.
BITS: one-device fits are unchanged; multi-device GLM fits equal them.

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
from core.device_plan import DevicePlan, device_plan_current
from std.os import getenv
from std.sys.compile import is_defined
from max.gpu.host import DeviceContext, DeviceBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.multi_gpu import peer_clone, copy_columns_kernel
from glm.impl.qn.qn_tiled_rule import qntm_ws_floats
from glm.impl.qn.qn_tiled_multi import qntm_tiled_into


@fieldwise_init
struct GradientColumnShard(Movable):
    var ctx: DeviceContext
    var x: DeviceBuffer[DType.float32]
    var dz: DeviceBuffer[DType.float32]
    var output: DeviceBuffer[DType.float32]
    # the tile partials of this shard's cells (`qntm_ws_floats`)
    var ws: DeviceBuffer[DType.float32]
    var first: Int
    var width: Int

    def __deinit__(deinit self):
        _ = self.x^
        _ = self.dz^
        _ = self.output^
        _ = self.ws^
        try:
            self.ctx.synchronize()
        except:
            pass
        _ = self.ctx^


def glm_device_plan() raises -> DevicePlan:
    """The GLM family's device plan, read once at a family entry
    (docs/plans/MULTI_GPU_ENCAPSULATION_20261103.md). Its `setting` is what
    `device_plan_setting("MOJOLEARN_GLM_DEVICE_COUNT")` returned, so
    `gradient_columns` tests the same text as before."""
    return device_plan_current("MOJOLEARN_GLM_DEVICE_COUNT", "")


def gradient_columns(
    dplan: DevicePlan,
    ctx: DeviceContext, mut output: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32], mut dz: DeviceBuffer[DType.float32],
    rows: Int, columns: Int, classes: Int,
) raises -> Bool:
    """The GLM gradient dispatcher: False (the caller's single-device route
    runs) when the plan is single (`DevicePlan.single()`: setting "" or "1"),
    else `output[c + C * j]` = the un-scaled `X^T dZ` cell in the tile order,
    computed over column shards, one per device, and True. Synchronous."""
    if dplan.single():
        return False
    var value = dplan.setting
    var count = Int(value)
    if count < 1 or count > 64 or GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("parallel GLM requires IDENTICAL and 1..64 devices")
    var active = min(count, columns)
    if active < 1:
        raise Error("parallel GLM needs features")
    ctx.synchronize()
    var shards = List[GradientColumnShard]()
    for rank in range(active):  # driver-loop: over the plan's device shards
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
        var device = DeviceContext(device_id=dplan.device(rank))
        var local_x = peer_clone(ctx, device, packed)
        _ = packed^
        var local_dz = peer_clone(ctx, device, dz)
        var out = device.enqueue_create_buffer[DType.float32](width * classes)
        var ws = device.enqueue_create_buffer[DType.float32](max(1, qntm_ws_floats(rows, width * classes)))
        device.synchronize()
        shards.append(GradientColumnShard(device^, local_x^, local_dz^, out^, ws^, first, width))
    for rank in range(active):  # driver-loop: over the plan's device shards
        ref shard = shards[rank]
        # the one-device tile order over this shard's packed columns: cell
        # `c + C * (j - first)` here is cell `c + C * j` of the full call
        qntm_tiled_into(shard.ctx, shard.output, shard.x, shard.dz, shard.ws, rows, shard.width, classes)
    for rank in range(active):  # driver-loop: over the plan's device shards
        ref shard = shards[rank]
        var destination = output.create_sub_buffer[DType.float32](shard.first * classes, shard.width * classes)
        shard.output.enqueue_copy_to(destination)
        shard.ctx.synchronize()
    _ = shards^
    ctx.synchronize()
    return True
