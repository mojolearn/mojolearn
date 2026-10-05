# SPDX-License-Identifier: Apache-2.0
"""Row scratch is shared across leaf counts, never across concurrent tasks."""
from max.gpu.host import DeviceContext
from core.device_zero import enqueue_fill
from gbdt.gpu_util.arena import BufferArena
from gbdt.methods.leaves_estimation.pointwise_oracle import (
    make_oracle_device_scratch_in,
    make_oracle_device_scratch_sharing_rows_in,
)


def main() raises:
    var ctx = DeviceContext()
    for n in [17, 97, 113]:
        var arena = BufferArena()
        var first = make_oracle_device_scratch_in(ctx, arena, n, 2, 1, 1, 1, 32)
        var next_shape = make_oracle_device_scratch_sharing_rows_in(
            ctx, arena, first, 3, 1, 32
        )
        var other_task = make_oracle_device_scratch_in(ctx, arena, n, 2, 1, 1, 1, 32)
        enqueue_fill(ctx, first.d_bins, UInt32(7))
        enqueue_fill(ctx, other_task.d_bins, UInt32(11))
        enqueue_fill(ctx, next_shape.d_bins, UInt32(99))
        enqueue_fill(ctx, first.d_leaves, UInt32(5))
        enqueue_fill(ctx, next_shape.d_leaves, UInt32(9))
        var h_first = ctx.enqueue_create_host_buffer[DType.uint32](n)
        var h_other = ctx.enqueue_create_host_buffer[DType.uint32](n)
        var h_leaves = ctx.enqueue_create_host_buffer[DType.uint32](2)
        var h_next = ctx.enqueue_create_host_buffer[DType.uint32](3)
        ctx.enqueue_copy(dst_ptr=h_first.unsafe_ptr(), src_buf=first.d_bins)
        ctx.enqueue_copy(dst_ptr=h_other.unsafe_ptr(), src_buf=other_task.d_bins)
        ctx.enqueue_copy(dst_ptr=h_leaves.unsafe_ptr(), src_buf=first.d_leaves)
        ctx.enqueue_copy(dst_ptr=h_next.unsafe_ptr(), src_buf=next_shape.d_leaves)
        ctx.synchronize()
        for i in range(n):
            if h_first[i] != UInt32(99):
                raise Error("leaf shapes did not share their row plane")
            if h_other[i] != UInt32(11):
                raise Error("concurrent tasks shared a row plane")
        for i in range(2):
            if h_leaves[i] != UInt32(5):
                raise Error("leaf-specific planes overlap")
        for i in range(3):
            if h_next[i] != UInt32(9):
                raise Error("new leaf plane has wrong extent")
        if not next_shape.matches(n, 3, 1, 1, 1, 32):
            raise Error("shared scratch lost its exact shape metadata")
    print("CTR_ARENA_SCRATCH_PASS shapes=3 sharing=within_task isolation=across_tasks")
