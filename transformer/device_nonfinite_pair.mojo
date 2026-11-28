# SPDX-License-Identifier: Apache-2.0
# Two independent input refusals share one completion wait, without changing
# their existing predicate, scan geometry or first-index reduction.
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from core.device_scan import NONFINITE_NONE, SCAN_TPB, _scan_blocks, _fold_partials, nonfinite_partial_kernel
from core.step_phase import step_count_device_alloc, step_count_host_alloc, step_count_launch, step_count_d2h, step_count_sync


struct DeviceNonfinitePair(Movable):
    """Operation-owned scratch for two same-length scans on one live context.

    Allocate at the largest input length; each call scans/copies only its live
    prefix. The optional first slot supports a missing normalization stage.
    Every call rereads current inputs. Callers inspect the first result before
    the second and must not commit subsequent work before both checks pass.
    This owner is private to one operation, with no overlapping scan calls.
    """
    var capacity: Int
    var first_part: Optional[DeviceBuffer[DType.int32]]
    var first_host: Optional[HostBuffer[DType.int32]]
    var second_part: DeviceBuffer[DType.int32]
    var second_host: HostBuffer[DType.int32]

    def __init__(out self, ctx: DeviceContext, capacity: Int, scan_first: Bool) raises:
        if capacity < 0 or capacity > Int(NONFINITE_NONE):
            raise Error("nonfinite pair: length outside Int32 index range")
        self.capacity = capacity
        self.first_part = None
        self.first_host = None
        var blocks = max(1, _scan_blocks(capacity))
        if scan_first:
            step_count_device_alloc()
            self.first_part = ctx.enqueue_create_buffer[DType.int32](blocks)
            step_count_host_alloc()
            self.first_host = ctx.enqueue_create_host_buffer[DType.int32](blocks)
        step_count_device_alloc()
        self.second_part = ctx.enqueue_create_buffer[DType.int32](blocks)
        step_count_host_alloc()
        self.second_host = ctx.enqueue_create_host_buffer[DType.int32](blocks)

    def first_nonfinite(mut self, ctx: DeviceContext,
                        mut first: DeviceBuffer[DType.float32],
                        mut second: DeviceBuffer[DType.float32], n: Int) raises -> Tuple[Int, Int]:
        if n < 0 or n > self.capacity or n > len(second) or (self.first_part and n > len(first)):
            raise Error("nonfinite pair: live prefix exceeds admitted capacity")
        if n == 0:
            return (-1, -1)
        var blocks = _scan_blocks(n)
        var first_view = Optional[DeviceBuffer[DType.int32]]()
        var second_view = Optional[DeviceBuffer[DType.int32]]()
        try:
            if self.first_part:
                step_count_launch()
                ctx.enqueue_function[nonfinite_partial_kernel](
                    self.first_part.value().unsafe_ptr(), first.unsafe_ptr(), Int32(n),
                    grid_dim=(blocks, 1, 1), block_dim=(SCAN_TPB, 1, 1))
                first_view = self.first_part.value().create_sub_buffer[DType.int32](0, blocks)
                step_count_d2h()
                ctx.enqueue_copy(dst_ptr=self.first_host.value().unsafe_ptr(), src_buf=first_view.value())
            step_count_launch()
            ctx.enqueue_function[nonfinite_partial_kernel](
                self.second_part.unsafe_ptr(), second.unsafe_ptr(), Int32(n),
                grid_dim=(blocks, 1, 1), block_dim=(SCAN_TPB, 1, 1))
            second_view = self.second_part.create_sub_buffer[DType.int32](0, blocks)
            step_count_d2h()
            ctx.enqueue_copy(dst_ptr=self.second_host.unsafe_ptr(), src_buf=second_view.value())
            # Both copies share the same in-order queue. Only these host result
            # reads require a wait; no selection/progress mutation is enqueued.
            step_count_sync()
            ctx.synchronize()
            var first_bad = -1
            if self.first_part:
                first_bad = _fold_partials(self.first_host.value(), blocks)
            return (first_bad, _fold_partials(self.second_host, blocks))
        except problem:
            # Retain every input, partial, host mirror and borrowed copy span
            # until queued work drains; cleanup cannot replace the first error.
            try:
                ctx.synchronize()
            except:
                pass
            raise problem^
        finally:
            _ = first_view
            _ = second_view
            _ = first
            _ = second
            _ = self
