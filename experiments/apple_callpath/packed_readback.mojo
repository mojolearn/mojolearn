# SPDX-License-Identifier: Apache-2.0
"""Uncompiled opt-in experiment: several results, one completion boundary."""
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer


struct PackedReadback(Movable):
    var host: HostBuffer[DType.float32]
    var capacity: Int
    var used: Int
    var ready: Bool
    # Retain aliases until completion: no temporary sub-buffer lifetime guess.
    var views: List[DeviceBuffer[DType.float32]]

    def __init__(out self, ctx: DeviceContext, capacity: Int) raises:
        if capacity < 0:
            raise Error("negative packed readback capacity")
        self.host = ctx.enqueue_create_host_buffer[DType.float32](max(capacity, 1))
        self.capacity = capacity
        self.used = 0
        self.ready = False
        self.views = List[DeviceBuffer[DType.float32]]()
        ctx.synchronize()

    def append(mut self, ctx: DeviceContext,
               mut source: DeviceBuffer[DType.float32], offset: Int,
               count: Int) raises -> Int:
        if self.ready or offset < 0 or count < 0 or offset > len(source) - count:
            raise Error("invalid readback source or already completed batch")
        if count > self.capacity - self.used:
            raise Error("packed readback capacity exceeded")
        var start = self.used
        if count > 0:
            self.views.append(source.create_sub_buffer[DType.float32](offset, count))
            ctx.enqueue_copy(dst_ptr=self.host.unsafe_ptr().unsafe_offset(start),
                             src_buf=self.views[len(self.views) - 1])
        self.used += count
        return start

    def finish(mut self, ctx: DeviceContext) raises:
        ctx.synchronize()
        self.ready = True

    def collect_into(self, offset: Int, mut result: List[Float32]) raises:
        if not self.ready or offset < 0 or offset > self.used - len(result):
            raise Error("readback region is not ready or out of bounds")
        for i in range(len(result)):
            result[i] = self.host.unsafe_ptr()[offset + i]

    def reset(mut self, ctx: DeviceContext) raises:
        # Also valid during exception cleanup; synchronize BEFORE alias release.
        if not self.ready:
            ctx.synchronize()
        self.views.clear()
        self.used = 0
        self.ready = False
