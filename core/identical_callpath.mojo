# SPDX-License-Identifier: Apache-2.0
"""SOURCE ONLY: opt-in vendor-neutral storage and completion for any GPU algorithm.

No math, replacement kernels, implicit zeroing, global cache, or hidden queue.
Existing algorithms enqueue their exact kernels against the typed device slots.
Public calls require serialized host ownership. Never mutate internal banks.
"""
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


struct CallBufferBank[dtype: DType](Movable):
    var device: List[DeviceBuffer[dtype]]
    var upload_host: List[HostBuffer[dtype]]
    var download_host: List[HostBuffer[dtype]]
    var sizes: List[Int]
    var staged: List[Bool]
    var uploaded: List[Bool]
    var requested: List[Bool]
    var ready: List[Bool]

    def __init__(out self):
        self.device = List[DeviceBuffer[dtype]]()
        self.upload_host = List[HostBuffer[dtype]]()
        self.download_host = List[HostBuffer[dtype]]()
        self.sizes = List[Int]()
        self.staged = List[Bool]()
        self.uploaded = List[Bool]()
        self.requested = List[Bool]()
        self.ready = List[Bool]()

    def check(self, index: Int) raises:
        if index < 0 or index >= len(self.sizes):
            raise Error("call buffer index out of range")

    def reserve(mut self, ctx: DeviceContext, count: Int) raises -> Int:
        if count < 0:
            raise Error("negative call buffer extent")
        var index = len(self.sizes)
        self.device.append(ctx.enqueue_create_buffer[dtype](max(count, 1)))
        self.upload_host.append(ctx.enqueue_create_host_buffer[dtype](max(count, 1)))
        self.download_host.append(ctx.enqueue_create_host_buffer[dtype](max(count, 1)))
        self.sizes.append(count)
        self.staged.append(False)
        self.uploaded.append(False)
        self.requested.append(False)
        self.ready.append(False)
        # Allocation completion is a setup boundary, never a warmed call.
        ctx.synchronize()
        return index

    def stage(mut self, index: Int, values: List[Scalar[dtype]]) raises:
        self.check(index)
        if len(values) != self.sizes[index]:
            raise Error("call buffer shape mismatch")
        for i in range(len(values)):
            self.upload_host[index].unsafe_ptr()[i] = values[i]
        self.staged[index] = True

    def begin(mut self):
        for i in range(len(self.sizes)):
            self.uploaded[i] = False
            self.requested[i] = False
            self.ready[i] = False

    def upload(mut self, ctx: DeviceContext, index: Int) raises:
        self.check(index)
        if not self.staged[index] or self.uploaded[index]:
            raise Error("upload requires staged data and one submission per batch")
        self.uploaded[index] = True
        if self.sizes[index] > 0:
            ctx.enqueue_copy(dst_buf=self.device[index],
                             src_ptr=self.upload_host[index].unsafe_ptr())

    def readback(mut self, ctx: DeviceContext, index: Int) raises:
        self.check(index)
        if self.requested[index]:
            raise Error("one readback per buffer per batch")
        self.requested[index] = True
        if self.sizes[index] > 0:
            ctx.enqueue_copy(dst_ptr=self.download_host[index].unsafe_ptr(),
                             src_buf=self.device[index])

    def complete(mut self):
        for i in range(len(self.sizes)):
            self.ready[i] = self.requested[i]

    def collect(mut self, index: Int, mut result: List[Scalar[dtype]]) raises:
        self.check(index)
        if not self.ready[index] or len(result) != self.sizes[index]:
            raise Error("readback not ready or result shape mismatch")
        for i in range(len(result)):
            result[i] = self.download_host[index].unsafe_ptr()[i]

    def release(mut self):
        self.download_host.clear()
        self.upload_host.clear()
        self.device.clear()


struct IdenticalCallSession(Movable):
    """One queue and arbitrary heterogeneous buffers, shared by all families.

    reserve/stage -> begin -> upload/existing kernels/readback -> finish/collect.
    Persistent model/state slots need no upload on subsequent batches.
    Scratch initialization remains the algorithm's existing operation.
    Use finish/collect before every required host decision; then begin again.
    """
    var ctx: DeviceContext
    # 0 idle/completed, 1 active, 2 poisoned. No automatic reuse after failure.
    var phase: Int
    var f32: CallBufferBank[DType.float32]
    var f64: CallBufferBank[DType.float64]
    var f16: CallBufferBank[DType.float16]
    var bf16: CallBufferBank[DType.bfloat16]
    var i32: CallBufferBank[DType.int32]
    var i64: CallBufferBank[DType.int64]
    var u32: CallBufferBank[DType.uint32]
    var u64: CallBufferBank[DType.uint64]
    var i8: CallBufferBank[DType.int8]
    var i16: CallBufferBank[DType.int16]
    var u16: CallBufferBank[DType.uint16]
    var u8: CallBufferBank[DType.uint8]

    def __init__(out self, device_id: Int = -1) raises:
        comptime if not is_defined["MOJOLEARN_EXPERIMENT_IDENTICAL_CALLPATH"]():
            raise Error("shared callpath requires explicit experimental opt-in")
        comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
            raise Error("shared callpath requires IDENTICAL numerical mode")
        if device_id < -1:
            raise Error("invalid call session device id")
        if device_id == -1:
            self.ctx = DeviceContext()
        else:
            self.ctx = DeviceContext(device_id=device_id)
        self.phase = 0
        self.f32 = CallBufferBank[DType.float32]()
        self.f64 = CallBufferBank[DType.float64]()
        self.f16 = CallBufferBank[DType.float16]()
        self.bf16 = CallBufferBank[DType.bfloat16]()
        self.i32 = CallBufferBank[DType.int32]()
        self.i64 = CallBufferBank[DType.int64]()
        self.u32 = CallBufferBank[DType.uint32]()
        self.u64 = CallBufferBank[DType.uint64]()
        self.i8 = CallBufferBank[DType.int8]()
        self.i16 = CallBufferBank[DType.int16]()
        self.u16 = CallBufferBank[DType.uint16]()
        self.u8 = CallBufferBank[DType.uint8]()

    def require_idle(self) raises:
        if self.phase != 0:
            raise Error("call session must be completed and unpoisoned")

    def require_active(self) raises:
        if self.phase != 1:
            raise Error("call session must be active")

    def begin(mut self) raises:
        self.require_idle()
        self.f32.begin()
        self.f64.begin()
        self.f16.begin()
        self.bf16.begin()
        self.i32.begin()
        self.i64.begin()
        self.u32.begin()
        self.u64.begin()
        self.i8.begin()
        self.i16.begin()
        self.u16.begin()
        self.u8.begin()
        self.phase = 1

    def finish(mut self) raises:
        self.require_active()
        # Poison before waiting so a failed wait cannot permit unsafe reuse.
        self.phase = 2
        self.ctx.synchronize()
        self.f32.complete()
        self.f64.complete()
        self.f16.complete()
        self.bf16.complete()
        self.i32.complete()
        self.i64.complete()
        self.u32.complete()
        self.u64.complete()
        self.i8.complete()
        self.i16.complete()
        self.u16.complete()
        self.u8.complete()
        self.phase = 0

    def abort(mut self) raises:
        # Required after an external algorithm/kernel enqueue raises.
        # A drained failed session remains poisoned; construct a new one.
        self.phase = 2
        self.ctx.synchronize()

    def reserve_f32(mut self, count: Int) raises -> Int:
        self.require_idle()
        try:
            return self.f32.reserve(self.ctx, count)
        except e:
            self.phase = 2
            raise e

    def stage_f32(mut self, index: Int, values: List[Scalar[DType.float32]]) raises:
        self.require_idle()
        self.f32.stage(index, values)

    def upload_f32(mut self, index: Int) raises:
        self.require_active()
        try:
            self.f32.upload(self.ctx, index)
        except e:
            self.phase = 2
            raise e

    def readback_f32(mut self, index: Int) raises:
        self.require_active()
        try:
            self.f32.readback(self.ctx, index)
        except e:
            self.phase = 2
            raise e

    def collect_f32(mut self, index: Int, mut result: List[Scalar[DType.float32]]) raises:
        self.require_idle()
        self.f32.collect(index, result)

    def reserve_f64(mut self, count: Int) raises -> Int:
        self.require_idle()
        try:
            return self.f64.reserve(self.ctx, count)
        except e:
            self.phase = 2
            raise e

    def stage_f64(mut self, index: Int, values: List[Scalar[DType.float64]]) raises:
        self.require_idle()
        self.f64.stage(index, values)

    def upload_f64(mut self, index: Int) raises:
        self.require_active()
        try:
            self.f64.upload(self.ctx, index)
        except e:
            self.phase = 2
            raise e

    def readback_f64(mut self, index: Int) raises:
        self.require_active()
        try:
            self.f64.readback(self.ctx, index)
        except e:
            self.phase = 2
            raise e

    def collect_f64(mut self, index: Int, mut result: List[Scalar[DType.float64]]) raises:
        self.require_idle()
        self.f64.collect(index, result)

    def reserve_f16(mut self, count: Int) raises -> Int:
        self.require_idle()
        try:
            return self.f16.reserve(self.ctx, count)
        except e:
            self.phase = 2
            raise e

    def stage_f16(mut self, index: Int, values: List[Scalar[DType.float16]]) raises:
        self.require_idle()
        self.f16.stage(index, values)

    def upload_f16(mut self, index: Int) raises:
        self.require_active()
        try:
            self.f16.upload(self.ctx, index)
        except e:
            self.phase = 2
            raise e

    def readback_f16(mut self, index: Int) raises:
        self.require_active()
        try:
            self.f16.readback(self.ctx, index)
        except e:
            self.phase = 2
            raise e

    def collect_f16(mut self, index: Int, mut result: List[Scalar[DType.float16]]) raises:
        self.require_idle()
        self.f16.collect(index, result)

    def reserve_bf16(mut self, count: Int) raises -> Int:
        self.require_idle()
        try:
            return self.bf16.reserve(self.ctx, count)
        except e:
            self.phase = 2
            raise e

    def stage_bf16(mut self, index: Int, values: List[Scalar[DType.bfloat16]]) raises:
        self.require_idle()
        self.bf16.stage(index, values)

    def upload_bf16(mut self, index: Int) raises:
        self.require_active()
        try:
            self.bf16.upload(self.ctx, index)
        except e:
            self.phase = 2
            raise e

    def readback_bf16(mut self, index: Int) raises:
        self.require_active()
        try:
            self.bf16.readback(self.ctx, index)
        except e:
            self.phase = 2
            raise e

    def collect_bf16(mut self, index: Int, mut result: List[Scalar[DType.bfloat16]]) raises:
        self.require_idle()
        self.bf16.collect(index, result)

    def reserve_i32(mut self, count: Int) raises -> Int:
        self.require_idle()
        try:
            return self.i32.reserve(self.ctx, count)
        except e:
            self.phase = 2
            raise e

    def stage_i32(mut self, index: Int, values: List[Scalar[DType.int32]]) raises:
        self.require_idle()
        self.i32.stage(index, values)

    def upload_i32(mut self, index: Int) raises:
        self.require_active()
        try:
            self.i32.upload(self.ctx, index)
        except e:
            self.phase = 2
            raise e

    def readback_i32(mut self, index: Int) raises:
        self.require_active()
        try:
            self.i32.readback(self.ctx, index)
        except e:
            self.phase = 2
            raise e

    def collect_i32(mut self, index: Int, mut result: List[Scalar[DType.int32]]) raises:
        self.require_idle()
        self.i32.collect(index, result)

    def reserve_i64(mut self, count: Int) raises -> Int:
        self.require_idle()
        try:
            return self.i64.reserve(self.ctx, count)
        except e:
            self.phase = 2
            raise e

    def stage_i64(mut self, index: Int, values: List[Scalar[DType.int64]]) raises:
        self.require_idle()
        self.i64.stage(index, values)

    def upload_i64(mut self, index: Int) raises:
        self.require_active()
        try:
            self.i64.upload(self.ctx, index)
        except e:
            self.phase = 2
            raise e

    def readback_i64(mut self, index: Int) raises:
        self.require_active()
        try:
            self.i64.readback(self.ctx, index)
        except e:
            self.phase = 2
            raise e

    def collect_i64(mut self, index: Int, mut result: List[Scalar[DType.int64]]) raises:
        self.require_idle()
        self.i64.collect(index, result)

    def reserve_u32(mut self, count: Int) raises -> Int:
        self.require_idle()
        try:
            return self.u32.reserve(self.ctx, count)
        except e:
            self.phase = 2
            raise e

    def stage_u32(mut self, index: Int, values: List[Scalar[DType.uint32]]) raises:
        self.require_idle()
        self.u32.stage(index, values)

    def upload_u32(mut self, index: Int) raises:
        self.require_active()
        try:
            self.u32.upload(self.ctx, index)
        except e:
            self.phase = 2
            raise e

    def readback_u32(mut self, index: Int) raises:
        self.require_active()
        try:
            self.u32.readback(self.ctx, index)
        except e:
            self.phase = 2
            raise e

    def collect_u32(mut self, index: Int, mut result: List[Scalar[DType.uint32]]) raises:
        self.require_idle()
        self.u32.collect(index, result)

    def reserve_u64(mut self, count: Int) raises -> Int:
        self.require_idle()
        try:
            return self.u64.reserve(self.ctx, count)
        except e:
            self.phase = 2
            raise e

    def stage_u64(mut self, index: Int, values: List[Scalar[DType.uint64]]) raises:
        self.require_idle()
        self.u64.stage(index, values)

    def upload_u64(mut self, index: Int) raises:
        self.require_active()
        try:
            self.u64.upload(self.ctx, index)
        except e:
            self.phase = 2
            raise e

    def readback_u64(mut self, index: Int) raises:
        self.require_active()
        try:
            self.u64.readback(self.ctx, index)
        except e:
            self.phase = 2
            raise e

    def collect_u64(mut self, index: Int, mut result: List[Scalar[DType.uint64]]) raises:
        self.require_idle()
        self.u64.collect(index, result)

    def reserve_u8(mut self, count: Int) raises -> Int:
        self.require_idle()
        try:
            return self.u8.reserve(self.ctx, count)
        except e:
            self.phase = 2
            raise e

    def stage_u8(mut self, index: Int, values: List[Scalar[DType.uint8]]) raises:
        self.require_idle()
        self.u8.stage(index, values)

    def upload_u8(mut self, index: Int) raises:
        self.require_active()
        try:
            self.u8.upload(self.ctx, index)
        except e:
            self.phase = 2
            raise e

    def readback_u8(mut self, index: Int) raises:
        self.require_active()
        try:
            self.u8.readback(self.ctx, index)
        except e:
            self.phase = 2
            raise e

    def collect_u8(mut self, index: Int, mut result: List[Scalar[DType.uint8]]) raises:
        self.require_idle()
        self.u8.collect(index, result)

    def reserve_i8(mut self, count: Int) raises -> Int:
        self.require_idle()
        try:
            return self.i8.reserve(self.ctx, count)
        except e:
            self.phase = 2
            raise e

    def stage_i8(mut self, index: Int, values: List[Scalar[DType.int8]]) raises:
        self.require_idle()
        self.i8.stage(index, values)

    def upload_i8(mut self, index: Int) raises:
        self.require_active()
        try:
            self.i8.upload(self.ctx, index)
        except e:
            self.phase = 2
            raise e

    def readback_i8(mut self, index: Int) raises:
        self.require_active()
        try:
            self.i8.readback(self.ctx, index)
        except e:
            self.phase = 2
            raise e

    def collect_i8(mut self, index: Int, mut result: List[Scalar[DType.int8]]) raises:
        self.require_idle()
        self.i8.collect(index, result)

    def reserve_i16(mut self, count: Int) raises -> Int:
        self.require_idle()
        try:
            return self.i16.reserve(self.ctx, count)
        except e:
            self.phase = 2
            raise e

    def stage_i16(mut self, index: Int, values: List[Scalar[DType.int16]]) raises:
        self.require_idle()
        self.i16.stage(index, values)

    def upload_i16(mut self, index: Int) raises:
        self.require_active()
        try:
            self.i16.upload(self.ctx, index)
        except e:
            self.phase = 2
            raise e

    def readback_i16(mut self, index: Int) raises:
        self.require_active()
        try:
            self.i16.readback(self.ctx, index)
        except e:
            self.phase = 2
            raise e

    def collect_i16(mut self, index: Int, mut result: List[Scalar[DType.int16]]) raises:
        self.require_idle()
        self.i16.collect(index, result)

    def reserve_u16(mut self, count: Int) raises -> Int:
        self.require_idle()
        try:
            return self.u16.reserve(self.ctx, count)
        except e:
            self.phase = 2
            raise e

    def stage_u16(mut self, index: Int, values: List[Scalar[DType.uint16]]) raises:
        self.require_idle()
        self.u16.stage(index, values)

    def upload_u16(mut self, index: Int) raises:
        self.require_active()
        try:
            self.u16.upload(self.ctx, index)
        except e:
            self.phase = 2
            raise e

    def readback_u16(mut self, index: Int) raises:
        self.require_active()
        try:
            self.u16.readback(self.ctx, index)
        except e:
            self.phase = 2
            raise e

    def collect_u16(mut self, index: Int, mut result: List[Scalar[DType.uint16]]) raises:
        self.require_idle()
        self.u16.collect(index, result)

    def __deinit__(deinit self):
        # Device-loss teardown remains unverified, as does partial construction.
        try:
            self.ctx.synchronize()
        except:
            pass
        self.f32.release()
        self.f64.release()
        self.f16.release()
        self.bf16.release()
        self.i32.release()
        self.i64.release()
        self.u32.release()
        self.u64.release()
        self.i8.release()
        self.i16.release()
        self.u16.release()
        self.u8.release()
        try:
            self.ctx.synchronize()
        except:
            pass
        _ = self.f32^
        _ = self.f64^
        _ = self.f16^
        _ = self.bf16^
        _ = self.i32^
        _ = self.i64^
        _ = self.u32^
        _ = self.u64^
        _ = self.i8^
        _ = self.i16^
        _ = self.u16^
        _ = self.u8^
        _ = self.ctx^
