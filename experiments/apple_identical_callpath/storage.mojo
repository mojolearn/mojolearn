# SPDX-License-Identifier: Apache-2.0
"""Source-only candidate: context-owned, fixed-shape reusable call storage.

Internal to synchronous adapters. No public asynchronous result handles and
no caller-supplied contexts: queued work cannot accidentally switch queues.
"""
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


struct IdenticalCallStorage(Movable):
    var ctx: DeviceContext
    var inputs: List[DeviceBuffer[DType.float32]]
    var outputs: List[DeviceBuffer[DType.float32]]
    var host_inputs: List[HostBuffer[DType.float32]]
    var host_outputs: List[HostBuffer[DType.float32]]
    var count: Int
    var slots: Int
    var pending: Bool
    var poisoned: Bool

    def __init__(out self, count: Int, slots: Int) raises:
        comptime if not is_defined["MOJOLEARN_EXPERIMENT_IDENTICAL_CALLPATH"]():
            raise Error("experimental callpath requires explicit opt-in")
        comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
            raise Error("experimental callpath requires IDENTICAL numerical mode")
        comptime if not has_apple_gpu_accelerator():
            raise Error("experimental callpath is restricted to Apple GPU targets")
        if count <= 0 or count > 2147483647 or slots <= 0:
            raise Error("positive fixed shape and slot capacity required")
        self.ctx = DeviceContext()
        self.inputs = List[DeviceBuffer[DType.float32]]()
        self.outputs = List[DeviceBuffer[DType.float32]]()
        self.host_inputs = List[HostBuffer[DType.float32]]()
        self.host_outputs = List[HostBuffer[DType.float32]]()
        self.count = count
        self.slots = slots
        self.pending = False
        self.poisoned = False
        for _ in range(slots):
            self.inputs.append(self.ctx.enqueue_create_buffer[DType.float32](count))
            self.outputs.append(self.ctx.enqueue_create_buffer[DType.float32](count))
            self.host_inputs.append(self.ctx.enqueue_create_host_buffer[DType.float32](count))
            self.host_outputs.append(self.ctx.enqueue_create_host_buffer[DType.float32](count))
        self.ctx.synchronize()

    def stage(mut self, values: List[List[Float32]]) raises:
        if self.pending or self.poisoned:
            raise Error("pending or poisoned call storage cannot be reused")
        if len(values) <= 0 or len(values) > self.slots:
            raise Error("batch exceeds fixed slot capacity")
        # Validate every item before altering any staging buffer.
        for i in range(len(values)):
            if len(values[i]) != self.count:
                raise Error("batch item has wrong fixed shape")
        for i in range(len(values)):
            for j in range(self.count):
                self.host_inputs[i].unsafe_ptr()[j] = values[i][j]

    def finish(mut self) raises:
        # Clear only after success. A failed drain makes reuse unsafe.
        self.ctx.synchronize()
        self.pending = False

    def __deinit__(deinit self):
        # Adapters finish or drain before returning. This is a final defensive
        # drain; a device failure is not a successful completion certificate.
        try:
            self.ctx.synchronize()
        except:
            pass
        _ = self.host_outputs^
        _ = self.host_inputs^
        _ = self.outputs^
        _ = self.inputs^
        # Match the repository's buffer-before-context destruction discipline.
        try:
            self.ctx.synchronize()
        except:
            pass
        _ = self.ctx^
