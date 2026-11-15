# SPDX-License-Identifier: Apache-2.0
"""Experimental candidate: context-owned, fixed-shape reusable call storage.

Internal to synchronous adapters. No public asynchronous result handles and
no caller-supplied contexts: queued work cannot accidentally switch queues.
"""
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer


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
        # TOMBSTONE: MOJOLEARN_EXPERIMENT_IDENTICAL_CALLPATH (no A/B recorded for the opt-in arm) deleted 2026-11-15 by lane/expdel2-07-base; code recoverable at 0278c32300.
        # Restore: git apply experiments/removed/MOJOLEARN_EXPERIMENT_IDENTICAL_CALLPATH.expdel2-07-base.patch
        raise Error("experimental callpath requires explicit opt-in")

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
