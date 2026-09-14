# SPDX-License-Identifier: Apache-2.0
"""Ordered FP32 data parallelism and single-device replay for ByteTrainer.

The logical shard count and order are independent of the physical device count.
Each shard produces a mean-CE gradient. Reduction is a SUM, never an average:
copy g[0], then add g[1], ..., g[K-1] with the IDENTICAL FTZ/FMA primitive.
No collective chooses the arithmetic order. Qualification is recorded separately.
"""
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from checks.numerics import ftz, identical_mul_add
from training.byte_lm import (
    ByteTrainer, byte_gradient_device, byte_update_device,
    byte_validate_tokens, byte_rollback, _require_device_finite,
)
from training.byte_lm_config import ByteConfig
from training.checks.optimizer_oracle import OptimizerConfig
from training.checks.train_loop import _copy_into


def _ordered_add_kernel(
    total: MutPointer[Float32, MutAnyOrigin],
    shard: MutPointer[Float32, MutAnyOrigin], n: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        total[i] = ftz(identical_mul_add(Float32(1), ftz(total[i]), ftz(shard[i])))


struct ByteParallelTrainer(Movable, Writable):
    var contexts: List[DeviceContext]
    var trainers: List[ByteTrainer]
    var total: Optional[DeviceBuffer[DType.float32]]
    var incoming: Optional[DeviceBuffer[DType.float32]]
    var logical_shards: Int
    var busy: Bool
    var usable: Bool

    def __init__(out self):
        self.contexts = List[DeviceContext]()
        self.trainers = List[ByteTrainer]()
        self.total = Optional[DeviceBuffer[DType.float32]]()
        self.incoming = Optional[DeviceBuffer[DType.float32]]()
        self.logical_shards = 0
        self.busy = False
        self.usable = False

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ByteParallelTrainer")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("ByteParallelTrainer")

    def __deinit__(deinit self):
        # Drain buffer destruction before destroying any device context.
        _ = self.trainers^
        _ = self.total^
        _ = self.incoming^
        for i in range(len(self.contexts)):
            try:
                self.contexts[i].synchronize()
            except:
                pass
        _ = self.contexts^

    def close(mut self) raises:
        if self.busy:
            raise Error("byte LM parallel: busy")
        self.usable = False
        self.trainers = List[ByteTrainer]()
        self.total = None
        self.incoming = None
        for i in range(len(self.contexts)):
            self.contexts[i].synchronize()
        self.contexts = List[DeviceContext]()

    def open(mut self, devices: List[Int], shards: Int,
             p: List[Float32], m: List[Float32], v: List[Float32],
             flags: List[Bool], completed: Int, opt: OptimizerConfig,
             shape: ByteConfig) raises:
        if self.busy or len(self.contexts) != 0:
            raise Error("byte LM parallel: already open")
        if shards < 1 or shards > 1024 or len(devices) < 1 or len(devices) > shards:
            raise Error("byte LM parallel: require 1 <= devices <= shards <= 1024")
        for i in range(len(devices)):
            if devices[i] < 0:
                raise Error("byte LM parallel: negative device index")
            for j in range(i):
                if devices[i] == devices[j]:
                    raise Error("byte LM parallel: duplicate device index")
        self.logical_shards = shards
        # Each physical replica begins from exactly the same host bytes.
        try:
            for i in range(len(devices)):
                self.contexts.append(DeviceContext(device_id=devices[i]))
                self.trainers.append(ByteTrainer(self.contexts[i], p, m, v,
                    flags, completed, opt, shape))
            self.total = self.contexts[0].enqueue_create_buffer[DType.float32](shape.n_total())
            self.incoming = self.contexts[0].enqueue_create_buffer[DType.float32](shape.n_total())
            self.contexts[0].synchronize()
        except error:
            self.close()
            raise error
        self.usable = True

    def require_open(self) raises:
        if self.busy or not self.usable or len(self.trainers) == 0:
            raise Error("byte LM parallel: closed, busy or lost; restore an export")

    def rollback(mut self) raises:
        # Attempt EVERY replica even when one device is lost. Never expose a
        # surviving subset as a usable group after partial recovery.
        var lost = False
        for i in range(len(self.trainers)):
            try:
                self.trainers[i].grad_step = -1
                if self.trainers[i].shadow_valid:
                    _ = byte_rollback(self.contexts[i], self.trainers[i])
                else:
                    self.trainers[i].validate_device_state(self.contexts[i], self.trainers[i].completed_steps)
                self.trainers[i].healthy = True
            except:
                lost = True
        if lost:
            self.usable = False
            raise Error("byte LM parallel: replica recovery failed; restore an export")

    def step(mut self, shards: List[List[Int32]]) raises -> List[Float32]:
        self.require_open()
        if len(shards) != self.logical_shards:
            raise Error("byte LM parallel: logical shard count mismatch")
        var completed = self.trainers[0].completed_steps
        if completed >= 999999:
            raise Error("byte LM parallel: step bound reached")
        for i in range(len(shards)):
            byte_validate_tokens(shards[i], self.trainers[0].config)
        for i in range(len(self.trainers)):
            if not self.trainers[i].healthy or self.trainers[i].completed_steps != completed:
                raise Error("byte LM parallel: replica state mismatch")
        self.busy = True
        for i in range(len(self.trainers)):
            self.trainers[i].shadow_valid = False
            self.trainers[i].grad_step = -1
            self.trainers[i].healthy = False
        var losses = List[Float32]()
        var n = self.trainers[0].config.n_total()
        try:
            # The existing gradient body synchronizes internally. This initial
            # correctness path schedules shards serially; no speedup claim.
            # Assignment changes with device count, but the fold never does.
            for shard in range(self.logical_shards):
                var rank = shard % len(self.trainers)
                losses.append(byte_gradient_device(self.contexts[rank], self.trainers[rank], shards[shard]))
                _require_device_finite(self.contexts[rank], self.trainers[rank].scan,
                    self.trainers[rank].buffers.grad, n, "shard gradients")
                # Copy runs on the source stream. The source wait MUST finish
                # before the root launches a kernel reading the destination.
                self.trainers[rank].buffers.grad.enqueue_copy_to(self.incoming.value())
                self.contexts[rank].synchronize()
                if shard == 0:
                    _copy_into(self.contexts[0], self.total.value(), self.incoming.value(), 0, 0, n)
                else:
                    self.contexts[0].enqueue_function[_ordered_add_kernel](
                        self.total.value().unsafe_ptr(), self.incoming.value().unsafe_ptr(), Int32(n),
                        grid_dim=((n + 127) // 128, 1, 1), block_dim=(128, 1, 1))
                self.contexts[0].synchronize()
            # Complete every broadcast and scan BEFORE any replica updates.
            for i in range(len(self.trainers)):
                self.total.value().enqueue_copy_to(self.trainers[i].buffers.grad)
                self.contexts[0].synchronize()
                _require_device_finite(self.contexts[i], self.trainers[i].scan,
                    self.trainers[i].buffers.grad, n, "summed gradients")
            for i in range(len(self.trainers)):
                byte_update_device(self.contexts[i], self.trainers[i])
            for i in range(len(self.trainers)):
                self.contexts[i].synchronize()
                self.trainers[i].healthy = True
        except error:
            self.busy = False
            self.rollback()
            raise error
        self.busy = False
        return losses^
