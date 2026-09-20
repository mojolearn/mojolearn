# SPDX-License-Identifier: Apache-2.0
"""Ordered FP32 data parallelism and single-device replay for ByteTrainer.

The logical shard count and order are independent of the physical device count.
Each shard produces a mean-CE gradient. Reduction is a SUM, never an average:
copy g[0], then add g[1], ..., g[K-1] with the IDENTICAL FTZ/FMA primitive.
No collective chooses the arithmetic order. Qualification is recorded separately.
"""
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from max.algorithm import sync_parallelize
from core.step_phase import STEP_PHASE_TIMERS
from checks.numerics import ftz, identical_mul_add
from training.byte_lm import (
    ByteTrainer, byte_gradient_device, byte_update_device,
    byte_validate_tokens, byte_rollback, _require_device_finite,
    _FAULT_NAN,
)
from training.byte_lm_optimizer_pool import pool_snapshot, pool_update, pool_restore, pool_maybe_fault
from training.checks.optimizer import OPT_RECORD_INTERMEDIATES
from training.byte_lm_config import ByteConfig
from training.checks.optimizer_oracle import OptimizerConfig
from training.checks.train_loop import _copy_into
from core.multi_gpu import transfer_bytes


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
    var pool_totals: List[DeviceBuffer[DType.float32]]
    var pool_incoming: List[DeviceBuffer[DType.float32]]
    var pool_optimizer: Bool
    var logical_shards: Int
    var busy: Bool
    var usable: Bool

    def __init__(out self):
        self.contexts = List[DeviceContext]()
        self.trainers = List[ByteTrainer]()
        self.total = Optional[DeviceBuffer[DType.float32]]()
        self.pool_totals = List[DeviceBuffer[DType.float32]]()
        self.pool_incoming = List[DeviceBuffer[DType.float32]]()
        self.pool_optimizer = False
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
        _ = self.pool_totals^
        _ = self.pool_incoming^
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
        self.pool_totals = List[DeviceBuffer[DType.float32]]()
        self.pool_incoming = List[DeviceBuffer[DType.float32]]()
        for i in range(len(self.contexts)):
            self.contexts[i].synchronize()
        self.contexts = List[DeviceContext]()

    def open(mut self, devices: List[Int], shards: Int,
             p: List[Float32], m: List[Float32], v: List[Float32],
             flags: List[Bool], completed: Int, opt: OptimizerConfig,
             shape: ByteConfig, pool_optimizer: Bool = False) raises:
        if self.busy or len(self.contexts) != 0:
            raise Error("byte LM parallel: already open")
        if shards < 1 or shards > 1024 or len(devices) < 1 or len(devices) > shards:
            raise Error("byte LM parallel: require 1 <= devices <= shards <= 1024")
        comptime if STEP_PHASE_TIMERS:
            if len(devices) > 1:
                raise Error("byte LM parallel: process-global phase counters cannot profile concurrent devices")
        for i in range(len(devices)):
            if devices[i] < 0:
                raise Error("byte LM parallel: negative device index")
            for j in range(i):
                if devices[i] == devices[j]:
                    raise Error("byte LM parallel: duplicate device index")
        if pool_optimizer and len(devices) > shape.n_total():
            raise Error("byte LM optimizer pool: empty ownership range")
        comptime if OPT_RECORD_INTERMEDIATES:
            if pool_optimizer:
                raise Error("byte LM optimizer pool: recorded intermediates are unsupported")
        self.pool_optimizer = pool_optimizer
        self.logical_shards = shards
        # Each physical replica begins from exactly the same host bytes.
        try:
            for i in range(len(devices)):
                self.contexts.append(DeviceContext(device_id=devices[i]))
                self.trainers.append(ByteTrainer(self.contexts[i], p, m, v,
                    flags, completed, opt, shape,
                    shape.n_total()*i//len(devices) if pool_optimizer else 0,
                    shape.n_total()*(i+1)//len(devices)-shape.n_total()*i//len(devices) if pool_optimizer else -1))
            if pool_optimizer:
                for i in range(len(devices)):
                    var owned = self.trainers[i].buffers.optimizer_count
                    self.pool_totals.append(self.contexts[i].enqueue_create_buffer[DType.float32](owned))
                    self.pool_incoming.append(self.contexts[i].enqueue_create_buffer[DType.float32](owned))
                    self.contexts[i].synchronize()
            else:
                self.total = self.contexts[0].enqueue_create_buffer[DType.float32](shape.n_total())
                self.contexts[0].synchronize()
        except error:
            self.close()
            raise error
        self.usable = True

    def require_open(self) raises:
        if self.busy or not self.usable or len(self.trainers) == 0:
            raise Error("byte LM parallel: closed, busy or lost; restore an export")

    def broadcast_parameters(mut self) raises:
        # Every source range is authoritative on exactly one device. Copies
        # never overwrite another device's owned range.
        for source in range(len(self.trainers)):
            var first = self.trainers[source].buffers.optimizer_first
            var n = self.trainers[source].buffers.optimizer_count
            var part = self.trainers[source].buffers.param.create_sub_buffer[DType.float32](first,n)
            for target in range(len(self.trainers)):
                if target == source:
                    continue
                var dest = self.trainers[target].buffers.param.create_sub_buffer[DType.float32](first,n)
                transfer_bytes(self.contexts[source], self.contexts[target], part, dest, n, True)
        for rank in range(len(self.trainers)):
            self.contexts[rank].synchronize()

    def rollback(mut self) raises:
        # Attempt EVERY replica even when one device is lost. Never expose a
        # surviving subset as a usable group after partial recovery.
        var lost = False
        for i in range(len(self.trainers)):
            try:
                self.trainers[i].grad_step = -1
                if self.pool_optimizer:
                    pool_restore(self.contexts[i], self.trainers[i])
                elif self.trainers[i].shadow_valid:
                    _ = byte_rollback(self.contexts[i], self.trainers[i])
                else:
                    self.trainers[i].validate_device_state(self.contexts[i], self.trainers[i].completed_steps)
                self.trainers[i].healthy = True
            except:
                lost = True
        if self.pool_optimizer and not lost:
            try:
                self.broadcast_parameters()
                for i in range(len(self.trainers)):
                    self.trainers[i].validate_device_state(self.contexts[i], self.trainers[i].completed_steps)
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
        var losses = List[Float32](length=self.logical_shards, fill=Float32(0))
        var n = self.trainers[0].config.n_total()
        try:
            # A wave owns one trainer/context per task. Lists cannot resize
            # until the join. No task touches a different task's state, and
            # every task has finished before reduction, reuse or rollback.
            var width = len(self.trainers)
            var start = 0
            while start < self.logical_shards:
                var active = min(width, self.logical_shards - start)
                var failed = List[Int](length=active, fill=0)
                var cp = rebind[MutPointer[DeviceContext, MutUntrackedOrigin]](self.contexts.unsafe_ptr())
                var tp = rebind[MutPointer[ByteTrainer, MutUntrackedOrigin]](self.trainers.unsafe_ptr())
                var sp = rebind[MutPointer[List[Int32], MutUntrackedOrigin]](shards.unsafe_ptr())
                var lp = rebind[MutPointer[Float32, MutUntrackedOrigin]](losses.unsafe_ptr())
                var fp = rebind[MutPointer[Int, MutUntrackedOrigin]](failed.unsafe_ptr())
                var base = start

                def _gradient_task(rank: Int) {imm cp, imm tp, imm sp, imm lp, imm fp, imm base, imm n}:
                    try:
                        lp[base + rank] = byte_gradient_device(cp[rank], tp[rank], sp[base + rank])
                        _require_device_finite(cp[rank], tp[rank].scan,
                            tp[rank].buffers.grad, n, "shard gradients")
                    except:
                        fp[rank] = 1

                if active == 1:
                    _gradient_task(0)
                else:
                    sync_parallelize(_gradient_task, active)
                for rank in range(active):
                    if failed[rank] != 0:
                        raise Error("byte LM parallel: gradient shard " + String(start + rank) + " failed")
                # Every parameter keeps the same logical left fold. Owners
                # reduce disjoint ranges, so no cross-owner sum exists.
                for rank in range(active):
                    if self.pool_optimizer:
                        for owner in range(width):
                            var first = self.trainers[owner].buffers.optimizer_first
                            var owned = self.trainers[owner].buffers.optimizer_count
                            var part = self.trainers[rank].buffers.grad.create_sub_buffer[DType.float32](first,owned)
                            transfer_bytes(self.contexts[rank], self.contexts[owner], part,
                                self.pool_incoming[owner], owned, rank != owner)
                            if start + rank == 0:
                                _copy_into(self.contexts[owner],self.pool_totals[owner],self.pool_incoming[owner],0,0,owned)
                            else:
                                self.contexts[owner].enqueue_function[_ordered_add_kernel](
                                    self.pool_totals[owner].unsafe_ptr(),self.pool_incoming[owner].unsafe_ptr(),Int32(owned),
                                    grid_dim=((owned+127)//128,1,1),block_dim=(128,1,1))
                            self.contexts[owner].synchronize()
                    else:
                        # Rank zero's gradient has entered the fold before a
                        # later rank may overwrite it.  Reuse that now-dead
                        # full gradient as the transfer staging buffer rather
                        # than retaining a second full-model allocation on
                        # device zero.  `total` remains a distinct accumulator,
                        # so the logical left fold and every FP operation are
                        # unchanged.  On later waves rank zero first computes
                        # its new shard into the same buffer, consumes it, and
                        # only then can a remote transfer overwrite it again.
                        ref incoming = self.trainers[0].buffers.grad
                        if rank != 0:
                            var source = self.trainers[rank].buffers.grad.create_sub_buffer[DType.float32](0, n)
                            transfer_bytes(self.contexts[rank], self.contexts[0], source, incoming, n, True)
                        if start + rank == 0:
                            _copy_into(self.contexts[0], self.total.value(), incoming, 0, 0, n)
                        else:
                            self.contexts[0].enqueue_function[_ordered_add_kernel](
                                self.total.value().unsafe_ptr(), incoming.unsafe_ptr(), Int32(n),
                                grid_dim=((n + 127) // 128, 1, 1), block_dim=(128, 1, 1))
                        self.contexts[0].synchronize()
                start += active
            # Assemble each committed full-gradient replica from disjoint
            # owner ranges. Complete ALL copies/scans before any update.
            for i in range(len(self.trainers)):
                if self.pool_optimizer:
                    for owner in range(width):
                        var first = self.trainers[owner].buffers.optimizer_first
                        var owned = self.trainers[owner].buffers.optimizer_count
                        var target = self.trainers[i].buffers.grad.create_sub_buffer[DType.float32](first,owned)
                        transfer_bytes(self.contexts[owner], self.contexts[i], self.pool_totals[owner], target,
                            owned, owner != i)
                    self.contexts[i].synchronize()
                else:
                    transfer_bytes(self.contexts[0], self.contexts[i], self.total.value(),
                        self.trainers[i].buffers.grad, n, i != 0)
                if self.pool_optimizer:
                    pool_maybe_fault(self.contexts[i], self.trainers[i].buffers.grad, "grad_nonfinite", 0, _FAULT_NAN, self.trainers[i].buffers.optimizer_first)
                _require_device_finite(self.contexts[i], self.trainers[i].scan,
                    self.trainers[i].buffers.grad, n, "summed gradients")
            if self.pool_optimizer:
                for i in range(len(self.trainers)):
                    pool_snapshot(self.contexts[i], self.trainers[i])
                for i in range(len(self.trainers)):
                    pool_update(self.contexts[i], self.trainers[i])
                self.broadcast_parameters()
                for i in range(len(self.trainers)):
                    self.trainers[i].validate_device_state(self.contexts[i], completed + 1)
                for i in range(len(self.trainers)):
                    self.trainers[i].completed_steps = completed + 1
                    self.trainers[i].grad_step = completed + 1
            else:
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
