# SPDX-License-Identifier: Apache-2.0
# Native wavefront execution of a logically partitioned captured DAG.
# Stages from different microbatches and independent tensor branches execute
# concurrently on distinct physical workers. Node VJPs recompute locally;
# activation bytes are staged on the host, while arithmetic stays in Exec.
from std.time import perf_counter_ns
from sequence.exec_trait import Exec
from sequence.ops import FP, dummy_ptr
from autograd.partition_plan import GraphBytes, PartitionPlan
from autograd.partition_stage import ModelStage, NodeGradients
from autograd.optim import OptimizerConfig
from autograd.model import epoch_lr
from autograd.rankn import ShapeN
from autograd.program import PROGRAM_IN_IDS
from autograd.cast_ops import upload_cast, upload_cast_i32
from autograd.data import epoch_order, gather_rows, DATA_MAX_ROWS
from core.cast_in import CastIn
from core.host_parallel import host_parallelize
from checks.numerics import ftz, identical_div, identical_mul_add


@fieldwise_init
struct PartitionEvent(ImplicitlyCopyable, Movable):
    var node: Int
    var microbatch: Int
    var stage: Int
    var device: Int
    var backward: Bool
    var start_ns: Int64
    var end_ns: Int64


def empty_graph_bytes(n: Int) -> List[GraphBytes]:
    var out = List[GraphBytes]()
    for k in range(n):  # driver-loop: graph value slots, no tensor data
        out.append(GraphBytes())
    return out^


struct PartitionedProgram[E: Exec & Movable & Deinitable](Movable):
    var plan: PartitionPlan
    var workers: List[ModelStage[Self.E]]
    var devices: List[Int]
    var cfg: OptimizerConfig
    var optimizer_ints: List[Int]
    var optimizer_floats: List[Float32]
    var microbatches: Int
    var seed: UInt64
    var epoch: Int
    var step: Int
    var busy: Bool
    var usable: Bool
    var events: List[PartitionEvent]

    def __init__(out self, var plan: PartitionPlan, devices: List[Int], var executors: List[Self.E],
                 cfg: OptimizerConfig, oi: List[Int], of: List[Float32], microbatches: Int,
                 seed: UInt64) raises:
        if len(devices) < 1 or len(devices) > plan.stages or microbatches < 1:
            raise Error("partitioned program needs physical devices <= logical stages and positive microbatches")
        if plan.program.n_in < 1 or len(plan.program.outs) != 1 or len(plan.parameters) < 1 or plan.program.n_nodes() < 1:
            raise Error("partitioned training needs inputs, parameters and one output")
        self.plan = plan^
        self.workers = List[ModelStage[Self.E]]()
        self.devices = devices.copy()
        self.cfg = cfg
        self.optimizer_ints = oi.copy()
        self.optimizer_floats = of.copy()
        self.microbatches = microbatches
        self.seed = seed
        self.epoch = 0
        self.step = 0
        self.busy = False
        self.usable = True
        self.events = List[PartitionEvent]()
        for rank in range(len(devices)):  # driver-loop: physical stage construction
            var ex = executors.pop(0)
            self.workers.append(ModelStage[Self.E](ex^, self.plan, cfg, rank, len(devices)))

    def ready(self) raises:
        if self.busy or not self.usable:
            raise Error("partitioned program is busy, closed or failed; restore a checkpoint")

    def wave(mut self, mut values: List[GraphBytes], mut gradients: List[GraphBytes],
             mut parameter_grads: List[GraphBytes], micros: Int, wave: Int,
             backward: Bool, training: Bool) raises:
        var p = self.plan.program.copy()
        var placement = self.plan.placement.copy()
        var levels = self.plan.levels.copy()
        var parameters = self.plan.parameters.copy()
        var width = len(self.workers)
        var nv = p.n_values()
        var np = len(p.params)
        var wp = rebind[MutPointer[ModelStage[Self.E], MutUntrackedOrigin]](self.workers.unsafe_ptr())
        var vp = rebind[MutPointer[GraphBytes, MutUntrackedOrigin]](values.unsafe_ptr())
        var gp = rebind[MutPointer[GraphBytes, MutUntrackedOrigin]](gradients.unsafe_ptr())
        var failed = List[String](length=width, fill=String())
        var fp = rebind[MutPointer[String, MutUntrackedOrigin]](failed.unsafe_ptr())
        var results = List[NodeGradients]()
        var marks = List[PartitionEvent]()
        for k in range(micros * p.n_nodes()):  # driver-loop: disjoint result slots
            results.append(NodeGradients())
            marks.append(PartitionEvent(-1, -1, -1, -1, backward, 0, 0))
        var rp = rebind[MutPointer[NodeGradients, MutUntrackedOrigin]](results.unsafe_ptr())
        var ep = rebind[MutPointer[PartitionEvent, MutUntrackedOrigin]](marks.unsafe_ptr())
        var step = self.step
        var logical_micros = self.microbatches
        var devices = self.devices.copy()
        # Dependencies have lower DAG levels, so work in a wave has no edges
        # within that wave. Each host task owns exactly one physical executor.
        def task(rank: Int) {imm wp, imm vp, imm gp, imm fp, imm rp, imm ep, imm p, imm placement,
                             imm levels, imm parameters, imm width, imm nv, imm micros,
                             imm wave, imm backward, imm training, imm step, imm logical_micros, imm devices}:
            try:
                for micro in range(micros):  # driver-loop: pipeline microbatches
                    var level = wave - micro
                    for node in range(p.n_nodes()):  # driver-loop: ready DAG operations
                        if levels[node] != level or placement[node] % width != rank:
                            continue
                        var slot = micro * p.n_nodes() + node
                        var result_ref = p.n_in + node
                        if backward and len(gp[micro * nv + result_ref].words) == 0:
                            continue
                        var args = List[GraphBytes]()
                        for j in range(p.a_at[node], p.a_at[node + 1]):  # driver-loop: node inputs
                            var rf = p.args[j]
                            args.append(parameters[-rf - 1].copy() if rf < 0 else vp[micro * nv + rf].copy())
                        var seed = gp[micro * nv + result_ref].copy() if backward else GraphBytes()
                        var stream = step * logical_micros + micro if training else -1
                        var begin = Int64(perf_counter_ns())
                        wp[rank].evaluate(p, node, args, seed, backward, stream)
                        wp[rank].session.ex.sync()
                        var end = Int64(perf_counter_ns())
                        ep[slot] = PartitionEvent(node, micro, placement[node], devices[rank], backward, begin, end)
                        if backward:
                            rp[slot] = wp[rank].gradients.copy()
                        else:
                            vp[micro * nv + result_ref] = wp[rank].output.copy()
            except err:
                fp[rank] = String(err)
        if width == 1:
            task(0)
        else:
            host_parallelize(task, width)
        for rank in range(width):  # driver-loop: every worker failure is retained
            if failed[rank].byte_length() > 0:
                raise Error("partitioned physical device " + String(self.devices[rank]) + ": " + failed[rank])
        for k in range(len(marks)):  # driver-loop: completed native operation events
            if marks[k].node >= 0:
                self.events.append(marks[k])
        if not backward:
            return
        # Contributions from independent nodes are folded in reverse graph
        # order regardless of which physical GPU finished first. Microbatch
        # sums are separate and later folded in ascending microbatch order.
        for micro in range(micros):  # driver-loop: independent gradient tapes
            var node = p.n_nodes() - 1
            while node >= 0:  # driver-loop: canonical consumer order
                var slot = micro * p.n_nodes() + node
                for j in range(len(results[slot].refs)):  # driver-loop: node VJP results
                    var rf = results[slot].refs[j]
                    if rf < 0:
                        if self.plan.trainable[-rf - 1]:
                            var owner = self.plan.owners[-rf - 1] % width
                            wp[owner].accumulate(parameter_grads[micro * np - rf - 1], results[slot].values[j])
                    else:
                        var owner = 0 if rf < p.n_in else placement[rf - p.n_in] % width
                        wp[owner].accumulate(gradients[micro * nv + rf], results[slot].values[j])
                node -= 1

    def run(mut self, arrays: List[GraphBytes]) raises -> GraphBytes:
        self.ready()
        var p = self.plan.program.copy()
        if len(arrays) != p.n_in:
            raise Error("partitioned run: input count differs from graph")
        var values = empty_graph_bytes(p.n_values())
        var gradients = empty_graph_bytes(p.n_values())
        var parameter_grads = empty_graph_bytes(len(p.params))
        for k in range(len(arrays)):  # driver-loop: inference input references
            if arrays[k].shape.ndim != p.in_shapes[k].ndim:
                raise Error("partitioned run: input rank differs from graph")
            values[k] = arrays[k].copy()
        self.busy = True
        self.events.clear()
        try:
            for wave in range(1, self.plan.max_level + 1):  # driver-loop: DAG-ready tensor/stage waves
                self.wave(values, gradients, parameter_grads, 1, wave, False, False)
        except err:
            self.busy = False
            self.usable = False
            raise err
        self.busy = False
        var rf = p.outs[0]
        return self.plan.parameters[-rf - 1].copy() if rf < 0 else values[rf].copy()

    def fit(mut self, srcs: List[CastIn], shapes: List[ShapeN], target: CastIn,
            loss_kind: Int, delta: Float32, epochs: Int, batch: Int, shuffle: Bool,
            curve: FP) raises:
        self.ready()
        var p = self.plan.program.copy()
        if len(srcs) != p.n_in or epochs < 1 or batch < 1:
            raise Error("partitioned fit: invalid input count, epochs or batch size")
        var n = shapes[0].d[0]
        if n < 1 or n >= DATA_MAX_ROWS or (loss_kind != 0 and target.rows != n):
            raise Error("partitioned fit: invalid sample count or targets")
        for k in range(len(srcs)):  # driver-loop: input shape contracts
            if shapes[k].ndim < 2 or shapes[k].d[0] != n or shapes[k].ndim != p.in_shapes[k].ndim:
                raise Error("partitioned fit: inputs must retain ranks and share their sample count")
        var bs = min(n, batch)
        var full = List[FP]()
        var batches = List[FP]()
        var storage = List[FP]()
        var ty = dummy_ptr()
        var yb = dummy_ptr()
        var indices = dummy_ptr()
        self.busy = True
        self.events.clear()
        try:
            ty = self.workers[0].session.ex.alloc(max(1, target.count()))
            storage.append(ty)
            yb = self.workers[0].session.ex.alloc(max(1, bs * target.cols))
            storage.append(yb)
            indices = self.workers[0].session.ex.alloc(n)
            storage.append(indices)
            for k in range(len(srcs)):  # driver-loop: call-scoped dataset buffers
                var buf = self.workers[0].session.ex.alloc(srcs[k].count())
                storage.append(buf)
                full.append(buf)
                var batch_buf = self.workers[0].session.ex.alloc(bs * (shapes[k].count() // n))
                storage.append(batch_buf)
                batches.append(batch_buf)
                if p.in_kinds[k] == PROGRAM_IN_IDS:
                    upload_cast_i32(self.workers[0].session.ex, buf, srcs[k], "partitioned ids")
                else:
                    upload_cast(self.workers[0].session.ex, buf, srcs[k])
            if loss_kind != 0:
                upload_cast(self.workers[0].session.ex, ty, target)
            for e in range(epochs):  # driver-loop: epochs
                epoch_order(self.workers[0].session.ex, shuffle, self.seed, self.epoch, n, indices)
                var loss_total = Float32(0)
                var start = 0
                while start < n:  # driver-loop: optimizer minibatches
                    var count = min(bs, n - start)
                    var micros = min(self.microbatches, count)
                    var values = empty_graph_bytes(micros * p.n_values())
                    var gradients = empty_graph_bytes(micros * p.n_values())
                    var param_grads = empty_graph_bytes(micros * len(p.params))
                    var targets = empty_graph_bytes(micros)
                    for micro in range(micros):  # driver-loop: native microbatch assembly
                        var lo = count * micro // micros
                        var rows = count * (micro + 1) // micros - lo
                        for k in range(len(srcs)):  # driver-loop: input arrays
                            var sh = shapes[k]
                            var width = sh.count() // n
                            sh.d[0] = rows
                            var data = List[Float32](length=sh.count(), fill=Float32(0))
                            gather_rows(self.workers[0].session.ex, full[k], indices, start + lo, rows, width, batches[k])
                            self.workers[0].session.ex.download(FP(unsafe_from_address=Int(data.unsafe_ptr())), batches[k], sh.count())
                            values[micro * p.n_values() + k] = GraphBytes(sh, data^, p.in_kinds[k] == PROGRAM_IN_IDS)
                        var cols = max(1, target.cols)
                        var y = List[Float32](length=rows * cols, fill=Float32(0))
                        if loss_kind != 0:
                            gather_rows(self.workers[0].session.ex, ty, indices, start + lo, rows, cols, yb)
                            self.workers[0].session.ex.download(FP(unsafe_from_address=Int(y.unsafe_ptr())), yb, rows * cols)
                        targets[micro] = GraphBytes(ShapeN.matrix(rows, cols), y^)
                    var last_wave = self.plan.max_level + micros - 1
                    for wave in range(1, last_wave + 1):  # driver-loop: overlapping forward pipeline
                        self.wave(values, gradients, param_grads, micros, wave, False, True)
                    for micro in range(micros):  # driver-loop: objective seeds
                        var rows = count * (micro + 1) // micros - count * micro // micros
                        var rf = p.outs[0]
                        var output = self.plan.parameters[-rf - 1].copy() if rf < 0 else values[micro * p.n_values() + rf].copy()
                        var owner = self.plan.owners[-rf - 1] if rf < 0 else (0 if rf < p.n_in else self.plan.placement[rf - p.n_in])
                        var result = self.workers[owner % len(self.workers)].loss_seed(output, targets[micro], loss_kind, delta,
                            ftz(identical_div(Float32(rows), Float32(count))))
                        if rf < 0:
                            param_grads[micro * len(p.params) - rf - 1] = result[1].copy()
                        else:
                            gradients[micro * p.n_values() + rf] = result[1].copy()
                        loss_total = ftz(identical_mul_add(Float32(rows), result[0], loss_total))
                    var wave = last_wave
                    while wave > 0:  # driver-loop: overlapping reverse pipeline
                        self.wave(values, gradients, param_grads, micros, wave, True, True)
                        wave -= 1
                    var totals = empty_graph_bytes(len(p.params))
                    for micro in range(micros):  # driver-loop: immutable logical microbatch fold
                        for j in range(len(p.params)):  # driver-loop: parameter gradient owners
                            if self.plan.trainable[j] and len(param_grads[micro * len(p.params) + j].words) > 0:
                                self.workers[self.plan.owners[j] % len(self.workers)].accumulate(
                                    totals[j], param_grads[micro * len(p.params) + j])
                    var lr = epoch_lr(self.cfg.lr, self.optimizer_ints[2], self.optimizer_ints[3], self.optimizer_floats[6], self.epoch)
                    for rank in range(len(self.workers)):  # driver-loop: disjoint model and optimizer updates
                        self.workers[rank].update(self.plan.parameters, totals, self.cfg, self.step + 1, lr)
                    self.step += 1
                    start += count
                curve[e] = ftz(identical_div(loss_total, Float32(n)))
                self.epoch += 1
        except err:
            self.usable = False
            self.busy = False
            self.release_inputs(storage)
            raise err
        self.release_inputs(storage)
        self.busy = False

    def release_inputs(mut self, storage: List[FP]) raises:
        # One acquisition ledger handles partial setup without parallel-list
        # indexing and releases call storage before a failed object is closed.
        for k in range(len(storage)):  # driver-loop: call-scoped input owners
            self.workers[0].session.ex.free(storage[k])

    def close(mut self) raises:
        if self.busy:
            raise Error("partitioned program is busy")
        self.usable = False
        for rank in range(len(self.workers)):  # driver-loop: physical owner lifetimes
            self.workers[rank].close()
        self.workers.clear()
