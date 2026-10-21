# SPDX-License-Identifier: Apache-2.0
# Native data-parallel training of user-captured graphs. Logical microbatches,
# their gradient fold order and dropout counters never depend on GPU count.
# Parameters are replicated; optimizer moments and gradient accumulators are
# physically partitioned into disjoint contiguous ranges. Transport stages
# bytes through native host memory (also valid on AMD without peer access).
# No speedup or cross-vendor qualification is inferred from this source.
from std.memory import memcpy
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, OP_COPY, OP_FILL, OP_SCALE, dummy_ptr
from autograd.ops import OP_AG_ACCUMULATE
from autograd.ops_api import OpsSession
from autograd.program import OpsProgram, PROGRAM_IN_IDS, same_shape
from autograd.program_runtime import parameter_buffers, replay_values, program_loss
from autograd.rankn import ShapeN
from autograd.optim import OptimizerConfig, OptimizerState, step_lr, optimizer_slots
from autograd.model import epoch_lr
from autograd.import_api import ImportRules
from autograd.data import epoch_order, gather_rows, DATA_MAX_ROWS
from autograd.cast_ops import upload_cast, upload_cast_i32
from core.cast_in import CastIn
from core.host_parallel import host_parallelize
from checks.numerics import identical_mul, identical_div, identical_mul_add, ftz


struct ProgramReplica[E: Exec & Movable & Deinitable](Movable):
    var session: OpsSession[Self.E]
    var program: OpsProgram
    var params: List[FP]
    var trainable: List[Bool]
    var full: List[FP]
    var dataset_storage: List[FP]
    var batches: List[FP]
    var shapes: List[ShapeN]
    var target: FP
    var batch_target: FP
    var indices: FP
    var dataset_ready: Bool
    var target_cols: Int
    var gradients: List[FP]
    var temps: List[Int]
    var loss: Float32
    var states: List[OptimizerState]
    var totals: List[FP]
    var incoming: List[FP]
    var first: List[Int]
    var width: List[Int]

    def __init__(out self, var ex: Self.E, p: OpsProgram, values: List[List[Float32]],
                 trainable: List[Bool], cfg: OptimizerConfig, rank: Int, world: Int) raises:
        self.session = OpsSession[Self.E](ex^, False)
        self.program = p.copy()
        self.params = List[FP]()
        self.trainable = trainable.copy()
        self.full = List[FP]()
        self.dataset_storage = List[FP]()
        self.batches = List[FP]()
        self.shapes = List[ShapeN]()
        self.target = dummy_ptr()
        self.batch_target = dummy_ptr()
        self.indices = dummy_ptr()
        self.dataset_ready = False
        self.target_cols = 1
        self.gradients = List[FP]()
        self.temps = List[Int]()
        self.loss = Float32(0)
        self.states = List[OptimizerState]()
        self.totals = List[FP]()
        self.incoming = List[FP]()
        self.first = List[Int]()
        self.width = List[Int]()
        for j in range(len(p.params)):  # driver-loop: parameter ownership table
            var sh = p.param_shapes[j]
            var src = CastIn.f32_c(Int(values[j].unsafe_ptr()), sh.rows(), sh.cols())
            var h = self.session.upload_nd(src, sh, False)
            self.program.params[j] = h
            self.params.append(FP(unsafe_from_address=self.session.owned[self.session._slot(h)]))
            self.gradients.append(dummy_ptr())
            # A shape-independent partition: adjacent owners cover the whole
            # tensor exactly once, including uneven widths and empty owners.
            var lo = sh.count() * rank // world
            var n = sh.count() * (rank + 1) // world - lo
            self.first.append(lo)
            self.width.append(n)
            self.states.append(OptimizerState(self.session.ex, cfg, max(1, n)) if trainable[j] else OptimizerState())
            self.totals.append(self.session.ex.alloc(max(1, n)))
            self.incoming.append(self.session.ex.alloc(max(1, n)))

    def release_dataset(mut self) raises:
        # Register every acquisition immediately, including a partial setup.
        # Full/batch lists borrow these buffers and need not have equal lengths.
        for j in range(len(self.dataset_storage)):  # driver-loop: call-scoped allocation owners
            self.session.ex.free(self.dataset_storage[j])
        self.dataset_storage.clear()
        self.full.clear()
        self.batches.clear()
        self.shapes.clear()
        self.target = dummy_ptr()
        self.batch_target = dummy_ptr()
        self.indices = dummy_ptr()
        self.dataset_ready = False

    def prepare(mut self, srcs: List[CastIn], shapes: List[ShapeN], target: CastIn,
                batch: Int, loss_kind: Int) raises:
        self.release_dataset()
        var rows = shapes[0].d[0]
        try:
            for k in range(len(srcs)):  # driver-loop: input arrays
                var buf = self.session.ex.alloc(srcs[k].count())
                self.dataset_storage.append(buf)
                self.full.append(buf)
                var batch_buf = self.session.ex.alloc(batch * (shapes[k].count() // rows))
                self.dataset_storage.append(batch_buf)
                self.batches.append(batch_buf)
                self.shapes.append(shapes[k])
                if self.program.in_kinds[k] == PROGRAM_IN_IDS:
                    upload_cast_i32(self.session.ex, buf, srcs[k], "distributed program ids")
                else:
                    upload_cast(self.session.ex, buf, srcs[k])
            self.target_cols = max(1, target.cols)
            self.target = self.session.ex.alloc(max(1, target.count()))
            self.dataset_storage.append(self.target)
            self.batch_target = self.session.ex.alloc(max(1, batch * self.target_cols))
            self.dataset_storage.append(self.batch_target)
            self.indices = self.session.ex.alloc(rows)
            self.dataset_storage.append(self.indices)
            if loss_kind != 0:
                upload_cast(self.session.ex, self.target, target)
            self.dataset_ready = True
        except error:
            self.release_dataset()
            raise error

    def gradient(mut self, start: Int, count: Int, loss_kind: Int, delta: Float32,
                 stream: Int) raises:
        self.session._drop_temps(self.temps)
        self.temps.clear()
        self.session.tape.reset()
        self.session.tape.record()
        var shapes = self.shapes.copy()
        for k in range(len(self.full)):  # driver-loop: input arrays
            var width = shapes[k].count() // shapes[k].d[0]
            gather_rows(self.session.ex, self.full[k], self.indices, start, count, width, self.batches[k])
            shapes[k].d[0] = count
        if loss_kind != 0:
            gather_rows(self.session.ex, self.target, self.indices, start, count,
                        self.target_cols, self.batch_target)
        var phs = List[Int]()
        var outs = replay_values(self.session, self.program, self.batches, shapes, self.params,
                                 self.trainable, self.temps, phs, stream)
        var output = self.session.tensor(outs[0])
        var y = self.session.tape.variable(self.session.ex, self.batch_target, count, self.target_cols, False)
        var loss = program_loss(self.session.tape, self.session.ex, output, y, loss_kind, delta)
        if self.session.tape.requires_grad(loss):
            self.session.tape.backward_scalar_with_rules(self.session.ex, loss, ImportRules())
        for j in range(len(self.params)):  # driver-loop: parameter gradients
            if self.trainable[j]:
                var parameter = self.session.tensor(phs[j])
                if self.session.tape._nodes[parameter._index].grad_ready:
                    self.gradients[j] = self.session.tape.gradient(parameter)
                else:
                    # A selected parameter behind detach/copy has the zero
                    # derivative. Keep the existing global optimizer clock:
                    # momentum/decay still use this explicit zero gradient.
                    var n = self.program.param_shapes[j].count()
                    var zero = self.session.tape.reserve(self.session.ex, n)
                    var fill = Args()
                    fill.p0 = zero
                    fill.f0 = Float32(0)
                    self.session.ex.launch[OP_FILL](fill, n)
                    self.gradients[j] = zero
        var word = List[Float32](length=1, fill=Float32(0))
        self.session.ex.download(FP(unsafe_from_address=Int(word.unsafe_ptr())),
                                 self.session.tape.value(loss), 1)
        self.session.ex.sync()
        self.loss = word[0]

    def clear_totals(mut self) raises:
        for j in range(len(self.params)):  # driver-loop: owned gradient ranges
            if self.trainable[j] and self.width[j] > 0:
                var a = Args()
                a.p0 = self.totals[j]
                a.f0 = Float32(0)
                self.session.ex.launch[OP_FILL](a, self.width[j])

    def update(mut self, cfg: OptimizerConfig, step: Int, lr: Float32) raises:
        for j in range(len(self.params)):  # driver-loop: disjoint optimizer ranges
            if self.trainable[j] and self.width[j] > 0:
                step_lr(self.session.ex, cfg, self.states[j], self.params[j] + self.first[j],
                        self.totals[j], self.width[j], step, lr)
        self.session.ex.sync()

    def close(mut self) raises:
        self.release_dataset()
        self.session._drop_temps(self.temps)
        self.temps.clear()
        for j in range(len(self.states)):  # driver-loop: optimizer range owners
            if self.trainable[j]:
                self.states[j].release(self.session.ex)
            self.session.ex.free(self.totals[j])
            self.session.ex.free(self.incoming[j])
        self.states.clear()
        self.session.close()


struct DistributedProgram[E: Exec & Movable & Deinitable](Movable):
    var workers: List[ProgramReplica[Self.E]]
    var program: OpsProgram
    var trainable: List[Bool]
    var devices: List[Int]
    var logical_shards: Int
    var cfg: OptimizerConfig
    var optimizer_ints: List[Int]
    var optimizer_floats: List[Float32]
    var epoch: Int
    var step: Int
    var seed: UInt64
    var busy: Bool
    var usable: Bool

    def __init__(out self, p: OpsProgram, trainable: List[Bool], devices: List[Int],
                 logical_shards: Int, cfg: OptimizerConfig, oi: List[Int], of: List[Float32],
                 seed: UInt64) raises:
        if logical_shards < 1 or len(devices) < 1 or len(devices) > logical_shards:
            raise Error("distributed program requires 1 <= physical devices <= logical_shards")
        if len(p.params) < 1 or len(p.outs) != 1 or p.n_in < 1:
            raise Error("distributed training requires inputs, parameters and one output")
        for i in range(len(devices)):  # small-loop(devices: physical device IDs): admission
            if devices[i] < 0:
                raise Error("distributed program: negative device ID")
            for j in range(i):  # small-loop(i: preceding device IDs): distinct ownership
                if devices[i] == devices[j]:
                    raise Error("distributed program: duplicate physical device ID")
        self.workers = List[ProgramReplica[Self.E]]()
        self.program = p.copy()
        self.trainable = trainable.copy()
        self.devices = devices.copy()
        self.logical_shards = logical_shards
        self.cfg = cfg
        self.optimizer_ints = oi.copy()
        self.optimizer_floats = of.copy()
        self.epoch = 0
        self.step = 0
        self.seed = seed
        self.busy = False
        self.usable = True

    def ready(self) raises:
        if self.busy or not self.usable or len(self.workers) != len(self.devices):
            raise Error("distributed program is busy, closed or failed; restore a checkpoint")

    def fold(mut self, source: Int, weight: Float32) raises:
        # Reduction always visits logical shard 0,1,... . Physical owners
        # operate on disjoint element ranges and never reduce each other.
        # Host staging is a byte copy; arithmetic stays on the owning Exec.
        var wp = rebind[MutPointer[ProgramReplica[Self.E], MutUntrackedOrigin]](self.workers.unsafe_ptr())
        for j in range(len(self.program.params)):  # driver-loop: parameters
            if not self.trainable[j]:
                continue
            for rank in range(len(self.workers)):  # driver-loop: optimizer slice owners
                var n = wp[rank].width[j]
                if n == 0:
                    continue
                var bytes = List[Float32](length=n, fill=Float32(0))
                var ptr = FP(unsafe_from_address=Int(bytes.unsafe_ptr()))
                wp[source].session.ex.download(ptr, wp[source].gradients[j] + wp[rank].first[j], n)
                wp[rank].session.ex.upload(wp[rank].incoming[j], ptr, n)
                # FP erases the List origin. Keep its owner alive until both
                # transfers finish; Mojo otherwise destroys it at unsafe_ptr.
                _ = bytes^
                var a = Args()
                a.p0 = wp[rank].incoming[j]
                a.f0 = weight
                wp[rank].session.ex.launch[OP_SCALE](a, n)
                a.p1 = wp[rank].totals[j]
                wp[rank].session.ex.launch[OP_AG_ACCUMULATE](a, n)

    def broadcast(mut self) raises:
        var wp = rebind[MutPointer[ProgramReplica[Self.E], MutUntrackedOrigin]](self.workers.unsafe_ptr())
        for j in range(len(self.program.params)):  # driver-loop: parameters
            if not self.trainable[j]:
                continue
            for owner in range(len(self.workers)):  # driver-loop: authoritative parameter ranges
                var n = wp[owner].width[j]
                if n == 0:
                    continue
                var lo = wp[owner].first[j]
                var bytes = List[Float32](length=n, fill=Float32(0))
                var ptr = FP(unsafe_from_address=Int(bytes.unsafe_ptr()))
                wp[owner].session.ex.download(ptr, wp[owner].params[j] + lo, n)
                for rank in range(len(self.workers)):  # driver-loop: parameter replicas
                    if rank != owner:
                        wp[rank].session.ex.upload(wp[rank].params[j] + lo, ptr, n)
                _ = bytes^
        for rank in range(len(self.workers)):  # driver-loop: transfer completion
            self.workers[rank].session.ex.sync()

    def fit(mut self, srcs: List[CastIn], shapes: List[ShapeN], target: CastIn,
            loss_kind: Int, delta: Float32, epochs: Int, batch: Int, shuffle: Bool,
            curve: FP) raises:
        self.ready()
        if epochs < 1 or batch < 1 or len(srcs) != self.program.n_in or len(shapes) != len(srcs):
            raise Error("distributed fit: invalid epochs, batch size or input count")
        var n = shapes[0].d[0]
        if n < 1 or n >= DATA_MAX_ROWS:
            raise Error("distributed fit: row count must be in [1, 2^24)")
        for k in range(len(srcs)):  # driver-loop: shape contracts
            if shapes[k].ndim < 2 or shapes[k].d[0] != n or shapes[k].ndim != self.program.in_shapes[k].ndim:
                raise Error("distributed fit: inputs must retain their captured ranks and share a sample count")
        if loss_kind < 0 or loss_kind > 5 or (loss_kind != 0 and target.rows != n):
            raise Error("distributed fit: invalid loss or target sample count")
        var bs = min(n, batch)
        self.busy = True
        try:
            for rank in range(len(self.workers)):  # driver-loop: input residency
                self.workers[rank].prepare(srcs, shapes, target, bs, loss_kind)
            var wp = rebind[MutPointer[ProgramReplica[Self.E], MutUntrackedOrigin]](self.workers.unsafe_ptr())
            for e in range(epochs):  # driver-loop: epochs
                for rank in range(len(self.workers)):  # driver-loop: native row permutations
                    epoch_order(wp[rank].session.ex, shuffle, self.seed, self.epoch, n, wp[rank].indices)
                var epoch_loss = Float32(0)
                var start = 0
                while start < n:  # driver-loop: optimizer minibatches
                    var count = min(bs, n - start)
                    var shards = min(self.logical_shards, count)
                    for rank in range(len(self.workers)):  # driver-loop: reset disjoint accumulators
                        wp[rank].clear_totals()
                    var base = 0
                    while base < shards:  # driver-loop: fixed logical shard waves
                        var active = min(len(self.workers), shards - base)
                        var failed = List[String](length=active, fill=String())
                        var fp = rebind[MutPointer[String, MutUntrackedOrigin]](failed.unsafe_ptr())
                        var step = self.step
                        var logical = self.logical_shards
                        def task(rank: Int) {imm wp, imm fp, imm base, imm start, imm count,
                                             imm shards, imm loss_kind, imm delta, imm step, imm logical}:
                            try:
                                var shard = base + rank
                                var lo = count * shard // shards
                                var hi = count * (shard + 1) // shards
                                # Stream coordinates are logical, not physical.
                                wp[rank].gradient(start + lo, hi - lo, loss_kind, delta, step * logical + shard)
                                # One native executor per rank; join queued work before folding.
                                wp[rank].session.ex.sync()
                            except err:
                                fp[rank] = String(err)
                        if active == 1:
                            task(0)
                        else:
                            host_parallelize(task, active)
                        for rank in range(active):  # driver-loop: propagate every failed shard
                            if failed[rank].byte_length() > 0:
                                raise Error("distributed logical shard " + String(base + rank) + ": " + failed[rank])
                        for rank in range(active):  # driver-loop: immutable logical left fold
                            var shard = base + rank
                            var rows = count * (shard + 1) // shards - count * shard // shards
                            var weight = ftz(identical_div(Float32(rows), Float32(count)))
                            self.fold(rank, weight)
                            epoch_loss = ftz(identical_mul_add(Float32(rows), wp[rank].loss, epoch_loss))
                        base += active
                    var lr = epoch_lr(self.cfg.lr, self.optimizer_ints[2], self.optimizer_ints[3],
                                      self.optimizer_floats[6], self.epoch)
                    for rank in range(len(self.workers)):  # driver-loop: sharded optimizer updates
                        wp[rank].update(self.cfg, self.step + 1, lr)
                    self.broadcast()
                    self.step += 1
                    start += count
                curve[e] = ftz(identical_div(epoch_loss, Float32(n)))
                self.epoch += 1
            for rank in range(len(self.workers)):  # driver-loop: free call-scoped data
                wp[rank].release_dataset()
        except err:
            # A failed update may have touched one owner. No later use or save
            # may expose a mixed step: restore the last complete checkpoint.
            self.usable = False
            self.busy = False
            for rank in range(len(self.workers)):  # driver-loop: failed-call dataset owners
                self.workers[rank].release_dataset()
            raise err
        self.busy = False

    def close(mut self) raises:
        if self.busy:
            raise Error("distributed program is busy")
        self.usable = False
        for rank in range(len(self.workers)):  # driver-loop: native worker lifetimes
            self.workers[rank].close()
        self.workers.clear()
