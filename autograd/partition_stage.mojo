# SPDX-License-Identifier: Apache-2.0
# One physical worker of a partitioned captured graph. Only parameters owned
# by this worker have persistent device storage; other inputs are transient.
# Node-local recomputation supplies exact native VJPs without a Python tape.
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, OP_SCALE, OP_FILL, dummy_ptr
from autograd.ops import OP_AG_ACCUMULATE
from autograd.ops_api import OpsSession, OPS_DROPOUT
from autograd.program import OpsProgram
from autograd.partition_plan import GraphBytes, PartitionPlan
from autograd.optim import OptimizerConfig, OptimizerState, step_lr
from autograd.import_api import ImportRules
from autograd.program_runtime import program_loss
from autograd.rankn import ShapeN


struct NodeGradients(Copyable, Movable):
    var refs: List[Int]
    var values: List[GraphBytes]

    def __init__(out self):
        self.refs = List[Int]()
        self.values = List[GraphBytes]()


struct ModelStage[E: Exec & Movable & Deinitable](Movable):
    var session: OpsSession[Self.E]
    var params: List[FP]
    var owns: List[Bool]
    var states: List[OptimizerState]
    var trainable: List[Bool]
    var temps: List[Int]
    var scratch: List[FP]
    var output: GraphBytes
    var gradients: NodeGradients

    def __init__(out self, var ex: Self.E, plan: PartitionPlan, cfg: OptimizerConfig,
                 rank: Int, world: Int) raises:
        self.session = OpsSession[Self.E](ex^, False)
        self.params = List[FP]()
        self.owns = List[Bool]()
        self.states = List[OptimizerState]()
        self.trainable = plan.trainable.copy()
        self.temps = List[Int]()
        self.scratch = List[FP]()
        self.output = GraphBytes()
        self.gradients = NodeGradients()
        for j in range(len(plan.parameters)):  # driver-loop: model/optimizer residency
            var own = plan.owners[j] % world == rank
            self.owns.append(own)
            if own:
                var n = plan.parameters[j].shape.count()
                var buf = self.session.ex.alloc(n)
                self.session.ex.upload(buf, FP(unsafe_from_address=Int(plan.parameters[j].words.unsafe_ptr())), n)
                self.params.append(buf)
                if plan.trainable[j]:
                    self.states.append(OptimizerState(self.session.ex, cfg, n))
                else:
                    self.states.append(OptimizerState())
            else:
                self.params.append(dummy_ptr())
                self.states.append(OptimizerState())

    def clear(mut self) raises:
        self.session._drop_temps(self.temps)
        self.temps.clear()
        self.session.tape.reset()
        self.session.oplog.clear()
        for j in range(len(self.scratch)):  # driver-loop: transient input owners
            self.session.ex.free(self.scratch[j])
        self.scratch.clear()

    def leaf(mut self, a: GraphBytes, rf: Int, recording: Bool) raises -> Int:
        var buf = dummy_ptr()
        if rf < 0 and self.owns[-rf - 1]:
            buf = self.params[-rf - 1]
        else:
            buf = self.session.ex.alloc(a.shape.count())
            self.scratch.append(buf)
            self.session.ex.upload(buf, FP(unsafe_from_address=Int(a.words.unsafe_ptr())), a.shape.count())
        var grad = recording and not a.integer
        var t = self.session.tape.variable_nd(self.session.ex, buf, a.shape, grad)
        var h = self.session._add(t, 0, grad, 0)
        self.session.int_words[self.session._slot(h)] = a.integer
        self.temps.append(h)
        return h

    def read(mut self, h: Int) raises -> GraphBytes:
        var sh = self.session.shapes[self.session._slot(h)]
        var data = List[Float32](length=sh.count(), fill=Float32(0))
        self.session.download(h, FP(unsafe_from_address=Int(data.unsafe_ptr())))
        return GraphBytes(sh, data^, self.session.int_words[self.session._slot(h)])

    def evaluate(mut self, p: OpsProgram, node: Int, arguments: List[GraphBytes],
                 seed: GraphBytes, backward: Bool, stream: Int) raises:
        self.clear()
        if backward:
            self.session.tape.record()
        else:
            self.session.tape.no_grad()
        var hs = List[Int]()
        var refs = List[Int]()
        var unique = List[Bool]()
        for k in range(len(arguments)):  # driver-loop: operation arguments
            var rf = p.args[p.a_at[node] + k]
            var earlier = -1
            for j in range(k):  # driver-loop: aliased operation operands
                if refs[j] == rf:
                    earlier = j
                    break
            refs.append(rf)
            unique.append(earlier < 0)
            hs.append(hs[earlier] if earlier >= 0 else self.leaf(arguments[k], rf, backward))
        var ints = List[Int]()
        var fs = List[Float32]()
        for k in range(p.i_at[node], p.i_at[node + 1]):  # driver-loop: operation metadata
            ints.append(p.ints[k])
        for k in range(p.f_at[node], p.f_at[node + 1]):  # driver-loop: operation metadata
            fs.append(p.fs[k])
        if stream >= 0 and p.code[node] == OPS_DROPOUT:
            ints[len(ints) - 1] += stream * p.n_nodes() + node
        var h = self.session._op_run(p.code[node], hs, ints, fs)
        self.temps.append(h)
        if not backward:
            self.output = self.read(h)
            self.session.ex.sync()
            return
        var out = self.session.tensor(h)
        self.gradients = NodeGradients()
        if not self.session.tape.requires_grad(out):
            # A detached node stops its incoming adjoint. Other graph paths
            # retain their own contributions in the coordinator's fold.
            self.session.ex.sync()
            return
        if seed.shape.count() != self.session.shapes[self.session._slot(h)].count():
            raise Error("partitioned backward: output seed shape mismatch")
        var seed_buf = self.session.ex.alloc(seed.shape.count())
        self.scratch.append(seed_buf)
        self.session.ex.upload(seed_buf, FP(unsafe_from_address=Int(seed.words.unsafe_ptr())), seed.shape.count())
        var seed_tensor = self.session.tape.variable_nd(self.session.ex, seed_buf, seed.shape, False)
        self.session.tape.backward_with_rules(self.session.ex, out, seed_tensor, ImportRules(), False)
        self.gradients = NodeGradients()
        for k in range(len(hs)):  # driver-loop: one VJP per distinct operand
            if not unique[k] or arguments[k].integer:
                continue
            var t = self.session.tensor(hs[k])
            # Some operands (indices/conditions) deliberately have no VJP.
            # Inspect readiness rather than swallowing derivative failures.
            if not self.session.tape._nodes[t._index].grad_ready:
                continue
            var sh = arguments[k].shape
            var data = List[Float32](length=sh.count(), fill=Float32(0))
            self.session.ex.download(FP(unsafe_from_address=Int(data.unsafe_ptr())),
                                     self.session.tape.gradient(t), sh.count())
            self.gradients.refs.append(refs[k])
            self.gradients.values.append(GraphBytes(sh, data^))
        self.session.ex.sync()

    def accumulate(mut self, mut total: GraphBytes, value: GraphBytes) raises:
        if len(total.words) == 0:
            total = value.copy()
            return
        if total.shape.count() != value.shape.count():
            raise Error("partitioned gradient accumulation: shape mismatch")
        var n = value.shape.count()
        var a = self.session.ex.alloc(n)
        var b = self.session.ex.alloc(n)
        self.session.ex.upload(a, FP(unsafe_from_address=Int(value.words.unsafe_ptr())), n)
        self.session.ex.upload(b, FP(unsafe_from_address=Int(total.words.unsafe_ptr())), n)
        var args = Args()
        args.p0 = a
        args.p1 = b
        self.session.ex.launch[OP_AG_ACCUMULATE](args, n)
        self.session.ex.download(FP(unsafe_from_address=Int(total.words.unsafe_ptr())), b, n)
        self.session.ex.free(a)
        self.session.ex.free(b)

    def loss_seed(mut self, output: GraphBytes, target: GraphBytes, kind: Int,
                  delta: Float32, weight: Float32) raises -> Tuple[Float32, GraphBytes]:
        self.clear()
        self.session.tape.record()
        var h = self.leaf(output, 0, True)
        var y = self.leaf(target, 1, False)
        var loss = program_loss(self.session.tape, self.session.ex, self.session.tensor(h),
                                self.session.tensor(y), kind, delta)
        self.session.tape.backward_scalar_with_rules(self.session.ex, loss, ImportRules())
        var lossword = List[Float32](length=1, fill=Float32(0))
        self.session.ex.download(FP(unsafe_from_address=Int(lossword.unsafe_ptr())), self.session.tape.value(loss), 1)
        var grad = self.session.tape.gradient(self.session.tensor(h))
        var args = Args()
        args.p0 = grad
        args.f0 = weight
        self.session.ex.launch[OP_SCALE](args, output.shape.count())
        var words = List[Float32](length=output.shape.count(), fill=Float32(0))
        self.session.ex.download(FP(unsafe_from_address=Int(words.unsafe_ptr())), grad, output.shape.count())
        return (lossword[0], GraphBytes(output.shape, words^))

    def update(mut self, mut parameters: List[GraphBytes], gradients: List[GraphBytes],
               cfg: OptimizerConfig, step: Int, lr: Float32) raises:
        self.clear()
        for j in range(len(self.params)):  # driver-loop: authoritative parameter owners
            if not self.owns[j] or not self.trainable[j]:
                continue
            var n = parameters[j].shape.count()
            if len(gradients[j].words) != 0 and len(gradients[j].words) != n:
                raise Error("partitioned update: parameter gradient shape mismatch")
            var grad = self.session.ex.alloc(n)
            try:
                if len(gradients[j].words) == 0:
                    # No differentiable path is an explicit zero gradient;
                    # selected optimizer states keep their global step clock.
                    var fill = Args()
                    fill.p0 = grad
                    fill.f0 = Float32(0)
                    self.session.ex.launch[OP_FILL](fill, n)
                else:
                    self.session.ex.upload(grad, FP(unsafe_from_address=Int(gradients[j].words.unsafe_ptr())), n)
                step_lr(self.session.ex, cfg, self.states[j], self.params[j], grad, n, step, lr)
                self.session.ex.download(FP(unsafe_from_address=Int(parameters[j].words.unsafe_ptr())), self.params[j], n)
            except error:
                self.session.ex.free(grad)
                raise error
            self.session.ex.free(grad)
        self.session.ex.sync()

    def close(mut self) raises:
        self.clear()
        for j in range(len(self.params)):  # driver-loop: physical parameter/state owners
            if self.owns[j]:
                self.session.ex.free(self.params[j])
                if self.trainable[j]:
                    self.states[j].release(self.session.ex)
        self.session.close()
