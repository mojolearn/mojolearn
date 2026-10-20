# SPDX-License-Identifier: Apache-2.0
"""THE OPERATION SESSION: the tape's forward operations, one call each, for
composing new inference and hybrid algorithms (lane ag-ops-api, 2026-10-19).

An `OpsSession` owns one executor and one `Tape` and a table of live
tensors by integer handle. Every operation is a tape method (or one of the
tape's losses, layers, data steps or metrics) on the session's executor, so
a composition built here runs the kernels the trainers run and gives the
same bits on Apple, NVIDIA, AMD and the host. In inference mode (the
default) the tape keeps no graph; with recording on, the same composition is
differentiable: `backward` then `gradient_into` per leaf.

Handles: `(generation << 24) | (slot + 1)`, so a freed handle never names
the tensor that later takes its slot. Storage: a computed tensor's values
live in tape storage until `reset` or `close`; an uploaded tensor owns its
buffer, which `reset` keeps (re-registered as a fresh leaf of the next
graph) and `free` returns at the next `reset` or `close` (a recorded graph
may still read it before then).

Sessions live in a process-wide table (`std.ffi._Global`, one per numeric
tier, as `sequence/seq_tensor.mojo`); each entry records its executor kind,
so a host-binding call never reaches a device session. No new kernels: the
operations launch the tape's, the losses', the metrics' and the trainers'
existing opcodes (autograd/ops.mojo, autograd/loss_ops.mojo,
autograd/data_ops.mojo, sequence/ops.mojo OP_FILL / OP_COPY /
OP_GATHER_ROWS / OP_MLP_PERM).

Optimizers (lane ag-session-optim, 2026-10-19): `optimizer` makes an
optimizer of the session (SGD with momentum and Nesterov, Adam, AdamW, with
weight decay and Sequential.compile's learning-rate schedules); `step`
updates trainable uploaded tensors IN PLACE on the executor from the last
backward's gradients, one OP_OPT launch per parameter (autograd/optim.mojo
step_lr, the element body and the host scalars every sequence trainer
runs), with the moment buffers and the step count held in the session, per
(optimizer, parameter), allocated at the parameter's first step. The update
is elementwise and its host scalars depend on the step count and the
learning rate alone, so a parameter stepped here gets the words
Sequential.fit's one flat launch gives it (sequence/recurrent.mojo
opt_step: the same OP_OPT arguments, opt_scalars word for word).

Higher order (lane ag-ho-session, docs/plans/AUTOGRAD_HIGHER_ORDER.md
section 5): `backward_graph` records the backward pass itself
(autograd/higher.mojo backward_scalar_recorded), so `grad_handle` returns a
gradient as a session tensor that later operations and a second backward
can use; `backward_seed`, `vjp`, `hvp` and `jvp` are built from it with tape
operations only (no new kernels)."""
from std.ffi import _Global
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, OP_FILL, OP_COPY, OPT_SGD, OPT_ADAM, OPT_ADAMW
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from autograd.tape import Tape, Tensor
from autograd.ops import RELU, SIGMOID, TANH, SOFTMAX
from autograd.layers import NeuralRules, layer_norm
from autograd.higher import backward_recorded, backward_scalar_recorded
from autograd.losses import mse, mae, huber, binary_cross_entropy_with_logits, cross_entropy
from autograd.data import permutation, gather_rows
from autograd.metrics import accuracy, accuracy_from_scores, log_loss, roc_auc
from core.cast_in import CastIn, cast_in_kind
from autograd.cast_ops import upload_cast, upload_cast_i32
from autograd.optim import OptimizerConfig, OptimizerState, optimizer_config_packed, optimizer_slots, step_lr
from autograd.model import SCHED_CONSTANT, SCHED_STEP, SCHED_EXPONENTIAL, epoch_lr

# Executor kinds of a session (the binding that opened it).
# Lane ag-pass6-host (2026-10-19, PASS6_IDEAS idea G): the packed argument
# block of one operation call (OpsSession.op_packed), int64 words written by
# the Python shell into a session-owned buffer and read here by address, so
# a call crosses no Python list: word 0 the handle count, 1 the integer
# count, 2 the float count, then OPS_PACK_HANDLES handle words, then
# OPS_PACK_INTS integer words, then OPS_PACK_FLOATS float32 values in the
# words from OPS_PACK_F_WORD on (two a word, little-endian order as numpy's
# float32 view of the same buffer).
comptime OPS_PACK_HANDLES = 4
comptime OPS_PACK_INTS = 8
comptime OPS_PACK_FLOATS = 4
comptime OPS_PACK_H_WORD = 3
comptime OPS_PACK_I_WORD = OPS_PACK_H_WORD + OPS_PACK_HANDLES
comptime OPS_PACK_F_WORD = OPS_PACK_I_WORD + OPS_PACK_INTS
comptime OPS_PACK_WORDS = OPS_PACK_F_WORD + (OPS_PACK_FLOATS + 1) // 2

comptime OPS_DEVICE = 1
comptime OPS_HOST = 2

# Operation codes of ops_op (python/mojolearn/ops.py mirrors them).
comptime OPS_MATMUL = 1
comptime OPS_ADD = 2
comptime OPS_SUB = 3
comptime OPS_MUL = 4
comptime OPS_DIV = 5
comptime OPS_RELU = 6
comptime OPS_SIGMOID = 7
comptime OPS_TANH = 8
comptime OPS_EXP = 9
comptime OPS_LOG = 10
comptime OPS_SQRT = 11
comptime OPS_SCALE = 12
comptime OPS_TRANSPOSE = 13
comptime OPS_RESHAPE = 14
comptime OPS_SUM = 15
comptime OPS_MEAN = 16
comptime OPS_SOFTMAX = 17
comptime OPS_LOG_SOFTMAX = 18
comptime OPS_LINEAR = 19
comptime OPS_LAYER_NORM = 20
comptime OPS_DROPOUT = 21
comptime OPS_GATHER_ROWS = 22
comptime OPS_PERMUTATION = 23
comptime OPS_MSE = 24
comptime OPS_MAE = 25
comptime OPS_HUBER = 26
comptime OPS_BCE_LOGITS = 27
comptime OPS_CROSS_ENTROPY = 28
comptime OPS_DETACH = 29
comptime OPS_FULL = 30
comptime OPS_COPY = 31
comptime OPS_EMBEDDING = 32

# Metric codes of ops_metric.
comptime OPS_METRIC_ACCURACY = 1
comptime OPS_METRIC_LOG_LOSS = 2
comptime OPS_METRIC_ROC_AUC = 3

# The packed optimizer options of `optimizer`.
comptime OPS_OPT_INTS = 4    # kind, nesterov, schedule, step size
comptime OPS_OPT_FLOATS = 7  # lr, momentum, beta1, beta2, eps, weight_decay, gamma

comptime _SLOT_BITS = 24
comptime _SLOT_MASK = (1 << _SLOT_BITS) - 1


def _activation(code: Int) raises -> Int:
    """The tape activation of a linear op's code: 0 none, 1 relu, 2 sigmoid,
    3 tanh, 4 softmax."""
    if code == 0:
        return 0
    if code == 1:
        return RELU
    if code == 2:
        return SIGMOID
    if code == 3:
        return TANH
    if code == 4:
        return SOFTMAX
    raise Error("ops: linear activation must be 0 (none), 1 (relu), 2 (sigmoid), 3 (tanh) or 4 (softmax)")


def _need(what: String, have: Int, inputs: Int, ints: Int, have_ints: Int, floats: Int, have_floats: Int) raises:
    if have != inputs or have_ints < ints or have_floats < floats:
        raise Error("ops: " + what + " takes " + String(inputs) + " tensors, " + String(ints)
                    + " integers and " + String(floats) + " floats")


def _seed(lo: Int, hi: Int) raises -> UInt64:
    if lo < 0 or hi < 0 or lo > 0xFFFFFFFF or hi > 0xFFFFFFFF:
        raise Error("ops: the seed words must be 32-bit non-negative values")
    return (UInt64(hi) << 32) | UInt64(lo)


@fieldwise_init
struct _OpsOptimizer(ImplicitlyCopyable, Movable):
    """One optimizer of a session: its OP_OPT configuration (packed as
    Sequential.compile packs it), its learning-rate schedule
    (autograd/model.mojo SCHED_*) and the schedule's epoch."""
    var cfg: OptimizerConfig
    var sched: Int
    var sched_step: Int
    var gamma: Float32
    var epoch: Int


@fieldwise_init
struct _OpsOptState(ImplicitlyCopyable, Movable):
    """The state of one (optimizer, parameter) pair: the executor buffers
    of an autograd/optim.mojo OptimizerState (owned by the session), the
    parameter's step count and the running host scalars."""
    var opt: Int
    var param: Int
    var s1: FP
    var s2: FP
    var s3: FP
    var parts: FP
    var cell: FP
    var n: Int
    var block: Int
    var nparts: Int
    var t: Int
    var pw1: Float32
    var pw2: Float32
    var mu_prod: Float32


struct OpsSession[E: Exec & Movable & Deinitable](Movable):
    """One executor, one tape and the live tensors of a composition."""
    var ex: Self.E
    var tape: Tape
    var tensors: List[Tensor]
    var live: List[Bool]
    var generation: List[Int]
    var rows: List[Int]
    var cols: List[Int]
    #: address of a session-owned buffer (an uploaded tensor), 0 otherwise
    var owned: List[Int]
    var wants_grad: List[Bool]
    #: a permutation output: the row count its indices address (else 0)
    var index_rows: List[Int]
    var free_slots: List[Int]
    #: owned buffers of freed handles, returned at the next reset or close
    var retired: List[Int]
    #: the session's optimizers (id = index + 1) and their per-parameter state
    var opts: List[_OpsOptimizer]
    var opt_states: List[_OpsOptState]

    def __init__(out self, var ex: Self.E, record: Bool):
        self.ex = ex^
        self.tape = Tape()
        if not record:
            self.tape.no_grad()
        self.tensors = List[Tensor]()
        self.live = List[Bool]()
        self.generation = List[Int]()
        self.rows = List[Int]()
        self.cols = List[Int]()
        self.owned = List[Int]()
        self.wants_grad = List[Bool]()
        self.index_rows = List[Int]()
        self.free_slots = List[Int]()
        self.retired = List[Int]()
        self.opts = List[_OpsOptimizer]()
        self.opt_states = List[_OpsOptState]()

    # ------------------------------------------------------------ handles
    def _slot(self, h: Int) raises -> Int:
        var slot = (h & _SLOT_MASK) - 1
        var gen = h >> _SLOT_BITS
        if h <= 0 or slot < 0 or slot >= len(self.live) or not self.live[slot] or self.generation[slot] != gen:
            raise Error("ops: handle " + String(h) + " is not a live tensor of this session")
        return slot

    def _add(mut self, t: Tensor, owned: Int, wants_grad: Bool, index_rows: Int) raises -> Int:
        var r = self.tape.rows(t)
        var c = self.tape.cols(t)
        var slot: Int
        if len(self.free_slots) > 0:
            slot = self.free_slots.pop()
            self.tensors[slot] = t
            self.live[slot] = True
            self.rows[slot] = r
            self.cols[slot] = c
            self.owned[slot] = owned
            self.wants_grad[slot] = wants_grad
            self.index_rows[slot] = index_rows
        else:
            slot = len(self.live)
            if slot >= _SLOT_MASK:
                raise Error("ops: too many live tensors in one session")
            self.tensors.append(t)
            self.live.append(True)
            self.generation.append(0)
            self.rows.append(r)
            self.cols.append(c)
            self.owned.append(owned)
            self.wants_grad.append(wants_grad)
            self.index_rows.append(index_rows)
        return (self.generation[slot] << _SLOT_BITS) | (slot + 1)

    def _retire(mut self, slot: Int):
        if self.owned[slot] != 0:
            # a freed parameter's optimizer state goes with it (its buffers
            # are returned with the freed uploads, after the queued updates)
            self._drop_states((self.generation[slot] << _SLOT_BITS) | (slot + 1))
        self.live[slot] = False
        self.generation[slot] += 1
        if self.owned[slot] != 0:
            self.retired.append(self.owned[slot])
            self.owned[slot] = 0
        self.free_slots.append(slot)

    def tensor(self, h: Int) raises -> Tensor:
        return self.tensors[self._slot(h)]

    def shape(self, h: Int) raises -> Tuple[Int, Int]:
        var slot = self._slot(h)
        return (self.rows[slot], self.cols[slot])

    def recording(self) -> Bool:
        return self.tape.is_recording()

    # ------------------------------------------------------------ data in and out
    def upload(mut self, src: CastIn, requires_grad: Bool) raises -> Int:
        """A new (src.rows, src.cols) tensor holding the caller's array as
        float32 (core/cast_in.mojo: any accepted dtype, C or Fortran order,
        cast on the executor by autograd/cast_ops.mojo upload_cast); the
        caller's memory may change once this returns."""
        src.check("ops: tensor")
        var rows = src.rows
        var cols = src.cols
        if rows <= 0 or cols <= 0 or rows > ((1 << 60) // 4) // cols:
            raise Error("ops: tensor dimensions must be positive")
        if requires_grad and not self.tape.is_recording():
            raise Error("ops: requires_grad needs a recording session (Session(record=True))")
        var n = rows * cols
        var buf = self.ex.alloc(n)
        upload_cast(self.ex, buf, src)
        var t = self.tape.variable(self.ex, buf, rows, cols, requires_grad)
        return self._add(t, Int(buf), requires_grad, 0)

    def download(mut self, h: Int, dst: FP) raises:
        """The tensor's rows x cols floats (row-major) into the caller's memory."""
        var slot = self._slot(h)
        var src_t = self.tensors[slot]
        var src = self.tape.value_of(self.ex, src_t)
        var count = self.rows[slot] * self.cols[slot]
        self.ex.download(dst, src, count)

    def scalar(mut self, h: Int) raises -> Float32:
        """The one value of a (1, 1) tensor, read back to the caller."""
        var slot = self._slot(h)
        if self.rows[slot] * self.cols[slot] != 1:
            raise Error("ops: scalar() takes a tensor of one value")
        var src_t = self.tensors[slot]
        var src = self.tape.value_of(self.ex, src_t)
        var out = List[Float32](length=1, fill=Float32(0))
        self.ex.download(FP(unsafe_from_address=Int(out.unsafe_ptr())), src, 1)
        return out[0]

    def free(mut self, h: Int) raises:
        self._retire(self._slot(h))

    def reset(mut self) raises:
        """Drop every computed tensor and the recorded graph; uploaded tensors
        stay (each a fresh leaf of the next graph)."""
        self.tape.reset()
        for slot in range(len(self.live)):  # driver-loop: over session handles
            if not self.live[slot]:
                continue
            if self.owned[slot] == 0:
                self._retire(slot)
            else:
                var buf = FP(unsafe_from_address=self.owned[slot])
                var r = self.rows[slot]
                var c = self.cols[slot]
                var grad = self.wants_grad[slot] and self.tape.is_recording()
                var leaf = self.tape.variable(self.ex, buf, r, c, grad)
                self.tensors[slot] = leaf
        for i in range(len(self.retired)):  # driver-loop: over freed uploads
            var old = FP(unsafe_from_address=self.retired[i])
            self.ex.free(old)
        self.retired.clear()

    def close(mut self) raises:
        """Return every buffer to the executor and wait for its queue."""
        self.tape.release(self.ex)
        for slot in range(len(self.live)):  # driver-loop: over session handles
            if self.live[slot] and self.owned[slot] != 0:
                var buf = FP(unsafe_from_address=self.owned[slot])
                self.ex.free(buf)
            self.live[slot] = False
        for i in range(len(self.retired)):  # driver-loop: over freed uploads
            var old = FP(unsafe_from_address=self.retired[i])
            self.ex.free(old)
        self.retired.clear()
        for i in range(len(self.opt_states)):  # driver-loop: over optimizer states
            var r = self.opt_states[i]
            self.ex.free(r.s1)
            self.ex.free(r.s2)
            self.ex.free(r.s3)
            self.ex.free(r.parts)
            self.ex.free(r.cell)
        self.opt_states.clear()
        self.ex.sync()

    # ------------------------------------------------------------ gradients
    def backward(mut self, h: Int, accumulate: Bool) raises:
        """Differentiate the (1, 1) tensor `h` (a recording session)."""
        if not self.tape.is_recording():
            raise Error("ops: backward needs a recording session (Session(record=True))")
        var t = self.tensor(h)
        self.tape.backward_scalar_with_rules(self.ex, t, NeuralRules(), accumulate)

    def gradient_into(mut self, h: Int, dst: FP) raises:
        """The gradient of tensor `h` from the last backward into the caller's memory."""
        var slot = self._slot(h)
        var g = self.tape.gradient(self.tensors[slot])
        var count = self.rows[slot] * self.cols[slot]
        self.ex.download(dst, g, count)

    def zero_grad(mut self) raises:
        """Forget the gradient of every trainable tensor (no launch): the
        next backward with accumulate=True starts each from zero."""
        if not self.tape.is_recording():
            raise Error("ops: zero_grad needs a recording session (Session(record=True))")
        for slot in range(len(self.live)):  # driver-loop: over session handles
            if self.live[slot] and self.wants_grad[slot]:
                var leaf = self.tensors[slot]
                self.tape.clear_gradient(leaf)

    # ------------------------------------------------------------ higher order
    # Lane ag-ho-session (docs/plans/AUTOGRAD_HIGHER_ORDER.md section 5, H3):
    # the session surface of the recorded backward (autograd/higher.mojo,
    # lane H1). A recorded backward expresses every gradient through tape
    # operations, so a gradient is a session tensor that later operations
    # (and a second backward) can use. No new kernels here: the helpers call
    # backward_recorded / backward_scalar_recorded, Tape.grad_of, the tape's
    # multiply / sum / add, the first-order backward, and OP_FILL / OP_COPY
    # for the returned values. Host loops run over handle lists only.
    def _wrt(self, hs: List[Int], what: String) raises -> List[Tensor]:
        """The tensors of `hs`, each differentiable (an input of a VJP)."""
        if len(hs) < 1:
            raise Error("ops: " + what + " takes at least one input")
        var ts = List[Tensor](capacity=len(hs))
        for i in range(len(hs)):  # driver-loop: over the call's input handles
            var t = self.tensor(hs[i])
            if not self.tape.requires_grad(t):
                raise Error("ops: every input of " + what + " must require a gradient (requires_grad=True or computed from one)")
            ts.append(t)
        return ts^

    def _same_shapes(self, hs: List[Int], vs: List[Int], what: String) raises:
        if len(vs) != len(hs):
            raise Error("ops: " + what + " takes one direction per input")
        for i in range(len(hs)):  # driver-loop: over the call's input handles
            var a = self.shape(hs[i])
            var b = self.shape(vs[i])
            if a[0] != b[0] or a[1] != b[1]:
                raise Error("ops: each direction of " + what + " must have its input's shape")

    def _copy_constant(mut self, src: FP, rows: Int, cols: Int) raises -> Int:
        """A new constant session tensor holding a copy of `src`."""
        var dst = self.tape.reserve(self.ex, rows * cols)
        var a = Args()
        a.p0 = src
        a.p1 = dst
        self.ex.launch[OP_COPY](a, rows * cols)
        return self._constant(dst, rows, cols, 0)

    def _zeros(mut self, rows: Int, cols: Int) raises -> Int:
        """A new constant session tensor of zeros."""
        var dst = self.tape.reserve(self.ex, rows * cols)
        var a = Args()
        a.p0 = dst
        a.f0 = Float32(0)
        self.ex.launch[OP_FILL](a, rows * cols)
        return self._constant(dst, rows, cols, 0)

    def _directional(mut self, gs: List[Tensor], vs: List[Int]) raises -> Tensor:
        """sum_i sum(gs[i] * vs[i]) as one recorded (1, 1) tensor, the
        terms added in list order."""
        var g0 = gs[0]
        var v0 = self.tensor(vs[0])
        var p0 = self.tape.multiply(self.ex, g0, v0)
        var d = self.tape.sum(self.ex, p0, -1)
        for i in range(1, len(gs)):  # driver-loop: over the call's input handles
            var gi = gs[i]
            var vi = self.tensor(vs[i])
            var pi = self.tape.multiply(self.ex, gi, vi)
            var term = self.tape.sum(self.ex, pi, -1)
            d = self.tape.add(self.ex, d, term)
        return d

    def backward_graph(mut self, h: Int) raises:
        """Differentiate the (1, 1) tensor `h` and record the gradient
        computation itself, so each gradient is a tensor (grad_handle) that
        can be differentiated again. No accumulate form: add the previous
        gradient tensor with an operation instead."""
        if not self.tape.is_recording():
            raise Error("ops: backward needs a recording session (Session(record=True))")
        var t = self.tensor(h)
        backward_scalar_recorded(self.tape, self.ex, t)

    def backward_seed(mut self, h: Int, seed_h: Int, create_graph: Bool) raises:
        """Differentiate tensor `h` against the seed `seed_h` (a tensor of
        h's shape): the vector-Jacobian product of h for that seed, recorded
        when `create_graph` is set."""
        if not self.tape.is_recording():
            raise Error("ops: backward needs a recording session (Session(record=True))")
        var t = self.tensor(h)
        var seed = self.tensor(seed_h)
        var a = self.shape(h)
        var b = self.shape(seed_h)
        if a[0] != b[0] or a[1] != b[1]:
            raise Error("ops: the seed must have the output's shape")
        if create_graph:
            backward_recorded(self.tape, self.ex, t, seed)
        else:
            self.tape.backward_with_rules(self.ex, t, seed, NeuralRules(), False)

    def grad_handle(mut self, h: Int) raises -> Int:
        """The gradient of tensor `h` from the last recorded backward, as a
        new session handle (a computed tensor: it lives until reset)."""
        var t = self.tensor(h)
        var g = self.tape.grad_of(t)
        return self._add(g, 0, False, 0)

    def vjp(mut self, out_h: Int, hs: List[Int], seed_h: Int, create_graph: Bool) raises -> List[Int]:
        """The vector-Jacobian product of `out_h` for the seed `seed_h` with
        respect to each tensor of `hs`; one new handle per input. With
        `create_graph` the results are recorded (differentiable) tensors;
        otherwise they are constant copies of the gradients."""
        var xs = self._wrt(hs, "vjp")
        self.backward_seed(out_h, seed_h, create_graph)
        var res = List[Int](capacity=len(hs))
        for i in range(len(xs)):  # driver-loop: over the call's input handles
            var x = xs[i]
            if create_graph:
                var g = self.tape.grad_of(x)
                res.append(self._add(g, 0, False, 0))
            else:
                var s = self.shape(hs[i])
                var gbuf = self.tape.gradient(x)
                res.append(self._copy_constant(gbuf, s[0], s[1]))
        return res^

    def hvp(mut self, loss_h: Int, hs: List[Int], vs: List[Int]) raises -> List[Int]:
        """The Hessian-vector product of the (1, 1) tensor `loss_h` with
        respect to the tensors `hs` along the directions `vs` (one per
        input, of its shape); one new constant handle per input. A recorded
        backward gives the gradients g_i as tensors; d = sum_i sum(g_i * v_i)
        is recorded from them; a first-order backward of d gives H v. When
        no gradient depends on an input (zero curvature) the result is
        zeros."""
        if not self.tape.is_recording():
            raise Error("ops: hvp needs a recording session (Session(record=True))")
        var xs = self._wrt(hs, "hvp")
        self._same_shapes(hs, vs, "hvp")
        var t = self.tensor(loss_h)
        backward_scalar_recorded(self.tape, self.ex, t)
        var gs = List[Tensor](capacity=len(xs))
        for i in range(len(xs)):  # driver-loop: over the call's input handles
            var x = xs[i]
            gs.append(self.tape.grad_of(x))
        var d = self._directional(gs, vs)
        var res = List[Int](capacity=len(hs))
        if not self.tape.requires_grad(d):
            for i in range(len(hs)):  # driver-loop: over the call's input handles
                var s = self.shape(hs[i])
                res.append(self._zeros(s[0], s[1]))
            return res^
        self.tape.backward_scalar_with_rules(self.ex, d, NeuralRules(), False)
        for i in range(len(xs)):  # driver-loop: over the call's input handles
            var x = xs[i]
            var s = self.shape(hs[i])
            var gbuf = self.tape.gradient(x)
            res.append(self._copy_constant(gbuf, s[0], s[1]))
        return res^

    def jvp(mut self, out_h: Int, hs: List[Int], us: List[Int]) raises -> Int:
        """The Jacobian-vector product of tensor `out_h` along the
        directions `us` (one per tensor of `hs`, of its shape); a new
        constant handle of out_h's shape. By the double-VJP identity: with a
        seed w of out_h's shape (a differentiable leaf; its value does not
        matter, the VJP is linear in it) the recorded VJP gives J_i^T w as
        tensors, and the gradient of sum_i sum(J_i^T w * u_i) with respect
        to w is sum_i J_i u_i."""
        if not self.tape.is_recording():
            raise Error("ops: jvp needs a recording session (Session(record=True))")
        var xs = self._wrt(hs, "jvp")
        self._same_shapes(hs, us, "jvp")
        var t = self.tensor(out_h)
        var oshape = self.shape(out_h)
        var rows = oshape[0]
        var cols = oshape[1]
        var wbuf = self.tape.reserve(self.ex, rows * cols)
        var fa = Args()
        fa.p0 = wbuf
        fa.f0 = Float32(0)
        self.ex.launch[OP_FILL](fa, rows * cols)
        var w = self.tape.variable(self.ex, wbuf, rows, cols, True)
        backward_recorded(self.tape, self.ex, t, w)
        var gs = List[Tensor](capacity=len(xs))
        for i in range(len(xs)):  # driver-loop: over the call's input handles
            var x = xs[i]
            gs.append(self.tape.grad_of(x))
        var d = self._directional(gs, us)
        if not self.tape.requires_grad(d):
            return self._zeros(rows, cols)
        self.tape.backward_scalar_with_rules(self.ex, d, NeuralRules(), False)
        var jbuf = self.tape.gradient(w)
        return self._copy_constant(jbuf, rows, cols)

    # ------------------------------------------------------------ optimizers
    def optimizer(mut self, ints: List[Int], fs: List[Float32]) raises -> Int:
        """A new optimizer; returns its id (>= 1).

        ints = [kind (OPT_SGD, OPT_ADAM or OPT_ADAMW), nesterov, schedule
        (autograd/model.mojo SCHED_*), step size]; fs = [lr, momentum,
        beta1, beta2, eps, weight_decay, gamma]. The configuration is packed
        as Sequential.compile packs it (sgd: f1 momentum, f2 0; adam and
        adamw: f1 beta1, f2 beta2; f7 0), so the update's arguments are
        model_fit's words."""
        if len(ints) != OPS_OPT_INTS or len(fs) != OPS_OPT_FLOATS:
            raise Error("ops: optimizer takes 4 integers and 7 floats")
        var kind = ints[0]
        if kind != OPT_SGD and kind != OPT_ADAM and kind != OPT_ADAMW:
            raise Error("ops: the optimizer must be sgd, adam or adamw")
        var lr = fs[0]
        var momentum = fs[1]
        var beta1 = fs[2]
        var beta2 = fs[3]
        var eps = fs[4]
        var wd = fs[5]
        var gamma = fs[6]
        if not (lr > Float32(0)):
            raise Error("ops: the learning rate must be positive")
        if not (momentum >= Float32(0) and momentum < Float32(1)):
            raise Error("ops: momentum must be in [0, 1)")
        if not (beta1 >= Float32(0) and beta1 < Float32(1)) or not (beta2 >= Float32(0) and beta2 < Float32(1)):
            raise Error("ops: beta1 and beta2 must be in [0, 1)")
        if not (eps > Float32(0)):
            raise Error("ops: eps must be positive")
        if not (wd >= Float32(0)):
            raise Error("ops: weight_decay must be >= 0")
        var nesterov = ints[1] != 0
        if nesterov and (kind != OPT_SGD or not (momentum > Float32(0))):
            raise Error("ops: Nesterov momentum needs sgd with momentum > 0")
        var sched = ints[2]
        if sched < SCHED_CONSTANT or sched > SCHED_EXPONENTIAL:
            raise Error("ops: unknown learning-rate schedule code " + String(sched))
        if sched != SCHED_CONSTANT and not (gamma > Float32(0)):
            raise Error("ops: the schedule's gamma must be positive")
        if sched == SCHED_STEP and ints[3] < 1:
            raise Error("ops: the step schedule's step size must be at least 1")
        var f1 = momentum if kind == OPT_SGD else beta1
        var f2 = Float32(0) if kind == OPT_SGD else beta2
        var flags = 1 if nesterov else 0
        var cfg = optimizer_config_packed(kind, flags, f1, f2, eps, wd, Float32(0), lr)
        var sched_step = ints[3] if sched == SCHED_STEP else 1
        self.opts.append(_OpsOptimizer(cfg, sched, sched_step, gamma, 0))
        return len(self.opts)

    def _opt_index(self, o: Int) raises -> Int:
        if o < 1 or o > len(self.opts):
            raise Error("ops: " + String(o) + " is not an optimizer of this session")
        return o - 1

    def learning_rate(self, o: Int) raises -> Float32:
        """Optimizer `o`'s learning rate at its schedule's current epoch."""
        var op = self.opts[self._opt_index(o)]
        return epoch_lr(op.cfg.lr, op.sched, op.sched_step, op.gamma, op.epoch)

    def scheduler_step(mut self, o: Int) raises:
        """Advance optimizer `o`'s schedule by one epoch."""
        var oi = self._opt_index(o)
        self.opts[oi].epoch += 1

    def _param_slot(self, h: Int, what: String) raises -> Int:
        var slot = self._slot(h)
        if self.owned[slot] == 0 or not self.wants_grad[slot]:
            raise Error("ops: " + what + " takes tensors made with requires_grad=True")
        return slot

    def _state_of(mut self, oi: Int, h: Int, n: Int) raises -> Int:
        """The index of (optimizer oi, parameter h)'s state, allocated (zero
        moment buffers, step 0) on first use."""
        for i in range(len(self.opt_states)):  # driver-loop: over optimizer states
            if self.opt_states[i].opt == oi and self.opt_states[i].param == h:
                return i
        var cfg = self.opts[oi].cfg
        var st = OptimizerState(self.ex, cfg, n)
        self.opt_states.append(_OpsOptState(
            oi, h, st.s1, st.s2, st.s3, st.parts, st.cell, st.n, st.block, st.nparts, 0,
            st.pw1, st.pw2, st.mu_prod,
        ))
        return len(self.opt_states) - 1

    def _state_view(self, i: Int) -> OptimizerState:
        """An OptimizerState over state i's buffers and running scalars (it
        owns nothing; the session frees the buffers)."""
        var r = self.opt_states[i]
        var st = OptimizerState()
        st.s1 = r.s1
        st.s2 = r.s2
        st.s3 = r.s3
        st.parts = r.parts
        st.cell = r.cell
        st.n = r.n
        st.block = r.block
        st.nparts = r.nparts
        st.pw1 = r.pw1
        st.pw2 = r.pw2
        st.mu_prod = r.mu_prod
        return st^

    def _drop_states(mut self, h: Int):
        """Retire every optimizer state of parameter handle h."""
        var i = len(self.opt_states) - 1
        while i >= 0:  # driver-loop: over optimizer states
            if self.opt_states[i].param == h:
                var r = self.opt_states[i]
                self.retired.append(Int(r.s1))
                self.retired.append(Int(r.s2))
                self.retired.append(Int(r.s3))
                self.retired.append(Int(r.parts))
                self.retired.append(Int(r.cell))
                _ = self.opt_states.pop(i)
            i -= 1

    def step(mut self, o: Int, hs: List[Int]) raises:
        """One update of each parameter in `hs` by optimizer `o`, in place,
        from the last backward's gradients: per parameter ONE OP_OPT launch
        (autograd/optim.mojo step_lr; no clip, no read-back, no sync) at the
        schedule's current learning rate and the parameter's next step
        count. Every parameter is checked before any is updated."""
        var oi = self._opt_index(o)
        if not self.tape.is_recording():
            raise Error("ops: step needs a recording session (Session(record=True))")
        if len(hs) < 1:
            raise Error("ops: step takes at least one parameter")
        var slots = List[Int](capacity=len(hs))
        var grads = List[FP](capacity=len(hs))
        for i in range(len(hs)):  # driver-loop: over the step's parameters
            var slot = self._param_slot(hs[i], "step")
            for j in range(len(slots)):  # driver-loop: over the step's parameters
                if slots[j] == slot:
                    raise Error("ops: step lists a parameter twice")
            slots.append(slot)
            grads.append(self.tape.gradient(self.tensors[slot]))
        var cfg = self.opts[oi].cfg
        var lr = self.learning_rate(o)
        for i in range(len(slots)):  # driver-loop: over the step's parameters
            var slot = slots[i]
            var n = self.rows[slot] * self.cols[slot]
            var si = self._state_of(oi, hs[i], n)
            var t = self.opt_states[si].t + 1
            var st = self._state_view(si)
            step_lr(self.ex, cfg, st, FP(unsafe_from_address=self.owned[slot]), grads[i], n, t, lr)
            self.opt_states[si].t = t
            self.opt_states[si].pw1 = st.pw1
            self.opt_states[si].pw2 = st.pw2
            self.opt_states[si].mu_prod = st.mu_prod

    def _snapshot(mut self, src: FP, slot: Int) raises -> Int:
        """A copy of a state buffer shaped as parameter `slot`, as a new
        tensor (valid until reset)."""
        var r = self.rows[slot]
        var c = self.cols[slot]
        var dst = self.tape.reserve(self.ex, r * c)
        var a = Args()
        a.p0 = src
        a.p1 = dst
        self.ex.launch[OP_COPY](a, r * c)
        return self._constant(dst, r, c, 0)

    def optimizer_state(mut self, o: Int, h: Int) raises -> Tuple[Int, Float32, Float32, Float32, Int, Int, Int]:
        """(step count, beta1^t, beta2^t, NAdam's mu product, then one
        handle per state buffer: a copy of each buffer the kind uses, valid
        until reset, and 0 for one it does not use). A parameter never
        stepped reads step 0 and zero buffers."""
        var oi = self._opt_index(o)
        var slot = self._param_slot(h, "optimizer_state")
        var si = self._state_of(oi, h, self.rows[slot] * self.cols[slot])
        var r = self.opt_states[si]
        var used = optimizer_slots(self.opts[oi].cfg)
        var h1 = 0
        var h2 = 0
        var h3 = 0
        if used[0]:
            h1 = self._snapshot(r.s1, slot)
        if used[1]:
            h2 = self._snapshot(r.s2, slot)
        if used[2]:
            h3 = self._snapshot(r.s3, slot)
        return (r.t, r.pw1, r.pw2, r.mu_prod, h1, h2, h3)

    def load_optimizer_state(mut self, o: Int, h: Int, ints: List[Int], fs: List[Float32]) raises:
        """Set (optimizer o, parameter h)'s state: ints = [step count, then
        one tensor handle per state buffer (0 for a buffer the kind does
        not use)], fs = [beta1^t, beta2^t, NAdam's mu product], the words
        `optimizer_state` returns. Each buffer is copied on the executor."""
        if len(ints) != 4 or len(fs) != 3:
            raise Error("ops: load_optimizer_state takes 4 integers and 3 floats")
        if ints[0] < 0:
            raise Error("ops: the optimizer state's step count must be >= 0")
        var oi = self._opt_index(o)
        var slot = self._param_slot(h, "load_optimizer_state")
        var n = self.rows[slot] * self.cols[slot]
        var si = self._state_of(oi, h, n)
        var r = self.opt_states[si]
        var u = optimizer_slots(self.opts[oi].cfg)
        var used = List[Bool]()
        used.append(u[0])
        used.append(u[1])
        used.append(u[2])
        var dsts = List[FP]()
        dsts.append(r.s1)
        dsts.append(r.s2)
        dsts.append(r.s3)
        # every handle checked before any copy
        var srcs = List[FP]()
        for k in range(3):  # driver-loop: over the three state buffers
            var hk = ints[1 + k]
            if not used[k]:
                if hk != 0:
                    raise Error("ops: this optimizer keeps no state buffer " + String(k + 1))
                srcs.append(r.s1)
                continue
            if hk == 0:
                raise Error("ops: the optimizer state lacks state buffer " + String(k + 1))
            var ks = self._slot(hk)
            if self.rows[ks] != self.rows[slot] or self.cols[ks] != self.cols[slot]:
                raise Error("ops: an optimizer state buffer must have its parameter's shape")
            var src_t = self.tensors[ks]
            srcs.append(self.tape.value_of(self.ex, src_t))
        for k in range(3):  # driver-loop: over the three state buffers
            if used[k]:
                var a = Args()
                a.p0 = srcs[k]
                a.p1 = dsts[k]
                self.ex.launch[OP_COPY](a, n)
        self.opt_states[si].t = ints[0]
        self.opt_states[si].pw1 = fs[0]
        self.opt_states[si].pw2 = fs[1]
        self.opt_states[si].mu_prod = fs[2]

    # ------------------------------------------------------------ operations
    def _constant(mut self, values: FP, rows: Int, cols: Int, index_rows: Int) raises -> Int:
        var t = self.tape.variable(self.ex, values, rows, cols, False)
        return self._add(t, 0, False, index_rows)

    def op(mut self, code: Int, hs: List[Int], ints: List[Int], fs: List[Float32]) raises -> Int:
        """Run operation `code` on the tensors `hs` with integer and float
        arguments; returns the new tensor's handle."""
        var x = List[Tensor]()
        for i in range(len(hs)):  # small-loop(hs: an op's input handles, at most three): handle lookup, no data
            x.append(self.tensor(hs[i]))
        var n = len(x)
        var ni = len(ints)
        var nf = len(fs)
        var t: Tensor
        if code == OPS_MATMUL:
            _need("matmul", n, 2, 0, ni, 0, nf)
            t = self.tape.matmul(self.ex, x[0], x[1])
        elif code == OPS_ADD:
            _need("add", n, 2, 0, ni, 0, nf)
            t = self.tape.add(self.ex, x[0], x[1])
        elif code == OPS_SUB:
            _need("sub", n, 2, 0, ni, 0, nf)
            t = self.tape.subtract(self.ex, x[0], x[1])
        elif code == OPS_MUL:
            _need("mul", n, 2, 0, ni, 0, nf)
            t = self.tape.multiply(self.ex, x[0], x[1])
        elif code == OPS_DIV:
            _need("div", n, 2, 0, ni, 0, nf)
            t = self.tape.divide(self.ex, x[0], x[1])
        elif code == OPS_RELU:
            _need("relu", n, 1, 0, ni, 0, nf)
            t = self.tape.relu(self.ex, x[0])
        elif code == OPS_SIGMOID:
            _need("sigmoid", n, 1, 0, ni, 0, nf)
            t = self.tape.sigmoid(self.ex, x[0])
        elif code == OPS_TANH:
            _need("tanh", n, 1, 0, ni, 0, nf)
            t = self.tape.tanh(self.ex, x[0])
        elif code == OPS_EXP:
            _need("exp", n, 1, 0, ni, 0, nf)
            t = self.tape.exp(self.ex, x[0])
        elif code == OPS_LOG:
            _need("log", n, 1, 0, ni, 0, nf)
            t = self.tape.log(self.ex, x[0])
        elif code == OPS_SQRT:
            _need("sqrt", n, 1, 0, ni, 0, nf)
            t = self.tape.sqrt(self.ex, x[0])
        elif code == OPS_SCALE:
            _need("scale", n, 1, 0, ni, 1, nf)
            t = self.tape.scale(self.ex, x[0], fs[0])
        elif code == OPS_TRANSPOSE:
            _need("transpose", n, 1, 0, ni, 0, nf)
            t = self.tape.transpose(self.ex, x[0])
        elif code == OPS_RESHAPE:
            _need("reshape", n, 1, 2, ni, 0, nf)
            t = self.tape.reshape(self.ex, x[0], ints[0], ints[1])
        elif code == OPS_SUM:
            _need("sum", n, 1, 1, ni, 0, nf)
            t = self.tape.sum(self.ex, x[0], ints[0])
        elif code == OPS_MEAN:
            _need("mean", n, 1, 1, ni, 0, nf)
            t = self.tape.mean(self.ex, x[0], ints[0])
        elif code == OPS_SOFTMAX:
            _need("softmax", n, 1, 0, ni, 0, nf)
            t = self.tape.softmax(self.ex, x[0])
        elif code == OPS_LOG_SOFTMAX:
            _need("log_softmax", n, 1, 0, ni, 0, nf)
            t = self.tape.log_softmax(self.ex, x[0])
        elif code == OPS_LINEAR:
            if n != 2 and n != 3:
                raise Error("ops: linear takes an input, a weight and an optional bias")
            _need("linear", 2, 2, 1, ni, 0, nf)
            var act = _activation(ints[0])
            if n == 3:
                t = self.tape.linear(self.ex, x[0], x[1], Optional[Tensor](x[2]), act)
            else:
                t = self.tape.linear(self.ex, x[0], x[1], None, act)
        elif code == OPS_LAYER_NORM:
            _need("layer_norm", n, 3, 0, ni, 1, nf)
            t = layer_norm(self.tape, self.ex, x[0], x[1], x[2], fs[0])
        elif code == OPS_DROPOUT:
            _need("dropout", n, 1, 3, ni, 1, nf)
            t = self.tape.dropout(self.ex, x[0], fs[0], _seed(ints[0], ints[1]), ints[2])
        elif code == OPS_GATHER_ROWS:
            # The gather kernel (OP_GATHER_ROWS) reads X[idx] unchecked, so the
            # indices must be a permutation output: every value lies in
            # [0, its row count), checked here against X's rows, no download.
            _need("gather_rows", n, 2, 2, ni, 0, nf)
            var gx = self._slot(hs[0])
            var gj = self._slot(hs[1])
            var g_bound = self.index_rows[gj]
            if g_bound == 0 or self.cols[gj] != 1:
                raise Error("ops: gather_rows takes row indices made by permutation")
            if g_bound > self.rows[gx]:
                raise Error("ops: the indices address more rows than the tensor has")
            if self.tape.is_recording() and self.tape.requires_grad(x[0]):
                # a gradient through a gather needs a scatter-add VJP (not on the tape yet)
                raise Error("ops: gather_rows carries no gradient; detach the input first")
            var g_start = ints[0]
            var g_count = ints[1] if ints[1] > 0 else self.rows[gj] - g_start
            if g_start < 0 or g_count < 1 or g_start + g_count > self.rows[gj]:
                raise Error("ops: gather_rows start and count must lie within the indices")
            var g_cols = self.cols[gx]
            var g_dst = self.tape.reserve(self.ex, g_count * g_cols)
            var g_src = self.tape.value_of(self.ex, x[0])
            var g_idx = self.tape.value_of(self.ex, x[1])
            gather_rows(self.ex, g_src, g_idx, g_start, g_count, g_cols, g_dst)
            return self._constant(g_dst, g_count, g_cols, 0)
        elif code == OPS_PERMUTATION:
            _need("permutation", n, 0, 4, ni, 0, nf)
            var p_rows = ints[0]
            var p_dst = self.tape.reserve(self.ex, p_rows if p_rows > 0 else 1)
            permutation(self.ex, _seed(ints[1], ints[2]), ints[3], p_rows, p_dst)
            return self._constant(p_dst, p_rows, 1, p_rows)
        elif code == OPS_MSE:
            _need("mse", n, 2, 1, ni, 0, nf)
            t = mse(self.tape, self.ex, x[0], x[1], ints[0])
        elif code == OPS_MAE:
            _need("mae", n, 2, 1, ni, 0, nf)
            t = mae(self.tape, self.ex, x[0], x[1], ints[0])
        elif code == OPS_HUBER:
            _need("huber", n, 2, 1, ni, 1, nf)
            t = huber(self.tape, self.ex, x[0], x[1], fs[0], ints[0])
        elif code == OPS_BCE_LOGITS:
            _need("bce_with_logits", n, 2, 1, ni, 0, nf)
            t = binary_cross_entropy_with_logits(self.tape, self.ex, x[0], x[1], ints[0])
        elif code == OPS_CROSS_ENTROPY:
            _need("cross_entropy", n, 2, 1, ni, 0, nf)
            t = cross_entropy(self.tape, self.ex, x[0], x[1], ints[0])
        elif code == OPS_DETACH:
            _need("detach", n, 1, 0, ni, 0, nf)
            t = self.tape.detach(self.ex, x[0])
        elif code == OPS_FULL:
            _need("full", n, 0, 2, ni, 1, nf)
            var f_rows = ints[0]
            var f_cols = ints[1]
            if f_rows <= 0 or f_cols <= 0 or f_rows > ((1 << 60) // 4) // f_cols:
                raise Error("ops: full needs positive dimensions")
            var f_dst = self.tape.reserve(self.ex, f_rows * f_cols)
            var fa = Args()
            fa.p0 = f_dst
            fa.f0 = fs[0]
            self.ex.launch[OP_FILL](fa, f_rows * f_cols)
            return self._constant(f_dst, f_rows, f_cols, 0)
        elif code == OPS_EMBEDDING:
            # lane ag-embedding-op: table[ids] (Tape.embedding). ints: the
            # caller's ids as an input descriptor (lane ag-thin-glue2: address,
            # NumPy dtype number, item size, rows, cols, row and column byte
            # strides; any integer dtype and strides, flattened row-major) and
            # padding_idx (-1 none). The ids are cast on the executor to int32
            # words (autograd/cast_ops.mojo upload_cast_i32; values outside
            # int32 saturate and the embedding's id check refuses them) in a
            # session buffer that lives with the graph: it is returned at the
            # next reset or close, like a freed upload.
            _need("embedding", n, 1, 8, ni, 0, nf)
            var e_src = CastIn(
                ints[0], cast_in_kind(ints[1], ints[2], "ops: embedding ids"), ints[2],
                ints[3], ints[4], ints[5], ints[6],
            )
            e_src.check("ops: embedding ids")
            var e_n = e_src.count()
            var e_buf = self.ex.alloc(e_n)
            self.retired.append(Int(e_buf))
            upload_cast_i32(self.ex, e_buf, e_src, "ops: embedding ids")
            var e_ids = self.tape.ids(self.ex, e_buf, e_n)
            t = self.tape.embedding(self.ex, x[0], e_ids, ints[7])
        elif code == OPS_COPY:
            _need("copy", n, 1, 0, ni, 0, nf)
            var cx = self._slot(hs[0])
            var c_count = self.rows[cx] * self.cols[cx]
            var c_rows = self.rows[cx]
            var c_cols = self.cols[cx]
            var c_dst = self.tape.reserve(self.ex, c_count)
            var ca = Args()
            ca.p0 = self.tape.value_of(self.ex, x[0])
            ca.p1 = c_dst
            self.ex.launch[OP_COPY](ca, c_count)
            return self._constant(c_dst, c_rows, c_cols, 0)
        else:
            raise Error("ops: unknown operation code " + String(code))
        return self._add(t, 0, False, 0)

    def op_packed(mut self, code: Int, block: Int) raises -> Int:
        """`op` with its arguments in a packed block at address `block`
        (OPS_PACK_* layout): the same checks and the same call."""
        if block == 0:
            raise Error("ops: null argument block")
        var w = UnsafePointer[Int64, MutUntrackedOrigin](unsafe_from_address=block)
        var nh = Int(w[0])
        var ni = Int(w[1])
        var nf = Int(w[2])
        if nh < 0 or nh > OPS_PACK_HANDLES or ni < 0 or ni > OPS_PACK_INTS or nf < 0 or nf > OPS_PACK_FLOATS:
            raise Error("ops: the argument block's counts are out of range")
        var hs = List[Int](capacity=nh)
        for i in range(nh):  # small-loop(nh: an op's input handles, at most four): argument parsing, no data
            hs.append(Int(w[OPS_PACK_H_WORD + i]))
        var ints = List[Int](capacity=ni)
        for i in range(ni):  # small-loop(ni: an op's integer arguments, at most eight): argument parsing, no data
            ints.append(Int(w[OPS_PACK_I_WORD + i]))
        var f = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=block + 8 * OPS_PACK_F_WORD)
        var fs = List[Float32](capacity=nf)
        for i in range(nf):  # small-loop(nf: an op's float arguments, at most four): argument parsing, no data
            fs.append(f[i])
        return self.op(code, hs, ints, fs)

    def metric(mut self, kind: Int, hs: List[Int]) raises -> Float32:
        """A forward-only metric of (y_true, prediction); one value read back."""
        if len(hs) != 2:
            raise Error("ops: a metric takes y_true and a prediction")
        var ys = self._slot(hs[0])
        var ps = self._slot(hs[1])
        var rows = self.rows[ys]
        if self.cols[ys] != 1 or self.rows[ps] != rows:
            raise Error("ops: y_true must be (n, 1) and the prediction must have n rows")
        var yt_t = self.tensors[ys]
        var yt = self.tape.value_of(self.ex, yt_t)
        var pv_t = self.tensors[ps]
        var pv = self.tape.value_of(self.ex, pv_t)
        var cols = self.cols[ps]
        if kind == OPS_METRIC_ACCURACY:
            if cols == 1:
                return accuracy(self.ex, yt, pv, rows)
            return accuracy_from_scores(self.ex, yt, pv, rows, cols)
        if kind == OPS_METRIC_LOG_LOSS:
            return log_loss(self.ex, yt, pv, rows, cols)
        if kind == OPS_METRIC_ROC_AUC:
            if cols != 1:
                raise Error("ops: roc_auc takes one score column")
            return roc_auc(self.ex, yt, pv, rows)
        raise Error("ops: unknown metric code " + String(kind))


# ---------------------------------------------------------------- the session table
struct _OpsStore(Defaultable, Movable):
    var ids: List[Int]
    var addr: List[Int]
    var kind: List[Int]
    var next_id: Int

    def __init__(out self):
        self.ids = List[Int]()
        self.addr = List[Int]()
        self.kind = List[Int]()
        self.next_id = 1


comptime _OPS_NAME = "MojoAutogradOpsSessionsIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoAutogradOpsSessionsFast"
comptime _OPS_SESSIONS = _Global[StorageType=_OpsStore, name=_OPS_NAME, init_fn=_OpsStore.__init__]


def ops_session_open[E: Exec & Movable & Deinitable, KIND: Int](var ex: E, record: Bool) raises -> Int:
    """A new session on `ex`; returns its id (>= 1)."""
    var p = alloc[OpsSession[E]](1)
    p.init_pointee_move(OpsSession[E](ex^, record))
    var s = _OPS_SESSIONS.get_or_create_ptr()
    var sid = s[].next_id
    s[].next_id += 1
    s[].ids.append(sid)
    s[].addr.append(Int(p))
    s[].kind.append(KIND)
    return sid


def _find(sid: Int, kind: Int) raises -> Int:
    var s = _OPS_SESSIONS.get_or_create_ptr()
    for i in range(len(s[].ids)):  # small-loop(ids: open sessions): a session lookup, no data
        if s[].ids[i] == sid:
            if s[].kind[i] != kind:
                raise Error("ops: session " + String(sid) + " belongs to another binding")
            return i
    raise Error("ops: session " + String(sid) + " is not open")


def ops_session[E: Exec & Movable & Deinitable, KIND: Int](sid: Int) raises -> UnsafePointer[OpsSession[E], MutUntrackedOrigin]:
    """The open session `sid` of this executor kind."""
    var s = _OPS_SESSIONS.get_or_create_ptr()
    var i = _find(sid, KIND)
    return UnsafePointer[OpsSession[E], MutUntrackedOrigin](unsafe_from_address=s[].addr[i])


def ops_session_close[E: Exec & Movable & Deinitable, KIND: Int](sid: Int) raises:
    """Release session `sid`'s storage and drop it from the table."""
    var i = _find(sid, KIND)
    var s = _OPS_SESSIONS.get_or_create_ptr()
    var p = UnsafePointer[OpsSession[E], MutUntrackedOrigin](unsafe_from_address=s[].addr[i])
    _ = s[].ids.pop(i)
    _ = s[].addr.pop(i)
    _ = s[].kind.pop(i)
    try:
        p[].close()
    except e:
        p.destroy_pointee()
        p.free()
        raise e
    p.destroy_pointee()
    p.free()
