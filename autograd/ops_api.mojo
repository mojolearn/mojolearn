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
OP_GATHER_ROWS / OP_MLP_PERM)."""
from std.ffi import _Global
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, OP_FILL, OP_COPY
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from autograd.tape import Tape, Tensor
from autograd.ops import RELU, SIGMOID, TANH, SOFTMAX
from autograd.layers import NeuralRules, layer_norm
from autograd.losses import mse, mae, huber, binary_cross_entropy_with_logits, cross_entropy
from autograd.data import permutation, gather_rows
from autograd.metrics import accuracy, accuracy_from_scores, log_loss, roc_auc

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
    def upload(mut self, src: FP, rows: Int, cols: Int, requires_grad: Bool) raises -> Int:
        """A new tensor holding the caller's rows x cols floats (row-major);
        the caller's memory may change once this returns."""
        if rows <= 0 or cols <= 0 or rows > ((1 << 60) // 4) // cols:
            raise Error("ops: tensor dimensions must be positive")
        if requires_grad and not self.tape.is_recording():
            raise Error("ops: requires_grad needs a recording session (Session(record=True))")
        var n = rows * cols
        var buf = self.ex.alloc(n)
        self.ex.upload(buf, src, n)
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
            # host address of the caller's int32 ids, their count, and
            # padding_idx (-1 none). The ids are copied (bits) into a session
            # buffer that lives with the graph: it is returned at the next
            # reset or close, like a freed upload.
            _need("embedding", n, 1, 3, ni, 0, nf)
            var e_n = ints[1]
            if ints[0] == 0 or e_n < 1:
                raise Error("ops: embedding needs at least one id")
            var e_buf = self.ex.alloc(e_n)
            self.retired.append(Int(e_buf))
            self.ex.upload(e_buf, FP(unsafe_from_address=ints[0]), e_n)
            var e_ids = self.tape.ids(self.ex, e_buf, e_n)
            t = self.tape.embedding(self.ex, x[0], e_ids, ints[2])
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
