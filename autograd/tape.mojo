# SPDX-License-Identifier: Apache-2.0
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, dummy_ptr, OP_COPY, OP_FILL, OP_GEMM
from sequence.ops import OP_ACT, OP_ACT_BWD, OP_COLSUM, OP_COLSUM_DIV, OP_GEMM_EPI, OP_SOFTMAX
from sequence.mlp import (
    ACT_IDENTITY, ACT_LOGISTIC, ACT_TANH, ACT_RELU, ACT_SOFTMAX,
    EPI_BIAS_ACT, EPI_L2GRAD, EPI_ACT_BWD,
)
from checks.numerics import identical_div
from autograd.fold import blocked_fold, fold_partials
from autograd.ops import (
    OP_AG_BIN_FWD, OP_AG_BIN_VJP, OP_AG_UNARY_FWD, OP_AG_UNARY_VJP, OP_AG_TRANSPOSE,
    OP_AG_REDUCE_PART, OP_AG_REDUCE_FOLD, OP_AG_REDUCE_VJP, OP_AG_SOFTMAX_FWD,
    OP_AG_SOFTMAX_VJP, OP_AG_ACCUMULATE,
    LEAF, ADD, SUBTRACT, MULTIPLY,
    DIVIDE, RELU, SIGMOID, TANH, EXP, LOG, SQRT, SCALE, TRANSPOSE, RESHAPE,
    REDUCE, SOFTMAX, LOG_SOFTMAX, MATMUL, LINEAR, CUSTOM, AXIS_ALL, AXIS_ROWS, AXIS_COLS,
    ag_fold_block, unary_vjp_reads_input, unary_vjp_reads_output, LOSS_ELEM, LOSS_CE, LOSS_SCE,
)
from autograd.loss_ops import (
    OP_LS_ELEM_FWD, OP_LS_ELEM_VJP, OP_LS_CE_FWD, OP_LS_CE_VJP, LS_MSE, LS_MAE, LS_HUBER,
    LS_BCE_LOGITS,
)
from autograd.data_ops import OP_AG_DROPOUT_FWD, OP_AG_DROPOUT_VJP, DROPOUT, dropout_args
from autograd.pass1_ops import OP_P1_SCE_VJP

comptime _INFERRED_REFUSAL = "tensor was computed in inference mode (no_grad) and has no graph: call record() and run the forward again to differentiate it"


@fieldwise_init
struct Tensor(ImplicitlyCopyable, Movable):
    """A matrix or scalar recorded by a Tape; scalars have shape (1, 1)."""
    var _owner: Int
    var _generation: Int
    var _index: Int


@fieldwise_init
struct _Node(ImplicitlyCopyable, Movable):
    var value: FP
    var grad: FP
    var rows: Int
    var cols: Int
    var requires_grad: Bool
    var op: Int
    var start: Int
    var count: Int
    var saved: Args
    # The gradient buffer holds a value: written by a backward (or zeroed for
    # an unreached leaf), or caller storage from variable_into.
    var grad_ready: Bool
    # Produced in inference mode (no_grad): a value with no parents, no
    # gradient storage and no backward.
    var inferred: Bool
    # A LINEAR node's gradient buffer holds dL/d(pre-activation), not
    # dL/d(output): its activation derivative was already applied, by its
    # one consumer's dX epilogue or in place by its own backward.
    var grad_preact: Bool
    # The gradient buffer is caller storage (variable_into, custom_into,
    # binary_recorded with caller_grad): never re-pointed by a backward.
    var caller_grad: Bool


trait BackwardRules:
    # Native extensions receive one distinct destination per input, zeroed
    # unless zeroed_destination says the rule needs no zeros there. They
    # write VJPs there; the tape owns graph traversal and accumulation.
    # Return False for unknown tags so a backward refuses BEFORE touching grads.
    def supports(self, tag: Int) -> Bool:
        ...

    # True: zero-fill destination `edge` before apply (the safe default).
    # False: the rule overwrites every element of that destination, or never
    # touches it and the input needs no gradient, so the fill is skipped.
    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        ...

    # True: destination `edge` is the input's gradient buffer itself, never
    # scratch, and the tape adds nothing to it afterwards. The rule writes
    # that input's whole gradient there (or, for a graph whose rules share
    # the join, leaves it to the rule that does), including whatever other
    # consumers contributed. It is never zero-filled. A driver that records
    # a fixed graph uses it to run a fan-in or residual join inside the
    # kernel that already reads the operands, as a hand-written backward
    # does. False everywhere is the plain contract above.
    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        ...

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        ...


@fieldwise_init
struct NoCustomRules(BackwardRules):
    def supports(self, tag: Int) -> Bool:
        return False

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        return True

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        return False

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        raise Error("custom backward rule not registered")


def _fill[E: Exec](mut ex: E, p: FP, n: Int, value: Float32 = 0) raises:
    var a = Args()
    a.p0 = p
    a.f0 = value
    ex.launch[OP_FILL](a, n)


def _copy[E: Exec](mut ex: E, dst: FP, src: FP, n: Int) raises:
    var a = Args()
    a.p0 = src
    a.p1 = dst
    ex.launch[OP_COPY](a, n)


def _accumulate[E: Exec](mut ex: E, dst: FP, src: FP, n: Int) raises:
    var a = Args()
    a.p0 = src
    a.p1 = dst
    ex.launch[OP_AG_ACCUMULATE](a, n)


def _transpose[E: Exec](mut ex: E, dst: FP, src: FP, src_rows: Int, src_cols: Int) raises:
    var a = Args()
    a.p0 = src
    a.p1 = dst
    a.i0 = src_rows
    a.i1 = src_cols
    ex.launch[OP_AG_TRANSPOSE](a, src_rows * src_cols)


def _gemm_epi[E: Exec](
    mut ex: E, A: FP, Bm: FP, C: FP, M: Int, N: Int, K: Int,
    sam: Int, sak: Int, sbk: Int, sbn: Int, epi: Int, p3: FP, i10: Int, f0: Float32, f1: Float32,
) raises:
    """A GEMM and its elementwise follower in one launch, with the argument
    layout of the MLP trainer's gemm_epi (sequence/mlp_fit.mojo), so the
    fused linear node launches what the hand-assembled MLP launches."""
    var a = Args()
    a.p0 = A
    a.p1 = Bm
    a.p2 = C
    a.p3 = p3
    a.i0 = M
    a.i1 = N
    a.i2 = K
    a.i3 = sam
    a.i4 = sak
    a.i5 = sbk
    a.i6 = sbn
    a.i8 = N
    a.i9 = epi
    a.i10 = i10
    a.f0 = f0
    a.f1 = f1
    ex.launch[OP_GEMM_EPI](a, M * N)


def _linear_act(activation: Int) raises -> Int:
    """The MLP activation code of a linear node's activation."""
    if activation == 0:
        return ACT_IDENTITY
    if activation == SIGMOID:
        return ACT_LOGISTIC
    if activation == TANH:
        return ACT_TANH
    if activation == RELU:
        return ACT_RELU
    if activation == SOFTMAX:
        return ACT_SOFTMAX
    raise Error("linear takes activation 0 (none), RELU, SIGMOID, TANH or SOFTMAX")


def _gemm[E: Exec](
    mut ex: E, dst: FP, a: FP, b: FP, m: Int, n: Int, k: Int,
    ar: Int, ac: Int, br: Int, bc: Int,
) raises:
    var args = Args()
    args.p0 = a
    args.p1 = b
    args.p2 = dst
    args.i0 = m
    args.i1 = n
    args.i2 = k
    args.i3 = ar
    args.i4 = ac
    args.i5 = br
    args.i6 = bc
    args.i8 = n
    ex.launch[OP_GEMM](args, m * n)


@always_inline
def _passes_upstream(n: _Node, edge: Int, p: _Node) -> Bool:
    """True when binary node n's VJP into operand `edge` (node p) is n's
    upstream gradient itself: either operand of an add, or the left operand
    of a subtraction, at the output's shape (no broadcast)."""
    return (n.op == ADD or (n.op == SUBTRACT and edge == 0)) and p.rows == n.rows and p.cols == n.cols


struct Tape(Movable):
    """Record native FP32 tensor operations and compute first derivatives.

    Use the same executor for every operation on a tape. It owns tensor
    storage and must outlive the tape. reset() invalidates tensor handles
    and reuses storage for the next graph; release() also returns that
    storage to the executor. Operations are out of place.

    Example::

        var ex = HostExec()
        var tape = Tape()
        var x = tape.scalar(ex, 2.0, True)
        var loss = tape.multiply(ex, x, x)
        tape.backward(ex, loss)
        var dx = tape.gradients(ex, x)

    Inference: after no_grad() every operation computes its value exactly
    as when recording (the same kernels, so the same bits) but keeps no
    graph and no gradient storage; record() switches recording back on.
    Tensors made in inference mode are constants: backward from one, or
    gradient() of one, raises. Example::

        tape.no_grad()
        var y = tape.relu(ex, tape.matmul(ex, x, w))
        var prediction = tape.values(ex, y)
        tape.record()
    """
    var _nodes: List[_Node]
    var _parents: List[Int]
    var _buffers: List[FP]
    var _capacities: List[Int]
    var _cursor: Int
    var _identity: Optional[MutPointer[Int, MutUntrackedOrigin]]
    var _generation: Int
    var _recording: Bool

    def __init__(out self):
        self._nodes = List[_Node]()
        self._parents = List[Int]()
        self._buffers = List[FP]()
        self._capacities = List[Int]()
        self._cursor = 0
        self._identity = None
        self._generation = 0
        self._recording = True

    def no_grad(mut self):
        """Enter inference mode: later operations keep no graph or gradients.

        Values are computed exactly as when recording. The mode lasts until
        record() and is kept across reset() and release().
        """
        self._recording = False

    def record(mut self):
        """Leave inference mode: later operations are recorded for backward."""
        self._recording = True

    def is_recording(self) -> Bool:
        """True unless no_grad() is in effect."""
        return self._recording

    def __deinit__(deinit self):
        # The tape keeps no executor reference, so it cannot free executor
        # storage here: call release(ex) when done. Anything not released is
        # reclaimed when the executor ends (it must outlive the tape).
        if self._identity:
            self._identity.value().free()

    def release[E: Exec](mut self, mut ex: E) raises:
        """Return every buffer this tape allocated to the executor.

        Invalidates all tensor handles, like reset(). Caller-owned storage
        (variable, variable_into and the *_into outputs) is left alone. The
        tape can record again afterwards; it allocates fresh storage.
        """
        for i in range(len(self._buffers)):  # driver-loop: over tape storage slots
            ex.free(self._buffers[i])
        self._buffers.clear()
        self._capacities.clear()
        self.reset()

    def reset(mut self):
        """Discard the recorded graph and reuse its executor storage."""
        self._nodes.clear()
        self._parents.clear()
        self._generation += 1
        self._cursor = 0

    def _check(self, x: Tensor) raises -> Int:
        if not self._identity:
            raise Error("tensor belongs to an empty tape")
        if x._owner != Int(self._identity.value()) or x._generation != self._generation:
            raise Error("tensor belongs to another tape or an expired graph")
        if x._index < 0 or x._index >= len(self._nodes):
            raise Error("invalid tensor handle")
        return x._index

    def _alloc[E: Exec](mut self, mut ex: E, n: Int) raises -> FP:
        var slot = self._cursor
        if slot == len(self._buffers):
            self._buffers.append(ex.alloc(n))
            self._capacities.append(n)
        elif self._capacities[slot] < n:
            # A slot is reused only by a later graph (after reset) or by the
            # next node's backward scratch, so nothing still names the old
            # buffer: return it (a device keeps it until its queue drains).
            # Doubling capacity bounds reallocations for growing shapes.
            var capacity = max(n, self._capacities[slot] * 2)
            ex.free(self._buffers[slot])
            self._buffers[slot] = ex.alloc(capacity)
            self._capacities[slot] = capacity
        self._cursor += 1
        return self._buffers[slot]

    def _shape(self, rows: Int, cols: Int) raises:
        if rows <= 0 or cols <= 0 or rows > ((1 << 60) // 4) // cols:
            raise Error("tensor dimensions must be positive and fit in addressable storage")

    def reserve[E: Exec](mut self, mut ex: E, elements: Int) raises -> FP:
        """Reserve executor storage for a custom operation until the next reset."""
        self._shape(1, elements)
        return self._alloc(ex, elements)

    def _record[E: Exec](
        mut self, mut ex: E, value: FP, rows: Int, cols: Int,
        op: Int, parents: List[Tensor], saved: Args, leaf_grad: Bool = False,
        grad_storage: FP = dummy_ptr(), caller_grad: Bool = False,
    ) raises -> Tensor:
        # ONE VALUE PATH (lane ag-forward). Every operation computes its value
        # (allocation through _alloc, or caller storage for *_into, then the
        # same launch helper) BEFORE calling _record, and _record never
        # launches a value kernel. Inference mode differs from recording only
        # here: the node keeps no parents, gets no gradient storage and is
        # marked inferred. So a recorded and a non-recorded forward run the
        # same kernels on the same inputs and produce the same bits by
        # construction. (Recording also takes gradient slots from _alloc, so
        # the two modes may place values in different slots; slot placement
        # never changes a value.)
        self._shape(rows, cols)
        if not self._identity:
            self._identity = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(alloc[Int](1)))
        if not self._recording:
            for p in parents:  # driver-loop: over node inputs
                _ = self._check(p)
            var at = len(self._nodes)
            self._nodes.append(_Node(value, dummy_ptr(), rows, cols, False, op, len(self._parents), 0,
                                     saved, False, True, False, False))
            return Tensor(Int(self._identity.value()), self._generation, at)
        var requires = leaf_grad
        var start = len(self._parents)
        for p in parents:
            var i = self._check(p)
            requires = requires or self._nodes[i].requires_grad
        for p in parents:
            self._parents.append(p._index)
        # No fill here: backward writes each gradient's first contribution
        # straight into it (see the invariant in _backward).
        var grad = dummy_ptr()
        var ready = False
        if requires:
            if caller_grad:
                grad = grad_storage
                ready = True
            else:
                grad = self._alloc(ex, rows * cols)
        var index = len(self._nodes)
        self._nodes.append(_Node(value, grad, rows, cols, requires, op, start, len(parents), saved, ready, False, False,
                                 requires and caller_grad))
        return Tensor(Int(self._identity.value()), self._generation, index)

    def variable[E: Exec](
        mut self, mut ex: E, data: FP, rows: Int, cols: Int, requires_grad: Bool = True,
    ) raises -> Tensor:
        """Record borrowed executor memory as a leaf; keep it unchanged until backward finishes."""
        return self._record(ex, data, rows, cols, LEAF, List[Tensor](), Args(), requires_grad)

    def variable_into[E: Exec](
        mut self, mut ex: E, data: FP, grad_destination: FP, rows: Int, cols: Int,
    ) raises -> Tensor:
        """Record a differentiable leaf whose gradient lands in caller storage.

        `grad_destination` holds rows * cols floats of this executor. A
        backward writes the gradient there directly (accumulate=True adds to
        what it holds). Keep `data` unchanged until backward finishes, and
        do not let `grad_destination` overlap any tensor of the graph.
        """
        return self._record(ex, data, rows, cols, LEAF, List[Tensor](), Args(), True,
                            grad_destination, True)

    def tensor[E: Exec](
        mut self, mut ex: E, values: List[Float32], rows: Int, cols: Int,
        requires_grad: Bool = False,
    ) raises -> Tensor:
        """Copy a row-major host list onto the executor and record a leaf."""
        self._shape(rows, cols)
        if len(values) != rows * cols:
            raise Error("tensor data length does not match shape")
        var data = self._alloc(ex, rows * cols)
        ex.upload(data, FP(unsafe_from_address=Int(values.unsafe_ptr())), rows * cols)
        return self.variable(ex, data, rows, cols, requires_grad)

    def scalar[E: Exec](mut self, mut ex: E, value: Float32, requires_grad: Bool = False) raises -> Tensor:
        """Create a scalar on the executor."""
        var p = self._alloc(ex, 1)
        _fill(ex, p, 1, value)
        return self.variable(ex, p, 1, 1, requires_grad)

    def value(self, x: Tensor) raises -> FP:
        """Return the executor pointer for a tensor's values."""
        return self._nodes[self._check(x)].value

    def gradient(self, x: Tensor) raises -> FP:
        """Return the executor pointer for a differentiable tensor's gradient."""
        var n = self._nodes[self._check(x)]
        if n.inferred:
            raise Error(_INFERRED_REFUSAL)
        if not n.requires_grad:
            raise Error("tensor does not require a gradient")
        if not n.grad_ready:
            raise Error("tensor has no gradient: run backward from an output that depends on it")
        if n.grad_preact:
            raise Error("this linear output holds its pre-activation gradient: use preactivation_gradient")
        return n.grad

    def preactivation_gradient(self, x: Tensor) raises -> FP:
        """Return the gradient with respect to a linear output's pre-activation.

        Available after a backward for an output recorded by linear or
        linear_into whose activation derivative was applied in place, and
        for one without an activation.
        """
        var n = self._nodes[self._check(x)]
        if n.op != LINEAR:
            raise Error("preactivation_gradient takes an output of linear or linear_into")
        if not n.requires_grad or not n.grad_ready:
            raise Error("tensor has no gradient: run backward from an output that depends on it")
        if not n.grad_preact and n.saved.i0 != ACT_IDENTITY:
            raise Error("the pre-activation gradient of this output was not kept")
        return n.grad

    def requires_grad(self, x: Tensor) raises -> Bool:
        """Return whether a backward computes a gradient for this tensor."""
        return self._nodes[self._check(x)].requires_grad

    def rows(self, x: Tensor) raises -> Int:
        return self._nodes[self._check(x)].rows

    def cols(self, x: Tensor) raises -> Int:
        return self._nodes[self._check(x)].cols

    def values[E: Exec](self, mut ex: E, x: Tensor) raises -> List[Float32]:
        """Download a tensor into a row-major host list."""
        var n = self._nodes[self._check(x)]
        var result = List[Float32](length=n.rows * n.cols, fill=0)
        ex.download(FP(unsafe_from_address=Int(result.unsafe_ptr())), n.value, len(result))
        return result^

    def gradients[E: Exec](self, mut ex: E, x: Tensor) raises -> List[Float32]:
        """Download a differentiable tensor's gradient into a host list."""
        var p = self.gradient(x)
        var n = self._nodes[self._check(x)]
        var result = List[Float32](length=n.rows * n.cols, fill=0)
        ex.download(FP(unsafe_from_address=Int(result.unsafe_ptr())), p, len(result))
        return result^

    def _binary[E: Exec](mut self, mut ex: E, x: Tensor, y: Tensor, op: Int) raises -> Tensor:
        return self._binary_to(ex, x, y, op, dummy_ptr(), False)

    def binary_into[E: Exec](mut self, mut ex: E, x: Tensor, y: Tensor, op: Int, output: FP) raises -> Tensor:
        """add/subtract/multiply/divide (op ADD, SUBTRACT, MULTIPLY or DIVIDE
        from autograd.ops) written to caller storage of the broadcast shape.

        Values equal the matching method's. Keep `output` unchanged until
        backward finishes; it must not overlap an input.
        """
        if op != ADD and op != SUBTRACT and op != MULTIPLY and op != DIVIDE:
            raise Error("binary_into takes ADD, SUBTRACT, MULTIPLY or DIVIDE")
        return self._binary_to(ex, x, y, op, output, True)

    def _binary_to[E: Exec](
        mut self, mut ex: E, x: Tensor, y: Tensor, op: Int, into: FP, use_into: Bool,
    ) raises -> Tensor:
        var a = self._nodes[self._check(x)]
        var b = self._nodes[self._check(y)]
        if (a.rows != b.rows and a.rows != 1 and b.rows != 1) or (a.cols != b.cols and a.cols != 1 and b.cols != 1):
            raise Error("incompatible tensor shapes for broadcasting")
        var rows = max(a.rows, b.rows)
        var cols = max(a.cols, b.cols)
        self._shape(rows, cols)
        var output = into if use_into else self._alloc(ex, rows * cols)
        # Every other slot keeps the Args() placeholder; the body touches
        # p0, p1 and p2 only. x * x binds its one buffer once (i10 = 1).
        var args = Args()
        args.i1 = op
        args.i2 = rows
        args.i3 = cols
        args.i4 = a.rows
        args.i5 = a.cols
        args.i6 = b.rows
        args.i7 = b.cols
        args.p0 = a.value
        if Int(b.value) == Int(a.value):
            args.i10 = 1
        else:
            args.p1 = b.value
        args.p2 = output
        ex.launch[OP_AG_BIN_FWD](args, rows * cols)
        return self._record(ex, output, rows, cols, op, [x, y], args)

    def add[E: Exec](mut self, mut ex: E, x: Tensor, y: Tensor) raises -> Tensor:
        """Add tensors with scalar, row, or column broadcasting."""
        return self._binary(ex, x, y, ADD)

    def subtract[E: Exec](mut self, mut ex: E, x: Tensor, y: Tensor) raises -> Tensor:
        return self._binary(ex, x, y, SUBTRACT)

    def multiply[E: Exec](mut self, mut ex: E, x: Tensor, y: Tensor) raises -> Tensor:
        return self._binary(ex, x, y, MULTIPLY)

    def divide[E: Exec](mut self, mut ex: E, x: Tensor, y: Tensor) raises -> Tensor:
        return self._binary(ex, x, y, DIVIDE)

    def _unary[E: Exec](mut self, mut ex: E, x: Tensor, op: Int, scale: Float32 = 1) raises -> Tensor:
        return self._unary_to(ex, x, op, scale, dummy_ptr(), False)

    def unary_into[E: Exec](
        mut self, mut ex: E, x: Tensor, op: Int, output: FP, scale: Float32 = 1,
    ) raises -> Tensor:
        """An elementwise or row operation written to caller storage.

        op is RELU, SIGMOID, TANH, EXP, LOG, SQRT, SCALE (factor `scale`),
        SOFTMAX or LOG_SOFTMAX from autograd.ops. `output` holds x's
        rows * cols floats; values equal the matching method's. Keep it
        unchanged until backward finishes; it must not overlap x.
        """
        if (op < RELU or op > SCALE) and op != SOFTMAX and op != LOG_SOFTMAX:
            raise Error("unary_into takes an elementwise or softmax operation")
        return self._unary_to(ex, x, op, scale, output, True)

    def _unary_to[E: Exec](
        mut self, mut ex: E, x: Tensor, op: Int, scale: Float32, into: FP, use_into: Bool,
    ) raises -> Tensor:
        var n = self._nodes[self._check(x)]
        var output = into if use_into else self._alloc(ex, n.rows * n.cols)
        var a = Args()
        a.p0 = n.value
        a.p1 = output
        if op == SOFTMAX or op == LOG_SOFTMAX:
            a.i0 = 1 if op == LOG_SOFTMAX else 0
            a.i1 = n.cols
            ex.launch[OP_AG_SOFTMAX_FWD](a, n.rows)
        else:
            a.i0 = op
            a.f0 = scale
            ex.launch[OP_AG_UNARY_FWD](a, n.rows * n.cols)
        return self._record(ex, output, n.rows, n.cols, op, [x], a)

    def relu[E: Exec](mut self, mut ex: E, x: Tensor) raises -> Tensor:
        return self._unary(ex, x, RELU)

    def sigmoid[E: Exec](mut self, mut ex: E, x: Tensor) raises -> Tensor:
        return self._unary(ex, x, SIGMOID)

    def tanh[E: Exec](mut self, mut ex: E, x: Tensor) raises -> Tensor:
        return self._unary(ex, x, TANH)

    def exp[E: Exec](mut self, mut ex: E, x: Tensor) raises -> Tensor:
        return self._unary(ex, x, EXP)

    def log[E: Exec](mut self, mut ex: E, x: Tensor) raises -> Tensor:
        return self._unary(ex, x, LOG)

    def sqrt[E: Exec](mut self, mut ex: E, x: Tensor) raises -> Tensor:
        return self._unary(ex, x, SQRT)

    def scale[E: Exec](mut self, mut ex: E, x: Tensor, factor: Float32) raises -> Tensor:
        return self._unary(ex, x, SCALE, factor)

    def softmax[E: Exec](mut self, mut ex: E, x: Tensor) raises -> Tensor:
        """Normalize each row with a numerically stable softmax."""
        return self._unary(ex, x, SOFTMAX)

    def log_softmax[E: Exec](mut self, mut ex: E, x: Tensor) raises -> Tensor:
        return self._unary(ex, x, LOG_SOFTMAX)

    def dropout[E: Exec](
        mut self, mut ex: E, x: Tensor, p: Float32, seed: UInt64, counter: Int,
    ) raises -> Tensor:
        """Zero each element with probability p and scale the rest by 1 / (1 - p).

        The mask is a pure function of (seed, counter, element index), so the
        same seed and counter give the same mask on every executor and every
        run. Use a new counter for every training step (for example the step
        number) and the same one when a step is replayed. 0 <= p < 1 and
        0 <= counter < 2^32.

        Example::

            var h = tape.relu(ex, tape.matmul(ex, x, w))
            var d = tape.dropout(ex, h, 0.1, seed, step)
        """
        if not (p >= Float32(0) and p < Float32(1)):
            raise Error("dropout probability must be in [0, 1)")
        if counter < 0 or counter > 0xFFFFFFFF:
            raise Error("dropout counter must be in [0, 2^32)")
        var n = self._nodes[self._check(x)]
        var output = self._alloc(ex, n.rows * n.cols)
        # s = 1 / (1 - p) once here in float32; every element then takes one
        # rounded multiply by s (autograd/data_ops.mojo).
        var scale = identical_div(Float32(1), Float32(1) - p)
        var a = dropout_args(seed, counter, p, scale)
        a.p0 = n.value
        a.p1 = output
        ex.launch[OP_AG_DROPOUT_FWD](a, n.rows * n.cols)
        return self._record(ex, output, n.rows, n.cols, DROPOUT, [x], dropout_args(seed, counter, p, scale))

    def matmul[E: Exec](mut self, mut ex: E, x: Tensor, y: Tensor) raises -> Tensor:
        """Multiply two matrices using the existing native GEMM executor."""
        var a = self._nodes[self._check(x)]
        var b = self._nodes[self._check(y)]
        if a.cols != b.rows:
            raise Error("matmul inner dimensions differ")
        self._shape(a.rows, b.cols)
        var output = self._alloc(ex, a.rows * b.cols)
        _gemm(ex, output, a.value, b.value, a.rows, b.cols, a.cols, a.cols, 1, b.cols, 1)
        return self._record(ex, output, a.rows, b.cols, MATMUL, [x, y], Args())

    def linear[E: Exec](
        mut self, mut ex: E, x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
        activation: Int = 0, l2: Float32 = 0, divisor: Float32 = 0,
    ) raises -> Tensor:
        """A dense layer, activation(x @ weight + bias), recorded as one node.

        Use it for dense layers instead of matmul, add and an activation:
        it runs one forward launch and three backward launches per layer.

        Args:
            x: The (rows, inputs) input.
            weight: The (inputs, outputs) weights.
            bias: An optional (1, outputs) bias.
            activation: 0 (none), RELU, SIGMOID, TANH or SOFTMAX from
                autograd.ops.
            l2: L2 penalty on the weights; needs a positive `divisor`.
            divisor: When positive, this layer's weight gradient is
                (x^T dZ + l2 * weight) / divisor and its bias gradient is
                the column sum of dZ divided by `divisor` (dZ the gradient
                with respect to the pre-activation). 0 leaves both unscaled.

        Returns:
            The (rows, outputs) activated output.

        Example::

            var h = tape.linear(ex, x, w1, b1, RELU)
            var logits = tape.linear(ex, h, w2, b2)
        """
        return self._linear_to(ex, x, weight, bias, activation, l2, divisor, dummy_ptr(), False)

    def linear_into[E: Exec](
        mut self, mut ex: E, x: Tensor, weight: Tensor, bias: Optional[Tensor], activation: Int,
        output: FP, l2: Float32 = 0, divisor: Float32 = 0,
    ) raises -> Tensor:
        """linear() with its output written to caller storage.

        `output` holds rows * outputs floats of this executor and must not
        overlap x. Keep it unchanged until backward finishes. Record
        `weight` and `bias` with variable_into to have their gradients land
        in caller storage as well.
        """
        return self._linear_to(ex, x, weight, bias, activation, l2, divisor, output, True)

    def _linear_to[E: Exec](
        mut self, mut ex: E, x: Tensor, weight: Tensor, bias: Optional[Tensor], activation: Int,
        l2: Float32, divisor: Float32, into: FP, use_into: Bool,
    ) raises -> Tensor:
        # LAUNCHES PER DENSE LAYER (lane ag-fusion2, read from the code).
        # Hand-assembled MLP (sequence/mlp_fit.mojo, AUTOGRAD_MLP_OFF):
        #   forward 1 (gemm_epi EPI_BIAS_ACT; + OP_SOFTMAX for a softmax
        #   output); backward 3: gemm_epi EPI_L2GRAD (dW), OP_COLSUM_DIV
        #   (db), gemm_epi EPI_ACT_BWD (dX times the layer below's act');
        #   2 for the first layer (no dX).
        # Unfused tape (matmul + add + activation): forward 3 (2 without an
        #   activation); backward activation VJP 1, the product's add VJP 1,
        #   the bias's add VJP 3 (elementwise VJP + two-launch blocked fold),
        #   GEMMs 2 (1 for the first layer), the MLP's L2GRAD + DIVS export
        #   2: 9 (8 for the first layer); plus 1 seed copy a backward.
        # This node: forward 1 (gemm_epi EPI_BIAS_ACT, the hand path's call;
        #   + OP_SOFTMAX for softmax; without a bias OP_GEMM + OP_ACT).
        #   Backward 3: dW = gemm_epi EPI_L2GRAD (OP_GEMM when unscaled),
        #   db = OP_COLSUM_DIV (OP_COLSUM when unscaled), dX = gemm_epi
        #   EPI_ACT_BWD with the input layer's activation derivative in the
        #   epilogue when the input is a linear node consumed only here
        #   (else OP_GEMM). 2 for a layer whose input needs no gradient.
        #   The dZ materialisation launch (OP_ACT_BWD in place, or the
        #   softmax VJP into scratch) runs only when no consumer fused the
        #   activation derivative and the seed is not the pre-activation
        #   gradient (backward_from_preactivation): 0 in the MLP. With
        #   backward_from_preactivation the root needs no seed copy either.
        #   So the MLP runs exactly the hand path's launches with the hand
        #   path's arguments.
        var a = self._nodes[self._check(x)]
        var w = self._nodes[self._check(weight)]
        if a.cols != w.rows:
            raise Error("linear: input columns must equal weight rows")
        var act = _linear_act(activation)
        if divisor < 0 or (l2 != 0 and not (divisor > 0)):
            raise Error("linear: divisor must be positive when l2 is set, and never negative")
        var rows = a.rows
        var cols = w.cols
        var inner = a.cols
        self._shape(rows, cols)
        var parents = List[Tensor]()
        parents.append(x)
        parents.append(weight)
        var output = into if use_into else self._alloc(ex, rows * cols)
        if bias:
            var bt = bias.value()
            var b = self._nodes[self._check(bt)]
            if b.rows != 1 or b.cols != cols:
                raise Error("linear: bias must have shape (1, outputs)")
            parents.append(bt)
            # the GEMM, its bias and (not softmax) its activation: one launch
            _gemm_epi(ex, a.value, w.value, output, rows, cols, inner, inner, 1, cols, 1,
                      EPI_BIAS_ACT, b.value, 0 if act == ACT_SOFTMAX else act, Float32(0.0), Float32(0.0))
        else:
            _gemm(ex, output, a.value, w.value, rows, cols, inner, inner, 1, cols, 1)
            if act != ACT_IDENTITY and act != ACT_SOFTMAX:
                var f = Args()
                f.p0 = output
                f.i0 = act
                ex.launch[OP_ACT](f, rows * cols)
        if act == ACT_SOFTMAX:
            # in place, as the MLP trainer's _act_launch
            var s = Args()
            s.p0 = output
            s.p1 = output
            s.i0 = act
            s.i1 = cols
            ex.launch[OP_SOFTMAX](s, rows)
        var saved = Args()
        saved.i0 = act
        if bias:
            saved.i1 = 1
        saved.f0 = l2
        saved.f1 = divisor
        return self._record(ex, output, rows, cols, LINEAR, parents, saved)

    def transpose[E: Exec](mut self, mut ex: E, x: Tensor) raises -> Tensor:
        var n = self._nodes[self._check(x)]
        var output = self._alloc(ex, n.rows * n.cols)
        _transpose(ex, output, n.value, n.rows, n.cols)
        return self._record(ex, output, n.cols, n.rows, TRANSPOSE, [x], Args())

    def reshape[E: Exec](mut self, mut ex: E, x: Tensor, rows: Int, cols: Int) raises -> Tensor:
        self._shape(rows, cols)
        var n = self._nodes[self._check(x)]
        if rows * cols != n.rows * n.cols:
            raise Error("reshape must preserve element count")
        return self._record(ex, n.value, rows, cols, RESHAPE, [x], Args())

    def detach[E: Exec](mut self, mut ex: E, x: Tensor) raises -> Tensor:
        """Create a constant view of x that stops gradient propagation."""
        var n = self._nodes[self._check(x)]
        return self.variable(ex, n.value, n.rows, n.cols, False)

    def _reduce[E: Exec](mut self, mut ex: E, x: Tensor, axis: Int, mean: Bool) raises -> Tensor:
        if axis < -1 or axis > 1:
            raise Error("reduction axis must be -1 (all), 0, or 1")
        var n = self._nodes[self._check(x)]
        var rows = n.rows if axis == 1 else 1
        var cols = n.cols if axis == 0 else 1
        var count = n.rows * n.cols if axis == -1 else (n.rows if axis == 0 else n.cols)
        var output = self._alloc(ex, rows * cols)
        var factor = identical_div(Float32(1), Float32(count)) if mean else Float32(1)
        self._fold(ex, output, n.value, n.rows, n.cols, axis, factor)
        var a = Args()
        a.i0 = axis
        a.f0 = factor
        return self._record(ex, output, rows, cols, REDUCE, [x], a)

    def _fold[E: Exec](
        mut self, mut ex: E, dst: FP, src: FP, rows: Int, cols: Int, axis: Int, factor: Float32,
    ) raises:
        """dst = factor * (src summed over `axis`) as a blocked fold: blocks of
        ag_fold_block(extent) consecutive terms, each added ascending from
        +0.0 (OP_AG_REDUCE_PART, parallel over blocks x outputs), then each
        output's block partials added ascending from +0.0 (OP_AG_REDUCE_FOLD,
        parallel over outputs). One order on every executor. The partials
        are tape scratch (released with the graph's storage at reset)."""
        # The launches live in autograd/fold.mojo (blocked_fold), shared with
        # the losses and metrics so every sum in autograd has one order.
        var partials = self._alloc(ex, fold_partials(rows, cols, axis))
        blocked_fold(ex, dst, src, rows, cols, axis, factor, partials)

    def sum[E: Exec](mut self, mut ex: E, x: Tensor, axis: Int = -1) raises -> Tensor:
        """Sum all elements or one matrix axis, retaining reduced dimensions."""
        return self._reduce(ex, x, axis, False)

    def mean[E: Exec](mut self, mut ex: E, x: Tensor, axis: Int = -1) raises -> Tensor:
        return self._reduce(ex, x, axis, True)

    def mse_loss[E: Exec](mut self, mut ex: E, prediction: Tensor, target: Tensor) raises -> Tensor:
        var difference = self.subtract(ex, prediction, target)
        var square = self.multiply(ex, difference, difference)
        return self.mean(ex, square)

    def cross_entropy[E: Exec](mut self, mut ex: E, logits: Tensor, targets: Tensor) raises -> Tensor:
        """Mean cross-entropy with dense class weights, one row per example."""
        var a = self._nodes[self._check(logits)]
        var b = self._nodes[self._check(targets)]
        if a.rows != b.rows or a.cols != b.cols:
            raise Error("cross-entropy targets must have the logits' shape")
        var logp = self.log_softmax(ex, logits)
        var weighted = self.multiply(ex, logp, targets)
        var rows = self.sum(ex, weighted, 1)
        var loss = self.mean(ex, rows)
        return self.scale(ex, loss, -1)

    def loss_terms[E: Exec](
        mut self, mut ex: E, prediction: Tensor, target: Tensor, kind: Int, delta: Float32 = 1,
    ) raises -> Tensor:
        """Per-element loss terms of two same-shape tensors, differentiable in
        both. kind is LS_MSE, LS_MAE, LS_HUBER (threshold `delta`) or
        LS_BCE_LOGITS (prediction holds logits) from autograd.loss_ops.
        autograd/losses.mojo reduces the terms with mean or sum."""
        if kind != LS_MSE and kind != LS_MAE and kind != LS_HUBER and kind != LS_BCE_LOGITS:
            raise Error("loss_terms takes LS_MSE, LS_MAE, LS_HUBER or LS_BCE_LOGITS")
        if kind == LS_HUBER and not (delta > 0):
            raise Error("Huber delta must be positive")
        var a = self._nodes[self._check(prediction)]
        var b = self._nodes[self._check(target)]
        if a.rows != b.rows or a.cols != b.cols:
            raise Error("loss target must have the prediction's shape")
        if Int(a.value) == Int(b.value):
            raise Error("loss prediction and target must be distinct tensors")
        var output = self._alloc(ex, a.rows * a.cols)
        var args = Args()
        args.p0 = a.value
        args.p1 = b.value
        args.p2 = output
        args.i0 = kind
        args.f0 = delta
        ex.launch[OP_LS_ELEM_FWD](args, a.rows * a.cols)
        var saved = Args()
        saved.i0 = kind
        saved.f0 = delta
        return self._record(ex, output, a.rows, a.cols, LOSS_ELEM, [prediction, target], saved)

    def cross_entropy_rows[E: Exec](mut self, mut ex: E, logits: Tensor, codes: Tensor) raises -> Tensor:
        """Per-row softmax cross-entropy (rows, 1) of logits (rows, classes)
        against class codes (rows, 1) held as float values. The codes take
        no gradient; a code outside [0, classes) gives NaN."""
        var a = self._nodes[self._check(logits)]
        var b = self._nodes[self._check(codes)]
        if b.rows != a.rows or b.cols != 1:
            raise Error("cross-entropy codes must have shape (rows, 1)")
        if b.requires_grad:
            raise Error("cross-entropy codes cannot require a gradient")
        if Int(a.value) == Int(b.value):
            raise Error("cross-entropy logits and codes must be distinct tensors")
        var output = self._alloc(ex, a.rows)
        var args = Args()
        args.p0 = a.value
        args.p1 = b.value
        args.p2 = output
        args.i1 = a.cols
        ex.launch[OP_LS_CE_FWD](args, a.rows)
        return self._record(ex, output, a.rows, 1, LOSS_CE, [logits, codes], Args())

    def softmax_cross_entropy[E: Exec](mut self, mut ex: E, logits: Tensor, codes: Tensor) raises -> Tensor:
        """Mean softmax cross-entropy of logits (rows, classes) against class
        codes (rows, 1, the class index as a float value), as one node.

        The value and the logits' gradient equal those of
        cross_entropy_rows followed by mean, bit for bit; the backward is a
        single launch that writes (softmax - onehot) / rows into the
        logits' gradient. The codes take no gradient; a code outside
        [0, classes) gives NaN.

        Example::

            var logits = tape.linear(ex, h, w, b)
            var loss = tape.softmax_cross_entropy(ex, logits, codes)
            tape.backward(ex, loss)
        """
        # LAUNCHES (read from the code). cross_entropy_rows + mean: forward
        # 3 (OP_LS_CE_FWD, the two-launch blocked fold), backward 2
        # (OP_AG_REDUCE_VJP, OP_LS_CE_VJP). Tape.cross_entropy (dense
        # targets): forward 7, backward 5. This node: forward 3 (the same
        # row losses and the same fold; a mean in one launch would need a
        # serial or an unpinned reduction), backward 1 (OP_P1_SCE_VJP).
        var a = self._nodes[self._check(logits)]
        var b = self._nodes[self._check(codes)]
        if b.rows != a.rows or b.cols != 1:
            raise Error("cross-entropy codes must have shape (rows, 1)")
        if b.requires_grad:
            raise Error("cross-entropy codes cannot require a gradient")
        if Int(a.value) == Int(b.value):
            raise Error("cross-entropy logits and codes must be distinct tensors")
        var terms = self._alloc(ex, a.rows)
        var args = Args()
        args.p0 = a.value
        args.p1 = b.value
        args.p2 = terms
        args.i1 = a.cols
        ex.launch[OP_LS_CE_FWD](args, a.rows)
        var output = self._alloc(ex, 1)
        # Tape.mean's factor and fold over the (rows, 1) row losses
        var factor = identical_div(Float32(1), Float32(a.rows))
        self._fold(ex, output, terms, a.rows, 1, AXIS_ALL, factor)
        var saved = Args()
        saved.f0 = factor
        return self._record(ex, output, 1, 1, LOSS_SCE, [logits, codes], saved)

    def custom[E: Exec](
        mut self, mut ex: E, inputs: List[Tensor], output: FP,
        rows: Int, cols: Int, tag: Int, saved: Args,
    ) raises -> Tensor:
        """Record an already executed native operation with a custom backward tag.

        Keep output and saved buffers alive and unchanged through backward.
        Pass its BackwardRules implementation to backward_with_rules().
        """
        if tag < 0 or tag > (1 << 30):
            raise Error("custom operation tag is out of range")
        var args = saved
        # The tag is metadata; all Args slots remain available to the rule.
        var result = self._record(ex, output, rows, cols, CUSTOM, inputs, args)
        self._nodes[result._index].op = CUSTOM + tag
        return result

    def custom_into[E: Exec](
        mut self, mut ex: E, inputs: List[Tensor], output: FP, grad_destination: FP,
        rows: Int, cols: Int, tag: Int, saved: Args,
    ) raises -> Tensor:
        """Like custom(), with the operation's gradient held in caller storage.

        `grad_destination` holds rows * cols floats of this executor. Pass
        the same storage as the seed of backward_with_rules() and the seed
        is used in place, without a copy. Keep it unchanged through
        backward, and do not let it overlap any tensor of the graph other
        than that seed.
        """
        if tag < 0 or tag > (1 << 30):
            raise Error("custom operation tag is out of range")
        var args = saved
        var result = self._record(ex, output, rows, cols, CUSTOM, inputs, args, False, grad_destination, True)
        self._nodes[result._index].op = CUSTOM + tag
        return result

    def binary_recorded[E: Exec](
        mut self, mut ex: E, x: Tensor, y: Tensor, op: Int, output: FP,
        grad_destination: FP = dummy_ptr(), caller_grad: Bool = False,
    ) raises -> Tensor:
        """Record an already executed add/subtract/multiply/divide.

        `output` already holds the broadcast result of x op y (computed by
        the caller's own kernel); nothing is launched. The backward is the
        tape's own binary VJP, which reads the operands' values (multiply,
        divide) and `output` (divide). With caller_grad=True the gradient
        lands in `grad_destination`, as in custom_into.
        """
        if op != ADD and op != SUBTRACT and op != MULTIPLY and op != DIVIDE:
            raise Error("binary_recorded takes ADD, SUBTRACT, MULTIPLY or DIVIDE")
        var a = self._nodes[self._check(x)]
        var b = self._nodes[self._check(y)]
        if (a.rows != b.rows and a.rows != 1 and b.rows != 1) or (a.cols != b.cols and a.cols != 1 and b.cols != 1):
            raise Error("incompatible tensor shapes for broadcasting")
        var rows = max(a.rows, b.rows)
        var cols = max(a.cols, b.cols)
        return self._record(ex, output, rows, cols, op, [x, y], Args(), False,
                            grad_destination, caller_grad)

    def _binary_vjp[E: Exec](
        mut self, mut ex: E, n: _Node, edge: Int, contribution: FP, rows: Int, cols: Int,
    ) raises:
        """Operand `edge`'s VJP of binary node n into `contribution` (the
        operand's shape rows x cols). The elementwise VJP is computed at the
        output's shape; a broadcast operand's gradient is that matrix summed
        over the broadcast axis (or all of it, for a scalar operand) by the
        blocked fold (_fold), never one thread per gradient element."""
        var left = self._nodes[self._parents[n.start]]
        var right = self._nodes[self._parents[n.start + 1]]
        var a = Args()
        a.i1 = n.op
        a.i2 = n.rows
        a.i3 = n.cols
        a.i4 = left.rows
        a.i5 = left.cols
        a.i6 = right.rows
        a.i7 = right.cols
        a.i8 = edge
        a.p3 = n.grad
        # Bind only the slots the kind reads (op_ag_bin_vjp), each buffer once.
        if n.op == MULTIPLY:
            a.p0 = left.value
            if Int(right.value) == Int(left.value):
                a.i10 = 1
            else:
                a.p1 = right.value
        elif n.op == DIVIDE:
            a.p1 = right.value
            if edge == 1:
                a.p2 = n.value
        var reduce_rows = rows < n.rows
        var reduce_cols = cols < n.cols
        if not reduce_rows and not reduce_cols:
            a.p4 = contribution
            ex.launch[OP_AG_BIN_VJP](a, n.rows * n.cols)
            return
        var axis = AXIS_ALL if reduce_rows and reduce_cols else (AXIS_ROWS if reduce_rows else AXIS_COLS)
        if n.op == ADD or (n.op == SUBTRACT and edge == 0):
            # The elementwise VJP is the upstream itself (ftz(g), stored
            # flushed), and the fold flushes each term on load, so folding
            # n.grad directly gives the same bits without the elementwise
            # launch or its scratch. (Not the subtraction's right operand:
            # -1 * fold(g) and fold(-g) differ in the sign of a zero sum.)
            self._fold(ex, contribution, n.grad, n.rows, n.cols, axis, Float32(1))
            return
        var elementwise = self._alloc(ex, n.rows * n.cols)
        a.p4 = elementwise
        ex.launch[OP_AG_BIN_VJP](a, n.rows * n.cols)
        self._fold(ex, contribution, elementwise, n.rows, n.cols, axis, Float32(1))

    def backward[E: Exec](mut self, mut ex: E, output: Tensor, accumulate: Bool = False) raises:
        """Differentiate a scalar, clearing gradients unless accumulate=True."""
        self.backward_scalar_with_rules(ex, output, NoCustomRules(), accumulate)

    def backward_scalar_with_rules[E: Exec, R: BackwardRules](
        mut self, mut ex: E, output: Tensor, rules: R, accumulate: Bool = False,
    ) raises:
        """Differentiate a scalar using native custom backward rules."""
        var n = self._nodes[self._check(output)]
        if n.inferred:
            raise Error(_INFERRED_REFUSAL)
        if n.rows * n.cols != 1:
            raise Error("non-scalar backward requires an explicit seed tensor")
        var cursor = self._cursor
        # The seed 1 is written straight into the root's own gradient (one
        # launch, no seed copy) unless the root is a leaf whose gradient
        # accumulates, which must add the 1 to what it holds.
        var seed: FP
        if accumulate and n.op == LEAF:
            seed = self._alloc(ex, 1)
        else:
            seed = n.grad
        _fill(ex, seed, 1, 1)
        try:
            self._backward(ex, output, seed, rules, accumulate)
        except e:
            self._cursor = cursor
            raise e
        self._cursor = cursor

    def backward_from[E: Exec](
        mut self, mut ex: E, output: Tensor, seed: Tensor, accumulate: Bool = False,
    ) raises:
        """Compute a vector-Jacobian product using a seed of the output's shape."""
        self.backward_with_rules(ex, output, seed, NoCustomRules(), accumulate)

    def backward_from_preactivation[E: Exec](
        mut self, mut ex: E, output: Tensor, seed: Tensor, accumulate: Bool = False,
    ) raises:
        """Backpropagate a seed that is the gradient with respect to a linear
        output's pre-activation (x @ W + b), not its activated output.

        Use it when the loss derivative already includes the output
        activation, as the softmax or sigmoid with log loss derivative
        p - y does. `output` comes from linear or linear_into and `seed` has
        its shape. The seed is read in place (no copy); keep it unchanged
        until backward finishes. The output's own gradient is not stored.
        """
        self.backward_from_preactivation_with_rules(ex, output, seed, NoCustomRules(), accumulate)

    def backward_from_preactivation_with_rules[E: Exec, R: BackwardRules](
        mut self, mut ex: E, output: Tensor, seed: Tensor, rules: R, accumulate: Bool = False,
    ) raises:
        var n = self._nodes[self._check(output)]
        var s = self._nodes[self._check(seed)]
        if n.op != LINEAR:
            raise Error("backward_from_preactivation takes an output of linear or linear_into")
        if n.rows != s.rows or n.cols != s.cols:
            raise Error("backward seed shape must equal output shape")
        var cursor = self._cursor
        try:
            self._backward(ex, output, s.value, rules, accumulate, True)
        except e:
            self._cursor = cursor
            raise e
        self._cursor = cursor

    def _linear_dz[E: Exec](mut self, mut ex: E, i: Int) raises -> FP:
        """dL/dZ of linear node i from its output gradient dY: dY itself
        without an activation; act'(Y) applied in place (OP_ACT_BWD, the
        hand path's body) for an elementwise one; the softmax VJP into
        backward scratch for softmax."""
        var n = self._nodes[i]
        var act = n.saved.i0
        if act == ACT_IDENTITY:
            return n.grad
        if act == ACT_SOFTMAX:
            var d = self._alloc(ex, n.rows * n.cols)
            var s = Args()
            s.p0 = n.value
            s.p1 = n.grad
            s.p2 = d
            s.i0 = 0
            s.i1 = n.cols
            ex.launch[OP_AG_SOFTMAX_VJP](s, n.rows)
            return d
        var a = Args()
        a.p0 = n.grad
        a.p1 = n.value
        a.i0 = act
        ex.launch[OP_ACT_BWD](a, n.rows * n.cols)
        self._nodes[i].grad_preact = True
        return n.grad

    def _linear_vjp[E: Exec](
        mut self, mut ex: E, n: _Node, edge: Int, dst: FP, first: Bool, dz: FP, consumers: List[Int],
    ) raises:
        """Edge `edge` of linear node n into `dst` (the parent's gradient
        for a first contribution, else scratch), from dz = dL/dZ. The calls
        and arguments are the hand MLP backward's (sequence/mlp_fit.mojo)."""
        var xi = self._parents[n.start]
        var x = self._nodes[xi]
        var w = self._nodes[self._parents[n.start + 1]]
        var rows = n.rows
        var fi = x.cols
        var fo = n.cols
        var divisor = n.saved.f1
        if edge == 1:
            # dW = x^T dz (+ l2 W) / divisor
            if divisor > 0:
                _gemm_epi(ex, x.value, dz, dst, fi, fo, rows, 1, fi, fo, 1,
                          EPI_L2GRAD, w.value, 0, n.saved.f0, divisor)
            else:
                _gemm(ex, dst, x.value, dz, fi, fo, rows, 1, fi, fo, 1)
        elif edge == 2:
            # db = column sum of dz over the rows (/ divisor)
            var c = Args()
            c.p0 = dz
            c.p1 = dst
            c.i0 = rows
            c.i2 = fo
            if divisor > 0:
                c.f0 = divisor
                ex.launch[OP_COLSUM_DIV](c, fo)
            else:
                ex.launch[OP_COLSUM](c, fo)
        else:
            # dX = dz W^T. When x is a linear node whose only consumer is
            # this node, its activation derivative rides in the epilogue
            # and x's gradient holds its pre-activation gradient: the hand
            # path's EPI_ACT_BWD (act' read from x's output values).
            var x_act = x.saved.i0
            if first and consumers[xi] == 1 and x.op == LINEAR and x_act != ACT_SOFTMAX:
                _gemm_epi(ex, dz, w.value, dst, rows, fi, fo, fo, 1, 1, fo,
                          EPI_ACT_BWD, x.value, x_act, Float32(0.0), Float32(0.0))
                if x_act != ACT_IDENTITY:
                    self._nodes[xi].grad_preact = True
            else:
                _gemm(ex, dst, dz, w.value, rows, fi, fo, fo, 1, 1, fo)

    def backward_with_rules[E: Exec, R: BackwardRules](
        mut self, mut ex: E, output: Tensor, seed: Tensor, rules: R, accumulate: Bool = False,
    ) raises:
        var n = self._nodes[self._check(output)]
        var s = self._nodes[self._check(seed)]
        if n.inferred:
            raise Error(_INFERRED_REFUSAL)
        if n.rows != s.rows or n.cols != s.cols:
            raise Error("backward seed shape must equal output shape")
        var cursor = self._cursor
        try:
            self._backward(ex, output, s.value, rules, accumulate)
        except e:
            self._cursor = cursor
            raise e
        self._cursor = cursor

    def _backward[E: Exec, R: BackwardRules](
        mut self, mut ex: E, output: Tensor, seed: FP, rules: R, accumulate: Bool,
        preact: Bool = False,
    ) raises:
        var root = self._check(output)
        if self._nodes[root].inferred:
            raise Error(_INFERRED_REFUSAL)
        if not self._nodes[root].requires_grad:
            raise Error("output has no differentiable ancestors")
        var active = List[Bool](length=len(self._nodes), fill=False)
        # Edges from differentiated nodes into each node (a linear node's
        # dX epilogue may apply its input's act' only for a sole consumer).
        var consumers = List[Int](length=len(self._nodes), fill=0)
        active[root] = True
        # Creation order is topological. Mark only ancestors of this output;
        # disconnected invalid-domain operations are never differentiated.
        for reverse in range(root + 1):
            var i = root - reverse
            if not active[i]:
                continue
            var n = self._nodes[i]
            if n.op >= CUSTOM and not rules.supports(n.op - CUSTOM):
                raise Error("custom backward rule not registered")
            for edge in range(n.count):
                var p = self._parents[n.start + edge]
                if self._nodes[p].requires_grad:
                    active[p] = True
                    consumers[p] += 1
        # GRADIENT WRITE INVARIANT (lane ag-fusion). Gradients are never
        # zero-filled before use. touched[i] says node i's gradient already
        # holds a contribution. The FIRST contribution to a parent is written
        # by the VJP (or matmul GEMM) directly into the parent's grad; only
        # later contributions go to scratch and are added with ACCUMULATE in
        # the same order as before. That is bit for bit the old zero,
        # copy-first, add-the-rest schedule. A VJP reads parent VALUES and
        # the node's own grad, never a parent's grad, so writing the parent's
        # grad in place is safe. Leaves start touched only when
        # accumulate=True (a leaf never written before is zeroed first, as
        # record time used to). After the traversal an unreached
        # differentiable leaf is zeroed (gradient 0); an unreached
        # intermediate has no gradient (gradient() refuses).
        var touched = List[Bool](length=len(self._nodes), fill=False)
        for i in range(len(self._nodes)):  # driver-loop: over graph nodes
            self._nodes[i].grad_preact = False
            var n = self._nodes[i]
            if not n.requires_grad:
                continue
            if accumulate and n.op == LEAF:
                # A leaf never written before holds no value to add to: it
                # stays untouched, so its first contribution is written in
                # place like any other (no fill before use) and, if no path
                # reaches it, the zero fill after the traversal gives it 0.
                if n.grad_ready:
                    touched[i] = True
            else:
                self._nodes[i].grad_ready = False
        # Preserve the seed before overwriting a leaf root's gradient. A
        # pre-activation seed (backward_from_preactivation) is read in place
        # as the root linear node's dZ: no copy.
        if preact:
            pass
        elif touched[root]:
            _accumulate(ex, self._nodes[root].grad, seed, self._nodes[root].rows * self._nodes[root].cols)
        elif Int(self._nodes[root].grad) != Int(seed):
            _copy(ex, self._nodes[root].grad, seed, self._nodes[root].rows * self._nodes[root].cols)
        touched[root] = True
        var scratch_start = self._cursor
        for reverse in range(root + 1):  # driver-loop: over graph nodes, reverse creation order
            var i = root - reverse
            if not active[i]:
                continue
            var n = self._nodes[i]
            if n.op == LEAF:
                continue
            self._cursor = scratch_start
            # Destinations: the parent's own grad for a first contribution,
            # scratch for a later one (and, for a custom rule, for an input
            # that needs no gradient: the rule writes every destination).
            var first = List[Bool](length=n.count, fill=False)
            var total = List[Bool](length=n.count, fill=False)
            var contributions = List[FP]()
            for edge in range(n.count):  # driver-loop: over node inputs
                var parent = self._parents[n.start + edge]
                var p = self._nodes[parent]
                if p.requires_grad and n.op >= CUSTOM and rules.writes_total(n.op - CUSTOM, n.saved, edge):
                    # the rule writes this input's whole gradient in place
                    first[edge] = True
                    total[edge] = True
                    touched[parent] = True
                    contributions.append(p.grad)
                elif p.requires_grad and not touched[parent]:
                    first[edge] = True
                    touched[parent] = True
                    if (n.op == RESHAPE and consumers[parent] == 1 and p.op != LEAF and p.op < CUSTOM
                            and p.op != LINEAR and not p.caller_grad):
                        # A reshape's only-consumed, tape-owned operand takes
                        # the reshape's gradient buffer as its own: the copy
                        # it replaces is a copy of bits, and no later write
                        # reaches either (one consumer; its backward never
                        # writes its own gradient, unlike LINEAR's act' in
                        # place). No launch.
                        self._nodes[parent].grad = n.grad
                    contributions.append(self._nodes[parent].grad)
                elif p.requires_grad and (n.op == RESHAPE or _passes_upstream(n, edge, p)):
                    # a reshape's VJP, and a same-shape add's, is its upstream
                    # gradient itself: add it straight from n.grad, no scratch
                    # and no VJP launch (ACCUMULATE flushes on load exactly as
                    # the add's VJP would have before storing)
                    contributions.append(n.grad)
                elif p.requires_grad or n.op >= CUSTOM:
                    contributions.append(self._alloc(ex, p.rows * p.cols))
                else:
                    contributions.append(dummy_ptr())
            # A linear node's dL/dZ: the pre-activation seed at the root, its
            # gradient when a consumer already applied act', else formed here.
            var dz = n.grad
            if n.op == LINEAR:
                if i == root and preact:
                    dz = seed
                elif not n.grad_preact:
                    dz = self._linear_dz(ex, i)
            if n.op >= CUSTOM:
                var inputs = List[FP]()
                var rows = List[Int]()
                var cols = List[Int]()
                for edge in range(n.count):  # driver-loop: over node inputs
                    var p = self._nodes[self._parents[n.start + edge]]
                    inputs.append(p.value)
                    rows.append(p.rows)
                    cols.append(p.cols)
                    # BackwardRules receive distinct destinations, zeroed
                    # unless the rule overwrites (or ignores) that one, or
                    # writes the input's whole gradient there (writes_total:
                    # it may already hold other consumers' terms).
                    if not total[edge] and rules.zeroed_destination(n.op - CUSTOM, n.saved, edge):
                        _fill(ex, contributions[edge], p.rows * p.cols)
                rules.apply(ex, n.op - CUSTOM, n.saved, n.value, n.grad, inputs, contributions, rows, cols)
            for edge in range(n.count):  # driver-loop: over node inputs
                var parent = self._parents[n.start + edge]
                var p = self._nodes[parent]
                if not p.requires_grad:
                    continue
                # contribution is p.grad itself for a first contribution
                # (every body below, including _binary_vjp's blocked fold,
                # then writes the parent's gradient directly), else scratch.
                var contribution = contributions[edge]
                if n.op == RESHAPE:
                    # (no copy when the operand took this buffer above)
                    if first[edge] and Int(contribution) != Int(n.grad):
                        _copy(ex, contribution, n.grad, p.rows * p.cols)
                elif not first[edge] and _passes_upstream(n, edge, p):
                    pass  # the accumulate below adds n.grad itself
                elif n.op < CUSTOM:
                    if n.op == LINEAR:
                        self._linear_vjp(ex, n, edge, contribution, first[edge], dz, consumers)
                    elif n.op == MATMUL:
                        var left = self._nodes[self._parents[n.start]]
                        var right = self._nodes[self._parents[n.start + 1]]
                        if edge == 0:
                            _gemm(ex, contribution, n.grad, right.value, left.rows, left.cols, right.cols,
                                  right.cols, 1, 1, right.cols)
                        else:
                            _gemm(ex, contribution, left.value, n.grad, right.rows, right.cols, left.rows,
                                  1, left.cols, right.cols, 1)
                    elif n.op <= DIVIDE:
                        self._binary_vjp(ex, n, edge, contribution, p.rows, p.cols)
                    elif n.op == TRANSPOSE:
                        # n is (p.cols, p.rows); its upstream transposes back.
                        _transpose(ex, contribution, n.grad, n.rows, n.cols)
                    elif n.op == REDUCE:
                        var a = Args()
                        a.p0 = n.grad
                        a.p1 = contribution
                        a.i0 = n.saved.i0
                        a.i2 = p.cols
                        a.f0 = n.saved.f0
                        ex.launch[OP_AG_REDUCE_VJP](a, p.rows * p.cols)
                    elif n.op == DROPOUT:
                        # The mask is re-derived from the saved (seed,
                        # counter) slots, never stored (autograd/data_ops.mojo).
                        var a = n.saved
                        a.p0 = n.grad
                        a.p1 = contribution
                        ex.launch[OP_AG_DROPOUT_VJP](a, p.rows * p.cols)
                    elif n.op == LOSS_ELEM:
                        var a = Args()
                        a.p0 = self._nodes[self._parents[n.start]].value
                        a.p1 = self._nodes[self._parents[n.start + 1]].value
                        a.p2 = n.grad
                        a.p3 = contribution
                        a.i0 = n.saved.i0
                        a.i8 = edge
                        a.f0 = n.saved.f0
                        ex.launch[OP_LS_ELEM_VJP](a, p.rows * p.cols)
                    elif n.op == LOSS_CE:
                        # only the logits (edge 0): the codes never require a gradient
                        var a = Args()
                        a.p0 = p.value
                        a.p1 = self._nodes[self._parents[n.start + 1]].value
                        a.p2 = n.grad
                        a.p3 = contribution
                        a.i1 = p.cols
                        ex.launch[OP_LS_CE_VJP](a, p.rows)
                    elif n.op == LOSS_SCE:
                        # the logits only (edge 0), in one launch: the mean's
                        # factor rides in the row body (autograd/pass1_ops.mojo)
                        var a = Args()
                        a.p0 = p.value
                        a.p1 = self._nodes[self._parents[n.start + 1]].value
                        a.p2 = n.grad
                        a.p3 = contribution
                        a.i1 = p.cols
                        a.f0 = n.saved.f0
                        ex.launch[OP_P1_SCE_VJP](a, p.rows)
                    elif n.op == SOFTMAX or n.op == LOG_SOFTMAX:
                        var a = Args()
                        a.p0 = n.value
                        a.p1 = n.grad
                        a.p2 = contribution
                        a.i0 = 1 if n.op == LOG_SOFTMAX else 0
                        a.i1 = p.cols
                        ex.launch[OP_AG_SOFTMAX_VJP](a, p.rows)
                    else:
                        var a = Args()
                        if unary_vjp_reads_input(n.op):
                            a.p0 = p.value
                        if unary_vjp_reads_output(n.op):
                            a.p1 = n.value
                        a.p2 = n.grad
                        a.p3 = contribution
                        a.i0 = n.op
                        a.f0 = n.saved.f0
                        ex.launch[OP_AG_UNARY_VJP](a, p.rows * p.cols)
                if not first[edge]:
                    _accumulate(ex, p.grad, contribution, p.rows * p.cols)
        self._cursor = scratch_start
        for i in range(len(self._nodes)):  # driver-loop: over graph nodes
            var n = self._nodes[i]
            if not n.requires_grad:
                continue
            if touched[i]:
                self._nodes[i].grad_ready = True
            elif n.op == LEAF:
                # unreached differentiable leaf: its gradient is zero
                _fill(ex, n.grad, n.rows * n.cols)
                self._nodes[i].grad_ready = True
        if preact:
            # the root's gradient buffer was never written: the seed was used
            self._nodes[root].grad_ready = False
