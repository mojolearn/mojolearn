# SPDX-License-Identifier: Apache-2.0
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, dummy_ptr, OP_COPY, OP_FILL, OP_GEMM
from checks.numerics import identical_div
from autograd.ops import (
    OP_AUTOGRAD, FORWARD, VJP, ACCUMULATE, LEAF, ADD, SUBTRACT, MULTIPLY,
    DIVIDE, RELU, SIGMOID, TANH, EXP, LOG, SQRT, SCALE, TRANSPOSE, RESHAPE,
    REDUCE, SOFTMAX, LOG_SOFTMAX, MATMUL, CUSTOM,
)


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


trait BackwardRules:
    # Native extensions receive one distinct, zeroed destination per input.
    # They write VJPs there; the tape owns graph traversal and accumulation.
    # Return False for unknown tags so a backward refuses BEFORE touching grads.
    def supports(self, tag: Int) -> Bool:
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


def _element_args(anchor: FP) -> Args:
    # Bind unused slots to live storage as well. The retained Metal argument
    # probe uses this same convention, ruling out dummy addresses as the
    # cause of its binding failure (docs/plans/NATIVE_AUTOGRAD.md).
    var a = Args()
    a.p0 = anchor
    a.p1 = anchor
    a.p2 = anchor
    a.p3 = anchor
    a.p4 = anchor
    a.p5 = anchor
    a.p6 = anchor
    a.p7 = anchor
    a.p8 = anchor
    a.p9 = anchor
    a.p10 = anchor
    a.p11 = anchor
    return a


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
    """
    var _nodes: List[_Node]
    var _parents: List[Int]
    var _buffers: List[FP]
    var _capacities: List[Int]
    var _cursor: Int
    var _identity: Optional[MutPointer[Int, MutUntrackedOrigin]]
    var _generation: Int

    def __init__(out self):
        self._nodes = List[_Node]()
        self._parents = List[Int]()
        self._buffers = List[FP]()
        self._capacities = List[Int]()
        self._cursor = 0
        self._identity = None
        self._generation = 0

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
        self._shape(rows, cols)
        if not self._identity:
            self._identity = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(alloc[Int](1)))
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
        self._nodes.append(_Node(value, grad, rows, cols, requires, op, start, len(parents), saved, ready))
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
        if not n.requires_grad:
            raise Error("tensor does not require a gradient")
        if not n.grad_ready:
            raise Error("tensor has no gradient: run backward from an output that depends on it")
        return n.grad

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
        var args = _element_args(output)
        args.i1 = op
        args.i2 = rows
        args.i3 = cols
        args.i4 = a.rows
        args.i5 = a.cols
        args.i6 = b.rows
        args.i7 = b.cols
        args.p0 = a.value
        args.p1 = b.value
        args.p2 = output
        ex.launch[OP_AUTOGRAD](args, rows * cols)
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
        var a = _element_args(output)
        a.i1 = op
        a.i2 = n.rows
        a.i3 = n.cols
        a.i4 = n.rows
        a.i5 = n.cols
        a.p0 = n.value
        a.p2 = output
        a.f0 = scale
        var count = n.rows if op == SOFTMAX or op == LOG_SOFTMAX else n.rows * n.cols
        ex.launch[OP_AUTOGRAD](a, count)
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

    def transpose[E: Exec](mut self, mut ex: E, x: Tensor) raises -> Tensor:
        var n = self._nodes[self._check(x)]
        var output = self._alloc(ex, n.rows * n.cols)
        var a = _element_args(output)
        a.i1 = TRANSPOSE
        a.i2 = n.cols
        a.i3 = n.rows
        a.i4 = n.rows
        a.i5 = n.cols
        a.p0 = n.value
        a.p2 = output
        ex.launch[OP_AUTOGRAD](a, n.rows * n.cols)
        return self._record(ex, output, n.cols, n.rows, TRANSPOSE, [x], a)

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
        var a = _element_args(output)
        a.i1 = REDUCE
        a.i2 = rows
        a.i3 = cols
        a.i4 = n.rows
        a.i5 = n.cols
        a.i9 = axis
        a.f0 = identical_div(Float32(1), Float32(count)) if mean else Float32(1)
        a.p0 = n.value
        a.p2 = output
        ex.launch[OP_AUTOGRAD](a, rows * cols)
        return self._record(ex, output, rows, cols, REDUCE, [x], a)

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

    def backward[E: Exec](mut self, mut ex: E, output: Tensor, accumulate: Bool = False) raises:
        """Differentiate a scalar, clearing gradients unless accumulate=True."""
        self.backward_scalar_with_rules(ex, output, NoCustomRules(), accumulate)

    def backward_scalar_with_rules[E: Exec, R: BackwardRules](
        mut self, mut ex: E, output: Tensor, rules: R, accumulate: Bool = False,
    ) raises:
        """Differentiate a scalar using native custom backward rules."""
        var n = self._nodes[self._check(output)]
        if n.rows * n.cols != 1:
            raise Error("non-scalar backward requires an explicit seed tensor")
        var cursor = self._cursor
        var seed = self._alloc(ex, 1)
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

    def backward_with_rules[E: Exec, R: BackwardRules](
        mut self, mut ex: E, output: Tensor, seed: Tensor, rules: R, accumulate: Bool = False,
    ) raises:
        var n = self._nodes[self._check(output)]
        var s = self._nodes[self._check(seed)]
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
    ) raises:
        var root = self._check(output)
        if not self._nodes[root].requires_grad:
            raise Error("output has no differentiable ancestors")
        var active = List[Bool](length=len(self._nodes), fill=False)
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
            var n = self._nodes[i]
            if not n.requires_grad:
                continue
            if accumulate and n.op == LEAF:
                if not n.grad_ready:
                    _fill(ex, n.grad, n.rows * n.cols)
                    self._nodes[i].grad_ready = True
                touched[i] = True
            else:
                self._nodes[i].grad_ready = False
        # Preserve the seed before overwriting a leaf root's gradient.
        if touched[root]:
            var a = _element_args(self._nodes[root].grad)
            a.i0 = ACCUMULATE
            a.p0 = seed
            a.p1 = self._nodes[root].grad
            ex.launch[OP_AUTOGRAD](a, self._nodes[root].rows * self._nodes[root].cols)
        else:
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
            var contributions = List[FP]()
            for edge in range(n.count):  # driver-loop: over node inputs
                var parent = self._parents[n.start + edge]
                var p = self._nodes[parent]
                if p.requires_grad and not touched[parent]:
                    first[edge] = True
                    touched[parent] = True
                    contributions.append(p.grad)
                elif p.requires_grad or n.op >= CUSTOM:
                    contributions.append(self._alloc(ex, p.rows * p.cols))
                else:
                    contributions.append(dummy_ptr())
            if n.op >= CUSTOM:
                var inputs = List[FP]()
                var rows = List[Int]()
                var cols = List[Int]()
                for edge in range(n.count):  # driver-loop: over node inputs
                    var p = self._nodes[self._parents[n.start + edge]]
                    inputs.append(p.value)
                    rows.append(p.rows)
                    cols.append(p.cols)
                    # BackwardRules receive zeroed, distinct destinations.
                    _fill(ex, contributions[edge], p.rows * p.cols)
                rules.apply(ex, n.op - CUSTOM, n.saved, n.value, n.grad, inputs, contributions, rows, cols)
            for edge in range(n.count):  # driver-loop: over node inputs
                var parent = self._parents[n.start + edge]
                var p = self._nodes[parent]
                if not p.requires_grad:
                    continue
                var contribution = contributions[edge]
                if n.op < CUSTOM:
                    if n.op == MATMUL:
                        var left = self._nodes[self._parents[n.start]]
                        var right = self._nodes[self._parents[n.start + 1]]
                        if edge == 0:
                            _gemm(ex, contribution, n.grad, right.value, left.rows, left.cols, right.cols,
                                  right.cols, 1, 1, right.cols)
                        else:
                            _gemm(ex, contribution, left.value, n.grad, right.rows, right.cols, left.rows,
                                  1, left.cols, right.cols, 1)
                    else:
                        var a = n.saved
                        a.i0 = VJP
                        a.i1 = n.op
                        a.i8 = edge
                        a.p0 = self._nodes[self._parents[n.start]].value
                        a.p1 = self._nodes[self._parents[n.start + 1]].value if n.count > 1 else a.p0
                        a.p2 = n.value
                        a.p3 = n.grad
                        a.p4 = contribution
                        a.p5 = n.value
                        a.p6 = n.value
                        a.p7 = n.value
                        a.p8 = n.value
                        a.p9 = n.value
                        a.p10 = n.value
                        a.p11 = n.value
                        var count = p.rows if n.op == SOFTMAX or n.op == LOG_SOFTMAX else p.rows * p.cols
                        ex.launch[OP_AUTOGRAD](a, count)
                if not first[edge]:
                    var a = _element_args(p.grad)
                    a.i0 = ACCUMULATE
                    a.p0 = contribution
                    a.p1 = p.grad
                    ex.launch[OP_AUTOGRAD](a, p.rows * p.cols)
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
