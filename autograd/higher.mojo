# SPDX-License-Identifier: Apache-2.0
# HIGHER-ORDER GRADIENTS: THE RECORDED BACKWARD (lane ag-ho-core, H1 of
# docs/plans/AUTOGRAD_HIGHER_ORDER.md).
#
# backward_recorded walks the graph exactly as the first-order
# Tape._backward does (the active ancestors of the root, reverse creation
# order, the edges of one node ascending) but writes every VJP as recorded
# tape operations (lower_vjp, the plan's section 1.2 table): each gradient
# is then a node of the graph, and a later backward from a scalar built on
# it gives second (or third) derivatives. The first-order path is untouched:
# nothing here writes a gradient buffer of an existing interior node,
# borrows a buffer, pairs GEMMs or rewinds scratch. Every recorded gradient
# lives in tape storage until reset().
#
# Same bits on every vendor by construction: every recorded VJP is a
# composition of tape operations whose kernels are already cross-vendor
# identical (one fold order through Tape.sum's blocked fold, elementwise
# bodies with ftz), the new element bodies are compares and selects
# (autograd/higher_ops.mojo), and several contributions to one gradient are
# added by tape.add in the first-order canonical order (decreasing
# consumer, increasing edge), the arithmetic OP_AG_ACCUMULATE performs. No
# vendor branch and no size rule anywhere in this module.
#
# GPU path: every launch is a tape operation's parallel kernel; the loops
# here are driver loops over graph nodes and node inputs. No download.
#
# First-order values under create_graph may differ from the fused path in a
# few words (one version, every vendor together; reported, never gated):
# relu's recorded VJP g * indicator(x) gives -0.0 where the fused kernel
# stores +0.0 for a negative upstream at x <= 0; the softmax VJP's row dot is
# the blocked fold here and a serial chain in OP_AG_SOFTMAX_VJP (equal up to
# one fold block of columns); intermediates stored here are flushed (ftz)
# where the fused kernel keeps them in a register.
from sequence.exec_trait import Exec
from sequence.ops import FP, Args
from autograd.tape import Tape, Tensor, _fill, _copy, _INFERRED_REFUSAL
from autograd.ops import (
    LEAF, ADD, SUBTRACT, MULTIPLY, DIVIDE, RELU, SIGMOID, TANH, EXP, LOG, SQRT, SCALE, TRANSPOSE,
    RESHAPE, REDUCE, SOFTMAX, LOG_SOFTMAX, MATMUL, LINEAR, CUSTOM, AXIS_ALL, AXIS_ROWS, AXIS_COLS,
    LOSS_ELEM, LOSS_CE, LOSS_SCE, LOSS_REDUCED, OP_AG_REDUCE_VJP,
)
from autograd.data_ops import DROPOUT
from autograd.embedding_ops import EMBEDDING
from autograd.higher_ops import (
    EXPAND, CLAMP, OP_HO_INDICATOR, OP_HO_SIGN, OP_HO_CLAMP_FWD, OP_HO_INSIDE, higher_refusal,
)
from autograd.higher_fused import lower_fused_vjp
from autograd.layers import LAYER_NORM


# ---------------------------------------------------------------- operations
def expand[E: Exec](
    mut tape: Tape, mut ex: E, g: Tensor, rows: Int, cols: Int, axis: Int, factor: Float32 = 1,
) raises -> Tensor:
    """Broadcast a reduced tensor back to rows x cols, times `factor`.

    `axis` is the axis a sum or mean reduced: -1 (all; g is 1 x 1), 0
    (rows; g is 1 x cols) or 1 (columns; g is rows x 1). Element (r, c) of
    the result is g's element on that row or column times `factor`. Its
    gradient sums the upstream gradient over the same axis, times
    `factor`.
    """
    if axis < -1 or axis > 1:
        raise Error("expand: axis must be -1 (all), 0, or 1")
    tape._shape(rows, cols)
    var gi = tape._use(ex, g)
    var n = tape._nodes[gi]
    var want_rows = rows if axis == AXIS_COLS else 1
    var want_cols = cols if axis == AXIS_ROWS else 1
    if n.rows != want_rows or n.cols != want_cols:
        raise Error("expand: the tensor's shape is not the target shape reduced over the axis")
    var count = rows * cols
    var output = tape._alloc(ex, count)
    # the reduce VJP's body: one thread per output element, the broadcast read
    var a = Args()
    a.p0 = n.value
    a.p1 = output
    a.i0 = axis
    a.i2 = cols
    a.f0 = factor
    ex.launch[OP_AG_REDUCE_VJP](a, count)
    var saved = Args()
    saved.i0 = axis
    saved.f0 = factor
    return tape._record(ex, output, rows, cols, EXPAND, [g], saved)


def clamp[E: Exec](mut tape: Tape, mut ex: E, x: Tensor, bound: Float32) raises -> Tensor:
    """Clip every element of x to [-bound, bound] (bound > 0).

    The gradient passes where |x| < bound and is zero elsewhere.
    """
    if not (bound > 0):
        raise Error("clamp: the bound must be positive")
    var xi = tape._use(ex, x)
    var n = tape._nodes[xi]
    var count = n.rows * n.cols
    var output = tape._alloc(ex, count)
    var a = Args()
    a.p0 = n.value
    a.p1 = output
    a.f0 = bound
    ex.launch[OP_HO_CLAMP_FWD](a, count)
    var saved = Args()
    saved.f0 = bound
    return tape._record(ex, output, n.rows, n.cols, CLAMP, [x], saved)


def _constant[E: Exec](mut tape: Tape, mut ex: E, x: Tensor, op: Int, bound: Float32) raises -> Tensor:
    """An elementwise function of x's values recorded as a constant leaf
    (no graph, as detach records one): the second derivative through it is
    zero, which is the mathematics of a step function."""
    var xi = tape._use(ex, x)
    var n = tape._nodes[xi]
    var count = n.rows * n.cols
    var output = tape._alloc(ex, count)
    var a = Args()
    a.p0 = n.value
    a.p1 = output
    a.f0 = bound
    if op == OP_HO_INDICATOR:
        ex.launch[OP_HO_INDICATOR](a, count)
    elif op == OP_HO_SIGN:
        ex.launch[OP_HO_SIGN](a, count)
    else:
        ex.launch[OP_HO_INSIDE](a, count)
    return tape.variable(ex, output, n.rows, n.cols, False)


def indicator[E: Exec](mut tape: Tape, mut ex: E, x: Tensor) raises -> Tensor:
    """A constant tensor: 1 where x > 0, else 0 (no gradient flows through it)."""
    return _constant(tape, ex, x, OP_HO_INDICATOR, Float32(0))


def sign[E: Exec](mut tape: Tape, mut ex: E, x: Tensor) raises -> Tensor:
    """A constant tensor: 1 where x > 0, -1 where x < 0, 0 where x is zero."""
    return _constant(tape, ex, x, OP_HO_SIGN, Float32(0))


def inside[E: Exec](mut tape: Tape, mut ex: E, x: Tensor, bound: Float32) raises -> Tensor:
    """A constant tensor: 1 where |x| < bound, else 0."""
    return _constant(tape, ex, x, OP_HO_INSIDE, bound)


def _fold_to[E: Exec](mut tape: Tape, mut ex: E, c: Tensor, rows: Int, cols: Int) raises -> Tensor:
    """c (a binary node's output shape) summed over the axes an operand of
    shape rows x cols was broadcast along: the blocked fold the first-order
    _binary_vjp launches, as a recorded sum. c itself when nothing broadcast."""
    var reduce_rows = rows < tape.rows(c)
    var reduce_cols = cols < tape.cols(c)
    if not reduce_rows and not reduce_cols:
        return c
    var axis = AXIS_ALL if reduce_rows and reduce_cols else (AXIS_ROWS if reduce_rows else AXIS_COLS)
    return tape.sum(ex, c, axis)


# ---------------------------------------------------------------- lowering
def _layer_norm_vjp[E: Exec](
    mut tape: Tape, mut ex: E, i: Int, upstream: Tensor, edge: Int,
) raises -> Tensor:
    # Recompute normalized inputs as recorded primitives. The ordinary
    # backward keeps its existing fused kernels; only create_graph uses this
    # expression, whose reductions have the tape's vendor-independent order.
    var n = tape._nodes[i]
    var x = tape._handle(tape._parents[n.start])
    var w = tape._handle(tape._parents[n.start + 1])
    var xm = tape.reshape(ex, x, n.rows, n.cols)
    var g = tape.reshape(ex, upstream, n.rows, n.cols)
    if edge == 2:
        return tape.sum(ex, g, AXIS_ROWS)
    var center = tape.subtract(ex, xm, tape.mean(ex, xm, AXIS_COLS))
    var square = tape.multiply(ex, center, center)
    var variance = tape.mean(ex, square, AXIS_COLS)
    var eps = tape.scalar(ex, n.saved.f0)
    var denom = tape.sqrt(ex, tape.add(ex, variance, eps))
    var norm = tape.divide(ex, center, denom)
    if edge == 1:
        return tape.sum(ex, tape.multiply(ex, g, norm), AXIS_ROWS)
    var weighted = tape.multiply(ex, g, w)
    var average = tape.mean(ex, weighted, AXIS_COLS)
    var cov = tape.mean(ex, tape.multiply(ex, weighted, norm), AXIS_COLS)
    var residual = tape.subtract(ex, tape.subtract(ex, weighted, average), tape.multiply(ex, norm, cov))
    return tape.divide(ex, residual, denom)


def lower_vjp[E: Exec](mut tape: Tape, mut ex: E, i: Int, upstream: Tensor, edge: Int) raises -> Tensor:
    """The VJP of node i into its input `edge` for the upstream gradient
    tensor `upstream` (node i's shape), as recorded tape operations: a
    tensor of the input's shape (docs/plans/AUTOGRAD_HIGHER_ORDER.md 1.2).
    The fused kinds go to lower_fused_vjp; custom rules are refused."""
    var n = tape._nodes[i]
    var op = n.op
    if op == CUSTOM + LAYER_NORM:
        return _layer_norm_vjp(tape, ex, i, upstream, edge)
    if op >= CUSTOM:
        raise Error(higher_refusal(op))
    if (op == LINEAR or op == LOSS_ELEM or op == LOSS_REDUCED or op == LOSS_CE or op == LOSS_SCE
            or op == DROPOUT or op == EMBEDDING):
        return lower_fused_vjp(tape, ex, i, upstream, edge)
    var parent = tape._parents[n.start + edge]
    var p = tape._nodes[parent]
    var g = upstream
    var z = tape._handle(i)
    if op >= ADD and op <= DIVIDE:
        var left = tape._handle(tape._parents[n.start])
        var right = tape._handle(tape._parents[n.start + 1])
        if op == ADD or (op == SUBTRACT and edge == 0):
            return _fold_to(tape, ex, g, p.rows, p.cols)
        if op == SUBTRACT:
            # fold(-g), as the kernel: the sign of a zero sum follows the terms
            return _fold_to(tape, ex, tape.scale(ex, g, Float32(-1)), p.rows, p.cols)
        if op == MULTIPLY:
            if edge == 0:
                return _fold_to(tape, ex, tape.multiply(ex, g, right), p.rows, p.cols)
            return _fold_to(tape, ex, tape.multiply(ex, g, left), p.rows, p.cols)
        if edge == 0:
            return _fold_to(tape, ex, tape.divide(ex, g, right), p.rows, p.cols)
        # -(g z) / y: the kernel's _bin_vjp_at algebra (a multiply by -1 is
        # the exact negation)
        var q = tape.divide(ex, tape.multiply(ex, g, z), right)
        return _fold_to(tape, ex, tape.scale(ex, q, Float32(-1)), p.rows, p.cols)
    var x = tape._handle(parent)
    if op == RELU:
        return tape.multiply(ex, g, indicator(tape, ex, x))
    if op == SIGMOID:
        var one = tape.scalar(ex, Float32(1))
        return tape.multiply(ex, tape.multiply(ex, g, z), tape.subtract(ex, one, z))
    if op == TANH:
        var one = tape.scalar(ex, Float32(1))
        return tape.multiply(ex, g, tape.subtract(ex, one, tape.multiply(ex, z, z)))
    if op == EXP:
        return tape.multiply(ex, g, z)
    if op == LOG:
        return tape.divide(ex, g, x)
    if op == SQRT:
        return tape.divide(ex, g, tape.scale(ex, z, Float32(2)))
    if op == SCALE:
        return tape.scale(ex, g, n.saved.f0)
    if op == TRANSPOSE:
        return tape.transpose(ex, g)
    if op == RESHAPE:
        return tape.reshape(ex, g, p.rows, p.cols)
    if op == REDUCE:
        return expand(tape, ex, g, p.rows, p.cols, n.saved.i0, n.saved.f0)
    if op == SOFTMAX:
        var dot = tape.sum(ex, tape.multiply(ex, g, z), AXIS_COLS)
        return tape.multiply(ex, z, tape.subtract(ex, g, dot))
    if op == LOG_SOFTMAX:
        var total = tape.sum(ex, g, AXIS_COLS)
        return tape.subtract(ex, g, tape.multiply(ex, tape.exp(ex, z), total))
    if op == MATMUL:
        if edge == 0:
            # dX = G @ Y^T, Y^T a view: the first-order GEMM's strides
            var w = tape._handle(tape._parents[n.start + 1])
            return tape.matmul(ex, g, tape.transpose(ex, w))
        var xl = tape._handle(tape._parents[n.start])
        return tape.matmul(ex, tape.transpose(ex, xl), g)
    if op == EXPAND:
        var s = tape.sum(ex, g, n.saved.i0)
        if n.saved.f0 != Float32(1):
            return tape.scale(ex, s, n.saved.f0)
        return s
    if op == CLAMP:
        return tape.multiply(ex, g, inside(tape, ex, x, n.saved.f0))
    raise Error("autograd: no backward rule for operation kind " + String(op))


def _recorded_kind(op: Int) -> Bool:
    """A node kind below CUSTOM that lower_vjp knows (the fused kinds go to
    lower_fused_vjp, which may refuse them)."""
    return (op == LEAF or (op >= ADD and op <= DIVIDE) or (op >= RELU and op <= SCALE) or op == TRANSPOSE
            or op == RESHAPE or op == REDUCE or op == SOFTMAX or op == LOG_SOFTMAX or op == MATMUL
            or op == EXPAND or op == CLAMP or op == LINEAR or op == LOSS_ELEM or op == LOSS_REDUCED
            or op == LOSS_CE or op == LOSS_SCE or op == DROPOUT or op == EMBEDDING)


# ---------------------------------------------------------------- traversal
def backward_recorded[E: Exec](mut tape: Tape, mut ex: E, output: Tensor, seed: Tensor) raises:
    """Differentiate `output` along `seed` (output's shape) and record the
    computation, so every gradient is a differentiable tensor.

    Afterwards tape.grad_of(x) is x's gradient as a tensor of the graph and
    tape.gradient(x) / gradients(ex, x) read its values. A backward from a
    scalar built on those tensors gives second derivatives. The seed may
    itself require a gradient. Every recorded tensor lives until reset().
    """
    var root = tape._check(output)
    var s_at = tape._use(ex, seed)
    var rn = tape._nodes[root]
    var sn = tape._nodes[s_at]
    if rn.inferred or sn.inferred:
        raise Error(_INFERRED_REFUSAL)
    if not tape.is_recording():
        raise Error("create_graph records the gradient computation: call record() before backward")
    if rn.rows != sn.rows or rn.cols != sn.cols:
        raise Error("backward seed shape must equal output shape")
    tape._refuse_stale(root)
    if not rn.requires_grad:
        raise Error("output has no differentiable ancestors")
    # a later first-order backward writes every recorded gradient buffer:
    # the seed (a node value of the extended graph) may not be one
    tape._refuse_overlap(sn.value, sn.rows * sn.cols, False, "backward seed")
    var active = List[Bool]()
    var consumers = List[Int]()
    tape._mark_active(root, active, consumers)
    # Refusals before any launch: a custom rule (section 2) or a kind with
    # no rule. The fused kinds refuse inside lower_fused_vjp until lane H2.
    for i in range(root + 1):  # driver-loop: over graph nodes
        if not active[i]:
            continue
        var kind = tape._nodes[i].op
        if kind == CUSTOM + LAYER_NORM:
            continue
        if kind >= CUSTOM:
            raise Error(higher_refusal(kind))
        if not _recorded_kind(kind):
            raise Error("autograd: no backward rule for operation kind " + String(kind))
    var nodes0 = len(tape._nodes)
    var parents0 = len(tape._parents)
    var cursor0 = tape._cursor
    # grad_node[j]: the node holding node j's gradient so far (-1: none)
    var grad_node = List[Int](length=nodes0, fill=-1)
    grad_node[root] = s_at
    try:
        for reverse in range(root + 1):  # driver-loop: over graph nodes, reverse creation order
            var i = root - reverse
            if not active[i]:
                continue
            var n = tape._nodes[i]
            if n.op == LEAF:
                continue
            if grad_node[i] < 0:
                raise Error("autograd: internal: an active node has no recorded gradient")
            var upstream = tape._handle(grad_node[i])
            for edge in range(n.count):  # driver-loop: over node inputs
                var parent = tape._parents[n.start + edge]
                var p = tape._nodes[parent]
                if not p.requires_grad:
                    continue
                var c = lower_vjp(tape, ex, i, upstream, edge)
                if tape.rows(c) != p.rows or tape.cols(c) != p.cols:
                    raise Error("autograd: internal: a recorded VJP has the wrong shape for operation kind "
                                + String(n.op))
                # the canonical order: the first contribution is the
                # gradient, later ones are added to it ((c1 + c2) + c3)
                if grad_node[parent] < 0:
                    grad_node[parent] = c._index
                else:
                    grad_node[parent] = tape.add(ex, tape._handle(grad_node[parent]), c)._index
        # Differentiable leaves: an unreached one gets a recorded zero; each
        # leaf's gradient buffer also receives the values (caller storage
        # from variable_into keeps its contract; accumulate=True later adds
        # to them).
        for i in range(nodes0):  # driver-loop: over graph nodes
            var n = tape._nodes[i]
            if n.op != LEAF or n.inferred or not n.requires_grad:
                continue
            var count = n.rows * n.cols
            if grad_node[i] < 0:
                var zero = tape._alloc(ex, count)
                _fill(ex, zero, count)
                grad_node[i] = tape.variable(ex, zero, n.rows, n.cols, False)._index
                _fill(ex, n.own_grad, count)
            else:
                var held = tape.value_of(ex, tape._handle(grad_node[i]))
                if Int(held) != Int(n.own_grad):
                    _copy(ex, n.own_grad, held, count)
            tape._nodes[i].grad = n.own_grad
            tape._nodes[i].grad_ready = True
            tape._nodes[i].grad_preact = False
        # a gradient recorded as a transpose view is copied once, so its
        # value buffer holds it for gradient() and every dense reader
        for i in range(nodes0):  # driver-loop: over graph nodes
            if grad_node[i] >= 0:
                tape._materialize_at(ex, grad_node[i])
    except e:
        tape._rollback(nodes0, parents0, cursor0)
        raise e
    var table = List[Int](length=len(tape._nodes), fill=-1)
    for i in range(nodes0):  # driver-loop: over graph nodes
        table[i] = grad_node[i]
    tape._grad_node = table^
    tape._recorded_live = True


def backward_scalar_recorded[E: Exec](mut tape: Tape, mut ex: E, output: Tensor) raises:
    """backward_recorded from a scalar output with the seed 1.

    Example::

        var y = tape.multiply(ex, tape.multiply(ex, x, x), x)
        backward_scalar_recorded(tape, ex, y)          # dy/dx = 3 x^2, a tensor
        tape.backward(ex, tape.grad_of(x))             # d2y/dx2 = 6 x
        var second = tape.gradients(ex, x)
    """
    var at = tape._check(output)
    var n = tape._nodes[at]
    if n.inferred:
        raise Error(_INFERRED_REFUSAL)
    if n.rows * n.cols != 1:
        raise Error("non-scalar backward requires an explicit seed tensor")
    tape._refuse_stale(at)
    if not n.requires_grad:
        raise Error("output has no differentiable ancestors")
    if not tape.is_recording():
        raise Error("create_graph records the gradient computation: call record() before backward")
    var nodes0 = len(tape._nodes)
    var parents0 = len(tape._parents)
    var cursor0 = tape._cursor
    # the seed is a constant leaf of the graph
    var seed = tape.scalar(ex, Float32(1))
    try:
        backward_recorded(tape, ex, output, seed)
    except e:
        tape._rollback(nodes0, parents0, cursor0)
        raise e
