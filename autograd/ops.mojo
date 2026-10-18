# SPDX-License-Identifier: Apache-2.0
# Native element bodies shared by HostExec and DeviceExec. Graph traversal
# and gradient accumulation order live in tape.mojo, not inside GPU atomics.
#
# One operation code per body (the library's design, sequence/dispatch.mojo).
# Each body reads and writes only the pointer slots listed in its comment;
# the tape binds every other slot to the Args() placeholder, as every other
# sequence operation does, and never binds one buffer to two slots of a
# launch. No body reads back a slot it writes, except OP_AG_ACCUMULATE's
# element-wise read-modify-write of its own destination word.
from sequence.ops import Args, FP, ld, st, add, sub, mul
from checks.numerics import (
    ftz, identical_div, identical_exp, identical_log, identical_sigmoid, identical_sqrt,
    identical_tanh,
)

# Sequence codes 120/121 are recurrent scan kernels, intercepted before the
# generic dispatcher. Codes OP_AG_FIRST..OP_AG_LAST are this family, on both
# executors (sequence/dispatch.mojo asserts the range is free).
comptime OP_AG_BIN_FWD = 1000
comptime OP_AG_BIN_VJP = 1001
comptime OP_AG_UNARY_FWD = 1002
comptime OP_AG_UNARY_VJP = 1003
comptime OP_AG_TRANSPOSE = 1004
comptime OP_AG_REDUCE_PART = 1005
comptime OP_AG_REDUCE_FOLD = 1006
comptime OP_AG_REDUCE_VJP = 1007
comptime OP_AG_SOFTMAX_FWD = 1008
comptime OP_AG_SOFTMAX_VJP = 1009
comptime OP_AG_ACCUMULATE = 1010
comptime OP_AG_FIRST = OP_AG_BIN_FWD
comptime OP_AG_LAST = OP_AG_ACCUMULATE

# Node kinds recorded by the tape. Binary and unary kinds also travel as a
# runtime integer (a.i1 / a.i0) inside their operation's body.
comptime ADD = 1
comptime SUBTRACT = 2
comptime MULTIPLY = 3
comptime DIVIDE = 4
comptime RELU = 10
comptime SIGMOID = 11
comptime TANH = 12
comptime EXP = 13
comptime LOG = 14
comptime SQRT = 15
comptime SCALE = 16
comptime TRANSPOSE = 20
comptime RESHAPE = 21
comptime REDUCE = 22
comptime SOFTMAX = 23
comptime LOG_SOFTMAX = 24
comptime MATMUL = 30
comptime CUSTOM = 40
comptime LEAF = 0

# Reduction axes (a.i0 of the reduce bodies).
comptime AXIS_ALL = -1
comptime AXIS_ROWS = 0
comptime AXIS_COLS = 1


@always_inline
def ag_fold_block(extent: Int) -> Int:
    """Elements per block of a blocked fold over `extent` terms: the smallest
    power of two R >= 64 with R * R >= extent (the ln_fold_rows rule,
    sequence/layernorm.mojo). It depends on the folded extent alone, never on
    a vendor or a benchmark shape: the block count and the block length both
    grow as sqrt(extent), so the partial launch has ~sqrt(extent) threads per
    output and the fold chain ~sqrt(extent) terms. An extent of at most 64
    is one block (one ascending chain)."""
    var r = 64
    while r * r < extent:
        r *= 2
    return r


@always_inline
def ag_fold_blocks(extent: Int) -> Int:
    var b = ag_fold_block(extent)
    return (extent + b - 1) // b


@always_inline
def div(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(a, b))


@always_inline
def _index(row: Int, col: Int, rows: Int, cols: Int) -> Int:
    return (0 if rows == 1 else row) * cols + (0 if cols == 1 else col)


@always_inline
def _right(a: Args) -> FP:
    # i10 = 1: the right operand is the left operand's buffer (x * x). The
    # tape then leaves p1 unbound instead of binding one buffer twice.
    return a.p0 if a.i10 == 1 else a.p1


@always_inline
def _unary(op: Int, x: Float32, scale: Float32) -> Float32:
    if op == RELU:
        return x if x > 0 else Float32(0)
    if op == SIGMOID:
        return ftz(identical_sigmoid(x))
    if op == TANH:
        return ftz(identical_tanh(x))
    if op == EXP:
        return ftz(identical_exp(x))
    if op == LOG:
        return ftz(identical_log(x))
    if op == SQRT:
        return ftz(identical_sqrt(x))
    return mul(x, scale)


@always_inline
def unary_vjp_reads_input(op: Int) -> Bool:
    return op == RELU or op == LOG


@always_inline
def unary_vjp_reads_output(op: Int) -> Bool:
    return op == SIGMOID or op == TANH or op == EXP or op == SQRT


@always_inline
def op_ag_bin_fwd(t: Int, a: Args):
    """p0 x, p1 y (unbound when i10 = 1), p2 z. i1 kind, i3 output cols,
    i4/i5 x shape, i6/i7 y shape. Thread t = output element t."""
    var row = t // a.i3
    var col = t % a.i3
    var x = ld(a.p0, _index(row, col, a.i4, a.i5))
    var y = ld(_right(a), _index(row, col, a.i6, a.i7))
    var op = a.i1
    var z: Float32
    if op == ADD:
        z = add(x, y)
    elif op == SUBTRACT:
        z = sub(x, y)
    elif op == MULTIPLY:
        z = mul(x, y)
    else:
        z = div(x, y)
    st(a.p2, t, z)


@always_inline
def op_ag_bin_vjp(t: Int, a: Args):
    """The elementwise VJP at output element t, for operand i8, written at
    output index t of p4 (output shape i2 x i3). A broadcast operand's
    gradient is then a blocked row/column/total reduction of p4 (tape.mojo).
    Slots read by kind: ADD/SUBTRACT p3 only; MULTIPLY p0 and p1 (the other
    operand); DIVIDE p1 and, for operand 1, p2 (the forward output z).
    p0 x, p1 y (unbound when i10 = 1), p2 z, p3 upstream, p4 destination."""
    var op = a.i1
    var row = t // a.i3
    var col = t % a.i3
    var g = ld(a.p3, t)
    if op == SUBTRACT and a.i8 == 1:
        g = -g
    elif op == MULTIPLY:
        if a.i8 == 0:
            g = mul(g, ld(_right(a), _index(row, col, a.i6, a.i7)))
        else:
            g = mul(g, ld(a.p0, _index(row, col, a.i4, a.i5)))
    elif op == DIVIDE:
        var y = ld(a.p1, _index(row, col, a.i6, a.i7))
        if a.i8 == 0:
            g = div(g, y)
        else:
            g = -div(mul(g, ld(a.p2, t)), y)
    st(a.p4, t, g)


@always_inline
def op_ag_unary_fwd(t: Int, a: Args):
    """p0 x, p1 y. i0 kind, f0 scale (SCALE). Thread t = element t."""
    st(a.p1, t, _unary(a.i0, ld(a.p0, t), a.f0))


@always_inline
def op_ag_unary_vjp(t: Int, a: Args):
    """p0 forward input (RELU, LOG only), p1 forward output (SIGMOID, TANH,
    EXP, SQRT only), p2 upstream, p3 destination. i0 kind, f0 scale."""
    var op = a.i0
    var g = ld(a.p2, t)
    var r: Float32
    if op == RELU:
        r = g if ld(a.p0, t) > 0 else Float32(0)
    elif op == SIGMOID:
        var y = ld(a.p1, t)
        r = mul(mul(g, y), sub(Float32(1), y))
    elif op == TANH:
        var y = ld(a.p1, t)
        r = mul(g, sub(Float32(1), mul(y, y)))
    elif op == EXP:
        r = mul(g, ld(a.p1, t))
    elif op == LOG:
        r = div(g, ld(a.p0, t))
    elif op == SQRT:
        r = div(g, mul(Float32(2), ld(a.p1, t)))
    else:
        r = mul(g, a.f0)
    st(a.p3, t, r)


@always_inline
def op_ag_transpose(t: Int, a: Args):
    """p0 source (i0 rows x i1 cols), p1 destination (i1 x i0). Forward and
    VJP are the same transpose. Thread t = destination element t."""
    st(a.p1, t, ld(a.p0, (t % a.i0) * a.i1 + t // a.i0))


@always_inline
def op_ag_reduce_part(t: Int, a: Args):
    """Partial sums of a blocked fold. p0 source (i1 rows x i2 cols), p1
    partials, i0 axis, i3 block length B (ag_fold_block of the folded
    extent). Each partial is its block's terms added ascending from +0.0.
      AXIS_ALL:  thread t = block t of the i1*i2 elements; p1[t].
      AXIS_ROWS: thread t = (block t // i2, column t % i2), folding rows;
                 p1[t] (block-major).
      AXIS_COLS: thread t = (row t // nb, block t % nb), folding columns;
                 p1[t] (row-major), nb = ceil(i2 / B)."""
    var rows = a.i1
    var cols = a.i2
    var B = a.i3
    var s = Float32(0)
    if a.i0 == AXIS_ALL:
        var lo = t * B
        var hi = min(lo + B, rows * cols)
        for k in range(lo, hi):
            s = add(s, ld(a.p0, k))
    elif a.i0 == AXIS_ROWS:
        var c = t % cols
        var lo = (t // cols) * B
        var hi = min(lo + B, rows)
        for r in range(lo, hi):
            s = add(s, ld(a.p0, r * cols + c))
    else:
        var nb = (cols + B - 1) // B
        var r = t // nb
        var lo = (t % nb) * B
        var hi = min(lo + B, cols)
        for k in range(lo, hi):
            s = add(s, ld(a.p0, r * cols + k))
    st(a.p1, t, s)


@always_inline
def op_ag_reduce_fold(t: Int, a: Args):
    """Second launch of the blocked fold: p1[t] = (the partials of output t,
    p0, added ascending from +0.0) * f0. i0 axis, i1 rows, i2 cols, i3 B as
    in op_ag_reduce_part. AXIS_ALL: one output; AXIS_ROWS: thread t =
    column t; AXIS_COLS: thread t = row t."""
    var rows = a.i1
    var cols = a.i2
    var B = a.i3
    var s = Float32(0)
    if a.i0 == AXIS_ALL:
        var nb = (rows * cols + B - 1) // B
        for k in range(nb):
            s = add(s, ld(a.p0, k))
    elif a.i0 == AXIS_ROWS:
        var nb = (rows + B - 1) // B
        for k in range(nb):
            s = add(s, ld(a.p0, k * cols + t))
    else:
        var nb = (cols + B - 1) // B
        for k in range(nb):
            s = add(s, ld(a.p0, t * nb + k))
    st(a.p1, t, mul(s, a.f0))


@always_inline
def op_ag_reduce_vjp(t: Int, a: Args):
    """p0 upstream (reduced shape), p1 destination (input shape, i2 cols).
    i0 axis, f0 factor. Thread t = input element t."""
    var i = 0
    if a.i0 == AXIS_ROWS:
        i = t % a.i2
    elif a.i0 == AXIS_COLS:
        i = t // a.i2
    st(a.p1, t, mul(ld(a.p0, i), a.f0))


@always_inline
def op_ag_softmax_fwd(t: Int, a: Args):
    """p0 x, p1 y, i0 = 1 for log-softmax, i1 cols. Thread t = row t. The
    exponentials are recomputed in the last pass rather than read back
    from p1, so the body never reads a slot it writes."""
    var cols = a.i1
    var base = t * cols
    var maximum = ld(a.p0, base)
    for k in range(1, cols):
        maximum = max(maximum, ld(a.p0, base + k))
    var total = Float32(0)
    for k in range(cols):
        total = add(total, ftz(identical_exp(sub(ld(a.p0, base + k), maximum))))
    if a.i0 == 0:
        for k in range(cols):
            var e = ftz(identical_exp(sub(ld(a.p0, base + k), maximum)))
            st(a.p1, base + k, div(e, total))
    else:
        var log_total = ftz(identical_log(total))
        for k in range(cols):
            st(a.p1, base + k, sub(sub(ld(a.p0, base + k), maximum), log_total))


@always_inline
def op_ag_softmax_vjp(t: Int, a: Args):
    """p0 forward output y, p1 upstream, p2 destination, i0 = 1 for
    log-softmax, i1 cols. Thread t = row t: the row's dot product is
    computed once, O(rows * cols)."""
    var cols = a.i1
    var base = t * cols
    var total = Float32(0)
    for k in range(cols):
        var g = ld(a.p1, base + k)
        if a.i0 == 0:
            g = mul(g, ld(a.p0, base + k))
        total = add(total, g)
    for k in range(cols):
        var i = base + k
        if a.i0 == 0:
            st(a.p2, i, mul(ld(a.p0, i), sub(ld(a.p1, i), total)))
        else:
            st(a.p2, i, sub(ld(a.p1, i), mul(ftz(identical_exp(ld(a.p0, i))), total)))


@always_inline
def op_ag_accumulate(t: Int, a: Args):
    """p1[t] += p0[t]. p0 contribution, p1 gradient."""
    st(a.p1, t, add(ld(a.p1, t), ld(a.p0, t)))


@always_inline
def apply_autograd[OP: Int](t: Int, a: Args):
    comptime if OP == OP_AG_BIN_FWD:
        op_ag_bin_fwd(t, a)
    elif OP == OP_AG_BIN_VJP:
        op_ag_bin_vjp(t, a)
    elif OP == OP_AG_UNARY_FWD:
        op_ag_unary_fwd(t, a)
    elif OP == OP_AG_UNARY_VJP:
        op_ag_unary_vjp(t, a)
    elif OP == OP_AG_TRANSPOSE:
        op_ag_transpose(t, a)
    elif OP == OP_AG_REDUCE_PART:
        op_ag_reduce_part(t, a)
    elif OP == OP_AG_REDUCE_FOLD:
        op_ag_reduce_fold(t, a)
    elif OP == OP_AG_REDUCE_VJP:
        op_ag_reduce_vjp(t, a)
    elif OP == OP_AG_SOFTMAX_FWD:
        op_ag_softmax_fwd(t, a)
    elif OP == OP_AG_SOFTMAX_VJP:
        op_ag_softmax_vjp(t, a)
    elif OP == OP_AG_ACCUMULATE:
        op_ag_accumulate(t, a)
