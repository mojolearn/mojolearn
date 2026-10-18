# SPDX-License-Identifier: Apache-2.0
# Native element bodies shared by HostExec and DeviceExec. Graph traversal
# and gradient accumulation order live in tape.mojo, not inside GPU atomics.
from sequence.ops import Args, ld, st, add, sub, mul
from checks.numerics import (
    ftz, identical_div, identical_exp, identical_log, identical_sigmoid, identical_sqrt,
    identical_tanh,
)

# Sequence codes 120/121 are recurrent scan kernels, intercepted before the
# generic dispatcher. Reserve 1000 for this dispatch family on both executors.
comptime OP_AUTOGRAD = 1000
comptime FORWARD = 0
comptime VJP = 1
comptime ACCUMULATE = 2
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


@always_inline
def div(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(a, b))


@always_inline
def _index(row: Int, col: Int, rows: Int, cols: Int) -> Int:
    return (0 if rows == 1 else row) * cols + (0 if cols == 1 else col)


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
def _unary_vjp(op: Int, x: Float32, y: Float32, g: Float32, scale: Float32) -> Float32:
    if op == RELU:
        return g if x > 0 else Float32(0)
    if op == SIGMOID:
        return mul(mul(g, y), sub(Float32(1), y))
    if op == TANH:
        return mul(g, sub(Float32(1), mul(y, y)))
    if op == EXP:
        return mul(g, y)
    if op == LOG:
        return div(g, x)
    if op == SQRT:
        return div(g, mul(Float32(2), y))
    return mul(g, scale)


def apply_autograd(t: Int, a: Args):
    # Forward: p0/p1 inputs, p2 output. VJP: p0/p1 inputs,
    # p2 forward output, p3 upstream, p4 this input's contribution.
    # i2/i3 output shape, i4/i5 left shape, i6/i7 right shape,
    # i8 operand number, i9 reduction axis. f0 reduction/scale factor.
    var op = a.i1
    if a.i0 == ACCUMULATE:
        st(a.p1, t, add(ld(a.p1, t), ld(a.p0, t)))
        return
    if a.i0 == FORWARD:
        if op <= DIVIDE:
            var row = t // a.i3
            var col = t % a.i3
            var x = ld(a.p0, _index(row, col, a.i4, a.i5))
            var y = ld(a.p1, _index(row, col, a.i6, a.i7))
            var z = Float32(0)
            if op == ADD:
                z = add(x, y)
            elif op == SUBTRACT:
                z = sub(x, y)
            elif op == MULTIPLY:
                z = mul(x, y)
            else:
                z = div(x, y)
            st(a.p2, t, z)
        elif op < TRANSPOSE:
            st(a.p2, t, _unary(op, ld(a.p0, t), a.f0))
        elif op == TRANSPOSE:
            st(a.p2, t, ld(a.p0, (t % a.i3) * a.i5 + t // a.i3))
        elif op == REDUCE:
            var total = Float32(0)
            if a.i9 == -1:
                for k in range(a.i4 * a.i5):
                    total = add(total, ld(a.p0, k))
            elif a.i9 == 0:
                for k in range(a.i4):
                    total = add(total, ld(a.p0, k * a.i5 + t))
            else:
                for k in range(a.i5):
                    total = add(total, ld(a.p0, t * a.i5 + k))
            st(a.p2, t, mul(total, a.f0))
        elif op == SOFTMAX or op == LOG_SOFTMAX:
            var base = t * a.i5
            var maximum = ld(a.p0, base)
            for k in range(1, a.i5):
                maximum = max(maximum, ld(a.p0, base + k))
            var total = Float32(0)
            for k in range(a.i5):
                var e = ftz(identical_exp(sub(ld(a.p0, base + k), maximum)))
                st(a.p2, base + k, e)
                total = add(total, e)
            if op == SOFTMAX:
                for k in range(a.i5):
                    st(a.p2, base + k, div(ld(a.p2, base + k), total))
            else:
                var log_total = ftz(identical_log(total))
                for k in range(a.i5):
                    st(a.p2, base + k, sub(sub(ld(a.p0, base + k), maximum), log_total))
        return

    if op <= DIVIDE:
        var rows = a.i4 if a.i8 == 0 else a.i6
        var cols = a.i5 if a.i8 == 0 else a.i7
        var row = t // cols
        var col = t % cols
        # One writer per gradient element, with a fixed ascending fold over
        # every broadcast use. No atomic scatter and no hardware-sized tree.
        var r0 = 0 if rows == 1 else row
        var r1 = a.i2 if rows == 1 else row + 1
        var c0 = 0 if cols == 1 else col
        var c1 = a.i3 if cols == 1 else col + 1
        var total = Float32(0)
        var first = True
        for r in range(r0, r1):
            for c in range(c0, c1):
                var g = ld(a.p3, r * a.i3 + c)
                if op == SUBTRACT and a.i8 == 1:
                    g = -g
                elif op == MULTIPLY:
                    if a.i8 == 0:
                        g = mul(g, ld(a.p1, _index(r, c, a.i6, a.i7)))
                    else:
                        g = mul(g, ld(a.p0, _index(r, c, a.i4, a.i5)))
                elif op == DIVIDE:
                    var y = ld(a.p1, _index(r, c, a.i6, a.i7))
                    if a.i8 == 0:
                        g = div(g, y)
                    else:
                        g = -div(mul(g, ld(a.p2, r * a.i3 + c)), y)
                total = g if first else add(total, g)
                first = False
        st(a.p4, t, total)
    elif op < TRANSPOSE:
        st(a.p4, t, _unary_vjp(op, ld(a.p0, t), ld(a.p2, t), ld(a.p3, t), a.f0))
    elif op == TRANSPOSE:
        st(a.p4, t, ld(a.p3, (t % a.i5) * a.i4 + t // a.i5))
    elif op == RESHAPE:
        st(a.p4, t, ld(a.p3, t))
    elif op == REDUCE:
        var i = 0
        if a.i9 == 0:
            i = t % a.i5
        elif a.i9 == 1:
            i = t // a.i5
        st(a.p4, t, mul(ld(a.p3, i), a.f0))
    elif op == SOFTMAX or op == LOG_SOFTMAX:
        # One thread owns a row, so the dot product is computed once rather
        # than once per column. This is O(rows * cols), not O(rows * cols^2).
        var base = t * a.i5
        var total = Float32(0)
        for k in range(a.i5):
            var g = ld(a.p3, base + k)
            if op == SOFTMAX:
                g = mul(g, ld(a.p2, base + k))
            total = add(total, g)
        for k in range(a.i5):
            var i = base + k
            if op == SOFTMAX:
                st(a.p4, i, mul(ld(a.p2, i), sub(ld(a.p3, i), total)))
            else:
                st(a.p4, i, sub(ld(a.p3, i), mul(ftz(identical_exp(ld(a.p2, i))), total)))
