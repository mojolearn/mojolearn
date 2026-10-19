# SPDX-License-Identifier: Apache-2.0
# Canonical gradient accumulation order (docs/plans/NATIVE_AUTOGRAD.md,
# "Accumulation order"). One leaf x (1 x 4) feeds 3 to 5 consumers recorded
# in several orders; the consumers are then summed by a chain of adds and a
# backward_from runs with a fixed seed. Every consumer's contribution to dx
# is exact (powers of two times small dyadics, a one-hot GEMM), so the only
# rounding is the order the tape adds them in, and the host oracle below
# reproduces that order with the kernel's own add (sequence.ops.add). The
# check asserts:
#   (a) a given record order gives the oracle's bits, and the same bits on a
#       repeated backward and on a re-recorded graph;
#   (b) a different record order changes bits only in the columns where the
#       oracle's sums round differently (columns 0 and 2 sit next to 2^24 and
#       2^25, where adding 1 or 2 rounds; columns 1 and 3 are exact sums and
#       must not change), and the data does make columns 0/2 change;
#   (c) the printed digests are the oracle's, so HostExec, Metal, NVIDIA and
#       AMD print the same ORDER lines (diff the cpu and gpu outputs).
from std.memory import bitcast
from autograd import Tape, Tensor
from sequence.exec_trait import Exec
from sequence.ops import add, mul
from autograd.checks.lifecycle import run_alias_checks

comptime _N = 4
# consumer kinds
comptime _MUL = 0     # x * w        dx += s * w
comptime _SCALE = 1   # 0.5 x        dx += s * 0.5
comptime _ADD = 2     # x + c        dx += s
comptime _SQUARE = 3  # x * x        dx += s * x, then s * x (edges 0, 1)
comptime _LINEAR = 4  # x @ diag(w)  dx += s * w (one-hot GEMM: exact)
comptime _RSUB = 5    # c - x        dx += -s


def _x() -> List[Float32]:
    return [Float32(1.5), Float32(-2.25), Float32(3.0), Float32(0.75)]


def _seed() -> List[Float32]:
    return [Float32(1.0), Float32(0.5), Float32(2.0), Float32(0.25)]


def _kind(id: Int) -> Int:
    if id == 0 or id == 1:
        return _MUL
    if id == 2:
        return _SCALE
    if id == 3:
        return _ADD
    if id == 4:
        return _SQUARE
    if id == 5:
        return _LINEAR
    return _RSUB


def _weights(id: Int) -> List[Float32]:
    """Per-column weights of MUL (ids 0, 1) and LINEAR (id 5) consumers."""
    if id == 0:
        return [Float32(16777216.0), Float32(3.0), Float32(1.0), Float32(0.5)]
    if id == 1:
        return [Float32(1.0), Float32(5.0), Float32(-16777216.0), Float32(0.25)]
    return [Float32(4.0), Float32(-1.0), Float32(8.0), Float32(0.5)]


def _contributions(id: Int, col: Int) -> List[Float32]:
    """The contributions consumer `id` makes to dx[col], in edge order,
    computed with the kernels' own mul (every one is exact)."""
    var s = _seed()[col]
    var k = _kind(id)
    if k == _MUL or k == _LINEAR:
        return [mul(s, _weights(id)[col])]
    if k == _SCALE:
        return [mul(s, Float32(0.5))]
    if k == _ADD:
        return [s]
    if k == _SQUARE:
        var v = mul(s, _x()[col])
        return [v, v]
    return [-s]


def _oracle(order: List[Int]) -> List[Float32]:
    """dx under the stated order: consumers in decreasing record position,
    one consumer's edges ascending; the first written, the rest added with
    the accumulate kernel's add."""
    var out = List[Float32]()
    for col in range(_N):
        var acc = Float32(0)
        var started = False
        for back in range(len(order)):
            var id = order[len(order) - 1 - back]
            var parts = _contributions(id, col)
            for e in range(len(parts)):
                if started:
                    acc = add(acc, parts[e])
                else:
                    acc = parts[e]
                    started = True
        out.append(acc)
    return out^


def _consumer[E: Exec](mut ex: E, mut tape: Tape, x: Tensor, id: Int) raises -> Tensor:
    var k = _kind(id)
    if k == _MUL:
        var w = tape.tensor(ex, _weights(id), 1, _N)
        return tape.multiply(ex, x, w)
    if k == _SCALE:
        return tape.scale(ex, x, 0.5)
    if k == _ADD:
        var c = tape.tensor(ex, [Float32(0.5), Float32(-1.0), Float32(2.0), Float32(0.125)], 1, _N)
        return tape.add(ex, x, c)
    if k == _SQUARE:
        return tape.multiply(ex, x, x)
    if k == _LINEAR:
        var d = _weights(id)
        var w = List[Float32](length=_N * _N, fill=0)
        for j in range(_N):
            w[j * _N + j] = d[j]
        var wt = tape.tensor(ex, w, _N, _N)
        return tape.linear(ex, x, wt)
    var c = tape.tensor(ex, [Float32(1.0), Float32(1.0), Float32(1.0), Float32(1.0)], 1, _N)
    return tape.subtract(ex, c, x)


def _run[E: Exec](mut ex: E, mut tape: Tape, order: List[Int]) raises -> List[Float32]:
    """Record x, its consumers in `order`, their add chain; return dx. A
    second backward on the same graph must give the same bits."""
    tape.reset()
    var x = tape.tensor(ex, _x(), 1, _N, True)
    var outs = List[Tensor]()
    for i in range(len(order)):
        outs.append(_consumer(ex, tape, x, order[i]))
    var total = outs[0]
    for i in range(1, len(outs)):
        total = tape.add(ex, total, outs[i])
    var seed = tape.tensor(ex, _seed(), 1, _N)
    tape.backward_from(ex, total, seed)
    var dx = tape.gradients(ex, x)
    tape.backward_from(ex, total, seed)
    _same("repeat backward", tape.gradients(ex, x), dx)
    return dx^


def _same(name: String, actual: List[Float32], expected: List[Float32]) raises:
    if len(actual) != len(expected):
        raise Error(name + ": wrong length")
    for i in range(len(actual)):
        if bitcast[DType.uint32](actual[i]) != bitcast[DType.uint32](expected[i]):
            print("FAIL", name, "index", i, "actual", actual[i], "expected", expected[i])
            raise Error(name + ": bits differ from the canonical order")


def _digest(v: List[Float32]) -> UInt64:
    var digest = UInt64(1469598103934665603)
    for i in range(len(v)):
        digest = (digest ^ UInt64(bitcast[DType.uint32](v[i]))) * UInt64(1099511628211)
    return digest


def _label(order: List[Int]) -> String:
    var s = String("")
    for i in range(len(order)):
        s += String(order[i])
    return s


def run_order_cases[E: Exec](mut ex: E) raises:
    var tape = Tape()
    # Each set holds the same consumers in several record orders.
    var sets = List[List[List[Int]]]()
    sets.append([[0, 1, 3], [3, 1, 0], [1, 0, 3]])
    sets.append([[0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1]])
    sets.append([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [1, 4, 0, 3, 2]])
    sets.append([[0, 1, 3, 4, 5], [5, 4, 3, 1, 0], [3, 5, 1, 0, 4]])
    sets.append([[0, 1, 4, 5, 6], [6, 5, 4, 1, 0], [4, 0, 6, 1, 5]])
    var changed = 0
    var cases = 0
    var summary = UInt64(1469598103934665603)
    for si in range(len(sets)):
        var first = List[Float32]()
        var first_oracle = List[Float32]()
        for oi in range(len(sets[si])):
            var order = sets[si][oi].copy()
            var name = String("order/") + String(len(order)) + "/" + _label(order)
            var expected = _oracle(order)
            var got = _run(ex, tape, order)
            _same(name, got, expected)
            # (a) a re-recorded graph gives the same bits
            _same(name + "/rerecord", _run(ex, tape, order), got)
            var d = _digest(got)
            summary = (summary ^ d) * UInt64(1099511628211)
            print("ORDER", name, "digest", d)
            cases += 1
            if oi == 0:
                first = got.copy()
                first_oracle = expected.copy()
                continue
            # (b) a different order moves bits only where the oracle's sums
            # round differently; the exact columns (1, 3) never move.
            for col in range(_N):
                var moved = bitcast[DType.uint32](got[col]) != bitcast[DType.uint32](first[col])
                var predicted = bitcast[DType.uint32](expected[col]) != bitcast[DType.uint32](first_oracle[col])
                if moved != predicted:
                    raise Error(name + ": record order changed bits the oracle does not predict")
                if moved and (col == 1 or col == 3):
                    raise Error(name + ": an exact sum changed with the record order")
                if moved:
                    changed += 1
    if changed == 0:
        raise Error("order cases never exercised a rounding difference")
    print("PASS order cases", cases, "order-sensitive columns", changed, "summary", summary)
    tape.release(ex)


def run_order_checks[E: Exec](mut ex: E) raises:
    run_order_cases(ex)
    run_alias_checks(ex)
