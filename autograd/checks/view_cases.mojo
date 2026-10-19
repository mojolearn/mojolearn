# SPDX-License-Identifier: Apache-2.0
# Lane ag-pass6-fold (2026-10-19): bit-equality checks for the pass-6 tape
# paths against the paths they replace, on any executor (HostExec, Metal,
# NVIDIA, AMD). Each case compares two recordings of the same mathematics
# word for word and prints one VIEW line with the digest of the compared
# words, so the cpu and gpu outputs can also be diffed across columns.
#   transpose view (AUTOGRAD_TRANSPOSE_VIEW): matmul(transpose(x), y) and
#     matmul(a, transpose(b)) with the view read through GEMM strides, vs
#     the same products after materialize (a dense operand): the value and
#     every gradient; value() refuses a pending view, value_of and values
#     give the materialized words.
#   loss_reduced (AUTOGRAD_LOSS_REDUCED): mse / mae / huber / bce, mean and
#     sum, at 5 x 7 (one launch) and 70 x 130 (leaves, a tree level, the
#     last level) vs loss_terms then mean / sum: the value and both
#     gradients.
#   the curve word (AUTOGRAD_CURVE_WORD): arm_fold_word before a mean vs
#     tape.scale(mean, B): the same word.
# Shapes are small, odd and off any board row (size reasoning: they cover a
# one-block fold, a multi-level fold and a non-square GEMM).
from std.memory import bitcast
from autograd.tape import Tape, Tensor
from autograd.loss_ops import LS_MSE, LS_MAE, LS_HUBER, LS_BCE_LOGITS
from sequence.exec_trait import Exec
from sequence.ops import FP


def _data(n: Int, salt: Int) -> List[Float32]:
    var v = List[Float32](length=n, fill=Float32(0))
    for i in range(n):  # small-loop(n: check data, host side)
        v[i] = Float32(((i * 7919 + salt * 104729) % 1000) - 500) * Float32(0.00123)
    return v^


def _digest(v: List[Float32]) -> UInt64:
    var digest = UInt64(1469598103934665603)
    for i in range(len(v)):  # small-loop(len(v): check output digest)
        digest = (digest ^ UInt64(bitcast[DType.uint32](v[i]))) * UInt64(1099511628211)
    return digest


def _same(name: String, got: List[Float32], want: List[Float32]) raises -> UInt64:
    if len(got) != len(want):
        raise Error(name + ": lengths differ")
    for i in range(len(got)):  # small-loop(len(got): check comparison)
        if bitcast[DType.uint32](got[i]) != bitcast[DType.uint32](want[i]):
            print("FAIL", name, "index", i, "got", got[i], "want", want[i])
            raise Error(name + ": bits differ")
    return _digest(got)


def _matmul_case[E: Exec](mut ex: E, mut tape: Tape, left_view: Bool, dense: Bool) raises -> List[List[Float32]]:
    """[value, d first operand's source, d second operand's source] of
    matmul(transpose(x), y) (left_view) or matmul(a, transpose(b)), the
    transposed operand materialized first when `dense`."""
    tape.reset()
    comptime M = 23
    comptime K = 37
    comptime N = 19
    var out: Tensor
    var p: Tensor
    var q: Tensor
    if left_view:
        p = tape.tensor(ex, _data(K * M, 1), K, M, True)   # x, transposed to M x K
        q = tape.tensor(ex, _data(K * N, 2), K, N, True)   # y
        var t = tape.transpose(ex, p)
        if dense:
            tape.materialize(ex, t)
        out = tape.matmul(ex, t, q)
    else:
        p = tape.tensor(ex, _data(M * K, 3), M, K, True)   # a
        q = tape.tensor(ex, _data(N * K, 4), N, K, True)   # b, transposed to K x N
        var t = tape.transpose(ex, q)
        if dense:
            tape.materialize(ex, t)
        out = tape.matmul(ex, p, t)
    var seed = tape.tensor(ex, _data(M * N, 5), M, N)
    tape.backward_from(ex, out, seed)
    var r = List[List[Float32]]()
    r.append(tape.values(ex, out))
    r.append(tape.gradients(ex, p))
    r.append(tape.gradients(ex, q))
    return r^


def _view_reads[E: Exec](mut ex: E, mut tape: Tape) raises -> UInt64:
    """value() refuses a pending view; values() and value_of give its words."""
    tape.reset()
    var x = tape.tensor(ex, _data(5 * 3, 6), 5, 3)
    var t = tape.transpose(ex, x)
    var first = tape.values(ex, t)
    var refused = False
    try:
        _ = tape.value(t)
    except:
        refused = True
    _ = tape.value_of(ex, t)
    var again = tape.values(ex, t)
    var d = _same("view/reads", again, first)
    var r = tape.reshape(ex, t, 1, 15)
    _ = _same("view/reshape", tape.values(ex, r), first)
    # x is 5 x 3; t[c][r] = x[r][c]
    var xs = _data(5 * 3, 6)
    for i in range(3):  # small-loop(3: check rows)
        for j in range(5):  # small-loop(5: check columns)
            if bitcast[DType.uint32](first[i * 5 + j]) != bitcast[DType.uint32](xs[j * 3 + i]):
                raise Error("view/reads: the view's words are not the transpose")
    print("VIEW view/reads refused", refused, "digest", d)
    return d


def _loss_case[E: Exec](
    mut ex: E, mut tape: Tape, kind: Int, rows: Int, cols: Int, mean: Bool, reduced: Bool,
) raises -> List[List[Float32]]:
    tape.reset()
    var p = tape.tensor(ex, _data(rows * cols, 7 + kind), rows, cols, True)
    var yv = _data(rows * cols, 11 + kind)
    if kind == LS_BCE_LOGITS:
        for i in range(len(yv)):  # small-loop(len(yv): check targets in [0, 1])
            yv[i] = Float32(1) if yv[i] > 0 else Float32(0)
    var y = tape.tensor(ex, yv, rows, cols, True)
    var delta = Float32(0.3)
    var loss: Tensor
    if reduced:
        loss = tape.loss_reduced(ex, p, y, kind, delta, mean)
    else:
        var terms = tape.loss_terms(ex, p, y, kind, delta)
        if mean:
            loss = tape.mean(ex, terms)
        else:
            loss = tape.sum(ex, terms)
    tape.backward(ex, loss)
    var r = List[List[Float32]]()
    r.append(tape.values(ex, loss))
    r.append(tape.gradients(ex, p))
    r.append(tape.gradients(ex, y))
    return r^


def _curve_word[E: Exec](mut ex: E, mut tape: Tape, rows: Int, cols: Int) raises -> UInt64:
    tape.reset()
    var p = tape.tensor(ex, _data(rows * cols, 21), rows, cols, True)
    var q = tape.tensor(ex, _data(rows * cols, 22), rows, cols)
    var B = Float32(rows)
    var word = tape.reserve(ex, 1)
    tape.arm_fold_word(word, B)
    var d = tape.subtract(ex, p, q)
    var loss = tape.mean(ex, tape.multiply(ex, d, d))
    var wrote = tape.disarm_fold_word(loss)
    var want = tape.values(ex, tape.scale(ex, loss, B))
    var got = List[Float32](length=1, fill=Float32(0))
    ex.download(FP(unsafe_from_address=Int(got.unsafe_ptr())), word, 1)
    if not wrote:
        print("VIEW curve_word", rows, "x", cols, "not taken (MOJOLEARN_AUTOGRAD_CURVE_WORD_OFF)")
        return 0
    var dg = _same("curve_word/" + String(rows) + "x" + String(cols), got, want)
    print("VIEW curve_word", rows, "x", cols, "digest", dg)
    return dg


def run_view_checks[E: Exec](mut ex: E) raises:
    var tape = Tape()
    var summary = UInt64(1469598103934665603)
    var cases = 0
    for side in range(2):  # small-loop(2: the transposed operand's side)
        var name = String("view/matmul_tx_y")
        if side == 1:
            name = String("view/matmul_a_tb")
        var v = _matmul_case(ex, tape, side == 0, False)
        var w = _matmul_case(ex, tape, side == 0, True)
        for k in range(3):  # small-loop(3: value and two gradients)
            var d = _same(name + "/" + String(k), v[k], w[k])
            summary = (summary ^ d) * UInt64(1099511628211)
            print("VIEW", name, k, "digest", d)
            cases += 1
    summary = (summary ^ _view_reads(ex, tape)) * UInt64(1099511628211)
    cases += 1
    var kinds: List[Int] = [LS_MSE, LS_MAE, LS_HUBER, LS_BCE_LOGITS]
    var shapes: List[Int] = [5, 7, 70, 130]
    for ki in range(len(kinds)):  # small-loop(4: loss kinds)
        for si in range(2):  # small-loop(2: shapes)
            for mi in range(2):  # small-loop(2: mean, sum)
                var rows = shapes[2 * si]
                var cols = shapes[2 * si + 1]
                var mean = mi == 0
                var a = _loss_case(ex, tape, kinds[ki], rows, cols, mean, True)
                var b = _loss_case(ex, tape, kinds[ki], rows, cols, mean, False)
                var name = String("loss_reduced/") + String(kinds[ki]) + "/" + String(rows) + "x" + String(cols)
                if mean:
                    name += "/mean"
                else:
                    name += "/sum"
                for k in range(3):  # small-loop(3: value and two gradients)
                    var d = _same(name + "/" + String(k), a[k], b[k])
                    summary = (summary ^ d) * UInt64(1099511628211)
                    print("VIEW", name, k, "digest", d)
                    cases += 1
    summary = (summary ^ _curve_word(ex, tape, 5, 7)) * UInt64(1099511628211)
    summary = (summary ^ _curve_word(ex, tape, 70, 130)) * UInt64(1099511628211)
    cases += 2
    print("PASS view cases", cases, "summary", summary)
    tape.release(ex)
