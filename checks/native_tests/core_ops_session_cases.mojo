# SPDX-License-Identifier: Apache-2.0
"""The operation session's compositions, moved from test_ops_session.py
(2026-11-14). Each fixture runs on two independent sessions of one executor
and the second must give the same float32 words (the former digest check).

  0 mixture of two experts (inference): y0 = linear(x, W0, b0),
    y1 = tanh linear, gate = softmax linear; the hard route picks the expert
    with the larger mean gate by one scalar read back; the soft route weights
    each row's experts by its gate. Shapes, finite values, gate rows sum to
    1 within 1e-5, the hard route is >= 0, both sessions pick the same expert.
  1 losses and metrics on frozen predictions (inference): cross entropy mean
    and sum, mse, huber (delta 0.5); accuracy, log loss, ROC AUC. Finite,
    cross entropy > 0 and its sum larger than its mean, accuracy and AUC in
    [0, 1], log loss > 0.
  2 permutation and row gather: permutation(10, seed 7, epoch 0) holds every
    index once; the gathered rows are X's rows (their first column sorted is
    0, 4, ..., 36); the batch gather from 2 of 3 rows equals rows[2:5].
  3 recording session gradient: d mse(X w, Y)/dw within rtol 1e-4, atol 1e-5
    of (2 / Y.size) X^T (X W - Y); the loss within rtol 1e-5 (atol 1e-8) of
    the float64 mean squared error.

The NumPy draws are replaced by core_ops_common's seeded values of the same
shapes. The refusals, the ndarray program input views and the operation-name
table stay in the Python test.
"""
from std.math import exp
from autograd.ops_api import OpsSession
from sequence.exec_trait import Exec
from checks.native_tests.ops_higher_cases import (
    require, tensor, op1, op2, total, values, gradient, same_words,
)
from checks.native_tests.core_ops_common import (
    normal, uniform, wide, allclose, exact, all_finite, dims2, same_shape, opi,
)

comptime CORE_OPS_SESSION_FIXTURES = 4

comptime _MATMUL = 1
comptime _ADD = 2
comptime _MUL = 4
comptime _RELU = 6
comptime _LINEAR = 19
comptime _GATHER_ROWS = 22
comptime _PERMUTATION = 23
comptime _MSE = 24
comptime _HUBER = 26
comptime _CROSS_ENTROPY = 28
comptime _SOFTMAX = 17
comptime _MEAN_N = 36
comptime _ACT_NONE = 0
comptime _ACT_TANH = 3
comptime _ACT_SOFTMAX = 4
comptime _ACCURACY = 1
comptime _LOG_LOSS = 2
comptime _ROC_AUC = 3


def core_ops_session_records(fixture_id: Int) -> Bool:
    """Whether the fixture's sessions record (only the gradient does)."""
    return fixture_id == 3


def _linear[E: Exec & Movable & Deinitable](mut s: OpsSession[E], x: Int, w: Int, b: Int, act: Int) raises -> Int:
    var hs: List[Int] = [x, w, b]
    var ints: List[Int] = [act]
    return opi(s, _LINEAR, hs, ints, List[Float32]())


def _moe[E: Exec & Movable & Deinitable](mut s: OpsSession[E]) raises -> Tuple[List[List[Float32]], Bool]:
    var x = tensor(s, normal(64, 0), dims2(16, 4))
    var w0 = tensor(s, normal(12, 1), dims2(4, 3))
    var w1 = tensor(s, normal(12, 2), dims2(4, 3))
    var wg = tensor(s, normal(8, 3), dims2(4, 2))
    var b0 = tensor(s, normal(3, 4), dims2(1, 3))
    var b1 = tensor(s, normal(3, 5), dims2(1, 3))
    var bg = tensor(s, normal(2, 6), dims2(1, 2))
    var y0 = _linear(s, x, w0, b0, _ACT_NONE)
    var y1 = _linear(s, x, w1, b1, _ACT_TANH)
    var gate = _linear(s, x, wg, bg, _ACT_SOFTMAX)
    # hard route: mean(gate, axis=0) (keepdims default) @ [[1], [-1]], one scalar read back
    var hs: List[Int] = [gate]
    var ints: List[Int] = [-1, 0]
    var mean_gate = opi(s, _MEAN_N, hs, ints, List[Float32]())
    var plus_minus: List[Float32] = [1.0, -1.0]
    var margin = op2(s, _MATMUL, mean_gate, tensor(s, plus_minus, dims2(2, 1)))
    same_shape(s, gate, dims2(16, 2), "moe gate")
    same_shape(s, margin, dims2(1, 1), "moe margin")
    var first = s.scalar(margin) > 0
    var hard = op1(s, _RELU, y0 if first else y1)
    var e0: List[Float32] = [1.0, 0.0]
    var e1: List[Float32] = [0.0, 1.0]
    var g0 = op2(s, _MATMUL, gate, tensor(s, e0, dims2(2, 1)))
    var g1 = op2(s, _MATMUL, gate, tensor(s, e1, dims2(2, 1)))
    var soft = op2(s, _ADD, op2(s, _MUL, g0, y0), op2(s, _MUL, g1, y1))
    var out: List[List[Float32]] = [values(s, gate, dims2(16, 2)), values(s, hard, dims2(16, 3)), values(s, soft, dims2(16, 3))]
    return (out^, first)


def mixture_of_two_experts[E: Exec & Movable & Deinitable](mut s: OpsSession[E], mut again: OpsSession[E]) raises:
    var r = _moe(s)
    var got = r[0].copy()
    for k in range(3):
        all_finite(got[k], "moe output " + String(k))
    for row in range(16):
        var t = Float64(got[0][2 * row]) + Float64(got[0][2 * row + 1])
        require(abs(t - 1.0) <= 1e-5, "moe gate row " + String(row) + " sums to " + String(t))
    for v in got[1]:
        require(v >= 0, "moe hard route is a relu")
    var r2 = _moe(again)
    require(r[1] == r2[1], "moe: the two sessions pick different experts")
    for k in range(3):
        same_words(got[k], r2[0][k], "moe output " + String(k))


def _frozen[E: Exec & Movable & Deinitable](mut s: OpsSession[E]) raises -> List[Float32]:
    var logits = normal(96, 10)
    var codes = List[Float32](length=32, fill=0)
    var picks = uniform(32, 11, 0.0, 3.0)
    for i in range(32):
        codes[i] = Float32(Int(picks[i]))
    var target = normal(96, 12)
    var binary = List[Float32](length=32, fill=0)
    var coins = uniform(32, 13, 0.0, 1.0)
    for i in range(32):
        binary[i] = 1.0 if coins[i] > 0.5 else 0.0
    binary[0] = 0.0
    binary[1] = 1.0
    var first_column = List[Float32](length=32, fill=0)
    for i in range(32):
        first_column[i] = logits[3 * i]
    var z = tensor(s, logits, dims2(32, 3))
    var c = tensor(s, codes, dims2(32, 1))
    var t = tensor(s, target, dims2(32, 3))
    var hs: List[Int] = [z, c]
    var mean_words: List[Int] = [0]
    var sum_words: List[Int] = [1]
    var ce = opi(s, _CROSS_ENTROPY, hs, mean_words, List[Float32]())
    var ce_sum = opi(s, _CROSS_ENTROPY, hs, sum_words, List[Float32]())
    var zt: List[Int] = [z, t]
    var mse = opi(s, _MSE, zt, mean_words, List[Float32]())
    var delta: List[Float32] = [0.5]
    var hub = opi(s, _HUBER, zt, mean_words, delta)
    var prob = op1(s, _SOFTMAX, z)
    var acc_hs: List[Int] = [c, z]
    var acc = s.metric(_ACCURACY, acc_hs)
    var ll_hs: List[Int] = [c, prob]
    var ll = s.metric(_LOG_LOSS, ll_hs)
    var auc_hs: List[Int] = [tensor(s, binary, dims2(32, 1)), tensor(s, first_column, dims2(32, 1))]
    var auc = s.metric(_ROC_AUC, auc_hs)
    var out: List[Float32] = [s.scalar(ce), s.scalar(ce_sum), s.scalar(mse), s.scalar(hub), acc, ll, auc]
    return out^


def losses_and_metrics_on_frozen_predictions[E: Exec & Movable & Deinitable](mut s: OpsSession[E], mut again: OpsSession[E]) raises:
    var r = _frozen(s)
    all_finite(r, "frozen losses and metrics")
    require(r[0] > 0 and r[1] > r[0], "cross entropy is positive and its sum exceeds its mean")
    require(r[4] >= 0 and r[4] <= 1, "accuracy in [0, 1]")
    require(r[5] > 0, "log loss is positive")
    require(r[6] >= 0 and r[6] <= 1, "ROC AUC in [0, 1]")
    same_words(r, _frozen(again), "frozen losses and metrics")


def _gather[E: Exec & Movable & Deinitable](mut s: OpsSession[E]) raises -> List[List[Float32]]:
    var xv = List[Float32](length=40, fill=0)
    for i in range(40):
        xv[i] = Float32(i)
    var x = tensor(s, xv, dims2(10, 4))
    var perm_words: List[Int] = [10, 7, 0, 0]
    var perm = opi(s, _PERMUTATION, List[Int](), perm_words, List[Float32]())
    var hs: List[Int] = [x, perm]
    var all_rows: List[Int] = [0, 0]
    var some: List[Int] = [2, 3]
    var rows = opi(s, _GATHER_ROWS, hs, all_rows, List[Float32]())
    var batch = opi(s, _GATHER_ROWS, hs, some, List[Float32]())
    same_shape(s, rows, dims2(10, 4), "gathered rows")
    same_shape(s, batch, dims2(3, 4), "gathered batch")
    var out: List[List[Float32]] = [values(s, perm, dims2(10, 1)), values(s, rows, dims2(10, 4)), values(s, batch, dims2(3, 4))]
    return out^


def permutation_and_gather[E: Exec & Movable & Deinitable](mut s: OpsSession[E], mut again: OpsSession[E]) raises:
    var r = _gather(s)
    var seen = List[Int](length=10, fill=0)
    for v in r[0]:
        require(v >= 0 and v < 10 and v == Float32(Int(v)), "permutation index range")
        seen[Int(v)] += 1
    var firsts = List[Int](length=10, fill=0)
    for row in range(10):
        var v = r[1][4 * row]
        require(v == Float32(Int(v)) and Int(v) % 4 == 0 and Int(v) < 40, "a gathered row is a row of X")
        firsts[Int(v) // 4] += 1
    for k in range(10):
        require(seen[k] == 1, "permutation holds every index once")
        require(firsts[k] == 1, "the gathered rows' first column sorted is 0, 4, ..., 36")
    var middle = List[Float32](length=12, fill=0)
    for i in range(12):
        middle[i] = r[1][8 + i]
    exact(r[2], middle, "the batch gather is rows[2:5]")
    var r2 = _gather(again)
    for k in range(3):
        same_words(r[k], r2[k], "gather output " + String(k))


def _grad_run[E: Exec & Movable & Deinitable](mut s: OpsSession[E], xv: List[Float32], wv: List[Float32], yv: List[Float32]) raises -> List[Float32]:
    var w = tensor(s, wv, dims2(5, 2), True)
    var pred = op2(s, _MATMUL, tensor(s, xv, dims2(24, 5)), w)
    var hs: List[Int] = [pred, tensor(s, yv, dims2(24, 2))]
    var mean_words: List[Int] = [0]
    var l = opi(s, _MSE, hs, mean_words, List[Float32]())
    s.backward(l, False)
    var out = gradient(s, w, dims2(5, 2))
    out.append(s.scalar(l))
    return out^


def recording_session_gradient[E: Exec & Movable & Deinitable](mut s: OpsSession[E], mut again: OpsSession[E]) raises:
    var xv = normal(120, 20)
    var wv = normal(10, 21)
    var yv = normal(48, 22)
    var r = _grad_run(s, xv, wv, yv)
    all_finite(r, "gradient and loss")
    var x = wide(xv)
    var w = wide(wv)
    var y = wide(yv)
    var resid = List[Float64](length=48, fill=0)
    var mse = Float64(0)
    for i in range(24):
        for j in range(2):
            var p = Float64(0)
            for k in range(5):
                p += x[5 * i + k] * w[2 * k + j]
            resid[2 * i + j] = p - y[2 * i + j]
            mse += resid[2 * i + j] * resid[2 * i + j]
    mse /= 48.0
    var expected = List[Float64](length=10, fill=0)
    for k in range(5):
        for j in range(2):
            var g = Float64(0)
            for i in range(24):
                g += x[5 * i + k] * resid[2 * i + j]
            expected[2 * k + j] = (2.0 / 48.0) * g
    var grad = List[Float32](length=10, fill=0)
    for i in range(10):
        grad[i] = r[i]
    allclose(grad, expected, 1e-4, 1e-5, "d mse / dw against (2 / n) X^T (X W - Y)")
    var loss: List[Float32] = [r[10]]
    var want: List[Float64] = [mse]
    allclose(loss, want, 1e-5, 1e-8, "the mse loss")
    same_words(r, _grad_run(again, xv, wv, yv), "gradient and loss")


def run_core_ops_session_fixture[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], mut again: OpsSession[E], fixture_id: Int
) raises:
    if fixture_id == 0:
        mixture_of_two_experts(s, again)
    elif fixture_id == 1:
        losses_and_metrics_on_frozen_predictions(s, again)
    elif fixture_id == 2:
        permutation_and_gather(s, again)
    elif fixture_id == 3:
        recording_session_gradient(s, again)
    else:
        raise Error("unknown core_ops_session fixture " + String(fixture_id))
