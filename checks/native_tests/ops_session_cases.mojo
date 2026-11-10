# SPDX-License-Identifier: Apache-2.0
"""Native session correctness fixtures migrated from test_ops_session_coverage.

No Python imports, bindings, objects or callbacks. Both executors use these
same fixtures. Float64 formulas below are independent test oracles; they do
not call product math helpers. Tolerances match the former NumPy tests.
"""
from std.math import exp, log, log1p
from autograd.ops_api import (
    OpsSession, OPS_SUB, OPS_DIV, OPS_MINIMUM, OPS_POW, OPS_COPY,
    OPS_COMPARE, OPS_LOGICAL, OPS_CEIL, OPS_SIGN, OPS_ROUND, OPS_RESHAPE_N,
    OPS_MAE, OPS_HUBER, OPS_BCE_LOGITS, OPS_METRIC_ACCURACY,
    OPS_METRIC_ROC_AUC, OPS_METRIC_LOG_LOSS, OPS_PERMUTATION, OPS_SUM,
    OPS_AVGPOOL2D,
)
from autograd.losses import REDUCTION_MEAN, REDUCTION_SUM
from autograd.model import SCHED_STEP
from autograd.rankn import ShapeN
from core.cast_in import CastIn, CAST_KIND_FLOAT
from core.hostptr import f32_ptr
from sequence.exec_trait import Exec
from sequence.ops import OPT_SGD

comptime OPS_SESSION_FIXTURES = 28


def require(condition: Bool, message: String) raises:
    if not condition:
        raise Error(message)


def _close(actual: Float32, expected: Float64, rtol: Float64, atol: Float64, label: String) raises:
    # The negated comparison rejects NaNs too.
    if not (abs(Float64(actual) - expected) <= atol + rtol * abs(expected)):
        raise Error(label + ": got " + String(actual) + ", expected " + String(expected))


def _upload[E: Exec & Movable & Deinitable](mut s: OpsSession[E], values: List[Float32], rows: Int, cols: Int) raises -> Int:
    var source = CastIn(Int(values.unsafe_ptr()), CAST_KIND_FLOAT, 4, rows, cols, cols * 4, 4)
    var h = s.upload(source, False)
    _ = values
    return h


def _check_tensor[E: Exec & Movable & Deinitable](mut s: OpsSession[E], h: Int, expected: List[Float64], dims: List[Int], rtol: Float64 = 1e-6, atol: Float64 = 1e-6) raises:
    require(s.shape_nd(h).same(ShapeN.of(dims)), "tensor result shape")
    var actual = List[Float32](length=len(expected), fill=0)
    s.download(h, f32_ptr(Int(actual.unsafe_ptr())))
    for i in range(len(expected)):
        _close(actual[i], expected[i], rtol, atol, "tensor element " + String(i))


def _elementwise[E: Exec & Movable & Deinitable](mut s: OpsSession[E], fixture_id: Int, a: Int, b: Int) raises:
    var av: List[Float64] = [1.25, -2.75, 3.5, 4.0]
    var bv: List[Float64] = [2.0, 2.0, 1.0, 0.5]
    var expected = List[Float64](length=4, fill=0)
    var hs: List[Int] = [a, b]
    var ints = List[Int]()
    var fs = List[Float32]()
    var dims: List[Int] = [2, 2]
    var code = OPS_SUB
    if fixture_id < 4:
        if fixture_id == 1:
            code = OPS_DIV
        elif fixture_id == 2:
            code = OPS_MINIMUM
        elif fixture_id == 3:
            code = OPS_POW
            hs = [b, a]
        for i in range(4):
            if fixture_id == 0:
                expected[i] = av[i] - bv[i]
            elif fixture_id == 1:
                expected[i] = av[i] / bv[i]
            elif fixture_id == 2:
                expected[i] = min(av[i], bv[i])
            else:
                expected[i] = exp(av[i] * log(bv[i]))
    elif fixture_id < 8:
        code = OPS_COMPARE
        var kinds: List[Int] = [0, 5, 3, 4]
        ints = [kinds[fixture_id - 4]]
        if fixture_id == 4:
            var copy_hs: List[Int] = [a]
            var copied = s.run_op(OPS_COPY, copy_hs, List[Int](), List[Float32]())
            hs = [a, copied]
        for i in range(4):
            var truth = True
            if fixture_id == 5:
                truth = av[i] != bv[i]
            elif fixture_id == 6:
                truth = av[i] >= bv[i]
            elif fixture_id == 7:
                truth = av[i] <= bv[i]
            expected[i] = 1.0 if truth else 0.0
    elif fixture_id == 8 or fixture_id == 9 or fixture_id == 13:
        code = OPS_LOGICAL
        var zero_hs: List[Int] = [a, a]
        if fixture_id == 8:
            zero_hs = [b, b]
        var zero = s.run_op(OPS_SUB, zero_hs, List[Int](), List[Float32]())
        if fixture_id == 8:
            hs = [a, zero]
            ints = [1]
        elif fixture_id == 9:
            hs = [zero, b]
            ints = [2]
        else:
            hs = [zero]
            ints = [0]
        for i in range(4):
            expected[i] = 0.0 if fixture_id == 8 else 1.0
    else:
        hs = [a]
        if fixture_id == 10:
            code = OPS_CEIL
            expected = [2.0, -2.0, 4.0, 4.0]
        elif fixture_id == 11:
            code = OPS_SIGN
            expected = [1.0, -1.0, 1.0, 1.0]
        elif fixture_id == 12:
            code = OPS_ROUND
            expected = [1.0, -3.0, 4.0, 4.0]
        elif fixture_id == 14:
            code = OPS_POW
            fs = [2.0]
            for i in range(4):
                expected[i] = av[i] * av[i]
        else:
            code = OPS_RESHAPE_N
            ints = [3, 0]
            dims = [1, 2, 2]
            expected = av.copy()
    var result = s.run_op(code, hs, ints, fs)
    _check_tensor(s, result, expected, dims)


def _loss[E: Exec & Movable & Deinitable](mut s: OpsSession[E], fixture_id: Int, a: Int, b: Int) raises:
    var av: List[Float64] = [1.25, -2.75, 3.5, 4.0]
    var bv: List[Float64] = [2.0, 2.0, 1.0, 0.5]
    var tv: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    var hs: List[Int] = [a, b]
    var code = OPS_MAE
    var reduction = REDUCTION_SUM if fixture_id == 17 or fixture_id == 20 else REDUCTION_MEAN
    var ints: List[Int] = [reduction]
    var fs = List[Float32]()
    if fixture_id == 18:
        code = OPS_HUBER
        fs = [1.0]
    elif fixture_id >= 19:
        code = OPS_BCE_LOGITS
        hs = [a, _upload(s, tv, 2, 2)]
    var expected = Float64(0)
    for i in range(4):
        var d = abs(av[i] - bv[i])
        if fixture_id == 18:
            expected += 0.5 * d * d if d <= 1.0 else d - 0.5
        elif fixture_id >= 19:
            expected += max(av[i], 0.0) - av[i] * Float64(tv[i]) + log1p(exp(-abs(av[i])))
        else:
            expected += d
    if reduction == REDUCTION_MEAN:
        expected /= 4.0
    var result = s.run_op(code, hs, ints, fs)
    var dims: List[Int] = [1, 1]
    var expected_values: List[Float64] = [expected]
    var rtol = Float64(1e-5) if fixture_id >= 19 else Float64(1e-6)
    _check_tensor(s, result, expected_values, dims, rtol, 0.0)
    _close(s.scalar(result), expected, rtol, 0.0, "loss scalar")


def _metric[E: Exec & Movable & Deinitable](mut s: OpsSession[E], fixture_id: Int) raises:
    var y: List[Float32] = [1.0, 0.0, 1.0, 0.0]
    var score: List[Float32] = [0.9, 0.2, 0.4, 0.6]
    var pred: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    var yt = _upload(s, y, 4, 1)
    var sc = _upload(s, score, 4, 1)
    var hs: List[Int] = [yt, sc]
    var kind = OPS_METRIC_ROC_AUC
    var expected = Float64(0.75)  # Three of the four positive/negative pairs are ordered correctly.
    if fixture_id == 21:
        kind = OPS_METRIC_ACCURACY
        hs = [yt, _upload(s, pred, 4, 1)]
        expected = 0.5  # Two correct labels of four.
    elif fixture_id == 23:
        kind = OPS_METRIC_LOG_LOSS
        expected = 0.0
        for i in range(4):
            var p = Float64(score[i])
            var target = Float64(y[i])
            expected -= (target * log(p) + (1.0 - target) * log(1.0 - p)) / 4.0
    var actual = s.metric(kind, hs)
    if fixture_id == 21:
        require(actual == 0.5, "accuracy counts two of four labels")
    else:
        _close(actual, expected, 1e-5 if fixture_id == 23 else 1e-6, 0.0, "metric")


def run_ops_session_fixture[E: Exec & Movable & Deinitable](mut s: OpsSession[E], fixture_id: Int) raises:
    require(fixture_id >= 0 and fixture_id < OPS_SESSION_FIXTURES, "unknown fixture")
    if fixture_id < 21:
        var av: List[Float32] = [1.25, -2.75, 3.5, 4.0]
        var bv: List[Float32] = [2.0, 2.0, 1.0, 0.5]
        var a = _upload(s, av, 2, 2)
        var b = _upload(s, bv, 2, 2)
        if fixture_id < 16:
            _elementwise(s, fixture_id, a, b)
        else:
            _loss(s, fixture_id, a, b)
    elif fixture_id < 24:
        _metric(s, fixture_id)
    elif fixture_id == 24:
        var ints: List[Int] = [7, 3, 0, 0]
        var h = s.run_op(OPS_PERMUTATION, List[Int](), ints, List[Float32]())
        var dims: List[Int] = [7, 1]
        require(s.shape_nd(h).same(ShapeN.of(dims)), "permutation shape")
        var values = List[Float32](length=7, fill=-1)
        s.download(h, f32_ptr(Int(values.unsafe_ptr())))
        var seen = List[Int](length=7, fill=0)
        for v in values:
            require(v >= 0 and v < 7 and v == Float32(Int(v)), "permutation index range and integrality")
            seen[Int(v)] += 1
        for count in seen:
            require(count == 1, "permutation contains each index exactly once")
    elif fixture_id == 25:
        var av: List[Float32] = [1.25, -2.75, 3.5, 4.0]
        var hs: List[Int] = [_upload(s, av, 2, 2)]
        var ints: List[Int] = [-1]
        var h = s.run_op(OPS_SUM, hs, ints, List[Float32]())
        require(s.scalar(h) == 6.0, "sum scalar")
    elif fixture_id == 26:
        var values = List[Float32](length=16, fill=0)
        for i in range(16):
            values[i] = Float32(i)
        var dims: List[Int] = [1, 1, 4, 4]
        var source = CastIn(Int(values.unsafe_ptr()), CAST_KIND_FLOAT, 4, 4, 4, 16, 4)
        var a = s.upload_nd(source, ShapeN.of(dims), False)
        _ = values
        var hs: List[Int] = [a]
        var ints: List[Int] = [2, 2]
        var h = s.run_op(OPS_AVGPOOL2D, hs, ints, List[Float32]())
        var expected: List[Float64] = [2.5, 4.5, 10.5, 12.5]
        dims = [1, 1, 2, 2]
        _check_tensor(s, h, expected, dims, 1e-6, 0.0)
    else:
        var ints: List[Int] = [OPT_SGD, 0, SCHED_STEP, 1]
        var fs: List[Float32] = [0.5, 0.0, 0.9, 0.999, 1e-8, 0.0, 0.5]
        var optimizer = s.optimizer(ints, fs)
        require(s.learning_rate(optimizer) == 0.5, "initial step-scheduler learning rate")
        s.scheduler_step(optimizer)
        require(s.learning_rate(optimizer) == 0.25, "advanced step-scheduler learning rate")
