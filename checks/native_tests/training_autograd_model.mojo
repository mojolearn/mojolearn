# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""Sequential training over the autograd tape (moved from
python/mojolearn/tests/test_autograd_model.py, lane tmig-training-2,
2026-11-15), through the entries the Sequential door calls
(training_sequential_common.mojo: model_fit_native, model_predict).

Data: X (120, 5) standard normals (the counter generator of
core_ops_common.normal; the original drew NumPy's default_rng(0), and its
assertions never depended on the stream), yr = 2 x0 - x1 + 0.5 x2,
y3 = [x0 > 0] + [x1 > 0.5], yb = [x0 + x3 > 0].

  0 regression: linear(8).tanh.linear(1), mse, adam lr 0.02, 30 epochs of
    32 (seed 0): 30 epochs ran, the last loss is under half the first, 120
    predictions; linear(4).relu.linear(2), huber, momentum lr 0.01 on
    [yr, -yr], 10 epochs of 40 (seed 1): (120, 2) predictions, the loss falls.
  1 linear(6).layer_norm.relu.linear(6).residual(2).linear(1), mse, adamw
    lr 0.01 weight decay 0.01, step schedule (5, 0.5), 15 epochs of 30
    (seed 2): the loss falls.
  2 classifiers: linear(8).relu.linear(3), cross_entropy, adam lr 0.02, 20
    epochs of 32: the loss falls, softmax rows sum to 1 (atol 1e-5), the
    argmax codes are classes; linear(4).tanh.linear(1), bce_with_logits,
    sgd lr 0.1, 20 epochs of 20 without shuffle: the loss falls, (120, 2)
    probabilities, 120 class codes in {0, 1}.
  3 the same seed gives the same bits: linear(6).relu.linear(w) (mse w = 1,
    cross_entropy w = 3), adam lr 0.01, 5 epochs of 16, seed 7, fitted twice:
    the parameters and the loss curves are word for word equal.
  4 early stop: linear(1), sgd lr 1e-6, tol 10, n_iter_no_change 2, 50
    epochs of 60: fewer than 50 epochs run.
  5 dropout: linear(8).relu.dropout(0.2).linear(1), mse, lr 0.01, 5 epochs
    of 32 (seed 0): predictions are repeatable; 2 epochs then 3 resumed
    give the 5-epoch parameters word for word.
  6 rank-4 conv: X (48, 2, 7, 9), y = [mean x[0, :3] > mean x[1, 4:]];
    conv2d(3, 3, padding 1).relu.max_pool2d(2).flatten.linear(2),
    cross_entropy, adam lr 0.02, 12 epochs of 16 (seed 5), twice: the same
    bits, finite parameters, the loss falls, 3*2*9 + 3 + 36*2 + 2
    parameters, softmax rows sum to 1.
  7 rank-3: X (40, 5, 6), y = sum_t x[t, 0] - mean_c x[0, c];
    linear(4).tanh.permute(1, 0).reshape(-1).linear(1), mse, adam lr 0.02,
    10 epochs of 8 (seed 1): the loss falls, 40 predictions.

The inputs-as-they-are (dtype, order, strides, float16, labels' dtypes),
the refusals by name and the layer listing stay in the Python test.
"""
from std.math import isfinite
from std.memory import bitcast
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from autograd.model import (
    LOSS_MSE, LOSS_CROSS_ENTROPY, LOSS_BCE_WITH_LOGITS, LOSS_HUBER, TARGET_DENSE, TARGET_CODES,
    TARGET_CODE_VALUES, OUT_RAW, OUT_SOFTMAX, OUT_PROBA2, OUT_ARGMAX, OUT_THRESHOLD,
)
from checks.native_tests.core_ops_common import normal
from checks.native_tests.training_sequential_common import (
    SeqModel, codes_as_words, L_LINEAR, L_RELU, L_TANH, L_LAYER_NORM, L_DROPOUT, L_RESIDUAL, L_RESHAPE,
    L_PERMUTE, L_FLATTEN, L_CONV2D, L_MAXPOOL2D,
)
from checks.native_tests.training_check_common import same_words
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, CPU_COLUMN_BUILD, check_cases_serial, require


comptime AUTOGRAD_MODEL_CASES = 8
comptime _N = 120
comptime _D = 5


struct _Data(Movable):
    var x: List[Float32]
    var yr: List[Float32]
    var y3: List[Float32]   # int32 codes in float32 words
    var yb: List[Float32]   # int32 codes in float32 words

    def __init__(out self):
        var x = normal(_N * _D, 0)
        var yr = List[Float32](length=_N, fill=0)
        var c3 = List[Int](length=_N, fill=0)
        var cb = List[Int](length=_N, fill=0)
        for i in range(_N):  # small-loop(fixture targets): fixture-sized, no product work
            var r = _D * i
            yr[i] = x[r] * 2 - x[r + 1] + Float32(0.5) * x[r + 2]
            c3[i] = (1 if x[r] > 0 else 0) + (1 if x[r + 1] > Float32(0.5) else 0)
            cb[i] = 1 if x[r] + x[r + 3] > 0 else 0
        self.x = x^
        self.yr = yr^
        self.y3 = codes_as_words(c3)
        self.yb = codes_as_words(cb)


def _falls(m: SeqModel, what: String) raises:
    var c = m.losses()
    require(len(c) >= 2, what + ": fewer than two epochs")
    require(c[len(c) - 1] < c[0], what + ": the loss did not fall (" + String(c[0]) + " -> " + String(c[len(c) - 1]) + ")")


def _rows_sum_to_one(p: List[Float32], width: Int, what: String) raises:
    for r in range(len(p) // width):  # small-loop(check comparison): fixture-sized, no product work
        var s = Float64(0)
        for k in range(width):
            s += Float64(p[r * width + k])
        require(abs(s - 1.0) <= 1e-5, what + ": row " + String(r) + " sums to " + String(s))


def _codes_below(p: List[Float32], count: Int, what: String) raises:
    for i in range(len(p)):  # small-loop(check comparison): fixture-sized, no product work
        var c = p[i]
        require(c >= 0 and c < Float32(count) and Float32(Int(c)) == c, what + ": " + String(c) + " is not a class code")


def _two_layer(width: Int, code: Int, n_out: Int) -> SeqModel:
    var m = SeqModel()
    m.linear(width)
    m.act(code)
    m.linear(n_out)
    return m^


def run_autograd_model_case[E: Exec](mut ex: E, unit: Int) raises:
    var data = _Data()
    if unit == 0:
        var m = _two_layer(8, L_TANH, 1)
        m.compile(LOSS_MSE, "adam", 0.02)
        _ = m.fit(ex, data.x, _N, _D, data.yr, 1, TARGET_DENSE, 30, 32, -1, 0)
        require(m.n_iter == 30 and len(m.losses()) == 30, "30 epochs ran")
        var c = m.losses()
        require(c[29] < Float32(0.5) * c[0], "the regression loss fell by half")
        require(len(m.predict(ex, data.x, _N, _D, OUT_RAW)) == _N, "120 predictions")
        var y2 = List[Float32](length=2 * _N, fill=0)
        for i in range(_N):  # small-loop(fixture targets)
            y2[2 * i] = data.yr[i]
            y2[2 * i + 1] = -data.yr[i]
        var m2 = _two_layer(4, L_RELU, 2)
        m2.compile(LOSS_HUBER, "momentum", 0.01)
        _ = m2.fit(ex, data.x, _N, _D, y2, 2, TARGET_DENSE, 10, 40, -1, 1)
        require(len(m2.predict(ex, data.x, _N, _D, OUT_RAW)) == 2 * _N, "(120, 2) predictions")
        _falls(m2, "huber momentum")
        _ = y2^
    elif unit == 1:
        var m = SeqModel()
        m.linear(6)
        m.layer(L_LAYER_NORM, f=1e-5)
        m.act(L_RELU)
        m.linear(6)
        m.layer(L_RESIDUAL, 2)
        m.linear(1)
        m.compile(LOSS_MSE, "adamw", 0.01, weight_decay=0.01, sched=1, step_size=5, gamma=0.5)
        _ = m.fit(ex, data.x, _N, _D, data.yr, 1, TARGET_DENSE, 15, 30, -1, 2)
        _falls(m, "layer norm and residual")
        require(len(m.predict(ex, data.x, _N, _D, OUT_RAW)) == _N, "120 predictions")
    elif unit == 2:
        var c = _two_layer(8, L_RELU, 3)
        c.compile(LOSS_CROSS_ENTROPY, "adam", 0.02)
        _ = c.fit(ex, data.x, _N, _D, data.y3, 1, TARGET_CODES, 20, 32, -1, 0)
        _falls(c, "cross_entropy")
        var p = c.predict(ex, data.x, _N, _D, OUT_SOFTMAX)
        require(len(p) == 3 * _N, "(120, 3) probabilities")
        _rows_sum_to_one(p, 3, "softmax")
        _codes_below(c.predict(ex, data.x, _N, _D, OUT_ARGMAX), 3, "predict")
        var b = _two_layer(4, L_TANH, 1)
        b.compile(LOSS_BCE_WITH_LOGITS, "sgd", 0.1)
        _ = b.fit(ex, data.x, _N, _D, data.yb, 1, TARGET_CODE_VALUES, 20, 20, 0, 0)
        _falls(b, "bce_with_logits")
        require(len(b.predict(ex, data.x, _N, _D, OUT_PROBA2)) == 2 * _N, "(120, 2) probabilities")
        var labels = b.predict(ex, data.x, _N, _D, OUT_THRESHOLD)
        require(len(labels) == _N, "120 labels")
        _codes_below(labels, 2, "binary predict")
    elif unit == 3:
        for which in range(2):  # small-loop(2: the two small models)
            var width = 1 if which == 0 else 3
            var params = List[List[Float32]]()
            var curves = List[List[Float32]]()
            for _ in range(2):  # small-loop(2: two fits with one seed)
                var m = _two_layer(6, L_RELU, width)
                m.compile(LOSS_MSE if which == 0 else LOSS_CROSS_ENTROPY, "adam", 0.01)
                if which == 0:
                    _ = m.fit(ex, data.x, _N, _D, data.yr, 1, TARGET_DENSE, 5, 16, -1, 7)
                else:
                    _ = m.fit(ex, data.x, _N, _D, data.y3, 1, TARGET_CODES, 5, 16, -1, 7)
                params.append(m.params.copy())
                curves.append(m.losses())
            require(same_words(params[0], params[1]), "the same seed gives the same parameter bits (" + String(which) + ")")
            require(same_words(curves[0], curves[1]), "the same seed gives the same loss curve (" + String(which) + ")")
    elif unit == 4:
        var m = SeqModel()
        m.linear(1)
        m.compile(LOSS_MSE, "sgd", 1e-6, tol=10.0, patience=2)
        _ = m.fit(ex, data.x, _N, _D, data.yr, 1, TARGET_DENSE, 50, 60, -1, 0)
        require(m.n_iter < 50, "early stopping ended the fit before 50 epochs")
    elif unit == 5:
        var m = SeqModel()
        m.linear(8)
        m.act(L_RELU)
        m.layer(L_DROPOUT, f=0.2)
        m.linear(1)
        m.compile(LOSS_MSE, "adam", 0.01)
        _ = m.fit(ex, data.x, _N, _D, data.yr, 1, TARGET_DENSE, 5, 32, -1, 0)
        var first = m.predict(ex, data.x, _N, _D, OUT_RAW)
        require(len(first) == _N, "120 predictions")
        require(same_words(first, m.predict(ex, data.x, _N, _D, OUT_RAW)), "dropout is off in predict")
        var other = SeqModel()
        other.linear(8)
        other.act(L_RELU)
        other.layer(L_DROPOUT, f=0.2)
        other.linear(1)
        other.compile(LOSS_MSE, "adam", 0.01)
        _ = other.fit(ex, data.x, _N, _D, data.yr, 1, TARGET_DENSE, 2, 32, -1, 0)
        _ = other.fit(ex, data.x, _N, _D, data.yr, 1, TARGET_DENSE, 3, 32, -1, 0, True)
        require(same_words(m.params, other.params), "2 + 3 resumed epochs equal 5 epochs")
    elif unit == 6:
        var n = 48
        var cols = 2 * 7 * 9
        var x = normal(n * cols, 3)
        var codes = List[Int](length=n, fill=0)
        for i in range(n):  # small-loop(fixture targets)
            var a = Float64(0)
            var b = Float64(0)
            for r in range(3):
                for c in range(9):
                    a += Float64(x[i * cols + r * 9 + c])
            for r in range(4, 7):
                for c in range(9):
                    b += Float64(x[i * cols + 63 + r * 9 + c])
            codes[i] = 1 if a / 27.0 > b / 27.0 else 0
        var y = codes_as_words(codes)
        var params = List[List[Float32]]()
        var model = SeqModel()
        for _ in range(2):  # small-loop(2: two fits with one seed)
            var m = SeqModel()
            m.sample = [2, 7, 9]
            m.layer(L_CONV2D, 3, 3, 1, 1, 1, 1)
            m.layer(L_RELU)
            m.layer(L_MAXPOOL2D, 2, 2)
            m.layer(L_FLATTEN)
            m.layer(L_LINEAR, 2, 1)
            m.compile(LOSS_CROSS_ENTROPY, "adam", 0.02)
            _ = m.fit(ex, x, n, cols, y, 1, TARGET_CODES, 12, 16, -1, 5)
            params.append(m.params.copy())
            model = m^
        require(same_words(params[0], params[1]), "the conv model's same seed gives the same bits")
        require(len(model.params) == 3 * 2 * 9 + 3 + 36 * 2 + 2, "the conv model's parameter count")
        for i in range(len(model.params)):  # small-loop(check comparison)
            require(isfinite(model.params[i]), "finite conv parameters")
        _falls(model, "the conv model")
        var p = model.predict(ex, x, n, cols, OUT_SOFTMAX)
        require(len(p) == 2 * n, "(48, 2) probabilities")
        _rows_sum_to_one(p, 2, "conv softmax")
        _ = x^
        _ = y^
    else:
        require(unit == 7, "unknown autograd model case")
        var n = 40
        var cols = 30
        var x = normal(n * cols, 4)
        var y = List[Float32](length=n, fill=0)
        for i in range(n):  # small-loop(fixture targets)
            var s = Float64(0)
            for t in range(5):
                s += Float64(x[i * cols + t * 6])
            var mean = Float64(0)
            for c in range(6):
                mean += Float64(x[i * cols + c])
            y[i] = Float32(s - mean / 6.0)
        var m = SeqModel()
        m.sample = [5, 6]
        m.layer(L_LINEAR, 4, 1)
        m.layer(L_TANH)
        m.layer(L_PERMUTE, 1, 0, -1, -1, -1, -1)
        m.layer(L_RESHAPE, -1, 0, 0, 0, 0, 0)
        m.layer(L_LINEAR, 1, 1)
        m.compile(LOSS_MSE, "adam", 0.02)
        _ = m.fit(ex, x, n, cols, y, 1, TARGET_DENSE, 10, 8, -1, 1)
        _falls(m, "the rank-3 model")
        require(len(m.predict(ex, x, n, cols, OUT_RAW)) == n, "40 predictions")
        _ = x^
        _ = y^
    _ = data^


# ---- The host and device columns ----
# The host column: the eight independent cases run across host cores, each on its own HostExec. The device column:
# the cases run in order on one DeviceExec: they share the one device and its queue, and each fit's kernels already
# spread across the device.
def test_autograd_model() raises:
    comptime if CPU_COLUMN_BUILD:
        def host_case(unit: Int) raises:
            var ex = HostExec()
            run_autograd_model_case(ex, unit)
        check_cases("autograd_model", AUTOGRAD_MODEL_CASES, host_case)
    else:
        var ex = DeviceExec()
        def device_case(unit: Int) raises {mut ex}:
            run_autograd_model_case(ex, unit)
        check_cases_serial("autograd_model", AUTOGRAD_MODEL_CASES, device_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
