# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The losses of lane nn-losses (2026-11-23) for `Sequential.compile(loss=...)`: l1, smooth_l1 (beta),
nll (on log-probabilities), kl_div (log_target=False; mean, sum, batchmean) and bce_with_logits with pos_weight,
each under reduction mean and sum (autograd/nn_losses.mojo bodies through autograd/loss_ops.mojo).

  - test_nn_losses_tape: every loss's value and its gradients in the prediction AND the target (Tape.loss_reduced
    through autograd/losses.mojo) against a float64 oracle written here; the oracle's closed-form derivatives
    against central finite differences of the oracle's reduced loss, element by element (h = 1e-6, 1e-5); the
    trainers' loss_value_into the same bits as the tape. bce_with_logits with pos_weight 1 skips the weighted
    term in the body (autograd/nn_losses.mojo), so its words are the unweighted loss's.
  - test_nn_losses_fit: small Sequential fits through the public fit entry (model_fit_native): l1 and smooth_l1
    (beta 0.5, reduction sum) regressions and a bce_with_logits (pos_weight 2) classifier, each with finite loss,
    parameters and predictions and a stated accuracy; nll and kl_div fits of a linear log-softmax model on the
    tape (Sequential has no log_softmax layer yet), each with a stated accuracy or loss.
  - one `NN_LOSSES <case> digest` line over the value and gradient words.
Columns: the CPU column runs on HostExec. The device column runs each case on DeviceExec AND HostExec and requires
the same words.
Sabotage: `-D MOJOLEARN_NN_LOSSES_SABOTAGE_SMOOTH_L1_HALF` drops the 0.5 of smooth_l1's quadratic part (the value
only; the gradient body is unchanged, so the fit's parameters and its pinned digest do not move); the oracle value
checks of smooth_l1 must FAIL (CAUGHT at f503904838 on the CPU column: smooth_l1_b0.5/mean/value 0.65125 against
the float64 oracle 0.650625).
"""
from std.math import exp, log, log1p
from std.memory import bitcast
from std.testing import TestSuite
from autograd.tape import Tape, Tensor
from autograd.losses import (
    REDUCTION_MEAN, REDUCTION_SUM, REDUCTION_BATCHMEAN, smooth_l1, nll, kl_div, mae,
    binary_cross_entropy_with_logits, loss_scratch, release_loss_scratch, loss_value_into, ls_per_row,
)
from autograd.loss_ops import LS_MAE, LS_BCE_LOGITS
from autograd.nn_losses import LS_SMOOTH_L1, LS_NLL, LS_KL_DIV
from autograd.model import (
    LOSS_L1, LOSS_SMOOTH_L1, LOSS_BCE_WITH_LOGITS, LOSS_WORD_REDUCTION, TARGET_DENSE, TARGET_CODE_VALUES,
    OUT_RAW, OUT_THRESHOLD,
)
from sequence.ops import FP
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from checks.native_tests.training_sequential_common import SeqModel, L_LINEAR, L_TANH
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD


def _fail(message: String) raises:
    print("FAIL NN_LOSSES", message)
    raise Error(message)


def _require(ok: Bool, message: String) raises:
    if not ok:
        _fail(message)


def _digest(words: List[Float32]) -> UInt64:
    var d = UInt64(1469598103934665603)
    for i in range(len(words)):  # small-loop(check digest): fixture-sized, no product work
        d = (d ^ UInt64(bitcast[DType.uint32](words[i]))) * UInt64(1099511628211)
    return d


def _close(name: String, actual: List[Float32], expected: List[Float64]) raises:
    if len(actual) != len(expected):
        _fail(name + ": wrong length")
    for i in range(len(actual)):  # small-loop(check comparison): fixture-sized
        var e = expected[i]
        var tol = 3.0e-4 + 3.0e-4 * abs(e)
        if not (abs(Float64(actual[i]) - e) <= tol):
            _fail(name + ": index " + String(i) + " is " + String(actual[i]) + ", the float64 oracle " + String(e))


def _same_words(name: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        _fail(name + ": host and device word counts differ")
    for i in range(len(a)):  # small-loop(check comparison): fixture-sized
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            _fail(name + ": host and device words differ at " + String(i) + " (" + String(a[i]) + " vs "
                  + String(b[i]) + ")")


# ---------------------------------------------------------------- float64 oracle
def _sp_neg64(x: Float64) -> Float64:
    """softplus(-x)."""
    return max(-x, 0.0) + log1p(exp(-abs(x)))


def _sigmoid64(x: Float64) -> Float64:
    return 1.0 / (1.0 + exp(-x))


def _term64(kind: Int, p: Float64, y: Float64, a: Float64) -> Float64:
    if kind == LS_MAE:
        return abs(p - y)
    if kind == LS_SMOOTH_L1:
        var d = p - y
        if abs(d) < a:
            return 0.5 * d * d / a
        return abs(d) - 0.5 * a
    if kind == LS_NLL:
        return -y * p
    if kind == LS_KL_DIV:
        if y > 0:
            return y * (log(y) - p)
        return 0.0
    # LS_BCE_LOGITS, a = pos_weight: -[w y log s + (1 - y) log(1 - s)]
    return (1.0 - y) * p + (1.0 + (a - 1.0) * y) * _sp_neg64(p)


def _dp64(kind: Int, p: Float64, y: Float64, a: Float64) -> Float64:
    if kind == LS_MAE:
        return 1.0 if p > y else (-1.0 if p < y else 0.0)
    if kind == LS_SMOOTH_L1:
        var d = p - y
        if abs(d) < a:
            return d / a
        return 1.0 if d > 0 else -1.0
    if kind == LS_NLL or kind == LS_KL_DIV:
        return -y
    return (1.0 + (a - 1.0) * y) * _sigmoid64(p) - a * y


def _dy64(kind: Int, p: Float64, y: Float64, a: Float64) -> Float64:
    if kind == LS_MAE or kind == LS_SMOOTH_L1:
        return -_dp64(kind, p, y, a)
    if kind == LS_NLL:
        return -p
    if kind == LS_KL_DIV:
        if y > 0:
            return log(y) + 1.0 - p
        return 0.0
    return -p + (a - 1.0) * _sp_neg64(p)


def _factor64(kind: Int, reduction: Int, rows: Int, count: Int) -> Float64:
    if reduction == REDUCTION_SUM:
        return 1.0
    if ls_per_row(kind, reduction):
        return 1.0 / Float64(rows)
    return 1.0 / Float64(count)


def _total64(kind: Int, p: List[Float64], y: List[Float64], a: Float64, factor: Float64) -> Float64:
    var s = 0.0
    for i in range(len(p)):  # small-loop(oracle): fixture-sized
        s += _term64(kind, p[i], y[i], a)
    return s * factor


def _fd_check(name: String, kind: Int, p: List[Float64], y: List[Float64], a: Float64, factor: Float64) raises:
    """Every element's closed-form derivative (prediction and target) against central differences of the
    oracle's reduced loss (inputs chosen away from the kinks)."""
    comptime H = 1.0e-6
    for i in range(len(p)):  # small-loop(oracle): fixture-sized
        var pp = p.copy()
        var pm = p.copy()
        pp[i] += H
        pm[i] -= H
        var fd_p = (_total64(kind, pp, y, a, factor) - _total64(kind, pm, y, a, factor)) / (2.0 * H)
        if abs(fd_p - factor * _dp64(kind, p[i], y[i], a)) > 1.0e-5:
            _fail(name + ": d/dprediction " + String(i) + " disagrees with its central difference")
        if kind == LS_KL_DIV and y[i] <= 0.0:
            continue  # y log y has no derivative at 0; the body's convention there (0) is checked by value
        var yp = y.copy()
        var ym = y.copy()
        yp[i] += H
        ym[i] -= H
        var fd_y = (_total64(kind, p, yp, a, factor) - _total64(kind, p, ym, a, factor)) / (2.0 * H)
        if abs(fd_y - factor * _dy64(kind, p[i], y[i], a)) > 1.0e-5:
            _fail(name + ": d/dtarget " + String(i) + " disagrees with its central difference")


# ---------------------------------------------------------------- tape cases
@fieldwise_init
struct LossCase(Copyable, Movable):
    var name: String
    var kind: Int
    var reduction: Int
    var param: Float32
    var rows: Int
    var cols: Int
    var pred: List[Float32]
    var target: List[Float32]


def _log_softmax_rows(z: List[Float64], rows: Int, cols: Int) -> List[Float32]:
    var out = List[Float32]()
    for r in range(rows):  # small-loop(fixture)
        var m = z[r * cols]
        for k in range(cols):  # small-loop(fixture)
            m = max(m, z[r * cols + k])
        var s = 0.0
        for k in range(cols):  # small-loop(fixture)
            s += exp(z[r * cols + k] - m)
        for k in range(cols):  # small-loop(fixture)
            out.append(Float32(z[r * cols + k] - m - log(s)))
    return out^


def _cases() -> List[LossCase]:
    var pred: List[Float32] = [0.3, -0.8, 1.2, 0.7, 0.2, -0.4, 2.1, -1.7]
    var target: List[Float32] = [0.9, 1.3, 0.6, -0.1, 0.25, -0.35, 0.4, -2.6]
    var bce_target: List[Float32] = [0.0, 1.0, 0.25, 1.0, 0.0, 0.6, 1.0, 0.0]
    var z: List[Float64] = [0.3, -0.8, 1.2, 0.7, 2.5, 0.1, -1.0, 0.0, -0.4, -0.4, 0.9, 3.0]
    var logp = _log_softmax_rows(z, 3, 4)
    var onehot: List[Float32] = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    # target distributions, one zero entry (KLDivLoss's xlogy convention)
    var dist: List[Float32] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.25, 0.25, 0.05, 0.15, 0.6, 0.2]
    var cases = List[LossCase]()
    for reduction in [REDUCTION_MEAN, REDUCTION_SUM]:
        var r = String("mean") if reduction == REDUCTION_MEAN else String("sum")
        cases.append(LossCase("l1/" + r, LS_MAE, reduction, 1, 2, 4, pred.copy(), target.copy()))
        cases.append(LossCase("smooth_l1_b0.5/" + r, LS_SMOOTH_L1, reduction, 0.5, 2, 4, pred.copy(),
                              target.copy()))
        cases.append(LossCase("smooth_l1_b2/" + r, LS_SMOOTH_L1, reduction, 2, 2, 4, pred.copy(), target.copy()))
        cases.append(LossCase("bce_pos1/" + r, LS_BCE_LOGITS, reduction, 1, 2, 4, pred.copy(), bce_target.copy()))
        cases.append(LossCase("bce_pos2.5/" + r, LS_BCE_LOGITS, reduction, 2.5, 2, 4, pred.copy(),
                              bce_target.copy()))
        cases.append(LossCase("bce_pos0.3/" + r, LS_BCE_LOGITS, reduction, 0.3, 2, 4, pred.copy(),
                              bce_target.copy()))
        cases.append(LossCase("nll/" + r, LS_NLL, reduction, 1, 3, 4, logp.copy(), onehot.copy()))
        cases.append(LossCase("kl_div/" + r, LS_KL_DIV, reduction, 1, 3, 4, logp.copy(), dist.copy()))
    cases.append(LossCase("kl_div/batchmean", LS_KL_DIV, REDUCTION_BATCHMEAN, 1, 3, 4, logp.copy(), dist.copy()))
    return cases^


def run_tape_case[E: Exec](mut ex: E, c: LossCase, oracle: Bool) raises -> List[Float32]:
    """The case's value, prediction gradient and target gradient words."""
    var tape = Tape()
    var x = tape.tensor(ex, c.pred, c.rows, c.cols, True)
    var y = tape.tensor(ex, c.target, c.rows, c.cols, True)
    var out: Tensor
    if c.kind == LS_MAE:
        out = mae(tape, ex, x, y, c.reduction)
    elif c.kind == LS_SMOOTH_L1:
        out = smooth_l1(tape, ex, x, y, c.param, c.reduction)
    elif c.kind == LS_NLL:
        out = nll(tape, ex, x, y, c.reduction)
    elif c.kind == LS_KL_DIV:
        out = kl_div(tape, ex, x, y, c.reduction)
    else:
        out = binary_cross_entropy_with_logits(tape, ex, x, y, c.reduction, c.param)
    var value = tape.values(ex, out)
    tape.backward(ex, out)
    var gp = tape.gradients(ex, x)
    var gy = tape.gradients(ex, y)
    var count = c.rows * c.cols
    if oracle:
        var p64 = List[Float64]()
        var y64 = List[Float64]()
        for i in range(count):  # small-loop(fixture)
            p64.append(Float64(c.pred[i]))
            y64.append(Float64(c.target[i]))
        var a = Float64(c.param)
        var factor = _factor64(c.kind, c.reduction, c.rows, count)
        _fd_check(c.name, c.kind, p64, y64, a, factor)
        var ev: List[Float64] = [_total64(c.kind, p64, y64, a, factor)]
        var ep = List[Float64]()
        var ey = List[Float64]()
        for i in range(count):  # small-loop(fixture)
            ep.append(factor * _dp64(c.kind, p64[i], y64[i], a))
            ey.append(factor * _dy64(c.kind, p64[i], y64[i], a))
        _close(c.name + "/value", value, ev)
        _close(c.name + "/d_prediction", gp, ep)
        _close(c.name + "/d_target", gy, ey)
        # the trainers' path: the same body and fold, the same word
        var ws = loss_scratch(ex, count)
        var word = ex.alloc(1)
        loss_value_into(ex, c.kind, tape.value(x), tape.value(y), c.rows, c.cols, c.reduction, word, ws, c.param)
        var direct = List[Float32](length=1, fill=Float32(0))
        ex.download(FP(unsafe_from_address=Int(direct.unsafe_ptr())), word, 1)
        if bitcast[DType.uint32](direct[0]) != bitcast[DType.uint32](value[0]):
            _fail(c.name + ": loss_value_into differs in bits from the tape loss")
        release_loss_scratch(ex, ws)
        ex.free(word)
    var words = List[Float32]()
    words.append(value[0])
    for i in range(count):  # small-loop(fixture)
        words.append(gp[i])
    for i in range(count):  # small-loop(fixture)
        words.append(gy[i])
    _finite(value, c.name + " value")
    print("NN_LOSSES", c.name, "digest", _digest(words))
    tape.release(ex)
    return words^


def test_nn_losses_tape() raises:
    var cases = _cases()
    for k in range(len(cases)):  # driver-loop: over the loss cases
        var host = HostExec()
        var hw = run_tape_case(host, cases[k], True)
        comptime if not CPU_COLUMN_BUILD:
            var dev = DeviceExec()
            var dw = run_tape_case(dev, cases[k], True)
            _same_words(cases[k].name, hw, dw)
    print("PASS NN_LOSSES tape cases", len(cases))


# ---------------------------------------------------------------- Sequential fits
comptime _N = 96
# the smooth_l1 fit's words on the CPU column at 80efc181cf (cpc2); the device column must give the same words
comptime _SMOOTH_L1_FIT_DIGEST = UInt64(5497763799716909111)
comptime _D = 4


def _data() -> Tuple[List[Float32], List[Float32], List[Float32]]:
    """X (96, 4) in [-1, 1), a real target y = 0.8 x0 - 0.5 x1 + 0.3 x2 x3, and binary labels x0 + x1 > 0 as
    int32 code words."""
    var state = UInt64(20261123)
    var x = List[Float32]()
    for _ in range(_N * _D):  # small-loop(fixture)
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        x.append(Float32(Int((state >> 33) % UInt64(2000)) - 1000) * Float32(0.001))
    var yr = List[Float32]()
    var yb = List[Float32]()
    for i in range(_N):  # small-loop(fixture)
        var b = i * _D
        yr.append(Float32(0.8) * x[b] - Float32(0.5) * x[b + 1] + Float32(0.3) * x[b + 2] * x[b + 3])
        yb.append(bitcast[DType.float32](Int32(1 if x[b] + x[b + 1] > 0 else 0)))
    return (x^, yr^, yb^)


def _finite(words: List[Float32], what: String) raises:
    for i in range(len(words)):  # small-loop(check): fixture-sized
        var v = words[i]
        if not (v == v) or abs(v) > Float32(3.0e38):
            _fail(what + ": word " + String(i) + " is not finite")


def _regressor(loss_word: Int, scalar: Float64) raises -> SeqModel:
    var m = SeqModel()
    m.layer(L_LINEAR, 8, 1)
    m.layer(L_TANH)
    m.layer(L_LINEAR, 1, 1)
    m.compile(LOSS_L1, "adam", 0.03)
    m.loss = loss_word
    m.huber_delta = scalar
    return m^


def run_fit_case[E: Exec](mut ex: E, which: Int) raises -> List[Float32]:
    """0: l1 (mean); 1: smooth_l1 (beta 0.5, sum); 2: bce_with_logits (pos_weight 2). The parameters and
    predictions, checked finite and against the stated accuracy."""
    var data = _data()
    var x = data[0].copy()
    var yr = data[1].copy()
    var yb = data[2].copy()
    var words = List[Float32]()
    if which < 2:
        var word = LOSS_L1 if which == 0 else LOSS_SMOOTH_L1 + LOSS_WORD_REDUCTION * REDUCTION_SUM
        var m = _regressor(word, 1.0 if which == 0 else 0.5)
        if which == 1:
            m.lr = 0.002
        _ = m.fit(ex, x, _N, _D, yr, 1, TARGET_DENSE, 60, 16, -1, 3)
        var p = m.predict(ex, x, _N, _D, OUT_RAW)
        _finite(m.losses(), "fit " + String(which) + " loss curve")
        _finite(m.params, "fit " + String(which) + " parameters")
        _finite(p, "fit " + String(which) + " predictions")
        var mae = 0.0
        var spread = 0.0
        for i in range(_N):  # small-loop(check): fixture-sized
            mae += abs(Float64(p[i]) - Float64(yr[i]))
            spread += abs(Float64(yr[i]))
        mae /= Float64(_N)
        spread /= Float64(_N)
        # stated accuracy: the mean absolute error is under a quarter of the targets' mean magnitude
        _require(mae < 0.25 * spread, "fit " + String(which) + ": mean absolute error " + String(mae)
                 + " is not under a quarter of the targets' mean magnitude " + String(spread))
        var c = m.losses()
        _require(c[len(c) - 1] < c[0], "fit " + String(which) + ": the loss did not fall")
        print("NN_LOSSES fit", which, "mae", mae, "of", spread, "loss", c[0], "->", c[len(c) - 1])
        words = m.params.copy()
        for i in range(len(p)):  # small-loop(fixture)
            words.append(p[i])
    else:
        var m = _regressor(LOSS_BCE_WITH_LOGITS, 2.0)
        _ = m.fit(ex, x, _N, _D, yb, 1, TARGET_CODE_VALUES, 40, 16, -1, 5)
        var labels = m.predict(ex, x, _N, _D, OUT_THRESHOLD)
        _finite(m.losses(), "bce fit loss curve")
        _finite(m.params, "bce fit parameters")
        var right = 0
        for i in range(_N):  # small-loop(check): fixture-sized
            var truth = Int(bitcast[DType.int32](yb[i]))
            if Int(labels[i]) == truth:
                right += 1
        var accuracy = Float64(right) / Float64(_N)
        # stated accuracy: at least 0.9 on the separable labels
        _require(accuracy >= 0.9, "bce_with_logits pos_weight 2: accuracy " + String(accuracy) + " under 0.9")
        print("NN_LOSSES fit bce pos_weight 2 accuracy", accuracy)
        words = m.params.copy()
        for i in range(len(labels)):  # small-loop(fixture)
            words.append(labels[i])
    var digest = _digest(words)
    print("NN_LOSSES fit_" + String(which), "digest", digest)
    if which == 1 and digest != _SMOOTH_L1_FIT_DIGEST:
        _fail("fit_1 (smooth_l1): digest " + String(digest) + " is not this version's recorded "
              + String(_SMOOTH_L1_FIT_DIGEST) + " (a changed smooth_l1 body: the sabotage arm, or a product change "
              + "whose new words must be recorded here)")
    return words^


def run_tape_fit[E: Exec](mut ex: E, kind: Int) raises -> List[Float32]:
    """A linear log-softmax model W (4, 3) fitted by full-batch gradient descent on the tape (60 steps, lr 0.5):
    nll on the labels argmax(X T) (stated accuracy >= 0.9), or kl_div batchmean on the teacher's softmax(X T)
    (stated final loss < 0.25 of the first). Sequential has no log_softmax layer yet, so these fits run here."""
    var data = _data()
    var x = data[0].copy()
    var teacher: List[Float32] = [2.0, -1.0, 0.5, -1.5, 2.0, 0.0, 1.0, 1.0, -2.0, 0.5, -0.5, 1.5]
    var C = 3
    var zt = List[Float64]()
    for i in range(_N):  # small-loop(fixture)
        for k in range(C):  # small-loop(fixture)
            var s = 0.0
            for d in range(_D):  # small-loop(fixture)
                s += Float64(x[i * _D + d]) * Float64(teacher[d * C + k])
            zt.append(s)
    var target = List[Float32]()
    var labels = List[Int]()
    if kind == LS_NLL:
        for i in range(_N):  # small-loop(fixture)
            var best = 0
            for k in range(1, C):  # small-loop(fixture)
                if zt[i * C + k] > zt[i * C + best]:
                    best = k
            labels.append(best)
            for k in range(C):  # small-loop(fixture)
                target.append(Float32(1) if k == best else Float32(0))
    else:
        var lp = _log_softmax_rows(zt, _N, C)
        for i in range(len(lp)):  # small-loop(fixture)
            target.append(Float32(exp(Float64(lp[i]))))
    var w = List[Float32](length=_D * C, fill=Float32(0))
    var tape = Tape()
    var first = Float32(0)
    var last = Float32(0)
    for step in range(60):  # driver-loop: over the check's gradient steps
        tape.reset()
        var xt = tape.tensor(ex, x, _N, _D, False)
        var wt = tape.tensor(ex, w, _D, C, True)
        var yt = tape.tensor(ex, target, _N, C, False)
        var logp = tape.log_softmax(ex, tape.matmul(ex, xt, wt))
        var loss: Tensor
        if kind == LS_NLL:
            loss = nll(tape, ex, logp, yt)
        else:
            loss = kl_div(tape, ex, logp, yt, REDUCTION_BATCHMEAN)
        tape.backward(ex, loss)
        var lv = tape.values(ex, loss)[0]
        if step == 0:
            first = lv
        last = lv
        var g = tape.gradients(ex, wt)
        for k in range(len(w)):  # small-loop(check update): fixture-sized
            w[k] = w[k] - Float32(0.5) * g[k]
    _finite(w, "tape fit weights")
    _require(last == last and last < first, "tape fit: the loss did not fall")
    if kind == LS_NLL:
        var right = 0
        for i in range(_N):  # small-loop(check)
            var best = 0
            var bz = Float64(-1.0e30)
            for k in range(C):  # small-loop(check)
                var s = 0.0
                for d in range(_D):  # small-loop(check)
                    s += Float64(x[i * _D + d]) * Float64(w[d * C + k])
                if s > bz:
                    bz = s
                    best = k
            if best == labels[i]:
                right += 1
        var accuracy = Float64(right) / Float64(_N)
        _require(accuracy >= 0.9, "nll fit: accuracy " + String(accuracy) + " under 0.9")
        print("NN_LOSSES tape fit nll accuracy", accuracy, "loss", first, "->", last)
    else:
        _require(last < Float32(0.25) * first, "kl_div fit: the loss " + String(last) + " is not under a quarter of "
                 + String(first))
        print("NN_LOSSES tape fit kl_div loss", first, "->", last)
    tape.release(ex)
    var words = w.copy()
    words.append(last)
    print("NN_LOSSES tape_fit_" + String(kind), "digest", _digest(words))
    return words^


def test_nn_losses_fit() raises:
    for which in range(3):  # driver-loop: over the Sequential fits
        var host = HostExec()
        var hw = run_fit_case(host, which)
        comptime if not CPU_COLUMN_BUILD:
            var dev = DeviceExec()
            var dw = run_fit_case(dev, which)
            _same_words("fit_" + String(which), hw, dw)
    for kind in [LS_NLL, LS_KL_DIV]:
        var host = HostExec()
        var hw = run_tape_fit(host, kind)
        comptime if not CPU_COLUMN_BUILD:
            var dev = DeviceExec()
            var dw = run_tape_fit(dev, kind)
            _same_words("tape_fit_" + String(kind), hw, dw)
    print("PASS NN_LOSSES fits 5")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
