# SPDX-License-Identifier: Apache-2.0
# Value and gradient checks of autograd/losses.mojo and value checks of
# autograd/metrics.mojo against float64 oracles written here (host loops in
# a check are the oracle, not runtime). Each PASS line carries the digest
# of the float32 result (autograd/checks/support.mojo), so the CPU and GPU
# runs can be compared digest for digest.
from std.math import exp, log, log1p, sqrt
from std.memory import bitcast
from autograd import Tape
from autograd.losses import (
    mse, mae, huber, binary_cross_entropy_with_logits, cross_entropy, REDUCTION_MEAN,
    REDUCTION_SUM, loss_scratch, release_loss_scratch, loss_value_into,
)
from autograd.loss_ops import (
    LS_MSE, LS_MAE, LS_HUBER, LS_BCE_LOGITS, LS_LOG_PROB, LS_BINARY_LOG_PROB, LS_CROSS_ENTROPY,
)
from autograd.metrics import (
    accuracy, accuracy_from_scores, log_loss, mean_squared_error, root_mean_squared_error,
    mean_absolute_error, r2_score, roc_auc,
)
from autograd.checks.support import check_values
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, OP_MSE, OP_CE


# ------------------------------------------------------------------ oracles
def _clip64(p: Float64) -> Float64:
    comptime EPS = 1.1920928955078125e-07
    if p < EPS:
        return EPS
    if p > 1.0 - EPS:
        return 1.0 - EPS
    return p


def _term64(kind: Int, p: Float64, y: Float64, delta: Float64) -> Float64:
    var d = p - y
    if kind == LS_MSE:
        return d * d
    if kind == LS_MAE:
        return abs(d)
    if kind == LS_HUBER:
        if abs(d) < delta:
            return 0.5 * d * d
        return delta * (abs(d) - 0.5 * delta)
    if kind == LS_BCE_LOGITS:
        return max(p, 0.0) - p * y + log1p(exp(-abs(p)))
    if kind == LS_LOG_PROB:
        if y == 0.0:
            return 0.0
        return -y * log(_clip64(p))
    var pc = _clip64(p)
    var acc = 0.0
    if y != 0.0:
        acc += y * log(pc)
    if 1.0 - y != 0.0:
        acc += (1.0 - y) * log(1.0 - pc)
    return -acc


def _sign64(x: Float64) -> Float64:
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return 0.0


def _dp64(kind: Int, p: Float64, y: Float64, delta: Float64) -> Float64:
    var d = p - y
    if kind == LS_MSE:
        return 2.0 * d
    if kind == LS_MAE:
        return _sign64(d)
    if kind == LS_HUBER:
        if abs(d) < delta:
            return d
        return delta * _sign64(d)
    return 1.0 / (1.0 + exp(-p)) - y


def _dy64(kind: Int, p: Float64, y: Float64, delta: Float64) -> Float64:
    if kind == LS_BCE_LOGITS:
        return -p
    return -_dp64(kind, p, y, delta)


def _check_oracle_derivative(name: String, kind: Int, p: Float64, y: Float64, delta: Float64) raises:
    # The closed-form oracle derivatives against central differences of
    # the oracle loss itself (inputs are chosen away from kinks).
    comptime H = 1.0e-6
    var fd_p = (_term64(kind, p + H, y, delta) - _term64(kind, p - H, y, delta)) / (2.0 * H)
    var fd_y = (_term64(kind, p, y + H, delta) - _term64(kind, p, y - H, delta)) / (2.0 * H)
    if abs(fd_p - _dp64(kind, p, y, delta)) > 1.0e-5 or abs(fd_y - _dy64(kind, p, y, delta)) > 1.0e-5:
        raise Error(name + ": oracle derivative disagrees with its central difference")


def _same_bits(a: Float32, b: Float32) -> Bool:
    return bitcast[DType.uint32](a) == bitcast[DType.uint32](b)


def _upload[E: Exec](mut ex: E, values: List[Float32]) raises -> FP:
    var p = ex.alloc(len(values))
    ex.upload(p, FP(unsafe_from_address=Int(values.unsafe_ptr())), len(values))
    return p


def _download[E: Exec](mut ex: E, p: FP, n: Int) raises -> List[Float32]:
    var out = List[Float32](length=n, fill=Float32(0))
    ex.download(FP(unsafe_from_address=Int(out.unsafe_ptr())), p, n)
    return out^


def _f32(x: Float64) -> Float32:
    return Float32(x)


# ------------------------------------------------------------------ tape losses
def _elementwise_case[E: Exec](
    mut ex: E, mut tape: Tape, name: String, kind: Int, reduction: Int, delta: Float32,
    pred: List[Float32], target: List[Float32],
) raises:
    tape.reset()
    var rows = 2
    var cols = len(pred) // 2
    var count = len(pred)
    var x = tape.tensor(ex, pred, rows, cols, True)
    var y = tape.tensor(ex, target, rows, cols, True)
    var out = tape.scalar(ex, 0)
    if kind == LS_MSE:
        out = mse(tape, ex, x, y, reduction)
    elif kind == LS_MAE:
        out = mae(tape, ex, x, y, reduction)
    elif kind == LS_HUBER:
        out = huber(tape, ex, x, y, delta, reduction)
    else:
        out = binary_cross_entropy_with_logits(tape, ex, x, y, reduction)
    var factor = 1.0 / Float64(count) if reduction == REDUCTION_MEAN else 1.0
    var value = 0.0
    var gp = List[Float32]()
    var gy = List[Float32]()
    for i in range(count):
        var p = Float64(pred[i])
        var t = Float64(target[i])
        _check_oracle_derivative(name, kind, p, t, Float64(delta))
        value += _term64(kind, p, t, Float64(delta))
        gp.append(_f32(factor * _dp64(kind, p, t, Float64(delta))))
        gy.append(_f32(factor * _dy64(kind, p, t, Float64(delta))))
    var tape_value = tape.values(ex, out)
    check_values(name + "/value", tape_value, [_f32(value * factor)])
    tape.backward(ex, out)
    check_values(name + "/prediction", tape.gradients(ex, x), gp)
    check_values(name + "/target", tape.gradients(ex, y), gy)
    # The trainer path: the same terms and fold over the same buffers.
    var ws = loss_scratch(ex, count)
    var word = ex.alloc(1)
    loss_value_into(ex, kind, tape.value(x), tape.value(y), rows, cols, reduction, word, ws, delta)
    var direct = _download(ex, word, 1)
    if not _same_bits(direct[0], tape_value[0]):
        raise Error(name + ": loss_value_into differs in bits from the tape loss")
    print("PASS", name + "/loss_value_into same bits as the tape")
    release_loss_scratch(ex, ws)
    ex.free(word)


def _mse_matches_tape[E: Exec](mut ex: E, mut tape: Tape, pred: List[Float32], target: List[Float32]) raises:
    """losses.mse equals Tape.mse_loss bit for bit (value and gradients)."""
    tape.reset()
    var x = tape.tensor(ex, pred, 2, len(pred) // 2, True)
    var y = tape.tensor(ex, target, 2, len(pred) // 2, True)
    var a = mse(tape, ex, x, y)
    tape.backward(ex, a)
    var va = tape.values(ex, a)
    var ga = tape.gradients(ex, x)
    var ha = tape.gradients(ex, y)
    tape.reset()
    var x2 = tape.tensor(ex, pred, 2, len(pred) // 2, True)
    var y2 = tape.tensor(ex, target, 2, len(pred) // 2, True)
    var b = tape.mse_loss(ex, x2, y2)
    tape.backward(ex, b)
    var vb = tape.values(ex, b)
    var gb = tape.gradients(ex, x2)
    var hb = tape.gradients(ex, y2)
    if not _same_bits(va[0], vb[0]):
        raise Error("mse: value differs in bits from Tape.mse_loss")
    for i in range(len(ga)):
        if not _same_bits(ga[i], gb[i]) or not _same_bits(ha[i], hb[i]):
            raise Error("mse: gradient differs in bits from Tape.mse_loss")
    print("PASS mse same bits as Tape.mse_loss")


def _cross_entropy_case[E: Exec](mut ex: E, mut tape: Tape, reduction: Int) raises:
    tape.reset()
    var rows = 3
    var cols = 4
    var logits: List[Float32] = [
        0.3, -0.8, 1.2, 0.7,
        2.5, 0.1, -1.0, 0.0,
        -0.4, -0.4, 0.9, 30.0,
    ]
    var codes: List[Float32] = [2.0, 0.0, 1.0]
    var z = tape.tensor(ex, logits, rows, cols, True)
    var c = tape.tensor(ex, codes, rows, 1, False)
    var out = cross_entropy(tape, ex, z, c, reduction)
    var factor = 1.0 / Float64(rows) if reduction == REDUCTION_MEAN else 1.0
    var value = 0.0
    var grad = List[Float32]()
    for r in range(rows):
        var m = Float64(logits[r * cols])
        for k in range(cols):
            m = max(m, Float64(logits[r * cols + k]))
        var s = 0.0
        for k in range(cols):
            s += exp(Float64(logits[r * cols + k]) - m)
        var y = Int(codes[r])
        value += log(s) + m - Float64(logits[r * cols + y])
        for k in range(cols):
            var p = exp(Float64(logits[r * cols + k]) - m) / s
            if k == y:
                p -= 1.0
            grad.append(_f32(factor * p))
    var name = String("cross_entropy/") + (String("mean") if reduction == REDUCTION_MEAN else String("sum"))
    var tape_value = tape.values(ex, out)
    check_values(name + "/value", tape_value, [_f32(value * factor)])
    tape.backward(ex, out)
    check_values(name + "/logits", tape.gradients(ex, z), grad)
    var ws = loss_scratch(ex, rows)
    var word = ex.alloc(1)
    loss_value_into(ex, LS_CROSS_ENTROPY, tape.value(z), tape.value(c), rows, cols, reduction, word, ws)
    var direct = _download(ex, word, 1)
    if not _same_bits(direct[0], tape_value[0]):
        raise Error(name + ": loss_value_into differs in bits from the tape loss")
    print("PASS", name + "/loss_value_into same bits as the tape")
    release_loss_scratch(ex, ws)
    ex.free(word)
    if reduction == REDUCTION_MEAN:
        # Report (not a gate): does the loss VJP give op_ce's seed bits?
        var seed = ex.alloc(rows * cols)
        var rowloss = ex.alloc(rows)
        var a = Args()
        a.p0 = tape.value(z)
        a.p1 = tape.value(c)
        a.p2 = seed
        a.p3 = rowloss
        a.i0 = rows
        a.i1 = cols
        a.f0 = Float32(1.0) / Float32(rows)
        ex.launch[OP_CE](a, rows)
        var hand = _download(ex, seed, rows * cols)
        var tape_grad = tape.gradients(ex, z)
        var same = True
        for i in range(rows * cols):
            same = same and _same_bits(hand[i], tape_grad[i])
        print("SEEDBITS cross_entropy op_ce", String("SAME") if same else String("DIFFER"))
        ex.free(seed)
        ex.free(rowloss)


def _mse_seed_report[E: Exec](mut ex: E, mut tape: Tape, pred: List[Float32], target: List[Float32]) raises:
    """Report (not a gate): does the mse VJP give op_mse's seed bits?"""
    tape.reset()
    var n = len(pred)
    var x = tape.tensor(ex, pred, 2, n // 2, True)
    var y = tape.tensor(ex, target, 2, n // 2, False)
    var out = mse(tape, ex, x, y)
    tape.backward(ex, out)
    var tape_grad = tape.gradients(ex, x)
    var seed = ex.alloc(n)
    var sq = ex.alloc(n)
    var a = Args()
    a.p0 = tape.value(x)
    a.p1 = tape.value(y)
    a.p2 = seed
    a.p3 = sq
    a.f0 = Float32(2.0) / Float32(n)
    ex.launch[OP_MSE](a, n)
    var hand = _download(ex, seed, n)
    var same = True
    for i in range(n):
        same = same and _same_bits(hand[i], tape_grad[i])
    print("SEEDBITS mse op_mse", String("SAME") if same else String("DIFFER"))
    ex.free(seed)
    ex.free(sq)


def _probability_terms[E: Exec](mut ex: E) raises:
    """The MLP trainer's clipped-probability kinds (forward only), with
    probabilities at and beyond the clip bounds."""
    var p: List[Float32] = [0.0, 0.25, 1.0, 0.6, 0.999999999, 0.1]
    var y: List[Float32] = [1.0, 0.0, 0.0, 1.0, 1.0, 0.5]
    var dp = _upload(ex, p)
    var dy = _upload(ex, y)
    var ws = loss_scratch(ex, len(p))
    var word = ex.alloc(1)
    for kind in [LS_LOG_PROB, LS_BINARY_LOG_PROB]:
        var total = 0.0
        for i in range(len(p)):
            total += _term64(kind, Float64(p[i]), Float64(y[i]), 1.0)
        loss_value_into(ex, kind, dp, dy, 2, 3, REDUCTION_SUM, word, ws)
        var name = String("log_prob/sum") if kind == LS_LOG_PROB else String("binary_log_prob/sum")
        check_values(name, _download(ex, word, 1), [_f32(total)])
    release_loss_scratch(ex, ws)
    ex.free(word)
    ex.free(dp)
    ex.free(dy)


# ------------------------------------------------------------------ metrics
def _metric(name: String, actual: Float32, expected: Float64) raises:
    check_values(name, [actual], [_f32(expected)])


def _auc64(y: List[Float32], s: List[Float32]) raises -> Float64:
    var pairs = 0.0
    var pos = 0
    var neg = 0
    for i in range(len(y)):
        if y[i] == 1:
            pos += 1
        else:
            neg += 1
    for i in range(len(y)):
        if y[i] != 1:
            continue
        for j in range(len(y)):
            if y[j] == 1:
                continue
            if s[i] > s[j]:
                pairs += 1.0
            elif s[i] == s[j]:
                pairs += 0.5
    return pairs / (Float64(pos) * Float64(neg))


def _auc_case[E: Exec](mut ex: E, name: String, y: List[Float32], s: List[Float32]) raises:
    var dy = _upload(ex, y)
    var ds = _upload(ex, s)
    _metric(String("roc_auc/") + name, roc_auc(ex, dy, ds, len(y)), _auc64(y, s))
    ex.free(dy)
    ex.free(ds)


def _metric_checks[E: Exec](mut ex: E) raises:
    # accuracy on codes
    var yt: List[Float32] = [0.0, 1.0, 2.0, 1.0, 0.0]
    var yp: List[Float32] = [0.0, 2.0, 2.0, 1.0, 1.0]
    var dyt = _upload(ex, yt)
    var dyp = _upload(ex, yp)
    _metric("accuracy", accuracy(ex, dyt, dyp, 5), 0.6)
    # argmax accuracy; rows 1 and 3 tie, the first maximum wins
    var scores: List[Float32] = [
        0.2, 0.5, 0.3,
        0.4, 0.4, 0.2,
        0.1, 0.1, 0.8,
        0.3, 0.3, 0.3,
        0.6, 0.2, 0.2,
    ]
    var dsc = _upload(ex, scores)
    var hits = 0.0
    for r in range(5):
        var best = 0
        for k in range(1, 3):
            if scores[r * 3 + k] > scores[r * 3 + best]:
                best = k
        if best == Int(yt[r]):
            hits += 1.0
    _metric("accuracy_from_scores", accuracy_from_scores(ex, dyt, dsc, 5, 3), hits / 5.0)
    # k-class log loss, with probabilities 0 and 1 hitting the clip
    var proba: List[Float32] = [
        0.7, 0.2, 0.1,
        0.0, 0.0, 1.0,
        0.1, 0.1, 0.8,
        0.25, 0.5, 0.25,
        1.0, 0.0, 0.0,
    ]
    var dpr = _upload(ex, proba)
    var ll = 0.0
    for r in range(5):
        ll += -log(_clip64(Float64(proba[r * 3 + Int(yt[r])])))
    _metric("log_loss/k3", log_loss(ex, dyt, dpr, 5, 3), ll / 5.0)
    # binary log loss from the positive column
    var yb: List[Float32] = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    var pb: List[Float32] = [0.9, 0.2, 0.0, 1.0, 0.55, 0.45]
    var dyb = _upload(ex, yb)
    var dpb = _upload(ex, pb)
    var bl = 0.0
    for i in range(6):
        var q = Float64(pb[i]) if yb[i] == 1 else 1.0 - Float64(pb[i])
        bl += -log(_clip64(q))
    _metric("log_loss/binary", log_loss(ex, dyb, dpb, 6, 1), bl / 6.0)
    # regression metrics
    var rt: List[Float32] = [1.5, -0.25, 3.0, 0.75, 2.0, -1.0, 0.5]
    var rp: List[Float32] = [1.25, 0.0, 2.5, 1.0, 2.5, -0.5, 0.5]
    var drt = _upload(ex, rt)
    var drp = _upload(ex, rp)
    var n = len(rt)
    var se = 0.0
    var ae = 0.0
    var mean = 0.0
    for i in range(n):
        var d = Float64(rt[i]) - Float64(rp[i])
        se += d * d
        ae += abs(d)
        mean += Float64(rt[i])
    mean /= Float64(n)
    var st = 0.0
    for i in range(n):
        st += (Float64(rt[i]) - mean) * (Float64(rt[i]) - mean)
    _metric("mean_squared_error", mean_squared_error(ex, drt, drp, n), se / Float64(n))
    _metric("root_mean_squared_error", root_mean_squared_error(ex, drt, drp, n), sqrt(se / Float64(n)))
    _metric("mean_absolute_error", mean_absolute_error(ex, drt, drp, n), ae / Float64(n))
    _metric("r2_score", r2_score(ex, drt, drp, n), 1.0 - se / st)
    var flat: List[Float32] = [2.0, 2.0, 2.0]
    var dflat = _upload(ex, flat)
    _metric("r2_score/constant_perfect", r2_score(ex, dflat, dflat, 3), 1.0)
    _metric("r2_score/constant_miss", r2_score(ex, dflat, drp, 3), 0.0)
    # AUC with ties, a length that is not a power of two, larger generated
    # cases over several scan blocks, all scores tied, and a refusal
    var ya: List[Float32] = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    var sa: List[Float32] = [0.1, 0.4, 0.35, 0.8, 0.4, 0.4, 0.9, 0.1, 0.65, 0.2, 0.9, 0.1, 0.5]
    _auc_case(ex, "ties13", ya, sa)
    var tied = List[Float32](length=13, fill=Float32(0.5))
    _auc_case(ex, "all_tied", ya, tied)
    for size in [300, 5000]:
        var state = UInt64(12345 + size)
        var yg = List[Float32]()
        var sg = List[Float32]()
        for _ in range(size):
            state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            var u = Int((state >> 33) % UInt64(1000))
            var label = 1 if (u + Int((state >> 20) % UInt64(400))) > 700 else 0
            yg.append(Float32(label))
            sg.append(Float32(u // 25) * Float32(0.025))  # 40 score levels: many ties
        _auc_case(ex, String("generated") + String(size), yg, sg)
    var one_class = List[Float32](length=4, fill=Float32(1))
    var d1 = _upload(ex, one_class)
    var refused = False
    try:
        _ = roc_auc(ex, d1, d1, 4)
    except:
        refused = True
    if not refused:
        raise Error("roc_auc accepted a single class")
    print("PASS roc_auc/refuses_one_class")
    for p in [dyt, dyp, dsc, dpr, dyb, dpb, drt, drp, dflat, d1]:
        ex.free(p)


def run_loss_checks[E: Exec](mut ex: E) raises:
    var tape = Tape()
    var pred: List[Float32] = [0.3, -0.8, 1.2, 0.7, 0.2, -0.4]
    var target: List[Float32] = [0.9, 1.3, 0.6, -0.1, 0.25, -0.35]
    var bce_target: List[Float32] = [0.0, 1.0, 0.25, 1.0, 0.0, 0.6]
    var cases = 0
    for reduction in [REDUCTION_MEAN, REDUCTION_SUM]:
        var r = String("mean") if reduction == REDUCTION_MEAN else String("sum")
        _elementwise_case(ex, tape, String("mse/") + r, LS_MSE, reduction, 1, pred, target)
        _elementwise_case(ex, tape, String("mae/") + r, LS_MAE, reduction, 1, pred, target)
        _elementwise_case(ex, tape, String("huber/") + r, LS_HUBER, reduction, 0.5, pred, target)
        _elementwise_case(ex, tape, String("bce_logits/") + r, LS_BCE_LOGITS, reduction, 1, pred, bce_target)
        _cross_entropy_case(ex, tape, reduction)
        cases += 5
    _mse_matches_tape(ex, tape, pred, target)
    _mse_seed_report(ex, tape, pred, target)
    _probability_terms(ex)
    _metric_checks(ex)
    tape.release(ex)
    print("PASS loss_checks cases=" + String(cases))
