# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""THE TYPED METRIC BUILDERS the search scorers run (lane mf-metric4, 2026-11-23), in a process with no Python, over
the metrics program entry of the host column (x_metrics/descriptor_host.mojo `metrics_program_host_v1`), against
the scikit-learn definitions computed here in binary64 from the same float32 inputs:

1. regression (bindings/metrics_reg_run.mojo `reg_metric_run`, the builders bindings/metrics_reg_door.mojo calls):
   y [3, -0.5, 2, 7, 4.5], p [2.5, 0, 2, 8, 4]: r2, MSE, MAE, RMSE, MSLE, RMSLE, MAPE, median AE, explained
   variance, max error, Poisson and Gamma deviance (y, p > 0 rows only: y [3, 2, 7, 4.5]), d2 absolute error; a NaN
   prediction raises "y_pred contains NaN or infinity".
2. labels (bindings/mcls_classify_run.mojo `label_metric_run`, the sums builder both label doors call): t [0, 1, 2,
   2, 1, 0, 2], p [0, 2, 2, 2, 1, 1, 0]: balanced accuracy, matthews, precision / recall / f1 / jaccard x micro,
   macro, weighted; a label present only in p (code 3 of k = 4) counts as the doors' union does.
3. log_loss (`log_loss_run`, `mc_log_loss_stages`): 4 rows x 3 classes, and the one-column binary form; a fold
   that lacks a class raises the door's words.
4. roc_auc (bindings/mcls_ranking_run.mojo `roc_auc_run`): binary (Mann-Whitney with ties as 1/2), one-vs-rest
   macro and weighted, one-vs-one macro and weighted (scikit-learn's pair rule) over 6 rows x 3 classes.

Every value must match to a relative 1e-6 (the programs accumulate in binary64 over float32 words; the oracle does
the same arithmetic in a different order).

FAIL-PROOF EDITS (each must make a case fail): in `reg_r2_stages` `r.fin(res_sums, _RE_DIV)` for `r.fin(res_sums, 0)`
(case 1, r2); in `mc_sums_stages` `[a, b, match_off, labels, 0]` for the pair key's last word (case 2);
`McEpi(2, labels, avg, 0, 2, 4.0)` in `label_metric_run` (case 2, f1: beta 2); in `log_loss_run`
`codes[i] = Int32(t[i])` without the rank (case 3, the lacking-class form); `mc_roc_ovr_stages(..., True,
not weighted)` swapped (case 4, ovr weighted); `pair_b[g0 + t]` read as `pair_a` in `roc_ovo_run` (case 4).
"""
from std.math import sqrt, log, isnan
from std.memory import bitcast
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from bindings.arena_prog_run import MetricRoute
from bindings.metrics_reg_run import (
    reg_metric_run, REG_R2, REG_MSE, REG_MAE, REG_RMSE, REG_MSLE, REG_RMSLE, REG_MAPE, REG_MEDAE,
    REG_EXPLAINED_VARIANCE, REG_MAX_ERROR, REG_POISSON, REG_GAMMA, REG_D2_ABSOLUTE,
)
from bindings.mcls_classify_run import (
    label_metric_run, log_loss_run, LBL_JACCARD, LBL_BALANCED, LBL_PRECISION, LBL_RECALL, LBL_F1, LBL_MCC,
    LBL_MICRO, LBL_MACRO, LBL_WEIGHTED,
)
from bindings.mcls_ranking_run import roc_auc_run, ROC_BINARY, ROC_OVR, ROC_OVR_WEIGHTED, ROC_OVO, ROC_OVO_WEIGHTED
from x_metrics.descriptor_host import metrics_program_host_v1

comptime CASES = 4


def _route() -> MetricRoute:
    return MetricRoute(metrics_program_host_v1, False, False)


def _near(got: Float64, want: Float64, what: String) raises:
    var tol = 1e-6 * max(abs(want), 1.0)
    require(abs(got - want) <= tol, what + ": " + String(got) + " expected " + String(want))


def _f(values: List[Float32], i: Int) -> Float64:
    return Float64(values[i])


# ---------------------------------------------------------------- 1. regression


def _reg(metric: Int, y: List[Float32], p: List[Float32]) raises -> Float64:
    return reg_metric_run(_route(), metric, Int(y.unsafe_ptr()), Int(p.unsafe_ptr()), len(y))


def _median(var v: List[Float64]) -> Float64:
    var n = len(v)
    for i in range(1, n):  # small-loop(oracle: insertion sort)
        var key = v[i]
        var j = i - 1
        while j >= 0 and v[j] > key:
            v[j + 1] = v[j]
            j -= 1
        v[j + 1] = key
    if n % 2 == 1:
        return v[n // 2]
    return (v[n // 2 - 1] + v[n // 2]) / 2.0


def check_regression() raises:
    var y: List[Float32] = [3.0, -0.5, 2.0, 7.0, 4.5]
    var p: List[Float32] = [2.5, 0.0, 2.0, 8.0, 4.0]
    var n = len(y)
    var se = 0.0
    var ae = 0.0
    var sle = 0.0
    var ape = 0.0
    var mean_y = 0.0
    var mx = 0.0
    var absd = List[Float64]()
    var diffs = List[Float64]()
    for i in range(n):  # small-loop(oracle)
        var d = _f(y, i) - _f(p, i)
        se += d * d
        ae += abs(d)
        var l = log(1.0 + _f(y, i)) - log(1.0 + _f(p, i))
        sle += l * l
        ape += abs(d) / max(abs(_f(y, i)), 2.220446049250313e-16)
        mean_y += _f(y, i)
        mx = max(mx, abs(d))
        absd.append(abs(d))
        diffs.append(d)
    mean_y /= Float64(n)
    var tot = 0.0
    var mean_d = 0.0
    for i in range(n):  # small-loop(oracle)
        tot += (_f(y, i) - mean_y) * (_f(y, i) - mean_y)
        mean_d += diffs[i]
    mean_d /= Float64(n)
    var var_d = 0.0
    for i in range(n):  # small-loop(oracle)
        var_d += (diffs[i] - mean_d) * (diffs[i] - mean_d)
    var med_y = _median([3.0, -0.5, 2.0, 7.0, 4.5])
    var d2_den = 0.0
    for i in range(n):  # small-loop(oracle)
        d2_den += abs(_f(y, i) - med_y)
    _near(_reg(REG_R2, y, p), 1.0 - se / tot, "r2")
    _near(_reg(REG_MSE, y, p), se / Float64(n), "mse")
    _near(_reg(REG_MAE, y, p), ae / Float64(n), "mae")
    _near(_reg(REG_RMSE, y, p), sqrt(se / Float64(n)), "rmse")
    _near(_reg(REG_MSLE, y, p), sle / Float64(n), "msle")
    _near(_reg(REG_RMSLE, y, p), sqrt(sle / Float64(n)), "rmsle")
    _near(_reg(REG_MAPE, y, p), ape / Float64(n), "mape")
    _near(_reg(REG_MEDAE, y, p), _median(absd^), "median absolute error")
    _near(_reg(REG_EXPLAINED_VARIANCE, y, p), 1.0 - var_d / tot, "explained variance")
    _near(_reg(REG_MAX_ERROR, y, p), mx, "max error")
    _near(_reg(REG_D2_ABSOLUTE, y, p), 1.0 - ae / d2_den, "d2 absolute error")
    var yp: List[Float32] = [3.0, 2.0, 7.0, 4.5]
    var pp: List[Float32] = [2.5, 2.0, 8.0, 4.0]
    var poisson = 0.0
    var gamma = 0.0
    for i in range(len(yp)):  # small-loop(oracle: the deviances)
        var a = _f(yp, i)
        var b = _f(pp, i)
        poisson += 2.0 * (a * log(a / b) - a + b)
        gamma += 2.0 * (log(b / a) + a / b - 1.0)
    _near(_reg(REG_POISSON, yp, pp), poisson / Float64(len(yp)), "poisson deviance")
    _near(_reg(REG_GAMMA, yp, pp), gamma / Float64(len(yp)), "gamma deviance")
    var bad: List[Float32] = [2.5, 0.0, 2.0, 8.0, 4.0]
    bad[1] = bitcast[DType.float32](UInt32(0x7FC00000))
    var raised = String("")
    try:
        _ = _reg(REG_MSE, y, bad)
    except error:
        raised = String(error)
    require(raised.find("y_pred contains NaN or infinity") >= 0, "a non-finite prediction: " + raised)


# ---------------------------------------------------------------- 2. labels


def _counts(t: List[Int32], p: List[Int32], k: Int, mut tp: List[Float64], mut tr: List[Float64],
            mut pr: List[Float64]):
    tp = List[Float64](length=k, fill=0.0)
    tr = List[Float64](length=k, fill=0.0)
    pr = List[Float64](length=k, fill=0.0)
    for i in range(len(t)):  # small-loop(oracle)
        tr[Int(t[i])] += 1.0
        pr[Int(p[i])] += 1.0
        if t[i] == p[i]:
            tp[Int(t[i])] += 1.0


def _div(a: Float64, b: Float64) -> Float64:
    return 0.0 if b == 0.0 else a / b


def _set_wise(which: Int, avg: Int, tp: List[Float64], tr: List[Float64], pr: List[Float64],
              present: List[Int]) -> Float64:
    """precision (0), recall (1), f1 (2), jaccard (3) over the present labels."""
    if avg == LBL_MICRO:
        var a = 0.0
        var t = 0.0
        var q = 0.0
        for i in range(len(present)):  # small-loop(oracle)
            var c = present[i]
            a += tp[c]
            t += tr[c]
            q += pr[c]
        if which == 0:
            return _div(a, q)
        if which == 1:
            return _div(a, t)
        if which == 2:
            return _div(2.0 * a, t + q)
        return _div(a, t + q - a)
    var total = 0.0
    var weight = 0.0
    for i in range(len(present)):  # small-loop(oracle)
        var c = present[i]
        var v: Float64
        if which == 0:
            v = _div(tp[c], pr[c])
        elif which == 1:
            v = _div(tp[c], tr[c])
        elif which == 2:
            v = _div(2.0 * tp[c], tr[c] + pr[c])
        else:
            v = _div(tp[c], tr[c] + pr[c] - tp[c])
        var w = tr[c] if avg == LBL_WEIGHTED else 1.0
        total += w * v
        weight += w
    return total / weight


def check_labels() raises:
    var t: List[Int32] = [0, 1, 2, 2, 1, 0, 2]
    var p: List[Int32] = [0, 2, 2, 2, 1, 1, 0]
    var k = 3
    var n = len(t)
    var tp = List[Float64]()
    var tr = List[Float64]()
    var pr = List[Float64]()
    _counts(t, p, k, tp, tr, pr)
    var present: List[Int] = [0, 1, 2]
    var route = _route()
    var ta = Int(t.unsafe_ptr())
    var pa = Int(p.unsafe_ptr())
    var bal = 0.0
    for c in range(k):  # small-loop(oracle: balanced accuracy, the mean recall)
        bal += tp[c] / tr[c]
    _near(label_metric_run(route, LBL_BALANCED, 0, ta, pa, n, k), bal / 3.0, "balanced accuracy")
    # matthews: sklearn's multiclass formula
    var s = Float64(n)
    var cc = 0.0
    var pk2 = 0.0
    var tk2 = 0.0
    var pt = 0.0
    for c in range(k):  # small-loop(oracle)
        cc += tp[c]
        pk2 += pr[c] * pr[c]
        tk2 += tr[c] * tr[c]
        pt += pr[c] * tr[c]
    var mcc = (cc * s - pt) / sqrt((s * s - pk2) * (s * s - tk2))
    _near(label_metric_run(route, LBL_MCC, 0, ta, pa, n, k), mcc, "matthews")
    var metrics: List[Int] = [LBL_PRECISION, LBL_RECALL, LBL_F1, LBL_JACCARD]
    var avgs: List[Int] = [LBL_MICRO, LBL_MACRO, LBL_WEIGHTED]
    for i in range(4):  # small-loop(check: the four set-wise metrics)
        for a in range(3):  # small-loop(check: the three averages)
            _near(label_metric_run(route, metrics[i], avgs[a], ta, pa, n, k),
                  _set_wise(i, avgs[a], tp, tr, pr, present),
                  "set-wise metric " + String(i) + " average " + String(avgs[a]))
    # code 3 of k = 4 present only in the predictions: the union holds it (recall 0 there, tr 0)
    var p4: List[Int32] = [0, 3, 2, 2, 1, 1, 0]
    var tp4 = List[Float64]()
    var tr4 = List[Float64]()
    var pr4 = List[Float64]()
    _counts(t, p4, 4, tp4, tr4, pr4)
    var present4: List[Int] = [0, 1, 2, 3]
    _near(label_metric_run(route, LBL_PRECISION, LBL_MACRO, ta, Int(p4.unsafe_ptr()), n, 4),
          _set_wise(0, LBL_MACRO, tp4, tr4, pr4, present4), "macro precision over the label union")
    _ = len(t)  # the runs read the codes through their addresses
    _ = len(p)
    _ = len(p4)


# ---------------------------------------------------------------- 3. log_loss


def check_log_loss() raises:
    var t: List[Int32] = [0, 1, 2, 1]
    var proba: List[Float32] = [0.7, 0.2, 0.1, 0.1, 0.8, 0.1, 0.25, 0.25, 0.5, 0.3, 0.6, 0.1]
    var want = 0.0
    for i in range(4):  # small-loop(oracle)
        want -= log(_f(proba, i * 3 + Int(t[i])))
    _near(log_loss_run(_route(), Int(t.unsafe_ptr()), Int(proba.unsafe_ptr()), 4, 3), want / 4.0, "log loss k=3")
    var tb: List[Int32] = [0, 1, 1, 0, 1]
    var col: List[Float32] = [0.2, 0.9, 0.6, 0.4, 0.7]
    var wb = 0.0
    for i in range(5):  # small-loop(oracle: the positive column)
        wb -= log(_f(col, i)) if tb[i] == 1 else log(1.0 - _f(col, i))
    _near(log_loss_run(_route(), Int(tb.unsafe_ptr()), Int(col.unsafe_ptr()), 5, 2), wb / 5.0, "log loss binary")
    var lacking: List[Int32] = [0, 1, 1, 0]
    var raised = String("")
    try:
        _ = log_loss_run(_route(), Int(lacking.unsafe_ptr()), Int(proba.unsafe_ptr()), 4, 3)
    except error:
        raised = String(error)
    require(raised == "probability columns must match the label count (at least two)", "a lacking class: " + raised)
    _ = len(t)  # the runs read the words through their addresses
    _ = len(proba)
    _ = len(tb)
    _ = len(col)
    _ = len(lacking)


# ---------------------------------------------------------------- 4. roc_auc


def _auc(flags: List[Bool], scores: List[Float64]) -> Float64:
    """Mann-Whitney: the share of (positive, negative) pairs ordered right, ties as 1/2."""
    var good = 0.0
    var pairs = 0.0
    for i in range(len(flags)):  # small-loop(oracle)
        if not flags[i]:
            continue
        for j in range(len(flags)):  # small-loop(oracle)
            if flags[j]:
                continue
            pairs += 1.0
            if scores[i] > scores[j]:
                good += 1.0
            elif scores[i] == scores[j]:
                good += 0.5
    return good / pairs


def check_roc() raises:
    var tb: List[Int32] = [0, 0, 1, 1, 0, 1]
    var sb: List[Float32] = [0.1, 0.4, 0.35, 0.8, 0.2, 0.9]
    var fb = List[Bool]()
    var vb = List[Float64]()
    for i in range(6):  # small-loop(oracle)
        fb.append(tb[i] == 1)
        vb.append(_f(sb, i))
    _near(roc_auc_run(_route(), ROC_BINARY, Int(tb.unsafe_ptr()), Int(sb.unsafe_ptr()), 6, 2), _auc(fb, vb),
          "binary roc_auc")
    var t: List[Int32] = [0, 1, 2, 2, 1, 0]
    var s: List[Float32] = [0.6, 0.3, 0.1, 0.2, 0.5, 0.3, 0.1, 0.2, 0.7, 0.3, 0.3, 0.4, 0.2, 0.6, 0.2, 0.5, 0.25, 0.25]
    var k = 3
    var n = 6
    var macro = 0.0
    var weighted = 0.0
    for c in range(k):  # small-loop(oracle: one vs rest)
        var f = List[Bool]()
        var v = List[Float64]()
        var support = 0.0
        for i in range(n):  # small-loop(oracle)
            f.append(Int(t[i]) == c)
            v.append(_f(s, i * k + c))
            if Int(t[i]) == c:
                support += 1.0
        var a = _auc(f, v)
        macro += a / Float64(k)
        weighted += a * support / Float64(n)
    var ta = Int(t.unsafe_ptr())
    var sa = Int(s.unsafe_ptr())
    _near(roc_auc_run(_route(), ROC_OVR, ta, sa, n, k), macro, "roc_auc ovr macro")
    _near(roc_auc_run(_route(), ROC_OVR_WEIGHTED, ta, sa, n, k), weighted, "roc_auc ovr weighted")
    var ovo = 0.0
    var ovo_w = 0.0
    var w_total = 0.0
    for a in range(k):  # small-loop(oracle: one vs one, scikit-learn's pair rule)
        for b in range(a + 1, k):  # small-loop(oracle)
            var fa = List[Bool]()
            var va = List[Float64]()
            var fbb = List[Bool]()
            var vbb = List[Float64]()
            var count = 0.0
            for i in range(n):  # small-loop(oracle)
                var c = Int(t[i])
                if c != a and c != b:
                    continue
                count += 1.0
                fa.append(c == a)
                va.append(_f(s, i * k + a))
                fbb.append(c == b)
                vbb.append(_f(s, i * k + b))
            var pair = (_auc(fa, va) + _auc(fbb, vbb)) / 2.0
            ovo += pair / 3.0
            ovo_w += pair * count
            w_total += count
    _near(roc_auc_run(_route(), ROC_OVO, ta, sa, n, k), ovo, "roc_auc ovo macro")
    _near(roc_auc_run(_route(), ROC_OVO_WEIGHTED, ta, sa, n, k), ovo_w / w_total, "roc_auc ovo weighted")
    _ = len(t)  # the runs read the words through their addresses
    _ = len(s)
    _ = len(tb)
    _ = len(sb)


def test_metrics_scorer_builders() raises:
    def run_case(which: Int) raises:
        if which == 0:
            check_regression()
        elif which == 1:
            check_labels()
        elif which == 2:
            check_log_loss()
        else:
            check_roc()
    check_cases("metrics_scorer_builders", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
