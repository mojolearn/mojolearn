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

5. average_precision (bindings/metrics_exp_curve_run.mojo `average_precision_run`, lane mf-score5): binary over
   t == positive (scikit-learn's step sum over the distinct thresholds, descending), the one-vs-rest macro mean over
   4 rows x 3 classes, and the door's refusal of a column count that is not the classes y_true holds.
6. brier_score_loss (bindings/metrics_exp_rank_run.mojo `brier_run`): the one-column binary form (labels {0, 1}:
   positive code 1; labels {1, 2}: the largest label y_true holds) and the 3-class form (no half);
   top_k_accuracy_score (`top_k_run`, k = 2): 5 rows x 4 classes, and a binary column (k >= n_classes: 1.0 and
   the warning bit).
7. normalized_mutual_info_score and adjusted_mutual_info_score (bindings/metrics_exp_common_run.mojo
   `cluster_mi_run`, arithmetic mean): NMI against the entropy definition; AMI of a relabelled identical partition
   is 1; one cluster is 0 (AMI) and both one cluster is 1.
8. THE WARNINGS AS NOTICES (core/msel_scorer_run.mojo `msel_scorer_value`): r2 of one sample is NaN with notice
   MEMBER_NOTICE_METRIC_FEW; macro precision with a never-predicted label gives MEMBER_NOTICE_METRIC_PRF (a = 0);
   balanced accuracy with a predicted class y_true lacks gives MEMBER_NOTICE_METRIC_BALANCED; a binary top_k gives
   MEMBER_NOTICE_METRIC_TOP_K (a = 2, b = 2); a clean macro f1 gives none.

9. THE CLUSTER LABEL-PAIR SCORERS (lane mf-cluster5; core/msel_scorer_run.mojo over core/metrics_cluster_table.mojo
   `cluster_pair_run` and the CPU column's entry, metrics/host/cluster_entry_host.mojo `metrics_cluster_host_v1`):
   rand (pair agreements over C(n, 2)), adjusted rand (scikit-learn's pair-count formula), mutual information
   (nats), homogeneity MI / H(true), completeness MI / H(pred), v-measure (beta 1), Fowlkes-Mallows
   TK / sqrt(PK QK), each against its definition over the case-7 codes; the same labels shifted (+5, the raw
   labels the doors see) give the same values.
10. A CLUSTERER'S SIDE (`classes` 0): the true labels as float32 words of integers, the member's cluster ids as
   int32: the same values as case 9 for every clustering scorer (NMI, AMI and the label-pair seven); a float32
   label 0.5 refuses.
11. A MAKE_SCORER FLIP and AN ABSENT pos_label (lane mf-cluster5): the options' `flip` of 'neg_mean_squared_error'
   scores +MSE; 'f1' with SCORER_POSITIVE_ABSENT over two present labels raises the door's "pos_label=1 is not a
   valid label. It should be one of [0, 2]" (label words ['0', '2']), over one present label scores 0.0 with the
   PRF notice, and SCORER_POSITIVE_KIND raises "pos_label must use the same type as the targets"; three present
   labels raise the door's "Target is multiclass but average='binary'..." words.

Every value must match to a relative 1e-6 (the programs accumulate in binary64 over float32 words; the oracle does
the same arithmetic in a different order).

FAIL-PROOF EDITS (each must make a case fail): in `reg_r2_stages` `r.fin(res_sums, _RE_DIV)` for `r.fin(res_sums, 0)`
(case 1, r2); in `mc_sums_stages` `[a, b, match_off, labels, 0]` for the pair key's last word (case 2);
`McEpi(2, labels, avg, 0, 2, 4.0)` in `label_metric_run` (case 2, f1: beta 2); in `log_loss_run`
`codes[i] = Int32(t[i])` without the rank (case 3, the lacking-class form); `mc_roc_ovr_stages(..., True,
not weighted)` swapped (case 4, ovr weighted); `pair_b[g0 + t]` read as `pair_a` in `roc_ovo_run` (case 4); `flags[i] = ... == positive` as `!=` in
`average_precision_run` (case 5); `brier_marks_positive` answering `last` always (case 6, labels {0, 1} with only
code 0 present: no row is positive); `kk = k` replaced by `kk = 1` in `top_k_stages`
(case 6); `cluster_mi_run` swapping `ka` / `kb` in the early return (case 7, the one-cluster AMI); dropping
`member_notice_add` for MEMBER_NOTICE_METRIC_PRF in `_prf_notices` (case 8); in `cluster_union_codes` copy `truth`
into both halves (case 9: MI, homogeneity, completeness, v-measure, FM move); in `metrics_cluster_host_run` pass
`(yt, yp)` to the completeness branch (case 9); in `_cluster_value` read the clusterer's float32 truth as int32 bits
(case 10); drop `row.sign = -row.sign` in `msel_scorer_row` (case 11); in `_binary_value` drop the `present >= 2`
refusal (case 11: the absent positive scores instead of raising).
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
from bindings.metrics_exp_curve_run import average_precision_run, top_k_run
from bindings.metrics_exp_rank_run import brier_run, BRIER_MARK_ZERO, BRIER_MARK_ONE
from bindings.metrics_exp_common_run import cluster_mi_run
from core.msel_scorer_run import msel_scorer_row, msel_scorer_code, msel_scorer_value
from core.member_notices import (
    MEMBER_NOTICE_WORDS, MEMBER_NOTICE_METRIC_PRF, MEMBER_NOTICE_METRIC_BALANCED, MEMBER_NOTICE_METRIC_FEW,
    MEMBER_NOTICE_METRIC_TOP_K,
)
from x_metrics.descriptor_host import metrics_program_host_v1
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.entry_table import EntryRef
from core.metrics_cluster_table import cluster_ops_publish, cluster_ops_entry
from core.msel_scorer_run import MselScorerOptions, SCORER_POSITIVE_ABSENT, SCORER_POSITIVE_KIND
from metrics.host.cluster_entry_host import METRICS_CLUSTER_HOST_API, metrics_cluster_host_v1

comptime CASES = 11


def _route() -> MetricRoute:
    return MetricRoute(metrics_program_host_v1, False, False)


def _near(got: Float64, want: Float64, what: String) raises:
    var tol = 1e-6 * max(abs(want), 1.0)
    require(abs(got - want) <= tol, what + ": " + String(got) + " expected " + String(want))


def _f(values: List[Float32], i: Int) -> Float64:
    return Float64(values[i])


# ---------------------------------------------------------------- 1. regression


def _reg(metric: Int, y: List[Float32], p: List[Float32]) raises -> Float64:
    var warn = 0
    return reg_metric_run(_route(), metric, Int(y.unsafe_ptr()), Int(p.unsafe_ptr()), len(y), warn)


def _lbl(route: MetricRoute, metric: Int, avg: Int, t: Int, p: Int, n: Int, k: Int) raises -> Float64:
    var warn = 0
    return label_metric_run(route, metric, avg, t, p, n, k, warn)


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
    _near(_lbl(route, LBL_BALANCED, 0, ta, pa, n, k), bal / 3.0, "balanced accuracy")
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
    _near(_lbl(route, LBL_MCC, 0, ta, pa, n, k), mcc, "matthews")
    var metrics: List[Int] = [LBL_PRECISION, LBL_RECALL, LBL_F1, LBL_JACCARD]
    var avgs: List[Int] = [LBL_MICRO, LBL_MACRO, LBL_WEIGHTED]
    for i in range(4):  # small-loop(check: the four set-wise metrics)
        for a in range(3):  # small-loop(check: the three averages)
            _near(_lbl(route, metrics[i], avgs[a], ta, pa, n, k),
                  _set_wise(i, avgs[a], tp, tr, pr, present),
                  "set-wise metric " + String(i) + " average " + String(avgs[a]))
    # code 3 of k = 4 present only in the predictions: the union holds it (recall 0 there, tr 0)
    var p4: List[Int32] = [0, 3, 2, 2, 1, 1, 0]
    var tp4 = List[Float64]()
    var tr4 = List[Float64]()
    var pr4 = List[Float64]()
    _counts(t, p4, 4, tp4, tr4, pr4)
    var present4: List[Int] = [0, 1, 2, 3]
    _near(_lbl(route, LBL_PRECISION, LBL_MACRO, ta, Int(p4.unsafe_ptr()), n, 4),
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


# ---------------------------------------------------------------- 5. average_precision


def _ap(flags: List[Bool], scores: List[Float64]) -> Float64:
    """scikit-learn: sum over the distinct thresholds (descending) of (R_n - R_{n-1}) P_n."""
    var positives = 0.0
    for i in range(len(flags)):  # small-loop(oracle)
        if flags[i]:
            positives += 1.0
    var done = List[Bool](length=len(scores), fill=False)
    var ap = 0.0
    var recall_prev = 0.0
    while True:  # small-loop(oracle: over the distinct thresholds, descending)
        var top = -1.0e300
        var found = False
        for i in range(len(scores)):  # small-loop(oracle: the next threshold)
            if not done[i] and scores[i] > top:
                top = scores[i]
                found = True
        if not found:
            break
        var tp = 0.0
        var fp = 0.0
        for i in range(len(scores)):  # small-loop(oracle: the rows at or above it)
            if scores[i] >= top:
                done[i] = True
                if flags[i]:
                    tp += 1.0
                else:
                    fp += 1.0
        var recall = tp / positives
        ap += (recall - recall_prev) * (tp / (tp + fp))
        recall_prev = recall
    return ap


def check_average_precision() raises:
    var route = _route()
    var tb: List[Int32] = [0, 1, 1, 0, 1, 0]
    var sb: List[Float32] = [0.2, 0.7, 0.4, 0.5, 0.9, 0.1]
    var fb = List[Bool]()
    var vb = List[Float64]()
    for i in range(6):  # small-loop(oracle)
        fb.append(tb[i] == 1)
        vb.append(_f(sb, i))
    _near(average_precision_run(route, Int(tb.unsafe_ptr()), Int(sb.unsafe_ptr()), 6, 1, 2, 1), _ap(fb, vb),
          "binary average precision")
    var t: List[Int32] = [0, 1, 2, 1]
    var s: List[Float32] = [0.6, 0.3, 0.1, 0.2, 0.5, 0.3, 0.1, 0.2, 0.7, 0.3, 0.4, 0.3]
    var macro = 0.0
    for c in range(3):  # small-loop(oracle: one vs rest)
        var f = List[Bool]()
        var v = List[Float64]()
        for i in range(4):  # small-loop(oracle)
            f.append(Int(t[i]) == c)
            v.append(_f(s, i * 3 + c))
        macro += _ap(f, v) / 3.0
    _near(average_precision_run(route, Int(t.unsafe_ptr()), Int(s.unsafe_ptr()), 4, 3, 3, -1), macro,
          "one-vs-rest macro average precision")
    var four: List[Float32] = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                               0.25, 0.25]
    var raised = String("")
    try:
        _ = average_precision_run(route, Int(t.unsafe_ptr()), Int(four.unsafe_ptr()), 4, 4, 4, -1)
    except error:
        raised = String(error)
    require(raised == "Number of classes in y_true not equal to the number of columns in 'y_score'",
            "a column count that is not the classes y_true holds: " + raised)
    _ = len(tb)  # the runs read the words through their addresses
    _ = len(sb)
    _ = len(t)
    _ = len(s)
    _ = len(four)


# ---------------------------------------------------------------- 6. brier and top_k


def check_brier_top_k() raises:
    var route = _route()
    var tb: List[Int32] = [0, 1, 1, 0, 1]
    var col: List[Float32] = [0.2, 0.9, 0.6, 0.4, 0.7]
    var want = 0.0
    for i in range(5):  # small-loop(oracle: (y - p)^2, y = t == 1)
        var y = 1.0 if tb[i] == 1 else 0.0
        want += (y - _f(col, i)) * (y - _f(col, i))
    var zero_one = BRIER_MARK_ZERO + (BRIER_MARK_ONE << 2)
    _near(brier_run(route, Int(tb.unsafe_ptr()), Int(col.unsafe_ptr()), 5, 2, 1, zero_one), want / 5.0,
          "binary brier, labels {0, 1}")
    # labels {1, 2} (marks: 1 at code 0) with both present: not {0,1} / {-1,1}: the largest label (code 1)
    _near(brier_run(route, Int(tb.unsafe_ptr()), Int(col.unsafe_ptr()), 5, 2, 1, BRIER_MARK_ONE), want / 5.0,
          "binary brier, labels {1, 2}")
    # labels {0, 1} with only code 0 held: {0} is pos_label-consistent, the label 1 is code 1: no row is positive
    var t0: List[Int32] = [0, 0, 0, 0, 0]
    var want0 = 0.0
    for i in range(5):  # small-loop(oracle: p^2)
        want0 += _f(col, i) * _f(col, i)
    _near(brier_run(route, Int(t0.unsafe_ptr()), Int(col.unsafe_ptr()), 5, 2, 1, zero_one), want0 / 5.0,
          "binary brier, labels {0, 1}, only 0 held")
    var t3: List[Int32] = [0, 1, 2, 1]
    var p3: List[Float32] = [0.7, 0.2, 0.1, 0.1, 0.8, 0.1, 0.25, 0.25, 0.5, 0.3, 0.6, 0.1]
    var want3 = 0.0
    for i in range(4):  # small-loop(oracle: the sum over the classes, no half)
        for c in range(3):  # small-loop(oracle)
            var y = 1.0 if Int(t3[i]) == c else 0.0
            want3 += (y - _f(p3, i * 3 + c)) * (y - _f(p3, i * 3 + c))
    _near(brier_run(route, Int(t3.unsafe_ptr()), Int(p3.unsafe_ptr()), 4, 3, 3, 0), want3 / 4.0, "3-class brier")
    var t4: List[Int32] = [0, 1, 2, 3, 1]
    var s4: List[Float32] = [0.5, 0.3, 0.1, 0.1, 0.1, 0.2, 0.6, 0.1, 0.4, 0.1, 0.3, 0.2, 0.1, 0.2, 0.3, 0.4,
                             0.6, 0.1, 0.2, 0.15]
    var hits = 0.0
    for i in range(5):  # small-loop(oracle: the true class among the two largest scores)
        var mine = _f(s4, i * 4 + Int(t4[i]))
        var above = 0
        for c in range(4):  # small-loop(oracle)
            if _f(s4, i * 4 + c) > mine:
                above += 1
        if above < 2:
            hits += 1.0
    var warn = 0
    _near(top_k_run(route, Int(t4.unsafe_ptr()), Int(s4.unsafe_ptr()), 5, 4, 4, 2, warn), hits / 5.0, "top_k k=2")
    require(warn == 0, "top_k over 4 classes does not warn")
    _near(top_k_run(route, Int(tb.unsafe_ptr()), Int(col.unsafe_ptr()), 5, 1, 2, 2, warn), 1.0, "binary top_k")
    require(warn == 1, "a binary top_k with k = 2 warns that k >= n_classes")
    _ = len(tb)  # the runs read the words through their addresses
    _ = len(col)
    _ = len(t0)
    _ = len(t3)
    _ = len(p3)
    _ = len(t4)
    _ = len(s4)


# ---------------------------------------------------------------- 7. mutual information


def check_mutual_info() raises:
    var route = _route()
    var a: List[Int32] = [0, 0, 1, 1, 2, 2, 2]
    var b: List[Int32] = [0, 1, 1, 1, 2, 2, 0]
    var n = 7
    var mi = 0.0
    var ha = 0.0
    var hb = 0.0
    for x in range(3):  # small-loop(oracle: the contingency definitions)
        var px = 0.0
        var py = 0.0
        for i in range(n):  # small-loop(oracle)
            if Int(a[i]) == x:
                px += 1.0
            if Int(b[i]) == x:
                py += 1.0
        ha -= (px / 7.0) * log(px / 7.0)
        hb -= (py / 7.0) * log(py / 7.0)
        for y in range(3):  # small-loop(oracle)
            var pxy = 0.0
            var qy = 0.0
            for i in range(n):  # small-loop(oracle)
                if Int(a[i]) == x and Int(b[i]) == y:
                    pxy += 1.0
                if Int(b[i]) == y:
                    qy += 1.0
            if pxy > 0.0:
                mi += (pxy / 7.0) * log((pxy * 7.0) / (px * qy))
    _near(cluster_mi_run(route, False, Int(a.unsafe_ptr()), Int(b.unsafe_ptr()), n, 3, 2), mi / ((ha + hb) / 2.0),
          "normalized mutual information")
    var same: List[Int32] = [1, 1, 2, 2, 0, 0, 0]
    _near(cluster_mi_run(route, True, Int(a.unsafe_ptr()), Int(same.unsafe_ptr()), n, 3, 2), 1.0,
          "adjusted mutual information of a relabelled partition")
    var one: List[Int32] = [2, 2, 2, 2, 2, 2, 2]
    require(cluster_mi_run(route, True, Int(a.unsafe_ptr()), Int(one.unsafe_ptr()), n, 3, 2) == 0.0,
            "adjusted mutual information against one cluster")
    require(cluster_mi_run(route, False, Int(one.unsafe_ptr()), Int(one.unsafe_ptr()), n, 3, 2) == 1.0,
            "normalized mutual information of one cluster each")
    _ = len(a)  # the runs read the codes through their addresses
    _ = len(b)
    _ = len(same)
    _ = len(one)


# ---------------------------------------------------------------- 8. the warnings as notices


def _notice_codes(words: List[Float64]) -> List[Int]:
    var codes = List[Int]()
    for i in range(len(words) // MEMBER_NOTICE_WORDS):  # small-loop(check: the notices)
        codes.append(Int(words[MEMBER_NOTICE_WORDS * i]))
    return codes^


def check_notices() raises:
    var route = _route()
    var y1: List[Float32] = [2.0]
    var p1: List[Float32] = [1.5]
    var notes = List[Float64]()
    var r2 = msel_scorer_row(msel_scorer_code("r2", False))
    var value = msel_scorer_value(route, r2, Int(y1.unsafe_ptr()), Int(p1.unsafe_ptr()), 1, 0, 1, -1, notes)
    require(isnan(value), "r2 of one sample is NaN")
    var codes = _notice_codes(notes)
    require(len(codes) == 1 and codes[0] == MEMBER_NOTICE_METRIC_FEW and notes[1] == 0.0, "r2 of one sample warns")
    var t: List[Int32] = [0, 1, 1]
    var pz: List[Int32] = [0, 0, 0]
    notes = List[Float64]()
    var precision = msel_scorer_row(msel_scorer_code("precision_macro", False))
    _ = msel_scorer_value(route, precision, Int(t.unsafe_ptr()), Int(pz.unsafe_ptr()), 3, 2, 1, -1, notes)
    codes = _notice_codes(notes)
    require(len(codes) == 1 and codes[0] == MEMBER_NOTICE_METRIC_PRF and notes[1] == 0.0,
            "macro precision with a never-predicted label warns")
    var tb: List[Int32] = [0, 0, 1]
    var pb: List[Int32] = [0, 1, 2]
    notes = List[Float64]()
    var balanced = msel_scorer_row(msel_scorer_code("balanced_accuracy", False))
    _ = msel_scorer_value(route, balanced, Int(tb.unsafe_ptr()), Int(pb.unsafe_ptr()), 3, 3, 1, -1, notes)
    codes = _notice_codes(notes)
    require(len(codes) == 1 and codes[0] == MEMBER_NOTICE_METRIC_BALANCED, "balanced accuracy warns")
    var tk: List[Int32] = [0, 1, 1, 0]
    var sk: List[Float32] = [0.1, 0.8, 0.6, 0.3]
    notes = List[Float64]()
    var top_k = msel_scorer_row(msel_scorer_code("top_k_accuracy", False))
    _ = msel_scorer_value(route, top_k, Int(tk.unsafe_ptr()), Int(sk.unsafe_ptr()), 4, 2, 1, -1, notes)
    codes = _notice_codes(notes)
    require(len(codes) == 1 and codes[0] == MEMBER_NOTICE_METRIC_TOP_K and notes[1] == 2.0 and notes[2] == 2.0,
            "a binary top_k warns")
    var tc: List[Int32] = [0, 1, 2, 1]
    var pc: List[Int32] = [0, 1, 2, 2]
    notes = List[Float64]()
    var f1 = msel_scorer_row(msel_scorer_code("f1_macro", False))
    _ = msel_scorer_value(route, f1, Int(tc.unsafe_ptr()), Int(pc.unsafe_ptr()), 4, 3, 1, -1, notes)
    require(len(notes) == 0, "a clean macro f1 raises nothing")
    _ = len(y1)  # the runs read the words through their addresses
    _ = len(p1)
    _ = len(t)
    _ = len(pz)
    _ = len(tb)
    _ = len(pb)
    _ = len(tk)
    _ = len(sk)
    _ = len(tc)
    _ = len(pc)


# ---------------------------------------------------------------- 9-11. the cluster scorers (lane mf-cluster5)


def _cluster_entry() raises -> EntryRef:
    """The CPU column's cluster entry, published as its binding's module init publishes it."""
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var api = METRICS_CLUSTER_HOST_API.get_or_create_ptr()
    cluster_ops_publish(mode, 0, Int(api[].entries.unsafe_ptr()), metrics_cluster_host_v1)
    return cluster_ops_entry(mode, 0)


def _c2(x: Float64) -> Float64:
    return x * (x - 1.0) / 2.0


def _cluster_oracle(a: List[Int32], b: List[Int32], k: Int) -> List[Float64]:
    """[rand, adjusted rand, MI, homogeneity, completeness, v-measure, FM] by their definitions (codes 0..k-1)."""
    var n = len(a)
    var agree = 0.0
    for i in range(n):  # small-loop(oracle: the sample pairs)
        for j in range(i + 1, n):  # small-loop(oracle)
            if (a[i] == a[j]) == (b[i] == b[j]):
                agree += 1.0
    var pairs = _c2(Float64(n))
    var cont = List[Float64](length=k * k, fill=0.0)
    var ra = List[Float64](length=k, fill=0.0)
    var rb = List[Float64](length=k, fill=0.0)
    for i in range(n):  # small-loop(oracle: the contingency)
        cont[Int(a[i]) * k + Int(b[i])] += 1.0
        ra[Int(a[i])] += 1.0
        rb[Int(b[i])] += 1.0
    var sum_comb = 0.0
    var sq = 0.0
    for c in range(k * k):  # small-loop(oracle)
        sum_comb += _c2(cont[c])
        sq += cont[c] * cont[c]
    var sa = 0.0
    var sb = 0.0
    var pk = 0.0
    var qk = 0.0
    var ha = 0.0
    var hb = 0.0
    var count_f = Float64(n)
    for c in range(k):  # small-loop(oracle)
        sa += _c2(ra[c])
        sb += _c2(rb[c])
        qk += ra[c] * ra[c]
        pk += rb[c] * rb[c]
        if ra[c] > 0.0:
            ha -= (ra[c] / count_f) * log(ra[c] / count_f)
        if rb[c] > 0.0:
            hb -= (rb[c] / count_f) * log(rb[c] / count_f)
    var mi = 0.0
    for x in range(k):  # small-loop(oracle)
        for y in range(k):  # small-loop(oracle)
            var c = cont[x * k + y]
            if c > 0.0:
                mi += (c / count_f) * log((c * count_f) / (ra[x] * rb[y]))
    var expected = sa * sb / pairs
    var ari = (sum_comb - expected) / ((sa + sb) / 2.0 - expected)
    var h = mi / ha if ha > 0.0 else 1.0
    var cc = mi / hb if hb > 0.0 else 1.0
    var v = 2.0 * h * cc / (h + cc) if h + cc > 0.0 else 0.0
    var fm = (sq - count_f) / sqrt((pk - count_f) * (qk - count_f))
    var out: List[Float64] = [agree / pairs, ari, mi, h, cc, v, fm]
    return out^


def _pair_names() -> List[String]:
    var names: List[String] = ["rand_score", "adjusted_rand_score", "mutual_info_score", "homogeneity_score",
                               "completeness_score", "v_measure_score", "fowlkes_mallows_score"]
    return names^


def check_cluster_pairs() raises:
    var route = _route()
    var cluster = _cluster_entry()
    var a: List[Int32] = [0, 0, 1, 1, 2, 2, 2]
    var b: List[Int32] = [0, 1, 1, 1, 2, 2, 0]
    var a5: List[Int32] = [5, 5, 6, 6, 7, 7, 7]
    var b5: List[Int32] = [5, 6, 6, 6, 7, 7, 5]
    var want = _cluster_oracle(a, b, 3)
    var names = _pair_names()
    for m in range(len(names)):  # small-loop(check: the seven scorers)
        var row = msel_scorer_row(msel_scorer_code(names[m], False))
        var notes = List[Float64]()
        var got = msel_scorer_value(route, cluster, List[String](), row, Int(a.unsafe_ptr()), Int(b.unsafe_ptr()),
                                    7, 3, 1, -1, notes)
        _near(got, want[m], names[m])
        var raw = msel_scorer_value(route, cluster, List[String](), row, Int(a5.unsafe_ptr()),
                                    Int(b5.unsafe_ptr()), 7, 8, 1, -1, notes)
        require(raw == got, names[m] + ": raw labels give the codes' value")
        require(len(notes) == 0, names[m] + " raises no warning")
    _ = len(a)  # the runs read the codes through their addresses
    _ = len(b)
    _ = len(a5)
    _ = len(b5)


def check_clusterer_side() raises:
    var route = _route()
    var cluster = _cluster_entry()
    var a: List[Int32] = [0, 0, 1, 1, 2, 2, 2]
    var b: List[Int32] = [0, 1, 1, 1, 2, 2, 0]
    var af: List[Float32] = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    var names = _pair_names()
    names.append("normalized_mutual_info_score")
    names.append("adjusted_mutual_info_score")
    for m in range(len(names)):  # small-loop(check: the clustering scorers)
        var row = msel_scorer_row(msel_scorer_code(names[m], False))
        var notes = List[Float64]()
        var codes = msel_scorer_value(route, cluster, List[String](), row, Int(a.unsafe_ptr()), Int(b.unsafe_ptr()),
                                      7, 3, 1, -1, notes)
        var ids = msel_scorer_value(route, cluster, List[String](), row, Int(af.unsafe_ptr()), Int(b.unsafe_ptr()),
                                    7, 0, 1, -1, notes)
        require(ids == codes, names[m] + ": a clusterer's float32 labels give the codes' value")
    var bad: List[Float32] = [0.5, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    var refused = False
    try:
        var notes = List[Float64]()
        _ = msel_scorer_value(route, cluster, List[String](), msel_scorer_row(msel_scorer_code("rand_score", False)),
                              Int(bad.unsafe_ptr()), Int(b.unsafe_ptr()), 7, 0, 1, -1, notes)
    except e:
        refused = String(e).find("integer labels") >= 0
    require(refused, "a non-integer clusterer label refuses")
    _ = len(a)
    _ = len(b)
    _ = len(af)
    _ = len(bad)


def _raises_with(route: MetricRoute, positive: Int, t: List[Int32], p: List[Int32], words: String) raises -> Bool:
    var f1 = msel_scorer_row(msel_scorer_code("f1", False))
    var labels: List[String] = ["0", "2", "3"]
    try:
        var notes = List[Float64]()
        _ = msel_scorer_value(route, EntryRef(metrics_cluster_host_v1, 0), labels, f1, Int(t.unsafe_ptr()),
                              Int(p.unsafe_ptr()), len(t), 3, 1, positive, notes)
    except e:
        return String(e) == words
    return False


def check_flip_absent() raises:
    var route = _route()
    var y: List[Float32] = [1.0, 2.0, 4.0]
    var p: List[Float32] = [1.5, 2.0, 2.0]
    var notes = List[Float64]()
    var mse = msel_scorer_code("neg_mean_squared_error", False)
    var neg = msel_scorer_value(route, msel_scorer_row(mse), Int(y.unsafe_ptr()), Int(p.unsafe_ptr()), 3, 0, 1, -1,
                                notes)
    var flip = MselScorerOptions()
    flip.flip = True
    var pos = msel_scorer_value(route, msel_scorer_row(mse, flip, 0), Int(y.unsafe_ptr()), Int(p.unsafe_ptr()),
                                3, 0, 1, -1, notes)
    _near(pos, (0.25 + 0.0 + 4.0) / 3.0, "make_scorer(mean_squared_error) (greater_is_better=True)")
    require(pos == -neg, "the flip negates the row's sign only")
    var t2: List[Int32] = [0, 1, 1, 0]
    var p2: List[Int32] = [0, 1, 0, 0]
    require(_raises_with(route, SCORER_POSITIVE_ABSENT, t2, p2,
                         "pos_label=1 is not a valid label. It should be one of [0, 2]"),
            "an absent pos_label among two present labels refuses in the door's words")
    require(_raises_with(route, SCORER_POSITIVE_KIND, t2, p2, "pos_label must use the same type as the targets"),
            "a pos_label of another type refuses")
    var t3: List[Int32] = [0, 1, 2, 0]
    var p3: List[Int32] = [0, 1, 2, 2]
    require(_raises_with(route, SCORER_POSITIVE_ABSENT, t3, p3,
                         "Target is multiclass but average='binary'. Please choose another average setting, one of "
                         + "[None, 'micro', 'macro', 'weighted']."),
            "three present labels refuse")
    var t1: List[Int32] = [0, 0, 0]
    var p1: List[Int32] = [0, 0, 0]
    var labels: List[String] = ["0", "2", "3"]
    notes = List[Float64]()
    var f1 = msel_scorer_row(msel_scorer_code("f1", False))
    var value = msel_scorer_value(route, EntryRef(metrics_cluster_host_v1, 0), labels, f1, Int(t1.unsafe_ptr()),
                                  Int(p1.unsafe_ptr()), 3, 3, 1, SCORER_POSITIVE_ABSENT, notes)
    var codes = _notice_codes(notes)
    require(value == 0.0 and len(codes) == 1 and codes[0] == MEMBER_NOTICE_METRIC_PRF,
            "an absent pos_label over one present label scores 0.0 with the ill-defined warning")
    _ = len(y)
    _ = len(p)
    _ = len(t2)
    _ = len(p2)
    _ = len(t3)
    _ = len(p3)
    _ = len(t1)
    _ = len(p1)


def test_metrics_scorer_builders() raises:
    def run_case(which: Int) raises:
        if which == 0:
            check_regression()
        elif which == 1:
            check_labels()
        elif which == 2:
            check_log_loss()
        elif which == 3:
            check_roc()
        elif which == 4:
            check_average_precision()
        elif which == 5:
            check_brier_top_k()
        elif which == 6:
            check_mutual_info()
        elif which == 7:
            check_notices()
        elif which == 8:
            check_cluster_pairs()
        elif which == 9:
            check_clusterer_side()
        else:
            check_flip_absent()
    check_cases("metrics_scorer_builders", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
