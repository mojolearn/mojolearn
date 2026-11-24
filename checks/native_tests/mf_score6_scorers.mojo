# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""THE LAST SCORERS OFF THE PYTHON UNITS (lane mf-score6, 2026-11-23), in a process with no Python, over the metrics
program entry of the host column (x_metrics/descriptor_host.mojo `metrics_program_host_v1`), against the
scikit-learn definitions computed here in binary64 from the same float32 inputs:

1. average=None (bindings/mcls_classify_run.mojo `label_metric_options_run`, avg LBL_NONE): t [0, 1, 2, 2, 1, 0, 2],
   p [0, 2, 2, 2, 1, 1, 0]: the per-label f1, precision and jaccard values against 2tp / (2tp + fp + fn), tp /
   pred and tp / (true + pred - tp). Through core/msel_scorer_run.mojo `msel_scorer_value` (an f1_macro row with
   average=None): three labels raise scikit-learn's `_score` words, "ValueError: scoring must return a number, got
   [...] (<class 'numpy.ndarray'>) instead. (scorer=<the scorer's label word>)"; a `labels` list of one label
   scores that label's value. `_ndarray_words` prints [0.5, 1.0] as "[0.5 1. ]" and [2/3, 0] as
   "[0.66666667 0.        ]", as NumPy's str does.
2. sample_weight as a make_scorer kwarg (the weights' float32 bits among the selection words, `w_from` / `w_n`):
   weighted accuracy, macro precision, neg MSE, neg log_loss (one column), roc_auc (binary, weighted pair count),
   neg brier (one column) and top_k (k = 2, 4 rows x 3 classes) against their weighted definitions; average
   precision with weight 2 on one row equals the unweighted value with that row repeated.
3. The weights' refusals, in the door's order and words: a length other than the side's ("mojolearn accuracy_score:
   sample_weight has 3 entries for 7 samples"), a negative entry ("... must have finite nonnegative entries"), all
   zero ("... must have positive total weight"), a metric without the parameter (`w_n` -1: "TypeError: native
   metric received an unexpected keyword argument"), an empty array (SCORER_WEIGHTS_EMPTY: "... has 0 entries for
   7 samples"), a refusal word read at UNPACK (raised as is), roc_auc one-vs-one ("sample_weight is not supported
   for multiclass one-vs-one ROC AUC, ..."). The length refusal comes before average='samples'.
4. average='samples' on 1-D targets: "NotImplementedError: mojolearn precision_recall_fscore_support:
   average='samples' applies to multilabel targets, which are NOT IMPLEMENTED (metrics/NOT_IMPLEMENTED.tsv)".
5. A float pos_label (`pos_float`): 'f1' raises SCORER_FLOAT_POS_WORDS after its multiclass check (three present
   labels raise the multiclass words first); 'neg_brier_score' raises it; 'average_precision' with the same
   positive code is unaffected (its door compares by value).
6. The scorer word group (`msel_scorer_group_words` / `msel_scorer_group_read`): average=None's word, samples,
   the weights' words and refusal word and a float pos_label round-trip; a refusal word below SCORER_WEIGHTS_EMPTY
   is refused as malformed.

Every value must match to a relative 1e-6.

FAIL-PROOF EDITS (each must make a case fail): in `label_metric_options_run` read the per-label values at
`v + 2 * chosen` for every metric (case 1, f1 and precision move); in `_ndarray_words` drop the right padding
(case 1, "[0.5 1.]"); in `label_accuracy_value` pass `False` to `label_accuracy_plan` (case 2, accuracy unweighted);
in `roc_auc_run` drop `addresses.append(w)` and pass `False` (case 2, roc_auc); in `brier_options_run` pass
`False` to `brier_stages` (case 2); in `_check_weights` drop the length check (case 3: the program reads past the
side); in `msel_scorer_value` move the `row.samples` refusal before the weights' checks (case 3, the order); in
`_binary_value` drop the `row.pos_float` refusal (case 5); in `msel_scorer_group_read` drop the `pos_float` read
(case 6).
"""
from std.math import log, isnan
from std.memory import bitcast
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from bindings.arena_prog_run import MetricRoute
from bindings.mcls_classify_run import label_metric_options_run, LBL_PRECISION, LBL_F1, LBL_JACCARD, LBL_NONE
from bindings.metrics_exp_curve_run import average_precision_run
from core.msel_scorer_run import (
    msel_scorer_row, msel_scorer_code, msel_scorer_value, MselScorer, MselScorerOptions, msel_scorer_group_words,
    msel_scorer_group_read, _ndarray_words, SCORER_NUMBER_WORDS, SCORER_FLOAT_POS_WORDS, SCORER_WEIGHTS_OK,
    SCORER_WEIGHTS_EMPTY, SCORER_GROUP_WORDS,
)
from core.entry_table import entry_ref_unresolved
from x_metrics.descriptor_host import metrics_program_host_v1

comptime CASES = 6


def _route() -> MetricRoute:
    return MetricRoute(metrics_program_host_v1, False, False)


def _near(got: Float64, want: Float64, what: String) raises:
    var tol = 1e-6 * max(abs(want), 1.0)
    require(abs(got - want) <= tol, what + ": " + String(got) + " expected " + String(want))


def _bits(values: List[Float32]) -> List[Int32]:
    """The weights' float32 bits as the UNPACK appends them to the selection words."""
    var out = List[Int32]()
    for i in range(len(values)):  # small-loop(fixture: the weights)
        out.append(bitcast[DType.int32](values[i]))
    return out^


def _weighted(code: Int, sels: List[Int32], n: Int) raises -> MselScorer:
    """The row of `code` with `n` sample_weight words at the start of `sels`."""
    var opts = MselScorerOptions()
    opts.w_from = 0
    opts.w_n = n
    return msel_scorer_row(code, opts, Int(sels.unsafe_ptr()))


def _value(row: MselScorer, labels: List[String], truth: Int, response: Int, count: Int, classes: Int, cols: Int,
           positive: Int) raises -> Float64:
    var notes = List[Float64]()
    return msel_scorer_value(_route(), entry_ref_unresolved(), labels, row, truth, response, count, classes, cols,
                             positive, notes)


def _refusal(row: MselScorer, labels: List[String], truth: Int, response: Int, count: Int, classes: Int,
             cols: Int, positive: Int) -> String:
    """The words `msel_scorer_value` raises, or "" when it answers."""
    try:
        _ = _value(row, labels, truth, response, count, classes, cols, positive)
    except e:
        return String(e)
    return String("")


# ---------------------------------------------------------------- 1. average=None


def _counts(t: List[Int32], p: List[Int32], c: Int, w: List[Float32]) -> Tuple[Float64, Float64, Float64]:
    """(tp, true, pred) sums of class c, weighted by w (empty: 1)."""
    var tp = 0.0
    var tr = 0.0
    var pr = 0.0
    for i in range(len(t)):  # small-loop(oracle: the rows)
        var wi = Float64(w[i]) if len(w) > 0 else 1.0
        if Int(t[i]) == c:
            tr += wi
        if Int(p[i]) == c:
            pr += wi
        if Int(t[i]) == c and Int(p[i]) == c:
            tp += wi
    return (tp, tr, pr)


def check_average_none() raises:
    var route = _route()
    var t: List[Int32] = [0, 1, 2, 2, 1, 0, 2]
    var p: List[Int32] = [0, 2, 2, 2, 1, 1, 0]
    var none_w = List[Float32]()
    var metrics: List[Int] = [LBL_F1, LBL_PRECISION, LBL_JACCARD]
    for m in range(3):  # small-loop(fixture: the metrics)
        var per = List[Float64]()
        var warn = 0
        _ = label_metric_options_run(route, metrics[m], LBL_NONE, Int(t.unsafe_ptr()), Int(p.unsafe_ptr()), 7, 3, 0,
                                     2, 1.0, 0, 0, 0, per, warn)
        require(len(per) == 3, "average=None answers one value per present label")
        for c in range(3):  # small-loop(oracle: the labels)
            var s = _counts(t, p, c, none_w)
            var want: Float64
            if metrics[m] == LBL_F1:
                want = 2.0 * s[0] / (s[1] + s[2])
            elif metrics[m] == LBL_PRECISION:
                want = s[0] / s[2]
            else:
                want = s[0] / (s[1] + s[2] - s[0])
            _near(per[c], want, "average=None label " + String(c))
    var labels: List[String] = ["0", "1", "2", "make_scorer(f1_score, average=None)"]
    var opts = MselScorerOptions()
    opts.none_word = 3
    var sels = List[Int32]()
    var row = msel_scorer_row(msel_scorer_code("f1_macro", False), opts, Int(sels.unsafe_ptr()))
    var words = _refusal(row, labels, Int(t.unsafe_ptr()), Int(p.unsafe_ptr()), 7, 3, 1, -1)
    require(words.startswith(SCORER_NUMBER_WORDS) and words.endswith(
        " (<class 'numpy.ndarray'>) instead. (scorer=make_scorer(f1_score, average=None))"),
            "three labels: scikit-learn's _score refusal, got " + words)
    var one: List[Int32] = [2]
    var opts1 = MselScorerOptions()
    opts1.none_word = 3
    opts1.sel_from = 0
    opts1.sel_n = 1
    var row1 = msel_scorer_row(msel_scorer_code("f1_macro", False), opts1, Int(one.unsafe_ptr()))
    var s2 = _counts(t, p, 2, none_w)
    _near(_value(row1, labels, Int(t.unsafe_ptr()), Int(p.unsafe_ptr()), 7, 3, 1, -1), 2.0 * s2[0] / (s2[1] + s2[2]),
          "labels=[2] with average=None: its one value")
    var half: List[Float64] = [0.5, 1.0]
    require(_ndarray_words(half, 1.0) == "[0.5 1. ]", "NumPy's str of [0.5, 1.0]")
    var third: List[Float64] = [2.0 / 3.0, 0.0]
    require(_ndarray_words(third, 1.0) == "[0.66666667 0.        ]", "NumPy's str of [2/3, 0]")
    _ = len(t)
    _ = len(p)
    _ = len(sels)
    _ = len(one)


# ---------------------------------------------------------------- 2. sample_weight


def check_weights() raises:
    var t: List[Int32] = [0, 1, 2, 2, 1, 0, 2]
    var p: List[Int32] = [0, 2, 2, 2, 1, 1, 0]
    var w: List[Float32] = [1.0, 2.0, 0.5, 1.0, 3.0, 1.0, 2.0]
    var bits = _bits(w)
    var none = List[String]()
    var total = 0.0
    var hit = 0.0
    for i in range(7):  # small-loop(oracle: the rows)
        total += Float64(w[i])
        if t[i] == p[i]:
            hit += Float64(w[i])
    var acc = _weighted(msel_scorer_code("accuracy", False), bits, 7)
    _near(_value(acc, none, Int(t.unsafe_ptr()), Int(p.unsafe_ptr()), 7, 3, 1, -1), hit / total, "weighted accuracy")
    var macro = 0.0
    for c in range(3):  # small-loop(oracle: the labels)
        var s = _counts(t, p, c, w)
        macro += s[0] / s[2]
    var prec = _weighted(msel_scorer_code("precision_macro", False), bits, 7)
    _near(_value(prec, none, Int(t.unsafe_ptr()), Int(p.unsafe_ptr()), 7, 3, 1, -1), macro / 3.0,
          "weighted macro precision")
    # regression
    var y: List[Float32] = [3.0, -0.5, 2.0, 7.0, 4.5, 1.0, 0.0]
    var q: List[Float32] = [2.5, 0.0, 2.0, 8.0, 4.0, 1.5, 0.5]
    var se = 0.0
    for i in range(7):  # small-loop(oracle: the rows)
        var d = Float64(y[i]) - Float64(q[i])
        se += Float64(w[i]) * d * d
    var mse = _weighted(msel_scorer_code("neg_mean_squared_error", False), bits, 7)
    _near(_value(mse, none, Int(y.unsafe_ptr()), Int(q.unsafe_ptr()), 7, 0, 1, -1), -se / total, "weighted neg MSE")
    # binary: log_loss, roc_auc, brier over one column
    var tb: List[Int32] = [0, 1, 1, 0, 1, 0, 1]
    var sb: List[Float32] = [0.2, 0.8, 0.4, 0.4, 0.9, 0.1, 0.6]
    var ll = 0.0
    var br = 0.0
    var auc = 0.0
    var wp = 0.0
    var wn = 0.0
    for i in range(7):  # small-loop(oracle: the rows)
        var pi = Float64(sb[i])
        var yi = 1.0 if tb[i] == 1 else 0.0
        ll -= Float64(w[i]) * (yi * log(pi) + (1.0 - yi) * log(1.0 - pi))
        br += Float64(w[i]) * (pi - yi) * (pi - yi)
        if tb[i] == 1:
            wp += Float64(w[i])
        else:
            wn += Float64(w[i])
        for j in range(7):  # small-loop(oracle: the pairs)
            if tb[i] == 1 and tb[j] == 0:
                var pair = Float64(w[i]) * Float64(w[j])
                if sb[i] > sb[j]:
                    auc += pair
                elif sb[i] == sb[j]:
                    auc += 0.5 * pair
    var lrow = _weighted(msel_scorer_code("neg_log_loss", False), bits, 7)
    _near(_value(lrow, none, Int(tb.unsafe_ptr()), Int(sb.unsafe_ptr()), 7, 2, 1, -1), -ll / total,
          "weighted neg log_loss")
    var rrow = _weighted(msel_scorer_code("roc_auc", True), bits, 7)
    _near(_value(rrow, none, Int(tb.unsafe_ptr()), Int(sb.unsafe_ptr()), 7, 2, 1, -1), auc / (wp * wn),
          "weighted roc_auc")
    var brow = _weighted(msel_scorer_code("neg_brier_score", False), bits, 7)
    var opts = MselScorerOptions()
    opts.w_from = 0
    opts.w_n = 7
    opts.pos_given = True
    brow = msel_scorer_row(msel_scorer_code("neg_brier_score", False), opts, Int(bits.unsafe_ptr()))
    _near(_value(brow, none, Int(tb.unsafe_ptr()), Int(sb.unsafe_ptr()), 7, 2, 1, 1), -br / total,
          "weighted neg brier (pos_label 1)")
    # top_k, k = 2: rows 0, 2, 3 hit
    var tk: List[Int32] = [0, 1, 2, 1]
    var sk: List[Float32] = [0.5, 0.3, 0.2, 0.6, 0.1, 0.3, 0.1, 0.3, 0.6, 0.2, 0.3, 0.5]
    var wk: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var kbits = _bits(wk)
    var krow = _weighted(msel_scorer_code("top_k_accuracy", False), kbits, 4)
    _near(_value(krow, none, Int(tk.unsafe_ptr()), Int(sk.unsafe_ptr()), 4, 3, 3, -1), 8.0 / 10.0, "weighted top_k")
    # average precision: weight 2 on row 1 is row 1 twice
    var ta: List[Int32] = [0, 1, 1, 0, 1]
    var sa: List[Float32] = [0.1, 0.7, 0.4, 0.35, 0.8]
    var wa: List[Float32] = [1.0, 2.0, 1.0, 1.0, 1.0]
    var td: List[Int32] = [0, 1, 1, 1, 0, 1]
    var sd: List[Float32] = [0.1, 0.7, 0.7, 0.4, 0.35, 0.8]
    var got = average_precision_run(_route(), Int(ta.unsafe_ptr()), Int(sa.unsafe_ptr()), 5, 1, 2, 1,
                                    Int(wa.unsafe_ptr()))
    var want = average_precision_run(_route(), Int(td.unsafe_ptr()), Int(sd.unsafe_ptr()), 6, 1, 2, 1)
    _near(got, want, "weighted average precision (a weight 2 row is the row twice)")
    _ = len(t)
    _ = len(p)
    _ = len(bits)
    _ = len(y)
    _ = len(q)
    _ = len(tb)
    _ = len(sb)
    _ = len(tk)
    _ = len(sk)
    _ = len(kbits)
    _ = len(ta)
    _ = len(sa)
    _ = len(wa)
    _ = len(td)
    _ = len(sd)


# ---------------------------------------------------------------- 3. the weights' refusals


def check_weight_refusals() raises:
    var t: List[Int32] = [0, 1, 2, 2, 1, 0, 2]
    var p: List[Int32] = [0, 2, 2, 2, 1, 1, 0]
    var none = List[String]()
    var acc = msel_scorer_code("accuracy", False)
    var short: List[Float32] = [1.0, 1.0, 1.0]
    var short_bits = _bits(short)
    var words = _refusal(_weighted(acc, short_bits, 3), none, Int(t.unsafe_ptr()), Int(p.unsafe_ptr()), 7, 3, 1, -1)
    require(words == "mojolearn accuracy_score: sample_weight has 3 entries for 7 samples", "length: " + words)
    var neg: List[Float32] = [1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    var neg_bits = _bits(neg)
    words = _refusal(_weighted(acc, neg_bits, 7), none, Int(t.unsafe_ptr()), Int(p.unsafe_ptr()), 7, 3, 1, -1)
    require(words == "mojolearn accuracy_score: sample_weight must have finite nonnegative entries", "negative: " + words)
    var zero = List[Float32](length=7, fill=0.0)
    var zero_bits = _bits(zero)
    words = _refusal(_weighted(acc, zero_bits, 7), none, Int(t.unsafe_ptr()), Int(p.unsafe_ptr()), 7, 3, 1, -1)
    require(words == "mojolearn accuracy_score: sample_weight must have positive total weight", "zero: " + words)
    var opts = MselScorerOptions()
    opts.w_n = -1
    var sels = List[Int32]()
    words = _refusal(msel_scorer_row(msel_scorer_code("max_error", False), opts, Int(sels.unsafe_ptr())), none,
                     Int(t.unsafe_ptr()), Int(p.unsafe_ptr()), 7, 0, 1, -1)
    require(words == "TypeError: native metric received an unexpected keyword argument", "no parameter: " + words)
    var empty = MselScorerOptions()
    empty.w_word = SCORER_WEIGHTS_EMPTY
    words = _refusal(msel_scorer_row(acc, empty, Int(sels.unsafe_ptr())), none, Int(t.unsafe_ptr()),
                     Int(p.unsafe_ptr()), 7, 3, 1, -1)
    require(words == "mojolearn accuracy_score: sample_weight has 0 entries for 7 samples", "empty: " + words)
    var told = MselScorerOptions()
    told.w_word = 0
    var said: List[String] = ["mojolearn accuracy_score: sample_weight must be 1-D, got shape (7, 1)"]
    words = _refusal(msel_scorer_row(acc, told, Int(sels.unsafe_ptr())), said, Int(t.unsafe_ptr()),
                     Int(p.unsafe_ptr()), 7, 3, 1, -1)
    require(words == said[0], "the UNPACK's refusal word: " + words)
    # one-vs-one roc_auc: 6 rows x 3 classes
    var to: List[Int32] = [0, 1, 2, 0, 1, 2]
    var so = List[Float32](length=18, fill=Float32(1.0 / 3.0))
    var ones = List[Float32](length=6, fill=1.0)
    var one_bits = _bits(ones)
    words = _refusal(_weighted(msel_scorer_code("roc_auc_ovo", False), one_bits, 6), none, Int(to.unsafe_ptr()),
                     Int(so.unsafe_ptr()), 6, 3, 3, -1)
    require(words == "sample_weight is not supported for multiclass one-vs-one ROC AUC, 'sample_weight' must be "
            + "None in this case.", "ovo: " + words)
    # the length refusal before average='samples'
    var both = MselScorerOptions()
    both.samples = True
    both.w_from = 0
    both.w_n = 3
    words = _refusal(msel_scorer_row(msel_scorer_code("precision_macro", False), both, Int(short_bits.unsafe_ptr())),
                     none, Int(t.unsafe_ptr()), Int(p.unsafe_ptr()), 7, 3, 1, -1)
    require(words == "mojolearn precision_recall_fscore_support: sample_weight has 3 entries for 7 samples",
            "the weights before average='samples': " + words)
    _ = len(t)
    _ = len(p)
    _ = len(short_bits)
    _ = len(neg_bits)
    _ = len(zero_bits)
    _ = len(sels)
    _ = len(to)
    _ = len(so)
    _ = len(one_bits)


# ---------------------------------------------------------------- 4. average='samples'


def check_samples() raises:
    var t: List[Int32] = [0, 1, 2, 2]
    var p: List[Int32] = [0, 2, 2, 1]
    var opts = MselScorerOptions()
    opts.samples = True
    var sels = List[Int32]()
    var words = _refusal(msel_scorer_row(msel_scorer_code("f1_macro", False), opts, Int(sels.unsafe_ptr())),
                         List[String](), Int(t.unsafe_ptr()), Int(p.unsafe_ptr()), 4, 3, 1, -1)
    require(words == "NotImplementedError: mojolearn precision_recall_fscore_support: average='samples' applies to "
            + "multilabel targets, which are NOT IMPLEMENTED (metrics/NOT_IMPLEMENTED.tsv)", "samples: " + words)
    _ = len(t)
    _ = len(p)
    _ = len(sels)


# ---------------------------------------------------------------- 5. a float pos_label


def check_float_pos_label() raises:
    var t: List[Int32] = [0, 1, 1, 0]
    var p: List[Int32] = [0, 1, 0, 0]
    var labels: List[String] = ["0", "1", "1.0"]
    var opts = MselScorerOptions()
    opts.pos_given = True
    opts.pos_float = True
    opts.pos_word = 2
    var sels = List[Int32]()
    var f1 = msel_scorer_row(msel_scorer_code("f1", False), opts, Int(sels.unsafe_ptr()))
    var words = _refusal(f1, labels, Int(t.unsafe_ptr()), Int(p.unsafe_ptr()), 4, 2, 1, 1)
    require(words == SCORER_FLOAT_POS_WORDS, "f1 with pos_label=1.0: " + words)
    var t3: List[Int32] = [0, 1, 2, 0]
    var p3: List[Int32] = [0, 1, 2, 2]
    words = _refusal(f1, labels, Int(t3.unsafe_ptr()), Int(p3.unsafe_ptr()), 4, 3, 1, 1)
    require(words.startswith("Target is multiclass but average='binary'."), "the multiclass words first: " + words)
    var s: List[Float32] = [0.2, 0.8, 0.6, 0.3]
    var brier = msel_scorer_row(msel_scorer_code("neg_brier_score", False), opts, Int(sels.unsafe_ptr()))
    words = _refusal(brier, labels, Int(t.unsafe_ptr()), Int(s.unsafe_ptr()), 4, 2, 1, 1)
    require(words == SCORER_FLOAT_POS_WORDS, "brier with pos_label=1.0: " + words)
    var by_value = MselScorerOptions()
    by_value.pos_given = True
    by_value.pos_word = 2
    var ap = msel_scorer_row(msel_scorer_code("average_precision", False), by_value, Int(sels.unsafe_ptr()))
    var plain = msel_scorer_row(msel_scorer_code("average_precision", False))
    _near(_value(ap, labels, Int(t.unsafe_ptr()), Int(s.unsafe_ptr()), 4, 2, 1, 1),
          _value(plain, labels, Int(t.unsafe_ptr()), Int(s.unsafe_ptr()), 4, 2, 1, 1),
          "average_precision compares a float pos_label by value")
    _ = len(t)
    _ = len(p)
    _ = len(t3)
    _ = len(p3)
    _ = len(s)
    _ = len(sels)


# ---------------------------------------------------------------- 6. the word group


def check_group_words() raises:
    var sels: List[Int32] = [2, 1065353216, 1073741824]
    var opts = MselScorerOptions()
    opts.pos_given = True
    opts.pos_float = True
    opts.pos_word = 4
    opts.sel_from = 0
    opts.sel_n = 1
    opts.none_word = 5
    opts.samples = True
    opts.w_from = 1
    opts.w_n = 2
    opts.w_word = SCORER_WEIGHTS_OK
    var words: List[Float64] = [9.0]
    msel_scorer_group_words(words, 7, 1, opts, sels)
    require(len(words) == 1 + SCORER_GROUP_WORDS + 1 + 2, "the group is its fixed words, one label code, two weights")
    var code = 0
    var positive = -1
    var back = MselScorerOptions()
    var read = List[Int32]()
    var end = msel_scorer_group_read(words, 1, code, positive, back, read)
    require(end == len(words) and code == 7 and positive == 1, "the group's end, code and positive word")
    require(back.pos_given and back.pos_float and back.pos_word == 4, "a float pos_label round-trips")
    require(back.none_word == 5 and back.samples and back.w_n == 2 and back.w_word == SCORER_WEIGHTS_OK,
            "average None / samples / the weights round-trip")
    require(len(read) == 3 and read[back.sel_from] == 2 and read[back.w_from] == 1065353216
            and read[back.w_from + 1] == 1073741824, "the label code and the weights' bits round-trip")
    words[1 + 12] = -3.0
    var refused = False
    try:
        _ = msel_scorer_group_read(words, 1, code, positive, back, read)
    except:
        refused = True
    require(refused, "a refusal word below SCORER_WEIGHTS_EMPTY is malformed")


def test_mf_score6_scorers() raises:
    def run_case(which: Int) raises:
        if which == 0:
            check_average_none()
        elif which == 1:
            check_weights()
        elif which == 2:
            check_weight_refusals()
        elif which == 3:
            check_samples()
        elif which == 4:
            check_float_pos_label()
        else:
            check_group_words()
    check_cases("mf_score6_scorers", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
