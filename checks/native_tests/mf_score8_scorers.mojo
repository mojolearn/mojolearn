# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""EVERY REMAINING SCORER, TYPED (lane mf-score8, 2026-11-23), in a process with no Python, over the metrics
program entry of the host column (x_metrics/descriptor_host.mojo `metrics_program_host_v1`). Each new search
scorer row of core/msel_scorer_run.mojo runs the program its metric's public door runs; the expected values are
scikit-learn's definitions computed by hand from the same inputs (worked in each case):

0. mean_pinball_loss alpha=0.3, y [1, 2, 3, 4], p [2, 2, 2, 2]: (0.7 + 0 + 0.3 + 0.6) / 4 = 0.4; alpha 1.5 raises
   "alpha must be a real number in [0, 1]".
1. mean_tweedie_deviance: power 0 is the MSE 1.5; power 1.5 is 0.435393321 (the unit deviance's mean); power 0.5
   raises "mojolearn mean_tweedie_deviance: power in (0, 1) is not a Tweedie distribution"; power 1.5 over a
   negative y raises the domain words with the power printed from its label word ("power=1.5").
2. d2_tweedie_score power 0 equals r2_score on the same values (1e-6); d2_pinball_score alpha 0.5 equals
   d2_absolute_error_score; one sample answers NaN with the D^2 notice (MEMBER_NOTICE_METRIC_FEW, a = 1).
3. r2 force_finite: y [2, 2, 2], p [1, 2, 3]: True answers 0.0, False -inf; multioutput 'raw_values' and
   'variance_weighted' on y [1, 2, 3, 4], p [1.5, 2, 2.5, 4.5] answer the 'uniform_average' value.
4. accuracy_score(normalize=False) over t [0, 1, 2, 2, 1, 0, 2], p [0, 2, 2, 2, 1, 1, 0]: 4 correct; weighted
   [2, 1, 1, 1, 1, 3, 0.5]: 5.0. balanced_accuracy adjusted=True: (5/9 - 1/3) / (2/3) = 1/3.
5. cohen_kappa_score over the same codes: 11/32; weights='linear' 1 - 4 / (44/7) = 16/44; labels [2, 0] (codes in
   that order): 0.5; a side whose labels hold none of y1's raises the door's words; t = p = [0, 0, 0] with
   replace_undefined_by -1.0: -1.0 and a MEMBER_NOTICE_METRIC_KAPPA notice.
6. A regression scorer on a classifier: label values 10, 20, 30 (int labels' float32 words), t [0, 1, 2, 2],
   p [0, 2, 2, 1]: neg_mean_squared_error -50.0; string labels ['1.5', 'b', '2'] (kinds TEXT, UNREAD, TEXT):
   t [0, 1] raises "ValueError: could not convert string to float: 'b'"; neg_median_absolute_error over TEXT
   labels raises SCORER_NUMERIC_WORDS.
7. log_loss normalize=False, binary, t [0, 1, 1, 0], p1 [0.2, 0.7, 0.9, 0.4]: the summed loss 1.196004635 (1e-5);
   top_k_accuracy k=2 normalize=False over four 3-class rows: 2.0 (normalize True: 0.5).
8. The word group (`msel_scorer_group_words` / `msel_scorer_group_read`): the eight new words and two label value
   words per label round-trip; a refusal word is raised per side as its label word.

FAIL-PROOF EDITS (each must make a case fail): in `reg_metric_options_run` pass 0.5 instead of `alpha` to the
PINBALL stages (case 0); drop `shown = param_shown.copy()` (case 1, the printed power); pass `True` for
`force_finite` to `reg_r2_stages` (case 3); answer `out + 1` as f64 for CNT_CORRECT weighted (case 4); in
`kappa_stages` pass `0` for `wm` (case 5, linear); in `_label_floats` read `lv[2 * k + 1]` as the value (case 6);
in `log_loss_options_run` pass True to `mc_log_loss_stages` (case 7); in `msel_scorer_group_read` drop the
`lv_n` words (case 8).
"""
from std.memory import bitcast
from std.math import isnan, isinf
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from bindings.arena_prog_run import MetricRoute
from bindings.metrics_reg_run import REG_MO_RAW, REG_MO_VARIANCE
from core.msel_scorer_run import (
    msel_scorer_row, msel_scorer_value, msel_scorer_metric_code, msel_scorer_code, msel_scorer_group_words,
    msel_scorer_group_read, MselScorer, MselScorerOptions, SCORER_GROUP_WORDS, SCORER_LABEL_NUMBER,
    SCORER_LABEL_TEXT, SCORER_LABEL_UNREAD, SCORER_NUMERIC_WORDS,
)
from core.member_notices import MEMBER_NOTICE_METRIC_FEW, MEMBER_NOTICE_METRIC_KAPPA
from core.entry_table import entry_ref_unresolved
from x_metrics.descriptor_host import metrics_program_host_v1

comptime CASES = 9


def _route() -> MetricRoute:
    return MetricRoute(metrics_program_host_v1, False, False)


def _near(got: Float64, want: Float64, what: String, tol_rel: Float64 = 1e-6) raises:
    var tol = tol_rel * max(abs(want), 1.0)
    require(abs(got - want) <= tol, what + ": " + String(got) + " expected " + String(want))


def _metric_row(metric: String, opts: MselScorerOptions, sels: List[Int32]) raises -> MselScorer:
    var code = msel_scorer_metric_code(metric, True)
    require(code >= 1, metric + " has a typed row")
    return msel_scorer_row(code, opts, Int(sels.unsafe_ptr()))


def _value(row: MselScorer, words: List[String], truth: Int, response: Int, count: Int, classes: Int,
           mut notes: List[Float64]) raises -> Float64:
    return msel_scorer_value(_route(), entry_ref_unresolved(), words, row, truth, response, count, classes, 1, -1,
                             notes)


def _reg(metric: String, opts: MselScorerOptions, words: List[String], y: List[Float32], p: List[Float32],
         mut notes: List[Float64]) raises -> Float64:
    var none = List[Int32]()
    var row = _metric_row(metric, opts, none)
    var value = _value(row, words, Int(y.unsafe_ptr()), Int(p.unsafe_ptr()), len(y), 0, notes)
    _ = len(none)
    return value


def _raised(metric: String, opts: MselScorerOptions, words: List[String], y: List[Float32],
            p: List[Float32]) raises -> String:
    var notes = List[Float64]()
    try:
        _ = _reg(metric, opts, words, y, p, notes)
    except e:
        return String(e)
    return String("")


def _given(param: Float64, word: Int) -> MselScorerOptions:
    var opts = MselScorerOptions()
    opts.param = param
    opts.param_word = word
    return opts^


def check_pinball() raises:
    var y: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var p: List[Float32] = [2.0, 2.0, 2.0, 2.0]
    var words: List[String] = ["0.3"]
    var notes = List[Float64]()
    _near(_reg("mean_pinball_loss", _given(0.3, 0), words, y, p, notes), 0.4, "mean_pinball_loss alpha 0.3")
    var bad: List[String] = ["1.5"]
    var got = _raised("mean_pinball_loss", _given(1.5, 0), bad, y, p)
    require(got == "alpha must be a real number in [0, 1]", "alpha refusal: " + got)


def check_tweedie() raises:
    var y: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var p: List[Float32] = [2.0, 2.0, 2.0, 2.0]
    var notes = List[Float64]()
    var w0: List[String] = ["0"]
    _near(_reg("mean_tweedie_deviance", _given(0.0, 0), w0, y, p, notes), 1.5, "tweedie power 0")
    var w15: List[String] = ["1.5"]
    _near(_reg("mean_tweedie_deviance", _given(1.5, 0), w15, y, p, notes), 0.43539332147391163, "tweedie power 1.5",
          1e-5)
    var w05: List[String] = ["0.5"]
    var got = _raised("mean_tweedie_deviance", _given(0.5, 0), w05, y, p)
    require(got == "mojolearn mean_tweedie_deviance: power in (0, 1) is not a Tweedie distribution",
            "power refusal: " + got)
    var yn: List[Float32] = [-1.0, 2.0, 3.0, 4.0]
    got = _raised("mean_tweedie_deviance", _given(1.5, 0), w15, yn, p)
    require(got == ("Mean Tweedie deviance error with power=1.5 can only be used on non-negative y and strictly "
                    + "positive y_pred."), "domain words: " + got)


def check_d2() raises:
    var y: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var p: List[Float32] = [1.5, 2.0, 2.5, 4.5]
    var notes = List[Float64]()
    var none = List[String]()
    var w0: List[String] = ["0"]
    var r2 = _reg("r2_score", MselScorerOptions(), none, y, p, notes)
    _near(_reg("d2_tweedie_score", _given(0.0, 0), w0, y, p, notes), r2, "d2_tweedie power 0 is r2")
    var w5: List[String] = ["0.5"]
    var d2a = _reg("d2_absolute_error_score", MselScorerOptions(), none, y, p, notes)
    _near(_reg("d2_pinball_score", _given(0.5, 0), w5, y, p, notes), d2a, "d2_pinball alpha 0.5")
    var y1: List[Float32] = [1.0]
    var p1: List[Float32] = [2.0]
    var few = List[Float64]()
    var v = _reg("d2_pinball_score", _given(0.5, 0), w5, y1, p1, few)
    require(isnan(v), "one sample: NaN")
    require(len(few) == 3 and Int(few[0]) == MEMBER_NOTICE_METRIC_FEW and few[1] == 1.0, "the D^2 notice")


def check_r2_options() raises:
    var y: List[Float32] = [2.0, 2.0, 2.0]
    var p: List[Float32] = [1.0, 2.0, 3.0]
    var notes = List[Float64]()
    var none = List[String]()
    var finite = MselScorerOptions()
    finite.ff = 1
    _near(_reg("r2_score", finite, none, y, p, notes), 0.0, "force_finite True")
    var raw = MselScorerOptions()
    raw.ff = 0
    var v = _reg("r2_score", raw, none, y, p, notes)
    require(isinf(v) and v < 0.0, "force_finite False: -inf, got " + String(v))
    var y2: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var p2: List[Float32] = [1.5, 2.0, 2.5, 4.5]
    var plain = _reg("r2_score", MselScorerOptions(), none, y2, p2, notes)
    var mo_raw = MselScorerOptions()
    mo_raw.mo = REG_MO_RAW
    _near(_reg("r2_score", mo_raw, none, y2, p2, notes), plain, "raw_values on 1-D")
    var mo_var = MselScorerOptions()
    mo_var.mo = REG_MO_VARIANCE
    _near(_reg("r2_score", mo_var, none, y2, p2, notes), plain, "variance_weighted on 1-D")


def _codes(mut t: List[Int32], mut p: List[Int32]):
    t = [0, 1, 2, 2, 1, 0, 2]
    p = [0, 2, 2, 2, 1, 1, 0]


def check_counts() raises:
    var t = List[Int32]()
    var p = List[Int32]()
    _codes(t, p)
    var none = List[Int32]()
    var words = List[String]()
    var notes = List[Float64]()
    var code = msel_scorer_metric_code("accuracy_score", False)
    var row = msel_scorer_row(code, MselScorerOptions(), Int(none.unsafe_ptr()))
    _near(_value(row, words, Int(t.unsafe_ptr()), Int(p.unsafe_ptr()), 7, 3, notes), 4.0, "correct count")
    var w: List[Float32] = [2.0, 1.0, 1.0, 1.0, 1.0, 3.0, 0.5]
    var sels = List[Int32]()
    for i in range(len(w)):  # small-loop(fixture: the weights' bits)
        sels.append(bitcast[DType.int32](w[i]))
    var wopts = MselScorerOptions()
    wopts.w_from = 0
    wopts.w_n = 7
    var wrow = msel_scorer_row(code, wopts, Int(sels.unsafe_ptr()))
    _near(_value(wrow, words, Int(t.unsafe_ptr()), Int(p.unsafe_ptr()), 7, 3, notes), 5.0, "correct weight")
    var adjusted = MselScorerOptions()
    adjusted.flag = 1
    var brow = msel_scorer_row(msel_scorer_code("balanced_accuracy", False), adjusted, Int(none.unsafe_ptr()))
    _near(_value(brow, words, Int(t.unsafe_ptr()), Int(p.unsafe_ptr()), 7, 3, notes), 1.0 / 3.0, "adjusted")
    _ = len(t)
    _ = len(p)
    _ = len(none)
    _ = len(sels)


def check_kappa() raises:
    var t = List[Int32]()
    var p = List[Int32]()
    _codes(t, p)
    var none = List[Int32]()
    var words = List[String]()
    var notes = List[Float64]()
    var ta = Int(t.unsafe_ptr())
    var pa = Int(p.unsafe_ptr())
    _near(_value(_metric_row("cohen_kappa_score", MselScorerOptions(), none), words, ta, pa, 7, 3, notes),
          11.0 / 32.0, "kappa")
    var linear = MselScorerOptions()
    linear.wmode = 1
    _near(_value(_metric_row("cohen_kappa_score", linear, none), words, ta, pa, 7, 3, notes), 16.0 / 44.0,
          "linear kappa")
    var sel: List[Int32] = [2, 0]
    var chosen = MselScorerOptions()
    chosen.sel_from = 0
    chosen.sel_n = 2
    _near(_value(_metric_row("cohen_kappa_score", chosen, sel), words, ta, pa, 7, 3, notes), 0.5, "kappa labels")
    var t1: List[Int32] = [1, 1, 1]
    var p1: List[Int32] = [0, 0, 0]
    var only: List[Int32] = [0]
    var single = MselScorerOptions()
    single.sel_from = 0
    single.sel_n = 1
    var got = String("")
    try:
        _ = _value(_metric_row("cohen_kappa_score", single, only), words, Int(t1.unsafe_ptr()), Int(p1.unsafe_ptr()),
                   3, 2, notes)
    except e:
        got = String(e)
    require(got.startswith("At least one label in `labels` must be present in `y1`"), "labels refusal: " + got)
    var tz: List[Int32] = [0, 0, 0]
    var pz: List[Int32] = [0, 0, 0]
    var undefined = MselScorerOptions()
    undefined.param = -1.0
    undefined.param_word = 0
    var shown: List[String] = ["-1.0"]
    var kn = List[Float64]()
    var v = _value(_metric_row("cohen_kappa_score", undefined, none), shown, Int(tz.unsafe_ptr()), Int(pz.unsafe_ptr()),
                   3, 1, kn)
    require(v == -1.0, "replace_undefined_by: " + String(v))
    require(len(kn) == 3 and Int(kn[0]) == MEMBER_NOTICE_METRIC_KAPPA, "the kappa notice")
    _ = len(t)
    _ = len(p)
    _ = len(none)
    _ = len(sel)
    _ = len(t1)
    _ = len(p1)
    _ = len(only)
    _ = len(tz)
    _ = len(pz)


def _values(values: List[Float32], kinds: List[Int], mut sels: List[Int32]):
    for i in range(len(values)):  # small-loop(fixture: one value and kind word per label)
        sels.append(bitcast[DType.int32](values[i]))
        sels.append(Int32(kinds[i]))


def check_regression_on_classifier() raises:
    var lv = List[Int32]()
    var values: List[Float32] = [10.0, 20.0, 30.0]
    var kinds: List[Int] = [SCORER_LABEL_NUMBER, SCORER_LABEL_NUMBER, SCORER_LABEL_NUMBER]
    _values(values, kinds, lv)
    var opts = MselScorerOptions()
    opts.lv_from = 0
    opts.lv_n = 3
    var row = msel_scorer_row(msel_scorer_code("neg_mean_squared_error", False), opts, Int(lv.unsafe_ptr()))
    var t: List[Int32] = [0, 1, 2, 2]
    var p: List[Int32] = [0, 2, 2, 1]
    var words: List[String] = ["10", "20", "30"]
    var notes = List[Float64]()
    _near(_value(row, words, Int(t.unsafe_ptr()), Int(p.unsafe_ptr()), 4, 3, notes), -50.0, "MSE of label values")
    var text = List[Int32]()
    var tv: List[Float32] = [1.5, 0.0, 2.0]
    var tk: List[Int] = [SCORER_LABEL_TEXT, SCORER_LABEL_UNREAD, SCORER_LABEL_TEXT]
    _values(tv, tk, text)
    var trow = msel_scorer_row(msel_scorer_code("neg_mean_squared_error", False), opts, Int(text.unsafe_ptr()))
    var sw: List[String] = ["'1.5'", "'b'", "'2'"]
    var t2: List[Int32] = [0, 1, 2]
    var p2: List[Int32] = [0, 0, 2]
    var got = String("")
    try:
        _ = _value(trow, sw, Int(t2.unsafe_ptr()), Int(p2.unsafe_ptr()), 3, 3, notes)
    except e:
        got = String(e)
    require(got == "ValueError: could not convert string to float: 'b'", "NumPy's words: " + got)
    var mrow = msel_scorer_row(msel_scorer_code("neg_median_absolute_error", False), opts, Int(text.unsafe_ptr()))
    got = String("")
    try:
        _ = _value(mrow, sw, Int(p2.unsafe_ptr()), Int(p2.unsafe_ptr()), 3, 3, notes)
    except e:
        got = String(e)
    require(got == SCORER_NUMERIC_WORDS, "dtype='numeric' words: " + got)
    _ = len(lv)
    _ = len(t)
    _ = len(p)
    _ = len(text)
    _ = len(t2)
    _ = len(p2)


def check_normalize_flags() raises:
    var t: List[Int32] = [0, 1, 1, 0]
    var p1: List[Float32] = [0.2, 0.7, 0.9, 0.4]
    var none = List[Int32]()
    var words = List[String]()
    var notes = List[Float64]()
    var summed = MselScorerOptions()
    summed.flag = 0
    var lrow = msel_scorer_row(msel_scorer_code("neg_log_loss", False), summed, Int(none.unsafe_ptr()))
    _near(msel_scorer_value(_route(), entry_ref_unresolved(), words, lrow, Int(t.unsafe_ptr()), Int(p1.unsafe_ptr()),
                            4, 2, 1, -1, notes), -1.1960046346767592, "summed log_loss", 1e-5)
    var tk: List[Int32] = [0, 1, 2, 0]
    var scores: List[Float32] = [0.5, 0.3, 0.2, 0.1, 0.2, 0.7, 0.6, 0.3, 0.1, 0.2, 0.5, 0.3]
    var krow = msel_scorer_row(msel_scorer_code("top_k_accuracy", False), summed, Int(none.unsafe_ptr()))
    _near(msel_scorer_value(_route(), entry_ref_unresolved(), words, krow, Int(tk.unsafe_ptr()),
                            Int(scores.unsafe_ptr()), 4, 3, 3, -1, notes), 2.0, "top_k hits")
    var kfrac = msel_scorer_row(msel_scorer_code("top_k_accuracy", False), MselScorerOptions(), Int(none.unsafe_ptr()))
    _near(msel_scorer_value(_route(), entry_ref_unresolved(), words, kfrac, Int(tk.unsafe_ptr()),
                            Int(scores.unsafe_ptr()), 4, 3, 3, -1, notes), 0.5, "top_k fraction")
    _ = len(t)
    _ = len(p1)
    _ = len(none)
    _ = len(tk)
    _ = len(scores)


def check_group_words() raises:
    var sels: List[Int32] = [7, 1092616192, 1, 1101004800, 2]
    var opts = MselScorerOptions()
    opts.mo = 1
    opts.ff = 0
    opts.param = 0.25
    opts.param_word = 3
    opts.refuse_word = 4
    opts.flag = 1
    opts.wmode = 2
    opts.sel_from = 0
    opts.sel_n = 1
    opts.lv_from = 1
    opts.lv_n = 2
    var words: List[Float64] = [9.0]
    msel_scorer_group_words(words, 5, -1, opts, sels)
    require(len(words) == 1 + SCORER_GROUP_WORDS + 1 + 4,
            "the group is its fixed words, one label code, four value words")
    var code = 0
    var positive = 0
    var back = MselScorerOptions()
    var read = List[Int32]()
    var end = msel_scorer_group_read(words, 1, code, positive, back, read)
    require(end == len(words) and code == 5 and positive == -1, "the group's end, code and positive word")
    require(back.mo == 1 and back.ff == 0 and back.param == 0.25 and back.param_word == 3 and back.refuse_word == 4
            and back.flag == 1 and back.wmode == 2, "the eight words round-trip")
    require(back.lv_n == 2 and read[back.lv_from] == 1092616192 and read[back.lv_from + 3] == 2,
            "the label value words round-trip")
    var refuse = MselScorerOptions()
    refuse.refuse_word = 0
    var none = List[Int32]()
    var row = msel_scorer_row(msel_scorer_code("r2", False), refuse, Int(none.unsafe_ptr()))
    var y: List[Float32] = [1.0, 2.0]
    var said: List[String] = ["Custom weights are useful only in multi-output cases."]
    var notes = List[Float64]()
    var got = String("")
    try:
        _ = _value(row, said, Int(y.unsafe_ptr()), Int(y.unsafe_ptr()), 2, 0, notes)
    except e:
        got = String(e)
    require(got == "Custom weights are useful only in multi-output cases.", "the refusal word: " + got)
    _ = len(sels)
    _ = len(none)
    _ = len(y)


def test_mf_score8_scorers() raises:
    def run_case(which: Int) raises:
        if which == 0:
            check_pinball()
        elif which == 1:
            check_tweedie()
        elif which == 2:
            check_d2()
        elif which == 3:
            check_r2_options()
        elif which == 4:
            check_counts()
        elif which == 5:
            check_kappa()
        elif which == 6:
            check_regression_on_classifier()
        elif which == 7:
            check_normalize_flags()
        else:
            check_group_words()
    check_cases("mf_score8_scorers", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
