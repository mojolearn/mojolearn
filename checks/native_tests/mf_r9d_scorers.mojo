# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""THE SEARCH SCORER ROWS OF LANE mf-r9d (2026-11-23), in a process with no Python, over the metrics program entry
of the host column (x_metrics/descriptor_host.mojo `metrics_program_host_v1`). Each case calls the RUN function a
search side calls (core/msel_scorer_run.mojo `msel_scorer_value`, `msel_scorer_code`, `msel_scorer_row` and the
word group writer and reader) with the per-score words the search RUN sets (`MselScorer.cpu_fast`). Expected words
and values are scikit-learn 1.9.1's (metrics/_ranking.py `roc_auc_score` lines 708-747, `_binary_roc_auc_score`,
metrics/_base.py `_average_binary_score`, utils/validation.py `column_or_1d`) and the native scorer door's
(bindings/msel_scorer_native_door.mojo `MselMetricPrepared.prepare`, core/device_route.mojo `device_cpu_fast`),
worked by hand from the same inputs:

0. the word group round trip of the three new words: an r2 row over predict_proba (`resp`) with the other
   numeric_mode tier (`tier` 1), its refusal word 3 and predict_log_proba (`log`) reads back as written and the
   row carries them; a log flag on a row that reads no predict_proba is refused; a tier word of 2 is refused.
1. the tier's refusals per side, in the door's order: an unexpected sample_weight first ("TypeError: native metric
   received an unexpected keyword argument"), then the refused image's words (label word 0), then device='cpu'
   with the fast tier; a row of the default tier with neither scores r2 as before (truth [1, 2, 3, 4], prediction
   [1, 2, 3, 5]: SS_res 1, SS_tot 5, r2 0.8).
2. 'roc_auc' (decision half) on a NON-classifier: y [-1, 1, 1, -1] (binary: -1 negative, 1 positive), decision
   [0.1, 0.9, 0.2, 0.3]: positive scores 0.9 and 0.2 against negative 0.1 and 0.3, three of four pairs ordered:
   0.75; y [0, 1, 2, 1]: "multi_class must be in ('ovo', 'ovr')"; y [0.5, 1, 1, 0]: "ValueError: continuous
   format is not supported"; binary y with two score columns: "ValueError: y should be a 1d array, got an array
   of shape (4, 2) instead."; y [1, 1, 1, 1]: nan and one MEMBER_NOTICE_METRIC_ROC_ONE notice.

FAIL-PROOF EDITS (each must make a case fail): in `msel_scorer_group_read` drop the `tier` read (case 0's tier
stays 0); in `msel_scorer_row(code, opts, sel_base)` drop `row.log = True` (case 0); in `msel_scorer_value` drop
the `tier_word` raise (case 1 scores instead of the image's words) or move it before the `w_n < 0` check (case 1's
first words change); in `_float_roc_value` drop the union's ascending swap (case 2 answers 0.25), drop the
two-column refusal (case 2 scores the (4, 2) block), or code a continuous y (case 2 scores 0.5's side).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from bindings.arena_prog_run import MetricRoute
from core.msel_scorer_run import (
    msel_scorer_row, msel_scorer_value, msel_scorer_code, MselScorer, MselScorerOptions, msel_scorer_group_words,
    msel_scorer_group_read, SCORER_CPU_FAST_WORDS,
)
from core.member_abi import MEMBER_OP_PREDICT_PROBA, MEMBER_OP_DECISION
from core.member_notices import MEMBER_NOTICE_METRIC_ROC_ONE
from core.entry_table import entry_ref_unresolved
from x_metrics.descriptor_host import metrics_program_host_v1

comptime CASES = 3


def _route() -> MetricRoute:
    return MetricRoute(metrics_program_host_v1, False, False)


def _near(got: Float64, want: Float64, what: String) raises:
    require(abs(got - want) <= 1e-6 * max(abs(want), 1.0), what + ": " + String(got) + " expected " + String(want))


def _words(row: MselScorer, words: List[String], truth: Int, response: Int, count: Int, classes: Int,
           cols: Int) raises -> String:
    var notes = List[Float64]()
    try:
        _ = msel_scorer_value(_route(), entry_ref_unresolved(), words, row, truth, response, count, classes, cols,
                              -1, notes)
    except e:
        return String(e)
    return String("")


def check_group_words() raises:
    var code = msel_scorer_code("r2", False)
    var opts = MselScorerOptions()
    opts.resp = MEMBER_OP_PREDICT_PROBA
    opts.tier = 1
    opts.tier_word = 3
    opts.log = True
    var words = List[Float64]()
    var sels = List[Int32]()
    msel_scorer_group_words(words, code, -1, opts, sels)
    var got_code = 0
    var positive = 0
    var back = MselScorerOptions()
    var read_sels = List[Int32]()
    var after = msel_scorer_group_read(words, 0, got_code, positive, back, read_sels)
    require(after == len(words), "the group is read to its end")
    require(got_code == code and back.tier == 1 and back.tier_word == 3 and back.log, "the three words read back")
    var row = msel_scorer_row(code, back, Int(read_sels.unsafe_ptr()))
    require(row.tier == 1 and row.tier_word == 3 and row.log and row.op == MEMBER_OP_PREDICT_PROBA,
            "the row carries the tier, its word and the log response")
    var plain = MselScorerOptions()
    plain.log = True
    var refused = String("")
    try:
        _ = msel_scorer_row(code, plain, Int(read_sels.unsafe_ptr()))
    except e:
        refused = String(e)
    require(refused == "RuntimeError: predict_log_proba on a search scorer that reads no predict_proba",
            "log on a predict row: " + refused)
    words[22] = 2.0
    var bad = String("")
    try:
        _ = msel_scorer_group_read(words, 0, got_code, positive, back, read_sels)
    except e:
        bad = String(e)
    require(bad == "member ABI: a search scorer's numeric_mode tier word is not 0 or 1", "tier 2: " + bad)
    _ = len(sels)
    _ = len(read_sels)


def check_tier_refusals() raises:
    var code = msel_scorer_code("r2", False)
    var truth: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var pred: List[Float32] = [1.0, 2.0, 3.0, 5.0]
    var label_words: List[String] = ["ImportError: mojolearn: the fast metrics image is not built"]
    var row = msel_scorer_row(code)
    row.tier = 1
    row.tier_word = 0
    row.cpu_fast = True
    row.w_n = -1
    var got = _words(row, label_words, Int(truth.unsafe_ptr()), Int(pred.unsafe_ptr()), 4, 0, 1)
    require(got == "TypeError: native metric received an unexpected keyword argument", "kwargs first: " + got)
    row.w_n = 0
    got = _words(row, label_words, Int(truth.unsafe_ptr()), Int(pred.unsafe_ptr()), 4, 0, 1)
    require(got == label_words[0], "the image's words: " + got)
    row.tier_word = -1
    got = _words(row, label_words, Int(truth.unsafe_ptr()), Int(pred.unsafe_ptr()), 4, 0, 1)
    require(got == SCORER_CPU_FAST_WORDS, "device='cpu' with the fast tier: " + got)
    var plain = msel_scorer_row(code)
    var notes = List[Float64]()
    var value = msel_scorer_value(_route(), entry_ref_unresolved(), label_words, plain, Int(truth.unsafe_ptr()),
                                  Int(pred.unsafe_ptr()), 4, 0, 1, -1, notes)
    _near(value, 0.8, "r2 on the default tier")
    _ = len(truth)  # read through their addresses by the scorer
    _ = len(pred)


def check_float_roc() raises:
    var row = msel_scorer_row(msel_scorer_code("roc_auc", True))
    require(row.op == MEMBER_OP_DECISION, "the decision half of 'roc_auc'")
    var y: List[Float32] = [-1.0, 1.0, 1.0, -1.0]
    var score: List[Float32] = [0.1, 0.9, 0.2, 0.3]
    var notes = List[Float64]()
    var value = msel_scorer_value(_route(), entry_ref_unresolved(), List[String](), row, Int(y.unsafe_ptr()),
                                  Int(score.unsafe_ptr()), 4, 0, 1, -1, notes)
    _near(value, 0.75, "roc_auc of an outlier detector's decision")
    var multi: List[Float32] = [0.0, 1.0, 2.0, 1.0]
    var got = _words(row, List[String](), Int(multi.unsafe_ptr()), Int(score.unsafe_ptr()), 4, 0, 1)
    require(got == "multi_class must be in ('ovo', 'ovr')", "multiclass y: " + got)
    var cont: List[Float32] = [0.5, 1.0, 1.0, 0.0]
    got = _words(row, List[String](), Int(cont.unsafe_ptr()), Int(score.unsafe_ptr()), 4, 0, 1)
    require(got == "ValueError: continuous format is not supported", "continuous y: " + got)
    var block: List[Float32] = [0.9, 0.1, 0.2, 0.8, 0.4, 0.6, 0.7, 0.3]
    got = _words(row, List[String](), Int(y.unsafe_ptr()), Int(block.unsafe_ptr()), 4, 0, 2)
    require(got == "ValueError: y should be a 1d array, got an array of shape (4, 2) instead.", "two columns: " + got)
    var one: List[Float32] = [1.0, 1.0, 1.0, 1.0]
    var one_notes = List[Float64]()
    var undefined = msel_scorer_value(_route(), entry_ref_unresolved(), List[String](), row, Int(one.unsafe_ptr()),
                                      Int(score.unsafe_ptr()), 4, 0, 1, -1, one_notes)
    require(undefined != undefined, "one class: nan")
    require(len(one_notes) > 0 and Int(one_notes[0]) == MEMBER_NOTICE_METRIC_ROC_ONE, "one class: its warning")
    _ = len(y)  # read through their addresses by the scorer
    _ = len(score)
    _ = len(multi)
    _ = len(cont)
    _ = len(block)
    _ = len(one)


def test_mf_r9d_scorers() raises:
    def run_case(which: Int) raises:
        if which == 0:
            check_group_words()
        elif which == 1:
            check_tier_refusals()
        else:
            check_float_roc()
    check_cases("mf_r9d_scorers", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
