# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""THE LAST SEARCH SCORER ROWS (lane mf-r9c, 2026-11-23), in a process with no Python, over the metrics program
entry of the host column (x_metrics/descriptor_host.mojo `metrics_program_host_v1`). Each case calls the RUN
function a search side calls (core/msel_scorer_run.mojo `msel_scorer_value`, `msel_scorer_code`, the word group
writer and reader) with the per-score words the search RUN sets (`MselScorer.outs`, `counts_at`, `mheld_at`,
`fold`). Expected words and values are scikit-learn 1.9.1's (utils/_response.py lines 199-236, utils/multiclass.py
`type_of_target`, utils/validation.py line 1083, metrics/_ranking.py `roc_auc_score` / `average_precision_score` /
`top_k_accuracy_score`, metrics/_base.py line 73, metrics/_classification.py lines 178-215 and 3947,
preprocessing/_label.py line 325, metrics/_regression.py line 128), worked by hand from the same inputs:

0. roc_auc (predict_proba) on a multi-output classifier whose outputs hold 2 and 3 classes (a ragged classes_
   list, 'unknown': the probability list reaches the metric), 3 rows: NumPy's "setting an array element with a
   sequence. The requested array has an inhomogeneous shape after 2 dimensions. The detected shape was (2, 3) +
   inhomogeneous part.".
1. the same with 3 classes per output (multiclass-multioutput): "Found array with dim 3, while dim <= 2 is
   required."; top_k_accuracy: "y should be a 1d array, got an array of shape (3, 2) instead.".
2. outputs [0, 1] and [0, 1] (multilabel-indicator classes), y_true [[0, 1], [1, 0], [1, 1]] (multilabel):
   roc_auc and neg_log_loss the doors' NOT IMPLEMENTED words; average_precision with a pos_label other than 1
   (`wmode` 1): "Parameter pos_label is fixed to 1 for multilabel-indicator y_true. ...".
3. outputs [5] and [5] (one class each: 'binary' classes): roc_auc "AttributeError: 'list' object has no
   attribute 'shape'"; average_precision (pos_label 1: `pos_given`) "... no attribute 'tolist'".
4. outputs [0, 1, 2] and [0, 1, 2], the fold holding [0, 1] in each (`mheld_at`, fold 0): multilabel classes;
   y_true codes holding a 2 (multiclass-multioutput): neg_brier_score "Multioutput target data is not supported
   with label binarization"; d2_brier_score on a 1-row side: nan and its warning.
5. the dual 'roc_auc' multi_class rows: `msel_scorer_code("roc_auc(multi_class='ovr')", True)` is a
   decision_function row of ROC_OVR, False its predict_proba row; the micro row's decision half exists; a 3-class
   decision side whose rows do not sum to one raises roc_auc_score's "Target scores need to be probabilities for
   multiclass roc_auc, i.e. they should sum up to 1.0 over classes"; rows that do score as the proba row does.
6. r2 over predict_proba (`resp`, the word group round trip): a binary classifier, labels [0, 1], truth [0, 1, 1,
   0], probabilities [[.8, .2], [.3, .7], [.1, .9], [.6, .4]]: y_pred = column 1 [.2, .7, .9, .4], SS_res .30,
   SS_tot 1.0, r2 0.7; a 3-column block: "y_true and y_pred have different number of output (1!=3)"; a
   non-classifier's 1-D decision [1, 2, 3, 4] against y [1, 2, 3, 4]: 1.0.

FAIL-PROOF EDITS (each must make a case fail): in `_multi_classes` drop the unequal-widths return (case 0 says
dim 3); drop its `w0 == 1` binary returns (case 3 gets the list words); in `_multi_proba_value` drop the top_k
branch (case 1's top_k gets the dim-3 words); in `_multi_classes` ignore `row.mheld_at` (case 4 reads 3-class
outputs: dim-3 words); in `_dual` drop the multi_class name test (case 5's True code is the proba row); in
`_response_regression` drop the binary column pick (case 6 raises "(1!=2)"); in `msel_scorer_group_read` drop the
`resp` read (case 6's row stays predict and maps the probabilities as codes).
"""
from std.testing import TestSuite
from std.memory import bitcast
from checks.native_tests.testsuite_cases import check_cases, require
from bindings.arena_prog_run import MetricRoute
from core.msel_scorer_run import (
    msel_scorer_row, msel_scorer_value, msel_scorer_code, MselScorer, MselScorerOptions, msel_scorer_group_words,
    msel_scorer_group_read, SCORER_LABEL_NUMBER,
)
from core.member_abi import MEMBER_OP_PREDICT_PROBA, MEMBER_OP_DECISION, MEMBER_OP_PREDICT
from core.entry_table import entry_ref_unresolved
from x_metrics.descriptor_host import metrics_program_host_v1

comptime CASES = 7


def _route() -> MetricRoute:
    return MetricRoute(metrics_program_host_v1, False, False)


def _near(got: Float64, want: Float64, what: String) raises:
    require(abs(got - want) <= 1e-6 * max(abs(want), 1.0), what + ": " + String(got) + " expected " + String(want))


def _not_implemented(caller: String) -> String:
    return ("NotImplementedError: mojolearn " + caller + ": y_true is 2-D; multilabel-indicator targets are NOT "
            + "IMPLEMENTED (metrics/NOT_IMPLEMENTED.tsv), only binary and multiclass 1-D labels")


def _number_sels(values: List[Float32]) -> List[Int32]:
    var sels = List[Int32]()
    for i in range(len(values)):
        sels.append(bitcast[DType.int32](values[i]))
        sels.append(Int32(SCORER_LABEL_NUMBER))
    return sels^


def _multi_row(name: String, sels: List[Int32], opts: MselScorerOptions, counts: List[Int]) raises -> MselScorer:
    var row = msel_scorer_row(msel_scorer_code(name, False), opts, Int(sels.unsafe_ptr()))
    row.outs = len(counts)
    row.counts_at = Int(counts.unsafe_ptr())
    return row


def _label_opts(n: Int) -> MselScorerOptions:
    var opts = MselScorerOptions()
    opts.lv_from = 0
    opts.lv_n = n
    return opts


def _words(row: MselScorer, truth: Int, response: Int, count: Int, classes: Int, cols: Int) raises -> String:
    var notes = List[Float64]()
    try:
        _ = msel_scorer_value(_route(), entry_ref_unresolved(), List[String](), row, truth, response, count, classes,
                              cols, -1, notes)
    except e:
        return String(e)
    return String("")


def check_ragged() raises:
    var values: List[Float32] = [0.0, 1.0, 0.0, 1.0, 2.0]
    var sels = _number_sels(values)
    var counts: List[Int] = [2, 3]
    var row = _multi_row("roc_auc", sels, _label_opts(5), counts)
    var truth: List[Int32] = [0, 0, 1, 2, 1, 1]
    var proba = List[Float32](length=15, fill=0.2)
    var got = _words(row, Int(truth.unsafe_ptr()), Int(proba.unsafe_ptr()), 3, 0, 5)
    require(got == "ValueError: setting an array element with a sequence. The requested array has an inhomogeneous "
            + "shape after 2 dimensions. The detected shape was (2, 3) + inhomogeneous part.", "ragged: " + got)
    _ = len(truth)  # read through their addresses by the scorer
    _ = len(proba)
    _ = len(counts)
    _ = len(sels)


def check_equal_blocks() raises:
    var values: List[Float32] = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    var sels = _number_sels(values)
    var counts: List[Int] = [3, 3]
    var truth: List[Int32] = [0, 0, 1, 2, 2, 1]
    var proba = List[Float32](length=18, fill=0.2)
    var roc = _words(_multi_row("roc_auc", sels, _label_opts(6), counts), Int(truth.unsafe_ptr()),
                     Int(proba.unsafe_ptr()), 3, 0, 6)
    require(roc == "ValueError: Found array with dim 3, while dim <= 2 is required.", "equal blocks: " + roc)
    var top = _words(_multi_row("top_k_accuracy", sels, _label_opts(6), counts), Int(truth.unsafe_ptr()),
                     Int(proba.unsafe_ptr()), 3, 0, 6)
    require(top == "ValueError: y should be a 1d array, got an array of shape (3, 2) instead.", "top_k: " + top)
    _ = len(truth)  # read through their addresses by the scorer
    _ = len(proba)
    _ = len(counts)
    _ = len(sels)


def check_multilabel() raises:
    var values: List[Float32] = [0.0, 1.0, 0.0, 1.0]
    var sels = _number_sels(values)
    var counts: List[Int] = [2, 2]
    var truth: List[Int32] = [0, 1, 1, 0, 1, 1]
    var proba = List[Float32](length=12, fill=0.5)
    var roc = _words(_multi_row("roc_auc", sels, _label_opts(4), counts), Int(truth.unsafe_ptr()),
                     Int(proba.unsafe_ptr()), 3, 0, 4)
    require(roc == _not_implemented("roc_auc_score"), "multilabel roc: " + roc)
    var ll = _words(_multi_row("neg_log_loss", sels, _label_opts(4), counts), Int(truth.unsafe_ptr()),
                    Int(proba.unsafe_ptr()), 3, 0, 4)
    require(ll == _not_implemented("log_loss"), "multilabel log_loss: " + ll)
    var ap_opts = _label_opts(4)
    ap_opts.pos_given = True
    ap_opts.wmode = 1
    var ap = _words(_multi_row("average_precision", sels, ap_opts, counts), Int(truth.unsafe_ptr()),
                    Int(proba.unsafe_ptr()), 3, 0, 4)
    require(ap == "ValueError: Parameter pos_label is fixed to 1 for multilabel-indicator y_true. Do not set "
            + "pos_label or set pos_label to 1.", "multilabel AP: " + ap)
    _ = len(truth)  # read through their addresses by the scorer
    _ = len(proba)
    _ = len(counts)
    _ = len(sels)


def check_binary_classes() raises:
    var values: List[Float32] = [5.0, 5.0]
    var sels = _number_sels(values)
    var counts: List[Int] = [1, 1]
    var truth: List[Int32] = [0, 0, 0, 0]
    var proba = List[Float32](length=4, fill=1.0)
    var roc = _words(_multi_row("roc_auc", sels, _label_opts(2), counts), Int(truth.unsafe_ptr()),
                     Int(proba.unsafe_ptr()), 2, 0, 2)
    require(roc == "AttributeError: 'list' object has no attribute 'shape'", "binary classes roc: " + roc)
    var ap_opts = _label_opts(2)
    ap_opts.pos_given = True
    var ap = _words(_multi_row("average_precision", sels, ap_opts, counts), Int(truth.unsafe_ptr()),
                    Int(proba.unsafe_ptr()), 2, 0, 2)
    require(ap == "AttributeError: 'list' object has no attribute 'tolist'", "binary classes AP: " + ap)
    _ = len(truth)  # read through their addresses by the scorer
    _ = len(proba)
    _ = len(counts)
    _ = len(sels)


def check_fold_classes() raises:
    var values: List[Float32] = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    var sels = _number_sels(values)
    var counts: List[Int] = [3, 3]
    var held = List[List[Int32]]()
    var h0: List[Int32] = [0, 1]
    var h1: List[Int32] = [0, 1]
    held.append(h0^)
    held.append(h1^)
    var truth: List[Int32] = [0, 2, 1, 0]
    var proba = List[Float32](length=8, fill=0.5)
    var brier = _multi_row("neg_brier_score", sels, _label_opts(6), counts)
    brier.mheld_at = Int(held.unsafe_ptr())
    brier.fold = 0
    var got = _words(brier, Int(truth.unsafe_ptr()), Int(proba.unsafe_ptr()), 2, 0, 4)
    require(got == "ValueError: Multioutput target data is not supported with label binarization", "fold brier: " + got)
    var d2 = _multi_row("d2_brier_score", sels, _label_opts(6), counts)
    d2.mheld_at = Int(held.unsafe_ptr())
    d2.fold = 0
    var notes = List[Float64]()
    var value = msel_scorer_value(_route(), entry_ref_unresolved(), List[String](), d2, Int(truth.unsafe_ptr()),
                                  Int(proba.unsafe_ptr()), 1, 0, 4, -1, notes)
    require(value != value, "one-row D^2 brier is nan: " + String(value))
    require(len(notes) > 0, "one-row D^2 brier warns")
    _ = len(truth)  # read through their addresses by the scorer
    _ = len(proba)
    _ = len(counts)
    _ = len(held)
    _ = len(sels)


def check_multi_class_rows() raises:
    var dec = msel_scorer_code("roc_auc(multi_class='ovr')", True)
    var pro = msel_scorer_code("roc_auc(multi_class='ovr')", False)
    require(dec >= 0 and pro >= 0 and dec != pro, "multi_class row codes")
    require(msel_scorer_row(dec).op == MEMBER_OP_DECISION, "ovr decision row")
    require(msel_scorer_row(pro).op == MEMBER_OP_PREDICT_PROBA, "ovr proba row")
    var micro = msel_scorer_code("roc_auc(multi_class='ovr', average='micro')", True)
    require(micro >= 0 and msel_scorer_row(micro).op == MEMBER_OP_DECISION, "micro decision row")
    var truth: List[Int32] = [0, 1, 2, 0, 1, 2]
    var raw: List[Float32] = [2.0, -1.0, 0.5, -1.0, 2.0, 0.5, 0.1, 0.2, 3.0, 1.5, 0.0, -0.5, 0.0, 1.0, 0.0,
                              -1.0, 0.0, 2.0]
    var notes = List[Float64]()
    var words = String("")
    try:
        _ = msel_scorer_value(_route(), entry_ref_unresolved(), List[String](), msel_scorer_row(dec),
                              Int(truth.unsafe_ptr()), Int(raw.unsafe_ptr()), 6, 3, 3, -1, notes)
    except e:
        words = String(e)
    require(words.find("Target scores need to be probabilities for multiclass roc_auc, i.e. they should sum up to "
                       + "1.0 over classes") >= 0, "decision rows not summing to one: " + words)
    var probs: List[Float32] = [0.7, 0.2, 0.1, 0.1, 0.8, 0.1, 0.2, 0.2, 0.6, 0.5, 0.3, 0.2, 0.3, 0.4, 0.3,
                                0.1, 0.3, 0.6]
    var a = msel_scorer_value(_route(), entry_ref_unresolved(), List[String](), msel_scorer_row(dec),
                              Int(truth.unsafe_ptr()), Int(probs.unsafe_ptr()), 6, 3, 3, -1, notes)
    var b = msel_scorer_value(_route(), entry_ref_unresolved(), List[String](), msel_scorer_row(pro),
                              Int(truth.unsafe_ptr()), Int(probs.unsafe_ptr()), 6, 3, 3, -1, notes)
    _near(a, b, "decision row scores as the proba row")
    _ = len(truth)  # read through their addresses by the scorer
    _ = len(raw)
    _ = len(probs)


def check_response_regression() raises:
    var values: List[Float32] = [0.0, 1.0]
    var label_sels = _number_sels(values)
    var opts = _label_opts(2)
    opts.resp = MEMBER_OP_PREDICT_PROBA
    var words = List[Float64]()
    var src = label_sels.copy()
    msel_scorer_group_words(words, msel_scorer_code("r2", False), -1, opts, src)
    var code = 0
    var positive = 0
    var back = MselScorerOptions()
    var sels = List[Int32]()
    _ = msel_scorer_group_read(words, 0, code, positive, back, sels)
    require(back.resp == MEMBER_OP_PREDICT_PROBA, "the resp word round trip")
    var row = msel_scorer_row(code, back, Int(sels.unsafe_ptr()))
    require(row.op == MEMBER_OP_PREDICT_PROBA, "the regression row reads predict_proba")
    var truth: List[Int32] = [0, 1, 1, 0]
    var proba: List[Float32] = [0.8, 0.2, 0.3, 0.7, 0.1, 0.9, 0.6, 0.4]
    var notes = List[Float64]()
    var value = msel_scorer_value(_route(), entry_ref_unresolved(), List[String](), row, Int(truth.unsafe_ptr()),
                                  Int(proba.unsafe_ptr()), 4, 2, 2, -1, notes)
    _near(value, 0.7, "r2 over classes[-1]'s column")
    var three = List[Float32](length=12, fill=0.25)
    var three_values: List[Float32] = [0.0, 1.0, 2.0]
    var three_label = _number_sels(three_values)
    var row3 = msel_scorer_row(code, _label_opts(3), Int(three_label.unsafe_ptr()))
    row3.op = MEMBER_OP_PREDICT_PROBA
    var got = _words(row3, Int(truth.unsafe_ptr()), Int(three.unsafe_ptr()), 4, 3, 3)
    require(got == "ValueError: y_true and y_pred have different number of output (1!=3)", "3 columns: " + got)
    var plain = msel_scorer_row(code)
    plain.op = MEMBER_OP_DECISION
    var y: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var dv = msel_scorer_value(_route(), entry_ref_unresolved(), List[String](), plain, Int(y.unsafe_ptr()),
                               Int(y.unsafe_ptr()), 4, 0, 1, -1, notes)
    _near(dv, 1.0, "r2 over a non-classifier's decision")
    require(msel_scorer_row(code).op == MEMBER_OP_PREDICT, "the table row stays predict")
    _ = len(truth)  # read through their addresses by the scorer
    _ = len(proba)
    _ = len(three)
    _ = len(three_label)
    _ = len(y)
    _ = len(sels)
    _ = len(label_sels)


def test_mf_r9c_scorers() raises:
    def run_case(which: Int) raises:
        if which == 0:
            check_ragged()
        elif which == 1:
            check_equal_blocks()
        elif which == 2:
            check_multilabel()
        elif which == 3:
            check_binary_classes()
        elif which == 4:
            check_fold_classes()
        elif which == 5:
            check_multi_class_rows()
        else:
            check_response_regression()
    check_cases("mf_r9c_scorers", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
