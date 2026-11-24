# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""THE MULTI-OUTPUT SEARCH SCORER ROWS (lane mf-rows8b, 2026-11-23), in a process with no Python, over the metrics
program entry of the host column (x_metrics/descriptor_host.mojo `metrics_program_host_v1`). Each case calls the
RUN function the search's sides call (core/msel_scorer_run.mojo `msel_scorer_value`, `msel_multi_output_type`),
with the per-score words the search RUN sets (`MselScorer.outs`, `counts_at`, `fpred`); the expected words and
values are scikit-learn 1.9's (metrics/_classification.py `_check_targets` lines 116-145, confusion_matrix line 555,
matthews_corrcoef line 1319; metrics/cluster/_supervised.py line 72; utils/multiclass.py `type_of_target`), worked
by hand from the same inputs:

0. accuracy on a multi-output regressor's side, y [[0.5, 1], [2, 3], [1, 1.5]] and p [[0.5, 1], [2, 3], [1, 1.5]]:
   "ValueError: continuous-multioutput is not supported".
1. y [[0, 1], [2, 0], [1, 2]] (integral: multiclass-multioutput), p continuous: "ValueError: Classification
   metrics can't handle a mix of multiclass-multioutput and continuous-multioutput targets".
2. y = p = [[0, 1], [1, 0], [1, 1]] (multilabel-indicator): accuracy raises the label doors' NOT IMPLEMENTED words;
   matthews_corrcoef raises "ValueError: multilabel-indicator is not supported".
3. adjusted_rand_score on a 3-row, 2-output side: "ValueError: labels_true must be 1D: shape is (3, 2)".
4. r2 on a multi-output classifier (output labels [10, 20] and [1, 2, 3]): truth codes [[0, 0], [1, 2]] (values
   [[10, 1], [20, 3]]), predicted [[1, 0], [1, 1]] (values [[20, 1], [20, 2]]): output 0 r2 = 1 - 100 / 50 = -1,
   output 1 r2 = 1 - 1 / 2 = 0.5, uniform average -0.25. Truth = predicted scores 1.0.
5. r2 on a multi-output classifier whose label 'x' does not convert: "ValueError: could not convert string to
   float: 'x'" (its value word names label word 0).
6. adjusted_rand_score on a regressor's 1-D side whose predict is float32 (`fpred` 1): the public cluster doors'
   words "mojolearn metrics: labels_pred must be an integer label array, got buffer format 'f'; the ported
   kernels take int32 labels"; float64 (`fpred` 2) names 'd'.
7. `msel_multi_output_type`: [0.5, 1, 2, 3] continuous-multioutput; [1, 2, 1, 2] multilabel-indicator;
   [0, 1, 2, 0] multiclass-multioutput.

FAIL-PROOF EDITS (each must make a case fail): in `_multi_type` drop the `distinct < 3` multilabel return (cases 2,
7); in `_multi_output_refusal` drop the mix check (case 1 then says "not supported"); in `_multi_floats` leave every
offset 0 (case 4 reads output 1's codes as output 0's labels); drop the `row.fpred` check in `_cluster_value`
(case 6 scores); in `msel_scorer_value` move the `row.outs` check after `_float_label_value` (cases 0-2 score).
"""
from std.testing import TestSuite
from std.memory import bitcast
from checks.native_tests.testsuite_cases import check_cases, require
from bindings.arena_prog_run import MetricRoute
from core.msel_scorer_run import (
    msel_scorer_row, msel_scorer_value, msel_scorer_code, MselScorer, MselScorerOptions, msel_multi_output_type,
    SCORER_LABEL_NUMBER, SCORER_LABEL_UNREAD,
)
from core.entry_table import entry_ref_unresolved
from x_metrics.descriptor_host import metrics_program_host_v1

comptime CASES = 8
comptime _MULTILABEL_WORDS = ("NotImplementedError: mojolearn accuracy_score: y_true is 2-D; multilabel-indicator "
                              + "targets are NOT IMPLEMENTED (metrics/NOT_IMPLEMENTED.tsv), only binary and "
                              + "multiclass 1-D labels")


def _route() -> MetricRoute:
    return MetricRoute(metrics_program_host_v1, False, False)


def _near(got: Float64, want: Float64, what: String) raises:
    require(abs(got - want) <= 1e-6 * max(abs(want), 1.0), what + ": " + String(got) + " expected " + String(want))


def _row(name: String, outs: Int, sels: List[Int32], opts: MselScorerOptions) raises -> MselScorer:
    var row = msel_scorer_row(msel_scorer_code(name, False), opts, Int(sels.unsafe_ptr()))
    row.outs = outs
    return row


def _words(row: MselScorer, label_words: List[String], truth: Int, response: Int, count: Int,
           cols: Int) raises -> String:
    var notes = List[Float64]()
    try:
        _ = msel_scorer_value(_route(), entry_ref_unresolved(), label_words, row, truth, response, count, 0, cols, -1,
                              notes)
    except e:
        return String(e)
    return String("")


def check_continuous() raises:
    var none = List[Int32]()
    var y: List[Float32] = [0.5, 1.0, 2.0, 3.0, 1.0, 1.5]
    var got = _words(_row("accuracy", 2, none, MselScorerOptions()), List[String](), Int(y.unsafe_ptr()),
                     Int(y.unsafe_ptr()), 3, 2)
    require(got == "ValueError: continuous-multioutput is not supported", "continuous: " + got)
    _ = len(y)  # read through its address by the scorer


def check_mix() raises:
    var none = List[Int32]()
    var y: List[Float32] = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    var p: List[Float32] = [0.5, 1.0, 2.0, 0.0, 1.0, 2.0]
    var got = _words(_row("accuracy", 2, none, MselScorerOptions()), List[String](), Int(y.unsafe_ptr()),
                     Int(p.unsafe_ptr()), 3, 2)
    require(got == "ValueError: Classification metrics can't handle a mix of multiclass-multioutput and "
            + "continuous-multioutput targets", "mix: " + got)
    _ = len(y)  # read through their addresses by the scorer
    _ = len(p)


def check_multilabel() raises:
    var none = List[Int32]()
    var y: List[Float32] = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    var acc = _words(_row("accuracy", 2, none, MselScorerOptions()), List[String](), Int(y.unsafe_ptr()),
                     Int(y.unsafe_ptr()), 3, 2)
    require(acc == _MULTILABEL_WORDS, "multilabel accuracy: " + acc)
    var mcc = _words(_row("matthews_corrcoef", 2, none, MselScorerOptions()), List[String](), Int(y.unsafe_ptr()),
                     Int(y.unsafe_ptr()), 3, 2)
    require(mcc == "ValueError: multilabel-indicator is not supported", "multilabel matthews: " + mcc)
    _ = len(y)  # read through its address by the scorer


def check_cluster_2d() raises:
    var none = List[Int32]()
    var y: List[Float32] = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    var got = _words(_row("adjusted_rand_score", 2, none, MselScorerOptions()), List[String](), Int(y.unsafe_ptr()),
                     Int(y.unsafe_ptr()), 3, 2)
    require(got == "ValueError: labels_true must be 1D: shape is (3, 2)", "cluster 2-D: " + got)
    _ = len(y)  # read through its address by the scorer


def _label_sels(values: List[Float32], kinds: List[Int]) -> List[Int32]:
    var sels = List[Int32]()
    for i in range(len(values)):
        if kinds[i] == SCORER_LABEL_UNREAD:
            sels.append(Int32(Int(values[i])))  # the value word names its label word
        else:
            sels.append(bitcast[DType.int32](values[i]))
        sels.append(Int32(kinds[i]))
    return sels^


def check_classifier_r2() raises:
    var values: List[Float32] = [10.0, 20.0, 1.0, 2.0, 3.0]
    var kinds: List[Int] = [SCORER_LABEL_NUMBER, SCORER_LABEL_NUMBER, SCORER_LABEL_NUMBER, SCORER_LABEL_NUMBER,
                            SCORER_LABEL_NUMBER]
    var sels = _label_sels(values, kinds)
    var counts: List[Int] = [2, 3]
    var opts = MselScorerOptions()
    opts.lv_from = 0
    opts.lv_n = 5
    var row = _row("r2", 2, sels, opts)
    row.counts_at = Int(counts.unsafe_ptr())
    var truth: List[Int32] = [0, 0, 1, 2]
    var pred: List[Int32] = [1, 0, 1, 1]
    var notes = List[Float64]()
    var value = msel_scorer_value(_route(), entry_ref_unresolved(), List[String](), row, Int(truth.unsafe_ptr()),
                                  Int(pred.unsafe_ptr()), 2, 0, 2, -1, notes)
    _near(value, -0.25, "multi-output classifier r2")
    var same = msel_scorer_value(_route(), entry_ref_unresolved(), List[String](), row, Int(truth.unsafe_ptr()),
                                 Int(truth.unsafe_ptr()), 2, 0, 2, -1, notes)
    _near(same, 1.0, "multi-output classifier r2, truth = predicted")
    _ = len(truth)  # read through their addresses by the scorer
    _ = len(pred)
    _ = len(counts)
    _ = len(sels)


def check_classifier_unread() raises:
    var values: List[Float32] = [0.0, 1.0, 1.0]
    var kinds: List[Int] = [SCORER_LABEL_UNREAD, SCORER_LABEL_NUMBER, SCORER_LABEL_NUMBER]
    var sels = _label_sels(values, kinds)
    var counts: List[Int] = [2, 1]
    var opts = MselScorerOptions()
    opts.lv_from = 0
    opts.lv_n = 3
    var row = _row("r2", 2, sels, opts)
    row.counts_at = Int(counts.unsafe_ptr())
    var words: List[String] = ["'x'"]
    var truth: List[Int32] = [0, 0, 1, 0]
    var got = _words(row, words, Int(truth.unsafe_ptr()), Int(truth.unsafe_ptr()), 2, 2)
    require(got == "ValueError: could not convert string to float: 'x'", "unread label: " + got)
    _ = len(truth)  # read through its address by the scorer
    _ = len(counts)
    _ = len(sels)


def check_float_clusters() raises:
    var none = List[Int32]()
    var y: List[Float32] = [0.0, 1.0, 1.0]
    var p: List[Float32] = [0.5, 1.5, 1.5]
    var row = _row("adjusted_rand_score", 0, none, MselScorerOptions())
    row.fpred = 1
    var f = _words(row, List[String](), Int(y.unsafe_ptr()), Int(p.unsafe_ptr()), 3, 1)
    require(f == "mojolearn metrics: labels_pred must be an integer label array, got buffer format 'f'; the ported "
            + "kernels take int32 labels", "float32 clusters: " + f)
    row.fpred = 2
    var d = _words(row, List[String](), Int(y.unsafe_ptr()), Int(p.unsafe_ptr()), 3, 1)
    require(d.find("buffer format 'd'") >= 0, "float64 clusters: " + d)
    _ = len(y)  # read through their addresses by the scorer
    _ = len(p)


def check_types() raises:
    var c: List[Float32] = [0.5, 1.0, 2.0, 3.0]
    var m: List[Float32] = [1.0, 2.0, 1.0, 2.0]
    var k: List[Float32] = [0.0, 1.0, 2.0, 0.0]
    require(msel_multi_output_type(Int(c.unsafe_ptr()), 4) == "continuous-multioutput", "continuous type")
    require(msel_multi_output_type(Int(m.unsafe_ptr()), 4) == "multilabel-indicator", "multilabel type")
    require(msel_multi_output_type(Int(k.unsafe_ptr()), 4) == "multiclass-multioutput", "multiclass type")
    _ = len(c)  # read through their addresses by the type scan
    _ = len(m)
    _ = len(k)


def test_mf_rows8b_multi_output() raises:
    def run_case(which: Int) raises:
        if which == 0:
            check_continuous()
        elif which == 1:
            check_mix()
        elif which == 2:
            check_multilabel()
        elif which == 3:
            check_cluster_2d()
        elif which == 4:
            check_classifier_r2()
        elif which == 5:
            check_classifier_unread()
        elif which == 6:
            check_float_clusters()
        else:
            check_types()
    check_cases("mf_rows8b_multi_output", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
