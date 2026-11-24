# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""THE METRIC ROWS (lane mf-fitadm7, 2026-11-23), in a process with no Python, over the metrics program entry of
the host column (x_metrics/descriptor_host.mojo `metrics_program_host_v1`): a native make_scorer over
zero_one_loss (normalize True / False) or hamming_loss is a typed search scorer row (core/msel_scorer_run.mojo
`msel_scorer_metric_code`), scored by bindings/mcls_classify_run.mojo `label_count_run` (the program
bindings/metrics_exp_cls_door.mojo `_cm_count` runs: the sums over the present labels, then the COUNT tail),
against scikit-learn's definitions computed here from the same codes:

1. Unweighted, t [0, 1, 2, 2, 1, 0, 2], p [0, 2, 2, 2, 1, 1, 0] (three rows differ): zero_one_loss 3/7,
   zero_one_loss(normalize=False) 3.0, hamming_loss 3/7; the make_scorer's greater_is_better=False (`flip`)
   negates each.
2. Weighted, w [1, 2, 1, 1, 1, 3, 0.5] (the weights' float32 bits as the UNPACK appends them): the
   misclassified weight 5.5 of 9.5: zero_one_loss 5.5 / 9.5, normalize=False 5.5, hamming_loss 5.5 / 9.5.
3. The lookup and the words: `msel_scorer_metric_code` answers three distinct codes and -1 for a metric no row
   serves (cohen_kappa_score); `msel_scorer_caller` names "zero_one_loss" and "hamming_loss"; a sample_weight
   of the wrong length raises "mojolearn zero_one_loss: sample_weight has 3 entries for 7 samples" (the doors'
   `_reg_weights` words through `mx_weights`).

Every value must match to a relative 1e-6.

FAIL-PROOF EDITS (each must make a case fail): in `label_count_run` read `out + 4` for CNT_LOSS (case 1, the
fraction); pass `False` to `mc_sums_stages` there (case 2, the weights ignored); in `msel_scorer_metric_code`
map normalize False to "zero_one_loss" (case 1, the count); in `msel_scorer_caller` drop the COUNT branch (case
3, the words).
"""
from std.memory import bitcast
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from bindings.arena_prog_run import MetricRoute
from core.msel_scorer_run import (
    msel_scorer_row, msel_scorer_value, msel_scorer_metric_code, msel_scorer_caller, MselScorer, MselScorerOptions,
)
from core.entry_table import entry_ref_unresolved
from x_metrics.descriptor_host import metrics_program_host_v1

comptime CASES = 3


def _route() -> MetricRoute:
    return MetricRoute(metrics_program_host_v1, False, False)


def _near(got: Float64, want: Float64, what: String) raises:
    var tol = 1e-6 * max(abs(want), 1.0)
    require(abs(got - want) <= tol, what + ": " + String(got) + " expected " + String(want))


def _row(metric: String, normalize: Bool, flip: Bool, sels: List[Int32], n_w: Int) raises -> MselScorer:
    var code = msel_scorer_metric_code(metric, normalize)
    require(code >= 2, metric + " has a metric row")
    var opts = MselScorerOptions()
    opts.flip = flip
    opts.w_from = 0
    opts.w_n = n_w
    return msel_scorer_row(code, opts, Int(sels.unsafe_ptr()))


def _value(row: MselScorer, truth: Int, response: Int, count: Int, classes: Int) raises -> Float64:
    var notes = List[Float64]()
    return msel_scorer_value(_route(), entry_ref_unresolved(), List[String](), row, truth, response, count, classes,
                             1, -1, notes)


def _fixture(mut t: List[Int32], mut p: List[Int32]):
    t = [0, 1, 2, 2, 1, 0, 2]
    p = [0, 2, 2, 2, 1, 1, 0]


def check_unweighted() raises:
    var t = List[Int32]()
    var p = List[Int32]()
    _fixture(t, p)
    var none = List[Int32]()
    var ta = Int(t.unsafe_ptr())
    var pa = Int(p.unsafe_ptr())
    _near(_value(_row("zero_one_loss", True, False, none, 0), ta, pa, 7, 3), 3.0 / 7.0, "zero_one_loss")
    _near(_value(_row("zero_one_loss", False, False, none, 0), ta, pa, 7, 3), 3.0, "zero_one_loss(normalize=False)")
    _near(_value(_row("hamming_loss", True, False, none, 0), ta, pa, 7, 3), 3.0 / 7.0, "hamming_loss")
    _near(_value(_row("zero_one_loss", True, True, none, 0), ta, pa, 7, 3), -3.0 / 7.0, "flipped zero_one_loss")
    _near(_value(_row("hamming_loss", True, True, none, 0), ta, pa, 7, 3), -3.0 / 7.0, "flipped hamming_loss")
    _ = len(t)
    _ = len(p)
    _ = len(none)


def check_weighted() raises:
    var t = List[Int32]()
    var p = List[Int32]()
    _fixture(t, p)
    var w: List[Float32] = [1.0, 2.0, 1.0, 1.0, 1.0, 3.0, 0.5]
    var sels = List[Int32]()
    for i in range(len(w)):  # small-loop(fixture: the weights' bits)
        sels.append(bitcast[DType.int32](w[i]))
    var ta = Int(t.unsafe_ptr())
    var pa = Int(p.unsafe_ptr())
    _near(_value(_row("zero_one_loss", True, False, sels, 7), ta, pa, 7, 3), 5.5 / 9.5, "weighted zero_one_loss")
    _near(_value(_row("zero_one_loss", False, False, sels, 7), ta, pa, 7, 3), 5.5, "weighted count")
    _near(_value(_row("hamming_loss", True, False, sels, 7), ta, pa, 7, 3), 5.5 / 9.5, "weighted hamming_loss")
    _ = len(t)
    _ = len(p)
    _ = len(sels)


def check_lookup_words() raises:
    var a = msel_scorer_metric_code("zero_one_loss", True)
    var c = msel_scorer_metric_code("zero_one_loss", False)
    var h = msel_scorer_metric_code("hamming_loss", True)
    require(a >= 2 and c >= 2 and h >= 2 and a != c and a != h and c != h, "three distinct metric rows")
    require(msel_scorer_metric_code("cohen_kappa_score", True) == -1, "no row for cohen_kappa_score")
    require(msel_scorer_caller(msel_scorer_row(a)) == "zero_one_loss", "zero_one_loss caller")
    require(msel_scorer_caller(msel_scorer_row(h)) == "hamming_loss", "hamming_loss caller")
    var t = List[Int32]()
    var p = List[Int32]()
    _fixture(t, p)
    var w: List[Float32] = [1.0, 1.0, 1.0]
    var sels = List[Int32]()
    for i in range(len(w)):  # small-loop(fixture: the weights' bits)
        sels.append(bitcast[DType.int32](w[i]))
    var words = String("")
    try:
        _ = _value(_row("zero_one_loss", True, False, sels, 3), Int(t.unsafe_ptr()), Int(p.unsafe_ptr()), 7, 3)
    except e:
        words = String(e)
    require(words == "mojolearn zero_one_loss: sample_weight has 3 entries for 7 samples", "length words: " + words)
    _ = len(t)
    _ = len(p)
    _ = len(sels)


def test_mf_fitadm7_count_scorers() raises:
    def run_case(which: Int) raises:
        if which == 0:
            check_unweighted()
        elif which == 1:
            check_weighted()
        else:
            check_lookup_words()
    check_cases("mf_fitadm7_count_scorers", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
