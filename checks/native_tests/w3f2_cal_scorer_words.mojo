# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""THE PRINTED y_true AND THE y_prob RANGE OF LANE w3f2-cal (2026-11-24), in a process with no Python, over the
metrics program entry of the host column (x_metrics/descriptor_host.mojo `metrics_program_host_v1`). Each case calls
the RUN function a search side calls (core/msel_scorer_run.mojo `msel_scorer_value`) with the per-score words the
search RUN sets (`MselScorer.given`, `given_dtype`, `side_rows`). Expected words are scikit-learn 1.9.1's
(metrics/_classification.py `_validate_multiclass_probabilistic_prediction` lines 272-275,
`_validate_binary_probabilistic_prediction` lines 3675-3678, `_one_hot_encoding_multiclass_target`;
utils/multiclass.py line 111) and NumPy 2's default array repr, worked by hand:

0. neg_log_loss on a NON-classifier, probabilities [[0.5, 0.5], [-0.25, 1.25], ...]: "y_prob contains values greater
   than 1: 1.25" (the largest first); with every word in [0, 1] but one -0.25 (and no word above 1): "... lower than
   0: -0.25"; brier over one column [0.2, -0.5, 0.7, 0.4] with binary y: "... less than 0: -0.5". The range checks
   come before the target checks: a continuous y with an out-of-range word still gets the range words.
1. neg_log_loss, a continuous y GIVEN as float64 [0.5, 1.25, 2.0, 0.5] at side rows [0, 1, 2, 3] of the given y
   (the float32 targets the same): "ValueError: Unknown label type: (array([0.5 , 1.25, 2.  , 0.5 ]),)"; NOT given
   (the float32 targets are y): "... (array([0.5 , 1.25, 2.  , 0.5 ], dtype=float32),)"; side rows [2, 0] of a
   given y [0.5, 1e-05, 3.0]: only the side's rows print, "(array([3. , 0.5]),)".
2. an integer y GIVEN as int64: one label prints "(1)" not "(1.0)"; three classes against two columns print
   "Classes found in y_true: [0 1 2]".

FAIL-PROOF EDITS (each must make a case fail): drop `_y_prob_range` in the log_loss branch of `_float_proba_value`
(case 0 gets the class-count words) or swap its two tests (case 0's first words change); print "lower" in the binary
branch (case 0, brier); read the float32 targets when given (case 1's first line gains the dtype); drop the
`_np_float_elements` padding (case 1 prints "0.5, 1.25, 2., 0.5"); drop `_given_int` (case 2 prints "1.0").
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from bindings.arena_prog_run import MetricRoute
from core.msel_scorer_run import msel_scorer_row, msel_scorer_value, msel_scorer_code, MselScorer
from core.member_abi import MEMBER_F64, MEMBER_I64
from core.entry_table import entry_ref_unresolved
from x_metrics.descriptor_host import metrics_program_host_v1

comptime CASES = 3


def _route() -> MetricRoute:
    return MetricRoute(metrics_program_host_v1, False, False)


def _words(row: MselScorer, truth: Int, response: Int, count: Int, cols: Int) raises -> String:
    var notes = List[Float64]()
    try:
        _ = msel_scorer_value(_route(), entry_ref_unresolved(), List[String](), row, truth, response, count, 0, cols,
                              -1, notes)
    except e:
        return String(e)
    return String("")


def check_range() raises:
    var log_row = msel_scorer_row(msel_scorer_code("neg_log_loss", False))
    var y: List[Float32] = [0.0, 1.0, 0.0, 1.0]
    var high: List[Float32] = [0.5, 0.5, -0.25, 1.25, 0.3, 0.7, 0.6, 0.4]
    var got = _words(log_row, Int(y.unsafe_ptr()), Int(high.unsafe_ptr()), 4, 2)
    require(got == "ValueError: y_prob contains values greater than 1: 1.25", "log_loss high: " + got)
    var low: List[Float32] = [0.5, 0.5, -0.25, 1.0, 0.3, 0.7, 0.6, 0.4]
    got = _words(log_row, Int(y.unsafe_ptr()), Int(low.unsafe_ptr()), 4, 2)
    require(got == "ValueError: y_prob contains values lower than 0: -0.25", "log_loss low: " + got)
    var cont: List[Float32] = [0.5, 1.25, 2.0, 0.5]
    got = _words(log_row, Int(cont.unsafe_ptr()), Int(high.unsafe_ptr()), 4, 2)
    require(got == "ValueError: y_prob contains values greater than 1: 1.25", "log_loss range first: " + got)
    var brier = msel_scorer_row(msel_scorer_code("neg_brier_score", False))
    var column: List[Float32] = [0.2, -0.5, 0.7, 0.4]
    got = _words(brier, Int(y.unsafe_ptr()), Int(column.unsafe_ptr()), 4, 1)
    require(got == "ValueError: y_prob contains values less than 0: -0.5", "brier low: " + got)
    _ = len(y)  # read through their addresses by the scorer
    _ = len(high)
    _ = len(low)
    _ = len(cont)
    _ = len(column)


def check_continuous() raises:
    var log_row = msel_scorer_row(msel_scorer_code("neg_log_loss", False))
    var cont: List[Float32] = [0.5, 1.25, 2.0, 0.5]
    var given: List[Float64] = [0.5, 1.25, 2.0, 0.5]
    var rows: List[Int32] = [0, 1, 2, 3]
    var proba: List[Float32] = [0.5, 0.5, 0.25, 0.75, 0.3, 0.7, 0.6, 0.4]
    var row = log_row
    row.given = Int(given.unsafe_ptr())
    row.given_dtype = MEMBER_F64
    row.side_rows = Int(rows.unsafe_ptr())
    var got = _words(row, Int(cont.unsafe_ptr()), Int(proba.unsafe_ptr()), 4, 2)
    require(got == "ValueError: Unknown label type: (array([0.5 , 1.25, 2.  , 0.5 ]),)", "given float64: " + got)
    got = _words(log_row, Int(cont.unsafe_ptr()), Int(proba.unsafe_ptr()), 4, 2)
    require(got == "ValueError: Unknown label type: (array([0.5 , 1.25, 2.  , 0.5 ], dtype=float32),)",
            "float32 y: " + got)
    var wide: List[Float64] = [0.5, 1.0e-05, 3.0]
    var picked: List[Int32] = [2, 0]
    var side: List[Float32] = [3.0, 0.5]
    row.given = Int(wide.unsafe_ptr())
    row.side_rows = Int(picked.unsafe_ptr())
    got = _words(row, Int(side.unsafe_ptr()), Int(proba.unsafe_ptr()), 2, 2)
    require(got == "ValueError: Unknown label type: (array([3. , 0.5]),)", "side rows: " + got)
    _ = len(cont)  # read through their addresses by the scorer
    _ = len(given)
    _ = len(rows)
    _ = len(proba)
    _ = len(wide)
    _ = len(picked)
    _ = len(side)


def check_integer_labels() raises:
    var log_row = msel_scorer_row(msel_scorer_code("neg_log_loss", False))
    var given: List[Int64] = [1, 1, 1, 1]
    var rows: List[Int32] = [0, 1, 2, 3]
    var one: List[Float32] = [1.0, 1.0, 1.0, 1.0]
    var two: List[Float32] = [0.8, 0.2, 0.3, 0.7, 0.1, 0.9, 0.6, 0.4]
    var row = log_row
    row.given = Int(given.unsafe_ptr())
    row.given_dtype = MEMBER_I64
    row.side_rows = Int(rows.unsafe_ptr())
    var got = _words(row, Int(one.unsafe_ptr()), Int(two.unsafe_ptr()), 4, 2)
    require(got == "ValueError: y_true contains only one label (1). Please provide the list of all expected class "
            + "labels explicitly through the labels argument.", "int one label: " + got)
    var multi: List[Float32] = [0.0, 1.0, 2.0, 1.0]
    got = _words(row, Int(multi.unsafe_ptr()), Int(two.unsafe_ptr()), 4, 2)
    require(got == "ValueError: y_true and y_prob contain different number of classes: 3 vs 2. Please provide the "
            + "true labels explicitly through the labels argument. Classes found in y_true: [0 1 2]",
            "int classes: " + got)
    _ = len(given)  # read through their addresses by the scorer
    _ = len(rows)
    _ = len(one)
    _ = len(two)
    _ = len(multi)


def test_w3f2_cal_scorer_words() raises:
    def run_case(which: Int) raises:
        if which == 0:
            check_range()
        elif which == 1:
            check_continuous()
        else:
            check_integer_labels()
    check_cases("w3f2_cal_scorer_words", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
