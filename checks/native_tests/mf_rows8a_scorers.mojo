# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""THE LAST SEARCH SCORER ROWS, TYPED (lane mf-rows8a, 2026-11-23), in a process with no Python, over the metrics
program entry of the host column (x_metrics/descriptor_host.mojo `metrics_program_host_v1`). Each case calls the
RUN function the search's sides call (core/msel_scorer_run.mojo `msel_scorer_value`; core/msel_search_run.mojo
`_classes_words`); the expected values and words are scikit-learn 1.9's, worked by hand from the same inputs:

0. f1 (average 'binary') on a regressor's side, pos_label 1 (its value word "1"): y [0, 1, 1, 0], p [0, 1, 0, 0]
   (float32, integral): tp 1, fp 0, fn 1, so precision 1, recall 0.5, f1 = 2/3.
1. pos_label 1 not among two present labels: y [0, 2, 2, 0], p [0, 2, 0, 0] raises
   "ValueError: pos_label=1 is not a valid label. It should be one of [0. 2.]" (`_check_set_wise_labels`'
   f-string over NumPy's float array).
2. Three labels: y [0, 1, 2], p [0, 1, 2] raises "ValueError: Target is multiclass but average='binary'. ...".
3. One label present and pos_label absent: y = p = [0, 0]: recall scored 0.0 as `labels=[pos_label]` scores it.
4. A pos_label the door reads as no label (`pos_float`) raises SCORER_FLOAT_POS_WORDS; a string pos_label
   (SCORER_POSITIVE_KIND) raises "pos_label must use the same type as the targets".
5. `_numpy_float_words`: [-1, 2] "[-1.  2.]"; [0, 10] "[ 0. 10.]"; [1, 2000] "[1.e+00 2.e+03]"; [1, 1500]
   "[1.0e+00 1.5e+03]"; [0, 1e8] "[0.e+00 1.e+08]" (numpy/_core/arrayprint.py `FloatingFormat.fillFormat`).
6. cohen_kappa_score whose replace_undefined_by is no number (flag word 1): t = p = [0, 0, 0] raises
   "TypeError: scoring must return a real scalar"; the defined t [0, 1], p [0, 1] still scores 1.0.
7. `_classes_words`: ["0", "10"] of 2 classes "[ 0 10]"; ["'a'", "'bb'", "5"] of 2 classes "['a' 'bb']".

FAIL-PROOF EDITS (each must make a case fail): in `_float_label_value` pass `positive` instead of `side_positive`
(cases 0, 1, 3: the absent word reaches `_binary_value`, which answers 0.0 / the other words); in
`_float_binary_positive` drop the `len(union) > 2` check (case 2 scores instead); compare `union[j] == 1.0` instead
of `row.param` (case 0 with param 2.0, below); in `_numpy_float_words` take `max_a > 1000.0 * min_a` out of `sci`
(case 5, the 2000 / 1500 rows); in the kappa branch drop `if row.flag == 1` (case 6 answers NaN); in
`_classes_words` set `ints = False` (case 7).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from bindings.arena_prog_run import MetricRoute
from core.msel_scorer_run import (
    msel_scorer_row, msel_scorer_value, msel_scorer_code, msel_scorer_metric_code, MselScorer, MselScorerOptions,
    SCORER_POSITIVE_ABSENT, SCORER_POSITIVE_KIND, SCORER_FLOAT_POS_WORDS, SCORER_REAL_WORDS, _numpy_float_words,
)
from core.msel_search_run import _classes_words
from core.entry_table import entry_ref_unresolved
from x_metrics.descriptor_host import metrics_program_host_v1

comptime CASES = 8


def _route() -> MetricRoute:
    return MetricRoute(metrics_program_host_v1, False, False)


def _near(got: Float64, want: Float64, what: String) raises:
    require(abs(got - want) <= 1e-6 * max(abs(want), 1.0), what + ": " + String(got) + " expected " + String(want))


def _binary_row(name: String, param: Float64, pos_float: Bool, sels: List[Int32]) raises -> MselScorer:
    var opts = MselScorerOptions()
    opts.param = param
    opts.param_word = 0
    opts.pos_given = pos_float
    opts.pos_float = pos_float
    return msel_scorer_row(msel_scorer_code(name, False), opts, Int(sels.unsafe_ptr()))


def _side(row: MselScorer, words: List[String], y: List[Float32], p: List[Float32], positive: Int,
          mut notes: List[Float64]) raises -> Float64:
    return msel_scorer_value(_route(), entry_ref_unresolved(), words, row, Int(y.unsafe_ptr()), Int(p.unsafe_ptr()),
                             len(y), 0, 1, positive, notes)


def _side_words(row: MselScorer, words: List[String], y: List[Float32], p: List[Float32],
                positive: Int) raises -> String:
    var notes = List[Float64]()
    try:
        _ = _side(row, words, y, p, positive, notes)
    except e:
        return String(e)
    return String("")


def check_f1_on_a_regressor() raises:
    var none = List[Int32]()
    var words: List[String] = ["1"]
    var notes = List[Float64]()
    var y: List[Float32] = [0.0, 1.0, 1.0, 0.0]
    var p: List[Float32] = [0.0, 1.0, 0.0, 0.0]
    _near(_side(_binary_row("f1", 1.0, False, none), words, y, p, SCORER_POSITIVE_ABSENT, notes), 2.0 / 3.0,
          "f1 of float labels, pos_label 1")
    # pos_label 0.0 (a real equal to a label): precision 2/3, recall 1, f1 0.8
    var zero: List[String] = ["0.0"]
    _near(_side(_binary_row("f1", 0.0, False, none), zero, y, p, SCORER_POSITIVE_ABSENT, notes), 0.8,
          "f1 of float labels, pos_label 0.0")
    _ = len(none)


def check_absent_two_labels() raises:
    var none = List[Int32]()
    var words: List[String] = ["1"]
    var y: List[Float32] = [0.0, 2.0, 2.0, 0.0]
    var p: List[Float32] = [0.0, 2.0, 0.0, 0.0]
    var got = _side_words(_binary_row("precision", 1.0, False, none), words, y, p, SCORER_POSITIVE_ABSENT)
    require(got == "ValueError: pos_label=1 is not a valid label. It should be one of [0. 2.]", "absent words: " + got)
    _ = len(none)


def check_multiclass() raises:
    var none = List[Int32]()
    var words: List[String] = ["1"]
    var y: List[Float32] = [0.0, 1.0, 2.0]
    var got = _side_words(_binary_row("recall", 1.0, False, none), words, y, y, SCORER_POSITIVE_ABSENT)
    require(got == ("ValueError: Target is multiclass but average='binary'. Please choose another average setting, "
                    + "one of [None, 'micro', 'macro', 'weighted']."), "multiclass words: " + got)
    _ = len(none)


def check_one_label() raises:
    var none = List[Int32]()
    var words: List[String] = ["1"]
    var notes = List[Float64]()
    var y: List[Float32] = [0.0, 0.0]
    _near(_side(_binary_row("recall", 1.0, False, none), words, y, y, SCORER_POSITIVE_ABSENT, notes), 0.0,
          "recall of an absent pos_label")
    _ = len(none)


def check_pos_kinds() raises:
    var none = List[Int32]()
    var words: List[String] = ["None"]
    var y: List[Float32] = [0.0, 1.0]
    var got = _side_words(_binary_row("f1", 0.0, True, none), words, y, y, SCORER_POSITIVE_ABSENT)
    require(got == String(SCORER_FLOAT_POS_WORDS), "no-label pos_label words: " + got)
    var text: List[String] = ["a"]
    got = _side_words(_binary_row("f1", 0.0, False, none), text, y, y, SCORER_POSITIVE_KIND)
    require(got == "pos_label must use the same type as the targets", "string pos_label words: " + got)
    _ = len(none)


def check_numpy_words() raises:
    var a: List[Float64] = [-1.0, 2.0]
    require(_numpy_float_words(a) == "[-1.  2.]", "[-1, 2]: " + _numpy_float_words(a))
    var b: List[Float64] = [0.0, 10.0]
    require(_numpy_float_words(b) == "[ 0. 10.]", "[0, 10]: " + _numpy_float_words(b))
    var c: List[Float64] = [1.0, 2000.0]
    require(_numpy_float_words(c) == "[1.e+00 2.e+03]", "[1, 2000]: " + _numpy_float_words(c))
    var d: List[Float64] = [1.0, 1500.0]
    require(_numpy_float_words(d) == "[1.0e+00 1.5e+03]", "[1, 1500]: " + _numpy_float_words(d))
    var e: List[Float64] = [0.0, 1.0e8]
    require(_numpy_float_words(e) == "[0.e+00 1.e+08]", "[0, 1e8]: " + _numpy_float_words(e))


def check_kappa_no_number() raises:
    var none = List[Int32]()
    var opts = MselScorerOptions()
    opts.flag = 1
    var row = msel_scorer_row(msel_scorer_metric_code("cohen_kappa_score", True), opts, Int(none.unsafe_ptr()))
    var words = List[String]()
    var notes = List[Float64]()
    var t: List[Int32] = [0, 0, 0]
    var got = String("")
    try:
        _ = msel_scorer_value(_route(), entry_ref_unresolved(), words, row, Int(t.unsafe_ptr()), Int(t.unsafe_ptr()),
                              3, 1, 1, -1, notes)
    except e:
        got = String(e)
    require(got == String(SCORER_REAL_WORDS), "undefined kappa with no number: " + got)
    var u: List[Int32] = [0, 1]
    _near(msel_scorer_value(_route(), entry_ref_unresolved(), words, row, Int(u.unsafe_ptr()), Int(u.unsafe_ptr()), 2,
                            2, 1, -1, notes), 1.0, "defined kappa")
    _ = len(none)
    _ = len(t)
    _ = len(u)


def check_classes_words() raises:
    var ints: List[String] = ["0", "10"]
    require(_classes_words(ints, 2) == "[ 0 10]", "int classes: " + _classes_words(ints, 2))
    var strs: List[String] = ["'a'", "'bb'", "5"]
    require(_classes_words(strs, 2) == "['a' 'bb']", "string classes: " + _classes_words(strs, 2))


def test_mf_rows8a_scorers() raises:
    def run_case(which: Int) raises:
        if which == 0:
            check_f1_on_a_regressor()
        elif which == 1:
            check_absent_two_labels()
        elif which == 2:
            check_multiclass()
        elif which == 3:
            check_one_label()
        elif which == 4:
            check_pos_kinds()
        elif which == 5:
            check_numpy_words()
        elif which == 6:
            check_kappa_no_number()
        else:
            check_classes_words()
    check_cases("mf_rows8a_scorers", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
