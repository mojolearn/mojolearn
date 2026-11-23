# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Lane mf-del5, in a process with no Python: CalibratedClassifierCV's two cv refusals
(xtrees/trees_calibrated_fit_run.mojo `cal_cv_check`, the one body of its door and its META kind) and a
regressor's stratified splitter on the member path (bindings/msel_split_spec_run.mojo `msel_target_codes_run`,
`msel_spec_sides_targets`). Oracles: scikit-learn 1.9 calibration.py's words and np.unique's inverse, written
out here as literals; the splitter door's own StratifiedKFold over the same codes (the one-body check).

 1. cal_cv_check over codes [0, 0, 0, 1, 1, 2, 2, 2] (k = 3): n_folds 2 passes; n_folds 3 raises "Requesting
    3-fold cross-validation but provided less than 3 examples for at least one class."; a code no row holds
    (k = 4) is not a class; n_folds 0 never counts.
 2. cal_cv_check with loo: scikit-learn's LeaveOneOut words, verbatim (its "allowall").
 3. msel_target_codes_run over [2.5, -1.0, 2.5, 0.0, -0.0, 7.0]: codes [2, 0, 2, 1, 1, 3], m = 4 (-0.0 and 0.0
    one value); a NaN target refused.
 4. msel_spec_sides_targets with StratifiedKFold(2) over targets [1, 2, 1, 2, 1, 2] == msel_spec_sides over the
    codes [0, 1, 0, 1, 0, 1] (the door's split of a regressor's y), side for side; continuous targets
    [0.1 .. 0.6] meet the door's class-size refusal; a KFold spec reads no targets (== msel_spec_sides).

Lane mf-cal5 (THE FALLBACK RULE, Gbdt cat_features index outside X with an eval_set):
 5. core/gbdt_cat_words.mojo `gbdt_cat_outside_words` over a feature-list tail [w0, w1, cat..., hot..., C, H]:
    every index inside X answers ""; the FIRST index outside X in the given order (not the sorted order) is
    named: tail cats [1, 7, -2] with d = 5 answers "mojolearn: cat_features index 7 out of range for 5
    features" (the fit door's `_cat_slots` words, bindings/gbdt_category_unpack.mojo, literal here); a
    negative index alone answers its words; one-hot indices are not read.

FAIL-PROOF EDITS (each must make the named case fail): in `cal_cv_check` test `total <= n_folds` (case 1, the
n_folds 2 pass) or drop the `total > 0` guard (case 1, the k = 4 row); drop the `if loo` raise (case 2); in
`msel_target_codes_run` keep float32 bits uncast (pass `y` to `host_unique_inverse` as if float64) (case 3); in
`msel_spec_sides_targets` pass `y` itself as the codes (case 4); in `gbdt_cat_outside_words` test `index > d`
(case 5, the index 5 row) or read the indices sorted (case 5, the [1, 7, -2] row).
"""
from std.math import nan
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.msel_entry_table import MselEntryKey
from core.msel_helpers_host import publish_msel_helpers_host
from core.msel_helpers_calls import msel_helpers_ref
from core.base_entry_table import BaseEntryKey
from core.base_helpers_host import publish_base_helpers_host
from core.base_helpers_calls import base_helpers_ref
from x_metrics.descriptor_host import metrics_program_host_v1
from bindings.msel_split_plan_run import MselSplitRoute
from bindings.msel_split_spec_run import (
    MselSplitSpec, MselSides, MselGroupCodes, msel_spec_sides, msel_spec_sides_targets, msel_target_codes_run,
    MSEL_SPEC_KFOLD, MSEL_SPEC_STRATIFIED_KFOLD,
)
from xtrees.trees_calibrated_fit_run import cal_cv_check, CAL_LOO_WORDS
from core.gbdt_cat_words import gbdt_cat_outside_words

comptime CASES = 5


def _route() raises -> MselSplitRoute:
    var mode = Int(GLOBAL_NUMERIC_MODE)
    return MselSplitRoute(msel_helpers_ref(MselEntryKey(mode, 0)), base_helpers_ref(BaseEntryKey(mode, 0)),
                          metrics_program_host_v1, False, False)


def _spec(kind: Int, k: Int) raises -> MselSplitSpec:
    var spec = MselSplitSpec()
    spec.kind = kind
    spec.k = k
    spec.route = _route()
    spec.rows_helpers = base_helpers_ref(BaseEntryKey(Int(GLOBAL_NUMERIC_MODE), 0))
    return spec^


def _raises_words(codes: List[Int32], k: Int, n_folds: Int, loo: Bool) -> String:
    try:
        cal_cv_check(Int(codes.unsafe_ptr()), len(codes), k, n_folds, loo)
    except error:
        _ = len(codes)
        return String(error)
    _ = len(codes)
    return String("")


def check_requesting() raises:
    var codes: List[Int32] = [0, 0, 0, 1, 1, 2, 2, 2]
    require(_raises_words(codes, 3, 2, False) == "", "n_folds 2: every class has 2 rows or more")
    var want = String("Requesting 3-fold cross-validation but provided less than 3 examples for at least one class.")
    require(_raises_words(codes, 3, 3, False) == want, "n_folds 3: scikit-learn's words")
    var full: List[Int32] = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    require(_raises_words(full, 4, 3, False) == "", "a code no row holds is not a class")
    require(_raises_words(codes, 3, 0, False) == "", "n_folds 0 reads no counts")


def check_loo() raises:
    var codes: List[Int32] = [0, 1, 0, 1]
    var got = _raises_words(codes, 2, 0, True)
    require(got == String(CAL_LOO_WORDS), "LeaveOneOut: " + got)
    require(got.startswith("LeaveOneOut cross-validation does not allowall classes"), "the words are verbatim")


def check_target_codes() raises:
    var y: List[Float32] = [2.5, -1.0, 2.5, 0.0, -0.0, 7.0]
    var got = msel_target_codes_run(Int(y.unsafe_ptr()), len(y))
    _ = len(y)
    var want: List[Int] = [2, 0, 2, 1, 1, 3]
    require(got.m == 4, "four distinct values, got " + String(got.m))
    for i in range(len(want)):  # small-loop(oracle: six literal codes)
        require(Int(got.codes[i]) == want[i], "code " + String(i) + " is " + String(Int(got.codes[i])))
    var bad: List[Float32] = [1.0, nan[DType.float32]()]
    var refused = False
    try:
        _ = msel_target_codes_run(Int(bad.unsafe_ptr()), len(bad))
    except error:
        refused = String(error) == "Input y contains NaN."
    _ = len(bad)
    require(refused, "a NaN target is refused")


def _same(a: MselSides, b: MselSides, what: String) raises:
    require(a.splits() == b.splits(), what + ": split counts differ")
    for s in range(a.splits()):  # small-loop(a few splits)
        require(len(a.test[s]) == len(b.test[s]) and len(a.train[s]) == len(b.train[s]), what + ": side sizes")
        for i in range(len(a.test[s])):  # small-loop(a few rows)
            require(a.test[s][i] == b.test[s][i], what + ": test row " + String(i))
        for i in range(len(a.train[s])):  # small-loop(a few rows)
            require(a.train[s][i] == b.train[s][i], what + ": train row " + String(i))


def check_targets_sides() raises:
    var spec = _spec(MSEL_SPEC_STRATIFIED_KFOLD, 2)
    var y: List[Float32] = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    var codes: List[Int32] = [0, 1, 0, 1, 0, 1]
    var got = msel_spec_sides_targets(spec, 6, Int(y.unsafe_ptr()), MselGroupCodes())
    var want = msel_spec_sides(spec, 6, Int(codes.unsafe_ptr()), 2, MselGroupCodes())
    _ = len(y)
    _ = len(codes)
    _same(got, want, "stratified on the targets")
    var cont: List[Float32] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    var refused = False
    try:
        _ = msel_spec_sides_targets(spec, 6, Int(cont.unsafe_ptr()), MselGroupCodes())
    except error:
        refused = String(error).find("cannot be greater than the number of members in each class") >= 0
    _ = len(cont)
    require(refused, "continuous targets: the door's class-size refusal")
    var kf = _spec(MSEL_SPEC_KFOLD, 3)
    var plain = msel_spec_sides_targets(kf, 6, Int(cont.unsafe_ptr()), MselGroupCodes())
    var plain_want = msel_spec_sides(kf, 6, 0, 0, MselGroupCodes())
    _ = len(cont)
    _same(plain, plain_want, "KFold reads no targets")


def _tail(cats: List[Float64], hot: List[Float64]) -> List[Float64]:
    """NEW's words ending in the feature-list tail: two leading words, the cat and one-hot indices, C, H."""
    var words: List[Float64] = [3.0, 0.5]
    for i in range(len(cats)):  # small-loop(cats)
        words.append(cats[i])
    for i in range(len(hot)):  # small-loop(hot)
        words.append(hot[i])
    words.append(Float64(len(cats)))
    words.append(Float64(len(hot)))
    return words^


def check_cat_outside() raises:
    var inside: List[Float64] = [0.0, 4.0, 2.0]
    var none: List[Float64] = []
    require(gbdt_cat_outside_words(_tail(inside, none), 5) == "", "every index inside X")
    var edge: List[Float64] = [5.0]
    require(gbdt_cat_outside_words(_tail(edge, none), 5)
            == "mojolearn: cat_features index 5 out of range for 5 features", "index d is outside X")
    var order: List[Float64] = [1.0, 7.0, -2.0]
    require(gbdt_cat_outside_words(_tail(order, none), 5)
            == "mojolearn: cat_features index 7 out of range for 5 features", "the first in the given order")
    var negative: List[Float64] = [-1.0]
    require(gbdt_cat_outside_words(_tail(negative, none), 5)
            == "mojolearn: cat_features index -1 out of range for 5 features", "a negative index")
    var hot: List[Float64] = [9.0]
    require(gbdt_cat_outside_words(_tail(inside, hot), 5) == "", "one-hot indices are not read")


def test_mf_del5_cv() raises:
    publish_msel_helpers_host()
    publish_base_helpers_host()

    def run_case(which: Int) raises:
        if which == 0:
            check_requesting()
        elif which == 1:
            check_loo()
        elif which == 2:
            check_target_codes()
        elif which == 3:
            check_targets_sides()
        else:
            check_cat_outside()

    check_cases("mf_del5_cv", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
