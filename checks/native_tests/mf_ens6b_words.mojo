# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Lane mf-ens6b: the search kind's refusal words (core/msel_refusal_words.mojo), built with no Python, equal to
the text scikit-learn 1.9 and the search door raise (the oracle strings below are written from their sources, not
from the module under test).

 1. An empty multi-metric scoring: a list (`[]`), a set (`set()`) and a dict (`{}`), metrics/_scorer.py
    `_check_multimetric_scoring` lines 634 and 644 (the list form keeps scikit-learn's two spaces).
 2. cross_val_predict's `method` outside its options (model_selection/_validation.py line 1018; options sorted).
 3. KFold's n_splits < 2 for 1 and 0 (model_selection/_split.py `_BaseKFold`), and the search door's refit
    callback and no-fits words (bindings/msel_search_door.mojo `msel_search_fit`).

FAIL-PROOF EDITS (each must make the named case fail): in core/msel_refusal_words.mojo
`msel_empty_scoring_words` drop one of the two spaces before "Empty list" (case 1); in `msel_cvp_method_words`
list the options in scikit-learn's declaration order instead of sorted (case 2); in `msel_kfold_words` print
`n_splits + 1` (case 3).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from core.msel_refusal_words import (
    msel_empty_scoring_words, msel_cvp_method_words, msel_kfold_words, MSEL_REFIT_CALLBACK_WORDS, MSEL_NO_FITS_WORDS,
)

comptime CASES = 3


def _same(got: String, want: String, label: String) raises:
    require(got == want, label + ": got [" + got + "] want [" + want + "]")


def check_empty_scoring() raises:
    var head = String("ValueError: The list/tuple elements must be unique strings of predefined scorers. ")
    _same(msel_empty_scoring_words(False, "[]"), head + " Empty list was given. []", "empty list")
    _same(msel_empty_scoring_words(False, "set()"), head + " Empty list was given. set()", "empty set")
    _same(msel_empty_scoring_words(True, "{}"), "ValueError: An empty dict was passed. {}", "empty dict")


def check_method() raises:
    var want = String("ValueError: The 'method' parameter of cross_val_predict must be a str among ")
    want += "{'decision_function', 'predict', 'predict_log_proba', 'predict_proba'}. Got 'transform' instead."
    _same(msel_cvp_method_words("'transform'"), want, "method 'transform'")


def check_kfold_and_door() raises:
    var head = String("ValueError: k-fold cross-validation requires at least one train/test split by setting ")
    _same(msel_kfold_words(1), head + "n_splits=2 or more, got n_splits=1.", "n_splits 1")
    _same(msel_kfold_words(0), head + "n_splits=2 or more, got n_splits=0.", "n_splits 0")
    _same(String(MSEL_REFIT_CALLBACK_WORDS),
          "TypeError: Python refit callbacks are not supported; use a native bool or scorer-name refit policy",
          "refit callback")
    _same(String(MSEL_NO_FITS_WORDS),
          "ValueError: No fits were performed. Was the CV iterator empty? Were there no candidates?", "no fits")


def test_mf_ens6b_words() raises:
    def run_case(which: Int) raises:
        if which == 0:
            check_empty_scoring()
        elif which == 1:
            check_method()
        else:
            check_kfold_and_door()

    check_cases("mf_ens6b_words", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
