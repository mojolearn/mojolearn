# SPDX-License-Identifier: Apache-2.0
"""The refusal words of the search kind's admission (lane mf-ens6b), typed: each a function of the values the
UNPACK reads (their `repr`s as strings), so a native check builds the same text with no Python and the doors
raise one body.

- `msel_empty_scoring_words`: scikit-learn 1.9 metrics/_scorer.py `_check_multimetric_scoring` (lines 634, 644),
  an empty multi-metric scoring list / tuple / set, or dict: scikit-learn's own fit fails.
- `msel_cvp_method_words`: scikit-learn 1.9 model_selection/_validation.py line 1018, cross_val_predict's
  `method` StrOptions (the options in sorted order: scikit-learn prints a set in hash order).
- `msel_kfold_words`: KFold's n_splits < 2 refusal (sklearn model_selection/_split.py `_BaseKFold.__init__`).
- `MSEL_REFIT_CALLBACK_WORDS` / `MSEL_NO_FITS_WORDS`: the search door's own words (bindings/msel_search_door.mojo
  `msel_search_fit`).
"""

comptime MSEL_REFIT_CALLBACK_WORDS = "TypeError: Python refit callbacks are not supported; use a native bool or scorer-name refit policy"
comptime MSEL_NO_FITS_WORDS = "ValueError: No fits were performed. Was the CV iterator empty? Were there no candidates?"


def msel_empty_scoring_words(mapping: Bool, given: String) -> String:
    """The words for an empty multi-metric scoring; `given` is its `repr`."""
    if mapping:
        return "ValueError: An empty dict was passed. " + given
    # scikit-learn's `err_msg` ends in a space and the f-string adds one more
    return ("ValueError: The list/tuple elements must be unique strings of predefined scorers.  Empty list was given. "
            + given)


def msel_cvp_method_words(given: String) -> String:
    """cross_val_predict's `method` outside its options; `given` is its `repr`."""
    return ("ValueError: The 'method' parameter of cross_val_predict must be a str among {'decision_function', "
            + "'predict', 'predict_log_proba', 'predict_proba'}. Got " + given + " instead.")


def msel_kfold_words(n_splits: Int) -> String:
    """KFold's refusal of fewer than two splits."""
    return ("ValueError: k-fold cross-validation requires at least one train/test split by setting n_splits=2 or "
            + "more, got n_splits=" + String(n_splits) + ".")
