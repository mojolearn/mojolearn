"""Before/after digests of the scorer options lane mf-score6 (2026-11-23) moved off the search's Python units onto
the typed search RUN, host column. One line per configuration and output: name and sha256 of the output bytes
(dtype and shape included); a refusal is digested by its exception class and words. Base: origin/lane/member-fits-6
28608532e4 (each scorer below made the search admission answer False and the door's Python units scored it), head
lane/mf-score6 (the RUN scores it, or raises the metric door's refusal per side). Test tool, never product runtime.

Every search is GridSearchCV over LogisticRegression(max_iter=50), grid {'C': [1.0, 0.5]}, refit=False, on a fixed
9-row, 3-class set (and its 2-class subset for pos_label), with cv=[(every, every)] (train = test = every row, so a
make_scorer sample_weight has the side's length) or cv=3; `error_score` as named. Outputs: cv_results_'s
mean_test_score and split0_test_score, or the raised words. `call_*` units call the scorer itself on the fitted
estimator (the public scorer call).

EXPECTED:
- `none_one_label`, `w_accuracy`, `w_balanced`, `w_f1_macro`, `w_neg_log_loss`, `w_roc_auc_ovr`,
  `w_top_k`, `w_ap`, `w_brier`: values equal to the last bits or within 1e-6 relative (the RUN's binary64 programs
  against the door's metric calls: the unweighted label rows' public door is a float32 entry, the weighted one the
  same binary64 program as the RUN).
- `none_three_labels`: base, error_score nan: nan scores with a warning (the door caught its own TypeError
  "scoring must return a real scalar"); head: ValueError "scoring must return a number, got [...] ..." (scikit-
  learn's `_score`, whatever the error_score): a stated behaviour change.
- `w_short_raise`, `samples_raise`, `float_pos_raise`: the same exception class and words
  (the RUN raises the door's words; the order of checks is the door's).
- `w_short_nan`, `samples_nan`, `float_pos_nan`: nan scores on both sides.
- `call_none`, `call_none_neg`: base TypeError "scoring must return a real scalar"; head the signed per-label array
  (float32 from the unweighted door): a stated behaviour change.
- `callable`: TypeError on both sides; head words name "a scoring string ... or make_scorer over a native mojolearn
  metric".

Configurations are independent: one process each across the host cores.

    python tools/mf_score6_digests.py > digests.txt
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
import sys
import warnings

import numpy as np


def _digest(value):
    a = np.asarray(value)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}{a.shape}".encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:24]


def _data():
    X = np.array([[0.0, 0.1], [0.2, 0.0], [1.0, 1.1], [1.2, 0.9], [2.0, 2.1], [2.2, 1.9],
                  [0.1, 0.3], [1.1, 1.0], [2.1, 2.0]], dtype=np.float32)
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2], dtype=np.int32)
    return X, y


W = np.array([1.0, 2.0, 0.5, 1.0, 3.0, 1.0, 2.0, 1.0, 0.5], dtype=np.float32)


def _scoring(name):
    """(scoring, binary, cv, error_score) of configuration `name`."""
    from mojolearn import metrics
    from mojolearn import model_selection as ms
    mk = ms.make_scorer
    if name == "none_three_labels":
        return mk(metrics.f1_score, average=None), False, None, np.nan
    if name == "none_one_label":
        return mk(metrics.f1_score, average=None, labels=[1]), False, None, np.nan
    if name == "w_accuracy":
        return mk(metrics.accuracy_score, sample_weight=W), False, None, 'raise'
    if name == "w_balanced":
        return mk(metrics.balanced_accuracy_score, sample_weight=W), False, None, 'raise'
    if name == "w_f1_macro":
        return mk(metrics.f1_score, average='macro', sample_weight=W), False, None, 'raise'
    if name == "w_neg_log_loss":
        return (mk(metrics.log_loss, greater_is_better=False, response_method='predict_proba', sample_weight=W),
                False, None, 'raise')
    if name == "w_roc_auc_ovr":
        return (mk(metrics.roc_auc_score, response_method='predict_proba', multi_class='ovr', sample_weight=W),
                False, None, 'raise')
    if name == "w_top_k":
        return (mk(metrics.top_k_accuracy_score, response_method=('decision_function', 'predict_proba'),
                   sample_weight=W), False, None, 'raise')
    if name == "w_ap":
        return (mk(metrics.average_precision_score, response_method=('decision_function', 'predict_proba'),
                   sample_weight=W[:6]), True, None, 'raise')
    if name == "w_brier":
        return (mk(metrics.brier_score_loss, greater_is_better=False, response_method='predict_proba',
                   sample_weight=W[:6]), True, None, 'raise')
    if name == "w_short_raise":
        return mk(metrics.accuracy_score, sample_weight=W), False, 3, 'raise'
    if name == "w_short_nan":
        return mk(metrics.accuracy_score, sample_weight=W), False, 3, np.nan
    if name == "samples_raise":
        return mk(metrics.precision_score, average='samples'), False, None, 'raise'
    if name == "samples_nan":
        return mk(metrics.precision_score, average='samples'), False, None, np.nan
    if name == "float_pos_raise":
        return mk(metrics.f1_score, pos_label=1.0), True, None, 'raise'
    if name == "float_pos_nan":
        return mk(metrics.f1_score, pos_label=1.0), True, None, np.nan
    if name == "callable":
        return (lambda est, X, y: 0.0), False, None, 'raise'
    raise KeyError(name)


def _unit(name):
    out = []
    warnings.simplefilter("ignore")
    try:
        import mojolearn as ml
        from mojolearn import metrics
        from mojolearn import model_selection as ms
        X, y = _data()
        if name.startswith("call_"):
            est = ml.LogisticRegression(max_iter=50).fit(X, y)
            scorer = ms.make_scorer(metrics.precision_score, average=None,
                                    greater_is_better=name != "call_none_neg")
            out.append((name + "_value", _digest(np.asarray(scorer(est, X, y)))))
            return out
        scoring, binary, cv, error_score = _scoring(name)
        if binary:
            keep = y < 2
            X, y = X[keep], y[keep]
        every = np.arange(len(X))
        search = ms.GridSearchCV(ml.LogisticRegression(max_iter=50), {'C': [1.0, 0.5]},
                                 cv=cv if cv is not None else [(every, every)], scoring=scoring,
                                 error_score=error_score, refit=False).fit(X, y)
        out.append((name + "_mean", _digest(search.cv_results_['mean_test_score'])))
        out.append((name + "_split0", _digest(search.cv_results_['split0_test_score'])))
    except Exception as error:  # a refusal is digested by its class and words
        out.append((name + "_raised", hashlib.sha256(f"{type(error).__name__}: {error}".encode()).hexdigest()[:24]))
    return out


UNITS = ["none_three_labels", "none_one_label", "w_accuracy", "w_balanced", "w_f1_macro",
         "w_neg_log_loss", "w_roc_auc_ovr", "w_top_k", "w_ap", "w_brier", "w_short_raise", "w_short_nan",
         "samples_raise", "samples_nan", "float_pos_raise", "float_pos_nan", "callable",
         "call_none", "call_none_neg"]


def main():
    with ProcessPoolExecutor(max_workers=min(len(UNITS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_unit, UNITS):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
