"""Before/after digests of lane mf-r9c (2026-11-23): the last scorer rows of a search over native members, host
column. One line per configuration and output: name and sha256 of the output bytes (dtype and shape included); a
refusal is digested by its exception class and words. Base: origin/lane/member-fits-8 be11679002 (each
configuration made the search's FIT admission answer False: the door's units ran it), head lane/mf-r9c (the typed
RUN runs it, or raises the words). Test tool, never product runtime.

Searches are GridSearchCV over one grid, three fixed folds of four test rows, refit=False, error_score='raise'
unless the name ends in `_tol` (error_score=-3.0). Outputs: split0..2 test scores, or the words.

EXPECTED (base vs head):
- `mo_ml_roc` (KNN, two 0/1 outputs, 'roc_auc'): head the doors' NOT IMPLEMENTED words (scikit-learn serves a
  multilabel-indicator target); base the units' door words for a 2-D y_true (NOT IMPLEMENTED too).
- `mo_mc_logloss` (KNN, outputs of 3 and 4 labels, 'neg_log_loss'): head NumPy's inhomogeneous-shape words, as
  scikit-learn's check_array raises them; base the door's 2-D words (a stated change to scikit-learn's words).
- `mo_mc_topk_tol`: three -3.0 on head (column_or_1d's words, error_score); base: the units' refusal, -3.0 too.
- `dual_ovr_knn`: head scikit-learn's ovr AUC of KNN's predict_proba; base: the door's units (equal values
  expected up to float rounding).
- `dual_ovr_logreg`: head roc_auc_score's "Target scores need to be probabilities" words; base the same words
  from the door's metric.
- `r2_proba_binary`: head r2 of the labels against classes[-1]'s column (scikit-learn's value); base the units'
  value (equal up to float rounding).
- `r2_proba_multi`: head "y_true and y_pred have different number of output (1!=3)"; base the door's words.

Configurations are independent: one process each across the host cores.

    python tools/mf_r9c_digests.py > digests.txt
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
import sys
import warnings

import numpy as np

_TESTS = [[0, 2, 4, 9], [1, 3, 5, 10], [6, 7, 8, 11]]


def _digest(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()[:24]


def _folds():
    every = np.arange(12)
    return [(np.setdiff1d(every, np.array(t)), np.array(t)) for t in _TESTS]


def _data(name):
    X = np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(12, 2)
    i = np.arange(12)
    if name.startswith("mo_ml"):
        return X, np.stack([i % 2, (i // 2) % 2], axis=1).astype(np.int64)
    if name.startswith("mo_mc"):
        return X, np.stack([i % 3, i % 4], axis=1).astype(np.int64)
    if name == "r2_proba_binary":
        return X, (i % 2).astype(np.int64)
    return X, (i % 3).astype(np.int64)


def _unit(name):
    out = []
    warnings.simplefilter("ignore")
    try:
        import mojolearn as ml
        from mojolearn import metrics
        from mojolearn import model_selection as ms
        X, y = _data(name)
        error_score = -3.0 if name.endswith("_tol") else 'raise'
        base = name[:-4] if name.endswith("_tol") else name
        dual = ms.make_scorer(metrics.roc_auc_score, response_method=("decision_function", "predict_proba"),
                              multi_class="ovr")
        r2_proba = ms.make_scorer(metrics.r2_score, response_method="predict_proba")
        scoring = {"mo_ml_roc": 'roc_auc', "mo_mc_logloss": 'neg_log_loss', "mo_mc_topk": 'top_k_accuracy',
                   "dual_ovr_knn": dual, "dual_ovr_logreg": dual, "r2_proba_binary": r2_proba,
                   "r2_proba_multi": r2_proba}[base]
        if base == "dual_ovr_logreg":
            est, grid = ml.LogisticRegression(max_iter=30), {'C': [1.0]}
        else:
            est, grid = ml.KNeighborsClassifier(), {'n_neighbors': [3]}
        search = ms.GridSearchCV(est, grid, cv=_folds(), scoring=scoring, error_score=error_score,
                                 refit=False).fit(X, y)
        for f in range(3):  # glue: the three folds' scores
            out.append((f"{name}_split{f}", _digest(search.cv_results_[f'split{f}_test_score'])))
    except Exception as error:  # a refusal is digested by its class and words
        out.append((name + "_raised", hashlib.sha256(f"{type(error).__name__}: {error}".encode()).hexdigest()[:24]))
    return out


UNITS = ["mo_ml_roc", "mo_mc_logloss", "mo_mc_topk_tol", "dual_ovr_knn", "dual_ovr_logreg", "r2_proba_binary",
         "r2_proba_multi"]


def main():
    with ProcessPoolExecutor(max_workers=min(len(UNITS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_unit, UNITS):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
