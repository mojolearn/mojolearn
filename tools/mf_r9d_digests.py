"""Before/after digests of lane mf-r9d (2026-11-23): the scorer rows of a search over native members this lane
typed, host column. One line per configuration and output: name and sha256 of the output bytes (dtype and shape
included); a refusal is digested by its exception class and words. Base: origin/lane/member-fits-8 ea2e0c0139 (each
configuration made the search's FIT admission answer False: the door's units ran it), head lane/mf-r9d (the typed
RUN runs it, or raises the words). Test tool, never product runtime.

Searches are GridSearchCV over one grid, three fixed folds of four test rows, refit=False, error_score='raise'
unless the name ends in `_tol` (error_score=-3.0). Outputs: split0..2 test scores, or the words.

EXPECTED (base vs head):
- `tier_fast`, `tier_identical` (KNeighborsRegressor, make_scorer(r2_score, numeric_mode=...)): the default tier
  is unchanged; the other tier on head scores on that tier's image (base: the door's units, the same image), or
  both raise its loader's words. Equal values expected.
- `logproba_r2` (LogisticRegression, two classes, r2 over predict_log_proba): head the log of classes[-1]'s
  column in the member's precision; base the units' np.log of predict_proba. Equal up to float rounding.
- `tuple_hinge` (hinge_loss over ('decision_function', 'predict')): head the decision row; base the units.
- `gmm_roc_one` (GaussianMixture n_components=1, float 0/1 y, 'roc_auc'): 0.5 per split both.
- `gmm_roc_two_tol`: -3.0 per split both (column_or_1d's words; error_score).
- `gmm_roc_two`: head scikit-learn's column_or_1d words; base the door metric's words.
- `gmm_roc_continuous`: head "continuous format is not supported"; base the door metric's words.
- ROUND b `gmm_logloss` (3 target values, 2 components): head scikit-learn's class-count words; base the door's.
- ROUND b `wrong_accuracy` (accuracy over ('predict_proba', 'predict')): head the mix words; base the door's.

Configurations are independent: one process each across the allocated cores.

    python tools/mf_r9d_digests.py > digests.txt
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
import sys
import warnings

import numpy as np

_TESTS = [[0, 2, 4, 9], [1, 3, 5, 10], [6, 7, 8, 11]]


def _cores():
    try:
        return len(os.sched_getaffinity(0))  # the allocation, not the host's visible CPUs
    except AttributeError:
        return os.cpu_count() or 1


def _digest(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()[:24]


def _folds():
    every = np.arange(12)
    return [(np.setdiff1d(every, np.array(t)), np.array(t)) for t in _TESTS]


def _unit(name):
    out = []
    warnings.simplefilter("ignore")
    try:
        import mojolearn as ml
        from mojolearn import metrics
        from mojolearn import model_selection as ms
        X = np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(12, 2)
        i = np.arange(12)
        error_score = -3.0 if name.endswith("_tol") else 'raise'
        base = name[:-4] if name.endswith("_tol") else name
        if base.startswith("tier_"):
            est, grid, y = ml.KNeighborsRegressor(numeric_mode="identical"), {'n_neighbors': [3]}, i / 6.0
            scoring = ms.make_scorer(metrics.r2_score, numeric_mode=base[len("tier_"):])
        elif base == "logproba_r2":
            est, grid, y = ml.LogisticRegression(max_iter=30), {'C': [1.0]}, i % 2
            scoring = ms.make_scorer(metrics.r2_score, response_method="predict_log_proba")
        elif base == "gmm_logloss":
            est, grid, y = ml.GaussianMixture(random_state=0, numeric_mode="identical"), {'n_components': [2]}, \
                (i % 3).astype(np.float64)
            scoring = 'neg_log_loss'
        elif base == "wrong_accuracy":
            est, grid, y = ml.LogisticRegression(max_iter=30), {'C': [1.0]}, i % 2
            scoring = ms.make_scorer(metrics.accuracy_score, response_method=("predict_proba", "predict"))
        elif base == "tuple_hinge":
            est, grid, y = ml.LogisticRegression(max_iter=30), {'C': [1.0]}, i % 2
            scoring = ms.make_scorer(metrics.hinge_loss, response_method=("decision_function", "predict"),
                                     greater_is_better=False)
        else:
            components = 1 if base == "gmm_roc_one" or base == "gmm_roc_continuous" else 2
            est = ml.GaussianMixture(random_state=0, numeric_mode="identical")
            grid = {'n_components': [components]}
            y = (i / 12.0 + 0.25) if base == "gmm_roc_continuous" else (i % 2).astype(np.float64)
            scoring = 'roc_auc'
        search = ms.GridSearchCV(est, grid, cv=_folds(), scoring=scoring, error_score=error_score,
                                 refit=False).fit(X, y)
        for f in range(3):  # glue: the three folds' scores
            out.append((f"{name}_split{f}", _digest(search.cv_results_[f'split{f}_test_score'])))
    except Exception as error:  # a refusal is digested by its class and words
        out.append((name + "_raised", hashlib.sha256(f"{type(error).__name__}: {error}".encode()).hexdigest()[:24]))
    return out


UNITS = ["tier_fast", "tier_identical", "logproba_r2", "tuple_hinge", "gmm_roc_one", "gmm_roc_two_tol",
         "gmm_roc_two", "gmm_roc_continuous", "gmm_logloss", "wrong_accuracy"]


def main():
    with ProcessPoolExecutor(max_workers=min(len(UNITS), _cores())) as pool:
        for lines in pool.map(_unit, UNITS):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
