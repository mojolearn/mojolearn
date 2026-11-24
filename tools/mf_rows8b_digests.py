"""Before/after digests of lane mf-rows8b (2026-11-23): the multi-output and refusal search rows, host column. One
line per configuration and output: name and sha256 of the output bytes (dtype and shape included); a refusal is
digested by its exception class and words. Base: origin/lane/member-fits-8 d2b0f3aa3c (each configuration made the
search's FIT admission answer False: the door's units ran it), head lane/mf-rows8b (the typed RUN runs it, or
raises the words). Test tool, never product runtime.

Searches are GridSearchCV over {'n_neighbors': [3]}, three fixed folds, refit=False, error_score='raise' unless the
name ends in `_tol` (error_score=-3.0). Outputs: split0..2 test scores, a cross_val_predict array, or the words.

EXPECTED (base vs head):
- `mo_reg_accuracy` (+ `_tol`): head "continuous-multioutput is not supported" (scikit-learn's words), `_tol`
  three -3.0; base: the label door's 2-D NOT IMPLEMENTED words (a stated change to scikit-learn's words).
- `mo_reg_ari`: head "labels_true must be 1D: shape is (3, 2)"; base the cluster door's 1-D words.
- `mo_cls_multilabel_acc`: NOT IMPLEMENTED words on both; `mo_cls_multilabel_mcc`: head scikit-learn's
  "multilabel-indicator is not supported", base the door's NOT IMPLEMENTED words.
- `mo_cls_r2`: equal values (scikit-learn's r2 of the labels' values) on both.
- `mo_cls_strat`: head StratifiedKFold's words; base the door's split words.
- `one_out_knn_acc`: equal values on both (head: the typed RUN; base: the units).
- `gmm_cvp_proba`: equal arrays up to float rounding (head: the member's own columns, float64).
- `reg_float_ari`: the cluster door's labels_pred words on both.
- `nested_some_refused_tol`: head scores the refused candidate -3.0 and the other natively; base: the units.

Configurations are independent: one process each across the host cores.

    python tools/mf_rows8b_digests.py > digests.txt
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
import sys
import warnings

import numpy as np

_TESTS = [[0, 2, 4], [1, 3, 5], [6, 7, 8]]


def _digest(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()[:24]


def _folds():
    every = np.arange(12)
    return [(np.setdiff1d(every, np.array(t)), np.array(t)) for t in _TESTS]


def _data(name):
    X = np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(12, 2)
    i = np.arange(12)
    if name.startswith("mo_reg"):
        return X, np.stack([np.linspace(0.1, 2.3, 12), np.linspace(1.7, 0.2, 12)], axis=1).astype(np.float32)
    if "multilabel" in name:
        return X, np.stack([i % 2, (i // 2) % 2], axis=1).astype(np.int64)
    if name.startswith("mo_cls"):
        return X, np.stack([i % 3 * 10, i % 4], axis=1).astype(np.int64)
    return X, (i % 3).astype(np.int64)


def _unit(name):
    out = []
    warnings.simplefilter("ignore")
    try:
        import mojolearn as ml
        from mojolearn import model_selection as ms
        X, y = _data(name)
        error_score = -3.0 if name.endswith("_tol") else 'raise'
        base = name[:-4] if name.endswith("_tol") else name
        if base == "gmm_cvp_proba":
            got = ms.cross_val_predict(ml.GaussianMixture(n_components=2, random_state=0), X, None, cv=_folds(),
                                       method='predict_proba')
            return [(name, _digest(got))]
        if base == "nested_some_refused":
            est = ml.BaggingClassifier(ml.DecisionTreeClassifier(random_state=0), random_state=0)
            grid = {'n_estimators': [3, -1]}  # the second candidate's nested admission refuses
            search = ms.GridSearchCV(est, grid, cv=_folds(), scoring='accuracy', error_score=error_score,
                                     refit=False).fit(X, y)
        else:
            scoring = {"mo_reg_accuracy": 'accuracy', "mo_reg_ari": 'adjusted_rand_score',
                       "mo_cls_multilabel_acc": 'accuracy', "mo_cls_multilabel_mcc": 'matthews_corrcoef',
                       "mo_cls_r2": 'r2', "mo_cls_strat": 'accuracy', "one_out_knn_acc": 'accuracy',
                       "reg_float_ari": 'adjusted_rand_score'}[base]
            regressor = base.startswith("mo_reg") or base == "reg_float_ari"
            est = ml.KNeighborsRegressor() if regressor else ml.KNeighborsClassifier()
            cv = ms.StratifiedKFold(3) if base == "mo_cls_strat" else _folds()
            search = ms.GridSearchCV(est, {'n_neighbors': [3]}, cv=cv, scoring=scoring, error_score=error_score,
                                     refit=False).fit(X, y)
        for f in range(3):  # glue: the three folds' scores
            out.append((f"{name}_split{f}", _digest(search.cv_results_[f'split{f}_test_score'])))
    except Exception as error:  # a refusal is digested by its class and words
        out.append((name + "_raised", hashlib.sha256(f"{type(error).__name__}: {error}".encode()).hexdigest()[:24]))
    return out


UNITS = ["mo_reg_accuracy", "mo_reg_accuracy_tol", "mo_reg_ari", "mo_cls_multilabel_acc", "mo_cls_multilabel_mcc",
         "mo_cls_r2", "mo_cls_strat", "one_out_knn_acc", "gmm_cvp_proba", "reg_float_ari",
         "nested_some_refused_tol"]


def main():
    with ProcessPoolExecutor(max_workers=min(len(UNITS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_unit, UNITS):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
