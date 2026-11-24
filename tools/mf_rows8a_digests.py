"""Before/after digests of lane mf-rows8a (2026-11-23): the last search scorer rows, typed, host column. One line
per configuration and output: name and sha256 of the output bytes (dtype and shape included); a refusal is
digested by its exception class and words. Base: origin/lane/member-fits-8 3a9a031adc (each configuration below
made the search's FIT admission answer False: the door's units scored it), head lane/mf-rows8a (the typed RUN
scores it, or raises the words per side). Test tool, never product runtime.

Searches are GridSearchCV, one candidate, three fixed folds, refit=False, error_score='raise' unless the name ends
in `_tol` (error_score=-7.0). Outputs: cv_results_'s split0..2 test scores, or the raised words.

EXPECTED (base vs head):
- `reg_f1`, `reg_precision`, `reg_recall`, `reg_jaccard` (DecisionTreeRegressor over integral float targets
  {0, 1}): head equals scikit-learn's metric of the fold predictions; base raised the label door's "floating
  labels ... unsupported" words (the door reads no float labels): a stated behaviour change (scikit-learn serves
  it).
- `reg_f1_absent` (targets {0, 2}): head raises "pos_label=1 is not a valid label. It should be one of [0. 2.]";
  base raised the door's float-label words. `reg_f1_absent_tol`: three -7.0 on both sides.
- `bool_pos`: equal to `int_pos` on head; base: the door's units (same values).
- `brier_absent`: head raises `_get_response_values`' words "... It should be one of [0 1]"; base: the door
  scored the column against no positive (a stated behaviour change toward scikit-learn).
- `labels_empty`, `labels_float`, `labels_repeat`, `labels_repeat_absent`, `labels_kind`: the same words on both
  sides, except `labels_repeat_absent` (head "labels must be unique"; base the door's same words) and
  `labels_float` (head and base: the doors' kind words); `*_tol` three -7.0.
- `kwarg_unexpected` (+ `_tol`): the native scorer door's words on both sides.
- `vm_beta_text`: head equals `vm_beta_real`; base: the door (same values).
- `kappa_none`: base the door's TypeError where kappa is undefined; head the same words from the RUN.

Configurations are independent: one process each across the host cores.

    python tools/mf_rows8a_digests.py > digests.txt
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
import sys
import warnings

import numpy as np

_TESTS = [[0, 2, 4], [1, 3, 5], [6, 7, 8]]


def _digest(value):
    a = np.asarray(value)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}{a.shape}".encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:24]


def _reg01(high=1.0):
    X = np.arange(12, dtype=np.float32).reshape(12, 1)
    y = np.array([0, high, 0, high, 0, high, high, 0, high, 0, high, 0], dtype=np.float32)
    return X, y


def _bin_data():
    X = np.array([[0.0, 0.1], [0.2, 0.0], [1.0, 1.1], [1.2, 0.9], [0.1, 0.2], [1.1, 1.0],
                  [0.3, 0.1], [1.3, 1.2], [0.2, 0.3], [1.2, 1.1], [0.0, 0.3], [1.0, 1.3]], dtype=np.float32)
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
    return X, y


def _cls_data():
    X = np.array([[0.0, 0.1], [0.2, 0.0], [1.0, 1.1], [1.2, 0.9], [2.0, 2.1], [2.2, 1.9],
                  [0.1, 0.3], [1.1, 1.0], [2.1, 2.0], [0.3, 0.2], [1.3, 1.2], [2.3, 2.2]], dtype=np.float32)
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2], dtype=np.int32)
    return X, y


def _config(name):
    import mojolearn as ml
    from mojolearn import metrics
    from mojolearn import model_selection as ms
    base = name[:-4] if name.endswith("_tol") else name
    tree = (ml.DecisionTreeRegressor(random_state=0), {'max_depth': [None]})
    lr = (ml.LogisticRegression(max_iter=50), {'C': [1.0]})
    km = (ml.KMeans(n_clusters=3, n_init=1, random_state=0), {'n_clusters': [3]})
    Xr, yr = _reg01()
    Xa, ya = _reg01(high=2.0)
    Xb, yb = _bin_data()
    Xc, yc = _cls_data()
    f1m = lambda labels: ms.make_scorer(metrics.f1_score, average='macro', labels=labels)  # noqa: E731
    table = {
        "reg_f1": tree + (Xr, yr, "f1"),
        "reg_precision": tree + (Xr, yr, "precision"),
        "reg_recall": tree + (Xr, yr, "recall"),
        "reg_jaccard": tree + (Xr, yr, "jaccard"),
        "reg_f1_absent": tree + (Xa, ya, "f1"),
        "bool_pos": lr + (Xb, yb, ms.make_scorer(metrics.f1_score, pos_label=True)),
        "int_pos": lr + (Xb, yb, ms.make_scorer(metrics.f1_score, pos_label=1)),
        "brier_absent": lr + (Xb, yb, ms.make_scorer(metrics.brier_score_loss, response_method='predict_proba',
                                                     greater_is_better=False, pos_label=5)),
        "labels_empty": lr + (Xc, yc, f1m([])),
        "labels_float": lr + (Xc, yc, f1m([0.5])),
        "labels_repeat": lr + (Xc, yc, f1m([0, 0])),
        "labels_repeat_absent": lr + (Xc, yc, f1m([7, 7])),
        "labels_kind": lr + (Xc, yc, f1m(['a'])),
        "kwarg_unexpected": (ml.Ridge(), {'alpha': [1.0]}, Xr, yr, ms.make_scorer(metrics.r2_score,
                                                                                   not_a_parameter=1)),
        "vm_beta_text": km + (Xc, yc, ms.make_scorer(metrics.v_measure_score, beta='0.5')),
        "vm_beta_real": km + (Xc, yc, ms.make_scorer(metrics.v_measure_score, beta=0.5)),
        "kappa_none": lr + (Xb, np.zeros(12, dtype=np.int32) + (np.arange(12) >= 11),
                            ms.make_scorer(metrics.cohen_kappa_score, replace_undefined_by=None)),
    }
    est, grid, X, y, scorer = table[base]
    return est, grid, X, y, scorer, (-7.0 if name.endswith("_tol") else 'raise')


def _unit(name):
    out = []
    warnings.simplefilter("ignore")
    try:
        from mojolearn import model_selection as ms
        est, grid, X, y, scorer, error_score = _config(name)
        every = np.arange(len(X))
        folds = [(np.setdiff1d(every, np.array(t)), np.array(t)) for t in _TESTS]
        search = ms.GridSearchCV(est, grid, cv=folds, scoring=scorer, error_score=error_score, refit=False).fit(X, y)
        for f in range(3):  # glue: the three fixed folds' scores
            out.append((f"{name}_split{f}", _digest(search.cv_results_[f'split{f}_test_score'])))
    except Exception as error:  # a refusal is digested by its class and words
        out.append((name + "_raised", hashlib.sha256(f"{type(error).__name__}: {error}".encode()).hexdigest()[:24]))
    return out


UNITS = ["reg_f1", "reg_precision", "reg_recall", "reg_jaccard", "reg_f1_absent", "reg_f1_absent_tol", "bool_pos",
         "int_pos", "brier_absent", "labels_empty", "labels_empty_tol", "labels_float", "labels_repeat",
         "labels_repeat_absent", "labels_kind", "kwarg_unexpected", "kwarg_unexpected_tol", "vm_beta_text",
         "vm_beta_real", "kappa_none", "kappa_none_tol"]


def main():
    with ProcessPoolExecutor(max_workers=min(len(UNITS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_unit, UNITS):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
