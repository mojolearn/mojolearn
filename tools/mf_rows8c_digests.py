"""Before/after digests of lane mf-rows8c (2026-11-23): the search admission rows :3442, :3447, :3769 (scoring None
over a kind with no member score), host column. One line per configuration and output: name and sha256 of the output
bytes (dtype and shape included); a refusal is digested by its exception class and words. Base:
origin/lane/member-fits-8 3a9a031adc (each configuration made the search's FIT admission answer False: the door's
Python units scored it, or refused it in their words), head lane/mf-rows8c. Test tool, never product runtime.

Searches are GridSearchCV, refit=False, three fixed folds over 30 rows x 3 features, one or two candidates,
error_score='raise' unless the name ends in `_tol` (error_score=np.nan). Outputs: cv_results_'s split0..2 test
scores, or the raised words. `*_ref` lines are the same search with the named scorer ('r2' / 'accuracy'): the
head's `none_*` line must equal its `*_ref` line.

EXPECTED (base vs head):
- `none_rf_reg`, `none_knn_reg`, `none_kridge`, `none_svr`, `none_gpr`, `none_pls`, `none_knn_cls`,
  `none_nc`, `none_gpc`, `none_envelope`, `none_bag_reg`: base scored by the door's units (each estimator's
  public `score`: the same metric program over its predict); head scored by the typed 'r2' / 'accuracy' scorer.
  Equal within 1e-6 relative to base; equal to its `*_ref` line bit for bit on head.
- `none_nb`: equal on both sides (base: `_mean_accuracy_score`; head: `_own_score`; the same 'accuracy' code).
- `none_gmm`: base door units (public GaussianMixture.score); head the member's MEMBER_OP_SCORE over the same entry
  words: equal within 1e-6.
- `none_pca`: base the door's units' words; head scikit-learn's check_scoring TypeError: a stated words change.
- `foreign`: base the door's units fitted a scikit-learn Ridge through Python; head refuses natively
  (MEMBER_SET_FOREIGN): a stated behaviour change (AGENTS.md "No calls back into Python").
- `nested_raise`, `nested_tol`: base the door's units' words; head the Stacking admission's callback words, and
  scikit-learn's all-fits-failed ValueError over them.

Configurations are independent: one process each across the host cores.

    python tools/mf_rows8c_digests.py > digests.txt
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
import sys
import warnings

import numpy as np

_TESTS = [list(range(0, 10)), list(range(10, 20)), list(range(20, 30))]


def _digest(value):
    a = np.asarray(value)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}{a.shape}".encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:24]


def _data():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(30, 3)).astype(np.float32)
    yr = (X[:, 0] * 2.0 - X[:, 1] + 0.1 * rng.normal(size=30)).astype(np.float32)
    yc = (np.arange(30) % 2).astype(np.int64)
    X[:, 0] += yc
    yo = np.where(np.abs(X[:, 1]) < 1.5, 1, -1)
    return X, yr, yc, yo


class _UserSplitter:
    def split(self, X, y=None, groups=None):  # never called by the head
        every = np.arange(len(X))
        yield every[:15], every[15:]

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1


def _table(name):
    import mojolearn as ml
    X, yr, yc, yo = _data()
    t = {
        "rf_reg": (lambda: ml.RandomForestRegressor(n_estimators=4, random_state=0), {"max_depth": [2, 3]}, yr, "r2"),
        "knn_reg": (lambda: ml.KNeighborsRegressor(), {"n_neighbors": [3, 5]}, yr, "r2"),
        "kridge": (lambda: ml.KernelRidge(), {"alpha": [0.5, 1.0]}, yr, "r2"),
        "svr": (lambda: ml.SVR(), {"C": [0.5, 1.0]}, yr, "r2"),
        "gpr": (lambda: ml.GaussianProcessRegressor(), {"alpha": [1e-6, 1e-3]}, yr, "r2"),
        "pls": (lambda: ml.PLSRegression(), {"n_components": [1, 2]}, yr, "r2"),
        "bag_reg": (lambda: ml.BaggingRegressor(ml.DecisionTreeRegressor(random_state=0), n_estimators=3,
                                                random_state=0), {"max_samples": [0.8, 1.0]}, yr, "r2"),
        "knn_cls": (lambda: ml.KNeighborsClassifier(), {"n_neighbors": [3, 5]}, yc, "accuracy"),
        "nc": (lambda: ml.NearestCentroid(), {"shrink_threshold": [None]}, yc, "accuracy"),
        "gpc": (lambda: ml.GaussianProcessClassifier(), {"max_iter_predict": [50, 100]}, yc, "accuracy"),
        "nb": (lambda: ml.GaussianNB(), {"var_smoothing": [1e-9, 1e-6]}, yc, "accuracy"),
        "envelope": (lambda: ml.EllipticEnvelope(random_state=0), {"contamination": [0.1, 0.2]}, yo, "accuracy"),
    }
    return X, t[name]


_NONE = ["rf_reg", "knn_reg", "kridge", "svr", "gpr", "pls", "bag_reg", "knn_cls", "nc", "gpc", "nb", "envelope"]


def _folds(n):
    every = np.arange(n)
    return [(np.setdiff1d(every, np.array(t)), np.array(t)) for t in _TESTS]


def _search_lines(name, make_search, X, y):
    out = []
    try:
        search = make_search().fit(X, y) if y is not None else make_search().fit(X)
        for f in range(3):  # glue: the three fixed folds' scores
            out.append((f"{name}_split{f}", _digest(search.cv_results_[f'split{f}_test_score'])))
    except Exception as error:  # a refusal is digested by its class and words
        out.append((name + "_raised", hashlib.sha256(f"{type(error).__name__}: {error}".encode()).hexdigest()[:24]))
    return out


def _unit(name):
    warnings.simplefilter("ignore")
    import mojolearn as ml
    from mojolearn import model_selection as ms
    if name.startswith("none_") or name.endswith("_ref"):
        key = name[5:] if name.startswith("none_") else name[:-4]
        X, (make, grid, y, ref) = _table(key)
        scoring = None if name.startswith("none_") else ref
        return _search_lines(name, lambda: ms.GridSearchCV(make(), grid, cv=_folds(len(X)), scoring=scoring,
                                                           refit=False, error_score="raise"), X, y)
    X, yr, _, _ = _data()
    if name == "none_gmm_x":
        return _search_lines("none_gmm", lambda: ms.GridSearchCV(ml.GaussianMixture(random_state=0),
                                                                 {"n_components": [1, 2]}, cv=_folds(len(X)),
                                                                 refit=False), X, None)
    if name == "pca":
        return _search_lines("none_pca", lambda: ms.GridSearchCV(ml.PCA(), {"n_components": [1, 2]},
                                                                 cv=_folds(len(X))), X, None)
    if name == "foreign":
        from sklearn.linear_model import Ridge
        return _search_lines(name, lambda: ms.GridSearchCV(Ridge(), {"alpha": [0.5, 1.0]}, cv=_folds(len(X)),
                                                           scoring="r2", refit=False), X, yr)
    stack = ml.StackingRegressor([("r", ml.Ridge())], cv=_UserSplitter())
    tolerate = np.nan if name == "nested_tol" else "raise"
    return _search_lines(name, lambda: ms.GridSearchCV(stack, {"passthrough": [False, True]}, cv=_folds(len(X)),
                                                       scoring="r2", refit=False, error_score=tolerate), X, yr)


UNITS = (["none_" + k for k in _NONE] + [k + "_ref" for k in _NONE]
         + ["none_gmm_x", "pca", "foreign", "nested_raise", "nested_tol"])


def main():
    with ProcessPoolExecutor(max_workers=min(len(UNITS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_unit, UNITS):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
