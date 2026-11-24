"""Before/after digests of lane mf-rows8d (2026-11-23): the last search admission rows, host column. One line per
configuration and output: name and sha256 of the output bytes (dtype and shape included); a refusal is digested by
its exception class and words. Base: origin/lane/member-fits-8 73027304ea (each configuration made the search's
FIT admission answer False: the door's units ran it), head lane/mf-rows8d (the typed RUN runs it, or raises the
words). Test tool, never product runtime.

EXPECTED (base vs head):
- `sss_mo_cls_nested`, `sss_mo_reg_nested`: an outer GridSearchCV (one all-rows fold) over an inner GridSearchCV
  with StratifiedShuffleSplit(2, test_size=8, random_state=0) on a multi-output target; the outer's refit fits
  the inner search as a member. Head: the inner's mean_test_score equal to `sss_mo_*_own` (the inner fitted on its
  own: the door's split of the same joined labels); base: the units (the same values expected).
- `sss_mo_cls_lone`: head "The least populated classes in y have only 1 member, ..."; base the units' words.
- `cvp_mo_bool`: head rows of bool labels per row (the label decoder); base: the units' placement. Values equal.
- `search_none`: head scikit-learn's "The 'estimator' parameter of GridSearchCV must be ..."; base the units'.
- `rand_np_seed`, `rand_big_seed`: head the sampler's draws with the seed as given; base the units'. Params equal.

Configurations are independent: one process each across the host cores.

    python tools/mf_rows8d_digests.py > digests.txt
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
import sys
import warnings

import numpy as np


def _digest(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()[:24]


def _x():
    return np.linspace(0.0, 1.0, 32, dtype=np.float32).reshape(16, 2)


def _labels():
    i = np.arange(16)
    return np.stack([i % 2, (i // 2) % 2 + 3], axis=1).astype(np.int64)


def _values():
    i = np.arange(16)
    return np.stack([i % 2 * 0.5, (i // 2) % 2 + 1.25], axis=1)


def _unit(name):
    warnings.simplefilter("ignore")
    try:
        import mojolearn as ml
        from mojolearn import model_selection as ms
        X = _x()
        if name.startswith("sss_"):
            classifier = "_cls_" in name
            y = _labels() if classifier else _values()
            if name.endswith("_lone"):
                y[0] = [1, 9]
            est = ml.MultiOutputClassifier(ml.KNeighborsClassifier()) if classifier else ml.KNeighborsRegressor()
            grid = {'estimator__n_neighbors': [1, 3]} if classifier else {'n_neighbors': [1, 3]}
            inner = ms.GridSearchCV(est, grid, scoring='r2', refit=True,
                                    cv=ms.StratifiedShuffleSplit(n_splits=2, test_size=8, random_state=0))
            if name.endswith("_own"):
                return [(name, _digest(inner.fit(X, y).cv_results_['mean_test_score']))]
            every = np.arange(16)
            outer = ms.GridSearchCV(inner, {'refit': [True]}, cv=[(every, every)], scoring='r2').fit(X, y)
            return [(name, _digest(outer.best_estimator_.cv_results_['mean_test_score']))]
        if name == "cvp_mo_bool":
            folds = [(np.arange(8, 16), np.arange(8)), (np.arange(8), np.arange(8, 16))]
            got = ms.cross_val_predict(ml.MultiOutputClassifier(ml.KNeighborsClassifier(n_neighbors=3)), X,
                                       _labels() % 2 == 1, cv=folds)
            return [(name, _digest(np.array([[bool(v) for v in row] for row in got])))]
        if name == "search_none":
            ms.GridSearchCV(None, {'n_neighbors': [1]}).fit(X, np.arange(16) % 2)
            return [(name, "no refusal")]
        state = np.int64(7) if name == "rand_np_seed" else 2 ** 60 + 3
        got = ms.RandomizedSearchCV(ml.KNeighborsClassifier(), {'n_neighbors': [1, 2, 3, 4, 5]}, n_iter=3,
                                    random_state=state, cv=2).fit(X, np.arange(16) % 2)
        return [(name, hashlib.sha256(repr(got.cv_results_['params']).encode()).hexdigest()[:24]),
                (name + "_mean", _digest(got.cv_results_['mean_test_score']))]
    except Exception as error:  # a refusal is digested by its class and words
        return [(name + "_raised", hashlib.sha256(f"{type(error).__name__}: {error}".encode()).hexdigest()[:24])]


UNITS = ["sss_mo_cls_nested", "sss_mo_cls_own", "sss_mo_reg_nested", "sss_mo_reg_own", "sss_mo_cls_lone",
         "cvp_mo_bool", "search_none", "rand_np_seed", "rand_big_seed"]


def main():
    with ProcessPoolExecutor(max_workers=min(len(UNITS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_unit, UNITS):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
