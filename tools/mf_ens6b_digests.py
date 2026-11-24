"""Before/after digests of the search-kind refusals and routes lane mf-ens6b changed (2026-11-23), host column. One
line per configuration and output: name and sha256 of the output bytes (dtype and shape included); a refusal is
digested by its exception type and words. Base: origin/lane/member-fits-6 946945c36b (each setting below made the
search kind's admission answer False and the door's units ran or refused it), head lane/mf-ens6b. Test tool, never
product runtime.

Rows covered (see ~/mojolearn-evidence/briefs-20261122/mf-ens6b-work/rows.tsv): the search kind's empty
multi-metric scoring, refit callback, cross_val_predict method outside the options and missing predict_log_proba,
integral NumPy cv (served and < 2), the trees module import; IterativeImputer sample_posterior.

EXPECTED (stated per unit): `empty_list` / `empty_dict`: base the door's words (refit=True: "For multi-metric
scoring, refit must be False or a scorer key."), head scikit-learn's ValueError words (a behaviour change);
`empty_list_norefit`: base a fitted search (the door scored no scorer), head the same ValueError (a behaviour
change: scikit-learn fails); `refit_callback`: equal words both sides (TypeError, the door's); `cvp_transform`:
base the units' answer (LogisticRegression has no transform: an AttributeError) or words, head scikit-learn's
ValueError; `cvp_log_proba_missing`: equal AttributeError words (head raises before any fit); `np_cv2_*`: equal
scores and best params (base the door's units, head the kind; float32 last bits may differ: "bits change" on the
head only if the kind's scorer program differs from the units', stated in the result); `np_cv1`: equal KFold words;
`iterimp_posterior_*`: equal (BayesianRidge served by the member on both sides). Configurations are independent:
one process each across the host cores.

    python tools/mf_ens6b_digests.py > digests.txt
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
import sys

import numpy as np


def _digest(value):
    a = np.asarray(value)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}{a.shape}".encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:24]


def _data(n=40, d=3, seed=11):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] - 0.5 * X[:, 1] > 0).astype(np.int64)
    return X, y


def _grid(**kw):
    import mojolearn
    from mojolearn.model_selection import GridSearchCV
    return GridSearchCV(mojolearn.LogisticRegression(max_iter=50), {"C": [1.0, 0.5]}, **kw)


def _run(name):
    """The configuration's outputs as (suffix, value) pairs."""
    import mojolearn
    from mojolearn import model_selection as ms
    X, y = _data()
    if name == "empty_list":
        return [("scores", _grid(scoring=[], cv=3).fit(X, y).cv_results_["params"])]
    if name == "empty_dict":
        return [("scores", _grid(scoring={}, cv=3).fit(X, y).cv_results_["params"])]
    if name == "empty_list_norefit":
        return [("n_splits", _grid(scoring=[], cv=3, refit=False).fit(X, y).n_splits_)]
    if name == "refit_callback":
        return [("best", _grid(cv=3, refit=lambda r: 0).fit(X, y).best_index_)]
    if name == "cvp_transform":
        return [("pred", ms.cross_val_predict(mojolearn.LogisticRegression(max_iter=50), X, y, cv=3,
                                              method="transform"))]
    if name == "cvp_log_proba_missing":
        return [("pred", ms.cross_val_predict(mojolearn.Ridge(), X, y.astype(np.float32), cv=3,
                                              method="predict_log_proba"))]
    if name in ("np_cv2_grid", "np_cv2_int"):
        cv = np.int64(3) if name == "np_cv2_grid" else 3
        search = _grid(cv=cv).fit(X, y)
        return [("scores", np.asarray(search.cv_results_["mean_test_score"], dtype=np.float64)),
                ("best", np.asarray([search.best_index_])),
                ("coef", search.best_estimator_.coef_)]
    if name == "np_cv1":
        return [("scores", _grid(cv=np.int64(1)).fit(X, y).cv_results_["mean_test_score"])]
    if name in ("iterimp_posterior_br", "iterimp_posterior_default"):
        Xm = X.copy()
        Xm[::5, 1] = np.nan
        est = mojolearn.BayesianRidge() if name == "iterimp_posterior_br" else None
        imp = mojolearn.IterativeImputer(estimator=est, sample_posterior=True, max_iter=3, random_state=0)
        return [("xt", imp.fit_transform(Xm))]
    raise ValueError(name)


def _unit(name):
    out = []
    try:
        for suffix, value in _run(name):
            out.append((f"{name}_{suffix}", _digest(value)))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((name + "_raised", f"{type(e).__name__}: {e}"))
    return out


UNITS = ["empty_list", "empty_dict", "empty_list_norefit", "refit_callback", "cvp_transform", "cvp_log_proba_missing",
         "np_cv2_grid", "np_cv2_int", "np_cv1", "iterimp_posterior_br", "iterimp_posterior_default"]


def main():
    with ProcessPoolExecutor(max_workers=min(len(UNITS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_unit, UNITS):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
