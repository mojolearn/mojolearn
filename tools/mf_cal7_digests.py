"""Before/after digests of CalibratedClassifierCV over the DeviceRows bases (lane mf-cal7, 2026-11-23). One line per
configuration and output: name and sha256 of the output bytes (dtype and shape included); a refusal is digested by
its exception type and words. Base: origin/lane/member-fits-7 455a6707b5 (n_jobs=1 took the door's DeviceRows fold
units `_CalUnits`: each fold side an `_arena_io.DeviceRows` in one resident scope; n_jobs>1 the typed RUN over
gathered rows), head lane/mf-cal7 (every n_jobs takes the typed RUN; on an image that gathers on the device the fold
fits and scores read X by row index in one resident scope: THE RESIDENT ROWS of
xtrees/trees_calibrated_fit_run.mojo). Test tool, never product runtime.

Units: {GaussianNB, MultinomialNB (X shifted non-negative), LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
CategoricalNB (X binned)} x {sigmoid, isotonic} x {k = 2, k = 3} x ensemble {True, False} x n_jobs {1, 4}: for each,
predict_proba and predict over X, the calibrators' words (Platt a, b; isotonic knots) and each pair member's fitted
attributes (theta_ / var_ / class_count_, or means_ / coef_ / priors_ where they exist).

EXPECTED, host column: every line equal base vs head (the n_jobs=1 lines moved from the door units to the RUN: the
same member programs over the same rows in the same order, the same calibrator glue ops), and head's n_jobs=1 line
equal to its n_jobs=4 line. L40S (device image): the same; a difference is a defect of the row-index route, not an
accepted bit change. Configurations are independent: one process each across the allocated cores.

    python tools/mf_cal7_digests.py > digests.txt
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import os
import sys

import numpy as np

BASES = ["gnb", "mnb", "lda", "qda", "cnb"]
METHODS = ["sigmoid", "isotonic"]
CLASSES = [2, 3]
ENSEMBLE = [True, False]
JOBS = [1, 4]
ATTRS = ["theta_", "var_", "class_count_", "class_prior_", "feature_log_prob_", "means_", "coef_", "intercept_",
         "priors_", "scalings_", "rotations_"]


def _digest(value):
    a = np.asarray(value)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}{a.shape}".encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:24]


def _data(base, k, n=240, d=6, seed=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (np.arange(n) % k).astype(np.int64)
    X[:, 0] += 1.25 * y
    X[:, 2] -= 0.75 * y
    if base == "mnb":
        X = X - X.min() + np.float32(0.5)
    if base == "cnb":
        X = np.floor(np.clip(X + 3.0, 0.0, 5.99)).astype(np.float32)
    return X, y


def _base(name):
    import mojolearn as ml
    return {"gnb": ml.GaussianNB, "mnb": ml.MultinomialNB, "lda": ml.LinearDiscriminantAnalysis,
            "qda": ml.QuadraticDiscriminantAnalysis, "cnb": ml.CategoricalNB}[name]()


def _cal_words(cals):
    words = []
    for kind, par in cals:
        if kind == "sigmoid":
            words.append(np.asarray(par, dtype=np.float64))
        else:
            words.append(np.asarray(par[0], dtype=np.float64))
            words.append(np.asarray(par[1], dtype=np.float64))
    return np.concatenate(words) if words else np.zeros(0)


def _run(config):
    from mojolearn.calibration import CalibratedClassifierCV
    base, method, k, ensemble, jobs = config
    X, y = _data(base, k)
    m = CalibratedClassifierCV(_base(base), method=method, cv=3, ensemble=ensemble, n_jobs=jobs).fit(X, y)
    out = [("proba", m.predict_proba(X)), ("predict", m.predict(X))]
    for p, (member, cals) in enumerate(m.calibrated_classifiers_):
        out.append((f"pair{p}_cal", _cal_words(cals)))
        for attr in ATTRS:
            value = getattr(member, attr, None)
            if value is not None:
                out.append((f"pair{p}_{attr}", np.asarray(value)))
    return out


def _unit(config):
    base, method, k, ensemble, jobs = config
    name = f"cal7_{base}_{method}_k{k}_e{int(ensemble)}_j{jobs}"
    out = []
    try:
        for suffix, value in _run(config):
            out.append((f"{name}_{suffix}", _digest(value)))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((name + "_raised", f"{type(e).__name__}: {e}"))
    return out


def _workers(units):
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    return max(1, min(units, cores))


def main():
    configs = list(itertools.product(BASES, METHODS, CLASSES, ENSEMBLE, JOBS))
    with ProcessPoolExecutor(max_workers=_workers(len(configs))) as pool:
        for lines in pool.map(_unit, configs):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
