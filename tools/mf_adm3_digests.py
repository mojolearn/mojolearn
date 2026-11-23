"""Before/after digests of the member admissions lane mf-adm3 opened (2026-11-23): class_weight ('balanced' and a
dict) on the classifier members (LogisticRegression, LinearSVC, SVC, SGDClassifier, RidgeClassifier,
LogisticRegressionCV) and kernel='precomputed' on SVC and KernelRidge, through the callers that fit members:
CalibratedClassifierCV, IterativeImputer (estimator=), AdaBoostClassifier, BaggingRegressor. Host column. Prints one
line per configuration and output: name and sha256 of the output bytes (dtype and shape included); a refusal is
digested by its words. Base: origin/lane/member-fits a8ca0c5aec (these configurations took the door paths: the
admits refused them), head lane/mf-adm3 (the native members). Test tool, never product runtime.

EQUAL (every line): 'balanced' everywhere; a dict over integer labels 0 .. k-1 in CalibratedClassifierCV (the
base's door units fitted the clones on the ensemble's codes, which are those labels); every IterativeImputer line
(the base's door rounds fit on the column's float values, the head's steps read the dict keyed by those values);
every precomputed line (fits on a square kernel; the folds' non-square rows raise the same square refusal words).

WHAT MOVES (stated, bits change):
  * `cal_*_dstr_*` (a dict keyed by STRING labels): the base's door units fitted the clones on the codes 0 .. k-1, so
    the dict's labels were absent from that y and the fit raised "class_weight contains labels absent from y" (or,
    for the x_linear kinds, "The classes, [...], are not in class_weight"; SVC "class_weight names the label ...");
    scikit-learn fits each fold on the labels, and so does the head (the table is read over `classes_`).
  * `cal_*_gap_*dict*` where the dict names the class fold 0 lacks: both raise; the head's words are
    "class_weight contains labels absent from y" for every family (the x_linear and SVC doors name the key, which a
    member fit does not hold: a words difference to send back if it matters).
Configurations are independent: one process each across the host cores.

    python tools/mf_adm3_digests.py > digests.txt
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import os
import sys

import numpy as np


def _digest(value):
    a = np.asarray(value)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}{a.shape}".encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:24]


def _data(seed, n, k, d=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    s = X @ rng.normal(size=d) + 0.3 * rng.normal(size=n)
    edges = np.quantile(s, np.linspace(0, 1, k + 1)[1:-1] ** 1.3)  # unbalanced classes
    return X, np.searchsorted(edges, s).astype(np.int64)


def _member(name, class_weight=None):
    import mojolearn
    kw = {} if class_weight is None else {"class_weight": class_weight}
    if name == "lr":
        return mojolearn.LogisticRegression(max_iter=200, **kw)
    if name == "svc":
        return mojolearn.LinearSVC(C=0.5, **kw)
    if name == "svk":
        return mojolearn.SVC(C=1.0, gamma=0.5, **kw)
    if name == "sgd":
        return mojolearn.SGDClassifier(max_iter=20, random_state=3, **kw)
    if name == "ridgec":
        return mojolearn.RidgeClassifier(alpha=0.7, **kw)
    return mojolearn.LogisticRegressionCV(Cs=3, cv=3, max_iter=100, **kw)


def _weights(kind, labels):
    if kind == "bal":
        return "balanced"
    values = (0.4, 2.5, 1.7)
    return {labels[c]: values[c] for c in range(len(labels) - 1)}  # the last class keeps 1.0


MEMBERS = ("lr", "svc", "svk", "sgd", "ridgec", "lrcv")
CAL = list(itertools.product(MEMBERS, ("bal", "dint", "dstr"), (2, 3), (True, False), (False, True)))
IMP = list(itertools.product(("lr", "svc", "svk", "sgd", "ridgec"), ("bal", "dint")))
PRE = ("ada_svc", "cal_svc", "bag_krr", "imp_krr")


def _gap_cv(y, k, n):
    """Three folds of i % 3; the last class moved into fold 0's test rows only, so fold 0's train rows lack it."""
    rows = np.arange(n)
    y = y.copy()
    y[y == k - 1] = 0
    y[rows % 3 == 0] = np.where(rows[rows % 3 == 0] % 2 == 0, k - 1, y[rows % 3 == 0])
    return y, [(rows[rows % 3 != f], rows[rows % 3 == f]) for f in range(3)]


def _cal(index):
    from mojolearn.calibration import CalibratedClassifierCV
    member, weights, k, ensemble, gap = CAL[index]
    tag = f"cal_{member}_{weights}_k{k}_e{int(ensemble)}_{'gap' if gap else 'cv3'}"
    X, codes = _data(41 + k, 210, k)
    Xt, _ = _data(97, 61, k)
    cv = 3
    if gap:
        codes, cv = _gap_cv(codes, k, len(codes))
    labels = list("abc"[:k]) if weights == "dstr" else list(range(k))
    y = np.asarray(labels, dtype=object if weights == "dstr" else np.int64)[codes]
    out = []
    try:
        m = CalibratedClassifierCV(_member(member, _weights(weights, labels)), cv=cv, ensemble=ensemble).fit(X, y)
        out.append((tag + "_predict", _digest(np.asarray(m.predict(Xt)).astype(str))))
        out.append((tag + "_proba", _digest(m.predict_proba(Xt))))
        for p, (est, cals) in enumerate(m.calibrated_classifiers_):
            for attr in ("coef_", "intercept_", "dual_coef_"):
                if hasattr(est, attr):
                    out.append((f"{tag}_pair{p}_{attr}", _digest(getattr(est, attr))))
            for j, (kind, params) in enumerate(cals):
                if kind == "sigmoid":
                    out.append((f"{tag}_pair{p}_cal{j}", _digest(np.asarray(params[:2], dtype=np.float64))))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((tag + "_raised", f"{type(e).__name__}: {e}"))
    return out


def _imp(index):
    from mojolearn.impute import IterativeImputer
    member, weights = IMP[index]
    tag = f"imp_{member}_{weights}"
    rng = np.random.default_rng(7)
    X = rng.integers(0, 3, size=(150, 4)).astype(np.float32)
    X[:, 1] = (X[:, 0] + X[:, 2]) % 3
    X[rng.random(X.shape) < 0.15] = np.nan
    out = []
    try:
        m = IterativeImputer(estimator=_member(member, _weights(weights, [0.0, 1.0, 2.0])), max_iter=3,
                             random_state=0).fit(X)
        out.append((tag + "_transform", _digest(m.transform(X))))
        for s, (feat, _, est) in enumerate(m.imputation_sequence_):
            for attr in ("coef_", "intercept_", "dual_coef_"):
                if hasattr(est, attr):
                    out.append((f"{tag}_step{s}_{feat}_{attr}", _digest(getattr(est, attr))))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((tag + "_raised", f"{type(e).__name__}: {e}"))
    return out


def _pre(index):
    import mojolearn
    from mojolearn.calibration import CalibratedClassifierCV
    from mojolearn.impute import IterativeImputer
    name = PRE[index]
    X, y = _data(11, 90, 2)
    K = (X @ X.T).astype(np.float32)
    out = []
    try:
        if name == "ada_svc":
            m = mojolearn.AdaBoostClassifier(mojolearn.SVC(kernel="precomputed"), n_estimators=3,
                                             random_state=0).fit(K, y)
            out.append((f"pre_{name}_predict", _digest(m.predict(K))))
        elif name == "cal_svc":
            m = CalibratedClassifierCV(mojolearn.SVC(kernel="precomputed"), cv=3).fit(K, y)
            out.append((f"pre_{name}_proba", _digest(m.predict_proba(K))))
        elif name == "bag_krr":
            m = mojolearn.BaggingRegressor(mojolearn.KernelRidge(kernel="precomputed"), n_estimators=3,
                                           random_state=0).fit(K, y.astype(np.float32))
            out.append((f"pre_{name}_predict", _digest(m.predict(K))))
        else:
            Kn = K.copy()
            Kn[3, 4] = np.nan
            m = IterativeImputer(estimator=mojolearn.KernelRidge(kernel="precomputed"), max_iter=2).fit(Kn)
            out.append((f"pre_{name}_transform", _digest(m.transform(Kn))))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((f"pre_{name}_raised", f"{type(e).__name__}: {e}"))
    return out


def _run(index):
    if index < len(CAL):
        return _cal(index)
    index -= len(CAL)
    if index < len(IMP):
        return _imp(index)
    return _pre(index - len(IMP))


def main():
    total = len(CAL) + len(IMP) + len(PRE)
    with ProcessPoolExecutor(max_workers=min(total, os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, range(total)):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
