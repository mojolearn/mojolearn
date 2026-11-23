"""Before/after digests of CalibratedClassifierCV.fit, the operation lane mf-cal split into UNPACK / typed RUN / PACK
(xtrees/trees_calibrated_fit_run.mojo), host column. Prints one line per configuration and output: name and sha256
of the output bytes (dtype and shape included); a refusal is digested by its words. Round 1: base
origin/lane/member-fits 4d76a9e16e, head lane/mf-cal. Round 2 (lane mf-cal2): base d3a8864541, head lane/mf-cal2.
Every line must be equal, EXCEPT the k = 3 `gap` lines of a member whose decision has one column for two classes
(LogisticRegression, LinearSVC, LinearDiscriminantAnalysis): fold 0 holds two classes, and the base door read two
columns of that member's one-column decision (past the block's end); the head places it in the positive class's
column (bits change there; the base words were not defined). Test tool, never product runtime.

Configurations: method {sigmoid, isotonic} x ensemble {True, False} x k {2, 3, 4} x member {LogisticRegression,
LinearSVC, GaussianNB and LinearDiscriminantAnalysis (the resident DeviceRows route when the folds run on one
thread), DecisionTreeClassifier, KNeighborsClassifier, CategoricalNB (no member entry: its doors)} x n_jobs {1, 4} x
{cv=3, an explicit cv whose fold 0 train rows lack the last class (k = 2: fold 0 holds one class, its door unit;
k = 3: two classes; k = 4: three, fitted natively on them since mf-cal2)}, plus one non-finite X refusal per
member (mf-cal2: refused before any fit, in the doors' words). Outputs:
predict and predict_proba of a held-out X, and per (member, calibrators) pair: the member's scores on that X
(decision_function, else predict_proba), its coef_/intercept_ when it has them, and every calibrator word.

The configurations are independent: one process each across the host cores.

    python tools/mf_cal_digests.py > digests.txt
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


def _data(seed, n, k, d=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    s = X @ rng.normal(size=d) + 0.3 * rng.normal(size=n)
    edges = np.quantile(s, np.linspace(0, 1, k + 1)[1:-1])
    return X, np.searchsorted(edges, s).astype(np.int64)


def _member(name):
    import mojolearn
    if name == "lr":
        return mojolearn.LogisticRegression(max_iter=200)
    if name == "svc":
        return mojolearn.LinearSVC(C=0.5)
    if name == "gnb":
        return mojolearn.GaussianNB()
    if name == "lda":
        return mojolearn.LinearDiscriminantAnalysis()
    if name == "knn":
        return mojolearn.KNeighborsClassifier(n_neighbors=7)
    if name == "cnb":
        return mojolearn.CategoricalNB()
    return mojolearn.DecisionTreeClassifier(max_depth=4, random_state=1)


def _for(member, X):
    """CategoricalNB reads category indices: small non-negative integers."""
    if member == "cnb":
        return np.floor(np.abs(X) * 2.0).astype(np.float32)
    return X


MEMBERS = ("lr", "svc", "gnb", "lda", "dt", "knn", "cnb")
CONFIGS = list(itertools.product(("sigmoid", "isotonic"), (True, False), (2, 3, 4), MEMBERS, (1, 4), (False, True)))


def _gap_cv(y, k, n):
    """Three folds of i % 3; the last class moved into fold 0's test rows only, so fold 0's train rows lack it."""
    rows = np.arange(n)
    y = y.copy()
    y[y == k - 1] = 0
    y[rows % 3 == 0] = np.where(rows[rows % 3 == 0] % 2 == 0, k - 1, y[rows % 3 == 0])
    return y, [(rows[rows % 3 != f], rows[rows % 3 == f]) for f in range(3)]


def _config(index):
    from mojolearn.calibration import CalibratedClassifierCV
    method, ensemble, k, member, jobs, gap = CONFIGS[index]
    tag = f"cal_{method}_e{int(ensemble)}_k{k}_{member}_j{jobs}_{'gap' if gap else 'cv3'}"
    X, y = _data(31 + k, 240, k)
    Xt, _ = _data(77, 83, k)
    X, Xt = _for(member, X), _for(member, Xt)
    cv = 3
    if gap:
        y, cv = _gap_cv(y, k, len(y))
    out = []
    try:
        m = CalibratedClassifierCV(_member(member), method=method, cv=cv, ensemble=ensemble, n_jobs=jobs).fit(X, y)
        out.append((tag + "_predict", _digest(m.predict(Xt))))
        out.append((tag + "_proba", _digest(m.predict_proba(Xt))))
        out.append((tag + "_pairs", repr(len(m.calibrated_classifiers_))))
        for p, (est, cals) in enumerate(m.calibrated_classifiers_):
            scores = est.decision_function(Xt) if hasattr(est, "decision_function") else est.predict_proba(Xt)
            out.append((f"{tag}_pair{p}_scores", _digest(scores)))
            out.append((f"{tag}_pair{p}_classes", _digest(np.asarray(est.classes_))))
            for attr in ("coef_", "intercept_", "theta_", "var_"):
                if hasattr(est, attr):
                    out.append((f"{tag}_pair{p}_{attr}", _digest(getattr(est, attr))))
            for j, (kind, params) in enumerate(cals):
                if kind == "sigmoid":
                    words = np.asarray([params[0], params[1]], dtype=np.float64)
                else:
                    mm = int(params[2])
                    words = np.concatenate([np.asarray(params[0], dtype=np.float64)[:mm],
                                            np.asarray(params[1], dtype=np.float64)[:mm]])
                out.append((f"{tag}_pair{p}_cal{j}_{kind}", _digest(words)))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((tag + "_raised", f"{type(e).__name__}: {e}"))
    return out


def _refusal(index):
    from mojolearn.calibration import CalibratedClassifierCV
    member = MEMBERS[index]
    X, y = _data(5, 120, 2)
    X = _for(member, X)
    X[7, 1] = np.nan
    try:
        CalibratedClassifierCV(_member(member), cv=3).fit(X, y)
        return [(f"cal_nan_{member}", "no-refusal")]
    except Exception as e:  # noqa: BLE001 - the words are the digest
        return [(f"cal_nan_{member}", f"{type(e).__name__}: {e}")]


def _run(index):
    return _config(index) if index < len(CONFIGS) else _refusal(index - len(CONFIGS))


def main():
    total = len(CONFIGS) + len(MEMBERS)
    with ProcessPoolExecutor(max_workers=min(total, os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, range(total)):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
