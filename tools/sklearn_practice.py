# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""scikit-learn in practice: every public mojolearn estimator that has a
scikit-learn twin, used the way a scikit-learn user would, on a real dataset.

Test tooling, not runtime. Per estimator: a `Pipeline(StandardScaler, est)`
under `cross_val_score` (cv=3) next to the scikit-learn twin built with the
same shared parameters, a full fit then predict / transform, a pickle round
trip that must predict the same output, and `clone` + refit. Scores are shown,
never a gate; a status other than ok is a failure to fix.

    python tools/sklearn_practice.py <out.tsv> [name ...]
"""

import inspect
import pickle
import signal
import sys
import traceback
import warnings

import numpy as np
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import all_estimators

from mojolearn.tests import test_sklearn_compat as T

LIMIT_S = 240


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout(f"over {LIMIT_S}s")


def _twin(name, ours):
    sk = dict(all_estimators()).get(name)
    if sk is None:
        return None
    try:
        names = set(inspect.signature(sk.__init__).parameters) - {"self"}
        shared = {k: v for k, v in ours.get_params(deep=False).items() if k in names}
        for k, v in list(shared.items()):
            if hasattr(v, "get_params") and type(v).__module__.startswith("mojolearn"):
                shared.pop(k)  # a mojolearn sub-estimator: let the twin use its own default
        return sk(**shared)
    except Exception:
        try:
            return sk()
        except Exception:
            return None


def _data(est):
    if is_classifier(est):
        d = load_breast_cancer()
        return d.data.astype(np.float32), d.target, "breast_cancer"
    if is_regressor(est):
        d = load_diabetes()
        return d.data.astype(np.float32), d.target.astype(np.float32), "diabetes"
    d = load_iris()
    return d.data.astype(np.float32), None, "iris"


def _output(fitted, X):
    for m in ("predict", "transform", "fit_predict"):
        if hasattr(fitted, m) and m != "fit_predict":
            return m, np.asarray(getattr(fitted, m)(X))
    if hasattr(fitted, "labels_"):
        return "labels_", np.asarray(fitted.labels_)
    return "none", None


def run_one(name):
    row = dict(name=name, kind="", data="", status="ok", cv_ours="", cv_sklearn="", pickle="", clone="", error="")
    signal.alarm(LIMIT_S)
    try:
        est = T._make(name)
        row["kind"] = "classifier" if is_classifier(est) else "regressor" if is_regressor(est) else "other"
        X, y, row["data"] = _data(est)
        if y is not None:
            pipe = Pipeline([("scale", StandardScaler()), ("est", clone(est))])
            row["cv_ours"] = f"{cross_val_score(pipe, X, y, cv=3).mean():.4f}"
            twin = _twin(name, est)
            if twin is not None:
                try:
                    tp = Pipeline([("scale", StandardScaler()), ("est", twin)])
                    row["cv_sklearn"] = f"{cross_val_score(tp, X, y, cv=3).mean():.4f}"
                except Exception as e:  # the twin's failure is context only
                    row["cv_sklearn"] = f"twin-error:{type(e).__name__}"
        fitted = clone(est).fit(X, y) if y is not None else clone(est).fit(X)
        how, out = _output(fitted, X)
        back = pickle.loads(pickle.dumps(fitted))
        if out is not None:
            out2 = np.asarray(getattr(back, how)(X)) if how != "labels_" else np.asarray(back.labels_)
            row["pickle"] = "same" if np.array_equal(out, out2) else "DIFFERS"
            if row["pickle"] != "same":
                row["status"] = "pickle-differs"
        params = fitted.get_params(deep=False)
        again = clone(fitted).set_params(**params)
        again.fit(X, y) if y is not None else again.fit(X)
        row["clone"] = "refit"
    except _Timeout as e:
        row["status"], row["error"] = "timeout", str(e)
    except Exception as e:
        row["status"] = "error"
        tb = traceback.extract_tb(e.__traceback__)
        where = next((f"{f.filename.split('/mojolearn/')[-1]}:{f.lineno}" for f in reversed(tb)
                      if "/mojolearn/" in f.filename), "")
        row["error"] = f"{type(e).__name__}: {str(e)[:200]} @ {where}".replace("\t", " ").replace("\n", " ")
    finally:
        signal.alarm(0)
    return row


def main():
    out_path = sys.argv[1]
    names = sys.argv[2:] or sorted(T.SHARED_NAMES)
    signal.signal(signal.SIGALRM, _alarm)
    warnings.simplefilter("ignore")
    cols = ["name", "kind", "data", "status", "cv_ours", "cv_sklearn", "pickle", "clone", "error"]
    bad = 0
    with open(out_path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for name in names:
            row = run_one(name)
            bad += row["status"] != "ok"
            f.write("\t".join(str(row[c]) for c in cols) + "\n")
            f.flush()
            print(f"{row['status']:>14} {name} {row['cv_ours']} {row['cv_sklearn']} {row['error'][:120]}", flush=True)
    print(f"PRACTICE total={len(names)} not_ok={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
