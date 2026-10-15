# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""scikit-learn in practice: every public mojolearn estimator that has a
scikit-learn twin, used the way a scikit-learn user would, on a real dataset,
step for step next to that twin.

Test tooling, not runtime. Steps, run on ours and on the twin with the same
data and the same shared parameters:
  cv      `cross_val_score(Pipeline([scaler,] est), X, y, cv=3, error_score="raise")`
          (supervised estimators with a score)
  fit     a full fit, then the twin's own output method (predict, transform,
          fit_predict or labels_)
  pickle  ours only: a pickle round trip must give the same output
  refit   ours only: `clone` + `set_params(**get_params())` + fit
A step that fails on ours while the twin passes it is a failure to fix
(status error). A step the twin fails too is recorded as twin-fails (the
scenario does not apply to this estimator), never counted as passing.
Scores are shown, never a gate.

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
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_iris
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import all_estimators

from mojolearn.tests import test_sklearn_compat as T

LIMIT_S = 240
# Data the estimator's own contract needs (the twin gets the same).
COUNT_DATA = {"MultinomialNB", "ComplementNB", "CategoricalNB", "BernoulliNB"}
ONE_FEATURE = {"IsotonicRegression"}
MULTI_TARGET = {"MultiOutputRegressor", "MultiOutputClassifier", "RegressorChain", "ClassifierChain", "CCA",
                "PLSCanonical", "PLSRegression", "PLSSVD"}
LABEL_ONLY = {"LabelBinarizer", "LabelEncoder", "MultiLabelBinarizer"}
IRIS_CLASSES = {"QuadraticDiscriminantAnalysis"}  # full-rank class covariances
PARAMS = {"GaussianRandomProjection": {"n_components": 3}, "SparseRandomProjection": {"n_components": 3},
          "CategoricalNB": {"min_categories": 17},  # digits pixels take 0..16
          "SparseCoder": {"dictionary": np.random.default_rng(0).standard_normal((8, 4)).astype(np.float32)}}


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout(f"over {LIMIT_S}s")


def _twin(name, ours):
    sk = dict(all_estimators()).get(name)
    if sk is None:
        return None
    names = set(inspect.signature(sk.__init__).parameters) - {"self"}
    shared = {}
    for k, v in ours.get_params(deep=False).items():
        if k not in names:
            continue
        if callable(v) and not hasattr(v, "get_params") and getattr(v, "__module__", "").startswith("mojolearn"):
            continue  # a mojolearn function (score_func, kernel): the twin keeps its own default
        if hasattr(v, "get_params") and type(v).__module__.startswith("mojolearn"):
            tw = _twin(type(v).__name__, v)  # a mojolearn sub-estimator: its own twin
            if tw is None:
                continue
            v = tw
        elif isinstance(v, (list, tuple)) and v and all(isinstance(p, tuple) and len(p) == 2 for p in v):
            members = []
            for n, e in v:  # Voting/Stacking members (never `twin or e`: an ensemble's len() needs a fit)
                tw = _twin(type(e).__name__, e)
                members.append((n, e if tw is None else tw))
            v = members
        shared[k] = v
    import re
    for _ in range(len(shared) + 1):  # drop each shared value scikit-learn's own validation refuses
        try:
            twin = sk(**shared)
            if hasattr(twin, "_validate_params"):
                twin._validate_params()
            return twin
        except Exception as e:
            m = re.search(r"The '(\w+)' parameter", str(e))
            if not m or m.group(1) not in shared:
                return sk()
            shared.pop(m.group(1))
    return sk()


def _needs_y(est):
    if "score_func" in est.get_params(deep=False):
        return True  # univariate feature selection scores features against y
    p = inspect.signature(est.fit).parameters.get("y")
    return p is not None and p.default is inspect.Parameter.empty


def _data(name, est):
    if name in COUNT_DATA:
        d = load_digits()
        return d.data.astype(np.float32), d.target, "digits", False
    if name in IRIS_CLASSES:
        d = load_iris()
        return d.data.astype(np.float32), d.target, "iris", True
    if is_classifier(est):
        d = load_breast_cancer()
        X, y, tag = d.data.astype(np.float32), d.target, "breast_cancer"
    elif is_regressor(est) or _needs_y(est):
        d = load_diabetes()
        X, y, tag = d.data.astype(np.float32), d.target.astype(np.float32), "diabetes"
        if _needs_y(est) and not is_regressor(est):
            y = (y > np.median(y)).astype(np.int64)  # a supervised transformer / selector: class labels
            tag += "-binary"
    else:
        d = load_iris()
        return d.data.astype(np.float32), None, "iris", True
    if name in ONE_FEATURE:
        X, tag = X[:, :1], tag + "-1col"
    if name in MULTI_TARGET:
        rng = np.random.default_rng(0)
        y = np.column_stack([y, y[rng.permutation(len(y))]])
        tag += "-2targets"
    return X, y, tag, True


def _output_method(fitted):
    for m in ("predict", "transform"):
        if hasattr(fitted, m):
            return m
    if hasattr(fitted, "labels_"):
        return "labels_"
    return None


def _out(est, how, X):
    return np.asarray(est.labels_ if how == "labels_" else getattr(est, how)(X))


def _steps(est, X, y, scale, ours):
    """Yields (step, None | error string); stops at the first failure."""
    res = {}
    try:
        if y is not None and hasattr(est, "score"):
            pre = [("scale", StandardScaler() if scale else MinMaxScaler())]
            pipe = Pipeline(pre + [("est", clone(est))])
            res["cv"] = f"{cross_val_score(pipe, X, y, cv=3, error_score='raise').mean():.4f}"
        fitted = clone(est).fit(X, y) if y is not None else clone(est).fit(X)
        how = _output_method(fitted)
        out = _out(fitted, how, X) if how else None
        res["how"] = how
        if ours:
            back = pickle.loads(pickle.dumps(fitted))
            if out is not None and not np.array_equal(out, _out(back, how, X)):
                return res, "pickle", "pickle round trip changed the output"
            again = clone(fitted).set_params(**fitted.get_params(deep=False))
            again.fit(X, y) if y is not None else again.fit(X)
        return res, None, None
    except _Timeout:
        raise
    except Exception as e:
        step = "fit" if "cv" in res or y is None or not hasattr(est, "score") else "cv"
        tb = traceback.extract_tb(e.__traceback__)
        where = next((f"{f.filename.split('/mojolearn/')[-1]}:{f.lineno}" for f in reversed(tb)
                      if "/mojolearn/" in f.filename and "sklearn_practice" not in f.filename), "")
        msg = f"{type(e).__name__}: {str(e)[:220]} @ {where}".replace("\t", " ").replace("\n", " ")
        return res, step, msg


def run_one(name):
    row = dict(name=name, kind="", data="", status="ok", cv_ours="", cv_sklearn="", output="", error="")
    signal.alarm(LIMIT_S)
    try:
        est = T._make(name)
        if name in PARAMS:
            est.set_params(**PARAMS[name])
        row["kind"] = "classifier" if is_classifier(est) else "regressor" if is_regressor(est) else "other"
        if name in LABEL_ONLY:
            y = load_iris().target
            if name == "MultiLabelBinarizer":
                y = [(int(c),) if c else (0, 2) for c in y]  # label sets
            out = clone(est).fit(y).transform(y)
            pickle.loads(pickle.dumps(clone(est).fit(y)))
            row.update(data="iris-labels", output=f"shape={np.shape(out)}")
            return row
        X, y, row["data"], scale = _data(name, est)
        res, step, err = _steps(est, X, y, scale, True)
        row["cv_ours"], row["output"] = res.get("cv", ""), res.get("how") or ""
        twin = _twin(name, est)
        if twin is not None:
            if name in PARAMS:
                twin.set_params(**{k: v for k, v in PARAMS[name].items() if k in twin.get_params()})
            tres, tstep, terr = _steps(twin, X, y, scale, False)
            row["cv_sklearn"] = tres.get("cv", "")
        else:
            tstep, terr = None, None
        if step is not None:
            if twin is not None and tstep == step:
                row["status"], row["error"] = "twin-fails", f"{step}: ours {err} | sklearn {terr}"
            else:
                row["status"], row["error"] = "error", f"{step}: {err}"
    except _Timeout as e:
        row["status"], row["error"] = "timeout", str(e)
    except Exception as e:
        row["status"], row["error"] = "error", f"setup: {type(e).__name__}: {str(e)[:200]}"
    finally:
        signal.alarm(0)
    return row


def main():
    out_path = sys.argv[1]
    names = sys.argv[2:] or sorted(T.SHARED_NAMES)
    signal.signal(signal.SIGALRM, _alarm)
    warnings.simplefilter("ignore")
    cols = ["name", "kind", "data", "status", "cv_ours", "cv_sklearn", "output", "error"]
    counts = {}
    with open(out_path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for name in names:
            row = run_one(name)
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            f.write("\t".join(str(row[c]) for c in cols) + "\n")
            f.flush()
            print(f"{row['status']:>11} {name} {row['cv_ours']} {row['cv_sklearn']} {row['error'][:140]}", flush=True)
    print("PRACTICE total=%d %s" % (len(names), " ".join(f"{k}={v}" for k, v in sorted(counts.items()))))
    return 0 if set(counts) <= {"ok"} else 1


if __name__ == "__main__":
    sys.exit(main())
