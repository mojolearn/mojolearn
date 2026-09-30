#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/py-bugs: one digest and one wall-clock time per case the lane
touched, on whatever the install resolves (MOJOLEARN_VENDOR=cpu for the CPU
column). Run it on the base tree and on the lane's tree, same box, same job,
and diff the two JSONs (`--diff A B`): a case's digest must not move unless the
lane moved it on purpose.

    probe.py --out FILE.json [--only case,case]
    probe.py --diff BASE.json NEW.json

The lanes of tools/identity_lanes/ carry the rest (the seeded searches, the
fits); this covers the outputs no lane hashes: predict_proba of the mixtures
and LogisticRegressionCV, the Cs grid, the ext GaussianMixture's bic, the
sample_posterior draw with a user estimator, the Adam loss curve, JL, the
unseeded search's folds, learning_curve's subsets, the RNN's 2^24 cap.
"""
import argparse, hashlib, json, os, sys, time, traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
#: the tree whose mojolearn runs (the base tree's, for the before column)
sys.path.insert(0, str(Path(os.environ.get("PROBE_TREE", ROOT)) / "python"))


def H(*arrays):
    m = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        m.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return m.hexdigest()[:16]


def data(n=3000, d=6, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X[:, 1] += 2.0 * (X[:, 0] > 0)
    y3 = ((X[:, 0] > 0).astype(np.int32) + (X[:, 2] > 0.5).astype(np.int32)).astype(np.int32)
    yr = (X @ np.linspace(0.5, -0.5, d).astype(np.float32) + 0.1 * rng.standard_normal(n)).astype(np.float32)
    return X, y3, yr


CASES = {}


def case(fn):
    CASES[fn.__name__] = fn
    return fn


@case
def bgmm_proba(ml):
    X, _, _ = data()
    m = ml.BayesianGaussianMixture(n_components=4, max_iter=20, random_state=3).fit(X[:, :4])
    return H(m.predict_proba(X[:, :4]), m.predict(X[:, :4]), m.score_samples(X[:, :4]))


@case
def gmm_ext_proba_bic(ml):
    X, _, _ = data()
    m = ml.GaussianMixture(n_components=3, covariance_type="diag", n_init=2, random_state=1).fit(X[:, :4])
    return H(m.predict_proba(X[:, :4]), m.log_det_chol_ if hasattr(m, "log_det_chol_") else 0,
             np.float64(m.bic(X[:, :4])), np.float64(m.aic(X[:, :4])))


@case
def logcv_proba(ml):
    X, y3, _ = data(1200)
    out = []
    for y, cs in (((y3 > 0).astype(np.int32), 16), (y3, 10)):
        m = ml.LogisticRegressionCV(Cs=cs, cv=3, max_iter=15).fit(X, y)
        out += [np.asarray(m.Cs_, dtype=np.float64), m.coef_, m.intercept_, m.predict_proba(X)]
    return H(*out)


@case
def autoarima_bic(ml):
    rng = np.random.default_rng(5)
    e = rng.standard_normal((4, 120)).astype(np.float32)
    y = np.cumsum(e, axis=1).astype(np.float32) * np.float32(0.3) + e
    m = ml.AutoARIMA(np.ascontiguousarray(y, dtype=np.float32)).search(d=range(2), p=range(2), q=range(2), ic="bic")
    return H(np.asarray(m.order_), np.asarray(m.ic_, dtype=np.float64))


@case
def cnn_adam(ml):
    rng = np.random.default_rng(2)
    X = rng.standard_normal((256, 36)).astype(np.float32)
    y = (X[:, 14] > 0).astype(np.int32)
    m = ml.CNNClassifier(input_shape=(1, 6, 6), conv_channels=(4,), optimizer="adam", learning_rate=1e-2,
                         batch_size=32, max_iter=3, random_state=1).fit(X, y)
    return H(np.asarray(m.loss_curve_, dtype=np.float64), m.predict_proba(X))


@case
def metrics_pow(ml):
    X, y3, _ = data()
    mt = ml.metrics
    P = np.abs(X[:, :3]) + np.float32(0.1)
    P = (P / P.sum(axis=1, keepdims=True)).astype(np.float32)
    return H(np.float64(mt.d2_brier_score(y3, P)), np.float64(mt.calinski_harabasz_score(X, y3)),
             np.float64(mt.davies_bouldin_score(X, y3)))


@case
def sgkf_and_search_std(ml):
    X, y3, _ = data(600)
    ms = ml.model_selection
    groups = (np.arange(600) % 37).astype(np.int64)
    folds = [np.asarray(te) for _, te in ms.StratifiedGroupKFold(4, shuffle=True, random_state=2).split(X, y3, groups)]
    est = ml.RidgeClassifier()
    gs = ms.GridSearchCV(est, {"alpha": [0.1, 1.0, 10.0]}, cv=ms.KFold(3, shuffle=True, random_state=4)).fit(X, y3)
    return H(*folds, np.asarray(gs.cv_results_["std_test_score"]), np.asarray(gs.cv_results_["mean_test_score"]))


class _RowSpy:
    """Scores a fold by the rows it trained on: equal scores across
    candidates means equal folds."""
    _estimator_type = "regressor"

    def __init__(self, c=0):
        self.c = c

    def get_params(self, deep=False):
        return {"c": self.c}

    def set_params(self, **p):
        self.c = p.get("c", self.c)
        return self

    def fit(self, X, y=None):
        self.key_ = float(np.asarray(X, dtype=np.float64)[:, 0].sum())
        return self

    def score(self, X, y=None):
        return self.key_


@case
def search_unseeded_folds(ml):
    """A shuffling KFold with random_state=None: every candidate must see the
    same folds (scikit-learn); returns SAME / DIFFERENT, not a digest."""
    X, _, yr = data(300)
    ms = ml.model_selection
    gs = ms.GridSearchCV(_RowSpy(), {"c": [0, 1, 2, 3]}, cv=ms.KFold(3, shuffle=True), refit=False).fit(X, yr)
    rows = [np.asarray(gs.cv_results_[f"split{i}_test_score"]) for i in range(3)]
    return "SAME" if all(len(set(r.tolist())) == 1 for r in rows) else "DIFFERENT"


@case
def learning_curve_nested(ml):
    """learning_curve(shuffle=True): each fold's subsets must be nested
    prefixes of one order (scikit-learn); NESTED / NOT NESTED plus a digest."""
    X, _, yr = data(300)
    ms = ml.model_selection
    seen = []

    class Spy(_RowSpy):
        def fit(self, X, y=None):
            seen.append(np.asarray(X, dtype=np.float32)[:, 0].copy())
            return super().fit(X, y)

    sizes, tr, te = ms.learning_curve(Spy(), X, yr, cv=3, train_sizes=[0.2, 0.5, 1.0], shuffle=True,
                                      random_state=0)
    k = 3
    nested = all(np.array_equal(seen[s * k + f][:len(seen[f])], seen[f]) for f in range(k) for s in range(3))
    return ("NESTED " if nested else "NOT NESTED ") + H(np.asarray(sizes), np.asarray(tr), np.asarray(te))


class _MeanStd:
    """A user estimator with predict(return_std=True) (pure Python)."""

    def get_params(self, deep=False):
        return {}

    def fit(self, X, y):
        v = [float(t) for t in (y.tolist() if hasattr(y, "tolist") else y)]
        self.mu = sum(v) / len(v)
        self.sd = (sum((t - self.mu) * (t - self.mu) for t in v) / len(v)) ** 0.5 + 0.25
        return self

    def predict(self, X, return_std=False):
        n = len(X.tolist()) if hasattr(X, "tolist") else len(X)
        mus = [self.mu + 0.01 * i for i in range(n)]
        return (mus, [self.sd] * n) if return_std else mus


@case
def imputer_posterior_user(ml):
    X, _, _ = data(800)
    Xm = X.copy()
    Xm.reshape(-1)[::5] = np.nan
    e = ml.IterativeImputer(estimator=_MeanStd(), sample_posterior=True, random_state=3, max_iter=2,
                            min_value=-1.5, max_value=1.5)
    return H(e.fit_transform(Xm))


@case
def jl_min_dim(ml):
    from mojolearn._expansion_decomp import johnson_lindenstrauss_min_dim as jl
    vals = [jl(n, eps=e) for n in (10, 1000, 10 ** 6, 123457) for e in (0.05, 0.1, 0.3, 0.5, 0.9)]
    return H(np.asarray(vals, dtype=np.int64))


@case
def svgp(ml):
    X, _, yr = data(512)
    m = ml.SVGP(n_inducing=24, kernel_variance=2.0, lengthscale=3.0, noise_variance=0.5).fit(X, yr)
    return H(np.asarray(m.predict(X[:64])))


@case
def x_linear_score(ml):
    X, _, yr = data(2000)
    m = ml.HuberRegressor(max_iter=30).fit(X, yr)
    return H(np.float64(m.score(X, yr)))


@case
def rnn_small(ml):
    rng = np.random.default_rng(4)
    X = rng.standard_normal((200, 5, 3)).astype(np.float32)
    y = X[:, -1, :2].astype(np.float32)
    m = ml.RNNRegressor(hidden_size=6, batch_size=32, max_epochs=3, random_state=1).fit(X, y)
    return H(m.params_, np.asarray(m.loss_curve_, dtype=np.float64))


@case
def rnn_17_epochs_1m(ml):
    """17 epochs at 1M rows (schedule 17M >= 2^24): refused before the fix."""
    n = 1_000_000
    X = np.linspace(-1, 1, n, dtype=np.float32).reshape(n, 1, 1)
    y = (X[:, 0, :] * np.float32(0.5)).astype(np.float32)
    try:
        m = ml.RNNRegressor(hidden_size=2, batch_size=n, max_epochs=17, random_state=1).fit(X, y)
    except Exception as e:
        return "REFUSED " + str(e).splitlines()[0][:120]
    return "FIT " + H(m.params_, np.asarray(m.loss_curve_, dtype=np.float64))


def run(out, only):
    import mojolearn as ml
    res = {"vendor": os.environ.get("MOJOLEARN_VENDOR", "default"), "cases": {}}
    for name, fn in CASES.items():
        if only and name not in only:
            continue
        t0 = time.perf_counter()
        try:
            v = fn(ml)
        except Exception as e:
            v = "ERROR " + type(e).__name__ + ": " + str(e).splitlines()[0][:200] if str(e) else type(e).__name__
            traceback.print_exc()
        dt = time.perf_counter() - t0
        res["cases"][name] = {"value": v, "seconds": round(dt, 4)}
        print(f"{name:28s} {dt:9.3f} s  {v}", flush=True)
    Path(out).write_text(json.dumps(res, indent=1))


def diff(a, b):
    A, B = json.loads(Path(a).read_text()), json.loads(Path(b).read_text())
    print(f"{'case':28s} {'base s':>9s} {'new s':>9s}  verdict  (base -> new)")
    for k in sorted(set(A["cases"]) | set(B["cases"])):
        x, y = A["cases"].get(k, {}), B["cases"].get(k, {})
        v = "SAME" if x.get("value") == y.get("value") else "MOVED"
        print(f"{k:28s} {x.get('seconds', float('nan')):9.3f} {y.get('seconds', float('nan')):9.3f}  {v:7s}"
              + ("" if v == "SAME" else f"  ({x.get('value')} -> {y.get('value')})"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--only", default="")
    ap.add_argument("--diff", nargs=2)
    a = ap.parse_args()
    if a.diff:
        diff(*a.diff)
    else:
        run(a.out, {x for x in a.only.split(",") if x})
