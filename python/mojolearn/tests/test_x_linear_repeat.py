# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Every `_mojolearn_x_linear` entry point (`x_linear_fit` for each of the
21 estimators, `x_linear_decision` through their predict/score doors),
called TWICE in one process, on the GPU binding and on the CPU host binding
(CURRENT DIRECTIVES 2026-09-27: x_cluster and x_neighbors hung on the
second GPU call of a process because each call built a DeviceContext whose
buffers outlived it; x_linear keeps ONE process-lifetime context,
x_linear/device.mojo `linear_ctx`). The two calls must return the same
bytes, and the GPU the host's.

    .pixi/envs/test/bin/python -m pytest python/mojolearn/tests/test_x_linear_repeat.py -q
"""
import array

import pytest

import mojolearn._expansion_linear as lm
from mojolearn import _backend
from mojolearn._expansion_linear import _BINDING


def _data(n=96, d=5, seed=3):
    x = seed * 2654435761 + 1
    X, y, yc, yp = [], [], [], []
    for i in range(n):
        row = []
        for _ in range(d):
            x = (x * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
            row.append(-1.0 + 2.0 * ((x >> 40) / float(1 << 24)))
        X.append(row)
        t = 0.8 * row[0] - 0.5 * row[1] + 0.25 * row[2] + 0.05 * row[3]
        y.append(t)
        yc.append(i % 3 if d > 4 and row[4] > 0.3 else int(t > 0))
        yp.append(1.0 + abs(t))
    return X, y, yc, [int(v > 0) for v in y], yp


def _cases():
    X, y, y3, y2, yp = _data()
    x1 = [r[0] for r in X]
    return [
        ("sgd-clf", lm.SGDClassifier(max_iter=20, tol=None, random_state=0), X, y2),
        ("sgd-reg", lm.SGDRegressor(max_iter=20, tol=None, random_state=0), X, y),
        ("poisson", lm.PoissonRegressor(), X, yp),
        ("gamma", lm.GammaRegressor(), X, yp),
        ("tweedie", lm.TweedieRegressor(power=1.5), X, yp),
        ("huber", lm.HuberRegressor(), X, y),
        ("bayes-ridge", lm.BayesianRidge(), X, y),
        ("ard", lm.ARDRegression(), X, y),
        ("lars", lm.Lars(), X, y),
        ("lasso-lars", lm.LassoLars(alpha=0.01), X, y),
        ("quantile", lm.QuantileRegressor(alpha=0.0), X, y),
        ("perceptron", lm.Perceptron(max_iter=20, random_state=0), X, y3),
        ("pa-clf", lm.PassiveAggressiveClassifier(max_iter=20, random_state=0), X, y2),
        ("pa-reg", lm.PassiveAggressiveRegressor(max_iter=20, random_state=0), X, y),
        ("sgd-ocsvm", lm.SGDOneClassSVM(max_iter=20, tol=None, random_state=0), X, None),
        ("ridge-clf", lm.RidgeClassifier(), X, y3),
        ("ridge-cv", lm.RidgeCV(), X, y),
        ("lasso-cv", lm.LassoCV(cv=3, n_alphas=5), X, y),
        ("enet-cv", lm.ElasticNetCV(cv=3, n_alphas=5, l1_ratio=[0.5, 0.9]), X, y),
        ("logistic-cv", lm.LogisticRegressionCV(Cs=3, cv=3, max_iter=30), X, y2),
        ("isotonic", lm.IsotonicRegression(), x1, y),
    ]


def _pin(est, module):
    """Route the estimator's x_linear calls to `module` (the host or the GPU
    binding); every other binding it asks for resolves as usual."""
    orig = est._bind
    est._bind = lambda name=None: module if (name or est._BINDING) == _BINDING else orig(name)
    return est


def _bytes(v):
    if hasattr(v, "tolist"):
        v = v.tolist()
    flat = []

    def walk(o):
        if isinstance(o, (list, tuple)):
            for e in o:
                walk(e)
        else:
            flat.append(float(o))

    walk(v)
    return bytes(array.array("d", flat))


def _run(module):
    out = []
    for name, est, X, y in _cases():
        _pin(est, module)
        est.fit(X) if y is None else est.fit(X, y)
        parts = [_bytes(est.predict(X))]
        for attr in ("coef_", "intercept_", "decision_function"):
            if attr == "decision_function":
                if hasattr(est, attr):
                    parts.append(_bytes(est.decision_function(X)))
            elif hasattr(est, attr):
                parts.append(_bytes(getattr(est, attr)))
        out.append((name, b"".join(parts)))
    return out


def _modules():
    host = _backend.load_host_module("_mojolearn_x_linear_host")
    try:
        gpu = _backend.binding(_BINDING, "identical")
    except Exception:  # a CPU-only install has no GPU binding
        gpu = None
    if gpu is not None and not hasattr(gpu, "x_linear_vendor"):
        gpu = None
    return host, gpu


def test_every_entry_twice_host_and_gpu():
    host, gpu = _modules()
    h1, h2 = _run(host), _run(host)
    assert h1 == h2, [n for (n, a), (_, b) in zip(h1, h2) if a != b]
    if gpu is None or gpu is host:
        pytest.skip("no GPU binding in this process")
    g1, g2 = _run(gpu), _run(gpu)
    assert g1 == g2, [n for (n, a), (_, b) in zip(g1, g2) if a != b]
    assert g1 == h1, [n for (n, a), (_, b) in zip(g1, h1) if a != b]
