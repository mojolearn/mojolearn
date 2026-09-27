# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MLPClassifier / MLPRegressor against scikit-learn's own (shuffle=False,
the same random_state, so the same initial weights and sample order): the
loss curves and predictions agree to float32 tolerance. Skipped without
scikit-learn; on the lane's pod it held to 3e-6 over Adam, SGD invscaling and
SGD adaptive (docs/lanes/progress/sequence.md)."""
import numpy as np

import mojolearn as ml


def _data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 5)).astype(np.float32)
    yr = (X[:, 0] * 2 - X[:, 1]).astype(np.float32)
    y3 = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0.5)
    return X, yr, y3


def test_against_sklearn():
    try:
        from sklearn import neural_network as sk
    except ImportError:
        return
    X, yr, y3 = _data()
    for ours, theirs, y, kw in (
            (ml.MLPRegressor, sk.MLPRegressor, yr, dict(hidden_layer_sizes=(8, 4), activation="tanh", max_iter=15)),
            (ml.MLPClassifier, sk.MLPClassifier, y3, dict(hidden_layer_sizes=(6,), solver="sgd", max_iter=15))):
        a = ours(shuffle=False, random_state=0, batch_size=50, **kw).fit(X, y)
        b = theirs(shuffle=False, random_state=0, batch_size=50, **kw).fit(X, y)
        np.testing.assert_allclose(a.loss_curve_, b.loss_curve_, rtol=1e-4, atol=1e-5)


def test_shapes_and_proba():
    X, yr, y3 = _data()
    c = ml.MLPClassifier(hidden_layer_sizes=(5,), max_iter=5, random_state=0).fit(X, y3)
    p = c.predict_proba(X)
    assert p.shape == (200, 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)
    b = ml.MLPClassifier(hidden_layer_sizes=(5,), max_iter=5, random_state=0).fit(X, y3 > 0)
    assert b.predict_proba(X).shape == (200, 2)
    r = ml.MLPRegressor(hidden_layer_sizes=(5,), max_iter=5, random_state=0).fit(X, yr)
    assert r.predict(X).shape == (200,) and r.n_iter_ == 5


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
