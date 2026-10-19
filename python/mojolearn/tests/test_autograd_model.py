# SPDX-License-Identifier: Apache-2.0
"""Sequential, the model-description trainer over the autograd tape
(autograd/model.mojo): small models (scikit-learn-test sizes) whose loss
falls, prediction shapes, the same bits for the same seed across two fits,
and the refusals of bad descriptions. Runs on whichever sequence binding the
install resolves (CPU, Metal, NVIDIA or AMD); skipped when the binding
predates the model entries."""
import hashlib

import numpy as np
import pytest

import mojolearn as ml
from mojolearn import _backend


def _has_model_entries():
    try:
        return hasattr(_backend.binding("_mojolearn_x_sequence"), "model_fit")
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_model_entries(),
                                reason="the sequence binding has no model_fit entry (rebuild it)")


def _data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 5)).astype(np.float32)
    yr = (X[:, 0] * 2 - X[:, 1] + 0.5 * X[:, 2]).astype(np.float32)
    y3 = (X[:, 0] > 0).astype(np.int64) + (X[:, 1] > 0.5)
    yb = (X[:, 0] + X[:, 3] > 0).astype(np.int64)
    return X, yr, y3, yb


def _digest(model):
    return hashlib.sha256(np.ascontiguousarray(model.params_).tobytes()).hexdigest()[:16]


def test_regression_loss_falls_and_shapes():
    X, yr, _, _ = _data()
    m = ml.Sequential().linear(8).tanh().linear(1).compile(loss="mse", optimizer="adam", lr=0.02)
    m.fit(X, yr, epochs=30, batch_size=32, seed=0)
    assert m.n_iter_ == 30 and len(m.loss_curve_) == 30
    assert m.loss_curve_[-1] < 0.5 * m.loss_curve_[0]
    assert m.predict(X).shape == (120,)
    y2 = np.stack([yr, -yr], axis=1)
    m2 = ml.Sequential().linear(4).relu().linear(2).compile(loss="huber", optimizer="momentum", lr=0.01)
    m2.fit(X, y2, epochs=10, batch_size=40, seed=1)
    assert m2.predict(X).shape == (120, 2)
    assert m2.loss_curve_[-1] < m2.loss_curve_[0]


def test_layer_norm_and_residual():
    X, yr, _, _ = _data()
    m = (ml.Sequential().linear(6).layer_norm().relu().linear(6).residual(2).linear(1)
         .compile(loss="mse", optimizer="adamw", lr=0.01, weight_decay=0.01, schedule="step", step_size=5,
                  gamma=0.5))
    m.fit(X, yr, epochs=15, batch_size=30, seed=2)
    assert m.loss_curve_[-1] < m.loss_curve_[0]
    assert m.predict(X).shape == (120,)


def test_classifiers():
    X, _, y3, yb = _data()
    c = ml.Sequential().linear(8).relu().linear(3).compile(loss="cross_entropy", optimizer="adam", lr=0.02)
    c.fit(X, y3, epochs=20, batch_size=32, seed=0)
    assert c.loss_curve_[-1] < c.loss_curve_[0]
    p = c.predict_proba(X)
    assert p.shape == (120, 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)
    assert set(np.unique(c.predict(X))) <= set(c.classes_)
    b = ml.Sequential().linear(4).tanh().linear(1).compile(loss="bce_with_logits", optimizer="sgd", lr=0.1)
    b.fit(X, yb, epochs=20, batch_size=20, seed=0, shuffle=False)
    assert b.loss_curve_[-1] < b.loss_curve_[0]
    assert b.predict_proba(X).shape == (120, 2)
    assert b.predict(X).shape == (120,)


def test_same_seed_same_bits():
    X, yr, y3, _ = _data()
    for loss, y, width in (("mse", yr, 1), ("cross_entropy", y3, 3)):  # two small models
        digests, curves = [], []
        for _ in range(2):  # two fits with one seed
            m = ml.Sequential().linear(6).relu().linear(width).compile(loss=loss, lr=0.01)
            m.fit(X, y, epochs=5, batch_size=16, seed=7)
            digests.append(_digest(m))
            curves.append(m.loss_curve_)
            print(f"MODEL_DIGEST {loss} {digests[-1]}")
        assert digests[0] == digests[1] and curves[0] == curves[1]


def test_early_stop():
    X, yr, _, _ = _data()
    m = ml.Sequential().linear(1).compile(loss="mse", optimizer="sgd", lr=1e-6, tol=10.0, n_iter_no_change=2)
    m.fit(X, yr, epochs=50, batch_size=60, seed=0)
    assert m.n_iter_ < 50


@pytest.mark.parametrize("build, message", [
    (lambda: ml.Sequential().relu(), "Linear"),
    (lambda: ml.Sequential().linear(4).residual(0).linear(2).residual(-1), "width"),
    (lambda: ml.Sequential().linear(4).residual(3), "earlier layer"),
])
def test_bad_descriptions(build, message):
    X, yr, _, _ = _data()
    m = build().compile(loss="mse")
    with pytest.raises(Exception, match=message):
        m.fit(X, yr, epochs=1)


def test_bad_options():
    X, yr, y3, _ = _data()
    with pytest.raises(ValueError):
        ml.Sequential().linear(1).compile(loss="nope")
    with pytest.raises(ValueError):
        ml.Sequential().linear(0)
    with pytest.raises(ValueError):
        ml.Sequential().linear(1).fit(X, yr)  # not compiled
    with pytest.raises(ValueError, match="outputs"):
        ml.Sequential().linear(2).compile(loss="mse").fit(X, yr, epochs=1)
    with pytest.raises(Exception, match="logits"):
        ml.Sequential().linear(3).softmax().compile(loss="cross_entropy").fit(X, y3, epochs=1)
    with pytest.raises(ValueError, match="classes"):
        ml.Sequential().linear(2).compile(loss="cross_entropy").fit(X, y3, epochs=1)


def test_dropout_trains_or_is_refused():
    X, yr, _, _ = _data()
    m = ml.Sequential().linear(8).relu().dropout(0.2).linear(1).compile(loss="mse", lr=0.01)
    try:
        m.fit(X, yr, epochs=5, batch_size=32, seed=0)
    except Exception as exc:  # a build without the tape's dropout refuses by name
        assert "Dropout is not available" in str(exc)
        return
    assert m.predict(X).shape == (120,)
