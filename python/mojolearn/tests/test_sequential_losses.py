# SPDX-License-Identifier: Apache-2.0
"""Public boundary of the Sequential losses added by lane nn-losses (2026-11-23): l1, smooth_l1, nll, kl_div,
bce_with_logits with pos_weight, and reduction 'sum' / 'batchmean'. Admission, refusals in clear words,
finite small fits, and the save_checkpoint / from_checkpoint round trip of the loss, its reduction and its
scalar. The numerics are checked natively (checks/native_tests/nn_losses.mojo)."""
import numpy as np
import pytest

from mojolearn import Sequential


def data(n=64, d=4, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, d)).astype(np.float32)
    y = (0.8 * x[:, 0] - 0.5 * x[:, 1]).astype(np.float32)
    return x, y


def regressor():
    return Sequential(numeric_mode="identical").linear(6).tanh().linear(1)


@pytest.mark.parametrize("loss, kw", [
    ("l1", {}), ("smooth_l1", {"beta": 0.5}), ("smooth_l1", {"beta": 0.0, "reduction": "sum"}),
    ("mse", {"reduction": "sum"}), ("huber", {"reduction": "sum", "huber_delta": 0.5}),
])
def test_regression_losses_fit_finite(loss, kw):
    x, y = data()
    m = regressor().compile(loss=loss, optimizer="adam", lr=0.01, **kw).fit(x, y, epochs=5, seed=1)
    curve = np.asarray(m.loss_curve_)
    assert len(curve) == 5 and np.all(np.isfinite(curve))
    assert np.all(np.isfinite(m.params_))
    assert np.all(np.isfinite(m.predict(x)))


def test_bce_pos_weight_and_nll_classify():
    x, y = data()
    labels = (y > 0).astype(np.int64)
    b = regressor().compile(loss="bce_with_logits", pos_weight=2.0, lr=0.02).fit(x, labels, epochs=10, seed=2)
    assert set(np.unique(b.predict(x))) <= {0, 1}
    assert np.all(np.isfinite(b.predict_proba(x)))
    three = np.digitize(y, [-0.3, 0.3]).astype(np.int32)
    n = Sequential(numeric_mode="identical").linear(6).tanh().linear(3).compile(loss="nll", lr=0.01)
    n = n.fit(x, three, epochs=2, seed=3)
    assert np.all(np.isfinite(n.loss_curve_))
    assert n.predict(x).shape == (len(x),)


def test_kl_div_batchmean_fits():
    x, _ = data()
    rng = np.random.default_rng(4)
    p = rng.uniform(0.05, 1.0, (len(x), 3)).astype(np.float32)
    p /= p.sum(axis=1, keepdims=True)
    m = Sequential(numeric_mode="identical").linear(6).tanh().linear(3)
    m = m.compile(loss="kl_div", reduction="batchmean", lr=0.01).fit(x, p, epochs=3, seed=5)
    assert np.all(np.isfinite(m.loss_curve_))
    assert m.predict(x).shape == (len(x), 3)


@pytest.mark.parametrize("compile_kw, words", [
    ({"loss": "l1", "reduction": "average"}, "reduction"),
    ({"loss": "l1", "reduction": "batchmean"}, "batchmean"),
    ({"loss": "smooth_l1", "beta": -1.0}, "beta"),
    ({"loss": "bce_with_logits", "pos_weight": 0.0}, "pos_weight"),
])
def test_refusals_in_words(compile_kw, words):
    x, y = data()
    target = (y > 0).astype(np.int64) if compile_kw["loss"] == "bce_with_logits" else y
    with pytest.raises(Exception, match=words):
        regressor().compile(**compile_kw).fit(x, target, epochs=1)


def test_log_probability_losses_refuse_a_final_softmax():
    x, y = data()
    three = np.digitize(y, [-0.3, 0.3]).astype(np.int32)
    m = Sequential(numeric_mode="identical").linear(3).softmax().compile(loss="nll")
    with pytest.raises(Exception, match="log-probabilities"):
        m.fit(x, three, epochs=1)


@pytest.mark.parametrize("loss, kw", [("smooth_l1", {"beta": 0.25, "reduction": "sum"}), ("l1", {})])
def test_checkpoint_round_trip_keeps_the_loss(tmp_path, loss, kw):
    x, y = data()
    m = regressor().compile(loss=loss, optimizer="adam", lr=0.01, **kw).fit(x, y, epochs=2, seed=6)
    path = tmp_path / "model.mlx"
    m.save_checkpoint(str(path))
    back = Sequential.from_checkpoint(str(path))
    np.testing.assert_array_equal(back.predict(x), m.predict(x))
    # resuming the loaded model continues with the saved loss, reduction and beta: the same words as resuming
    # the original
    a = m.fit(x, y, epochs=1, resume=True)
    b = back.fit(x, y, epochs=1, resume=True)
    np.testing.assert_array_equal(a.params_, b.params_)
    np.testing.assert_array_equal(np.asarray(a.loss_curve_), np.asarray(b.loss_curve_))
