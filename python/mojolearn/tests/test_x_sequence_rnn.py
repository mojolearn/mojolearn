# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The sequence lane's recurrent models against a float64 NumPy restatement of
PyTorch's cell equations (nn.RNN / nn.LSTM / nn.GRU), at a tolerance; and the
fit's loss going down. Runs on whatever binding the install resolves (GPU or
CPU host). Parity with torch's training itself was checked on the lane's pod."""
import numpy as np

import mojolearn as ml
from mojolearn import _x_sequence_rnn as R


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def reference_forward(model, X):
    """Top layer h_1..h_T in float64, PyTorch's equations and gate orders."""
    sd = {k: v.astype(np.float64) for k, v in model.state_dict().items()}
    cell = model._cell()
    H = model.hidden_size
    inp = X.astype(np.float64)
    n, T, _ = inp.shape
    for l in range(model.num_layers):
        Wi, Wh = sd[f"weight_ih_l{l}"], sd[f"weight_hh_l{l}"]
        bi, bh = sd[f"bias_ih_l{l}"], sd[f"bias_hh_l{l}"]
        h = np.zeros((n, H))
        c = np.zeros((n, H))
        outs = []
        for t in range(T):
            gx = inp[:, t] @ Wi.T + bi
            gh = h @ Wh.T + bh
            if cell == R.CELLS["lstm"]:
                g = gx + gh
                i, f, gg, o = (_sig(g[:, :H]), _sig(g[:, H:2 * H]), np.tanh(g[:, 2 * H:3 * H]),
                               _sig(g[:, 3 * H:]))
                c = f * c + i * gg
                h = o * np.tanh(c)
            elif cell == R.CELLS["gru"]:
                r = _sig(gx[:, :H] + gh[:, :H])
                z = _sig(gx[:, H:2 * H] + gh[:, H:2 * H])
                nn_ = np.tanh(gx[:, 2 * H:] + r * gh[:, 2 * H:])
                h = (1 - z) * nn_ + z * h
            elif cell == R.CELLS["rnn_relu"]:
                h = np.maximum(gx + gh, 0.0)
            else:
                h = np.tanh(gx + gh)
            outs.append(h)
        inp = np.stack(outs, axis=1)
    return inp


def _data(n=48, T=5, D=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, T, D)).astype(np.float32)
    y = (X[:, -1, 0] + 0.5 * X[:, -2, 1]).astype(np.float32)
    return X, y


def check_cls(cls, **kw):
    X, y = _data()
    m = cls(hidden_size=7, num_layers=2, learning_rate=1e-2, batch_size=16, max_epochs=8, **kw).fit(X, y)
    np.testing.assert_allclose(m.hidden_sequence(X), reference_forward(m, X), atol=2e-5)
    assert m.loss_curve_[-1] < m.loss_curve_[0]
    assert m.predict(X).shape == (len(X),)
    return m


def test_lstm_regressor():
    check_cls(ml.LSTMRegressor)


def test_lstm_classifier():
    X, y = _data()
    yc = (y > 0).astype(np.int64)
    m = ml.LSTMClassifier(hidden_size=5, learning_rate=5e-2, batch_size=16, max_epochs=10).fit(X, yc)
    p = m.predict_proba(X)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)
    assert m.loss_curve_[-1] < m.loss_curve_[0]
    assert set(m.predict(X)) <= {0, 1}


def test_gru_regressor():
    check_cls(ml.GRURegressor)


def test_gru_classifier():
    X, y = _data()
    m = ml.GRUClassifier(hidden_size=5, learning_rate=5e-2, batch_size=16, max_epochs=10).fit(X, (y > 0).astype(np.int64))
    np.testing.assert_allclose(m.hidden_sequence(X), reference_forward(m, X), atol=2e-5)
    assert m.loss_curve_[-1] < m.loss_curve_[0]


def test_rnn_tanh_and_relu():
    check_cls(ml.RNNRegressor)
    check_cls(ml.RNNRegressor, nonlinearity="relu")


def test_state_dict_round_trip():
    X, y = _data()
    m = ml.LSTMRegressor(hidden_size=4, max_epochs=1).fit(X, y)
    m2 = ml.LSTMRegressor(hidden_size=4)
    m2.n_features_in_, m2.n_outputs_, m2._y_1d = m.n_features_in_, m.n_outputs_, True
    m2.load_state_dict(m.state_dict())
    np.testing.assert_array_equal(m2.predict(X), m.predict(X))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
