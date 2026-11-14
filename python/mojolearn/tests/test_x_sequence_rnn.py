# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The recurrent models' public state_dict round trip. The hidden sequences
against a float64 restatement of PyTorch's cell equations, the fit's loss
going down and the classifier's probabilities are checked natively in
checks/native_tests/sequence_rnn_{cpu,gpu}.mojo."""
import numpy as np

import mojolearn as ml


def _data(n=48, T=5, D=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, T, D)).astype(np.float32)
    y = (X[:, -1, 0] + 0.5 * X[:, -2, 1]).astype(np.float32)
    return X, y


def test_state_dict_round_trip():
    X, y = _data()
    m = ml.LSTMRegressor(hidden_size=4, max_epochs=1).fit(X, y)
    m2 = ml.LSTMRegressor(hidden_size=4)
    m2.n_features_in_, m2.n_outputs_, m2._y_1d = m.n_features_in_, m.n_outputs_, True
    m2.load_state_dict(m.state_dict())
    np.testing.assert_array_equal(m2.predict(X), m.predict(X))


def test_public_predict_shapes_and_labels():
    X, y = _data()
    for cls in (ml.LSTMRegressor, ml.GRURegressor, ml.RNNRegressor):
        m = cls(hidden_size=7, num_layers=2, learning_rate=1e-2, batch_size=16, max_epochs=8).fit(X, y)
        assert m.predict(X).shape == (len(X),)
    m = ml.LSTMClassifier(hidden_size=5, learning_rate=5e-2, batch_size=16, max_epochs=10).fit(
        X, (y > 0).astype(np.int64))
    assert set(m.predict(X)) <= {0, 1}
