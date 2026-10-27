# SPDX-License-Identifier: Apache-2.0
"""Bounded candidate public proof; needs rebuilt host + x_sequence bindings."""
from types import BuiltinFunctionType

import numpy as np
import pytest
import mojolearn as ml
from mojolearn import _backend
from mojolearn._native_xseq_bridge import connect
from mojolearn._verify_native_runtime import audit_native_call


def poison_packed_entries(monkeypatch):
    _, context = connect()
    entries = context['xseq_bindings'][_backend.default_mode()]
    assert isinstance(entries['model_native_api_v1'], BuiltinFunctionType)
    seen = []

    def forbidden(*args, **kwargs):
        seen.append(True)
        raise AssertionError('numeric model reentered a packed CPython engine')

    for name in ('rnn_fit', 'rnn_predict', 'mlp_fit', 'mlp_predict', 'model_labels'):
        monkeypatch.setitem(entries, name, forbidden)
    for name in ('uniform_init_f32', 'argmax_rows_f32', 'reduce_stat', 'threshold_labels_i64'):
        monkeypatch.setitem(context['xseq_base'], name, forbidden)
    return seen


@pytest.mark.parametrize('name', ['RNNRegressor', 'LSTMRegressor', 'GRURegressor'])
def test_recurrent_native_table_retains_fit_and_changed_prediction_inputs(monkeypatch, name):
    cls = getattr(ml, name)
    options = dict(hidden_size=4, num_layers=1, max_epochs=1, batch_size=4,
                   random_state=7, shuffle=True, numeric_mode='identical')
    x = np.linspace(-1, 1, 48, dtype='f4').reshape(8, 3, 2)
    y = x[:, -1, 0].copy()
    reference = cls(**options).fit(x, y)
    expected = reference.predict(x[:3]).copy()
    seen = poison_packed_entries(monkeypatch)
    model = cls(**options)
    audit_native_call(model.fit, x.copy(), y.copy()).assert_native()
    assert model.params_.tobytes() == reference.params_.tobytes()
    assert model.loss_curve_.tobytes() == reference.loss_curve_.tobytes()
    got = audit_native_call(model.predict, x[:3].copy()).assert_native()
    assert got.tobytes() == expected.tobytes()
    assert model.hidden_sequence(x[:2].copy()).shape[0] == 2
    assert not seen


@pytest.mark.parametrize('classes', [2, 3])
def test_mlp_native_table_retains_numeric_labels_outputs_and_refit(monkeypatch, classes):
    x = np.linspace(-1, 1, 36, dtype='f4').reshape(12, 3)
    y = (np.arange(12) % classes * 9 - 7).astype('i8')
    options = dict(hidden_layer_sizes=(4,), max_iter=2, batch_size=4,
                   random_state=7, shuffle=True, tol=0.0, n_iter_no_change=10,
                   numeric_mode='identical')
    reference = ml.MLPClassifier(**options).fit(x, y)
    expected = reference.predict(x[:5]).copy()
    probabilities = reference.predict_proba(x[:5]).copy()
    seen = poison_packed_entries(monkeypatch)
    model = ml.MLPClassifier(**options)
    audit_native_call(model.fit, x.copy(), y.copy()).assert_native()
    for actual, wanted in zip(model.coefs_ + model.intercepts_, reference.coefs_ + reference.intercepts_):
        assert actual.tobytes() == wanted.tobytes()
    assert model.best_loss_ == reference.best_loss_
    assert audit_native_call(model.predict, x[:5].copy()).assert_native().tobytes() == expected.tobytes()
    assert audit_native_call(model.predict_proba, x[:5].copy()).assert_native().tobytes() == probabilities.tobytes()
    audit_native_call(model.fit, x[::-1].copy(), y[::-1].copy()).assert_native()
    assert model.predict(x[:2]).shape == (2,)
    assert not seen
