# SPDX-License-Identifier: Apache-2.0
"""The CNN classifier's one-call fit (`x_cnn_fit_d`) and one-call predict
(`x_cnn_predict_r`) produce the same words as the per-epoch and per-chunk
entries they replace, and the admission record never changes a result."""
import contextlib

import numpy as np
import pytest


def _bindings():
    from mojolearn import _native_xcnn_bridge as bridge
    _, context = bridge.connect()
    tables = context.get("xcnn_bindings")
    if not tables:
        pytest.skip("the CNN binding is not connected")
    return [t for t in tables.values() if isinstance(t, dict)]


@contextlib.contextmanager
def _without(name):
    """Hide one compiled entry so the classifier takes its older route."""
    saved = []
    for table in _bindings():
        if name in table["symbols"]:
            saved.append((table, table["symbols"].pop(name)))
    try:
        yield bool(saved)
    finally:
        for table, fn in saved:
            table["symbols"][name] = fn


def _data(rows=150):
    rng = np.random.default_rng(3)
    X = rng.standard_normal((rows, 1, 6, 6)).astype(np.float32)
    y = (X[:, 0, :3].sum((1, 2)) > X[:, 0, 3:].sum((1, 2))).astype(int)
    return X.reshape(rows, -1), y


def _fit(optimizer, X, y):
    import mojolearn as ml
    return ml.CNNClassifier(input_shape=(1, 6, 6), conv_channels=(4,), learning_rate=0.05, batch_size=32,
                            max_iter=4, optimizer=optimizer, random_state=0).fit(X, y)


def _weight_bytes(model):
    return [np.asarray(w).tobytes() for w in model.weights()]


@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_fit_one_call_equals_epoch_calls(optimizer):
    X, y = _data()
    whole = _fit(optimizer, X, y)
    with _without("x_cnn_fit_d") as hidden:
        if not hidden:
            pytest.skip("this binding has no x_cnn_fit_d")
        epochs = _fit(optimizer, X, y)
    assert np.asarray(whole.loss_curve_).tobytes() == np.asarray(epochs.loss_curve_).tobytes()
    assert np.asarray(whole.losses_).tobytes() == np.asarray(epochs.losses_).tobytes()
    assert _weight_bytes(whole) == _weight_bytes(epochs)


@pytest.mark.parametrize("rows", [1, 150, 2048 + 37])
def test_predict_one_call_equals_chunk_calls(rows):
    X, y = _data()
    model = _fit("sgd", X, y)
    Q, _ = _data(rows)
    proba = model.predict_proba(Q)
    labels = model.predict(Q)
    with _without("x_cnn_predict_r") as hidden:
        if not hidden:
            pytest.skip("this binding has no x_cnn_predict_r")
        proba_chunks = model.predict_proba(Q)
        labels_chunks = model.predict(Q)
    assert proba.tobytes() == proba_chunks.tobytes()
    assert np.array_equal(labels, labels_chunks)


def test_admission_record_follows_field_changes():
    X, y = _data()
    model = _fit("sgd", X, y)
    first = model.predict_proba(X)
    again = model.predict_proba(X)          # a stamped admission hit
    assert first.tobytes() == again.tobytes()
    model.layers_ = "not a layer list"      # a field write bumps the generation: full admission refuses it
    with pytest.raises(TypeError):
        model.predict_proba(X)


def test_native_shape_rules_match_the_layers():
    """Conv / pool output shapes come from the native shape rules (no
    binding crossing); the shapes and the refusal are unchanged."""
    from mojolearn import _expansion_cnn as E
    x = np.zeros((2, 3, 7, 9), np.float32)
    conv = E.Conv2d(3, 5, kernel_size=3, stride=2, padding=1, dilation=1)
    assert conv.forward(x).shape == (2, 5, 4, 5)
    pool = E.MaxPool2d(2, ceil_mode=True)
    assert pool.forward(x).shape == (2, 3, 4, 5)
    big = E.Conv2d(3, 5, kernel_size=11)
    with pytest.raises(Exception, match="does not fit"):
        big.forward(x)
