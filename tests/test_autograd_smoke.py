# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Autograd smoke: a tiny MLP and a tiny LSTM fit through the installed bindings' default training
path (the MLP's and the recurrent models' backward runs on the autograd tape), in identical mode. Each fit must give finite
losses and predictions, and a second fit in the same process must give the same digest. Observed
digests are written to $MOJOLEARN_AUTOGRAD_SMOKE_OUT when set, so the vendor columns of one build can
be compared with each other (identical mode: the same bits on Apple, NVIDIA and AMD within one version;
never compared against an older version's digest).

Sizes are scikit-learn-test small (a correctness check, not a measurement): it runs in seconds,
almost all of it the binding load. tools/autograd_smoke.py runs it as a 5 s smoke entry."""
import hashlib
import json
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
ml = pytest.importorskip("mojolearn")


def _digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def _check(name, got, again):
    out = os.environ.get("MOJOLEARN_AUTOGRAD_SMOKE_OUT")
    if out:
        path = Path(out)
        seen = json.loads(path.read_text()) if path.is_file() else {}
        seen[name] = got
        path.write_text(json.dumps(seen, indent=1, sort_keys=True))
    assert got == again, "%s: a second identical fit gave %s, the first %s" % (name, again, got)


def _fit_mlp(X, y):
    m = ml.MLPClassifier(hidden_layer_sizes=(8,), batch_size=16, max_iter=3, shuffle=True,
                         random_state=0, tol=0.0, n_iter_no_change=1000, numeric_mode="identical")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X, y)
    return m


def test_mlp_fit_default_path():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 6)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    m = _fit_mlp(X, y)
    curve = np.asarray(m.loss_curve_, dtype=np.float64)
    assert curve.shape == (3,) and np.isfinite(curve).all()
    p = m.predict_proba(X)
    assert p.shape == (64, 2) and np.isfinite(p).all()
    m2 = _fit_mlp(X, y)
    _check("mlp", _digest(*m.coefs_, *m.intercepts_, p),
           _digest(*m2.coefs_, *m2.intercepts_, m2.predict_proba(X)))


def test_lstm_fit_default_path():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((32, 5, 4)).astype(np.float32)
    y = X[:, :, 0].sum(axis=1).astype(np.float32)
    def fit():
        return ml.LSTMRegressor(hidden_size=8, num_layers=1, batch_size=8, max_epochs=2, shuffle=True,
                                random_state=0, numeric_mode="identical").fit(X, y)
    m = fit()
    curve = np.asarray(m.loss_curve_, dtype=np.float64)
    assert m.n_iter_ == 8 and curve.shape == (8,) and np.isfinite(curve).all()
    p = m.predict(X)
    assert p.shape == (32,) and np.isfinite(p).all()
    m2 = fit()
    _check("lstm", _digest(m.params_, m.loss_curve_, p), _digest(m2.params_, m2.loss_curve_, m2.predict(X)))
