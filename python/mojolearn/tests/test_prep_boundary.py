# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Python API boundary tests of the prep family: refusals, exception types and words, returned types,
buffer admission and framework interop. Numerical results are held by the verify lanes."""


# moved from test_x_prep_scaler_options.py (vdv-pytest-b, 2026-11-24)
import os
import tempfile

import numpy as np
import pytest

import mojolearn as ml


def _so_data(seed=3, n=257, d=5, nan_every=7):
    rng = np.random.default_rng(seed)
    X = (rng.normal(size=(n, d)) * [1, 10, 0.1, 3, 100] + [0, 5, -2, 0, 1000]).astype(np.float32)[:, :d]
    X[:, 3] = 2.5                       # exactly constant: variance zero, scale one
    Xn = X.copy()
    Xn[::nan_every, 0] = np.nan
    Xn[1::nan_every + 4, 2] = np.nan
    return X, Xn


def _so_b(a):
    return np.asarray(a, dtype=np.float32).tobytes()


def test_standard_nan_free_fit_is_the_binding_path():
    X, _ = _so_data()
    a = ml.StandardScaler().fit(X)
    assert isinstance(a.n_samples_seen_, int) and a.n_samples_seen_ == X.shape[0]
    b = ml.StandardScaler().partial_fit(X)
    for name in ("mean_", "var_", "scale_"):
        assert _so_b(getattr(a, name)) == _so_b(getattr(b, name)), name
    assert b.n_samples_seen_ == a.n_samples_seen_


def test_copy_false_writes_in_place():
    X, Xn = _so_data()
    for est in (ml.StandardScaler(copy=False), ml.MinMaxScaler(copy=False)):
        est.fit(X)
        expect = np.asarray(type(est)().fit(X).transform(X))
        buf = X.copy()
        out = est.transform(buf)
        assert out is buf
        assert _so_b(buf) == _so_b(expect)
        bufn = Xn.copy()
        outn = est.transform(bufn)
        assert outn is bufn and np.array_equal(np.isnan(bufn), np.isnan(Xn))
    # a float64 input is converted, so it is not written (the reference copies too)
    X64 = X.astype(np.float64)
    before = X64.copy()
    s = ml.StandardScaler(copy=False).fit(X)
    with pytest.raises(TypeError):
        s.transform(X64)
    assert np.array_equal(X64, before)
    # StandardScaler.transform(copy=True) on a copy=False scaler leaves X alone
    buf = X.copy()
    s.transform(buf, copy=True)
    assert _so_b(buf) == _so_b(X)


def test_inf_refused_nan_allowed():
    X, _ = _so_data()
    Xi = X.copy()
    Xi[3, 1] = np.inf
    for est in (ml.StandardScaler(), ml.MinMaxScaler()):
        with pytest.raises(ValueError):
            est.fit(Xi)
    s = ml.StandardScaler().fit(X)
    with pytest.raises(ValueError):
        s.transform(Xi)
    with pytest.raises(ValueError):
        ml.StandardScaler().fit(X, sample_weight=np.full(X.shape[0], np.nan, np.float32))


def test_save_load_vector_counts():
    _, Xn = _so_data()
    w = np.random.default_rng(1).uniform(0.5, 2, size=Xn.shape[0]).astype(np.float32)
    for est in (ml.StandardScaler().fit(Xn), ml.StandardScaler().fit(Xn, sample_weight=w),
                ml.StandardScaler().fit(np.nan_to_num(Xn), sample_weight=w)):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.npz")
            est.save(path)
            back = ml.StandardScaler.load(path)
        assert np.array_equal(np.asarray(back.n_samples_seen_), np.asarray(est.n_samples_seen_))
        assert type(back.n_samples_seen_) is type(est.n_samples_seen_)
        assert _so_b(back.transform(Xn)) == _so_b(est.transform(Xn))


# moved from test_x_prep_selection.py (vdv-pytest-b, 2026-11-24): the refusal part of
# test_score_edges_and_rfe_getter (its scores and rankings are verify lane x-prep-score-edges).
def test_rfe_callable_importance_getter_refused():
    rng = np.random.default_rng(11)
    n = 300
    y = rng.integers(0, 3, n)
    Xg = (rng.standard_normal((n, 8)) + y[:, None] * np.linspace(0, 1.4, 8)).astype(np.float32)
    with pytest.raises(TypeError, match="a callable is not supported"):
        ml.RFE(ml.LinearDiscriminantAnalysis(), importance_getter=lambda e: e.coef_[0]).fit(Xg, y)
