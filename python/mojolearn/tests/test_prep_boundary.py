# SPDX-License-Identifier: Apache-2.0
"""The preprocessing family at the Python boundary: refusals by type. Their bits on each side are the verify lanes."""
import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
# LabelBinarizer inverse (moved from test_label_binarizer_inverse_window.py).
from mojolearn._expansion_prep import LabelBinarizer


def test_binary_rejects_three_columns():
    model = LabelBinarizer().fit([3, 7])
    try:
        model.inverse_transform(np.zeros((2, 3), dtype=np.float32))
    except ValueError:
        return
    raise AssertionError('binary inverse must reject three columns')


# ---- appended by vdv-pytest-b (2026-11-24) ----
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


# moved from python/mojolearn/tests/test_rfe_gbdt_fit_params.py
# RFE.fit(X, y, **fit_params) over a GradientBoosting member refuses with the member's own fit door's words.
import numpy as np
import pytest

import mojolearn as ml
from mojolearn._expansion_prep import RFE


def _rfeg_data(seed=3, n=160, d=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X @ np.arange(1, d + 1, dtype=np.float32) + 0.1 * rng.normal(size=n)).astype(np.float32)
    return X, y


def _rfeg_words(fit):
    with pytest.raises(Exception) as caught:
        fit()
    return f"{type(caught.value).__name__}: {caught.value}"


@pytest.mark.parametrize("params", [
    {"eval_set": [(np.zeros((4, 5), np.float32), np.zeros(4, np.float32))] * 2},
    {"eval_set": (np.zeros((4, 4), np.float32), np.zeros(4, np.float32))},
    {"eval_set": (np.zeros((4, 5), np.float32), np.zeros(3, np.float32))},
    {"pairs_weight": [1.0]},
    {"subgroup_id": np.zeros(160)},
    {"pairs": [[0, 1]], "group_id": np.zeros(160)},
])
def test_booster_refusals_are_the_fit_doors_words(params):
    X, y = _rfeg_data()
    ours = _rfeg_words(lambda: RFE(ml.GradientBoosting(n_estimators=4), n_features_to_select=5).fit(X, y, **params))
    own = _rfeg_words(lambda: ml.GradientBoosting(n_estimators=4).fit(X, y, **params))
    assert ours == own


def test_an_unknown_key_keeps_the_fit_door():
    X, y = _rfeg_data()
    ours = _rfeg_words(lambda: RFE(ml.GradientBoosting(n_estimators=4), n_features_to_select=5).fit(X, y, bogus=1))
    own = _rfeg_words(lambda: ml.GradientBoosting(n_estimators=4).fit(X, y, bogus=1))
    assert ours == own
