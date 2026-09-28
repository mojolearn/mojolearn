# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""StandardScaler / MinMaxScaler option parity (prep lane item 5): NaN in fit
and transform, StandardScaler sample_weight, partial_fit for both (the first
batch equals fit's bits) and copy=False, against scikit-learn. A NaN-free,
unweighted fit stays the binding's own path and bits (the identity lanes
standard-scaler*, minmax-scaler* hold those). Run on the GPU binding and with
MOJOLEARN_VENDOR=cpu on the host binding."""
import os
import tempfile

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn import preprocessing as skp  # noqa: E402

import mojolearn as ml  # noqa: E402

RTOL, ATOL = 2e-5, 2e-5


def _data(seed=3, n=257, d=5, nan_every=7):
    rng = np.random.default_rng(seed)
    X = (rng.normal(size=(n, d)) * [1, 10, 0.1, 3, 100] + [0, 5, -2, 0, 1000]).astype(np.float32)[:, :d]
    X[:, 3] = 2.5                       # exactly constant: variance zero, scale one
    Xn = X.copy()
    Xn[::nan_every, 0] = np.nan
    Xn[1::nan_every + 4, 2] = np.nan
    return X, Xn


def _b(a):
    return np.asarray(a, dtype=np.float32).tobytes()


def _close(ours, ref, msg):
    np.testing.assert_allclose(np.asarray(ours, dtype=np.float64), np.asarray(ref, dtype=np.float64),
                               rtol=RTOL, atol=ATOL, err_msg=msg)


def test_standard_nan_fit_and_transform():
    X, Xn = _data()
    ours = ml.StandardScaler().fit(Xn)
    ref = skp.StandardScaler().fit(Xn.astype(np.float64))
    _close(ours.mean_, ref.mean_, "mean_")
    _close(ours.var_, ref.var_, "var_")
    _close(ours.scale_, ref.scale_, "scale_")
    assert np.array_equal(np.asarray(ours.n_samples_seen_), ref.n_samples_seen_)
    assert np.asarray(ours.var_)[3] == 0 and np.asarray(ours.scale_)[3] == 1
    T = np.asarray(ours.transform(Xn))
    R = ref.transform(Xn.astype(np.float64))
    assert np.array_equal(np.isnan(T), np.isnan(Xn))
    _close(np.nan_to_num(T), np.nan_to_num(R), "transform")
    I = np.asarray(ours.inverse_transform(T))
    assert np.array_equal(np.isnan(I), np.isnan(Xn))
    # A finite entry takes the binding's own transform arithmetic: the same
    # bits as the NaN-free transform of the same row.
    clean = np.asarray(ours.transform(X))
    ok = ~np.isnan(Xn)
    assert _b(T[ok]) == _b(clean[ok])


def test_standard_nan_free_fit_is_the_binding_path():
    X, _ = _data()
    a = ml.StandardScaler().fit(X)
    assert isinstance(a.n_samples_seen_, int) and a.n_samples_seen_ == X.shape[0]
    b = ml.StandardScaler().partial_fit(X)
    for name in ("mean_", "var_", "scale_"):
        assert _b(getattr(a, name)) == _b(getattr(b, name)), name
    assert b.n_samples_seen_ == a.n_samples_seen_


def test_standard_sample_weight():
    X, Xn = _data()
    w = np.random.default_rng(5).uniform(0, 3, size=X.shape[0]).astype(np.float32)
    w[::11] = 0
    for data in (X, Xn):
        ours = ml.StandardScaler().fit(data, sample_weight=w)
        ref = skp.StandardScaler().fit(data.astype(np.float64), sample_weight=w.astype(np.float64))
        _close(ours.mean_, ref.mean_, "weighted mean_")
        _close(ours.var_, ref.var_, "weighted var_")
        _close(ours.scale_, ref.scale_, "weighted scale_")
        np.testing.assert_allclose(np.asarray(ours.n_samples_seen_, dtype=np.float64),
                                   ref.n_samples_seen_, rtol=1e-6)
        assert np.asarray(ours.scale_)[3] == 1
    # unit weights: the reference's unweighted statistics
    ones = ml.StandardScaler().fit(X, sample_weight=np.ones(X.shape[0], np.float32))
    _close(ones.mean_, skp.StandardScaler().fit(X.astype(np.float64)).mean_, "unit weights")
    # a scalar broadcasts
    s = ml.StandardScaler().fit(X, sample_weight=2.0)
    _close(s.mean_, ones.mean_, "scalar weight")


def test_standard_partial_fit():
    X, Xn = _data(n=600)
    w = np.random.default_rng(9).uniform(0.5, 2, size=X.shape[0]).astype(np.float32)
    for data, weight in ((X, None), (Xn, None), (X, w), (Xn, w)):
        ours, ref = ml.StandardScaler(), skp.StandardScaler()
        for a, b in ((0, 100), (100, 350), (350, 600)):
            sw = None if weight is None else weight[a:b]
            ours.partial_fit(data[a:b], sample_weight=sw)
            ref.partial_fit(data[a:b].astype(np.float64),
                            sample_weight=None if sw is None else sw.astype(np.float64))
        _close(ours.mean_, ref.mean_, "partial mean_")
        _close(ours.var_, ref.var_, "partial var_")
        _close(ours.scale_, ref.scale_, "partial scale_")
        np.testing.assert_allclose(np.asarray(ours.n_samples_seen_, dtype=np.float64),
                                   np.asarray(ref.n_samples_seen_, dtype=np.float64), rtol=1e-6)
    # the first batch is fit, bit for bit
    first = ml.StandardScaler().partial_fit(Xn[:100])
    fit = ml.StandardScaler().fit(Xn[:100])
    for name in ("mean_", "var_", "scale_"):
        assert _b(getattr(first, name)) == _b(getattr(fit, name)), name
    # with_mean=False / with_std=False keep only what they use
    nm = ml.StandardScaler(with_std=False).partial_fit(X[:100]).partial_fit(X[100:])
    assert nm.var_ is None and nm.scale_ is None
    _close(nm.mean_, X.astype(np.float64).mean(0), "with_std=False mean_")
    none = ml.StandardScaler(with_mean=False, with_std=False).partial_fit(X[:100]).partial_fit(X[100:])
    assert none.mean_ is None and none.n_samples_seen_ == X.shape[0]


def test_standard_all_nan_feature():
    X, _ = _data()
    X = X.copy()
    X[:, 1] = np.nan
    ours = ml.StandardScaler().fit(X)
    assert np.isnan(np.asarray(ours.mean_)[1]) and np.isnan(np.asarray(ours.scale_)[1])
    assert list(np.asarray(ours.n_samples_seen_))[1] == 0
    T = np.asarray(ours.transform(np.nan_to_num(X)))
    assert np.isnan(T[:, 1]).all() and not np.isnan(np.delete(T, 1, axis=1)).any()
    # a later batch that sees it takes the batch's statistics
    Y, _ = _data(seed=4)
    ours.partial_fit(Y)
    ref = skp.StandardScaler().fit(Y.astype(np.float64))
    _close(np.asarray(ours.mean_)[1], ref.mean_[1], "first sighting")


def test_minmax_nan_and_partial_fit():
    X, Xn = _data()
    for clip in (False, True):
        ours = ml.MinMaxScaler(feature_range=(-1, 2), clip=clip).fit(Xn)
        ref = skp.MinMaxScaler(feature_range=(-1, 2), clip=clip).fit(Xn.astype(np.float64))
        for name in ("data_min_", "data_max_", "data_range_", "scale_", "min_"):
            _close(getattr(ours, name), getattr(ref, name), name)
        T = np.asarray(ours.transform(Xn))
        assert np.array_equal(np.isnan(T), np.isnan(Xn))
        _close(np.nan_to_num(T), np.nan_to_num(ref.transform(Xn.astype(np.float64))), "minmax transform")
    # the NaN fit's extrema are those of the non-NaN entries: fit on the
    # rows without NaN in any column that has one gives the same bits
    rows = ~np.isnan(Xn).any(axis=1)
    a = ml.MinMaxScaler().fit(np.where(np.isnan(Xn), np.nanmin(Xn, axis=0), Xn).astype(np.float32))
    b = ml.MinMaxScaler().fit(Xn)
    for name in ("data_min_", "data_max_", "scale_", "min_"):
        assert _b(getattr(a, name)) == _b(getattr(b, name)), name
    assert rows.any()
    # partial_fit: the running extrema; first batch == fit; three batches == one fit on the concatenation
    first = ml.MinMaxScaler().partial_fit(X[:50])
    fit = ml.MinMaxScaler().fit(X[:50])
    for name in ("data_min_", "data_max_", "data_range_", "scale_", "min_"):
        assert _b(getattr(first, name)) == _b(getattr(fit, name)), name
    p = ml.MinMaxScaler()
    for a0, b0 in ((0, 50), (50, 120), (120, X.shape[0])):
        p.partial_fit(X[a0:b0])
    whole = ml.MinMaxScaler().fit(X)
    for name in ("data_min_", "data_max_", "data_range_", "scale_", "min_"):
        assert _b(getattr(p, name)) == _b(getattr(whole, name)), name
    assert p.n_samples_seen_ == X.shape[0]
    # np.minimum semantics: a batch with an all-NaN column poisons it
    Z = X[:20].copy()
    Z[:, 4] = np.nan
    q = ml.MinMaxScaler().fit(X[20:]).partial_fit(Z)
    r = skp.MinMaxScaler().fit(X[20:].astype(np.float64)).partial_fit(Z.astype(np.float64))
    assert np.isnan(np.asarray(q.scale_)[4]) and np.isnan(r.scale_[4])


def test_copy_false_writes_in_place():
    X, Xn = _data()
    for est in (ml.StandardScaler(copy=False), ml.MinMaxScaler(copy=False)):
        est.fit(X)
        expect = np.asarray(type(est)().fit(X).transform(X))
        buf = X.copy()
        out = est.transform(buf)
        assert out is buf
        assert _b(buf) == _b(expect)
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
    assert _b(buf) == _b(X)


def test_inf_refused_nan_allowed():
    X, _ = _data()
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
    _, Xn = _data()
    w = np.random.default_rng(1).uniform(0.5, 2, size=Xn.shape[0]).astype(np.float32)
    for est in (ml.StandardScaler().fit(Xn), ml.StandardScaler().fit(Xn, sample_weight=w),
                ml.StandardScaler().fit(np.nan_to_num(Xn), sample_weight=w)):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.npz")
            est.save(path)
            back = ml.StandardScaler.load(path)
        assert np.array_equal(np.asarray(back.n_samples_seen_), np.asarray(est.n_samples_seen_))
        assert type(back.n_samples_seen_) is type(est.n_samples_seen_)
        assert _b(back.transform(Xn)) == _b(est.transform(Xn))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
