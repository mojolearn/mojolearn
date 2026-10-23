# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Release gate 2026-10-22 (part 2): the metrics, model-selection helpers and
saved-model CPU prediction endpoints changed since 0.8.55, on the real binding
path: each result against a NumPy restatement on small data, and each
endpoint's refusal of a bad call.
"""
import os

import numpy as np
import pytest

import mojolearn as ml
from mojolearn import _serialize
from mojolearn import metrics
from mojolearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score, \
    permutation_test_score


def _reg_pair(n=50, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n).astype(np.float32)
    p = (y + 0.3 * rng.standard_normal(n)).astype(np.float32)
    return y, p


def _cls_pair(n=60, seed=1):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n).astype(np.int64)
    p = np.where(rng.random(n) < 0.8, y, 1 - y).astype(np.int64)
    return y, p


# ------------------------------------------------------------------ metrics

def test_mean_absolute_error_and_rmse_match_numpy():
    y, p = _reg_pair()
    d = y.astype(np.float64) - p.astype(np.float64)
    assert metrics.mean_absolute_error(y, p) == pytest.approx(np.abs(d).mean(), rel=1e-5)
    assert metrics.root_mean_squared_error(y, p) == pytest.approx(np.sqrt((d * d).mean()), rel=1e-5)


def test_mean_absolute_error_and_rmse_refusals():
    y, p = _reg_pair()
    with pytest.raises(TypeError):
        metrics.mean_absolute_error(y.astype(np.float64), p)
    with pytest.raises(ValueError):
        metrics.mean_absolute_error(y, p[:-1])
    with pytest.raises(TypeError):
        metrics.root_mean_squared_error(y, p.astype(np.float64))
    with pytest.raises(ValueError):
        metrics.root_mean_squared_error(y[:-2], p)


def test_precision_recall_f1_match_counts():
    y, p = _cls_pair()
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    assert metrics.precision_score(y, p) == pytest.approx(prec, rel=1e-6)
    assert metrics.recall_score(y, p) == pytest.approx(rec, rel=1e-6)
    assert metrics.f1_score(y, p) == pytest.approx(2 * prec * rec / (prec + rec), rel=1e-6)


def test_precision_recall_f1_refuse_bad_options():
    y, p = _cls_pair()
    with pytest.raises(ValueError):
        metrics.precision_score(y, p, average="median")
    with pytest.raises(ValueError):
        metrics.recall_score(y, p, zero_division=5)
    with pytest.raises(ValueError):
        metrics.f1_score(y, p, average="median")
    with pytest.raises(ValueError):
        metrics.precision_score(y, p[:-1])


def test_precision_recall_curve_matches_counts_at_each_threshold():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 40).astype(np.int64)
    s = (y * 0.6 + rng.random(40)).astype(np.float32)
    prec, rec, thr = (np.asarray(a, np.float64) for a in metrics.precision_recall_curve(y, s))
    assert prec.shape == rec.shape == (thr.size + 1,)
    assert prec[-1] == 1.0 and rec[-1] == 0.0
    for i, t in enumerate(thr):
        pred = s >= np.float32(t)
        tp = np.sum(pred & (y == 1))
        assert prec[i] == pytest.approx(tp / np.sum(pred), rel=1e-5)
        assert rec[i] == pytest.approx(tp / np.sum(y == 1), rel=1e-5)


def test_precision_recall_curve_refusals():
    s = np.linspace(0, 1, 6).astype(np.float32)
    with pytest.raises(ValueError):
        metrics.precision_recall_curve(np.array([2, 3, 2, 3, 2, 3]), s)
    with pytest.raises(ValueError):
        metrics.precision_recall_curve(np.array([0, 1, 0, 1, 0, 1]), s, drop_intermediate="yes")


# --------------------------------------------------------- model selection

def test_cross_val_predict_matches_per_fold_fits():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((30, 3)).astype(np.float32)
    y = (X @ np.array([1.0, -2.0, 0.5], np.float32) + 0.1 * rng.standard_normal(30)).astype(np.float32)
    got = np.asarray(cross_val_predict(ml.Ridge(alpha=1.0), X, y, cv=KFold(n_splits=3)), np.float64)
    ref = np.empty(30)
    for k in range(3):
        test = np.arange(10 * k, 10 * (k + 1))
        train = np.setdiff1d(np.arange(30), test)
        ref[test] = np.asarray(ml.Ridge(alpha=1.0).fit(X[train], y[train]).predict(X[test]))
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)


def test_cross_val_predict_refuses_parallel_jobs():
    X = np.zeros((12, 2), np.float32)
    y = np.arange(12, dtype=np.float32)
    with pytest.raises(NotImplementedError):
        cross_val_predict(ml.Ridge(), X, y, cv=3, n_jobs=2)


def test_permutation_test_score_counts_its_permutations():
    rng = np.random.default_rng(4)
    y = (np.arange(45) % 3).astype(np.int64)
    X = (rng.standard_normal((45, 3)) + y[:, None]).astype(np.float32)
    cv = StratifiedKFold(n_splits=3)
    score, perm, pvalue = permutation_test_score(ml.LinearDiscriminantAnalysis(), X, y, cv=cv,
                                                 n_permutations=5, random_state=0)
    perm = np.asarray(perm, np.float64)
    assert perm.shape == (5,)
    base = np.asarray(cross_val_score(ml.LinearDiscriminantAnalysis(), X, y, cv=cv), np.float64).mean()
    assert score == pytest.approx(base, abs=1e-6)
    assert pvalue == pytest.approx((np.sum(perm >= score) + 1.0) / 6.0)


def test_permutation_test_score_refuses_unknown_scoring():
    y = (np.arange(30) % 2).astype(np.int64)
    X = np.random.default_rng(5).standard_normal((30, 2)).astype(np.float32)
    with pytest.raises(ValueError):
        permutation_test_score(ml.LinearDiscriminantAnalysis(), X, y, cv=3, n_permutations=2,
                               scoring="not_a_scorer")


# ------------------------------------------------- saved-model CPU predict

def _forest(tmp_path):
    rng = np.random.default_rng(6)
    y = (np.arange(60) % 3).astype(np.int64)
    X = (rng.standard_normal((60, 4)) + y[:, None]).astype(np.float32)
    rf = ml.RandomForestClassifier(n_estimators=4, max_depth=5, random_state=7).fit(X, y)
    path = os.path.join(str(tmp_path), "rf.npz")
    rf.save(path)
    return rf, path, X


def test_host_predict_and_proba_match_the_fitted_forest(tmp_path):
    rf, path, X = _forest(tmp_path)
    np.testing.assert_array_equal(np.asarray(ml.host_predict(path, X)), np.asarray(rf.predict(X)))
    np.testing.assert_allclose(np.asarray(ml.host_predict_proba(path, X), np.float64),
                               np.asarray(rf.predict_proba(X), np.float64), rtol=0, atol=1e-6)


def test_host_predict_and_proba_refuse_an_unknown_format(tmp_path):
    path = os.path.join(str(tmp_path), "other.npz")
    _serialize.write_npz(path, {"format": "mojolearn-unknown-0", "estimator": "Unknown"})
    X = np.zeros((2, 4), np.float32)
    with pytest.raises(ValueError):
        ml.host_predict(path, X)
    with pytest.raises(ValueError):
        ml.host_predict_proba(path, X)
