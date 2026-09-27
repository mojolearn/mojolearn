# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Prep lane sanity: TargetEncoder against scikit-learn. `fit` + `transform`
at a float32 tolerance (continuous, binary, multiclass; auto and fixed
smooth); `fit_transform` against the reference's own cross-fit arithmetic
on OUR folds (the fold draw is ours by design)."""
import numpy as np
try:
    import pytest
    pytest.importorskip("sklearn")
except ImportError:
    pass
from sklearn.preprocessing import TargetEncoder as SkTE
import mojolearn as ml


def _data(seed=0, n=400, d=3):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 5, (n, d)).astype(np.float32)
    y = (X[:, 0] * 0.5 + rng.standard_normal(n)).astype(np.float32)
    return X, y


def test_fit_transform_like_reference():
    X, y = _data()
    Xh, _ = _data(1)
    Xh[0, 0] = 9
    yb = (y > 0.8).astype(np.int64)
    ym = np.digitize(y, [0.0, 1.0]).astype(np.int64)
    for target, kw in ((y, dict(smooth="auto")), (y, dict(smooth=2.0)), (yb, {}), (ym, {})):
        m = ml.TargetEncoder(**kw).fit(X, target)
        r = SkTE(**kw).fit(X, target)
        np.testing.assert_allclose(np.asarray(m.transform(Xh)), r.transform(Xh), rtol=2e-4, atol=2e-5)


def test_cross_fit_uses_fold_encodings():
    X, y = _data(3)
    m = ml.TargetEncoder(random_state=7)
    out = np.asarray(m.fit_transform(X, y))
    from mojolearn._expansion_prep import _kfold_assignment
    folds = np.asarray(_kfold_assignment(len(y), 5, 7))
    for k in range(5):
        tr, te = folds != k, folds == k
        r = SkTE().fit(X[tr], y[tr])
        np.testing.assert_allclose(out[te], r.transform(X[te]), rtol=2e-4, atol=2e-5)


def test_stratified_folds_unshuffled_are_the_reference():
    from sklearn.model_selection import StratifiedKFold
    from mojolearn._expansion_prep import _stratified_assignment
    X, y = _data(4)
    yb = (y > 0.8).astype(np.int64)
    ym = np.digitize(y, [0.0, 1.0]).astype(np.int64)[::-1].copy()
    for t in (yb, ym):
        want = np.empty(len(t), dtype=int)
        for k, (_tr, te) in enumerate(StratifiedKFold(5).split(X, t)):
            want[te] = k
        np.testing.assert_array_equal(np.asarray(_stratified_assignment(t.tolist(), 5, 0, False)), want)
        got = np.asarray(ml.TargetEncoder(shuffle=False).fit_transform(X, t))
        np.testing.assert_allclose(got, SkTE(shuffle=False).fit_transform(X, t), rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(np.asarray(ml.TargetEncoder(shuffle=False).fit_transform(X, y)),
                               SkTE(shuffle=False).fit_transform(X, y), rtol=2e-4, atol=2e-5)
    # shuffled: every class spread over the folds as evenly as the reference's allocation
    f = np.asarray(_stratified_assignment(ym.tolist(), 5, 11, True))
    for c in np.unique(ym):
        cnt = np.bincount(f[ym == c], minlength=5)
        assert cnt.max() - cnt.min() <= 1


if __name__ == "__main__":
    test_fit_transform_like_reference()
    test_cross_fit_uses_fold_encodings()
    test_stratified_folds_unshuffled_are_the_reference()
    print("PASS test_x_prep_target_encoder")
