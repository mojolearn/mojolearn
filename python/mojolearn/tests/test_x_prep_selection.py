# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Prep lane sanity: feature selection against scikit-learn."""
import numpy as np
import pytest
try:
    import pytest
    pytest.importorskip("sklearn")
except ImportError:
    pass
import sklearn.feature_selection as skfs
import mojolearn as ml


# _data, test_variance_threshold and test_scores_and_kbest moved to checks/native_tests/prep_feature_selection.mojo
# ::test_variance_threshold and ::test_scores_and_kbest (lane py-delete, 2026-11-23).


def test_mutual_info_discrete():
    rng = np.random.default_rng(6)
    n = 500
    y = rng.integers(0, 3, n)
    Xd = np.stack([rng.integers(0, 4, n), (y + rng.integers(0, 2, n)) % 4, y * 2.5,
                   np.full(n, 7)], axis=1).astype(np.float32)
    Xc = (rng.standard_normal((n, 2)) + y[:, None] * np.array([0.2, 1.0])).astype(np.float32)
    X = np.concatenate([Xd[:, :2], Xc[:, :1], Xd[:, 2:], Xc[:, 1:]], axis=1)
    yr = (X[:, 1] + 0.7 * rng.standard_normal(n)).astype(np.float32)
    Xd4 = X[:, [0, 1, 3, 4]]
    for A, df in ((Xd4, True), (X, [0, 1, 3, 4]), (X, np.array([True, True, False, True, True, False])),
                  (X, [-6, 1, 3, -2]), (X, True)):
        a = np.asarray(ml.mutual_info_classif(A, y, discrete_features=df, random_state=0))
        b = skfs.mutual_info_classif(A, y, discrete_features=df, random_state=0)
        np.testing.assert_allclose(a, b, atol=3e-3, err_msg=f"classif {df}")
        if A is X and df is True:     # continuous columns as categories: one sample per value
            for f in (ml.mutual_info_regression, skfs.mutual_info_regression):
                try:
                    f(A, yr, discrete_features=df, random_state=0)
                    raise AssertionError("no ValueError")
                except ValueError:
                    pass
            continue
        a = np.asarray(ml.mutual_info_regression(A, yr, discrete_features=df, random_state=0))
        b = skfs.mutual_info_regression(A, yr, discrete_features=df, random_state=0)
        np.testing.assert_allclose(a, b, atol=3e-3, err_msg=f"regression {df}")
    # discrete_features=False on the integer columns: exact ties, broken by the noise
    # (the reference's float64 draw, our noise word); the estimate is then a random
    # variable of the draw, so the two agree as seed means
    for f, g, t in ((ml.mutual_info_classif, skfs.mutual_info_classif, y),
                    (ml.mutual_info_regression, skfs.mutual_info_regression, yr)):
        a = np.mean([np.asarray(f(X, t, discrete_features=False, random_state=s)) for s in range(10)], axis=0)
        b = np.mean([g(X, t, discrete_features=False, random_state=s) for s in range(10)], axis=0)
        np.testing.assert_allclose(a, b, atol=2.5e-2, err_msg=f"{f.__name__} False (ties)")


if __name__ == "__main__":
    test_mutual_info_discrete()
    print("PASS test_x_prep_selection (mutual info)")
