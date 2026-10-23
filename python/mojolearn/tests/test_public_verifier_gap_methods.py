# SPDX-License-Identifier: Apache-2.0
"""Independent public-output checks for changed endpoints lacking verifier lanes."""
import math

import numpy as np
import pytest

import mojolearn as ml
from mojolearn import training


@pytest.mark.parametrize("cls", [ml.DynamicTheta, ml.OptimizedTheta])
@pytest.mark.parametrize("season", [1, 12])
def test_theta_variant_batch_forecasts_preserve_single_series(cls, season):
    time = np.arange(48, dtype=np.float32)
    y = np.stack([20 + time * np.float32(0.1), 30 + np.sin(time * np.float32(0.4))]).astype("f4")
    batch = np.asarray(cls(season_length=season, alpha=0.3).fit(y).predict(7)["mean"])
    assert batch.shape == (2, 7)
    for row in range(2):
        single = np.asarray(cls(season_length=season, alpha=0.3).fit(y[row]).predict(7)["mean"])
        np.testing.assert_array_equal(batch[row], single.reshape(-1))
    assert np.isfinite(batch).all()


@pytest.mark.parametrize("n,eps", [(1, 0.2), (17, 0.25), (1000, 0.4)])
def test_johnson_lindenstrauss_dimension_matches_stated_bound(n, eps):
    expected = int(4 * math.log(n) / (eps**2 / 2 - eps**3 / 3))
    assert ml.johnson_lindenstrauss_min_dim(n, eps=eps) == expected


@pytest.mark.parametrize("n,eps", [(0, 0.2), (5, 0.0), (5, 1.0)])
def test_johnson_lindenstrauss_refuses_invalid_bound_parameters(n, eps):
    with pytest.raises(ValueError):
        ml.johnson_lindenstrauss_min_dim(n, eps=eps)


@pytest.mark.parametrize("algorithm", ["threshold", "lasso_cd", "lasso_lars"])
def test_sparse_encode_orthonormal_dictionary_matches_soft_threshold(algorithm):
    rows = np.array([[2, -0.75, 0.0625], [-0.5, 1, -3]], dtype="f4")
    expected = np.sign(rows) * np.maximum(np.abs(rows) - np.float32(0.125), np.float32(0))
    got = ml.sparse_encode(rows, np.eye(3, dtype="f4"), algorithm=algorithm,
                           alpha=0.125, max_iter=100, numeric_mode="identical")
    np.testing.assert_allclose(np.asarray(got), expected, rtol=0, atol=1e-6)


def test_sparse_encode_omp_keeps_largest_orthonormal_atom():
    rows = np.array([[2, -0.75, 0.0625], [-0.5, 1, -3]], dtype="f4")
    got = ml.sparse_encode(rows, np.eye(3, dtype="f4"), algorithm="omp", n_nonzero_coefs=1,
                           numeric_mode="identical")
    np.testing.assert_array_equal(np.asarray(got), np.array([[2, 0, 0], [0, 0, -3]], dtype="f4"))


@pytest.mark.parametrize("probability", [0.0, 0.375])
def test_residual_dropout_forward_and_backward_obey_same_mask(probability):
    values = np.arange(1, 33, dtype="f4").reshape(4, 8) * np.float32(0.125)
    residual = np.full_like(values, 0.5)
    ones = np.ones_like(values)
    options = dict(p=probability, seed=42, stream=3, offset=17)
    forward = np.asarray(training.residual_dropout(values, residual, **options))
    dvalues, dresidual = training.residual_dropout_backward(ones, **options)
    dvalues = np.asarray(dvalues)
    np.testing.assert_array_equal(np.asarray(dresidual), ones)
    expected = np.asarray(residual + np.asarray(values * dvalues, dtype="f4"), dtype="f4")
    np.testing.assert_array_equal(forward.view("u4"), expected.view("u4"))
    if probability == 0:
        np.testing.assert_array_equal(dvalues, ones)
    else:
        assert np.any(dvalues == 0) and np.any(dvalues != 0)


@pytest.mark.parametrize("probability", [-0.1, 1.0])
def test_residual_dropout_rejects_invalid_probability(probability):
    values = np.ones((2, 3), dtype="f4")
    with pytest.raises(ValueError):
        training.residual_dropout(values, values, p=probability)
    with pytest.raises(ValueError):
        training.residual_dropout_backward(values, p=probability)


def test_training_vendor_is_the_selected_backend():
    assert training.vendor_used(numeric_mode="identical") == ml._backend.vendor()
