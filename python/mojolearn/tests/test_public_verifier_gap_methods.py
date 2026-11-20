# SPDX-License-Identifier: Apache-2.0
"""Independent public-output checks for changed endpoints lacking verifier lanes.

sparse_encode on an orthonormal dictionary (soft threshold, omp), the Theta
variants' batch forecasts and the residual-dropout mask run natively in
checks/native_tests/core_verifier_gap_methods.mojo,
sequence_theta_variants.mojo and training_residual_dropout_mask.mojo."""
import math

import numpy as np
import pytest

import mojolearn as ml
from mojolearn import training


@pytest.mark.parametrize("cls", [ml.DynamicTheta, ml.OptimizedTheta])
def test_theta_variant_public_batch_and_single_shapes(cls):
    """The public 2-D batch and 1-D single-series admission and returned shapes;
    the batch-equals-single numerics are sequence_theta_variants.mojo."""
    time = np.arange(48, dtype=np.float32)
    y = np.stack([20 + time * np.float32(0.1), 30 + time * np.float32(0.2)]).astype("f4")
    batch = np.asarray(cls(season_length=1, alpha=0.3).fit(y).predict(7)["mean"])
    single = np.asarray(cls(season_length=1, alpha=0.3).fit(y[0]).predict(7)["mean"])
    assert batch.shape == (2, 7)
    assert single.reshape(-1).shape == (7,)


@pytest.mark.parametrize("n,eps", [(1, 0.2), (17, 0.25), (1000, 0.4)])
def test_johnson_lindenstrauss_dimension_matches_stated_bound(n, eps):
    expected = int(4 * math.log(n) / (eps**2 / 2 - eps**3 / 3))
    assert ml.johnson_lindenstrauss_min_dim(n, eps=eps) == expected


@pytest.mark.parametrize("n,eps", [(0, 0.2), (5, 0.0), (5, 1.0)])
def test_johnson_lindenstrauss_refuses_invalid_bound_parameters(n, eps):
    with pytest.raises(ValueError):
        ml.johnson_lindenstrauss_min_dim(n, eps=eps)


@pytest.mark.parametrize("probability", [-0.1, 1.0])
def test_residual_dropout_rejects_invalid_probability(probability):
    values = np.ones((2, 3), dtype="f4")
    with pytest.raises(ValueError):
        training.residual_dropout(values, values, p=probability)
    with pytest.raises(ValueError):
        training.residual_dropout_backward(values, p=probability)


def test_training_vendor_is_the_selected_backend():
    assert training.vendor_used(numeric_mode="identical") == ml._backend.vendor()
