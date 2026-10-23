# SPDX-License-Identifier: Apache-2.0
"""The experimental FeatureFreq surface through the real public fit (small inputs)."""

import numpy as np
import pytest

from mojolearn import ExperimentalTwoLevelFeatureFreq


def test_experimental_feature_freq_validates_and_forwards():
    X = np.asarray(
        [[0, 0, -2.5], [0, 1, 0.25], [1, 0, 3.5], [1, 1, 8.0]],
        dtype=np.float32,
    )
    y = np.asarray([-5, -3, 3, 5], dtype=np.float32)

    def fit(**kw):
        est = ExperimentalTwoLevelFeatureFreq(sources=[0, 1], l2_leaf_reg=4.0, random_state=7, **kw)
        assert est.fit(X, y) is est
        return est

    est = fit(learning_rate=0.2)
    assert est.model_
    assert est.n_features_in_ == 3
    assert fit(learning_rate=0.2).model_ == est.model_
    # the options reach the fit
    assert fit(learning_rate=0.5).model_ != est.model_


@pytest.mark.parametrize(
    "X,sources,needle",
    [
        ([[0, 0], [0, 1], [2, 0], [2, 1]], [0, 1], "densely coded"),
        ([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 2], "outside"),
        ([[0, 0], [0, 1], [1.5, 0], [1, 1]], [0, 1], "integer"),
    ],
)
def test_experimental_feature_freq_refusals(X, sources, needle):
    est = ExperimentalTwoLevelFeatureFreq(sources=sources)
    with pytest.raises(ValueError, match=needle):
        est.fit(np.asarray(X, dtype=np.float32), np.zeros(4, np.float32))


@pytest.mark.parametrize(
    "weights,needle",
    [
        ([1.0, 1.0], "one value per row"),
        ([1.0, -1.0, 1.0, 1.0], "finite and non-negative"),
        ([0.0, 0.0, 0.0, 0.0], "positive sum"),
        ([1.0, float("nan"), 1.0, 1.0], "finite and non-negative"),
        ([1.0, float("inf"), 1.0, 1.0], "finite and non-negative"),
    ],
)
def test_experimental_feature_freq_weight_refusals(weights, needle):
    X = np.asarray(
        [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32
    )
    est = ExperimentalTwoLevelFeatureFreq(sources=[0, 1])
    with pytest.raises(ValueError, match=needle):
        est.fit(X, np.zeros(4, np.float32), sample_weight=weights)


def test_experimental_feature_freq_refuses_constant_numeric_column():
    X = np.asarray(
        [[0, 0, 7], [0, 1, 7], [1, 0, 7], [1, 1, 7]], dtype=np.float32
    )
    est = ExperimentalTwoLevelFeatureFreq(sources=[0, 1])
    with pytest.raises(ValueError, match="numeric columns must vary; column 2"):
        est.fit(X, np.zeros(4, np.float32))


def test_experimental_feature_freq_forwards_weights():
    X = np.asarray(
        [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32
    )
    y = np.asarray([-2, -1, 1, 2], dtype=np.float32)

    def model(weights):
        return ExperimentalTwoLevelFeatureFreq(sources=[0, 1]).fit(X, y, sample_weight=weights).model_

    unit = ExperimentalTwoLevelFeatureFreq(sources=[0, 1]).fit(X, y).model_
    assert model(np.ones(4, np.float32)) == unit
    assert model(np.asarray([1, 2, 3, 4], dtype=np.float32)) != unit
