# SPDX-License-Identifier: Apache-2.0
"""Lightweight Python-policy gate for the experimental FeatureFreq surface."""

import ctypes

import numpy as np
import pytest

from mojolearn import ExperimentalTwoLevelFeatureFreq


class _FakeBinding:
    def __init__(self):
        self.params = None

    def gbdt_fit_two_level_feature_freq(self, x, y, weights, sources, params):
        assert x and y and weights and sources
        self.weights = np.ctypeslib.as_array(
            (ctypes.c_float * params[2]).from_address(weights)
        ).copy()
        self.params = list(params)
        return "tensor-model-text"


def test_experimental_feature_freq_validates_and_forwards():
    X = np.asarray(
        [[0, 0, -2.5], [0, 1, 0.25], [1, 0, 3.5], [1, 1, 8.0]],
        dtype=np.float32,
    )
    y = np.asarray([-5, -3, 3, 5], dtype=np.float32)
    binding = _FakeBinding()
    est = ExperimentalTwoLevelFeatureFreq(
        sources=[0, 1], learning_rate=0.2, l2_leaf_reg=4.0, random_state=7
    )
    est._bind = lambda _name: binding
    assert est.fit(X, y) is est
    assert est.model_ == "tensor-model-text"
    assert binding.params == [4, 3, 0, 2, 0.2, 4.0, 7]
    assert est.n_features_in_ == 3


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


def test_experimental_feature_freq_forwards_weights():
    X = np.asarray(
        [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32
    )
    binding = _FakeBinding()
    est = ExperimentalTwoLevelFeatureFreq(sources=[0, 1])
    est._bind = lambda _name: binding
    est.fit(
        X,
        np.asarray([-2, -1, 1, 2], dtype=np.float32),
        sample_weight=np.asarray([1, 2, 3, 4], dtype=np.float32),
    )
    assert binding.params[:4] == [4, 2, 4, 2]
    np.testing.assert_array_equal(binding.weights, [1, 2, 3, 4])
