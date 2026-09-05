# SPDX-License-Identifier: Apache-2.0
"""Lightweight Python-policy gate for the experimental FeatureFreq surface."""

import numpy as np
import pytest

from mojolearn import ExperimentalTwoLevelFeatureFreq


class _FakeBinding:
    def __init__(self):
        self.params = None

    def gbdt_fit_two_level_feature_freq(self, x, y, sources, params):
        assert x and y and sources
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
    assert binding.params == [4, 3, 2, 0.2, 4.0, 7]
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
