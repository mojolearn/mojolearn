# SPDX-License-Identifier: Apache-2.0
"""Python boundary policy only; native ordered certification is a separate gate."""
import ctypes

import numpy as np
import pytest

from mojolearn.ensemble import OrderedRMSE


class _Binding:
    def gbdt_fit_ordered_rmse(self, x, y, weights, permutation, params):
        self.params = list(params)
        n, f, nw = params[:3]
        def copy(address, ctype, count):
            return np.ctypeslib.as_array((ctype * count).from_address(address)).copy()
        self.x = copy(x, ctypes.c_float, n * f)
        self.y = copy(y, ctypes.c_float, n)
        self.weights = copy(weights, ctypes.c_float, nw)
        self.permutation = copy(permutation, ctypes.c_uint32, n)
        return "ordered-model"


def test_ordered_buffers_are_original_order_and_column_major():
    binding = _Binding()
    model = OrderedRMSE(n_estimators=3, max_depth=2, border_count=7,
                        learning_rate=0.2, l2_leaf_reg=0, numeric_mode="identical")
    selected = []
    def bind(name):
        selected.append((name, model.numeric_mode))
        return binding
    model._bind = bind
    X = np.arange(8, dtype=np.float64).reshape(4, 2)
    y = np.asarray([8, 2, -1, 3], np.float64)
    assert model.fit(X, y, permutation=[3, 1, 0, 2], sample_weight=[2, 0, 1, 4]) is model
    assert selected == [("_mojolearn_gbdt", "identical")]
    assert binding.params == [4, 2, 4, 4, 3, 2, 7, 0.2, 0.0]
    np.testing.assert_array_equal(binding.x, X.ravel(order="F"))
    np.testing.assert_array_equal(binding.y, y)
    np.testing.assert_array_equal(binding.weights, [2, 0, 1, 4])
    np.testing.assert_array_equal(binding.permutation, [3, 1, 0, 2])
    assert model.model_ == "ordered-model"
    assert model.loss_curve_ is None and model.best_iteration_ is None


@pytest.mark.parametrize("permutation", [
    [0, 0, 2, 3], [0, 1, 2], [-1, 1, 2, 3], [0, 1, 2, 4],
    [0., 1., 2., 3.], [[0, 1], [2, 3]], [False, True, False, True],
    [0, 1, 2, 2**32 + 3],
])
def test_invalid_permutation_refused_before_binding(permutation):
    model = OrderedRMSE()
    model._bind = lambda name: pytest.fail("invalid input reached native binding")
    with pytest.raises(ValueError, match="integer bijection"):
        model.fit(np.arange(8).reshape(4, 2), np.zeros(4), permutation=permutation)


@pytest.mark.parametrize("kwargs", [
    {"max_depth": 9}, {"max_depth": 2.5}, {"max_depth": True},
    {"n_estimators": 0}, {"border_count": 256},
    {"learning_rate": 0}, {"learning_rate": 1e-100},
    {"learning_rate": float("inf")}, {"l2_leaf_reg": -1},
    {"l2_leaf_reg": float("nan")},
])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        OrderedRMSE(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"loss": "Logloss"}, {"cat_features": [0]},
    {"boosting_type": "Ordered"}, {"bootstrap_type": "Bayesian"},
    {"permutation_count": 2}, {"grow_policy": "Lossguide"},
])
def test_unsupported_options_are_not_silently_accepted(kwargs):
    with pytest.raises(TypeError):
        OrderedRMSE(**kwargs)


@pytest.mark.parametrize("weights", [[0, 0, 0, 0], [1, -1, 1, 1],
                                        [1, float("nan"), 1, 1], [1, 2]])
def test_invalid_weights(weights):
    model = OrderedRMSE()
    model._bind = lambda name: pytest.fail("invalid input reached native binding")
    with pytest.raises(ValueError, match="sample_weight"):
        model.fit(np.arange(8).reshape(4, 2), np.zeros(4),
                  permutation=[0, 1, 2, 3], sample_weight=weights)


def test_old_binary_refuses_with_actionable_error():
    model = OrderedRMSE()
    model._bind = lambda name: object()
    with pytest.raises(RuntimeError, match="predates OrderedRMSE"):
        model.fit(np.arange(8).reshape(4, 2), np.zeros(4), permutation=[0, 1, 2, 3])
