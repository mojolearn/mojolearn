# SPDX-License-Identifier: Apache-2.0
"""OrderedRMSE through the real public fit (small inputs); native ordered
certification is a separate gate."""
import numpy as np
import pytest

from mojolearn.ensemble import OrderedRMSE


def _fit(X, **kw):
    model = OrderedRMSE(n_estimators=3, max_depth=2, border_count=7,
                        learning_rate=0.2, l2_leaf_reg=0, numeric_mode="identical")
    y = np.asarray([8, 2, -1, 3], np.float64)
    assert model.fit(X, y, permutation=[3, 1, 0, 2], sample_weight=[2, 0, 1, 4], **kw) is model
    return model


def test_ordered_fit_reads_rows_in_original_order_any_layout():
    """The fit reads X in its original row order whatever its layout and
    dtype: C-order float64, Fortran-order float64 and float32 are one model."""
    X = np.arange(8, dtype=np.float64).reshape(4, 2)
    model = _fit(X)
    assert model.model_
    assert model.loss_curve_ is None and model.best_iteration_ is None
    assert model.n_features_in_ == 2
    for same in (np.asfortranarray(X), X.astype(np.float32)):
        assert _fit(same).model_ == model.model_


@pytest.mark.parametrize("permutation", [
    [0, 0, 2, 3], [0, 1, 2], [-1, 1, 2, 3], [0, 1, 2, 4],
    [0., 1., 2., 3.], [[0, 1], [2, 3]], [False, True, False, True],
    [0, 1, 2, 2**32 + 3],
])
def test_invalid_permutation_refused_before_binding(permutation):
    model = OrderedRMSE()
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
    with pytest.raises(ValueError, match="sample_weight"):
        model.fit(np.arange(8).reshape(4, 2), np.zeros(4),
                  permutation=[0, 1, 2, 3], sample_weight=weights)
