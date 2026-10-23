# SPDX-License-Identifier: Apache-2.0
"""Bagging* and AdaBoost* refuse a member that is not a mojolearn tree
estimator, by name, at fit (the real public path; small inputs)."""
import numpy as np
import pytest
from mojolearn import (AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier,
                       BaggingRegressor)


class _OtherEstimator:
    """A scikit-learn-style estimator that is not a mojolearn class."""

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def fit(self, X, y, sample_weight=None):
        return self

    def predict(self, X):
        return np.zeros(len(X))


def _data(classifier):
    rng = np.random.default_rng(0)
    X = rng.random((40, 4), dtype=np.float32)
    y = (X[:, 0] > 0.5).astype(np.int32) if classifier else X[:, 0].astype(np.float32)
    return X, y


@pytest.mark.parametrize("cls, classifier", [
    (BaggingClassifier, True), (BaggingRegressor, False),
    (AdaBoostClassifier, True), (AdaBoostRegressor, False),
])
def test_non_mojolearn_estimator_refused(cls, classifier):
    X, y = _data(classifier)
    model = cls(estimator=_OtherEstimator(), n_estimators=2, random_state=0)
    with pytest.raises(NotImplementedError, match="is not a mojolearn"):
        model.fit(X, y)
