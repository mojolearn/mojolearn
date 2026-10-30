"""Radius fit must use its connected neighbor-query attribute owner."""
import numpy as np
import pytest

import mojolearn as ml
from mojolearn._verify_native_runtime import audit_native_call


def test_fit_reads_settings_through_native_neighbor_owner():
    x = np.array([[0, 0], [1, 0], [0, 2]], dtype=np.float32)
    model = ml.RadiusNeighbors(radius=1.5, metric='minkowski', p=3)
    assert audit_native_call(model.fit, x).assert_native() is model
    assert model.n_samples_fit_ == 3 and model.n_features_in_ == 2
    assert model.radius == 1.5 and model.p == 3


def test_fit_does_not_execute_unconnected_setting_descriptor(monkeypatch):
    model = ml.RadiusNeighbors(radius=1.5)
    called = []

    def metric(instance):
        called.append(instance)
        return 'euclidean'

    monkeypatch.setattr(ml.RadiusNeighbors, 'metric', property(metric), raising=False)
    with pytest.raises(TypeError, match='property|descriptor|layout'):
        model.fit(np.zeros((3, 2), dtype=np.float32))
    assert called == []


def test_fit_keeps_current_setting_validation():
    model = ml.RadiusNeighbors(radius=1.5)
    model.radius = -1.0
    with pytest.raises(ValueError, match='radius'):
        model.fit(np.zeros((3, 2), dtype=np.float32))
