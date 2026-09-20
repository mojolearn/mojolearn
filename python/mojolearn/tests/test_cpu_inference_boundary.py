# SPDX-License-Identifier: Apache-2.0
"""CPU training is public: a fit on a CPU-only install runs, it does not refuse."""
import pytest
from mojolearn import _backend
from mojolearn._cpu_reference import reference_training, require_training
from mojolearn._mode import NumericModeMixin


class Estimator(NumericModeMixin):
    def fit(self, value):
        return value


@pytest.mark.parametrize('method', ['fit', 'partial_fit', 'fit_predict', 'fit_transform'])
def test_cpu_training_entrypoints_run(monkeypatch, method):
    monkeypatch.setattr(_backend, '_CPU_ONLY', 'test CPU')
    cls = type('TrainingEstimator', (NumericModeMixin,), {method: lambda self: 'ran'})
    assert getattr(cls(), method)() == 'ran'


def test_reference_context_is_harmless_and_restores(monkeypatch):
    monkeypatch.setattr(_backend, '_CPU_ONLY', 'test CPU')
    estimator = Estimator()
    with pytest.raises(ValueError):
        with reference_training():
            assert estimator.fit(7) == 7
            raise ValueError('caller failed')
    assert estimator.fit(10) == 10


def test_host_model_classes_fit(monkeypatch):
    monkeypatch.setattr(_backend, '_CPU_ONLY', None)
    class HostEstimator(Estimator):
        _HOST_INFERENCE_ONLY = True
    assert HostEstimator().fit(7) == 7
    assert require_training(HostEstimator()) is None
