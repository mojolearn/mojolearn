# SPDX-License-Identifier: Apache-2.0
"""Public fits refuse before doing work; private verification restores policy."""
import pytest
from mojolearn import _backend, host_surface
from mojolearn._cpu_reference import reference_training
from mojolearn._mode import NumericModeMixin


class Estimator(NumericModeMixin):
    def fit(self, value):
        return value


@pytest.mark.parametrize('method', ['fit', 'partial_fit', 'fit_predict', 'fit_transform'])
def test_cpu_training_entrypoints_refuse(monkeypatch, method):
    monkeypatch.setattr(_backend, '_CPU_ONLY', 'test CPU')
    cls = type('TrainingEstimator', (NumericModeMixin,), {method: lambda self: pytest.fail('fit executed')})
    with pytest.raises(NotImplementedError, match='inference from saved models'):
        getattr(cls(), method)()


def test_reference_context_restores_after_exception(monkeypatch):
    monkeypatch.setattr(_backend, '_CPU_ONLY', 'test CPU')
    estimator = Estimator()
    with pytest.raises(ValueError):
        with reference_training():
            assert estimator.fit(7) == 7
            with reference_training():
                assert estimator.fit(8) == 8
            assert estimator.fit(9) == 9
            raise ValueError('verification failed')
    with pytest.raises(NotImplementedError):
        estimator.fit(10)


def test_gpu_fit_and_explicit_host_model(monkeypatch):
    monkeypatch.setattr(_backend, '_CPU_ONLY', None)
    assert Estimator().fit(7) == 7
    class HostEstimator(Estimator):
        _HOST_INFERENCE_ONLY = True
    with pytest.raises(NotImplementedError):
        HostEstimator().fit(7)


def test_passive_cpu_fits_are_not_a_backdoor(monkeypatch):
    from mojolearn import NearestNeighbors, KernelDensity
    monkeypatch.setattr(_backend, '_CPU_ONLY', 'test CPU')
    for estimator in (NearestNeighbors(), KernelDensity()):
        with pytest.raises(NotImplementedError, match='internal bitwise verifier'):
            estimator.fit([[0.0], [1.0]])


def test_exponential_smoothing_fit_is_not_a_backdoor(monkeypatch):
    """ExponentialSmoothing is not a NumericModeMixin, so the mixin's guard
    never wrapped its fit; it refuses by name itself (2026-09-15)."""
    from mojolearn import ExponentialSmoothing
    monkeypatch.setattr(_backend, '_CPU_ONLY', 'test CPU')
    series = [1.0 + (i % 12) + 0.1 * i for i in range(48)]
    with pytest.raises(NotImplementedError, match='internal bitwise verifier'):
        ExponentialSmoothing(series, seasonal_periods=12).fit()


def test_saved_model_inference_classes_are_host_bound():
    """The saved ARIMA and UMAP host classes are inference-only: their fits
    refuse on any machine outside the reference context."""
    from mojolearn._classical_host import HostARIMA, HostUMAP
    with pytest.raises(NotImplementedError):
        HostARIMA().fit([[0.0, 1.0, 2.0, 3.0]])
    with pytest.raises(NotImplementedError):
        HostUMAP().fit([[0.0], [1.0]])


def test_byte_lm_published_trainer_remains_exported():
    import mojolearn
    assert hasattr(mojolearn.LanguageModelHostTrainer, 'train_step')
    assert 'byte_lm' in host_surface.wheel_families()
    assert 'byte_lm_host_train_step' in host_surface.family('byte_lm')['exports']
