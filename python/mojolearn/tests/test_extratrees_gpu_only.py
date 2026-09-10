# SPDX-License-Identifier: Apache-2.0
"""GPU-only ET parameter validation without loading native code."""
import pytest

from mojolearn import ExtraTreesClassifier, ExtraTreesRegressor


@pytest.mark.parametrize('cls', [ExtraTreesClassifier, ExtraTreesRegressor])
@pytest.mark.parametrize('device', ['cpu', 'cuda', '', None, 0, 1])
def test_non_gpu_device_refused_at_constructor_and_set_params(cls, device):
    with pytest.raises(ValueError, match='GPU-only'):
        cls(device=device)
    model = cls(device='gpu')
    with pytest.raises(ValueError, match='GPU-only'):
        model.set_params(device=device)
    assert model.device == 'gpu'


@pytest.mark.parametrize('cls', [ExtraTreesClassifier, ExtraTreesRegressor])
def test_direct_device_mutation_refused_before_input_or_binding(cls, monkeypatch):
    model = cls()
    model.device = 'cpu'
    def forbidden(*args, **kwargs):
        raise AssertionError('native binding must not load')
    monkeypatch.setattr(model, '_bind', forbidden)
    # Invalid objects also prove device validation precedes data conversion.
    with pytest.raises(ValueError, match='GPU-only'):
        model.fit(object(), object())


@pytest.mark.parametrize('cls', [ExtraTreesClassifier, ExtraTreesRegressor])
def test_gpu_parameter_remains_cloneable_and_abi_stable(cls):
    from mojolearn.extratrees import _fit_params
    model = cls(device='gpu')
    replica = cls(**model.get_params())
    assert replica.device == 'gpu'
    params = _fit_params(2, 1, 2, replica._cfg, replica.device,
                         replica._criterion_code)
    assert params[20] == 1
