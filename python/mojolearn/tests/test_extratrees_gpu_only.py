# SPDX-License-Identifier: Apache-2.0
"""GPU-only ET parameter validation at fit, before any input conversion."""
import pytest

from mojolearn import ExtraTreesClassifier, ExtraTreesRegressor


@pytest.mark.parametrize('cls', [ExtraTreesClassifier, ExtraTreesRegressor])
@pytest.mark.parametrize('device', ['cpu', 'cuda', '', None, 0, 1])
def test_non_gpu_device_refused_at_fit_after_constructor_and_set_params(cls, device):
    # scikit-learn's convention: the constructor and set_params store the
    # value verbatim and check nothing; fit refuses it, before any input
    # conversion or native binding.
    model = cls(device=device)
    assert model.device == device
    with pytest.raises(ValueError, match='GPU-only'):
        model.fit(object(), object())
    model = cls(device='gpu')
    model.set_params(device=device)
    assert model.device == device
    with pytest.raises(ValueError, match='GPU-only'):
        model.fit(object(), object())


@pytest.mark.parametrize('cls', [ExtraTreesClassifier, ExtraTreesRegressor])
def test_direct_device_mutation_refused_before_input_or_binding(cls):
    model = cls()
    model.device = 'cpu'
    # Invalid objects also prove device validation precedes data conversion.
    with pytest.raises(ValueError, match='GPU-only'):
        model.fit(object(), object())


@pytest.mark.parametrize('cls', [ExtraTreesClassifier, ExtraTreesRegressor])
def test_gpu_parameter_remains_cloneable_and_fits(cls):
    import numpy as np
    model = cls(device='gpu')
    replica = cls(**model.get_params())
    assert replica.device == 'gpu'
    rng = np.random.default_rng(0)
    X = rng.random((40, 4), dtype=np.float32)
    y = (X[:, 0] > 0.5).astype(np.int32)
    replica.set_params(n_estimators=2, max_depth=2, random_state=0)
    assert replica.fit(X, y) is replica
    assert replica.device == 'gpu'
