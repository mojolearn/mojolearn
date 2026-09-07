# SPDX-License-Identifier: Apache-2.0
"""Authored PCA whitening boundary checks; never run by a subagent.

Default cases use host binding sentinels, not numerical PCA implementations.
The explicit opt-in cases require root-only remote CUDA/HIP IDENTICAL work.

DEVIATION 2460: fitted attributes and transform outputs are `mojolearn.Array`;
in-place mutation and NumPy reductions go through `np.asarray` (zero-copy,
writable), so every assertion below tests exactly what it did before.
"""
import ctypes
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mojolearn import PCA, _backend


def floats(address, count):
    return np.ctypeslib.as_array((ctypes.c_float * count).from_address(address))


class FakePCA:
    def __init__(self):
        self.calls = []
        self.nonfinite = False

    def pca_fit(self, x, components, mean, explained, ratio, singular, params):
        self.calls.append(('fit', list(params)))
        rows, features, count = params
        floats(components, count * features)[:] = np.eye(count, features, dtype=np.float32).ravel()
        floats(mean, features)[:] = 0
        floats(explained, count)[:] = 1
        floats(ratio, count)[:] = 1 / count
        floats(singular, count)[:] = 2
        return 0.

    def pca_whiten_transform(self, x, mean, components, singular, out, params):
        self.calls.append(('whiten', list(params)))
        floats(out, params[0] * params[2])[:] = np.nan if self.nonfinite else .25

    def pca_whiten_inverse_transform(self, scores, components, singular, mean, out, params):
        self.calls.append(('inverse_whiten', list(params)))
        floats(out, params[0] * params[1])[:] = np.inf if self.nonfinite else .5

    def pca_transform(self, x, mean, components, out, params):
        self.calls.append(('plain', list(params)))
        floats(out, params[0] * params[2])[:] = 2

    def inverse_transform(self, scores, components, mean, out, params):
        self.calls.append(('inverse_plain', list(params)))
        floats(out, params[0] * params[1])[:] = 3


@pytest.fixture
def fake(monkeypatch):
    result = FakePCA()
    monkeypatch.setattr(PCA, '_bind', lambda self, name: result)
    return result


def test_whitening_requires_both_additive_exports_before_fit(monkeypatch):
    partial = SimpleNamespace(pca_whiten_transform=lambda *args: None)
    monkeypatch.setattr(PCA, '_bind', lambda self, name: partial)
    with pytest.raises(NotImplementedError, match='build_estimators.sh'):
        PCA(2, whiten=True).fit(np.zeros((8, 2), dtype=np.float32))


def test_whitening_abi_uses_fit_row_count_and_keeps_state(fake):
    x = np.arange(16, dtype=np.float32).reshape(8, 2)
    model = PCA(2, whiten=True).fit(x)
    state = {name: getattr(model, name).tobytes() for name in
             ('components_', 'singular_values_', 'mean_', 'explained_variance_')}
    before = x.tobytes()
    scores = model.transform(x[:3])
    model.inverse_transform(scores[:1])
    assert fake.calls == [('fit', [8, 2, 2]), ('whiten', [3, 2, 2, 8]),
                          ('inverse_whiten', [1, 2, 2, 8])]
    assert x.tobytes() == before
    assert all(getattr(model, name).tobytes() == value for name, value in state.items())


def test_plain_transform_abi_unchanged(fake):
    x = np.zeros((8, 2), dtype=np.float32)
    model = PCA(2, whiten=False).fit(x)
    model.inverse_transform(model.transform(x[:3]))
    assert fake.calls == [('fit', [8, 2, 2]), ('plain', [3, 2, 2]),
                          ('inverse_plain', [3, 2, 2, 1])]


@pytest.mark.parametrize('name,value', [('singular_values_', -1.),
    ('singular_values_', np.inf), ('explained_variance_', -1.),
    ('components_', np.nan), ('mean_', np.inf)])
def test_invalid_whitening_state_refused_before_native(fake, name, value):
    model = PCA(2, whiten=True).fit(np.zeros((8, 2), dtype=np.float32))
    np.asarray(getattr(model, name)).flat[0] = value
    with pytest.raises(ValueError, match='finite|nonnegative'):
        model.transform(np.zeros((1, 2), dtype=np.float32))
    assert len(fake.calls) == 1


def test_zero_singular_values_are_admitted(fake):
    model = PCA(2, whiten=True).fit(np.zeros((8, 2), dtype=np.float32))
    np.asarray(model.singular_values_)[:] = 0
    np.asarray(model.explained_variance_)[:] = 0
    assert np.isfinite(model.transform(np.zeros((1, 2), dtype=np.float32))).all()


def test_nonfinite_input_and_output_refused(fake):
    x = np.zeros((8, 2), dtype=np.float32)
    bad = x.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite X'):
        PCA(2, whiten=True).fit(bad)
    assert not fake.calls
    model = PCA(2, whiten=True).fit(x)
    with pytest.raises(ValueError, match='finite X'):
        model.transform(bad)
    with pytest.raises(ValueError, match='finite scores'):
        model.inverse_transform(bad)
    fake.nonfinite = True
    with pytest.raises(ValueError, match='nonfinite output'):
        model.transform(x)
    with pytest.raises(ValueError, match='nonfinite output'):
        model.inverse_transform(x)


def require_remote_gpu():
    if os.environ.get('MOJOLEARN_RUN_PCA_WHITEN_GPU') != '1':
        pytest.skip('root-only opt-in remote PCA whitening gate')
    assert sys.platform == 'linux'
    assert _backend.vendor() in ('cuda', 'hip')
    assert _backend.numeric_mode() == 'identical'


def test_remote_planted_zero_scale_and_round_trip():
    require_remote_gpu()
    model = PCA(2, whiten=True, numeric_mode='identical')
    model.n_samples_ = 5
    model.n_components_ = model.n_features_in_ = 2
    model.components_ = np.eye(2, dtype=np.float32)
    model.mean_ = np.zeros(2, dtype=np.float32)
    model.singular_values_ = np.array([2., 0.], dtype=np.float32)
    model.explained_variance_ = np.array([1., 0.], dtype=np.float32)
    x = np.array([[1., 2.], [3., 4.], [-1., -2.]], dtype=np.float32)
    # sqrt(5-1)=2 exactly. Ordinary component scale is2/2; zero s
    # skips the divide, so its forward scale is2 and inverse scale is1/2.
    expected = np.array([[1., 4.], [3., 8.], [-1., -4.]], dtype=np.float32)
    actual = model.transform(x)
    assert actual.tobytes() == expected.tobytes()
    assert model.transform(x[:1]).tobytes() == actual[:1].tobytes()
    assert model.transform(x).tobytes() == actual.tobytes()
    assert model.inverse_transform(actual).tobytes() == x.tobytes()


def test_remote_fit_unit_variance_and_inverse():
    require_remote_gpu()
    x = np.array([[-3., -1.], [-3., 1.], [-1., -1.], [-1., 1.],
                  [1., -1.], [1., 1.], [3., -1.], [3., 1.]], dtype=np.float32)
    original = x.tobytes()
    model = PCA(2, whiten=True, numeric_mode='identical').fit(x)
    components = model.components_.tobytes()
    actual = model.transform(x)
    assert np.allclose(np.asarray(actual).astype(np.float64).var(axis=0, ddof=1), 1., rtol=2e-5, atol=2e-5)
    assert model.transform(x[:3]).tobytes() == actual[:3].tobytes()
    assert np.allclose(model.inverse_transform(actual), x, rtol=2e-5, atol=2e-5)
    assert model.components_.tobytes() == components and x.tobytes() == original


def test_remote_strict_skip_threshold():
    require_remote_gpu()
    threshold = np.float32(1e-10)
    below = np.nextafter(threshold, np.float32(0))
    model = PCA(2, whiten=True, numeric_mode='identical')
    model.n_samples_ = 2  # scalar sqrt(n_fit-1) is exactly1
    model.n_components_ = model.n_features_in_ = 2
    model.components_ = np.eye(2, dtype=np.float32)
    model.mean_ = np.zeros(2, dtype=np.float32)
    model.singular_values_ = np.array([below, threshold], dtype=np.float32)
    model.explained_variance_ = np.square(model.singular_values_)
    actual = model.transform(np.eye(2, dtype=np.float32))
    assert np.asarray(actual)[0, 0].tobytes() == np.float32(1).tobytes()
    expected = np.float32(1) / threshold
    assert np.asarray(actual)[1, 1].tobytes() == expected.tobytes()
