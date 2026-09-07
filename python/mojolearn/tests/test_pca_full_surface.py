"""Authored additive full-PCA exposure gates; root executes, never agents.

Host sentinels prove dispatch/ABI only. Numerical case requires explicit
remote CUDA/HIP opt-in and does not confer cross-vendor certification.
"""
import ctypes
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mojolearn import PCA, _backend


def write(address, values):
    values = np.asarray(values, dtype=np.float32).ravel()
    np.ctypeslib.as_array((ctypes.c_float * len(values)).from_address(address))[:] = values


def test_full_uses_dedicated_export_and_all_output_slots(monkeypatch):
    calls = []
    def full(x, components, mean, explained, ratio, singular, params):
        calls.append(list(params))
        write(components, [[1, 0, 0], [0, 1, 0]])
        write(mean, [1, 2, 3])
        write(explained, [8, 4])
        write(ratio, [.5, .25])
        write(singular, [6, 3])
        return .125
    # No covariance export exists on this sentinel: falling back must fail.
    monkeypatch.setattr(PCA, '_bind', lambda self, name: SimpleNamespace(pca_fit_full=full))
    x = np.arange(24, dtype=np.float32).reshape(8, 3)
    before = x.tobytes()
    model = PCA(2, svd_solver='full').fit(x)
    assert calls == [[8, 3, 2]]
    np.testing.assert_array_equal(model.components_, [[1, 0, 0], [0, 1, 0]])
    np.testing.assert_array_equal(model.mean_, [1, 2, 3])
    np.testing.assert_array_equal(model.explained_variance_, [8, 4])
    np.testing.assert_array_equal(model.explained_variance_ratio_, [.5, .25])
    np.testing.assert_array_equal(model.singular_values_, [6, 3])
    assert model.noise_variance_ == .125 and model.n_samples_ == 8
    assert x.tobytes() == before


def test_full_missing_export_refuses_without_covariance_fallback(monkeypatch):
    monkeypatch.setattr(PCA, '_bind', lambda self, name: SimpleNamespace())
    with pytest.raises(NotImplementedError, match='build_estimators.sh'):
        PCA(2, svd_solver='full').fit(np.ones((8, 3), dtype=np.float32))


def test_full_wide_matrix_refuses_before_native_call(monkeypatch):
    def forbidden(*args):
        raise AssertionError('wide matrix reached native full fit')
    monkeypatch.setattr(PCA, '_bind', lambda self, name: SimpleNamespace(pca_fit_full=forbidden))
    with pytest.raises(NotImplementedError, match='at least as many samples'):
        PCA(2, svd_solver='full').fit(np.ones((3, 8), dtype=np.float32))


@pytest.mark.skipif(
    os.environ.get('MOJOLEARN_PCA_FULL_GATE') != '1', reason='root-only numerical opt-in')
def test_full_remote_small_reference_and_identical_repeat():
    expected_vendor = os.environ.get('MOJOLEARN_EXPECT_VENDOR')
    if sys.platform != 'linux' or expected_vendor not in ('cuda', 'hip'):
        pytest.fail('full PCA gate requires explicit remote CUDA/HIP vendor')
    if _backend.vendor() != expected_vendor or _backend.numeric_mode() != 'identical':
        pytest.fail('full PCA native mode/vendor mismatch')
    x = np.array([[0, 1, 2], [2, 0, 1], [1, 3, -1], [-2, 1, 0],
                  [4, -1, 2], [0, -3, 1], [3, 2, 4], [-1, 0, -2]], dtype=np.float32)
    original = x.tobytes()
    a = PCA(3, svd_solver='full').fit(x)
    b = PCA(3, svd_solver='full').fit(x)
    centered = x.astype(np.float64) - x.astype(np.float64).mean(axis=0)
    _, singular, _ = np.linalg.svd(centered, full_matrices=False)
    np.testing.assert_allclose(a.singular_values_, singular, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(a.explained_variance_, singular**2 / 7, rtol=4e-4, atol=2e-5)
    np.testing.assert_allclose(a.inverse_transform(a.transform(x)), x, rtol=2e-4, atol=2e-4)
    for name in ('components_', 'mean_', 'explained_variance_',
                 'explained_variance_ratio_', 'singular_values_'):
        assert getattr(a, name).tobytes() == getattr(b, name).tobytes()
    assert x.tobytes() == original
