# SPDX-License-Identifier: Apache-2.0
"""Exact gate for borrowed fitted buffers in classical host inference."""
import os

import numpy as np
import pytest


def _binding():
    # The INSTALLED package's host binding through the one resolution path
    # (`_backend.host_module_path`, honours MOJOLEARN_HOST_DIR, answers the
    # merged host file when the release merges them) and the product's own
    # loader, which reads back vendor, mode and column. The old form looked
    # beside this test file, so a run from a source checkout always skipped.
    from mojolearn import _backend
    if not os.path.exists(_backend.host_module_path("_mojolearn_estimators_host")):
        pytest.skip("_mojolearn_estimators_host.so is not built")
    return _backend.load_host_module("_mojolearn_estimators_host")


def _addr(array):
    return int(array.ctypes.data)


def test_borrowed_linear_and_decomposition_buffers_are_exact_and_live():
    binding = _binding()
    x = np.array([[1, 2, 3], [-1, 1, 2]], dtype=np.float32)
    components = np.array([[1, 0, 1], [0, 2, -1]], dtype=np.float32)
    mean = np.array([1, 1, 1], dtype=np.float32)
    out = np.empty((2, 2), dtype=np.float32)

    binding.tsvd_transform(_addr(x), _addr(components), _addr(out), [2, 3, 2])
    # The original 2x3 literal transform oracle now runs directly through
    # typed CPU/GPU providers in checks/native_tests/tsvd_native_*. Keep
    # the borrowed NumPy output and later caller-owned mutation seam here.
    assert out.shape == (2, 2) and out.dtype == np.float32
    binding.pca_transform(_addr(x), _addr(mean), _addr(components), _addr(out), [2, 3, 2])
    assert out.tobytes() == np.array([[2, 0], [-1, -1]], dtype=np.float32).tobytes()

    # A later call must observe caller-owned fitted storage, not a stale copy.
    components[0, 0] = 2
    binding.tsvd_transform(_addr(x), _addr(components), _addr(out), [2, 3, 2])
    assert out.tobytes() == np.array([[5, 1], [0, 0]], dtype=np.float32).tobytes()


def test_borrowed_regression_and_logistic_buffers_keep_contract():
    binding = _binding()
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    coef = np.array([2, -1], dtype=np.float32)
    out = np.empty(2, dtype=np.float32)
    binding.ols_predict(_addr(x), _addr(coef), _addr(out), [2, 2, 0.5])
    assert out.tobytes() == np.array([0.5, 2.5], dtype=np.float32).tobytes()
    binding.qn_decision_function(_addr(x), _addr(coef), _addr(out), [2, 2, 0])
    assert out.tobytes() == np.array([0, 2], dtype=np.float32).tobytes()
    with pytest.raises(Exception, match="params must contain"):
        binding.tsvd_transform(_addr(x), _addr(coef), _addr(out), [2, 2])
