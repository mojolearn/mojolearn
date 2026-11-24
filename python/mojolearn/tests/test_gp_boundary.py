# SPDX-License-Identifier: Apache-2.0
"""Gaussian process public boundary: refusals in words and the types, shapes and dtypes the methods return.

The numerical answers on each side are verify lanes (gp*, gpc*, x-neighbors-gp-cov); what stays here is what a
digest cannot see.
"""
import numpy as np
import pytest

import mojolearn as ml
from mojolearn import _gp_impl


# ---------------------------------------------------------------- moved from test_gp_return_cov.py (vdv-pytest-a g2)
# GaussianProcessRegressor.predict(return_cov=True) (lane x-neighbors-gp-cov). Skips, saying so, without a GP binding.

def _return_cov_fit_or_skip(normalize_y=False):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(60, 3)).astype(np.float32)
    y = (np.sin(x[:, 0]) + 0.1 * rng.normal(size=60) + (5.0 if normalize_y else 0.0)).astype(np.float32)
    k = ml.ConstantKernel(1.0) * ml.RBF(1.0) + ml.WhiteKernel(0.1)
    try:
        m = ml.GaussianProcessRegressor(kernel=k, normalize_y=normalize_y).fit(x, y)
    except (ImportError, NotImplementedError) as exc:
        pytest.skip(f"no GP binding on this install: {exc}")
    return m, x


def test_both_flags_are_refused():
    m, x = _return_cov_fit_or_skip()
    with pytest.raises(RuntimeError, match="at most one"):
        m.predict(x, return_std=True, return_cov=True)


@pytest.mark.parametrize("normalize_y", [False, True])
def test_covariance_public_arrays_and_mean_interface(normalize_y):
    m, x = _return_cov_fit_or_skip(normalize_y)
    q = x[:20] + np.float32(0.25)
    mean, cov = m.predict(q, return_cov=True)
    mean, cov = np.asarray(mean), np.asarray(cov)
    assert cov.shape == (20, 20) and cov.dtype == np.float32
    assert mean.tobytes() == np.asarray(m.predict(q)).tobytes()
    # Both normalization settings retain the original numerical bounds in native fixtures.
    again = np.asarray(m.predict(q, return_cov=True)[1])
    assert again.tobytes() == cov.tobytes()


# ---------------------------------------------------------------- moved from test_gp_normalize_y.py (vdv-pytest-a g2)
# GaussianProcessRegressor(normalize_y=True) (lane gp-normalize-y): the refusal of a non-bool flag, and the scalar
# metadata and float32 array a constant target publishes.

def test_non_bool_is_refused_by_name():
    with pytest.raises(TypeError, match="normalize_y must be a bool"):
        _gp_impl.GaussianProcessRegressor(normalize_y="yes").fit([[0.0], [1.0]], [0.0, 1.0])


def test_constant_target_publishes_scalar_metadata_and_array():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(96, 3)).astype(np.float32)
    q = rng.normal(size=(24, 3)).astype(np.float32)
    y = np.full(x.shape[0], 7.0, np.float32)
    kernel = _gp_impl.ConstantKernel(1.0) * _gp_impl.RBF(1.0) + _gp_impl.WhiteKernel(0.1)
    try:
        m = _gp_impl.GaussianProcessRegressor(kernel=kernel, normalize_y=True).fit(x, y)
    except (ImportError, NotImplementedError) as exc:
        pytest.skip(f"no GP binding that fits on this install: {exc}")
    assert isinstance(m._y_train_std, float) and isinstance(m._y_train_mean, float)
    mean = np.asarray(m.predict(q))
    assert mean.shape == (24,) and mean.dtype == np.float32


# ---- appended by vdv-pytest-b (2026-11-24) ----
# moved from python/mojolearn/tests/test_parallel_gaussian_process.py
import numpy as np
import pytest

from mojolearn import parallel_gaussian_process as _pgp_pg
from mojolearn._gpc_impl import GaussianProcessClassifier as _PgpGPC


@pytest.fixture
def _pgp_identical(monkeypatch):
    monkeypatch.setattr('mojolearn._backend.default_mode', lambda: 'identical')
    yield


@pytest.mark.parametrize('labels', [[1, 1], [1], []])
def test_parallel_gpc_invalid_targets(_pgp_identical, labels):
    with pytest.raises(ValueError):
        _pgp_pg.fit_gaussian_process_classifier(_PgpGPC(), np.ones((2, 3), np.float32), labels)


def test_parallel_gpc_prediction_validation(_pgp_identical):
    with pytest.raises(ValueError, match='method'):
        _pgp_pg.predict_gaussian_process_classifier(_PgpGPC(), np.ones((2, 3), np.float32), method='score')
    with pytest.raises(ValueError, match='fit'):
        _pgp_pg.predict_gaussian_process_classifier(_PgpGPC(), np.ones((2, 3), np.float32))
