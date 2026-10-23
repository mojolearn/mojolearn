# SPDX-License-Identifier: Apache-2.0
"""PCA whitening boundary checks.

Default cases check the native owner on the installed binding.
The explicit opt-in cases require root-only remote CUDA/HIP IDENTICAL work.

DEVIATION 2460: fitted attributes and transform outputs are `mojolearn.Array`;
in-place mutation and NumPy reductions go through `np.asarray` (zero-copy,
writable), so every assertion below tests exactly what it did before.
"""
import os
import sys

import numpy as np
import pytest

from mojolearn import PCA, _backend


@pytest.mark.parametrize('name,value', [('singular_values_', -1.),
    ('singular_values_', np.inf), ('components_', np.nan), ('mean_', np.inf)])
def test_invalid_whitening_state_refused_by_the_native_call(name, value):
    # The whitened transforms read components_, mean_ and singular_values_
    # and refuse non-finite values and negative singular values in the call
    # that transforms; nothing is written to the output.
    x = np.random.default_rng(3).random((8, 2), dtype=np.float32)
    model = PCA(2, whiten=True).fit(x)
    np.asarray(getattr(model, name)).flat[0] = value
    with pytest.raises(ValueError, match='finite|nonnegative'):
        model.transform(x[:1])


def test_whitening_state_layout_refused_before_the_kernel():
    x = np.random.default_rng(4).random((8, 2), dtype=np.float32)
    model = PCA(2, whiten=True).fit(x)
    model.singular_values_ = np.ones(2, dtype=np.float64)
    with pytest.raises(ValueError, match='float32 singular_values_'):
        model.transform(x[:1])


def test_zero_singular_values_are_admitted():
    x = np.random.default_rng(5).random((8, 2), dtype=np.float32)
    model = PCA(2, whiten=True).fit(x)
    np.asarray(model.singular_values_)[:] = 0
    np.asarray(model.explained_variance_)[:] = 0
    assert np.isfinite(model.transform(x[:1])).all()


def test_nonfinite_input_refused():
    x = np.random.default_rng(6).random((8, 2), dtype=np.float32)
    bad = x.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match='Input X contains NaN'):
        PCA(2, whiten=True).fit(bad)
    model = PCA(2, whiten=True).fit(x)
    with pytest.raises(ValueError, match='Input X contains NaN'):
        model.transform(bad)
    with pytest.raises(ValueError, match='Input X contains NaN'):
        model.inverse_transform(bad)


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
