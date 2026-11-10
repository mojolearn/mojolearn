# SPDX-License-Identifier: Apache-2.0
"""PCA whitening boundary checks.

These cases check actual writable-state and input admission on the installed
binding. Numerical whitening fixtures now call native CPU/GPU functions in
checks/native_tests/pca_whitening_{cpu,gpu}_suite.mojo.

DEVIATION 2460: fitted attributes and transform outputs are `mojolearn.Array`;
in-place mutation and NumPy reductions go through `np.asarray` (zero-copy,
writable), so every assertion below tests exactly what it did before.
"""
import numpy as np
import pytest

from mojolearn import PCA


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
