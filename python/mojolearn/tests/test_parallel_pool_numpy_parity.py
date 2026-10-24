# SPDX-License-Identifier: Apache-2.0
"""The native multi-GPU drivers run in the caller's process with its
optional dependency set.

`import mojolearn` no longer loads NumPy, so the old pool's fresh worker had
none and the pooled Gram PCA (`parallel_classical.fit_gram_estimator`, lane
par-gram-pca) failed there with "PCA requires optional NumPy support"
although the parent had NumPy (identity record, AMD MI325X, 5b1606ab1). The
cooperative drivers are now one native call in this process
(bindings/family_native.mojo `family_par_run`), so they see the caller's
NumPy and return the caller's native types.
"""
import sys

import numpy as np
import pytest

from mojolearn import _backend


def test_pooled_gram_pca_fits_in_a_fresh_worker():
    if _backend.vendor() in (None, "cpu"):
        pytest.skip("the cooperative Gram worker needs a GPU device")
    import mojolearn as ml
    from mojolearn.parallel_classical import fit_gram_estimator
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4096, 12)).astype(np.float32)
    par = fit_gram_estimator(ml.PCA(n_components=4), X, devices=(0,))
    plain = ml.PCA(n_components=4).fit(X)
    # The driver's model is the plain fit's, in this process.
    for name in ("components_", "mean_", "explained_variance_", "explained_variance_ratio_",
                 "singular_values_"):
        got, want = getattr(par, name), getattr(plain, name)
        assert type(got) is type(want), name
        assert got.dtype == want.dtype and got.shape == want.shape, name
        assert got.flags.c_contiguous == want.flags.c_contiguous, name
        assert np.asarray(got).tobytes() == np.asarray(want).tobytes(), name
    for name in ("n_components_", "n_features_in_", "noise_variance_"):
        assert getattr(par, name) == getattr(plain, name), name
    # transform and inverse_transform run on the driver's state.
    assert np.asarray(par.transform(X)).tobytes() == np.asarray(plain.transform(X)).tobytes()
    Z = np.asarray(plain.transform(X))
    assert np.asarray(par.inverse_transform(Z)).tobytes() == np.asarray(plain.inverse_transform(Z)).tobytes()


def test_native_kernel_ridge_driver_predict_is_the_plain_native_result():
    # fit_kernel_method / apply_kernel_method are one native call in this
    # process (bindings/kernel_methods_native.mojo `family_km_par_*`): no
    # worker, no worker NumPy flag; an internal caller gets the plain raw
    # float32 Array, byte for byte.
    if _backend.vendor() in (None, "cpu"):
        pytest.skip("the multi-GPU kernel-methods driver needs a GPU device")
    import mojolearn as ml
    from mojolearn import parallel_classical
    rng = np.random.default_rng(0)
    X = rng.standard_normal((256, 6)).astype(np.float32)
    y = rng.standard_normal(256).astype(np.float32)
    est = ml.KernelRidge(alpha=1.0, kernel="rbf").fit(X, y)
    # this module is mojolearn's own (`mojolearn.tests.*`): raw results
    plain = est.predict(X)
    got = parallel_classical.apply_kernel_method(est, X, devices=(0,))
    assert type(got) is type(plain)
    assert got.dtype == plain.dtype
    assert np.asarray(got).tobytes() == np.asarray(plain).tobytes()
