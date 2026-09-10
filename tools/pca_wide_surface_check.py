#!/usr/bin/env python3
"""Exercise the public IDENTICAL full-PCA wide route and spectral invariants."""
import hashlib
import numpy as np
from mojolearn import PCA

rng = np.random.default_rng(910)
for m, n in ((4, 8), (8, 17), (17, 33), (4, 129)):
    x = rng.uniform(-1, 1, (m, n)).astype(np.float32)
    fit = PCA(svd_solver='full', numeric_mode='identical').fit(x)
    assert fit.components_.shape == (m, n)
    ref = np.linalg.svd(x.astype(np.float64)-x.mean(axis=0, dtype=np.float64), compute_uv=False)
    np.testing.assert_allclose(fit.singular_values_, ref, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(fit.components_ @ fit.components_.T, np.eye(m), rtol=0, atol=3e-5)
    rebuilt = fit.inverse_transform(fit.transform(x))
    np.testing.assert_allclose(rebuilt, x, rtol=0, atol=3e-5)
    one = PCA(n_components=1, svd_solver='full', numeric_mode='identical').fit(x)
    np.testing.assert_allclose(one.noise_variance_, (ref[1:]**2).sum() / ((m-1)*(m-1)), rtol=3e-5, atol=1e-6)
    white = PCA(n_components=m-1, svd_solver='full', whiten=True, numeric_mode='identical').fit(x)
    z = white.transform(x)
    np.testing.assert_allclose(z.var(axis=0, ddof=1), 1, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(white.inverse_transform(z), x, rtol=0, atol=3e-5)
    print('PCA_WIDE_PUBLIC', m, n, hashlib.sha256(fit.components_.tobytes()).hexdigest())
try:
    PCA(n_components=5, svd_solver='full', numeric_mode='identical').fit(np.zeros((4,8), np.float32))
except ValueError as e:
    assert 'min(n_samples, n_features)' in str(e)
else:
    raise AssertionError('wide overlarge component count accepted')
print('PCA wide public PASS')
