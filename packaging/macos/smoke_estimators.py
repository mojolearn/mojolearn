"""End-to-end checks for the secondary estimator extension on a real GPU."""

import numpy as np

from mojolearn.decomposition import PCA, TruncatedSVD
from mojolearn.density import DBSCAN
from mojolearn.linear_model import LinearRegression

rng = np.random.default_rng(42)

# DBSCAN: two dense, separated clouds and four isolated noise points.
clouds = np.vstack([
    rng.normal((-4, -4), 0.08, (40, 2)),
    rng.normal((4, 4), 0.08, (40, 2)),
    [[-10, 8], [10, -8], [0, 10], [10, 0]],
]).astype(np.float32)
labels = DBSCAN(eps=0.35, min_samples=5).fit_predict(clouds)
assert len(set(labels[:40])) == 1
assert len(set(labels[40:80])) == 1
assert labels[0] != labels[40]
assert (labels[-4:] == -1).all()

# PCA: covariance eigenvalues and round trip against NumPy identities.
x = rng.normal(size=(512, 6)).astype(np.float32)
x[:, 1] += 0.7 * x[:, 0]
pca = PCA(n_components=6).fit(x)
expected = np.linalg.eigvalsh(np.cov(x, rowvar=False))[::-1]
np.testing.assert_allclose(pca.explained_variance_, expected, rtol=2e-3, atol=2e-4)
z = pca.transform(x)
np.testing.assert_allclose(pca.inverse_transform(z), x, rtol=2e-3, atol=2e-3)

# Truncated SVD: singular values and full-rank reconstruction.
svd = TruncatedSVD(n_components=6).fit(x)
np.testing.assert_allclose(
    svd.singular_values_, np.linalg.svd(x, compute_uv=False), rtol=2e-3, atol=2e-3
)
zs = svd.transform(x)
np.testing.assert_allclose(svd.inverse_transform(zs), x, rtol=2e-3, atol=2e-3)

# OLS: fit with an intercept, GPU prediction, and a planted exact model.
coef = np.array([1.5, -2.0, 0.25, 4.0, -1.0, 0.5], dtype=np.float32)
y = x @ coef + np.float32(3.25)
ols = LinearRegression().fit(x, y)
np.testing.assert_allclose(ols.coef_, coef, rtol=3e-3, atol=3e-3)
np.testing.assert_allclose(ols.predict(x), y, rtol=3e-3, atol=3e-3)

print("secondary estimators OK: DBSCAN, PCA, TruncatedSVD, LinearRegression")
