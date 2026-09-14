# SPDX-License-Identifier: Apache-2.0
"""Cooperative classical estimators with explicit per-algorithm partitions."""
from ._parallel_pool import DevicePool
from ._buffer import as_f32_c


def fit_kmeans(estimator, X, *, devices=(0,), sample_weight=None):
    """Distribute KMeans row assignments; retain full-data update/reduction order.

    Initialization, fixed-point centroid sums, restart choice and convergence
    use the existing implementation. Whole aligned row tiles compute distances
    concurrently. This first path stages buffers per assignment call.
    """
    from .cluster import KMeans
    if type(estimator) is not KMeans:
        raise TypeError('fit_kmeans requires mojolearn.KMeans')
    params = {name: getattr(estimator, name) for name in (
        'n_clusters', 'init', 'n_init', 'max_iter', 'tol', 'random_state', 'init_centroids')}
    mode = getattr(estimator, 'numeric_mode', None)
    if mode not in (None, 'identical'):
        raise ValueError('parallel KMeans requires IDENTICAL numeric mode')
    params['numeric_mode'] = 'identical'
    X, _ = as_f32_c(X, ndim=2, name='X')
    if sample_weight is not None:
        sample_weight, _ = as_f32_c(sample_weight, ndim=1, name='sample_weight')
    pool = DevicePool(devices, cooperative=True)
    try:
        result = pool.map([('kmeans_fit', params, (X, sample_weight))])[0]
    finally:
        pool.close()
    estimator.__dict__ = result.__dict__.copy()
    return estimator
