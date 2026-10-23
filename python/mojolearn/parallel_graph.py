# SPDX-License-Identifier: Apache-2.0
"""Fit graph-based clustering and embedding estimators with their distance and
neighbor rows spread across several GPUs.

``fit_graph`` accepts ``AgglomerativeClustering``, ``SpectralClustering`` and
``UMAP``; ``transform_umap`` embeds new rows with a fitted ``UMAP``. Results
are the same, bit for bit, as the one-GPU ``fit`` and ``transform``.

Example
-------
>>> import numpy as np
>>> from mojolearn import UMAP
>>> from mojolearn.parallel_graph import fit_graph, transform_umap
>>> X = np.random.default_rng(0).standard_normal((1024, 8)).astype(np.float32)
>>> model = fit_graph(UMAP(n_neighbors=8, random_state=3), X, devices=(0, 1))
>>> Y = transform_umap(model, X[:64], devices=(0, 1))
"""
# Native distance/query rows with each graph estimator's original global order.
__all__ = [
    'fit_graph',
    'transform_umap',
]

from ._parallel_pool import DevicePool


def _binding(estimator):
    from ._hierarchy_impl import AgglomerativeClustering
    from ._spectral_impl import SpectralClustering
    from ._umap_impl import UMAP
    if type(estimator) is AgglomerativeClustering:
        from . import _mojolearn_solver
        return _mojolearn_solver, 'hierarchy_parallel_available'
    if type(estimator) in (SpectralClustering, UMAP):
        from ._metrics_impl import _get_binding
        return _get_binding('identical'), 'graph_parallel_available'
    raise TypeError('parallel graph fits require AgglomerativeClustering, SpectralClustering or UMAP')


def fit_graph(estimator, X, *, devices=(0,)):
    """Fit ``AgglomerativeClustering``, ``SpectralClustering`` or ``UMAP`` on several GPUs.

    The pairwise-distance and nearest-neighbor rows are computed across the
    selected GPUs; the merge tree, eigensolver and layout optimization run as in
    a one-GPU fit.

    Parameters
    ----------
    estimator : AgglomerativeClustering, SpectralClustering or UMAP
        An unfitted estimator using the ``'identical'`` numeric mode (or the default).
    X : array-like of shape (n_samples, n_features)
        Training data, converted to float32. A ``SpectralClustering`` with
        ``affinity='precomputed'`` takes its affinity matrix as given.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.

    Returns
    -------
    estimator
        The same estimator, fitted, with the attributes a one-GPU ``fit`` sets.
    """
    # Distribute distance/neighbor rows before original graph/solver updates.
    #
    # AgglomerativeClustering retains its root MST and merge/tie order. Spectral
    # retains its eigensolver and distributes the existing KMeans assignment.
    # UMAP retains its graph construction, eigensolver and epoch/RNG order.
    # Full reference data, graph and solver state still need to fit on the root.
    from . import _backend
    from ._hierarchy_impl import AgglomerativeClustering
    from ._spectral_impl import SpectralClustering
    from ._umap_impl import UMAP
    from ._buffer import as_f32_c
    if type(estimator) not in (AgglomerativeClustering, SpectralClustering, UMAP):
        raise TypeError('unsupported parallel graph estimator')
    mode = getattr(estimator, 'numeric_mode', None) or _backend.default_mode()
    if mode != 'identical':
        raise ValueError('parallel graph estimators require IDENTICAL numeric mode')
    copied = None
    if not (type(estimator) is SpectralClustering and estimator.affinity == 'precomputed'):
        X, copied = as_f32_c(X, ndim=2, name='X')
    pool = DevicePool(devices, cooperative=True)
    try:
        result = pool.map([('graph_fit', estimator, (X,))])[0]
    finally:
        pool.close()
    if copied is not None and hasattr(result, 'input_copied_'):
        result.input_copied_ = copied
    estimator.__dict__ = result.__dict__.copy()
    return estimator


def transform_umap(estimator, X, *, devices=(0,)):
    """Embed new rows with a fitted ``UMAP``, the neighbor search spread across GPUs.

    Parameters
    ----------
    estimator : UMAP
        A ``UMAP`` fitted in the ``'identical'`` numeric mode.
    X : array-like of shape (n_samples, n_features)
        Rows to embed.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.

    Returns
    -------
    Array of shape (n_samples, n_components)
        The same embedding as ``estimator.transform(X)``.
    """
    # Distribute neighbor search only; retain the entire transform update order.
    from ._umap_impl import UMAP
    if type(estimator) is not UMAP or getattr(estimator, '_transform_mode', None) != 'identical':
        raise ValueError('requires an IDENTICAL fitted UMAP')
    pool = DevicePool(devices, cooperative=True)
    try:
        return pool.map([('umap_transform', estimator, (X,))])[0]
    finally:
        pool.close()
