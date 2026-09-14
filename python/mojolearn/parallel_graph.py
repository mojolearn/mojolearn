# SPDX-License-Identifier: Apache-2.0
"""Native distance/query rows with each graph estimator's original global order."""
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
    """Distribute distance/neighbor rows before original graph/solver updates.

    AgglomerativeClustering retains its root MST and merge/tie order. Spectral
    retains its eigensolver and distributes the existing KMeans assignment.
    UMAP retains its graph construction, eigensolver and epoch/RNG order.
    Full reference data, graph and solver state still need to fit on the root.
    """
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
    """Distribute neighbor search only; retain the entire transform update order."""
    from ._umap_impl import UMAP
    if type(estimator) is not UMAP or getattr(estimator, '_transform_mode', None) != 'identical':
        raise ValueError('requires an IDENTICAL fitted UMAP')
    pool = DevicePool(devices, cooperative=True)
    try:
        return pool.map([('umap_transform', estimator, (X,))])[0]
    finally:
        pool.close()
