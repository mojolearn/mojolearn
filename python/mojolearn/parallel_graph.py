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

import sys

from . import _backend, _native_family_bridge
from ._hierarchy_impl import AgglomerativeClustering
from ._spectral_impl import SpectralClustering
from ._umap_impl import UMAP
from ._parallel_pool import public_method_result
from . import _metrics_impl, _mojolearn_solver, _expansion_cluster

# The classical families' compiled owner: each driver is one `family_par_run`
# call (bindings/family_native.mojo).
_FAMILY, _FAMILY_CONTEXT = _native_family_bridge.connect()


# glue: binding acquisition only. Each entry packs the plain `fit`'s own
# owner arguments (its bindings, then X), word for word, for the estimator's
# type; no data value is touched.
_GRAPH_FIT = {
    AgglomerativeClustering: ('family_cfb_agglo_fit', lambda estimator, X: (
        estimator._bind(estimator._BINDING), _mojolearn_solver,
        _backend.binding(_expansion_cluster._XCluster._BINDING, getattr(estimator, "numeric_mode", None)),
        _backend.binding("_mojolearn_x_neighbors", getattr(estimator, "numeric_mode", None)), X)),
    SpectralClustering: ('family_cfb_sc_fit', lambda estimator, X: (
        _metrics_impl._get_binding(), _backend.binding("_mojolearn_x_decomp", _backend.default_mode()),
        _backend.binding("_mojolearn", "identical"),
        _backend.binding("_mojolearn_x_cluster", getattr(estimator, "numeric_mode", None)), X)),
    UMAP: ('family_mani_umap_fit', lambda estimator, X: (estimator._bind(), X, None)),
}

#: The bindings whose owner loops a graph fit or transform may reach (the old
#: worker's environment reached every native module of its process).
_GRAPH_BINDINGS = ('_mojolearn_metrics', '_mojolearn', '_mojolearn_estimators', '_mojolearn_solver',
                   '_mojolearn_x_cluster', '_mojolearn_x_neighbors', '_mojolearn_x_decomp',
                   '_mojolearn_hdbscan')


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
    # One native call (`family_par_run`, driver `graph_fit`): the plain fit
    # with its distance and neighbor rows on the selected devices.
    # AgglomerativeClustering retains its root MST and merge/tie order.
    # Spectral retains its eigensolver and distributes the existing KMeans
    # assignment. UMAP retains its graph construction, eigensolver and
    # epoch/RNG order. Full reference data, graph and solver state still need
    # to fit on the root. The plain fit converts X itself, so its
    # `input_copied_` is its own.
    if type(estimator) not in _GRAPH_FIT:
        raise TypeError('unsupported parallel graph estimator')
    if (getattr(estimator, 'numeric_mode', None) or _backend.default_mode()) != 'identical':
        raise ValueError('parallel graph estimators require IDENTICAL numeric mode')
    operation, owner_args = _GRAPH_FIT[type(estimator)]
    return _FAMILY(estimator, 'family_par_run',
                   ('graph_fit', devices, operation, owner_args(estimator, X), None, None,
                    _backend.plan_bindings(_GRAPH_BINDINGS)), _FAMILY_CONTEXT)


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
    # One native call (`family_par_run`, driver `umap_transform`): the
    # neighbor search on the selected devices, the transform's update order
    # unchanged; the plain method's public conversion for our caller.
    if type(estimator) is not UMAP or getattr(estimator, '_transform_mode', None) != 'identical':
        raise ValueError('requires an IDENTICAL fitted UMAP')
    return public_method_result(estimator, 'transform',
                                _FAMILY(estimator, 'family_par_run',
                                        ('umap_transform', devices, 'family_mani_umap_transform',
                                         (_backend.binding(estimator._BINDING, 'identical'), X), None, None,
                                         _backend.plan_bindings(_GRAPH_BINDINGS)), _FAMILY_CONTEXT),
                                sys._getframe(1))
