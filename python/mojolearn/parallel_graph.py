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

from . import _native_family_bridge, _native_parallel_public

_FAMILY, _FAMILY_CONTEXT = _native_family_bridge.connect()
_native_parallel_public.connect(_FAMILY_CONTEXT)


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
    return _FAMILY(estimator, 'family_par_public', ('fit_graph', (X, devices)), _FAMILY_CONTEXT)


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
    return _FAMILY(estimator, 'family_par_public', ('transform_umap', (X, devices)), _FAMILY_CONTEXT)
