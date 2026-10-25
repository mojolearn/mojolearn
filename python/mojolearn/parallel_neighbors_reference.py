# SPDX-License-Identifier: Apache-2.0
"""Exact k-nearest-neighbor search over a reference set split across GPUs.

Each GPU holds only a block of the fitted reference rows, so the index can
be larger than one GPU's memory. Neighbors, predictions and probabilities
are the same, bit for bit, as the estimator's own methods.

Example
-------
>>> import numpy as np
>>> from mojolearn import KNeighborsClassifier
>>> from mojolearn.parallel_neighbors_reference import ReferenceShardedNeighbors
>>> rng = np.random.default_rng(0)
>>> X = rng.standard_normal((4096, 16)).astype(np.float32)
>>> y = (X[:, 0] > 0).astype(np.int64)
>>> model = KNeighborsClassifier(n_neighbors=8).fit(X, y)
>>> with ReferenceShardedNeighbors(model, devices=(0, 1)) as rs:
...     distances, indices = rs.kneighbors(X[:64])
...     labels = rs.predict(X[:64])
"""
# Reference partitions with the native IDENTICAL top-k composite ordering.
__all__ = [
    'ReferenceShardedNeighbors',
]

from . import neighbors as _nb
from . import _native_parallel_shard as _shard
from ._native_parallel_shard import _native_parallel_shard

# Import-time class metadata; shard queries and votes are native operations.
_shard.register(nearest_class=_nb.NearestNeighbors, knc_class=_nb.KNeighborsClassifier,
                knr_class=_nb.KNeighborsRegressor, context=_nb._NATIVE_CONTEXT)


class ReferenceShardedNeighbors:
    """Brute-force k-NN with the fitted reference rows split across GPUs.

    Parameters
    ----------
    estimator : NearestNeighbors, KNeighborsClassifier or KNeighborsRegressor
        A fitted estimator with ``algorithm='auto'`` or ``'brute'``, using the
        ``'identical'`` numeric mode (or the default).
    devices : tuple of int, default=(0,)
        Distinct GPU indices; each runs one worker.
    reference_rows_per_shard : int, optional
        Number of reference rows per block. By default the reference set is
        split evenly over ``devices``.
    query_rows_per_shard : int, default=128
        Number of query rows answered per round.

    Attributes
    ----------
    last_shards_ : list of dict
        One record per (query block, reference block) of the last call.

    Use it as a context manager, or call :meth:`close` to stop the workers.
    """
    # Partition a fitted brute-force KNN reference matrix across GPU workers.
    #
    # Workers receive only their assigned reference shard. Whole-feature distance
    # cells and the native (distance-bit key, global index) ordering are retained.
    # Each reference shard plus the original native workspace must fit on a GPU.
    # The complete input lives on the host. Classification/regression currently
    # retain the complete target table on the voting GPU. Use a context manager.
    def __init__(self, estimator, *, devices=(0,), reference_rows_per_shard=None,
                 query_rows_per_shard=128):
        _native_parallel_shard('reference_init', (self, estimator, devices, reference_rows_per_shard, query_rows_per_shard))

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """Find the ``n_neighbors`` nearest reference rows of each query row.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Query rows.
        n_neighbors : int, optional
            Number of neighbors; defaults to the estimator's ``n_neighbors``.
        return_distance : bool, default=True
            Also return the distances.

        Returns
        -------
        distances, indices : arrays of shape (n_queries, n_neighbors)
            As ``estimator.kneighbors(X)``; only ``indices`` when
            ``return_distance=False``.
        """
        # One native call (bindings/parallel_shard_native.mojo `reference_query`).
        return _native_parallel_shard('reference_query', (self, X, 'kneighbors', n_neighbors, return_distance))

    def predict(self, X):
        """Predict labels (classifier) or values (regressor) for ``X``, as ``estimator.predict(X)``.
        """
        # One native call (bindings/parallel_shard_native.mojo `reference_query`).
        return _native_parallel_shard('reference_query', (self, X, 'predict', None, True))

    def predict_proba(self, X):
        """Class probabilities for ``X`` from a ``KNeighborsClassifier``, as ``estimator.predict_proba(X)``.
        """
        # One native call (bindings/parallel_shard_native.mojo `reference_query`).
        return _native_parallel_shard('reference_query', (self, X, 'predict_proba', None, True))

    def close(self):
        """Stop the worker processes."""
        _native_parallel_shard('holder_close', (self,))

    def __enter__(self):
        return _native_parallel_shard('holder_enter', (self,))

    def __exit__(self, *exc):
        _native_parallel_shard('holder_close', (self,))


_shard.register(reference_driver_class=ReferenceShardedNeighbors)
