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

import sys

from ._parallel_pool import public_method_result
from ._buffer import _native, empty, addr, addr_ro
from . import neighbors as _nb
from . import _parallel_shard_ops
from . import _native_parallel_shard as _shard
from ._native_parallel_shard import _native_parallel_shard

# The reference-sharded queries' owner (bindings/parallel_shard_native.mojo
# `reference_query`): the reference blocks, the waves, the candidate merge
# and the join are native; a block's search and the vote run in the shard
# workers (python/mojolearn/_parallel_shard_ops.py).
_shard.register(knc_class=_nb.KNeighborsClassifier, knr_class=_nb.KNeighborsRegressor,
                py_ops=_parallel_shard_ops.OPERATIONS)


def _vote(model, distances, indices, method):
    """Worker-side original vote half, with no reference matrix present."""
    from .neighbors import KNeighborsClassifier, KNeighborsRegressor
    nq, k = distances.shape
    no = model._y_cols.shape[0]
    native = model._bind('_mojolearn')
    params = [model.n_samples_fit_, nq, 0, k, 0, no]
    args = (addr_ro(distances, name='distances'), addr_ro(indices, name='indices'),
            addr_ro(model._y_cols, name='targets'))
    if type(model) is KNeighborsRegressor and method == 'predict':
        if not callable(getattr(native, 'knn_regress_neighbors', None)):
            raise ImportError('rebuild base binding for merged-neighbor regression')
        output = empty((nq, no), '<f4')
        native.knn_regress_neighbors(*args, addr(output, name='output'), params, model._dist_params())
        return output if model.outputs_2d_ else output.reshape((nq,))
    if type(model) is not KNeighborsClassifier or method not in ('predict', 'predict_proba'):
        raise ValueError('unsupported reference-sharded vote')
    if not callable(getattr(native, 'knn_classify_neighbors', None)):
        raise ImportError('rebuild base binding for merged-neighbor classification')
    counts = [len(c) for c in model._classes_list]  # glue: class counts per output column
    labels = empty((nq, no), '<i4')
    proba = empty((nq * sum(counts),), '<f4')  # glue: class counts per output column
    uniq = empty((sum(counts),), '<i4')  # glue: class counts per output column
    native.knn_classify_neighbors(*args, addr(labels, name='labels'),
        addr(proba, name='probabilities'), addr(uniq, name='classes'),
        params + [int(method == 'predict_proba')] + counts, model._dist_params())
    offset = 0
    flat = uniq.tolist()
    for count, expected in zip(counts, model._classes_list):  # glue: class set check per output column
        if flat[offset:offset+count] != expected:
            raise RuntimeError('native and fitted class sets differ')
        offset += count
    if method == 'predict':
        labels = labels.astype('<i8')
        return labels if model.outputs_2d_ else labels.reshape((nq,))
    offset = 0
    result = []
    for count in counts:  # glue: class counts per output column
        result.append(proba[offset:offset+nq*count].reshape((nq, count)))
        offset += nq*count
    return result if model.outputs_2d_ else result[0]


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
        if type(estimator) not in (_nb.NearestNeighbors, _nb.KNeighborsClassifier, _nb.KNeighborsRegressor):
            raise TypeError('reference shards require a brute-force KNN estimator')
        if estimator.algorithm not in ('auto', 'brute'):
            raise ValueError('reference sharding currently requires the IDENTICAL brute-force arm')
        if estimator.numeric_mode not in (None, 'identical'):
            raise ValueError('reference shards require IDENTICAL numeric mode')
        for name, value in (('reference_rows_per_shard', reference_rows_per_shard),  # glue: validates two named arguments
                            ('query_rows_per_shard', query_rows_per_shard)):
            if value is None and name == 'reference_rows_per_shard':
                continue
            if type(value) is not int or value < 1:
                raise ValueError(name + ' must be a positive integer')
        self.estimator = estimator
        self.reference_rows_per_shard = reference_rows_per_shard
        self.query_rows_per_shard = query_rows_per_shard
        self._pool = _native_parallel_shard('pool_open', (devices,))
        self.last_shards_ = []

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
        return public_method_result(self.estimator, 'kneighbors',
                                    _native_parallel_shard('reference_query', (self, X, 'kneighbors', n_neighbors,
                                                                               return_distance,
                                                                               _native('shard_topk_merge_f32'))),
                                    sys._getframe(1))

    def predict(self, X):
        """Predict labels (classifier) or values (regressor) for ``X``, as ``estimator.predict(X)``.
        """
        # One native call (bindings/parallel_shard_native.mojo `reference_query`).
        return public_method_result(self.estimator, 'predict',
                                    _native_parallel_shard('reference_query', (self, X, 'predict', None, True,
                                                                               _native('shard_topk_merge_f32'))),
                                    sys._getframe(1))

    def predict_proba(self, X):
        """Class probabilities for ``X`` from a ``KNeighborsClassifier``, as ``estimator.predict_proba(X)``.
        """
        # One native call (bindings/parallel_shard_native.mojo `reference_query`).
        return public_method_result(self.estimator, 'predict_proba',
                                    _native_parallel_shard('reference_query', (self, X, 'predict_proba', None, True,
                                                                               _native('shard_topk_merge_f32'))),
                                    sys._getframe(1))

    def close(self):
        """Stop the worker processes."""
        _native_parallel_shard('pool_close', (self._pool,))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
