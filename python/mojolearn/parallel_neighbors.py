# SPDX-License-Identifier: Apache-2.0
"""Answer neighbor and density queries with the query rows spread across GPUs.

Every GPU holds the whole fitted index and answers a block of query rows.
Results are the same, bit for bit, as the estimator's own query methods.

Example
-------
>>> import numpy as np
>>> from mojolearn import KNeighborsClassifier
>>> from mojolearn.parallel_neighbors import ParallelQueries
>>> rng = np.random.default_rng(0)
>>> X = rng.standard_normal((4096, 16)).astype(np.float32)
>>> y = (X[:, 0] > 0).astype(np.int64)
>>> model = KNeighborsClassifier(n_neighbors=8).fit(X, y)
>>> with ParallelQueries(model, devices=(0, 1), rows_per_shard=256) as pq:
...     labels = pq.query(X[:1000], method="predict")
"""
# Whole-query GPU partitions with the original complete reference-row order.
__all__ = [
    'ParallelQueries',
]

import sys

from ._parallel_pool import public_method_result
from . import neighbors as _nb
from . import density as _den
from . import _parallel_shard_ops
from . import _native_parallel_shard as _shard
from ._native_parallel_shard import _native_parallel_shard


#: The query methods each estimator answers (the first is the default).
_METHODS = {
    _nb.NearestNeighbors: ('kneighbors',),
    _nb.RadiusNeighbors: ('radius_neighbors',),
    _nb.KNeighborsClassifier: ('kneighbors', 'predict', 'predict_proba'),
    _nb.KNeighborsRegressor: ('kneighbors', 'predict'),
    _den.KernelDensity: ('score_samples',),
}

# The parallel queries' owner (bindings/parallel_shard_native.mojo
# `parallel_queries`): the snapshots, the query blocks, the scheduling and the
# join are native; a block's query runs in the shard workers
# (python/mojolearn/_parallel_shard_ops.py).
_shard.register(query_methods=_METHODS, py_ops=_parallel_shard_ops.OPERATIONS)


def _methods(estimator):
    return _METHODS.get(type(estimator), ())


class ParallelQueries:
    """GPU workers that answer queries for one fitted neighbors or density estimator.

    Parameters
    ----------
    estimator : NearestNeighbors, RadiusNeighbors, KNeighborsClassifier, KNeighborsRegressor or KernelDensity
        A fitted estimator using the ``'identical'`` numeric mode (or the default).
    devices : tuple of int, default=(0,)
        Distinct GPU indices; each runs one worker.
    rows_per_shard : int, default=128
        Number of query rows sent to a worker at a time.

    Attributes
    ----------
    last_shards_ : list of dict
        One record per block of the last query: its row range and device.

    Use it as a context manager, or call :meth:`close` to stop the workers.
    Calls on one instance must not overlap.
    """
    # Persistent GPU workers for fitted neighbors and KernelDensity estimators.
    #
    # Each query retains the complete index and the original distance, selection,
    # voting or density reduction. Only query rows are partitioned: the complete
    # host input and each worker's reference index must fit in memory. Use as a
    # context manager, or call close(). Calls on one driver must be serialized.
    # Estimator state is sent afresh on each call; fit the estimator normally.
    #
    # The state travels ONCE per worker per call (lane/py-decomp-nbrs,
    # 2026-09-28): `query` sends one `neighbor_state` request to every worker,
    # which keeps the unpickled estimator (and the resident index handle its
    # first shard builds) under a per-call token; each shard then carries only
    # its query slice and that token. Before, every shard pickled the whole
    # index and its worker uploaded it again (about 1 TB pickled for 1M
    # queries on a 1M x 32 index at 128 rows per shard). The shard cut, the
    # per-shard call and the merge are unchanged, so the bytes are too. The
    # workers drop the snapshot when the call ends.
    def __init__(self, estimator, *, devices=(0,), rows_per_shard=128):
        if not _methods(estimator):
            raise TypeError('parallel queries require an exact neighbors or KernelDensity estimator')
        if estimator.numeric_mode not in (None, 'identical'):
            raise ValueError('parallel queries require IDENTICAL numeric mode')
        if type(rows_per_shard) is not int or rows_per_shard < 1:
            raise ValueError('rows_per_shard must be a positive integer')
        self.estimator = estimator
        self.rows_per_shard = rows_per_shard
        self._pool = _native_parallel_shard('pool_open', (devices,))
        self.last_shards_ = []

    def query(self, X=None, *, method=None, **kwargs):
        """Run one query method of the estimator over ``X``.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), optional
            Query rows. ``RadiusNeighbors`` with ``X=None`` queries the fitted
            rows themselves, as ``radius_neighbors()`` does.
        method : str, optional
            ``'kneighbors'`` (NearestNeighbors and the k-NN estimators),
            ``'radius_neighbors'`` (RadiusNeighbors), ``'predict'`` and
            ``'predict_proba'`` (classifier; ``'predict'`` also for the regressor) or
            ``'score_samples'`` (KernelDensity). Defaults to the first method the
            estimator supports in that order.
        **kwargs
            Passed to the estimator's method, for example ``n_neighbors`` or
            ``sort_results``.

        Returns
        -------
        The same value the estimator's method returns for ``X``.
        """
        # One native call (bindings/parallel_shard_native.mojo `parallel_queries`).
        return public_method_result(self.estimator, _methods(self.estimator)[0] if method is None else method,
                                    _native_parallel_shard('parallel_queries', (self, X, method, kwargs)),
                                    sys._getframe(1))

    def close(self):
        """Stop the worker processes."""
        _native_parallel_shard('pool_close', (self._pool,))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
