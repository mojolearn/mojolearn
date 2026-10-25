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

from . import neighbors as _nb
from . import density as _den
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

# Import-time exact classes/methods; workers dispatch directly to native owners.
_shard.register(query_methods=_METHODS,
                query_classes={cls.__name__: cls for cls in _METHODS},  # glue: fixed supported class metadata
                nearest_class=_nb.NearestNeighbors, knc_class=_nb.KNeighborsClassifier,
                knr_class=_nb.KNeighborsRegressor, kde_class=_den.KernelDensity,
                context=_nb._NATIVE_CONTEXT)


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
        _native_parallel_shard('queries_init', (self, estimator, devices, rows_per_shard))

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
        return _native_parallel_shard('parallel_queries', (self, X, method, kwargs))

    def close(self):
        """Stop the worker processes."""
        _native_parallel_shard('holder_close', (self,))

    def __enter__(self):
        return _native_parallel_shard('holder_enter', (self,))

    def __exit__(self, *exc):
        _native_parallel_shard('holder_close', (self,))


_shard.register(query_driver_class=ParallelQueries)
