# SPDX-License-Identifier: Apache-2.0
"""Whole-query GPU partitions with the original complete reference-row order."""
from ._parallel_pool import DevicePool, driver_read_shift
from ._buffer import as_f32_c, empty, addr, addr_ro
from ._bufcheck import memcopy


def _methods(estimator):
    from .neighbors import (NearestNeighbors, RadiusNeighbors,
                            KNeighborsClassifier, KNeighborsRegressor)
    from .density import KernelDensity
    return {
        NearestNeighbors: ('kneighbors',),
        RadiusNeighbors: ('radius_neighbors',),
        KNeighborsClassifier: ('kneighbors', 'predict', 'predict_proba'),
        KNeighborsRegressor: ('kneighbors', 'predict'),
        KernelDensity: ('score_samples',),
    }.get(type(estimator), ())


def _join(parts, *, ragged=False):
    first = parts[0]
    if isinstance(first, tuple):
        return tuple(_join([p[i] for p in parts], ragged=ragged)
                     for i in range(len(first)))
    if isinstance(first, list):
        if ragged:
            return [row for part in parts for row in part]
        # Multi-output classification: one probability matrix per target.
        return [_join([p[i] for p in parts]) for i in range(len(first))]
    shape = (sum(p.shape[0] for p in parts),) + first.shape[1:]
    result = empty(shape, first.dtype)
    destination = addr(result, name='query output')
    offset = 0
    for part in parts:
        if part.dtype != first.dtype or part.shape[1:] != first.shape[1:]:
            raise RuntimeError('inconsistent query shard result')
        if part.nbytes:
            memcopy(destination + offset, addr_ro(part, name='query shard'), part.nbytes)
        offset += part.nbytes
    return result


class ParallelQueries:
    """Persistent GPU workers for fitted neighbors and KernelDensity estimators.

    Each query retains the complete index and the original distance, selection,
    voting or density reduction. Only query rows are partitioned: the complete
    host input and each worker's reference index must fit in memory. Use as a
    context manager, or call close(). Calls on one driver must be serialized.
    Estimator state is sent afresh on each call; fit the estimator normally.
    """
    def __init__(self, estimator, *, devices=(0,), rows_per_shard=128):
        if not _methods(estimator):
            raise TypeError('parallel queries require an exact neighbors or KernelDensity estimator')
        if estimator.numeric_mode not in (None, 'identical'):
            raise ValueError('parallel queries require IDENTICAL numeric mode')
        if type(rows_per_shard) is not int or rows_per_shard < 1:
            raise ValueError('rows_per_shard must be a positive integer')
        self.estimator = estimator
        self.rows_per_shard = rows_per_shard
        self._pool = DevicePool(devices)
        self.last_shards_ = []

    def query(self, X=None, *, method=None, **kwargs):
        import copy
        methods = _methods(self.estimator)
        method = methods[0] if method is None else method
        if method not in methods:
            raise ValueError('unsupported parallel query method: ' + str(method))
        if self.estimator.numeric_mode not in (None, 'identical'):
            raise ValueError('parallel queries require IDENTICAL numeric mode')
        index = getattr(self.estimator, '_index', getattr(self.estimator, '_x', None))
        if index is None:
            raise ValueError('fit the estimator before querying')
        if X is None and method == 'radius_neighbors':
            X = index  # Preserve the original inclusion of self edges.
        data, _ = as_f32_c(X, ndim=2, name='X')
        if data.shape[1] != self.estimator.n_features_in_:
            raise ValueError('input feature count differs from fit')
        n = data.shape[0]
        ranges = [(i, min(i + self.rows_per_shard, n))
                  for i in range(0, n, self.rows_per_shard)] or [(0, 0)]
        state = copy.copy(self.estimator)
        state.numeric_mode = 'identical'
        # The READ is shifted by `driver_read_shift` (0 unless the driver
        # sabotage switch is on, and 0 for the first shard either way); the
        # join below is still in shard order over unchanged widths.
        results = self._pool.map([
            ('neighbor_query', state,
             (data[start - driver_read_shift(index, start):
                   end - driver_read_shift(index, start)], method, kwargs))
            for index, (start, end) in enumerate(ranges)])
        output = _join([r[0] for r in results], ragged=method == 'radius_neighbors')
        # Diagnostics describe actual shard calls, not a fictitious global tile.
        diagnostics = [dict(start=start, end=end,
                            device=self._pool.devices[i % len(self._pool.devices)],
                            **result[1])
                       for i, ((start, end), result) in enumerate(zip(ranges, results))]
        self.last_shards_ = diagnostics
        return output

    def close(self):
        self._pool.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
