# SPDX-License-Identifier: Apache-2.0
"""Reference partitions with the native IDENTICAL top-k composite ordering."""
import copy
import ctypes
import heapq
import struct

from ._parallel_pool import DevicePool, driver_read_shift
from ._buffer import as_f32_c, empty, addr, addr_ro
from ._bufcheck import memcopy
from .parallel_neighbors import _join


def _key(bits, index):
    # neighbors/checks/select_radix_identical.mojo::composite_key:
    # preserve signed zero and NaN payload order, without float conversion.
    twiddled = bits ^ (0xffffffff if bits & 0x80000000 else 0x80000000)
    return (twiddled << 32) | index


def _merge(parts, ranges, n_queries, k):
    distances = empty((n_queries, k), '<f4')
    indices = empty((n_queries, k), '<u4')
    db = addr(distances, name='merged distances')
    ib = addr(indices, name='merged indices')
    values = [p[0].tobytes() for p in parts]
    ids = [p[1].tolist() for p in parts]
    for row in range(n_queries):
        candidates = []
        for rank, ((first, end), (dist, _)) in enumerate(zip(ranges, parts)):
            width = dist.shape[1]
            for j in range(width):
                local_id = ids[rank][row][j]
                if not 0 <= local_id < end - first:
                    raise RuntimeError('reference worker returned an invalid neighbor index')
                index = first + local_id
                offset = (row * width + j) * 4
                bits = struct.unpack_from('<I', values[rank], offset)[0]
                candidates.append((_key(bits, index), rank, offset, index))
        chosen = heapq.nsmallest(k, candidates)
        if len(chosen) != k:
            raise RuntimeError('reference shards returned too few neighbors')
        # The native selector ranks composite keys, then estimator.mojo applies
        # this host insertion sort. Its float comparisons intentionally differ
        # from radix ordering for signed zero and NaNs. Preserve both stages.
        for a in range(1, k):
            token = chosen[a]
            dv = struct.unpack_from('<f', values[token[1]], token[2])[0]
            b = a - 1
            while b >= 0:
                previous = chosen[b]
                dbits = struct.unpack_from('<f', values[previous[1]], previous[2])[0]
                if dbits < dv or (dbits == dv and previous[3] <= token[3]):
                    break
                chosen[b+1] = previous
                b -= 1
            chosen[b+1] = token
        for j, (_, rank, offset, index) in enumerate(chosen):
            # Float bits travel unchanged from the winning GPU slot.
            memcopy(db + (row * k + j) * 4,
                    addr_ro(parts[rank][0], name='distance shard') + offset, 4)
            packed = struct.pack('<I', index)
            ctypes.memmove(ib + (row * k + j) * 4, packed, 4)
    return distances, indices


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
    counts = [len(c) for c in model._classes_list]
    labels = empty((nq, no), '<i4')
    proba = empty((nq * sum(counts),), '<f4')
    uniq = empty((sum(counts),), '<i4')
    native.knn_classify_neighbors(*args, addr(labels, name='labels'),
        addr(proba, name='probabilities'), addr(uniq, name='classes'),
        params + [int(method == 'predict_proba')] + counts, model._dist_params())
    offset = 0
    flat = uniq.tolist()
    for count, expected in zip(counts, model._classes_list):
        if flat[offset:offset+count] != expected:
            raise RuntimeError('native and fitted class sets differ')
        offset += count
    if method == 'predict':
        labels = labels.astype('<i8')
        return labels if model.outputs_2d_ else labels.reshape((nq,))
    offset = 0
    result = []
    for count in counts:
        result.append(proba[offset:offset+nq*count].reshape((nq, count)))
        offset += nq*count
    return result if model.outputs_2d_ else result[0]


class ReferenceShardedNeighbors:
    """Partition a fitted brute-force KNN reference matrix across GPU workers.

    Workers receive only their assigned reference shard. Whole-feature distance
    cells and the native (distance-bit key, global index) ordering are retained.
    Each reference shard plus the original native workspace must fit on a GPU.
    The complete input lives on the host. Classification/regression currently
    retain the complete target table on the voting GPU. Use a context manager.
    """
    def __init__(self, estimator, *, devices=(0,), reference_rows_per_shard=None,
                 query_rows_per_shard=128):
        from .neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
        if type(estimator) not in (NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor):
            raise TypeError('reference shards require a brute-force KNN estimator')
        if estimator.algorithm not in ('auto', 'brute'):
            raise ValueError('reference sharding currently requires the IDENTICAL brute-force arm')
        if estimator.numeric_mode not in (None, 'identical'):
            raise ValueError('reference shards require IDENTICAL numeric mode')
        for name, value in (('reference_rows_per_shard', reference_rows_per_shard),
                            ('query_rows_per_shard', query_rows_per_shard)):
            if value is None and name == 'reference_rows_per_shard':
                continue
            if type(value) is not int or value < 1:
                raise ValueError(name + ' must be a positive integer')
        self.estimator = estimator
        self.reference_rows_per_shard = reference_rows_per_shard
        self.query_rows_per_shard = query_rows_per_shard
        self._pool = DevicePool(devices)
        self.last_shards_ = []

    def _query(self, X, *, method, n_neighbors=None, return_distance=True):
        from .neighbors import KNeighborsClassifier, KNeighborsRegressor
        model = self.estimator
        if model.numeric_mode not in (None, 'identical') or model.algorithm not in ('auto', 'brute'):
            raise ValueError('reference shards require IDENTICAL brute-force KNN')
        if getattr(model, '_index', None) is None:
            raise ValueError('fit the estimator before querying')
        if method != 'kneighbors':
            if type(model) not in (KNeighborsClassifier, KNeighborsRegressor):
                raise TypeError('prediction requires a classifier or regressor')
            if method == 'predict_proba' and type(model) is not KNeighborsClassifier:
                raise TypeError('predict_proba requires a classifier')
            if model._y_cols is None:
                raise ValueError('fit the estimator before prediction')
        data, _ = as_f32_c(X, ndim=2, name='X')
        if data.shape[1] != model.n_features_in_:
            raise ValueError('input feature count differs from fit')
        n = model.n_samples_fit_
        k = model.n_neighbors if n_neighbors is None else n_neighbors
        if type(k) is not int or not 1 <= k <= n:
            raise ValueError('n_neighbors must be an integer in [1, reference rows]')
        if n > 0x7fffffff:
            raise ValueError('reference row count exceeds the native signed-int32 shape contract')
        cell_limit = 0x7fffffff // model.n_features_in_
        if cell_limit < 1:
            raise ValueError('feature count exceeds the native signed-int32 shape contract')
        width = self.reference_rows_per_shard
        if width is None:
            width = min(cell_limit, (n + len(self._pool.devices) - 1) // len(self._pool.devices))
        if min(width, n) > cell_limit:
            raise ValueError('reference shard exceeds the native signed-int32 cell-count contract')
        ranges = [(i, min(n, i + width)) for i in range(0, n, width)]
        params = dict(n_neighbors=k, query_tile=model.query_tile, metric=model.metric,
                      algorithm='brute', p=model.p, numeric_mode='identical')
        outputs, diagnostics = [], []
        vote_model = None
        if method != 'kneighbors':
            vote_model = copy.copy(model)
            vote_model._index = None  # Never send the complete reference matrix.
            vote_model.numeric_mode = 'identical'
        for first in range(0, data.shape[0], self.query_rows_per_shard):
            end = min(data.shape[0], first + self.query_rows_per_shard)
            query = data[first:end]
            parts = []
            # Materialize only one device wave of reference slices on the host.
            # Retain just its small candidate output before staging the next.
            for wave in range(0, len(ranges), len(self._pool.devices)):
                # The READ is shifted by `driver_read_shift` (0 unless the
                # driver sabotage switch is on, and 0 for the first reference
                # range either way); the merge below still folds by the true
                # `ranges`, and `stop-start` keeps every shard's width.
                parts.extend(self._pool.map([
                    ('neighbor_reference', params,
                     (model._index[start - driver_read_shift(index, start):
                                   stop - driver_read_shift(index, start)],
                      query, min(k, stop-start)))
                    for index, (start, stop) in enumerate(ranges[wave:wave+len(self._pool.devices)],
                                                          start=wave)]))
            distances, indices = _merge(parts, ranges, end-first, k)
            if method == 'kneighbors':
                indices = indices.astype('<i8')
                output = (distances, indices) if return_distance else indices
            else:
                output = self._pool.map([('neighbor_vote', vote_model, (distances, indices, method))])[0]
            outputs.append(output)
            diagnostics.extend(dict(query_start=first, query_end=end, reference_start=start,
                reference_end=stop, reference_bytes=(stop-start)*model.n_features_in_*4,
                device=self._pool.devices[i % len(self._pool.devices)])
                for i, (start, stop) in enumerate(ranges))
        result = _join(outputs)
        self.last_shards_ = diagnostics
        return result

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        return self._query(X, method='kneighbors', n_neighbors=n_neighbors, return_distance=return_distance)

    def predict(self, X):
        return self._query(X, method='predict')

    def predict_proba(self, X):
        return self._query(X, method='predict_proba')

    def close(self):
        self._pool.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
