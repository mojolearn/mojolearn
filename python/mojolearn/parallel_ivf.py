# SPDX-License-Identifier: Apache-2.0
"""Disjoint IVF candidate storage with replicated coarse centers.

A fitted/loaded index is partitioned without retraining its quantizer. Worker
processes retain only their candidate rows; each search uploads only that shard
to its GPU. Coarse centers and queries are replicated. Building the original
index and extending a distributed index are outside this API.
"""
__all__ = ['DistributedIVFIndex']

from ._parallel_pool import DevicePool
from ._buffer import Array, as_f32_c, empty, addr, addr_ro
from ._ivf_impl import IVFIndex


def _partial_search(index, queries):
    q, _ = as_f32_c(queries, ndim=2, name='queries')
    if q.shape[1] != index.n_features_in_:
        raise ValueError('query features differ from index')
    m, k = q.shape[0], index.n_neighbors
    dist, ids, counts = empty((m, k), '<f4'), empty((m, k), '<i4'), empty((m,), '<i4')
    native = index._extension()
    fn = index._entry(native, 'ivf_flat_partial_search')
    fn([addr_ro(index.centers_, name='centers'), addr_ro(index.center_norms_, name='norms'),
        addr_ro(index.list_offsets_, name='offsets'), addr_ro(index.list_indices_, name='ids'),
        addr_ro(index.list_data_, name='data'), addr_ro(q, name='queries'),
        addr(dist, name='distances'), addr(ids, name='ids'), addr(counts, name='counts')],
       [index.n_rows_, index.n_features_in_, index.n_lists_, index.metric_code_, m, k, index.n_probes])
    return dist, ids, counts


class DistributedIVFIndex:
    """Context-managed search over disjoint stored candidate shards.

    ``from_index(index, devices=(0, 1))`` preserves the global coarse quantizer
    and its selected probes. Local IDs are monotonic maps of original IDs, so
    exact-distance ties retain original row order. Final selection uses squared
    distances; Euclidean rooting occurs only after the global order is fixed.
    Individual shards and all coarse centers must each fit on one GPU.
    """
    def __init__(self):
        raise TypeError('use DistributedIVFIndex.from_index(index, devices=...)')

    @classmethod
    def from_index(cls, index, *, devices=(0, 1)):
        from . import _backend
        if type(index) is not IVFIndex:
            raise TypeError('requires mojolearn.IVFIndex')
        if not hasattr(index, 'list_data_'):
            raise ValueError('fit or load the index first')
        if (getattr(index, 'numeric_mode', None) or _backend.default_mode()) != 'identical':
            raise ValueError('distributed IVF requires IDENTICAL numeric mode')
        if index._metric_code() != index.metric_code_:
            raise ValueError('metric differs from the fitted index')
        from ._ivf_impl import _int_param
        for name in ('n_lists', 'n_probes', 'n_neighbors'):
            _int_param(name, getattr(index, name))
        if not 1 <= index.n_probes <= index.n_lists_ or index.n_neighbors < 1:
            raise ValueError('invalid IVF probe or neighbor count')
        original_ids = [int(v) for v in index.list_indices_.tolist()]
        if sorted(original_ids) != list(range(index.n_rows_)):
            raise ValueError('IVF original IDs must be a permutation of stored row IDs')
        if index.list_data_.shape != (index.n_rows_, index.n_features_in_):
            raise ValueError('IVF stored data shape differs from index metadata')
        offsets = [int(v) for v in index.list_offsets_.tolist()]
        # Partitioning clips offsets to local bounds. Validate BEFORE clipping:
        # otherwise a corrupted saved index can be silently repaired differently
        # from the ordinary native search, which rejects its original layout.
        if (len(offsets) != index.n_lists_ + 1 or offsets[0] != 0
                or offsets[-1] != index.n_rows_
                or any(a > b for a, b in zip(offsets, offsets[1:]))):
            raise ValueError('invalid global IVF list offsets')
        devices = tuple(devices)
        pool = DevicePool(devices)  # validates every requested device before slicing
        pool.close()
        devices = devices[:index.n_rows_]
        pool = DevicePool(devices)
        obj = object.__new__(cls)
        obj._pool, obj._closed = pool, False
        obj.n_features_in_, obj.n_neighbors = index.n_features_in_, index.n_neighbors
        obj.metric_code_, obj.devices = index.metric_code_, devices
        obj.n_candidates_ = None
        obj._id_maps = []
        requests = []
        for part in range(len(devices)):
            lo = part * index.n_rows_ // len(devices)
            hi = (part + 1) * index.n_rows_ // len(devices)
            original = original_ids[lo:hi]
            mapping = sorted(original)
            local = {value: i for i, value in enumerate(mapping)}
            shard = IVFIndex(index.n_lists_, index.n_probes, index.n_neighbors,
                             metric=index.metric, numeric_mode='identical')
            shard.n_rows_, shard.n_lists_ = hi - lo, index.n_lists_
            shard.n_features_in_, shard.metric_code_ = index.n_features_in_, index.metric_code_
            shard.centers_, shard.center_norms_ = index.centers_, index.center_norms_
            shard.list_offsets_ = Array.from_list([max(0, min(v, hi) - lo) for v in offsets], '<i4')
            shard.list_indices_ = Array.from_list([local[v] for v in original], '<i4')
            shard.list_data_ = index.list_data_[lo:hi]
            obj._id_maps.append(mapping)
            requests.append(('ivf_store', shard, ()))
        try:
            receipts = pool.map(requests)
            if receipts != [len(ids) for ids in obj._id_maps]:
                raise ValueError('IVF storage receipts differ from shard sizes')
        except BaseException:
            obj.close()
            raise
        return obj

    def search(self, queries):
        if self._closed:
            raise RuntimeError('distributed IVF index is closed')
        q, _ = as_f32_c(queries, ndim=2, name='queries')
        if q.shape[0] < 1:
            raise ValueError('at least one query is required')
        if q.shape[1] != self.n_features_in_:
            raise ValueError('query features differ from index')
        try:
            parts = self._pool.map([('ivf_search_stored', None, (q,)) for _ in self.devices])
            if len(parts) != len(self._id_maps):
                raise ValueError('incomplete IVF result shards')
            distances, ids, counts = [], [], []
            for row in range(q.shape[0]):
                candidates, total = [], 0
                for mapping, (d, ix, count) in zip(self._id_maps, parts):
                    if d.shape != (q.shape[0], self.n_neighbors) or ix.shape != d.shape or count.shape != (q.shape[0],):
                        raise ValueError('invalid IVF shard output shapes')
                    n = int(count[row])
                    if not 0 <= n <= len(mapping):
                        raise ValueError('invalid IVF candidate count')
                    total += n
                    for j in range(min(self.n_neighbors, n)):
                        local = int(ix[row, j])
                        if not 0 <= local < len(mapping):
                            raise ValueError('invalid IVF local row id')
                        candidates.append((float(d[row, j]), mapping[local]))
                if total < self.n_neighbors:
                    raise ValueError('global probed candidate count is smaller than n_neighbors')
                candidates.sort()
                selected = candidates[:self.n_neighbors]
                distances.append([v for v, _ in selected])
                ids.append([i for _, i in selected])
                counts.append(total)
            result = Array.from_list(distances, '<f4')
            if self.metric_code_ == 1:
                result = self._pool.map([('ivf_finalize', result, (self.metric_code_,))])[0]
            self.n_candidates_ = Array.from_list(counts, '<i4')
            return result, Array.from_list(ids, '<i4')
        except BaseException:
            self.close()
            raise

    def close(self):
        self._pool.close()
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError('distributed IVF index is closed')
        return self

    def __exit__(self, *exc):
        self.close()
