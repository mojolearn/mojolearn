# SPDX-License-Identifier: Apache-2.0
"""IVF storage ownership and squared-key composition tests."""
import numpy as np
import pytest
from mojolearn import parallel_ivf as pi
from mojolearn._ivf_impl import IVFIndex
from mojolearn._buffer import Array


def ar(x, dtype='<f4'):
    return Array.from_list(np.asarray(x).tolist(), dtype)


class StoragePool:
    closed = False
    shards = []
    fail = False
    native = False
    def __init__(self, devices):
        if not devices or len(set(devices)) != len(devices):
            raise ValueError('invalid devices')
        self.shards = []
        type(self).closed = False
    def close(self):
        type(self).closed = True
    def map(self, requests):
        if self.fail:
            raise RuntimeError('worker failed')
        if requests[0][0] == 'ivf_store':
            self.shards = [state for _, state, _ in requests]
            type(self).shards = self.shards
            return [state.n_rows_ for state in self.shards]
        if requests[0][0] == 'ivf_finalize':
            from mojolearn._buffer import addr
            state = requests[0][1]
            if self.native:
                self.shards[0]._extension().ivf_finalize_distances(addr(state, name='d'), state.size, 1)
                return [state]
            return [ar(np.sqrt(np.asarray(state)))]
        results = []
        for shard, (_, _, args) in zip(self.shards, requests):
            if self.native:
                results.append(pi._partial_search(shard, args[0]))
                continue
            data, queries = np.asarray(shard.list_data_), np.asarray(args[0])
            rows, ids, counts = [], [], []
            for query in queries:
                # All probes test oracle; native tests cover actual coarse selection.
                found = sorted((float(np.sum((point - query)**2)), int(i))
                               for point, i in zip(data, shard.list_indices_.tolist()))
                valid = found[:shard.n_neighbors]
                counts.append(len(found))
                valid += [(0.0, 0)] * (shard.n_neighbors - len(valid))
                rows.append([v for v, _ in valid])
                ids.append([i for _, i in valid])
            results.append((ar(rows), ar(ids, '<i4'), ar(counts, '<i4')))
        return results


@pytest.fixture
def pool(monkeypatch):
    StoragePool.fail, StoragePool.native = False, False
    monkeypatch.setattr(pi, 'DevicePool', StoragePool)
    monkeypatch.setattr('mojolearn._backend.default_mode', lambda: 'identical')


def index(metric='sqeuclidean'):
    m = IVFIndex(3, 3, 4, metric=metric)
    m.n_rows_, m.n_lists_, m.n_features_in_ = 7, 3, 2
    m.metric_code_ = 0 if metric == 'sqeuclidean' else 1
    m.centers_, m.center_norms_ = ar([[0, 0], [1, 1], [4, 4]]), ar([0, 2, 32])
    m.list_offsets_ = ar([0, 3, 3, 7], '<i4')
    m.list_indices_ = ar([1, 4, 6, 0, 2, 3, 5], '<i4')
    m.list_data_ = ar([[0, 0], [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    return m


@pytest.mark.parametrize('devices', [(0,), (0, 1), (2, 1, 0), (0, 1, 2, 3)])
@pytest.mark.parametrize('metric', ['sqeuclidean', 'euclidean'])
def test_disjoint_storage_and_global_ties(pool, devices, metric):
    original = index(metric)
    before = original.list_data_.tobytes()
    q = ar([[0, 0], [3, 3]])
    with pi.DistributedIVFIndex.from_index(original, devices=devices) as model:
        d, ids = model.search(q)
        assert ids.tolist() == [[1, 4, 6, 0], [2, 0, 3, 5]]
        assert sum(s.n_rows_ for s in StoragePool.shards) == 7
        assert len(StoragePool.shards) == len(devices)
        assert model.n_candidates_.tolist() == [7, 7]
        assert original.list_data_.tobytes() == before
        if len(devices) > 1:
            assert all(s.n_rows_ < original.n_rows_ for s in StoragePool.shards)
        reference = np.asarray([[0, 0, 2, 8], [0, 2, 2, 8]], dtype=np.float32)
        if metric == 'euclidean':
            reference = np.sqrt(reference)
        assert np.asarray(d).tobytes() == reference.tobytes()
    assert StoragePool.closed
    with pytest.raises(RuntimeError, match='closed'):
        model.search(q)


def test_failure_closes_stored_workers(pool):
    model = pi.DistributedIVFIndex.from_index(index())
    StoragePool.fail = True
    with pytest.raises(RuntimeError, match='worker failed'):
        model.search(ar([[0, 0]]))
    assert model._closed and StoragePool.closed


@pytest.mark.parametrize('devices', [(0,), (0, 1), (3, 2, 1, 0)])
@pytest.mark.parametrize('metric', ['sqeuclidean', 'euclidean'])
@pytest.mark.parametrize('probes', [1, 3])
def test_native_partial_storage_matches_full_index(pool, monkeypatch, devices, metric, probes):
    import os
    import importlib.util
    path = os.environ.get('MOJOLEARN_NATIVE_IVF_PARTIAL')
    if not path:
        pytest.skip('requires freshly built partial IVF binding under native slot')
    spec = importlib.util.spec_from_file_location('_mojolearn_ivf', path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    monkeypatch.setattr(IVFIndex, '_extension', lambda self: native)
    StoragePool.native = True
    model = index(metric)
    model.n_probes, model.n_neighbors = probes, 2
    queries = ar([[0, 0], [3, 3], [0, 0]])
    expected = model.search(queries)
    expected_count = model.n_candidates_.tobytes()
    with pi.DistributedIVFIndex.from_index(model, devices=devices) as distributed:
        actual = distributed.search(queries)
        for a, b in zip(actual, expected):
            assert a.tobytes() == b.tobytes()
        assert distributed.n_candidates_.tobytes() == expected_count
        repeat = distributed.search(queries)
        assert repeat[0].tobytes() == actual[0].tobytes()


@pytest.mark.parametrize('rows,features', [(33, 3), (65, 17)])
def test_native_fitted_index_uneven_tiles(pool, monkeypatch, rows, features):
    import os
    import importlib.util
    from mojolearn._cpu_reference import reference_training
    path = os.environ.get('MOJOLEARN_NATIVE_IVF_PARTIAL')
    if not path:
        pytest.skip('requires freshly built partial IVF binding under native slot')
    spec = importlib.util.spec_from_file_location('_mojolearn_ivf', path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    monkeypatch.setattr(IVFIndex, '_extension', lambda self: native)
    StoragePool.native = True
    x = np.random.default_rng(97).normal(size=(rows, features)).astype(np.float32)
    x[1] = x[0]
    with reference_training():
        model = IVFIndex(4, 4, 5, kmeans_n_iters=3, numeric_mode='identical').fit(x)
    expected = model.search(x[:7])
    for devices in [(0, 1), (2, 1, 0)]:
        with pi.DistributedIVFIndex.from_index(model, devices=devices) as distributed:
            actual = distributed.search(x[:7])
            for a, b in zip(actual, expected):
                assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize('fault', ['missing', 'shape', 'negative-count', 'large-count', 'invalid-id'])
def test_invalid_shard_receipts_fail_closed(pool, monkeypatch, fault):
    original_map = StoragePool.map
    def corrupt(self, requests):
        result = original_map(self, requests)
        if requests[0][0] != 'ivf_search_stored':
            return result
        if fault == 'missing':
            return result[:-1]
        d, ix, count = result[0]
        if fault == 'shape':
            d = ar([[1]])
        elif fault == 'negative-count':
            count = ar([-1], '<i4')
        elif fault == 'large-count':
            count = ar([100], '<i4')
        else:
            ix = ar([[100, 0, 0, 0]], '<i4')
        result[0] = (d, ix, count)
        return result
    monkeypatch.setattr(StoragePool, 'map', corrupt)
    model = pi.DistributedIVFIndex.from_index(index())
    with pytest.raises(ValueError):
        model.search(ar([[0, 0]]))
    assert model._closed and StoragePool.closed


def test_duplicate_global_ids_rejected_before_partition(pool):
    m = index()
    m.list_indices_ = ar([1, 4, 6, 0, 2, 3, 1], '<i4')
    with pytest.raises(ValueError, match='permutation'):
        pi.DistributedIVFIndex.from_index(m)


def test_store_failure_closes_workers(pool):
    StoragePool.fail = True
    with pytest.raises(RuntimeError, match='worker failed'):
        pi.DistributedIVFIndex.from_index(index())
    assert StoragePool.closed


@pytest.mark.parametrize('offsets', [[-1, 3, 3, 7], [0, 3, 3, 8], [0, 4, 3, 7], [0, 3, 7]])
def test_corrupt_global_offsets_are_not_repaired_by_partition_clipping(pool, monkeypatch, offsets):
    def unexpected(*args):
        raise AssertionError('invalid layout started GPU workers')
    monkeypatch.setattr(pi, 'DevicePool', unexpected)
    m = index()
    m.list_offsets_ = ar(offsets, '<i4')
    with pytest.raises(ValueError, match='global IVF list offsets'):
        pi.DistributedIVFIndex.from_index(m)
