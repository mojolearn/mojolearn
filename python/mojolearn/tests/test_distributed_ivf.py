# SPDX-License-Identifier: Apache-2.0
"""DistributedIVFIndex against the plain index.

`from_index` and `search` are one native call each
(bindings/parallel_shard_native.mojo `ivf_from_index` / `ivf_search`): the
native shard plan gives every device its contiguous stored rows, each worker
process stores its shard, and every search merges the shards' partial
results natively. On a CPU-only install each device index is one worker
process, so the multi-device cases run there; on a GPU install they need as
many GPUs.
"""
import numpy as np
import pytest
from mojolearn import _backend
from mojolearn import parallel_ivf as pi
from mojolearn._ivf_impl import IVFIndex
from mojolearn._buffer import Array
from mojolearn._native_parallel_shard import _native_parallel_shard


def ar(x, dtype='<f4'):
    return Array.from_list(np.asarray(x).tolist(), dtype)


@pytest.fixture
def identical(monkeypatch):
    monkeypatch.setattr('mojolearn._backend.default_mode', lambda: 'identical')


def fitted(metric, rows=33, features=3):
    from mojolearn._cpu_reference import reference_training
    x = np.random.default_rng(rows + features).normal(size=(rows, features)).astype(np.float32)
    x[1] = x[0]
    with reference_training():
        return IVFIndex(4, 4, 5, kmeans_n_iters=3, metric=metric, numeric_mode='identical').fit(x), x


def _devices(count):
    if count > 1 and _backend._CPU_ONLY is None:
        pytest.skip('a multi-device GPU column runs on a multi-GPU box (verify --par)')
    return tuple(range(count))


@pytest.mark.parametrize('count', [1, 2, 3])
@pytest.mark.parametrize('metric', ['sqeuclidean', 'euclidean'])
def test_disjoint_storage_matches_the_plain_index(identical, count, metric):
    devices = _devices(count)
    index, x = fitted(metric)
    before = index.list_data_.tobytes()
    expected = index.search(x[:7])
    _native_parallel_shard('trace_take', ())
    with pi.DistributedIVFIndex.from_index(index, devices=devices) as model:
        got = model.search(x[:7])
        assert [np.asarray(a).tobytes() for a in got] == [np.asarray(a).tobytes() for a in expected]
        assert index.list_data_.tobytes() == before
        trace = _native_parallel_shard('trace_take', ())
        assert len(trace) == 1 and trace[0]['devices'] == list(devices)
        calls = [c['operation'] for c in trace[0]['calls']]
        assert calls == ['ivf_store'] * len(devices) + ['ivf_search'] * len(devices)
    with pytest.raises(RuntimeError, match='closed'):
        model.search(x[:2])


def test_failure_closes_stored_workers(identical, monkeypatch):
    index, x = fitted('sqeuclidean')
    devices = _devices(2)
    model = pi.DistributedIVFIndex.from_index(index, devices=devices)
    monkeypatch.setenv('MOJOLEARN_PAR_TRANSPORT_FAULT', 'drop_result')
    monkeypatch.setenv('MOJOLEARN_HOST_ALLOW_SABOTAGE', '1')
    _native_parallel_shard('faults_take', ())
    with pytest.raises(ValueError, match='incomplete'):
        model.search(x[:3])
    assert model._closed


def synthetic():
    m = IVFIndex(3, 3, 4, metric='sqeuclidean')
    m.n_rows_, m.n_lists_, m.n_features_in_ = 7, 3, 2
    m.metric_code_ = 0
    m.centers_, m.center_norms_ = ar([[0, 0], [1, 1], [4, 4]]), ar([0, 2, 32])
    m.list_offsets_ = ar([0, 3, 3, 7], '<i4')
    m.list_indices_ = ar([1, 4, 6, 0, 2, 3, 5], '<i4')
    m.list_data_ = ar([[0, 0], [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    return m


def test_duplicate_global_ids_rejected_before_partition(identical):
    m = synthetic()
    m.list_indices_ = ar([1, 4, 6, 0, 2, 3, 1], '<i4')
    _native_parallel_shard('trace_take', ())
    with pytest.raises(ValueError, match='permutation'):
        pi.DistributedIVFIndex.from_index(m, devices=(0,))
    assert _native_parallel_shard('trace_take', ()) == [], 'a worker started before the refusal'


@pytest.mark.parametrize('offsets', [[-1, 3, 3, 7], [0, 3, 3, 8], [0, 4, 3, 7], [0, 3, 7]])
def test_corrupt_global_offsets_are_not_repaired_by_partition_clipping(identical, offsets):
    m = synthetic()
    m.list_offsets_ = ar(offsets, '<i4')
    _native_parallel_shard('trace_take', ())
    with pytest.raises(ValueError, match='global IVF list offsets'):
        pi.DistributedIVFIndex.from_index(m, devices=(0,))
    assert _native_parallel_shard('trace_take', ()) == [], 'an invalid layout started workers'
