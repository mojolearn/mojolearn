# SPDX-License-Identifier: Apache-2.0
"""Scheduler contracts with isolated fake fits; not physical GPU qualification."""
import ctypes
import os
import pickle

import pytest

from mojolearn import Array, _backend, _buffer
from mojolearn import model_selection as serial
from mojolearn import parallel_model_selection as parallel
from mojolearn._cpu_reference import reference_training
from mojolearn._parallel_pool import DevicePool, _cpu_refusal
from mojolearn._parallel_worker import execute


class Estimator:
    _estimator_type = 'regressor'

    def __init__(self, settings=None, numeric_mode='identical'):
        self.settings, self.numeric_mode = settings, numeric_mode

    def get_params(self, deep=True):
        return dict(settings=self.settings, numeric_mode=self.numeric_mode)

    def fit(self, X, y):
        assert not hasattr(self, 'fitted_'), 'a fitted estimator crossed a fold boundary'
        if self.settings is not None:
            assert self.settings == {'values': [7]}
            self.settings['values'].append(8)
        self.fitted_ = True
        self.train_sum_ = sum(row[0] for row in X.tolist()) + sum(y.tolist())
        return self

    def score(self, X, y):
        return self.train_sum_ + sum(row[0] * 3 for row in X.tolist()) + sum(y.tolist())


def negative_score(estimator, X, y):
    return -estimator.score(X, y)


def fail_score(estimator, X, y):
    raise ValueError('score failure')


def nonscalar_score(estimator, X, y):
    return [1.0]


@pytest.fixture
def setup(monkeypatch):
    def gather(src, dst, indices, source_rows, output_rows, row_bytes):
        for output, source in enumerate((ctypes.c_int64 * output_rows).from_address(indices)):
            ctypes.memmove(dst + output * row_bytes, src + source * row_bytes, row_bytes)
    monkeypatch.setitem(_buffer._NATIVE, 'gather_rows_bytes', gather)
    monkeypatch.setattr(_backend, 'vendor', lambda: 'cuda')
    # THE PRETENDED INSTALL MUST BE CONSISTENT WITH THE PRETENDED VENDOR
    # (lane/cpu-routes-gpu-only-four, 2026-09-20). This fixture emulates a
    # CUDA box; a real one has `_CPU_ONLY is None`, and leaving the host
    # box's real value in place made the worker's new `require_training`
    # refuse every fold of a test that claims to be on a GPU. Saying 'cuda'
    # and 'CPU-only install' at once is not a state that exists.
    monkeypatch.setattr(_backend, '_CPU_ONLY', None)
    instances = []

    class Pool(DevicePool):
        def __init__(self, devices):
            super().__init__(devices)
            self.widths, self.assignments, self.closed = [], [], False
            instances.append(self)

        def map(self, requests):
            if all(request[0] == 'device_inventory' for request in requests):
                return [dict(kind='visible-device-inventory', vendor='cuda', pid=100 + device,
                             devices=[dict(ordinal=0, uuid=f'{device + 1:032x}',
                                           pci_bus_id=f'0000:{device + 1:02x}:00.0')])
                        for device in self.devices]
            # Pickle boundaries prevent a fake in-process worker from hiding
            # accidental fitted-state or mutable-parameter sharing.
            self.widths.append(len(requests))
            results = [None] * len(requests)
            for i in reversed(range(len(requests))):
                request = pickle.loads(pickle.dumps(requests[i], protocol=5))
                self.assignments.append((self.devices[i], request[2][2].tolist()))
                results[i] = execute(request)
            return results

        def close(self):
            self.closed = True
            super().close()

    monkeypatch.setattr(parallel, 'DevicePool', Pool)
    X = Array.from_list([[float(i), float(100 + i)] for i in range(11)], '<f4')
    y = Array.from_list([float(i * 2) for i in range(11)], '<f4')
    return X, y, instances


@pytest.mark.parametrize('devices', [(0,), (0, 1), (3, 1, 0)])
@pytest.mark.parametrize('scoring', [None, negative_score])
def test_scores_fold_order_and_fresh_clones_match_serial(setup, devices, scoring):
    X, y, pools = setup
    original = Estimator({'values': [7]})
    original.fitted_ = True  # Caller may supply an already-fitted estimator.
    expected = serial.cross_val_score(original, X, y, cv=5, scoring=scoring)
    actual = parallel.cross_val_score(original, X, y, devices=devices, cv=5, scoring=scoring)
    assert actual.tobytes() == expected.tobytes()
    assert original.settings == {'values': [7]}
    assert original.fitted_ is True
    pool = pools[-1]
    assert pool.closed
    width = len(devices)
    assert pool.widths == [min(width, 5 - start) for start in range(0, 5, width)]
    assert {device for device, _ in pool.assignments} == set(devices)


def test_validate_all_folds_before_any_worker_fit(setup):
    X, y, pools = setup
    with pytest.raises(ValueError, match='overlap'):
        parallel.cross_val_score(Estimator(), X, y, devices=(0, 1),
                                 cv=[([0, 1], [2]), ([0, 1], [1])])
    assert not pools[-1].widths


@pytest.mark.parametrize('scoring,match', [(fail_score, 'score failure'),
                                         (nonscalar_score, 'real scalar')])
def test_score_failure_closes_pool_without_partial_result(setup, scoring, match):
    X, y, pools = setup
    with pytest.raises((ValueError, TypeError), match=match):
        parallel.cross_val_score(Estimator(), X, y, devices=(0, 1), cv=3, scoring=scoring)
    assert pools[-1].closed
    assert pools[-1].widths == [2]


def test_unpickleable_scorer_fails_before_launch(setup):
    X, y, pools = setup
    with pytest.raises(TypeError, match='pickleable'):
        parallel.cross_val_score(Estimator(), X, y, devices=(0, 1), scoring=lambda *args: 1.)
    assert not pools[-1].widths


@pytest.mark.parametrize('vendor', ['cpu', 'metal'])
def test_no_implicit_cpu_or_metal_pool(setup, monkeypatch, vendor):
    """A GPU INSTALL whose vendor is not CUDA or HIP still refuses.

    `_CPU_ONLY is None` here (the fixture's emulated GPU box), so this is the
    Metal case and the case of a GPU install that reports `cpu` -- not the
    CPU-only host route, which `test_cpu_only_install_takes_the_host_route`
    below covers separately. The two must not be confused: one is "the wrong
    vendor", the other is "no vendor at all, and one worker is one process"."""
    X, y, pools = setup
    monkeypatch.setattr(_backend, 'vendor', lambda: vendor)
    with pytest.raises(NotImplementedError, match='CUDA or HIP'):
        parallel.cross_val_score(Estimator(), X, y, devices=(0, 1))
    assert not pools[-1].widths
    if vendor == 'cpu':
        with pytest.raises(NotImplementedError, match='CUDA, HIP or Metal'):
            execute(('cross_val_fold', Estimator(), (X, y, X, y, None)))


def test_single_metal_worker_matches_serial_without_device_isolation_claim(setup, monkeypatch):
    X, y, pools = setup
    monkeypatch.setattr(_backend, 'vendor', lambda: 'metal')
    original = parallel.DevicePool
    asked = []

    class MetalPool(original):
        def map(self, requests):
            asked.append(requests[0][0])
            if requests[0][0] == 'metal_worker_identity':
                return [dict(kind='single-metal-worker', vendor='metal',
                             pid=1000, ppid=os.getpid())]
            return super().map(requests)

    monkeypatch.setattr(parallel, 'DevicePool', MetalPool)
    expected = serial.cross_val_score(Estimator({'values': [7]}), X, y, cv=5)
    actual = parallel.cross_val_score(Estimator({'values': [7]}), X, y, devices=(0,), cv=5)
    assert actual.tobytes() == expected.tobytes()
    assert asked[0] == 'metal_worker_identity'
    assert 'device_inventory' not in asked and 'worker_identity' not in asked
    assert pools[-1].widths == [1] * 5
    assert pools[-1].closed


@pytest.mark.parametrize('devices', [(1,), (0, 1), (1, 0)])
def test_metal_requires_exactly_device_zero_before_launch(setup, monkeypatch, devices):
    X, y, pools = setup
    monkeypatch.setattr(_backend, 'vendor', lambda: 'metal')
    with pytest.raises(NotImplementedError, match=r'Metal devices=\(0,\)'):
        parallel.cross_val_score(Estimator(), X, y, devices=devices)
    assert not pools[-1].widths


@pytest.mark.parametrize('record', [
    dict(kind='worker-process-identity', vendor='cpu', pid=1000),
    dict(kind='single-metal-worker', vendor='cuda', pid=1000),
    dict(kind='single-metal-worker', vendor='metal', pid=True),
])
def test_metal_worker_identity_rejects_wrong_route(record):
    with pytest.raises(RuntimeError, match='identity is invalid'):
        parallel._require_single_metal_worker([dict(record, ppid=os.getpid())])


def test_metal_worker_identity_checks_parent_process():
    with pytest.raises(RuntimeError, match='identity is invalid'):
        parallel._require_single_metal_worker([
            dict(kind='single-metal-worker', vendor='metal', pid=1000, ppid=-1)])


def test_cpu_only_install_takes_the_host_route_with_a_process_witness(setup, monkeypatch):
    """The CPU-only install DOES run the driver, with the process witness.

    lane/cpu-routes-gpu-only-four (2026-09-20). `cross_val_fold` and
    `worker_identity` joined `_parallel_pool.CPU_OPERATIONS`, so the refusal
    is gone; what replaces it is `require_distinct_processes` over one
    `worker-process-identity` record per index. The fixture's fake pool
    answers `device_inventory`, so this test makes it answer the host
    operation instead and checks the driver ASKED for the host one, which is
    the whole difference between the two routes at this level."""
    X, y, pools = setup
    monkeypatch.setattr(_backend, 'vendor', lambda: 'cpu')
    monkeypatch.setattr(_backend, '_CPU_ONLY', 'no identical binding on this box')
    asked = []
    original = parallel.DevicePool

    class HostPool(original):
        def map(self, requests):
            asked.append(requests[0][0])
            if requests[0][0] == 'worker_identity':
                return [dict(kind='worker-process-identity', vendor='cpu',
                             pid=1000 + device, ppid=os.getpid())
                        for device in self.devices]
            return super().map(requests)

    monkeypatch.setattr(parallel, 'DevicePool', HostPool)
    # INSIDE `reference_training()`, which is the only scope a CPU fold fits
    # in. The fixture's pool runs `execute` in this process rather than
    # through `DevicePool.map`'s `cpu_reference` wrapper, so the scope is
    # opened here; `test_the_cpu_fold_guard_is_load_bearing` below is the
    # arm that shows the guard is what closes it.
    with reference_training():
        scores = parallel.cross_val_score(Estimator(), X, y, devices=(0, 1), cv=3)
    assert asked[0] == 'worker_identity'
    assert 'device_inventory' not in asked
    expected = serial.cross_val_score(Estimator(), X, y, cv=3)
    assert scores.tobytes() == expected.tobytes()
    assert pools[-1].closed


def test_the_cpu_route_is_still_gated_by_cpu_operations():
    """`cross_val_fold` is admitted; a neighbouring operation is not.

    A one-sided assertion here would read the same whether the frozenset had
    two new names or every name."""
    assert _cpu_refusal([('cross_val_fold', None, ())], False) is None
    assert _cpu_refusal([('worker_identity', None, ())], False) is None
    assert _cpu_refusal([('gbdt_fit', None, ())], False) is not None
    assert _cpu_refusal([('cross_val_fold', None, ())], True) is not None


@pytest.mark.parametrize('devices', [(), (0, 0), (-1,), (True,), ('0',)])
def test_invalid_device_lists_refused(setup, devices):
    X, y, _ = setup
    with pytest.raises(ValueError, match='distinct nonnegative'):
        parallel.cross_val_score(Estimator(), X, y, devices=devices)


def test_fast_mode_refused_before_workers(setup):
    X, y, pools = setup
    with pytest.raises(ValueError, match='IDENTICAL'):
        parallel.cross_val_score(Estimator(numeric_mode='fast'), X, y, devices=(0, 1))
    assert not pools[-1].widths


def test_custom_splitter_groups_and_string_labels(setup):
    X, _, pools = setup
    y = ['one', 'two'] * 5 + ['one']

    class Splitter:
        def split(self, X, labels, groups):
            assert labels == y and groups == list(range(11))
            yield [0, 2, 4], [1, 3]
            yield [1, 3], [0, 2, 4]
    # Prepare once in the parent; splitters need not be pickleable.
    _, labels, folds = serial._prepare_folds(Estimator(), X, y, Splitter(), None,
                                            list(range(11)), 'raise')
    assert labels == y and len(folds) == 2
    assert serial._take_rows(labels, folds[0][0]) == ['one'] * 3


class FailingEstimator(Estimator):
    def fit(self, X, y):
        raise ValueError('fit failure')


def test_fit_failure_closes_pool(setup):
    X, y, pools = setup
    with pytest.raises(RuntimeError, match='GPU worker failed'):
        # Exercise the parent's handling of the real RPC error wrapper.
        def failed_map(requests):
            try:
                execute(requests[0])
            except ValueError as exc:
                raise RuntimeError('GPU worker failed: ' + str(exc)) from exc
        class FailingPool(parallel.DevicePool):
            def map(self, requests):
                if requests[0][0] == 'device_inventory':
                    return super().map(requests)
                return failed_map(requests)
        original = parallel.DevicePool
        parallel.DevicePool = FailingPool
        try:
            parallel.cross_val_score(FailingEstimator(), X, y, devices=(0, 1), cv=3)
        finally:
            parallel.DevicePool = original
    assert pools[-1].closed




def test_physical_alias_refused_before_fits_and_pool_closed(setup, monkeypatch):
    X, y, pools = setup
    class AliasedPool(parallel.DevicePool):
        def map(self, requests):
            assert all(request[0] == 'device_inventory' for request in requests)
            records = super().map(requests)
            records[1]['devices'] = records[0]['devices']
            return records
    monkeypatch.setattr(parallel, 'DevicePool', AliasedPool)
    with pytest.raises(RuntimeError, match='repeated physical device'):
        parallel.cross_val_score(Estimator(), X, y, devices=(0, 1))
    assert pools[-1].closed and not pools[-1].widths


@pytest.mark.parametrize('extra', [False, True])
def test_incomplete_or_excess_fold_batch_closes_pool(setup, monkeypatch, extra):
    X, y, instances = setup
    original = parallel.DevicePool
    class CorruptPool(original):
        def map(self, requests):
            result = super().map(requests)
            if requests[0][0] == 'cross_val_fold':
                return result + [1.0] if extra else result[:-1]
            return result
    monkeypatch.setattr(parallel, 'DevicePool', CorruptPool)
    with pytest.raises(ValueError, match='incomplete fold batch'):
        parallel.cross_val_score(Estimator(), X, y, devices=(0, 1), cv=5)
    assert instances[-1].closed
