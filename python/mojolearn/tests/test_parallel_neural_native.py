"""Whole native parallel training, independent calculus and child audits."""
import json
import os
import sys

import numpy as np
import pytest

import mojolearn
from mojolearn import ParallelNeuralTrainer, _backend
from mojolearn._parallel_mlp import _native_parallel_mlp as mlp
from mojolearn._parallel_neural import _native_parallel_neural as native
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_parallel_mlp_native import model_fixture, independent_gradient


def checked(operation, *args):
    return audit_native_call(native, operation, args).assert_native()


def training_state(checkpoint):
    state = checkpoint['state']
    arrays = [np.asarray(state['weights'][key]).tobytes() for key in state['parameter_order']]
    arrays.extend(np.asarray(state['optimizer'][key]).tobytes() for key in ('m', 'v', 'flags'))
    return (arrays, state['optimizer']['step'], state['config'], state['data_schedule'])


@pytest.fixture(autouse=True)
def identical_mode():
    if _backend.default_mode() != 'identical':
        pytest.skip('existing parallel neural interface requires identical mode')


def audited_child_module(tmp_path, monkeypatch):
    (tmp_path / 'native_neural_audit_child.py').write_text('''
import json, os, sys
from pathlib import Path
directory = Path(os.environ['NATIVE_NEURAL_AUDIT_DIRECTORY'])
sys.stdout = (directory / (str(os.getpid()) + '.stdout.log')).open('w', buffering=1)
sys.stderr = (directory / (str(os.getpid()) + '.stderr.log')).open('w', buffering=1)
from mojolearn import _backend
from mojolearn._parallel_neural import _native_parallel_neural
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn._gpu_witness import visible_gpu_inventory, worker_process_inventory
vendor = _backend.vendor()
# This independent inventory is test work, outside the complete native loop.
inventory = visible_gpu_inventory(vendor) if vendor in ('cuda', 'hip') else worker_process_inventory()
audit = audit_native_call(_native_parallel_neural, 'worker', sys.argv)
report = audit.report()
report['exception'] = None if audit.exception is None else repr(audit.exception)
report['inventory'] = inventory
Path(os.environ['NATIVE_NEURAL_AUDIT_DIRECTORY'], str(os.getpid()) + '.json').write_text(json.dumps(report, indent=2))
audit.assert_native()
''')
    monkeypatch.setenv('NATIVE_NEURAL_AUDIT_DIRECTORY', str(tmp_path))
    package_parent = str(__import__('pathlib').Path(mojolearn.__file__).resolve().parent.parent)
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join((str(tmp_path), package_parent, os.environ.get('PYTHONPATH', ''))))
    return 'native_neural_audit_child'


def test_complete_training_oracle_checkpoint_replay_and_child_audit(tmp_path, monkeypatch):
    child_module = audited_child_module(tmp_path, monkeypatch)
    devices = tuple(map(int, os.environ.get('MOJOLEARN_PARALLEL_NEURAL_DEVICES', '0').split(',')))
    if _backend.vendor() == 'cpu' and len(devices) > 1:
        pytest.skip('existing cooperative CPU update supports one device only')
    logical = max(2, len(devices))
    model = model_fixture()
    initial = audit_native_call(mlp, 'snapshot', (model,)).assert_native()
    shards = []
    expected_losses, expected_grads = [], []
    for i in range(logical):
        x = np.zeros((3, 8), np.float32)
        x[:, 0] = [1. + .125*i, -.25, .5]
        y = np.array([0, 1, 2], np.int64)
        shards.append((x, y))
        loss, grads = independent_gradient(initial, x, y)
        expected_losses.append(loss)
        expected_grads.append(grads)
    summed = [sum(part[j] for part in expected_grads) for j in range(4)]
    owner = object.__new__(ParallelNeuralTrainer)
    checked('init_with_worker', owner, model, devices, logical, child_module)
    replay = None
    pids = []
    gradient_pids = []
    update_pid = None
    try:
        result = audit_native_call(owner.train_step, shards).assert_native()
        np.testing.assert_allclose(result['losses'], expected_losses, rtol=3e-6, atol=3e-7)
        assert result['completed_steps'] == 1
        retained = audit_native_call(owner.export_gradients).assert_native()
        for observed, expected in zip(retained, summed):
            np.testing.assert_allclose(np.asarray(observed), expected, rtol=5e-6, atol=5e-7)
        checkpoint = audit_native_call(owner.checkpoint).assert_native()
        cfg = initial['config']
        for key, gradient in zip(initial['parameter_order'], summed):
            expected = np.asarray(initial['weights'][key], dtype=np.float64) * (1-cfg['lr']*cfg['weight_decay'])
            expected -= cfg['lr'] * gradient / (np.abs(gradient) + cfg['eps'])
            np.testing.assert_allclose(np.asarray(checkpoint['state']['weights'][key]), expected, rtol=6e-6, atol=6e-7)
        replay = audit_native_call(ParallelNeuralTrainer.from_checkpoint, checkpoint, devices=(devices[0],)).assert_native()
        # The portable checkpoint and both restored/live owners have distinct
        # parameter storage; changing the saved bytes cannot alter replay.
        np.asarray(checkpoint['state']['weights']['weight1'])[:] = 99
        audit_native_call(owner.train_step, shards).assert_native()
        audit_native_call(replay.train_step, shards).assert_native()
        assert training_state(audit_native_call(owner.checkpoint).assert_native()) == training_state(audit_native_call(replay.checkpoint).assert_native())
        gradient_pids = [worker['pid'] for worker in owner._pool['workers']]
        update_pid = owner._update_pool['workers'][0]['pid']
        pids = gradient_pids + [update_pid]
        assert len(set(pids)) == len(devices) + 1
    finally:
        primary = sys.exception()
        cleanup_errors = []
        # Capture identities even when training or the independent oracle
        # fails, then give every child time to retain its complete audit.
        workers = owner._pool['workers'] + owner._update_pool['workers']
        pids = sorted(set(pids + [worker['pid'] for worker in workers]))
        for active in (owner, replay):
            if active is not None:
                try:
                    audit_native_call(active.close).assert_native()
                except BaseException as error:
                    cleanup_errors.append(error)
        if primary is not None:
            primary.add_note('Full native child stdout/stderr and audit reports: ' + str(tmp_path))
            for path in sorted(tmp_path.glob('*.stderr.log')):
                # The entire diagnostic stays on disk; retain the final
                # relevant lines on the original failure without masking it.
                primary.add_note(str(path) + '\n' + '\n'.join(path.read_text().splitlines()[-12:]))
            for error in cleanup_errors:
                primary.add_note('Native child cleanup also failed: ' + repr(error))
        elif cleanup_errors:
            raise ExceptionGroup('native child cleanup failures; evidence: ' + str(tmp_path), cleanup_errors)
    reports = {}
    for pid in pids:
        path = tmp_path / (str(pid) + '.json')
        assert path.exists(), 'complete native training worker audit missing: ' + str(path)
        report = json.loads(path.read_text())
        assert report['exception'] is None, report
        for key in ('python_frames', 'forbidden_native_calls', 'wrapper_work', 'unobserved_existing_threads'):
            assert not report[key], report
        reports[pid] = report
    if _backend.vendor() in ('cuda', 'hip'):
        from mojolearn._gpu_witness import require_distinct_workers
        inventories = [reports[pid]['inventory'] for pid in gradient_pids]
        require_distinct_workers(inventories, _backend.vendor(), len(devices))
        expected_uuids = {row['devices'][0]['uuid'] for row in inventories}
        actual_uuids = {row['uuid'] for row in reports[update_pid]['inventory']['devices']}
        assert actual_uuids == expected_uuids


def test_worker_failure_rolls_back_and_retry_restarts_owned_processes():
    model = model_fixture()
    np.asarray(model._opt.params[0])[0, 0] = 2
    owner = object.__new__(ParallelNeuralTrainer)
    audit_native_call(owner.__init__, model, devices=(0,), logical_shards=1).assert_native()
    x = np.zeros((1, 8), np.float32)
    x[0, 0] = np.finfo(np.float32).max
    y = np.array([0], np.int32)
    try:
        before = audit_native_call(owner.checkpoint).assert_native()
        failed = audit_native_call(owner.train_step, [(x, y)])
        assert isinstance(failed.exception, RuntimeError), failed.report()
        for key in ('python_frames', 'forbidden_native_calls', 'wrapper_work', 'unobserved_existing_threads'):
            assert not failed.report()[key], failed.report()
        assert training_state(audit_native_call(owner.checkpoint).assert_native()) == training_state(before)
        assert owner._pool['workers'] == []
        assert owner._pool['closed'] is False
        x[0, 0] = 1
        assert audit_native_call(owner.train_step, [(x, y)]).assert_native()['completed_steps'] == 1
    finally:
        audit_native_call(owner.close).assert_native()


def test_samba_parallel_dropout_clipping_schedule_and_checkpoint_replay(tmp_path, monkeypatch):
    from mojolearn import _training_impl as training
    from mojolearn.tests.test_samba_gradients_native import _model, _reference
    from mojolearn.tests.test_samba_state_native import _assert_state_equal

    model = _model(['attention', 'mamba3'], True)
    model.generator.counter = 9
    model.max_norm = .01
    model.optimizer.lr_schedule = training.WarmupCosineLR(.003, 3, 17)
    initial = model.state_dict()
    shards = [(np.array([[1, 2, 3]], np.int64), np.array([[2, -100, 4]], np.int64)),
              (np.array([[4, 5, 6]], np.int16), np.array([[5, 6, 7]], np.int16))]
    reference = []
    offset = 0
    for inputs, targets in shards:
        reference.append(_reference(model, inputs.astype(np.int32), targets.astype(np.int32),
                                    None, 9, offset))
        offset += inputs.size
    # Independent per-shard model chain plus explicit left-associated float32
    # addition. The coordinator/worker/fold implementation is not the oracle.
    expected = [np.frombuffer(a, np.float32) + np.frombuffer(b, np.float32)
                for a, b in zip(reference[0][1], reference[1][1])]
    gradient = np.concatenate(expected).astype(np.float64)
    norm = np.linalg.norm(gradient)
    assert norm > model.max_norm
    clipped = gradient * (model.max_norm / (norm + 1e-6))
    config = initial['optimizer']
    rate = float(np.float32(np.float32(.003) / np.float32(3)))
    parameters = np.asarray(initial['parameters'], dtype=np.float64)
    parameters = parameters * (1 - rate * config['weight_decay']) - rate * clipped / (
        np.abs(clipped) + config['eps'])

    owner = object.__new__(ParallelNeuralTrainer)
    child = audited_child_module(tmp_path, monkeypatch)
    checked('init_with_worker', owner, model, (0,), 2, child)
    replay = None
    pids = []
    try:
        result = audit_native_call(owner.train_step, shards).assert_native()
        np.testing.assert_array_equal(np.asarray(result['losses'], np.float32).view(np.uint32),
                                      np.asarray([part[0] for part in reference], np.float32).view(np.uint32))
        actual_gradients = audit_native_call(owner.export_gradients).assert_native()
        for actual, want in zip(actual_gradients, expected):
            np.testing.assert_array_equal(np.asarray(actual).reshape(-1).view(np.uint32), want.view(np.uint32))
        checkpoint = audit_native_call(owner.checkpoint).assert_native()
        state = checkpoint['state']
        assert state['rng']['counter'] == initial['rng']['counter'] + 1
        assert state['rng']['seed'] == initial['rng']['seed']
        assert state['schedule'] == initial['schedule']
        assert state['t'] == 1
        np.testing.assert_allclose(np.asarray(state['parameters']), parameters, rtol=3e-5, atol=2e-7)
        np.testing.assert_allclose(np.asarray(state['exp_avg']), (1-config['beta1']) * clipped,
                                   rtol=2e-5, atol=2e-9)
        np.testing.assert_allclose(np.asarray(state['exp_avg_sq']), (1-config['beta2']) * clipped**2,
                                   rtol=3e-5, atol=2e-12)
        replay = audit_native_call(ParallelNeuralTrainer.from_checkpoint, checkpoint, devices=(0,)).assert_native()
        audit_native_call(owner.train_step, shards).assert_native()
        audit_native_call(replay.train_step, shards).assert_native()
        _assert_state_equal(audit_native_call(owner.checkpoint).assert_native()['state'],
                            audit_native_call(replay.checkpoint).assert_native()['state'])
        pids = [row['pid'] for row in owner._pool['workers'] + owner._update_pool['workers']]
    finally:
        primary = sys.exception()
        for active in (owner, replay):
            if active is not None:
                try:
                    audit_native_call(active.close).assert_native()
                except BaseException as error:
                    if primary is None:
                        raise
                    primary.add_note('Child cleanup also failed: ' + repr(error))
        if primary is not None:
            primary.add_note('Full Samba worker diagnostics and audit reports: ' + str(tmp_path))
            for path in sorted(tmp_path.glob('*.stderr.log')):
                primary.add_note(str(path) + '\n' + '\n'.join(path.read_text().splitlines()[-12:]))
    assert len(pids) == 2
    for pid in pids:
        path = tmp_path / (str(pid) + '.json')
        assert path.exists(), str(path)
        report = json.loads(path.read_text())
        assert report['exception'] is None, report
        for key in ('python_frames', 'forbidden_native_calls', 'wrapper_work', 'unobserved_existing_threads'):
            assert not report[key], report
