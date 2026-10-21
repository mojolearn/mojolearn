"""Public distributed training refuses Python protocols before state/process work."""
from collections import defaultdict

import numpy as np
import pytest

from mojolearn import ParallelNeuralTrainer, SmallMLPTrainer, _backend
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_parallel_mlp_native import model_fixture, independent_gradient
from mojolearn.tests.test_parallel_neural_native import training_state


@pytest.fixture(autouse=True)
def identical_mode():
    if _backend.default_mode() != 'identical':
        pytest.skip('parallel neural training retains its identical-only contract')


class Hook:
    calls = []

    def fail(self, name):
        Hook.calls.append(name)
        raise AssertionError('Python protocol reached: ' + name)

    def __iter__(self):
        return self.fail('iter')

    def __getattr__(self, name):
        return self.fail('getattr')

    def __float__(self):
        return self.fail('float')

    def __index__(self):
        return self.fail('index')

    def __buffer__(self, flags):
        return self.fail('buffer')


def rejected(operation, *args, **kwargs):
    Hook.calls.clear()
    observed = audit_native_call(operation, *args, **kwargs)
    assert isinstance(observed.exception, TypeError), observed.report()
    assert Hook.calls == []
    for name in ('python_frames', 'forbidden_native_calls', 'wrapper_work', 'unobserved_existing_threads'):
        assert not observed.report()[name], observed.report()
    return observed


@pytest.mark.parametrize('field', ['model', 'devices', 'logical_shards'])
def test_public_constructor_refuses_protocols_before_snapshot(field):
    args = dict(model=model_fixture(), devices=(0,), logical_shards=1)
    args[field] = Hook()
    rejected(ParallelNeuralTrainer, **args)


def test_model_subclass_and_spoofed_class_are_never_entered():
    class Subclass(SmallMLPTrainer):
        def __getattribute__(self, name):
            Hook.calls.append('getattribute')
            raise AssertionError('subclass attribute hook reached')
        def __del__(self):
            pass  # test fixture has no initialized model state to release
    class Spoof:
        @property
        def __class__(self):
            Hook.calls.append('class descriptor')
            return SmallMLPTrainer
    foreign = object.__new__(Subclass)
    rejected(ParallelNeuralTrainer, foreign)
    rejected(ParallelNeuralTrainer, Spoof())


@pytest.mark.parametrize('location', ['outer', 'pair', 'inputs', 'targets', 'array_subclass'])
def test_public_train_refuses_hooks_without_launch_or_state_change(location):
    owner = audit_native_call(ParallelNeuralTrainer, model_fixture()).assert_native()
    x, y = np.zeros((1, 8), np.float32), np.zeros(1, np.int32)
    shards = [(x, y)]
    if location == 'outer':
        shards = Hook()
    elif location == 'pair':
        shards = [Hook()]
    elif location == 'inputs':
        shards = [(Hook(), y)]
    elif location == 'targets':
        shards = [(x, Hook())]
    else:
        class CustomArray(np.ndarray):
            def __array__(self, *args):
                Hook.calls.append('array')
                raise AssertionError('subclass array conversion reached')
        shards = [(x.view(CustomArray), y)]
    try:
        before = audit_native_call(owner.checkpoint).assert_native()
        rejected(owner.train_step, shards)
        assert owner._pool['workers'] == owner._update_pool['workers'] == []
        assert training_state(audit_native_call(owner.checkpoint).assert_native()) == training_state(before)
    finally:
        audit_native_call(owner.close).assert_native()


@pytest.mark.parametrize('field', ['optimizer', 'config', 'schedule', 'default_factory'])
def test_mutated_model_metadata_is_refused_but_native_cleanup_remains_available(field):
    model = model_fixture()
    owner = audit_native_call(ParallelNeuralTrainer, model).assert_native()
    if field == 'optimizer':
        model._opt = Hook()
    elif field == 'config':
        model._config['lr'] = Hook()
    elif field == 'schedule':
        model._schedule['callback'] = Hook()
    else:
        model._schedule = defaultdict(lambda: Hook().fail('default factory'), model._schedule)
    rejected(owner.checkpoint)
    assert owner._pool['workers'] == owner._update_pool['workers'] == []
    # Closing touches native locks/process owners, not rejected model data.
    audit_native_call(owner.close).assert_native()


@pytest.mark.parametrize('field', ['mapping', 'devices', 'nested_step'])
def test_public_checkpoint_restore_refuses_python_protocols(field):
    owner = audit_native_call(ParallelNeuralTrainer, model_fixture()).assert_native()
    try:
        checkpoint = audit_native_call(owner.checkpoint).assert_native()
    finally:
        audit_native_call(owner.close).assert_native()
    devices = (0,)
    if field == 'mapping':
        class Mapping(dict):
            def get(self, *args):
                Hook.calls.append('get')
                raise AssertionError('Python checkpoint mapping method reached')
        checkpoint = Mapping(checkpoint)
    elif field == 'devices':
        devices = Hook()
    else:
        checkpoint['state']['optimizer']['step'] = Hook()
    rejected(ParallelNeuralTrainer.from_checkpoint, checkpoint, devices=devices)


def test_public_training_native_buffers_calculus_and_checkpoint_continuation():
    model = model_fixture()
    owner = audit_native_call(ParallelNeuralTrainer, model, devices=range(1), logical_shards=2).assert_native()
    replay = None
    x = np.zeros((3, 8), np.float32)
    x[:, 0] = [1., -.25, .5]
    y = np.array([0, 1, 2], np.int16)
    shards = [[np.asfortranarray(x), y], (x.copy(), y.astype(np.int64))]
    try:
        initial = audit_native_call(owner.checkpoint).assert_native()
        expected_loss, gradient = independent_gradient(initial['state'], x, y)
        result = audit_native_call(owner.train_step, shards).assert_native()
        np.testing.assert_allclose(result['losses'], [expected_loss] * 2, rtol=3e-6, atol=3e-7)
        for actual, expected in zip(audit_native_call(owner.export_gradients).assert_native(), gradient):
            np.testing.assert_allclose(np.asarray(actual), expected * 2, rtol=5e-6, atol=5e-7)
        checkpoint = audit_native_call(owner.checkpoint).assert_native()
        replay = audit_native_call(ParallelNeuralTrainer.from_checkpoint, checkpoint, devices=[0]).assert_native()
        audit_native_call(owner.train_step, shards).assert_native()
        audit_native_call(replay.train_step, shards).assert_native()
        assert training_state(audit_native_call(owner.checkpoint).assert_native()) == training_state(audit_native_call(replay.checkpoint).assert_native())
    finally:
        audit_native_call(owner.close).assert_native()
        if replay is not None:
            audit_native_call(replay.close).assert_native()
