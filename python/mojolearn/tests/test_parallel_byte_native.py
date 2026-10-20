# SPDX-License-Identifier: Apache-2.0
"""Legacy parallel byte-LM whole-operation audits and independent oracles.

Input/checkpoint construction and NumPy oracles are test-controller work.
Physical multi-device evidence requires explicit installed GPU device IDs.
"""
import math
import os

import numpy as np
import pytest

from mojolearn import ByteLanguageModelConfig, ParallelByteLanguageModelTrainer
from mojolearn import _backend, _byte_lm_impl
from mojolearn._verify_native_runtime import audit_native_call


FIELDS = ('parameters', 'm', 'v', 'flags')


def state_fixture():
    shape = ByteLanguageModelConfig(batch=1, length=4, d_model=8, n_heads=2,
                                   n_kv=1, head_dim=4, intermediate=16,
                                   n_layers=1, vocab_size=16)
    state = dict(schema='mojolearn.small-byte-lm-state.v1',
                 profile=_byte_lm_impl._runtime_profile(shape, 'identical'),
                 numeric_mode='identical', model_shape=shape.to_dict(),
                 parameter_names=list(shape.parameter_names),
                 parameter_shapes=[list(x) for x in shape.parameter_shapes],
                 parameter_offsets=list(shape.offsets),
                 parameters=np.zeros(shape.n_total, np.float32),
                 m=np.zeros(shape.n_total, np.float32),
                 v=np.zeros(shape.n_total, np.float32),
                 flags=np.zeros(shape.n_tensors, np.int32),
                 completed_steps=0, next_batch_index=0,
                 config=dict(kind=2, lr=.125, beta1=.5, beta2=.75,
                             eps=1e-8, weight_decay=.25, momentum=0.,
                             dampening=0., nesterov=False, max_norm=0.),
                 data_schedule={'name': 'parallel-native-independent',
                                'offsets': [0, 4], 'seed': 21})
    # With all norm/lm-head weights zero, every gradient is analytically zero,
    # including this nonzero embedding; only decoupled weight decay changes it.
    state['parameters'][0] = .25
    return state, shape


def construct(state, **kwargs):
    owner = object.__new__(ParallelByteLanguageModelTrainer)
    audit_native_call(owner.__init__, state, **kwargs).assert_native()
    return owner


def native_failure(operation, exception, *args, **kwargs):
    observed = audit_native_call(operation, *args, **kwargs)
    assert isinstance(observed.exception, exception), observed.report()
    report = observed.report()
    for key in ('python_frames', 'forbidden_native_calls', 'wrapper_work',
                'unobserved_existing_threads'):
        assert not report[key], report
    return str(observed.exception)


def test_constructor_owns_arrays_schedule_and_accepts_portable_registry():
    state, shape = state_fixture()
    owner = construct(state, logical_shards=3)
    try:
        assert owner._shape.to_dict() == shape.to_dict()
        assert owner._state['parameter_offsets'] == list(shape.offsets)
        state['parameters'][0] = 11
        state['data_schedule']['offsets'][0] = 99
        assert np.asarray(owner._state['parameters'])[0] == .25
        assert owner._state['data_schedule']['offsets'][0] == 0
        assert audit_native_call(type(owner).step_.fget, owner).assert_native() == 0
        rate = audit_native_call(owner.set_lr, .1).assert_native()
        assert rate == float(np.float32(.1))
        assert owner._state['config']['lr'] == rate
        native_failure(owner.train_step, ValueError, [])
    finally:
        audit_native_call(owner.close).assert_native()
    native_failure(owner._open, RuntimeError)


@pytest.mark.parametrize('field,value', [('completed_steps', True),
                                        ('next_batch_index', -1),
                                        ('profile', 'wrong'),
                                        ('numeric_mode', 'unknown')])
def test_state_admission_failure_has_no_python_runtime(field, value):
    state, _ = state_fixture()
    state[field] = value
    owner = object.__new__(ParallelByteLanguageModelTrainer)
    native_failure(owner.__init__, ValueError, state)


def test_nonfinite_and_wrong_dtype_arrays_refused_before_device_open():
    state, _ = state_fixture()
    owner = object.__new__(ParallelByteLanguageModelTrainer)
    state['parameters'][0] = np.nan
    native_failure(owner.__init__, ValueError, state)
    state['parameters'][0] = .25
    state['flags'] = state['flags'].astype(np.float32)
    native_failure(owner.__init__, TypeError, state)


def test_cpu_cooperative_session_refusal_preserves_public_contract():
    if _backend.vendor() != 'cpu':
        pytest.skip('CPU refusal is specific to the installed CPU binding')
    state, _ = state_fixture()
    owner = construct(state)
    try:
        assert 'cooperative GPU byte-LM session' in native_failure(owner._open, NotImplementedError)
    finally:
        audit_native_call(owner.close).assert_native()


def test_gpu_parallel_step_independent_decay_loss_and_complete_replay():
    if _backend.vendor() == 'cpu':
        pytest.skip('native cooperative byte-LM session requires a GPU')
    devices = tuple(map(int, os.environ.get('MOJOLEARN_PARALLEL_BYTE_DEVICES', '0').split(',')))
    state, shape = state_fixture()
    shards = [np.arange(5, dtype=np.int32).reshape(1, 5) for _ in range(3)]
    owner = construct(state, devices=devices, logical_shards=3)
    replay = None
    try:
        result = audit_native_call(owner.train_step, shards).assert_native()
        np.testing.assert_allclose(result['losses'], [math.log(shape.vocab_size)] * 3,
                                   rtol=0, atol=2e-6)
        assert result['completed_steps'] == 1
        exported = audit_native_call(owner.state_dict).assert_native()
        expected = state['parameters'] * np.float32(1 - .125 * .25)
        np.testing.assert_array_equal(np.asarray(exported['parameters']), expected)
        for key in ('m', 'v', 'flags'):
            np.testing.assert_array_equal(np.asarray(exported[key]), state[key])
        gradient = audit_native_call(owner.export_gradients).assert_native()
        np.testing.assert_array_equal(np.asarray(gradient), np.zeros(shape.n_total, np.float32))
        into = {k: np.empty_like(state[k]) for k in FIELDS}
        assert audit_native_call(owner.export_raw, into=into).assert_native() is into
        for key in FIELDS:
            np.testing.assert_array_equal(into[key], np.asarray(exported[key]))
        ownership = audit_native_call(owner.optimizer_ownership).assert_native()
        assert len(ownership) == len(devices)
        assert sum(row['count'] for row in ownership) == shape.n_total
        checkpoint = audit_native_call(owner.checkpoint).assert_native()
        replay = audit_native_call(ParallelByteLanguageModelTrainer.from_checkpoint,
                                   checkpoint, devices=(devices[0],)).assert_native()
        audit_native_call(owner.train_step, shards).assert_native()
        audit_native_call(replay.train_step, shards).assert_native()
        a = audit_native_call(owner.state_dict).assert_native()
        b = audit_native_call(replay.state_dict).assert_native()
        for key in FIELDS:
            assert np.asarray(a[key]).tobytes() == np.asarray(b[key]).tobytes()
        for key in ('completed_steps', 'next_batch_index', 'config', 'data_schedule'):
            assert a[key] == b[key]
    finally:
        audit_native_call(owner.close).assert_native()
        if replay is not None:
            audit_native_call(replay.close).assert_native()
