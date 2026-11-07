# SPDX-License-Identifier: Apache-2.0
"""The resident byte-LM session lifecycle against the installed compiled binding.

The resident session owns the device state; these tests open, step, export,
close and restore it through the real binding and read the trainer's own
session record (`_session_open`, `_state`) where the old fake counted session
objects. No numerical claim beyond finiteness and same-bits restores.

DELETED (2026-11-07) with the retired Python owner mock, because each needed
a stand-in binding the native owner now refuses before any entry is read:
- test_native_and_python_failures_roll_back_and_keep_the_session and
  test_lost_session_raises_with_the_last_export_step: injected step faults
  and a failed rollback. The rollback and lost-session rules are native
  (bindings/byte_lm_trainer_native.mojo, training/byte_lm_host_session.mojo);
  the compiled binding has no fault hook, so they have no real-binding test
  (a coverage gap, reported). A real refusal keeping the session is tested in
  test_invalid_restore_preserves_live_session and
  test_byte_lm_native_state_restore::test_invalid_restore_preserves_live_session_and_state.
- test_missing_session_binding_never_falls_back_silently: a binding without
  the session entries; a missing compiled entry is refused by name at
  connection (test_byte_lm_host_trainer::test_a_binding_without_the_training_entry_is_refused_by_name).
"""
import hashlib
import math

import numpy as np
import pytest

from mojolearn import SmallByteLanguageModelTrainer as Trainer
from mojolearn import _byte_lm_impl as impl


def initial():
    return (np.random.default_rng(29).standard_normal(34944) * .02).astype(np.float32)


def ids():
    return (np.arange(66, dtype=np.int32) % 256).reshape(2, 33)


def state_digest(model):
    state = model.state_dict()
    for key in ('parameters', 'm', 'v', 'flags'):
        value = np.asarray(state[key])
        state[key] = {'dtype': value.dtype.str, 'shape': value.shape,
                      'hex': value.tobytes().hex()}
    return hashlib.sha256(impl._canonical(state)).hexdigest()


def model(step_result='full'):
    return Trainer(initial(), data_schedule={'dataset': 'session-test'}, resident=True,
                   step_result=step_result)


def is_open(trainer):
    return vars(trainer)['_session_open'] is True


def test_lazy_reuse_close_and_restore():
    trainer = model()
    assert not is_open(trainer)
    assert trainer.run_metadata()['device_state_lifetime'] == 'resident'
    assert not is_open(trainer)
    trainer.train_step(ids())
    assert is_open(trainer)
    checkpoint = trainer.state_dict()
    trainer.evaluate(ids())
    trainer.train_step(ids())
    assert is_open(trainer) and trainer.step_ == 2
    trainer.close()
    trainer.close()
    assert not is_open(trainer)
    trainer.load_state_dict(checkpoint)
    trainer.train_step(ids())
    assert trainer.step_ == 2 and is_open(trainer)
    trainer.load_state_dict(checkpoint)
    assert not is_open(trainer) and trainer.step_ == 1
    trainer.close()


def test_state_arrays_live_on_the_device_while_open():
    trainer = model()
    assert trainer._state['parameters'] is not None
    trainer.train_step(ids())
    assert all(trainer._state[key] is None for key in ('parameters', 'm', 'v'))
    assert trainer._state['flags'] is not None
    parameters = np.asarray(trainer.parameters_).copy()
    assert np.isfinite(parameters).all()
    trainer.close()
    assert all(trainer._state[key] is not None for key in ('parameters', 'm', 'v'))
    assert np.asarray(trainer._state['parameters']).tobytes() == parameters.tobytes()
    assert np.asarray(trainer._state['m']).any()


def test_lean_result_keys_and_export_gradients():
    trainer = model('lean')
    result = trainer.train_step(ids())
    assert set(result) == {'loss', 'step', 'completed_steps', 'next_batch_index', 'flags'}
    assert math.isfinite(result['loss']) and result['step'] == 1
    assert np.asarray(result['flags']).shape == (20,)
    exported = trainer.export_gradients()
    assert set(exported) == {'flat_gradients', 'gradients'}
    flat = np.asarray(exported['flat_gradients'])
    assert flat.shape == (34944,) and np.isfinite(flat).all()
    assert set(exported['gradients']) == set(Trainer.parameter_registry()[i]['name'] for i in range(20))
    assert set(trainer.export_gradients(named=False)) == {'flat_gradients'}
    assert trainer.run_metadata()['step_result'] == 'lean'
    assert trainer.run_metadata()['last_export_step'] == 0
    trainer.state_dict()
    assert trainer.run_metadata()['last_export_step'] == 1
    trainer.close()


def test_full_result_on_a_resident_session_is_the_lean_step_plus_the_gradient_export():
    full, lean = model('full'), model('lean')
    result = full.train_step(ids())
    lean_result = lean.train_step(ids())
    assert set(result) == {'loss', 'step', 'completed_steps', 'next_batch_index',
                           'flat_gradients', 'gradients'}
    assert result['loss'] == lean_result['loss']
    assert (np.asarray(result['flat_gradients']).tobytes()
            == np.asarray(lean.export_gradients()['flat_gradients']).tobytes())
    assert np.asarray(result['gradients']['embed']).shape == (256, 32)
    full.close()
    lean.close()


def test_exports_are_copies_and_mutation_cannot_reach_the_device():
    trainer = model('lean')
    twin = model('lean')
    trainer.train_step(ids())
    twin.train_step(ids())
    state = trainer.export_state()
    np.asarray(state['parameters'])[0] += .25
    np.asarray(state['v'])[0] = -1
    np.asarray(state['flags'])[0] = 0
    gradients = trainer.export_gradients()
    np.asarray(gradients['flat_gradients'])[0] = 7
    np.asarray(gradients['gradients']['embed'])[0, 0] = 7
    tokens = ids()
    trainer.train_step(tokens)
    twin.train_step(ids())
    tokens[:] = 0
    assert state_digest(trainer) == state_digest(twin)
    assert (np.asarray(trainer.export_gradients()['flat_gradients']).tobytes()
            == np.asarray(twin.export_gradients()['flat_gradients']).tobytes())
    with pytest.raises(ValueError):
        trainer.load_state_dict(state)
    assert is_open(trainer)
    trainer.train_step(ids())
    assert trainer.step_ == 3
    trainer.close()
    twin.close()


def test_export_gradients_is_refused_before_any_step_and_on_the_stateless_path():
    trainer = model('lean')
    with pytest.raises(RuntimeError, match='open resident session'):
        trainer.export_gradients()
    trainer.evaluate(ids())
    with pytest.raises(RuntimeError, match='no gradient'):
        trainer.export_gradients()
    trainer.close()
    # The stateless full result returns its gradients; there is no session
    # to export from.
    stateless = Trainer(initial(), data_schedule={'dataset': 'session-test'}, resident=False)
    stateless.train_step(ids())
    with pytest.raises(RuntimeError, match='open resident session'):
        stateless.export_gradients()


def test_close_exports_then_releases_and_the_next_call_readmits(tmp_path):
    trainer = model('lean')
    trainer.train_step(ids())
    parameters = np.asarray(trainer.parameters_).copy()
    trainer.close()
    assert not is_open(trainer)
    assert trainer.run_metadata()['last_export_step'] == 1
    snapshot = trainer.state_dict()
    assert np.asarray(snapshot['parameters']).tobytes() == parameters.tobytes()
    assert snapshot['completed_steps'] == 1
    trainer.export_checkpoint(tmp_path / 'closed.json')
    restored = Trainer.from_checkpoint(tmp_path / 'closed.json', resident=True)
    assert state_digest(restored) == state_digest(trainer)
    trainer.train_step(ids())
    assert is_open(trainer) and trainer.step_ == 2
    trainer.export_checkpoint(tmp_path / 'open.json')
    assert Trainer.from_checkpoint(tmp_path / 'open.json').step_ == 2
    trainer.close()


def test_invalid_restore_preserves_live_session():
    trainer = model()
    trainer.train_step(ids())
    before = state_digest(trainer)
    state = trainer.state_dict()
    np.asarray(state['v'])[0] = -1
    with pytest.raises(ValueError):
        trainer.load_state_dict(state)
    assert is_open(trainer)
    assert math.isfinite(trainer.evaluate(ids()))
    assert state_digest(trainer) == before
    trainer.close()


def test_checkpoint_restores_resident_preference_without_serializing_device_state(tmp_path):
    trainer = model()
    trainer.train_step(ids())
    path = tmp_path / 'session.json'
    trainer.save_checkpoint(path)
    restored = Trainer.from_checkpoint(path, resident=True)
    assert not is_open(restored)
    assert state_digest(restored) == state_digest(trainer)
    restored.train_step(ids())
    trainer.train_step(ids())
    assert is_open(restored) and restored.step_ == 2
    assert state_digest(restored) == state_digest(trainer)
    restored.close()
    trainer.close()


@pytest.mark.parametrize('value', [1, 'yes', np.bool_(True)])  # None is AUTO (cpu2-l11-neural)
def test_resident_requires_boolean(value):
    with pytest.raises(TypeError, match='resident'):
        Trainer(initial(), data_schedule={'dataset': 'test'}, resident=value)


@pytest.mark.parametrize('kwargs', [dict(resident=False, step_result='lean'),
                                    dict(resident=True, step_result='terse'),
                                    dict(resident=True, step_result=7)])
def test_lean_requires_resident_and_the_result_kind_is_checked(kwargs):
    with pytest.raises(ValueError, match='step_result'):
        Trainer(initial(), data_schedule={'dataset': 'test'}, **kwargs)


def test_default_step_result_is_lean_for_resident_and_full_for_stateless():
    # DEVIATION 2514 step 9: the default flipped on gate G5 (H100, 2026-09-11).
    # Andrew 2026-11-06: resident=None (AUTO) is lean too; only an explicit
    # resident=False keeps the full result.
    assert Trainer(initial(), data_schedule={'dataset': 'test'}).run_metadata()['step_result'] == 'lean'
    assert Trainer(initial(), data_schedule={'dataset': 'test'},
                   resident=False).run_metadata()['step_result'] == 'full'
    assert Trainer(initial(), data_schedule={'dataset': 'test'},
                   step_result='lean').run_metadata()['step_result'] == 'lean'
    assert Trainer(initial(), data_schedule={'dataset': 'test'},
                   resident=True).run_metadata()['step_result'] == 'lean'
    assert Trainer(initial(), data_schedule={'dataset': 'test'}, resident=True,
                   step_result='full').run_metadata()['step_result'] == 'full'
