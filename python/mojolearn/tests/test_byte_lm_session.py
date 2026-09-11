"""Host lifecycle checks; native arithmetic and ownership are checked separately.

DEVIATION 2514: the resident session owns the device state; the fake binding
(`FakeByteLM.enable_sessions`) keeps a fake device state per session so the
lean result, the exports, the rollback and the lost-session error are
observable here without a GPU. No numerical claim.
"""
import numpy as np
import pytest

from mojolearn import SmallByteLanguageModelTrainer as Trainer
from mojolearn.tests.test_byte_lm_surface import host, initial, ids, state_digest


def sessions(fake):
    return fake.enable_sessions()


def model(step_result='full'):
    return Trainer(initial(), data_schedule={'dataset': 'session-test'}, resident=True,
                   step_result=step_result)


def test_lazy_reuse_close_and_restore(host):
    created, closed, calls = sessions(host)
    trainer = model()
    assert created == []
    assert trainer.run_metadata()['device_state_lifetime'] == 'resident'
    assert created == []
    trainer.train_step(ids())
    checkpoint = trainer.state_dict()
    trainer.evaluate(ids())
    trainer.train_step(ids())
    assert calls == [created[0]] * 3
    trainer.close()
    trainer.close()
    assert closed == [created[0]]
    trainer.load_state_dict(checkpoint)
    trainer.train_step(ids())
    assert trainer.step_ == 2 and len(created) == 2
    trainer.load_state_dict(checkpoint)
    assert closed == created


def test_state_arrays_live_on_the_device_while_open(host):
    created, closed, calls = sessions(host)
    trainer = model()
    assert trainer._state['parameters'] is not None
    trainer.train_step(ids())
    assert all(trainer._state[key] is None for key in ('parameters', 'm', 'v'))
    assert trainer._state['flags'] is not None
    np.testing.assert_array_equal(trainer.parameters_, np.ones(34944, np.float32))
    trainer.close()
    assert all(trainer._state[key] is not None for key in ('parameters', 'm', 'v'))
    np.testing.assert_array_equal(trainer._state['m'], np.full(34944, 2, np.float32))
    assert closed == [created[0]]


def test_lean_result_keys_and_export_gradients(host):
    created, closed, calls = sessions(host)
    trainer = model('lean')
    result = trainer.train_step(ids())
    assert set(result) == {'loss', 'step', 'completed_steps', 'next_batch_index', 'flags'}
    assert result['loss'] == 1.25 and result['step'] == 1
    np.testing.assert_array_equal(result['flags'], np.ones(20, np.int32))
    exported = trainer.export_gradients()
    assert set(exported) == {'flat_gradients', 'gradients'}
    np.testing.assert_array_equal(exported['flat_gradients'], np.full(34944, .125, np.float32))
    assert set(exported['gradients']) == set(Trainer.parameter_registry()[i]['name'] for i in range(20))
    assert set(trainer.export_gradients(named=False)) == {'flat_gradients'}
    assert trainer.run_metadata()['step_result'] == 'lean'
    assert trainer.run_metadata()['last_export_step'] == 0
    trainer.state_dict()
    assert trainer.run_metadata()['last_export_step'] == 1


def test_full_result_on_a_resident_session_is_the_lean_step_plus_the_gradient_export(host):
    created, closed, calls = sessions(host)
    trainer = model('full')
    result = trainer.train_step(ids())
    assert set(result) == {'loss', 'step', 'completed_steps', 'next_batch_index',
                           'flat_gradients', 'gradients'}
    np.testing.assert_array_equal(result['flat_gradients'], np.full(34944, .125, np.float32))
    assert result['gradients']['embed'].shape == (256, 32)


def test_exports_are_copies_and_mutation_cannot_reach_the_device(host):
    created, closed, calls = sessions(host)
    trainer = model('lean')
    trainer.train_step(ids())
    state = trainer.export_state()
    np.asarray(state['parameters'])[0] += .25
    np.asarray(state['v'])[0] = -1
    np.asarray(state['flags'])[0] = 0
    gradients = trainer.export_gradients()
    np.asarray(gradients['flat_gradients'])[0] = 7
    np.asarray(gradients['gradients']['embed'])[0, 0] = 7
    tokens = ids()
    trainer.train_step(tokens)
    tokens[:] = 0
    clean = trainer.export_state()
    np.testing.assert_array_equal(clean['parameters'], np.full(34944, 2, np.float32))
    np.testing.assert_array_equal(clean['v'], np.full(34944, 6, np.float32))
    np.testing.assert_array_equal(clean['flags'], np.ones(20, np.int32))
    np.testing.assert_array_equal(trainer.export_gradients()['flat_gradients'],
                                  np.full(34944, .125, np.float32))
    with pytest.raises(ValueError):
        trainer.load_state_dict(state)
    assert closed == [] and len(created) == 1
    trainer.train_step(ids())
    assert trainer.step_ == 3


@pytest.mark.parametrize('fault', ['fail_after_write', 'wrong_step',
                                  'nonfinite_after_write', 'leave_gradient_unwritten'])
def test_native_and_python_failures_roll_back_and_keep_the_session(host, fault):
    created, closed, calls = sessions(host)
    trainer = model()
    trainer.train_step(ids())
    before = state_digest(trainer)
    setattr(host, fault, True)
    with pytest.raises((RuntimeError, ValueError)):
        trainer.train_step(ids())
    assert state_digest(trainer) == before
    assert closed == [] and len(created) == 1
    assert trainer.step_ == 1
    with pytest.raises(RuntimeError, match='no gradient'):
        trainer.export_gradients()
    setattr(host, fault, False)
    trainer.train_step(ids())
    assert trainer.step_ == 2 and len(created) == 1 and closed == []
    np.testing.assert_array_equal(trainer.export_gradients()['flat_gradients'],
                                  np.full(34944, .125, np.float32))


def test_export_gradients_is_refused_before_any_step_and_on_the_stateless_path(host):
    created, closed, calls = sessions(host)
    trainer = model('lean')
    with pytest.raises(RuntimeError, match='open resident session'):
        trainer.export_gradients()
    trainer.evaluate(ids())
    with pytest.raises(RuntimeError, match='no gradient'):
        trainer.export_gradients()
    stateless = Trainer(initial(), data_schedule={'dataset': 'session-test'})
    stateless.train_step(ids())
    with pytest.raises(RuntimeError, match='open resident session'):
        stateless.export_gradients()


def test_close_exports_then_releases_and_the_next_call_readmits(host, tmp_path):
    created, closed, calls = sessions(host)
    trainer = model('lean')
    trainer.train_step(ids())
    trainer.close()
    assert closed == [created[0]]
    assert trainer.run_metadata()['last_export_step'] == 1
    snapshot = trainer.state_dict()
    np.testing.assert_array_equal(snapshot['parameters'], np.ones(34944, np.float32))
    assert snapshot['completed_steps'] == 1
    trainer.export_checkpoint(tmp_path / 'closed.json')
    restored = Trainer.from_checkpoint(tmp_path / 'closed.json', resident=True)
    assert state_digest(restored) == state_digest(trainer)
    trainer.train_step(ids())
    assert len(created) == 2 and trainer.step_ == 2
    trainer.export_checkpoint(tmp_path / 'open.json')
    assert Trainer.from_checkpoint(tmp_path / 'open.json').step_ == 2


def test_lost_session_raises_with_the_last_export_step(host):
    created, closed, calls = sessions(host)
    trainer = model('lean')
    trainer.train_step(ids())
    trainer.train_step(ids())
    trainer.export_state()
    trainer.train_step(ids())
    host.wrong_step = True
    host.lose_on_rollback = True
    with pytest.raises(RuntimeError):
        trainer.train_step(ids())
    host.wrong_step = False
    for call in (lambda: trainer.train_step(ids()), lambda: trainer.evaluate(ids()),
                 trainer.export_state, trainer.export_gradients, trainer.state_dict):
        with pytest.raises(RuntimeError, match='session lost at step 3; last exported state is step 2'):
            call()
    assert closed == [] and len(created) == 1
    trainer.close()
    assert closed == [created[0]]
    with pytest.raises(RuntimeError, match='session lost'):
        trainer.train_step(ids())
    trainer.load_state_dict(Trainer(initial(), data_schedule={'dataset': 'session-test'}).state_dict())
    host.lose_on_rollback = False
    trainer.train_step(ids())
    assert trainer.step_ == 1 and len(created) == 2


def test_invalid_restore_preserves_live_session(host):
    created, closed, calls = sessions(host)
    trainer = model()
    trainer.train_step(ids())
    state = trainer.state_dict()
    np.asarray(state['v'])[0] = -1
    with pytest.raises(ValueError):
        trainer.load_state_dict(state)
    assert closed == []
    trainer.evaluate(ids())
    assert calls == [created[0]] * 2


def test_missing_session_binding_never_falls_back_silently(host):
    trainer = model()
    with pytest.raises(ImportError, match='owned sessions'):
        trainer.train_step(ids())
    assert trainer.step_ == 0 and host.calls == []


def test_checkpoint_restores_resident_preference_without_serializing_device_state(host, tmp_path):
    created, closed, calls = sessions(host)
    trainer = model()
    trainer.train_step(ids())
    path = tmp_path / 'session.json'
    trainer.save_checkpoint(path)
    restored = Trainer.from_checkpoint(path, resident=True)
    assert len(created) == 1
    assert state_digest(restored) == state_digest(trainer)
    restored.train_step(ids())
    assert len(created) == 2 and restored.step_ == 2


@pytest.mark.parametrize('value', [1, 'yes', np.bool_(True), None])
def test_resident_requires_boolean(host, value):
    with pytest.raises(TypeError, match='resident'):
        Trainer(initial(), data_schedule={'dataset': 'test'}, resident=value)


@pytest.mark.parametrize('kwargs', [dict(resident=False, step_result='lean'),
                                    dict(step_result='lean'),
                                    dict(resident=True, step_result='terse'),
                                    dict(resident=True, step_result=7)])
def test_lean_requires_resident_and_the_result_kind_is_checked(host, kwargs):
    with pytest.raises(ValueError, match='step_result'):
        Trainer(initial(), data_schedule={'dataset': 'test'}, **kwargs)


def test_default_step_result_is_lean_for_resident_and_full_for_stateless(host):
    # DEVIATION 2514 step 9: the default flipped on gate G5 (H100, 2026-09-11).
    assert Trainer(initial(), data_schedule={'dataset': 'test'}).run_metadata()['step_result'] == 'full'
    assert Trainer(initial(), data_schedule={'dataset': 'test'},
                   resident=True).run_metadata()['step_result'] == 'lean'
    assert Trainer(initial(), data_schedule={'dataset': 'test'}, resident=True,
                   step_result='full').run_metadata()['step_result'] == 'full'
