"""Host lifecycle checks; native arithmetic and ownership are checked separately."""
import numpy as np
import pytest

from mojolearn import SmallByteLanguageModelTrainer as Trainer
from mojolearn.tests.test_byte_lm_surface import host, initial, ids, state_digest


def sessions(fake):
    created, closed, calls = [], [], []

    def create():
        session = object()
        created.append(session)
        return session

    def run(session, addresses, parameters, shape):
        assert session not in closed
        calls.append(session)
        return fake.byte_lm_run(addresses, parameters)

    fake.byte_lm_session_create = create
    fake.byte_lm_session_close = closed.append
    fake.byte_lm_session_run = run
    return created, closed, calls


def model():
    return Trainer(initial(), data_schedule={'dataset': 'session-test'}, resident=True)


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


@pytest.mark.parametrize('fault', ['fail_after_write', 'wrong_step',
                                  'nonfinite_after_write', 'modify_input',
                                  'leave_gradient_unwritten'])
def test_native_and_python_failures_discard_advanced_session(host, fault):
    created, closed, calls = sessions(host)
    trainer = model()
    trainer.train_step(ids())
    before = state_digest(trainer)
    setattr(host, fault, True)
    with pytest.raises((RuntimeError, ValueError)):
        trainer.train_step(ids())
    assert state_digest(trainer) == before
    assert closed == created
    setattr(host, fault, False)
    trainer.train_step(ids())
    assert trainer.step_ == 2 and len(created) == 2


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
