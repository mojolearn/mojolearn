"""Actual installed CPU ByteLM ownership/materialization regression.

Requires native bindings. No fake providers, monkeypatches or skip-on-missing
behavior; run with MOJOLEARN_VENDOR=cpu in a separate installed environment.
"""
import hashlib
import numpy as np
import pytest
import mojolearn as ml


def _shape():
    return ml.ByteLanguageModelConfig(batch=1, length=4, d_model=32, n_heads=4,
                                     n_kv=2, head_dim=8, intermediate=64,
                                     n_layers=1, vocab_size=32)


@pytest.mark.parametrize('threaded', [False, True], ids=['serial', 'threaded'])
def test_inference_owned_parameters_and_fresh_logits(threaded):
    shape = _shape()
    source = np.zeros(shape.n_total, dtype='f4')
    expected = hashlib.sha256(source.tobytes()).hexdigest()
    model = ml.LanguageModelInference(source, shape=shape, threaded=threaded, threads=2)
    assert model.parameters_sha256() == expected
    source.fill(7)
    assert model.parameters_sha256() == expected
    ids = np.array([[1, 3, 5, 7]], dtype='i4')
    first = model.logits(ids)
    second = model.logits(ids)
    assert type(first) is ml.Array and type(second) is ml.Array
    assert first is not second
    for result in (first, second):
        values = np.asarray(result)
        assert values.shape == (1, 4, 32) and values.dtype == np.float32
        assert np.isfinite(values).all()
    assert np.asarray(first).tobytes() == np.asarray(second).tobytes()


def test_host_training_uses_same_owned_array_materializer():
    shape = _shape()
    source = np.zeros(shape.n_total, dtype='f4')
    trainer = ml.LanguageModelHostTrainer(source, shape=shape)
    source.fill(7)
    assert not np.asarray(trainer.parameters_).any()
    result = trainer.train_step(np.array([[1, 3, 5, 7, 9]], dtype='i4'))
    assert np.isfinite(float(result))
    assert trainer.completed_steps == 1
    for value in (trainer.parameters_, trainer.m_, trainer.v_, trainer.gradient_):
        assert type(value) is ml.Array
        assert np.asarray(value).shape == (shape.n_total,)
        assert np.isfinite(np.asarray(value)).all()


def test_invalid_parameters_keep_native_validation_error():
    with pytest.raises(ValueError, match='parameters') as caught:
        ml.LanguageModelInference(np.zeros(3, dtype='f4'), shape=_shape())
    assert 'arena_native_types' not in str(caught.value)


def test_user_array_hook_is_not_admitted_by_context_repair():
    calls = []
    class UserArray:
        def __array__(self, *args, **kwargs):
            calls.append(True)
            raise AssertionError('user conversion hook executed')
    with pytest.raises(TypeError):
        ml.LanguageModelInference(UserArray(), shape=_shape())
    assert calls == []
