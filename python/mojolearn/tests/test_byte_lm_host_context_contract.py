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
    # CPU training is LanguageModelTrainer(device="cpu") (the former
    # LanguageModelHostTrainer's step, lane neural-consolidate 2026-11-22).
    trainer = ml.LanguageModelTrainer(source, shape=shape, data_schedule={'test': 'host-context'},
                                      device='cpu', resident=False, step_result='full')
    source.fill(7)
    assert not np.asarray(trainer.parameters_).any()
    result = trainer.train_step(np.array([[1, 3, 5, 7, 9]], dtype='i4'))
    assert np.isfinite(float(result['loss']))
    assert trainer.step_ == 1
    state = trainer.export_state()
    for value in (trainer.parameters_, state['m'], state['v'], result['flat_gradients']):
        assert type(value) is ml.Array
        assert np.asarray(value).shape == (shape.n_total,)
        assert np.isfinite(np.asarray(value)).all()
