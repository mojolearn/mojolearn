"""Bounded root-only public host provider probes; runtime results must be retained separately."""
import types

import numpy as np
import pytest


# TOMBSTONE: test_host_native_provider_reuse_and_transactional_outputs and
# test_host_public_constructor_restore_and_consumed_metadata (LanguageModelHostTrainer)
# deleted 2026-11-22 by lane neural-consolidate with the class: CPU training is
# LanguageModelTrainer(device="cpu"), whose step is the same composed host step.
def _model():
    from mojolearn._byte_lm_config import ByteLanguageModelConfig
    shape = ByteLanguageModelConfig(batch=1, length=4, d_model=32, n_heads=4,
                                   n_kv=2, head_dim=8, intermediate=64,
                                   n_layers=1, vocab_size=32)
    parameters = (np.random.default_rng(91).standard_normal(shape.n_total) * .02).astype('f4')
    return types.SimpleNamespace(parameters_=parameters, shape=shape)


def test_host_inference_native_calls_ragged_and_refreshed_inputs():
    from mojolearn._byte_lm_host import LanguageModelInference
    from mojolearn._verify_native_runtime import audit_native_call
    template = _model()
    model = LanguageModelInference(template.parameters_, shape=template.shape, threaded=False, threads=1)
    ids = np.array([[1, 3, 5, 7], [2, 4, 6, 8]], dtype='i4')
    actual = np.asarray(audit_native_call(model.logits, ids).assert_native())
    expected = np.zeros_like(actual)
    parameters = np.asarray(template.parameters_)
    count = model._binding.byte_lm_host_logits([parameters.ctypes.data, ids.ctypes.data, expected.ctypes.data],
                                               list(ids.shape), model._native, 0, 1)
    assert count == expected.size and actual.tobytes() == expected.tobytes()
    assert audit_native_call(model.next_bytes, ids).assert_native() == list(np.argmax(expected[:, -1], axis=1))
    assert audit_native_call(model.next_bytes, ids, threaded=True, threads=1).assert_native() == list(np.argmax(expected[:, -1], axis=1))
    padded = ids.copy()
    padded[0, 2:] = 999
    ragged = np.asarray(audit_native_call(model.logits, padded, lengths=[np.int64(2), np.int64(4)]).assert_native())
    assert ragged[0, :2].tobytes() == expected[0, :2].tobytes()
    assert ragged[0, 2:].tobytes() == np.zeros_like(ragged[0, 2:]).tobytes()
    assert ragged[1].tobytes() == expected[1].tobytes()
    assert audit_native_call(model.next_bytes, padded, lengths=[2, 4]).assert_native() == [int(np.argmax(expected[0, 1])), int(np.argmax(expected[1, 3]))]
    short = ids[:, :2].astype('i8')[:, ::-1]
    fresh = np.asarray(audit_native_call(model.logits, short).assert_native())
    assert fresh.shape == (2, 2, template.shape.vocab_size)
    train = np.array([[1, 3, 5, 7, -100]], dtype='i4')
    bits = audit_native_call(model.loss_bits, train).assert_native()
    direct = model._binding.byte_lm_host_loss([parameters.ctypes.data, train.ctypes.data], model._native, 0, 1)
    assert bits == direct
    import struct
    assert struct.pack('<f', audit_native_call(model.loss, train).assert_native()) == struct.pack('<I', bits)


@pytest.mark.parametrize('resident', [False, True])
def test_cpu_language_model_public_provider_matches_host_owner(monkeypatch, tmp_path, resident):
    from mojolearn import _byte_lm_impl as impl
    from mojolearn._byte_lm_host import _load
    from mojolearn._verify_native_runtime import audit_native_call
    host = _load()
    context = dict(impl._LM_TRAINER_CONTEXT, byte_lm_bindings={'identical': host}, byte_lm_vendors=('cpu',))
    monkeypatch.setattr(impl, '_LM_TRAINER_CONTEXT', context)
    expected = _model()
    ids = np.array([[1, 3, 5, 7, 9]], dtype='i4')
    trainer = audit_native_call(impl.SmallByteLanguageModelTrainer, expected.parameters_,
                               data_schedule={'corpus': 'host-provider-test'},
                               shape=expected.shape, resident=resident).assert_native()
    # The baseline is the stateless host step (the former LanguageModelHostTrainer's
    # route; lane neural-consolidate 2026-11-22): both sessions give its bytes.
    baseline = impl.SmallByteLanguageModelTrainer(expected.parameters_, data_schedule={'corpus': 'host-provider-test'},
                                                  shape=expected.shape, resident=False, step_result='full')
    baseline_loss = baseline.train_step(ids)['loss']
    reference = baseline.state_dict()
    actual = audit_native_call(trainer.train_step, ids).assert_native()
    import struct
    assert struct.pack('<f', actual['loss']) == struct.pack('<f', baseline_loss)
    state = audit_native_call(trainer.state_dict).assert_native()
    for name in ('parameters', 'm', 'v'):
        assert np.asarray(state[name]).tobytes() == np.asarray(reference[name]).tobytes()
    snapshot = tuple(np.asarray(state[k]).tobytes() for k in ('parameters', 'm', 'v'))
    invalid = ids.copy()
    invalid[0, 0] = expected.shape.vocab_size
    with pytest.raises(ValueError):
        trainer.train_step(invalid)
    after = trainer.state_dict()
    assert after['completed_steps'] == 1
    assert tuple(np.asarray(after[k]).tobytes() for k in ('parameters', 'm', 'v')) == snapshot
    logits = audit_native_call(trainer.logits, ids[:, :-1]).assert_native()
    assert np.asarray(logits).shape == (1, 4, expected.shape.vocab_size)
    path = tmp_path / 'cpu.json'
    audit_native_call(trainer.export_checkpoint, path).assert_native()
    restored = audit_native_call(impl.SmallByteLanguageModelTrainer.from_checkpoint, path, resident=resident).assert_native()
    assert np.asarray(restored.state_dict()['parameters']).tobytes() == snapshot[0]
    audit_native_call(restored.close).assert_native()
    audit_native_call(trainer.close).assert_native()
