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


def test_inference_constructor_native_flat_named_packed_and_atomic_admission():
    from mojolearn._byte_lm_host import LanguageModelInference
    from mojolearn._verify_native_runtime import audit_native_call
    from mojolearn import lowbit
    template = _model()
    shape = template.shape
    flat = np.asarray(template.parameters_).copy()
    named = {name: flat[shape.offsets[i]:shape.offsets[i + 1]].reshape(dims).copy()
             for i, (name, dims) in enumerate(zip(shape.parameter_names, shape.parameter_shapes))}
    model = audit_native_call(LanguageModelInference, named, shape=shape, threads=1).assert_native()
    assert model.weight_format == 'float32' and np.asarray(model._parameters).tobytes() == flat.tobytes()
    for fmt in ('bfloat16', 'int8'):
        packed = lowbit.pack(named, fmt)
        # `unpack` returns the float32 dict and the stored format.
        materialized, stored = lowbit.unpack(packed)
        assert stored == fmt
        expected = np.concatenate([np.asarray(materialized[name]).ravel() for name in shape.parameter_names])
        result = audit_native_call(LanguageModelInference, packed, shape=shape, threads=1).assert_native()
        assert result.weight_format == fmt and np.asarray(result._parameters).tobytes() == expected.tobytes()
    before = dict(model.__dict__)
    broken = dict(named)
    broken['embed'] = np.full_like(named['embed'], np.nan)
    with pytest.raises(ValueError):
        model.__init__(broken, shape=shape)
    assert model.__dict__ == before
    calls = []
    class MappingHook:
        def keys(self):
            calls.append('keys')
            return named.keys()
    with pytest.raises((TypeError, ValueError)):
        LanguageModelInference(MappingHook(), shape=shape)
    assert not calls


def test_host_inference_checkpoint_native_integrity_and_file_errors(tmp_path):
    from mojolearn._byte_lm_host import LanguageModelInference
    from mojolearn._byte_lm_impl import SmallByteLanguageModelTrainer
    from mojolearn._verify_native_runtime import audit_native_call
    template = _model()
    trainer = SmallByteLanguageModelTrainer(template.parameters_, data_schedule={'corpus': 'host-provider-test'},
                                            shape=template.shape, resident=False)
    path = tmp_path / 'state.json'
    trainer.export_checkpoint(path)
    restored = audit_native_call(LanguageModelInference.from_checkpoint, path, threads=1).assert_native()
    assert restored.shape == template.shape
    assert np.asarray(restored._parameters).tobytes() == np.asarray(template.parameters_).tobytes()
    with pytest.raises(FileNotFoundError) as failure:
        LanguageModelInference.from_checkpoint(tmp_path / 'missing.json')
    assert failure.value.errno == 2
    bad = tmp_path / 'bad.json'
    encoded = bytearray(path.read_bytes())
    position = encoded.index(b'payload_sha256') + len(b'payload_sha256') + 3
    encoded[position] = ord('0') if encoded[position] != ord('0') else ord('1')
    bad.write_bytes(encoded)
    with pytest.raises(ValueError, match='integrity'):
        LanguageModelInference.from_checkpoint(bad)
    trainer.close()


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
    baseline_bits = expected.train_step(ids)
    actual = audit_native_call(trainer.train_step, ids).assert_native()
    import struct
    assert struct.pack('<f', actual['loss']) == struct.pack('<I', baseline_bits)
    state = audit_native_call(trainer.state_dict).assert_native()
    assert np.asarray(state['parameters']).tobytes() == np.asarray(expected.parameters_).tobytes()
    assert np.asarray(state['m']).tobytes() == np.asarray(expected.m_).tobytes()
    assert np.asarray(state['v']).tobytes() == np.asarray(expected.v_).tobytes()
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
