"""Bounded root-only host provider runtime probe; constructors remain audited debt."""
import numpy as np
import pytest


def _model():
    from mojolearn._byte_lm_host import LanguageModelHostTrainer
    from mojolearn._byte_lm_config import ByteLanguageModelConfig
    shape = ByteLanguageModelConfig(batch=1, length=4, d_model=32, n_heads=4,
                                   n_kv=2, head_dim=8, intermediate=64,
                                   n_layers=1, vocab_size=32)
    parameters = (np.random.default_rng(91).standard_normal(shape.n_total) * .02).astype('f4')
    return LanguageModelHostTrainer(parameters, shape=shape)


def test_host_native_provider_reuse_and_transactional_outputs():
    from mojolearn._verify_native_runtime import audit_native_call
    from mojolearn._byte_lm_host import _HOST_TRAIN, _HOST_TRAIN_CONTEXT
    model, legacy = _model(), _model()
    ids = np.array([[1, 3, 5, 7, 9]], dtype='i4')
    arrays = [np.zeros_like(np.asarray(model.parameters_)) for _ in range(4)]
    # A direct call to the preserved old native door verifies the new provider
    # conveys identical addresses/settings and consumes every output correctly.
    inputs = [np.asarray(legacy.parameters_), np.asarray(legacy.m_), np.asarray(legacy.v_), ids]
    addresses = [a.ctypes.data for a in inputs + arrays]
    expected_bits = legacy._binding.byte_lm_host_train_step(addresses, legacy._native, legacy._scalars, 0)
    bits = audit_native_call(model.train_step, ids).assert_native()
    assert bits == expected_bits
    for actual, expected in zip((model.gradient_, model.parameters_, model.m_, model.v_), arrays):
        assert np.asarray(actual).tobytes() == expected.tobytes()
    provider = model._native_provider
    audit_native_call(model.loss, ids).assert_native()
    assert model._native_provider is provider
    assert _HOST_TRAIN(model, 'native_provider_info', (), _HOST_TRAIN_CONTEXT)[1][3] == 2
    before = tuple(np.asarray(x).tobytes() for x in (model.parameters_, model.m_, model.v_))
    completed = model.completed_steps
    factory = model._binding.byte_lm_host_api_v1
    calls = []
    def forbidden():
        calls.append(True)
        raise AssertionError('Python factory executed')
    try:
        model._binding.byte_lm_host_api_v1 = forbidden
        with pytest.raises(TypeError):
            model.train_step(ids)
    finally:
        model._binding.byte_lm_host_api_v1 = factory
    assert not calls and model.gradient_ is None and model.completed_steps == completed
    assert tuple(np.asarray(x).tobytes() for x in (model.parameters_, model.m_, model.v_)) == before


def test_host_public_constructor_restore_and_consumed_metadata():
    import hashlib
    from mojolearn._byte_lm_host import LanguageModelHostTrainer
    from mojolearn._verify_native_runtime import audit_native_call
    template = _model()
    expected = np.asarray(template.parameters_).copy()
    storage = np.zeros(expected.size * 2, dtype='f8')
    source = storage[::2]
    source[:] = expected
    model = audit_native_call(LanguageModelHostTrainer, source, shape=template.shape,
                              betas=np.array([.9, .999], dtype='f8')).assert_native()
    assert np.asarray(model.parameters_).tobytes() == expected.tobytes()
    source.fill(0)
    assert np.asarray(model.parameters_).tobytes() == expected.tobytes()
    assert audit_native_call(model.parameters_sha256).assert_native() == hashlib.sha256(expected.tobytes()).hexdigest()
    assert model.profile == model._binding.byte_lm_host_profile(model._native)
    restored = audit_native_call(LanguageModelHostTrainer.from_state, model.parameters_, model.m_, model.v_,
                                 completed_steps=3, shape=model.shape).assert_native()
    assert restored.completed_steps == 3 and restored.gradient_ is None
    assert np.asarray(restored.parameters_).tobytes() == expected.tobytes()
    calls = []
    class Hook:
        def __float__(self):
            calls.append('float')
            return .1
    with pytest.raises((TypeError, ValueError)):
        LanguageModelHostTrainer(expected, shape=model.shape, lr=Hook())
    assert calls == []


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
        materialized = lowbit.materialize(packed)
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
