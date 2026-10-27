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
