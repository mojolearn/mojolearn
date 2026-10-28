"""Focused public probes for root's compiled Mac pooled-output repair."""
import gc
import numpy as np
import pytest


def _model():
    from mojolearn import LanguageModelTrainer, LanguageModelConfig
    shape = LanguageModelConfig(batch=1, length=4, d_model=32, n_heads=4,
                                n_kv=2, head_dim=8, intermediate=64,
                                n_layers=1, vocab_size=32)
    parameters = (np.random.default_rng(81).standard_normal(shape.n_total) * .02).astype('f4')
    return LanguageModelTrainer(parameters, data_schedule={'batch_size': 1},
                                shape=shape, resident=True, step_result='lean')


def test_resident_logits_address_is_native_and_held_outputs_survive_pool_reuse():
    from mojolearn._verify_native_runtime import audit_native_call
    model = _model()
    ids = np.array([[1, 3, 5, 7]], dtype='i4')
    held = audit_native_call(model.logits, ids).assert_native()
    original = np.asarray(held).copy()
    for _ in range(3):
        temporary = audit_native_call(model.logits, ids).assert_native()
        assert np.asarray(temporary).tobytes() == original.tobytes()
        del temporary
        gc.collect()
    assert np.asarray(held).tobytes() == original.tobytes()
    padded = ids.copy()
    padded[0, 2:] = 999
    ragged = audit_native_call(model.logits, padded, lengths=[2]).assert_native()
    assert np.asarray(ragged)[0, 2:].tobytes() == np.zeros_like(original[0, 2:]).tobytes()
    assert audit_native_call(model.next_bytes, ids).assert_native() == list(np.argmax(original[:, -1], axis=1))
    expected = [int(np.argmax(np.asarray(ragged)[0, 1]))]
    assert audit_native_call(model.next_bytes, padded, lengths=[2]).assert_native() == expected
    audit_native_call(model.close).assert_native()
    assert np.asarray(held).tobytes() == original.tobytes()


@pytest.mark.parametrize('name', ['__array_interface__', '_addr'])
def test_replaced_pooled_python_descriptor_is_refused_without_invoking_it(name):
    from mojolearn import _byte_lm_impl as impl
    model = _model()
    ids = np.array([[1, 3, 5, 7]], dtype='i4')
    calls = []
    def read(self):
        calls.append('read')
        raise AssertionError('a pooled Python descriptor executed')
    def write(self, value):
        calls.append('write')
        raise AssertionError('a pooled Python descriptor executed')
    cls = impl._PooledLogits
    missing = object()
    previous = vars(cls).get(name, missing)
    setattr(cls, name, property(read, write))
    try:
        with pytest.raises(TypeError, match='pooled logits storage hook changed'):
            model.logits(ids)
    finally:
        if previous is missing:
            delattr(cls, name)
        else:
            setattr(cls, name, previous)
    assert calls == []
    model.close()
