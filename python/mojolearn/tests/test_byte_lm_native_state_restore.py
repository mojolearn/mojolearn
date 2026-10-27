"""Actual public candidate probes; root runs with the rebuilt Mac artifact."""
import copy

import numpy as np
import pytest


def _model():
    from mojolearn import LanguageModelTrainer
    from mojolearn._byte_lm_config import ByteLanguageModelConfig
    shape = ByteLanguageModelConfig(batch=1, length=4, d_model=32, n_heads=4,
                                   n_kv=2, head_dim=8, intermediate=64,
                                   n_layers=1, vocab_size=32)
    parameters = (np.random.default_rng(81).standard_normal(shape.n_total) * .02).astype('f4')
    return LanguageModelTrainer(parameters, data_schedule={'batch_size': 1},
                                shape=shape, resident=True, step_result='lean')


def _detached(state):
    result = {key: copy.deepcopy(value) for key, value in state.items()
              if key not in ('parameters', 'm', 'v', 'flags')}
    for key in ('parameters', 'm', 'v', 'flags'):
        result[key] = np.asarray(state[key]).copy()
    return result


def _same(left, right):
    for key in ('parameters', 'm', 'v', 'flags'):
        a, b = np.asarray(left[key]), np.asarray(right[key])
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), key
    for key in ('completed_steps', 'next_batch_index', 'config', 'data_schedule',
                'parameter_names', 'parameter_shapes', 'parameter_offsets'):
        assert left[key] == right[key], key


def test_restore_is_one_native_call_detaches_and_resumes():
    from mojolearn._verify_native_runtime import audit_native_call
    first, restored = _model(), _model()
    ids = np.array([[1, 3, 5, 7, 9]], dtype='i4')
    first.train_step(ids)
    saved = _detached(first.export_state())
    expected = _detached(saved)
    audit = audit_native_call(restored.load_state_dict, saved)
    assert audit.assert_native() is restored
    saved['parameters'].fill(0)
    saved['m'].fill(1)
    saved['flags'].fill(1)
    saved['data_schedule']['batch_size'] = 999
    _same(restored.export_state(), expected)
    a, b = first.train_step(ids), restored.train_step(ids)
    assert np.asarray(a['loss']).tobytes() == np.asarray(b['loss']).tobytes()
    _same(first.export_state(), restored.export_state())


def test_invalid_restore_preserves_live_session_and_state():
    model = _model()
    model.train_step(np.array([[1, 3, 5, 7, 9]], dtype='i4'))
    before = _detached(model.export_state())
    session = model._native_session
    mutations = (
        lambda s: s['v'].__setitem__(0, -1),
        lambda s: s['flags'].__setitem__(0, 2),
        lambda s: s.__setitem__('next_batch_index', 44),
        lambda s: s['parameter_offsets'].__setitem__(0, 1),
        lambda s: s['model_shape'].__setitem__('head_dim', 7),
        lambda s: s.__setitem__('parameters', s['parameters'].astype('f8')),
    )
    for mutate in mutations:
        state = _detached(before)
        mutate(state)
        with pytest.raises((TypeError, ValueError)):
            model.load_state_dict(state)
        assert model._native_session is session
        _same(model.export_state(), before)


def test_restore_strided_int32_flags_and_rejects_python_hooks():
    model = _model()
    saved = _detached(model.export_state())
    flags = np.zeros(saved['flags'].size * 2, dtype='i4')[::2]
    flags[:] = saved['flags']
    saved['flags'] = flags
    model.load_state_dict(saved)
    _same(model.export_state(), saved)
    calls = []
    class IndexHook:
        def __index__(self):
            calls.append('index')
            raise AssertionError('Python scalar hook reached')
    bad = _detached(saved)
    bad['completed_steps'] = IndexHook()
    with pytest.raises(TypeError):
        model.load_state_dict(bad)
    assert calls == []


def test_binary_checkpoint_public_roundtrip_and_original_io_errors(tmp_path):
    from mojolearn import LanguageModelTrainer
    from mojolearn._verify_native_runtime import audit_native_call
    model = _model()
    ids = np.array([[1, 3, 5, 7, 9]], dtype='i4')
    model.train_step(ids)
    expected = _detached(model.export_state())
    path = tmp_path / 'state.bin'
    digest = audit_native_call(model.export_checkpoint_binary, path).assert_native()
    import hashlib
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    restored = audit_native_call(LanguageModelTrainer.from_checkpoint_binary, path, resident=True).assert_native()
    _same(restored.export_state(), expected)
    model.train_step(ids)
    restored.train_step(ids)
    _same(model.export_state(), restored.export_state())
    missing = tmp_path / 'missing' / 'state.bin'
    with pytest.raises(FileNotFoundError) as failure:
        LanguageModelTrainer.from_checkpoint_binary(missing, resident=True)
    assert failure.value.errno == 2 and str(missing) in str(failure.value.filename)
    original = path.read_bytes()
    path.write_bytes(original[:-1])
    with pytest.raises(ValueError):
        LanguageModelTrainer.from_checkpoint_binary(path, resident=True)
    assert path.read_bytes() == original[:-1]
