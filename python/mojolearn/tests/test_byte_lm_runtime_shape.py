# SPDX-License-Identifier: Apache-2.0
"""Runtime-shape plumbing against the installed compiled byte-LM binding.

The native trainer owner admits only the exact compiled module, so these use
the real binding at tiny shapes; they assert the registry, the shape carried
through state and checkpoints, and the refusals, not model quality."""
import math

import numpy as np
import pytest
from mojolearn import ByteLanguageModelConfig as Shape, SmallByteLanguageModelTrainer as Trainer
from mojolearn import _byte_lm_impl as impl


def shape():
    return Shape(batch=1, length=17, d_model=16, n_heads=2, n_kv=1,
                 head_dim=8, intermediate=24)


def model(cfg=None):
    cfg = cfg or shape()
    values = (np.random.default_rng(11).standard_normal(cfg.n_total) * .02).astype(np.float32)
    return Trainer(values, shape=cfg, step_result='full',
                   data_schedule={'dataset': 'host-sentinel'})


def _bits(state):
    return {k: np.asarray(state[k]).tobytes() for k in ('parameters', 'm', 'v', 'flags')}


def test_default_registry_and_serialization_stay_v1():
    cfg = Shape()
    assert cfg.n_total == 34944
    assert cfg.profile == impl.PROFILE
    assert cfg.parameter_shapes == impl.PARAMETER_SHAPES
    assert cfg.offsets == impl._OFFSETS
    assert 'model_shape' not in model(cfg).state_dict()


@pytest.mark.parametrize('kwargs', [dict(batch=0), dict(length=8193), dict(batch=True),
    dict(length=1.5), dict(head_dim=7), dict(n_kv=3), dict(d_model=64),
    dict(batch=1 << 20), dict(intermediate=1 << 21), dict(batch=np.bool_(True))])
def test_invalid_shapes_refused_before_binding(kwargs):
    with pytest.raises(ValueError):
        Shape(**kwargs)


def test_runtime_abi_registry_full_gradient_and_eval():
    cfg = shape()
    m = model(cfg)
    ids = np.zeros((1, 18), np.int32)
    before = ids.tobytes()
    result = m.train_step(ids)
    assert math.isfinite(result['loss']) and m.step_ == 1
    registry = Trainer.parameter_registry(cfg)
    assert list(result['gradients']) == [entry['name'] for entry in registry]
    for entry in registry:
        gradient = np.asarray(result['gradients'][entry['name']])
        assert gradient.shape == tuple(entry['shape']) and np.isfinite(gradient).all()
    state = _bits(m.state_dict())
    loss = m.evaluate(ids)
    assert math.isfinite(loss) and m.step_ == 1
    assert _bits(m.state_dict()) == state
    assert ids.tobytes() == before
    with pytest.raises(ValueError, match='shape'):
        m.train_step(np.zeros((2, 33), np.int32))


def test_configured_checkpoint_roundtrip_and_shape_integrity(tmp_path):
    m = model()
    ids = np.zeros((1, 18), np.int32)
    m.train_step(ids)
    path = tmp_path / 'runtime.json'
    m.save_checkpoint(path)
    restored = Trainer.from_checkpoint(path)
    assert restored.state_dict()['profile'] == shape().profile
    assert restored.state_dict()['model_shape'] == shape().to_dict()
    assert np.asarray(restored.parameters_).tobytes() == np.asarray(m.parameters_).tobytes()
    assert restored.evaluate(ids) == m.evaluate(ids)
    state = m.state_dict()
    state['model_shape']['length'] = 18
    with pytest.raises(ValueError, match='profile'):
        m.load_state_dict(state)
    assert m.state_dict()['model_shape']['length'] == 17
    state = m.state_dict()
    del state['model_shape']
    with pytest.raises(ValueError, match='profile'):
        m.load_state_dict(state)


def test_named_runtime_parameters_and_snapshot_ownership():
    cfg = shape()
    registry = Trainer.parameter_registry(cfg)
    supplied = {entry['name']: np.full(entry['shape'], i, np.float32) for i, entry in enumerate(registry)}
    m = Trainer(supplied, shape=cfg, data_schedule={'dataset': 'host'})
    for i, entry in enumerate(registry):
        assert np.all(m.parameters_[entry['offset']:entry['offset'] + entry['size']] == i)
    state = m.state_dict()
    state['model_shape']['batch'] = 99
    assert m.state_dict()['model_shape']['batch'] == 1


# test_old_extension_refuses_runtime_before_pointer_call and
# test_profile_negotiation_refuses_mismatch are DELETED (2026-11-07): each
# installed a Python stand-in binding (one without the runtime-shape entry,
# one answering a different profile), and the native owner now refuses any
# provider that is not the exact compiled module before reading either
# (providers/python/neural_descriptor_pyapi.mojo `neural_descriptor_entry`).
# A binding without an entry is refused by name in
# test_byte_lm_host_trainer::test_a_binding_without_the_training_entry_is_refused_by_name,
# a wrong mode or vendor in test_byte_lm_host::test_wrong_mode_or_vendor_refused.


def test_checkpoint_size_refused_before_snapshot_copy(monkeypatch, tmp_path):
    cfg = Shape(d_model=64, n_heads=4, n_kv=2, head_dim=16, intermediate=128)
    m = model(cfg)
    assert cfg.n_total * 24 > impl._CHECKPOINT_LIMIT
    monkeypatch.setattr(m, 'state_dict', lambda: pytest.fail('oversize save copied state'))
    with pytest.raises(ValueError, match='2 MiB'):
        m.save_checkpoint(tmp_path / 'too-large.json')
    assert not (tmp_path / 'too-large.json').exists()


def test_load_state_can_replace_shape_without_cached_dimensions():
    m = model(Shape())
    source = model()
    m.load_state_dict(source.state_dict())
    ids = np.zeros((1, 18), np.int32)
    assert m.state_dict()['model_shape'] == shape().to_dict()
    assert m.evaluate(ids) == source.evaluate(ids)


@pytest.mark.parametrize('layers,vocab', [(1, 257), (3, 513), (12, 50257)])
def test_generalized_registry_training_and_restore(tmp_path, layers, vocab):
    cfg = Shape(batch=1, length=3, d_model=8, n_heads=1, n_kv=1,
                head_dim=8, intermediate=8, n_layers=layers, vocab_size=vocab)
    m = model(cfg)
    registry = Trainer.parameter_registry(cfg)
    assert len(registry) == 2 + 9 * layers
    assert registry[-1]['name'] == 'lm_head'
    assert registry[-2]['name'] == f'block{layers - 1}.w_down'
    assert registry[0]['shape'] == (vocab, 8)
    ids = np.array([[0, 256, vocab - 1, 7]], np.int32)
    result = m.train_step(ids)
    assert len(result['gradients']) == cfg.n_tensors
    for entry in registry:
        assert np.asarray(result['gradients'][entry['name']]).shape == tuple(entry['shape'])
    restored = model()
    restored.load_state_dict(m.state_dict())
    assert restored.evaluate(ids) == m.evaluate(ids)
    assert np.asarray(restored.state_dict()['flags']).shape == (cfg.n_tensors,)
    invalid = ids.copy()
    invalid[0, 0] = vocab
    with pytest.raises(ValueError, match='IDs'):
        restored.train_step(invalid)
    if cfg.n_total * 24 < impl._CHECKPOINT_LIMIT:
        path = tmp_path / 'generalized.json'
        m.save_checkpoint(path)
        assert Trainer.from_checkpoint(path).state_dict()['model_shape'] == cfg.to_dict()


def test_legacy_seven_field_state_remains_readable():
    m = model()
    state = m.state_dict()
    del state['model_shape']['n_layers']
    del state['model_shape']['vocab_size']
    m.load_state_dict(state)
    assert m.state_dict()['model_shape'] == shape().to_dict()


@pytest.mark.parametrize('kwargs', [dict(n_layers=0), dict(vocab_size=0),
                                   dict(n_layers=True), dict(vocab_size=1.5)])
def test_generalized_invalid_dimensions(kwargs):
    with pytest.raises(ValueError):
        Shape(**kwargs)


# Large host-only registry arithmetic is tested directly through ByteConfig in
# training/checks/byte_lm_config_check.mojo (no Python trainer or model allocation).
