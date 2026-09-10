# SPDX-License-Identifier: Apache-2.0
"""Host plumbing checks only; fake arithmetic is not a model qualification."""
import numpy as np
import pytest
from mojolearn import ByteLanguageModelConfig as Shape, SmallByteLanguageModelTrainer as Trainer
from mojolearn import _byte_lm_impl as impl
from mojolearn.tests.test_byte_lm_surface import host, buffer


def shape():
    return Shape(batch=1, length=17, d_model=16, n_heads=2, n_kv=1,
                 head_dim=8, intermediate=24)


def model(cfg=None):
    cfg = cfg or shape()
    return Trainer(np.zeros(cfg.n_total, np.float32), shape=cfg,
                   data_schedule={'dataset': 'host-sentinel'})


def configured(host):
    host.byte_lm_config_profile = lambda dimensions: Shape(*dimensions).profile
    def run(addresses, params, dimensions):
        cfg = Shape(*dimensions)
        host.calls.append((list(addresses), list(params), list(dimensions)))
        for source, dest in zip(addresses[:3], addresses[5:8]):
            buffer(dest, cfg.n_total)[:] = buffer(source, cfg.n_total)
        buffer(addresses[9], 20, True)[:] = buffer(addresses[3], 20, True)
        buffer(addresses[10], 1)[:] = 2.5
        if params[0]:
            buffer(addresses[8], cfg.n_total)[:] = np.arange(cfg.n_total, dtype=np.float32)
            buffer(addresses[5], cfg.n_total)[:] += 1
        return params[1] + params[0]
    host.byte_lm_run_configured = run


def test_default_registry_and_serialization_stay_v1(host):
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


def test_runtime_abi_registry_full_gradient_and_eval(host):
    configured(host)
    cfg = shape()
    m = model(cfg)
    ids = np.zeros((1, 18), np.int32)
    before = ids.tobytes()
    result = m.train_step(ids)
    assert result['loss'] == 2.5 and m.step_ == 1
    assert host.calls[-1][2] == list(cfg.native_shape)
    for entry in Trainer.parameter_registry(cfg):
        np.testing.assert_array_equal(result['gradients'][entry['name']].ravel(),
            np.arange(entry['offset'], entry['offset'] + entry['size'], dtype=np.float32))
    state = m.state_dict()
    assert m.evaluate(ids) == 2.5 and m.step_ == 1
    assert all(state[k].tobytes() == m.state_dict()[k].tobytes() for k in ('parameters', 'm', 'v', 'flags'))
    assert ids.tobytes() == before
    with pytest.raises(ValueError, match='shape'):
        m.train_step(np.zeros((2, 33), np.int32))


def test_configured_checkpoint_roundtrip_and_shape_integrity(host, tmp_path):
    configured(host)
    m = model()
    m.train_step(np.zeros((1, 18), np.int32))
    path = tmp_path / 'runtime.json'
    m.save_checkpoint(path)
    restored = Trainer.from_checkpoint(path)
    assert restored.state_dict()['profile'] == shape().profile
    assert restored.state_dict()['model_shape'] == shape().to_dict()
    assert restored.parameters_.tobytes() == m.parameters_.tobytes()
    assert restored.evaluate(np.zeros((1, 18), np.int32)) == 2.5
    state = m.state_dict()
    state['model_shape']['length'] = 18
    with pytest.raises(ValueError, match='profile'):
        m.load_state_dict(state)
    assert m.state_dict()['model_shape']['length'] == 17
    state = m.state_dict()
    del state['model_shape']
    with pytest.raises(ValueError, match='profile'):
        m.load_state_dict(state)


def test_named_runtime_parameters_and_snapshot_ownership(host):
    cfg = shape()
    registry = Trainer.parameter_registry(cfg)
    supplied = {entry['name']: np.full(entry['shape'], i, np.float32) for i, entry in enumerate(registry)}
    m = Trainer(supplied, shape=cfg, data_schedule={'dataset': 'host'})
    for i, entry in enumerate(registry):
        assert np.all(m.parameters_[entry['offset']:entry['offset'] + entry['size']] == i)
    state = m.state_dict()
    state['model_shape']['batch'] = 99
    assert m.state_dict()['model_shape']['batch'] == 1


def test_old_extension_refuses_runtime_before_pointer_call(host):
    m = model()
    with pytest.raises(ImportError, match='runtime shapes'):
        m.evaluate(np.zeros((1, 18), np.int32))
    assert not host.calls


def test_profile_negotiation_refuses_mismatch(host):
    configured(host)
    host.byte_lm_config_profile = lambda dimensions: impl.PROFILE
    with pytest.raises(RuntimeError, match='profile mismatch'):
        model().evaluate(np.zeros((1, 18), np.int32))
    assert not host.calls


def test_checkpoint_size_refused_before_snapshot_copy(host, monkeypatch, tmp_path):
    cfg = Shape(d_model=64, n_heads=4, n_kv=2, head_dim=16, intermediate=128)
    m = model(cfg)
    assert cfg.n_total * 24 > impl._CHECKPOINT_LIMIT
    monkeypatch.setattr(m, 'state_dict', lambda: pytest.fail('oversize save copied state'))
    with pytest.raises(ValueError, match='2 MiB'):
        m.save_checkpoint(tmp_path / 'too-large.json')
    assert not (tmp_path / 'too-large.json').exists()


def test_load_state_can_replace_shape_without_cached_dimensions(host):
    configured(host)
    m = model(Shape())
    m.load_state_dict(model().state_dict())
    assert m.evaluate(np.zeros((1, 18), np.int32)) == 2.5
    assert host.calls[-1][2] == list(shape().native_shape)
