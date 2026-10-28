"""Authored public config probe; execute only in the final root Mac verifier."""
import dataclasses
import pickle
import pytest


def test_native_config_preserves_dataclass_value_contract():
    import numpy as np
    from mojolearn._byte_lm_config import ByteLanguageModelConfig as Config, require_shape, state_shape
    from mojolearn._verify_native_runtime import audit_native_call
    cfg = audit_native_call(Config, batch=np.int64(1), length=4, n_layers=1, vocab_size=32).assert_native()
    assert dataclasses.is_dataclass(cfg)
    assert cfg.native_shape == (1, 4, 32, 4, 2, 8, 64, 1, 32)
    fields = ('batch', 'length', 'd_model', 'n_heads', 'n_kv', 'head_dim', 'intermediate', 'n_layers', 'vocab_size', 'chunked_lm_head_v2')
    assert tuple(f.name for f in dataclasses.fields(cfg)) == fields
    assert cfg.to_dict() == dataclasses.asdict(cfg)
    assert cfg.n_tensors == len(cfg.parameter_shapes) == len(cfg.parameter_names) == 11
    assert cfg.offsets[0] == 0 and cfg.offsets[-1] == cfg.n_total
    assert cfg.profile == 'mojolearn.byte-lm.b1-l4-d32-h4-kv2-hd8-ff64-v32-blocks1.fp32.v3'
    same = dataclasses.replace(cfg)
    assert cfg == same and hash(cfg) == hash(tuple(cfg.to_dict().values()))
    assert cfg == pickle.loads(pickle.dumps(cfg))
    assert repr(cfg) == 'ByteLanguageModelConfig(batch=1, length=4, d_model=32, n_heads=4, n_kv=2, head_dim=8, intermediate=64, n_layers=1, vocab_size=32, chunked_lm_head_v2=False)'
    assert cfg != object()
    assert audit_native_call(require_shape, cfg).assert_native() is cfg
    assert audit_native_call(state_shape, {'model_shape': cfg.to_dict()}).assert_native() == cfg
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.batch = 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        del cfg.batch
    for options in ({'head_dim': 7}, {'batch': True}, {'length': 8193}, {'batch': 1 << 21},
                    {'batch': 1 << 20, 'length': 8192}, {'chunked_lm_head_v2': 1}):
        with pytest.raises((TypeError, ValueError)):
            Config(**options)
    calls = []
    class IndexHook:
        def __index__(self):
            calls.append(True)
            return 1
    with pytest.raises((TypeError, ValueError)):
        Config(batch=IndexHook())
    assert not calls
