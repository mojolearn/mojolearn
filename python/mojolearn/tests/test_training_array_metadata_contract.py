"""Actual native training-copy metadata and NumPy ownership contracts.

Requires installed bindings. No fake providers or skip-on-missing behavior.
These tests intentionally inspect returned metadata before NumPy conversion.
"""
import gc
import numpy as np
import mojolearn as ml


def _shape():
    return ml.LanguageModelConfig(batch=1, length=5, d_model=16, n_heads=2,
                                  n_kv=1, head_dim=8, intermediate=24,
                                  n_layers=1, vocab_size=17)


def _numpy_view(value):
    assert type(value) is ml.Array
    assert type(value.shape) is tuple
    assert type(value.strides) is tuple
    interface = value.__array_interface__
    assert interface["shape"] is value.shape
    assert type(interface["shape"]) is tuple
    result = np.asarray(value)
    assert result.shape == value.shape
    assert result.dtype.str == value.dtype
    assert result.ctypes.data == value._addr
    assert result.flags.writeable == (not value._readonly)
    return result


def test_strict_training_state_export_load_and_numpy_owner():
    shape = _shape()
    source = np.linspace(-0.01, 0.01, shape.n_total, dtype="f4")
    wanted = source.tobytes()
    trainer = ml.LanguageModelTrainer(source, shape=shape, resident=False,
                                     data_schedule={"dataset": "array-metadata"})
    try:
        source.fill(7)
        state = trainer.state_dict()  # Pre-step export must admit its own Arrays.
        assert _numpy_view(state["parameters"]).tobytes() == wanted
        for name in ("parameters", "m", "v", "flags"):
            _numpy_view(state[name])
        trainer.load_state_dict(state)
        restored = trainer.state_dict()
        for name in ("parameters", "m", "v", "flags"):
            assert _numpy_view(restored[name]).tobytes() == _numpy_view(state[name]).tobytes()
        held = _numpy_view(restored["parameters"])
        expected = held.tobytes()
        del restored, state
    finally:
        trainer.close()
    del trainer
    gc.collect()
    assert held.tobytes() == expected  # NumPy must retain the Array/storage owner.


def test_cast_host_training_copy_keeps_canonical_metadata_and_owned_bytes():
    shape = _shape()
    source = np.linspace(-0.01, 0.01, shape.n_total, dtype="f8")
    expected = source.astype("f4").tobytes()
    trainer = ml.LanguageModelHostTrainer(source, shape=shape)
    source.fill(7)
    result = trainer.parameters_
    held = _numpy_view(result)
    assert held.tobytes() == expected
    for value in (trainer.m_, trainer.v_):
        assert type(value.shape) is tuple
        assert not _numpy_view(value).any()
    del result, trainer
    gc.collect()
    assert held.tobytes() == expected
