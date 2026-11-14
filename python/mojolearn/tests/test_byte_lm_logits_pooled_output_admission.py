"""Boundary test: resident LanguageModelTrainer.logits returns its pooled Array.

The smoke failure 'Array native storage descriptors changed' came from the
shared Array output admission refusing the connected pooled logits subclass.
"""
import numpy as np


def _model():
    from mojolearn import LanguageModelTrainer, LanguageModelConfig
    shape = LanguageModelConfig(batch=2, length=4, d_model=32, n_heads=4,
                                n_kv=2, head_dim=8, intermediate=64,
                                n_layers=1, vocab_size=32)
    parameters = (np.random.default_rng(7).standard_normal(shape.n_total) * .02).astype('f4')
    return LanguageModelTrainer(parameters, shape=shape, resident=True, step_result='lean',
                                data_schedule={'batch_size': 2})


def test_resident_logits_returns_pooled_array_and_reuses_the_pool():
    from mojolearn import Array
    from mojolearn import _byte_lm_impl as impl
    model = _model()
    ids = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    first = model.logits(ids)
    assert type(first) is impl._PooledLogits
    assert isinstance(first, Array)
    assert first.shape == (2, 4, 32) and first.dtype == '<f4'
    values = np.asarray(first).copy()
    assert np.isfinite(values).all()
    del first
    second = model.logits(ids)
    assert type(second) is impl._PooledLogits
    assert np.asarray(second).tobytes() == values.tobytes()
    model.close()


def test_resident_logits_unfilled_store_holds_only_downloaded_bytes():
    # Lane fix-lm-forward-door (2026-11-13): a fresh pooled logits store is an
    # UNFILLED bytearray (bindings/byte_lm_trainer_native_door.mojo
    # BYTE_LM_LOGITS_ZERO_FILL off). Every byte must come from the download:
    # recycle freed allocator blocks of the output's size filled with 0xFF
    # (NaN bit patterns) between calls while the previous result is held (so
    # the pool misses, as on the board), and the bytes must not move.
    from mojolearn import _byte_lm_impl as impl
    model = _model()
    ids = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    first = model.logits(ids)
    want = np.asarray(first).tobytes()
    assert type(first._store) is bytearray and len(first._store) == first.nbytes
    assert first._mv.format == 'f' and first._readonly is False
    # the "f" view is over the store itself (hot-path audit 2026-11-13: one
    # C-API memoryview, its cast through the connect-time descriptor)
    assert first._mv.obj is first._store and first._mv.nbytes == first.nbytes
    ctx = impl._LM_TRAINER_CONTEXT
    assert ctx["byte_lm_view_cast"] is memoryview.cast and ctx["byte_lm_store_type"] is bytearray
    held = first
    for _ in range(3):
        junk = [bytearray(b'\xff' * held.nbytes) for _ in range(4)]
        del junk
        got = model.logits(ids)
        assert type(got) is impl._PooledLogits and got.shape == (2, 4, 32)
        assert got._store is not held._store
        assert np.asarray(got).tobytes() == want
        held = got
    del first, held, got
    again = model.logits(ids)
    assert type(again._store) is bytearray
    assert np.asarray(again).tobytes() == want
    model.close()
