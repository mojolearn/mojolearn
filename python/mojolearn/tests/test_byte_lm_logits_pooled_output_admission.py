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
