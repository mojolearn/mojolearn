# SPDX-License-Identifier: Apache-2.0
"""NumPy is the independent oracle; product calls run with its imports blocked."""
import builtins
import hashlib
import json
from contextlib import contextmanager

import numpy as np
import pytest

from mojolearn import Array, RBFSampler, lm_corpus, parallel_classical


@contextmanager
def no_numpy():
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split('.')[0] == 'numpy':
            raise AssertionError('product attempted to import NumPy')
        return original(name, *args, **kwargs)

    builtins.__import__ = guarded
    try:
        yield
    finally:
        builtins.__import__ = original


def test_token_payload_and_mapped_batches_match_independent_oracle(tmp_path):
    data = b'abc\ndefgh\nijkl\nmnop\n'
    ranges = {'train_range': [0, 14], 'validation_range': [14, len(data)]}
    identity = {'schema': 'test-vocabulary', 'sha256': 'a' * 64, 'n_vocab': 1024}

    class Tokenizer:
        def encode_batch(self, documents):
            return [[value + 256 for value in doc] for doc in documents]

    oracle = np.asarray([value + 256 for value in data], dtype='<i4')
    with no_numpy():
        lm_corpus._tokenize(data, {'sha256': hashlib.sha256(data).hexdigest()},
                           ranges, Tokenizer(), identity, tmp_path, 5, 2)
        batches = lm_corpus.TokenBatches(tmp_path, batch=3, length=3)
        actual = [batches.ids(step) for step in (0, 1, 9, 1000001)]
    assert (tmp_path / 'tokens.i32').read_bytes() == oracle.tobytes()
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['train_range'] == [0, 14]
    assert manifest['ids_above_255'] == len(data)
    assert manifest['max_id'] == int(oracle.max())
    assert batches.ids_all.flags['WRITEABLE'] is False
    for step, result in zip((0, 1, 9, 1000001), actual):
        starts = [(step * 3 * 3 + b * 3) % (14 - 3 - 1) for b in range(3)]
        expected = np.stack([oracle[start:start + 4] for start in starts])
        assert isinstance(result, Array)
        assert result.shape == expected.shape
        assert result.tobytes() == expected.tobytes()
    # Returned batches own storage; later reads cannot overwrite earlier batches.
    assert actual[0].tobytes() == np.stack([oracle[s:s + 4] for s in (0, 3, 6)]).tobytes()


def test_token_batch_prefetch_is_ordered_bounded_and_propagates(monkeypatch, tmp_path):
    data = b'abcdefghijklmnopqrstuvwxyz\n'
    ranges = {'train_range': [0, len(data)]}
    identity = {'schema': 'test-vocabulary', 'sha256': 'b' * 64, 'n_vocab': 1024}

    class Tokenizer:
        def encode_batch(self, documents):
            return [[value + 256 for value in doc] for doc in documents]

    lm_corpus._tokenize(data, {'sha256': hashlib.sha256(data).hexdigest()},
                       ranges, Tokenizer(), identity, tmp_path, 8, 2)
    batches = lm_corpus.TokenBatches(tmp_path, batch=2, length=4)
    expected = [(step, batches.ids(step).tobytes()) for step in range(3, 10)]
    actual = [(step, ids.tobytes()) for step, ids in batches.prefetch(3, 7, depth=2)]
    assert actual == expected
    for args in ((-1, 1, 2), (0, -1, 2), (0, 1, 0), (0, 1, True)):
        with pytest.raises(ValueError):
            list(batches.prefetch(args[0], args[1], depth=args[2]))

    original = batches.ids
    def fail_at_five(step):
        if step == 5:
            raise RuntimeError('prefetch-step-five')
        return original(step)
    monkeypatch.setattr(batches, 'ids', fail_at_five)
    with pytest.raises(RuntimeError, match='prefetch-step-five'):
        list(batches.prefetch(3, 7, depth=2))


@pytest.mark.parametrize('shard_rows', [1, 3, 20])
@pytest.mark.parametrize('fault', [None, 'width', 'missing', 'worker'])
def test_parallel_rbf_copy_order_and_cleanup(monkeypatch, shard_rows, fault):
    expected = np.arange(35, dtype='<f4').reshape(7, 5)[::-1]
    x = Array.from_list(expected.tolist(), '<f4')
    model = RBFSampler(n_components=5)
    model.n_features_in_ = 5
    model.random_weights_ = Array((5, 5), '<f4')
    seen, closed = [], []

    class Pool:
        def __init__(self, devices):
            pass

        def map(self, requests):
            parts = []
            for operation, estimator, (shard,) in requests:
                assert operation == 'rbf_sampler_rows' and estimator is model
                seen.append(shard)
                parts.append(shard.copy())
            if fault == 'worker':
                raise RuntimeError('test worker failure')
            if fault == 'width':
                parts[0] = Array((1, 4), '<f4')
            if fault == 'missing':
                parts.pop()
            return parts

        def close(self):
            closed.append(True)

    monkeypatch.setattr(parallel_classical, 'DevicePool', Pool)
    with no_numpy():
        if fault:
            with pytest.raises(RuntimeError if fault == 'worker' else ValueError):
                parallel_classical.transform_rbf_sampler(model, x, rows_per_shard=shard_rows)
        else:
            result = parallel_classical.transform_rbf_sampler(model, x, rows_per_shard=shard_rows)
            assert isinstance(result, Array)
            assert result.tobytes() == expected.tobytes()
    assert closed == [True]
    assert b''.join(shard.tobytes() for shard in seen) == expected.tobytes()
