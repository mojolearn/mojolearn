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


def _byte_vocabulary():
    from mojolearn.tokenizer import BpeTokenizer
    try:
        return BpeTokenizer.from_token_bytes([bytes([b]) for b in range(256)])
    except ImportError as exc:
        pytest.skip(f"tokenizer host binding not built: {exc}")


def test_token_payload_and_mapped_batches_match_independent_oracle(tmp_path):
    data = b'abc\ndefgh\nijkl\nmnop\n'
    ranges = {'train_range': [0, 14], 'validation_range': [14, len(data)]}
    identity = {'schema': 'test-vocabulary', 'sha256': 'a' * 64, 'n_vocab': 1024}

    # The byte vocabulary: every id is its byte, so the payload is the bytes.
    tok = _byte_vocabulary()
    oracle = np.asarray(list(data), dtype='<i4')
    with no_numpy():
        lm_corpus._tokenize(data, {'sha256': hashlib.sha256(data).hexdigest()},
                           ranges, tok, identity, tmp_path, 5, 2)
        batches = lm_corpus.TokenBatches(tmp_path, batch=3, length=3)
        actual = [batches.ids(step) for step in (0, 1, 9, 1000001)]
    assert (tmp_path / 'tokens.i32').read_bytes() == oracle.tobytes()
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['train_range'] == [0, 14]
    assert manifest['ids_above_255'] == 0
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

    lm_corpus._tokenize(data, {'sha256': hashlib.sha256(data).hexdigest()},
                       ranges, _byte_vocabulary(), identity, tmp_path, 8, 2)
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
@pytest.mark.parametrize('fault', [None, 'drop_result', 'reverse_results'])
def test_parallel_rbf_copy_order_and_cleanup(monkeypatch, shard_rows, fault):
    # transform_rbf_sampler is one native call (bindings/parallel_shard_native.mojo);
    # it needs no NumPy, joins row blocks in input order, and its checks or
    # the bytes catch a dropped or reordered block (the transport negative
    # control, bindings/parallel_shard_common.mojo `shard_map`).
    from mojolearn._native_parallel_shard import _native_parallel_shard
    rows = np.arange(35, dtype='<f4').reshape(7, 5)[::-1]
    x = Array.from_list(rows.tolist(), '<f4')
    from mojolearn._cpu_reference import reference_training
    with reference_training():
        model = RBFSampler(n_components=5, random_state=3).fit(x)
    expected = model.transform(x)
    if fault:
        monkeypatch.setenv('MOJOLEARN_PAR_TRANSPORT_FAULT', fault)
        monkeypatch.setenv('MOJOLEARN_HOST_ALLOW_SABOTAGE', '1')
    _native_parallel_shard('faults_take', ())
    with no_numpy():
        if fault == 'drop_result' and len(range(0, 7, shard_rows)) > 1:
            with pytest.raises(ValueError):
                parallel_classical.transform_rbf_sampler(model, x, rows_per_shard=shard_rows)
        else:
            result = parallel_classical.transform_rbf_sampler(model, x, rows_per_shard=shard_rows)
            assert isinstance(result, Array)
            if fault == 'reverse_results' and len(range(0, 7, shard_rows)) > 1:
                assert result.tobytes() != expected.tobytes()
            else:
                assert result.tobytes() == expected.tobytes()
    assert _native_parallel_shard('faults_take', ()) == (1 if fault and len(range(0, 7, shard_rows)) > 1 else 0)
