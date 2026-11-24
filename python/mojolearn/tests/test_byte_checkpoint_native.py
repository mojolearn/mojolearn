# SPDX-License-Identifier: Apache-2.0
"""Independent v1 archive encoding against the native save/load (the refusals are in test_byte_lm_boundary.py)."""
import hashlib
import json
import struct

import numpy as np
import pytest

from mojolearn import _byte_lm_checkpoint as storage
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_parallel_byte_native import state_fixture


def fixture(large=False):
    state, _ = state_fixture()
    if large:
        from mojolearn import ByteLanguageModelConfig
        shape = ByteLanguageModelConfig(batch=1, length=4, d_model=128, n_heads=4,
                                       n_kv=2, head_dim=32, intermediate=512,
                                       n_layers=1, vocab_size=512)
        state.update(profile=shape.profile, model_shape=shape.to_dict(),
                     parameter_names=list(shape.parameter_names),
                     parameter_shapes=[list(value) for value in shape.parameter_shapes],
                     parameter_offsets=list(shape.offsets),
                     parameters=np.zeros(shape.n_total, np.float32),
                     m=np.zeros(shape.n_total, np.float32), v=np.zeros(shape.n_total, np.float32),
                     flags=np.zeros(shape.n_tensors, np.int32))
        assert state['parameters'].nbytes > storage.CHUNK
    words = np.array([0, 0x80000000, 1, 0x80000001, 0x7fc12345, 0xffc23456,
                      0x7f800000, 0xff800000], np.uint32)
    for offset, name in enumerate(('parameters', 'm', 'v')):
        state[name].view(np.uint32)[:len(words)] = np.roll(words, offset)
    state['flags'][:] = np.arange(len(state['flags']), dtype=np.int32) % 2
    state['completed_steps'] = state['next_batch_index'] = 7
    return state


def independent_archive(state):
    metadata = {key: value for key, value in state.items() if key not in dict(storage.ARRAYS)}
    payloads = [state[name].tobytes() for name, _ in storage.ARRAYS]
    entries = [dict(name=name, dtype=dtype, shape=[len(state[name])], nbytes=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest())
               for (name, dtype), raw in zip(storage.ARRAYS, payloads)]
    header = json.dumps(dict(schema=storage.SCHEMA, state=metadata, arrays=entries),
                        sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                        allow_nan=False).encode('ascii')
    return storage.MAGIC + struct.pack('<Q', len(header)) + header + hashlib.sha256(header).digest() + b''.join(payloads)


@pytest.mark.parametrize('large', [False, True])
def test_native_archive_matches_independent_existing_format_and_raw_words(tmp_path, large):
    state = fixture(large)
    expected = independent_archive(state)
    path = tmp_path / 'state.blm'
    digest = audit_native_call(storage.save, path, state).assert_native()
    assert path.read_bytes() == expected
    assert digest == hashlib.sha256(expected).hexdigest()
    # Read an independently written fixture, rather than trusting round trip.
    oracle_path = tmp_path / 'independent.blm'
    oracle_path.write_bytes(expected)
    restored = audit_native_call(storage.load, oracle_path).assert_native()
    for name, _ in storage.ARRAYS:
        assert np.asarray(restored[name]).tobytes() == state[name].tobytes()
    for key in state.keys() - dict(storage.ARRAYS).keys():
        assert restored[key] == state[key]
