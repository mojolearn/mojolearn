# SPDX-License-Identifier: Apache-2.0
"""Independent v1 archive encoding and strict native save/load boundaries."""
import hashlib
import json
import struct
import errno

import numpy as np
import pytest

from mojolearn import _byte_lm_checkpoint as storage
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_parallel_byte_native import state_fixture
from mojolearn.tests.test_cross_vendor_admission_native import PythonHook, refused


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


@pytest.mark.parametrize('damage,match', [('magic', 'magic'), ('header', 'integrity'),
                                        ('array', 'integrity'), ('short', 'truncated or trailing'),
                                        ('trailing', 'truncated or trailing')])
def test_native_archive_refuses_independent_corruption(tmp_path, damage, match):
    raw = bytearray(independent_archive(fixture()))
    if damage == 'magic':
        raw[0] ^= 1
    elif damage == 'header':
        raw[len(storage.MAGIC) + 8] ^= 1
    elif damage == 'array':
        raw[-5] ^= 1
    elif damage == 'short':
        del raw[-4:]
    else:
        raw += b'junk'
    path = tmp_path / 'bad.blm'
    path.write_bytes(raw)
    refused(storage.load, path, match=match)


def test_native_archive_rejects_python_path_buffer_and_metadata_hooks(tmp_path):
    state = fixture()
    refused(storage.save, PythonHook(), state, match='path hooks')
    refused(storage.load, PythonHook(), match='path hooks')
    state['parameters'] = PythonHook()
    refused(storage.save, tmp_path / 'bad.blm', state, match='buffer hooks')
    state = fixture()
    state['config']['lr'] = PythonHook()
    refused(storage.save, tmp_path / 'bad.blm', state, match='Python protocols')
    assert not (tmp_path / 'bad.blm').exists()


@pytest.mark.parametrize('case,kind,number', [
    ('load_missing', FileNotFoundError, errno.ENOENT),
    ('load_directory', IsADirectoryError, errno.EISDIR),
    ('save_missing_parent', FileNotFoundError, errno.ENOENT),
    ('save_non_directory_parent', NotADirectoryError, errno.ENOTDIR),
    ('save_over_directory', IsADirectoryError, errno.EISDIR),
])
def test_native_archive_preserves_os_exception_errno_and_atomic_destination(tmp_path, case, kind, number):
    state = fixture()
    marker = tmp_path / 'marker'
    marker.write_bytes(b'destination must survive')
    if case == 'load_missing':
        target = tmp_path / 'missing.blm'
        operation, args = storage.load, (target,)
    elif case == 'load_directory':
        target = tmp_path
        operation, args = storage.load, (target,)
    elif case == 'save_missing_parent':
        target = tmp_path / 'absent' / 'state.blm'
        operation, args = storage.save, (target, state)
    elif case == 'save_non_directory_parent':
        target = marker / 'state.blm'
        operation, args = storage.save, (target, state)
    else:
        target = tmp_path / 'directory.blm'
        target.mkdir()
        (target / 'retained').write_bytes(b'existing child')
        operation, args = storage.save, (target, state)
    observed = audit_native_call(operation, *args)
    error = observed.exception
    assert type(error) is kind, observed.report()
    assert error.errno == number
    if case.startswith('load'):
        assert error.filename == str(target)
    else:
        assert error.filename.startswith(str(target) + '.tmp-')
    if case == 'save_over_directory':
        assert error.filename2 == str(target)
        assert (target / 'retained').read_bytes() == b'existing child'
    for name in ('python_frames', 'forbidden_native_calls', 'wrapper_work', 'unobserved_existing_threads'):
        assert not observed.report()[name], observed.report()
    assert marker.read_bytes() == b'destination must survive'
    assert list(tmp_path.glob('*.tmp-*')) == []
    # A captured I/O error belongs only to its active native call. It must not
    # poison a later successful archive operation or replace corruption errors.
    valid = tmp_path / 'recovered.blm'
    audit_native_call(storage.save, valid, state).assert_native()
    restored = audit_native_call(storage.load, valid).assert_native()
    assert np.asarray(restored['parameters']).tobytes() == state['parameters'].tobytes()
    valid.write_bytes(b'bad magic')
    refused(storage.load, valid, match='magic')
