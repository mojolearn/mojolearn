# SPDX-License-Identifier: Apache-2.0
"""The byte language model at the Python boundary: checkpoint refusals and OS errors, buffer and Python-hook
admission, exception words. The numbers the model computes are the byte-lm verify lanes."""
import errno
import hashlib
import struct
import types

import numpy as np
import pytest

from mojolearn import _byte_lm_checkpoint as storage
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_byte_checkpoint_native import fixture, independent_archive
from mojolearn.tests.test_cross_vendor_boundary import PythonHook, refused


# ---------------------------------------------------------------- moved from test_byte_checkpoint_native.py

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


# ---------------------------------------------------------------- moved from test_byte_lm_checkpoint_binary.py

_BINARY_SCHEDULE = {'corpus_sha256': '1' * 64, 'token_schedule_sha256': '2' * 64,
                    'batch_offsets': [0, 32], 'planned_steps': 4}


def _binary_written(tmp_path, name='state.bin'):
    from mojolearn import SmallByteLanguageModelTrainer
    path = tmp_path / name
    SmallByteLanguageModelTrainer(np.zeros(34944, dtype=np.float32),
                                  data_schedule=_BINARY_SCHEDULE).export_checkpoint_binary(path)
    return path, path.read_bytes()


def test_refuses_a_foreign_magic(tmp_path):
    from mojolearn import SmallByteLanguageModelTrainer
    path, raw = _binary_written(tmp_path)
    path.write_bytes(b'MOJOLEARN-SAMBA\x02\n' + raw[len(storage.MAGIC):])
    with pytest.raises(ValueError, match='magic'):
        SmallByteLanguageModelTrainer.from_checkpoint_binary(path)


def test_refuses_a_tampered_header(tmp_path):
    from mojolearn import SmallByteLanguageModelTrainer
    path, raw = _binary_written(tmp_path)
    start = len(storage.MAGIC) + 8
    length = struct.unpack('<Q', raw[len(storage.MAGIC):start])[0]
    header = raw[start:start + length]
    edited = header.replace(b'"completed_steps":0', b'"completed_steps":7')
    assert edited != header and len(edited) == len(header)
    path.write_bytes(raw[:start] + edited + raw[start + length:])
    with pytest.raises(ValueError, match='header integrity'):
        SmallByteLanguageModelTrainer.from_checkpoint_binary(path)


def test_refuses_a_flipped_payload_byte(tmp_path):
    from mojolearn import SmallByteLanguageModelTrainer
    path, raw = _binary_written(tmp_path)
    index = len(raw) - 40
    path.write_bytes(raw[:index] + bytes([raw[index] ^ 0x01]) + raw[index + 1:])
    with pytest.raises(ValueError, match='integrity mismatch'):
        SmallByteLanguageModelTrainer.from_checkpoint_binary(path)


def test_refuses_truncation_and_trailing_bytes(tmp_path):
    from mojolearn import SmallByteLanguageModelTrainer
    path, raw = _binary_written(tmp_path)
    path.write_bytes(raw[:-4])
    with pytest.raises(ValueError, match='truncated or trailing'):
        SmallByteLanguageModelTrainer.from_checkpoint_binary(path)
    path.write_bytes(raw + b'\x00')
    with pytest.raises(ValueError, match='truncated or trailing'):
        SmallByteLanguageModelTrainer.from_checkpoint_binary(path)


def test_an_array_length_comes_from_the_registry_not_from_the_file(tmp_path):
    """A descriptor claiming a huge array must be refused BEFORE allocation."""
    from mojolearn import SmallByteLanguageModelTrainer
    path, raw = _binary_written(tmp_path)
    start = len(storage.MAGIC) + 8
    length = struct.unpack('<Q', raw[len(storage.MAGIC):start])[0]
    header = raw[start:start + length]
    edited = header.replace(b'"shape":[34944]', b'"shape":[99999999]', 1)
    assert edited != header
    rebuilt = (storage.MAGIC + struct.pack('<Q', len(edited)) + edited
               + hashlib.sha256(edited).digest() + raw[start + length + 32:])
    path.write_bytes(rebuilt)
    with pytest.raises(ValueError, match='descriptor mismatch'):
        SmallByteLanguageModelTrainer.from_checkpoint_binary(path)


def test_refuses_a_header_length_past_the_bound(tmp_path):
    from mojolearn import SmallByteLanguageModelTrainer
    path, raw = _binary_written(tmp_path)
    path.write_bytes(raw[:len(storage.MAGIC)] + struct.pack('<Q', storage.HEADER_LIMIT + 1)
                     + raw[len(storage.MAGIC) + 8:])
    with pytest.raises(ValueError, match='header length'):
        SmallByteLanguageModelTrainer.from_checkpoint_binary(path)


def test_refuses_a_state_whose_registry_does_not_match_its_shape(tmp_path):
    """The regex pins the CODEC's spelling. `_validate_state` refuses the same
    field with 'Byte-LM state profile/registry/mode mismatch', so matching the
    shared half would pass with this check deleted -- which is what the
    sabotage arm showed before the word 'checkpoint' was added to it."""
    from mojolearn import SmallByteLanguageModelTrainer
    path, raw = _binary_written(tmp_path)
    start = len(storage.MAGIC) + 8
    length = struct.unpack('<Q', raw[len(storage.MAGIC):start])[0]
    header = raw[start:start + length]
    edited = header.replace(b'"numeric_mode":"identical"', b'"numeric_mode":"fastxxxxx"', 1)
    assert edited != header and len(edited) == len(header)
    rebuilt = (storage.MAGIC + struct.pack('<Q', len(edited)) + edited
               + hashlib.sha256(edited).digest() + raw[start + length + 32:])
    path.write_bytes(rebuilt)
    with pytest.raises(ValueError, match='checkpoint profile/registry/mode'):
        SmallByteLanguageModelTrainer.from_checkpoint_binary(path)


# ---------------------------------------------------------------- moved from test_byte_lm_host_context_contract.py

def _host_shape():
    import mojolearn as ml
    return ml.ByteLanguageModelConfig(batch=1, length=4, d_model=32, n_heads=4,
                                      n_kv=2, head_dim=8, intermediate=64,
                                      n_layers=1, vocab_size=32)


def test_invalid_parameters_keep_native_validation_error():
    import mojolearn as ml
    with pytest.raises(ValueError, match='parameters') as caught:
        ml.LanguageModelInference(np.zeros(3, dtype='f4'), shape=_host_shape())
    assert 'arena_native_types' not in str(caught.value)


def test_user_array_hook_is_not_admitted_by_context_repair():
    import mojolearn as ml
    calls = []
    class UserArray:
        def __array__(self, *args, **kwargs):
            calls.append(True)
            raise AssertionError('user conversion hook executed')
    with pytest.raises(TypeError):
        ml.LanguageModelInference(UserArray(), shape=_host_shape())
    assert calls == []


# ---------------------------------------------------------------- moved from test_byte_lm_host_native_provider.py

def _provider_model():
    from mojolearn._byte_lm_config import ByteLanguageModelConfig
    shape = ByteLanguageModelConfig(batch=1, length=4, d_model=32, n_heads=4,
                                   n_kv=2, head_dim=8, intermediate=64,
                                   n_layers=1, vocab_size=32)
    parameters = (np.random.default_rng(91).standard_normal(shape.n_total) * .02).astype('f4')
    return types.SimpleNamespace(parameters_=parameters, shape=shape)


def test_inference_constructor_native_flat_named_packed_and_atomic_admission():
    from mojolearn._byte_lm_host import LanguageModelInference
    from mojolearn import lowbit
    template = _provider_model()
    shape = template.shape
    flat = np.asarray(template.parameters_).copy()
    named = {name: flat[shape.offsets[i]:shape.offsets[i + 1]].reshape(dims).copy()
             for i, (name, dims) in enumerate(zip(shape.parameter_names, shape.parameter_shapes))}
    model = audit_native_call(LanguageModelInference, named, shape=shape, threads=1).assert_native()
    assert model.weight_format == 'float32' and np.asarray(model._parameters).tobytes() == flat.tobytes()
    for fmt in ('bfloat16', 'int8'):
        packed = lowbit.pack(named, fmt)
        # `unpack` returns the float32 dict and the stored format.
        materialized, stored = lowbit.unpack(packed)
        assert stored == fmt
        expected = np.concatenate([np.asarray(materialized[name]).ravel() for name in shape.parameter_names])
        result = audit_native_call(LanguageModelInference, packed, shape=shape, threads=1).assert_native()
        assert result.weight_format == fmt and np.asarray(result._parameters).tobytes() == expected.tobytes()
    before = dict(model.__dict__)
    broken = dict(named)
    broken['embed'] = np.full_like(named['embed'], np.nan)
    with pytest.raises(ValueError):
        model.__init__(broken, shape=shape)
    assert model.__dict__ == before
    calls = []
    class MappingHook:
        def keys(self):
            calls.append('keys')
            return named.keys()
    with pytest.raises((TypeError, ValueError)):
        LanguageModelInference(MappingHook(), shape=shape)
    assert not calls


def test_host_inference_checkpoint_native_integrity_and_file_errors(tmp_path):
    from mojolearn._byte_lm_host import LanguageModelInference
    from mojolearn._byte_lm_impl import SmallByteLanguageModelTrainer
    template = _provider_model()
    trainer = SmallByteLanguageModelTrainer(template.parameters_, data_schedule={'corpus': 'host-provider-test'},
                                            shape=template.shape, resident=False)
    path = tmp_path / 'state.json'
    trainer.export_checkpoint(path)
    restored = audit_native_call(LanguageModelInference.from_checkpoint, path, threads=1).assert_native()
    assert restored.shape == template.shape
    assert np.asarray(restored._parameters).tobytes() == np.asarray(template.parameters_).tobytes()
    with pytest.raises(FileNotFoundError) as failure:
        LanguageModelInference.from_checkpoint(tmp_path / 'missing.json')
    assert failure.value.errno == 2
    bad = tmp_path / 'bad.json'
    encoded = bytearray(path.read_bytes())
    position = encoded.index(b'payload_sha256') + len(b'payload_sha256') + 3
    encoded[position] = ord('0') if encoded[position] != ord('0') else ord('1')
    bad.write_bytes(encoded)
    with pytest.raises(ValueError, match='integrity'):
        LanguageModelInference.from_checkpoint(bad)
    trainer.close()


# ------------------------------------------- moved from test_byte_lm_native_state_restore.py and
# ------------------------------------------- test_byte_lm_pooled_logits_native.py (the same resident model)

def _resident_model():
    from mojolearn import LanguageModelTrainer
    from mojolearn._byte_lm_config import ByteLanguageModelConfig
    shape = ByteLanguageModelConfig(batch=1, length=4, d_model=32, n_heads=4,
                                   n_kv=2, head_dim=8, intermediate=64,
                                   n_layers=1, vocab_size=32)
    parameters = (np.random.default_rng(81).standard_normal(shape.n_total) * .02).astype('f4')
    return LanguageModelTrainer(parameters, data_schedule={'batch_size': 1},
                                shape=shape, resident=True, step_result='lean')


def _restore_detached(state):
    import copy
    result = {key: copy.deepcopy(value) for key, value in state.items()
              if key not in ('parameters', 'm', 'v', 'flags')}
    for key in ('parameters', 'm', 'v', 'flags'):
        result[key] = np.asarray(state[key]).copy()
    return result


def _restore_same(left, right):
    for key in ('parameters', 'm', 'v', 'flags'):
        a, b = np.asarray(left[key]), np.asarray(right[key])
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), key
    for key in ('completed_steps', 'next_batch_index', 'config', 'data_schedule',
                'parameter_names', 'parameter_shapes', 'parameter_offsets'):
        assert left[key] == right[key], key


def test_restore_strided_int32_flags_and_rejects_python_hooks():
    model = _resident_model()
    saved = _restore_detached(model.export_state())
    flags = np.zeros(saved['flags'].size * 2, dtype='i4')[::2]
    flags[:] = saved['flags']
    saved['flags'] = flags
    model.load_state_dict(saved)
    _restore_same(model.export_state(), saved)
    calls = []
    class IndexHook:
        def __index__(self):
            calls.append('index')
            raise AssertionError('Python scalar hook reached')
    bad = _restore_detached(saved)
    bad['completed_steps'] = IndexHook()
    with pytest.raises(TypeError):
        model.load_state_dict(bad)
    assert calls == []


@pytest.mark.parametrize('name', ['__array_interface__', '_addr'])
def test_replaced_pooled_python_descriptor_is_refused_without_invoking_it(name):
    from mojolearn import _byte_lm_impl as impl
    model = _resident_model()
    ids = np.array([[1, 3, 5, 7]], dtype='i4')
    calls = []
    def read(self):
        calls.append('read')
        raise AssertionError('a pooled Python descriptor executed')
    def write(self, value):
        calls.append('write')
        raise AssertionError('a pooled Python descriptor executed')
    cls = impl._PooledLogits
    missing = object()
    previous = vars(cls).get(name, missing)
    setattr(cls, name, property(read, write))
    try:
        with pytest.raises(TypeError, match='pooled logits storage hook changed'):
            model.logits(ids)
    finally:
        if previous is missing:
            delattr(cls, name)
        else:
            setattr(cls, name, previous)
    assert calls == []
    model.close()


# ---------------------------------------------------------------- moved from test_byte_lm_train_steps.py

def _steps_trainer(seed=0):
    import mojolearn
    shape = mojolearn.LanguageModelConfig()
    rng = np.random.default_rng(seed)
    weights = (rng.standard_normal(shape.n_total) * 0.02).astype(np.float32)
    try:
        return shape, mojolearn.LanguageModelTrainer(weights, data_schedule={"corpus": "train-steps-test"})
    except (ImportError, RuntimeError) as error:
        pytest.skip(f"no byte LM binding: {error}")


def test_train_steps_refuses_a_wrong_shape_and_out_of_vocabulary_ids():
    shape, trainer = _steps_trainer()
    with pytest.raises(ValueError):
        trainer.train_steps(np.zeros((2, shape.batch, shape.length), dtype=np.int32))
    with pytest.raises(ValueError):
        trainer.train_steps(np.zeros((0, shape.batch, shape.length + 1), dtype=np.int32))
    bad = np.random.default_rng(1).integers(0, shape.vocab_size, (2, shape.batch, shape.length + 1),
                                            dtype=np.int32)
    bad[1, 0, 0] = shape.vocab_size
    with pytest.raises(ValueError):
        trainer.train_steps(bad)
    with pytest.raises(TypeError):
        trainer.train_steps(np.zeros((2, shape.batch, shape.length + 1), dtype=np.int64))
