# SPDX-License-Identifier: Apache-2.0
"""The streamed binary byte-LM checkpoint: round trip, bit exactness, refusals.

Host only. No model, GPU kernel, build or benchmark runs here; the native
binding is the same mock the rest of the byte-LM wrapper tests use, and no
test in this file completes a training step. What is asserted is that the
FILE carries the state, which is a storage claim and not a learning,
performance or cross-vendor identity claim.

THE POINT OF THE SHAPE IN `large()`. Its flat count is over the 87,381
parameters at which the JSON/hex envelope's 2 MiB bound stops, so the two
formats are exercised on the SAME state at a size where one refuses and the
other does not. A round-trip test at the tiny default shape alone would pass
just as well if the bound had never been the problem.
"""
import hashlib
import struct

import pytest

from mojolearn import SmallByteLanguageModelTrainer
from mojolearn import _byte_lm_checkpoint as storage
from mojolearn._byte_lm_config import ByteLanguageModelConfig

# `host` is a fixture; pytest resolves it by name in this module's namespace.
from mojolearn.tests.test_byte_lm_surface import host, ids, initial, trainer  # noqa: F401

SCHEDULE = {'corpus_sha256': '1' * 64, 'token_schedule_sha256': '2' * 64,
            'batch_offsets': [0, 32], 'planned_steps': 4}


def large():
    """A shape whose hex envelope exceeds 2 MiB (see the module docstring)."""
    return ByteLanguageModelConfig(batch=1, length=8, d_model=64, n_heads=4,
                                   n_kv=2, head_dim=16, intermediate=128,
                                   n_layers=2, vocab_size=512)


def large_trainer():
    """Deliberately NON-UNIFORM parameters. An all-zero vector hides every
    single-cell corruption: a sabotage that zeroed one element round-tripped
    clean against it and the test still passed."""
    shape = large()
    from mojolearn import _buffer
    from mojolearn._bufcheck import flat_view
    values = _buffer.zeros(shape.n_total, '<f4')
    view = flat_view(values, 'f')
    state = 0x2545F491
    for index in range(shape.n_total):
        state = (1103515245 * state + 12345) & 0xFFFFFFFF
        view[index] = float(state % 2003) * 9.765625e-4 - 1.0
    return (SmallByteLanguageModelTrainer(values, data_schedule=SCHEDULE, shape=shape), shape)


def arrays(state):
    return {key: state[key].tobytes() for key in ('parameters', 'm', 'v', 'flags')}


def metadata(state):
    return {key: value for key, value in state.items()
            if key not in ('parameters', 'm', 'v', 'flags')}


def test_round_trip_is_byte_for_byte_at_a_shape_the_json_envelope_refuses(host, tmp_path):
    model, shape = large_trainer()
    assert shape.n_total * 24 > 2 * 1024 * 1024, 'the shape must exceed the JSON bound'
    with pytest.raises(ValueError, match='2 MiB'):
        model.export_checkpoint(tmp_path / 'refused.json')

    path = tmp_path / 'state.byte-lm.bin'
    digest = model.export_checkpoint_binary(path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    # Raw bytes, not hex: 12 per parameter against the envelope's 24, plus
    # the four-byte flags and a header well under the 1 MiB bound.
    payload = 12 * shape.n_total + 4 * shape.n_tensors
    assert payload < path.stat().st_size <= payload + storage.HEADER_LIMIT

    restored = SmallByteLanguageModelTrainer.from_checkpoint_binary(path)
    before, after = model.state_dict(), restored.state_dict()
    assert arrays(after) == arrays(before)
    assert metadata(after) == metadata(before)


def test_the_moments_and_the_step_cursor_cross_the_file_not_just_the_parameters(host, tmp_path):
    """A restore that dropped `m`/`v` would still match on `parameters`."""
    model = trainer()
    model.train_step(ids())
    state = model.state_dict()
    assert state['completed_steps'] == 1
    assert any(state['m'].tobytes()[i] for i in range(len(state['m'].tobytes())))
    assert any(state['v'].tobytes()[i] for i in range(len(state['v'].tobytes())))

    path = tmp_path / 'after-one-step.bin'
    model.export_checkpoint_binary(path)
    restored = SmallByteLanguageModelTrainer.from_checkpoint_binary(path)
    after = restored.state_dict()
    assert after['m'].tobytes() == state['m'].tobytes()
    assert after['v'].tobytes() == state['v'].tobytes()
    assert after['flags'].tobytes() == state['flags'].tobytes()
    assert after['completed_steps'] == after['next_batch_index'] == 1


def test_hostile_float_bit_patterns_survive_the_file(host, tmp_path):
    """Decimal text would lose these; raw little-endian bytes do not."""
    from mojolearn import _buffer
    from mojolearn._bufcheck import flat_view
    values = _buffer.zeros(34944, '<f4')
    view = flat_view(values, 'f')
    hostile = struct.unpack('<8f', struct.pack(
        '<8I',
        0x80000000,  # -0.0
        0x00000001,  # smallest positive subnormal
        0x80000001,  # smallest negative subnormal
        0x00800000,  # smallest positive normal
        0x3F7FFFFF,  # 1.0 - 1 ulp
        0xBF800000,  # -1.0
        0x007FFFFF,  # largest subnormal
        0x33D6BF95,  # an arbitrary odd mantissa
    ))
    for index, value in enumerate(hostile):
        view[index] = value
    model = SmallByteLanguageModelTrainer(values, data_schedule=SCHEDULE)
    path = tmp_path / 'hostile.bin'
    model.export_checkpoint_binary(path)
    restored = SmallByteLanguageModelTrainer.from_checkpoint_binary(path)
    saved = model.state_dict()['parameters'].tobytes()
    assert restored.state_dict()['parameters'].tobytes() == saved
    assert saved[:32] == struct.pack(
        '<8I', 0x80000000, 0x00000001, 0x80000001, 0x00800000,
        0x3F7FFFFF, 0xBF800000, 0x007FFFFF, 0x33D6BF95)


def test_two_saves_of_one_state_are_identical_files(host, tmp_path):
    model = trainer()
    first, second = tmp_path / 'a.bin', tmp_path / 'b.bin'
    assert model.export_checkpoint_binary(first) == model.export_checkpoint_binary(second)
    assert first.read_bytes() == second.read_bytes()


def written(tmp_path, name='state.bin'):
    path = tmp_path / name
    trainer().export_checkpoint_binary(path)
    return path, path.read_bytes()


def test_refuses_a_foreign_magic(host, tmp_path):
    path, raw = written(tmp_path)
    path.write_bytes(b'MOJOLEARN-SAMBA\x02\n' + raw[len(storage.MAGIC):])
    with pytest.raises(ValueError, match='magic'):
        SmallByteLanguageModelTrainer.from_checkpoint_binary(path)


def test_refuses_a_tampered_header(host, tmp_path):
    path, raw = written(tmp_path)
    start = len(storage.MAGIC) + 8
    length = struct.unpack('<Q', raw[len(storage.MAGIC):start])[0]
    header = raw[start:start + length]
    edited = header.replace(b'"completed_steps":0', b'"completed_steps":7')
    assert edited != header and len(edited) == len(header)
    path.write_bytes(raw[:start] + edited + raw[start + length:])
    with pytest.raises(ValueError, match='header integrity'):
        SmallByteLanguageModelTrainer.from_checkpoint_binary(path)


def test_refuses_a_flipped_payload_byte(host, tmp_path):
    path, raw = written(tmp_path)
    index = len(raw) - 40
    path.write_bytes(raw[:index] + bytes([raw[index] ^ 0x01]) + raw[index + 1:])
    with pytest.raises(ValueError, match='integrity mismatch'):
        SmallByteLanguageModelTrainer.from_checkpoint_binary(path)


def test_refuses_truncation_and_trailing_bytes(host, tmp_path):
    path, raw = written(tmp_path)
    path.write_bytes(raw[:-4])
    with pytest.raises(ValueError, match='truncated or trailing'):
        SmallByteLanguageModelTrainer.from_checkpoint_binary(path)
    path.write_bytes(raw + b'\x00')
    with pytest.raises(ValueError, match='truncated or trailing'):
        SmallByteLanguageModelTrainer.from_checkpoint_binary(path)


def test_an_array_length_comes_from_the_registry_not_from_the_file(host, tmp_path):
    """A descriptor claiming a huge array must be refused BEFORE allocation."""
    path, raw = written(tmp_path)
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


def test_refuses_a_header_length_past_the_bound(host, tmp_path):
    path, raw = written(tmp_path)
    path.write_bytes(raw[:len(storage.MAGIC)] + struct.pack('<Q', storage.HEADER_LIMIT + 1)
                     + raw[len(storage.MAGIC) + 8:])
    with pytest.raises(ValueError, match='header length'):
        SmallByteLanguageModelTrainer.from_checkpoint_binary(path)


def test_refuses_a_state_whose_registry_does_not_match_its_shape(host, tmp_path):
    """The regex pins the CODEC's spelling. `_validate_state` refuses the same
    field with 'Byte-LM state profile/registry/mode mismatch', so matching the
    shared half would pass with this check deleted -- which is what the
    sabotage arm showed before the word 'checkpoint' was added to it."""
    path, raw = written(tmp_path)
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
