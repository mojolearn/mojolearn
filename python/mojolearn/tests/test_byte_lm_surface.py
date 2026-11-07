# SPDX-License-Identifier: Apache-2.0
"""The public byte-LM trainer surface against the installed compiled binding.

The native trainer owner admits only the exact compiled module (it refuses a
Python stand-in with "native provider requires an exact compiled module"),
so these tests construct, train, evaluate, save and restore through the real
binding at the default shape. They assert copies, refusals, transactional
state and checkpoint bytes, not model quality; learning, gradient and
cross-vendor identity qualification are separate.

DELETED (2026-11-07) with the retired Python owner mock, because each needed
a stand-in binding to inject a fault the compiled module cannot produce:
- test_evaluate_refused_on_a_signed_zero_fullstate_mutation_commits_nothing
  (an evaluate entry that rewrote state bits) and
  test_native_failure_and_partial_writes_never_commit (a step that failed after
  writing, wrote a NaN, changed its input, returned the wrong step or left the
  gradient unwritten). The transactional rule they guarded is native
  (bindings/byte_lm_trainer_native.mojo: results commit only after the step's
  checks); the real refusals that reach it are tested below
  (test_tokens_require_exact_int32_shape_and_range,
  test_bad_state_refuses_transactionally). No real-binding test injects a
  device failure: a coverage gap, reported.
- test_checkpoint_file_api_delegates_the_exact_bounded_capture: it replaced
  the Python classmethod `from_checkpoint_bytes`, which `from_checkpoint` no
  longer calls (both are one native call); the same bytes reaching both is
  tested in test_checkpoint_file_and_bytes_restore_the_same_state.
"""
import ctypes
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from mojolearn import SmallByteLanguageModelTrainer
from mojolearn import _byte_lm_impl as impl


def buffer(address, count, integer=False):
    kind = ctypes.c_int32 if integer else ctypes.c_float
    return np.ctypeslib.as_array((kind * count).from_address(address))


def initial():
    # Nonzero parameters: a real step at all-zero parameters has a zero
    # gradient and leaves both moments zero.
    return (np.random.default_rng(23).standard_normal(34944) * .02).astype(np.float32)


def ids():
    return (np.arange(66, dtype=np.int32) % 256).reshape(2, 33)


SCHEDULE = {'corpus_sha256': '1' * 64, 'token_schedule_sha256': '2' * 64,
            'batch_offsets': [0, 32], 'planned_steps': 4}


def trainer(**options):
    return SmallByteLanguageModelTrainer(initial(), data_schedule=dict(SCHEDULE, batch_offsets=[0, 32]),
                                         **options)


def state_digest(model):
    state = model.state_dict()
    for key in ('parameters', 'm', 'v', 'flags'):
        value = np.asarray(state[key])
        state[key] = {'dtype': value.dtype.str, 'shape': value.shape,
                      'hex': value.tobytes().hex()}
    return hashlib.sha256(impl._canonical(state)).hexdigest()


def test_registry_and_named_parameter_packing():
    registry = SmallByteLanguageModelTrainer.parameter_registry()
    assert len(registry) == 20
    assert registry[0] == {'name': 'embed', 'shape': (256, 32), 'offset': 0, 'size': 8192}
    assert registry[-1]['name'] == 'lm_head'
    assert registry[-1]['offset'] + registry[-1]['size'] == 34944
    supplied = {entry['name']: np.full(entry['shape'], index, np.float32)
                for index, entry in enumerate(reversed(registry))}
    model = SmallByteLanguageModelTrainer(supplied, data_schedule={'dataset': 'test'})
    for entry in registry:
        expected = supplied[entry['name']].ravel()
        actual = np.asarray(model.parameters_)[entry['offset']:entry['offset'] + entry['size']]
        np.testing.assert_array_equal(actual, expected)
    supplied['embed'][:] = 999
    assert not np.any(np.asarray(model.parameters_)[:8192] == 999)
    del supplied['lm_head']
    with pytest.raises(ValueError, match='registry'):
        SmallByteLanguageModelTrainer(supplied, data_schedule={'dataset': 'test'})


def test_train_copies_inputs_and_returns_all_preupdate_gradients():
    source = initial()
    model = SmallByteLanguageModelTrainer(source, data_schedule={'dataset': 'test'}, step_result='full')
    twin = SmallByteLanguageModelTrainer(initial(), data_schedule={'dataset': 'test'}, step_result='full')
    source[:] = 999
    tokens = ids()
    tokens.setflags(write=False)
    before = tokens.tobytes()
    result = model.train_step(tokens)
    expected = twin.train_step(ids())
    assert math.isfinite(result['loss']) and result['loss'] == expected['loss']
    assert result['step'] == result['completed_steps'] == result['next_batch_index'] == 1
    assert list(result['gradients']) == list(impl.PARAMETER_NAMES)
    assert tuple(np.asarray(result['flat_gradients']).shape) == (34944,)
    for name, shape in zip(impl.PARAMETER_NAMES, impl.PARAMETER_SHAPES):
        gradient = np.asarray(result['gradients'][name])
        assert gradient.shape == shape and np.isfinite(gradient).all()
    assert model.step_ == 1
    parameters = np.asarray(model.parameters_).copy()
    assert parameters.tobytes() == np.asarray(twin.parameters_).tobytes()
    assert tokens.tobytes() == before
    # Returned gradients are copies: writing them cannot reach the state.
    np.asarray(result['flat_gradients'])[:] = 900
    np.asarray(result['gradients']['embed'])[:] = 800
    assert np.asarray(model.parameters_).tobytes() == parameters.tobytes()


def test_evaluate_preserves_all_state_and_has_no_gradient_output():
    model = trainer()
    model.train_step(ids())
    before = state_digest(model)
    loss = model.evaluate(ids())
    assert math.isfinite(loss)
    assert state_digest(model) == before


def test_state_load_snapshot_isolation_and_cursor_validation():
    model = trainer()
    model.train_step(ids())
    state = model.state_dict()
    clone = trainer().load_state_dict(state)
    assert state_digest(model) == state_digest(clone)
    np.asarray(state['parameters'])[:] = 7
    np.asarray(state['m'])[:] = 8
    np.asarray(state['v'])[:] = 9
    np.asarray(state['flags'])[:] = 0
    state['data_schedule']['batch_offsets'][0] = 99
    assert state_digest(model) == state_digest(clone)
    invalid = clone.state_dict()
    invalid['next_batch_index'] += 1
    before = state_digest(clone)
    with pytest.raises(ValueError, match='cursor'):
        clone.load_state_dict(invalid)
    assert state_digest(clone) == before


def test_mode_refuses_without_overriding_process(monkeypatch):
    # The native owner reads the backend's resolved `_DEFAULT_MODE`
    # (bindings/byte_lm_trainer_native.mojo `_mode`); a value that is not a
    # tier is refused, never replaced by a default.
    model = trainer()
    before = state_digest(model)
    monkeypatch.setattr(impl._backend, '_DEFAULT_MODE', 'unselected')
    with pytest.raises(RuntimeError, match='process-selected'):
        model.train_step(ids())
    with pytest.raises(RuntimeError, match='process-selected'):
        trainer()
    assert state_digest(model) == before


def test_checkpoint_bytes_survive_source_buffer_and_path_replacement(tmp_path, monkeypatch):
    model = trainer()
    path = tmp_path / 'source.json'
    model.save_checkpoint(path)
    mutable = bytearray(path.read_bytes())
    captured = bytes(mutable)
    digest = hashlib.sha256(captured).hexdigest()
    mutable[:] = b'x' * len(mutable)
    path.write_bytes(b'{"changed":true}')
    def forbidden_open(*args, **kwargs):
        raise AssertionError('immutable loader reopened a checkpoint path')
    monkeypatch.setattr(impl.Path, 'open', forbidden_open)
    restored = SmallByteLanguageModelTrainer.from_checkpoint_bytes(captured)
    assert state_digest(restored) == state_digest(model)
    assert hashlib.sha256(captured).hexdigest() == digest


@pytest.mark.parametrize('resident', [False, True])
def test_checkpoint_file_and_bytes_restore_the_same_state(tmp_path, resident):
    model = trainer()
    model.train_step(ids())
    path = tmp_path / 'source.json'
    model.save_checkpoint(path)
    from_file = SmallByteLanguageModelTrainer.from_checkpoint(path, resident=resident)
    from_bytes = SmallByteLanguageModelTrainer.from_checkpoint_bytes(path.read_bytes(), resident=resident)
    assert state_digest(from_file) == state_digest(from_bytes) == state_digest(model)


@pytest.mark.parametrize('encoded', [bytearray(b'{}'), memoryview(b'{}'), '{}', None])
def test_checkpoint_bytes_reject_mutable_or_implicit_conversion(encoded):
    with pytest.raises(TypeError, match='immutable bytes'):
        SmallByteLanguageModelTrainer.from_checkpoint_bytes(encoded)


def test_checkpoint_bytes_limit_precedes_json_parsing(monkeypatch):
    def forbidden_parse(*args, **kwargs):
        raise AssertionError('oversized capture reached JSON parser')
    monkeypatch.setattr(impl.json, 'loads', forbidden_parse)
    with pytest.raises(ValueError, match='2 MiB'):
        SmallByteLanguageModelTrainer.from_checkpoint_bytes(b' ' * (impl._CHECKPOINT_LIMIT + 1))


@pytest.mark.parametrize('encoded', [b'', b'{}', b'{"schema":1,"schema":2}',
                                   b'{"schema":"wrong","payload":{},"payload_sha256":"0"}'])
def test_checkpoint_bytes_keep_strict_json_and_schema_admission(encoded):
    with pytest.raises(ValueError):
        SmallByteLanguageModelTrainer.from_checkpoint_bytes(encoded)


class _Witness:
    """The four witness entries `_byte_lm_impl._load` reads, and nothing
    else: `_load` is Python admission over them, so a stand-in for exactly
    those reads is enough. The native trainer owner never sees it."""

    def __init__(self, vendor):
        self.mode = 1
        self.vendor = vendor
        self.profile = impl.PROFILE

    def byte_lm_numeric_mode(self):
        return self.mode

    def byte_lm_vendor(self):
        return self.vendor

    def byte_lm_profile(self):
        return self.profile

    def byte_lm_arithmetic_suffix(self):
        return ''

    def byte_lm_run(self, *args):
        raise AssertionError('admission ran the binding')


def _witness(monkeypatch, vendor):
    witness = _Witness(vendor)
    monkeypatch.setattr(impl._backend, 'default_mode', lambda: 'identical')
    monkeypatch.setattr(impl._backend, 'numeric_mode', lambda: 'identical')
    monkeypatch.setattr(impl._backend, 'binding', lambda name, mode='identical': witness)
    return witness


@pytest.mark.parametrize('field,value', [('mode', 0), ('vendor', 'none'), ('profile', 'different-profile')])
def test_native_mode_vendor_profile_witnesses_refuse_before_model(monkeypatch, field, value):
    monkeypatch.setattr(impl._backend, '_CPU_ONLY', None)
    witness = _witness(monkeypatch, 'cuda')
    assert impl._load() is witness
    setattr(witness, field, value)
    with pytest.raises(RuntimeError, match='native profile'):
        impl._load()


def test_metal_witness_is_admitted_without_weakening_profile_or_mode(monkeypatch):
    witness = _witness(monkeypatch, 'metal')
    # A GPU install (no CPU-only set loaded) admits the Metal witness ...
    monkeypatch.setattr(impl._backend, '_CPU_ONLY', None)
    assert impl._load() is witness
    # ... a CPU-only install admits only the CPU byte LM binding.
    monkeypatch.setattr(impl._backend, '_CPU_ONLY', 'no GPU set')
    with pytest.raises(RuntimeError, match='native profile'):
        impl._load()
    monkeypatch.setattr(impl._backend, '_CPU_ONLY', None)
    witness.profile = 'wrong-profile'
    with pytest.raises(RuntimeError, match='native profile'):
        impl._load()


def test_tokens_require_exact_int32_shape_and_range():
    model = trainer()
    before = state_digest(model)
    candidates = [ids().astype(np.int64), ids().astype(np.float32), ids().ravel(),
                  ids()[:, :-1], np.full((2, 33), -1, np.int32),
                  np.full((2, 33), 256, np.int32)]
    for value in candidates:
        with pytest.raises((TypeError, ValueError)):
            model.train_step(value)
    assert state_digest(model) == before and model.step_ == 0


@pytest.mark.parametrize('options', [dict(lr=0), dict(eps=0), dict(lr=float('nan')),
                                    dict(weight_decay=-1), dict(betas=(.9, 1)),
                                    dict(eps=1e-100), dict(betas=(True, .9))])
def test_configuration_refuses_before_construction(options):
    with pytest.raises(ValueError):
        SmallByteLanguageModelTrainer(initial(), data_schedule={'dataset': 'test'}, **options)


def test_bad_state_refuses_transactionally():
    model = trainer()
    before = state_digest(model)
    for mutate in (lambda state: np.asarray(state['v']).__setitem__(0, -1),
                   lambda state: np.asarray(state['flags']).__setitem__(0, 2),
                   lambda state: state['config'].__setitem__('max_norm', 1),
                   lambda state: state.__setitem__('m', np.zeros(1, np.float32)),
                   lambda state: state.__setitem__('completed_steps', True)):
        state = model.state_dict()
        mutate(state)
        with pytest.raises(ValueError):
            model.load_state_dict(state)
        assert state_digest(model) == before


def test_checkpoint_canonical_complete_and_resume(tmp_path):
    model = trainer()
    model.train_step(ids())
    first, second = tmp_path / 'first.byte-lm.json', tmp_path / 'second.byte-lm.json'
    model.save_checkpoint(first)
    model.save_checkpoint(second)
    assert first.read_bytes() == second.read_bytes()
    assert len(first.read_bytes()) <= 2 * 1024 * 1024
    restored = SmallByteLanguageModelTrainer.from_checkpoint(first)
    assert state_digest(restored) == state_digest(model)
    restored.train_step(ids())
    model.train_step(ids())
    assert state_digest(restored) == state_digest(model)


def test_checkpoint_refuses_corruption_wrong_schema_and_excess_size(tmp_path):
    path = tmp_path / 'state.json'
    trainer().save_checkpoint(path)
    original = path.read_bytes()
    envelope = json.loads(original)
    envelope['payload']['completed_steps'] = 5
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match='integrity'):
        SmallByteLanguageModelTrainer.from_checkpoint(path)
    envelope = json.loads(original)
    envelope['schema'] = 'native-binary-v1'
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match='schema'):
        SmallByteLanguageModelTrainer.from_checkpoint(path)
    envelope = json.loads(original)
    envelope['payload']['m']['shape'] = [1000000000]
    envelope['payload_sha256'] = hashlib.sha256(impl._canonical(envelope['payload'])).hexdigest()
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match='descriptor'):
        SmallByteLanguageModelTrainer.from_checkpoint(path)
    path.write_bytes(b' ' * (impl._CHECKPOINT_LIMIT + 1))
    with pytest.raises(ValueError, match='2 MiB'):
        SmallByteLanguageModelTrainer.from_checkpoint(path)


def test_per_state_lock_serializes_steps():
    model = trainer()
    tokens = ids()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(model.train_step, tokens) for _ in range(2)]
        assert sorted(future.result()['step'] for future in futures) == [1, 2]
    assert model.step_ == 2


def test_terminal_state_can_evaluate_but_cannot_train():
    model = trainer()
    terminal = model.state_dict()
    terminal['completed_steps'] = terminal['next_batch_index'] = 999999
    model.load_state_dict(terminal)
    before = state_digest(model)
    assert math.isfinite(model.evaluate(ids()))
    assert state_digest(model) == before
    with pytest.raises(ValueError, match='exhausted'):
        model.train_step(ids())
    assert state_digest(model) == before


def test_run_metadata_retains_binding_source_profile_dataset_and_config():
    model = trainer()
    metadata = model.run_metadata()
    assert metadata['profile'] == metadata['native_profile'] == impl.PROFILE
    assert metadata['native_vendor'] in ('cpu', 'cuda', 'hip', 'metal')
    assert metadata['native_numeric_mode'] == 1
    sha = metadata['binding_sha256']
    assert type(sha) is str and len(sha) == 64 and int(sha, 16) >= 0
    assert metadata['source_sha256']
    assert metadata['config'] == model.state_dict()['config']
    assert metadata['data_schedule']['corpus_sha256'] == '1' * 64
    metadata['data_schedule']['batch_offsets'][0] = 999
    assert model.run_metadata()['data_schedule']['batch_offsets'][0] == 0
