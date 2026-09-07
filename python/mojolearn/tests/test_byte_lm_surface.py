# SPDX-License-Identifier: Apache-2.0
"""Root-only host mocks for the authored byte-LM wrapper; no numerical claim.

No model, GPU kernel, build, benchmark or independent numerical reference is
executed by these tests. Native two-block learning/gradient/identity/resume
qualification is a separate root-only remote CUDA/HIP task.
"""
import ctypes
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from mojolearn import SmallByteLanguageModelTrainer
from mojolearn import _byte_lm_impl as impl


def buffer(address, count, integer=False):
    kind = ctypes.c_int32 if integer else ctypes.c_float
    return np.ctypeslib.as_array((kind * count).from_address(address))


def initial():
    return np.zeros(34944, dtype=np.float32)


def ids():
    return (np.arange(66, dtype=np.int32) % 256).reshape(2, 33)


def trainer():
    return SmallByteLanguageModelTrainer(initial(), data_schedule={
        'corpus_sha256': '1' * 64, 'token_schedule_sha256': '2' * 64,
        'batch_offsets': [0, 32], 'planned_steps': 4})


def state_digest(model):
    state = model.state_dict()
    for key in ('parameters', 'm', 'v', 'flags'):
        # DEVIATION 2460: state buffers are mojolearn.Array; the typestr read
        # through np.asarray is the same '<f4' / '<u1' string as before.
        value = np.asarray(state[key])
        state[key] = {'dtype': value.dtype.str, 'shape': value.shape,
                      'hex': value.tobytes().hex()}
    return hashlib.sha256(impl._canonical(state)).hexdigest()


class FakeByteLM:
    def __init__(self):
        self.mode = 1
        self.vendor = 'cuda'
        self.profile = impl.PROFILE
        self.calls = []
        self.fail_after_write = False
        self.nonfinite_after_write = False
        self.modify_input = False
        self.eval_mutation = None
        self.wrong_step = False
        self.leave_gradient_unwritten = False

    def byte_lm_numeric_mode(self):
        return self.mode

    def byte_lm_vendor(self):
        return self.vendor

    def byte_lm_profile(self):
        return self.profile

    def byte_lm_run(self, addresses, params):
        assert len(addresses) == 11
        assert len(params) == 12
        assert params[2] == 2
        assert params[8:] == [0., 0., 0, 0.]
        self.calls.append((list(addresses), list(params)))
        # Copy all input state to fresh output buffers, then deliberately
        # write before injecting a failure. These are plumbing sentinels,
        # not an implementation or numerical reference for a language model.
        for source, target in zip(addresses[:3], addresses[5:8]):
            assert source != target
            buffer(target, 34944)[:] = buffer(source, 34944)
        buffer(addresses[9], 20, True)[:] = buffer(addresses[3], 20, True)
        buffer(addresses[10], 1)[0] = 1.25
        if params[0] == 1:
            buffer(addresses[5], 34944)[:] += 1
            buffer(addresses[6], 34944)[:] += 2
            buffer(addresses[7], 34944)[:] += 3
            buffer(addresses[9], 20, True)[:] = 1
            assert addresses[8] > 0
            if not self.leave_gradient_unwritten:
                buffer(addresses[8], 34944)[:] = .125
        else:
            assert addresses[8] == 0
            if self.eval_mutation in ('parameters', 'm', 'v'):
                index = {'parameters': 5, 'm': 6, 'v': 7}[self.eval_mutation]
                # +0.0 and -0.0 compare equal numerically; raw invariance
                # must still detect this mutation on the initial state.
                buffer(addresses[index], 34944)[0] = -0.0
            elif self.eval_mutation == 'flags':
                buffer(addresses[9], 20, True)[0] = 1
        if self.modify_input:
            buffer(addresses[0], 34944)[0] = 77
            buffer(addresses[4], 66, True)[0] = 99
        if self.nonfinite_after_write:
            buffer(addresses[7], 34944)[0] = np.nan
        if self.fail_after_write:
            raise RuntimeError('injected byte-LM failure after writes')
        return params[1] + params[0] + int(self.wrong_step)


@pytest.fixture
def host(monkeypatch):
    fake = FakeByteLM()
    monkeypatch.setattr(impl._backend, 'default_mode', lambda: 'identical')
    monkeypatch.setattr(impl._backend, 'numeric_mode', lambda: 'identical')
    def binding(name, mode):
        assert name == '_mojolearn_byte_lm' and mode == 'identical'
        return fake
    monkeypatch.setattr(impl._backend, 'binding', binding)
    monkeypatch.setattr(impl, '_binding_metadata', lambda value: {
        'binding_file': 'fake-byte-lm.so', 'binding_sha256': '0' * 64,
        'native_profile': value.profile, 'native_vendor': value.vendor,
        'native_numeric_mode': value.mode, 'source_sha256': {'fake': '3' * 64}})
    return fake


def test_registry_and_named_parameter_packing(host):
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
        actual = model.parameters_[entry['offset']:entry['offset'] + entry['size']]
        np.testing.assert_array_equal(actual, expected)
    supplied['embed'][:] = 999
    assert not np.any(model.parameters_[:8192] == 999)
    del supplied['lm_head']
    with pytest.raises(ValueError, match='registry'):
        SmallByteLanguageModelTrainer(supplied, data_schedule={'dataset': 'test'})
    assert host.calls == []


def test_train_copies_inputs_and_returns_all_preupdate_gradients(host):
    source = initial()
    model = SmallByteLanguageModelTrainer(source, data_schedule={'dataset': 'test'})
    source[:] = 999
    tokens = ids()
    tokens.setflags(write=False)
    before = tokens.tobytes()
    result = model.train_step(tokens)
    assert result['loss'] == 1.25
    assert result['step'] == result['completed_steps'] == result['next_batch_index'] == 1
    assert list(result['gradients']) == list(impl.PARAMETER_NAMES)
    assert result['flat_gradients'].shape == (34944,)
    for name, shape in zip(impl.PARAMETER_NAMES, impl.PARAMETER_SHAPES):
        assert result['gradients'][name].shape == shape
        np.testing.assert_array_equal(result['gradients'][name], np.full(shape, .125, np.float32))
    assert model.step_ == 1
    np.testing.assert_array_equal(model.parameters_, np.ones(34944, np.float32))
    assert tokens.tobytes() == before
    # DEVIATION 2460: returned gradients are mojolearn.Array; np.asarray is
    # a zero-copy writable view, so the isolation check is the same as before.
    np.asarray(result['flat_gradients'])[:] = 900
    np.asarray(result['gradients']['embed'])[:] = 800
    np.testing.assert_array_equal(model.parameters_, np.ones(34944, np.float32))


def test_evaluate_preserves_all_state_and_has_no_gradient_output(host):
    model = trainer()
    model.train_step(ids())
    before = state_digest(model)
    assert model.evaluate(ids()) == 1.25
    assert state_digest(model) == before
    addresses, params = host.calls[-1]
    assert params[0] == 0 and params[1] == 1 and addresses[8] == 0


@pytest.mark.parametrize('field', ['parameters', 'm', 'v', 'flags'])
def test_evaluate_detects_even_signed_zero_fullstate_mutation(host, field):
    model = trainer()
    before = state_digest(model)
    host.eval_mutation = field
    with pytest.raises(RuntimeError, match='evaluation changed'):
        model.evaluate(ids())
    assert state_digest(model) == before


@pytest.mark.parametrize('flag', ['fail_after_write', 'nonfinite_after_write',
                                 'modify_input', 'wrong_step', 'leave_gradient_unwritten'])
def test_native_failure_and_partial_writes_never_commit(host, flag):
    model = trainer()
    model.train_step(ids())
    before = state_digest(model)
    tokens = ids()
    token_bytes = tokens.tobytes()
    setattr(host, flag, True)
    with pytest.raises((ValueError, RuntimeError)):
        model.train_step(tokens)
    assert state_digest(model) == before
    assert tokens.tobytes() == token_bytes
    assert model.step_ == 1


def test_state_load_snapshot_isolation_and_cursor_validation(host):
    model = trainer()
    model.train_step(ids())
    state = model.state_dict()
    clone = trainer().load_state_dict(state)
    assert state_digest(model) == state_digest(clone)
    np.asarray(state['parameters'])[:] = 7  # DEVIATION 2460: zero-copy views
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


def test_mode_refuses_without_overriding_process(host, monkeypatch):
    model = trainer()
    before = state_digest(model)
    monkeypatch.setattr(impl._backend, 'default_mode', lambda: 'fast')
    with pytest.raises(RuntimeError, match='process-selected'):
        model.train_step(ids())
    with pytest.raises(RuntimeError, match='process-selected'):
        trainer()
    assert host.calls == []
    assert state_digest(model) == before


def test_checkpoint_bytes_survive_source_buffer_and_path_replacement(host, tmp_path, monkeypatch):
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
    assert host.calls == []


def test_checkpoint_file_api_delegates_the_exact_bounded_capture(host, tmp_path, monkeypatch):
    path = tmp_path / 'source.json'
    trainer().save_checkpoint(path)
    raw = path.read_bytes()
    seen = []
    sentinel = object()
    def decode(cls, encoded):
        seen.append(encoded)
        return sentinel
    monkeypatch.setattr(SmallByteLanguageModelTrainer, 'from_checkpoint_bytes', classmethod(decode))
    assert SmallByteLanguageModelTrainer.from_checkpoint(path) is sentinel
    assert seen == [raw] and type(seen[0]) is bytes


@pytest.mark.parametrize('encoded', [bytearray(b'{}'), memoryview(b'{}'), '{}', None])
def test_checkpoint_bytes_reject_mutable_or_implicit_conversion(host, encoded):
    with pytest.raises(TypeError, match='immutable bytes'):
        SmallByteLanguageModelTrainer.from_checkpoint_bytes(encoded)
    assert host.calls == []


def test_checkpoint_bytes_limit_precedes_json_parsing(host, monkeypatch):
    def forbidden_parse(*args, **kwargs):
        raise AssertionError('oversized capture reached JSON parser')
    monkeypatch.setattr(impl.json, 'loads', forbidden_parse)
    with pytest.raises(ValueError, match='2 MiB'):
        SmallByteLanguageModelTrainer.from_checkpoint_bytes(b' ' * (impl._CHECKPOINT_LIMIT + 1))
    assert host.calls == []


@pytest.mark.parametrize('encoded', [b'', b'{}', b'{"schema":1,"schema":2}',
                                   b'{"schema":"wrong","payload":{},"payload_sha256":"0"}'])
def test_checkpoint_bytes_keep_strict_json_and_schema_admission(host, encoded):
    with pytest.raises(ValueError):
        SmallByteLanguageModelTrainer.from_checkpoint_bytes(encoded)
    assert host.calls == []


@pytest.mark.parametrize('field,value', [('mode', 0), ('vendor', 'none'), ('profile', 'different-profile')])
def test_native_mode_vendor_profile_witnesses_refuse_before_model(host, field, value):
    model = trainer()
    setattr(host, field, value)
    with pytest.raises(RuntimeError, match='native profile'):
        model.evaluate(ids())
    assert host.calls == []


def test_metal_witness_is_admitted_without_weakening_profile_or_mode(host):
    host.vendor = 'metal'
    assert impl._load() is host
    host.profile = 'wrong-profile'
    with pytest.raises(RuntimeError, match='native profile'):
        impl._load()
    assert host.calls == []


def test_tokens_require_exact_int32_shape_and_range_before_gpu(host):
    model = trainer()
    candidates = [ids().astype(np.int64), ids().astype(np.float32), ids().ravel(),
                  ids()[:, :-1], np.full((2, 33), -1, np.int32),
                  np.full((2, 33), 256, np.int32)]
    for value in candidates:
        with pytest.raises((TypeError, ValueError)):
            model.train_step(value)
    assert host.calls == []


@pytest.mark.parametrize('options', [dict(lr=0), dict(eps=0), dict(lr=float('nan')),
                                    dict(weight_decay=-1), dict(betas=(.9, 1)),
                                    dict(eps=1e-100), dict(betas=(True, .9))])
def test_configuration_refuses_before_gpu(host, options):
    with pytest.raises(ValueError):
        SmallByteLanguageModelTrainer(initial(), data_schedule={'dataset': 'test'}, **options)
    assert host.calls == []


def test_bad_state_refuses_transactionally_before_gpu(host):
    model = trainer()
    before = state_digest(model)
    for mutate in (lambda state: np.asarray(state['v']).__setitem__(0, -1),  # DEVIATION 2460
                   lambda state: np.asarray(state['flags']).__setitem__(0, 2),
                   lambda state: state['config'].__setitem__('max_norm', 1),
                   lambda state: state.__setitem__('m', np.zeros(1, np.float32)),
                   lambda state: state.__setitem__('completed_steps', True)):
        state = model.state_dict()
        mutate(state)
        with pytest.raises(ValueError):
            model.load_state_dict(state)
        assert state_digest(model) == before
    assert host.calls == []


def test_checkpoint_canonical_complete_and_resume(host, tmp_path):
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


def test_checkpoint_refuses_corruption_wrong_schema_and_excess_size(host, tmp_path):
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


def test_per_state_lock_serializes_steps(host):
    model = trainer()
    tokens = ids()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(model.train_step, tokens) for _ in range(2)]
        assert sorted(future.result()['step'] for future in futures) == [1, 2]
    assert model.step_ == 2


def test_terminal_state_can_evaluate_but_cannot_train(host):
    model = trainer()
    terminal = model.state_dict()
    terminal['completed_steps'] = terminal['next_batch_index'] = 999999
    model.load_state_dict(terminal)
    before = state_digest(model)
    assert model.evaluate(ids()) == 1.25
    assert state_digest(model) == before
    with pytest.raises(ValueError, match='exhausted'):
        model.train_step(ids())
    assert len(host.calls) == 1


def test_run_metadata_retains_binding_source_profile_dataset_and_config(host):
    model = trainer()
    metadata = model.run_metadata()
    assert metadata['profile'] == metadata['native_profile'] == impl.PROFILE
    assert metadata['native_vendor'] == 'cuda' and metadata['native_numeric_mode'] == 1
    assert metadata['binding_sha256'] == '0' * 64 and metadata['source_sha256']
    assert metadata['config'] == model.state_dict()['config']
    assert metadata['data_schedule']['corpus_sha256'] == '1' * 64
    metadata['data_schedule']['batch_offsets'][0] = 999
    assert model.run_metadata()['data_schedule']['batch_offsets'][0] == 0
    assert host.calls == []
