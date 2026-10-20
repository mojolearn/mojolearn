# SPDX-License-Identifier: Apache-2.0
"""Actual native layer owners, independent arithmetic and complete child audits.

These small correctness fixtures are not performance or physical multi-GPU
qualification. CUDA/HIP inventory is required before claiming device isolation.
"""
import hashlib
import json
import os
import signal
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import mojolearn
from mojolearn import Array, _backend
from mojolearn._causal_lm_fixtures import family_fixture, _llama_config, _llama_tensors, _write_checkpoint
from mojolearn._verify_causal_lm import state_digest
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.models import CausalLM
from mojolearn.models import parallel_causal_lm as mod


AUDIT_FIELDS = ('python_frames', 'forbidden_native_calls', 'wrapper_work',
                'unobserved_existing_threads')


def checked(operation, *args, **kwargs):
    return audit_native_call(operation, *args, **kwargs).assert_native()


def refused(operation, exception, match, *args, **kwargs):
    audit = audit_native_call(operation, *args, **kwargs)
    assert isinstance(audit.exception, exception), audit.report()
    assert match in str(audit.exception), audit.report()
    for key in AUDIT_FIELDS:
        assert not audit.report()[key], audit.report()
    return audit.exception


def owners():
    default = '0,1' if _backend.vendor() == 'cpu' else '0,0'
    return tuple(map(int, os.environ.get('MOJOLEARN_PARALLEL_CAUSAL_DEVICES', default).split(',')))


def bits(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape and a.dtype == b.dtype
    np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))


def child_module(tmp_path, monkeypatch):
    (tmp_path / 'native_causal_audit_child.py').write_text('''
import json, os, sys
from pathlib import Path
root = Path(os.environ['NATIVE_CAUSAL_AUDIT_DIRECTORY'])
sys.stdout = (root / (str(os.getpid()) + '.stdout.log')).open('w', buffering=1)
sys.stderr = (root / (str(os.getpid()) + '.stderr.log')).open('w', buffering=1)
from mojolearn import _backend
from mojolearn.models.parallel_causal_lm import _native_parallel_causal
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn._gpu_witness import visible_gpu_inventory, worker_process_inventory
vendor = _backend.vendor()
# Test-only device inventory is outside the complete compiled worker loop.
inventory = visible_gpu_inventory(vendor) if vendor in ('cuda', 'hip') else worker_process_inventory()
audit = audit_native_call(_native_parallel_causal, 'worker', sys.argv)
report = audit.report()
report['inventory'] = inventory
report['numeric_mode'] = _backend.default_mode()
Path(root, str(os.getpid()) + '.json').write_text(json.dumps(report, indent=2))
audit.assert_native()
''')
    monkeypatch.setenv('NATIVE_CAUSAL_AUDIT_DIRECTORY', str(tmp_path))
    package = str(Path(mojolearn.__file__).resolve().parent.parent)
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join((str(tmp_path), package, os.environ.get('PYTHONPATH', ''))))
    return 'native_causal_audit_child'


@contextmanager
def actual_owner(tmp_path, monkeypatch, checkpoint, *, profile='fp32_v1', weight_format='float32'):
    module = child_module(tmp_path, monkeypatch)
    model = None
    pids = []
    try:
        model = checked(mod._native_parallel_causal, 'load_with_worker',
                        (mod.ParallelCausalLM, str(checkpoint), owners(),
                         (weight_format, None, profile), module), mod._CAUSAL_CONTEXT)
        pids = [w['pid'] for w in model._pool['workers']]
        assert len(set(pids)) == len(set(owners()))
        assert all(pid != os.getpid() for pid in pids)
        assert model._native_numeric_mode == 1
        assert model._native_vendor == _backend.vendor()
        yield model
    finally:
        primary = sys.exception()
        cleanup = None
        if model is not None:
            try:
                checked(model.close)
            except BaseException as error:
                cleanup = error
        if primary is not None:
            primary.add_note('Native causal child evidence: ' + str(tmp_path))
            for path in sorted(tmp_path.glob('*.stderr.log')):
                primary.add_note(str(path) + '\n' + '\n'.join(path.read_text().splitlines()[-12:]))
            if cleanup is not None:
                primary.add_note('Native child cleanup failed: ' + repr(cleanup))
        elif cleanup is not None:
            raise cleanup
    reports = []
    for pid in pids:
        with pytest.raises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)
        path = tmp_path / (str(pid) + '.json')
        assert path.exists(), 'Complete child audit is missing: ' + str(path)
        report = json.loads(path.read_text())
        assert report['exception'] is None and report['numeric_mode'] == 'identical', report
        for key in AUDIT_FIELDS:
            assert not report[key], report
        reports.append(report)
    if _backend.vendor() in ('cuda', 'hip'):
        from mojolearn._gpu_witness import require_distinct_workers
        require_distinct_workers([r['inventory'] for r in reports], _backend.vendor(), len(set(owners())))


def checkpoint(tmp_path, family='llama'):
    cfg, tensors = family_fixture(family, False)
    root = tmp_path / 'checkpoint'
    _write_checkpoint(root, cfg, tensors)
    return root, cfg, tensors


def local_model(root, profile, weight_format='float32'):
    route = 'cpu' if _backend.vendor() == 'cpu' else 'gpu'
    return CausalLM.load(root, device=route, numeric_profile=profile, weight_format=weight_format)


def block_reference(model, ids, state=None, *, step=False):
    """Independent test driver through retained block primitives, not CausalLM.forward."""
    b, length = ids.shape
    x = model._prims.embedding(model._embed, ids.reshape((b * length,)))
    x = x.reshape((b, length, model.d_model))
    for i, block in enumerate(model.blocks):
        x = block._call(x, None if state is None else state.layers[i], step=step)
    x = x.reshape((b * length, model.d_model))
    if model._norm is not None:
        x = model._prims.rms_norm(x, model._norm, model.norm_eps)
    result = (model._prims.linear(x, model._head) if model._head_int15 is None else
              model._prims.linear_int15(x, model._head_int15))
    return np.asarray(result).reshape(b, length, model.vocab_size)


@pytest.mark.parametrize('devices', [(0,), (), (0, -1), (0, True)])
def test_invalid_map_precedes_model_or_worker_construction(devices):
    refused(mod.ParallelCausalLM, ValueError, 'per layer',
            SimpleNamespace(n_layers=2), {}, layer_devices=devices)


def test_native_route_admission_precedes_checkpoint_read(monkeypatch):
    selected = _backend.vendor()
    assert checked(mod._admit_route, (0, 0)) == ('cpu' if selected == 'cpu' else 'gpu')
    refused(mod.ParallelCausalLM.load, FileNotFoundError, 'checkpoint directory',
            '/does/not/exist', layer_devices=(0, 0))
    if selected == 'metal':
        refused(mod.ParallelCausalLM.load, ValueError, 'device 0',
                '/does/not/exist', layer_devices=(0, 1))
    # Native connection metadata is injected only by this refusal fixture.
    with monkeypatch.context() as patch:
        patch.setitem(mod._CAUSAL_CONTEXT, 'selected_vendor', 'unsupported-vendor')
        refused(mod.ParallelCausalLM.load, NotImplementedError, 'CUDA or HIP',
                '/does/not/exist', layer_devices=(0, 0))


@pytest.mark.parametrize('family,profile', [('llama', None), ('llama', 'fp32_v1'),
                                           ('mamba', None), ('mamba2', None)])
def test_native_owners_match_independent_blocks_and_continue_states(tmp_path, monkeypatch, family, profile):
    root, cfg, tensors = checkpoint(tmp_path, family)
    plain = local_model(root, profile)
    ids = Array.from_list([[1, 3, 7], [2, 4, 8]], '<i4')
    expected = block_reference(plain, ids)
    with actual_owner(tmp_path, monkeypatch, root, profile=profile) as split:
        assert split.numeric_profile == plain.numeric_profile
        bits(checked(split.forward, ids), expected)
        reference_state = plain.allocate_state(2, 12)
        state = checked(split.allocate_state, 2, 12)
        bits(checked(split.forward, ids, state), block_reference(plain, ids, reference_state))
        token = Array.from_list([[5], [6]], '<i4')
        bits(checked(split.step, token, state), block_reference(plain, token, reference_state, step=True)[:, 0])
        assert state.positions == 4
        assert state_digest(state) == state_digest(reference_state)
        checked(split.reset_state, state)
        assert state.positions == 0
        bits(checked(split.forward, ids, state), expected)
        # Greedy choices are independently consumed by the retained block driver.
        greedy_state = plain.allocate_state(2, 7)
        logits = block_reference(plain, ids, greedy_state)
        continuations = []
        for i in range(4):
            next_ids = logits[:, -1].argmax(axis=1).astype('i4')
            continuations.append(next_ids)
            if i != 3:
                logits = block_reference(plain, Array.from_buffer(next_ids[:, None]), greedy_state, step=True)
        want = np.concatenate((np.asarray(ids), np.stack(continuations, axis=1)), axis=1)
        bits(checked(split.generate, ids, 4), want)
        actual_params = checked(split.parameters)
        expected_params = plain.parameters()
        assert set(actual_params) == set(expected_params)
        for key in actual_params:
            bits(actual_params[key], expected_params[key])
        # Each worker genuinely lacks unassigned layers. A model error keeps
        # its process and sequence alive; later good requests still work.
        pids = [w['pid'] for w in split._pool['workers']]
        if len(set(owners())) > 1:
            refused(split._rpc, RuntimeError, 'worker failed', owners()[0], 'parameters', 1)
        bits(checked(split.forward, ids), expected)
        assert [w['pid'] for w in split._pool['workers']] == pids
        blocks = split.blocks
        remote = state.layers[0]
        refused(blocks[1].step, ValueError, 'another layer', np.zeros((2, 1, plain.d_model), 'f4'), remote)
        checked(remote.close)
        checked(remote.close)
        refused(remote.snapshot, ValueError, 'released')
        # Reconstruct a stream on the other topology and continue exactly.
        stream = checked(split.stream, ids, 5, greedy=False, seed=29, top_k=7)
        checked(stream.read, 2)
        saved = checked(stream.state_dict)
        local_replay = checked(plain.resume_generation, saved)
        remote_replay = checked(split.resume_generation, saved)
        tail = checked(stream.read, 10)
        bits(checked(local_replay.read, 10), tail)
        bits(checked(remote_replay.read, 10), tail)
        reverse = checked(plain.stream, ids, 5, greedy=False, seed=31, top_k=5)
        checked(reverse.read, 1)
        from_local = checked(split.resume_generation, checked(reverse.state_dict))
        bits(checked(from_local.read, 10), checked(reverse.read, 10))


@pytest.mark.parametrize('weight_format', ['bfloat16', 'int8'])
def test_packed_checkpoint_weights_keep_native_owner_semantics(tmp_path, monkeypatch, weight_format):
    root, _, _ = checkpoint(tmp_path)
    plain = local_model(root, 'fp32_v1', weight_format)
    ids = Array.from_list([[1, 3, 7]], '<i4')
    with actual_owner(tmp_path, monkeypatch, root, weight_format=weight_format) as split:
        bits(checked(split.forward, ids), block_reference(plain, ids))
        assert all(block.weight_format == weight_format for block in split.blocks)
        for key, value in checked(split.parameters).items():
            bits(value, plain.parameters()[key])


def test_analytic_residual_head_oracle_and_ephemeral_cleanup(tmp_path, monkeypatch):
    cfg = _llama_config()
    tensors = _llama_tensors(cfg)
    d, v = cfg['hidden_size'], cfg['vocab_size']
    embedding = np.arange(1, v*d+1, dtype='f4').reshape(v, d) / np.float32(v*d)
    head = np.zeros((v, d), 'f4')
    head[:d] = np.eye(d, dtype='f4')
    tensors['model.embed_tokens.weight'] = ('F32', embedding.shape, embedding.tobytes())
    tensors['lm_head.weight'] = ('F32', head.shape, head.tobytes())
    for name, (dtype, shape, raw) in list(tensors.items()):
        if name.startswith('model.layers.') and len(shape) == 2:
            tensors[name] = (dtype, shape, bytes(len(raw)))
    root = tmp_path / 'checkpoint'
    _write_checkpoint(root, cfg, tensors)
    ids = np.array([[1, 4], [2, 5]], dtype='i4')
    hidden = embedding[ids].astype(np.float64)
    normalized = hidden / np.sqrt(np.mean(hidden*hidden, axis=-1, keepdims=True) + cfg['rms_norm_eps'])
    expected = normalized @ head.astype(np.float64).T
    with actual_owner(tmp_path, monkeypatch, root) as split:
        np.testing.assert_allclose(np.asarray(checked(split.forward, ids)), expected, rtol=3e-6, atol=2e-7)
        # Stateless forward allocated token0 in each owner and released it.
        # A valid error frame must not poison the process that owns the model.
        for device in set(owners()):
            refused(split._rpc, RuntimeError, 'state released', device, 'state', 0)
        checked(split.generate, ids, 2)
        assert checked(split.generate, ids, 0).shape == ids.shape


def checkpoint_hash(saved):
    """Independent test encoder for the published generation-v1 integrity record."""
    digest = hashlib.sha256()
    def scalar(value):
        digest.update((str(value) + '\n').encode())
    for name in ('schema', 'model', 'options', 'maximum', 'produced', 'positions',
                 'lengths', 'finished', 'text', 'tokenizer'):
        scalar(saved[name])
    digest.update(saved['pending_bytes'])
    for name in ('prompt', 'current'):
        digest.update(np.asarray(saved[name]).tobytes())
    for layer in saved['layers']:
        for name in sorted(layer):
            scalar(name)
            value = layer[name]
            if isinstance(value, Array):
                scalar(value.shape)
                scalar(value.dtype)
                digest.update(np.asarray(value).tobytes())
            else:
                scalar(value)
    return digest.hexdigest()


def test_invalid_restore_releases_new_states_preserves_live_stream(tmp_path, monkeypatch):
    root, _, _ = checkpoint(tmp_path)
    with actual_owner(tmp_path, monkeypatch, root) as split:
        ids = np.array([[1, 2]], 'i4')
        stream = checked(split.stream, ids, 5, greedy=False, seed=9)
        checked(stream.read, 1)
        saved = checked(stream.state_dict)
        assert checkpoint_hash(saved) == saved['checksum']
        plain = local_model(root, 'fp32_v1')
        replay = checked(plain.resume_generation, saved)
        # Shape corruption in layer1 occurs after newly allocated layer0 was
        # populated. Recompute integrity to exercise semantic validation.
        bad = dict(saved)
        bad['layers'] = [dict(layer) for layer in saved['layers']]
        bad['layers'][1]['k_cache'] = Array.from_list([0.], '<f4')
        bad['checksum'] = checkpoint_hash(bad)
        refused(split.resume_generation, ValueError, 'shape or dtype', bad)
        # Monotonic worker tokens identify exactly the failed restore's new
        # states. Existing stream uses token0 (and token1 for one shared owner).
        failed_tokens = (1,) if len(set(owners())) == 2 else (2, 3)
        for device in set(owners()):
            for token in failed_tokens:
                refused(split._rpc, RuntimeError, 'state released', device, 'state', token)
        bits(checked(stream.read, 10), checked(replay.read, 10))
        refused(split.stream, ValueError, 'seed', ids, 3, greedy=False)
        # Native metadata and process identity remain usable after refusals.
        assert checked(split.parameters)


def test_uniform_sampling_eos_has_independent_integer_oracle(tmp_path, monkeypatch):
    cfg = _llama_config(vocab_size=256)
    tensors = _llama_tensors(cfg)
    dtype, shape, raw = tensors['lm_head.weight']
    tensors['lm_head.weight'] = (dtype, shape, bytes(len(raw)))
    root = tmp_path / 'checkpoint'
    _write_checkpoint(root, cfg, tensors)
    with actual_owner(tmp_path, monkeypatch, root) as split:
        prompt = np.array([[33], [34]], 'i4')
        options = dict(greedy=False, seed=0, top_k=4)
        expected = [[3, 1, 0, 3, 0, 1, 0, 3], [1, 0, 2, 2, 0, 0, 3, 0]]
        assert np.asarray(checked(split.generate, prompt, 8, **options))[:, 1:].tolist() == expected
        stream = checked(split.stream, prompt, 8, **options)
        chunks = [np.asarray(checked(stream.read, n)) for n in (1, 3, 2, 9)]
        assert np.concatenate(chunks, axis=1).tolist() == expected
        stopped = checked(split.stream, prompt, 8, eos_token_id=3, pad_token_id=9, **options)
        assert np.asarray(checked(stopped.read, 20)).tolist() == [[3, 9, 9, 9, 9, 9, 9], [1, 0, 2, 2, 0, 0, 3]]
        assert checked(stopped.info)['lengths'] == [1, 7]


def test_public_load_and_transport_failure_refuse_lost_state(tmp_path):
    root, _, _ = checkpoint(tmp_path)
    split = checked(mod.ParallelCausalLM.load, root, layer_devices=owners(), numeric_profile='fp32_v1')
    pids = [w['pid'] for w in split._pool['workers']]
    try:
        ids = np.array([[1, 2]], 'i4')
        state = checked(split.allocate_state, 1, 5)
        checked(split.forward, ids, state)
        os.kill(pids[0], signal.SIGKILL)  # Test-only fault injection.
        audit = audit_native_call(split.step, np.array([3], 'i4'), state)
        assert audit.exception is not None
        for key in AUDIT_FIELDS:
            assert not audit.report()[key], audit.report()
        assert split._failed and not split._pool['workers']
        refused(split.forward, RuntimeError, 'worker state was lost', ids)
    finally:
        checked(split.close)
        checked(split.close)
    for pid in pids:
        with pytest.raises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)
    refused(split.forward, RuntimeError, 'closed', np.array([[1]], 'i4'))


def test_apple_fast_parent_preserves_existing_identical_worker_contract(tmp_path):
    if _backend.vendor() != 'metal':
        pytest.skip('fast is Apple only; this checks the existing Metal parent route')
    import subprocess
    cfg = _llama_config()
    tensors = _llama_tensors(cfg)
    dtype, shape, raw = tensors['lm_head.weight']
    tensors['lm_head.weight'] = (dtype, shape, bytes(len(raw)))
    root = tmp_path / 'checkpoint'
    _write_checkpoint(root, cfg, tensors)
    script = tmp_path / 'fast-parent.py'
    script.write_text('''
import json, sys
import numpy as np
from pathlib import Path
from mojolearn import _backend
from mojolearn.models import ParallelCausalLM
from mojolearn._verify_native_runtime import audit_native_call
assert _backend.vendor() == 'metal' and _backend.default_mode() == 'fast'
reports = {}
model = None
try:
    audit = audit_native_call(ParallelCausalLM.load, sys.argv[1], layer_devices=(0, 0), numeric_profile='fp32_v1')
    model = audit.result
    reports['load'] = audit.report()
    audit.assert_native()
    assert model._native_numeric_mode == 1
    audit = audit_native_call(model.forward, np.array([[1, 2]], 'i4'))
    reports['forward'] = audit.report()
    result = audit.assert_native()
    assert not np.count_nonzero(np.asarray(result))
finally:
    if model is not None:
        audit = audit_native_call(model.close)
        reports['close'] = audit.report()
    Path(sys.argv[2]).write_text(json.dumps(reports, indent=2))
    if model is not None:
        audit.assert_native()
''')
    env = dict(os.environ, MOJOLEARN_NUMERIC_MODE='fast', MOJOLEARN_VENDOR='metal')
    env['PYTHONPATH'] = str(Path(mojolearn.__file__).resolve().parent.parent) + os.pathsep + env.get('PYTHONPATH', '')
    stdout, stderr = tmp_path / 'fast-parent.stdout.log', tmp_path / 'fast-parent.stderr.log'
    with stdout.open('w') as out, stderr.open('w') as err:
        result = subprocess.run([sys.executable, str(script), str(root), str(tmp_path / 'fast-parent.json')],
                                env=env, stdout=out, stderr=err, timeout=180)
    assert result.returncode == 0, str(stderr) + '\n' + '\n'.join(stderr.read_text().splitlines()[-16:])
