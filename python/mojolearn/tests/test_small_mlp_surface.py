# SPDX-License-Identifier: Apache-2.0
"""Root-only SmallMLP gates. Host mocks never claim numerical correctness.

The opt-in independent-reference gate runs only on remote Linux CUDA/HIP,
with MOJOLEARN_RUN_SMALL_MLP_GPU=1 and process-selected IDENTICAL bindings.
Never execute this suite in a subagent or on Apple hardware.
"""
import ctypes
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from mojolearn._array import Array
from mojolearn import SmallMLPTrainer
from mojolearn import _mlp_impl as impl


def weights():
    return [((np.arange(np.prod(shape), dtype=np.float32) % 7 - 3) / 32).reshape(shape)
            for shape in ((16, 8), (16,), (3, 16), (3,))]


def batch(rows=7):
    x = ((np.arange(rows * 8, dtype=np.float32) % 13 - 6) / 8).reshape(rows, 8)
    y = np.arange(rows, dtype=np.int32) % 3
    return x, y


def state_bytes(model):
    return impl._canonical(impl._encode_state(model.state_dict()))


def pointer(address, size, integer=False):
    dtype = ctypes.c_int32 if integer else ctypes.c_float
    return np.ctypeslib.as_array((dtype * size).from_address(address))


class FakeTraining:
    def __init__(self):
        self.calls = []
        self.fail_after_write = False
        self.nonfinite_after_write = False

    def mlp_bias_activation(self, values, bias, out, params):
        rows, cols, relu = params
        self.calls.append(('bias', rows, cols, relu))
        pointer(out, rows * cols)[:] = .5
        return rows * cols

    def mlp_relu_backward(self, activation, incoming, out, params):
        rows, cols = params
        self.calls.append(('relu', rows, cols))
        pointer(out, rows * cols)[:] = .125
        return rows * cols

    def mlp_sum_rows(self, values, out, params):
        rows, cols = params
        self.calls.append(('sum', rows, cols))
        pointer(out, cols)[:] = .25
        return cols

    def optimizer_step(self, params, grads, m, v, offsets, flags, info, config):
        self.calls.append(('optimizer', list(config)))
        np.testing.assert_array_equal(pointer(offsets, 5, True), [0, 128, 144, 192, 195])
        # Deliberately mutate every carried buffer before fault injection.
        pointer(params, 195)[:] += 1
        pointer(m, 195)[:] += 2
        pointer(v, 195)[:] += 3
        pointer(flags, 4, True)[:] = 1
        pointer(info, 3)[:] = 0
        if self.nonfinite_after_write:
            pointer(v, 195)[0] = np.nan
        if self.fail_after_write:
            raise RuntimeError('injected failure after native mutation')


@pytest.fixture
def host(monkeypatch):
    fake = FakeTraining()
    monkeypatch.setattr(impl._backend, 'default_mode', lambda: 'identical')
    monkeypatch.setattr(impl._backend, 'numeric_mode', lambda: 'identical')
    monkeypatch.setattr(impl._linalg_impl, 'require_identical', lambda: None)
    monkeypatch.setattr(impl._training_impl, '_load', lambda mode=None: fake)

    def gemm(a, b, *, transpose_a=False, transpose_b=False, identical=True):
        assert identical is True
        m, k = a.shape[::-1] if transpose_a else a.shape
        kb, n = b.shape[::-1] if transpose_b else b.shape
        assert k == kb
        fake.calls.append(('gemm', m, n, k))
        # DEVIATION 2460: the real matmul returns mojolearn.Array and the
        # trainer checks it with `_buffer.all_finite`, so the fake does too
        # (a zero-copy view over the ndarray, same bytes as before).
        return Array.from_buffer(np.full((m, n), .125, dtype=np.float32))

    def loss(logits, labels, **kwargs):
        assert kwargs == dict(reduction='mean', return_grad=True, numeric_mode='identical')
        fake.calls.append(('loss', len(labels)))
        return .75, Array.from_buffer(np.full(logits.shape, .125, np.float32))
    monkeypatch.setattr(impl._linalg_impl, 'matmul', gemm)
    monkeypatch.setattr(impl._training_impl, 'cross_entropy', loss)
    return fake


def trainer():
    return SmallMLPTrainer(*weights(), data_schedule={'dataset': 'fixed-dyadic.v1', 'order': 'sequential'})


def test_host_copies_inputs_and_returns_all_gradients(host):
    supplied = weights()
    saved = [value.copy() for value in supplied]
    model = SmallMLPTrainer(*supplied, data_schedule={'dataset': 'test', 'order': [0, 1]})
    supplied[0][:] = 900
    np.testing.assert_array_equal(model.weights_['weight1'], saved[0])
    x, y = batch()
    x.setflags(write=False)
    y.setflags(write=False)
    before = (x.tobytes(), y.tobytes())
    result = model.train_step(x, y, return_input_grad=True)
    assert result['step'] == model.step_ == 1
    assert result['loss'] == .75
    assert result['logits'].shape == (7, 3)
    assert result['input_grad'].shape == (7, 8)
    assert list(result['gradients']) == list(impl._NAMES)
    for name, shape in zip(impl._NAMES, impl._SHAPES):
        assert result['gradients'][name].shape == shape
        # DEVIATION 2460: both are mojolearn.Array; np.asarray views them zero-copy
        assert not np.shares_memory(np.asarray(result['gradients'][name]),
                                    np.asarray(model.weights_[name]))
    assert before == (x.tobytes(), y.tobytes())
    assert ('gemm', 16, 8, 7) in host.calls
    assert ('gemm', 3, 16, 7) in host.calls


def test_host_default_skips_input_gradient_and_predict_does_not_update(host):
    model = trainer()
    x, y = batch(1)
    before = state_bytes(model)
    assert model.predict_logits(x).shape == (1, 3)
    assert state_bytes(model) == before
    result = model.train_step(x, y)
    assert 'input_grad' not in result
    assert ('gemm', 1, 8, 16) not in host.calls


@pytest.mark.parametrize('failure', ['fail_after_write', 'nonfinite_after_write'])
def test_host_failed_update_is_transactional(host, failure):
    model = trainer()
    model.train_step(*batch())
    before = state_bytes(model)
    setattr(host, failure, True)
    with pytest.raises((ValueError, RuntimeError)):
        model.train_step(*batch(), return_input_grad=True)
    assert state_bytes(model) == before
    assert model.step_ == 1


def test_host_state_snapshots_and_load_are_isolated(host):
    model = trainer()
    model.train_step(*batch())
    snapshot = model.state_dict()
    other = trainer().load_state_dict(snapshot)
    assert state_bytes(other) == state_bytes(model)
    # DEVIATION 2460: snapshot buffers are mojolearn.Array; np.asarray is a
    # zero-copy writable view, so the isolation check tests the same bytes.
    np.asarray(snapshot['weights']['weight1'])[:] = 7
    np.asarray(snapshot['optimizer']['m'])[:] = 8
    np.asarray(snapshot['optimizer']['v'])[:] = 9
    np.asarray(snapshot['optimizer']['flags'])[:] = 0
    snapshot['data_schedule']['order'] = 'changed'
    assert state_bytes(other) == state_bytes(model)
    bad = model.state_dict()
    np.asarray(bad['optimizer']['flags'])[0] = 2
    before = state_bytes(other)
    with pytest.raises(ValueError):
        other.load_state_dict(bad)
    assert state_bytes(other) == before


def test_host_serializes_concurrent_state_updates(host):
    model = trainer()
    x, y = batch()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(model.train_step, x, y) for _ in range(2)]
        assert sorted(future.result()['step'] for future in futures) == [1, 2]
    assert model.step_ == 2


def test_host_rejects_invalid_inputs_before_device_calls(host):
    model = trainer()
    x, y = batch()
    cases = [(x.astype(np.float64), y), (x[:0], y[:0]),
             (np.zeros((257, 8), np.float32), np.zeros(257, np.int32)),
             (np.full_like(x, np.nan), y), (x[:, :7], y),
             (x, np.full_like(y, -100)), (x, np.full_like(y, 3)),
             (x, np.full_like(y, 1 << 32, dtype=np.int64)),
             (x, y.astype(np.float32)), (x, y.reshape(-1, 1))]
    before = state_bytes(model)
    for bad_x, bad_y in cases:
        with pytest.raises((TypeError, ValueError)):
            model.train_step(bad_x, bad_y)
    assert host.calls == []
    assert state_bytes(model) == before


def test_host_mode_change_refuses_without_switching_it(host, monkeypatch):
    model = trainer()
    before = state_bytes(model)
    monkeypatch.setattr(impl._backend, 'default_mode', lambda: 'fast')
    with pytest.raises(RuntimeError, match='process-selected'):
        model.train_step(*batch())
    assert host.calls == []
    assert state_bytes(model) == before
    with pytest.raises(RuntimeError, match='process-selected'):
        trainer()


@pytest.mark.parametrize('options', [dict(lr=0), dict(eps=0), dict(weight_decay=-1),
                                    dict(betas=(.9, 1)), dict(lr=np.inf),
                                    dict(eps=1e-100), dict(betas=(.9, np.nan))])
def test_host_optimizer_config_refuses_before_device(host, options):
    with pytest.raises(ValueError):
        SmallMLPTrainer(*weights(), data_schedule={'dataset': 'test'}, **options)
    assert host.calls == []


def test_host_checkpoint_is_canonical_complete_and_resumable(host, tmp_path):
    model = trainer()
    model.train_step(*batch())
    first, second = tmp_path / 'first.mlp.json', tmp_path / 'second.mlp.json'
    model.save_checkpoint(first)
    model.save_checkpoint(second)
    assert first.read_bytes() == second.read_bytes()
    assert len(first.read_bytes()) < impl._FILE_LIMIT
    restored = SmallMLPTrainer.from_checkpoint(first)
    assert state_bytes(restored) == state_bytes(model)
    model.train_step(*batch())
    restored.train_step(*batch())
    assert state_bytes(restored) == state_bytes(model)


def test_host_checkpoint_refuses_corruption_wrong_schema_and_oversize(host, tmp_path):
    path = tmp_path / 'state.json'
    trainer().save_checkpoint(path)
    original = path.read_bytes()
    envelope = json.loads(original)
    envelope['payload']['optimizer']['step'] = 4
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match='integrity'):
        SmallMLPTrainer.from_checkpoint(path)
    envelope = json.loads(original)
    envelope['schema'] = 'llama-checkpoint'
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match='schema'):
        SmallMLPTrainer.from_checkpoint(path)
    envelope = json.loads(original)
    envelope['payload']['weights']['extra'] = envelope['payload']['weights']['bias2']
    envelope['payload_sha256'] = hashlib.sha256(impl._canonical(envelope['payload'])).hexdigest()
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match='registry'):
        SmallMLPTrainer.from_checkpoint(path)
    path.write_bytes(b' ' * (impl._FILE_LIMIT + 1))
    with pytest.raises(ValueError, match='size'):
        SmallMLPTrainer.from_checkpoint(path)


@pytest.mark.skipif(os.environ.get('MOJOLEARN_RUN_SMALL_MLP_GPU') != '1',
                    reason='root-only opt-in remote NVIDIA/AMD numerical gate')
@pytest.mark.parametrize('rows', [1, 7, 256])
def test_remote_independent_reference_and_resume(rows, tmp_path):
    if sys.platform != 'linux':
        pytest.fail('SmallMLP GPU gate refuses Apple/non-Linux execution')
    import torch
    from mojolearn import _backend
    if _backend.default_mode() != 'identical':
        pytest.fail('Set process IDENTICAL before importing mojolearn')
    binding = impl._training_impl._load('identical')
    if _backend.read_vendor(binding) not in ('cuda', 'hip'):
        pytest.fail('SmallMLP remote gate requires CUDA or HIP')
    # CPU float64 autograd is an independent numerical oracle, never timed.
    # AdamW implementations have different bias-correction arithmetic; only
    # logits/loss/gradients use this reference. Resume is a raw-byte contract.
    x, y = batch(rows)
    supplied = weights()
    model = SmallMLPTrainer(*supplied, data_schedule={'dataset': 'dyadic.v1', 'rows': rows})
    tx = torch.tensor(x.astype(np.float64), requires_grad=True)
    tw = [torch.tensor(value.astype(np.float64), requires_grad=True) for value in supplied]
    logits = torch.nn.functional.linear(
        torch.relu(torch.nn.functional.linear(tx, tw[0], tw[1])), tw[2], tw[3])
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor(y.astype(np.int64)))
    loss.backward()
    result = model.train_step(x, y, return_input_grad=True)
    np.testing.assert_allclose(result['logits'], logits.detach().numpy(), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(result['loss'], loss.detach().numpy(), rtol=2e-5, atol=2e-6)
    for name, tensor in zip(impl._NAMES, tw):
        np.testing.assert_allclose(result['gradients'][name], tensor.grad.numpy(), rtol=2e-4, atol=3e-6)
    np.testing.assert_allclose(result['input_grad'], tx.grad.numpy(), rtol=2e-4, atol=3e-6)
    independent = SmallMLPTrainer(*supplied, data_schedule={'dataset': 'dyadic.v1', 'rows': rows})
    repeated = independent.train_step(x, y, return_input_grad=True)
    assert state_bytes(model) == state_bytes(independent)
    for key in ('logits', 'input_grad'):
        assert result[key].tobytes() == repeated[key].tobytes()
    for name in impl._NAMES:
        assert result['gradients'][name].tobytes() == repeated['gradients'][name].tobytes()
    checkpoint = tmp_path / 'small-mlp.json'
    model.save_checkpoint(checkpoint)
    resumed = SmallMLPTrainer.from_checkpoint(checkpoint)
    model.train_step(x, y)
    resumed.train_step(x, y)
    assert state_bytes(model) == state_bytes(resumed)
