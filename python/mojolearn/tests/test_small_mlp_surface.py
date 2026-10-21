# SPDX-License-Identifier: Apache-2.0
"""SmallMLPTrainer surface gates on the selected native route.

Every public method is one native call (bindings/small_mlp_native.mojo); these
tests run it on whatever binding this process selected and check the contract:
copies, shapes, refusals before any state change, transactional updates,
serialized concurrent steps, canonical checkpoints and the one-call boundary.
They are not numerical qualification; the opt-in independent-reference gate at
the end runs only on remote Linux CUDA/HIP with MOJOLEARN_RUN_SMALL_MLP_GPU=1.
"""
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from mojolearn import SmallMLPTrainer, _backend
from mojolearn import _mlp_impl as impl
from mojolearn._verify_native_runtime import audit_native_call


def weights():
    return [((np.arange(np.prod(shape), dtype=np.float32) % 7 - 3) / 32).reshape(shape)
            for shape in ((16, 8), (16,), (3, 16), (3,))]


def batch(rows=7):
    x = ((np.arange(rows * 8, dtype=np.float32) % 13 - 6) / 8).reshape(rows, 8)
    y = np.arange(rows, dtype=np.int32) % 3
    return x, y


def state_bytes(model):
    return impl._canonical(impl._encode_state(model.state_dict()))


def trainer():
    return SmallMLPTrainer(*weights(), data_schedule={'dataset': 'fixed-dyadic.v1', 'order': 'sequential'})


def overflowing(model):
    """Load a valid state whose first layer overflows float32 on any batch of ones."""
    state = model.state_dict()
    np.asarray(state['weights']['weight1'])[:] = np.float32(3e38)
    model.load_state_dict(state)
    return np.ones((4, 8), np.float32), np.zeros(4, np.int32)


def test_copies_inputs_and_returns_all_gradients():
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
    assert isinstance(result['loss'], float) and np.isfinite(result['loss'])
    assert result['logits'].shape == (7, 3)
    assert result['input_grad'].shape == (7, 8)
    assert list(result['gradients']) == list(impl._NAMES)
    for name, shape in zip(impl._NAMES, impl._SHAPES):
        assert result['gradients'][name].shape == shape
        assert not np.shares_memory(np.asarray(result['gradients'][name]),
                                    np.asarray(model.weights_[name]))
    assert before == (x.tobytes(), y.tobytes())


def test_default_skips_input_gradient_and_predict_does_not_update():
    model = trainer()
    x, y = batch(1)
    before = state_bytes(model)
    assert model.predict_logits(x).shape == (1, 3)
    assert state_bytes(model) == before
    result = model.train_step(x, y)
    assert 'input_grad' not in result
    assert set(result) == {'step', 'loss', 'logits', 'gradients'}


def test_predict_matches_the_logits_of_the_next_step():
    model = trainer()
    x, y = batch(5)
    predicted = np.asarray(model.predict_logits(x)).tobytes()
    assert np.asarray(model.train_step(x, y)['logits']).tobytes() == predicted


@pytest.mark.parametrize('operation', ['train_step', 'apply_gradients'])
def test_failed_update_is_transactional(operation):
    model = trainer()
    model.train_step(*batch())
    x, y = overflowing(model)
    before = state_bytes(model)
    with pytest.raises((ValueError, RuntimeError)):
        if operation == 'train_step':
            model.train_step(x, y, return_input_grad=True)
        else:
            huge = [np.full(shape, 3e38, np.float32) for shape in impl._SHAPES]
            model.apply_gradients(huge)
            model.apply_gradients(huge)
            model.apply_gradients([np.full(shape, np.float32(3.4e38), np.float32) for shape in impl._SHAPES])
            raise RuntimeError('three huge updates did not overflow')
    if operation == 'train_step':
        assert state_bytes(model) == before
        assert model.step_ == 1


def test_state_snapshots_and_load_are_isolated():
    model = trainer()
    model.train_step(*batch())
    snapshot = model.state_dict()
    other = trainer().load_state_dict(snapshot)
    assert state_bytes(other) == state_bytes(model)
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


def test_serializes_concurrent_state_updates():
    model = trainer()
    x, y = batch()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(model.train_step, x, y) for _ in range(2)]
        assert sorted(future.result()['step'] for future in futures) == [1, 2]
    assert model.step_ == 2


def test_rejects_invalid_inputs_before_state_changes():
    model = trainer()
    x, y = batch()
    cases = [(x.astype(np.float64), y), (x[:0], y[:0]),
             (np.zeros((257, 8), np.float32), np.zeros(257, np.int32)),
             (np.full_like(x, np.nan), y), (x[:, :7], y),
             (x, np.full_like(y, -100)), (x, np.full_like(y, 3)),
             (x, np.full_like(y, 1 << 32, dtype=np.int64)),
             (x, y.astype(np.float32)), (x, y.reshape(-1, 1)),
             ([[0.0] * 8] * 7, y)]
    before = state_bytes(model)
    for bad_x, bad_y in cases:
        with pytest.raises((TypeError, ValueError)):
            model.train_step(bad_x, bad_y)
    with pytest.raises(TypeError):
        model.train_step(x, y, return_input_grad=1)
    assert state_bytes(model) == before
    assert model.step_ == 0


def test_unselected_mode_refuses_without_state_change(monkeypatch):
    model = trainer()
    before = state_bytes(model)
    with monkeypatch.context() as changed:
        changed.setattr(_backend, '_DEFAULT_MODE', 'unknown')
        with pytest.raises(RuntimeError, match='process-selected'):
            model.train_step(*batch())
        with pytest.raises(RuntimeError, match='process-selected'):
            trainer()
    assert state_bytes(model) == before


@pytest.mark.parametrize('options', [dict(lr=0), dict(eps=0), dict(weight_decay=-1),
                                    dict(betas=(.9, 1)), dict(lr=np.inf),
                                    dict(eps=1e-100), dict(betas=(.9, np.nan)),
                                    dict(lr=True), dict(betas=(.9,))])
def test_optimizer_config_refuses(options):
    with pytest.raises(ValueError):
        SmallMLPTrainer(*weights(), data_schedule={'dataset': 'test'}, **options)


@pytest.mark.parametrize('schedule', [{}, [1], {'a': float('nan')}, {'a': object()},
                                      {'a': 'x' * 3000}, {1: 'x'}])
def test_data_schedule_refuses(schedule):
    with pytest.raises(ValueError):
        SmallMLPTrainer(*weights(), data_schedule=schedule)


def test_checkpoint_is_canonical_complete_and_resumable(tmp_path):
    model = trainer()
    model.train_step(*batch())
    first, second = tmp_path / 'first.mlp.json', tmp_path / 'second.mlp.json'
    model.save_checkpoint(first)
    model.save_checkpoint(str(second))
    assert first.read_bytes() == second.read_bytes()
    assert len(first.read_bytes()) < impl._FILE_LIMIT
    assert first.read_bytes().endswith(b'\n')
    envelope = json.loads(first.read_bytes())
    assert envelope['payload_sha256'] == hashlib.sha256(impl._canonical(envelope['payload'])).hexdigest()
    assert first.read_bytes() == impl._canonical(envelope) + b'\n'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['first.mlp.json', 'second.mlp.json']
    restored = SmallMLPTrainer.from_checkpoint(first)
    assert type(restored) is SmallMLPTrainer
    assert state_bytes(restored) == state_bytes(model)
    model.train_step(*batch())
    restored.train_step(*batch())
    assert state_bytes(restored) == state_bytes(model)


def test_checkpoint_refuses_corruption_wrong_schema_and_oversize(tmp_path):
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
    path.write_bytes(original.replace(b'{"schema"', b'{"schema":"x","schema"', 1)
                     if original.startswith(b'{"schema"') else original[:1] + b'"schema":"x",' + original[1:])
    with pytest.raises(ValueError, match='JSON'):
        SmallMLPTrainer.from_checkpoint(path)
    path.write_bytes(b' ' * (impl._FILE_LIMIT + 1))
    with pytest.raises(ValueError, match='size'):
        SmallMLPTrainer.from_checkpoint(path)
    with pytest.raises(FileNotFoundError):
        SmallMLPTrainer.from_checkpoint(tmp_path / 'missing.json')


def test_per_operation_route_has_the_fused_bits(monkeypatch):
    x, y = batch(9)
    results = []
    for fused in ('1', '0'):
        monkeypatch.setenv('MOJOLEARN_MLP_FUSED', fused)
        model = trainer()
        first = model.train_step(x, y, return_input_grad=True)
        loss, grads = model.loss_and_grads(x, y)
        step = model.apply_gradients(grads)
        results.append((state_bytes(model), step, loss, np.asarray(model.predict_logits(x)).tobytes(),
                        np.asarray(first['input_grad']).tobytes(), first['loss']))
    assert results[0] == results[1]


def test_sessions_concatenate_the_per_batch_logits():
    model = trainer()
    a, _ = batch(3)
    b, _ = batch(5)
    joined = np.asarray(model.predict_logits_sessions([a, b]))
    assert joined.shape == (8, 3)
    np.testing.assert_array_equal(joined[:3], np.asarray(model.predict_logits(a)))
    np.testing.assert_array_equal(joined[3:], np.asarray(model.predict_logits(b)))


def test_closed_copy_refuses_instead_of_aliasing():
    import copy
    model = trainer()
    alias = copy.copy(model)
    del model
    with pytest.raises(RuntimeError, match='closed'):
        alias.state_dict()
    alias._handle = None


@pytest.mark.parametrize('method', ['init', 'train_step', 'train_step_dx', 'predict_logits',
                                    'loss_and_grads', 'apply_gradients', 'state_dict',
                                    'load_state_dict', 'step_', 'weights_', 'sessions',
                                    'save_checkpoint', 'from_checkpoint'])
def test_public_method_is_one_native_call(method, tmp_path):
    model = trainer()
    x, y = batch()
    path = tmp_path / 'audit.json'
    model.save_checkpoint(path)
    grads = model.loss_and_grads(x, y)[1]
    state = model.state_dict()
    calls = {
        'init': (SmallMLPTrainer, (*weights(),), dict(data_schedule={'dataset': 'audit'})),
        'train_step': (model.train_step, (x, y), {}),
        'train_step_dx': (model.train_step, (x, y), dict(return_input_grad=True)),
        'predict_logits': (model.predict_logits, (x,), {}),
        'loss_and_grads': (model.loss_and_grads, (x, y), {}),
        'apply_gradients': (model.apply_gradients, (grads,), {}),
        'state_dict': (model.state_dict, (), {}),
        'load_state_dict': (model.load_state_dict, (state,), {}),
        'step_': (type(model).step_.fget, (model,), {}),
        'weights_': (type(model).weights_.fget, (model,), {}),
        'sessions': (model.predict_logits_sessions, ([x, x],), {}),
        'save_checkpoint': (model.save_checkpoint, (tmp_path / 'saved.json',), {}),
        'from_checkpoint': (SmallMLPTrainer.from_checkpoint, (path,), {}),
    }
    operation, args, kwargs = calls[method]
    audit_native_call(operation, *args, **kwargs).assert_native()


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
