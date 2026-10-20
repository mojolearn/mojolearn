# SPDX-License-Identifier: Apache-2.0
"""Whole standard training steps against independent primitive-chain oracles."""
import numpy as np
import pytest

from mojolearn import SambaConfig, SambaStack
from mojolearn import _native_samba_bridge as bridge
from mojolearn import _training_impl as training
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_causal_model_native_loading import _poison
from mojolearn.tests.test_samba_gradients_native import _reference
from mojolearn.tests.test_samba_state_native import _assert_state_equal


def _step(model, inputs, targets):
    return audit_native_call(model.train_step, inputs, targets).assert_native()


def _reference_step(model, inputs, targets):
    # Testing-only explicit recipe, separate from both Samba step coordinators.
    count = int(np.count_nonzero(targets != -100))
    a = model.optimizer.accumulation_steps
    rows = inputs.shape[0] // a
    stream = model.generator.next_stream() if model.config.dropout else None
    losses, parts = [], []
    for k in range(a):
        start, end = k * rows, (k + 1) * rows
        loss, gradient_bytes = _reference(model, inputs[start:end], targets[start:end],
                                           count, stream, start * inputs.shape[1])
        losses.append(np.array([loss], dtype="f4"))
        parts.append([np.frombuffer(value, dtype="f4").copy().reshape(model.shapes[name])
                      for name, value in zip(model.names, gradient_bytes)])
    gradients = parts[0] if a == 1 else training.accumulate_grads(parts, tokens=inputs.size)
    loss = losses[0][0] if a == 1 else training.accumulate_grads(losses, tokens=None)[0]
    norm = model.optimizer.step(gradients, max_norm=model.max_norm)
    return {"loss": float(loss), "lr": model.optimizer.lr_,
            "step": model.optimizer.t, "total_norm": norm}


@pytest.mark.parametrize("layers,dm,dropout,accumulation,tied,shape", [
    (["attention"], 8, 0.0, 1, True, (2, 3)),
    (["mamba3", "attention"], 32, 0.125, 1, False, (2, 3)),
    (["attention"], 8, 0.125, 2, True, (4, 64)),
])
@pytest.mark.parametrize("resident", [False, True])
def test_native_standard_step_matches_independent_training_recipe(
        monkeypatch, layers, dm, dropout, accumulation, tied, shape, resident):
    monkeypatch.setenv("MOJOLEARN_OPTIMIZER_RESIDENT", "1" if resident else "0")
    config = SambaConfig(19, dm, layers, n_heads=2, n_kv_heads=1,
                         intermediate=48 if dm == 32 else 11,
                         tie_embeddings=tied, dropout=dropout)
    rng = np.random.default_rng(641)
    weights = {name: rng.uniform(-0.02, 0.02, dims).astype("f4")
               for name, dims in config.registry()}
    for name in weights:
        if "norm" in name or name.endswith(".D"):
            weights[name].fill(1)
    models = [SambaStack(config, weights=weights, lr=0.003,
                         betas=(0.8, 0.95), max_norm=0.25, accumulation_steps=accumulation,
                         lr_schedule=training.WarmupCosineLR(0.003, 3, 11))
              for _ in range(2)]
    expected_model, actual_model = models
    inputs = rng.integers(0, config.vocab, size=shape, dtype="i4")
    targets = rng.integers(0, config.vocab, size=shape, dtype="i4")
    targets[0, 1] = -100
    for _ in range(2):
        expected = _reference_step(expected_model, inputs, targets)
        with monkeypatch.context() as patch:
            _poison(patch)
            patch.setattr(SambaStack, "_legacy_train_step", lambda *a: pytest.fail("legacy Python coordinator"))
            patch.setattr(SambaStack, "loss_and_grads", lambda *a, **k: pytest.fail("Python gradient wrapper"))
            patch.setattr(training.AdamW, "step", lambda *a, **k: pytest.fail("Python optimizer step"))
            actual = _step(actual_model, inputs, targets)
        assert actual == expected
        assert actual == actual_model.last_ and actual is not actual_model.last_
        expected_state = expected_model.state_dict()
        actual_state = audit_native_call(bridge.RUNTIME.samba_native_call, "snapshot",
                                         (actual_model, True), bridge.CONTEXT).assert_native()
        _assert_state_equal(actual_state, expected_state)


def test_native_step_rejects_misalignment_before_state_changes():
    config = SambaConfig(13, 8, ["attention"], n_heads=2, intermediate=11, dropout=0.125)
    model = SambaStack(config, generator=training.Generator(8), accumulation_steps=2)
    inputs = np.array([[1, 2, 3], [4, 5, 6]], dtype="i4")
    before = model.state_dict()
    result = audit_native_call(model.train_step, inputs, inputs)
    assert isinstance(result.exception, ValueError), result.report()
    assert "MISALIGNED" in str(result.exception)
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work",
                "unobserved_existing_threads"):
        assert not result.report()[key], result.report()
    _assert_state_equal(model.state_dict(), before)


def test_public_native_step_never_invokes_custom_schedule():
    class CustomSchedule:
        def lr_at(self, step):
            pytest.fail("public native owner invoked a Python schedule")

    config = SambaConfig(13, 8, ["attention"], n_heads=2, intermediate=11, dropout=0.125)
    model = SambaStack(config, generator=training.Generator(8), lr_schedule=CustomSchedule())
    before = model.flat.tobytes(), model.generator.state_dict(), model.optimizer.t
    inputs = np.array([[1, 2, 3]], dtype="i4")
    result = audit_native_call(model.train_step, inputs, inputs)
    assert isinstance(result.exception, NotImplementedError), result.report()
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work",
                "unobserved_existing_threads"):
        assert not result.report()[key], result.report()
    assert (model.flat.tobytes(), model.generator.state_dict(), model.optimizer.t) == before


def _owner_models():
    config = SambaConfig(13, 8, ["attention"], n_heads=2,
                         intermediate=11, dropout=0.125)
    rng = np.random.default_rng(919)
    weights = {name: rng.uniform(-0.04, 0.04, shape).astype("f4")
               for name, shape in config.registry()}
    for name in weights:
        if "norm" in name:
            weights[name].fill(1)
    return [SambaStack(config, weights=weights, lr=0.002,
                       max_norm=0.3, weight_decay=0.02) for _ in range(2)]


@pytest.mark.parametrize("resident", [False, True])
def test_native_step_preserves_rebound_optimizer_parameter_owners(monkeypatch, resident):
    monkeypatch.setenv("MOJOLEARN_OPTIMIZER_RESIDENT", "1" if resident else "0")
    expected, actual = _owner_models()
    for model in (expected, actual):
        # These are separate writable buffers. The model still reads its
        # original flat weights, as the established public API does.
        model.optimizer.params = [np.asarray(value).copy() + np.float32(0.1)
                                  for value in model.optimizer.params]
    aliases = tuple(actual.optimizer.params)
    original_flat = actual.flat.tobytes()
    inputs = np.array([[1, 2, 4], [3, 5, 6]], dtype="i4")
    targets = np.array([[2, -100, 6], [5, 6, 7]], dtype="i4")
    for _ in range(2):
        metrics = _reference_step(expected, inputs, targets)
        with monkeypatch.context() as patch:
            _poison(patch)
            patch.setattr(training.AdamW, "step", lambda *a, **k: pytest.fail("Python optimizer"))
            assert _step(actual, inputs, targets) == metrics
        assert actual.flat.tobytes() == original_flat
        assert all(a is b for a, b in zip(actual.optimizer.params, aliases))
        for a, b in zip(actual.optimizer.params, expected.optimizer.params):
            assert a.tobytes() == b.tobytes()
        assert actual.optimizer.exp_avg.tobytes() == expected.optimizer.exp_avg.tobytes()
        assert actual.optimizer.exp_avg_sq.tobytes() == expected.optimizer.exp_avg_sq.tobytes()
        assert actual.generator.state_dict() == expected.generator.state_dict()


@pytest.mark.parametrize("clipping", [None, 0.3])
def test_native_step_preserves_device_parameter_owner(monkeypatch, clipping):
    expected, actual = _owner_models()
    if not expected.optimizer.params_resident_available():
        pytest.skip("native optimizer device-parameter entry is unavailable")
    for model in (expected, actual):
        model.max_norm = clipping
        model.optimizer.params_to_device()
    inputs = np.array([[1, 2, 4], [3, 5, 6]], dtype="i4")
    targets = np.array([[2, -100, 6], [5, 6, 7]], dtype="i4")
    for _ in range(2):
        stale = actual.flat.tobytes()
        metrics = _reference_step(expected, inputs, targets)
        with monkeypatch.context() as patch:
            _poison(patch)
            patch.setattr(training.AdamW, "step_device", lambda *a, **k: pytest.fail("Python device optimizer"))
            assert _step(actual, inputs, targets) == metrics
        assert actual.flat.tobytes() == stale
        assert actual.optimizer._p_dev and actual.optimizer._g_dev
        actual.optimizer.params_to_host()
        expected.optimizer.params_to_host()
        assert actual.flat.tobytes() == expected.flat.tobytes()
        assert actual.optimizer.exp_avg.tobytes() == expected.optimizer.exp_avg.tobytes()
        assert actual.optimizer.exp_avg_sq.tobytes() == expected.optimizer.exp_avg_sq.tobytes()
