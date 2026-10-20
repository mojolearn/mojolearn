# SPDX-License-Identifier: Apache-2.0
"""Native Samba state bridge against the existing independent Python state oracle."""
import copy

import numpy as np
import pytest

from mojolearn import SambaConfig, SambaStack
from mojolearn import _native_samba_bridge as bridge
from mojolearn import _training_impl as training
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_causal_model_native_loading import _poison


ARRAY_FIELDS = ("parameters", "exp_avg", "exp_avg_sq", "buf_initialized")


def _call(operation, *values):
    return audit_native_call(bridge.RUNTIME.samba_native_call, operation,
                             values, bridge.CONTEXT).assert_native()


def _stack(tied=True):
    config = SambaConfig(19, 32, ["mamba3", "attention"], n_heads=2,
                         n_kv_heads=1, intermediate=48, tie_embeddings=tied,
                         dropout=0.125)
    rng = np.random.default_rng(718)
    weights = {name: rng.uniform(-0.1, 0.1, shape).astype("f4")
               for name, shape in config.registry()}
    stack = SambaStack(config, weights=weights, lr=0.003,
                       betas=(0.8, 0.95), weight_decay=0.02, max_norm=1.5,
                       lr_schedule=training.WarmupCosineLR(0.003, 3, 17),
                       accumulation_steps=2)
    stack.generator.counter = 11
    stack.optimizer.t = 7
    stack.optimizer.exp_avg = np.linspace(-0.01, 0.01, stack.n_total, dtype="f4")
    stack.optimizer.exp_avg_sq = np.linspace(0, 0.02, stack.n_total, dtype="f4")
    return stack



def _reference_state(model):
    # Testing-only independent descriptor and public optimizer metadata recipe.
    # It does not call either Samba snapshot implementation under test.
    o = model.optimizer
    binding = training._load(model.numeric_mode)
    experiment = getattr(binding, "training_experiment_profile", lambda: "baseline")()
    return {
        "schema": "mojolearn.samba-stack-state.v1",
        "profile": "mojolearn.samba-stack.fp32.v1",
        "experiment_profile": experiment,
        "numeric_mode": "identical" if binding.training_numeric_mode() == 1 else "fast",
        "config": model.config.to_dict(),
        "registry": [{"name": name, "shape": list(model.shapes[name]),
                      "offset": model.offsets[i], "size": model.offsets[i+1] - model.offsets[i]}
                     for i, name in enumerate(model.names)],
        "parameters": model.flat.copy(), "exp_avg": o.exp_avg.copy(),
        "exp_avg_sq": o.exp_avg_sq.copy(), "buf_initialized": o.buf_initialized.copy(),
        "t": o.t,
        "optimizer": {"kind": "adamw", "lr": o.lr, "beta1": o.betas[0], "beta2": o.betas[1],
                      "eps": o.eps, "weight_decay": o.weight_decay, "max_norm": model.max_norm,
                      "accumulation_steps": o.accumulation_steps},
        "schedule": None if o.lr_schedule is None else o.lr_schedule.config(),
        "rng": {"seed": model.generator.seed, "counter": model.generator.counter},
    }

def _assert_state_equal(actual, expected):
    assert set(actual) == set(expected)
    for name in actual:
        if name in ARRAY_FIELDS:
            assert actual[name].shape == expected[name].shape, name
            assert actual[name].tobytes() == expected[name].tobytes(), name
        else:
            assert actual[name] == expected[name], name


@pytest.mark.parametrize("tied", [True, False])
def test_native_samba_snapshot_and_reconstruction_preserve_complete_state(monkeypatch, tied):
    stack = _stack(tied)
    expected = _reference_state(stack)
    with monkeypatch.context() as patch:
        _poison(patch)
        patch.setattr(SambaStack, "state_dict", lambda *a, **k: pytest.fail("Python snapshot"))
        actual = _call("snapshot", stack, True)
        clone = _call("from_state", actual)
        resumed = _call("snapshot", clone, True)
    _assert_state_equal(actual, expected)
    _assert_state_equal(resumed, expected)
    # The saved final-normalization epsilon is separate from each attention
    # block's fixed backward profile (1e-6). Reusing checkpoint-interface
    # defaults here used to mark restored attention blocks as extended.
    assert clone.config.norm_eps == stack.config.norm_eps
    assert clone._blocks[1].norm_eps == stack._blocks[1].norm_eps == 1e-6
    assert clone._blocks[1]._opts_tail == stack._blocks[1]._opts_tail
    assert clone._blocks[1]._extended is False
    for item in expected["registry"]:
        view = clone.arrays[item["name"]]
        assert view._addr == clone.flat._addr + 4 * item["offset"]
        assert view._base is clone.flat
        assert view.shape == tuple(item["shape"])
    for name in ARRAY_FIELDS:
        assert actual[name]._addr != resumed[name]._addr


def test_native_samba_publish_keeps_existing_parameter_aliases(monkeypatch):
    target = _stack()
    state = target.state_dict()
    state["parameters"] = state["parameters"] + np.float32(0.025)
    state["rng"]["counter"] = 23
    state["t"] = 9
    old_addresses = {name: value._addr for name, value in target.arrays.items()}
    old_flat = target.flat
    with monkeypatch.context() as patch:
        _poison(patch)
        patch.setattr(SambaStack, "load_state_dict", lambda *a: pytest.fail("Python publish"))
        assert _call("publish", target, state) is target
    assert target.flat is old_flat
    assert {name: value._addr for name, value in target.arrays.items()} == old_addresses
    _assert_state_equal(_call("snapshot", target, True), state)
    assert target.optimizer.lr_schedule.lr_at(10) == training.WarmupCosineLR(0.003, 3, 17).lr_at(10)


def test_native_samba_state_roundtrips_exhausted_generator():
    model = _stack()
    model.generator.load_state_dict({"seed": 9, "counter": (1 << 32) - 1})
    assert model.generator.next_stream() == (1 << 32) - 1
    state = _call("snapshot", model, True)
    assert state["rng"] == {"seed": 9, "counter": 1 << 32}
    clone = _call("from_state", state)
    assert _call("snapshot", clone, True)["rng"] == state["rng"]
    before = clone.flat.tobytes()
    with pytest.raises(RuntimeError, match="exhausted"):
        clone.generator.next_stream()
    assert clone.flat.tobytes() == before
    assert clone.generator.state_dict() == state["rng"]


@pytest.mark.parametrize("corruption", ["registry", "moment_shape", "flags", "rng"])
def test_native_samba_state_validation_refuses_corruption_before_publish(corruption):
    target = _stack()
    before = target.flat.tobytes()
    state = target.state_dict()
    if corruption == "registry":
        state["registry"] = copy.deepcopy(state["registry"])
        state["registry"][1]["offset"] += 1
    elif corruption == "moment_shape":
        state["exp_avg"] = np.zeros(target.n_total - 1, dtype="f4")
    elif corruption == "flags":
        state["buf_initialized"] = np.zeros(len(target.names), dtype="i8")
    else:
        state["rng"]["counter"] = 2 ** 32 + 1
    result = audit_native_call(bridge.RUNTIME.samba_native_call, "publish", (target, state), bridge.CONTEXT)
    assert isinstance(result.exception, ValueError), result.report()
    assert target.flat.tobytes() == before
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not result.report()[key], result.report()


@pytest.mark.parametrize("schedule", ["constant", "linear", "cosine"])
def test_native_samba_update_matches_existing_optimizer_and_schedule(monkeypatch, schedule):
    stack = _stack()
    cls = {"constant": training.ConstantLR, "linear": training.WarmupLinearLR,
           "cosine": training.WarmupCosineLR}[schedule]
    stack.optimizer.lr_schedule = cls(0.003, warmup_steps=3, total_steps=17, min_lr=0.0001)
    state = stack.state_dict()
    before = {name: state[name].tobytes() for name in ARRAY_FIELDS}
    rng = np.random.default_rng(935)
    gradients = [rng.normal(0, 0.02, stack.shapes[name]).astype("f4") for name in stack.names]
    reference_gradients = [value.copy() for value in gradients]
    stack.optimizer.step(reference_gradients, max_norm=stack.max_norm)
    expected = _reference_state(stack)
    with monkeypatch.context() as patch:
        _poison(patch)
        patch.setattr(training.AdamW, "step", lambda *a, **k: pytest.fail("Python optimizer loop"))
        patch.setattr(training._Schedule, "lr_at", lambda *a: pytest.fail("Python schedule loop"))
        actual = _call("apply", state, gradients)
    _assert_state_equal(actual, expected)
    for name in ARRAY_FIELDS:
        assert state[name].tobytes() == before[name]
    assert state["t"] == 7 and actual["t"] == 8
    assert state["rng"] == actual["rng"]


def test_public_state_snapshot_restore_are_native_and_keep_aliases(monkeypatch):
    model = _stack()
    expected = _reference_state(model)
    with monkeypatch.context() as patch:
        _poison(patch)
        actual = audit_native_call(model.state_dict).assert_native()
    _assert_state_equal(actual, expected)
    old_flat = model.flat
    actual["parameters"] = np.asarray(actual["parameters"]).copy() + np.float32(0.015)
    actual["rng"]["counter"] = 29
    with monkeypatch.context() as patch:
        _poison(patch)
        assert audit_native_call(model.load_state_dict, actual).assert_native() is model
    assert model.flat is old_flat
    _assert_state_equal(_reference_state(model), actual)
