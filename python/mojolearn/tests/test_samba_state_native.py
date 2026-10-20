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
    expected = stack.state_dict()
    with monkeypatch.context() as patch:
        _poison(patch)
        patch.setattr(SambaStack, "state_dict", lambda *a, **k: pytest.fail("Python snapshot"))
        actual = _call("snapshot", stack, True)
        clone = _call("from_state", actual)
        resumed = _call("snapshot", clone, True)
    _assert_state_equal(actual, expected)
    _assert_state_equal(resumed, expected)
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
        state["rng"]["counter"] = 2 ** 32
    result = audit_native_call(bridge.RUNTIME.samba_native_call, "publish", (target, state), bridge.CONTEXT)
    assert isinstance(result.exception, ValueError), result.report()
    assert target.flat.tobytes() == before
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not result.report()[key], result.report()
