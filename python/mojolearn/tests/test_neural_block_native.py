# SPDX-License-Identifier: Apache-2.0
"""Standalone native owners versus the retained independent block drivers."""
import os

import numpy as np
import pytest

from mojolearn import Array, _ragged
from mojolearn._causal_lm_fixtures import family_fixture, _write_checkpoint
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.models import CausalLM
from mojolearn.models import causal_lm
from mojolearn.tests.test_causal_model_native_loading import _poison

DEVICE = os.environ.get("MOJOLEARN_GENERATION_DEVICE", "cpu")


def block_inputs(tmp_path, family):
    config, tensors = family_fixture(family, False)
    _write_checkpoint(str(tmp_path), config, tensors)
    model = CausalLM.load(tmp_path, device=DEVICE, numeric_profile="fp32_v1")
    old = model.blocks[0]
    weights = dict(zip(old._W_NAMES, old._w))
    kwargs = {}
    if model.kind == "transformer":
        for name, value in zip(causal_lm._NATIVE_CONTEXT["optional_names"], old._wopt):
            if value is not None:
                weights[name] = value
        kwargs = dict(model.plan.block_options, n_heads=model.plan.n_heads, numeric_profile="fp32_v1")
    elif model.kind == "mamba2":
        kwargs = model.plan.block_kwargs
    cls = causal_lm._NATIVE_CONTEXT["block_classes"][DEVICE][model.kind]
    x = np.random.default_rng(123).normal(size=(2, 4, model.d_model)).astype("f4") * np.float32(0.05)
    return cls, weights, kwargs, x


@pytest.mark.parametrize("family", ["llama", "qwen2", "qwen3", "mistral", "mamba", "mamba2"])
def test_native_block_construct_forward_step_ragged(tmp_path, monkeypatch, family):
    cls, weights, kwargs, x = block_inputs(tmp_path, family)
    instance = cls.__new__(cls)
    constructor = cls.__init__
    with monkeypatch.context() as patch:
        _poison(patch)
        audit_native_call(constructor, instance, weights, **kwargs).assert_native()
    # These retained drivers independently spell kernel arguments and layout.
    expected = instance._call(x, None, step=False)
    wanted_ragged, _ = _ragged.ragged_forward(lambda xp: instance._call(xp, None, step=False),
                                             Array.from_buffer(x), None, [2, 4], "<f4", "test")
    with monkeypatch.context() as patch:
        _poison(patch)
        patch.setattr(cls, "_call", lambda *a, **k: pytest.fail("legacy driver executed"))
        actual = audit_native_call(instance.forward, x).assert_native()
        ragged = audit_native_call(instance.forward, x, lengths=[2, 4]).assert_native()
        state = audit_native_call(instance.allocate_state, 2, 8).assert_native() if family not in ("mamba", "mamba2") else audit_native_call(instance.allocate_state, 2).assert_native()
        prefix = audit_native_call(instance.forward, x[:, :3], state).assert_native()
        tail = audit_native_call(instance.step, x[:, 3], state).assert_native()
    assert actual.tobytes() == expected.tobytes()
    assert ragged.tobytes() == wanted_ragged.tobytes()
    joined = np.concatenate((np.asarray(prefix), np.asarray(tail)), axis=1)
    assert joined.tobytes() == expected.tobytes()


@pytest.mark.parametrize("bad", [[True, 2], [1.5, 2], [0, 2], [1, 5], [1]])
def test_native_lengths_keep_integer_and_range_refusals(tmp_path, bad):
    cls, weights, kwargs, x = block_inputs(tmp_path, "llama")
    block = cls(weights, **kwargs)
    with pytest.raises((TypeError, ValueError)):
        block.forward(x, lengths=bad)
