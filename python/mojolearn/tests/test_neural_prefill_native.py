# SPDX-License-Identifier: Apache-2.0
"""Native immutable prefill ownership and retained no-tape arithmetic paths."""
from types import SimpleNamespace

import numpy as np
import pytest

from mojolearn import Mamba3Block
from mojolearn import _native_neural_bridge as bridge
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_causal_model_native_loading import _poison
from mojolearn.tests.test_mamba23_backward_surface import host_fixture
from mojolearn.tests.test_neural_backward_native import _case


@pytest.mark.parametrize("family", ["transformer", "mamba3"])
@pytest.mark.parametrize("disable_tape", [False, True])
def test_native_prefill_and_backward_match_existing_independent_entries(tmp_path, monkeypatch, family, disable_tape):
    if family == "transformer":
        block, inputs, gradient = _case(tmp_path, "llama")
    else:
        weights, inputs = host_fixture(3)
        block = Mamba3Block(weights)
        gradient = np.random.default_rng(521).normal(size=inputs.shape).astype("f4")
    expected_output = block.forward(inputs).tobytes()
    expected_gradients = {name: value.tobytes() for name, value in block.backward(inputs, gradient).items()}
    selected = bridge.CONTEXT["training_bindings"][block._native_route + ":identical"]
    binding_family = "transformer" if family == "transformer" else "mamba"
    if disable_tape:
        replacement = SimpleNamespace(**vars(selected[binding_family]))
        replacement.__dict__.pop(family + "_forward_tape_enabled", None)
        monkeypatch.setitem(selected, binding_family, replacement)
    with monkeypatch.context() as patch:
        _poison(patch)
        patch.setattr(type(block), "_extension", lambda *a: pytest.fail("Python binding selection"))
        patch.setattr(type(block), "forward", lambda *a, **k: pytest.fail("Python forward callback"))
        output, tape = audit_native_call(block._forward_with_tape, inputs).assert_native()
        if tape is None:
            gradients = audit_native_call(block.backward, inputs, gradient).assert_native()
        else:
            try:
                gradients = audit_native_call(block._backward_from_tape, tape, gradient).assert_native()
            finally:
                audit_native_call(block._close_forward_tape, tape).assert_native()
        if disable_tape:
            assert tape is None
    assert output.tobytes() == expected_output
    assert set(gradients) == set(expected_gradients)
    for name in gradients:
        assert gradients[name].tobytes() == expected_gradients[name], name
