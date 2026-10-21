# SPDX-License-Identifier: Apache-2.0
"""Whole-operation ownership checks against independent raw binding calls."""
import numpy as np
import pytest

from mojolearn import TransformerBlock, Mamba1Block, Mamba2Block
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_causal_model_native_loading import _poison
from mojolearn.tests.test_neural_block_native import block_inputs


def _case(tmp_path, family):
    _, weights, options, inputs = block_inputs(tmp_path, family)
    if family == "llama":
        block = TransformerBlock(weights, n_heads=2, n_kv_heads=1,
                                 numeric_profile="fp32_v1")
    elif family == "mamba":
        block = Mamba1Block(weights)
    else:
        block = Mamba2Block(weights, **options)
    gradient = np.random.default_rng(553).normal(size=inputs.shape).astype("f4")
    return block, inputs, gradient


def _raw_reference(block, inputs, gradient, family):
    # Testing-only controller: spell the established native ABI independently
    # of the new model coordinator, including fresh output owners.
    weights = [np.ascontiguousarray(value) for value in block._w]
    outputs = [np.empty_like(inputs)] + [np.empty_like(value) for value in weights]
    addresses = [inputs.ctypes.data] + [value.ctypes.data for value in weights]
    addresses += [gradient.ctypes.data] + [value.ctypes.data for value in outputs]
    params = [inputs.shape[0], inputs.shape[1], inputs.shape[2]]
    if family == "llama":
        params += [2, 1, 16, 64, 0]
        name = "transformer_backward"
    elif family == "mamba":
        name = "mamba1_backward"
    else:
        params += list(block.dt_limit)
        name = "mamba2_backward"
    getattr(block._extension(), name)(addresses, params)
    return {name: value.tobytes() for name, value in zip(("x",) + block._W_NAMES, outputs)}


@pytest.mark.parametrize("family", ["llama", "mamba", "mamba2"])
def test_backward_is_one_native_call_and_matches_raw_abi(tmp_path, monkeypatch, family):
    block, inputs, gradient = _case(tmp_path, family)
    expected = _raw_reference(block, inputs, gradient, family)
    with monkeypatch.context() as patch:
        _poison(patch)
        patch.setattr(type(block), "_extension", lambda *a: pytest.fail("Python binding selection"))
        actual = audit_native_call(block.backward, inputs, gradient).assert_native()
    assert set(actual) == set(expected)
    for name, raw in expected.items():
        assert actual[name].tobytes() == raw, name


@pytest.mark.parametrize("failure", ["gradient_dtype", "gradient_shape", "nonfinite", "weight_shape"])
def test_backward_refuses_bad_inputs_inside_native_boundary(tmp_path, monkeypatch, failure):
    block, inputs, gradient = _case(tmp_path, "mamba2")
    if failure == "gradient_dtype":
        gradient = gradient.astype("f8")
    elif failure == "gradient_shape":
        gradient = gradient[:, :-1].copy()
    elif failure == "nonfinite":
        gradient[0, 0, 0] = np.nan
    else:
        block._w[1] = np.zeros((1, 1), dtype="f4")
    with monkeypatch.context() as patch:
        _poison(patch)
        result = audit_native_call(block.backward, inputs, gradient)
    assert isinstance(result.exception, (TypeError, ValueError)), result.report()
    report = result.report()
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not report[key], report


def test_backward_preserves_extended_transformer_refusal(tmp_path):
    _, weights, options, inputs = block_inputs(tmp_path, "llama")
    # The shared Llama fixture uses every default option. Make the refusal
    # trigger explicit: its RMSNorm epsilon differs from the backward profile.
    options["norm_eps"] = 1e-5
    block = TransformerBlock(weights, **options)
    assert block._extended is True
    result = audit_native_call(block.backward, inputs, np.ones_like(inputs))
    assert isinstance(result.exception, NotImplementedError), result.report()
    assert "default float32 block options" in str(result.exception)


def test_mamba3_backward_is_native_and_matches_raw_abi(monkeypatch):
    from mojolearn import Mamba3Block
    from mojolearn.tests.test_mamba23_backward_surface import host_fixture
    weights, inputs = host_fixture(3)
    block = Mamba3Block(weights)
    gradient = np.random.default_rng(724).normal(size=inputs.shape).astype("f4")
    outputs = [np.empty_like(inputs)] + [np.empty_like(weights[name]) for name in block._W_NAMES]
    addresses = [inputs.ctypes.data] + [weights[name].ctypes.data for name in block._W_NAMES]
    addresses += [gradient.ctypes.data] + [value.ctypes.data for value in outputs]
    block._extension().mamba3_backward(addresses, list(inputs.shape))
    expected = {name: value.tobytes() for name, value in zip(("x",) + block._W_NAMES, outputs)}
    with monkeypatch.context() as patch:
        _poison(patch)
        patch.setattr(Mamba3Block, "_extension", lambda *a: pytest.fail("Python binding selection"))
        patch.setattr(Mamba3Block, "_prefill_backward", lambda *a: pytest.fail("Python backward loop"))
        actual = audit_native_call(block.backward, inputs, gradient).assert_native()
    assert set(actual) == set(expected)
    for name in expected:
        assert actual[name].tobytes() == expected[name], name
