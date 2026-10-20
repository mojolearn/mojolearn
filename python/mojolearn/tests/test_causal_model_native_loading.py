# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-to-model operations contain no package constructor callbacks."""
import json
import os

import numpy as np
import pytest

from mojolearn import Array
from mojolearn._causal_lm_fixtures import family_fixture, _write_checkpoint
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.models import CausalLM, Checkpoint, HFConfig, plan_for
from mojolearn.models import causal_lm
from mojolearn import _backend, _buffer, _linalg_impl
from mojolearn._mode import PublicOutputsMixin

DEVICE = os.environ.get("MOJOLEARN_GENERATION_DEVICE", "cpu")


def _forbidden(*args, **kwargs):
    raise AssertionError("Python model-loading callback executed")


def _poison(patch):
    for name in ("__init__", "copy", "from_buffer", "from_list", "reshape"):
        patch.setattr(Array, name, _forbidden)
    patch.setattr(PublicOutputsMixin, "__getattribute__", _forbidden)
    for route in causal_lm._NATIVE_CONTEXT["block_classes"].values():
        for cls in route.values():
            patch.setattr(cls, "__init__", _forbidden)
    for name in ("_route", "_block_classes", "_block_kwargs", "_float_weight"):
        patch.setattr(causal_lm, name, _forbidden)
    for name in ("binding", "load_host_module", "default_mode", "vendor"):
        patch.setattr(_backend, name, _forbidden)
    for name in ("quantize_int8", "dequantize_int8", "quantize_int15", "to_bf16", "from_bf16"):
        patch.setattr(_linalg_impl, name, _forbidden)
    for name in ("empty", "as_f32_c", "addr", "addr_ro", "all_finite"):
        patch.setattr(_buffer, name, _forbidden)
    patch.setattr(json, "loads", _forbidden)
    patch.setattr(np, "empty", _forbidden)
    patch.setattr(np, "array", _forbidden)


@pytest.mark.parametrize("family", ["llama", "mistral", "qwen2", "qwen3", "phi3", "mamba", "mamba2"])
def test_native_load_all_mappings_against_checkpoint_bytes(tmp_path, monkeypatch, family):
    config, tensors = family_fixture(family, False)
    _write_checkpoint(str(tmp_path), config, tensors)
    # Independent byte oracle, including fused projection row slicing.
    plan = plan_for(HFConfig(config))
    expected = []
    for layer in range(plan.n_layers):
        values = {}
        for key, name, rows in plan.layer_weights(layer):
            dtype, shape, raw = tensors[name]
            assert dtype == "F32"
            if rows is not None:
                raw = raw[rows[0] * shape[1] * 4:rows[1] * shape[1] * 4]
            values[key] = raw
        expected.append(values)
    with monkeypatch.context() as patch:
        _poison(patch)
        model = audit_native_call(CausalLM.load, tmp_path, device=DEVICE,
                                  numeric_profile="fp32_v1").assert_native()
    assert model._native_vendor in ({"cpu"} if DEVICE == "cpu" else {"metal", "cuda", "hip"})
    assert model._native_numeric_mode == 1
    assert model.unused_names == []
    assert model._embed.tobytes() == tensors[plan.embed_name][2]
    for block, weights in zip(model.blocks, expected):
        for name, value in zip(block._W_NAMES, block._w):
            assert value.tobytes() == weights[name]
        if model.kind == "transformer":
            for name, value in zip(causal_lm._NATIVE_CONTEXT["optional_names"], block._wopt):
                assert (value is None) == (name not in weights)
                if value is not None:
                    assert value.tobytes() == weights[name]
    if plan.head_name is None:
        assert model._head is model._embed


@pytest.mark.parametrize("family", ["llama", "mamba", "mamba2"])
@pytest.mark.parametrize("format", ["bfloat16", "int8"])
def test_native_packed_constructor_matches_established_block_reference(tmp_path, monkeypatch, family, format):
    config, tensors = family_fixture(family, False)
    _write_checkpoint(str(tmp_path), config, tensors)
    with monkeypatch.context() as patch:
        _poison(patch)
        model = audit_native_call(CausalLM.load, tmp_path, device=DEVICE,
                                  weight_format=format, numeric_profile="fp32_v1").assert_native()
    assert all(block.weight_format == format for block in model.blocks)
    from mojolearn.tests.test_generation_native import block_reference
    ids = Array.from_list([[1, 4, 7]], "<i4")
    expected = block_reference(model, ids)
    actual = audit_native_call(model.forward, ids).assert_native()
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("profile", ["fp32_v1", "fixed15_v1"])
def test_direct_native_constructor_preserves_weights_and_fixed15(tmp_path, monkeypatch, profile):
    config, tensors = family_fixture("llama", True)
    _write_checkpoint(str(tmp_path), config, tensors)
    plan = plan_for(HFConfig(config))
    with Checkpoint.open(tmp_path) as checkpoint:
        weights = CausalLM._read_weights(checkpoint, plan, "float32")
    # An independent old quantizer provides the exact expected planes. CPU
    # tests use the retained CPU kernel directly, preventing a GPU head leak.
    context = causal_lm._NATIVE_CONTEXT
    mode = "identical"
    binding = context["bindings"][f"{DEVICE}:{mode}"]["linalg"]
    source = weights["embed"]
    rows, cols = source.shape
    hi = _buffer.empty((rows, cols), "<i1")
    lo = _buffer.empty((rows, cols), "<i1")
    exponents = _buffer.empty((rows,), "<i4")
    binding.quantize_int15([_buffer.addr(hi, name="hi"), _buffer.addr(lo, name="lo"), _buffer.addr(exponents, name="exponents"),
                           _buffer.addr_ro(source, name="source")], [rows, cols])
    model = CausalLM.__new__(CausalLM)
    with monkeypatch.context() as patch:
        _poison(patch)
        audit_native_call(CausalLM.__init__, model, plan, weights, device=DEVICE,
                          numeric_profile=profile).assert_native()
    assert model._embed is source and model._head is source
    if profile == "fixed15_v1":
        assert tuple(v.tobytes() for v in model._head_int15) == tuple(v.tobytes() for v in (hi, lo, exponents))
    else:
        assert model._head_int15 is None
