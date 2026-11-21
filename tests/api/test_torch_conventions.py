# SPDX-License-Identifier: Apache-2.0
"""PyTorch conventions at the Python boundary (docs/api/TORCH_CONVENTIONS.md).

Device words: `cuda` / `cuda:0` on Linux and `mps` / `mps:0` on macOS name the GPU route of `device=`; the other
OS's word, another GPU index and a `torch.device` object are refused in words. Neural blocks: `state_dict`,
`load_state_dict`, `parameters`, `named_parameters` and `to(device, dtype)` with PyTorch's meaning. The tests check
the boundary only (names, refusals, identity of the returned arrays, the route after `to`); the numerics of a rebuilt
block are the constructor's and are tested there.
"""
import platform

import numpy as np
import pytest

import mojolearn
from mojolearn._causal_lm_fixtures import family_fixture, _write_checkpoint
from mojolearn.models import CausalLM
from mojolearn.models import causal_lm

FAMILIES = ["llama", "mamba", "mamba2"]
MACOS = platform.system() == "Darwin"
GPU_WORDS = ["mps", "mps:0"] if MACOS else ["cuda", "cuda:0"]
OTHER_OS_WORDS = ["cuda", "cuda:0"] if MACOS else ["mps", "mps:0"]
HAS_GPU = mojolearn.vendor() != "cpu"
needs_gpu = pytest.mark.skipif(not HAS_GPU, reason="a CPU-only installation has no GPU route")


def block_inputs(tmp_path, family):
    """The GPU block class of `family`, its weights, its keyword arguments and an input."""
    config, tensors = family_fixture(family, False)
    _write_checkpoint(str(tmp_path), config, tensors)
    model = CausalLM.load(tmp_path, device="cpu", numeric_profile="fp32_v1")
    old = model.blocks[0]
    weights = dict(zip(old._W_NAMES, old._w))
    kwargs = {}
    if model.kind == "transformer":
        for name, value in zip(causal_lm._NATIVE_CONTEXT["optional_names"], old._wopt):
            if value is not None:
                weights[name] = value
        kwargs = dict(model.plan.block_options, n_heads=model.plan.n_heads, numeric_profile="fp32_v1")
    elif model.kind == "mamba2":
        kwargs = dict(model.plan.block_kwargs)
    cls = causal_lm._NATIVE_CONTEXT["block_classes"]["gpu"][model.kind]
    cpu_cls = causal_lm._NATIVE_CONTEXT["block_classes"]["cpu"][model.kind]
    x = np.random.default_rng(7).normal(size=(2, 4, model.d_model)).astype("f4") * np.float32(0.05)
    return cls, cpu_cls, weights, kwargs, x


# ---- device words -----------------------------------------------------------

@needs_gpu
@pytest.mark.parametrize("word", GPU_WORDS)
def test_pytorch_gpu_word_is_the_gpu_route(tmp_path, word):
    cls, _, weights, kwargs, x = block_inputs(tmp_path, "llama")
    block = cls(weights, device=word, **kwargs)
    assert block.get_params()["device"] == word
    assert block.vendor_used() == mojolearn.vendor()
    gpu = cls(weights, device="gpu", **kwargs)
    assert np.asarray(block.forward(x)).tobytes() == np.asarray(gpu.forward(x)).tobytes()


@pytest.mark.parametrize("word", OTHER_OS_WORDS + ["cuda:1", "mps:1", "xpu", "meta"])
def test_device_words_without_a_device_are_refused_in_words(tmp_path, word):
    cls, _, weights, kwargs, _ = block_inputs(tmp_path, "llama")
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        cls(weights, device=word, **kwargs)


def test_refusal_names_the_pytorch_word_of_this_os(tmp_path):
    cls, _, weights, kwargs, _ = block_inputs(tmp_path, "llama")
    with pytest.raises(ValueError, match="'mps'" if MACOS else "'cuda'"):
        cls(weights, device=OTHER_OS_WORDS[0], **kwargs)
    with pytest.raises(ValueError, match="VISIBLE_DEVICES"):
        cls(weights, device=("mps:1" if MACOS else "cuda:1"), **kwargs)


def test_torch_device_object_is_refused_with_the_string_hint(tmp_path):
    torch = pytest.importorskip("torch")
    cls, _, weights, kwargs, _ = block_inputs(tmp_path, "llama")
    with pytest.raises(ValueError, match="str"):
        cls(weights, device=torch.device("cpu"), **kwargs)


@needs_gpu
@pytest.mark.parametrize("word", GPU_WORDS)
def test_pytorch_gpu_word_reaches_estimators(word):
    est = mojolearn.Ridge(device=word)
    assert est.get_params()["device"] == word
    X = np.random.default_rng(0).random((64, 3), dtype=np.float32)
    y = X @ np.array([1.0, -2.0, 0.5], dtype=np.float32)
    got = np.asarray(est.fit(X, y).predict(X)).tobytes()
    assert got == np.asarray(mojolearn.Ridge(device="gpu").fit(X, y).predict(X)).tobytes()


# ---- neural block state ------------------------------------------------------

@pytest.mark.parametrize("family", FAMILIES)
def test_state_dict_is_the_weights_by_name_over_the_same_arrays(tmp_path, family):
    cls, _, weights, kwargs, _ = block_inputs(tmp_path, family)
    block = cls(weights, device="cpu", **kwargs)
    sd = block.state_dict()
    assert type(sd) is dict and sd is not weights
    assert list(sd) == list(weights)
    assert all(sd[k] is weights[k] for k in weights)
    assert [p is weights[k] for p, k in zip(block.parameters(), weights)] == [True] * len(weights)
    named = block.named_parameters()
    assert [n for n, _ in named] == list(weights)
    assert all(a is weights[n] for n, a in named)


@pytest.mark.parametrize("family", FAMILIES)
def test_load_state_dict_rebuilds_and_returns_the_block(tmp_path, family):
    cls, _, weights, kwargs, x = block_inputs(tmp_path, family)
    block = cls(weights, device="cpu", **kwargs)
    before = np.asarray(block.forward(x)).tobytes()
    scaled = {k: (v * np.float32(0.5)).astype("f4") for k, v in weights.items()}
    assert block.load_state_dict(scaled) is block
    assert block.state_dict()[next(iter(scaled))] is scaled[next(iter(scaled))]
    assert np.asarray(block.forward(x)).tobytes() == np.asarray(cls(scaled, device="cpu", **kwargs).forward(x)).tobytes()
    assert block.load_state_dict(weights).forward(x) is not None
    assert np.asarray(block.forward(x)).tobytes() == before


@pytest.mark.parametrize("family", FAMILIES)
def test_load_state_dict_refuses_incomplete_and_non_strict(tmp_path, family):
    cls, _, weights, kwargs, x = block_inputs(tmp_path, family)
    block = cls(weights, device="cpu", **kwargs)
    before = np.asarray(block.forward(x)).tobytes()
    partial = dict(list(weights.items())[1:])
    with pytest.raises((ValueError, KeyError)):
        block.load_state_dict(partial)
    with pytest.raises(ValueError, match="strict=True"):
        block.load_state_dict(weights, strict=False)
    with pytest.raises(TypeError):
        block.load_state_dict(list(weights.items()))
    assert np.asarray(block.forward(x)).tobytes() == before


# ---- to(device, dtype) ---------------------------------------------------------

@pytest.mark.parametrize("family", FAMILIES)
def test_to_float32_is_accepted_and_other_dtypes_refused(tmp_path, family):
    cls, _, weights, kwargs, _ = block_inputs(tmp_path, family)
    block = cls(weights, device="cpu", **kwargs)
    assert block.to("float32") is block
    assert block.to(dtype=np.dtype("float32")) is block
    for bad in ("float16", "bfloat16", "float64", np.dtype("float64")):
        with pytest.raises(ValueError, match="float32 only"):
            block.to(dtype=bad)
    with pytest.raises(TypeError):
        block.to(memory_format="channels_last")


def test_to_accepts_torch_dtypes(tmp_path):
    torch = pytest.importorskip("torch")
    cls, _, weights, kwargs, _ = block_inputs(tmp_path, "llama")
    block = cls(weights, device="cpu", **kwargs)
    assert block.to(torch.float32) is block
    assert block.to("cpu", torch.float32, non_blocking=True) is block
    with pytest.raises(ValueError, match="float32 only"):
        block.to(torch.bfloat16)


@needs_gpu
@pytest.mark.parametrize("family", FAMILIES)
def test_to_moves_the_block_between_routes(tmp_path, family):
    cls, _, weights, kwargs, x = block_inputs(tmp_path, family)
    block = cls(weights, device="cpu", **kwargs)
    assert block.vendor_used() == "cpu"
    assert block.to(GPU_WORDS[0]) is block
    assert block.get_params()["device"] == GPU_WORDS[0]
    assert block.vendor_used() == mojolearn.vendor()
    on_gpu = np.asarray(block.forward(x)).tobytes()
    assert block.to("cpu").vendor_used() == "cpu"
    assert np.asarray(block.forward(x)).tobytes() == on_gpu


@pytest.mark.parametrize("family", FAMILIES)
def test_inference_twin_refuses_a_gpu_word_in_to(tmp_path, family):
    _, cpu_cls, weights, kwargs, _ = block_inputs(tmp_path, family)
    block = cpu_cls(weights, **kwargs)
    assert block.to("cpu") is block
    with pytest.raises(ValueError):
        block.to("gpu")


def test_blocks_built_by_a_whole_model_say_so(tmp_path):
    config, tensors = family_fixture("llama", False)
    _write_checkpoint(str(tmp_path), config, tensors)
    model = CausalLM.load(tmp_path, device="cpu", numeric_profile="fp32_v1")
    with pytest.raises(RuntimeError, match="whole model"):
        model.blocks[0].state_dict()
