# SPDX-License-Identifier: Apache-2.0
"""PyTorch conventions at the Python boundary (docs/api/TORCH_CONVENTIONS.md).

Device: only None, 'gpu' and 'cpu' (Andrew, 2026-11-21); PyTorch's device words are refused like any unknown value.
Neural blocks: `state_dict`,
`load_state_dict`, `parameters`, `named_parameters` and `to(device, dtype)` with PyTorch's meaning. The tests check
the boundary only (names, refusals, identity of the returned arrays, the route after `to`); the numerics of a rebuilt
block are the constructor's and are tested there.
"""
import numpy as np
import pytest

import mojolearn
from mojolearn._causal_lm_fixtures import family_fixture, _write_checkpoint
from mojolearn.models import CausalLM
from mojolearn.models import causal_lm

FAMILIES = ["llama", "mamba", "mamba2"]
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

@pytest.mark.parametrize("word", ["cuda", "cuda:0", "mps", "mps:0", "xpu"])
def test_pytorch_device_words_are_refused(tmp_path, word):
    cls, _, weights, kwargs, _ = block_inputs(tmp_path, "llama")
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        cls(weights, device=word, **kwargs)
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        cls(weights, device="cpu", **kwargs).to(word)


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
    assert iter(block.parameters()) is not None and not isinstance(block.parameters(), list)
    named = list(block.named_parameters())
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
    assert block.to("gpu") is block
    assert block.get_params()["device"] == "gpu"
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


# ---- round 2: Sequential, parameters() as torch's iterators --------------------

def _sequential_state():
    rng = np.random.default_rng(0)
    return {"0.weight": rng.standard_normal((8, 4), dtype=np.float32), "0.bias": rng.standard_normal(8, dtype=np.float32),
            "2.weight": rng.standard_normal((3, 8), dtype=np.float32), "2.bias": np.zeros(3, dtype=np.float32)}


def test_sequential_state_dict_round_trips_in_torch_layout():
    state = _sequential_state()
    model = mojolearn.Sequential().linear(8).relu().linear(3).load_state_dict(state)
    sd = model.state_dict()
    assert list(sd) == ["0.weight", "0.bias", "2.weight", "2.bias"]
    for name, value in state.items():
        assert sd[name].shape == value.shape and sd[name].dtype == np.float32
        assert sd[name].tobytes() == value.tobytes()
    again = mojolearn.Sequential().linear(8).relu().linear(3).load_state_dict(sd)
    assert np.asarray(again.params_).tobytes() == np.asarray(model.params_).tobytes()
    assert [a.shape for a in model.parameters()] == [v.shape for v in state.values()]
    assert [n for n, _ in model.named_parameters()] == list(state)


def test_sequential_state_dict_loads_into_torch():
    torch = pytest.importorskip("torch")
    model = mojolearn.Sequential().linear(8).relu().linear(3).load_state_dict(_sequential_state())
    net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3))
    net.load_state_dict({k: torch.from_numpy(np.array(v)) for k, v in model.state_dict().items()})
    X = np.random.default_rng(1).random((5, 4), dtype=np.float32)
    np.testing.assert_allclose(net(torch.from_numpy(X)).detach().numpy(), np.asarray(model.predict(X)), rtol=1e-5, atol=1e-6)


def test_sequential_state_dict_needs_parameters():
    with pytest.raises(ValueError, match="no parameters yet"):
        mojolearn.Sequential().linear(2).state_dict()


def test_sequential_to_takes_gpu_cpu_and_float32_only():
    model = mojolearn.Sequential().linear(8).relu().linear(3).load_state_dict(_sequential_state())
    assert model.to("cpu") is model and model.vendor_used() == "cpu"
    assert model.to(dtype="float32") is model
    with pytest.raises(ValueError, match="float32 only"):
        model.to(dtype="float64")
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None"):
        model.to("cuda")
    X = np.random.default_rng(1).random((5, 4), dtype=np.float32)
    on_cpu = np.asarray(model.predict(X)).tobytes()
    if HAS_GPU:
        assert model.to("gpu").vendor_used() == mojolearn.vendor()
        assert np.asarray(model.predict(X)).tobytes() == on_cpu


def test_causal_lm_parameters_are_torch_iterators(tmp_path):
    config, tensors = family_fixture("llama", False)
    _write_checkpoint(str(tmp_path), config, tensors)
    model = CausalLM.load(tmp_path, device="cpu", numeric_profile="fp32_v1")
    named = dict(model.named_parameters())
    assert "lm_head.weight" in named
    arrays = list(model.parameters())
    assert len(arrays) == len(named)


# ---- round 3: CausalLM state_dict / load_state_dict ---------------------------

def _causal_logits(model):
    ids = np.array([[1, 2, 3, 4]], dtype=np.int32)
    return np.asarray(model.forward(ids)).tobytes()


@pytest.mark.parametrize("family", FAMILIES)
def test_causal_lm_state_dict_is_named_parameters(tmp_path, family):
    config, tensors = family_fixture(family, False)
    _write_checkpoint(str(tmp_path), config, tensors)
    model = CausalLM.load(tmp_path, device="cpu", numeric_profile="fp32_v1")
    sd = model.state_dict()
    assert isinstance(sd, dict)
    assert sorted(sd) == sorted(dict(model.named_parameters()))


@pytest.mark.parametrize("family", FAMILIES)
def test_causal_lm_load_state_dict_round_trips_and_moves_the_weights(tmp_path, family):
    config, tensors = family_fixture(family, False)
    _write_checkpoint(str(tmp_path), config, tensors)
    model = CausalLM.load(tmp_path, device="cpu", numeric_profile="fp32_v1")
    before = _causal_logits(model)
    sd = {k: np.array(np.asarray(v), dtype=np.float32) for k, v in model.state_dict().items()}
    assert model.load_state_dict(sd) is None
    assert _causal_logits(model) == before
    name = sorted(sd)[0]
    sd[name] = sd[name] * np.float32(0.5)
    model.load_state_dict(sd)
    assert np.asarray(model.state_dict()[name]).tobytes() == sd[name].tobytes()
    # The same weights from a checkpoint give the same bits.
    fresh = CausalLM.load(tmp_path, device="cpu", numeric_profile="fp32_v1")
    fresh.load_state_dict(sd)
    assert _causal_logits(fresh) == _causal_logits(model)


def test_causal_lm_load_state_dict_is_strict_and_all_or_nothing(tmp_path):
    config, tensors = family_fixture("llama", False)
    _write_checkpoint(str(tmp_path), config, tensors)
    model = CausalLM.load(tmp_path, device="cpu", numeric_profile="fp32_v1")
    before = _causal_logits(model)
    sd = {k: np.array(np.asarray(v), dtype=np.float32) for k, v in model.state_dict().items()}
    name = sorted(sd)[0]
    with pytest.raises(ValueError, match="missing key"):
        model.load_state_dict({k: v for k, v in sd.items() if k != name})
    with pytest.raises(ValueError, match="unexpected key"):
        model.load_state_dict({**sd, "extra.weight": sd[name]})
    bad = dict(sd)
    bad[name] = np.zeros((1, 1), dtype=np.float32)
    with pytest.raises((ValueError, TypeError)):
        model.load_state_dict(bad)
    with pytest.raises(TypeError):
        model.load_state_dict(list(sd.items()))
    assert _causal_logits(model) == before


# ---- round 2: CNN layer state_dict / load_state_dict --------------------------

def test_conv_state_dict_round_trips_and_matches_torch_names():
    conv = mojolearn.Conv2d(3, 4, 3, device="cpu")
    sd = conv.state_dict()
    assert sorted(sd) == ["bias", "weight"] and sd["weight"].shape == (4, 3, 3, 3)
    other = mojolearn.Conv2d(3, 4, 3, device="cpu").load_state_dict(sd)
    assert np.asarray(other.weight_).tobytes() == sd["weight"].tobytes()
    with pytest.raises(ValueError, match="no entry 'bias'"):
        other.load_state_dict({"weight": sd["weight"]})
    with pytest.raises(ValueError, match="unexpected keys"):
        other.load_state_dict({**sd, "extra": sd["bias"]})


def test_conv1d_state_dict_is_torch_rank_three():
    conv = mojolearn.Conv1d(2, 5, 3, device="cpu")
    sd = conv.state_dict()
    assert sd["weight"].shape == (5, 2, 3)
    assert mojolearn.Conv1d(2, 5, 3, device="cpu").load_state_dict(sd).state_dict()["weight"].tobytes() == sd["weight"].tobytes()


def test_batchnorm_state_dict_has_torch_names():
    bn = mojolearn.BatchNorm2d(4, device="cpu")
    sd = bn.state_dict()
    assert sorted(sd) == ["bias", "num_batches_tracked", "running_mean", "running_var", "weight"]
    sd["running_mean"] = np.full(4, 0.5, dtype=np.float32)
    bn.load_state_dict(sd)
    assert np.asarray(bn.running_mean_).tobytes() == sd["running_mean"].tobytes()
    with pytest.raises(ValueError, match="num_features"):
        bn.load_state_dict({**sd, "weight": np.ones(3, dtype=np.float32)})


def test_basic_block_state_dict_uses_torchvision_names():
    block = mojolearn.BasicBlock(4, 4, device="cpu")
    sd = block.state_dict()
    assert "conv1.weight" in sd and "bn2.running_var" in sd and "conv1.bias" not in sd
    again = mojolearn.BasicBlock(4, 4, random_state=1, device="cpu").load_state_dict(sd)
    assert {k: np.asarray(v).tobytes() for k, v in again.state_dict().items()} == {k: np.asarray(v).tobytes() for k, v in sd.items()}
