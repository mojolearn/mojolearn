# SPDX-License-Identifier: Apache-2.0
"""`device=None` (default), `device="gpu"` and `device="cpu"` on the block classes, at the Python boundary.

TransformerBlock, Mamba1Block, Mamba2Block and Mamba3Block take `device=` with the rule and the words of Ridge and
LinearRegression (test_linear_device_boundary.py): None is this installation's device, an unknown value is refused,
an explicit `device="gpu"` on a CPU-only installation is refused in words that name `device="cpu"`. The value is read
once, at construction, into the block's route; `get_params`, `set_params`, clone and pickle carry it. On a GPU
installation `device="cpu"` runs the host kernels (`vendor_used() == "cpu"`) and returns
the bits the GPU block returns; `backward` on it is refused in the shared words. What needs two devices is
skipped by name on a CPU-only installation.
"""
import pickle

import numpy as np
import pytest

import mojolearn
from mojolearn._causal_lm_fixtures import family_fixture, _write_checkpoint
from mojolearn.models import CausalLM
from mojolearn.models import causal_lm

FAMILIES = ["llama", "mamba", "mamba2"]
TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")


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
    cls = causal_lm._NATIVE_CONTEXT["block_classes"][model.kind]
    x = np.random.default_rng(123).normal(size=(2, 4, model.d_model)).astype("f4") * np.float32(0.05)
    return cls, weights, kwargs, x


@pytest.mark.parametrize("family", FAMILIES)
def test_device_is_a_constructor_parameter_with_default_none(tmp_path, family):
    cls, weights, kwargs, _ = block_inputs(tmp_path, family)
    assert "device" in cls._get_param_names()
    assert cls(weights, **kwargs).get_params()["device"] is None
    assert cls(weights, device=None, **kwargs).get_params()["device"] is None
    assert cls(weights, device="cpu", **kwargs).get_params()["device"] == "cpu"


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("bad", ["tpu", "GPU", " cpu", "", "None", "auto", 0, False, b"cpu"])
def test_unknown_device_is_refused_in_words_at_construction(tmp_path, family, bad):
    cls, weights, kwargs, _ = block_inputs(tmp_path, family)
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        cls(weights, device=bad, **kwargs)


@pytest.mark.parametrize("family", FAMILIES)
def test_default_device_is_this_installations_device(tmp_path, family):
    cls, weights, kwargs, x = block_inputs(tmp_path, family)
    block = cls(weights, **kwargs)
    assert block.vendor_used() == mojolearn.vendor()
    assert block.numeric_mode_used() in ("identical", "fast")
    assert np.asarray(block.forward(x)).shape == x.shape


@pytest.mark.parametrize("family", FAMILIES)
def test_explicit_gpu_runs_on_a_gpu_installation_and_is_refused_in_words_on_a_cpu_only_one(tmp_path, family):
    cls, weights, kwargs, x = block_inputs(tmp_path, family)
    if TWO_DEVICES:
        block = cls(weights, device="gpu", **kwargs)
        assert block.vendor_used() == mojolearn.vendor() and block.get_params()["device"] == "gpu"
        return
    with pytest.raises(ValueError, match="device='gpu' needs a GPU build of mojolearn.*Use device='cpu'"):
        cls(weights, device="gpu", **kwargs)
    block = cls(weights, **kwargs)
    with pytest.raises(ValueError, match="Use device='cpu'"):
        block.set_params(device="gpu")
    assert block.get_params()["device"] is None
    assert np.asarray(block.forward(x)).shape == x.shape      # the refused set_params left the block whole


@pytest.mark.parametrize("family", FAMILIES)
def test_set_params_refusal_leaves_the_block_unchanged(tmp_path, family):
    cls, weights, kwargs, x = block_inputs(tmp_path, family)
    block = cls(weights, device="cpu", **kwargs)
    before = np.asarray(block.forward(x)).tobytes()
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        block.set_params(device="tpu")
    assert block.get_params()["device"] == "cpu" and block.vendor_used() == "cpu"
    assert np.asarray(block.forward(x)).tobytes() == before


@pytest.mark.parametrize("family", FAMILIES)
def test_cpu_device_runs_the_host_kernels(tmp_path, family):
    # The CPU-by-class `*BlockInference` twins this compared against were
    # removed 2026-11-22 (same kernels, same bits); the whole model's CPU
    # route builds the same class on the CPU.
    cls, weights, kwargs, x = block_inputs(tmp_path, family)
    block = cls(weights, device="cpu", **kwargs)
    again = cls(weights, device="cpu", **kwargs)
    assert block.vendor_used() == "cpu" and block.numeric_mode_used() == "identical"
    assert np.asarray(block.forward(x)).tobytes() == np.asarray(again.forward(x)).tobytes()
    assert np.asarray(block.forward(x, lengths=[2, 4])).tobytes() == np.asarray(again.forward(x, lengths=[2, 4])).tobytes()


@two_devices
@pytest.mark.parametrize("family", FAMILIES)
def test_two_blocks_on_two_devices_interleaved_return_identical_bits(tmp_path, family):
    cls, weights, kwargs, x = block_inputs(tmp_path, family)
    gpu = cls(weights, numeric_mode="identical", **kwargs)
    cpu = cls(weights, numeric_mode="identical", device="cpu", **kwargs)
    assert gpu.vendor_used() == mojolearn.vendor() and cpu.vendor_used() == "cpu"
    for _ in range(3):                                        # neither block's owner record serves the other
        assert np.asarray(gpu.forward(x)).tobytes() == np.asarray(cpu.forward(x)).tobytes()
    states = [gpu.allocate_state(2, 8) if family == "llama" else gpu.allocate_state(2),
              cpu.allocate_state(2, 8) if family == "llama" else cpu.allocate_state(2)]
    for t in range(x.shape[1]):
        assert (np.asarray(gpu.step(x[:, t], states[0])).tobytes()
                == np.asarray(cpu.step(x[:, t], states[1])).tobytes())


@two_devices
@pytest.mark.parametrize("family", FAMILIES)
def test_set_params_moves_the_block_to_the_other_device(tmp_path, family):
    cls, weights, kwargs, x = block_inputs(tmp_path, family)
    block = cls(weights, numeric_mode="identical", **kwargs)
    on_gpu = np.asarray(block.forward(x)).tobytes()
    assert block.set_params(device="cpu") is block
    assert block.vendor_used() == "cpu" and block.get_params()["device"] == "cpu"
    assert np.asarray(block.forward(x)).tobytes() == on_gpu
    block.set_params(device=None)
    assert block.vendor_used() == mojolearn.vendor()


@two_devices
@pytest.mark.parametrize("family", FAMILIES)
def test_device_is_the_authority_over_state_built_for_another_device(tmp_path, family):
    """A block whose fields were handed over from the same block on the other device (what `check` does) runs its
    own device: the constructor parameter wins and the derived state is rebuilt once."""
    cls, weights, kwargs, x = block_inputs(tmp_path, family)
    gpu = cls(weights, numeric_mode="identical", **kwargs)
    wanted = np.asarray(gpu.forward(x)).tobytes()
    twin = cls(weights, numeric_mode="identical", device="cpu", **kwargs)
    params = set(twin.get_params())
    state = object.__getattribute__(gpu, "__dict__")
    target = object.__getattribute__(twin, "__dict__")
    for key, value in state.items():
        if key not in params:
            target[key] = value
    assert np.asarray(twin.forward(x)).tobytes() == wanted
    assert twin.vendor_used() == "cpu" and gpu.vendor_used() == mojolearn.vendor()
    assert np.asarray(gpu.forward(x)).tobytes() == wanted     # the GPU block kept its own state


@two_devices
@pytest.mark.parametrize("family", FAMILIES)
def test_cpu_with_fast_mode_is_refused_in_words(tmp_path, family):
    cls, weights, kwargs, _ = block_inputs(tmp_path, family)
    with pytest.raises(ValueError, match="device='cpu' runs in identical mode only"):
        cls(weights, numeric_mode="fast", device="cpu", **kwargs)


@two_devices
@pytest.mark.parametrize("family", FAMILIES)
def test_backward_on_the_cpu_device_is_refused_in_words_on_a_gpu_installation(tmp_path, family):
    cls, weights, kwargs, x = block_inputs(tmp_path, family)
    block = cls(weights, device="cpu", **kwargs)
    with pytest.raises(ValueError, match="device='cpu' is not available for backward"):
        block.backward(x, np.ones_like(x))


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("device", [None, "cpu"])
def test_pickle_carries_the_device(tmp_path, family, device):
    cls, weights, kwargs, x = block_inputs(tmp_path, family)
    block = cls(weights, device=device, **kwargs)
    wanted = np.asarray(block.forward(x)).tobytes()
    restored = pickle.loads(pickle.dumps(block))
    assert restored.get_params()["device"] == device
    assert restored.vendor_used() == block.vendor_used()
    assert np.asarray(restored.forward(x)).tobytes() == wanted


@pytest.mark.parametrize("family", FAMILIES)
def test_clone_through_get_params_carries_the_device(tmp_path, family):
    cls, weights, kwargs, x = block_inputs(tmp_path, family)
    block = cls(weights, device="cpu", **kwargs)
    twin = cls(**block.get_params(deep=False))
    assert twin.get_params()["device"] == "cpu" and twin.vendor_used() == "cpu"
    assert np.asarray(twin.forward(x)).tobytes() == np.asarray(block.forward(x)).tobytes()


@two_devices
@pytest.mark.parametrize("family", FAMILIES)
def test_check_runs_the_block_on_both_devices_in_this_process(tmp_path, family, capsys):
    cls, weights, kwargs, x = block_inputs(tmp_path, family)
    block = cls(weights, numeric_mode="identical", **kwargs)
    result = block.check(x, method="forward")
    capsys.readouterr()
    assert result["ran"], result.get("reason")
    assert result["transfer"] == "device"
    assert result["verdict"]["verdict"] == "MATCH"
    sides = result["sides"]
    assert sides["gpu"]["route"]["vendor_used"] == mojolearn.vendor()
    assert sides["cpu"]["route"]["vendor_used"] == "cpu"


# moved from python/mojolearn/tests/test_neural_inference.py
# TransformerBlock and Mamba blocks with device="cpu": rows alone, the causal
# prefix, ragged lengths, returned shape/dtype and the refusals in words.
import numpy as np
import pytest

import mojolearn as ml
from mojolearn import _backend

_NINFB_T_NAMES = ("input_layernorm.weight", "post_attention_layernorm.weight", "q_proj.weight",
                  "k_proj.weight", "v_proj.weight", "o_proj.weight", "gate_proj.weight",
                  "up_proj.weight", "down_proj.weight")


@pytest.fixture(scope="module")
def ninfb_binding():
    try:
        return _backend.load_host_module("_mojolearn_neural_host")
    except ImportError as exc:
        pytest.skip(f"neural host binding not built: {exc}")


def _ninfb_block_weights(dm=32, nh=2, nkv=1, it=64, seed=1):
    hd = dm // nh
    shapes = ((dm,), (dm,), (nh * hd, dm), (nkv * hd, dm), (nkv * hd, dm), (dm, nh * hd), (it, dm), (it, dm), (dm, it))
    rng = np.random.default_rng(seed)
    return {n: (rng.standard_normal(s) * 0.1).astype(np.float32) for n, s in zip(_NINFB_T_NAMES, shapes)}


def test_transformer_forward_rows_ragged_and_refusals(ninfb_binding):
    tw = _ninfb_block_weights()
    for window in (0, 8):
        blk = ml.TransformerBlock(tw, n_heads=2, n_kv_heads=1, window=window, device="cpu")
        x = np.random.default_rng(5).standard_normal((4, 16, 32)).astype(np.float32)
        y = np.asarray(blk.forward(x))
        assert y.shape == (4, 16, 32)
        # rows alone and the causal prefix: the native check training_neural_inference
        lengths = [16, 7, 1, 12]
        yr = np.asarray(blk.forward(x, lengths=lengths))
        for i, n in enumerate(lengths):
            alone = np.asarray(blk.forward(np.ascontiguousarray(x[i:i + 1, :n])))
            assert yr[i:i + 1, :n].tobytes() == alone.tobytes(), (window, i)
            assert not yr[i, n:].any()
    blk = ml.TransformerBlock(tw, n_heads=2, n_kv_heads=1, device="cpu")
    with pytest.raises(ValueError, match="state is required"):
        blk.step(np.zeros((1, 1, 32), np.float32), None)
    with pytest.raises(ValueError, match="d_model must equal n_heads"):
        ml.TransformerBlock(tw, n_heads=3, device="cpu")


def _ninfb_mamba_weights(kind, dm=32, seed=21):
    di = 2 * dm
    rng = np.random.default_rng(seed)
    if kind == "mamba1":
        r = -(-dm // 16)
        shapes = {"norm.weight": (dm,), "in_proj.weight": (2 * di, dm), "conv1d.weight": (di, 1, 4),
                  "conv1d.bias": (di,), "x_proj.weight": (r + 32, di), "dt_proj.weight": (di, r),
                  "dt_proj.bias": (di,), "A_log": (di, 16), "D": (di,), "out_proj.weight": (dm, di)}
        ones = ("norm.weight",)
    elif kind == "mamba2":
        nh = di // 64
        cd, dip = di + 256, 2 * di + 256 + nh
        shapes = {"block_norm.weight": (dm,), "in_proj.weight": (dip, dm), "conv1d.weight": (cd, 1, 4),
                  "conv1d.bias": (cd,), "dt_bias": (nh,), "A_log": (nh,), "D": (nh,), "norm.weight": (di,),
                  "out_proj.weight": (dm, di)}
        ones = ("block_norm.weight", "norm.weight")
    else:
        nh = di // 64
        shapes = {"block_norm.weight": (dm,), "in_proj.weight": (2 * di + 256 + 3 * nh + 32, dm),
                  "dt_bias": (nh,), "B_norm.weight": (128,), "C_norm.weight": (128,), "B_bias": (nh, 128),
                  "C_bias": (nh, 128), "D": (nh,), "out_proj.weight": (dm, di)}
        ones = ("block_norm.weight", "B_norm.weight", "C_norm.weight")
    return {n: (np.ones(s, np.float32) if n in ones else (rng.standard_normal(s) * 0.1).astype(np.float32))
            for n, s in shapes.items()}


_NINFB_MAMBA = (("mamba1", "Mamba1Block", {"device": "cpu"}), ("mamba2", "Mamba2Block", {"device": "cpu"}),
                ("mamba2", "Mamba2Block", {"dt_limit": (0.01, 0.1), "device": "cpu"}),
                ("mamba3", "Mamba3Block", {"device": "cpu"}))


@pytest.mark.parametrize("kind,cls,kw", _NINFB_MAMBA)
def test_mamba_forward_rows_prefix_ragged_and_refusals(ninfb_binding, kind, cls, kw):
    blk = getattr(ml, cls)(_ninfb_mamba_weights(kind), **kw)
    x = np.random.default_rng(3).standard_normal((4, 16, 32)).astype(np.float32)
    y = np.asarray(blk.forward(x))
    assert y.shape == (4, 16, 32) and y.dtype == np.float32
    for i in range(4):
        assert np.asarray(blk.forward(x[i:i + 1])).tobytes() == y[i:i + 1].tobytes(), i
    assert np.asarray(blk(np.ascontiguousarray(x[:, :7]))).tobytes() == np.ascontiguousarray(y[:, :7]).tobytes()
    lengths = [16, 7, 1, 12]
    yr = np.asarray(blk.forward(x, lengths=lengths))
    for i, n in enumerate(lengths):
        assert yr[i:i + 1, :n].tobytes() == np.asarray(blk.forward(np.ascontiguousarray(x[i:i + 1, :n]))).tobytes()
        assert not yr[i, n:].any()
    with pytest.raises(ValueError, match="state is required"):
        blk.step(np.ascontiguousarray(x[:, :1]), None)
