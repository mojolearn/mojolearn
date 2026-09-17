# SPDX-License-Identifier: Apache-2.0
"""Low-bit WEIGHT STORAGE for the inference classes (mojolearn.lowbit),
gated BITWISE: a block built from packed weights computes what the fp32
block computes from the exactly materialized weights, and the three
materialization spellings (GPU kernels, host binding, NumPy) agree.
Lane lane/identical-lowbit-inference, 2026-09-17.

Needs the identical GPU builds of the transformer, mamba, training and
linalg bindings; each test skips by name otherwise.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")

import mojolearn as ml  # noqa: E402
from mojolearn import _backend, lowbit  # noqa: E402


def _need(*modules):
    for name in modules:
        mod = getattr(ml, name, None)
        if mod is None or not callable(getattr(mod, "gpu_arch", None) if False else lambda: 0):
            pass
    try:
        ml.linalg.require_identical()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"identical linalg extension not loaded: {e}")


def _hw(shape, seed, lo=-0.125, hi=0.125):
    rng = np.random.default_rng(seed)
    return (lo + (hi - lo) * rng.random(shape)).astype(np.float32)


def _bits(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32)).view(np.uint32)


def _same(a, b):
    return np.array_equal(_bits(a), _bits(b))


@pytest.mark.parametrize("fmt", ["bfloat16", "int8"])
def test_pack_unpack_are_exact_and_the_three_spellings_agree(fmt):
    _need()
    w = {"a": _hw((40, 32), 1), "n": _hw((32,), 2), "b": _hw((32, 24), 3)}
    packed = lowbit.pack(w, fmt)
    assert lowbit.format_of(packed) == fmt
    assert lowbit.is_packed(packed["a"]) and not lowbit.is_packed(packed["n"])
    f32, got = lowbit.unpack(packed)
    assert got == fmt and f32["n"] is packed["n"]
    # idempotent: packing the materialized weights again changes nothing
    again = lowbit.pack(f32, fmt)
    for k in ("a", "b"):
        if fmt == "bfloat16":
            assert np.array_equal(again[k].bits, packed[k].bits)
        else:
            assert np.array_equal(again[k].codes, packed[k].codes)
            assert np.array_equal(again[k].exponents, packed[k].exponents)
    # the NumPy spelling of both conversions equals the GPU kernels' bits
    if fmt == "bfloat16":
        assert np.array_equal(lowbit._to_bf16_numpy(w["a"]), packed["a"].bits)
        assert _same(lowbit._from_bf16_numpy(packed["a"].bits), f32["a"])
    else:
        q, e = lowbit._quantize_int8_numpy(w["a"])
        assert np.array_equal(q, packed["a"].codes) and np.array_equal(e, packed["a"].exponents)
        assert _same(lowbit._dequantize_int8_numpy(q, e), f32["a"])
    # the host binding, when built, agrees too
    try:
        h = _backend.load_host_module("_mojolearn_linalg_host")
    except ImportError:
        return
    addr = lambda x: x.__array_interface__["data"][0]  # noqa: E731
    out = np.empty(w["a"].shape, np.float32)
    if fmt == "bfloat16":
        h.from_bf16(addr(out), addr(packed["a"].bits), [int(out.size)])
    else:
        h.dequantize_int8(addr(out), addr(packed["a"].codes), addr(packed["a"].exponents), [40, 32])
    assert _same(out, f32["a"])


def _transformer_weights(dm=32, nh=2, nkv=1, hd=16, it=64):
    return {
        "input_layernorm.weight": 1.0 + _hw((dm,), 11), "post_attention_layernorm.weight": 1.0 + _hw((dm,), 12),
        "q_proj.weight": _hw((nh * hd, dm), 13), "k_proj.weight": _hw((nkv * hd, dm), 14),
        "v_proj.weight": _hw((nkv * hd, dm), 15), "o_proj.weight": _hw((dm, nh * hd), 16),
        "gate_proj.weight": _hw((it, dm), 17), "up_proj.weight": _hw((it, dm), 18), "down_proj.weight": _hw((dm, it), 19),
    }


@pytest.mark.parametrize("fmt", ["bfloat16", "int8"])
def test_transformer_block_from_packed_weights_equals_fp32_on_materialized(fmt):
    _need()
    w = _transformer_weights()
    packed = lowbit.pack(w, fmt)
    f32, _ = lowbit.unpack(packed)
    x = _hw((2, 8, 32), 21, -1.0, 1.0)
    try:
        lo = ml.TransformerBlock(packed, n_heads=2, n_kv_heads=1)
    except ImportError as e:
        pytest.skip(str(e))
    hi = ml.TransformerBlock(f32, n_heads=2, n_kv_heads=1)
    assert lo.weight_format == fmt and hi.weight_format == "float32"
    assert _same(lo.forward(x), hi.forward(x))
    s1, s2 = lo.allocate_state(2, max_tokens=16), hi.allocate_state(2, max_tokens=16)
    assert _same(lo.forward(x, s1), hi.forward(x, s2))
    x1 = np.ascontiguousarray(x[:, :1, :])
    assert _same(lo.step(x1, s1), hi.step(x1, s2))
    # and the packed model is NOT the float32 model: the bits moved
    raw = ml.TransformerBlock(w, n_heads=2, n_kv_heads=1)
    assert not _same(lo.forward(x), raw.forward(x))


def _mamba1_weights(dm=32, di=64, r=2):
    return {"norm.weight": np.ones((dm,), np.float32), "in_proj.weight": _hw((2 * di, dm), 31),
            "conv1d.weight": _hw((di, 1, 4), 32), "conv1d.bias": _hw((di,), 33),
            "x_proj.weight": _hw((r + 32, di), 34), "dt_proj.weight": _hw((di, r), 35),
            "dt_proj.bias": _hw((di,), 36), "A_log": _hw((di, 16), 37), "D": _hw((di,), 38),
            "out_proj.weight": _hw((dm, di), 39)}


@pytest.mark.parametrize("fmt", ["bfloat16", "int8"])
def test_mamba1_block_from_packed_weights_equals_fp32_on_materialized(fmt):
    _need()
    w = _mamba1_weights()
    packed = lowbit.pack(w, fmt)
    f32, _ = lowbit.unpack(packed)
    x = _hw((2, 8, 32), 41, -1.0, 1.0)
    try:
        lo = ml.Mamba1Block(packed)
    except ImportError as e:
        pytest.skip(str(e))
    hi = ml.Mamba1Block(f32)
    assert lo.weight_format == fmt
    assert _same(lo.forward(x), hi.forward(x))
    s1, s2 = lo.allocate_state(2), hi.allocate_state(2)
    assert _same(lo.forward(x, s1), hi.forward(x, s2))
    x1 = np.ascontiguousarray(x[:, :1, :])
    assert _same(lo.step(x1, s1), hi.step(x1, s2))


@pytest.mark.parametrize("fmt", ["bfloat16", "int8"])
def test_mlp_inference_from_packed_weights(fmt):
    w = [((np.arange(int(np.prod(s)), dtype=np.float32) % 7 - 3) / 32).reshape(s).astype(np.float32)
         for s in ((16, 8), (16,), (3, 16), (3,))]
    packed = lowbit.pack({"weight1": w[0], "weight2": w[2]}, fmt)
    f32, _ = lowbit.unpack(packed)
    try:
        lo = ml.MLPInference(packed["weight1"], w[1], packed["weight2"], w[3])
        hi = ml.MLPInference(f32["weight1"], w[1], f32["weight2"], w[3])
    except ImportError as e:
        pytest.skip(str(e))
    assert lo.weight_format == fmt
    x = _hw((64, 8), 51, -1.0, 1.0)
    assert _same(lo.predict_logits(x), hi.predict_logits(x))
