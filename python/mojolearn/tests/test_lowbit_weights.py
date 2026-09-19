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


# ---------------------------------------------------------------------------
# THE NUMPY ORACLE SPELLINGS. They lived in `mojolearn/lowbit.py` until the
# NumPy-free rewrite (lane/model-loader, 2026-09-17); the package now carries
# a pure-Python third spelling (`lowbit._to_bf16_py` and friends, the
# `checks/numerics.mojo` seams over Python ints), and these four stay here
# as the vectorized oracle the kernels, the host binding and that spelling
# are all held to. `_quantize_int8_numpy` propagates a NaN through the row
# absmax where `row_absmax` ignores it, so the oracle is compared on finite
# data only.
# ---------------------------------------------------------------------------


def _to_bf16_numpy(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    sub = ((u & 0x7F800000) == 0) & ((u & 0x007FFFFF) != 0)
    u = np.where(sub, u & 0x80000000, u).astype(np.uint32)
    nan = ((u & 0x7F800000) == 0x7F800000) & ((u & 0x007FFFFF) != 0)
    rne = ((u + np.uint32(0x7FFF) + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    quiet = ((u >> 16) | 0x0040).astype(np.uint16)
    return np.where(nan, quiet, rne).astype(np.uint16)


def _from_bf16_numpy(bits):
    return (np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def _quantize_int8_numpy(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    sub = ((u & 0x7F800000) == 0) & ((u & 0x007FFFFF) != 0)
    xf = np.where(sub, (u & 0x80000000).astype(np.uint32), u).view(np.float32)
    absmax = np.max(np.abs(xf), axis=1)
    ub = absmax.view(np.uint32)
    exp = (((ub >> 23) & 0xFF).astype(np.int32) - 127) - 6
    exp = np.where(absmax == 0, np.int32(0), exp).astype(np.int32)
    scaled = (xf * np.ldexp(np.float32(1.0), -exp)[:, None].astype(np.float32)).astype(np.float32)
    us = scaled.view(np.uint32)
    sub2 = ((us & 0x7F800000) == 0) & ((us & 0x007FFFFF) != 0)
    scaled = np.where(sub2, (us & 0x80000000).astype(np.uint32), us).view(np.float32)
    magic = np.float32(12582912.0)
    r = np.where(scaled >= 0, (scaled + magic) - magic, (scaled - magic) + magic).astype(np.float32)
    r = np.where(np.isnan(scaled), np.float32(0.0), r)
    r = np.clip(r, -127.0, 127.0)
    return r.astype(np.int8), exp


def _dequantize_int8_numpy(codes, exponents):
    y = (codes.astype(np.float32) * np.ldexp(np.float32(1.0), exponents)[:, None].astype(np.float32)).astype(np.float32)
    u = y.view(np.uint32)
    sub = ((u & 0x7F800000) == 0) & ((u & 0x007FFFFF) != 0)
    return np.where(sub, (u & 0x80000000).astype(np.uint32), u).view(np.float32)


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
        assert np.array_equal(_to_bf16_numpy(w["a"]), np.asarray(packed["a"].bits))
        assert _same(_from_bf16_numpy(np.asarray(packed["a"].bits)), f32["a"])
    else:
        q, e = _quantize_int8_numpy(w["a"])
        assert np.array_equal(q, np.asarray(packed["a"].codes)) and np.array_equal(e, np.asarray(packed["a"].exponents))
        assert _same(_dequantize_int8_numpy(q, e), f32["a"])
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


@pytest.mark.parametrize("fmt", ["bfloat16", "int8"])
def test_repack_same_format_copies_packed_bytes_without_materializing(monkeypatch, fmt):
    if fmt == "bfloat16":
        original = lowbit.BF16Weight(np.arange(24, dtype=np.uint16).reshape(6, 4))
    else:
        original = lowbit.Int8Weight(
            np.arange(24, dtype=np.int8).reshape(6, 4),
            np.arange(6, dtype=np.int32) - 3,
        )
    monkeypatch.setattr(lowbit, "materialize_one", lambda *a, **k: pytest.fail("materialized"))
    copied = lowbit.pack({"w": original}, fmt)["w"]
    assert copied is not original
    if fmt == "bfloat16":
        assert np.array_equal(copied.bits, original.bits)
        assert copied.bits._addr != original.bits._addr
    else:
        assert np.array_equal(copied.codes, original.codes)
        assert np.array_equal(copied.exponents, original.exponents)
        assert copied.codes._addr != original.codes._addr
        assert copied.exponents._addr != original.exponents._addr
def _special_rows():
    """Finite float32 rows that reach every branch of the seams: zeros, a
    subnormal (flushed), negatives, exact halves (ties to even), a tiny row,
    a large row and one plain row."""
    rows = [
        [0.0, -0.0, 0.0, 0.0],
        [1e-40, -1e-40, 1.0, -1.0],
        [0.5, 1.5, 2.5, -2.5],
        [1e-30, 2e-30, -3e-30, 4e-30],
        [3e38, -1e38, 1.0, 2.0],
        [0.1, -0.2, 0.3, -0.4],
        [65504.0, 0.000030517578125, -0.000030517578125, 100.0],
    ]
    return np.array(rows, dtype=np.float32)


def test_pure_python_spelling_equals_numpy_oracle():
    """No binding needed: the pure-Python third spelling (`checks/numerics.mojo`'s
    seams over Python ints) equals the vectorized oracle bit for bit on
    hashed and on hand-picked finite data, and `pack`/`unpack` route through
    it on an install with neither the linalg kernels nor the host binding."""
    from mojolearn import Array
    for w in (_hw((40, 32), 7), _special_rows(), _hw((3, 5), 9, -1e3, 1e3)):
        a = Array.from_buffer(np.ascontiguousarray(w))
        bits = lowbit._to_bf16_py(a)
        assert bits.dtype == "<u2" and np.array_equal(np.asarray(bits), _to_bf16_numpy(w))
        back = lowbit._from_bf16_py(bits)
        assert back.dtype == "<f4" and _same(back, _from_bf16_numpy(_to_bf16_numpy(w)))
        q, e = lowbit._quantize_int8_py(a)
        qn, en = _quantize_int8_numpy(w)
        assert np.array_equal(np.asarray(q), qn) and np.array_equal(np.asarray(e), en)
        deq = lowbit._dequantize_int8_py(q, e)
        assert _same(deq, _dequantize_int8_numpy(qn, en))
    # the packed containers are Arrays, never NumPy
    packed = lowbit.pack({"a": _hw((8, 4), 3), "n": _hw((4,), 4)}, "int8")
    assert isinstance(packed["a"].codes, Array) and isinstance(packed["a"].exponents, Array)
    assert lowbit.is_packed(packed["a"]) and not lowbit.is_packed(packed["n"])
    packed = lowbit.pack({"a": _hw((8, 4), 3)}, "bfloat16")
    assert isinstance(packed["a"].bits, Array) and packed["a"].bits.dtype == "<u2"
    f32, fmt = lowbit.unpack(packed)
    assert fmt == "bfloat16" and isinstance(f32["a"], Array) and f32["a"].dtype == "<f4"


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
