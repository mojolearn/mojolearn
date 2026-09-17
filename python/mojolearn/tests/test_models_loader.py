# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The checkpoint loader (`mojolearn.models`, lane/model-loader, 2026-09-17)
on SYNTHETIC checkpoints only: every test writes a tiny config, shards and
tokenizer.json to a temp directory (nothing is downloaded; no real
checkpoint is loaded, that is lane C's run) and asserts

  - the safetensors reader: every dtype it admits round-trips bit for bit
    (F16 and BF16 widened against a struct/shift oracle), the sharded index
    is followed, an unknown dtype is refused by the tensor's name;
  - the option matrix: a Llama-shaped config plans the right shapes and
    tensor names, phi3's fused tensors split by rows, and every refusal
    names the model_type, the field and its value;
  - the live-signature rule: an option the block does not accept is refused
    by name unless its value is what today's block fixes;
  - loading: a two-layer Llama-shaped and a Mamba-shaped checkpoint assemble
    (block shapes, name mapping, nothing unused), `weight_format="bfloat16"`
    reads on every block, and greedy `generate` returns the same ids twice;
  - the tokenizer: the Llama 3 / Qwen 2 cut on hand-computed cases, encode
    and decode over a handful of merges, special tokens both ways, the GPT-2
    path against the package's own Python oracle, and the SentencePiece
    refusals by name.

The loading and GPT-2-door tests need a built binding (the neural host or
the GPU transformer/mamba extensions, the tokenizer host); each skips BY
NAME with the load error otherwise, and a skip is not a pass.

    cd python && python3 -m pytest -q mojolearn/tests/test_models_loader.py
"""
import array
import json
import math
import os
import struct
import tempfile

import pytest

from mojolearn import Array, lowbit
from mojolearn import _bpe_trainer
from mojolearn import _tokenizer_synthetic as syn
from mojolearn.models import (CausalLM, Checkpoint, HFConfig, SafetensorsFile, Tokenizer,
                              UnsupportedModel, plan_for)
from mojolearn.models import causal_lm as _cl
from mojolearn.models import tokenizer as _tk
from mojolearn.models.safetensors import write_safetensors, widen_f16
from mojolearn.tokenizer import _byte_to_char


# ---------------------------------------------------------------- helpers

def _lcg_floats(n, seed, lo=-0.125, hi=0.125):
    """`n` float32 values in [lo, hi) from a 64-bit LCG: hashed, not random,
    the same bytes on every machine."""
    x = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    out = array.array("f")
    for _ in range(n):
        x = (x * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        u = (x >> 11) / float(1 << 53)
        out.append(lo + (hi - lo) * u)
    return out


def _f32(shape, seed, lo=-0.125, hi=0.125):
    n = 1
    for s in shape:
        n *= s
    return ("F32", shape, _lcg_floats(n, seed, lo, hi).tobytes())


def _ones(shape):
    n = 1
    for s in shape:
        n *= s
    return ("F32", shape, array.array("f", [1.0] * n).tobytes())


def _to_bf16_trunc(f32_bytes):
    """The top halves of float32 bits: valid bf16 bits, no rounding needed
    for a synthetic checkpoint."""
    u = array.array("I")
    u.frombytes(f32_bytes)
    h = array.array("H", [v >> 16 for v in u])
    return h.tobytes()


def _llama_config(**over):
    cfg = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 32,
           "num_attention_heads": 2, "num_key_value_heads": 1, "intermediate_size": 64,
           "num_hidden_layers": 2, "vocab_size": 64, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
           "max_position_embeddings": 128, "tie_word_embeddings": False, "hidden_act": "silu",
           "attention_bias": False, "mlp_bias": False, "torch_dtype": "float32"}
    cfg.update(over)
    # A real Qwen2 config carries no `attention_bias` key (its modeling code
    # puts the bias on q, k and v unconditionally); the Llama default above
    # would otherwise say False and hide the qwen2 row's own default.
    if cfg.get("model_type") == "qwen2" and "attention_bias" not in over:
        del cfg["attention_bias"]
    return cfg


def _llama_tensors(cfg, seed=1):
    dm, nh, nkv = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = dm // nh
    it, v = cfg["intermediate_size"], cfg["vocab_size"]
    t = {"model.embed_tokens.weight": _f32((v, dm), seed + 1, -0.5, 0.5),
         "model.norm.weight": _ones((dm,))}
    if not cfg.get("tie_word_embeddings", False):
        t["lm_head.weight"] = _f32((v, dm), seed + 2, -0.5, 0.5)
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        s = seed + 10 * (i + 1)
        t[p + "input_layernorm.weight"] = _ones((dm,))
        t[p + "post_attention_layernorm.weight"] = _ones((dm,))
        t[p + "self_attn.q_proj.weight"] = _f32((nh * hd, dm), s + 1)
        t[p + "self_attn.k_proj.weight"] = _f32((nkv * hd, dm), s + 2)
        t[p + "self_attn.v_proj.weight"] = _f32((nkv * hd, dm), s + 3)
        t[p + "self_attn.o_proj.weight"] = _f32((dm, nh * hd), s + 4)
        t[p + "mlp.gate_proj.weight"] = _f32((it, dm), s + 5)
        t[p + "mlp.up_proj.weight"] = _f32((it, dm), s + 6)
        t[p + "mlp.down_proj.weight"] = _f32((dm, it), s + 7)
    return t


def _write_checkpoint(root, cfg, tensors, *, shards=2, bf16=False):
    """`config.json` plus either one `model.safetensors` or `shards` files
    and `model.safetensors.index.json`; `bf16=True` stores every 2-D
    projection as BF16 bits."""
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    if bf16:
        conv = {}
        for name, (dtype, shape, raw) in tensors.items():
            if len(shape) == 2 and "embed" not in name and "lm_head" not in name:
                conv[name] = ("BF16", shape, _to_bf16_trunc(raw))
            else:
                conv[name] = (dtype, shape, raw)
        tensors = conv
    names = sorted(tensors)
    if shards <= 1:
        write_safetensors(os.path.join(root, "model.safetensors"), tensors)
        return root
    weight_map = {}
    per = int(math.ceil(len(names) / shards))
    for k in range(shards):
        part = names[k * per:(k + 1) * per]
        if not part:
            continue
        fname = f"model-{k + 1:05d}-of-{shards:05d}.safetensors"
        write_safetensors(os.path.join(root, fname), {n: tensors[n] for n in part})
        for n in part:
            weight_map[n] = fname
    with open(os.path.join(root, "model.safetensors.index.json"), "w", encoding="utf-8") as fh:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, fh)
    return root


def _mamba_config(**over):
    cfg = {"model_type": "mamba", "hidden_size": 32, "num_hidden_layers": 2, "vocab_size": 64,
           "state_size": 16, "conv_kernel": 4, "expand": 2, "time_step_rank": "auto",
           "layer_norm_epsilon": 1e-5, "use_bias": False, "use_conv_bias": True,
           "tie_word_embeddings": True, "hidden_act": "silu"}
    cfg.update(over)
    return cfg


def _mamba_tensors(cfg, seed=5):
    dm, v = cfg["hidden_size"], cfg["vocab_size"]
    di, r = 2 * dm, int(math.ceil(dm / 16.0))
    t = {"backbone.embeddings.weight": _f32((v, dm), seed + 1, -0.5, 0.5),
         "backbone.norm_f.weight": _ones((dm,))}
    for i in range(cfg["num_hidden_layers"]):
        p = f"backbone.layers.{i}."
        s = seed + 10 * (i + 1)
        t[p + "norm.weight"] = _ones((dm,))
        t[p + "mixer.in_proj.weight"] = _f32((2 * di, dm), s + 1)
        t[p + "mixer.conv1d.weight"] = _f32((di, 1, 4), s + 2)
        t[p + "mixer.conv1d.bias"] = _f32((di,), s + 3)
        t[p + "mixer.x_proj.weight"] = _f32((r + 32, di), s + 4)
        t[p + "mixer.dt_proj.weight"] = _f32((di, r), s + 5)
        t[p + "mixer.dt_proj.bias"] = _f32((di,), s + 6)
        t[p + "mixer.A_log"] = _f32((di, 16), s + 7, 0.0, 1.0)
        t[p + "mixer.D"] = _f32((di,), s + 8)
        t[p + "mixer.out_proj.weight"] = _f32((dm, di), s + 9)
    return t


def _load_or_skip(root, **kw):
    try:
        return CausalLM.load(root, **kw)
    except ImportError as exc:
        pytest.skip(f"no block binding on this box: {exc}")


# --------------------------------------------------------- safetensors

def test_safetensors_reader_dtypes_round_trip_and_widen_exactly():
    with tempfile.TemporaryDirectory() as d:
        f32 = _lcg_floats(12, 3, -3.0, 3.0)
        # float16 bits that reach every branch: normal, subnormal, zero, -0, inf, -inf, nan, max
        h16 = array.array("H", [0x3C00, 0x0001, 0x03FF, 0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E01, 0x7BFF, 0xC000, 0x0400, 0x8001])
        bf = array.array("H", [0x3F80, 0x0001, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0x3F81, 0xBF80, 0x0000, 0x4049, 0x7F7F, 0x0080])
        i32 = array.array("i", [-5, 0, 7, 2 ** 31 - 1])
        i64 = array.array("q", [-5, 0, 7, 2 ** 40])
        path = os.path.join(d, "t.safetensors")
        write_safetensors(path, {
            "a.f32": ("F32", (3, 4), f32.tobytes()),
            "b.f16": ("F16", (12,), h16.tobytes()),
            "c.bf16": ("BF16", (3, 4), bf.tobytes()),
            "c1.bf16": ("BF16", (12,), bf.tobytes()),
            "d.i32": ("I32", (4,), i32.tobytes()),
            "e.i64": ("I64", (2, 2), i64.tobytes()),
            "z.f64": ("F64", (1,), struct.pack("<d", 1.0)),
        }, metadata={"format": "pt"})
        with SafetensorsFile(path) as sf:
            assert sf.metadata == {"format": "pt"}
            a = sf.read("a.f32")
            assert isinstance(a, Array) and a.dtype == "<f4" and a.shape == (3, 4) and a.tobytes() == f32.tobytes()
            b = sf.read("b.f16")
            assert b.dtype == "<f4" and b.shape == (12,)
            # oracle: struct's own half -> double -> float, exact since f16 is a subset of f32
            want = array.array("f", struct.unpack("<12e", h16.tobytes()))
            got = array.array("f")
            got.frombytes(b.tobytes())
            for k in range(12):
                if want[k] != want[k]:
                    assert got[k] != got[k]  # a NaN stays a NaN
                    assert (struct.unpack("<I", struct.pack("<f", got[k]))[0] & 0x80000000) == (h16[k] & 0x8000) << 16
                else:
                    assert struct.pack("<f", got[k]) == struct.pack("<f", want[k]), k
            c = sf.read("c.bf16")
            assert c.dtype == "<f4" and c.shape == (3, 4)
            shifted = array.array("I", [v << 16 for v in bf]).tobytes()
            assert c.tobytes() == shifted
            assert sf.read("c1.bf16").tobytes() == shifted
            bits = sf.read("c.bf16", bf16="bits")
            assert isinstance(bits, lowbit.BF16Weight) and bits.bits.tobytes() == bf.tobytes()
            assert sf.read("c1.bf16", bf16="bits").dtype == "<u2"  # 1-D bits stay a plain Array
            assert sf.read("d.i32").tolist() == [-5, 0, 7, 2 ** 31 - 1]
            assert sf.read("e.i64").tolist() == [[-5, 0], [7, 2 ** 40]]
            with pytest.raises(TypeError, match="z.f64.*F64"):
                sf.read("z.f64")
            with pytest.raises(KeyError, match="nope"):
                sf.read("nope")


def test_f16_widening_is_bit_exact_over_every_half():
    """All 65536 float16 bit patterns against struct's oracle (finite ones
    by float32 bytes, NaNs by sign and NaN-ness)."""
    h = array.array("H", range(65536))
    got = widen_f16(Array._owned(h, (65536,), "<u2", "C"))
    g = array.array("I")
    g.frombytes(got.tobytes())
    for k in range(65536):
        (w,) = struct.unpack("<e", struct.pack("<H", k))
        if w != w:
            assert (g[k] & 0x7F800000) == 0x7F800000 and (g[k] & 0x007FFFFF) != 0
            assert (g[k] >> 31) == (k >> 15)
        else:
            assert g[k] == struct.unpack("<I", struct.pack("<f", w))[0], k


def test_checkpoint_follows_the_sharded_index():
    with tempfile.TemporaryDirectory() as d:
        cfg = _llama_config()
        tensors = _llama_tensors(cfg)
        root = _write_checkpoint(os.path.join(d, "m"), cfg, tensors, shards=3)
        with Checkpoint.open(root) as ck:
            assert sorted(ck.names()) == sorted(tensors)
            for name, (_, shape, raw) in tensors.items():
                a = ck.read(name)
                assert a.shape == shape and a.tobytes() == raw
            assert "model.layers.1.mlp.up_proj.weight" in ck
        single = _write_checkpoint(os.path.join(d, "s"), cfg, tensors, shards=1)
        with Checkpoint.open(single) as ck:
            assert len(ck.names()) == len(tensors)
        with pytest.raises(FileNotFoundError):
            Checkpoint.open(os.path.join(d, "missing"))


# ------------------------------------------------------------- config

def test_plan_llama_shapes_names_and_options():
    plan = plan_for(HFConfig(_llama_config(num_key_value_heads=1)))
    assert plan.kind == "transformer" and plan.n_layers == 2 and plan.d_model == 32 and plan.vocab_size == 64
    assert plan.n_heads == 2 and plan.n_kv_heads == 1 and plan.head_dim == 16 and plan.intermediate == 64
    assert plan.embed_name == "model.embed_tokens.weight" and plan.norm_name == "model.norm.weight"
    assert plan.head_name == "lm_head.weight" and plan.norm_eps == 1e-6
    names = dict((k, n) for k, n, rows in plan.layer_weights(1))
    assert names["q_proj.weight"] == "model.layers.1.self_attn.q_proj.weight"
    assert names["down_proj.weight"] == "model.layers.1.mlp.down_proj.weight"
    assert names["input_layernorm.weight"] == "model.layers.1.input_layernorm.weight"
    assert all(rows is None for _, _, rows in plan.layer_weights(0))
    o = plan.block_options
    assert o["n_kv_heads"] == 1 and o["head_dim"] is None and o["window"] == 0
    assert o["norm_eps"] == 1e-6 and o["rope_theta"] == 10000.0 and o["qkv_bias"] is False
    tied = plan_for(HFConfig(_llama_config(tie_word_embeddings=True)))
    assert tied.head_name is None and "lm_head.weight" not in tied.checkpoint_names()
    assert plan_for(HFConfig(_llama_config(model_type="mistral", sliding_window=32))).block_options["window"] == 32


def test_plan_qwen_and_phi3_rows():
    q2 = plan_for(HFConfig(_llama_config(model_type="qwen2", rope_theta=1000000.0)))
    assert q2.block_options["qkv_bias"] is True
    assert "model.layers.0.self_attn.q_proj.bias" in q2.checkpoint_names()
    q3 = plan_for(HFConfig(_llama_config(model_type="qwen3", head_dim=16)))
    assert q3.block_options["qk_norm"] is True and q3.block_options["qkv_bias"] is False
    assert "model.layers.0.self_attn.k_norm.weight" in q3.checkpoint_names()
    p3 = plan_for(HFConfig(_llama_config(model_type="phi3")))
    rows = {k: (n, r) for k, n, r in p3.layer_weights(0)}
    assert rows["q_proj.weight"] == ("model.layers.0.self_attn.qkv_proj.weight", (0, 32))
    assert rows["k_proj.weight"] == ("model.layers.0.self_attn.qkv_proj.weight", (32, 48))
    assert rows["v_proj.weight"] == ("model.layers.0.self_attn.qkv_proj.weight", (48, 64))
    assert rows["gate_proj.weight"] == ("model.layers.0.mlp.gate_up_proj.weight", (0, 64))
    assert rows["up_proj.weight"] == ("model.layers.0.mlp.gate_up_proj.weight", (64, 128))
    m1 = plan_for(HFConfig(_mamba_config()))
    assert m1.kind == "mamba1" and m1.head_name is None
    assert dict((k, n) for k, n, _ in m1.layer_weights(1))["A_log"] == "backbone.layers.1.mixer.A_log"
    m2 = plan_for(HFConfig({"model_type": "mamba2", "hidden_size": 64, "num_hidden_layers": 1, "vocab_size": 16,
                            "time_step_limit": [0.0, 1.5]}))
    assert m2.kind == "mamba2" and m2.block_kwargs == {"dt_limit": (0.0, 1.5)}
    assert dict((k, n) for k, n, _ in m2.layer_weights(0))["block_norm.weight"] == "backbone.layers.0.norm.weight"


@pytest.mark.parametrize("cfg, words", [
    (_llama_config(num_local_experts=8), ["'llama'", "num_local_experts=8", "mixture of experts"]),
    (_llama_config(rope_scaling={"rope_type": "llama3", "factor": 8.0}), ["rope_scaling=", "llama3"]),
    (_llama_config(hidden_act="gelu"), ["hidden_act='gelu'", "SwiGLU"]),
    (_llama_config(model_type="qwen2", use_sliding_window=True, max_window_layers=21), ["use_sliding_window=True", "max_window_layers=21"]),
    (_llama_config(model_type="gemma", hidden_activation="gelu_pytorch_tanh", head_dim=256), ["'gemma'", "gelu_pytorch_tanh"]),
    (_llama_config(model_type="gemma2", attn_logit_softcapping=50.0), ["'gemma2'", "attn_logit_softcapping=50.0"]),
    (_llama_config(num_key_value_heads=3), ["num_key_value_heads=3"]),
    (_llama_config(attention_dropout=0.1), ["attention_dropout"]),
    (_mamba_config(state_size=8), ["'mamba'", "state_size=8", "16"]),
    (_mamba_config(layer_norm_epsilon=1e-6), ["layer_norm_epsilon=1e-06"]),
    ({"model_type": "mamba2", "hidden_size": 64, "num_hidden_layers": 1, "vocab_size": 16, "n_groups": 2}, ["'mamba2'", "n_groups=2"]),
    ({"model_type": "mamba2", "hidden_size": 48, "num_hidden_layers": 1, "vocab_size": 16}, ["hidden_size=48", "multiple of 32"]),
    ({"model_type": "mixtral", "hidden_size": 8}, ["'mixtral'", "not in the option matrix"]),
])
def test_refusals_name_the_model_type_field_and_value(cfg, words):
    with pytest.raises(UnsupportedModel) as e:
        plan_for(HFConfig(cfg))
    msg = str(e.value)
    for w in words:
        assert w in msg, (w, msg)


class _TodayBlock:
    def __init__(self, weights, *, n_heads, n_kv_heads=None, head_dim=None, window=0):
        pass


class _FullBlock:
    def __init__(self, weights, *, n_heads, n_kv_heads=None, head_dim=None, window=0,
                 rope_theta=10000.0, rope_scaling=None, rope_dim=None, max_positions=8192,
                 qkv_bias=False, o_bias=False, norm="rmsnorm", norm_eps=1e-5, norm_bias=False,
                 mlp="swiglu", mlp_bias=False, qk_norm=False, attn_softcap=None):
        pass


def test_block_kwargs_follow_the_live_signature():
    """Independent of whether lane B1 has landed: the rule is exercised on
    two stand-in signatures, today's and the full interface."""
    fixed = plan_for(HFConfig(_llama_config()))            # eps 1e-6, theta 10000: what today's block fixes
    assert _cl._block_kwargs(fixed, _TodayBlock) == {"n_kv_heads": 1}
    assert _cl._block_kwargs(fixed, _FullBlock) == {"n_kv_heads": 1, "norm_eps": 1e-6}
    l3 = plan_for(HFConfig(_llama_config(rms_norm_eps=1e-5, rope_theta=500000.0)))
    with pytest.raises(UnsupportedModel) as e:
        _cl._block_kwargs(l3, _TodayBlock)
    assert "rope_theta=500000.0" in str(e.value) and "_TodayBlock" in str(e.value) and "fixes rope_theta=10000.0" in str(e.value)
    assert _cl._block_kwargs(l3, _FullBlock) == {"n_kv_heads": 1, "rope_theta": 500000.0}
    eps = plan_for(HFConfig(_llama_config(rms_norm_eps=1e-5)))
    with pytest.raises(UnsupportedModel, match="norm_eps=1e-05.*rms_norm_eps"):
        _cl._block_kwargs(eps, _TodayBlock)
    assert _cl._block_kwargs(eps, _FullBlock) == {"n_kv_heads": 1}
    q2 = plan_for(HFConfig(_llama_config(model_type="qwen2")))
    with pytest.raises(UnsupportedModel, match="qkv_bias=True.*attention_bias"):
        _cl._block_kwargs(q2, _TodayBlock)
    assert _cl._block_kwargs(q2, _FullBlock) == {"n_kv_heads": 1, "norm_eps": 1e-6, "qkv_bias": True}
    win = plan_for(HFConfig(_llama_config(model_type="mistral", sliding_window=16)))
    assert _cl._block_kwargs(win, _TodayBlock) == {"n_kv_heads": 1, "window": 16}
    # the live class: its signature is read, whatever it is today
    from mojolearn._transformer_impl import TransformerBlock
    acc = _cl._accepted_options(TransformerBlock)
    assert acc is None or {"n_heads", "n_kv_heads", "head_dim", "window"} <= acc


def test_greedy_argmax_ties_go_to_the_lowest_index():
    logits = Array.from_list([[[0.0, 1.0, 1.0, -1.0], [2.0, 2.0, 2.0, 2.0]],
                              [[5.0, 5.0, 6.0, 6.0], [-1.0, -3.0, -1.0, -2.0]]], "<f4")
    assert _cl._argmax_last(logits, 2, 2, 4) == [0, 0]
    assert _cl._argmax_last(logits.reshape((4, 1, 4)), 4, 1, 4) == [1, 0, 2, 0]


# ------------------------------------------------------------- loading

def test_load_llama_checkpoint_shapes_names_and_greedy_generate_is_deterministic():
    with tempfile.TemporaryDirectory() as d:
        cfg = _llama_config()
        root = _write_checkpoint(os.path.join(d, "m"), cfg, _llama_tensors(cfg), shards=2)
        lm = _load_or_skip(root)
        assert lm.n_layers == 2 and lm.d_model == 32 and lm.vocab_size == 64 and lm.kind == "transformer"
        assert lm.max_positions == 128 and lm.weight_format == "float32" and lm.unused_names == []
        for blk in lm.blocks:
            assert blk.d_model == 32 and blk.n_heads == 2 and blk.n_kv_heads == 1 and blk.head_dim == 16
            assert blk.intermediate == 64 and blk.weight_format == "float32"
        assert len(lm.blocks) == 2 and lm.blocks[0] is lm.blocks[0]  # built once, kept
        params = lm.parameters()
        assert "model.layers.1.self_attn.k_proj.weight" in params and params["model.layers.1.self_attn.k_proj.weight"].shape == (16, 32)
        assert "lm_head.weight" in params and params["model.embed_tokens.weight"].shape == (64, 32)
        ids = Array.from_list([[1, 2, 3, 4, 5], [9, 8, 7, 6, 5]], "<i4")
        logits = lm.forward(ids)
        assert logits.shape == (2, 5, 64) and logits.dtype == "<f4"
        # generate twice: the same ids, and the prefix is the prompt
        a = lm.generate(ids, 4)
        b = lm.generate(ids, 4)
        assert a.shape == (2, 9) and a.dtype == "<i4" and a.tobytes() == b.tobytes()
        assert a.tolist()[0][:5] == [1, 2, 3, 4, 5]
        # and it is the stateful path token by token
        state = lm.allocate_state(2, 9)
        lg = lm.forward(ids, state)
        nxt = _cl._argmax_last(lg, 2, 5, 64)
        assert nxt == [row[5] for row in a.tolist()]
        step = lm.step(Array.from_list([[v] for v in nxt], "<i4"), state)
        assert step.shape == (2, 64) and state.positions == 6
        assert _cl._argmax_last(step.reshape((2, 1, 64)), 2, 1, 64) == [row[6] for row in a.tolist()]
        with pytest.raises(ValueError, match="max_positions"):
            lm.generate(ids, 200)
        with pytest.raises(ValueError, match=r"\[0, 64\)"):
            lm.forward(Array.from_list([[64]], "<i4"))
        with pytest.raises(NotImplementedError, match="greedy"):
            lm.generate(ids, 1, greedy=False)


@pytest.mark.parametrize("stored_bf16", [False, True])
def test_load_bfloat16_reads_on_every_block(stored_bf16):
    with tempfile.TemporaryDirectory() as d:
        cfg = _llama_config()
        tensors = _llama_tensors(cfg)
        root = _write_checkpoint(os.path.join(d, "m"), cfg, tensors, shards=2, bf16=stored_bf16)
        # the bits path needs no binding: a BF16 shard hands back its own bytes
        with Checkpoint.open(root) as ck:
            name = "model.layers.0.self_attn.q_proj.weight"
            got = ck.read(name, bf16="bits")
            if stored_bf16:
                assert isinstance(got, lowbit.BF16Weight)
                assert got.bits.tobytes() == _to_bf16_trunc(tensors[name][2])
            else:
                assert got.dtype == "<f4"
        lm = _load_or_skip(root, weight_format="bfloat16")
        assert lm.weight_format == "bfloat16"
        assert all(blk.weight_format == "bfloat16" for blk in lm.blocks)
        f32 = _load_or_skip(root, weight_format="float32")
        assert all(blk.weight_format == "float32" for blk in f32.blocks)
        ids = Array.from_list([[3, 1, 4, 1, 5]], "<i4")
        lo, hi = lm.forward(ids), f32.forward(ids)
        assert lo.shape == hi.shape == (1, 5, 64)
        if stored_bf16:
            # a BF16 checkpoint materializes to the same float32 either way: identical logits
            assert lo.tobytes() == hi.tobytes()
        else:
            # a float32 checkpoint packed to bf16 is a rounded model: the bits move
            assert lo.tobytes() != hi.tobytes()
        i8 = _load_or_skip(root, weight_format="int8")
        assert all(blk.weight_format == "int8" for blk in i8.blocks)


def test_load_mamba_checkpoint():
    with tempfile.TemporaryDirectory() as d:
        cfg = _mamba_config()
        root = _write_checkpoint(os.path.join(d, "m"), cfg, _mamba_tensors(cfg), shards=1)
        lm = _load_or_skip(root)
        assert lm.kind == "mamba1" and lm.max_positions is None and lm.n_layers == 2
        for blk in lm.blocks:
            assert blk.d_model == 32 and blk.d_inner == 64 and blk.dt_rank == 2
        assert lm.plan.head_name is None and lm._head is lm._embed
        ids = Array.from_list([[1, 2, 3]], "<i4")
        assert lm.forward(ids).shape == (1, 3, 64)
        a, b = lm.generate(ids, 3), lm.generate(ids, 3)
        assert a.shape == (1, 6) and a.tobytes() == b.tobytes()
        lo = _load_or_skip(root, weight_format="bfloat16")
        assert all(blk.weight_format == "bfloat16" for blk in lo.blocks)


def test_load_refuses_a_missing_tensor_and_a_bad_vocab_by_name():
    with tempfile.TemporaryDirectory() as d:
        cfg = _llama_config()
        tensors = _llama_tensors(cfg)
        del tensors["model.layers.1.mlp.up_proj.weight"]
        root = _write_checkpoint(os.path.join(d, "m"), cfg, tensors, shards=1)
        with pytest.raises(UnsupportedModel, match="model.layers.1.mlp.up_proj.weight"):
            CausalLM.load(root, device="cpu")
        cfg2 = _llama_config(vocab_size=65)
        root2 = _write_checkpoint(os.path.join(d, "v"), cfg2, _llama_tensors(_llama_config()), shards=1)
        try:
            CausalLM.load(root2, device="cpu")
        except ImportError as exc:
            pytest.skip(str(exc))
        except UnsupportedModel as exc:
            assert "vocab_size=65" in str(exc) and "model.embed_tokens.weight" in str(exc)
        else:
            raise AssertionError("a vocab_size that disagrees with the embedding rows must be refused")
        with pytest.raises(ValueError, match="weight_format"):
            CausalLM.load(root, weight_format="fp8")
        with pytest.raises(ValueError, match="device"):
            CausalLM.load(root, device="tpu")


# ----------------------------------------------------------- tokenizer

def _spell(token):
    c = _byte_to_char()
    return "".join(c[b] for b in token)


def _tokenizer_json(tokens, merges, regex, added=(), template=False, normalizer=None, **model_extra):
    """A `tokenizer.json` in the Hugging Face layout with a `Split Regex`
    pre-tokenizer, added tokens and optionally a TemplateProcessing BOS."""
    vocab = {_spell(t): i for i, t in enumerate(tokens)}
    for content, i in added:
        vocab[content] = i
    pre = ({"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True}
           if regex is None else
           {"type": "Sequence", "pretokenizers": [
               {"type": "Split", "pattern": {"Regex": regex}, "behavior": "Isolated", "invert": False},
               {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]})
    post = None
    if template and added:
        content, i = added[0]
        post = {"type": "TemplateProcessing",
                "single": [{"SpecialToken": {"id": content, "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
                "pair": [], "special_tokens": {content: {"id": content, "ids": [i], "tokens": [content]}}}
    model = {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
             "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
             "vocab": vocab, "merges": [[_spell(a), _spell(b)] for a, b in merges]}
    model.update(model_extra)
    return {"version": "1.0", "truncation": None, "padding": None,
            "added_tokens": [{"id": i, "content": c, "single_word": False, "lstrip": False, "rstrip": False,
                              "normalized": False, "special": True} for c, i in added],
            "normalizer": normalizer, "pre_tokenizer": pre, "post_processor": post,
            "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
            "model": model}


def _small_vocab():
    tokens = [bytes([b]) for b in range(256)]
    merges = [(b"h", b"e"), (b"l", b"l"), (b"he", b"ll"), (b"hell", b"o"), (b" ", b"w"), (b" w", b"o")]
    for a, b in merges:
        tokens.append(a + b)
    return tokens, merges


@pytest.mark.parametrize("pattern, text, pieces", [
    ("llama3", b"Hello world", [b"Hello", b" world"]),
    ("llama3", b"IT'S", [b"IT", b"'S"]),
    ("llama3", b"a  b", [b"a", b" ", b" b"]),
    ("llama3", b"12345", [b"123", b"45"]),
    ("llama3", b" 123", [b" ", b"123"]),
    ("llama3", b"x,\n\ny", [b"x", b",\n\n", b"y"]),
    ("llama3", b"a \n b", [b"a", b" \n", b" b"]),
    ("llama3", b"-hello", [b"-hello"]),
    ("llama3", b"hi\n", [b"hi", b"\n"]),
    ("llama3", b"a   ", [b"a", b"   "]),
    ("llama3", b"tab\tx", [b"tab", b"\tx"]),
    ("llama3", b"we're", [b"we", b"'re"]),
    ("llama3", b"caf\xc3\xa9 2024!", [b"caf\xc3\xa9", b" ", b"202", b"4", b"!"]),
    ("llama3", b"\xff\xfe", [b"\xff", b"\xfe"]),
    ("qwen2", b"12345", [b"1", b"2", b"3", b"4", b"5"]),
    ("qwen2", b"Hello world", [b"Hello", b" world"]),
    ("gpt2", b"IT'S", [b"IT", b"'", b"S"]),
    ("gpt2", b"a  b", [b"a", b" ", b" b"]),
    ("gpt2", b"12345", [b"12345"]),
])
def test_pretokenize_cuts_as_the_pattern_says(pattern, text, pieces):
    bounds = _tk.pretokenize(text, pattern)
    assert bounds[0] == 0 and bounds[-1] == len(text)
    assert [text[a:b] for a, b in zip(bounds, bounds[1:])] == pieces


def test_gpt2_pattern_in_python_is_the_package_oracle():
    for text in (b"Hello world", b"IT'S a  test\n\n", b"caf\xc3\xa9 2024!", b"\xff x"):
        assert _tk.pretokenize(text, "gpt2") == syn.pretokenize(text)


def test_tokenizer_llama3_encode_decode_and_specials():
    tokens, merges = _small_vocab()
    added = [("<|begin_of_text|>", 300), ("<|eot_id|>", 301)]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tokenizer.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_tokenizer_json(tokens, merges, _tk.PATTERNS["llama3"][0], added, template=True), fh)
        with open(os.path.join(d, "tokenizer_config.json"), "w", encoding="utf-8") as fh:
            json.dump({"bos_token": "<|begin_of_text|>", "eos_token": "<|eot_id|>"}, fh)
        tok = Tokenizer.from_pretrained(d)
        assert tok.pattern == "llama3" and tok.n_ranks == 262 and tok.n_vocab == 302
        assert tok.bos_token_id == 300 and tok.eos_token_id == 301 and tok.bos_ids == (300,) and tok.eos_ids == ()
        assert tok.pretokenize("hello world") == [b"hello", b" world"]
        ids = tok.encode("hello world", add_special_tokens=False)
        assert ids == [259, 261, 114, 108, 100]  # hello | " wo" r l d
        assert tok.encode("hello world") == [300] + ids
        assert tok.decode(ids) == "hello world" and tok.decode([300] + ids) == "<|begin_of_text|>hello world"
        # a special in the text is ordinary text unless allowed
        plain = tok.encode("hello<|eot_id|>", add_special_tokens=False)
        assert 301 not in plain and tok.decode(plain) == "hello<|eot_id|>"
        assert tok.encode("hello<|eot_id|>hello", add_special_tokens=False, allow_special=True) == [259, 301, 259]
        assert tok.encode_bytes(b"<|begin_of_text|><|eot_id|>", allow_special=True) == [300, 301]
        # decode refuses an id that is neither a rank nor an added token, by value and position
        with pytest.raises(ValueError, match="id 299 at position 1"):
            tok.decode([0, 299])
        # one digit at a time under qwen2, three under llama3
        q = _tokenizer_json(tokens, merges, _tk.PATTERNS["qwen2"][0], normalizer={"type": "NFC"})
        qt = Tokenizer._from_json(q, "qwen.json")
        assert qt.pattern == "qwen2" and qt.normalizer == "NFC"
        assert qt.pretokenize(b"2024") == [b"2", b"0", b"2", b"4"] and tok.pretokenize(b"2024") == [b"202", b"4"]
        # NFC: a decomposed e + U+0301 is normalized before the cut
        assert qt.encode("é", add_special_tokens=False) == qt.encode("é", add_special_tokens=False)
        assert tok.encode("é", add_special_tokens=False) != tok.encode("é", add_special_tokens=False)


def test_tokenizer_gpt2_pattern_uses_the_compiled_door_and_matches_the_oracle():
    corpus = [b"the quick brown fox jumps over the lazy dog " * 3, b"hello hello world, IT'S 2024\n\n"]
    tokens, merges, _ = _bpe_trainer.train(corpus, vocab_size=300, min_frequency=2)
    text = _bpe_trainer.render_tokenizer_json(tokens, merges)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tokenizer.json")
        with open(path, "w", encoding="ascii", newline="\n") as fh:
            fh.write(text)
        tok = Tokenizer.from_pretrained(path)
        assert tok.pattern == "gpt2" and tok.added == {"<|endoftext|>": len(tokens)}
        # the ByteLevel use_regex spelling names the same pattern
        data = json.loads(text)
        data["pre_tokenizer"] = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True}
        assert Tokenizer._from_json(data, "bl.json").pattern == "gpt2"
        sample = b"hello world, IT'S a  fox\n\n"
        want = syn.reference_encode(tokens, sample)
        try:
            got = tok.encode_bytes(sample)
        except ImportError as exc:
            pytest.skip(f"tokenizer host binding not built: {exc}")
        assert got == want and tok.decode_bytes(got) == sample


@pytest.mark.parametrize("edit, words", [
    ({"model": {"type": "Unigram", "vocab": []}}, ["Unigram", "SentencePiece"]),
    ({"model_extra": {"byte_fallback": True}}, ["byte_fallback", "SentencePiece"]),
    ({"pre_tokenizer": {"type": "Metaspace", "replacement": "▁"}}, ["Metaspace", "SentencePiece"]),
    ({"regex": "\\p{L}+|\\p{N}+"}, ["not one of the known patterns", "p{N}+", "refused by name"]),
    ({"normalizer": {"type": "Lowercase"}}, ["normalizer 'Lowercase'"]),
])
def test_sentencepiece_and_unknown_patterns_are_refused_by_name(edit, words):
    tokens, merges = _small_vocab()
    regex = edit.pop("regex", _tk.PATTERNS["llama3"][0])
    extra = edit.pop("model_extra", {})
    data = _tokenizer_json(tokens, merges, regex, **extra)
    for k, v in edit.items():
        if k == "model":
            data["model"].update(v)
        else:
            data[k] = v
    with pytest.raises(ValueError) as e:
        Tokenizer._from_json(data, "t.json")
    for w in words:
        assert w in str(e.value), (w, str(e.value))


def test_tokenizer_model_file_without_json_is_refused_by_name():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "tokenizer.model"), "wb") as fh:
            fh.write(b"\x0a\x00")
        with pytest.raises(ValueError, match="SentencePiece"):
            Tokenizer.from_pretrained(d)
        with pytest.raises(ValueError, match="▁"):
            tokens, merges = _small_vocab()
            data = _tokenizer_json(tokens, merges, _tk.PATTERNS["llama3"][0])
            data["model"]["vocab"]["▁the"] = 262
            Tokenizer._from_json(data, "sp.json")
