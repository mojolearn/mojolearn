# SPDX-License-Identifier: Apache-2.0
"""Deterministic tiny loaded-model fixtures; no pytest or network dependency."""
import array
import json
import math
import os
from .models.safetensors import write_safetensors

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


def family_fixture(architecture, tied):
    """Exercise each supported checkpoint family and its distinct mapping."""
    if architecture == 'mamba':
        cfg = _mamba_config()
        return cfg, _mamba_tensors(cfg)
    if architecture == 'mamba2':
        cfg = dict(model_type='mamba2', hidden_size=32, num_hidden_layers=2,
                   vocab_size=64, tie_word_embeddings=True, time_step_limit=[0.0, 1.5])
        t = {'backbone.embeddings.weight': _f32((64,32), 6, -0.5,0.5),
             'backbone.norm_f.weight': _ones((32,))}
        for i in range(2):
            p=f'backbone.layers.{i}.'; seed=20+10*i
            t[p+'norm.weight']=_ones((32,))
            for name,shape in [('in_proj.weight',(385,32)),('conv1d.weight',(320,1,4)),
                               ('conv1d.bias',(320,)),('dt_bias',(1,)),('A_log',(1,)),
                               ('D',(1,)),('out_proj.weight',(32,64))]:
                t[p+'mixer.'+name]=_f32(shape,seed); seed+=1
            t[p+'mixer.norm.weight']=_ones((64,))
        return cfg,t
    cfg = _llama_config(model_type=architecture, tie_word_embeddings=tied)
    if architecture == 'mistral': cfg['sliding_window']=3
    tensors = _llama_tensors(cfg)
    for i in range(cfg['num_hidden_layers']):
        p=f'model.layers.{i}.'
        if architecture == 'qwen2':
            for name,size in [('q',32),('k',16),('v',16)]:
                tensors[p+f'self_attn.{name}_proj.bias']=_f32((size,),30+i+size)
        elif architecture == 'qwen3':
            for name in ('q','k'):
                tensors[p+f'self_attn.{name}_norm.weight']=_f32((16,),40+i,0.75,1.25)
        elif architecture == 'phi3':
            for group,keys in [('self_attn.qkv_proj.weight', ['self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight']),
                               ('mlp.gate_up_proj.weight',['mlp.gate_proj.weight','mlp.up_proj.weight'])]:
                pieces=[tensors.pop(p+key) for key in keys]
                tensors[p+group]=('F32',(sum(x[1][0] for x in pieces),32),b''.join(x[2] for x in pieces))
    return cfg,tensors
