# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`HFConfig` and the OPTION MATRIX (lane/model-loader, 2026-09-17).

A Hugging Face `config.json` names a `model_type` and a bag of fields. This
module knows, per family, which fields map onto which block option, which
tensor names carry each weight, and which fields the blocks cannot honor.
Nothing here loads a tensor or runs arithmetic; the answer is a `ModelPlan`
that `causal_lm.py` assembles from, and every refusal is BY NAME: the
message carries the `model_type`, the field and its value.

THE BLOCK INTERFACE THIS CODES AGAINST. `TransformerBlock(weights, *,
n_heads, n_kv_heads=None, head_dim=None, window=0, rope_theta=10000.0,
rope_scaling=None, rope_dim=None, max_positions=8192, qkv_bias=False,
o_bias=False, norm="rmsnorm", norm_eps=1e-5, norm_bias=False, mlp="swiglu",
mlp_bias=False, qk_norm=False, attn_softcap=None)`. Lane B1 is adding the
second and later lines in parallel; today the block accepts the first line
only, so a plan lists EVERY option with its value and the loader passes a
keyword only when its value differs from the interface default, and refuses
by name any option the live block does not accept whose value differs from
what that block fixes today (`FIXED_TODAY`). `norm_eps` is the case that
matters: the block today fixes 1e-6 (contract section 3, `0x358637BD`) and
the interface's default is 1e-5, so a Llama 3 config (1e-5) is refused by
name until B1 lands and honored silently afterwards -- never run at the
wrong epsilon.

THE MAMBA INTERFACE. `Mamba1Block(weights)` and `Mamba2Block(weights, *,
dt_limit=(0.0, inf))` take no shape arguments: every constant is a profile
constant (`mamba/IDENTICAL_MAMBA_CONTRACT.md` section 3: d_state 16, d_conv
4, expand 2, dt_rank ceil(d_model/16), rms eps 1e-5; the Mamba-2 contract:
d_state 128, headdim 64, ngroups 1, chunk 256, eps 1e-5), so a config field
that names another value is refused by name here.

TENSOR NAMES. The Hugging Face names per family, with `{i}` the layer
index; the plan maps each block-dict key (the block classes' own names) to
the checkpoint name, and for phi3's fused `qkv_proj` / `gate_up_proj` to a
row slice of it (a slice copies bytes, no arithmetic).
"""
from ..tokenizer import _binding

_NATIVE = _binding()._module
import os

__all__ = ["HFConfig", "ModelPlan", "UnsupportedModel", "FAMILIES", "plan_for",
           "INTERFACE_DEFAULTS", "FIXED_TODAY", "POSITION_CEILING"]

#: DEVIATION 812's absolute-position ceiling, refused by name in Mojo; the
#: loader caps `max_positions` here and refuses a longer request itself.
POSITION_CEILING = 8192

#: The interface's defaults (lane B1's signature). A keyword is passed only
#: when the plan's value differs from this.
INTERFACE_DEFAULTS = {
    "n_kv_heads": None, "head_dim": None, "window": 0, "rope_theta": 10000.0,
    "rope_scaling": None, "rope_dim": None, "max_positions": POSITION_CEILING,
    "qkv_bias": False, "o_bias": False, "norm": "rmsnorm", "norm_eps": 1e-5,
    "norm_bias": False, "mlp": "swiglu", "mlp_bias": False, "qk_norm": False,
    "attn_softcap": None,
}

#: What TODAY's `TransformerBlock` fixes for an option it does not accept
#: (`_transformer_impl.py`'s "WHAT IS HONORED, WHAT IS FIXED, WHAT IS
#: REFUSED" table). An option absent from the live signature is admitted
#: only at this value.
FIXED_TODAY = {
    "rope_theta": 10000.0, "rope_scaling": None, "rope_dim": None,
    "max_positions": POSITION_CEILING, "qkv_bias": False, "o_bias": False,
    "norm": "rmsnorm", "norm_eps": 1e-6, "norm_bias": False, "mlp": "swiglu",
    "mlp_bias": False, "qk_norm": False, "attn_softcap": None,
}


class UnsupportedModel(ValueError):
    """A config field the blocks cannot honor, refused by name."""


class HFConfig:
    """A `config.json` as a read-only mapping with `model_type`, `get`,
    `[]` and `path`."""

    def __init__(self, data, path=None):
        return _NATIVE.model_config_init(self, data, path)

    @classmethod
    def from_json(cls, path):
        """`path` is `config.json` or the directory holding it."""
        return _NATIVE.model_config_load(cls, path)

    @property
    def model_type(self):
        return _NATIVE.model_config_access(self, 0, None, None, UnsupportedModel)

    def get(self, key, default=None):
        return _NATIVE.model_config_access(self, 1, key, default, UnsupportedModel)

    def __getitem__(self, key):
        return _NATIVE.model_config_access(self, 2, key, None, UnsupportedModel)

    def __contains__(self, key):
        return _NATIVE.model_config_access(self, 3, key, None, UnsupportedModel)

    def keys(self):
        return _NATIVE.model_config_access(self, 4, None, None, UnsupportedModel)

    def to_dict(self):
        return _NATIVE.model_config_access(self, 5, None, None, UnsupportedModel)

    def __repr__(self):
        return _NATIVE.model_config_repr(self, False)


class ModelPlan:
    """What `CausalLM` assembles from. Fields:

    `model_type`, `kind` ("transformer" | "mamba1" | "mamba2"), `n_layers`,
    `d_model`, `vocab_size`, `tie_embeddings`, `norm_eps` (the FINAL norm's
    epsilon, always passed to the norm primitive), `max_position_embeddings`,
    `block_options` (every interface option with its value, transformer
    only), `block_kwargs` (Mamba-2's `dt_limit`, else empty), `embed_name`,
    `norm_name`, `head_name` (None when tied), `layer_weights(i)` -> list of
    `(block_key, checkpoint_name, row_slice_or_None)`, `family` (the
    matrix row)."""

    def __init__(self, **kw):
        return _NATIVE.model_plan_init(self, kw)

    def layer_weights(self, i):
        return _NATIVE.model_layer_weights(self, i)

    def checkpoint_names(self):
        return _NATIVE.model_checkpoint_names(self)

    def __repr__(self):
        return _NATIVE.model_config_repr(self, True)


# --------------------------------------------------------------- families
#
# Each row: `kind`, `tokenizer` (what tokenizer.py can load for it), the
# checkpoint names, and `plan(cfg)`. The field reading, the refusals by name
# and the option arithmetic run in Mojo (`model_plan_for_binding`,
# bindings/causal_model_config_door.mojo: `_config_transformer`,
# `_config_mamba`, `_config_phi3`, the byte-LM rename). The README beside this
# file renders the same rows as a table; keep them in step.

_LLAMA_LAYER = [
    ("input_layernorm.weight", "model.layers.{i}.input_layernorm.weight", None),
    ("post_attention_layernorm.weight", "model.layers.{i}.post_attention_layernorm.weight", None),
    ("q_proj.weight", "model.layers.{i}.self_attn.q_proj.weight", None),
    ("k_proj.weight", "model.layers.{i}.self_attn.k_proj.weight", None),
    ("v_proj.weight", "model.layers.{i}.self_attn.v_proj.weight", None),
    ("o_proj.weight", "model.layers.{i}.self_attn.o_proj.weight", None),
    ("gate_proj.weight", "model.layers.{i}.mlp.gate_proj.weight", None),
    ("up_proj.weight", "model.layers.{i}.mlp.up_proj.weight", None),
    ("down_proj.weight", "model.layers.{i}.mlp.down_proj.weight", None),
]

_QKV_BIAS_LAYER = [
    ("q_proj.bias", "model.layers.{i}.self_attn.q_proj.bias", None),
    ("k_proj.bias", "model.layers.{i}.self_attn.k_proj.bias", None),
    ("v_proj.bias", "model.layers.{i}.self_attn.v_proj.bias", None),
]

_QK_NORM_LAYER = [
    ("q_norm.weight", "model.layers.{i}.self_attn.q_norm.weight", None),
    ("k_norm.weight", "model.layers.{i}.self_attn.k_norm.weight", None),
]

_MAMBA1_LAYER = [
    ("norm.weight", "backbone.layers.{i}.norm.weight", None),
    ("in_proj.weight", "backbone.layers.{i}.mixer.in_proj.weight", None),
    ("conv1d.weight", "backbone.layers.{i}.mixer.conv1d.weight", None),
    ("conv1d.bias", "backbone.layers.{i}.mixer.conv1d.bias", None),
    ("x_proj.weight", "backbone.layers.{i}.mixer.x_proj.weight", None),
    ("dt_proj.weight", "backbone.layers.{i}.mixer.dt_proj.weight", None),
    ("dt_proj.bias", "backbone.layers.{i}.mixer.dt_proj.bias", None),
    ("A_log", "backbone.layers.{i}.mixer.A_log", None),
    ("D", "backbone.layers.{i}.mixer.D", None),
    ("out_proj.weight", "backbone.layers.{i}.mixer.out_proj.weight", None),
]

_MAMBA2_LAYER = [
    ("block_norm.weight", "backbone.layers.{i}.norm.weight", None),
    ("in_proj.weight", "backbone.layers.{i}.mixer.in_proj.weight", None),
    ("conv1d.weight", "backbone.layers.{i}.mixer.conv1d.weight", None),
    ("conv1d.bias", "backbone.layers.{i}.mixer.conv1d.bias", None),
    ("dt_bias", "backbone.layers.{i}.mixer.dt_bias", None),
    ("A_log", "backbone.layers.{i}.mixer.A_log", None),
    ("D", "backbone.layers.{i}.mixer.D", None),
    ("norm.weight", "backbone.layers.{i}.mixer.norm.weight", None),
    ("out_proj.weight", "backbone.layers.{i}.mixer.out_proj.weight", None),
]

#: The byte-LM checkpoint format (`config.json` "format": "mojolearn.byte-lm",
#: tensors `embed`, `block{i}.*`, `lm_head`): the same Llama layer order and
#: (out, in) projection layout as `_LLAMA_LAYER`, rotate-half rotary, and NO
#: final norm (the head reads the last residual stream). `norm_name` is None.
BYTE_LM_FORMAT = "mojolearn.byte-lm"
BYTE_LM_MODEL_TYPE = "mojolearn-byte-lm"

_BYTE_LM_LAYER = [
    ("input_layernorm.weight", "block{i}.norm1_w", None),
    ("post_attention_layernorm.weight", "block{i}.norm2_w", None),
    ("q_proj.weight", "block{i}.w_q", None),
    ("k_proj.weight", "block{i}.w_k", None),
    ("v_proj.weight", "block{i}.w_v", None),
    ("o_proj.weight", "block{i}.w_o", None),
    ("gate_proj.weight", "block{i}.w_gate", None),
    ("up_proj.weight", "block{i}.w_up", None),
    ("down_proj.weight", "block{i}.w_down", None),
]

#: byte-LM config field -> the value the stack spells (anything else is refused by name)
_BYTE_LM_FIXED = {"positional_encoding": "rotary", "rope_pairing": "rotate_half", "norm": "rmsnorm",
                  "norm_placement": "pre-norm, no final norm", "mlp": "swiglu", "bias": False,
                  "tie_word_embeddings": False}


def _plan(cfg, mt=None):
    # Every row's planner: the native one. The Python `_plan_<family>`
    # bodies (field reads, refusals, head_dim/rank arithmetic) were dead
    # duplicates of `model_plan_for_binding` and are gone (lane
    # py-lm-neural, 2026-11-14).
    return plan_for(cfg)


#: model_type -> (plan function, tokenizer family, one-line note). `smollm`
#: is not a model_type: SmolLM and SmolLM2 ship `model_type: "llama"` and
#: land on the llama row; the alias is kept so a caller can look it up.
FAMILIES = {
    "llama": (_plan, "bytelevel-bpe (Llama 3) or sentencepiece (Llama 2, refused)",
              "Llama 2/3, TinyLlama, SmolLM, SmolLM2 (model_type llama)"),
    "smollm": (_plan, "bytelevel-bpe", "alias of llama; SmolLM ships model_type llama"),
    "mistral": (_plan, "sentencepiece (refused) or bytelevel-bpe (v3 tekken)",
                "sliding_window -> window"),
    "qwen2": (_plan, "bytelevel-bpe (Qwen 2 pattern)", "qkv_bias True"),
    "qwen3": (_plan, "bytelevel-bpe (Qwen 2 pattern)", "qk_norm True"),
    "gemma": (_plan, "sentencepiece (refused)", "REFUSED: GeGLU, (1+w) norm, embedding scale"),
    "gemma2": (_plan, "sentencepiece (refused)", "REFUSED: softcapping and the rest of gemma"),
    "phi3": (_plan, "sentencepiece (refused)", "fused qkv_proj / gate_up_proj split by rows"),
    "mamba": (_plan, "bytelevel-bpe (GPT-NeoX pattern = GPT-2's)", "profile constants checked"),
    "mamba2": (_plan, "bytelevel-bpe (GPT-NeoX pattern = GPT-2's)", "profile constants checked"),
}


#: The byte-LM format's row (not in FAMILIES: it is keyed by `format`, not `model_type`).
BYTE_LM_ROW = (_plan, "bytelevel-bpe (GPT-2 pattern)", "byte-LM format: no final norm, untied head")


def plan_for(config):
    """The `ModelPlan` of an `HFConfig` (or a dict), or an `UnsupportedModel`
    that names the unsupported model type or field."""
    return _NATIVE.model_plan_for(config, _NATIVE_CONTEXT)

# Static connection metadata; native planning never calls registry functions.
_NATIVE_CONTEXT = {
    "config_class": HFConfig, "plan_class": ModelPlan, "unsupported_class": UnsupportedModel,
    "families": FAMILIES, "defaults": INTERFACE_DEFAULTS, "llama_layer": _LLAMA_LAYER,
    "qkv_bias": _QKV_BIAS_LAYER, "qk_norm": _QK_NORM_LAYER, "mamba1_layer": _MAMBA1_LAYER,
    "mamba2_layer": _MAMBA2_LAYER, "byte_row": BYTE_LM_ROW, "byte_layer": _BYTE_LM_LAYER,
    "byte_fixed": _BYTE_LM_FIXED,
}

_NATIVE.model_config_connect(_NATIVE_CONTEXT)
