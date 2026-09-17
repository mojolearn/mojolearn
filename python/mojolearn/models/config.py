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
import json
import math
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


def _refuse(model_type, field, value, why):
    raise UnsupportedModel(
        f"mojolearn.models: model_type {model_type!r}: {field}={value!r} is not honored: {why}")


class HFConfig:
    """A `config.json` as a read-only mapping with `model_type`, `get`,
    `[]` and `path`."""

    def __init__(self, data, path=None):
        if not isinstance(data, dict):
            raise TypeError("mojolearn.models.HFConfig: the config must be a JSON object")
        self._d = dict(data)
        self.path = path

    @classmethod
    def from_json(cls, path):
        """`path` is `config.json` or the directory holding it."""
        path = os.fspath(path)
        if os.path.isdir(path):
            path = os.path.join(path, "config.json")
        with open(path, "r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except ValueError as exc:
                raise ValueError(f"mojolearn.models.HFConfig: {path} is not JSON: {exc}") from None
        return cls(data, path)

    @property
    def model_type(self):
        mt = self._d.get("model_type")
        if not isinstance(mt, str):
            raise UnsupportedModel(f"mojolearn.models: the config at {self.path!r} has no string model_type")
        return mt

    def get(self, key, default=None):
        return self._d.get(key, default)

    def __getitem__(self, key):
        return self._d[key]

    def __contains__(self, key):
        return key in self._d

    def keys(self):
        return self._d.keys()

    def to_dict(self):
        return dict(self._d)

    def __repr__(self):
        return f"HFConfig(model_type={self._d.get('model_type')!r}, path={self.path!r})"


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
        self.__dict__.update(kw)

    def layer_weights(self, i):
        out = []
        for key, template, rows in self._layer_map:
            out.append((key, template.format(i=i), rows))
        return out

    def checkpoint_names(self):
        names = [self.embed_name, self.norm_name]
        if self.head_name is not None:
            names.append(self.head_name)
        for i in range(self.n_layers):
            for _, name, _ in self.layer_weights(i):
                if name not in names:
                    names.append(name)
        return names

    def __repr__(self):
        return (f"ModelPlan({self.model_type!r}, kind={self.kind!r}, layers={self.n_layers}, "
                f"d_model={self.d_model}, vocab={self.vocab_size})")


# --------------------------------------------------------------- families
#
# Each row: `kind`, `tokenizer` (what tokenizer.py can load for it), the
# checkpoint names, and `plan(cfg)` which reads the fields, refuses by name
# and returns the option dict. The README beside this file renders the same
# rows as a table; keep them in step.

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

_LLAMA_TOP = ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")
_MAMBA_TOP = ("backbone.embeddings.weight", "backbone.norm_f.weight", "lm_head.weight")


def _int(cfg, mt, field, default=None, required=True):
    v = cfg.get(field, default)
    if v is None:
        if required:
            _refuse(mt, field, None, "the field is required and the config lacks it")
        return None
    if isinstance(v, bool) or not isinstance(v, int):
        _refuse(mt, field, v, "an integer is required")
    return v


def _float(cfg, mt, field, default):
    v = cfg.get(field, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        _refuse(mt, field, v, "a number is required")
    return float(v)


def _moe_check(cfg, mt):
    for field in ("num_local_experts", "num_experts", "num_experts_per_tok", "n_routed_experts"):
        v = cfg.get(field)
        if v not in (None, 0, 1):
            _refuse(mt, field, v, "mixture of experts: the block has one dense SwiGLU MLP")


def _rope_check(cfg, mt):
    rs = cfg.get("rope_scaling")
    if rs is not None:
        _refuse(mt, "rope_scaling", rs,
                "rope type 'default' is the only one the interface names a value for; "
                "linear, dynamic, yarn, longrope and llama3 scaling are refused")
    rt = cfg.get("rope_type", "default")
    if rt not in (None, "default"):
        _refuse(mt, "rope_type", rt, "rope type 'default' only")


def _act_check(cfg, mt, field="hidden_act"):
    act = cfg.get(field, "silu")
    if act not in ("silu", "swish"):
        _refuse(mt, field, act, "the block's MLP is SwiGLU (silu gate); mlp values beyond 'swiglu' are not in the interface")


def _sliding_alternation_check(cfg, mt):
    if cfg.get("use_sliding_window"):
        _refuse(mt, "use_sliding_window", True,
                f"sliding-window-alternating layers (max_window_layers={cfg.get('max_window_layers')!r}): "
                "the block takes ONE window for every layer")
    lt = cfg.get("layer_types")
    if lt is not None and len(set(lt)) > 1:
        _refuse(mt, "layer_types", lt, "alternating attention kinds: the block takes one window for every layer")


def _transformer_common(cfg, mt, *, qkv_bias_default=False, qk_norm=False):
    """The Llama-shaped fields, refusing by name what the block cannot take."""
    _moe_check(cfg, mt)
    _rope_check(cfg, mt)
    _act_check(cfg, mt)
    _sliding_alternation_check(cfg, mt)
    d_model = _int(cfg, mt, "hidden_size")
    n_heads = _int(cfg, mt, "num_attention_heads")
    n_kv = _int(cfg, mt, "num_key_value_heads", default=n_heads)
    head_dim = _int(cfg, mt, "head_dim", default=None, required=False)
    if head_dim is None:
        head_dim = d_model // n_heads
    if n_heads * head_dim != d_model and mt not in ("gemma", "gemma2"):
        _refuse(mt, "head_dim", head_dim, f"n_heads*head_dim ({n_heads}*{head_dim}) must equal hidden_size {d_model}")
    if n_heads % n_kv != 0:
        _refuse(mt, "num_key_value_heads", n_kv, f"num_attention_heads {n_heads} is not a multiple of it")
    if head_dim % 2 != 0:
        _refuse(mt, "head_dim", head_dim, "head_dim must be even (the rotary pairs)")
    inter = _int(cfg, mt, "intermediate_size")
    n_layers = _int(cfg, mt, "num_hidden_layers")
    vocab = _int(cfg, mt, "vocab_size")
    eps = _float(cfg, mt, "rms_norm_eps", 1e-6)
    theta = _float(cfg, mt, "rope_theta", 10000.0)
    window = cfg.get("sliding_window")
    if window is None:
        window = 0
    if isinstance(window, bool) or not isinstance(window, int) or window < 0:
        _refuse(mt, "sliding_window", window, "an integer width or null")
    attn_bias = bool(cfg.get("attention_bias", qkv_bias_default))
    mlp_bias = bool(cfg.get("mlp_bias", False))
    if float(cfg.get("attention_dropout", 0.0) or 0.0) != 0.0:
        _refuse(mt, "attention_dropout", cfg.get("attention_dropout"), "inference has no dropout and the profile refuses a nonzero value")
    tie = bool(cfg.get("tie_word_embeddings", False))
    maxpos = _int(cfg, mt, "max_position_embeddings", default=POSITION_CEILING)
    options = dict(INTERFACE_DEFAULTS)
    options.update({
        "n_kv_heads": n_kv if n_kv != n_heads else None,
        "head_dim": head_dim if head_dim != d_model // n_heads else None,
        "window": window,
        "rope_theta": theta,
        "norm_eps": eps,
        "qkv_bias": attn_bias,
        "o_bias": False,
        "mlp_bias": mlp_bias,
        "qk_norm": qk_norm,
    })
    layer = list(_LLAMA_LAYER)
    if attn_bias:
        layer += _QKV_BIAS_LAYER
    if qk_norm:
        layer += _QK_NORM_LAYER
    return dict(kind="transformer", n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                n_kv_heads=n_kv, head_dim=head_dim, intermediate=inter, vocab_size=vocab,
                tie_embeddings=tie, norm_eps=eps, max_position_embeddings=maxpos,
                block_options=options, block_kwargs={}, layer_map=layer,
                embed_name=_LLAMA_TOP[0], norm_name=_LLAMA_TOP[1],
                head_name=None if tie else _LLAMA_TOP[2])


def _plan_llama(cfg, mt):
    return _transformer_common(cfg, mt)


def _plan_mistral(cfg, mt):
    return _transformer_common(cfg, mt)


def _plan_qwen2(cfg, mt):
    # Qwen2 puts a bias on q, k and v and none on o (modeling_qwen2.py);
    # `attention_bias` is absent from its config, so the default is True.
    return _transformer_common(cfg, mt, qkv_bias_default=True)


def _plan_qwen3(cfg, mt):
    # Qwen3: per-head RMSNorm on q and k (q_norm.weight, k_norm.weight of
    # shape (head_dim,)), no attention bias, head_dim explicit.
    return _transformer_common(cfg, mt, qkv_bias_default=False, qk_norm=True)


def _plan_gemma(cfg, mt):
    act = cfg.get("hidden_activation", cfg.get("hidden_act"))
    _refuse(mt, "hidden_activation" if "hidden_activation" in cfg else "hidden_act", act,
            "Gemma's GeGLU (gelu_pytorch_tanh), its (1 + weight) RMSNorm and its sqrt(hidden_size) "
            "embedding scale are not values the interface names (mlp='swiglu', norm='rmsnorm' only)")


def _plan_gemma2(cfg, mt):
    for field in ("attn_logit_softcapping", "final_logit_softcapping", "query_pre_attn_scalar"):
        if cfg.get(field) is not None:
            _refuse(mt, field, cfg.get(field),
                    "Gemma 2's logit softcapping, query scaling, pre/post feed-forward norms and "
                    "alternating sliding window are not honored by the block")
    return _plan_gemma(cfg, mt)


def _plan_phi3(cfg, mt):
    p = _transformer_common(cfg, mt)
    if cfg.get("embd_pdrop", 0.0) or cfg.get("resid_pdrop", 0.0):
        pass  # dropout is a training knob; inference ignores it
    if cfg.get("original_max_position_embeddings") not in (None, p["max_position_embeddings"]):
        _refuse(mt, "original_max_position_embeddings", cfg.get("original_max_position_embeddings"),
                "a longrope-extended context; rope_scaling is refused above and so is its extension")
    nh, nkv, hd, it = p["n_heads"], p["n_kv_heads"], p["head_dim"], p["intermediate"]
    q_rows, kv_rows = nh * hd, nkv * hd
    layer = [
        ("input_layernorm.weight", "model.layers.{i}.input_layernorm.weight", None),
        ("post_attention_layernorm.weight", "model.layers.{i}.post_attention_layernorm.weight", None),
        ("q_proj.weight", "model.layers.{i}.self_attn.qkv_proj.weight", (0, q_rows)),
        ("k_proj.weight", "model.layers.{i}.self_attn.qkv_proj.weight", (q_rows, q_rows + kv_rows)),
        ("v_proj.weight", "model.layers.{i}.self_attn.qkv_proj.weight", (q_rows + kv_rows, q_rows + 2 * kv_rows)),
        ("o_proj.weight", "model.layers.{i}.self_attn.o_proj.weight", None),
        ("gate_proj.weight", "model.layers.{i}.mlp.gate_up_proj.weight", (0, it)),
        ("up_proj.weight", "model.layers.{i}.mlp.gate_up_proj.weight", (it, 2 * it)),
        ("down_proj.weight", "model.layers.{i}.mlp.down_proj.weight", None),
    ]
    p["layer_map"] = layer
    return p


def _plan_mamba(cfg, mt):
    d_model = _int(cfg, mt, "hidden_size")
    n_layers = _int(cfg, mt, "num_hidden_layers")
    vocab = _int(cfg, mt, "vocab_size")
    fixed = {"state_size": 16, "conv_kernel": 4, "expand": 2, "use_bias": False, "use_conv_bias": True}
    for field, want in fixed.items():
        got = cfg.get(field, want)
        if got != want:
            _refuse(mt, field, got, f"Mamba1Block fixes it at {want!r} (mamba/IDENTICAL_MAMBA_CONTRACT.md section 3)")
    rank = cfg.get("time_step_rank", "auto")
    want_rank = int(math.ceil(d_model / 16.0))
    if rank not in ("auto", want_rank):
        _refuse(mt, "time_step_rank", rank, f"Mamba1Block fixes dt_rank at ceil(hidden_size/16) = {want_rank}")
    eps = _float(cfg, mt, "layer_norm_epsilon", 1e-5)
    if eps != 1e-5:
        _refuse(mt, "layer_norm_epsilon", eps, "Mamba1Block fixes rms eps at 1e-5 (0x3727C5AC)")
    if cfg.get("hidden_act", "silu") != "silu":
        _refuse(mt, "hidden_act", cfg.get("hidden_act"), "silu only, as the reference asserts")
    tie = bool(cfg.get("tie_word_embeddings", True))
    return dict(kind="mamba1", n_layers=n_layers, d_model=d_model, vocab_size=vocab,
                tie_embeddings=tie, norm_eps=eps, max_position_embeddings=None,
                block_options={}, block_kwargs={}, layer_map=list(_MAMBA1_LAYER),
                embed_name=_MAMBA_TOP[0], norm_name=_MAMBA_TOP[1],
                head_name=None if tie else _MAMBA_TOP[2])


def _plan_mamba2(cfg, mt):
    d_model = _int(cfg, mt, "hidden_size")
    n_layers = _int(cfg, mt, "num_hidden_layers")
    vocab = _int(cfg, mt, "vocab_size")
    fixed = {"state_size": 128, "head_dim": 64, "expand": 2, "n_groups": 1, "chunk_size": 256,
             "conv_kernel": 4, "use_bias": False, "use_conv_bias": True, "rms_norm": True}
    for field, want in fixed.items():
        got = cfg.get(field, want)
        if got != want:
            _refuse(mt, field, got, f"Mamba2Block fixes it at {want!r} (mamba/IDENTICAL_MAMBA2_CONTRACT.md section 3)")
    if cfg.get("norm_before_gate", False):
        _refuse(mt, "norm_before_gate", True, "the gated norm is gate-then-norm (DEVIATION 787)")
    if d_model % 32 != 0:
        _refuse(mt, "hidden_size", d_model, "Mamba2Block needs a multiple of 32 (headdim 64, expand 2)")
    nh = cfg.get("num_heads")
    if nh is not None and nh != 2 * d_model // 64:
        _refuse(mt, "num_heads", nh, f"expand*hidden_size/head_dim = {2 * d_model // 64} heads is fixed by the profile")
    eps = _float(cfg, mt, "layer_norm_epsilon", 1e-5)
    if eps != 1e-5:
        _refuse(mt, "layer_norm_epsilon", eps, "Mamba2Block fixes rms eps at 1e-5 for both norms")
    limit = cfg.get("time_step_limit", [0.0, float("inf")])
    try:
        lo, hi = float(limit[0]), float(limit[1])
    except (TypeError, ValueError, IndexError):
        _refuse(mt, "time_step_limit", limit, "a pair (lo, hi)")
    tie = bool(cfg.get("tie_word_embeddings", True))
    return dict(kind="mamba2", n_layers=n_layers, d_model=d_model, vocab_size=vocab,
                tie_embeddings=tie, norm_eps=eps, max_position_embeddings=None,
                block_options={}, block_kwargs={"dt_limit": (lo, hi)}, layer_map=list(_MAMBA2_LAYER),
                embed_name=_MAMBA_TOP[0], norm_name=_MAMBA_TOP[1],
                head_name=None if tie else _MAMBA_TOP[2])


#: model_type -> (plan function, tokenizer family, one-line note). `smollm`
#: is not a model_type: SmolLM and SmolLM2 ship `model_type: "llama"` and
#: land on the llama row; the alias is kept so a caller can look it up.
FAMILIES = {
    "llama": (_plan_llama, "bytelevel-bpe (Llama 3) or sentencepiece (Llama 2, refused)",
              "Llama 2/3, TinyLlama, SmolLM, SmolLM2 (model_type llama)"),
    "smollm": (_plan_llama, "bytelevel-bpe", "alias of llama; SmolLM ships model_type llama"),
    "mistral": (_plan_mistral, "sentencepiece (refused) or bytelevel-bpe (v3 tekken)",
                "sliding_window -> window"),
    "qwen2": (_plan_qwen2, "bytelevel-bpe (Qwen 2 pattern)", "qkv_bias True"),
    "qwen3": (_plan_qwen3, "bytelevel-bpe (Qwen 2 pattern)", "qk_norm True"),
    "gemma": (_plan_gemma, "sentencepiece (refused)", "REFUSED: GeGLU, (1+w) norm, embedding scale"),
    "gemma2": (_plan_gemma2, "sentencepiece (refused)", "REFUSED: softcapping and the rest of gemma"),
    "phi3": (_plan_phi3, "sentencepiece (refused)", "fused qkv_proj / gate_up_proj split by rows"),
    "mamba": (_plan_mamba, "bytelevel-bpe (GPT-NeoX pattern = GPT-2's)", "profile constants checked"),
    "mamba2": (_plan_mamba2, "bytelevel-bpe (GPT-NeoX pattern = GPT-2's)", "profile constants checked"),
}


def plan_for(config):
    """The `ModelPlan` of an `HFConfig` (or a dict), or `UnsupportedModel`
    BY NAME."""
    if isinstance(config, dict):
        config = HFConfig(config)
    mt = config.model_type
    row = FAMILIES.get(mt)
    if row is None:
        raise UnsupportedModel(
            f"mojolearn.models: model_type {mt!r} is not in the option matrix; known: {sorted(FAMILIES)}")
    fn, _, _ = row
    fields = fn(config, mt)
    layer_map = fields.pop("layer_map")
    plan = ModelPlan(model_type=mt, config=config, family=row, _layer_map=layer_map, **fields)
    return plan
