# SPDX-License-Identifier: Apache-2.0
"""The block OPTIONS on the Python surface (lane/block-options, 2026-09-17):

    TransformerBlock(weights, *, n_heads, n_kv_heads=None, head_dim=None,
        window=0, rope_theta=10000.0, rope_scaling=None, rope_dim=None,
        max_positions=8192, qkv_bias=False, o_bias=False, norm="rmsnorm",
        norm_eps=1e-6, norm_bias=False, mlp="swiglu", mlp_bias=False,
        qk_norm=False, attn_softcap=None)

WHAT IS ASSERTED. (1) Every unsupported value or combination, and every
weight-dict presence/flag mismatch, is refused BY NAME with the value in
the message, with NO binding. (2) The default record sends the OLD lists
(no tail), and so does a record whose every value equals the default by
bits. (3) With the IDENTICAL transformer binding built: a record that is
non-default in the tail but arithmetically inert (`linear` scaling at
factor 1.0) reproduces the default block's bits, which proves the
extended lists and the options constructor move no bit on their own;
every option moves the output against the default (a wired-but-inert
option is the failure this exists to catch); every option's output agrees
with a float64 NumPy transcription of the reference arithmetic to an
adopted tolerance (rtol 1e-4, atol 1e-5: a tolerance ANCHOR, the bitwise
oracle is the lane's `transformer_options_check.mojo`); a token-by-token
decode equals the whole prefill bit for bit under a record that touches
positions (clause (d) through the public API); and `backward` refuses a
non-default record by name. (4) `TransformerBlockInference` takes the
same constructor.

RUN OWED. Written on a box that may not run it; nothing here has been
executed. `cd python && python3 -m pytest mojolearn/tests/test_transformer_options.py`
after `bash bindings/build_transformer.sh` (or the host binding for the
CPU column). Tests needing a binding skip by name without one.
"""
import math
import os

import numpy as np
import pytest

os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")

from mojolearn import _transformer_impl as impl  # noqa: E402
from mojolearn.transformer import TransformerBlock  # noqa: E402

DM, NH, NKV, HD, IT = 32, 2, 1, 16, 64
B, L = 2, 6
TOL = dict(rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------- fixtures

def _u(rng, shape, lo, hi):
    return (lo + (hi - lo) * rng.random(shape)).astype(np.float32)


def _weights(seed=0x424C4B, dm=DM, nh=NH, nkv=NKV, hd=HD, it=IT, **flags):
    """The nine, plus the optional tensors the flags switch on, in the
    shapes the constructor wants. `gated=False` drops gate_proj.weight."""
    rng = np.random.default_rng(seed)
    qw, kw = nh * hd, nkv * hd
    s_in, s_it = float(dm) ** -0.5, float(it) ** -0.5
    w = {
        "input_layernorm.weight": _u(rng, (dm,), 0.5, 1.5),
        "post_attention_layernorm.weight": _u(rng, (dm,), 0.5, 1.5),
        "q_proj.weight": _u(rng, (qw, dm), -s_in, s_in),
        "k_proj.weight": _u(rng, (kw, dm), -s_in, s_in),
        "v_proj.weight": _u(rng, (kw, dm), -s_in, s_in),
        "o_proj.weight": _u(rng, (dm, qw), -s_in, s_in),
        "gate_proj.weight": _u(rng, (it, dm), -s_in, s_in),
        "up_proj.weight": _u(rng, (it, dm), -s_in, s_in),
        "down_proj.weight": _u(rng, (dm, it), -s_it, s_it),
    }
    if not flags.get("gated", True):
        del w["gate_proj.weight"]
    if flags.get("qkv_bias"):
        w["q_proj.bias"] = _u(rng, (qw,), -0.25, 0.25)
        w["k_proj.bias"] = _u(rng, (kw,), -0.25, 0.25)
        w["v_proj.bias"] = _u(rng, (kw,), -0.25, 0.25)
    if flags.get("o_bias"):
        w["o_proj.bias"] = _u(rng, (dm,), -0.25, 0.25)
    if flags.get("norm_bias"):
        w["input_layernorm.bias"] = _u(rng, (dm,), -0.5, 0.5)
        w["post_attention_layernorm.bias"] = _u(rng, (dm,), -0.5, 0.5)
    if flags.get("mlp_bias"):
        w["up_proj.bias"] = _u(rng, (it,), -0.25, 0.25)
        w["down_proj.bias"] = _u(rng, (dm,), -0.25, 0.25)
        if flags.get("gated", True):
            w["gate_proj.bias"] = _u(rng, (it,), -0.25, 0.25)
    if flags.get("qk_norm"):
        w["q_norm.weight"] = _u(rng, (hd,), 0.5, 1.5)
        w["k_norm.weight"] = _u(rng, (hd,), 0.5, 1.5)
    return w


def _x(seed=0x58, b=B, l=L, dm=DM):
    return _u(np.random.default_rng(seed), (b, l, dm), -2.0, 2.0)


def _bits(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32)).ravel().view(np.uint32)


def _same(a, b):
    return np.shape(a) == np.shape(b) and np.array_equal(_bits(a), _bits(b))


def _binding_or_skip(blk):
    try:
        return blk._extension()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"transformer binding not loaded: {exc}")


# ------------------------------------------------ the float64 reference

_erf = np.vectorize(math.erf)


def _ref_inv_freq(cfg, rd):
    theta = float(cfg.get("rope_theta", 10000.0))
    inv = 1.0 / (theta ** (np.arange(0, rd, 2, dtype=np.float64) / rd))
    sc = cfg.get("rope_scaling")
    if sc is None:
        return inv
    if sc["type"] == "linear":
        return inv / float(sc["factor"])
    factor, lo, hi = float(sc["factor"]), float(sc["low_freq_factor"]), float(sc["high_freq_factor"])
    old = float(sc["original_max_position_embeddings"])
    low_wl, high_wl = old / lo, old / hi
    wl = 2 * math.pi / inv
    inv_l = np.where(wl > low_wl, inv / factor, inv)
    smooth = (old / wl - lo) / (hi - lo)
    smoothed = (1 - smooth) * inv_l / factor + smooth * inv_l
    medium = ~(wl < high_wl) & ~(wl > low_wl)
    return np.where(medium, smoothed, inv_l)


def _ref_rope(v, cfg, positions, hd):
    rd = int(cfg.get("rope_dim") or hd)
    inv = _ref_inv_freq(cfg, rd)
    ang = positions[:, None].astype(np.float64) * inv[None, :]  # (L, rd/2)
    cos, sin = np.cos(ang), np.sin(ang)
    out = v.copy()
    head = v[..., :rd]
    h = rd // 2
    x1, x2 = head[..., :h], head[..., h:]
    rot = np.concatenate([-x2, x1], axis=-1)
    c = np.concatenate([cos, cos], axis=-1)[None, :, None, :]
    s = np.concatenate([sin, sin], axis=-1)[None, :, None, :]
    out[..., :rd] = head * c + rot * s
    return out


def _ref_norm(x, w, b, cfg, eps):
    kind = cfg.get("norm", "rmsnorm")
    if kind == "layernorm":
        mean = x.mean(-1, keepdims=True)
        var = ((x - mean) ** 2).mean(-1, keepdims=True)
        y = (x - mean) / np.sqrt(var + eps) * w
        return y + b if b is not None else y
    rstd = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    ww = (1.0 + w) if kind == "rmsnorm_offset" else w
    return x * rstd * ww


def _ref_act(z, mlp):
    if mlp == "swiglu":
        return z / (1.0 + np.exp(-z))
    if mlp in ("gelu", "geglu"):
        return 0.5 * z * (1.0 + _erf(z / math.sqrt(2.0)))
    return 0.5 * z * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (z + 0.044715 * z ** 3)))


def _ref_block(w, x, cfg, n_heads=NH, n_kv=NKV, head_dim=HD):
    """The contract's block order in float64 with the options applied where
    the reference applies them (bias after the projection, qk norm before
    RoPE, softcap after the scale and before the mask, gelu/geglu in S20's
    place). A TOLERANCE anchor, not the oracle."""
    f = lambda k: np.asarray(w[k], np.float64)  # noqa: E731
    b64 = lambda k: (f(k) if k in w else None)  # noqa: E731
    eps = float(cfg.get("norm_eps", 1e-6))
    mlp = cfg.get("mlp", "swiglu")
    gated = mlp in ("swiglu", "geglu", "geglu_tanh")
    x = np.asarray(x, np.float64)
    b, l, dm = x.shape
    n_rep = n_heads // n_kv
    h = _ref_norm(x, f("input_layernorm.weight"), b64("input_layernorm.bias"), cfg, eps)
    q = h @ f("q_proj.weight").T
    k = h @ f("k_proj.weight").T
    v = h @ f("v_proj.weight").T
    if "q_proj.bias" in w:
        q, k, v = q + f("q_proj.bias"), k + f("k_proj.bias"), v + f("v_proj.bias")
    q = q.reshape(b, l, n_heads, head_dim)
    k = k.reshape(b, l, n_kv, head_dim)
    v = v.reshape(b, l, n_kv, head_dim)
    if cfg.get("qk_norm"):
        qcfg = {"norm": "rmsnorm_offset" if cfg.get("norm") == "rmsnorm_offset" else "rmsnorm"}
        q = _ref_norm(q, f("q_norm.weight"), None, qcfg, eps)
        k = _ref_norm(k, f("k_norm.weight"), None, qcfg, eps)
    pos = np.arange(l)
    q = _ref_rope(q, cfg, pos, head_dim)
    k = _ref_rope(k, cfg, pos, head_dim)
    k = np.repeat(k, n_rep, axis=2)
    v = np.repeat(v, n_rep, axis=2)
    scores = np.einsum("blhd,bmhd->bhlm", q, k) * (head_dim ** -0.5)
    cap = cfg.get("attn_softcap")
    if cap is not None:
        scores = np.tanh(scores / cap) * cap
    mask = np.triu(np.ones((l, l), bool), 1)
    scores = scores + np.where(mask, np.finfo(np.float32).min, 0.0)
    scores = scores - scores.max(-1, keepdims=True)
    e = np.exp(scores)
    p = e / e.sum(-1, keepdims=True)
    ctx = np.einsum("bhlm,bmhd->blhd", p, v).reshape(b, l, n_heads * head_dim)
    o = ctx @ f("o_proj.weight").T
    if "o_proj.bias" in w:
        o = o + f("o_proj.bias")
    r1 = x + o
    h2 = _ref_norm(r1, f("post_attention_layernorm.weight"),
                   b64("post_attention_layernorm.bias"), cfg, eps)
    up = h2 @ f("up_proj.weight").T
    if "up_proj.bias" in w:
        up = up + f("up_proj.bias")
    if gated:
        gate = h2 @ f("gate_proj.weight").T
        if "gate_proj.bias" in w:
            gate = gate + f("gate_proj.bias")
        inner = _ref_act(gate, mlp) * up
    else:
        inner = _ref_act(up, mlp)
    down = inner @ f("down_proj.weight").T
    if "down_proj.bias" in w:
        down = down + f("down_proj.bias")
    return r1 + down


# ---------------------------------------------------- (1) refusals, no binding

@pytest.mark.parametrize("kwargs, needle", [
    (dict(norm="batchnorm"), "norm='batchnorm'"),
    (dict(mlp="relu"), "mlp='relu'"),
    (dict(rope_scaling={"type": "yarn", "factor": 2.0}), "'yarn'"),
    (dict(rope_scaling={"type": "linear"}), "'factor'"),
    (dict(rope_scaling={"type": "llama3", "factor": 8.0}), "low_freq_factor"),
    (dict(rope_scaling={"type": "llama3", "factor": 8.0, "low_freq_factor": 4.0,
                        "high_freq_factor": 1.0, "original_max_position_embeddings": 8192}),
     "high_freq_factor > low_freq_factor"),
    (dict(rope_dim=7), "rope_dim"),   # odd; 6 is even and legal (half 3)
    (dict(rope_dim=18), "rope_dim"),
    (dict(norm_bias=True), "norm_bias=True needs norm='layernorm'"),
    (dict(norm="rmsnorm", qk_norm=True, norm_bias=True), "norm_bias=True"),
    (dict(norm="layernorm", qk_norm=True), "qk_norm=True with norm='layernorm'"),
    (dict(attn_softcap=0.0), "attn_softcap"),
    (dict(attn_softcap=-1.0), "attn_softcap"),
    (dict(attn_softcap=float("nan")), "attn_softcap"),
    (dict(rope_theta=0.0), "rope_theta"),
    (dict(norm_eps=0.0), "norm_eps"),
    (dict(max_positions=0), "max_positions"),
])
def test_unsupported_option_values_are_refused_by_name(kwargs, needle):
    with pytest.raises((ValueError, TypeError)) as e:
        TransformerBlock(_weights(), n_heads=NH, n_kv_heads=NKV, **kwargs)
    assert needle in str(e.value), str(e.value)


@pytest.mark.parametrize("flag, name", [
    ("qkv_bias", "q_proj.bias"), ("o_bias", "o_proj.bias"),
    ("norm_bias", "input_layernorm.bias"), ("mlp_bias", "up_proj.bias"),
    ("qk_norm", "q_norm.weight"),
])
def test_a_tensor_present_while_its_option_is_off_is_refused_by_its_name(flag, name):
    w = _weights(**{flag: True})
    with pytest.raises(ValueError) as e:
        TransformerBlock(w, n_heads=NH, n_kv_heads=NKV)
    assert name in str(e.value) and "option is off" in str(e.value)


@pytest.mark.parametrize("flag, kwargs, name", [
    ("qkv_bias", dict(qkv_bias=True), "k_proj.bias"),
    ("o_bias", dict(o_bias=True), "o_proj.bias"),
    ("norm_bias", dict(norm="layernorm", norm_bias=True), "post_attention_layernorm.bias"),
    ("mlp_bias", dict(mlp_bias=True), "gate_proj.bias"),
    ("qk_norm", dict(qk_norm=True), "k_norm.weight"),
])
def test_a_tensor_missing_while_its_option_is_on_is_refused_by_its_name(flag, kwargs, name):
    w = _weights(**{flag: True})
    del w[name]
    with pytest.raises(ValueError) as e:
        TransformerBlock(w, n_heads=NH, n_kv_heads=NKV, **kwargs)
    assert name in str(e.value) and "required" in str(e.value)


def test_gate_projection_presence_follows_the_mlp_form():
    with pytest.raises(ValueError) as e:
        TransformerBlock(_weights(), n_heads=NH, n_kv_heads=NKV, mlp="gelu")
    assert "gate_proj.weight is present" in str(e.value)
    with pytest.raises(ValueError) as e:
        TransformerBlock(_weights(gated=False), n_heads=NH, n_kv_heads=NKV)
    assert "missing" in str(e.value) and "gate_proj.weight" in str(e.value)
    blk = TransformerBlock(_weights(gated=False), n_heads=NH, n_kv_heads=NKV, mlp="gelu")
    assert blk.intermediate == IT and blk._w[6] is None and blk._extended
    blk2 = TransformerBlock(_weights(gated=False, mlp_bias=True), n_heads=NH,
                            n_kv_heads=NKV, mlp="gelu_tanh", mlp_bias=True)
    assert blk2._wopt[impl._OPT_NAMES.index("gate_proj.bias")] is None
    assert blk2._wopt[impl._OPT_NAMES.index("up_proj.bias")] is not None


def test_a_wrong_shaped_optional_tensor_is_refused_by_name():
    w = _weights(qkv_bias=True)
    w["k_proj.bias"] = w["q_proj.bias"]  # qw != kw under GQA
    with pytest.raises(ValueError) as e:
        TransformerBlock(w, n_heads=NH, n_kv_heads=NKV, qkv_bias=True)
    assert "k_proj.bias" in str(e.value) and "shape" in str(e.value)


# ---------------------------------- (2) the default record sends the old lists

def test_the_default_record_and_its_bitwise_equals_send_no_tail():
    a = TransformerBlock(_weights(), n_heads=NH, n_kv_heads=NKV)
    b = TransformerBlock(_weights(), n_heads=NH, n_kv_heads=NKV, rope_theta=10000.0,
                         norm_eps=1e-6, rope_dim=HD, max_positions=8192,
                         norm="rmsnorm", mlp="swiglu")
    assert not a._extended and not b._extended
    assert a._opts_tail == b._opts_tail == impl._default_tail()
    assert a._opts_tail[0] == 0x461C4000 and a._opts_tail[11] == 0x358637BD
    addrs, params = a._with_tails([1, 2, 3], [4, 5])
    assert addrs == [1, 2, 3] and params == [4, 5]
    c = TransformerBlock(_weights(), n_heads=NH, n_kv_heads=NKV, norm_eps=1e-5)
    assert c._extended and c._opts_tail[11] != 0x358637BD
    addrs, params = c._with_tails([1, 2, 3], [4, 5])
    assert len(addrs) == 3 + 11 and len(params) == 2 + 17
    assert addrs[3:] == [0] * 11


def test_the_params_tail_order_is_the_documented_one():
    blk = TransformerBlock(
        _weights(qkv_bias=True, o_bias=True, mlp_bias=True, qk_norm=True),
        n_heads=NH, n_kv_heads=NKV, rope_theta=500000.0,
        rope_scaling={"type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                      "high_freq_factor": 4.0, "original_max_position_embeddings": 8192},
        rope_dim=8, max_positions=16384, qkv_bias=True, o_bias=True,
        norm="rmsnorm_offset", norm_eps=1e-5, mlp="geglu_tanh", mlp_bias=True,
        qk_norm=True, attn_softcap=50.0)
    t = blk._opts_tail
    f32 = lambda v: np.float32(v).view(np.uint32).item()  # noqa: E731
    assert t == [f32(500000.0), 2, f32(8.0), f32(1.0), f32(4.0), 8192, 8, 16384,
                 1, 1, 2, f32(1e-5), 0, 4, 1, 1, f32(50.0)]
    tail = blk._tail_addrs()
    # rmsnorm_offset carries no norm bias (entries 4 and 5 are absent by
    # the record's own rule); every other tail slot is present here.
    assert len(tail) == 11 and all(tail[i] != 0 for i in range(11) if i not in (4, 5))
    assert tail[4] == 0 and tail[5] == 0
    assert [n for n in impl._OPT_NAMES] == [
        "q_proj.bias", "k_proj.bias", "v_proj.bias", "o_proj.bias",
        "input_layernorm.bias", "post_attention_layernorm.bias",
        "up_proj.bias", "down_proj.bias", "gate_proj.bias",
        "q_norm.weight", "k_norm.weight"]


# ------------------------------------------ (3) with the binding: the arithmetic

def test_an_arithmetically_inert_tail_reproduces_the_default_bits():
    """`linear` scaling at factor 1.0 divides every inverse frequency by
    exactly 1.0: a NON-default record (the tails are sent, the options
    constructor and the options rope table run) whose bits must be the
    default block's. If they are not, the extended path itself moves bits."""
    base = TransformerBlock(_weights(), n_heads=NH, n_kv_heads=NKV)
    _binding_or_skip(base)
    inert = TransformerBlock(_weights(), n_heads=NH, n_kv_heads=NKV,
                             rope_scaling={"type": "linear", "factor": 1.0})
    assert inert._extended
    x = _x()
    assert _same(base.forward(x), inert.forward(x))
    st_a, st_b = base.allocate_state(B, L + 2), inert.allocate_state(B, L + 2)
    assert _same(base.forward(x, st_a), inert.forward(x, st_b))
    assert _same(base.step(x[:, :1], st_a), inert.step(x[:, :1], st_b))
    assert _same(st_a.k_cache, st_b.k_cache) and _same(st_a.v_cache, st_b.v_cache)


_CASES = {
    "rope_theta": (dict(), dict(rope_theta=500000.0)),
    "rope_linear": (dict(), dict(rope_scaling={"type": "linear", "factor": 4.0})),
    "rope_llama3": (dict(), dict(rope_theta=500000.0, rope_scaling={
        "type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
        "high_freq_factor": 4.0, "original_max_position_embeddings": 8192})),
    "rope_dim": (dict(), dict(rope_dim=8)),
    "qkv_bias": (dict(qkv_bias=True), dict(qkv_bias=True)),
    "o_bias": (dict(o_bias=True), dict(o_bias=True)),
    "layernorm": (dict(), dict(norm="layernorm")),
    "layernorm_bias": (dict(norm_bias=True), dict(norm="layernorm", norm_bias=True)),
    "norm_eps": (dict(), dict(norm_eps=1e-5)),
    "rmsnorm_offset": (dict(), dict(norm="rmsnorm_offset")),
    "gelu": (dict(gated=False), dict(mlp="gelu")),
    "gelu_tanh": (dict(gated=False), dict(mlp="gelu_tanh")),
    "geglu": (dict(), dict(mlp="geglu")),
    "geglu_tanh": (dict(), dict(mlp="geglu_tanh")),
    "mlp_bias": (dict(mlp_bias=True), dict(mlp_bias=True)),
    "mlp_bias_gelu": (dict(gated=False, mlp_bias=True), dict(mlp="gelu", mlp_bias=True)),
    "qk_norm": (dict(qk_norm=True), dict(qk_norm=True)),
    "softcap": (dict(), dict(attn_softcap=50.0)),
    "everything": (dict(qkv_bias=True, o_bias=True, mlp_bias=True, qk_norm=True), dict(
        rope_theta=1000000.0, rope_scaling={"type": "linear", "factor": 2.0},
        rope_dim=8, qkv_bias=True, o_bias=True, norm="rmsnorm_offset",
        norm_eps=1e-5, mlp="geglu_tanh", mlp_bias=True, qk_norm=True,
        attn_softcap=30.0)),
}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_each_option_moves_the_output_and_matches_the_float64_reference(name):
    wflags, kwargs = _CASES[name]
    w = _weights(**wflags)
    blk = TransformerBlock(w, n_heads=NH, n_kv_heads=NKV, **kwargs)
    _binding_or_skip(blk)
    x = _x()
    y = np.asarray(blk.forward(x), np.float64)
    ref = _ref_block(w, x, kwargs)
    np.testing.assert_allclose(y, ref, **TOL)
    base = TransformerBlock(_weights(), n_heads=NH, n_kv_heads=NKV)
    y0 = np.asarray(base.forward(x), np.float64)
    assert not np.array_equal(y, y0), f"{name}: the option is wired but inert"


@pytest.mark.parametrize("name", ["rope_llama3", "rope_dim", "qk_norm", "softcap", "everything"])
def test_decode_equals_prefill_under_the_record(name):
    wflags, kwargs = _CASES[name]
    blk = TransformerBlock(_weights(**wflags), n_heads=NH, n_kv_heads=NKV, **kwargs)
    _binding_or_skip(blk)
    x = _x()
    whole = blk.forward(x)
    st = blk.allocate_state(B, L)
    parts = [blk.forward(x[:, :3], st)]
    for t in range(3, L):
        parts.append(blk.step(x[:, t:t + 1], st))
    assert st.cached_tokens == L
    assert _same(np.concatenate([np.asarray(p) for p in parts], axis=1), whole)
    st2 = blk.allocate_state(B, L)
    blk.forward(x, st2)
    assert _same(st.k_cache, st2.k_cache) and _same(st.v_cache, st2.v_cache)


def test_max_positions_is_the_declared_ceiling_and_the_angle_domain_is_refused_by_name():
    blk = TransformerBlock(_weights(), n_heads=NH, n_kv_heads=NKV, max_positions=4)
    _binding_or_skip(blk)
    with pytest.raises(Exception) as e:
        blk.forward(_x())  # 6 tokens past a ceiling of 4
    assert "max_positions" in str(e.value)
    wide = TransformerBlock(_weights(), n_heads=NH, n_kv_heads=NKV, max_positions=16384)
    st = wide.allocate_state(1, 9000)
    with pytest.raises(Exception) as e:
        wide.forward(_x(b=1, l=1), st)  # a 9000-position table at inv_freq[0] == 1
    assert "Cody-Waite" in str(e.value)
    scaled = TransformerBlock(_weights(), n_heads=NH, n_kv_heads=NKV, max_positions=16384,
                              rope_scaling={"type": "linear", "factor": 2.0})
    st2 = scaled.allocate_state(1, 9000)
    y = scaled.forward(_x(b=1, l=1), st2)  # largest angle 4499.5: admitted
    assert np.isfinite(np.asarray(y)).all() and st2.cached_tokens == 1


def test_backward_refuses_a_non_default_record_by_name():
    blk = TransformerBlock(_weights(), n_heads=NH, n_kv_heads=NKV, norm="layernorm")
    with pytest.raises(NotImplementedError) as e:
        blk.backward(_x(), _x(seed=1))
    assert "default block options only" in str(e.value) and "norm='layernorm'" in str(e.value)


def test_decode_session_carries_the_record():
    wflags, kwargs = _CASES["everything"]
    blk = TransformerBlock(_weights(**wflags), n_heads=NH, n_kv_heads=NKV, **kwargs)
    _binding_or_skip(blk)
    # A BARE `hasattr` PROBE TOOK THIS TEST DOWN ON THE CPU COLUMN. The
    # CPU-only stand-in raises ImportError by name from `__getattr__`
    # (`_backend.py::_HostBinding`) and `hasattr` swallows only
    # AttributeError, so the probe for the GPU session entry raised instead of
    # returning False -- the same trap `_transformer_impl._exports` was
    # written for. It no longer matters WHICH arm answers: since
    # lane/cpu-routes-gpu-only-four (2026-09-20) the host route carries the
    # session too, over `transformer_decode_step`/`transformer_forward`, and
    # the record this test checks (`_CASES["everything"]`, the full options
    # tail) has to survive both arms. So the probe is gone and the session is
    # asked for directly.
    x = _x()
    st = blk.allocate_state(B, L)
    with blk.decode_session(st) as sess:
        parts = [sess.forward(x[:, :3])]
        for t in range(3, L):
            parts.append(sess.step(x[:, t:t + 1]))
    assert _same(np.concatenate([np.asarray(p) for p in parts], axis=1), blk.forward(x))


# ------------------------------------------ (4) the CPU inference class

def test_inference_class_takes_the_same_constructor():
    from mojolearn.neural_inference import TransformerBlockInference
    w = _weights(qkv_bias=True)
    blk = TransformerBlockInference(w, n_heads=NH, n_kv_heads=NKV, qkv_bias=True,
                                    rope_theta=1000000.0)
    assert blk._extended and blk.qkv_bias and blk.rope_theta == 1000000.0
    with pytest.raises(ValueError):
        TransformerBlockInference(w, n_heads=NH, n_kv_heads=NKV)  # bias without its flag
    try:
        blk._extension()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"neural host binding not loaded: {exc}")
    x = _x()
    np.testing.assert_allclose(np.asarray(blk.forward(x), np.float64),
                               _ref_block(w, x, dict(rope_theta=1000000.0)), **TOL)
