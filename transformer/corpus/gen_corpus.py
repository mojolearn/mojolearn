# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Independent reference corpus for the IDENTICAL FP32 transformer block lane.

WHAT THIS IS. A deterministic generator of small Llama-shaped decoder-block
cases (inputs, weights, and per-stage reference outputs) whose expected values
come from SOMEBODY ELSE'S algorithm rather than from this repository's own
tally. The block order and every step is HuggingFace
`transformers/src/transformers/models/llama/modeling_llama.py` at commit
`d56c55bf564ddb176759eb6ec199442682564916`, re-implemented here in pure torch
with each step cited by symbol and line. The `transformers` package is NOT
imported; it is not installed on this machine and importing it would only add
a dependency on its own kernel dispatch. The stage list, the stage names and
the stage definitions are `transformer/IDENTICAL_TRANSFORMER_CONTRACT.md`
section 9, in card order.

WHY IT EXISTS. Every other gate in this lane compares our code against our own
code. A host oracle and a device kernel that agree prove they agree; they do
not prove either is a transformer. This corpus is the only instrument in the
fan-out that can say we ported the wrong algorithm. The mamba lane's sibling
corpus found exactly one such thing (a stage-naming disagreement about whether
`dt_proj.out` includes its bias) and no amount of internal agreement would
ever have surfaced it.

WHAT IT IS NOT. `ref64/` is a TOLERANCE reference, not a bitwise certificate.
The lane's bitwise oracle is its own pinned host oracle. `ref32/` is a plain
float32 CPU run of the same stages; it is INFORMATIVE ONLY, it is never a
target and never a gate, and its whole job is to calibrate how tight a
tolerance is honest. Nothing here has been compiled, run against a device, or
compared to a lane dump. NOTHING IN THIS FILE HAS BEEN EXECUTED BY ITS AUTHOR.
Every claim in every docstring is CONSTRUCTION or PREDICTION.

THE PRECISION RULE, which is the single most important decision in this file
(DEVIATION 1042). The corpus computes in float64 EXCEPT at the five places
where the profile pins an FP32 value as a CONSTANT OF THE ARITHMETIC, and
there it uses the FP32 bits widened to float64:

  1. the rotary inverse frequencies used to build the angle (contract S6,
     LRE:108, computed in FP32 by the reference);
  2. the rotary ANGLE `position * inv_freq` (contract S7, LRE:114-122, an FP32
     matmul with autocast EXPLICITLY DISABLED);
  3. the attention scale `head_dim ** -0.5` (contract section 3, LAR:226);
  4. the RMSNorm epsilon `1e-6` (contract section 3);
  5. the additive mask fill `-3.4028234663852886e+38` (contract section 3;
     note this is `finfo(float32).min` and NOT `finfo(dtype).min`, so the
     float64 reference uses the FP32 constant ON PURPOSE, because the profile
     pins the number and not the expression).

Items 1 and 2 are not stylistic. RoPE's angle at absolute position p is
`p * inv_freq`, so a one-ulp difference in `inv_freq` becomes a p-ulp
difference in the angle, and `d(cos)/d(angle) = 1`. At the profile's ceiling
(`p < 8192`, DEVIATION 812) that is an amplification of about 8000x, which
puts the disagreement between a float64 angle table and the reference's FP32
angle table orders of magnitude ABOVE float32 epsilon. A corpus built on the
float64 angle would fail every implementation on earth including torch's own
FP32, and it would fail hardest at exactly the case worth the most. `--verify`
control 5 MEASURES and PRINTS that amplification per case, so the number is on
the record rather than in this docstring.

WHAT WOULD FALSIFY THE PRECISION RULE. If control 5 prints a cos gap at or
below float32 epsilon (1.19e-07) even for the far-position case, the rule is
unnecessary and this file is more complicated than it needs to be. If
`--self-test` shows `rope.cos` needing a far looser rtol than every other
stage on the far-position case, the reference arm's own cosine is the limit
and the gate there is loose because of the reference, not because of us; the
`rtol_correctly_rounded` column exists to make that visible.

EVERY TENSOR ELEMENT IS A HASHED VALUE (standing rule: uniform test data hides
permutation). value = f32(lo + (hi - lo) * top24bits(splitmix64(key + i)))
with key = splitmix64(seed ^ (tensor_id << 32)); the arithmetic is exact in
float64 for every (lo, hi) used here and rounds ONCE to float32, so a Mojo
fixture can regenerate the identical bytes from README.md's spec with no
library RNG. The generator asserts that exactness against exact rationals for
every element of every tensor. Hash spec name:
`mojolearn.transformer.corpus.hash.v1`.

Run (see README.md):
    python transformer/corpus/gen_corpus.py [--out DIR] [--verify]
    python transformer/corpus/gen_corpus.py --self-test

DEVIATIONS 1040-1049 belong to this lane; see README.md, section "Deviations".
"""

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from fractions import Fraction

import numpy as np
import torch

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

HASH_SPEC = "mojolearn.transformer.corpus.hash.v1"
CORPUS_VERSION = "transformer-corpus-v1"
PROFILE = "mojolearn.identical.transformer.fp32.v1"
TRANSFORMERS_COMMIT = "d56c55bf564ddb176759eb6ec199442682564916"

# --------------------------------------------------------------------------
# Hash spec (mojolearn.transformer.corpus.hash.v1). README.md carries the
# C-like statement of this; the code below is the executable copy. It is the
# same splitmix64 construction as `mojolearn.mamba.corpus.hash.v1` with a
# different name, a different seed base and a different tensor-id table, so a
# Mojo fixture that already implements the mamba spec implements this one by
# changing three constants. All uint64 arithmetic wraps.
# --------------------------------------------------------------------------
M64 = (1 << 64) - 1
SM_GAMMA = 0x9E3779B97F4A7C15
SM_M1 = 0xBF58476D1CE4E5B9
SM_M2 = 0x94D049BB133111EB


def splitmix64_scalar(z):
    z = (z + SM_GAMMA) & M64
    z = ((z ^ (z >> 30)) * SM_M1) & M64
    z = ((z ^ (z >> 27)) * SM_M2) & M64
    return z ^ (z >> 31)


def splitmix64_array(z):
    """Vectorized splitmix64 over a uint64 numpy array (wrapping arithmetic)."""
    z = z.astype(np.uint64)
    with np.errstate(over="ignore"):
        z = z + np.uint64(SM_GAMMA)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(SM_M1)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(SM_M2)
        return z ^ (z >> np.uint64(31))


def hashed_unit(seed, tensor_id, n):
    """n values f_i = top24bits(splitmix64(key + i)) * 2^-24, key = splitmix64(seed ^ (tid << 32)).

    Returned as float64; every value is an exact multiple of 2^-24 in [0, 1)."""
    key = splitmix64_scalar((seed ^ (tensor_id << 32)) & M64)
    idx = np.arange(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        h = splitmix64_array(np.uint64(key) + idx)
    top24 = (h >> np.uint64(40)).astype(np.float64)
    return top24 * (2.0 ** -24)


def _is_dyadic_small(v):
    fr = Fraction(v)
    den = fr.denominator
    return den & (den - 1) == 0


def map_range(f_unit, lo, hi):
    """f32(lo + (hi - lo) * f), evaluated exactly in float64 and rounded ONCE.

    (lo, hi) must be dyadic rationals. The exactness is not assumed: it is
    checked against exact rationals for every element. A failure here means
    the hash spec as written in README.md is not implementable in Mojo, which
    is a defect in the spec and not in torch."""
    assert _is_dyadic_small(lo) and _is_dyadic_small(hi), (lo, hi)
    span = float(hi) - float(lo)
    v64 = float(lo) + span * f_unit
    flo, fspan = Fraction(lo), Fraction(hi) - Fraction(lo)
    for fv, v in zip(f_unit.tolist(), v64.tolist()):
        exact = flo + fspan * Fraction(fv)
        assert Fraction(v) == exact, ("float64 evaluation is not exact", lo, hi, fv, v)
    return v64.astype(np.float32)


# --------------------------------------------------------------------------
# Profile constants (contract section 3). Every one of these is FROZEN; a
# change here is a v2 of the profile, not an amendment to v1.
# --------------------------------------------------------------------------
RMS_EPS_F32 = np.float32(1e-6)                       # rms_norm_eps, 0x358637BD
ROPE_THETA_F32 = np.float32(10000.0)                 # rope base, 0x461C4000
MASK_FILL_F32 = np.float32(-3.4028234663852886e38)   # masking_utils.py:601-603, 0xFF7FFFFF
UNMASKED_FILL_F32 = np.float32(0.0)                  # ADDED, and may not be elided
FP32_MAX = 3.4028234663852886e38
FP32_MIN_NORMAL = 1.1754943508222875e-38
FP32_EPS = 1.1920928955078125e-07
MAX_ABS_POSITION = 8192                              # DEVIATION 812, strictly less than

TENSOR_IDS = {
    "x": 1,
    "norm1.weight": 2,
    "norm2.weight": 3,
    "q_proj.weight": 4,
    "k_proj.weight": 5,
    "v_proj.weight": 6,
    "o_proj.weight": 7,
    "gate_proj.weight": 8,
    "up_proj.weight": 9,
    "down_proj.weight": 10,
}

# Contract section 9, the thirty card stages IN CARD ORDER. This list is the
# contract's, not a convenience; a stage not on this list is not on the card.
STAGE_ORDER = [
    "input.x", "norm1.sumsq", "norm1.out",
    "q_proj.out", "k_proj.out", "v_proj.out",
    "rope.inv_freq", "rope.cos", "rope.sin",
    "q_rope.out", "k_rope.out",
    "kv.k_cache", "kv.v_cache",
    "attn.scores", "attn.masked", "attn.max", "attn.exp", "attn.denom",
    "attn.weights", "attn.ctx",
    "o_proj.out", "residual1.out",
    "norm2.sumsq", "norm2.out",
    "gate_proj.out", "up_proj.out", "silu.out", "mlp.gated",
    "down_proj.out", "residual2.out",
]

# Stages whose size is QUADRATIC in L. The reduced set drops exactly these and
# exists only to keep the committed corpus small on the longest case.
REDUCED_SKIP = ["attn.scores", "attn.masked", "attn.exp", "attn.weights"]

# Stages laid out batch-first rather than token-major [M, F]. Everything not
# here has M = B*L as its leading axis, which is the card's convention.
BATCH_FIRST = {"kv.k_cache", "kv.v_cache", "attn.scores", "attn.masked",
               "attn.exp", "attn.weights", "attn.max", "attn.denom"}
# Stages that are per-configuration rather than per-token (contract section 9
# records these once per configuration, not per call).
CONFIG_STAGES = {"rope.inv_freq", "rope.cos", "rope.sin"}

STAGE_DEFS = {
    "input.x": "the block input as given, token-major [M, d_model], M = B*L",
    "norm1.sumsq": "sum_j x_j^2 over d_model, the SUM and NOT the mean. The reference spells "
                   "hidden_states.pow(2).mean(-1) (LRN:65), which is this divided by d_model. "
                   "Contract S1. A dump carrying the mean differs by exactly a factor d_model "
                   "and the checker NAMES that rather than calling it an arithmetic defect. "
                   "This is deliberately the same trap the mamba corpus hit on dt_proj.out.",
    "norm1.out": "norm1.weight * (x * rsqrt(norm1.sumsq/d_model + 1e-6))  (LRN:65-67), S2-S4",
    "q_proj.out": "norm1.out @ q_proj.weight^T, bias=False  (LAR:230, :253), S5",
    "k_proj.out": "norm1.out @ k_proj.weight^T, bias=False  (LAR:233, :254), S5",
    "v_proj.out": "norm1.out @ v_proj.weight^T, bias=False  (LAR:236, :255), S5",
    "rope.inv_freq": "1 / theta**(2i/head_dim), i = 0..head_dim/2-1. ref64 is the exact "
                     "float64 value, a tolerance target for portable_powf. ref32 is the "
                     "REFERENCE'S OWN FP32 spelling (LRE:108) and it is the ANCHOR the "
                     "rope.cos and rope.sin references are built on. See the precision rule.",
    "rope.cos": "cos(f32(f32(position) * inv_freq_f32)) evaluated in float64, "
                "[n_positions, head_dim/2]. The HALF table, before the cat(freqs, freqs) of "
                "LRE:123 which is a COPY and not a seam. S7, S8",
    "rope.sin": "sin of the same FP32 angle, [n_positions, head_dim/2]. S7, S8",
    "q_rope.out": "(q*cos) + (rotate_half(q)*sin) (LAR:158, LRH:130-134), token-major "
                  "[M, n_heads*head_dim]. S9, S10",
    "k_rope.out": "the same for k, [M, n_kv_heads*head_dim]. S9, S10",
    "kv.k_cache": "k_rope laid out as [B, n_kv_heads, S, head_dim]. A COPY, not a seam; it is "
                  "on the card because a permutation here is invisible in every downstream "
                  "stage that sums over the key axis.",
    "kv.v_cache": "v_proj laid out as [B, n_kv_heads, S, head_dim]. v is NOT roped.",
    "attn.scores": "matmul(q, k^T) * scale, [B, n_heads, L, S] (EAF:204). S11, S12",
    "attn.masked": "attn.scores + mask; the mask is ADDED and not selected, fill "
                   "-3.4028234663852886e38 where masked and +0.0 where not (EAF:206, "
                   "masking_utils.py:601-603). Masking is by ABSOLUTE position. S13",
    "attn.max": "the row max over the WHOLE row INCLUDING masked cells (contract 5.2), "
                "[B, n_heads, L]. S14",
    "attn.exp": "exp(attn.masked - attn.max), [B, n_heads, L, S]. S15, S16",
    "attn.denom": "the sum of attn.exp over the key axis, ascending by ABSOLUTE key index, "
                  "[B, n_heads, L]. S17",
    "attn.weights": "attn.exp / attn.denom, ONE division per weight. S18",
    "attn.ctx": "sum_j attn.weights[j] * v[j], transposed to token-major then flattened, "
                "[M, n_heads*head_dim] (EAF:210 then :211). S19",
    "o_proj.out": "attn.ctx @ o_proj.weight^T, bias=False (LAR:239, :280). S5",
    "residual1.out": "input.x + o_proj.out (LDL:317). S22",
    "norm2.sumsq": "sum_j residual1.out_j^2, the SUM (see norm1.sumsq). S1",
    "norm2.out": "norm2.weight * (residual1.out * rsqrt(norm2.sumsq/d_model + 1e-6)). S2-S4",
    "gate_proj.out": "norm2.out @ gate_proj.weight^T (LMLP:175). S5",
    "up_proj.out": "norm2.out @ up_proj.weight^T (LMLP:175). S5",
    "silu.out": "gate_proj.out / (1 + exp(-gate_proj.out)), ATen's ONE-DIVISION spelling and "
                "NOT x*sigmoid(x), which is two roundings (contract S20). S20",
    "mlp.gated": "silu.out * up_proj.out (LMLP:175). S21",
    "down_proj.out": "mlp.gated @ down_proj.weight^T (LMLP:175). S5",
    "residual2.out": "residual1.out + down_proj.out (LDL:323). S23",
}

assert set(STAGE_DEFS) == set(STAGE_ORDER), "stage definitions and card order disagree"


# --------------------------------------------------------------------------
# The named perturbations. Each is a DELIBERATE WRONG ANSWER used to MEASURE
# what a case can see. No case is generated with one applied. They exist so
# that "what this case cannot certify" is a measurement rather than a guess,
# which is the single lesson the mamba lane's base_b1_l1_d8 taught (at L=1 a
# token-major and a channel-major dump are the same bytes, so that case could
# not certify a reindexing AT ALL).
# --------------------------------------------------------------------------
PERTURBATIONS = {
    "rope_relative_position": "index the rotary table from the start of the current slice "
                              "instead of by ABSOLUTE position (contract 7.2, sabotage "
                              "S07_ROPE_RELATIVE_POSITION). Bit-inert whenever pos0 == 0, "
                              "which is why one far-position case carries this whole clause.",
    "rope_halves_swapped": "rotate_half returns cat(x2, -x1) instead of cat(-x2, x1) "
                           "(LRH:130-134, sabotage S09_ROPE_HALVES_SWAPPED)",
    "rope_skipped": "no rotation at all; a coarse control that must move every case, and if it "
                    "does not, the case is not running RoPE",
    "mask_select": "write the mask value instead of adding it (contract 4.1(a), sabotage "
                   "S13_MASK_SELECT). Bit-inert unless a score is -0.0 or is large enough that "
                   "the add is not the identity.",
    "mask_neg_inf": "-inf instead of -FLT_MAX (contract 4.1(b), sabotage S13_MASK_NEG_INF)",
    "scale_into_q": "scale q before the dot instead of scaling the finished dot "
                    "(contract S12, sabotage S12_SCALE_INTO_Q)",
    "denom_descending": "fold the denominator descending instead of ascending (contract 5.3)",
    "reciprocal_mul": "e * (1/denom) instead of e/denom (contract 5.4, sabotage "
                      "S18_RECIPROCAL_MUL)",
    "no_max_subtract": "exp(s) with no row-max subtraction (contract S14, S15)",
    "max_over_unmasked_only": "take the row max over the unmasked prefix only (contract 5.2)",
    "denom_over_unmasked_only": "sum the denominator over the unmasked prefix only; this is "
                                "contract 7.1's masked-tail-is-inert theorem pointed at itself, "
                                "and it SHOULD be inert, which makes it a control on the "
                                "sensitivity machinery rather than on the block",
    "kv_head_shift": "head h reads kv head ((h // n_rep) + 1) % n_kv (contract DEVIATION 813). "
                     "Bit-inert whenever n_kv == 1.",
    "repeat_kv_tile": "head h reads kv head h % n_kv instead of h // n_rep, which is repeat_kv's "
                      "own index arithmetic (LRE:179-188). Bit-inert whenever n_rep == 1.",
    "ctx_head_transposed": "flatten the attention context head-major instead of transposing to "
                           "token-major first (EAF:211). Bit-inert whenever n_heads == 1.",
    "silu_mul_sigmoid": "x*sigmoid(x), two roundings, instead of x/(1+exp(-x)) (contract S20, "
                        "sabotage S20_SILU_MUL_SIGMOID)",
    "norm_mean_not_sumsq": "record the MEAN as norm*.sumsq. The definition-mismatch demo; this "
                           "is the mamba lane's dt_proj.out class of finding, planted so the "
                           "checker's explainer has something known to name.",
    "kcache_time_transposed": "lay the KV cache out [B, n_kv, head_dim, S] and read it as "
                              "[B, n_kv, S, head_dim]. Bit-inert whenever S == head_dim, and "
                              "SHAPE-invalid otherwise, which is why it is reported separately.",
}


def _rms(x, w, eps, pert):
    """LRN:62-67. Returns (recorded_sumsq, out). The reference's `variance` is
    the MEAN; the card's stage is the SUM (contract section 9)."""
    sumsq = (x * x).sum(-1)
    variance = sumsq / x.shape[-1]
    out = w * (x * torch.rsqrt(variance + eps).unsqueeze(-1))
    return (variance if pert == "norm_mean_not_sumsq" else sumsq), out


def _rotate_half(x, pert):
    """LRH:130-134."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    if pert == "rope_halves_swapped":
        return torch.cat((x2, -x1), dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_tables(head_dim, positions):
    """The rotary tables. See the precision rule in the module docstring.

    Returns a dict. `cos64`/`sin64` are the float64 HALF tables computed from
    the FP32 angle built from the REFERENCE'S FP32 inverse frequencies
    (LRE:108, LRE:114-122). `cos32`/`sin32` are a genuine float32 evaluation of
    the same angle, which is the ref32 arm. `cos_rn32`/`sin_rn32` are the
    float64 values rounded once to float32, which is the best any FP32
    implementation can do and is the FLOOR the calibration reports beside the
    reference arm. `cos64_from_inv64` exists ONLY to measure the amplification
    the precision rule is about; it is never a reference for anything."""
    half = head_dim // 2
    i = np.arange(half, dtype=np.float64)
    inv64 = 1.0 / (float(ROPE_THETA_F32) ** ((2.0 * i) / float(head_dim)))
    # LRE:108  inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    e32 = (np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim)).astype(np.float32)
    inv32 = (np.float32(1.0) / (ROPE_THETA_F32 ** e32)).astype(np.float32)
    pos = np.asarray(positions, dtype=np.int64)
    assert pos.size == 0 or ((pos >= 0).all() and (pos < MAX_ABS_POSITION).all()), (
        "absolute position outside the profile ceiling (DEVIATION 812)", positions)
    posf32 = pos.astype(np.float32)
    assert np.array_equal(posf32.astype(np.int64), pos), "position not exact in float32"
    # LRE:114-122  an FP32 k=1 matmul with autocast explicitly disabled. PINNED FP32.
    ang32 = (posf32[:, None] * inv32[None, :]).astype(np.float32)
    a64 = ang32.astype(np.float64)
    ang_alt = pos.astype(np.float64)[:, None] * inv64[None, :]
    return dict(
        inv64=inv64, inv32=inv32, ang32=ang32,
        cos64=np.cos(a64), sin64=np.sin(a64),
        cos32=np.cos(ang32).astype(np.float32), sin32=np.sin(ang32).astype(np.float32),
        cos_rn32=np.cos(a64).astype(np.float32), sin_rn32=np.sin(a64).astype(np.float32),
        cos64_from_inv64=np.cos(ang_alt), sin64_from_inv64=np.sin(ang_alt),
    )


def block_forward(cfg, P, x, positions, dtype, pert=None):
    """One decoder-block call over a prefill of `x` at absolute `positions`.

    `x` is token-major [M, d_model] with M = B*L, which is the card's shape.
    Returns a dict stage -> torch tensor in `dtype`, every stage of the card.
    Every FP32-pinned constant enters as its float32 bits widened to `dtype`
    (the precision rule). `pert` applies one named deliberate wrong answer;
    None is the reference."""
    B, L = cfg["B_L"]
    d_model = cfg["d_model"]
    H, HKV, hd = cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"]
    inter = cfg["intermediate_size"]
    n_rep = H // HKV
    M = B * L
    x = x.reshape(M, d_model).to(dtype)
    W = {k: v.to(dtype) for k, v in P.items()}
    eps = torch.tensor(float(RMS_EPS_F32), dtype=dtype)
    scale = torch.tensor(float(cfg["scale_f32"]), dtype=dtype)
    mask_fill = torch.tensor(float("-inf") if pert == "mask_neg_inf" else float(MASK_FILL_F32),
                             dtype=dtype)
    unmasked_fill = torch.tensor(float(UNMASKED_FILL_F32), dtype=dtype)
    neg_inf = torch.tensor(float("-inf"), dtype=dtype)
    zero = torch.zeros((), dtype=dtype)
    one = torch.ones((), dtype=dtype)
    out = {}

    # LDL:305-307  residual = hidden_states; hidden_states = self.input_layernorm(hidden_states)
    out["input.x"] = x
    residual = x
    sumsq1, h = _rms(x, W["norm1.weight"], eps, pert)
    out["norm1.sumsq"] = sumsq1
    out["norm1.out"] = h

    # LAR:253-255  q/k/v projections, bias=False, then view to heads and transpose
    q = torch.nn.functional.linear(h, W["q_proj.weight"])
    k = torch.nn.functional.linear(h, W["k_proj.weight"])
    v = torch.nn.functional.linear(h, W["v_proj.weight"])
    out["q_proj.out"], out["k_proj.out"], out["v_proj.out"] = q, k, v
    qh = q.reshape(B, L, H, hd).transpose(1, 2)      # [B, H, L, hd]
    kh = k.reshape(B, L, HKV, hd).transpose(1, 2)    # [B, HKV, L, hd]
    vh = v.reshape(B, L, HKV, hd).transpose(1, 2)    # [B, HKV, L, hd]

    # LRE:93-127 the tables, LAR:137-160 the rotation
    table_pos = list(range(L)) if pert == "rope_relative_position" else list(positions)
    t = rope_tables(hd, table_pos)
    out["rope.inv_freq"] = torch.from_numpy(t["inv64"]).to(dtype)
    cos_h = torch.from_numpy(t["cos64"]).to(dtype)
    sin_h = torch.from_numpy(t["sin64"]).to(dtype)
    out["rope.cos"], out["rope.sin"] = cos_h, sin_h
    # LRE:123  emb = torch.cat((freqs, freqs), dim=-1). A COPY, not a seam.
    cos = torch.cat((cos_h, cos_h), dim=-1).unsqueeze(0).unsqueeze(0)   # [1, 1, L, hd]
    sin = torch.cat((sin_h, sin_h), dim=-1).unsqueeze(0).unsqueeze(0)
    if pert == "rope_skipped":
        q_rope, k_rope = qh, kh
    else:
        q_rope = qh * cos + _rotate_half(qh, pert) * sin
        k_rope = kh * cos + _rotate_half(kh, pert) * sin
    out["q_rope.out"] = q_rope.transpose(1, 2).reshape(M, H * hd)
    out["k_rope.out"] = k_rope.transpose(1, 2).reshape(M, HKV * hd)

    # LAR:261-262  the cache append. This corpus's cache holds exactly this
    # call's tokens; a decode-step case is a prefill over the prefix. README.
    k_cache, v_cache = k_rope, vh
    if pert == "kcache_time_transposed" and L == hd:
        k_cache = k_cache.transpose(2, 3)
        v_cache = v_cache.transpose(2, 3)
    S = k_cache.shape[2]
    out["kv.k_cache"], out["kv.v_cache"] = k_cache, v_cache

    # LRE:179-188  repeat_kv, a declared COPY (contract DEVIATION 813)
    if pert == "repeat_kv_tile":
        idx = torch.tensor([hh % HKV for hh in range(H)])
    elif pert == "kv_head_shift":
        idx = torch.tensor([((hh // n_rep) + 1) % HKV for hh in range(H)])
    else:
        idx = torch.tensor([hh // n_rep for hh in range(H)])
    krep = k_cache.index_select(1, idx)   # [B, H, S, hd]
    vrep = v_cache.index_select(1, idx)

    # EAF:204  attn_weights = matmul(query, key.transpose(2,3)) * scaling
    if pert == "scale_into_q":
        scores = torch.matmul(q_rope * scale, krep.transpose(2, 3))
    else:
        scores = torch.matmul(q_rope, krep.transpose(2, 3)) * scale
    out["attn.scores"] = scores

    # EAF:206 + masking_utils.py:601-603  the ADDITIVE causal mask, by ABSOLUTE position
    qp = torch.tensor(list(positions), dtype=torch.int64).view(L, 1)
    kp = torch.tensor(list(positions), dtype=torch.int64).view(1, S)
    allowed = (kp <= qp)                                   # [L, S]
    if pert == "mask_select":
        masked = torch.where(allowed, scores, mask_fill)
    else:
        masked = scores + torch.where(allowed, unmasked_fill, mask_fill)
    out["attn.masked"] = masked

    # S14-S18, the softmax spelled out because every step is a card stage
    if pert == "max_over_unmasked_only":
        m = torch.where(allowed, masked, neg_inf).max(-1).values
    else:
        m = masked.max(-1).values                          # [B, H, L]
    out["attn.max"] = m
    e = torch.exp(masked) if pert == "no_max_subtract" else torch.exp(masked - m.unsqueeze(-1))
    out["attn.exp"] = e
    if pert == "denom_descending":
        denom = torch.flip(e, dims=[-1]).cumsum(-1)[..., -1]
    elif pert == "denom_over_unmasked_only":
        denom = torch.where(allowed, e, zero).sum(-1)
    else:
        denom = e.sum(-1)
    out["attn.denom"] = denom
    if pert == "reciprocal_mul":
        w = e * (one / denom.unsqueeze(-1))
    else:
        w = e / denom.unsqueeze(-1)
    out["attn.weights"] = w

    # EAF:210-211  attn_output = matmul(attn_weights, value); transpose(1,2); reshape
    ctx = torch.matmul(w, vrep)                            # [B, H, L, hd]
    if pert == "ctx_head_transposed":
        out["attn.ctx"] = ctx.reshape(B, H, L * hd).transpose(1, 2).reshape(M, H * hd)
    else:
        out["attn.ctx"] = ctx.transpose(1, 2).reshape(M, H * hd)

    # LAR:280  o_proj, then LDL:317  the residual add
    o = torch.nn.functional.linear(out["attn.ctx"], W["o_proj.weight"])
    out["o_proj.out"] = o
    r1 = residual + o
    out["residual1.out"] = r1

    # LDL:320-323  the second norm and the MLP
    sumsq2, h2 = _rms(r1, W["norm2.weight"], eps, pert)
    out["norm2.sumsq"], out["norm2.out"] = sumsq2, h2
    g = torch.nn.functional.linear(h2, W["gate_proj.weight"])
    u = torch.nn.functional.linear(h2, W["up_proj.weight"])
    out["gate_proj.out"], out["up_proj.out"] = g, u
    # LMLP:175 ACT2FN["silu"]; contract S20 pins ATen's ONE-division spelling
    s = (g * torch.sigmoid(g)) if pert == "silu_mul_sigmoid" else (g / (one + torch.exp(-g)))
    out["silu.out"] = s
    gated = s * u
    out["mlp.gated"] = gated
    dn = torch.nn.functional.linear(gated, W["down_proj.weight"])
    out["down_proj.out"] = dn
    out["residual2.out"] = r1 + dn

    for st in STAGE_ORDER:
        assert st in out, ("stage not produced", st)
    assert inter == g.shape[-1] and d_model == dn.shape[-1]
    return out


def slice_query_row(stages, B, L, t):
    """The decode VIEW of a prefill: query row t only, KV cache kept whole.

    Contract 7.2 claims a decode step at position t computes the same bits as
    prefill row t. A DECODE-STEP CASE in this corpus is a prefill over tokens
    0..t with only query row t recorded, which is what a driver holding a warm
    cache actually computes. The rotary tables and the cache stay whole
    because the driver needs all of them."""
    o = {}
    for s, v in stages.items():
        if s in CONFIG_STAGES or s in ("kv.k_cache", "kv.v_cache"):
            o[s] = v
        elif s in ("attn.scores", "attn.masked", "attn.exp", "attn.weights"):
            o[s] = v[:, :, t:t + 1, :]
        elif s in ("attn.max", "attn.denom"):
            o[s] = v[:, :, t:t + 1]
        elif s in ("norm1.sumsq", "norm2.sumsq"):
            o[s] = v.reshape(B, L)[:, t:t + 1].reshape(B)
        else:
            # reshape and not view: some stages arrive from a transposed
            # expression and a view would raise where a reshape copies.
            o[s] = v.reshape(B, L, -1)[:, t:t + 1, :].reshape(B, -1)
    return o


# --------------------------------------------------------------------------
# Comparison. THIS FUNCTION IS DUPLICATED, ON PURPOSE, in
# tools/transformer_corpus_check.py, which must run on numpy alone in the
# repository's pixi envs (torch lives in one env only). The two copies MUST
# agree; COMPARE_VERSION is written into every manifest and the checker
# refuses a manifest whose version it does not match, so a silent divergence
# between the copies is not possible.
# --------------------------------------------------------------------------
COMPARE_VERSION = "transformer-corpus-compare-v1"


def compare(dump, ref, rtol, atol):
    """Element comparison of an FP32 dump against a float64 reference.

    An element PASSES when |dump - ref| <= atol + rtol*|ref|, OR when it is an
    EXPLAINED FP32 OVERFLOW: the dump is an infinity of the same sign as the
    reference and |ref| exceeds the float32 maximum. That second class is not
    a courtesy. Contract 4.1(b) says a masked cell at an extreme score
    overflows to -inf in FP32 BY CONSTRUCTION, and a checker calling that a
    failure would be measuring the dtype rather than the port. Overflows are
    COUNTED and REPORTED and never silently absorbed.

    SIGNED ZEROS ARE COMPARED BY SIGN BIT, separately, because a tolerance
    pass is worthless for them: isclose treats +0.0 and -0.0 as equal. Where
    the reference is exactly +-0.0 the dump's sign bit must match. Contract
    section 8 is the policy. A sign-bit mismatch fails `ok` while leaving
    `tol_ok` true, so the two failures are never confused with each other.

    WHAT THIS CANNOT DO. It cannot see a fold ORDER. Reordering a float64 sum
    moves it by about 1e-16 relative, four orders below any tolerance an FP32
    dump can honestly be held to. Every clause of contract sections 5.3 and
    7.2 that is about ORDER is invisible here and belongs to the lane's own
    sabotages. The generator's sensitivity table prints exactly which ones."""
    d = np.asarray(dump, dtype=np.float64).ravel()
    r = np.asarray(ref, dtype=np.float64).ravel()
    close = np.isclose(d, r, rtol=rtol, atol=atol, equal_nan=True)
    overflow = np.isinf(d) & (np.abs(r) > FP32_MAX) & (np.signbit(d) == np.signbit(r))
    ok_elem = close | overflow
    finite = np.isfinite(r) & np.isfinite(d)
    diff = np.where(finite, np.abs(d - r), 0.0)
    max_abs = float(diff.max()) if diff.size else 0.0
    nz = finite & (r != 0)
    max_rel = float((diff[nz] / np.abs(r[nz])).max()) if nz.any() else 0.0
    ref_zero = (r == 0)
    zero_sign_bad = ref_zero & (np.signbit(d) != np.signbit(r))
    tol_ok = bool(ok_elem.all())
    return dict(
        ok=tol_ok and not bool(zero_sign_bad.any()),
        tol_ok=tol_ok,
        max_abs=max_abs, max_rel=max_rel, n=int(d.size),
        n_overflow=int(overflow.sum()), n_ref_zero=int(ref_zero.sum()),
        n_zero_sign_bad=int(zero_sign_bad.sum()),
        n_ref_nan=int(np.isnan(r).sum()),
        first_bad=(None if tol_ok else int(np.argmin(ok_elem))),
        first_zero_sign_bad=(None if not zero_sign_bad.any() else int(np.argmax(zero_sign_bad))),
    )


RTOL_LADDER = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
DEFAULT_ATOL = 1e-6


def smallest_passing_rtol(dump, ref, atol):
    for rt in RTOL_LADDER:
        if compare(dump, ref, rt, atol)["tol_ok"]:
            return rt
    return None


def one_rung_looser(rt):
    """The gate is ONE ladder rung looser than the measured reference arm.

    That is the mamba lane's conclusion restated: holding a dump to exactly
    the reference arm's own number makes the tolerance the defect the first
    time a stage's magnitudes move. Looser by one rung and no more."""
    if rt is None:
        return None
    i = RTOL_LADDER.index(rt)
    return RTOL_LADDER[min(i + 1, len(RTOL_LADDER) - 1)]


# --------------------------------------------------------------------------
# The cases. seed_k = SEED_BASE + 0x1000*k with k the case index.
#
# Standing rule: never build to datasets. No case here was picked, dropped,
# deferred or tuned by whether it flatters us. `expect_fail` names the cases
# this author PREDICTS the lane will fail, and an adversarial case we fail is
# worth more than one we pass.
# --------------------------------------------------------------------------
SEED_BASE = 0x58666D72436F7270   # "XfmrCorp"

BASE_RANGES = {
    "x": (-2.0, 2.0),
    "norm1.weight": (0.5, 1.5),
    "norm2.weight": (0.5, 1.5),
    "q_proj.weight": (-0.5, 0.5),
    "k_proj.weight": (-0.5, 0.5),
    "v_proj.weight": (-0.5, 0.5),
    "gate_proj.weight": (-0.25, 0.25),
    "up_proj.weight": (-0.25, 0.25),
}


def fan_in_scale(fan_in):
    """s = 0.5 / 2^ceil(log2(fan_in)/2), a dyadic stand-in for 0.5/sqrt(fan_in).

    Dyadic because the hash spec requires (lo, hi) to be exactly representable
    so that the float64 evaluation rounds exactly once."""
    return 0.5 / (2 ** math.ceil(math.log2(fan_in) / 2))


# 2^52 and 2^-68: dyadic, so the hash spec's exactness holds. Sizing, stated
# so a reader can check it rather than trust it. With x in [-2, 2] and
# d_model = 32, a projection weight uniform on [-w, w] gives |q| of order
# sqrt(32)*1.155*w/sqrt(3), and a head_dim=16 dot with scale 0.25 gives a
# score of order sqrt(16)*0.25*|q|*|k|.
#   BIG   = 2^52: |q| ~ 1.7e16, |score| ~ 3e32. That is about 29x above 2^103
#           (1.01e31), which is the threshold above which s + (-FLT_MAX) stops
#           being the identity, and about 1000x below the float32 maximum, so
#           the dot itself cannot overflow. Both margins are wanted.
#   SMALL = 2^-68: |q| ~ 1.3e-20, |score| ~ 1.6e-40, inside the float32
#           subnormal band [1.4e-45, 1.18e-38].
# `--verify` control 7 MEASURES both and says INTENT NOT REACHED if either
# estimate is wrong, because this arithmetic has not been run.
BIG = float(2 ** 52)
SMALL = float(2.0 ** -68)

CASES = [
    dict(name="base_b1_l1_d32_kv2", B=1, L=1, d_model=32, n_heads=2, head_dim=16,
         n_kv_heads=2, intermediate_size=64, pos0=0,
         certifies="the smallest shape. L=1 and S=1, which is also the shape of a decode step "
                   "with an empty cache. It reaches every one of the thirty stages once.",
         cannot_certify="anything about ordering along L or S. At L=1 the causal mask is a "
                        "single unmasked cell, the denominator is one term, and a "
                        "token-major to head-major reindexing of a [1, F] stage is the same "
                        "bytes. This is the mamba lane's base_b1_l1_d8 lesson stated in "
                        "advance, and this case's measured sensitivity table is the proof "
                        "rather than the promise.",
         expect_fail=False),
    dict(name="base_b2_l4_d32_kv2", B=2, L=4, d_model=32, n_heads=2, head_dim=16,
         n_kv_heads=2, intermediate_size=64, pos0=0,
         certifies="a real causal triangle at L=4, B=2, with n_rep == 1 so repeat_kv is the "
                   "identity and the head-to-kv-head map is live. Parent of the two "
                   "batch-composition rows.",
         cannot_certify="repeat_kv's own index arithmetic, which is the identity at n_rep == 1 "
                        "(contract DEVIATION 813 makes the same point from the other side). "
                        "Absolute position indexing, because pos0 == 0 makes relative and "
                        "absolute the same numbers.",
         expect_fail=False),
    dict(name="comp_row0_b1_l4_d32_kv2", B=1, L=4, d_model=32, n_heads=2, head_dim=16,
         n_kv_heads=2, intermediate_size=64, pos0=0, x_source=("base_b2_l4_d32_kv2", 0),
         certifies="batch composition: row 0 of the parent, run alone, SAME seed and SAME "
                   "weights, with x a byte slice of the parent's x. `--verify` control 2 "
                   "reports whether torch's own float64 reproduces the parent's row bit for "
                   "bit.",
         cannot_certify="the lane's batch invariance, which is clause (c) and is about OUR "
                        "execution plan. This shows only that the corpus is consistent with "
                        "itself, which is a precondition and not a result.",
         expect_fail=False),
    dict(name="comp_row1_b1_l4_d32_kv2", B=1, L=4, d_model=32, n_heads=2, head_dim=16,
         n_kv_heads=2, intermediate_size=64, pos0=0, x_source=("base_b2_l4_d32_kv2", 1),
         certifies="batch composition, row 1, same construction",
         cannot_certify="as comp_row0",
         expect_fail=False),
    dict(name="base_b1_l4_d32_kv1", B=1, L=4, d_model=32, n_heads=2, head_dim=16,
         n_kv_heads=1, intermediate_size=64, pos0=0,
         certifies="GQA at n_rep == 2, so repeat_kv actually copies and its index arithmetic "
                   "is live. Contract DEVIATION 813 requires both n_rep values to be gated.",
         cannot_certify="the head-to-kv-head ASSIGNMENT, which is vacuous at n_kv == 1 because "
                        "every head reads the one kv head. Only the n_kv == 2 cases see that. "
                        "This is a pair that must be read together and neither half is "
                        "sufficient alone.",
         expect_fail=False),
    dict(name="base_b2_l16_d32_kv1_i300", B=2, L=16, d_model=32, n_heads=2, head_dim=16,
         n_kv_heads=1, intermediate_size=300, pos0=0,
         certifies="intermediate_size 300, so down_proj contracts over k=300 and the GEMM "
                   "profile's balanced fold tree runs with P=3 and a ragged 44-element last "
                   "leaf (contract section 3). Without it that tree sits unexercised inside "
                   "the whole block. Also the longest L in the corpus.",
         cannot_certify="the GEMM's fold TOPOLOGY, which this corpus cannot see at all: a "
                        "different summation order moves a float64 result by about 1e-16 "
                        "relative, four orders below any honest FP32 tolerance. What this "
                        "case does is make the tree RUN, so a shape-dependent crash or a "
                        "wrong ragged-leaf length is reachable. Correctness of the fold is "
                        "the GEMM profile's, not this corpus's.",
         expect_fail=False),
    dict(name="base_b1_l4_d48_hd24_kv1", B=1, L=4, d_model=48, n_heads=2, head_dim=24,
         n_kv_heads=1, intermediate_size=64, pos0=0,
         certifies="head_dim 24, not a power of four, so the attention scale 1/sqrt(24) is "
                   "INEXACT. Contract section 3 says a power-of-four head_dim cannot see a "
                   "wrong scale spelling because the scale is exactly 0.25 or 0.125 there.",
         cannot_certify="which of the two spellings (`head_dim ** -0.5` rounded once, versus "
                        "f32div(1, f32sqrt(head_dim))) is right, if they agree at 24. "
                        "`--verify` control 6 prints both and says whether they differ. If "
                        "they agree, this case certifies the VALUE and not the SPELLING.",
         expect_fail=False),
    dict(name="adv_mask_extreme_b1_l4_d32_kv2", B=1, L=4, d_model=32, n_heads=2, head_dim=16,
         n_kv_heads=2, intermediate_size=64, pos0=0,
         overrides={"q_proj.weight": (-BIG, BIG), "k_proj.weight": (-BIG, BIG)},
         certifies="scores far above 2^103 in magnitude, so a masked cell's add is NOT the "
                   "identity: s + (-FLT_MAX) is a value distinct from -FLT_MAX for positive s "
                   "and overflows to -inf for large negative s. That separates the ADD from a "
                   "SELECT and separates -FLT_MAX from -inf, which is contract 4.1(b) and "
                   "sabotages S13_MASK_SELECT and S13_MASK_NEG_INF. It is the only case where "
                   "the FP32 arm legitimately holds an infinity, which is why compare() "
                   "carries an explained-overflow class at all.",
         cannot_certify="ordinary-magnitude behavior of anything downstream. The attention "
                        "here flattens to a near-uniform context, so the MLP half adds "
                        "nothing the base cases do not already have.",
         expect_fail=True),
    dict(name="adv_subnormal_scores_b1_l4_d32_kv2", B=1, L=4, d_model=32, n_heads=2,
         head_dim=16, n_kv_heads=2, intermediate_size=64, pos0=0,
         overrides={"q_proj.weight": (-SMALL, SMALL), "k_proj.weight": (-SMALL, SMALL)},
         certifies="scores that land in and below the float32 subnormal band, so the profile's "
                   "ftz at S12 and S14 is live and the row max is a flushed zero. It is the "
                   "closest a torch reference can get to contract 4.1(a).",
         cannot_certify="THE -0.0 MASK LAUNDERING ITSELF, and this is the honest limit of the "
                        "whole corpus. Contract 4.1(a) needs a score that is exactly -0.0, "
                        "reachable only through OUR ftz of a negative subnormal accumulator. "
                        "Torch has no ftz, so the reference holds a small negative number "
                        "where we hold -0.0, and no tolerance and no sign-bit check turns "
                        "that into a certificate. S13_MASK_SELECT is the lane's own oracle's "
                        "job and this corpus cannot help with it.",
         expect_fail=True),
    dict(name="adv_signed_zeros_b2_l4_d32_kv2", B=2, L=4, d_model=32, n_heads=2, head_dim=16,
         n_kv_heads=2, intermediate_size=64, pos0=0,
         zero_rule="x[b,t,:] = +0.0 for t%4==0; x[b,t,:] = -0.0 for t%4==2; "
                   "of the remaining elements, row-major flat index i%7==3 is -0.0",
         certifies="signed zeros in the INPUT, compared BY SIGN BIT and not by tolerance. "
                   "norm1.out carries the sign through, because w*(-0.0*rstd) is -0.0 for a "
                   "positive weight, so an implementation that laundered it is caught at that "
                   "stage and at input.x.",
         cannot_certify="anything downstream of the first GEMM. Torch's matmul seeds its "
                        "accumulator at +0.0, so a whole -0.0 token projects to +0.0 and the "
                        "sign is gone by q_proj.out IN THE REFERENCE TOO. That is the mamba "
                        "lane's finding repeated: the corpus certifies zero signs at input.x "
                        "and norm1.out and nowhere after them.",
         expect_fail=False),
    dict(name="adv_rope_far_pos_b1_l4_d32_kv2", B=1, L=4, d_model=32, n_heads=2, head_dim=16,
         n_kv_heads=2, intermediate_size=64, pos0=8188,
         certifies="absolute positions 8188 to 8191, one below the profile ceiling of 8192 "
                   "(DEVIATION 812, portable_cosf's Cody-Waite domain). This is the case that "
                   "makes the precision rule load-bearing, because the angle at 8191 "
                   "amplifies a one-ulp inv_freq error by about 8000x, so it is the only case "
                   "that can tell a correct rotary table from a plausible one. It is also the "
                   "only case in the corpus where `rope_relative_position` is not bit-inert, "
                   "so it is the only case that certifies absolute indexing at all.",
         cannot_certify="anything about rope.cos or rope.sin IF the lane's inv_freq bits "
                        "differ from the reference's. The checker verifies inv_freq against "
                        "the committed FP32 anchor FIRST and marks the cos and sin comparisons "
                        "CONDITIONAL when it does not match, because at this position a "
                        "one-ulp anchor difference dwarfs any tolerance worth setting.",
         expect_fail=True),
    dict(name="adv_softmax_saturation_b1_l8_d32_kv1", B=1, L=8, d_model=32, n_heads=2,
         head_dim=16, n_kv_heads=1, intermediate_size=64, pos0=0,
         token_scale_rule="x[b,t,:] *= 2^(t-3)",
         certifies="BOTH arms of the denominator chain in one case, by construction and not by "
                   "luck. Scaling token t of x by the exact power of two 2^(t-3) makes "
                   "score(t,j) proportional to 2^(t+j-6), so the late query rows span a score "
                   "range far wider than exp's 87.33655 underflow decade and most of their "
                   "keys go to exactly +0.0, while the early rows span almost nothing and "
                   "every key contributes. A power of two is exact in both dtypes, so the "
                   "hash spec stays reproducible bit for bit.",
         cannot_certify="the ORDER of the denominator fold. Adding the same nonnegative terms "
                        "in a different order moves a float64 sum by about 1e-16 relative, "
                        "invisible at any tolerance this corpus can honestly set. "
                        "S17_DENOM_HALVING_TREE is unreachable from here and belongs to the "
                        "lane's own oracle. What this case does certify is WHICH TERMS are in "
                        "the sum and that the max subtraction happened at all.",
         expect_fail=True),
    dict(name="decode_b1_l4_d32_kv1", B=1, L=4, d_model=32, n_heads=2, head_dim=16,
         n_kv_heads=1, intermediate_size=64, pos0=0, decode_steps=True,
         certifies="the structural claim of the whole profile (contract 7.2, clause (d)). It "
                   "emits four decode-step subcases; step t is an INDEPENDENT float64 run "
                   "over tokens 0..t with only query row t recorded, which is what a driver "
                   "with a warm cache computes. `--verify` control 3 then reports whether "
                   "each step equals the full prefill's row t bit for bit in torch's own "
                   "float64, which is contract 7.1's masked-tail-is-inert theorem measured "
                   "rather than argued, and control 4 is its negative control.",
         cannot_certify="that OUR decode equals OUR prefill, which is clause (d) and the "
                        "lane's. And it certifies nothing about absolute position indexing, "
                        "because pos0 == 0 here; adv_rope_far_pos is the case that does. A "
                        "gate built on this case alone would pass a driver whose position map "
                        "is broken by a constant offset, which is exactly the hole the mamba "
                        "lane's clause (c) needed a negative control to close.",
         expect_fail=False),
]

REDUCED_CASES = {"base_b2_l16_d32_kv1_i300"}


def case_index(name):
    for k, c in enumerate(CASES):
        if c["name"] == name:
            return k
    raise KeyError(name)


def case_seed(k):
    return (SEED_BASE + 0x1000 * k) & M64


def shapes_for(cfg, B, L):
    d, H, HKV, hd, inter = (cfg["d_model"], cfg["n_heads"], cfg["n_kv_heads"],
                            cfg["head_dim"], cfg["intermediate_size"])
    return {
        "x": [B, L, d],
        "norm1.weight": [d], "norm2.weight": [d],
        "q_proj.weight": [H * hd, d],
        "k_proj.weight": [HKV * hd, d],
        "v_proj.weight": [HKV * hd, d],
        "o_proj.weight": [d, H * hd],
        "gate_proj.weight": [inter, d],
        "up_proj.weight": [inter, d],
        "down_proj.weight": [d, inter],
    }


def ranges_for(cfg):
    r = dict(BASE_RANGES)
    s_o = fan_in_scale(cfg["n_heads"] * cfg["head_dim"])
    s_d = fan_in_scale(cfg["intermediate_size"])
    r["o_proj.weight"] = (-s_o, s_o)
    r["down_proj.weight"] = (-s_d, s_d)
    return r


def apply_zero_rule(x, rule):
    """x[b,t,:] = +0.0 for t%4==0; x[b,t,:] = -0.0 for t%4==2; remaining flat i%7==3 -> -0.0."""
    assert rule.startswith("x[b,t,:] = +0.0 for t%4==0"), rule
    B, L, d = x.shape
    flat = x.reshape(-1).copy()
    keep = np.ones(flat.shape[0], dtype=bool)
    for b in range(B):
        for t in range(L):
            base = (b * L + t) * d
            if t % 4 == 0:
                flat[base:base + d] = np.float32(0.0)
                keep[base:base + d] = False
            elif t % 4 == 2:
                flat[base:base + d] = np.float32(-0.0)
                keep[base:base + d] = False
    idx = np.arange(flat.shape[0])
    flat[keep & (idx % 7 == 3)] = np.float32(-0.0)
    return flat.reshape(x.shape)


def apply_token_scale(x, rule):
    """x[b,t,:] *= 2^(t-3). An exact power of two, so no rounding is introduced
    and a Mojo fixture reproduces the bytes from the README rule alone."""
    assert rule == "x[b,t,:] *= 2^(t-3)", rule
    out = x.copy()
    for t in range(x.shape[1]):
        exact = x[:, t, :].astype(np.float64) * (2.0 ** (t - 3))
        out[:, t, :] = exact.astype(np.float32)
        assert np.array_equal(out[:, t, :].astype(np.float64), exact), (
            "token scale is not exact in float32", t)
    return out


def build_case(case):
    """Materialize one case's inputs from the hash spec, plus its metadata."""
    k = case_index(case["name"])
    cfg = {kk: case[kk] for kk in ("d_model", "n_heads", "n_kv_heads",
                                   "head_dim", "intermediate_size")}
    assert cfg["d_model"] == cfg["n_heads"] * cfg["head_dim"], case["name"]
    assert cfg["n_heads"] % cfg["n_kv_heads"] == 0, case["name"]
    assert cfg["head_dim"] % 2 == 0 and cfg["intermediate_size"] > 0, case["name"]
    # contract section 3: the scale is pinned FP32 and computed ONCE on the host
    scale_ref = np.float32(float(cfg["head_dim"]) ** -0.5)                    # LAR:226
    scale_alt = np.float32(np.float32(1.0) / np.sqrt(np.float32(cfg["head_dim"])))
    cfg["scale_f32"] = scale_ref
    B, L = case["B"], case["L"]
    cfg["B_L"] = (B, L)
    src = case.get("x_source")
    seed = case_seed(case_index(src[0])) if src else case_seed(k)
    shapes = shapes_for(cfg, B, L)
    ranges = ranges_for(cfg)
    ranges.update(case.get("overrides", {}))
    tensors = {}
    for name, shape in shapes.items():
        lo, hi = ranges[name]
        if name == "x" and src:
            parent = CASES[case_index(src[0])]
            pshape = shapes_for(cfg, parent["B"], parent["L"])["x"]
            px = map_range(hashed_unit(seed, TENSOR_IDS["x"], int(np.prod(pshape))),
                           lo, hi).reshape(pshape)
            tensors["x"] = px[src[1]:src[1] + 1].copy()
            continue
        tensors[name] = map_range(hashed_unit(seed, TENSOR_IDS[name], int(np.prod(shape))),
                                  lo, hi).reshape(shape)
    if "zero_rule" in case:
        tensors["x"] = apply_zero_rule(tensors["x"], case["zero_rule"])
    if "token_scale_rule" in case:
        tensors["x"] = apply_token_scale(tensors["x"], case["token_scale_rule"])
    positions = [case["pos0"] + i for i in range(L)]
    meta = dict(
        name=case["name"], corpus=CORPUS_VERSION, hash_spec=HASH_SPEC, profile=PROFILE,
        compare_version=COMPARE_VERSION,
        seed=f"0x{seed:016X}", seed_case_index=(case_index(src[0]) if src else k),
        B=B, L=L, pos0=case["pos0"], positions=positions,
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_kv_heads=cfg["n_kv_heads"],
        head_dim=cfg["head_dim"], intermediate_size=cfg["intermediate_size"],
        n_rep=cfg["n_heads"] // cfg["n_kv_heads"],
        rms_eps=float(RMS_EPS_F32), rope_theta=float(ROPE_THETA_F32),
        mask_fill=float(MASK_FILL_F32), unmasked_fill=float(UNMASKED_FILL_F32),
        attention_scale=float(scale_ref),
        attention_scale_alt_spelling=float(scale_alt),
        attention_scale_spellings_agree=bool(scale_ref == scale_alt),
        certifies=case["certifies"], cannot_certify=case["cannot_certify"],
        expect_lane_failure=case["expect_fail"],
        tensors={name: dict(shape=shapes[name], dtype="float32", order="row-major",
                            tensor_id=TENSOR_IDS[name], lo=ranges[name][0], hi=ranges[name][1],
                            file=f"{name}.f32")
                 for name in shapes},
    )
    if src:
        meta["x_source"] = dict(case=src[0], batch_row=src[1])
        meta["tensors"]["x"]["derived"] = f"byte slice of {src[0]}/x.f32, batch row {src[1]}"
    if "zero_rule" in case:
        meta["zero_rule"] = case["zero_rule"]
        meta["tensors"]["x"]["derived"] = "hashed, then zero_rule applied"
    if "token_scale_rule" in case:
        meta["token_scale_rule"] = case["token_scale_rule"]
        meta["tensors"]["x"]["derived"] = "hashed, then token_scale_rule applied (exact)"
    return cfg, tensors, meta, positions


def expected_shape(stage, m, L, S):
    """Contract section 9's shapes, restated so a wrong reshape in THIS file is
    caught before it is written to disk and believed."""
    B, d, H, HKV, hd = m["B"], m["d_model"], m["n_heads"], m["n_kv_heads"], m["head_dim"]
    inter, half, npos = m["intermediate_size"], m["head_dim"] // 2, len(m["positions"])
    M = B * L
    return {
        "input.x": [M, d], "norm1.sumsq": [M], "norm2.sumsq": [M],
        "norm1.out": [M, d], "norm2.out": [M, d],
        "q_proj.out": [M, H * hd], "k_proj.out": [M, HKV * hd], "v_proj.out": [M, HKV * hd],
        "rope.inv_freq": [half], "rope.cos": [npos, half], "rope.sin": [npos, half],
        "q_rope.out": [M, H * hd], "k_rope.out": [M, HKV * hd],
        "kv.k_cache": [B, HKV, S, hd], "kv.v_cache": [B, HKV, S, hd],
        "attn.scores": [B, H, L, S], "attn.masked": [B, H, L, S],
        "attn.exp": [B, H, L, S], "attn.weights": [B, H, L, S],
        "attn.max": [B, H, L], "attn.denom": [B, H, L],
        "attn.ctx": [M, H * hd],
        "o_proj.out": [M, d], "residual1.out": [M, d],
        "gate_proj.out": [M, inter], "up_proj.out": [M, inter],
        "silu.out": [M, inter], "mlp.gated": [M, inter],
        "down_proj.out": [M, d], "residual2.out": [M, d],
    }[stage]


def write_raw(path, arr, dtype):
    a = np.ascontiguousarray(np.asarray(arr, dtype=dtype))
    with open(path, "wb") as fh:
        fh.write(a.astype("<" + a.dtype.str[1:]).tobytes())


def emit_stages(cdir, meta, r64, r32, stages, L, S, rope):
    """Write ref64 and ref32 for every stage and CALIBRATE per stage.

    Two calibration columns, both measured, both in the manifest:
      rtol_torch_fp32       the smallest ladder rung at which a plain float32
                            CPU run of the SAME algorithm passes. The gate is
                            one rung looser than this.
      rtol_correctly_rounded the smallest rung at which the float64 reference
                            ROUNDED ONCE to float32 passes. That is the best
                            any FP32 implementation can do. When the two
                            differ by more than a rung the reference arm is
                            the limit, not us, and the checker says so."""
    os.makedirs(os.path.join(cdir, "ref64"), exist_ok=True)
    os.makedirs(os.path.join(cdir, "ref32"), exist_ok=True)
    override32 = {"rope.inv_freq": rope["inv32"],
                  "rope.cos": rope["cos32"], "rope.sin": rope["sin32"]}
    stage_meta = {}
    for s in stages:
        a64 = np.ascontiguousarray(r64[s].detach().numpy())
        want = expected_shape(s, meta, L, S)
        assert list(a64.shape) == want, (meta["name"], s, list(a64.shape), want)
        a32 = override32.get(s)
        if a32 is None:
            a32 = np.ascontiguousarray(r32[s].detach().numpy())
        a32 = np.asarray(a32, dtype=np.float32).reshape(want)
        rn32 = a64.astype(np.float32)
        write_raw(os.path.join(cdir, "ref64", f"{s}.f64"), a64, np.float64)
        write_raw(os.path.join(cdir, "ref32", f"{s}.f32"), a32, np.float32)
        c = compare(a32, a64, RTOL_LADDER[-1], DEFAULT_ATOL)
        rt = smallest_passing_rtol(a32, a64, DEFAULT_ATOL)
        stage_meta[s] = dict(
            shape=want, order="row-major",
            ref64=f"ref64/{s}.f64", ref32=f"ref32/{s}.f32",
            rtol_torch_fp32=rt,
            rtol_correctly_rounded=smallest_passing_rtol(rn32, a64, DEFAULT_ATOL),
            rtol_gate=one_rung_looser(rt) or RTOL_LADDER[-1],
            atol=DEFAULT_ATOL,
            torch_max_abs=c["max_abs"], torch_max_rel=c["max_rel"],
            n_elements=c["n"], n_ref_zero=c["n_ref_zero"],
            n_ref_negative_zero=int(np.signbit(a64.ravel()[a64.ravel() == 0]).sum()),
            n_fp32_overflow=c["n_overflow"],
            n_fp32_zero_where_ref_nonzero=int(((a32.ravel() == 0) & (a64.ravel() != 0)).sum()),
            ref64_max_abs=float(np.abs(a64).max()) if a64.size else 0.0,
        )
    return stage_meta


def sensitivity(cfg, P, x, positions, base, stages, slicer=None):
    """Run every named perturbation and classify every stage.

    SEEN    the perturbation moves the stage by more than the tolerance the
            reference arm itself needs, so this case CAN catch it.
    SUBTOL  the perturbation moves the stage, but by less than that, so this
            case CANNOT catch it even though it is not bit-inert. This is the
            class that matters: it is where a clause looks covered and is not.
    SHAPE   the perturbation changes a stage's shape.
    (bit-inert stages are omitted.)

    This is the entire answer to "what would this case fail to catch, and how
    do I show it catches anything at all". It is computed rather than claimed,
    and a case with no SEEN entry for a perturbation must never be quoted as
    evidence about the clause that perturbation attacks."""
    res = {}
    for pert in PERTURBATIONS:
        try:
            alt = block_forward(cfg, P, x, positions, torch.float64, pert=pert)
            if slicer is not None:
                alt = slicer(alt)
        except Exception as exc:            # a perturbation that cannot run is REPORTED
            res[pert] = dict(error=f"{type(exc).__name__}: {exc}")
            continue
        seen, subtol, shape = [], [], []
        for s in stages:
            a = np.ascontiguousarray(alt[s].detach().numpy())
            b = np.ascontiguousarray(base[s].detach().numpy())
            if a.shape != b.shape:
                shape.append(s)
                continue
            if np.array_equal(a, b, equal_nan=True) and np.array_equal(np.signbit(a),
                                                                       np.signbit(b)):
                continue
            (subtol if compare(a, b, RTOL_LADDER[0], DEFAULT_ATOL)["tol_ok"] else seen).append(s)
        entry = dict(seen=seen)
        if subtol:
            entry["subtol"] = subtol
        if shape:
            entry["shape_changed"] = shape
        res[pert] = entry
    return res


def emit_case(case, out_root, want_sensitivity=True):
    cfg, tensors, meta, positions = build_case(case)
    B, L = case["B"], case["L"]
    cdir = os.path.join(out_root, case["name"])
    os.makedirs(cdir, exist_ok=True)
    for name, arr in tensors.items():
        write_raw(os.path.join(cdir, f"{name}.f32"), arr, np.float32)
    write_raw(os.path.join(cdir, "positions.i32"), np.asarray(positions, np.int32), np.int32)
    P = {k: torch.from_numpy(v.copy()) for k, v in tensors.items() if k != "x"}
    x = torch.from_numpy(tensors["x"].copy())

    r64 = block_forward(cfg, P, x, positions, torch.float64)
    r32 = block_forward(cfg, P, x, positions, torch.float32)
    stages = [s for s in STAGE_ORDER
              if not (case["name"] in REDUCED_CASES and s in REDUCED_SKIP)]
    rope = rope_tables(cfg["head_dim"], positions)
    meta["stages"] = emit_stages(cdir, meta, r64, r32, stages, L, L, rope)
    meta["stage_order"] = stages
    meta["reduced_stage_set"] = case["name"] in REDUCED_CASES
    meta["rope_anchor"] = rope_anchor_meta(rope)
    if want_sensitivity:
        meta["sensitivity"] = sensitivity(cfg, P, x, positions, r64, stages)
        meta["sensitivity_legend"] = (
            "seen = this case catches the perturbation at the tolerance the reference arm "
            "itself needs; subtol = it moves the stage but by less than that, so this case "
            "CANNOT catch it; shape_changed = it changes a stage's shape; omitted = bit-inert "
            "in float64.")
    write_manifest(cdir, meta)

    sub = []
    if case.get("decode_steps"):
        base_meta = {k: v for k, v in meta.items()
                     if k not in ("stages", "sensitivity", "stage_order", "rope_anchor",
                                  "sensitivity_legend", "reduced_stage_set")}
        for t in range(L):
            sub.append(emit_decode_step(case, cfg, tensors, P, x, positions, t,
                                        out_root, base_meta, want_sensitivity))
    return dict(cfg=cfg, meta=meta, r64=r64, r32=r32, P=P, x=x, rope=rope,
                positions=positions, tensors=tensors, subcases=sub)


def rope_anchor_meta(rope):
    return dict(
        note="rope.cos and rope.sin are cos and sin, evaluated in float64, of the FP32 angle "
             "f32(f32(position) * inv_freq_f32), where inv_freq_f32 is ref32/rope.inv_freq.f32, "
             "the reference's own FP32 spelling (LRE:108). A checker MUST verify a dump's "
             "rope.inv_freq against that anchor BEFORE believing any rope.cos comparison, "
             "because the angle amplifies an inv_freq error by the absolute position.",
        anchor_file="ref32/rope.inv_freq.f32",
        max_abs_angle=float(np.abs(rope["ang32"]).max()) if rope["ang32"].size else 0.0,
        cos_gap_f64_vs_f32_inv_freq=float(
            np.abs(rope["cos64_from_inv64"] - rope["cos64"]).max()) if rope["ang32"].size else 0.0,
        sin_gap_f64_vs_f32_inv_freq=float(
            np.abs(rope["sin64_from_inv64"] - rope["sin64"]).max()) if rope["ang32"].size else 0.0,
        float32_eps=FP32_EPS,
    )


def write_manifest(cdir, meta):
    with open(os.path.join(cdir, "manifest.json"), "w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
        fh.write("\n")


def emit_decode_step(case, cfg, tensors, P, x, positions, t, out_root, base_meta,
                     want_sensitivity):
    """Decode step t: an INDEPENDENT run over tokens 0..t, query row t recorded.

    Contract 7.2. This is deliberately NOT a slice of the full prefill.
    Computing it separately is the ONLY way `--verify` control 3's
    decode-equals-prefill report means anything; slicing would make the report
    a tautology."""
    name = f"{case['name']}_step{t}"
    B = case["B"]
    Lp = t + 1
    xs = x[:, :Lp, :].contiguous()
    ps = positions[:Lp]
    scfg = dict(cfg)
    scfg["B_L"] = (B, Lp)
    r64 = block_forward(scfg, P, xs, ps, torch.float64)
    r32 = block_forward(scfg, P, xs, ps, torch.float32)
    def sl(st):
        return slice_query_row(st, B, Lp, t)
    q64, q32 = sl(r64), sl(r32)
    meta = copy.deepcopy(base_meta)
    meta.update(name=name, L=1, prefix_len=Lp, decode_step=t, positions=ps,
                query_position=positions[t], parent_case=case["name"],
                certifies=f"decode step {t}: one query token at absolute position "
                          f"{positions[t]} against a cache of {Lp} keys, computed "
                          f"INDEPENDENTLY of the full prefill so that control 3's "
                          f"decode-equals-prefill report is a measurement.",
                cannot_certify="the parent case's limits, plus this: at pos0 == 0 a "
                               "relative-position index map is bit-inert here too, so these "
                               "steps do not certify absolute indexing. adv_rope_far_pos is "
                               "the only case that does.")
    meta["tensors"] = copy.deepcopy(base_meta["tensors"])
    meta["tensors"]["x"]["shape"] = [B, Lp, cfg["d_model"]]
    meta["tensors"]["x"]["derived"] = (
        f"the first {Lp} tokens of {case['name']}/x.f32; the QUERY is token {t}, and the "
        f"earlier tokens are the prefix whose keys and values are in the cache")
    cdir = os.path.join(out_root, name)
    os.makedirs(cdir, exist_ok=True)
    write_raw(os.path.join(cdir, "x.f32"), xs.numpy(), np.float32)
    for nm, arr in tensors.items():
        if nm != "x":
            write_raw(os.path.join(cdir, f"{nm}.f32"), arr, np.float32)
    write_raw(os.path.join(cdir, "positions.i32"), np.asarray(ps, np.int32), np.int32)
    rope = rope_tables(cfg["head_dim"], ps)
    stages = list(STAGE_ORDER)
    meta["stages"] = emit_stages(cdir, meta, q64, q32, stages, 1, Lp, rope)
    meta["stage_order"] = stages
    meta["reduced_stage_set"] = False
    meta["rope_anchor"] = rope_anchor_meta(rope)
    if want_sensitivity:
        meta["sensitivity"] = sensitivity(scfg, P, xs, ps, q64, stages, slicer=sl)
    write_manifest(cdir, meta)
    return dict(name=name, t=t, q64=q64, q32=q32, meta=meta)


def sha256_of_dir(root):
    h = hashlib.sha256()
    for dp, dn, fn in sorted(os.walk(root)):
        dn.sort()
        for f in sorted(fn):
            if f.endswith((".f32", ".f64", ".i32", ".json")):
                pth = os.path.join(dp, f)
                h.update(os.path.relpath(pth, root).encode())
                with open(pth, "rb") as fh:
                    h.update(fh.read())
    return h.hexdigest()


def generate(out_root, verify=False):
    os.makedirs(out_root, exist_ok=True)
    top = dict(
        corpus=CORPUS_VERSION, hash_spec=HASH_SPEC, profile=PROFILE,
        compare_version=COMPARE_VERSION, seed_base=f"0x{SEED_BASE:016X}",
        seed_rule="seed_k = seed_base + 0x1000*k with k the case index below; the x_source "
                  "cases and the decode-step subcases reuse the parent's seed",
        precision_rule="float64 everywhere EXCEPT the five FP32-pinned constants of the "
                       "arithmetic: the rotary inverse frequencies used for the angle, the "
                       "rotary angle, the attention scale, the RMS epsilon and the mask fill. "
                       "DEVIATION 1042; gen_corpus.py's module docstring carries the argument.",
        constants=dict(rms_eps=float(RMS_EPS_F32), rope_theta=float(ROPE_THETA_F32),
                       mask_fill=float(MASK_FILL_F32), unmasked_fill=float(UNMASKED_FILL_F32),
                       max_abs_position=MAX_ABS_POSITION, float32_eps=FP32_EPS),
        upstream=dict(transformers=f"https://github.com/huggingface/transformers @ "
                                   f"{TRANSFORMERS_COMMIT}, "
                                   f"src/transformers/models/llama/modeling_llama.py"),
        tensor_ids=TENSOR_IDS, stage_order=STAGE_ORDER, stage_definitions=STAGE_DEFS,
        batch_first_stages=sorted(BATCH_FIRST), config_stages=sorted(CONFIG_STAGES),
        perturbations=PERTURBATIONS, rtol_ladder=RTOL_LADDER, default_atol=DEFAULT_ATOL,
        cases=[])
    built = {}
    for k, case in enumerate(CASES):
        b = emit_case(case, out_root)
        built[case["name"]] = b
        top["cases"].append(dict(
            index=k, name=case["name"], seed=b["meta"]["seed"], B=case["B"], L=case["L"],
            pos0=case["pos0"], d_model=case["d_model"], n_heads=case["n_heads"],
            n_kv_heads=case["n_kv_heads"], head_dim=case["head_dim"],
            intermediate_size=case["intermediate_size"],
            reduced_stage_set=case["name"] in REDUCED_CASES,
            subcases=[s["name"] for s in b["subcases"]],
            expect_lane_failure=case["expect_fail"],
            certifies=case["certifies"], cannot_certify=case["cannot_certify"]))
    with open(os.path.join(out_root, "manifest.json"), "w") as fh:
        json.dump(top, fh, indent=1, sort_keys=True)
        fh.write("\n")
    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fn in os.walk(out_root) for f in fn)
    ncase = len(CASES) + sum(len(b["subcases"]) for b in built.values())
    print(f"wrote {ncase} case directories to {out_root}, "
          f"{total / 1e6:.3f} MB of tensor and manifest files")
    print(f"corpus sha256 (.f32 .f64 .i32 .json): {sha256_of_dir(out_root)}")
    print_calibration(built)
    return run_verify(out_root, built) if verify else 0


def _all_cases(built):
    for name, b in built.items():
        yield name, b["meta"]
        for sc in b["subcases"]:
            yield sc["name"], sc["meta"]


def print_calibration(built):
    """The tolerance table, per case per stage, from the corpus's own self test.

    Read this as the FLOOR. Tolerance is CALIBRATED PER CASE from a measurement
    of a plain FP32 run against the same float64 reference; it is never chosen
    globally. The mamba lane paid for that lesson once: its adv_gate_saturation
    case failed at rtol 1e-7 where TORCH'S OWN FP32 also failed, because the
    values reached 7.25e8 and 1e-7 is below float32 epsilon there. The
    tolerance was the defect, not the block."""
    print("\n== tolerance calibration: a plain FP32 run vs this corpus's float64 reference ==")
    print(f"   smallest ladder rung at which the FP32 arm passes, atol {DEFAULT_ATOL:g}. "
          f"NONE means it cannot pass at {RTOL_LADDER[-1]:g}, which is a finding about the "
          f"case, not a licence to widen the ladder.")
    print(f"   {'case':38s} {'stage':16s} {'fp32':>8s} {'best':>8s} {'gate':>8s} "
          f"{'max_rel':>10s}")
    for name, meta in _all_cases(built):
        loose = [(s, meta["stages"][s]) for s in meta["stage_order"]
                 if meta["stages"][s]["rtol_torch_fp32"] is None
                 or meta["stages"][s]["rtol_torch_fp32"] > RTOL_LADDER[0]]
        if not loose:
            print(f"   {name:38s} {'(every stage)':16s} {RTOL_LADDER[0]:8g} "
                  f"{'':>8s} {RTOL_LADDER[1]:8g}")
            continue
        for s, info in loose:
            rt = info["rtol_torch_fp32"]
            br = info["rtol_correctly_rounded"]
            print(f"   {name:38s} {s:16s} "
                  f"{('NONE' if rt is None else f'{rt:g}'):>8s} "
                  f"{('NONE' if br is None else f'{br:g}'):>8s} "
                  f"{info['rtol_gate']:8g} {info['torch_max_rel']:10.3e}")
    print("   fp32 = the reference arm's rung; best = the same value rounded once to float32, "
          "which is the floor no implementation beats; gate = the checker's default, one rung "
          "looser than fp32. fp32 far above best means the REFERENCE arm is the limit.")


def run_verify(out_root, built):
    """Every control this corpus has. Each one states what it proves.

    A checker with no negative control passes for ever. These are the controls
    that keep the clauses honest; the checker's own `--negative-control` is the
    other half."""
    rc = 0

    print("\n== control 1: determinism (regenerate into a temp dir, byte-compare) ==")
    print("   proves the corpus is a pure function of this file. An unseeded RNG, a dict")
    print("   ordering dependency or a thread-count dependency shows up here and nowhere else.")
    tmp = tempfile.mkdtemp(prefix="xfmr-corpus-")
    try:
        for case in CASES:
            emit_case(case, tmp, want_sensitivity=False)
        mism, n = [], 0
        for dp, _, fn in os.walk(tmp):
            for f in fn:
                if f == "manifest.json":
                    continue     # the real run carries a sensitivity table this one skips
                n += 1
                a = os.path.join(dp, f)
                bp = os.path.join(out_root, os.path.relpath(a, tmp))
                if not os.path.isfile(bp):
                    mism.append((os.path.relpath(a, tmp), "MISSING in corpus"))
                    continue
                with open(a, "rb") as fa, open(bp, "rb") as fb:
                    if fa.read() != fb.read():
                        mism.append((os.path.relpath(a, tmp), "DIFFERENT"))
        print(f"   data files compared: {n}, mismatches: {len(mism)}")
        for m in mism:
            print("    MISMATCH", m)
        rc |= 1 if mism else 0
    finally:
        shutil.rmtree(tmp)

    print("\n== control 2: batch composition, the two comp rows against their parent ==")
    print("   proves the corpus is consistent with itself across B. It does NOT prove the")
    print("   lane's batch invariance, which is clause (c) and is about our execution plan.")
    par = built["base_b2_l4_d32_kv2"]
    PB, PL = par["meta"]["B"], par["meta"]["L"]
    for b_ in (0, 1):
        ch = built[f"comp_row{b_}_b1_l4_d32_kv2"]
        for tag, key in (("ref64", "r64"), ("ref32", "r32")):
            for s in ("attn.weights", "attn.ctx", "residual2.out"):
                a, c = par[key][s], ch[key][s]
                a = a[b_:b_ + 1] if s in BATCH_FIRST else \
                    a.reshape(PB, PL, -1)[b_:b_ + 1].reshape(c.shape)
                print(f"   row{b_} {tag} {s:16s} bit-equal={bool(torch.equal(a, c))} "
                      f"maxabs={float((a - c).abs().max()):.3e}")

    print("\n== control 3: decode step t against prefill row t, in torch's own float64 ==")
    print("   this is contract 7.1's masked-tail-is-inert theorem MEASURED. If the extra")
    print("   masked terms were not exactly +0.0, or torch folded them in a different order,")
    print("   these would not be bit-equal and this corpus could not carry clause (d) at all.")
    dc = built["decode_b1_l4_d32_kv1"]
    for sc in dc["subcases"]:
        row = slice_query_row(dc["r64"], dc["meta"]["B"], dc["meta"]["L"], sc["t"])
        bad = []
        for s in ("attn.max", "attn.denom", "attn.ctx", "residual2.out"):
            a = np.ascontiguousarray(row[s].detach().numpy())
            b = np.ascontiguousarray(sc["q64"][s].detach().numpy())
            if a.shape != b.shape:
                bad.append((s, "SHAPE", list(a.shape), list(b.shape)))
            elif not np.array_equal(a, b):
                bad.append((s, "DIFFER", float(np.abs(a - b).max())))
        print(f"   step{sc['t']}: "
              f"{'bit-equal at every compared stage' if not bad else bad}")

    print("\n== control 4: the decode NEGATIVE control, two different steps must DIFFER ==")
    print("   without this a broken position map passes clause (d) for ever. This is the")
    print("   mamba lane's clause (c) lesson: a gate that only shows two things agreeing")
    print("   never shows the instrument moving. A bit-equal pair below makes the gate vacuous.")
    steps = dc["subcases"]
    for i in range(len(steps) - 1):
        a = np.ascontiguousarray(steps[i]["q64"]["residual2.out"].detach().numpy())
        b = np.ascontiguousarray(steps[i + 1]["q64"]["residual2.out"].detach().numpy())
        same = a.shape == b.shape and np.array_equal(a, b)
        print(f"   step{i} vs step{i + 1} residual2.out differ={not same}")
        rc |= 1 if same else 0

    print("\n== control 5: the rotary anchor, float64 inverse frequencies vs the FP32 ones ==")
    print("   the numbers below are WHY the precision rule exists. Each is the largest absolute")
    print("   difference between a cos table built from exact float64 inverse frequencies and")
    print("   one built from the reference's FP32 inverse frequencies, at the SAME positions.")
    print(f"   float32 epsilon is {FP32_EPS:.3e}. A gap far above it means a float64-anchored")
    print("   corpus would fail torch itself, hardest at the case that is worth the most.")
    for name, meta in _all_cases(built):
        ra = meta["rope_anchor"]
        print(f"   {name:38s} max|angle|={ra['max_abs_angle']:12.4f}  "
              f"cos gap={ra['cos_gap_f64_vs_f32_inv_freq']:.3e}  "
              f"sin gap={ra['sin_gap_f64_vs_f32_inv_freq']:.3e}")

    print("\n== control 6: the attention scale, both spellings ==")
    print("   contract section 3 measured agreement at seven head_dim values, none of them 24.")
    for name, b in built.items():
        m = b["meta"]
        print(f"   {name:38s} head_dim={m['head_dim']:3d} ref={m['attention_scale']!r} "
              f"alt={m['attention_scale_alt_spelling']!r} "
              f"agree={m['attention_scale_spellings_agree']}")

    print("\n== control 7: adversarial intent reached ==")
    print("   a case that does not reach its intent is decoration. Each line is a COUNT and a")
    print("   zero where a nonzero is described is a defect in THIS generator, not in the lane.")
    m = built["adv_mask_extreme_b1_l4_d32_kv2"]
    sc64 = m["r64"]["attn.scores"].detach().numpy()
    mk64 = m["r64"]["attn.masked"].detach().numpy()
    mk32 = m["r32"]["attn.masked"].detach().numpy()
    add_moved = int((np.abs(mk64 + FP32_MAX) > 2.0 ** 103).sum())
    print(f"   mask_extreme: |score| min={np.abs(sc64).min():.3e} max={np.abs(sc64).max():.3e}; "
          f"masked cells where the ADD is not the identity: {add_moved}; "
          f"FP32 infinities in attn.masked: {int(np.isinf(mk32).sum())} of {mk32.size}; "
          f"ref64 all finite: {bool(np.isfinite(mk64).all())}")
    if add_moved == 0:
        print("    INTENT NOT REACHED: widen this case's q_proj and k_proj ranges")
        rc |= 1
    m = built["adv_subnormal_scores_b1_l4_d32_kv2"]
    s64 = np.abs(m["r64"]["attn.scores"].detach().numpy())
    s32 = np.abs(m["r32"]["attn.scores"].detach().numpy())
    nsub = int(((s32 > 0) & (s32 < FP32_MIN_NORMAL)).sum())
    print(f"   subnormal_scores: |score| float64 min={s64.min():.3e} max={s64.max():.3e}; "
          f"FP32 subnormals={nsub}, FP32 exact zeros={int((s32 == 0).sum())} of {s32.size}")
    if nsub == 0 and int((s32 == 0).sum()) == 0:
        print("    INTENT NOT REACHED: this case's q_proj and k_proj ranges are too large")
        rc |= 1
    m = built["adv_signed_zeros_b2_l4_d32_kv2"]
    xn = m["x"].numpy()
    n1 = m["r64"]["norm1.out"].detach().numpy()
    qn = m["r64"]["q_proj.out"].detach().numpy()
    n1neg = int(np.signbit(n1[n1 == 0]).sum())
    print(f"   signed_zeros: x has {int((xn == 0).sum())} zeros, "
          f"{int(np.signbit(xn[xn == 0]).sum())} of them -0.0; ref64 norm1.out has "
          f"{int((n1 == 0).sum())} zeros, {n1neg} of them -0.0; ref64 q_proj.out has "
          f"{int((qn == 0).sum())} zeros, {int(np.signbit(qn[qn == 0]).sum())} of them -0.0 "
          f"(the reference's own GEMM launders the sign, which is the stated limit)")
    if n1neg == 0:
        print("    INTENT NOT REACHED: no negative zero survives to norm1.out")
        rc |= 1
    m = built["adv_softmax_saturation_b1_l8_d32_kv1"]
    e32 = m["r32"]["attn.exp"].detach().numpy()
    e64 = m["r64"]["attn.exp"].detach().numpy()
    mk = m["r64"]["attn.masked"].detach().numpy()
    mx = m["r64"]["attn.max"].detach().numpy()
    spread = mx[..., None] - mk
    unmasked = spread < 1e30
    per_row = np.where(unmasked, spread, 0.0).max(-1)
    rows_sat = int((per_row > 87.33655).sum())
    print(f"   softmax_saturation: rows={per_row.size}, rows whose UNMASKED spread exceeds "
          f"exp's FP32 underflow decade 87.33655: {rows_sat}; FP32 exact zeros in attn.exp="
          f"{int((e32 == 0).sum())} of {e32.size}; float64 min nonzero="
          f"{(e64[e64 > 0].min() if (e64 > 0).any() else float('nan')):.3e}")
    if rows_sat == 0 or rows_sat == per_row.size:
        print("    INTENT NOT REACHED: this case needs BOTH saturating and non-saturating rows")
        rc |= 1
    m = built["adv_rope_far_pos_b1_l4_d32_kv2"]
    print(f"   rope_far_pos: positions {m['positions']}, ceiling {MAX_ABS_POSITION} "
          f"(strictly less), max|angle|={m['meta']['rope_anchor']['max_abs_angle']:.4f}")

    print("\n== control 8: no nonfinite value in any float64 reference ==")
    print("   contract section 8 refuses a nonfinite INPUT. A nonfinite float64 REFERENCE would")
    print("   mean THIS GENERATOR produced a NaN and every stage after it is meaningless.")
    bad = []
    for name, b in built.items():
        for s, v in b["r64"].items():
            a = v.detach().numpy()
            if not np.isfinite(a).all():
                bad.append((name, s, int((~np.isfinite(a)).sum())))
        for sc in b["subcases"]:
            for s, v in sc["q64"].items():
                a = v.detach().numpy()
                if not np.isfinite(a).all():
                    bad.append((sc["name"], s, int((~np.isfinite(a)).sum())))
    print(f"   nonfinite float64 stages: {len(bad)}")
    for t in bad:
        print("    NONFINITE", t)
    rc |= 1 if bad else 0

    print("\n== control 9: sensitivity, what each case can and cannot see ==")
    print("   SEEN means a case catches that deliberate wrong answer at the tolerance the")
    print("   reference arm itself needs. A perturbation with NO seen stage anywhere is a")
    print("   clause THIS ENTIRE CORPUS cannot reach; it must be carried by the lane's own")
    print("   sabotage and must never be quoted as covered by the corpus.")
    cover = {p: [] for p in PERTURBATIONS}
    for name, meta in _all_cases(built):
        for p, v in meta.get("sensitivity", {}).items():
            if v.get("seen"):
                cover[p].append(name)
    for p in sorted(cover):
        who = cover[p]
        if who:
            print(f"   SEEN      {p:26s} by {len(who):2d} case(s), first: {who[0]}")
        else:
            print(f"   UNREACHED {p:26s} NO CASE IN THIS CORPUS CAN SEE THIS")
    print("\nverify exit code:", rc)
    return rc


def run_self_test(out_root):
    """Print the calibration table from the COMMITTED corpus, no recomputation.

    Same measurement `generate` prints, read back off disk, so it works against
    a corpus somebody else generated. numpy only, no torch needed for the read
    itself. The sibling `tools/transformer_corpus_check.py --self-test` prints
    the same table for ONE case."""
    top_path = os.path.join(out_root, "manifest.json")
    if not os.path.isfile(top_path):
        sys.exit(f"error: {top_path} not found; generate the corpus first")
    with open(top_path) as fh:
        top = json.load(fh)
    names = []
    for row in top["cases"]:
        names.append(row["name"])
        names.extend(row.get("subcases", []))
    print(f"self-test: {len(names)} cases, atol {top['default_atol']:g}, "
          f"ladder {top['rtol_ladder']}")
    print("  ref32 is a plain FP32 CPU run of the same algorithm. It is NOT a target. This")
    print("  table is the measured FLOOR: a lane dump held to a tighter rtol than the number")
    print("  below is being held to a bar the reference itself cannot clear, which is the")
    print("  tolerance being the defect rather than the block (mamba's adv_gate_saturation).")
    print(f"\n  {'case':38s} {'stage':16s} {'fp32':>8s} {'best':>8s} {'gate':>8s} "
          f"{'max_rel':>10s}")
    rc = 0
    for name in names:
        p = os.path.join(out_root, name, "manifest.json")
        if not os.path.isfile(p):
            print(f"  {name:38s} MANIFEST MISSING")
            rc = 1
            continue
        with open(p) as fh:
            m = json.load(fh)
        loose = [(s, m["stages"][s]) for s in m["stage_order"]
                 if m["stages"][s]["rtol_torch_fp32"] is None
                 or m["stages"][s]["rtol_torch_fp32"] > top["rtol_ladder"][0]]
        if not loose:
            print(f"  {name:38s} {'(every stage)':16s} {top['rtol_ladder'][0]:8g}")
            continue
        for s, info in loose:
            rt, br = info["rtol_torch_fp32"], info["rtol_correctly_rounded"]
            if rt is None:
                rc = 1
            print(f"  {name:38s} {s:16s} {('NONE' if rt is None else f'{rt:g}'):>8s} "
                  f"{('NONE' if br is None else f'{br:g}'):>8s} "
                  f"{info['rtol_gate']:8g} {info['torch_max_rel']:10.3e}")
    print("\n  NONE means the FP32 arm cannot pass at the loosest rung. That is a REPORTABLE")
    print("  finding about the case and not a licence to widen the ladder.")
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--verify", action="store_true",
                    help="run every control and print what each one proves")
    ap.add_argument("--self-test", action="store_true",
                    help="print the per-case per-stage tolerance calibration from the "
                         "committed corpus and exit; does not regenerate anything")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(run_self_test(a.out))
    print(f"torch {torch.__version__} numpy {np.__version__} python {sys.version.split()[0]}")
    print(f"profile {PROFILE}  hash spec {HASH_SPEC}  corpus {CORPUS_VERSION}")
    sys.exit(generate(a.out, verify=a.verify))


if __name__ == "__main__":
    main()
