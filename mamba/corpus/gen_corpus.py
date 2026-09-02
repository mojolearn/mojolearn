# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Independent reference corpus for the Mamba-1 block identity lane.

WHAT THIS IS. A deterministic generator of small Mamba-1 block cases (inputs,
parameters, and per-stage reference outputs) whose expected values come from
SOMEBODY ELSE'S algorithm, not from this repository's own tally. The math is
`mamba_ssm/ops/selective_scan_interface.py::selective_scan_ref` (state-spaces/
mamba, copied VERBATIM below, citation at the copy) and the block order of
HuggingFace `transformers/src/transformers/models/mamba/modeling_mamba.py`
(`MambaMixer.forward`, `causal_conv1d_fn`, `mamba_selective_scan`,
`MambaRMSNorm.forward`, `MambaBlock.forward`), re-implemented here in pure
torch with every step cited by file and line. Neither upstream package is
imported; the mamba_ssm wheel does not build on a Mac and the transformers
package would only add a dependency on its own kernel dispatch.

WHAT IT IS NOT. The float64 reference (`ref64/`) is a TOLERANCE reference for
the lane's stage dumps. The lane's bitwise oracle is its own pinned host
oracle; this corpus exists so the expected values are not solely our own
tally. The float32 torch run (`ref32/`) is informative only; plain torch FP32
on a CPU is not a target and not a gate. Nothing here is a bitwise
certificate of anything.

EVERY TENSOR ELEMENT IS A HASHED VALUE (uniform test data hides permutation):
value = f32(lo + (hi - lo) * top24bits(splitmix64(key + i)) * 2^-24) with
key = splitmix64(seed ^ (tensor_id << 32)); the arithmetic is exact in float64
for every (lo, hi) used here and rounds ONCE to float32, so a Mojo program can
regenerate the same bits from the README's spec with no library RNG. The
generator asserts that exactness with exact rationals for every element.

THIS FILE ALSO CARRIES THE MAMBA-2 (SSD) AND MAMBA-3 (SISO) CORPORA (one
corpus tool, the orchestrator's decision 2026-09-01). Each family lives in
its own clearly fenced section below (mamba2: profile
`mojolearn.identical.mamba2.fp32.v1`, `mamba/IDENTICAL_MAMBA2_CONTRACT.md`
section 8g; mamba3: profile `mojolearn.identical.mamba3.siso.fp32.v1`,
`mamba/IDENTICAL_MAMBA3_CONTRACT.md` section 8g, its case table MIRRORING
`mamba/checks/mamba3_fixture.mojo`, which landed FIRST and is normative),
emits into `mamba/corpus/mamba2/<case>/` / `mamba/corpus/mamba3/<case>/`
with its own top-level manifest, and leaves every Mamba-1 byte untouched
(the Mamba-1 sha256 is computed EXCLUDING `mamba2/` and `mamba3/`).

Run (see README.md for the scratch venv):
    python mamba/corpus/gen_corpus.py [--out mamba/corpus] [--verify] [--family mamba1|mamba2|mamba3|all]
`--verify` additionally runs the verbatim selective_scan_ref over the HF-shaped
path with z and D and compares against the composed stages, and regenerates
every file into a temp dir and byte-compares (determinism). The mamba2 family
has its own verify arm (verbatim HF mamba2_chunk_scan, verbatim mamba_ssm
ssd_minimal_discrete, composition rows, determinism, adversarial reach,
decode-prefix property).
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from fractions import Fraction
# `Optional` and `Tuple` exist ONLY because the verbatim mamba3 reference
# copies below (`mamba3_siso_step_ref`, `mamba3_siso_fwd_ref`) carry them in
# their signatures; the upstream test file imports them at its own top.
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# `einops` IS IMPORTED INSIDE `selective_scan_ref`, NOT HERE. DEVIATION 1934,
# 2026-08-28. It is used at five lines, all of them inside that one function
# (the reference SSM scan), and nothing else in this file touches it --
# splitmix64, the dyadic range helpers, shapes_for, default_ranges, rmsnorm
# and TENSOR_IDS are all pure numpy/torch.
#
# At module scope it took five unrelated lanes down with it. `speed_torch_seq.py`
# imports this file for those generic helpers and then dies at import when
# einops is absent, so on the Apple box the attention, mlp, rmsnorm,
# transformer and gemm torch arms ALL refused with "No module named 'einops'"
# and the whole Apple gemmseq board came home with NO OPPONENT ON THIS BOX on
# every row. The mamba reference path still needs einops and still refuses by
# name without it, which is correct; the other five lanes never needed it.
#
# Moving an import cannot move a bit: no corpus value changes.

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

HASH_SPEC = "mojolearn.mamba.corpus.hash.v1"
CORPUS_VERSION = "mamba-corpus-v1"

# --------------------------------------------------------------------------
# Hash spec (mojolearn.mamba.corpus.hash.v1). Written out in C-like pseudocode
# in README.md; this is the executable copy. All uint64 arithmetic wraps.
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
    """Vectorized splitmix64 on a uint64 numpy array (wrapping arithmetic)."""
    z = z.astype(np.uint64)
    with np.errstate(over="ignore"):
        z = z + np.uint64(SM_GAMMA)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(SM_M1)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(SM_M2)
        return z ^ (z >> np.uint64(31))


def hashed_unit(seed, tensor_id, n):
    """n values f_i = top24bits(splitmix64(key + i)) * 2^-24, key = splitmix64(seed ^ (tid << 32)).

    Returned as float64 (each value is an exact multiple of 2^-24 in [0, 1))."""
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
    """f32(lo + (hi - lo) * f) computed exactly in float64 and rounded ONCE.

    (lo, hi) must be dyadic rationals with few significant bits; the generator
    checks exactness for every element against the exact rational."""
    assert _is_dyadic_small(lo) and _is_dyadic_small(hi), (lo, hi)
    span = float(hi) - float(lo)
    v64 = float(lo) + span * f_unit
    # exactness check against exact rationals (cheap; tensors are small)
    flo, fspan = Fraction(lo), Fraction(hi) - Fraction(lo)
    for fv, v in zip(f_unit.tolist(), v64.tolist()):
        exact = flo + fspan * Fraction(fv)
        assert Fraction(v) == exact, ("float64 evaluation is not exact", lo, hi, fv, v)
    return v64.astype(np.float32)


# --------------------------------------------------------------------------
# Tensor ids and default ranges
# --------------------------------------------------------------------------
TENSOR_IDS = {
    "x": 1,
    "in_proj.weight": 2,
    "conv1d.weight": 3,
    "conv1d.bias": 4,
    "x_proj.weight": 5,
    "dt_proj.weight": 6,
    "dt_proj.bias": 7,
    "A_log": 8,
    "D": 9,
    "out_proj.weight": 10,
    "norm.weight": 11,
}


def fan_in_scale(fan_in):
    """s = 0.5 / 2^ceil(log2(fan_in)/2): a dyadic stand-in for 0.5/sqrt(fan_in)."""
    return 0.5 / (2 ** math.ceil(math.log2(fan_in) / 2))


def default_ranges(d_model, d_inner, dt_rank, d_state, d_conv):
    s_in = fan_in_scale(d_model)
    s_x = fan_in_scale(d_inner)
    s_out = fan_in_scale(d_inner)
    return {
        "x": (-2.0, 2.0),
        "norm.weight": (0.5, 1.5),
        "in_proj.weight": (-s_in, s_in),
        "conv1d.weight": (-0.5, 0.5),
        "conv1d.bias": (-0.125, 0.125),
        "x_proj.weight": (-s_x, s_x),
        "dt_proj.weight": (-1.0, 1.0),
        "dt_proj.bias": (-7.0, -2.0),  # softplus(dt) then spans about [0.0009, 0.127]
        "A_log": (0.0, 2.75),  # A = -exp(A_log) in [-15.64, -1]
        "D": (0.5, 1.5),
        "out_proj.weight": (-s_out, s_out),
    }


def shapes_for(d_model, d_inner, dt_rank, d_state, d_conv, B, L):
    return {
        "x": [B, L, d_model],
        "in_proj.weight": [2 * d_inner, d_model],
        "conv1d.weight": [d_inner, 1, d_conv],
        "conv1d.bias": [d_inner],
        "x_proj.weight": [dt_rank + 2 * d_state, d_inner],
        "dt_proj.weight": [d_inner, dt_rank],
        "dt_proj.bias": [d_inner],
        "A_log": [d_inner, d_state],
        "D": [d_inner],
        "out_proj.weight": [d_model, d_inner],
        "norm.weight": [d_model],
    }


# --------------------------------------------------------------------------
# Case table. seed_k = 0x4D616D6261436F72 ("MambaCor") + 0x1000 * k.
# Every case carries: B, L, d_model, optional range overrides, optional
# row-segment overrides, optional planted-zero rule, a stage set, and a note.
# --------------------------------------------------------------------------
SEED_BASE = 0x4D616D6261436F72
D_STATE = 16
D_CONV = 4
EXPAND = 2
EPS = 1e-5

FULL_STAGES = [
    "norm.out", "in_proj.out", "conv.out", "silu.out", "x_proj.out",
    "dt_proj.out", "softplus.out", "scan.y", "scan.h_last", "gate.out",
    "out_proj.out", "block.out",
]
REDUCED_STAGES = ["scan.y", "scan.h_last", "block.out"]

CASES = [
    # name, B, L, d_model, kind, extras
    dict(name="base_b1_l1_d8", B=1, L=1, d_model=8, note="smallest shape; L=1 is also the decode step shape"),
    dict(name="base_b2_l4_d8", B=2, L=4, d_model=8, note="L equals d_conv; conv window exactly fills; h_all written"),
    dict(name="base_b3_l16_d8", B=3, L=16, d_model=8, note="B=3 (odd batch), L=16"),
    dict(name="base_b1_l64_d8", B=1, L=64, d_model=8, note="L=64"),
    dict(name="base_b1_l1_d16", B=1, L=1, d_model=16, note="d_model 16, L=1"),
    dict(name="base_b3_l4_d16", B=3, L=4, d_model=16, note="d_model 16, B=3, L=d_conv; h_all written"),
    dict(name="base_b2_l16_d16", B=2, L=16, d_model=16, note="d_model 16, L=16"),
    dict(name="base_b1_l64_d16", B=1, L=64, d_model=16, note="d_model 16, L=64"),
    dict(name="adv_softplus_guard_b2_l8_d8", B=2, L=8, d_model=8,
         overrides={"dt_proj.bias": (19.875, 20.125)},
         note="(a) dt_proj outputs straddle torch's softplus threshold 20 (the CUDA kernel's delta <= 20 guard); bias in [19.875, 20.125], the projected part adds at most a few tenths"),
    dict(name="adv_a_very_negative_b1_l16_d16", B=1, L=16, d_model=16,
         overrides={"A_log": (6.0, 10.0)},
         note="(b) A in [-22026, -403]; exp(delta*A) crosses the FP32 denormal band [-103.97, -87.34] and underflows to zero; ref64 keeps the true tiny values"),
    dict(name="adv_a_near_zero_b3_l8_d8", B=3, L=8, d_model=8,
         overrides={"A_log": (-18.0, -12.0)},
         note="(c) A in [-6.1e-6, -1.5e-8]; exp(delta*A) is within a few ulp of 1 in FP32 and rounds to exactly 1 for the smallest products"),
    dict(name="adv_signed_zeros_b2_l8_d8", B=2, L=8, d_model=8,
         zero_rule="x[b,t,:] = +0.0 for t%4==0; x[b,t,:] = -0.0 for t%4==2; of the remaining elements, flat index i%7==3 is -0.0",
         note="(d) x carries whole +0.0 tokens, whole -0.0 tokens and scattered -0.0; RMSNorm of a zero token is 0*rsqrt(eps); a -0.0 token propagates -0.0 through in_proj only if the accumulator starts from the first product, not from +0.0 (the signed-zero hazard)"),
    dict(name="adv_gate_saturation_b1_l8_d16", B=1, L=8, d_model=16,
         segments={"in_proj.weight": [("gate_half", -1073741824.0, 1073741824.0)]},
         note="(e) the gate half of in_proj.weight (rows d_inner..2*d_inner) is in [-2^30, 2^30]; z is ~1e9, exp(-z) overflows to inf for negative z so sigmoid saturates to 0 (silu -> -0.0) and to 1 for positive z; the residual is lost in block.out"),
    dict(name="comp_b2_l257_d8", B=2, L=257, d_model=8, stages=REDUCED_STAGES,
         note="(f) long L=257 (one past a 256 chunk) with B=2; its two rows are comp_row0/comp_row1 with the SAME parameters; reduced stage set for size"),
    dict(name="comp_row0_b1_l257_d8", B=1, L=257, d_model=8, stages=REDUCED_STAGES,
         x_source=("comp_b2_l257_d8", 0),
         note="(f) row 0 of comp_b2_l257_d8 as a B=1 case (same seed, same parameters, x = parent x[0])"),
    dict(name="comp_row1_b1_l257_d8", B=1, L=257, d_model=8, stages=REDUCED_STAGES,
         x_source=("comp_b2_l257_d8", 1),
         note="(f) row 1 of comp_b2_l257_d8 as a B=1 case (same seed, same parameters, x = parent x[1])"),
]

# (g) decode: the per-token step reference IS the prefill reference read one
# token at a time; scan.h_all is written for the L <= 4 cases so the lane's
# decode step can also check its state after every token. See README.md.
H_ALL_MAX_L = 4


def case_seed(k):
    return (SEED_BASE + 0x1000 * k) & M64


def case_index(name):
    for k, c in enumerate(CASES):
        if c["name"] == name:
            return k
    raise KeyError(name)


# --------------------------------------------------------------------------
# selective_scan_ref, copied VERBATIM from
# https://github.com/state-spaces/mamba  commit e9594ce1c732d97440f0332fdc43170a2294dbfa
# mamba_ssm/ops/selective_scan_interface.py lines 127-194
# (Copyright (c) 2023, Tri Dao, Albert Gu; Apache-2.0). Only the indentation
# of this docstring line is ours. Do not edit.
# --------------------------------------------------------------------------
def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    from einops import rearrange, repeat  # DEVIATION 1934: lazy, see the note at the imports
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)
# ---- end of verbatim copy ------------------------------------------------


# NOTE on `.float()` inside selective_scan_ref: `Tensor.float()` casts to
# float32. For the float64 reference we call it on float64 tensors, which
# would silently downcast. We therefore call the verbatim function through
# `_scan_ref_dtype`, which temporarily rebinds `torch.Tensor.float` to a
# no-op cast to the working dtype. This keeps the copy verbatim and the
# reference in float64; the float32 run is unaffected (float() is identity).
class _scan_ref_dtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __enter__(self):
        self._orig = torch.Tensor.float
        dtype = self.dtype
        torch.Tensor.float = lambda t: t.to(dtype)
        return self

    def __exit__(self, *a):
        torch.Tensor.float = self._orig


# --------------------------------------------------------------------------
# The HF block, re-implemented step by step in pure torch. Citations are to
# huggingface/transformers commit d56c55bf564ddb176759eb6ec199442682564916,
# src/transformers/models/mamba/modeling_mamba.py (MM) and
# src/transformers/models/mamba/configuration_mamba.py (MC).
# (The brief named `MambaMixer.slow_forward`; at this commit that method no
# longer exists under that name. Its body is `MambaMixer.forward` MM:359-477
# plus the fallback functions `causal_conv1d_fn` MM:78-98 and
# `mamba_selective_scan` MM:162-258, which are what runs without the CUDA
# kernels.)
# --------------------------------------------------------------------------
def rmsnorm(x, w, eps):
    # MM:494-499 MambaRMSNorm.forward:
    #   variance = hidden_states.pow(2).mean(-1, keepdim=True)
    #   hidden_states = hidden_states * torch.rsqrt(variance + eps)
    #   return self.weight * hidden_states
    variance = x.pow(2).mean(-1, keepdim=True)
    return w * (x * torch.rsqrt(variance + eps))


def block_forward(p, x, dtype, want_h_all=False):
    """Returns an ordered dict of stage -> tensor, all in `dtype`.

    p: dict of parameter tensors (float32 on disk, cast to dtype here)."""
    P = {k: v.to(dtype) for k, v in p.items()}
    x = x.to(dtype)
    Bsz, L, d_model = x.shape
    d_inner = P["D"].shape[0]
    d_state = P["A_log"].shape[1]
    dt_rank = P["dt_proj.weight"].shape[1]
    out = {}

    # MM:521-527 MambaBlock.forward: residual = hidden_states;
    # hidden_states = self.norm(hidden_states); ... hidden_states = residual + mixer(...)
    residual = x
    h = rmsnorm(x, P["norm.weight"], EPS)  # MC:70 layer_norm_epsilon = 1e-5
    out["norm.out"] = h  # [B, L, d_model]

    # MM:371 projected_states = self.in_proj(hidden_states).transpose(1, 2)
    # in_proj: nn.Linear(hidden, 2*intermediate, bias=config.use_bias) MM:319, use_bias False MC:75
    proj = F.linear(h, P["in_proj.weight"]).transpose(1, 2)
    out["in_proj.out"] = proj  # [B, 2*d_inner, L]

    # MM:373 A = -torch.exp(self.A_log.float())
    A = -torch.exp(P["A_log"])
    # MM:396 hidden_states_B_C, gate = projected_states.chunk(2, dim=1)
    hs, gate = proj.chunk(2, dim=1)

    # MM:410-416 causal_conv1d_fn(hidden_states_B_C, conv1d.weight.squeeze(1), conv1d.bias, activation="silu")
    # MM:78-98: F.conv1d(x, weight.unsqueeze(1), bias, padding=K-1, groups=d_inner)[:, :, :seq_len]; then ACT2FN["silu"]
    # conv1d: nn.Conv1d(d_inner, d_inner, bias=use_conv_bias(True MC:76), kernel_size=4 (MC:74), groups=d_inner, padding=3) MM:303-310
    conv = F.conv1d(hs, P["conv1d.weight"], P["conv1d.bias"], padding=D_CONV - 1, groups=d_inner)[:, :, :L]
    out["conv.out"] = conv  # [B, d_inner, L]
    u = F.silu(conv)
    out["silu.out"] = u  # [B, d_inner, L]

    # MM:422-427 time_step, B, C = torch.split(self.x_proj(hidden_states_B_C.transpose(1,2)), [dt_rank, N, N], dim=-1)
    # x_proj: nn.Linear(d_inner, dt_rank + 2N, bias=False) MM:321; dt_rank = ceil(hidden/16) MC:96-98
    xdbl = F.linear(u.transpose(1, 2), P["x_proj.weight"])
    out["x_proj.out"] = xdbl  # [B, L, dt_rank + 2*d_state]
    dt_low, Bm, Cm = torch.split(xdbl, [dt_rank, d_state, d_state], dim=-1)

    # MM:429 time_step = self.dt_proj.weight @ time_step.transpose(1, 2)   -> [B, d_inner, L], NO bias yet
    # MM:430 time_proj_bias = self.dt_proj.bias.float()
    # MM:178-179 (mamba_selective_scan): dt = dt + delta_bias[..., None]; MM:180-181: dt = F.softplus(dt)
    # This corpus defines dt_proj.out = W @ dt + bias (bias folded in, as the brief states) and softplus.out = softplus(dt_proj.out).
    dt_raw = torch.matmul(P["dt_proj.weight"], dt_low.transpose(1, 2))
    dt_b = dt_raw + P["dt_proj.bias"][None, :, None]
    out["dt_proj.out"] = dt_b  # [B, d_inner, L]
    delta = F.softplus(dt_b)  # torch: x if x > 20 else log1p(exp(x))
    out["softplus.out"] = delta  # [B, d_inner, L]

    # MM:446-458 mamba_selective_scan(hidden_states_B_C.transpose(1,2) -> [B, d_inner, L], time_step, A,
    #   B.transpose(1,2) -> [B, N, L], C.transpose(1,2) -> [B, N, L], D=self.D.float(), z=gate,
    #   delta_bias=time_proj_bias, delta_softplus=True)
    # MM:183-188: discrete_A = exp(A * dt); deltaB_u = dt * B * u
    # MM:227-243: recurrent form h = dA*h + dBu; y_t = h @ C_t    (the same recurrence as selective_scan_ref lines 174-187)
    # MM:245-246: scan_output + hidden_states * D;  MM:248-249: * silu(z)
    # scan.y is defined as selective_scan_ref's `out` with z=None, i.e. y + u*D, BEFORE the gate.
    Bt = Bm.transpose(1, 2)  # [B, N, L]
    Ct = Cm.transpose(1, 2)  # [B, N, L]
    with _scan_ref_dtype(dtype):
        y_D, h_last = selective_scan_ref(u, dt_raw, A, Bt, Ct, D=P["D"], z=None,
                                         delta_bias=P["dt_proj.bias"], delta_softplus=True,
                                         return_last_state=True)
    out["scan.y"] = y_D  # [B, d_inner, L]
    out["scan.h_last"] = h_last  # [B, d_inner, d_state]
    if want_h_all:
        # the per-token state sequence (decode reference), same recurrence as the verbatim loop
        deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
        deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, Bt, u)
        hstate = A.new_zeros((Bsz, d_inner, d_state))
        hs_all = []
        for i in range(L):
            hstate = deltaA[:, :, i] * hstate + deltaB_u[:, :, i]
            hs_all.append(hstate)
        out["scan.h_all"] = torch.stack(hs_all, dim=1)  # [B, L, d_inner, d_state]
    g = y_D * F.silu(gate)
    out["gate.out"] = g  # [B, d_inner, L]

    # MM:476 contextualized_states = self.out_proj(scan_output.transpose(1, 2)); out_proj bias False (use_bias)
    o = F.linear(g.transpose(1, 2), P["out_proj.weight"])
    out["out_proj.out"] = o  # [B, L, d_model]
    # MM:527 hidden_states = residual + hidden_states
    out["block.out"] = residual + o  # [B, L, d_model]
    return out


def block_via_scan_ref_with_gate(p, x, dtype):
    """Cross-check path: the verbatim selective_scan_ref called WITH z and D as
    HF's mamba_selective_scan is called, so gate.out must equal its output."""
    P = {k: v.to(dtype) for k, v in p.items()}
    x = x.to(dtype)
    L = x.shape[1]
    d_inner = P["D"].shape[0]
    d_state = P["A_log"].shape[1]
    dt_rank = P["dt_proj.weight"].shape[1]
    h = rmsnorm(x, P["norm.weight"], EPS)
    proj = F.linear(h, P["in_proj.weight"]).transpose(1, 2)
    A = -torch.exp(P["A_log"])
    hs, gate = proj.chunk(2, dim=1)
    u = F.silu(F.conv1d(hs, P["conv1d.weight"], P["conv1d.bias"], padding=D_CONV - 1, groups=d_inner)[:, :, :L])
    xdbl = F.linear(u.transpose(1, 2), P["x_proj.weight"])
    dt_low, Bm, Cm = torch.split(xdbl, [dt_rank, d_state, d_state], dim=-1)
    dt_raw = torch.matmul(P["dt_proj.weight"], dt_low.transpose(1, 2))
    with _scan_ref_dtype(dtype):
        g, h_last = selective_scan_ref(u, dt_raw, A, Bm.transpose(1, 2), Cm.transpose(1, 2), D=P["D"], z=gate,
                                       delta_bias=P["dt_proj.bias"], delta_softplus=True, return_last_state=True)
    o = F.linear(g.transpose(1, 2), P["out_proj.weight"])
    return g, h_last, x + o


# --------------------------------------------------------------------------
# Tensor materialization for a case
# --------------------------------------------------------------------------
def gen_tensor(seed, name, shape, lo, hi, segments=None):
    n = int(np.prod(shape))
    f = hashed_unit(seed, TENSOR_IDS[name], n)
    if segments is None:
        v = map_range(f, lo, hi)
    else:
        v = map_range(f, lo, hi)
        arr = v.reshape(shape)
        for seg_name, slo, shi in segments:
            if seg_name == "gate_half":
                rows = arr.shape[0] // 2
                fseg = f.reshape(shape)[rows:].reshape(-1)
                arr[rows:] = map_range(fseg, slo, shi).reshape(arr[rows:].shape)
            else:
                raise ValueError(seg_name)
        v = arr.reshape(-1)
    return v.reshape(shape)


def apply_zero_rule(x, rule):
    """x[b,t,:] = +0.0 for t%4==0; x[b,t,:] = -0.0 for t%4==2; remaining flat i%7==3 -> -0.0."""
    assert rule.startswith("x[b,t,:] = +0.0 for t%4==0")
    Bsz, L, d = x.shape
    flat = x.reshape(-1).copy()
    keep = np.ones(flat.shape[0], dtype=bool)
    for b in range(Bsz):
        for t in range(L):
            if t % 4 == 0:
                base = (b * L + t) * d
                flat[base:base + d] = np.float32(0.0)
                keep[base:base + d] = False
            elif t % 4 == 2:
                base = (b * L + t) * d
                flat[base:base + d] = np.float32(-0.0)
                keep[base:base + d] = False
    idx = np.arange(flat.shape[0])
    sel = keep & (idx % 7 == 3)
    flat[sel] = np.float32(-0.0)
    return flat.reshape(x.shape)


def build_case(case):
    k = case_index(case["name"])
    d_model = case["d_model"]
    d_inner = EXPAND * d_model
    dt_rank = math.ceil(d_model / 16)
    B, L = case["B"], case["L"]
    src = case.get("x_source")
    seed = case_seed(case_index(src[0])) if src else case_seed(k)
    shapes = shapes_for(d_model, d_inner, dt_rank, D_STATE, D_CONV, B, L)
    ranges = default_ranges(d_model, d_inner, dt_rank, D_STATE, D_CONV)
    ranges.update(case.get("overrides", {}))
    segments = case.get("segments", {})
    tensors = {}
    for name, shape in shapes.items():
        if name == "x" and src:
            parent = next(c for c in CASES if c["name"] == src[0])
            pshape = shapes_for(d_model, d_inner, dt_rank, D_STATE, D_CONV, parent["B"], parent["L"])["x"]
            lo, hi = ranges["x"]
            px = gen_tensor(seed, "x", pshape, lo, hi)
            tensors["x"] = px[src[1]:src[1] + 1].copy()
            continue
        lo, hi = ranges[name]
        tensors[name] = gen_tensor(seed, name, shape, lo, hi, segments.get(name))
    if "zero_rule" in case:
        tensors["x"] = apply_zero_rule(tensors["x"], case["zero_rule"])
    meta = dict(
        name=case["name"], corpus=CORPUS_VERSION, hash_spec=HASH_SPEC,
        seed=f"0x{seed:016X}", seed_case_index=(case_index(src[0]) if src else k),
        B=B, L=L, d_model=d_model, d_inner=d_inner, d_state=D_STATE, d_conv=D_CONV,
        dt_rank=dt_rank, expand=EXPAND, rms_eps=EPS, softplus_threshold=20.0,
        tensors={name: dict(shape=shapes[name], dtype="float32", order="row-major",
                            tensor_id=TENSOR_IDS[name], lo=ranges[name][0], hi=ranges[name][1],
                            file=f"{name}.f32",
                            **({"segments": [dict(rows=s[0], lo=s[1], hi=s[2]) for s in segments[name]]}
                               if name in segments else {}))
                 for name in shapes},
        note=case["note"],
    )
    if src:
        meta["x_source"] = dict(case=src[0], batch_row=src[1])
        meta["tensors"]["x"]["derived"] = f"byte slice of {src[0]}/x.f32, batch row {src[1]}"
    if "zero_rule" in case:
        meta["zero_rule"] = case["zero_rule"]
        meta["tensors"]["x"]["derived"] = "hashed, then zero_rule applied"
    return tensors, meta


STAGE_SHAPES = {
    "norm.out": lambda m: [m["B"], m["L"], m["d_model"]],
    "in_proj.out": lambda m: [m["B"], 2 * m["d_inner"], m["L"]],
    "conv.out": lambda m: [m["B"], m["d_inner"], m["L"]],
    "silu.out": lambda m: [m["B"], m["d_inner"], m["L"]],
    "x_proj.out": lambda m: [m["B"], m["L"], m["dt_rank"] + 2 * m["d_state"]],
    "dt_proj.out": lambda m: [m["B"], m["d_inner"], m["L"]],
    "softplus.out": lambda m: [m["B"], m["d_inner"], m["L"]],
    "scan.y": lambda m: [m["B"], m["d_inner"], m["L"]],
    "scan.h_last": lambda m: [m["B"], m["d_inner"], m["d_state"]],
    "scan.h_all": lambda m: [m["B"], m["L"], m["d_inner"], m["d_state"]],
    "gate.out": lambda m: [m["B"], m["d_inner"], m["L"]],
    "out_proj.out": lambda m: [m["B"], m["L"], m["d_model"]],
    "block.out": lambda m: [m["B"], m["L"], m["d_model"]],
}


def write_raw(path, arr, dtype):
    a = np.ascontiguousarray(np.asarray(arr, dtype=dtype))
    with open(path, "wb") as fh:
        fh.write(a.astype("<" + a.dtype.str[1:]).tobytes())


def emit_case(case, out_root):
    tensors, meta = build_case(case)
    stages = list(case.get("stages", FULL_STAGES))
    if case["L"] <= H_ALL_MAX_L and "stages" not in case:
        stages.insert(stages.index("scan.h_last") + 1, "scan.h_all")
    cdir = os.path.join(out_root, case["name"])
    os.makedirs(os.path.join(cdir, "ref64"), exist_ok=True)
    os.makedirs(os.path.join(cdir, "ref32"), exist_ok=True)
    for name, arr in tensors.items():
        write_raw(os.path.join(cdir, f"{name}.f32"), arr, np.float32)
    p = {k: torch.from_numpy(v.copy()) for k, v in tensors.items() if k != "x"}
    x = torch.from_numpy(tensors["x"].copy())
    want_h_all = "scan.h_all" in stages
    r64 = block_forward(p, x, torch.float64, want_h_all)
    r32 = block_forward(p, x, torch.float32, want_h_all)
    stage_meta = {}
    for s in stages:
        shp = STAGE_SHAPES[s](meta)
        assert list(r64[s].shape) == shp, (s, r64[s].shape, shp)
        write_raw(os.path.join(cdir, "ref64", f"{s}.f64"), r64[s].numpy(), np.float64)
        write_raw(os.path.join(cdir, "ref32", f"{s}.f32"), r32[s].numpy(), np.float32)
        stage_meta[s] = dict(shape=shp, order="row-major", ref64=f"ref64/{s}.f64", ref32=f"ref32/{s}.f32")
    meta["stages"] = stage_meta
    meta["stage_order"] = stages
    meta["stage_definitions"] = "see ../manifest.json"
    with open(os.path.join(cdir, "manifest.json"), "w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return meta, r64, r32, p, x


STAGE_DEFS = {
    "norm.out": "weight * (x * rsqrt(mean(x^2, last) + 1e-5))  [B, L, d_model]",
    "in_proj.out": "(norm.out @ in_proj.weight^T).transpose(1,2)  [B, 2*d_inner, L]; rows 0..d_inner are the hidden half, rows d_inner..2*d_inner the gate z",
    "conv.out": "causal depthwise conv1d of the hidden half, kernel 4, padding 3, truncated to L, plus conv1d.bias  [B, d_inner, L]",
    "silu.out": "silu(conv.out)  [B, d_inner, L]  (this is u)",
    "x_proj.out": "silu.out.transpose(1,2) @ x_proj.weight^T  [B, L, dt_rank + 2*d_state]; columns split (dt_rank | B | C)",
    "dt_proj.out": "dt_proj.weight @ dt_low^T + dt_proj.bias  [B, d_inner, L]  (bias INCLUDED)",
    "softplus.out": "softplus(dt_proj.out) with torch's threshold 20 (x if x > 20 else log1p(exp(x)))  [B, d_inner, L]  (this is delta)",
    "scan.y": "selective_scan_ref(u, dt, A=-exp(A_log), B, C, D, z=None, delta_bias, delta_softplus=True) = y_t + u_t*D, BEFORE the gate  [B, d_inner, L]",
    "scan.h_last": "the state after the last token  [B, d_inner, d_state]",
    "scan.h_all": "the state after every token  [B, L, d_inner, d_state]  (decode reference; L <= 4 cases only)",
    "gate.out": "scan.y * silu(z)  [B, d_inner, L]",
    "out_proj.out": "gate.out.transpose(1,2) @ out_proj.weight^T  [B, L, d_model]",
    "block.out": "x + out_proj.out  [B, L, d_model]",
}


def sha256_of_dir(root, skip_dirs=()):
    h = hashlib.sha256()
    for dp, dn, fn in sorted(os.walk(root)):
        dn[:] = sorted(d for d in dn if d not in skip_dirs)
        for f in sorted(fn):
            if f.endswith((".f32", ".f64", ".json")):
                pth = os.path.join(dp, f)
                h.update(os.path.relpath(pth, root).encode())
                with open(pth, "rb") as fh:
                    h.update(fh.read())
    return h.hexdigest()


def generate(out_root, verify=False):
    os.makedirs(out_root, exist_ok=True)
    top = dict(corpus=CORPUS_VERSION, hash_spec=HASH_SPEC, seed_base=f"0x{SEED_BASE:016X}",
               seed_rule="seed_k = seed_base + 0x1000 * k, k = case index below; x_source cases reuse the parent's seed",
               d_state=D_STATE, d_conv=D_CONV, expand=EXPAND, rms_eps=EPS,
               upstream=dict(
                   mamba="https://github.com/state-spaces/mamba @ e9594ce1c732d97440f0332fdc43170a2294dbfa, mamba_ssm/ops/selective_scan_interface.py::selective_scan_ref L127-194",
                   transformers="https://github.com/huggingface/transformers @ d56c55bf564ddb176759eb6ec199442682564916, src/transformers/models/mamba/modeling_mamba.py",
               ),
               tensor_ids=TENSOR_IDS, stage_definitions=STAGE_DEFS, cases=[])
    total_bytes = 0
    all_meta = {}
    for k, case in enumerate(CASES):
        meta, r64, r32, p, x = emit_case(case, out_root)
        all_meta[case["name"]] = (meta, r64, r32, p, x)
        row = dict(index=k, name=case["name"], seed=meta["seed"], B=case["B"], L=case["L"],
                   d_model=case["d_model"], d_inner=meta["d_inner"], dt_rank=meta["dt_rank"],
                   stages=meta["stage_order"], note=case["note"])
        top["cases"].append(row)
        cdir = os.path.join(out_root, case["name"])
        total_bytes += sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fn in os.walk(cdir) for f in fn)
    with open(os.path.join(out_root, "manifest.json"), "w") as fh:
        json.dump(top, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"wrote {len(CASES)} cases to {out_root}, {total_bytes / 1e6:.3f} MB of tensor and manifest files")
    print(f"mamba1 corpus sha256 (files .f32 .f64 .json, mamba2/ and mamba3/ excluded): {sha256_of_dir(out_root, skip_dirs=('mamba2', 'mamba3'))}")
    if verify:
        run_verify(out_root, all_meta)


def run_verify(out_root, all_meta):
    print("\n== verify 1: verbatim selective_scan_ref WITH z and D (the HF call) vs the composed stages ==")
    worst = 0.0
    for name, (meta, r64, r32, p, x) in all_meta.items():
        for dtype, r in ((torch.float64, r64), (torch.float32, r32)):
            g, hl, bo = block_via_scan_ref_with_gate(p, x, dtype)
            same = (torch.equal(g, r["gate.out"]) and torch.equal(hl, r["scan.h_last"])
                    and torch.equal(bo, r["block.out"]))
            d = max(float((g - r["gate.out"]).abs().max()), float((hl - r["scan.h_last"]).abs().max()),
                    float((bo - r["block.out"]).abs().max()))
            # nan-safe
            if torch.isnan(g).any() or torch.isnan(r["gate.out"]).any():
                same = torch.equal(torch.nan_to_num(g), torch.nan_to_num(r["gate.out"]))
            worst = max(worst, d if d == d else 0.0)
            print(f"  {name:36s} {str(dtype):14s} bit-equal={same} maxabs={d:.3e}")
    print(f"  worst abs difference (both dtypes): {worst:.3e}")

    print("\n== verify 2: batch composition (comp rows vs parent, ref64 and ref32) ==")
    pm = all_meta["comp_b2_l257_d8"]
    for b in (0, 1):
        cm = all_meta[f"comp_row{b}_b1_l257_d8"]
        for tag, idx in (("ref64", 1), ("ref32", 2)):
            for s in ("scan.y", "scan.h_last", "block.out"):
                a = pm[idx][s][b:b + 1]
                c = cm[idx][s]
                print(f"  row{b} {tag} {s:12s} bit-equal={torch.equal(a, c)} maxabs={float((a - c).abs().max()):.3e}")

    print("\n== verify 3: determinism (regenerate into a temp dir, byte-compare) ==")
    tmp = tempfile.mkdtemp(prefix="mamba-corpus-")
    try:
        for case in CASES:
            emit_case(case, tmp)
        mism = []
        for dp, _, fn in os.walk(tmp):
            for f in fn:
                a = os.path.join(dp, f)
                bpath = os.path.join(out_root, os.path.relpath(a, tmp))
                with open(a, "rb") as fa, open(bpath, "rb") as fb:
                    if fa.read() != fb.read():
                        mism.append(os.path.relpath(a, tmp))
        print(f"  files compared: {sum(len(fn) for _, _, fn in os.walk(tmp))}, mismatches: {len(mism)}")
        for m in mism:
            print("   MISMATCH", m)
    finally:
        shutil.rmtree(tmp)

    print("\n== verify 4: adversarial intent reached ==")
    m, r64, r32 = all_meta["adv_softplus_guard_b2_l8_d8"][:3]
    dt = r64["dt_proj.out"]
    print(f"  softplus guard: dt_proj.out min={float(dt.min()):.4f} max={float(dt.max()):.4f} "
          f"count<=20: {int((dt <= 20).sum())} count>20: {int((dt > 20).sum())}")
    m, r64, r32 = all_meta["adv_a_very_negative_b1_l16_d16"][:3]
    A = -torch.exp(all_meta["adv_a_very_negative_b1_l16_d16"][3]["A_log"].double())
    dA = torch.exp(torch.einsum("bdl,dn->bdln", r64["softplus.out"], A))
    dA32 = dA.float()
    print(f"  A very negative: exp(delta*A) float32: zeros={int((dA32 == 0).sum())} "
          f"denormals={int(((dA32 != 0) & (dA32.abs() < 1.1754944e-38)).sum())} normals={int((dA32.abs() >= 1.1754944e-38).sum())} of {dA32.numel()}")
    m, r64, r32 = all_meta["adv_a_near_zero_b3_l8_d8"][:3]
    A = -torch.exp(all_meta["adv_a_near_zero_b3_l8_d8"][3]["A_log"].double())
    dA = torch.exp(torch.einsum("bdl,dn->bdln", r64["softplus.out"], A)).float()
    ulps = ((1.0 - dA) / np.float32(2 ** -24)).max()
    print(f"  A near zero: exp(delta*A) float32 min={float(dA.min()):.9f}, max distance from 1 = {float(ulps):.1f} ulp(1-), exactly 1: {int((dA == 1).sum())} of {dA.numel()}")
    m, r64, r32, p, x = all_meta["adv_signed_zeros_b2_l8_d8"]
    xn = x.numpy()
    print(f"  signed zeros: x has {int((xn == 0).sum())} zeros of which {int(np.signbit(xn[xn == 0]).sum())} are -0.0; "
          f"ref64 in_proj.out -0.0 count={int(np.signbit(r64['in_proj.out'].numpy()[r64['in_proj.out'].numpy() == 0]).sum())}, "
          f"ref32 in_proj.out -0.0 count={int(np.signbit(r32['in_proj.out'].numpy()[r32['in_proj.out'].numpy() == 0]).sum())}")
    m, r64, r32 = all_meta["adv_gate_saturation_b1_l8_d16"][:3]
    z = r32["in_proj.out"][:, m["d_inner"]:, :]
    sig = torch.sigmoid(z)
    print(f"  gate saturation: |z| min={float(z.abs().min()):.3e} max={float(z.abs().max()):.3e}; "
          f"sigmoid exactly 0: {int((sig == 0).sum())}, exactly 1: {int((sig == 1).sum())} of {sig.numel()}; "
          f"block.out max |.|={float(r32['block.out'].abs().max()):.3e}")


# ==========================================================================
# ==========================================================================
#                    MAMBA-2 (SSD) CORPUS EXTENSION
#
# Profile: mojolearn.identical.mamba2.fp32.v1
# (mamba/IDENTICAL_MAMBA2_CONTRACT.md; section 8g is the clause this section
# discharges). Same discipline as the Mamba-1 half above: every input element
# is a hashed value under mojolearn.mamba.corpus.hash.v1 -- NEW tensor names,
# NEW ids, NEW seed base -- torch float64 per-stage references spelled by the
# upstreams' OWN code (verbatim copies below, cited by line), float32 run
# informative only, nothing here a bitwise certificate of anything. Cases
# land in mamba/corpus/mamba2/<case>/ with their own top-level manifest; the
# Mamba-1 corpus above is byte-untouched.
#
# References (both checkouts in /Users/andrewhendel/CascadeProjects/upstream/):
#   state-spaces/mamba @ e9594ce1c732d97440f0332fdc43170a2294dbfa
#     mamba_ssm/modules/ssd_minimal.py::segsum (:23-32) and
#     ::ssd_minimal_discrete (:34-78), composed DISCRETIZE-FIRST as its own
#     test composes it (:94-103, `ssd_minimal_discrete(x*dt, A*dt, B, C)`)
#     -- the NORMATIVE reference, copied verbatim below;
#     mamba_ssm/modules/mamba2.py::Mamba2.forward non-mem-eff arm (:209-276),
#     ::step (:278-343), ::allocate_inference_cache (:345-355) -- block order
#     and state shapes;
#     mamba_ssm/ops/triton/ssd_chunk_state.py::_chunk_cumsum_fwd_kernel
#     (:72-86) -- the dt seam's order (bias :73-75, softplus with the <= 20
#     guard :76-77, clamp :78-81);
#     mamba_ssm/ops/triton/layernorm_gated.py::rms_norm_ref (:18-39) -- the
#     gated RMSNorm: gate BEFORE norm at norm_before_gate=False (:26-27),
#     rstd spelled `1 / torch.sqrt(...)` (:29), never torch.rsqrt.
#   huggingface/transformers @ d56c55bf564ddb176759eb6ec199442682564916
#     src/transformers/models/mamba2/modeling_mamba2.py::pad_tensor_by_size
#     (:42-50), ::reshape_into_chunks (:53-70), ::segment_sum (:73-90),
#     ::mamba2_chunk_scan (:254-348), ::MambaRMSNormGated (:105-120),
#     ::Mamba2RMSNorm (:591-605), ::Mamba2Block.forward (:617-631) -- the
#     second independent reference. Where it disagrees with mamba_ssm (its
#     torch.rsqrt at :118) the mamba_ssm spelling wins (contract section 1,
#     the same answer mamba1 DEVIATION 741 gave).
#
# The corpus reference spells the inter-chunk state pass as the references
# spell it (the decay-matrix pass, ssd_minimal.py:64-69 / HF :318-326). The
# PROFILE pins the SERIAL recurrence instead (DEVIATION 785); at float64
# tolerance scale the two agree, and the corpus is a TOLERANCE reference,
# never a bitwise oracle -- exactly the Mamba-1 corpus's standing.
# ==========================================================================

M2_PROFILE = "mojolearn.identical.mamba2.fp32.v1"
M2_CORPUS_VERSION = "mamba2-corpus-v1"
M2_SEED_BASE = 0x4D6D6232436F7270  # "Mmb2Corp"
M2_D_STATE = 128   # N, mamba2.py:41 (the shipped module's default)
M2_D_CONV = 4      # mamba2.py:42
M2_EXPAND = 2      # mamba2.py:44
M2_HEADDIM = 64    # P, mamba2.py:45
M2_NGROUPS = 1     # G, mamba2.py:47
M2_CHUNK = 256     # Q, mamba2.py:59 -- PART OF THE ARITHMETIC (DEVIATION 783)
M2_EPS = 1e-5      # both norms, mamba2.py:144 / HF layer_norm_epsilon
M2_INF = float("inf")

# New tensor names, new ids (contract 8g: "with new tensor names"); ids are
# disjoint from the Mamba-1 table above so no (seed, id) pair can alias.
M2_TENSOR_IDS = {
    "x": 21,
    "block_norm.weight": 22,   # the block's Mamba2RMSNorm weight (HF :597)
    "in_proj.weight": 23,      # [2*d_inner + 2*G*N + H, d_model], mamba2.py:96-98
    "conv1d.weight": 24,       # [CD, 1, 4], mamba2.py:105-113
    "conv1d.bias": 25,
    "dt_bias": 26,             # [H], mamba2.py:127
    "A_log": 27,               # [H], mamba2.py:133-135
    "D": 28,                   # [H] (D_has_hdim=False), mamba2.py:139
    "norm.weight": 29,         # gated RMSNorm weight [d_ssm], mamba2.py:144
    "out_proj.weight": 30,     # [d_model, d_inner], mamba2.py:148
    "initial_states": 31,      # [B, H, P, N], ssd_minimal.py:64-66 semantics
}


def m2_default_ranges(d_model):
    d_inner = M2_EXPAND * d_model
    s_in = fan_in_scale(d_model)
    s_out = fan_in_scale(d_inner)
    return {
        "x": (-2.0, 2.0),
        "block_norm.weight": (0.5, 1.5),
        "in_proj.weight": (-s_in, s_in),
        "conv1d.weight": (-0.5, 0.5),
        "conv1d.bias": (-0.125, 0.125),
        "dt_bias": (-7.0, -2.0),   # softplus(dt) then spans about [0.0009, 0.127]
        "A_log": (0.0, 2.75),      # A = -exp(A_log) in [-15.64, -1] (A_init_range (1,16) intent)
        "D": (0.5, 1.5),
        "norm.weight": (0.5, 1.5),
        "out_proj.weight": (-s_out, s_out),
        "initial_states": (-1.0, 1.0),
    }


def m2_shapes_for(d_model, B, L, with_init=False):
    assert d_model % 32 == 0, d_model  # contract section 3: d_model a multiple of 32
    d_inner = M2_EXPAND * d_model
    H = d_inner // M2_HEADDIM
    CD = d_inner + 2 * M2_NGROUPS * M2_D_STATE
    d_in_proj = 2 * d_inner + 2 * M2_NGROUPS * M2_D_STATE + H
    s = {
        "x": [B, L, d_model],
        "block_norm.weight": [d_model],
        "in_proj.weight": [d_in_proj, d_model],
        "conv1d.weight": [CD, 1, M2_D_CONV],
        "conv1d.bias": [CD],
        "dt_bias": [H],
        "A_log": [H],
        "D": [H],
        "norm.weight": [d_inner],
        "out_proj.weight": [d_model, d_inner],
    }
    if with_init:
        s["initial_states"] = [B, H, M2_HEADDIM, M2_D_STATE]
    return s


# Stage sets. Card order per contract section 7. seg.L and cb.G are recorded
# only where the size cap allows (the two tiny sub-chunk cases); everywhere
# else the elision is STATED in the manifest, never silent (section 7's rule).
M2_FULL_STAGES = [
    "input.x", "norm.sumsq", "norm.out", "in_proj.out", "A.out",
    "conv.out", "silu.out", "conv.window", "dt.out", "xd.out",
    "dacs.out", "ydiag.out", "decay.states", "cstate.out", "pass.states",
    "yoff.out", "scan.y", "skip.out", "gnorm.gate", "gnorm.sumsq",
    "gnorm.out", "out_proj.out", "residual.out", "ssd.h_last",
]
M2_SEG_STAGES = (M2_FULL_STAGES[:11] + ["seg.L", "cb.G"] + M2_FULL_STAGES[11:])
M2_REDUCED_STAGES = [
    "input.x", "dt.out", "dacs.out", "decay.states", "cstate.out",
    "pass.states", "scan.y", "ssd.h_last", "residual.out",
]

# Case table. Contract section 8g by name: L in {1, 4, 256, 257, 513, 770};
# softplus band [8, 14] plants; signed-zero plants; A-near-zero; saturating
# gate; active dt_limit ((0.001, 0.1) planted); nonzero initial_states;
# composition rows. The STATEPASS witnesses (section 8f) carry an A_log
# override so exp(dA_cs_last) stays a NORMAL FP32 number across chunk hops --
# under the default range some heads' chunk decay underflows to +0.0 in FP32
# and the STATEPASS_MATRIX / STATEPASS_UNFUSED sabotage arms would be bitwise
# inert on those cells (a fixture that cannot witness is a clause not gated).
M2_CASES = [
    dict(name="m2_base_b1_l1_d32", B=1, L=1, d_model=32, stages=M2_SEG_STAGES,
         note="smallest shape; L=1 under CHUNK_SIZE=256 exercises the sub-chunk arm (one padded chunk) and is the decode step shape; seg.L/cb.G recorded (size allows)"),
    dict(name="m2_base_b2_l4_d32", B=2, L=4, d_model=32, stages=M2_SEG_STAGES,
         note="L = d_conv, the conv window exactly fills; sub-chunk arm; seg.L/cb.G recorded (size allows)"),
    dict(name="m2_base_b3_l4_d64", B=3, L=4, d_model=64,
         note="d_model 64 (H=2, the first multi-head shape), odd batch, sub-chunk arm; full stages, seg.L/cb.G elided for size"),
    dict(name="m2_base_b1_l256_d32", B=1, L=256, d_model=32,
         note="exactly one chunk, zero padding; the chunk boundary NOT crossed (the L=257 cases are the crossing)"),
    dict(name="m2_base_b1_l257_d64", B=1, L=257, d_model=64, stages=M2_REDUCED_STAGES,
         note="one past the chunk boundary at d_model 64 (H=2); two chunks, second nearly all padding"),
    dict(name="m2_comp_b2_l257_d32", B=2, L=257, d_model=32, stages=M2_REDUCED_STAGES,
         note="composition parent: B=2 at L=257; its rows are m2_comp_row0/row1 with the SAME parameters (batch-composition corpus rows, contract 8c is the lane's gate)"),
    dict(name="m2_comp_row0_b1_l257_d32", B=1, L=257, d_model=32, stages=M2_REDUCED_STAGES,
         x_source=("m2_comp_b2_l257_d32", 0),
         note="row 0 of m2_comp_b2_l257_d32 as a B=1 case (same seed, same parameters, x = parent x[0])"),
    dict(name="m2_comp_row1_b1_l257_d32", B=1, L=257, d_model=32, stages=M2_REDUCED_STAGES,
         x_source=("m2_comp_b2_l257_d32", 1),
         note="row 1 of m2_comp_b2_l257_d32, same construction"),
    dict(name="m2_statepass_b1_l513_d32", B=1, L=513, d_model=32, stages=M2_REDUCED_STAGES,
         overrides={"A_log": (-4.0, -1.0)},
         note="three chunks -> a TWO-HOP inter-chunk decay, the STATEPASS_MATRIX/STATEPASS_UNFUSED witness (contract 8f); A_log in [-4,-1] so A is in [-0.368, -0.018] and exp(dA_cs_last) stays a normal FP32 number over >= 2 hops (default range underflows some heads and leaves the arms inert)"),
    dict(name="m2_statepass_b2_l770_d64", B=2, L=770, d_model=64, stages=M2_REDUCED_STAGES,
         overrides={"A_log": (-4.0, -1.0)},
         note="four chunks (770 = 3*256 + 2, sub-chunk tail), B=2, H=2; the longest composition row; same liveness override as m2_statepass_b1_l513_d32"),
    dict(name="m2_adv_softplus_band_b2_l8_d32", B=2, L=8, d_model=32,
         overrides={"dt_bias": (8.0, 14.0), "A_log": (-18.0, -12.0)},
         note="(a) biased dt lands IN the distinguishing band [8, 14] where softplus(x) - x = log1p(exp(-x)) is 1..340 ulp of x (the contract's band rule: a 20-straddling fixture is vacuous, the mamba1 adv_softplus_guard lesson); A_log tiny keeps dt*A ~ -6e-5 per token so the scan stays alive"),
    dict(name="m2_adv_a_near_zero_b3_l64_d32", B=3, L=64, d_model=32,
         overrides={"A_log": (-18.0, -12.0)},
         note="(b) A in [-6.1e-6, -1.5e-8]; per-chunk decay within a few ulp of 1 in FP32 and the S15 subtraction dA_cs_last - dA_cs is cancellation-prone (the contract's named A-near-zero case)"),
    dict(name="m2_adv_signed_zeros_b2_l8_d32", B=2, L=8, d_model=32,
         zero_rule="x[b,t,:] = +0.0 for t%4==0; x[b,t,:] = -0.0 for t%4==2; of the remaining elements, flat index i%7==3 is -0.0",
         note="(c) x carries whole +0.0 tokens, whole -0.0 tokens and scattered -0.0 (the mamba1 rule verbatim); zero-sign propagation through the S1 fold and the S4 GEMM cells is compared BY SIGN BIT by the lane, which the tolerance cannot see"),
    dict(name="m2_adv_gate_saturation_b1_l8_d64", B=1, L=8, d_model=64,
         segments={"in_proj.weight": [("z_rows", -1073741824.0, 1073741824.0)]},
         note="(d) the z rows of in_proj.weight (rows 0..d_inner, the zxbcdt order z | xBC | dt of mamba2.py:211-215) are in [-2^30, 2^30]; z is ~1e9-1e10, silu(z) saturates to -0.0 / z, and gnorm.gate is dominated by saturated products"),
    dict(name="m2_adv_dt_limit_b2_l8_d32", B=2, L=8, d_model=32,
         dt_limit=(0.001, 0.1),
         overrides={"dt_bias": (-9.0, -1.0)},
         note="(e) ACTIVE dt_limit (0.001, 0.1) planted (contract 8f, the CLAMP_BEFORE_SOFTPLUS witness); limits are the FP32 values (hex in the manifest); dt_bias widened to [-9,-1] so softplus output spans [7.5e-5, 0.47] and BOTH limits bind"),
    dict(name="m2_init_states_b1_l257_d32", B=1, L=257, d_model=32,
         init_states=True, stages=M2_REDUCED_STAGES,
         overrides={"A_log": (-4.0, -1.0)},
         note="(f) nonzero hashed initial_states [B,H,P,N] (ssd_minimal.py:64-66 semantics, prepended as chunk -1's state); at L=257 this is the OTHER STATEPASS witness (contract 8f: a zero-init two-chunk case is bitwise inert); same A_log liveness override"),
    dict(name="m2_decode_b1_l260p1_d32", B=1, L=261, d_model=32,
         kind="decode", prefill_len=260,
         note="(g) decode continuation: prefill 260 (one COMPLETED chunk + a 4-row open-chunk buffer), then ONE token; records the three-piece resumption state of contract section 5 (conv window, boundary h, intra-chunk buffer) and the decode token's per-stage rows. DEVIATION 786: decode IS prefill resumption, so the L=261 prefill IS the decode reference"),
]


def m2_case_seed(k):
    return (M2_SEED_BASE + 0x1000 * k) & M64


def m2_case_index(name):
    for k, c in enumerate(M2_CASES):
        if c["name"] == name:
            return k
    raise KeyError(name)


def m2_case_by_name(name):
    return M2_CASES[m2_case_index(name)]


# --------------------------------------------------------------------------
# segsum and ssd_minimal_discrete, copied VERBATIM from
# https://github.com/state-spaces/mamba  commit e9594ce1c732d97440f0332fdc43170a2294dbfa
# mamba_ssm/modules/ssd_minimal.py lines 23-32 and 34-78
# (Copyright (c) 2024, Albert Gu and Tri Dao; Apache-2.0). Our only edits:
# the lazy einops import line inside each body (the DEVIATION 1934 rule --
# einops must not be a module-scope import of this file) and this banner.
# Do not edit further.
# --------------------------------------------------------------------------
def segsum(x):
    """More stable segment sum calculation."""
    from einops import rearrange, repeat  # DEVIATION 1934: lazy, ours
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None):
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    """
    from einops import rearrange, repeat  # DEVIATION 1934: lazy, ours
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

    # Rearrange into blocks/chunks
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag  = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag+Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state
# ---- end of verbatim mamba_ssm copy --------------------------------------


# --------------------------------------------------------------------------
# pad_tensor_by_size, reshape_into_chunks, segment_sum and mamba2_chunk_scan,
# copied VERBATIM from https://github.com/huggingface/transformers
# commit d56c55bf564ddb176759eb6ec199442682564916,
# src/transformers/models/mamba2/modeling_mamba2.py lines 42-50, 53-70,
# 73-90 and 254-348 (Apache-2.0). Our only edits: the
# @use_kernel_func_from_hub_with_fallback decorator at :253 is omitted
# (it is HF's kernel-dispatch plumbing, not math), type annotations that
# need HF-internal imports are stripped from the mamba2_chunk_scan
# signature, and this banner. Do not edit further. In the float64 run,
# .float() inside mamba2_chunk_scan is rebound by _scan_ref_dtype exactly
# as the Mamba-1 half does for selective_scan_ref.
# --------------------------------------------------------------------------
def pad_tensor_by_size(input_tensor, pad_size):
    """
    Padding x tensor with `pad_size` on the seq_len dim (dim=1)

    Assumes that we only have tensors of either size 4 or 3
    """
    pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(input_tensor.shape) == 4 else (0, 0, 0, pad_size, 0, 0)

    return torch.nn.functional.pad(input_tensor, pad_shape, mode="constant", value=0)


def reshape_into_chunks(input_tensor, pad_size, chunk_size):
    """
    Padding input_tensor with `pad_size` on the seq_len dim (dim=1) and
    simultaneously splitting it into chunk sequences.

    Assumes that we only have tensors of either size 4 or 3
    """
    # [bsz, seq_len, ...] -> [bsz, seq_len multiple of chunk_size, ...]
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)

    if len(input_tensor.shape) == 3:
        # [bsz, seq_len multiple of chunk_size, num_heads] -> [bsz, -1, chunk_size, num_heads]
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    else:
        # [bsz, seq_len multiple of chunk_size, num_heads, head_dim or state_size] -> [bsz, -1, chunk_size, num_heads, head_dim or state_size]
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3]
        )


def segment_sum(input_tensor):
    """
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    """
    chunk_size = input_tensor.size(-1)
    # 1. expand input tensor to have an additional dimension and repeat along that dimension
    # [..., chunk_size] -> [..., chunk_size, chunk_size]
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    # 2. create a lower triangular mask with the diagonal set to 0 to 0 out elements above diag
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    # 3. compute actual cumsum
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)

    # 4. apply mask to keep only the lower triangular part of the cumulative sum result (incl diagonal this time)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum


def mamba2_chunk_scan(
    hidden_states,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    dt_bias=None,
    initial_states=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    return_final_states=False,
    **kwargs,
):
    batch_size, sequence_length, num_heads, head_dim = hidden_states.shape
    num_groups = B.shape[2]

    if dt_bias is not None:
        dt = dt + dt_bias.to(dt.dtype)
    if dt_softplus:
        dt = F.softplus(dt)
    dt = torch.clamp(dt, min=dt_limit[0], max=dt_limit[1])

    hidden_states = hidden_states.float()
    B = B.float().repeat_interleave(num_heads // num_groups, dim=2, output_size=num_heads)
    C = C.float().repeat_interleave(num_heads // num_groups, dim=2, output_size=num_heads)

    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    D_residual = None
    if D is not None:
        D_residual = D[..., None] * pad_tensor_by_size(hidden_states, pad_size)

    # Discretize x and A
    hidden_states = hidden_states * dt[..., None].float()
    A = A.to(hidden_states.dtype) * dt.float()

    # Rearrange into blocks/chunks
    hidden_states, A, B, C = [reshape_into_chunks(tensor, pad_size, chunk_size) for tensor in (hidden_states, A, B, C)]

    A = A.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    # This is the analog of a causal mask
    L = torch.exp(segment_sum(A))

    # Contraction of C and B to get G (attention-weights like)
    G = (C[:, :, :, None, :, :] * B[:, :, None, :, :, :]).sum(dim=-1)

    # Compute M, equivalent to applying attention mask to weights
    M = (G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]).sum(dim=-1)

    # Compute Y_diag (apply to values)
    Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
    states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    previous_states = (
        initial_states[:, None].to(dtype=states.dtype, device=states.device)
        if initial_states is not None
        else torch.zeros_like(states[:, :1])
    )
    states = torch.cat([previous_states, states], dim=1)
    decay_chunk = torch.exp(segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0)))).transpose(1, 3)
    new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    C_times_states = C[..., None, :] * states[:, :, None, ...]
    Y_off = C_times_states.sum(-1) * state_decay_out.permute(0, 2, 3, 1)[..., None]

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    output = Y_diag + Y_off
    output = output.reshape(batch_size, -1, num_heads, head_dim)

    if D_residual is not None:
        output = output + D_residual

    # Cutting off padded chunks
    if pad_size > 0:
        output = output[:, :sequence_length]

    if return_final_states:
        return output, final_state

    return output
# ---- end of verbatim HF copy ---------------------------------------------


# --------------------------------------------------------------------------
# The staged Mamba-2 block: every stage of contract section 7, spelled by the
# references' own ops, op for op (the HF chunk-scan sequence for S10-S20, so
# the verbatim mamba2_chunk_scan cross-check should be BIT-equal; the
# verbatim ssd_minimal_discrete cross-check is an independent spelling and
# agrees at roundoff scale). Citations: MM2 = modeling_mamba2.py at the HF
# pin; m2 = mamba_ssm/modules/mamba2.py; rmr = layernorm_gated.py::rms_norm_ref.
# --------------------------------------------------------------------------
def m2_forward(p, x, dtype, dt_limit=(0.0, M2_INF)):
    """Returns a dict of stage tag -> tensor (contract section 7's card
    shapes), all in `dtype`. `p` may carry "initial_states"."""
    P = {k: v.to(dtype) for k, v in p.items()}
    initial_states = P.pop("initial_states", None)
    x = x.to(dtype)
    Bsz, L, d_model = x.shape
    d_inner = M2_EXPAND * d_model
    H = d_inner // M2_HEADDIM
    N, G, Q, PP = M2_D_STATE, M2_NGROUPS, M2_CHUNK, M2_HEADDIM
    CD = d_inner + 2 * G * N
    d_in_proj = 2 * d_inner + 2 * G * N + H
    Mtok = Bsz * L
    nchunks = -(-L // Q)
    out = {}
    out["input.x"] = x.reshape(Mtok, d_model)

    # S1-S3 block RMSNorm (MM2 Mamba2RMSNorm :600-605 spells pow(2).mean and
    # torch.rsqrt; the card records the S1 fold's SUM, the S2 division, and
    # rstd as 1/sqrt per rms_norm_ref:29 -- the mamba_ssm spelling wins the
    # rsqrt disagreement, contract section 1 / mamba1 DEVIATION 741).
    residual = x  # MM2 :624
    sumsq = x.pow(2).sum(-1)
    out["norm.sumsq"] = sumsq.reshape(-1)
    rstd = 1 / torch.sqrt(sumsq / d_model + M2_EPS)
    hnorm = P["block_norm.weight"] * (x * rstd[..., None])  # MM2 :604-605 order
    out["norm.out"] = hnorm.reshape(Mtok, d_model)

    # S4 in_proj (m2 :178); column order z | xBC | dt_raw with d_mlp = 0
    # (m2 :210-215)
    zxbcdt = F.linear(hnorm, P["in_proj.weight"])
    out["in_proj.out"] = zxbcdt.reshape(Mtok, d_in_proj)
    z, xBC_raw, dt_raw = torch.split(zxbcdt, [d_inner, CD, H], dim=-1)

    # S5 A = -exp(A_log) (m2 :182)
    A = -torch.exp(P["A_log"])
    out["A.out"] = A

    # conv window AFTER the call (m2 :220-221: F.pad(xBC_t, (d_conv - L, 0)),
    # negative pad truncates from the left; zeros before the first token,
    # allocate_inference_cache :348-350)
    xt = xBC_raw.transpose(1, 2)
    out["conv.window"] = F.pad(xt, (M2_D_CONV - L, 0))

    # S6 causal depthwise conv1d, kernel 4, padding 3, truncated, bias
    # included (m2 :233, the torch fallback arm)
    conv = F.conv1d(xt, P["conv1d.weight"], P["conv1d.bias"],
                    padding=M2_D_CONV - 1, groups=CD)[:, :, :L].transpose(1, 2)
    out["conv.out"] = conv.reshape(Mtok, CD)

    # S7 SiLU (m2 :232 self.act)
    xBC = F.silu(conv)
    out["silu.out"] = xBC.reshape(Mtok, CD)

    # split x | B | C by copy (m2 :243)
    xs, Bs, Cs = torch.split(xBC, [d_inner, G * N, G * N], dim=-1)

    # S9 dt = clamp(softplus(dt_raw + dt_bias), dt_limit)
    # (_chunk_cumsum_fwd_kernel :73-81 order; MM2 :272-276 the same order in
    # torch; torch F.softplus threshold 20 == the kernel's <= 20 guard)
    dt = dt_raw + P["dt_bias"]
    dt = F.softplus(dt)
    dt = torch.clamp(dt, min=dt_limit[0], max=dt_limit[1])
    out["dt.out"] = dt.reshape(Mtok, H)

    # S10 DISCRETIZE FIRST (MM2 :288-289; the normative composition
    # ssd_minimal.py:103 `ssd_minimal_discrete(x*dt, A*dt, B, C)`; dt binds
    # to x and to A, NEVER to B -- DEVIATION 789)
    Xh = xs.reshape(Bsz, L, H, PP)
    Xd = Xh * dt[..., None]
    dA = A * dt
    out["xd.out"] = Xd.reshape(Mtok, H, PP)

    # ngroups broadcast of B and C by COPY (MM2 :279-280; a copy, not a seam)
    Bh = Bs.reshape(Bsz, L, G, N).repeat_interleave(H // G, dim=2, output_size=H)
    Ch = Cs.reshape(Bsz, L, G, N).repeat_interleave(H // G, dim=2, output_size=H)

    # padding to Q (MM2 :282, :292; contract section 3: pads are +0.0 inputs
    # and zero dt, the last chunk's fold is ALWAYS length Q)
    pad_size = (Q - L % Q) % Q
    Xc, dAc, Bc, Cc = [reshape_into_chunks(t, pad_size, Q) for t in (Xd, dA, Bh, Ch)]

    # S11 per-chunk cumsum (MM2 :294-295; zero-padded dA means padded
    # positions carry the last real value, the card's rule)
    dAp = dAc.permute(0, 3, 1, 2)  # [B, H, C, Q]
    A_cumsum = torch.cumsum(dAp, dim=-1)
    out["dacs.out"] = A_cumsum

    # S12 L = exp(segsum) (MM2 :299; exp(-inf) = +0.0 = the profile's
    # structural zero, DEVIATION 782)
    Lmat = torch.exp(segment_sum(dAp))  # [B, H, C, Q, Q]
    out["seg.L"] = Lmat.permute(0, 2, 1, 3, 4)  # card [B, C, H, Q, Q]

    # S12 G = C.B over n (MM2 :302); at G(roups)=1 every head column is a
    # bitwise COPY (broadcast by copy), so the card records one group row
    Gm = (Cc[:, :, :, None, :, :] * Bc[:, :, None, :, :, :]).sum(dim=-1)  # [B,C,Q,Q,H]
    for h in range(1, H):
        assert torch.equal(Gm[..., h], Gm[..., 0]), "ngroups=1 head copies must be bitwise identical"
    out["cb.G"] = Gm[..., :G].permute(0, 1, 4, 2, 3)  # card [B, C, G, Q, Q]

    # S13-S14 M = G (.) L, Y_diag = M . X_d (MM2 :305-308)
    Mm = (Gm[..., None] * Lmat.permute(0, 2, 3, 4, 1)[..., None]).sum(dim=-1)
    Y_diag = (Mm[..., None] * Xc[:, :, None]).sum(dim=3)  # [B, C, Q, H, P]
    out["ydiag.out"] = Y_diag.reshape(Bsz, nchunks * Q, H, PP)[:, :L].reshape(Mtok, H, PP)

    # S15 decay = exp(dA_cs_last - dA_cs), B_decay (MM2 :312-313)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    out["decay.states"] = decay_states  # card [B, H, C, Q]
    B_decay = Bc * decay_states.permute(0, -2, -1, 1)[..., None]

    # S16 chunk_states (MM2 :314)
    states = (B_decay[..., None, :] * Xc[..., None]).sum(dim=2)  # [B, C, H, P, N]
    out["cstate.out"] = states

    # S17 inter-chunk pass, REFERENCE spelling = the decay-matrix pass
    # (MM2 :318-326; ssd_minimal.py:64-69). The PROFILE pins the serial
    # recurrence (DEVIATION 785); this corpus is the float64 tolerance side.
    previous_states = (
        initial_states[:, None].to(dtype=states.dtype)
        if initial_states is not None
        else torch.zeros_like(states[:, :1])
    )
    states_cat = torch.cat([previous_states, states], dim=1)
    decay_chunk = torch.exp(segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0)))).transpose(1, 3)
    new_states = (decay_chunk[..., None, None] * states_cat[:, :, None, ...]).sum(dim=1)
    states_in, final_state = new_states[:, :-1], new_states[:, -1]
    out["pass.states"] = states_in   # card: the state ENTERING each chunk (h_{c-1})
    out["ssd.h_last"] = final_state  # section 5's report stage

    # S18 Y_off = (C . h_prev) (.) exp(dA_cs) (MM2 :330-332: contract over n
    # FIRST, scale AFTER -- HF's explicit order is the profile's)
    state_decay_out = torch.exp(A_cumsum)
    C_times_states = Cc[..., None, :] * states_in[:, :, None, ...]
    Y_off = C_times_states.sum(-1) * state_decay_out.permute(0, 2, 3, 1)[..., None]
    out["yoff.out"] = Y_off.reshape(Bsz, nchunks * Q, H, PP)[:, :L].reshape(Mtok, H, PP)

    # S19 Y = Y_diag + Y_off (MM2 :335-336)
    Y = (Y_diag + Y_off).reshape(Bsz, -1, H, PP)
    out["scan.y"] = Y[:, :L].reshape(Mtok, H, PP)

    # S20 D-residual from UNDISCRETIZED post-conv x, added LAST
    # (MM2 :284-285 and :338-339; truncation :342-343)
    D_residual = P["D"][..., None] * pad_tensor_by_size(Xh, pad_size)
    output = Y + D_residual
    if pad_size > 0:
        output = output[:, :L]
    out["skip.out"] = output.reshape(Mtok, H, PP)

    # S21 gated RMSNorm, gate BEFORE norm (m2 :268-270; rmr :26-30 at
    # norm_before_gate=False; group_size = the whole row at G = 1,
    # DEVIATION 787; eps 1e-5, m2 :144)
    y_flat = output.reshape(Bsz, L, d_inner)  # m2 :268 "b l h p -> b l (h p)"
    gate = y_flat * F.silu(z)                 # rmr :26-27
    out["gnorm.gate"] = gate.reshape(Mtok, d_inner)
    gsumsq = gate.pow(2).sum(-1)
    out["gnorm.sumsq"] = gsumsq.reshape(-1)
    grstd = 1 / torch.sqrt(gsumsq / d_inner + M2_EPS)  # rmr :29
    gout = gate * grstd[..., None] * P["norm.weight"]  # rmr :30 order (x * rstd * weight)
    out["gnorm.out"] = gout.reshape(Mtok, d_inner)

    # S4 out_proj (m2 :275), S22 residual (MM2 :630)
    o = F.linear(gout, P["out_proj.weight"])
    out["out_proj.out"] = o.reshape(Mtok, d_model)
    out["residual.out"] = (residual + o).reshape(Mtok, d_model)
    return out


def m2_cross_checks(meta, r, p, dtype, dt_limit):
    """The two verbatim upstream spellings run over the SAME staged front
    (norm/in_proj/conv/silu, read back from the staged run's own tensors):
    HF mamba2_chunk_scan (expected BIT-equal to the staged S9-S20) and
    mamba_ssm ssd_minimal_discrete composed discretize-first
    (ssd_minimal.py:103; an independent spelling, roundoff-scale agreement).
    Returns (hf_out [B,L,H,P], hf_final, ssd_y [B,L,H,P], ssd_final)."""
    Bsz, L = meta["B"], meta["L"]
    d_inner = meta["d_inner"]
    H = meta["nheads"]
    N, G, Q, PP = M2_D_STATE, M2_NGROUPS, M2_CHUNK, M2_HEADDIM
    CD = d_inner + 2 * G * N
    d_in_proj = 2 * d_inner + 2 * G * N + H
    ip = r["in_proj.out"].reshape(Bsz, L, d_in_proj)
    dt_raw = ip[..., d_in_proj - H:]
    xBC = r["silu.out"].reshape(Bsz, L, CD)
    Xh = xBC[..., :d_inner].reshape(Bsz, L, H, PP)
    B4 = xBC[..., d_inner:d_inner + G * N].reshape(Bsz, L, G, N)
    C4 = xBC[..., d_inner + G * N:].reshape(Bsz, L, G, N)
    A = r["A.out"]
    D = p["D"].to(dtype)
    dt_bias = p["dt_bias"].to(dtype)
    init = p["initial_states"].to(dtype) if "initial_states" in p else None
    with _scan_ref_dtype(dtype):
        hf_out, hf_final = mamba2_chunk_scan(
            Xh, dt_raw, A, B4, C4, chunk_size=Q, D=D, dt_bias=dt_bias,
            initial_states=init, dt_softplus=True, dt_limit=dt_limit,
            return_final_states=True)
    dt = r["dt.out"].reshape(Bsz, L, H)
    Xd = r["xd.out"].reshape(Bsz, L, H, PP)
    dA = A * dt
    ps = (Q - L % Q) % Q
    Bh = B4.repeat_interleave(H // G, dim=2, output_size=H)
    Ch = C4.repeat_interleave(H // G, dim=2, output_size=H)
    ssd_y, ssd_final = ssd_minimal_discrete(
        pad_tensor_by_size(Xd, ps), pad_tensor_by_size(dA, ps),
        pad_tensor_by_size(Bh, ps), pad_tensor_by_size(Ch, ps),
        Q, initial_states=(init[:, None] if init is not None else None))
    return hf_out, hf_final, ssd_y[:, :L], ssd_final


# --------------------------------------------------------------------------
# Mamba-2 tensor materialization, emission, manifests
# --------------------------------------------------------------------------
def f32_hex(v):
    return f"0x{int.from_bytes(np.float32(v).tobytes(), 'little'):08X}"


def m2_effective_dt_limit(case):
    """The case's dt_limit rounded through float32 ONCE (the FP32 profile
    clamps with float32 limits; the float64 reference uses the exact doubles
    of those float32 values, so both runs see the same bounds)."""
    lo, hi = case.get("dt_limit", (0.0, M2_INF))
    lo = float(np.float32(lo))
    hi = M2_INF if math.isinf(hi) else float(np.float32(hi))
    return (lo, hi)


def m2_dt_limit_meta(case):
    lo, hi = m2_effective_dt_limit(case)
    return [
        dict(value=("inf" if math.isinf(lo) else lo), f32_hex=f32_hex(lo)),
        dict(value=("inf" if math.isinf(hi) else hi), f32_hex=f32_hex(hi)),
    ]


def m2_gen_tensor(seed, name, shape, lo, hi, segments=None):
    n = int(np.prod(shape))
    f = hashed_unit(seed, M2_TENSOR_IDS[name], n)
    v = map_range(f, lo, hi)
    if segments is not None:
        arr = v.reshape(shape)
        for seg_name, slo, shi in segments:
            if seg_name == "z_rows":
                # rows 0..d_inner of in_proj.weight are the z block
                # (zxbcdt order z | xBC | dt, mamba2.py:211-215; d_inner =
                # expand * d_model = 2 * n_cols)
                rows = M2_EXPAND * shape[1]
                fseg = f.reshape(shape)[:rows].reshape(-1)
                arr[:rows] = map_range(fseg, slo, shi).reshape(arr[:rows].shape)
            else:
                raise ValueError(seg_name)
        v = arr.reshape(-1)
    return v.reshape(shape)


def m2_build_case(case):
    k = m2_case_index(case["name"])
    d_model = case["d_model"]
    d_inner = M2_EXPAND * d_model
    H = d_inner // M2_HEADDIM
    B, L = case["B"], case["L"]
    src = case.get("x_source")
    seed = m2_case_seed(m2_case_index(src[0])) if src else m2_case_seed(k)
    shapes = m2_shapes_for(d_model, B, L, with_init=case.get("init_states", False))
    ranges = m2_default_ranges(d_model)
    ranges.update(case.get("overrides", {}))
    segments = case.get("segments", {})
    tensors = {}
    for name, shape in shapes.items():
        if name == "x" and src:
            parent = m2_case_by_name(src[0])
            pshape = m2_shapes_for(d_model, parent["B"], parent["L"])["x"]
            lo, hi = ranges["x"]
            px = m2_gen_tensor(seed, "x", pshape, lo, hi)
            tensors["x"] = px[src[1]:src[1] + 1].copy()
            continue
        lo, hi = ranges[name]
        tensors[name] = m2_gen_tensor(seed, name, shape, lo, hi, segments.get(name))
    if "zero_rule" in case:
        tensors["x"] = apply_zero_rule(tensors["x"], case["zero_rule"])
    meta = dict(
        name=case["name"], family="mamba2", corpus=M2_CORPUS_VERSION,
        profile=M2_PROFILE, hash_spec=HASH_SPEC,
        seed=f"0x{seed:016X}", seed_case_index=(m2_case_index(src[0]) if src else k),
        B=B, L=L, d_model=d_model, d_inner=d_inner, nheads=H,
        headdim=M2_HEADDIM, d_state=M2_D_STATE, ngroups=M2_NGROUPS,
        d_conv=M2_D_CONV, expand=M2_EXPAND, chunk_size=M2_CHUNK,
        nchunks=-(-L // M2_CHUNK), rms_eps=M2_EPS, softplus_threshold=20.0,
        dt_limit=m2_dt_limit_meta(case),
        tensors={name: dict(shape=shapes[name], dtype="float32", order="row-major",
                            tensor_id=M2_TENSOR_IDS[name], lo=ranges[name][0], hi=ranges[name][1],
                            file=f"{name}.f32",
                            **({"segments": [dict(rows=s[0], lo=s[1], hi=s[2]) for s in segments[name]]}
                               if name in segments else {}))
                 for name in shapes},
        note=case["note"],
    )
    if src:
        meta["x_source"] = dict(case=src[0], batch_row=src[1])
        meta["tensors"]["x"]["derived"] = f"byte slice of {src[0]}/x.f32, batch row {src[1]}"
    if "zero_rule" in case:
        meta["zero_rule"] = case["zero_rule"]
        meta["tensors"]["x"]["derived"] = "hashed, then zero_rule applied"
    return tensors, meta


def m2_stage_shapes(meta):
    B, L, dm = meta["B"], meta["L"], meta["d_model"]
    d_inner, H, C = meta["d_inner"], meta["nheads"], meta["nchunks"]
    M = B * L
    Q, N, PP, G = M2_CHUNK, M2_D_STATE, M2_HEADDIM, M2_NGROUPS
    CD = d_inner + 2 * G * N
    d_in_proj = 2 * d_inner + 2 * G * N + H
    return {
        "input.x": [M, dm], "norm.sumsq": [M], "norm.out": [M, dm],
        "in_proj.out": [M, d_in_proj], "A.out": [H],
        "conv.out": [M, CD], "silu.out": [M, CD], "conv.window": [B, CD, M2_D_CONV],
        "dt.out": [M, H], "xd.out": [M, H, PP],
        "dacs.out": [B, H, C, Q], "seg.L": [B, C, H, Q, Q], "cb.G": [B, C, G, Q, Q],
        "ydiag.out": [M, H, PP], "decay.states": [B, H, C, Q],
        "cstate.out": [B, C, H, PP, N], "pass.states": [B, C, H, PP, N],
        "yoff.out": [M, H, PP], "scan.y": [M, H, PP], "skip.out": [M, H, PP],
        "gnorm.gate": [M, d_inner], "gnorm.sumsq": [M], "gnorm.out": [M, d_inner],
        "out_proj.out": [M, dm], "residual.out": [M, dm],
        "ssd.h_last": [B, H, PP, N],
    }


def m2_write_raw(path, arr, dtype):
    write_raw(path, arr, dtype)
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def m2_emit_case(case, out_root):
    if case.get("kind") == "decode":
        return m2_emit_decode_case(case, out_root)
    tensors, meta = m2_build_case(case)
    stages = list(case.get("stages", M2_FULL_STAGES))
    lim = m2_effective_dt_limit(case)
    cdir = os.path.join(out_root, case["name"])
    os.makedirs(os.path.join(cdir, "ref64"), exist_ok=True)
    os.makedirs(os.path.join(cdir, "ref32"), exist_ok=True)
    for name, arr in tensors.items():
        meta["tensors"][name]["sha256"] = m2_write_raw(os.path.join(cdir, f"{name}.f32"), arr, np.float32)
    p = {k: torch.from_numpy(v.copy()) for k, v in tensors.items() if k != "x"}
    x = torch.from_numpy(tensors["x"].copy())
    r64 = m2_forward(p, x, torch.float64, dt_limit=lim)
    r32 = m2_forward(p, x, torch.float32, dt_limit=lim)
    shapes = m2_stage_shapes(meta)
    stage_meta = {}
    for s in stages:
        shp = shapes[s]
        assert list(r64[s].shape) == shp, (s, tuple(r64[s].shape), shp)
        sha64 = m2_write_raw(os.path.join(cdir, "ref64", f"{s}.f64"), r64[s].numpy(), np.float64)
        sha32 = m2_write_raw(os.path.join(cdir, "ref32", f"{s}.f32"), r32[s].numpy(), np.float32)
        stage_meta[s] = dict(shape=shp, order="row-major", ref64=f"ref64/{s}.f64",
                             ref32=f"ref32/{s}.f32", sha256_ref64=sha64, sha256_ref32=sha32)
    meta["stages"] = stage_meta
    meta["stage_order"] = stages
    elided = [s for s in ("seg.L", "cb.G") if s not in stages]
    if elided:
        meta["elided_stages"] = dict(
            stages=elided,
            reason="size cap (contract section 7: the elision is stated, never silent); "
                   "recorded on m2_base_b1_l1_d32 and m2_base_b2_l4_d32")
    meta["stage_definitions"] = "see ../manifest.json"
    with open(os.path.join(cdir, "manifest.json"), "w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return dict(meta=meta, r64=r64, r32=r32, p=p, x=x)


# The decode-continuation case: contract section 5's three-piece resumption
# state and DEVIATION 786 (decode IS prefill resumption, so a prefill of
# L+1 tokens IS the decode step's reference).
M2_DECODE_PREFILL_STAGES = [
    "dt.out", "dacs.out", "pass.states", "scan.y", "residual.out",
    "conv.window", "ssd.h_last",
]
M2_DECODE_TOKEN_STAGES = [
    "norm.out", "in_proj.out", "conv.out", "silu.out", "dt.out", "xd.out",
    "ydiag.out", "yoff.out", "scan.y", "skip.out", "gnorm.gate",
    "gnorm.sumsq", "gnorm.out", "out_proj.out", "residual.out",
]


def m2_emit_decode_case(case, out_root):
    assert case["B"] == 1, "decode case is B=1"
    tensors, meta = m2_build_case(case)  # L = prefill_len + 1
    Lp = case["prefill_len"]
    L = case["L"]
    assert L == Lp + 1
    q = Lp % M2_CHUNK  # open-chunk buffer rows at the handoff
    d_inner, H = meta["d_inner"], meta["nheads"]
    CD = d_inner + 2 * M2_NGROUPS * M2_D_STATE
    d_in_proj = 2 * d_inner + 2 * M2_NGROUPS * M2_D_STATE + H
    lim = m2_effective_dt_limit(case)
    cdir = os.path.join(out_root, case["name"])
    os.makedirs(os.path.join(cdir, "ref64"), exist_ok=True)
    os.makedirs(os.path.join(cdir, "ref32"), exist_ok=True)
    for name, arr in tensors.items():
        meta["tensors"][name]["sha256"] = m2_write_raw(os.path.join(cdir, f"{name}.f32"), arr, np.float32)
    p = {k: torch.from_numpy(v.copy()) for k, v in tensors.items() if k != "x"}
    x = torch.from_numpy(tensors["x"].copy())
    pre64 = m2_forward(p, x[:, :Lp], torch.float64, dt_limit=lim)
    pre32 = m2_forward(p, x[:, :Lp], torch.float32, dt_limit=lim)
    full64 = m2_forward(p, x, torch.float64, dt_limit=lim)
    full32 = m2_forward(p, x, torch.float32, dt_limit=lim)

    recs = []  # (file stem, tensor64, tensor32)
    # (i) the prefill call's record (state of the world after token Lp)
    for s in M2_DECODE_PREFILL_STAGES:
        recs.append((f"prefill.{s}", pre64[s], pre32[s]))
    # (ii) the three-piece resumption state (contract section 5):
    # 1. conv window = prefill.conv.window above (m2 :220-221);
    # 2. the chunk-BOUNDARY state h = the S17 value entering the OPEN chunk
    #    (pass.states[:, -1] of the prefill call). NOTE it differs from
    #    prefill.ssd.h_last, which is the REPORT stage after the final
    #    PADDED chunk -- section 5's distinction, on disk;
    recs.append(("state.h_boundary", pre64["pass.states"][:, -1], pre32["pass.states"][:, -1]))
    # 3. the intra-chunk buffer: the open chunk's post-conv/post-SiLU xBC
    #    rows and raw dt rows (q rows of each).
    recs.append(("state.buffer_xbc",
                 pre64["silu.out"].reshape(1, Lp, CD)[:, Lp - q:],
                 pre32["silu.out"].reshape(1, Lp, CD)[:, Lp - q:]))
    recs.append(("state.buffer_dtraw",
                 pre64["in_proj.out"].reshape(1, Lp, d_in_proj)[:, Lp - q:, d_in_proj - H:],
                 pre32["in_proj.out"].reshape(1, Lp, d_in_proj)[:, Lp - q:, d_in_proj - H:]))
    # (iii) the decode token's per-stage rows = token Lp of the L-token
    # prefill (DEVIATION 786), plus the post-decode state.
    for s in M2_DECODE_TOKEN_STAGES:
        recs.append((f"decode.{s}", full64[s][Lp:Lp + 1], full32[s][Lp:Lp + 1]))
    recs.append(("decode.conv.window", full64["conv.window"], full32["conv.window"]))
    recs.append(("decode.ssd.h_last", full64["ssd.h_last"], full32["ssd.h_last"]))
    recs.append(("decode.pass.states", full64["pass.states"], full32["pass.states"]))

    stage_meta = {}
    order = []
    for stem, t64, t32 in recs:
        sha64 = m2_write_raw(os.path.join(cdir, "ref64", f"{stem}.f64"), t64.numpy(), np.float64)
        sha32 = m2_write_raw(os.path.join(cdir, "ref32", f"{stem}.f32"), t32.numpy(), np.float32)
        stage_meta[stem] = dict(shape=list(t64.shape), order="row-major",
                                ref64=f"ref64/{stem}.f64", ref32=f"ref32/{stem}.f32",
                                sha256_ref64=sha64, sha256_ref32=sha32)
        order.append(stem)
    meta["stages"] = stage_meta
    meta["stage_order"] = order
    meta["decode"] = dict(
        prefill_len=Lp,
        decode_token_index=Lp,
        open_chunk_rows=q,
        completed_chunks=Lp // M2_CHUNK,
        semantics=(
            "DEVIATION 786: decode is prefill resumption. prefill.* stages are the "
            f"{Lp}-token call; state.* is the three-piece resumption state after it "
            "(conv window; h entering the open chunk = pass.states[:, -1], NOT ssd.h_last, "
            "which is the after-the-padded-final-chunk REPORT; the open chunk's post-conv/"
            "post-SiLU xBC rows and raw in_proj dt rows); decode.* stages are token "
            f"{Lp}'s rows of the {L}-token prefill, which the profile's decode step must "
            "reproduce bit for bit against its own oracle (this corpus stays a tolerance "
            "reference)."),
    )
    meta["stage_definitions"] = "see ../manifest.json"
    with open(os.path.join(cdir, "manifest.json"), "w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return dict(meta=meta, r64=full64, r32=full32, pre64=pre64, pre32=pre32, p=p, x=x)


M2_STAGE_DEFS = {
    "input.x": "the block input, as given, token-major  [M, d_model]",
    "norm.sumsq": "S1: per-row sum of squares (HF Mamba2RMSNorm :603 spells mean(-1); recorded as the fold's SUM, the card's decomposition)  [M]",
    "norm.out": "S2-S3: x * (1/sqrt(sumsq/d_model + 1e-5)) * weight (rstd per rms_norm_ref:29, the mamba_ssm spelling; order HF :604-605)  [M, d_model]",
    "in_proj.out": "S4: norm.out @ in_proj.weight^T; columns z | xBC | dt_raw (mamba2.py:211-215, d_mlp = 0)  [M, 2*d_inner + 2*G*N + H]",
    "A.out": "S5: -exp(A_log) (mamba2.py:182)  [H]",
    "conv.out": "S6: causal depthwise conv1d over raw xBC, kernel 4, padding 3, truncated to L, bias included (mamba2.py:233)  [M, CD]",
    "silu.out": "S7: silu(conv.out) (mamba2.py:232); columns x | B | C split by copy (:243)  [M, CD]",
    "conv.window": "the conv window AFTER the call: last d_conv raw xBC columns per channel, zeros before the first token (mamba2.py:220-221, allocate_inference_cache:348-350)  [B, CD, 4]",
    "dt.out": "S9: clamp(softplus(dt_raw + dt_bias), dt_limit), in that order (_chunk_cumsum_fwd_kernel:73-81; HF :272-276; torch softplus threshold 20 == the kernel's <= 20 guard)  [M, H]",
    "xd.out": "S10: x * dt, DISCRETIZE FIRST (HF :288; ssd_minimal.py:103 composition; DEVIATION 789: dt binds to x and A, never to B)  [M, H, P]",
    "dacs.out": "S11: per-chunk cumsum of dA = A * dt (HF :289, :294-295); padded positions carry the last real value  [B, H, C, Q]",
    "seg.L": "S12: exp(segsum), +0.0 above the diagonal (HF :299 / ssd_minimal.py:54; exp(-inf) = +0.0 = the profile's structural zero, DEVIATION 782)  [B, C, H, Q, Q]",
    "cb.G": "S12: C.B contracted over n (HF :302); one group row at ngroups = 1 (heads are bitwise copies)  [B, C, G, Q, Q]",
    "ydiag.out": "S13-S14: (G (.) L) . X_d over chunk positions (HF :305-308), truncated to L  [M, H, P]",
    "decay.states": "S15: exp(dA_cs_last - dA_cs) (HF :312)  [B, H, C, Q]",
    "cstate.out": "S15-S16: (B (.) decay)^T . X_d per (b, chunk) (HF :313-314)  [B, C, H, P, N]",
    "pass.states": "S17: the state ENTERING each chunk (h_{c-1}); reference spelling is the decay-matrix pass (HF :318-326 / ssd_minimal.py:64-69); the PROFILE pins the serial fma recurrence (DEVIATION 785) and this float64 reference absorbs the difference in tolerance  [B, C, H, P, N]",
    "yoff.out": "S18: (C . h_prev) * exp(dA_cs), contract over n FIRST, scale AFTER (HF :330-332), truncated  [M, H, P]",
    "scan.y": "S19: Y_diag + Y_off (HF :335), truncated  [M, H, P]",
    "skip.out": "S20: Y + x * D[h], D from UNDISCRETIZED post-conv x, added LAST (HF :284-285, :338-339)  [M, H, P]",
    "gnorm.gate": "S21: y * silu(z), gate BEFORE norm (rms_norm_ref:26-27, norm_before_gate=False)  [M, d_ssm]",
    "gnorm.sumsq": "S21: the gated row's sum-of-squares fold  [M]",
    "gnorm.out": "S21: gate * (1/sqrt(sumsq/d_ssm + 1e-5)) * norm.weight (rms_norm_ref:29-30; eps mamba2.py:144; group_size = whole row at G = 1, DEVIATION 787)  [M, d_ssm]",
    "out_proj.out": "S4: gnorm.out @ out_proj.weight^T (mamba2.py:275)  [M, d_model]",
    "residual.out": "S22: residual + out_proj.out (HF Mamba2Block :630)  [M, d_model]",
    "ssd.h_last": "section 5's REPORT stage: the S17 value after the final (padded) chunk (ssd_minimal.py:69; mamba2.py:261-264); NOT the resumption state off a chunk boundary  [B, H, P, N]",
}


def m2_generate(out_root, verify=False):
    m2_root = os.path.join(out_root, "mamba2")
    os.makedirs(m2_root, exist_ok=True)
    top = dict(
        family="mamba2", corpus=M2_CORPUS_VERSION, profile=M2_PROFILE,
        contract="mamba/IDENTICAL_MAMBA2_CONTRACT.md",
        hash_spec=HASH_SPEC, seed_base=f"0x{M2_SEED_BASE:016X}",
        seed_rule="seed_k = seed_base + 0x1000 * k, k = case index below; x_source cases reuse the parent's seed",
        d_state=M2_D_STATE, d_conv=M2_D_CONV, expand=M2_EXPAND,
        headdim=M2_HEADDIM, ngroups=M2_NGROUPS, chunk_size=M2_CHUNK,
        rms_eps=M2_EPS, softplus_threshold=20.0,
        upstream=dict(
            mamba=("https://github.com/state-spaces/mamba @ e9594ce1c732d97440f0332fdc43170a2294dbfa, "
                   "mamba_ssm/modules/ssd_minimal.py::segsum L23-32, ::ssd_minimal_discrete L34-78 "
                   "(verbatim, NORMATIVE, composed discretize-first per L94-103); "
                   "mamba_ssm/modules/mamba2.py (block order, non-mem-eff arm L209-276); "
                   "mamba_ssm/ops/triton/ssd_chunk_state.py::_chunk_cumsum_fwd_kernel L72-86 (dt seam); "
                   "mamba_ssm/ops/triton/layernorm_gated.py::rms_norm_ref L18-39 (gated norm)"),
            transformers=("https://github.com/huggingface/transformers @ d56c55bf564ddb176759eb6ec199442682564916, "
                          "src/transformers/models/mamba2/modeling_mamba2.py::pad_tensor_by_size L42-50, "
                          "::reshape_into_chunks L53-70, ::segment_sum L73-90, ::mamba2_chunk_scan L254-348 "
                          "(verbatim), ::Mamba2RMSNorm L591-605, ::Mamba2Block.forward L617-631"),
        ),
        tensor_ids=M2_TENSOR_IDS, stage_definitions=M2_STAGE_DEFS, cases=[])
    total_bytes = 0
    all_meta = {}
    for k, case in enumerate(M2_CASES):
        entry = m2_emit_case(case, m2_root)
        all_meta[case["name"]] = entry
        meta = entry["meta"]
        row = dict(index=k, name=case["name"], seed=meta["seed"], B=case["B"], L=case["L"],
                   d_model=case["d_model"], d_inner=meta["d_inner"], nheads=meta["nheads"],
                   nchunks=meta["nchunks"], stages=meta["stage_order"], note=case["note"])
        if case.get("kind") == "decode":
            row["kind"] = "decode"
            row["prefill_len"] = case["prefill_len"]
        top["cases"].append(row)
        cdir = os.path.join(m2_root, case["name"])
        total_bytes += sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fn in os.walk(cdir) for f in fn)
    with open(os.path.join(m2_root, "manifest.json"), "w") as fh:
        json.dump(top, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"wrote {len(M2_CASES)} mamba2 cases to {m2_root}, {total_bytes / 1e6:.3f} MB of tensor and manifest files")
    print(f"mamba2 corpus sha256 (files .f32 .f64 .json): {sha256_of_dir(m2_root)}")
    if verify:
        m2_run_verify(m2_root, all_meta)


def m2_run_verify(m2_root, all_meta):
    print("\n== m2 verify 0: root input tensors on disk (the mamba2_check corpus gate opens "
          "mamba/corpus/mamba2/<case>/x.f32 RELATIVE TO THE REPO ROOT) ==")
    # 2026-09-01 incident: the check's gate_corpus refused MISSING FIXTURE
    # for a case whose x.f32 was present and committed at exactly that path.
    # Its try/except reports EVERY open/read failure as "missing" -- a wrong
    # working directory, or Mojo's TEXT-mode open("r") + read() over raw f32
    # bytes (not UTF-8), both refuse with the file on disk. This arm is the
    # corpus side's proof: every manifest-listed root tensor file exists and
    # x.f32 is byte-identical to the hashed derivation the refs were
    # computed from. HARD-asserted, not just printed.
    for name, entry in all_meta.items():
        cdir = os.path.join(m2_root, name)
        missing = [tm["file"] for tm in entry["meta"]["tensors"].values()
                   if not os.path.exists(os.path.join(cdir, tm["file"]))]
        assert not missing, (name, "missing root tensor files", missing)
        with open(os.path.join(cdir, "x.f32"), "rb") as fh:
            got = fh.read()
        want = np.ascontiguousarray(entry["x"].numpy().astype("<f4")).tobytes()
        assert got == want, (name, "x.f32 bytes differ from the hashed derivation")
        print(f"  {name:34s} {len(entry['meta']['tensors'])} root tensors present; "
              f"x.f32 {len(got)} bytes == hashed derivation")

    print("\n== m2 verify 1: verbatim HF mamba2_chunk_scan vs the composed stages (expect BIT-equal) ==")
    print("== m2 verify 2: verbatim mamba_ssm ssd_minimal_discrete, discretize-first (independent spelling, roundoff-scale) ==")
    worst_ssd = 0.0
    for name, entry in all_meta.items():
        case = m2_case_by_name(name)
        meta, p = entry["meta"], entry["p"]
        lim = m2_effective_dt_limit(case)
        M = meta["B"] * meta["L"]
        H, PP = meta["nheads"], M2_HEADDIM
        for dtype, r in ((torch.float64, entry["r64"]), (torch.float32, entry["r32"])):
            hf_out, hf_fin, ssd_y, ssd_fin = m2_cross_checks(meta, r, p, dtype, lim)
            hf_skip = hf_out.reshape(M, H, PP)
            eq_hf = torch.equal(hf_skip, r["skip.out"]) and torch.equal(hf_fin, r["ssd.h_last"])
            d_hf = max(float((hf_skip - r["skip.out"]).abs().max()),
                       float((hf_fin - r["ssd.h_last"]).abs().max()))
            ssd_ys = ssd_y.reshape(M, H, PP)
            eq_ssd = torch.equal(ssd_ys, r["scan.y"]) and torch.equal(ssd_fin, r["ssd.h_last"])
            d_ssd = max(float((ssd_ys - r["scan.y"]).abs().max()),
                        float((ssd_fin - r["ssd.h_last"]).abs().max()))
            worst_ssd = max(worst_ssd, d_ssd if d_ssd == d_ssd else 0.0)
            print(f"  {name:34s} {str(dtype):14s} hf bit-equal={eq_hf} maxabs={d_hf:.3e} | "
                  f"ssd_minimal bit-equal={eq_ssd} maxabs={d_ssd:.3e}")
    print(f"  worst ssd_minimal abs difference (both dtypes): {worst_ssd:.3e}")

    print("\n== m2 verify 3: batch composition (comp rows vs parent, ref64 and ref32) ==")
    pm = all_meta["m2_comp_b2_l257_d32"]
    Lc = 257
    m_lead = ("input.x", "dt.out", "scan.y", "residual.out")
    b_lead = ("dacs.out", "decay.states", "cstate.out", "pass.states", "ssd.h_last")
    for b in (0, 1):
        cm = all_meta[f"m2_comp_row{b}_b1_l257_d32"]
        for tag, key in (("ref64", "r64"), ("ref32", "r32")):
            for s in M2_REDUCED_STAGES:
                if s in m_lead:
                    a = pm[key][s][b * Lc:(b + 1) * Lc]
                elif s in b_lead:
                    a = pm[key][s][b:b + 1]
                else:
                    continue
                c = cm[key][s]
                print(f"  row{b} {tag} {s:14s} bit-equal={torch.equal(a, c)} maxabs={float((a - c).abs().max()):.3e}")
    # negative control: the two composed rows must DIFFER (a broken slicer
    # returning row 0 twice would pass the equality rows forever)
    r0 = all_meta["m2_comp_row0_b1_l257_d32"]["r64"]["scan.y"]
    r1 = all_meta["m2_comp_row1_b1_l257_d32"]["r64"]["scan.y"]
    print(f"  negative control: row0 vs row1 scan.y DIFFER = {not torch.equal(r0, r1)} (must be True)")

    print("\n== m2 verify 4: determinism (regenerate into a temp dir, byte-compare) ==")
    tmp = tempfile.mkdtemp(prefix="mamba2-corpus-")
    try:
        for case in M2_CASES:
            m2_emit_case(case, tmp)
        mism = []
        for dp, _, fn in os.walk(tmp):
            for f in fn:
                a = os.path.join(dp, f)
                bpath = os.path.join(m2_root, os.path.relpath(a, tmp))
                with open(a, "rb") as fa, open(bpath, "rb") as fb:
                    if fa.read() != fb.read():
                        mism.append(os.path.relpath(a, tmp))
        print(f"  files compared: {sum(len(fn) for _, _, fn in os.walk(tmp))}, mismatches: {len(mism)}")
        for m in mism:
            print("   MISMATCH", m)
    finally:
        shutil.rmtree(tmp)

    print("\n== m2 verify 5: adversarial and witness intent reached ==")
    # softplus band [8, 14]
    e = all_meta["m2_adv_softplus_band_b2_l8_d32"]
    meta = e["meta"]
    dip = 2 * meta["d_inner"] + 2 * M2_NGROUPS * M2_D_STATE + meta["nheads"]
    biased = (e["r64"]["in_proj.out"][:, dip - meta["nheads"]:]
              + e["p"]["dt_bias"].double())
    delta = torch.log1p(torch.exp(-biased))
    print(f"  softplus band: biased dt min={float(biased.min()):.4f} max={float(biased.max()):.4f} "
          f"in [8,14]: {int(((biased >= 8) & (biased <= 14)).sum())} of {biased.numel()}; "
          f"softplus-x delta in [{float(delta.min()):.3e}, {float(delta.max()):.3e}]")
    # A near zero
    e = all_meta["m2_adv_a_near_zero_b3_l64_d32"]
    dA32 = (e["r32"]["A.out"] * e["r32"]["dt.out"]).float()
    ddec = torch.exp(dA32)
    print(f"  A near zero: exp(dt*A) float32 exactly 1: {int((ddec == 1).sum())} of {ddec.numel()}, "
          f"max distance from 1 = {float(((1.0 - ddec) / np.float32(2 ** -24)).max()):.1f} ulp(1-)")
    # signed zeros
    e = all_meta["m2_adv_signed_zeros_b2_l8_d32"]
    xn = e["x"].numpy()
    ip64 = e["r64"]["in_proj.out"].numpy()
    ip32 = e["r32"]["in_proj.out"].numpy()
    print(f"  signed zeros: x has {int((xn == 0).sum())} zeros of which {int(np.signbit(xn[xn == 0]).sum())} are -0.0; "
          f"ref64 in_proj.out -0.0 count={int(np.signbit(ip64[ip64 == 0]).sum())}, "
          f"ref32 in_proj.out -0.0 count={int(np.signbit(ip32[ip32 == 0]).sum())}")
    # gate saturation
    e = all_meta["m2_adv_gate_saturation_b1_l8_d64"]
    meta = e["meta"]
    z = e["r32"]["in_proj.out"][:, :meta["d_inner"]]
    sig = torch.sigmoid(z)
    print(f"  gate saturation: |z| min={float(z.abs().min()):.3e} max={float(z.abs().max()):.3e}; "
          f"sigmoid exactly 0: {int((sig == 0).sum())}, exactly 1: {int((sig == 1).sum())} of {sig.numel()}")
    # active dt_limit: both limits must bind
    e = all_meta["m2_adv_dt_limit_b2_l8_d32"]
    case = m2_case_by_name("m2_adv_dt_limit_b2_l8_d32")
    lo, hi = m2_effective_dt_limit(case)
    meta = e["meta"]
    dip = 2 * meta["d_inner"] + 2 * M2_NGROUPS * M2_D_STATE + meta["nheads"]
    pre = F.softplus(e["r64"]["in_proj.out"][:, dip - meta["nheads"]:]
                     + e["p"]["dt_bias"].double())
    print(f"  active dt_limit: pre-clamp dt in [{float(pre.min()):.3e}, {float(pre.max()):.3e}]; "
          f"clamped at lo: {int((pre < lo).sum())}, at hi: {int((pre > hi).sum())} of {pre.numel()} "
          f"(both counts must be nonzero)")
    # STATEPASS witness liveness: chunk decays must be normal FP32 numbers
    for nm in ("m2_statepass_b1_l513_d32", "m2_statepass_b2_l770_d64", "m2_init_states_b1_l257_d32"):
        e = all_meta[nm]
        scale = torch.exp(e["r32"]["dacs.out"][..., -1])  # exp(dA_cs_last) per (b, h, chunk)
        print(f"  {nm}: exp(dA_cs_last) float32 min={float(scale.min()):.3e} max={float(scale.max()):.3e} "
              f"zeros={int((scale == 0).sum())} of {scale.numel()} (zeros must be 0 or the STATEPASS arms are inert)")

    print("\n== m2 verify 6: decode = prefill resumption (prefix property of the reference) ==")
    e = all_meta["m2_decode_b1_l260p1_d32"]
    Lp = m2_case_by_name("m2_decode_b1_l260p1_d32")["prefill_len"]
    for tag, pre_key, full_key in (("ref64", "pre64", "r64"), ("ref32", "pre32", "r32")):
        pre, full = e[pre_key], e[full_key]
        for s in ("scan.y", "residual.out", "dt.out"):
            a, c = pre[s], full[s][:Lp]
            print(f"  {tag} prefix {s:14s} bit-equal={torch.equal(a, c)} maxabs={float((a - c).abs().max()):.3e}")
        hb_pre = pre["pass.states"][:, -1]
        hb_full = full["pass.states"][:, -1]
        print(f"  {tag} h_boundary (pass.states[:, -1]) prefill-vs-full bit-equal={torch.equal(hb_pre, hb_full)}")
        print(f"  {tag} h_boundary != ssd.h_last (section 5's distinction) = "
              f"{not torch.equal(hb_pre, pre['ssd.h_last'])} (must be True)")


# ==========================================================================
# ==========================================================================
#                    MAMBA-3 (SISO) CORPUS EXTENSION
#
# Profile: mojolearn.identical.mamba3.siso.fp32.v1
# (mamba/IDENTICAL_MAMBA3_CONTRACT.md; section 8g is the clause this section
# discharges). Same discipline as the two families above: every input element
# is a hashed value under mojolearn.mamba.corpus.hash.v1 -- NEW seed base,
# NEW tensor names and ids (41-54) -- torch float64 per-stage references,
# float32 run informative only, nothing here a bitwise certificate of
# anything. Cases land in mamba/corpus/mamba3/<case>/ with their own
# top-level manifest; the Mamba-1 and Mamba-2 corpora are byte-untouched.
#
# THE CASE TABLE IS NOT THIS FILE'S TO INVENT. Unlike the two families
# above, the normative table landed FIRST, in Mojo:
# `mamba/checks/mamba3_fixture.mojo` (18 cases, seed base 0x4D6D6233436F7270
# "Mmb3Corp", ids 41-54, the plants of its module docstring). This section
# MIRRORS that table index for index, plant for plant, and the byte gate in
# `mamba/checks/mamba3_check.mojo` (`pixi run check-mamba3-block -- corpus`
# family arm: file bytes == the in-check generator's bytes) is the arbiter
# if the two tables ever drift.
#
# References (checkout /Users/andrewhendel/CascadeProjects/upstream/mamba @
# e9594ce1c732d97440f0332fdc43170a2294dbfa; there is NO second repository --
# HF transformers @ d56c55b has no mamba3 model, said out loud in contract
# section 0):
#   tests/ops/triton/test_mamba3_siso.py::_segsum (:22-31),
#     ::mamba3_siso_step_ref (:34-146), ::mamba3_siso_fwd_ref (:149-340)
#     -- the NORMATIVE math references, copied VERBATIM below;
#   mamba_ssm/modules/mamba3.py::heavy_tail_activation (:27-41),
#     ::Mamba3.forward (:160-278), ::__init__ defaults (:44-70), in_proj
#     layout (:106-107), B/C biases (:121-122), B/C norms eps 1e-5
#     (:126-127) -- block order and constants;
#   mamba_ssm/ops/triton/layernorm_gated.py::rms_norm_ref (:18-39) at
#     z=None, group_size=None -- the B/C norm spelling (rstd spelled
#     `1 / torch.sqrt(...)`, never torch.rsqrt);
#   mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py -- the chunked schedule
#     SHAPE the staged reference follows (phase structure; the contract's
#     citations).
#
# THE STAGED REFERENCE FOLLOWS THE PROFILE'S CHUNKED SCHEDULE, NOT THE
# UNCHUNKED REFERENCE'S (contract DEVIATION 827): per-chunk intra/state
# terms, a SERIAL inter-chunk pass, Q = CHUNK_SIZE = 64; the kernel's
# diagonal spelling (strict mask + gamma-scaled pre-rotation qk dot,
# DEVIATION 830); the per-token serial mod-2pi angle recurrence (DEVIATION
# 829). The verbatim `mamba3_siso_fwd_ref` computes ONE whole-sequence
# quadratic attention with the include-then-subtract diagonal and the
# mod-at-the-end angle placement -- equal in EXACT arithmetic, different
# bits -- so verify arm 1 reports its roundoff-scale distance and demands
# bit-equality of NOTHING (the mamba2 ssd_minimal arm's posture, not its
# HF arm's). Verify arm 2 does the same against `mamba3_siso_step_ref`'s
# per-token recurrence. A float64 tolerance reference absorbs all three
# spellings' differences, which is exactly why the corpus is never a
# bitwise oracle.
# ==========================================================================

M3_PROFILE = "mojolearn.identical.mamba3.siso.fp32.v1"
M3_CORPUS_VERSION = "mamba3-corpus-v1"
M3_SEED_BASE = 0x4D6D6233436F7270  # "Mmb3Corp", mamba3_fixture.mojo
M3_D_STATE = 128        # N, the QK head dim, mamba3.py:47
M3_EXPAND = 2           # mamba3.py:48
M3_HEADDIM = 64         # P, the V head dim, mamba3.py:49
M3_NGROUPS = 1          # G = num_bc_heads, mamba3.py:50
M3_CHUNK = 64           # Q, mamba3.py:64 -- PART OF THE ARITHMETIC (DEV 827/783)
M3_ROPE = 32            # R = num_rope_angles at rope_fraction 0.5, mamba3.py:98-103
M3_EPS = 1e-5           # block norm AND B/C norms, mamba3.py:126-127
M3_A_FLOOR = 1e-4       # mamba3.py:57, the S5 clamp bound

# Tensor names/ids: mamba3_fixture.mojo's M3_TID_* table verbatim. File
# names are `<name>.f32` at case root, exactly what
# mamba3_check.mojo::gate_corpus opens.
M3_TENSOR_IDS = {
    "x": 41,
    "block_norm.weight": 42,
    "in_proj.weight": 43,
    "dt_bias": 44,
    "B_norm.weight": 45,
    "C_norm.weight": 46,
    "B_bias": 47,
    "C_bias": 48,
    "D": 49,
    "out_proj.weight": 50,
    "init_theta": 51,
    "init_h": 52,
    "init_k": 53,
    "init_v": 54,
}


def m3_dims(d_model):
    """d_inner, nheads, d_in_proj (mamba3.py:92-107; column order
    z | x | B | C | dd_dt | dd_A | trap | angle)."""
    d_inner = M3_EXPAND * d_model
    assert d_inner % M3_HEADDIM == 0, d_model
    H = d_inner // M3_HEADDIM
    dip = 2 * d_inner + 2 * M3_NGROUPS * M3_D_STATE + 3 * H + M3_ROPE
    return d_inner, H, dip


def m3_cols(d_model):
    """Column offsets of the 8-way split (mamba3.py:106-107, :177-186;
    mamba3_fixture.mojo::Mamba3Dims col_* methods). The split is a COPY,
    not a seam."""
    d_inner, H, _ = m3_dims(d_model)
    col_dt = 2 * d_inner + 2 * M3_NGROUPS * M3_D_STATE
    return dict(
        z=0, x=d_inner, b=2 * d_inner,
        c=2 * d_inner + M3_NGROUPS * M3_D_STATE,
        dt=col_dt, a=col_dt + H, trap=col_dt + 2 * H, angle=col_dt + 3 * H,
    )


def m3_default_ranges(d_model):
    """mamba3_fixture.mojo::m3_case_weights defaults, verbatim."""
    d_inner, _, _ = m3_dims(d_model)
    s_in = fan_in_scale(d_model)
    s_out = fan_in_scale(d_inner)
    return {
        "x": (-2.0, 2.0),
        "block_norm.weight": (0.5, 1.5),
        "in_proj.weight": (-s_in, s_in),
        "dt_bias": (-7.0, -2.0),
        "B_norm.weight": (0.5, 1.5),
        "C_norm.weight": (0.5, 1.5),
        "B_bias": (0.5, 1.5),
        "C_bias": (0.5, 1.5),
        "D": (0.5, 1.5),
        "out_proj.weight": (-s_out, s_out),
        "init_theta": (0.0, 6.25),  # inside [0, 2pi), the S10 mod's invariant
        "init_h": (-1.0, 1.0),
        "init_k": (-1.0, 1.0),
        "init_v": (-1.0, 1.0),
    }


def m3_shapes_for(d_model, B, L, with_init=False):
    d_inner, H, dip = m3_dims(d_model)
    s = {
        "x": [B, L, d_model],
        "block_norm.weight": [d_model],
        "in_proj.weight": [dip, d_model],
        "dt_bias": [H],
        "B_norm.weight": [M3_D_STATE],
        "C_norm.weight": [M3_D_STATE],
        "B_bias": [H, M3_D_STATE],
        "C_bias": [H, M3_D_STATE],
        "D": [H],
        "out_proj.weight": [d_model, d_inner],
    }
    if with_init:
        s["init_theta"] = [B, H, M3_ROPE]
        s["init_h"] = [B, H, M3_HEADDIM, M3_D_STATE]
        s["init_k"] = [B, H, M3_D_STATE]
        s["init_v"] = [B, H, M3_HEADDIM]
    return s


def _m3_is_small_dyadic(v):
    """True only for map_range's DOCUMENTED precondition class: "dyadic
    rationals with few significant bits" -- v = m / 2^k with a SHORT m.
    `_is_dyadic_small` cannot make this call: EVERY binary float is a
    dyadic rational (Fraction(-1.8) is -8106479329266893 / 2^52, and 2^52
    passes its power-of-two test), so a guard built on it routes -1.8 to
    the asserting path, whose per-element exact-rational check then
    correctly refuses -- the measured first-run failure, 2026-09-01:
    AssertionError ('float64 evaluation is not exact', -2.5, -1.8,
    0.6609215140342712, ...). What actually makes `lo + (hi - lo) * f`
    exact in float64 is the bounds' significands staying short enough
    that span and every span * f product are exactly representable, so
    this predicate bounds the NUMERATOR (and, for symmetry, the
    denominator) at 24 bits -- comfortably covering every range the three
    corpus tables use (halves, small integers, 6.25, the fan-in scales,
    2^30, +-4096) and excluding any bound spelled from a non-terminating
    binary literal. Classifying too eagerly toward the asserting path
    fails LOUDLY there, never silently, so the bias of this bound is
    safe."""
    fr = Fraction(v)
    den = fr.denominator
    return (den & (den - 1) == 0
            and abs(fr.numerator).bit_length() <= 24
            and den <= (1 << 24))


def m3_map_range(f_unit, lo, hi):
    """map_range, minus the exact-rational assertion for the ONE range
    where it cannot apply: dt_bias in [-2.5, -1.8] (the angle-crossing
    plant, mamba3_fixture.mojo k=13; -1.8 has no short binary spelling).
    For that range the EXEMPTION'S ARGUMENT IS IEEE DETERMINISM, not
    rational exactness: the Mojo fixture's `corpus_tensor` computes
    `Float32(lo + span * unit)` with lo/hi the doubles nearest the
    literals, `span = hi - lo` one correctly rounded subtraction, one
    correctly rounded multiply and add per element, and one final cast --
    and this arm performs the IDENTICAL operation sequence on the
    identical doubles, so the two implementations agree bit for bit even
    though `lo + span * f` is NOT the exact rational value (the byte gate
    in mamba3_check.mojo remains the arbiter that they in fact agree).
    Every small-dyadic range -- all of them but this one -- still goes
    through the asserting map_range, exactness proven per element."""
    if _m3_is_small_dyadic(lo) and _m3_is_small_dyadic(hi):
        return map_range(f_unit, lo, hi)
    span = float(hi) - float(lo)
    return (float(lo) + span * np.asarray(f_unit, dtype=np.float64)).astype(np.float32)


def m3_gen_tensor(seed, name, shape, lo, hi):
    n = int(np.prod(shape))
    f = hashed_unit(seed, M3_TENSOR_IDS[name], n)
    return m3_map_range(f, lo, hi).reshape(shape)


# Stage sets. Card order per contract section 7 (28 stages). seg.L is
# recorded only on the two tiny cases; everywhere else the elision is
# STATED in the manifest, never silent (section 7's rule, the m2 pattern).
M3_FULL_STAGES = [
    "input.x", "norm.sumsq", "norm.out", "in_proj.out", "A.out",
    "dt.out", "adt.out", "trap.sigma", "trap.scale", "bcnorm.B",
    "bcnorm.C", "angle.theta", "rot.q", "rot.k", "qkdot.out",
    "kscale.out", "dacs.out", "yintra.out", "ystate.out", "skip.out",
    "gate.out", "out_proj.out", "residual.out", "ssd.h_last",
    "ssd.k_last", "ssd.v_last", "ssd.theta_last",
]
M3_SEG_STAGES = (M3_FULL_STAGES[:17] + ["seg.L"] + M3_FULL_STAGES[17:])
M3_REDUCED_STAGES = [
    "input.x", "dt.out", "adt.out", "trap.scale", "angle.theta",
    "dacs.out", "yintra.out", "ystate.out", "skip.out", "gate.out",
    "residual.out", "ssd.h_last", "ssd.k_last", "ssd.v_last",
    "ssd.theta_last",
]

# The case table: mamba3_fixture.mojo::m3_corpus_case MIRRORED index for
# index (18 cases; contract 8g's L set {1, 4, 63, 64, 65, 129, 257} plus
# the decode fixture's 70). `plant` names the in_proj.weight row-block
# overwrite the fixture spells (its module docstring records why each
# exists); `overrides` are range replacements; both must track the fixture
# EXACTLY -- the byte gate is the arbiter.
M3_CASES = [
    dict(name="m3_base_b1_l1_d32", B=1, L=1, d_model=32, stages=M3_SEG_STAGES,
         note="smallest shape; L=1 under Q=64 exercises the padded sub-chunk arm and is the decode step shape; seg.L recorded (size allows)"),
    dict(name="m3_base_b2_l4_d32", B=2, L=4, d_model=32, stages=M3_SEG_STAGES,
         note="B=2 L=4, the corpus-check default case (at L=1 token-major and channel-major are the same bytes -- the mamba1 reindexing lesson); seg.L recorded"),
    dict(name="m3_base_b3_l4_d64", B=3, L=4, d_model=64,
         note="d_model 64 (H=2, the first multi-head shape -- rotation is PER HEAD, so post-rotation K/Q differ per head even at G=1), odd batch"),
    dict(name="m3_base_b1_l63_d32", B=1, L=63, d_model=32,
         note="one row short of the chunk: the largest one-chunk shape"),
    dict(name="m3_base_b1_l64_d32", B=1, L=64, d_model=32,
         note="the exact one-chunk boundary (Q = 64), zero padding"),
    dict(name="m3_base_b1_l65_d64", B=1, L=65, d_model=64,
         note="the first chunk crossing at H=2; CHUNK_SIZE_32's witness shape (L > 32)"),
    dict(name="m3_comp_b2_l65_d32", B=2, L=65, d_model=32, stages=M3_REDUCED_STAGES,
         note="composition parent: B=2 at L=65 (one crossing); its rows are m3_comp_row0/row1 with the SAME parameters"),
    dict(name="m3_comp_row0_b1_l65_d32", B=1, L=65, d_model=32, stages=M3_REDUCED_STAGES,
         x_source=("m3_comp_b2_l65_d32", 0),
         note="row 0 of m3_comp_b2_l65_d32 as a B=1 case (same seed, same parameters, x = parent x[0])"),
    dict(name="m3_comp_row1_b1_l65_d32", B=1, L=65, d_model=32, stages=M3_REDUCED_STAGES,
         x_source=("m3_comp_b2_l65_d32", 1),
         note="row 1 of m3_comp_b2_l65_d32, same construction"),
    dict(name="m3_state_b1_l129_d32", B=1, L=129, d_model=32,
         note="THREE working chunks (129 = 2*64 + 1): STATE_TERM_SCALE_FIRST's witness (L > Q with nonzero incoming state at chunks 1 and 2)"),
    dict(name="m3_state_b2_l257_d64", B=2, L=257, d_model=64, stages=M3_REDUCED_STAGES,
         note="five chunks at H=2, B=2; the longest shape in the family"),
    dict(name="m3_adv_softplus_band_b2_l8_d32", B=2, L=8, d_model=32,
         overrides={"dt_bias": (8.0, 14.0)},
         note="(a) biased dt IN the softplus distinguishing band [8, 14], never straddling 20 (the mamba1 adv_softplus_guard lesson, inherited by contract section 4's closing note)"),
    dict(name="m3_adv_a_floor_b2_l8_d32", B=2, L=8, d_model=32, plant="a_floor",
         note="(b) the A_FLOOR_UNCLAMPED witness: dd_A rows of in_proj.weight PLANTED (column 0 = -30000.0, others +0.0), the ONLY region where the S5 clamp binds (dd_A < 1 - 1e4); dd_A = -30000 * norm.out[:, 0] so a healthy fraction of cells clamp AND a healthy fraction do not; verify arm 5 counts both and refuses zero as VACUOUS"),
    dict(name="m3_adv_angle_crossing_b1_l48_d32", B=1, L=48, d_model=32, plant="angle",
         overrides={"dt_bias": (-2.5, -1.8)},
         note="(c) the ANGLE_MOD_PER_CHUNK/_AT_END witness: angle rows of in_proj.weight widened to [-4096, 4096] (tanh saturates, angle rate ~ +-pi), dt_bias in [-2.5, -1.8] (dt ~ 0.08-0.15), so the serial angle recurrence random-walks across the [0, 2pi) seam INSIDE the first chunk; verify arm 5 counts non-boundary mod engagements and refuses zero as VACUOUS (contract 8f: a fixture that never crosses is vacuous for these arms)"),
    dict(name="m3_adv_trap_saturating_b2_l8_d32", B=2, L=8, d_model=32, plant="trap",
         note="(d) trap rows of in_proj.weight in [-1024, 1024]: sigma(trap) saturates to exact 0 and exact 1 in FP32, both directions (corpus row owed by name, contract 8g); TRAP_LEFT_ONLY's fixture must stay NONUNIFORM in trap, which saturation both ways guarantees"),
    dict(name="m3_adv_signed_zeros_b2_l8_d32", B=2, L=8, d_model=32,
         zero_rule="x[b,t,:] = +0.0 for t%4==0; x[b,t,:] = -0.0 for t%4==2; of the remaining elements, flat index i%7==3 is -0.0",
         note="(e) the mamba-1 zero rule verbatim; zero-sign propagation through the S1 fold and the S4 GEMM cells is compared BY SIGN BIT by the lane, which the tolerance cannot see"),
    dict(name="m3_init_states_b1_l65_d32", B=1, L=65, d_model=32, init_states=True,
         note="(f) nonzero hashed Input_States (theta in [0, 6.25] -- inside [0, 2pi) by the S10 mod's own invariant -- h/k/v in [-1, 1]): RESUME_KERNEL_ASSOC's witness and the section-5-claim-2 TOLERANCE continuation row; the S22 correction folds the scalar FIRST (fwd_ref :266-267, the normative association). NEVER a bitwise gate against an unbroken prefill (contract DEVIATION 831)"),
    dict(name="m3_decode_b1_l60p10_d32", B=1, L=70, d_model=32, prefix_len=60,
         note="(g) gate (d)'s chunk-crossing decode fixture (contract 8d: prefill 60, decode through 70, Q=64). As a CASE TABLE row it is the L=70 prefill, because DEVIATION 831 makes that prefill the decode reference; the decode CHAIN itself is the lane's gate (d), not a corpus fixture. Verify arm 6 checks the prefix property here"),
]


def m3_case_seed(k):
    return (M3_SEED_BASE + 0x1000 * k) & M64


def m3_case_index(name):
    for k, c in enumerate(M3_CASES):
        if c["name"] == name:
            return k
    raise KeyError(name)


def m3_case_by_name(name):
    return M3_CASES[m3_case_index(name)]


# --------------------------------------------------------------------------
# _segsum, mamba3_siso_step_ref and mamba3_siso_fwd_ref, copied VERBATIM
# from https://github.com/state-spaces/mamba
# commit e9594ce1c732d97440f0332fdc43170a2294dbfa
# tests/ops/triton/test_mamba3_siso.py lines 22-31, 34-146 and 149-340
# (Copyright (c) 2025, Dao AI Lab, Goombalab; Apache-2.0). Our only edits:
# the lazy einops import line inside each body (the DEVIATION 1934 rule --
# einops must not be a module-scope import of this file), the `typing`
# names imported at this file's top, and this banner. Do not edit further.
# --------------------------------------------------------------------------
def _segsum(x: torch.Tensor) -> torch.Tensor:
    """Segment sum helper for attention computation."""
    from einops import repeat  # DEVIATION 1934: lazy, ours
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def mamba3_siso_step_ref(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    ADT: torch.Tensor,
    DT: torch.Tensor,
    Trap: torch.Tensor,
    Q_bias: torch.Tensor,
    K_bias: torch.Tensor,
    Angles: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    Z: Optional[torch.Tensor] = None,
    Input_States: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Reference implementation of Mamba-3 in recurrent (step) mode.

    Args:
        Input_States: Optional tuple of (Angle_State, SSM_State, K_State, V_State)

    Returns:
        out: Output tensor (batch, seqlen, nheads, headdim_v)
        Final_States: Tuple of (Angle_State, SSM_State, K_State, V_State)
    """
    from einops import repeat  # DEVIATION 1934: lazy, ours
    batch, seqlen, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape
    headdim_angles = Angles.shape[-1]
    device = Q.device
    assert seqlen > 0
    Angles = torch.tanh(Angles) * math.pi

    # Expand Q/K for GQA
    if Q.shape[2] != V.shape[2]:
        Q = repeat(Q, "b s h_bc d -> b s (h_bc g) d", g=V.shape[2] // Q.shape[2])
    if K.shape[2] != V.shape[2]:
        K = repeat(K, "b s h_bc d -> b s (h_bc g) d", g=V.shape[2] // K.shape[2])

    def apply_rotary_emb(tensor, cos, sin):
        tensor_reshaped = tensor.view(*tensor.shape[:-1], -1, 2)
        tensor_0 = tensor_reshaped[..., 0]
        tensor_1 = tensor_reshaped[..., 1]
        if cos.shape[-1] < tensor_0.shape[-1]:
            pad_size = tensor_0.shape[-1] - cos.shape[-1]
            cos = F.pad(cos, (0, pad_size), value=1.0)
            sin = F.pad(sin, (0, pad_size), value=0.0)
        rotated_0 = tensor_0 * cos - tensor_1 * sin
        rotated_1 = tensor_0 * sin + tensor_1 * cos
        rotated = torch.stack([rotated_0, rotated_1], dim=-1).view_as(tensor)
        return rotated

    # Initialize states
    if Input_States is not None:
        Angle_State, SSM_State, K_State, V_State = Input_States
        Angle_State = Angle_State.clone()
        SSM_State = SSM_State.clone().to(torch.float32)
        K_State = K_State.clone()
        V_State = V_State.clone()
    else:
        Angle_State = torch.zeros((batch, nheads, headdim_angles), dtype=torch.float32, device=device)
        SSM_State = torch.zeros((batch, nheads, headdim_v, headdim_qk), dtype=torch.float32, device=device)
        K_State = torch.zeros((batch, nheads, headdim_qk), dtype=Q.dtype, device=device)
        V_State = torch.zeros((batch, nheads, headdim_v), dtype=V.dtype, device=device)

    TWO_PI = 2 * math.pi
    out_arr = []

    for idx in range(seqlen):
        q = Q[:, idx, :, :] + Q_bias.unsqueeze(0)
        k = K[:, idx, :, :] + K_bias.unsqueeze(0)
        v = V[:, idx, :, :]
        adt = ADT[:, :, idx]
        dt = DT[:, :, idx]
        trap = Trap[:, :, idx]
        z = Z[:, idx, :, :] if Z is not None else None
        angles = Angles[:, idx, :, :]

        # Update angle state with cumsum: Angle_State = (Angle_State + Angles * DT) mod 2π
        Angle_State = Angle_State + angles * dt.unsqueeze(-1)
        Angle_State = Angle_State - TWO_PI * torch.floor(Angle_State / TWO_PI)

        # Apply rotary embeddings to Q and K using cumulative angles
        cos_angles = torch.cos(Angle_State)
        sin_angles = torch.sin(Angle_State)
        q_rot = apply_rotary_emb(q, cos_angles, sin_angles)
        k_rot = apply_rotary_emb(k, cos_angles, sin_angles)

        trap = torch.sigmoid(trap)
        alpha = torch.exp(adt)
        beta = (1 - trap) * dt * alpha
        gamma = trap * dt

        # Update SSM state using previous K_State and V_State
        SSM_State = alpha.unsqueeze(-1).unsqueeze(-1) * SSM_State
        SSM_State = SSM_State + beta.unsqueeze(-1).unsqueeze(-1) * (K_State.unsqueeze(-2) * V_State.unsqueeze(-1))
        SSM_State = SSM_State + gamma.unsqueeze(-1).unsqueeze(-1) * (k_rot.unsqueeze(-2) * v.unsqueeze(-1))

        # Compute output
        out = torch.einsum("bhdD, bhD -> bhd", SSM_State, q_rot.to(SSM_State.dtype))

        if D is not None:
            out = out + D[None, :, None] * v

        if Z is not None:
            out = out * z * torch.sigmoid(z)

        out_arr.append(out)

        # Update K and V states for next step
        K_State = k_rot
        V_State = v

    out = torch.stack(out_arr, dim=1)
    Final_States = (Angle_State, SSM_State, K_State, V_State)
    return out, Final_States


def mamba3_siso_fwd_ref(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    ADT: torch.Tensor,
    DT: torch.Tensor,
    Trap: torch.Tensor,
    Q_bias: torch.Tensor,
    K_bias: torch.Tensor,
    Angles: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    Z: Optional[torch.Tensor] = None,
    Initial_States: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    chunk_size: int = 64,
    dtype: torch.dtype = torch.float32,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    """Reference implementation of Mamba-3 forward pass.

    Args:
        Initial_States: Optional tuple of (Angle_State, SSM_State, K_State, V_State)

    Returns:
        out_z: Output with Z gating applied
        final_states: (Final_Angle_State, Final_SSM_State, Final_K_State, Final_V_State)
    """
    from einops import repeat  # DEVIATION 1934: lazy, ours
    batch, total_seqlen, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape
    headdim_angles = Angles.shape[-1]
    device = Q.device

    is_varlen = cu_seqlens is not None
    if is_varlen:
        assert batch == 1

    # Cast inputs
    Q = Q.to(dtype)
    K = K.to(dtype)
    V = V.to(dtype)
    ADT = ADT.to(torch.float32)
    DT = DT.to(torch.float32)
    Trap = Trap.to(dtype)
    Q_bias = Q_bias.to(dtype)
    K_bias = K_bias.to(dtype)
    Angles = Angles.to(dtype)
    if D is not None:
        D = D.to(dtype)
    if Z is not None:
        Z = Z.to(dtype)
    if Initial_States is not None:
        Initial_Angle_State, Initial_SSM_State, Initial_K_State, Initial_V_State = Initial_States

    Angles = torch.tanh(Angles) * math.pi
    # Expand Q/K for GQA
    if Q.shape[2] != V.shape[2]:
        Q = repeat(Q, "b s h_bc d -> b s (h_bc g) d", g=V.shape[2] // Q.shape[2])
    if K.shape[2] != V.shape[2]:
        K = repeat(K, "b s h_bc d -> b s (h_bc g) d", g=V.shape[2] // K.shape[2])

    out_zs = []
    Final_Angle_States = []
    Final_SSM_States = []
    Final_K_States = []
    Final_V_States = []

    TWO_PI = 2 * math.pi

    def _rotary(tensor, cos, sin):
        tensor_reshaped = tensor.view(*tensor.shape[:-1], -1, 2)
        tensor_0 = tensor_reshaped[..., 0]
        tensor_1 = tensor_reshaped[..., 1]
        if cos.shape[-1] < tensor_0.shape[-1]:
            pad_size = tensor_0.shape[-1] - cos.shape[-1]
            cos = F.pad(cos, (0, pad_size), value=1.0)
            sin = F.pad(sin, (0, pad_size), value=0.0)
        rotated_0 = tensor_0 * cos - tensor_1 * sin
        rotated_1 = tensor_0 * sin + tensor_1 * cos
        return torch.stack([rotated_0, rotated_1], dim=-1).view_as(tensor)

    def compute_one_sequence(seq_idx):
        if is_varlen:
            start_idx, end_idx = cu_seqlens[seq_idx].item(), cu_seqlens[seq_idx + 1].item()
            Q_curr = Q[0, start_idx:end_idx, :, :]
            K_curr = K[0, start_idx:end_idx, :, :]
            V_curr = V[0, start_idx:end_idx, :, :]
            ADT_curr = ADT[0, :, start_idx:end_idx]
            DT_curr = DT[0, :, start_idx:end_idx]
            Trap_curr = Trap[0, :, start_idx:end_idx]
            Angles_curr = Angles[0, start_idx:end_idx, :, :]
            Z_curr = Z[0, start_idx:end_idx, :, :] if Z is not None else None
        else:
            Q_curr = Q[seq_idx]
            K_curr = K[seq_idx]
            V_curr = V[seq_idx]
            ADT_curr = ADT[seq_idx]
            DT_curr = DT[seq_idx]
            Trap_curr = Trap[seq_idx]
            Angles_curr = Angles[seq_idx]
            Z_curr = Z[seq_idx] if Z is not None else None

        Trap_curr = torch.sigmoid(Trap_curr)
        seqlen_curr = Q_curr.shape[0]

        Angles_scaled = Angles_curr.float() * DT_curr.transpose(0, 1).unsqueeze(-1)
        Angles_Cumsum = torch.cumsum(Angles_scaled, dim=0)
        if Initial_States is not None:
            Initial_Angle_State_curr = Initial_Angle_State[seq_idx]
            Angles_Cumsum = Angles_Cumsum + Initial_Angle_State_curr.unsqueeze(0)
        Angles_Cumsum = Angles_Cumsum - TWO_PI * torch.floor(Angles_Cumsum / TWO_PI)
        Final_Angle_States.append(Angles_Cumsum[-1])

        # Initialize acc_states
        if Initial_States is not None:
            Initial_SSM_State_curr = Initial_SSM_State[seq_idx]
            Initial_K_State_curr = Initial_K_State[seq_idx]
            Initial_V_State_curr = Initial_V_State[seq_idx]

            scalar = DT_curr[:, 0] * (1 - Trap_curr[:, 0])
            acc_states = Initial_SSM_State_curr + Initial_V_State_curr[:, :, None] * Initial_K_State_curr[:, None, :] * scalar[:, None, None]
        else:
            acc_states = torch.zeros((nheads, headdim_v, headdim_qk), device=device, dtype=torch.float32)

        # Compute shifted gamma and scale
        DT_shifted = F.pad(DT_curr[:, 1:], (0, 1))
        Trap_shifted = F.pad(Trap_curr[:, 1:], (0, 1))
        shifted_gamma = DT_shifted * (1 - Trap_shifted)
        scale = DT_curr * Trap_curr + DT_shifted * (1 - Trap_shifted)

        # Add biases
        Q_curr = Q_curr + Q_bias.unsqueeze(0)
        K_curr = K_curr + K_bias.unsqueeze(0)

        # Compute QK dot for skip connection
        QK_dot = torch.sum(K_curr * Q_curr, dim=-1) * shifted_gamma.transpose(0, 1)

        # Rotary embeddings using Angles_Cumsum
        cos_angles_curr = torch.cos(Angles_Cumsum).to(Q_curr.dtype)
        sin_angles_curr = torch.sin(Angles_Cumsum).to(Q_curr.dtype)
        Q_curr = _rotary(Q_curr, cos_angles_curr, sin_angles_curr)
        K_curr = _rotary(K_curr, cos_angles_curr, sin_angles_curr)

        Final_K_States.append(K_curr[-1])
        Final_V_States.append(V_curr[-1])

        K_curr_scaled = K_curr * scale.transpose(0, 1).unsqueeze(-1).to(K_curr.dtype)

        # Compute output via quadratic attention
        QK = torch.einsum("thd,shd->hts", Q_curr, K_curr_scaled)
        QK_causal = torch.tril(QK)
        QK_causal = (QK_causal * torch.exp(_segsum(ADT_curr))).to(QK_causal.dtype)
        out = torch.einsum("hts,shd->thd", QK_causal, V_curr)

        if Initial_States is not None:
            da_cs = torch.cumsum(ADT_curr, dim=-1)
            exp_da_cs = torch.exp(da_cs)
            out = out + torch.einsum("hDd,thd,ht->thD", acc_states.to(Q_curr.dtype), Q_curr, exp_da_cs.to(Q_curr.dtype))

        if D is not None:
            out = out + D[None, :, None] * V_curr

        out = out - V_curr * QK_dot.unsqueeze(-1)

        if Z_curr is not None:
            out = out * Z_curr * torch.sigmoid(Z_curr)
        out_zs.append(out)

        # Compute final state
        da_cs_last = torch.exp(torch.sum(ADT_curr, dim=-1))
        da_cs_rev = torch.exp(torch.sum(ADT_curr, dim=-1, keepdim=True) - torch.cumsum(ADT_curr, dim=-1))
        V_curr_scaled = V_curr * da_cs_rev.permute(1, 0).unsqueeze(-1).to(V_curr.dtype)
        final_acc_states = acc_states * da_cs_last.unsqueeze(-1).unsqueeze(-1) + torch.einsum(
            "thd,thD->hDd", K_curr_scaled, V_curr_scaled.to(K_curr_scaled.dtype))
        Final_SSM_States.append(final_acc_states)

    num_sequences = cu_seqlens.size(0) - 1 if is_varlen else batch
    for seq_idx in range(num_sequences):
        compute_one_sequence(seq_idx)

    if not is_varlen:
        out_zs = torch.stack(out_zs, dim=0)
        Final_Angle_States = torch.stack(Final_Angle_States, dim=0)
        Final_SSM_States = torch.stack(Final_SSM_States, dim=0)
        Final_K_States = torch.stack(Final_K_States, dim=0)
        Final_V_States = torch.stack(Final_V_States, dim=0)
    else:
        out_zs = torch.cat(out_zs, dim=0).unsqueeze(0)
        Final_Angle_States = torch.stack(Final_Angle_States, dim=0)
        Final_SSM_States = torch.stack(Final_SSM_States, dim=0)
        Final_K_States = torch.stack(Final_K_States, dim=0)
        Final_V_States = torch.stack(Final_V_States, dim=0)

    return out_zs, (Final_Angle_States, Final_SSM_States, Final_K_States, Final_V_States)
# ---- end of verbatim test_mamba3_siso.py copies ---------------------------


# NOTE on the hard `.to(torch.float32)` casts inside the verbatim copies
# (fwd_ref :188-189 for ADT/DT; step_ref :87 for the SSM state): for the
# float64 reference run those would silently halve the reference's
# precision. `_m3_ref_f32_as` therefore temporarily rebinds
# `torch.Tensor.to` so that an EXPLICIT torch.float32 dtype argument maps
# to the working dtype -- the same move, and the same justification, as
# `_scan_ref_dtype` above (`.float()` there, `.to(torch.float32)` here).
# The float32 run is unaffected (the mapping is the identity). The
# `torch.zeros(..., dtype=torch.float32)` zero-state constructions inside
# the copies are NOT Tensor.to calls and stay float32; the first arithmetic
# against a float64 operand promotes them, and the cross-checks are
# roundoff-scale reports either way.
class _m3_ref_f32_as:
    def __init__(self, dtype):
        self.dtype = dtype

    def __enter__(self):
        self._orig = torch.Tensor.to
        dtype = self.dtype
        orig = self._orig

        def to_mapped(t, *args, **kwargs):
            if args and args[0] is torch.float32:
                args = (dtype,) + args[1:]
            if kwargs.get("dtype") is torch.float32:
                kwargs = dict(kwargs, dtype=dtype)
            return orig(t, *args, **kwargs)

        torch.Tensor.to = to_mapped
        return self

    def __exit__(self, *a):
        torch.Tensor.to = self._orig


# --------------------------------------------------------------------------
# The staged Mamba-3 block: every stage of contract section 7, spelled by
# the profile's own schedule (the KERNEL's chunked two-phase shape,
# DEVIATION 827; the kernel's diagonal spelling, DEVIATION 830; the
# per-token serial mod, DEVIATION 829) in plain torch. Citations: m3 =
# mamba_ssm/modules/mamba3.py; fwd = ops/triton/mamba3/mamba3_siso_fwd.py;
# ref = tests/ops/triton/test_mamba3_siso.py::mamba3_siso_fwd_ref;
# rmr = ops/triton/layernorm_gated.py::rms_norm_ref -- all at the pin.
# --------------------------------------------------------------------------
def m3_pad_tokens(t, pad):
    """Zero-pad dim=1 (tokens) at the end; contract section 3: the last
    chunk is padded to Q with +0.0 q/k/v/dt/angle rows."""
    if pad == 0:
        return t
    spec = [0, 0] * (t.dim() - 2) + [0, pad]
    return F.pad(t, spec)


def m3_forward(p, x, dtype):
    """Returns a dict of stage tag -> tensor (contract section 7's card
    shapes), all in `dtype`. `p` may carry the four `init_*` pieces
    (`Input_States`, contract section 5 claim 2)."""
    P = {k: v.to(dtype) for k, v in p.items()}
    init_theta = P.pop("init_theta", None)
    init_h = P.pop("init_h", None)
    init_k = P.pop("init_k", None)
    init_v = P.pop("init_v", None)
    x = x.to(dtype)
    Bsz, L, dm = x.shape
    d_inner, H, dip = m3_dims(dm)
    G, N, Q, PP, R = M3_NGROUPS, M3_D_STATE, M3_CHUNK, M3_HEADDIM, M3_ROPE
    Mtok = Bsz * L
    nchunks = -(-L // Q)
    out = {}
    out["input.x"] = x.reshape(Mtok, dm)

    # S1-S3 block RMSNorm, mamba2 S1-S3 VERBATIM (contract section 4;
    # block.py Block.forward :51-53 non-fused arm; rstd per rmr :29).
    residual = x
    sumsq = x.pow(2).sum(-1)
    out["norm.sumsq"] = sumsq.reshape(-1)
    rstd = 1 / torch.sqrt(sumsq / dm + M3_EPS)
    hnorm = P["block_norm.weight"] * (x * rstd[..., None])
    out["norm.out"] = hnorm.reshape(Mtok, dm)

    # S4 in_proj (m3 :176); 8-way split z|x|B|C|dd_dt|dd_A|trap|angle
    # (m3 :106-107, :177-186; a COPY, not a seam)
    proj = F.linear(hnorm, P["in_proj.weight"])
    out["in_proj.out"] = proj.reshape(Mtok, dip)
    z, xs, Bs, Cs, dd_dt, dd_A, trap_raw, angle_raw = torch.split(
        proj, [d_inner, d_inner, G * N, G * N, H, H, H, R], dim=-1)

    # S5 data-dependent A: -heavy_tail(dd_A) then clamp(max=-A_floor)
    # (m3 :194-195; heavy_tail_activation :39-41 spelled branchless there,
    # bit-equal to the branch spelling -- contract S5's 782-style record)
    A = -(dd_A.clamp_min(0) + torch.reciprocal(1 - dd_A.clamp_max(0)))
    A = torch.clamp(A, max=-M3_A_FLOOR)
    out["A.out"] = A.reshape(Mtok, H)

    # S6 dt = softplus(dd_dt + dt_bias), NO clamp (m3 :196)
    dt = F.softplus(dd_dt + P["dt_bias"])
    out["dt.out"] = dt.reshape(Mtok, H)

    # S7 ADT = A * dt (m3 :197); always <= -A_floor*dt <= 0
    adt = A * dt
    out["adt.out"] = adt.reshape(Mtok, H)

    # S8-S9 trapezoid: sigma, gamma, beta', scale (ref :249, :272-275;
    # fwd :293-306); shifted operands are +0.0 past the sequence end
    sig = torch.sigmoid(trap_raw)
    out["trap.sigma"] = sig.reshape(Mtok, H)
    dt_sh = F.pad(dt[:, 1:, :], (0, 0, 0, 1))
    sig_sh = F.pad(sig[:, 1:, :], (0, 0, 0, 1))
    gamma = dt * sig
    scale = gamma + dt_sh * (1 - sig_sh)
    out["trap.scale"] = scale.reshape(Mtok, H)

    # S21 B/C RMSNorm over N per (b, l, g), eps 1e-5, learned weight, NO
    # gate NO bias (m3 :126-127, :204-206; rmr :29-30 at z=None)
    Bn = Bs.reshape(Bsz, L, G, N)
    Cn = Cs.reshape(Bsz, L, G, N)
    Bn = (Bn * (1 / torch.sqrt(Bn.pow(2).mean(-1, keepdim=True) + M3_EPS))) * P["B_norm.weight"]
    Cn = (Cn * (1 / torch.sqrt(Cn.pow(2).mean(-1, keepdim=True) + M3_EPS))) * P["C_norm.weight"]
    out["bcnorm.B"] = Bn.reshape(Mtok, G, N)
    out["bcnorm.C"] = Cn.reshape(Mtok, G, N)

    # S10 angle recurrence: a = tanh(angle_raw)*pi, inc = a*dt per head,
    # theta serial per token with the mod applied EVERY step (DEVIATION
    # 829's placement, step_ref :109-111; the head expand of angle_raw is
    # a COPY, m3 :202)
    TWO_PI = 2 * math.pi
    a = torch.tanh(angle_raw) * math.pi                     # [B, L, R]
    inc = a.unsqueeze(2) * dt.unsqueeze(-1)                 # [B, L, H, R]
    theta_state = (init_theta.clone() if init_theta is not None
                   else torch.zeros(Bsz, H, R, dtype=dtype))
    thetas = []
    for t in range(L):
        theta_state = theta_state + inc[:, t]
        theta_state = theta_state - TWO_PI * torch.floor(theta_state / TWO_PI)
        thetas.append(theta_state)
    theta = torch.stack(thetas, dim=1)                      # [B, L, H, R]
    out["angle.theta"] = theta.reshape(Mtok, H, R)

    # S12 Q/K bias AFTER the norm, BEFORE rotation (fwd :312-316; ref
    # :278-279); the G -> H broadcast is a COPY
    q = Cn.repeat_interleave(H // G, dim=2) + P["C_bias"]
    k = Bn.repeat_interleave(H // G, dim=2) + P["B_bias"]

    # S14 qk_dot PRE-rotation, times gamma -- the KERNEL's diagonal
    # spelling (fwd :319-325; DEVIATION 830 picks gamma over the ref's
    # beta')
    qkg = (q * k).sum(-1) * gamma
    out["qkdot.out"] = qkg.reshape(Mtok, H)

    # S13 rotation: interleaved pairs (2i, 2i+1), first R of the 64 pairs;
    # pairs >= R structurally unrotated (ref _rotary :216-226 pads
    # cos=1/sin=0, bit-equal to never computing them -- DEVIATION 828)
    cosA = torch.cos(theta)
    sinA = torch.sin(theta)
    cpad = F.pad(cosA, (0, N // 2 - R), value=1.0)
    spad = F.pad(sinA, (0, N // 2 - R), value=0.0)

    def _rot(t):
        tr = t.reshape(*t.shape[:-1], -1, 2)
        t0, t1 = tr[..., 0], tr[..., 1]
        return torch.stack([t0 * cpad - t1 * spad, t0 * spad + t1 * cpad], dim=-1).reshape(t.shape)

    q_rot = _rot(q)
    k_rot = _rot(k)
    out["rot.q"] = q_rot.reshape(Mtok, H, N)
    out["rot.k"] = k_rot.reshape(Mtok, H, N)  # PRE-scale, the carried spelling

    # S15 K scaling AFTER rotation, AFTER the k-state read (fwd :337-344)
    k_scaled = k_rot * scale.unsqueeze(-1)
    out["kscale.out"] = k_scaled.reshape(Mtok, H, N)

    v = xs.reshape(Bsz, L, H, PP)

    # S22 resumption correction, the NORMATIVE ref's association: fold the
    # scalar LAST onto (v x k) (ref :266-267; the kernel's fwd :371
    # association is refused silently -- contract S22)
    if init_h is not None:
        c1 = dt[:, 0, :] * (1 - sig[:, 0, :])               # [B, H]
        h = init_h + (init_v.unsqueeze(-1) * init_k.unsqueeze(-2)) * c1[..., None, None]
    else:
        h = torch.zeros(Bsz, H, PP, N, dtype=dtype)

    # The chunked SSD core (DEVIATION 827: the kernel's two-phase chunked
    # schedule at Q = 64, not the ref's whole-sequence attention).
    pad = (Q - L % Q) % Q
    qc = m3_pad_tokens(q_rot, pad).reshape(Bsz, nchunks, Q, H, N)
    kc = m3_pad_tokens(k_scaled, pad).reshape(Bsz, nchunks, Q, H, N)
    vc = m3_pad_tokens(v, pad).reshape(Bsz, nchunks, Q, H, PP)
    adtq = m3_pad_tokens(adt, pad).reshape(Bsz, nchunks, Q, H).permute(0, 3, 1, 2)

    # mamba2 S11 inherited: per-chunk cumsum of ADT; padded positions
    # carry the last real value (padded adt is +0.0)
    dacs = torch.cumsum(adtq, dim=-1)                       # [B, H, C, Q]
    out["dacs.out"] = dacs

    # S16 decay: L[i][j] = exp(sum_{s=j+1..i} adt_s) for j < i, STRUCTURAL
    # +0.0 for j >= i INCLUDING the diagonal (fwd :413-417 strict mask;
    # DEVIATION 830 moved the diagonal to S14/S18)
    mask_strict = torch.tril(torch.ones(Q, Q, dtype=torch.bool), diagonal=-1)
    xrep = adtq.unsqueeze(-1).expand(Bsz, H, nchunks, Q, Q)
    seg = torch.cumsum(xrep.masked_fill(~mask_strict, 0), dim=-2)
    Lmat = torch.where(mask_strict, torch.exp(seg), seg.new_zeros(()))
    out["seg.L"] = Lmat.permute(0, 2, 1, 3, 4)              # card [B, C, H, Q, Q]

    # S16 intra-chunk attention: s = q_rot . k_scaled^T over n, M = s (.) L,
    # Y_intra = M . v (fwd :411-418)
    s_qk = torch.einsum("bcthn,bcshn->bhcts", qc, kc)
    Mm = s_qk * Lmat
    Yintra = torch.einsum("bhcts,bcshp->bcthp", Mm, vc)
    out["yintra.out"] = Yintra.reshape(Bsz, nchunks * Q, H, PP)[:, :L].reshape(Mtok, H, PP)

    # S17 state read-out + S20 SERIAL inter-chunk pass (fwd :406-407,
    # :439-444; the profile's serial fma answer, mamba2 S17 inherited)
    ys_chunks = []
    for c in range(nchunks):
        exp_dacs = torch.exp(dacs[:, :, c])                 # [B, H, Q]
        ys = torch.einsum("bthn,bhpn->bthp", qc[:, c], h) * exp_dacs.permute(0, 2, 1)[..., None]
        ys_chunks.append(ys)
        d_last = dacs[:, :, c, -1]                          # [B, H]
        d_rev = d_last.unsqueeze(-1) - dacs[:, :, c]        # [B, H, Q]
        vdec = vc[:, c] * torch.exp(d_rev).permute(0, 2, 1)[..., None]
        incr = torch.einsum("bthp,bthn->bhpn", vdec, kc[:, c])
        h = torch.exp(d_last)[..., None, None] * h + incr
    Ystate = torch.stack(ys_chunks, dim=1).reshape(Bsz, nchunks * Q, H, PP)[:, :L]
    out["ystate.out"] = Ystate.reshape(Mtok, H, PP)
    out["ssd.h_last"] = h                                   # [B, H, P, N]

    # S18 diagonal + D skip in ONE add: t = D[h] + qk_gamma, Y += t * v
    # (fwd :421-422, the kernel's own association)
    Y = Yintra.reshape(Bsz, nchunks * Q, H, PP)[:, :L] + Ystate
    skip = Y + (P["D"] + qkg).unsqueeze(-1) * v
    out["skip.out"] = skip.reshape(Mtok, H, PP)

    # S19 Z gate, in-core, no output norm at defaults (fwd :430-431)
    z4 = z.reshape(Bsz, L, H, PP)
    gate = skip * F.silu(z4)
    out["gate.out"] = gate.reshape(Mtok, H, PP)

    # S4 out_proj (m3 :277), S23 residual (block.py :52/:67)
    o = F.linear(gate.reshape(Bsz, L, d_inner), P["out_proj.weight"])
    out["out_proj.out"] = o.reshape(Mtok, dm)
    out["residual.out"] = (residual + o).reshape(Mtok, dm)

    # section 5's four report pieces: k post-bias post-rotation PRE-scale
    # and v raw at the LAST REAL token (fwd wrapper :709-729; ref
    # :290-291), theta after the last real token
    out["ssd.k_last"] = k_rot[:, L - 1]                     # [B, H, N]
    out["ssd.v_last"] = v[:, L - 1]                         # [B, H, P]
    out["ssd.theta_last"] = theta[:, L - 1]                 # [B, H, R]
    return out


def m3_stage_shapes(meta):
    B, L, dm = meta["B"], meta["L"], meta["d_model"]
    d_inner, H, C = meta["d_inner"], meta["nheads"], meta["nchunks"]
    M = B * L
    dip = m3_dims(dm)[2]
    Q, N, PP, G, R = M3_CHUNK, M3_D_STATE, M3_HEADDIM, M3_NGROUPS, M3_ROPE
    return {
        "input.x": [M, dm], "norm.sumsq": [M], "norm.out": [M, dm],
        "in_proj.out": [M, dip], "A.out": [M, H], "dt.out": [M, H],
        "adt.out": [M, H], "trap.sigma": [M, H], "trap.scale": [M, H],
        "bcnorm.B": [M, G, N], "bcnorm.C": [M, G, N],
        "angle.theta": [M, H, R], "rot.q": [M, H, N], "rot.k": [M, H, N],
        "qkdot.out": [M, H], "kscale.out": [M, H, N],
        "dacs.out": [B, H, C, Q], "seg.L": [B, C, H, Q, Q],
        "yintra.out": [M, H, PP], "ystate.out": [M, H, PP],
        "skip.out": [M, H, PP], "gate.out": [M, H, PP],
        "out_proj.out": [M, dm], "residual.out": [M, dm],
        "ssd.h_last": [B, H, PP, N], "ssd.k_last": [B, H, N],
        "ssd.v_last": [B, H, PP], "ssd.theta_last": [B, H, R],
    }


M3_STAGE_DEFS = {
    "input.x": "the block input, as given, token-major  [M, d_model]",
    "norm.sumsq": "S1: per-row sum of squares (mamba2 S1 verbatim)  [M]",
    "norm.out": "S2-S3: x * (1/sqrt(sumsq/d_model + 1e-5)) * weight (mamba2 S2-S3 verbatim; rms_norm_ref:29 rstd)  [M, d_model]",
    "in_proj.out": "S4: norm.out @ in_proj.weight^T; columns z | x | B | C | dd_dt | dd_A | trap | angle (mamba3.py:106-107)  [M, d_in_proj]",
    "A.out": "S5: clamp(-heavy_tail(dd_A), max=-1e-4) (mamba3.py:194-195; heavy_tail :39-41)  [M, H]",
    "dt.out": "S6: softplus(dd_dt + dt_bias), NO clamp (mamba3.py:196)  [M, H]",
    "adt.out": "S7: A * dt (mamba3.py:197); always <= 0  [M, H]",
    "trap.sigma": "S8: sigmoid(trap) (fwd_ref :249)  [M, H]",
    "trap.scale": "S9: dt*sigma + dt_shifted*(1 - sigma_shifted), the trapezoid scale; shifted operands +0.0 past the sequence end (fwd:304-306; fwd_ref :272-275)  [M, H]",
    "bcnorm.B": "S21: RMSNorm over d_state with learned weight, eps 1e-5, no gate no bias (mamba3.py:126, :204; rms_norm_ref:29-30 at z=None)  [M, G, N]",
    "bcnorm.C": "S21, the C side (mamba3.py:127, :206)  [M, G, N]",
    "angle.theta": "S10: theta_t = mod2pi(theta_{t-1} + tanh(angle_raw)*pi * dt_t), SERIAL per token, mod EVERY step (DEVIATION 829; step_ref :109-111 placement)  [M, H, R]",
    "rot.q": "S12-S13: (C_normed + C_bias) rotated by theta, interleaved pairs, first 32 of 64 pairs (DEVIATION 828)  [M, H, N]",
    "rot.k": "S12-S13, the K side, PRE-scale (the carried k-state spelling, fwd:337-341)  [M, H, N]",
    "qkdot.out": "S14: (q . k) BEFORE rotation, times gamma -- the kernel's diagonal spelling (fwd:319-325; DEVIATION 830)  [M, H]",
    "kscale.out": "S15: k_rot * scale, AFTER rotation AFTER the state read (fwd:337-344)  [M, H, N]",
    "dacs.out": "mamba2 S11 inherited: per-chunk serial cumsum of ADT; padded positions carry the last real value  [B, H, C, Q]",
    "seg.L": "S16 decay: exp(segsum) STRICTLY below the diagonal, +0.0 on and above it (fwd:413-417; DEVIATION 830 moved the diagonal out)  [B, C, H, Q, Q]",
    "yintra.out": "S16: ((q_rot . k_scaled^T) (.) L) . v per chunk (fwd:411-418), truncated to L  [M, H, P]",
    "ystate.out": "S17: (q_rot . h_chunk_start^T) * exp(dacs), dacs INCLUSIVE of the token (fwd:406-407); h passed SERIALLY across chunks (S20, fwd:439-444)  [M, H, P]",
    "skip.out": "S18: Y + (D[h] + qk_gamma) * v -- diagonal and skip in ONE add (fwd:421-422)  [M, H, P]",
    "gate.out": "S19: skip * silu(z), in-core, no output norm at defaults (fwd:430-431)  [M, H, P]",
    "out_proj.out": "S4: gate.out @ out_proj.weight^T (mamba3.py:277)  [M, d_model]",
    "residual.out": "S23: residual + out_proj.out (block.py:52/:67; mamba2 S22 verbatim)  [M, d_model]",
    "ssd.h_last": "section 5 report: the SSM state after the final chunk of the serial pass  [B, H, P, N]",
    "ssd.k_last": "section 5 report: k post-bias post-rotation PRE-scale at the last real token (fwd wrapper :709-729)  [B, H, N]",
    "ssd.v_last": "section 5 report: v raw at the last real token  [B, H, P]",
    "ssd.theta_last": "section 5 report: theta after the last real token  [B, H, R]",
}


def m3_build_case(case):
    k = m3_case_index(case["name"])
    d_model = case["d_model"]
    d_inner, H, dip = m3_dims(d_model)
    B, L = case["B"], case["L"]
    src = case.get("x_source")
    seed = m3_case_seed(m3_case_index(src[0])) if src else m3_case_seed(k)
    with_init = case.get("init_states", False)
    shapes = m3_shapes_for(d_model, B, L, with_init=with_init)
    ranges = m3_default_ranges(d_model)
    ranges.update(case.get("overrides", {}))
    tensors = {}
    for name, shape in shapes.items():
        if name == "x" and src:
            parent = m3_case_by_name(src[0])
            pshape = m3_shapes_for(d_model, parent["B"], parent["L"])["x"]
            lo, hi = ranges["x"]
            px = m3_gen_tensor(seed, "x", pshape, lo, hi)
            tensors["x"] = px[src[1]:src[1] + 1].copy()
            continue
        lo, hi = ranges[name]
        tensors[name] = m3_gen_tensor(seed, name, shape, lo, hi)
    # The in_proj.weight PLANTS, mirroring mamba3_fixture.mojo::
    # m3_case_weights branch for branch. Each overwrites a row block of the
    # already-hashed base tensor, so everything outside the block keeps the
    # default derivation bit for bit.
    plant = case.get("plant")
    cols = m3_cols(d_model)
    if plant == "a_floor":
        # dd_A rows [col_a, col_a+H): column 0 = -30000.0, others +0.0
        w = tensors["in_proj.weight"]
        w[cols["a"]:cols["a"] + H, :] = np.float32(0.0)
        w[cols["a"]:cols["a"] + H, 0] = np.float32(-30000.0)
    elif plant == "angle":
        # angle rows [col_angle, col_angle+R) remapped through
        # [-4096, 4096] using the SAME hashed f values (the fixture's own
        # in-place respelling of corpus_tensor at those flat indices)
        f = hashed_unit(seed, M3_TENSOR_IDS["in_proj.weight"], dip * d_model).reshape(dip, d_model)
        g0 = cols["angle"]
        tensors["in_proj.weight"][g0:g0 + M3_ROPE] = m3_map_range(
            f[g0:g0 + M3_ROPE].reshape(-1), -4096.0, 4096.0).reshape(M3_ROPE, d_model)
    elif plant == "trap":
        f = hashed_unit(seed, M3_TENSOR_IDS["in_proj.weight"], dip * d_model).reshape(dip, d_model)
        t0 = cols["trap"]
        tensors["in_proj.weight"][t0:t0 + H] = m3_map_range(
            f[t0:t0 + H].reshape(-1), -1024.0, 1024.0).reshape(H, d_model)
    elif plant is not None:
        raise ValueError(plant)
    if "zero_rule" in case:
        tensors["x"] = apply_zero_rule(tensors["x"], case["zero_rule"])
    meta = dict(
        name=case["name"], family="mamba3", corpus=M3_CORPUS_VERSION,
        profile=M3_PROFILE, hash_spec=HASH_SPEC,
        seed=f"0x{seed:016X}", seed_case_index=(m3_case_index(src[0]) if src else k),
        B=B, L=L, d_model=d_model, d_inner=d_inner, nheads=H,
        headdim=M3_HEADDIM, d_state=M3_D_STATE, ngroups=M3_NGROUPS,
        num_rope_angles=M3_ROPE, expand=M3_EXPAND, chunk_size=M3_CHUNK,
        nchunks=-(-L // M3_CHUNK), rms_eps=M3_EPS, a_floor=M3_A_FLOOR,
        softplus_threshold=20.0,
        tensors={name: dict(shape=shapes[name], dtype="float32", order="row-major",
                            tensor_id=M3_TENSOR_IDS[name], lo=ranges[name][0], hi=ranges[name][1],
                            file=f"{name}.f32")
                 for name in shapes},
        note=case["note"],
    )
    if plant == "a_floor":
        meta["tensors"]["in_proj.weight"]["plant"] = (
            "dd_A rows [col_a, col_a+H): column 0 = -30000.0, all other columns +0.0 "
            "(mamba3_fixture.mojo k=12)")
    elif plant == "angle":
        meta["tensors"]["in_proj.weight"]["plant"] = (
            "angle rows [col_angle, col_angle+32) remapped through [-4096, 4096] "
            "from the same hashed stream (mamba3_fixture.mojo k=13)")
    elif plant == "trap":
        meta["tensors"]["in_proj.weight"]["plant"] = (
            "trap rows [col_trap, col_trap+H) remapped through [-1024, 1024] "
            "from the same hashed stream (mamba3_fixture.mojo k=14)")
    if src:
        meta["x_source"] = dict(case=src[0], batch_row=src[1])
        meta["tensors"]["x"]["derived"] = f"byte slice of {src[0]}/x.f32, batch row {src[1]}"
    if "zero_rule" in case:
        meta["zero_rule"] = case["zero_rule"]
        meta["tensors"]["x"]["derived"] = "hashed, then zero_rule applied"
    if "prefix_len" in case:
        meta["prefix_len"] = case["prefix_len"]
    return tensors, meta


def m3_emit_case(case, out_root):
    tensors, meta = m3_build_case(case)
    stages = list(case.get("stages", M3_FULL_STAGES))
    cdir = os.path.join(out_root, case["name"])
    os.makedirs(os.path.join(cdir, "ref64"), exist_ok=True)
    os.makedirs(os.path.join(cdir, "ref32"), exist_ok=True)
    for name, arr in tensors.items():
        meta["tensors"][name]["sha256"] = m2_write_raw(os.path.join(cdir, f"{name}.f32"), arr, np.float32)
    p = {k: torch.from_numpy(v.copy()) for k, v in tensors.items() if k != "x"}
    x = torch.from_numpy(tensors["x"].copy())
    r64 = m3_forward(p, x, torch.float64)
    r32 = m3_forward(p, x, torch.float32)
    shapes = m3_stage_shapes(meta)
    stage_meta = {}
    for s in stages:
        shp = shapes[s]
        assert list(r64[s].shape) == shp, (s, tuple(r64[s].shape), shp)
        sha64 = m2_write_raw(os.path.join(cdir, "ref64", f"{s}.f64"), r64[s].numpy(), np.float64)
        sha32 = m2_write_raw(os.path.join(cdir, "ref32", f"{s}.f32"), r32[s].numpy(), np.float32)
        stage_meta[s] = dict(shape=shp, order="row-major", ref64=f"ref64/{s}.f64",
                             ref32=f"ref32/{s}.f32", sha256_ref64=sha64, sha256_ref32=sha32)
    meta["stages"] = stage_meta
    meta["stage_order"] = stages
    if "seg.L" not in stages:
        meta["elided_stages"] = dict(
            stages=["seg.L"],
            reason="size cap (contract section 7: the elision is stated, never silent); "
                   "recorded on m3_base_b1_l1_d32 and m3_base_b2_l4_d32")
    meta["stage_definitions"] = "see ../manifest.json"
    with open(os.path.join(cdir, "manifest.json"), "w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return dict(meta=meta, r64=r64, r32=r32, p=p, x=x)


def m3_cross_inputs(meta, r, p, dtype):
    """The verbatim references' inputs, read back off the staged run's own
    tensors (the m2_cross_checks pattern): raw trap/angle from in_proj.out,
    normed B/C from bcnorm.*, dt/adt from their stages."""
    Bsz, L = meta["B"], meta["L"]
    dm = meta["d_model"]
    d_inner, H, dip = m3_dims(dm)
    G, N, PP, R = M3_NGROUPS, M3_D_STATE, M3_HEADDIM, M3_ROPE
    cols = m3_cols(dm)
    ip = r["in_proj.out"].reshape(Bsz, L, dip)
    z = ip[..., :d_inner].reshape(Bsz, L, H, PP)
    v = ip[..., cols["x"]:cols["x"] + d_inner].reshape(Bsz, L, H, PP)
    trap_raw = ip[..., cols["trap"]:cols["trap"] + H].permute(0, 2, 1)   # [B, H, L]
    angle_raw = ip[..., cols["angle"]:]                                   # [B, L, R]
    Angles = angle_raw.unsqueeze(2).expand(Bsz, L, H, R)
    Bn = r["bcnorm.B"].reshape(Bsz, L, G, N)
    Cn = r["bcnorm.C"].reshape(Bsz, L, G, N)
    adt = r["adt.out"].reshape(Bsz, L, H).permute(0, 2, 1)
    dt = r["dt.out"].reshape(Bsz, L, H).permute(0, 2, 1)
    init = None
    if "init_h" in p:
        init = tuple(p[n].to(dtype) for n in ("init_theta", "init_h", "init_k", "init_v"))
    return dict(Q=Cn, K=Bn, V=v, ADT=adt, DT=dt, Trap=trap_raw,
                Q_bias=p["C_bias"].to(dtype), K_bias=p["B_bias"].to(dtype),
                Angles=Angles, D=p["D"].to(dtype), Z=z, init=init)


def m3_generate(out_root, verify=False):
    m3_root = os.path.join(out_root, "mamba3")
    os.makedirs(m3_root, exist_ok=True)
    top = dict(
        family="mamba3", corpus=M3_CORPUS_VERSION, profile=M3_PROFILE,
        contract="mamba/IDENTICAL_MAMBA3_CONTRACT.md",
        normative_case_table="mamba/checks/mamba3_fixture.mojo (landed first; the byte gate in mamba3_check.mojo is the arbiter)",
        hash_spec=HASH_SPEC, seed_base=f"0x{M3_SEED_BASE:016X}",
        seed_rule="seed_k = seed_base + 0x1000 * k, k = case index below; x_source cases reuse the parent's seed",
        d_state=M3_D_STATE, expand=M3_EXPAND, headdim=M3_HEADDIM,
        ngroups=M3_NGROUPS, chunk_size=M3_CHUNK, num_rope_angles=M3_ROPE,
        rms_eps=M3_EPS, a_floor=M3_A_FLOOR, softplus_threshold=20.0,
        upstream=dict(
            mamba=("https://github.com/state-spaces/mamba @ e9594ce1c732d97440f0332fdc43170a2294dbfa, "
                   "tests/ops/triton/test_mamba3_siso.py::_segsum L22-31, ::mamba3_siso_step_ref L34-146, "
                   "::mamba3_siso_fwd_ref L149-340 (verbatim, NORMATIVE for values; the staged reference "
                   "follows the kernel's chunked schedule per contract DEVIATIONS 827/829/830); "
                   "mamba_ssm/modules/mamba3.py (block order, defaults, in_proj layout :106-107); "
                   "mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py (chunked schedule shape); "
                   "mamba_ssm/ops/triton/layernorm_gated.py::rms_norm_ref L18-39 (B/C norm)"),
            transformers=("NO SECOND REFERENCE: huggingface/transformers @ d56c55b has no mamba3 model "
                          "(contract section 0's one-repository note, said out loud)"),
        ),
        tensor_ids=M3_TENSOR_IDS, stage_definitions=M3_STAGE_DEFS, cases=[])
    total_bytes = 0
    all_meta = {}
    for k, case in enumerate(M3_CASES):
        entry = m3_emit_case(case, m3_root)
        all_meta[case["name"]] = entry
        meta = entry["meta"]
        row = dict(index=k, name=case["name"], seed=meta["seed"], B=case["B"], L=case["L"],
                   d_model=case["d_model"], d_inner=meta["d_inner"], nheads=meta["nheads"],
                   nchunks=meta["nchunks"], stages=meta["stage_order"], note=case["note"])
        if "prefix_len" in case:
            row["prefix_len"] = case["prefix_len"]
        top["cases"].append(row)
        cdir = os.path.join(m3_root, case["name"])
        total_bytes += sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fn in os.walk(cdir) for f in fn)
    with open(os.path.join(m3_root, "manifest.json"), "w") as fh:
        json.dump(top, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"wrote {len(M3_CASES)} mamba3 cases to {m3_root}, {total_bytes / 1e6:.3f} MB of tensor and manifest files")
    print(f"mamba3 corpus sha256 (files .f32 .f64 .json): {sha256_of_dir(m3_root)}")
    if verify:
        m3_run_verify(m3_root, all_meta)


def m3_run_verify(m3_root, all_meta):
    print("\n== m3 verify 0: root input tensors on disk (mamba3_check.mojo::gate_corpus opens "
          "mamba/corpus/mamba3/<case>/<tensor>.f32 RELATIVE TO THE REPO ROOT) ==")
    for name, entry in all_meta.items():
        cdir = os.path.join(m3_root, name)
        missing = [tm["file"] for tm in entry["meta"]["tensors"].values()
                   if not os.path.exists(os.path.join(cdir, tm["file"]))]
        assert not missing, (name, "missing root tensor files", missing)
        with open(os.path.join(cdir, "x.f32"), "rb") as fh:
            got = fh.read()
        want = np.ascontiguousarray(entry["x"].numpy().astype("<f4")).tobytes()
        assert got == want, (name, "x.f32 bytes differ from the hashed derivation")
        print(f"  {name:36s} {len(entry['meta']['tensors'])} root tensors present; "
              f"x.f32 {len(got)} bytes == hashed derivation")

    print("\n== m3 verify 1: verbatim mamba3_siso_fwd_ref (whole-sequence attention, mod at end, "
          "include-then-subtract diagonal) -- roundoff-scale agreement expected, NEVER bit-equality ==")
    print("== m3 verify 2: verbatim mamba3_siso_step_ref (per-token recurrence) -- same posture ==")
    for name, entry in all_meta.items():
        meta, p = entry["meta"], entry["p"]
        M = meta["B"] * meta["L"]
        H, PP = meta["nheads"], M3_HEADDIM
        for dtype, r in ((torch.float64, entry["r64"]), (torch.float32, entry["r32"])):
            ins = m3_cross_inputs(meta, r, p, dtype)
            with _m3_ref_f32_as(dtype):
                fo, (fa, fh_, fk, fv) = mamba3_siso_fwd_ref(
                    ins["Q"], ins["K"], ins["V"], ins["ADT"], ins["DT"], ins["Trap"],
                    ins["Q_bias"], ins["K_bias"], ins["Angles"], D=ins["D"], Z=ins["Z"],
                    Initial_States=ins["init"], chunk_size=M3_CHUNK, dtype=dtype)
                so, (sa, sh_, sk, sv) = mamba3_siso_step_ref(
                    ins["Q"], ins["K"], ins["V"], ins["ADT"], ins["DT"], ins["Trap"],
                    ins["Q_bias"], ins["K_bias"], ins["Angles"], D=ins["D"], Z=ins["Z"],
                    Input_States=ins["init"])
            g = r["gate.out"]
            d_f = max(float((fo.reshape(M, H, PP) - g).abs().max()),
                      float((fh_ - r["ssd.h_last"]).abs().max()),
                      float((fk - r["ssd.k_last"]).abs().max()),
                      float((fv - r["ssd.v_last"]).abs().max()),
                      float((fa - r["ssd.theta_last"]).abs().max()))
            d_s = max(float((so.reshape(M, H, PP) - g).abs().max()),
                      float((sh_ - r["ssd.h_last"]).abs().max()),
                      float((sk - r["ssd.k_last"]).abs().max()),
                      float((sv - r["ssd.v_last"]).abs().max()),
                      float((sa - r["ssd.theta_last"]).abs().max()))
            print(f"  {name:36s} {str(dtype):14s} fwd_ref maxabs={d_f:.3e} | step_ref maxabs={d_s:.3e}")

    print("\n== m3 verify 3: batch composition (comp rows vs parent, ref64 and ref32) ==")
    pm = all_meta["m3_comp_b2_l65_d32"]
    Lc = 65
    m_lead = ("input.x", "dt.out", "adt.out", "trap.scale", "angle.theta",
              "yintra.out", "ystate.out", "skip.out", "gate.out", "residual.out")
    b_lead = ("dacs.out", "ssd.h_last", "ssd.k_last", "ssd.v_last", "ssd.theta_last")
    for b in (0, 1):
        cm = all_meta[f"m3_comp_row{b}_b1_l65_d32"]
        for tag, key in (("ref64", "r64"), ("ref32", "r32")):
            for s in M3_REDUCED_STAGES:
                if s in m_lead:
                    a = pm[key][s][b * Lc:(b + 1) * Lc]
                elif s in b_lead:
                    a = pm[key][s][b:b + 1]
                else:
                    continue
                c = cm[key][s]
                print(f"  row{b} {tag} {s:14s} bit-equal={torch.equal(a, c)} maxabs={float((a - c).abs().max()):.3e}")
    r0 = all_meta["m3_comp_row0_b1_l65_d32"]["r64"]["gate.out"]
    r1 = all_meta["m3_comp_row1_b1_l65_d32"]["r64"]["gate.out"]
    print(f"  negative control: row0 vs row1 gate.out DIFFER = {not torch.equal(r0, r1)} (must be True)")

    print("\n== m3 verify 4: determinism (regenerate into a temp dir, byte-compare) ==")
    tmp = tempfile.mkdtemp(prefix="mamba3-corpus-")
    try:
        for case in M3_CASES:
            m3_emit_case(case, tmp)
        mism = []
        for dp, _, fn in os.walk(tmp):
            for f in fn:
                a = os.path.join(dp, f)
                bpath = os.path.join(m3_root, os.path.relpath(a, tmp))
                with open(a, "rb") as fa, open(bpath, "rb") as fb:
                    if fa.read() != fb.read():
                        mism.append(os.path.relpath(a, tmp))
        print(f"  files compared: {sum(len(fn) for _, _, fn in os.walk(tmp))}, mismatches: {len(mism)}")
        for m in mism:
            print("   MISMATCH", m)
    finally:
        shutil.rmtree(tmp)

    print("\n== m3 verify 5: adversarial and witness intent reached (a fixture that cannot "
          "witness its arm is a clause not gated -- HARD-asserted where the contract names vacuity) ==")
    # softplus band [8, 14]
    e = all_meta["m3_adv_softplus_band_b2_l8_d32"]
    meta = e["meta"]
    cols = m3_cols(meta["d_model"])
    H = meta["nheads"]
    ip = e["r64"]["in_proj.out"]
    biased = ip[:, cols["dt"]:cols["dt"] + H] + e["p"]["dt_bias"].double()
    n_band = int(((biased >= 8) & (biased <= 14)).sum())
    print(f"  softplus band: biased dt min={float(biased.min()):.4f} max={float(biased.max()):.4f} "
          f"in [8,14]: {n_band} of {biased.numel()}")
    assert n_band > 0, "softplus-band fixture is VACUOUS (no cell in [8, 14])"
    # A-floor clamp: both clamped and unclamped cells must exist
    e = all_meta["m3_adv_a_floor_b2_l8_d32"]
    meta = e["meta"]
    cols = m3_cols(meta["d_model"])
    H = meta["nheads"]
    dd_A = e["r32"]["in_proj.out"][:, cols["a"]:cols["a"] + H]
    n_clamp = int((dd_A < (1.0 - 1e4)).sum())   # heavy_tail(x) < 1e-4 iff x < 1 - 1e4 = -9999
    n_free = int(dd_A.numel()) - n_clamp
    a_out = e["r32"]["A.out"]
    n_at_floor = int((a_out == np.float32(-M3_A_FLOOR)).sum())
    print(f"  A-floor: clamp binds on {n_clamp} of {dd_A.numel()} cells ({n_free} unclamped); "
          f"A.out == -1e-4 exactly on {n_at_floor} cells (float32)")
    assert n_clamp > 0 and n_free > 0, "A-floor fixture is VACUOUS (contract 8f: the clamp must bind AND not bind)"
    # angle 2pi crossing INSIDE the first chunk, at non-boundary tokens
    e = all_meta["m3_adv_angle_crossing_b1_l48_d32"]
    meta = e["meta"]
    Bb, Ll, Hh = meta["B"], meta["L"], meta["nheads"]
    R = M3_ROPE
    cols = m3_cols(meta["d_model"])
    ip = e["r64"]["in_proj.out"].reshape(Bb, Ll, -1)
    a_rate = torch.tanh(ip[..., cols["angle"]:]) * math.pi
    dt64 = e["r64"]["dt.out"].reshape(Bb, Ll, Hh)
    inc = a_rate.unsqueeze(2) * dt64.unsqueeze(-1)
    TWO_PI = 2 * math.pi
    th = torch.zeros(Bb, Hh, R, dtype=torch.float64)
    engaged = 0
    for t in range(Ll):
        th = th + inc[:, t]
        fl = torch.floor(th / TWO_PI)
        engaged += int((fl != 0).sum())
        th = th - TWO_PI * fl
    print(f"  angle crossing: mod engagements over {Ll} tokens (L < Q = 64, so ALL inside chunk 0): {engaged}")
    assert engaged > 0, "angle-crossing fixture is VACUOUS (contract 8f: a fixture that never crosses 2pi)"
    # trap saturation both directions
    e = all_meta["m3_adv_trap_saturating_b2_l8_d32"]
    sig = e["r32"]["trap.sigma"]
    n0, n1 = int((sig == 0).sum()), int((sig == 1).sum())
    print(f"  trap saturation: sigma exactly 0: {n0}, exactly 1: {n1} of {sig.numel()} (both must be nonzero)")
    assert n0 > 0 and n1 > 0, "trap-saturating fixture is VACUOUS (one-sided)"
    # signed zeros
    e = all_meta["m3_adv_signed_zeros_b2_l8_d32"]
    xn = e["x"].numpy()
    ip64 = e["r64"]["in_proj.out"].numpy()
    ip32 = e["r32"]["in_proj.out"].numpy()
    print(f"  signed zeros: x has {int((xn == 0).sum())} zeros of which {int(np.signbit(xn[xn == 0]).sum())} are -0.0; "
          f"ref64 in_proj.out -0.0 count={int(np.signbit(ip64[ip64 == 0]).sum())}, "
          f"ref32 in_proj.out -0.0 count={int(np.signbit(ip32[ip32 == 0]).sum())}")
    # init-states liveness: theta inside [0, 2pi); chunk decays normal
    e = all_meta["m3_init_states_b1_l65_d32"]
    th0 = e["p"]["init_theta"]
    scale = torch.exp(e["r32"]["dacs.out"][..., -1])
    print(f"  init states: theta_in in [{float(th0.min()):.4f}, {float(th0.max()):.4f}] (must be inside [0, 2pi)); "
          f"exp(dacs_last) float32 min={float(scale.min()):.3e} zeros={int((scale == 0).sum())} of {scale.numel()}")
    assert float(th0.min()) >= 0.0 and float(th0.max()) < TWO_PI

    print("\n== m3 verify 6: the decode fixture's prefix property (DEVIATION 831: the L=70 "
          "prefill IS the decode reference; DEV 832's comparability clause is the negative control) ==")
    e = all_meta["m3_decode_b1_l60p10_d32"]
    Lp = m3_case_by_name("m3_decode_b1_l60p10_d32")["prefix_len"]
    for tag, key, dtype in (("ref64", "r64", torch.float64), ("ref32", "r32", torch.float32)):
        pre = m3_forward(e["p"], e["x"][:, :Lp], dtype)
        full = e[key]
        for s in ("dt.out", "adt.out", "angle.theta", "qkdot.out", "gate.out", "residual.out"):
            a, c = pre[s], full[s][:Lp]
            print(f"  {tag} prefix {s:14s} bit-equal={torch.equal(a, c)} maxabs={float((a - c).abs().max()):.3e}")
        # trap.scale rows 0..Lp-2 are prefix-stable; row Lp-1's beta' leg
        # exists only in the longer run (the trapezoid's seam, contract
        # section 3) -- it MUST differ or the comparability clause's
        # fixture is vacuous.
        ts_pre, ts_full = pre["trap.scale"], full["trap.scale"][:Lp]
        head_eq = torch.equal(ts_pre[:Lp - 1], ts_full[:Lp - 1])
        last_differs = not torch.equal(ts_pre[Lp - 1], ts_full[Lp - 1])
        print(f"  {tag} trap.scale rows 0..{Lp - 2} bit-equal={head_eq}; "
              f"row {Lp - 1} DIFFERS={last_differs} (must be True -- the beta' seam)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--family", choices=["mamba1", "mamba2", "mamba3", "all"], default="all",
                    help="which corpus family to (re)generate; mamba1 output is byte-stable "
                         "and mamba2/mamba3 land in <out>/mamba2/ and <out>/mamba3/, so 'all' is safe")
    a = ap.parse_args()
    print(f"torch {torch.__version__} numpy {np.__version__} python {sys.version.split()[0]}")
    if a.family in ("mamba1", "all"):
        generate(a.out, verify=a.verify)
    if a.family in ("mamba2", "all"):
        m2_generate(a.out, verify=a.verify)
    if a.family in ("mamba3", "all"):
        m3_generate(a.out, verify=a.verify)


if __name__ == "__main__":
    main()
