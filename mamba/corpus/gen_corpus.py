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

Run (see README.md for the scratch venv):
    python mamba/corpus/gen_corpus.py [--out mamba/corpus] [--verify]
`--verify` additionally runs the verbatim selective_scan_ref over the HF-shaped
path with z and D and compares against the composed stages, and regenerates
every file into a temp dir and byte-compares (determinism).
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


def sha256_of_dir(root):
    h = hashlib.sha256()
    for dp, dn, fn in sorted(os.walk(root)):
        dn.sort()
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
    print(f"corpus sha256 (files .f32 .f64 .json): {sha256_of_dir(out_root)}")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    print(f"torch {torch.__version__} numpy {np.__version__} python {sys.version.split()[0]}")
    generate(a.out, verify=a.verify)


if __name__ == "__main__":
    main()
