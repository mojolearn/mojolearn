# The IDENTICAL FP32 Mamba-3 (SISO) block contract

# PROFILE `mojolearn.identical.mamba3.siso.fp32.v1`

Written 2026-09-01, the mamba lane (DEVIATIONS 827-831 claimed; 832-839 of
the assigned band are EXPLICITLY UNCLAIMED and free). The shape of this
document is `mamba/IDENTICAL_MAMBA2_CONTRACT.md`'s, on purpose, and that
contract is the BASELINE: **wherever a numerical question recurs from
Mamba-2 (or Mamba-1 through it), this profile INHERITS the answer
unchanged and does not re-answer it.** Seam numbers `S*` are THIS
document's; Mamba-2's are cited as `mamba2 S17` etc. NOTHING IN THIS
DOCUMENT HAS RUN. No mamba3 .mojo file exists; every gate below is RUN
OWED with its command spelled.

**THE PROFILE NAME IS PART OF THE CONTRACT.** Changing any seam decision
in section 4, any constant in section 3 — **`CHUNK_SIZE = 64` included,
mamba2 DEVIATION 783's standing inherited** — or the stage list in
section 7 creates a v2. FAST is unversioned and makes no identity claim.

The completion claim is one sentence. One Mamba-3 SISO block, FP32, is
bit-identical across Apple, NVIDIA and AMD GPUs at every stage, at every
launch, at every batch composition, and between the prefill and decode
paths, under the declared profile. Not MIMO, not `is_outproj_norm=True`,
not training, not varlen, not bit-identical inference of a model. The
cross-vendor half is earned ONLY by an E-series leg (section 8) and never
by anything built on one machine.

**BUILD ORDER IS SEQUENTIAL BEHIND MAMBA-2 — see section 11.** This is
the orchestrator's staggering decision: the shared segsum / chunk-cumsum /
serial-state-passing substrate and the gemm-cell plumbing must exist ONCE,
in the Mamba-2 implementation round, and be certified (mamba2 contract
section 11 phases 0-2) before any Mamba-3 arithmetic lands on it. This
document exists so that when that gate opens, the builders start from a
contract, not from a reading assignment.

---

## 0. What Mamba-3 is, as deltas from Mamba-2

One line each; sections 1-6 carry the citations.

- **No conv1d.** The d_conv window, S6/S7 of both siblings, is GONE.
  in_proj feeds the SSD directly (mamba3.py:176-206). The conv window
  state piece is gone with it.
- **No dt clamp.** `dt = softplus(dd_dt + dt_bias)` and nothing else
  (mamba3.py:196); `dt_min`/`dt_max` are INITIALIZATION facts (:111-115).
  Do not import mamba2 S9's clamp; `identical_clamp` is still needed —
  by the A seam below.
- **Data-dependent A, per token per head.** `A_log` the parameter is
  gone; `_A = -heavy_tail_activation(dd_A)` then `clamp(max=-A_floor)`
  (mamba3.py:194-195, the activation :27-41). New elementwise seam, S5.
- **Trapezoidal discretization.** A third per-head projection `trap`,
  squashed by sigmoid; token t's K-row weight is
  `scale_t = dt_t*σ(trap_t) + dt_{t+1}*(1-σ(trap_{t+1}))` — the SHIFTED
  load (mamba3_siso_fwd.py:293-306) — with a separate diagonal term and
  a state-resumption correction (S9, S18, S22).
- **State rotation (RoPE-like), data-dependent.** Per-token angle rates
  from in_proj, `tanh(angle)*π`, times dt, accumulated serially mod 2π,
  applied to B and C as interleaved-pair rotations (angle_dt.py:94-117,
  mamba3_siso_fwd.py:327-350). Needs the portable trig PAIR — section 2a.
- **QK-norm-like B/C treatment.** RMSNorm over d_state on B and C with
  learned weight (mamba3.py:126-127, :204-206), then learned per-head
  biases added AFTER the norm, inside the kernel, ones-initialized
  (:121-122; mamba3_siso_fwd.py:312-316).
- **B/C become K/Q of a chunked attention.** The SSD core is spelled as
  strict-causal `(Q·K^T ⊙ decay) @ V` plus a `Q @ state` term plus a
  diagonal `γ·(q·k)·v` term (DEVIATION 830), with the same
  chunk/serial-pass skeleton mamba2 pinned.
- **MIMO exists and is OUT OF SCOPE** (`is_mimo`, mimo_rank, the
  TileLang kernel tree `ops/tilelang/mamba3/`); so is
  `is_outproj_norm=True` (grouped gated norm, group_size=headdim,
  norm_before_gate=True — a grouped fold with no v1 site in either
  sibling). Both refused by name, section 3.
- **No second reference.** HF transformers @ `d56c55b` has NO mamba3
  model (`src/transformers/models/` carries mamba, mamba2, falcon_mamba
  only — verified at the pin). This is a ONE-REPOSITORY profile; the
  independent cross-checks are the two pure-torch references inside the
  pinned repo's own test (section 1), which is weaker than mamba2's
  two-repo position and is said out loud here rather than papered over.

## 1. The reference, pinned

Checkouts live in `/Users/andrewhendel/CascadeProjects/upstream/`. Pin:
state-spaces/mamba `e9594ce` (a single root commit at this checkout; its
message is itself a mamba3 forward FIX — see the fragility note below).

| what | where | role |
|---|---|---|
| the block order, NORMATIVE | `mamba_ssm/modules/mamba3.py::Mamba3.forward` (:160-278): in_proj + 8-way split (:176-186, layout :106-107), heavy-tail A + clamp (:194-195), dt (:196), `ADT = _A * DT` pairing (:197), angle expand-to-heads (:202), B/C norms (:204-206), the SISO call (:249-265), out_proj (:277); `__init__` defaults (:44-70), B/C biases (:121-122), D (:140) | normative |
| the elementwise and whole-block MATH, NORMATIVE | `tests/ops/triton/test_mamba3_siso.py::mamba3_siso_fwd_ref` (:149-340): `tanh(Angles)*π` (:201), `σ(trap)` (:249), angle cumsum + init + mod 2π (:252-257), resumption correction (:261-267), shifted scale (:272-275), bias-then-QK-dot-then-rotate order (:278-288, `_rotary` :216-226), quadratic attention (:296-299), state term contract-then-decay (:301-304), D (:306-307), the QK-dot subtraction (:309), Z-gate (:311-312), final state (:315-321). fp32 by default (:163, :184-197) — the witness that fp32 semantics are upstream-spelled | normative for VALUES; its whole-sequence SCHEDULE and its diagonal SPELLING are replaced (DEVIATIONS 827, 830) |
| the chunked schedule SHAPE | `mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py::mamba3_siso_fwd_kernel`: phase 1 preprocessing (:276-351 — shifted dt/trap loads :293-302, scale :304-306, pre-rotation QK dot :319-325, rotations :332-335/:347-350, final-K stored post-rotation pre-scale :337-341, K scaled :343), phase 2 (:353-451 — resumption correction :367-371, strict-causal mask :413-417, diagonal-plus-D add :421-422, Z-gate :430-431, serial state update :440-444); wrapper's final-state pick (:709-729: k at `(seqlen-1) % chunk_size`, v = last token raw) | normative for STRUCTURE; its exp2·log2e respelling (:386, :407, :412, :440-442), its `min(·, 0.0)` guard (:412 — inert, since ADT <= -A_floor·dt <= 0 by S5/S6), its `tl.sum` for da_cs_last (:397) and its `tl.cumsum`/`tl.dot` trees are NOT pinned — the mamba2 S11/S12/DEVIATION-782 spellings are inherited over them |
| the angle chain | `mamba_ssm/ops/triton/mamba3/angle_dt.py::angle_dt_fwd_kernel` (:83-122): `tanh_approx(angle)*π` (:94), `*dt` (:101), chunked cumsum (:104-105), mod 2π as `x - 2π*floor(x/2π)` (:108), state modded per chunk (:115-117) | shape and mod SPELLING; its PER-CHUNK mod placement is refused (DEVIATION 829) |
| per-token recurrence SEMANTICS | `test_mamba3_siso.py::mamba3_siso_step_ref` (:34-146): the three-term update `S = α·S + β·(k_prev⊗v_prev) + γ·(k⊗v)` with `α = exp(adt)`, `β = (1-σ(trap))·dt·α`, `γ = σ(trap)·dt` (:119-127), per-token angle mod (:109-111); the module's `step` (mamba3.py:314-440, CuteDSL `mamba3_step_fn`, "Only tested on H100" :320) and `ops/triton/mamba3/mamba3_siso_step.py` | SEMANTICS ONLY; the rounding is the `STEP_UPSTREAM_RECURRENCE` required-RED arm (DEVIATION 831) |
| decode rotation, "what vendors run" | `mamba_ssm/ops/triton/mamba3/mamba3_mimo_rotary_step.py::rotary_qk_inference_kernel`: tanh respelled as `sigmoid(2x)*2-1`, `tl.cos`/`tl.sin`, and the updated angle state stored WITHOUT the mod-2π reduction | cited as a THIRD trig spelling and a mod-placement divergence; refused (DEVIATION 829). The prefill kernel's PTX `cos.approx`/`sin.approx`/`tanh.approx` (`ops/triton/mamba3/utils.py`:13-69) are a SECOND; the test refs' torch.cos/sin/tanh the first |
| the shipped surface's dtype | `mamba_ssm/ops/triton/mamba3/mamba3_siso_combined.py::mamba3_siso_combined` (:390-399): Q/K/V/Trap/Angles/Z force-cast to bfloat16 before the kernel | REFUSED — the profile is Float32 everywhere (Andrew's order, as mamba2); `mamba3_siso_fwd_ref`'s fp32 default is the witness that this is a surface fact, not the math |
| the B/C norm | `mamba_ssm/ops/triton/layernorm_gated.py::rms_norm_ref` (:18-39) at `z=None`, `group_size=None`: `rstd = 1/sqrt(mean(x²)+eps)` (:29), `x*rstd*weight` (:30); class `RMSNorm` (:415-437), no bias (:425) | mamba2 S1-S3/S21 machinery INHERITED; eps 1e-5 by construction (mamba3.py:126-127), not the ref's 1e-6 default |
| the residual/norm wrapper | `mamba_ssm/modules/block.py::Block.forward`, non-fused arm (:51-53, :67): `residual = hidden + residual`, `hidden = norm(residual)`, then mixer | mamba2 S1-S3/S22, INHERITED (mamba2 cited HF's Mamba2Block; with no HF mamba3, the in-repo Block is the citation — same arithmetic) |
| the projections' arithmetic | profile `mojolearn.identical.gemm.fp32.v1` (`gemm/IDENTICAL_FP32_CONTRACT.md`), certified three-vendor | this repository, INHERITED |
| exp, softplus, silu, sigmoid, tanh, div, rsqrt, trig | `checks/numerics.mojo`: `identical_exp`, `identical_softplus` (the `<= 20` guard — `F.softplus`'s own default threshold, so mamba3.py:196 spells the same guard through torch), `identical_silu` (ONE-division), `identical_sigmoid`, `identical_div`, `identical_rsqrt`, `identical_tanh` (DEVIATION 821, `portable_tanhf`), `portable_cosf` and `portable_sinf` (DEVIATION 820, shared `_cephes_sincosf_core`, domain |x| < 8192, gate `checks/portable_trig_check.mojo`) | this repository, INHERITED where certified; section 2a for what is still owed |

**Fragility note, from the pin itself.** The pin's HEAD commit message
records that `mamba3_siso_fwd_kernel` SILENTLY corrupted its forward
output on Blackwell at `num_stages > 1` (a Triton codegen fault across
three Triton versions), fixed by pruning the autotune space
(`_prune_mamba3_siso_fwd_configs`, mamba3_siso_fwd.py:19-31). Upstream's
own forward was execution-config-dependent and WRONG on one vendor
generation with no error raised. That is this contract's whole thesis
stated by the opposition: the execution plan must not be able to touch
the arithmetic (gemm contract section 6.1, mamba2 DEVIATION 784), and a
vendor's green run is not a correctness argument.

## 2. What one block call is

    hidden = residual + mixer(rmsnorm(residual))          Block.forward (:51-53, :67)

with mixer, in order (`Mamba3.forward`, SISO, defaults):

    in_proj -> split(z | x | B | C | dd_dt | dd_A | trap_raw | angle_raw)   [mamba3.py:106-107, :176-186]
    -> A = clamp(-heavy_tail(dd_A), max=-A_floor)                 [seam S5]
    -> dt = softplus(dd_dt + dt_bias)   (NO clamp)                [seam S6]
    -> ADT = A * dt                                               [seam S7]
    -> B/C RMSNorm over d_state (learned weight, eps 1e-5)        [seam S21]
    -> per token: σ(trap); γ, β', scale (the trapezoid)           [S8, S9]
    -> per token: θ += tanh(angle_raw)*π * dt, mod 2π (serial)    [S10]
    -> q = C + C_bias, k = B + B_bias  (bias AFTER the norm)      [S12]
    -> qk_dot = (q·k) BEFORE rotation, times γ                    [S14, DEVIATION 830]
    -> rotate q, k by θ (interleaved pairs, first 32)             [S13, DEVIATION 828]
    -> k_scaled = k_rot * scale                                   [S15]
    -> chunked SSD core (Q = CHUNK_SIZE = 64):
         per chunk: da_cs serial cumsum of ADT [mamba2 S11],
           Y_state = (q_rot · h_cstart) * exp(da_cs)              [S17]
           Y_intra = (strict-causal (q_rot·k_scaled^T) ⊙ exp(seg)) @ v   [S16]
         across chunks, SERIAL: h = fma(exp(da_cs_last), h, (v ⊙ exp(da_cs_rev))^T · k_scaled)  [S20, mamba2 S17's answer]
    -> Y += (D[h] + qk_dot_γ) * v      (diagonal + skip in ONE add)  [S18]
    -> Y *= silu(z)                    (gate; no output norm at defaults) [S19]
    -> out_proj                                                   [S4]

Inference only. Decode is DEVIATION 831. Weights carry no bias anywhere
(`in_proj`/`out_proj` bias=False, mamba3.py:108/:157; there is no conv).
dt/B/C-bias initializations (:111-122) are INITIALIZATION facts; weights
are inputs to this profile.

## 2a. New-primitive dependencies, named

| primitive | status | needed by |
|---|---|---|
| `portable_sinf` / `portable_cosf` | EXIST in `checks/numerics.mojo` (DEVIATION 820 built sinf and refactored cosf onto the shared core); gate `checks/portable_trig_check.mojo` / `pixi run` target per that file. CERTIFICATION of the pair on the device columns is a PRECONDITION of phase 2 — RUN OWED, not assumed | S13 rotation |
| `identical_tanh` | EXISTS (DEVIATION 821, `portable_tanhf`) | S10 angle squash |
| `identical_sigmoid` | EXISTS (row 52, `portable_sigmoidf`) | S8 trap |
| `identical_clamp` | DOES NOT EXIST. Already requested by mamba2 DEVIATION 788 (phase 0 of ITS build order); this profile is the SECOND consumer and the FIRST whose bound ALWAYS binds structurally (S5's `max=-A_floor`) | S5 |
| mod-2π reduction | NEW SPELLING, not a new function: composed from `identical_div`, exact `floor`, `pinned_mul`, one subtract (DEVIATION 829). No request to the identity lane needed | S10 |
| π / 2π constants | `Float32(π) = 0x40490FDB`, `Float32(2π) = 0x40C90FDB`, pinned BY BITS in the oracle and kernels; the upstream float64 literal `3.141592653589793` rounds to these in every f32 context the references use | S10, S13 |

The stale sentence in `IDENTICAL_SSM_NOTES.md` ("only `portable_cosf`
exists", "`portable_sinf` not asked for") predates DEVIATION 820 and is
flagged to the orchestrator for the fix-docs-on-discovery pass; that file
is not this contract's to edit.

## 3. Profile constants

| constant | value | source |
|---|---|---|
| d_state (N) — the QK head dim | 128 | mamba3.py:47 default |
| expand | 2 | :48; d_inner = 2·d_model |
| headdim (P) — the V head dim | 64 | :49; H = d_inner / headdim |
| ngroups = num_bc_heads (G) | 1 | :50, :95; B/C shared across heads by COPY, but rotation is PER HEAD (dt is per-head), so post-rotation K/Q DIFFER per head — the broadcast is a copy, what follows it is arithmetic |
| rope_fraction | 0.5 | :53, asserted in {0.5, 1.0} (:98); split_tensor_size = 64, **num_rope_angles = 32** (:100-103); pairs 32..63 of the 64 interleaved pairs are UNROTATED (structural cos=1/sin=0, DEVIATION 828) |
| A_floor | 1e-4 | :57; the S5 clamp bound. BINDS only when dd_A < -9999 (heavy_tail(x) < 1e-4 ⇔ x < 1-1e4) — present always, witnessed by a planted fixture |
| is_mimo / mimo_rank | False / 1 | :59, :87-88; MIMO refused by name |
| is_outproj_norm | False | :58; the Z-gate is applied raw in the core (S19); True (grouped gated norm :143-150) refused by name |
| **CHUNK_SIZE (Q)** | **64** | :64 default ("Recommended: 64 for SISO"). PART OF THE ARITHMETIC — mamba2 DEVIATION 783's standing inherited, at the NEW value |
| B/C norm eps | 1e-5 (`0x3727C5AC`) | :126-127 (construction overrides rms_norm_ref's 1e-6 default) |
| block norm eps | per config `layer_norm_epsilon`, default path as mamba2 | block.py; inherited |
| softplus threshold | 20.0, `<=` | `identical_softplus` inherited; F.softplus's own default threshold (mamba3.py:196) |
| in_proj layout | z, x, B, C, dd_dt, dd_A, trap, angle — widths d_inner, d_inner, G·N, G·N, H, H, H, 32 | mamba3.py:106-107, :177-186 |
| dtype | Float32 everywhere (weights, activations, all four state pieces) | Andrew's order; the shipped surface's bf16 casts (mamba3_siso_combined.py:390-399) REFUSED |

`is_mimo=True`, `is_outproj_norm=True`, `fuse_pregate_headwise_norm`,
`rope_fraction=1.0`, `ngroups > 1`, varlen (`cu_seqlens`/`seq_idx`) and
non-Float32 are REFUSED BY NAME, not silently defaulted. The inventory of
mamba3.py's surface knobs is product scoping and is DEFERRED to a
`FEATURE_PARITY.md` addendum row (the existing Mamba-3 row at :143 is the
anchor); this contract scopes arithmetic only.

**Invariance clause.** Bit-identity under this profile is invariant to
batch size, launch geometry, padding, autotune/execution config and
vendor. It is NOT invariant to CHUNK_SIZE, rope_fraction, A_floor, or any
constant above — each changes the bits and makes a v2.

**Padding is part of the arithmetic's shape**, as mamba2 section 3: the
last chunk is padded to Q with `+0.0` q/k/v/dt/angle rows; a padded
position has scale = +0.0 (its dt and its successor's dt are +0.0), so it
contributes exactly-zero products to every fold. σ(trap) of a padded
position is 0.5 and is INERT (it only ever multiplies a +0.0 dt).
**The shifted loads cross chunk boundaries but never the sequence end**:
`dt_{t+1}`/`trap_{t+1}` at the LAST REAL TOKEN are +0.0/inert
(mamba3_siso_fwd.py:294-296 masks at seqlen; fwd_ref :272-273 pads), so a
sequence's last K-row carries only its γ leg — the β' leg belongs to the
NEXT call (S22) and that is the trapezoid's seam, not a bug.

## 4. The seams

Rules inherited verbatim from mamba2 section 4: FMA contraction PER SEAM;
FUSED = one `identical_mul_add` rounding; PRODUCT = `pinned_mul`; every
seam result and every buffer load through `ftz`; copies (the 8-way split,
the head broadcast of B/C/angles, reshapes, state stores) are NOT seams.

| # | seam | reference spelling | pinned spelling | fused? | inherits |
|---|---|---|---|---|---|
| S1-S3 | block RMSNorm | Block.forward :53 | mamba2 S1-S3 VERBATIM | as mamba2 | mamba2 S1-S3 |
| S4 | in_proj, out_proj, and every core contraction of S14/S16/S17/S20 | `nn.Linear`; `tl.dot`/einsum | ALL gemm.fp32.v1 cells, `OP_NT` per call site. Cells and k: in_proj (k=d_model), out_proj (k=d_inner), qk_dot (k=N=128, one serial leaf), q·k^T (k=128), (M)·v (k=Q=64), q·h^T (k=128), v^T·k_scaled (k=Q=64) | per gemm v1 | mamba2 S4 / DEVIATION 784 |
| S5 | data-dependent A | `-heavy_tail_activation(dd_A)` then `clamp(max=-A_floor)` (mamba3.py:194-195; the function :39-41) | piecewise: x >= 0 → `ftz(1.0 + x)`; x < 0 → `identical_div(1.0, ftz(1.0 - x))`; then negate (exact); then `identical_clamp(·, -inf, -A_floor)`. The reference's branchless `clamp_min + reciprocal(1-clamp_max)` sum is bit-equal to the branch spelling (the inactive term is an exact +0.0 or the exact `x + 1` commutation) — recorded 782-style, spelling differs, bits do not | n/a | clamp primitive: mamba2 DEVIATION 788 |
| S6 | dt | `F.softplus(dd_dt + dt_bias)` (mamba3.py:196) — NO clamp | `identical_softplus(ftz(dd_dt + ftz(bias[h])))`. mamba2 S9 minus its clamp; do not add one | add then n/a | mamba1 S14 |
| S7 | ADT | `_A * DT` (mamba3.py:197) | `pinned_mul(A, dt)` per (token, head). Always <= -A_floor·dt <= 0, so every exp below is in (0, 1] — the kernel's `min(·, 0.0)` guard (fwd:412) is INERT and not pinned. This is NOT mamba2's DEVIATION-789 question: dt never binds to x/V here at all; it enters through γ/β'/scale instead | PRODUCT | — |
| S8 | trap sigmoid | `torch.sigmoid` (fwd_ref :249; step path mamba3.py:285); kernel `tl.sigmoid` (utils.py:114 — the PTX tanh spelling is commented OUT at the pin) | `identical_sigmoid(trap_raw)`, both the t and t+1 instances | n/a | row 52 |
| S9 | trapezoid scales | `gamma = dt*trap`, `shifted_gamma = dt₊*(1-trap₊)`, `scale = gamma + shifted_gamma` (fwd:304-306; ref :272-275) | `γ = pinned_mul(dt_t, σ_t)`; `β' = pinned_mul(dt_{t+1}, ftz(1.0 - σ_{t+1}))`; `scale = ftz(γ + β')`. Shifted operands +0.0 past the sequence end (section 3) | PRODUCTs + add | — |
| S10 | angle recurrence | three disagreeing spellings — DEVIATION 829 | `a = pinned_mul(identical_tanh(angle_raw), π)`; `inc = pinned_mul(a, dt_t)`; `θ_t = mod2π(ftz(θ_{t-1} + inc))` SERIAL over tokens, per (head, angle-index), θ₋₁ = angle state in (or +0.0); `mod2π(x) = ftz(x - pinned_mul(2π, floor(identical_div(x, 2π))))`, floor exact | add | DEVIATION 829 |
| S11 | cos/sin of θ | `torch.cos/sin` (ref :285-286); PTX approx (fwd:328-329); `tl.cos/sin` (decode rotary) | `portable_cosf(θ)`, `portable_sinf(θ)`; θ ∈ [0, 2π) ⊂ the pair's |x| < 8192 domain BY CONSTRUCTION of S10's mod | n/a | DEVIATION 820's pair |
| S12 | Q/K bias add | after the B/C norm, before rotation (fwd:312-316; ref :278-279) | `ftz(q + qb[h,n])`, `ftz(k + kb[h,n])` | add | — |
| S13 | rotation | interleaved pairs (2i, 2i+1), `x0·c - x1·s` / `x0·s + x1·c` (ref :216-226; fwd:332-335, :347-350) | `ro0 = ftz(ftz(pinned_mul(x0,c)) - ftz(pinned_mul(x1,s)))`; `ro1 = ftz(ftz(pinned_mul(x0,s)) + ftz(pinned_mul(x1,c)))` — UNFUSED, the references' two-product spelling. **THE INNER `ftz` IS THE THING THAT MAKES IT UNFUSED, added 2026-09-03.** This row already said UNFUSED and the code already said `ftz(pinned_mul(x0,c) - pinned_mul(x1,s))`, and that spelling DOES NOT ENFORCE IT: `pinned_mul(a,b)` is `identical_mul_add(a,b,-0.0)`, which a backend may simplify to `a*b` and then re-contract into `fma(a,b,-(c*d))` — one rounding where this row asks for three. It cost four stages of the AMD card (rot.k, kscale.out, ssd.h_last, ssd.k_last), and it was AMD that was RIGHT: its device produced the unfused answer, while Apple and NVIDIA contracted and every host oracle contracted with them, so the majority agreed with itself and the honest column was logged as the divergence. `ftz`'s bitcast-compare-select gives the `fmul` a second use and hands the subtraction a `select`, which no backend can fuse; it is bit-inert for normal values and is already the spelling S12 above uses. Pairs >= num_rope_angles: STRUCTURAL identity (never computed), DEVIATION 828 | PRODUCTs + add/sub | DEVIATION 828 |
| S14 | qk_dot | pre-rotation `sum(q*k)` then times γ (fwd:319-325; ref :282 uses β' instead — DEVIATION 830 picks γ) | gemm v1 cell over n, k = 128, ONE serial leaf; then `pinned_mul(·, γ_t)` | per gemm v1; PRODUCT | DEVIATION 830 |
| S15 | K scaling | `k_pre_block *= scale` AFTER rotation, AFTER the final-K-state store (fwd:337-344) | `pinned_mul(k_rot, scale_t)` per element; the carried k-state is the PRE-scale value (fwd:498) | PRODUCT | — |
| S16 | intra-chunk attention | `tl.dot(q,k^T)`, decay, strict mask, `tl.dot(·,v)` (fwd:411-418); ref quadratic (:296-299) | `s[i][j]` = gemm cell k=128; `L[i][j] = ftz(identical_exp(seg[i][j]))` for j < i, seg rebuilt SERIAL ASCENDING per mamba2 DEVIATION 782, STRUCTURAL +0.0 for j >= i — the DIAGONAL is structural too (DEVIATION 830 moves it to S14/S18); `M = pinned_mul(s, L)`; `Y_intra` = gemm cell over j, k = Q = 64 | per gemm v1; PRODUCT | mamba2 S12-S14 / DEVIATION 782 |
| S17 | state read-out | `dot(q, h^T) * exp(da_cs)` (fwd:406-407; ref :301-304 contract-then-scale) | gemm cell over n, k = 128, h = the chunk's INCOMING state; then `pinned_mul(·, ftz(identical_exp(da_cs_i)))` — da_cs INCLUSIVE of token i, the serial cumsum of S7's ADT per mamba2 S11 (the kernel's separate `tl.sum` for da_cs_last is refused under the same inheritance: da_cs_last IS the serial cumsum's last value) | per gemm v1; PRODUCT | mamba2 S11/S18 |
| S18 | diagonal + D skip | `acc_o += (D + qk_dot)*v` (fwd:421-422) | `t = ftz(D[h] + qkγ_i)`, `p = pinned_mul(t, v)`, `Y = ftz(Y + p)` — ONE add folds skip and diagonal, the kernel's own association | add; PRODUCT; add | mamba2 S20's D-last principle |
| S19 | Z gate | `out * silu(z)` in-core (fwd:430-431; ref :311-312 spells `z*σ(z)`) | `pinned_mul(Y, identical_silu(z))` — the ONE-division silu, the recurring silu question answered by mamba1 DEVIATION 744, inherited | PRODUCT | mamba1 744 |
| S20 | inter-chunk state pass, SERIAL | `acc = acc*exp2(da_cs_last) + dot((v*exp2(da_cs_rev))^T, k_scaled)` (fwd:439-444) | `d_rev = ftz(da_cs_last - da_cs_i)`; `pinned_mul(v, identical_exp(d_rev))`; increment = gemm cell over chunk rows, k = Q = 64, output [P, N]; `h = ftz(fma(identical_exp(ftz(da_cs_last)), h, increment))` — ONE rounding, mamba2 S17's fused answer to the identical decay·state+increment question | per gemm v1; FUSED | mamba2 S15-S17 / DEVIATION 785 |
| S21 | B/C RMSNorm | rms_norm_ref :29-30 at z=None, group_size=None, eps 1e-5 | mamba2 S1-S3 fold/rstd/product machinery over the N=128 row per (b, l, g); `identical_rsqrt` = 1/sqrt (the rsqrt question, answered mamba1 DEVIATION 741, inherited); weight product `pinned_mul`; NO gate, NO bias | as S1-S3 | mamba1 741 |
| S22 | resumption correction | `acc += V_st ⊗ K_st * dt_1 * (1-trap_1)` (fwd:367-371); ref folds the scalar FIRST (:266-267) | the NORMATIVE ref's association: `c = pinned_mul(dt_1, ftz(1 - σ(trap_1)))`; `t = pinned_mul(pinned_mul(v_st, k_st), c)` per (p, n); `h₀ = ftz(h_in + t)`. The kernel's `((v·k)·dt)·(1-σ)` association is refused silently — the normative reference wins where no argument overrides it | PRODUCTs + add | house rule |
| S23 | residual add | Block.forward :52/:67 | mamba2 S22 VERBATIM | add | mamba2 S22 |

The softplus in-band fixture note (plant in [8, 14], not at 20) applies
verbatim from the siblings.

## 5. State, decode, and what "resumable" means here

The upstream carried state is FOUR pieces (`allocate_inference_cache`,
mamba3.py:442-482): θ (angle state, [B, H, 32], fp32), h (SSM state,
[B, H, P, N], fp32), k_last (post-bias post-rotation PRE-scale,
[B, 1, H, N]), v_last ([B, H, P]). NO conv window exists.

DEVIATION 831 splits "resumable" into two claims the references conflate:

1. **Decode == prefill, bitwise, per token** is earned ONLY by the
   mamba2-DEVIATION-786 construction: an INTRA-CHUNK BUFFER (the open
   chunk's <= Q rows of rotated-unscaled q/k, v, dt, σ(trap), ADT, plus
   the four report pieces at the last chunk boundary), each decode step
   re-running the padded folds a prefill ending at that token runs. The
   trapezoid FITS this construction: a prefill of length t gives token
   s < t the scale `γ_s + β'_{s+1}` using only tokens <= t, so per-token
   outputs are prefix-stable and the resumption replay reproduces them.
2. **The upstream `Input_States` continuation** (four tensors in, S22's
   correction) is SUPPORTED, transliterated, corpus-checked — and NOT
   claimed bit-equal to an unbroken prefill, not even at a chunk
   boundary. The argument: in an unbroken run the boundary token's K-row
   is folded ONCE at `pinned_mul(k, ftz(γ+β'))`; across the seam it is
   folded at `pinned_mul(k, γ)` in call 1 with `β'`'s leg arriving as
   S22's separate product-and-add in call 2 — `ftz(a·(γ+β')) ≠
   ftz(a·γ) + ftz(a·β')` in general. mamba2's gate-d2 boundary-handoff
   bit-equality does NOT carry over; its analog here is a TOLERANCE
   corpus row, never a bitwise gate. Anyone who "fixes" gate D2m3 to
   bitwise has misread the trapezoid.

The upstream step spellings (`mamba3_siso_step_ref`'s three-term α/β/γ
recurrence, the CuteDSL `mamba3_step_fn`, `mamba3_siso_step.py`) round
differently from the chunked prefill and are kept as the required-RED
arm `STEP_UPSTREAM_RECURRENCE`, exactly the sibling's move.

DEFERRED, not refused: varlen (`cu_seqlens`, batch-1 concatenated),
MIMO, `is_outproj_norm`, multi-block caches. Recorded so a later round
starts from a sentence.

## 6. NaN, infinity, signed zero, denormals

Mamba-1 contract section 6 applies VERBATIM (refusal by name and BITS,
`refuse_nonfinite` before any recorded stage, `-0.0` admitted, `+0.0`
fold seeds, `ftz` everywhere), plus:

- Structural zeros (S16's j >= i triangle including the diagonal, S13's
  unrotated pairs) never exist in a buffer; the references' `-inf` mask
  (test `_segsum` :30) never exists here — mamba2 DEVIATION 782's
  bit-equality argument covers the strict triangle the same way.
- S5's clamp and S10's mod are the profile's ordered-select and
  floor-select on floats; both are built from the named primitives with
  defined zero-sign and NaN behavior (DEVIATION 788's requirement), never
  vendor min/max/fmod.
- θ ∈ [0, 2π) by construction keeps S11 inside the trig pair's domain; a
  θ that leaves it can only come from nonfinite inputs, which the
  refusal gate has already rejected.

## 7. The stages, in card order

Token-major `M = B*L`; H = nheads, P = 64, N = 128, Q = 64,
C = ceil(L/Q), R = num_rope_angles = 32.

    input.x        [M, d_model]
    norm.sumsq     [M]                 S1
    norm.out       [M, d_model]        S2-S3
    in_proj.out    [M, d_in_proj]      S4; columns per mamba3.py:106-107
    A.out          [M, H]              S5 (clamped)
    dt.out         [M, H]              S6
    adt.out        [M, H]              S7
    trap.sigma     [M, H]              S8
    trap.scale     [M, H]              S9 (γ and β' recoverable; scale carded)
    bcnorm.B       [M, G, N]           S21
    bcnorm.C       [M, G, N]           S21
    angle.theta    [M, H, R]           S10 (post-mod)
    rot.q          [M, H, N]           S12-S13 (post-bias post-rotation)
    rot.k          [M, H, N]           S12-S13 (PRE-scale)
    qkdot.out      [M, H]              S14 (γ-scaled)
    kscale.out     [M, H, N]           S15
    dacs.out       [B, H, C, Q]        mamba2 S11 inherited
    seg.L          [B, C, H, Q, Q]     S16 decay, +0.0 on and above the diagonal
    yintra.out     [M, H, P]           S16
    ystate.out     [M, H, P]           S17
    skip.out       [M, H, P]           S18
    gate.out       [M, H, P]           S19
    out_proj.out   [M, d_model]        S4
    residual.out   [M, d_model]        S23
    ssd.h_last     [B, H, P, N]        section 5 report
    ssd.k_last     [B, H, N]           section 5 report
    ssd.v_last     [B, H, P]           section 5 report
    ssd.theta_last [B, H, R]           section 5 report

Size-capped elisions stated on the card, never silent (mamba2 rule).

## 8. What "identical" is gated to mean

The mamba2 gate list transposed; every command RUN OWED.

(a) card == host oracle card, bitwise, every stage.
    `pixi run check-mamba3-block` (builders register
    `mamba/checks/mamba3_check.mojo`, shape via env, one compile at a
    time).
(b) repeat-run identity, 8 launches.
(c) batch-composition invariance with the sliced-input negative control.
(d) decode == prefill per token per stage via the DEVIATION-831 buffer
    construction, with the misalignment control, INCLUDING a decode run
    crossing a chunk boundary (prefill L=60, decode through token 70;
    Q=64). NO bitwise d2 arm exists (section 5, claim 2); the
    `Input_States` continuation is corpus-checked under tolerance
    instead.
(e) nonfinite refusal with reach measured, plus the clean-call control.
(f) sabotage arms, DELTAS from mamba2's list (inherited arms that
    survive: SEGSUM_DESCENDING, FOLD_SERIAL_ZERO_SEED, the gemm arms;
    retired with the conv: S6_BIAS_LAST, S6_TAPS_REVERSED; retired with
    the clamp-less dt: CLAMP_BEFORE_SOFTPLUS; retired with the changed
    pairing: PAIR_DT_B):

    | arm | must move | witnessed by |
    |---|---|---|
    | CHUNK_SIZE_32 (Q rebuilt at 32) | yintra.out onward | L > 32 (DEVIATION 783's falsifier at the new Q) |
    | DIAG_INCLUDE_SUBTRACT (the fwd_ref diagonal: inclusive tril at full scale, then subtract β'·(q·k)·v) | skip.out | any fixture with nonzero angles — rotation rounding separates q_rot·k_rot from q·k (DEVIATION 830's falsifier) |
    | ANGLE_MOD_PER_CHUNK (angle_dt.py's placement) | angle.theta onward | planted near-saturated angles, dt near 0.1, L >= 32 — a 2π crossing INSIDE a chunk; a fixture that never crosses 2π is vacuous for this arm and must be shown to cross |
    | ANGLE_MOD_AT_END (fwd_ref's placement) | angle.theta onward | same fixture |
    | ROTATE_HALF_SPLIT (the MIMO (i, i+N/2) pairing on SISO — mamba3.py:360-363's own note) | rot.q/rot.k onward | any nonzero-angle fixture with non-degenerate q |
    | TRAP_LEFT_ONLY (scale = γ only; Euler, not trapezoid) | kscale.out onward | nonuniform trap |
    | A_FLOOR_UNCLAMPED (S5 clamp dropped) | A.out onward | planted dd_A = -20000 (the ONLY region where the clamp binds; an unplanted fixture is vacuous) |
    | STEP_UPSTREAM_RECURRENCE (step_ref :119-127 spelling in decode) | gate (d) per token | L >= 2 decode |
    | RESUME_KERNEL_ASSOC (S22 spelled with the kernel's fwd:371 association) | corpus continuation rows | nonzero Input_States fixture |
    | STATE_TERM_SCALE_FIRST (S17 spelled decay-into-q before the contraction) | ystate.out | L > Q with nonzero incoming state |

    Each arm verified to move its own stage and no earlier one, on a
    fixture that WITNESSES it.
(g) corpus cross-check: `mamba/corpus/` mamba3 cases, torch float64
    per-stage references generated by `mamba3_siso_fwd_ref` and
    `mamba3_siso_step_ref` VERBATIM (copied, cited by line), hashed
    inputs, per-case tolerances from `--self-test`. Adversarial cases
    owed by name: softplus band [8,14]; dd_A near 0 AND at -20000;
    trap saturating both directions; angle 2π-crossing; signed-zero
    plants BY SIGN BIT; nonzero-Input_States continuation (tolerance,
    per section 5); L in {1, 4, 63, 64, 65, 129, 257}. 
    `pixi run check-mamba3-corpus`. RUN OWED.

FAST-mode arms recorded, not asserted. Nothing above earns a
cross-vendor sentence; only the E-series leg does.

## 9. Deviations 827-831 (832-839 unclaimed)

- **DEVIATION 827 — the chunked schedule is the kernel's, not the
  reference's.** The NORMATIVE math reference (`mamba3_siso_fwd_ref`)
  computes each sequence as ONE quadratic attention with no chunking
  (:296-299) — a different summation order from any chunked run from
  L > Q up. The profile pins the shipped kernel's two-phase chunked
  schedule (per-chunk S16/S17, serial S20) with Q = 64, exactly as
  mamba2's DEVIATION 785 pinned the kernel's serial pass over
  ssd_minimal's einsum. The unchunked spelling is kept as the corpus
  oracle (float64, tolerance), never the identity oracle. CHUNK_SIZE's
  standing is DEVIATION 783's, inherited; CHUNK_SIZE_32 is the
  falsifier at the new value.
- **DEVIATION 828 — the rotation seam and the portable trig pair.**
  Upstream spells the rotation's trig THREE ways at the one pin (torch
  cos/sin in the refs; PTX `cos.approx`/`sin.approx` in the prefill
  kernel, utils.py:13-50; `tl.cos`/`tl.sin` in the decode rotary
  kernel) — mutually bit-incompatible, so upstream's own prefill and
  decode already disagree. The profile pins `portable_cosf`/
  `portable_sinf` (DEVIATION 820's pair, one shared core) on BOTH
  paths, the interleaved (2i, 2i+1) pairing (the SISO spelling;
  mamba3.py:360-363 records half-split as the MIMO permutation —
  ROTATE_HALF_SPLIT is the arm), the UNFUSED two-product spelling of
  each rotated component, and STRUCTURAL identity for pairs >=
  num_rope_angles (the references' pad-with-cos-1/sin-0 and
  mask-angle-to-0 spellings both compute trig of nothing real; the
  structural spelling agrees with `cos(+0.0) = 1.0`, `sin(+0.0) = +0.0`
  exactly, so the three spellings' bits coincide — recorded 782-style).
- **DEVIATION 829 — the angle recurrence, serial per-token mod-2π.**
  Upstream places the mod-2π reduction THREE ways: per chunk on both
  running state and outputs (angle_dt.py:108, :117), once at the end of
  the whole cumsum (fwd_ref :253-257), and NOWHERE (the decode rotary
  kernel stores the advanced angle state unreduced) — three different
  bit-streams, and the third drifts toward the trig domain boundary.
  The profile pins the per-token serial recurrence with the mod applied
  every step (the step_ref :109-111 placement): one spelling for
  prefill and decode, bounded θ by construction, DEVIATION 831 made
  satisfiable. `mod2π` is composed, not a primitive: `identical_div`,
  exact floor, `pinned_mul` by the pinned 2π bits, one subtract. tanh
  through `identical_tanh` (the decode kernel's `sigmoid(2x)*2-1`
  respelling is refused with the rest). ANGLE_MOD_PER_CHUNK and
  ANGLE_MOD_AT_END are the arms, each requiring a witnessed 2π
  crossing.
- **DEVIATION 830 — the diagonal term rides the pre-rotation dot.**
  The kernel deliberately excludes the diagonal from the attention
  matrix (strict `>` mask, fwd:413-417) and adds it as
  `γ·(q_pre·k_pre)·v` (fwd:319-325, :421-422), its comment naming the
  reason ("prevent non-causal numerical leakage"); the test reference
  instead includes the diagonal at full `scale` and SUBTRACTS
  `β'·(q_pre·k_pre)·v` (:282, :297, :309). Equal in exact arithmetic
  (rotation preserves the dot), different bits. The profile takes the
  kernel's spelling — the one case in this contract where the kernel
  overrides the normative reference, because the kernel's version is a
  documented design decision that also removes a rotation-rounding
  dependency from the diagonal. DIAG_INCLUDE_SUBTRACT is the
  required-RED arm.
- **DEVIATION 831 — decode is prefill resumption; boundary handoff is
  NOT bitwise.** Section 5. The buffer construction (mamba2 DEVIATION
  786 inherited as the principle) earns per-token decode==prefill; the
  four-piece upstream state is a REPORT plus the S22 continuation
  input, and the trapezoid's split-scale rounding makes the
  continuation tolerance-checked, never bit-gated — with the
  one-rounding-versus-two argument written out so the missing gate is
  read as a theorem, not an omission. STEP_UPSTREAM_RECURRENCE and
  RESUME_KERNEL_ASSOC are the arms.

## 10. Not claimed

- "Bit-identical AI inference" is not claimed; ONE Mamba-3 SISO block is.
- Written before anything ran. As of 2026-09-02 the Apple column has run
  green (RUN RECORD 2026-09-01) and the AMD column has run RED on gate (a)
  (RUN RECORD 2026-09-02); NVIDIA is still unrun, and every remaining gate
  keeps its RUN OWED. NO cross-vendor identity sentence exists for this
  profile and none can until an E-series leg prints one -- and the AMD red
  is the reason there is none to print today.
- No mamba1/mamba2 claim is extended by this document and none of theirs
  earns anything here; the profiles are siblings, not transitive.
- No MIMO, no `is_outproj_norm`, no varlen, no training/backward, no
  BF16/FP16 (the shipped surface's own bf16 casts are refused, not
  reproduced), no multi-block model, no tokenizer, no CuteDSL/TileLang
  surface, no performance number (IDENTITY IS NOT FREE on any vendor).
- The upstream `Input_States` boundary continuation is supported but NOT
  claimed bit-equal to an unbroken prefill (DEVIATION 831).
- No claim is made about upstream's OWN cross-implementation agreement —
  at the pin its three trig spellings and two diagonal spellings
  disagree with each other; this profile picks, it does not referee.

## 11. Build order — SEQUENTIAL BEHIND THE MAMBA-2 SSD CORE

Nothing below starts until the mamba2 contract's phases 0-2 (primitives,
host oracle, SSD core on device) are CERTIFIED — the orchestrator's
staggering decision. The shared substrate (serial chunk cumsum, segsum
rebuild, serial state pass, gemm cells, the S1-S3 norm machinery,
`identical_clamp`) must exist once and be green before Mamba-3
arithmetic lands on it. Then, smallest first, one gate per phase:

1. **Phase M3-0 — trig and clamp preconditions.** `portable_sinf`/
   `portable_cosf` certification (`checks/portable_trig_check.mojo`'s
   gates, including the cos-bits-did-not-move arm) and `identical_clamp`
   landed via mamba2 phase 0. No new primitive request. RUN OWED.
2. **Phase M3-1 — host oracle.** `mamba/checks/mamba3_oracle.mojo`:
   sections 4-5 on CPU, stage-carded; corpus generator rows from
   `mamba3_siso_fwd_ref`/`step_ref` verbatim. Gate:
   `pixi run check-mamba3-corpus` base cases. RUN OWED.
3. **Phase M3-2 — SISO core on device.**
   `mamba/impl/mamba_ssm/ops/mamba3_siso.mojo` (S7-S20, S22 + reports).
   Gates: (a), (b), the core arms of (f) at B=1 L=4. RUN OWED.
4. **Phase M3-3 — the block, prefill.**
   `mamba/impl/mamba_ssm/modules/mamba3.mojo` (S1-S6, S21, S23 composed
   around phase M3-2). Gates: (a), (b), (c), (e), remaining (f). RUN
   OWED.
5. **Phase M3-4 — decode resumption.** The DEVIATION-831 buffer. Gates:
   (d), STEP_UPSTREAM_RECURRENCE, RESUME_KERNEL_ASSOC. RUN OWED.
6. **Phase M3-5 — shapes and corpus.** L set, adversarial rows,
   FAST recording, kernel-matrix rows. RUN OWED.
7. **Phase M3-6 — the E-series leg.** Three vendors, the
   `gemm/E1G_RUNBOOK.md` shape, reach measured not inferred. The
   completion claim lives or dies here. STARTED 2026-09-02 AND ITS
   SECOND COLUMN IS RED (AMD MI325X, gfx942 -- the RUN RECORD at the end
   of this file); the NVIDIA column is RUN OWED.

DERIVATION_MAP.tsv rows land WITH each file, not after.

---

## RUN RECORD — 2026-09-01 evening, Apple column (orchestrator runs; the implementation lane wrote, never ran)

Commit at run: 979302e7 (implementation commits 7ac1112e / 99a620d8 /
0eac05ed). Build `-D MOJOLEARN_NUMERIC_IDENTICAL=1`, M4, one process at a
time, niced.

- **Gates (a)+(b)+(c): PASS on the FIRST compile** — zero fix iterations
  (the M2 lane took three): every stage bit-identical to the host oracle,
  8 repeated launches identical, batch composition independent with the
  negative control moving 14,178 cells.
- **Gate (d): PASS** — decode == prefill bitwise per token
  (prefix-stable stages) plus the final-token trap.scale/kscale/chunk/
  state/report stages per DEV 832's comparability clause.
- **Gate (d) crossing: PASS** — prefill 60 + decode through 70 ==
  prefill 70 across the Q=64 boundary; the sealed buffer folded and
  refilled exactly as DEV 832 constructs.
- **Continuation: PASS** — four-piece Input_States bit-identical to the
  oracle on 8,192 nonzero incoming h cells, S22 correction included.
- **Gate (e): PASS** — 36 plants, each refused BY NAME with 0 stages.
- **Sabotage ladder: 11 of 11 witnessable arms RED naming their own
  first stage** (SEGSUM_DESCENDING→dacs.out, CHUNK_SIZE_32→yintra.out,
  TRAP_LEFT_ONLY→kscale.out, ANGLE_MOD_PER_CHUNK and _AT_END→
  angle.theta, ROTATE_HALF_SPLIT→rot.q, DIAG_INCLUDE_SUBTRACT→skip.out,
  STATE_TERM_SCALE_FIRST→ystate.out, A_FLOOR_UNCLAMPED→A.out,
  RESUME_KERNEL_ASSOC→pass.states, STEP_UPSTREAM_RECURRENCE→20,719
  cells on gate (d)). **FOLD_SERIAL_ZERO_SEED refused VACUOUS BY
  CONSTRUCTION naming DEVIATION 834** — the required outcome, not a
  pass and not a skip.
- **Shape sweep: 42/42 PASS** — B ∈ {1,2,3} × L ∈ {1,4,63,64,65,129,257}
  × d_model ∈ {32,64}, gate (a) asserted at every combination.

**STILL OWED — updated 2026-09-01 (write-side lane; every claim below is
UNVERIFIED, RUN OWED with its command):**

- **Corpus family: BUILT, runs owed.** `mamba/corpus/gen_corpus.py` now
  carries the mamba3 family MIRRORING `mamba3_fixture.mojo`'s normative
  table index for index (ids 41-54, seed base "Mmb3Corp", the four plants;
  verbatim `mamba3_siso_fwd_ref`/`step_ref`/`_segsum` copies cited by
  line; staged reference on the profile's chunked schedule per DEVIATIONS
  827/829/830), `tools/mamba_corpus_check.py` is documented
  family-agnostic, and `pixi run check-mamba3-corpus` exists
  (`mamba/corpus/mamba3/m3_base_b2_l4_d32`, tool-default tolerances until
  `--self-test` calibrates them). RUNS OWED, in order:
  1. generation (scratch venv per `mamba/corpus/README.md`):
     `python mamba/corpus/gen_corpus.py --family mamba3 --verify`
  2. the byte gate (two hash-spec implementations agreeing):
     `tools/with_identical_mode.sh pixi run mojo run -I . mamba/checks/mamba3_check.mojo corpus`
  3. the tolerance read, once the check grows a stage-dump path (the
     mamba1 `MOJOLEARN_MAMBA_CORPUS_DUMP` shape — a CHECK-LANE item, not
     the corpus lane's): `pixi run check-mamba3-corpus` with the dump dir
     in `MOJOLEARN_MAMBA_CORPUS_DUMP`, then `--self-test` calibration.
- **FAST recording: FIRST CARD LANDED 2026-09-02 ON AMD, APPLE STILL
  OWED.** The MI325X recorded a mamba3 FAST arm that diverged from the
  oracle on 10,497 cells, first stage `in_proj.out` -- expected, not a
  red, because FAST makes no bitwise promise. Apple has recorded none.
  The phase-8 pair exists
  (`tools/e1_bootstrap.sh`: `run_lane_arm mamba3` + `check-mamba3-block`;
  `tools/e3_round_judge.sh` expects and forgives
  `PHASE8-FINDING: mamba3 [fast]` and lists mamba3's FAST card as
  expected-missing until first recorded). The recording run is
  `MOJOLEARN_E1_LANES=mamba3 bash tools/e1_bootstrap.sh` (or any full
  leg) — a run, not a write.
- **Kernel-matrix rows: WRITTEN.** `IDENTITY_PATHS.md` rows 92 (mamba2)
  and 93 (this profile) carry the column-by-column ledger -- for this
  profile, Apple RECORDED PASS and, since 2026-09-02, AMD RECORDED RED
  (MI325X, gfx942; the RUN RECORD below), with NVIDIA still OWED; the
  block's kernels read no vendor-varying tunable, so
  `checks/kernel_matrix.mojo` gains no knob row (stated in row 93, not
  silently omitted). The cross-column READS are the E-leg's.
- **PHASE M3-6 — the three-vendor E-series leg** the completion claim
  lives on: `gemm/E1G_RUNBOOK.md` shape, reach measured not inferred.
  ITS SECOND COLUMN ARRIVED 2026-09-02 AND IT IS RED (the AMD RUN RECORD
  below); the NVIDIA column is still RUN OWED, and the leg itself stays
  open.

ONE COLUMN, ONE VENDOR AS OF 2026-09-01: nothing in this
2026-09-01 record is a cross-vendor claim. The second column is recorded
in the RUN RECORD below.


---

## RUN RECORD -- 2026-09-02, AMD column (MI325X, gfx942) -- GATE (a) RED

Commit at run: cd56e8ce, the SAME commit the Apple arm re-ran at on this
date, and the SAME command on both boxes:
`MOJOLEARN_IDENTITY_TRACE=... tools/with_identical_mode.sh pixi run mojo run -I . mamba/checks/mamba3_check.mojo`.
Box: a DigitalOcean MI325X droplet, gfx942. Orchestrator ran both arms.

- **APPLE (M4), rc 0.** `GATE A PASS: every stage bit-identical to the
  oracle at this shape.` The card carries 28 records, section 7 of this
  contract wants 28, and every tag is present once each in card order.
  **GATE B PASS** (8 repeated launches identical). **GATE C PASS** (row
  bits independent of launch companions; the negative control differed on
  14,178 cells).
- **AMD (MI325X, gfx942), rc 1.**
  `GATE A FAILED: 1179 cells differ between the device card and the host oracle (first stage: rot.k)`.
  FOUR stages moved, every other stage OK, and the four account for all
  1,179 cells:

  | stage | cells moved | first cell | device | oracle |
  | --- | --- | --- | --- | --- |
  | `rot.k` | 54 of 512 | 9 | `0x3fa66f4e` | `0x3fa66f4d` |
  | `kscale.out` | 49 of 512 | 9 | `0x3dc75964` | `0x3dc75963` |
  | `ssd.h_last` | 1,061 of 8,192 | 3 | `0x3ba5c22f` | `0x3ba5c230` |
  | `ssd.k_last` | 15 of 128 | 4 | `0x3fb318a0` | `0x3fb318a1` |

  Every quoted pair differs in the LAST MANTISSA BIT ONLY -- one ULP, and
  in BOTH directions (the device reads high on two stages and low on two).

**WHAT IS CLEAN IS WHAT NARROWS IT.** `m3.rot.q` is bit-identical on the
MI325X. rot.q and rot.k are the SAME ARITHMETIC in the same kernel
(`mamba/impl/mamba_ssm/ops/mamba3_siso.mojo`, the `if rotated:` branch):
both are `ftz(pinned_mul(x0, cv) - pinned_mul(x1, sv))` and
`ftz(pinned_mul(x0, sv) + pinned_mul(x1, cv))` with the SAME cv and sv,
and they differ only in which operand they read (q from `bcc` + `c_bias`,
k from `bcb` + `b_bias`). Both of those inputs are themselves traced and
both are bit-identical on the AMD box (`m3.bcnorm.B` and `m3.bcnorm.C`
OK), as is `m3.angle.theta`. So the divergence is DATA-DEPENDENT INSIDE
IDENTICAL CODE -- not a different code path, and not a bad input.

**LEADING HYPOTHESIS, AND IT IS A HYPOTHESIS, NOT THE CAUSE.**
`pinned_mul(a, b)` is `identical_mul_add(a, b, Float32(-0.0))`
(`mamba3_siso.mojo:127`). That is mathematically equal to `a * b` for
every input INCLUDING the zero signs, so a backend is free to simplify it
to `a * b` and then contract `a*b - c*d` into a fused multiply-add, which
rounds once where the oracle rounds twice; that would produce exactly this
signature. **UNCONFIRMED:** no disassembly was read and no contraction
flag was toggled on this run. **THE CONFIRMING EXPERIMENT, RUN OWED:**
read the gfx942 ISA for the rot kernel, or rebuild with contraction off
and re-run gate (a).

**THE FAST ARM ON THE SAME BOX DIVERGED AND THAT IS CORRECT.** mamba3
[fast] differed from the oracle on 10,497 cells, first stage
`in_proj.out`. FAST makes no bitwise promise, so this is recorded as
EXPECTED and is NOT a red.

**THIS IS THE SYSTEM WORKING.** The STILL OWED list above already named
the portable-trig device certification (phase M3-0) and every column that
is not this machine (phase M3-6). This is phase M3-6 arriving with a red
on its second column; no cross-vendor claim was ever made for this
profile, and none is made here. NVIDIA is STILL OWED.
