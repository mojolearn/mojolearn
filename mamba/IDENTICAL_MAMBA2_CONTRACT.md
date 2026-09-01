# The IDENTICAL FP32 Mamba-2 (SSD) block contract

# PROFILE `mojolearn.identical.mamba2.fp32.v1`

Written 2026-09-01, the mamba lane (DEVIATIONS 782-789; all eight are
claimed, none left unassigned). The shape of this document is
`mamba/IDENTICAL_MAMBA_CONTRACT.md`'s, which is `gemm/IDENTICAL_FP32_CONTRACT.md`'s,
on purpose. NOTHING IN THIS DOCUMENT HAS RUN. It is the builders'
instruction sheet; the code forms it names (`mamba/checks/mamba2_oracle.mojo`,
`mamba/impl/mamba_ssm/modules/ssd_minimal.mojo`, `mamba2.mojo`) do not exist
yet, and every gate below is RUN OWED with its command spelled.

**THE PROFILE NAME IS PART OF THE CONTRACT.** Every card, gate and claim
under this document names `mojolearn.identical.mamba2.fp32.v1`. Changing any
seam decision in section 4, any constant in section 3 — **`CHUNK_SIZE`
included, see DEVIATION 783** — or the stage list in section 7 creates a v2.
It does not amend v1. FAST is unversioned and makes no identity claim.

The completion claim this contract exists to support is one sentence. One
Mamba-2 (SSD) block, FP32, is bit-identical across Apple, NVIDIA and AMD
GPUs at every stage, at every launch, at every batch composition, and
between the prefill and decode paths, under the declared profile. Not
bit-identical inference of a model, not Mamba-3, not training, not
variable-length batching. The cross-vendor half of that sentence is earned
ONLY by an E-series leg (the `gemm/E1G_RUNBOOK.md` shape, judge section 8)
and is NOT earned by anything built on one machine.

Seam numbers `S*` in this document are THIS document's. Mamba-1's seams are
cited as `mamba1 S9` etc. Where a numerical question recurs from the Mamba-1
contract, **this profile INHERITS the Mamba-1 answer unchanged** — the two
Mamba lanes may not answer the same question twice (section 4's inheritance
column says which pin is inherited, seam by seam).

---

## 1. The reference, pinned

| what | where | pin |
|---|---|---|
| the SSD core's math, NORMATIVE | `mamba_ssm/modules/ssd_minimal.py::segsum` (:23-32) and `::ssd_minimal_discrete` (:34-78), composed discretize-first as its own test composes it (:94-103, `ssd_minimal_discrete(x*dt, A*dt, B, C)`) | state-spaces/mamba `e9594ce` |
| the block order, NORMATIVE | `mamba_ssm/modules/mamba2.py::Mamba2.forward`, non-mem-eff arm (:209-276): in_proj/zxbcdt split (:211-215), `A = -exp(A_log)` (:182), conv+SiLU (:230-242), xBC split (:243), gated norm (:269-270), out_proj (:275) | state-spaces/mamba `e9594ce` |
| the dt seam's order | `mamba_ssm/ops/triton/ssd_chunk_state.py::_chunk_cumsum_fwd_kernel` (:72-86): bias (:73-75), softplus with the `<= 20` guard (:76-77), clamp (:78-81); HF spells the same order in torch (`modeling_mamba2.py::mamba2_chunk_scan` :272-276) | both pins |
| the inter-chunk pass SHAPE | `mamba_ssm/ops/triton/ssd_state_passing.py::_state_passing_fwd_kernel` (:64-87): SERIAL over chunks, `states = scale * states + new_states` (:80), scale `= exp(dA_cs_last)` (:74-75) | state-spaces/mamba `e9594ce` |
| the decode step's SEMANTICS | `mamba_ssm/modules/mamba2.py::Mamba2.step` (:278-343, the torch arm :290-296 and :310-322), `::allocate_inference_cache` (:345-355); `ops/triton/selective_state_update.py::selective_state_update_ref` (:224-285). Semantics only — its ROUNDING is the DEVIATION 786 sabotage arm, not the profile | state-spaces/mamba `e9594ce` |
| the gated RMSNorm | `mamba_ssm/ops/triton/layernorm_gated.py::rms_norm_ref` (:18-39): gate BEFORE norm at `norm_before_gate=False` (:26-27), `1 / torch.sqrt(...)` (:29) | state-spaces/mamba `e9594ce` |
| the second independent reference | `src/transformers/models/mamba2/modeling_mamba2.py::segment_sum` (:73-90), `::mamba2_chunk_scan` (:254-348), `::MambaRMSNormGated` (:105-120), `::Mamba2RMSNorm` (:591-605), `::Mamba2Block.forward` (:617-631). Cross-reference and corpus source, not normative; where it disagrees with mamba_ssm (its `torch.rsqrt` at :118) the mamba_ssm spelling wins, exactly as mamba1 DEVIATION 741 decided the identical question | huggingface/transformers `d56c55b` |
| the "what vendors run" column | `mamba_ssm/ops/triton/ssd_combined.py::_mamba_chunk_scan_combined_fwd` (:343-395, the cumsum/bmm/chunk_state/state_passing/chunk_scan sequence) and the surface `mamba_chunk_scan_combined` (:628-648); `ssd_chunk_state.py::chunk_state_ref` (:1094-1122); `ssd_state_passing.py::state_passing_ref` (:327-350) | cited where their rounding DIFFERS from the profile's (DEVIATIONS 785, 786, 789) so a reader porting from the fused chain does not assume agreement |
| the projections' arithmetic | profile `mojolearn.identical.gemm.fp32.v1` (`gemm/IDENTICAL_FP32_CONTRACT.md`), entry `core/gemm.mojo::gemm_nt` / `::pinned_gemm_nt_kernel`, oracle `gemm/checks/gemm_oracle.mojo::gemm_oracle_cell` (`oracle_leaf_partial`, `fold_balanced_tree`, `contract_leaf_size`), certified three-vendor (IDENTITY_PATHS row 40) | this repository |
| the conv, silu, softplus, exp, rsqrt, div seams | mamba1 contract sections 3-4 and `checks/numerics.mojo` (`identical_exp`, `identical_softplus`, `identical_silu`, `identical_rsqrt`, `identical_div`, `identical_mul_add`, `pinned_mul`, `ftz`; IDENTITY_PATHS rows 49-54), certified as row 55 records | this repository, INHERITED |

Checkouts live in `/Users/andrewhendel/CascadeProjects/upstream/`. The house
rule is transliteration: their algorithm, statement by statement; every
departure is a numbered deviation below, and there are exactly eight.

## 2. What one block call is

    hidden = residual + mixer(rmsnorm(residual))          Mamba2Block.forward (HF :624-630)

with mixer, in order (`Mamba2.forward` non-mem-eff arm, `d_mlp = 0`):

    in_proj -> split(z | xBC | dt_raw)  [order z, xBC, dt; mamba2.py:211-215]
    -> causal depthwise conv1d over xBC (d_conv=4, padding 3, truncated) -> SiLU
    -> split(x | B | C)
    -> dt = clamp(softplus(dt_raw + dt_bias), dt_limit)        [seam S9]
    -> DISCRETIZE FIRST: X_d = x * dt, dA = dt * A, A = -exp(A_log)
    -> chunked SSD core (section 4, seams S10-S19):
         per chunk: dA_cs cumsum, L = exp(segsum), G = C.B, Y_diag = (G.L).X_d,
         chunk_states = (B.decay).X_d
         across chunks: SERIAL state pass  h_c = fma(exp(dA_cs_last), h_{c-1}, chunk_states_c)
         per chunk: Y_off = (C.h_{c-1}) * exp(dA_cs);  Y = Y_diag + Y_off
    -> + D * x (per head, D last)
    -> gated RMSNorm: norm(Y * silu(z)) * weight        [norm_before_gate=False]
    -> out_proj

Inference only. The decode step is the SAME chunked arithmetic resumed one
token at a time (section 5, DEVIATION 786). Weights carry no bias except
conv1d (`bias=False`, `conv_bias=True`, the Mamba2 defaults, mamba2.py:56-57).
`A_init_range=(1,16)` (mamba2.py:48) is an INITIALIZATION fact recorded for
completeness; weights are inputs to this profile and initialization is out of
scope.

## 3. Profile constants

| constant | value | source |
|---|---|---|
| d_state (N) | 128 | mamba2.py:41 default (mamba2_simple.py uses 64; the shipped module's 128 is the profile's) |
| d_conv | 4 | mamba2.py:42 |
| expand | 2 | mamba2.py:44; d_inner = 2 * d_model |
| headdim (P) | 64 | mamba2.py:45; nheads H = d_inner / headdim, so d_model must be a multiple of 32 |
| ngroups (G) | 1 | mamba2.py:47; B and C are shared across heads by COPY (broadcast, not arithmetic) |
| **CHUNK_SIZE (Q)** | **256** | mamba2.py:59 default. **PART OF THE ARITHMETIC, see DEVIATION 783** |
| dt_limit default | (0.0, +inf) | mamba2.py:55. `dt_limit` is a runtime INPUT; the clamp seam is pinned whether or not the limits bind |
| rms eps (both norms) | 1e-5 (`0x3727C5AC`) | mamba2.py:144 (gated norm); block norm per config `layer_norm_epsilon` |
| softplus threshold | 20.0, compared `<=` | `_chunk_cumsum_fwd_kernel`:77, `selective_state_update.py`:102; identical to mamba1's |
| D_has_hdim | False (D is per-head, shape [H]) | mamba2.py:49 |
| rmsnorm / norm_before_gate | True / False | mamba2.py:50-51 |
| d_ssm | d_inner (d_mlp = 0) | mamba2.py:46 default (d_ssm=None) |
| dtype | Float32 everywhere (weights, activations, state) | Andrew's order |

Changing any of these is a v2. `D_has_hdim=True`, `norm_before_gate=True`,
`rmsnorm=False` (z-gate inside the scan), `ngroups > 1`, `d_mlp > 0` and
non-Float32 are REFUSED BY NAME, not silently defaulted. Shapes covered by
the gates are B in {1, 2, 3}, L in {1, 4, 256, 257, 513, 770}, d_model in
{32, 64}; per-cell arithmetic reads none of B, L or the launch (section 4),
so shape coverage is about the gates, not the profile.

**Padding is part of the arithmetic's shape**: the last chunk is padded to Q
with `+0.0` inputs and zero dt (HF :42-50, :282; `_chunk_cumsum_fwd_kernel`:81
masks dt to 0 past seqlen), the output truncated to L (HF :342-343). Every
intra-chunk fold is therefore ALWAYS length Q; a padded position contributes
an exactly-zero product to a fold and moves no nonzero bit.

## 4. The seams, every one, with the fused-or-unfused decision

FMA contraction is PER SEAM. FUSED = one rounding through `identical_mul_add`.
PRODUCT = one rounding through `pinned_mul(a, b) = identical_mul_add(a, b, -0.0)`
(preserves a `-0.0` product; gemm fixture F6a's lesson). Every seam's RESULT
passes `ftz`, and every operand LOADED from a buffer passes `ftz` (row 10).
Copies — the zxbcdt split, the xBC split, the chunk reshape, the ngroups
broadcast of B and C, the window update, state stores — are NOT seams.

| # | seam | reference spelling | pinned spelling | fused? | inherits |
|---|---|---|---|---|---|
| S1 | block RMSNorm sum of squares | `hidden.pow(2).mean(-1)` (HF Mamba2RMSNorm :603) | serial ascending j from `+0.0`, `acc = ftz(fma(x_j, x_j, acc))`, one fold per row, no block fold | FUSED | mamba1 S1 |
| S2 | block RMSNorm mean, rstd | mean then rsqrt (HF :603-604) | `mean = ftz(identical_div(sumsq, d_model))`; `rstd = ftz(identical_rsqrt(ftz(mean + eps)))`; identical_rsqrt is `1/sqrt` as `rms_norm_ref`:29 spells it, never the hardware intrinsic | n/a | mamba1 S2, DEVIATION 741's answer |
| S3 | `hidden * rstd`, then `weight * hidden` (HF :604-605) | two products | `pinned_mul(x_j, rstd)`, then `pinned_mul(w_j, ·)` | PRODUCT | mamba1 S3-S4 |
| S4 | the three matmuls: in_proj, out_proj, and every SSD contraction of S12/S14/S16/S18 | `nn.Linear` (cuBLAS); `einsum`/`tl.dot` (vendor GEMM) | ALL are `gemm.fp32.v1` cells: `+0.0`-seeded serial-ascending leaves (`contract_leaf_size`: one leaf at k <= 128), `fold_balanced_tree` across leaves. `OP_NT` orientation per call site. DEVIATION 784 lists every cell and its k | per gemm v1 | mamba1 S17 |
| S5 | `A = -exp(A_log)` (mamba2.py:182) | negate after exp | `-ftz(identical_exp(ftz(A_log)))`; negation exact. A < 0 always, so every segment sum below is <= 0 and every `identical_exp` result is in (0, 1] — no overflow path exists in the SSD core | n/a | mamba1 S15 |
| S6 | conv tap chain over xBC (mamba2.py:230-242; d_ssm + 2·G·N channels) | `F.conv1d(padding=3, groups=conv_dim)` + bias | bias-SEEDED accumulator, taps k = 0..3 ascending (oldest first), `acc = ftz(fma(w[d,k], x[l-3+k], acc))`, window read for pre-sequence positions (zeros on a fresh prefill). Decode uses the SAME seed even though `Mamba2.step`:291-295 adds the bias last — the reference's two paths disagree with each other exactly as Mamba-1's did, and the answer is already given | FUSED | mamba1 S13 + DEVIATION 721, verbatim |
| S7 | SiLU after the conv | `self.act(xBC)` (mamba2.py:232, :296) | `identical_silu`, the ONE-division spelling `z / (1 + exp(-z))` | n/a | mamba1 S12's function, DEVIATION 744 |
| S8 | gate SiLU inside the norm | `F.silu(z)` (rms_norm_ref:27) | `identical_silu(z)` | n/a | same |
| S9 | `dt = clamp(softplus(dt_raw + dt_bias), lo, hi)` | bias, softplus, clamp IN THAT ORDER (`_chunk_cumsum_fwd_kernel`:73-81; HF :272-276) | `biased = ftz(dt_raw + ftz(bias[h]))`; `identical_softplus(biased)` (`x <= 20 ? log1p(exp(x)) : x`, DEVIATION 745's function); then `identical_clamp(·, lo, hi)` — DEVIATION 788, the one NEW primitive this profile needs. At the default (0, +inf) the clamp cannot move a bit (softplus of a finite input is >= +0.0) but it is PRESENT, matching the kernel, so an active `dt_limit` is the same code path | add, then n/a | mamba1 S14 for the softplus |
| S10 | DISCRETIZE FIRST: `X_d = x * dt`, `dA = dt * A` | `hidden_states * dt`, `A * dt` (HF :288-289); the normative composition `ssd_minimal_discrete(x*dt, A*dt, ...)` (ssd_minimal.py:103) | `pinned_mul(x[l,h,p], dt[l,h])`; `pinned_mul(dt[l,h], A[h])`. dt binds to x and to A, NEVER to B — DEVIATION 789. The fused chain rounds otherwise (`chunk_state_ref`:1122 keeps dt as a separate einsum factor) and is refused | PRODUCT | new question — see DEVIATION 789 for why this is NOT mamba1 S7/S8's question |
| S11 | per-chunk cumsum `dA_cs` and segsum | `torch.cumsum` (ssd_minimal.py:51, :29; `tl.cumsum` :85) | SERIAL ASCENDING within the chunk: `dA_cs[i] = ftz(dA_cs[i-1] + dA[i])`, seed the running value at the chunk's first element (no cross-chunk carry — the chunk boundary is a hard reset, that is the algorithm). segsum per DEVIATION 782: `seg[i][j]` for j < i is the serial ascending partial rebuilt as `seg[i][j] = ftz(seg[i-1][j] + dA[i])` with `seg[j][j] = +0.0`; entries above the diagonal are STRUCTURAL ZEROS, never a computed `exp(-inf)` | add | mamba1's serial-over-L discipline, applied to the cumsum |
| S12 | `L = exp(segsum)`, `G = C·B` | ssd_minimal.py:54-55; HF :299-302 | `L[i][j] = ftz(identical_exp(seg[i][j]))` for j <= i, `+0.0` above (structural; `identical_exp` saturates below -87.33655 to `+0.0`, so the masked `-inf` spelling and the structural spelling are the SAME bits — DEVIATION 782 records the argument); `G[i][j] = ` gemm v1 cell over n, k = d_state = 128, ONE serial ascending leaf | n/a; per gemm v1 | mamba1 S6's `identical_exp` |
| S13 | `M = G ⊙ L` | HF :305 | `pinned_mul(G[i][j], L[i][j])` | PRODUCT | — |
| S14 | `Y_diag = M · X_d` over chunk positions | ssd_minimal.py:55; HF :308 | gemm v1 cell, k = Q = 256 (two leaves of 128, one fold level). Structurally-zero M entries above the diagonal and padded positions enter the fold as exact zeros | per gemm v1 | — |
| S15 | `decay = exp(dA_cs_last - dA_cs)`, `B_decay = B ⊙ decay` | ssd_minimal.py:59; HF :312-313 | `d = ftz(dA_cs_last - dA_cs[i])` (one subtraction), `identical_exp(d)`, `pinned_mul(B[i,n], decay[i])` | sub; PRODUCT | — |
| S16 | `chunk_states = B_decay^T · X_d` over chunk positions | ssd_minimal.py:60; HF :314 | gemm v1 cell, k = Q = 256, output [H, P, N] per (b, chunk) | per gemm v1 | — |
| S17 | inter-chunk state pass, SERIAL | `states = scale * states + new_states`, serial over c (`_state_passing_fwd_kernel`:72-84); the NORMATIVE reference spells this as a decay-matrix einsum (ssd_minimal.py:64-69) and DEVIATION 785 replaces that spelling | `h_c = ftz(fma(scale_c, h_{c-1}, chunk_states_c))` per (b, h, p, n), `scale_c = ftz(identical_exp(ftz(dA_cs_last_c)))`, h_{-1} = initial_states or `+0.0`. ONE rounding, the same shape as mamba1 S9's `h = fma(deltaA, h, deltaB_u)` and pinned FUSED for the same reason | FUSED | mamba1 S9's answer to decay·state+increment |
| S18 | `Y_off = (C · h_prev) ⊙ exp(dA_cs)` | HF :330-332 (contract over n FIRST, scale by the decay AFTER); ssd_minimal.py:73-74 spells it as one einsum whose internal order torch owns — HF's explicit order is the profile's | gemm v1 cell over n, k = d_state = 128, then `pinned_mul(·, ftz(identical_exp(dA_cs[i])))` where h_prev is the chunk's INCOMING state (h_{c-1}) | per gemm v1; PRODUCT | — |
| S19 | `Y = Y_diag + Y_off` | ssd_minimal.py:77; HF :335 | `ftz(Y_diag + Y_off)` | add | — |
| S20 | `out = Y + x * D[h]` | D-residual from UNDISCRETIZED post-conv x (HF :284-285, added last :338-339; `Mamba2.step`:319) | `p = ftz(pinned_mul(x[l,h,p_], D[h]))`, then `ftz(Y + p)`. UNFUSED, product rounds on its own, D LAST — the reference's order, exactly mamba1 S11's decision | PRODUCT + add | mamba1 S11 |
| S21 | gated RMSNorm | `x = x * silu(z)` FIRST (norm_before_gate=False, rms_norm_ref:26-27), then sumsq/mean/rstd/weight (:29-30) | `y_g = pinned_mul(Y[j], identical_silu(z[j]))` (S8), then the S1-S3 fold/rstd/products over d_ssm with eps 1e-5. `group_size = d_ssm / ngroups` = the whole row at G = 1; a grouped fold has no site in v1 — DEVIATION 787 | as S1-S3 | mamba1 S1-S4 machinery |
| S22 | residual add | `residual + hidden_states` (HF :630) | `ftz(ftz(x) + out_proj)` | add | mamba1 S16 |

Inherited clause, restated so nobody rediscovers it: at k = 1 (a GEMM leaf of
one element, which S14/S16 hit at L = 1 only through the padded fold's real
row) the gemm v1 leaf is `ftz(fma(a, b, +0.0))`, so a `-0.0`-valued product
exits as `+0.0` (gemm section 9.2(a)). v1 GEMM behavior, inherited unchanged.

The softplus guard note from the Mamba-1 contract (distinguishing range
delta in about [8, 14], a 20-straddling fixture is vacuous) applies verbatim;
the mamba2 corpus must plant IN THE BAND, not at the boundary — the
`adv_softplus_guard` lesson was measured, not argued.

## 5. State, decode, and what "resumable" means here

The recurrent state between calls is THREE pieces, not two:

1. the conv WINDOW — last d_conv inputs of xBC per channel, oldest first,
   zeros before the first token (`allocate_inference_cache`:348-350);
2. the chunk-BOUNDARY SSM state `h` [B, H, P, N] — the S17 running value as
   of the last COMPLETED chunk (zeros, or `initial_states`, before the first);
3. the INTRA-CHUNK BUFFER — the current partial chunk's post-conv/post-SiLU
   xBC rows and raw dt rows (q <= Q of them), enough to re-derive every S10-S19
   quantity for the open chunk.

DEVIATION 786: decode is PREFILL RESUMPTION. A decode step appends its token
to the buffer, recomputes the open chunk's stages for its own row (the padded
folds over Q positions, zeros beyond q, are bitwise the folds a prefill
ending at that token runs), and when the buffer reaches Q tokens folds the
completed chunk through S16-S17 and clears it. `Mamba2.step`'s own
per-token recurrence (:310-322 / `selective_state_update_ref`:277-282)
rounds DIFFERENTLY from the chunked prefill — `dB = dt*B` pairing, `C·h_new`
association — and no choice of seams can make both spellings bit-equal, so
the profile has ONE spelling, the prefill's, and gate D verifies what the
construction promises. The upstream step spelling is kept in source as the
required-RED sabotage arm `STEP_UPSTREAM_RECURRENCE`.

The block ALSO returns `h_last` [B, H, P, N], the S17 value after the final
(padded) chunk — upstream's `final_state`/`ssm_state` (ssd_minimal.py:69,
mamba2.py:261-264) — recorded as a card stage and corpus-checked. It is a
REPORT, not the resumption state: feeding `h_last` to a fresh call as
`initial_states` is upstream's varlen-free continuation semantics and is
bit-equal to resumption ONLY when the handoff lands on a chunk boundary
(L a multiple of Q). Gate D2 covers exactly that boundary case;
`initial_states` in / final state out is otherwise supported as ssd_minimal
:64-66 spells it (prepended as chunk -1's state).

DEFERRED, not refused: variable length (`cu_seqlens` / `seq_idx`,
mamba2.py:154's plumbing), the mem-eff fused surface
(`mamba_split_conv1d_scan_combined`), and multi-sequence caches beyond one
block's state. Recorded here so a later round starts from a sentence, not a
silence.

## 6. NaN, infinity, signed zero, denormals

Section 6 of the Mamba-1 contract applies VERBATIM — refusal by name and by
BITS of any nonfinite input or weight before any recorded stage
(`refuse_nonfinite`), `-0.0` admitted everywhere and preserved where the
reference preserves it, `+0.0` fold seeds, `ftz` at every seam and on every
buffer load. Two additions:

- The segsum mask is STRUCTURAL (S11/S12), so the `-inf` the references
  write into masked positions (ssd_minimal.py:31, HF :89) never exists in a
  buffer and cannot trip the refusal audit. DEVIATION 782 carries the
  bit-equality argument.
- The clamp of S9 is the profile's only ordered-select on floats. Its
  operands are a softplus output (>= +0.0, never NaN after the refusal gate)
  and two finite-or-+inf limits; `identical_clamp`'s spelling (DEVIATION
  788) must still define its zero-sign and NaN behavior by construction, not
  by vendor min/max — row 13's selection hazard is why it is a named
  primitive and not an inline compare.

## 7. The stages, in card order

One record per stage per block call, tags prefixed by the driver
(`core/identity_trace.mojo` rules; tags carry no machine property).
Token-major `M = B * L`; H = nheads, P = headdim, N = d_state, Q =
CHUNK_SIZE, C = nchunks = ceil(L / Q), CD = conv_dim = d_ssm + 2·G·N.

    input.x        [M, d_model]        the block input, as given
    norm.sumsq     [M]                 S1
    norm.out       [M, d_model]        S2-S3
    in_proj.out    [M, 2*d_inner+2*G*N+H]  S4; columns z | xBC | dt_raw (mamba2.py:211-215 order)
    A.out          [H]                 S5
    conv.out       [M, CD]             S6, bias included
    silu.out       [M, CD]             S7 (columns x | B | C after the copy split)
    conv.window    [B, CD, 4]          the window AFTER the call (copies)
    dt.out         [M, H]              S9 (biased, softplussed, clamped)
    xd.out         [M, H, P]           S10, x * dt
    dacs.out       [B, H, C, Q]        S11 per-chunk cumsum (padded positions carry the last real value)
    seg.L          [B, C, H, Q, Q]     S12 exp(segsum), +0.0 above the diagonal
    cb.G           [B, C, G, Q, Q]     S12 C·B
    ydiag.out      [M, H, P]           S13-S14
    decay.states   [B, H, C, Q]        S15's exp
    cstate.out     [B, C, H, P, N]     S15-S16
    pass.states    [B, C, H, P, N]     S17, the state ENTERING each chunk (h_{c-1})
    yoff.out       [M, H, P]           S18
    scan.y         [M, H, P]           S19
    skip.out       [M, H, P]           S20
    gnorm.gate     [M, d_ssm]          S21's y * silu(z)
    gnorm.sumsq    [M]                 S21 fold
    gnorm.out      [M, d_ssm]          S21 out
    out_proj.out   [M, d_model]        S4
    residual.out   [M, d_model]        S22
    ssd.h_last     [B, H, P, N]        section 5's report stage

`seg.L` and `cb.G` are recorded at gate shapes only if the driver's size cap
allows; if elided, the elision is stated on the card, never silent. The
corpus's channel-major views are reindexed by the corpus gate, not recomputed
(the mamba1 corpus's transposed-dump control is the vacuity check to copy).

## 8. What "identical" is gated to mean

The Mamba-1 gate list, transposed, plus the chunk-boundary clauses. Every
command below is RUN OWED; none has run.

(a) **card == host oracle card**, bitwise, every stage, every gate shape.
    `pixi run check-mamba2-block` (builders register
    `mamba/checks/mamba2_check.mojo`; shape via MOJOLEARN_MAMBA2_CHECK_B/_L/_DM,
    one compile at a time — the 2026-08-23 crash rule).
(b) **repeat-run identity**: same bits on 8 repeated launches, fresh state
    and dispatches each. Same command, repeat arm.
(c) **batch-composition invariance**: a row's bits identical whether its
    sequence shares the launch with 0, 1 or 2 others, sliced from ONE input
    so row 0 is identical by construction, WITH the negative control (rows 0
    and 1 of the B=2 run must differ) — without it a broken slicer passes
    forever, the mamba1 lesson verbatim.
(d) **decode == prefill**, bitwise, per token, per stage: prefill L, then
    decode the same tokens one at a time from zero state; and the HANDOFF
    arm, prefill L1 + decode L2 == prefill L1+L2. Both with the deliberate
    misalignment control (compare decode t against prefill t+1, must differ).
    Fixture set MUST include a decode run that CROSSES a chunk boundary
    (L1 = 250, decode through token 262) — a boundary never crossed is a
    clause never gated.
    **(d2)** the boundary handoff: prefill L = 512, then a fresh call with
    `initial_states = h_last` == prefill L = 512 + more, per section 5.
(e) **nonfinite refusal**: every named input planted with quiet NaN and +inf
    BY BITS at cell len/2, read back off the device first (reach measured,
    not inferred), refused by name with 0 stages recorded; plus the clean-call
    control (must not raise, must record all stages).
(f) **every clause falsifiable by a named sabotage** that fails a gate, each
    verified to move the stage its own seam writes and no earlier one, on a
    fixture that WITNESSES it:

    | arm | must move | witnessed by |
    |---|---|---|
    | SEGSUM_DESCENDING (S11 fold reversed) | dacs.out / seg.L | any L >= 2 fixture with nonuniform dt |
    | CHUNK_SIZE_128 (Q rebuilt at 128) | ydiag.out onward | L > 128 fixture (DEVIATION 783's falsifier) |
    | STATEPASS_MATRIX (ssd_minimal:64-69's decay-matrix einsum) | pass.states | L >= 513 (a two-hop decay) AND the nonzero-`initial_states` fixture at L = 257 — a zero-init two-chunk case is bitwise inert |
    | STATEPASS_UNFUSED (mul then add at S17) | pass.states | same fixtures |
    | PAIR_DT_B (dt bound to B, the fused chain's pairing) | xd.out / cstate.out | hashed values, any shape (DEVIATION 789's falsifier) |
    | STEP_UPSTREAM_RECURRENCE (mamba2.py:310-322 spelling in decode) | gate (d) per-token | L >= 2 decode |
    | GATE_NORM_BEFORE (norm_before_gate=True spelling) | gnorm.gate onward | any fixture |
    | CLAMP_BEFORE_SOFTPLUS (S9 order swapped) | dt.out | the active-dt_limit fixture ((0.001, 0.1) planted) |
    | S6_BIAS_LAST, S6_TAPS_REVERSED | conv.out | inherited arms, mamba1's |
    | FOLD_SERIAL_ZERO_SEED (gemm fold sabotage) | ydiag/cstate at k = 256 | inherited from gemm; k = 256 is the first mamba shape with P = 2, so this lane is the fold's first consumer above one leaf in the SSM direction |

(g) **corpus cross-check**: `mamba/corpus/` grows mamba2 cases — torch
    float64 per-stage references generated by the two upstreams' own spellings
    (verbatim-copied, cited by line, the gen_corpus.py discipline), hashed
    inputs under the `mojolearn.mamba.corpus.hash.v1` element rule with new
    tensor names. Stage list to record = section 7's minus `seg.L`/`cb.G` when
    size-capped, per-case tolerances calibrated from the corpus `--self-test`
    (the `adv_gate_saturation` lesson: the tolerance is per case, and torch
    FP32's own distance from float64 is the yardstick). Adversarial cases
    owed by name: softplus band [8,14] plants; signed-zero plants compared BY
    SIGN BIT; an A-near-zero case (decay near 1, the cancellation-prone
    `dA_cs_last - dA_cs`); a saturating-gate case; an active-dt_limit case; a
    nonzero-initial_states case; L in {1, 4, 256, 257, 513, 770} composition
    rows. `pixi run check-mamba2-corpus` (tool: `tools/mamba_corpus_check.py`
    extended or a sibling).

FAST-mode arms of (a) are RECORDED, not asserted, where vendor-shaped.
Nothing above earns a cross-vendor sentence; only the E-series leg does.

## 9. Deviations 782-789, all claimed

- **DEVIATION 782 — pinned segsum/cumsum schedule.** The references spell
  segsum as repeat + masked_fill + `torch.cumsum` + `-inf` mask + exp
  (ssd_minimal.py:23-32, HF :73-90); torch's cumsum order is torch's own and
  `tl.cumsum`'s tree is Triton's. The profile pins SERIAL ASCENDING
  accumulation (S11) and replaces the `-inf` mask with structural zeros in L.
  Bit-equality of the replacement: `identical_exp` returns exactly `+0.0`
  for every argument below -87.33655 including -inf (`portable_expf`'s
  saturation branch), so exp-of-masked and never-computed agree bit for bit;
  the departure is recorded because the SPELLING differs, not the bits.
- **DEVIATION 783 — CHUNK_SIZE is part of the arithmetic.** Every reduction
  boundary in S11-S18 moves with Q: which terms flow through the intra-chunk
  GEMMs versus the serial pass IS the summation order. Bit-identity under
  this profile is invariant to batch size, launch geometry, padding and
  vendor, and is NOT invariant to CHUNK_SIZE. Q = 256 is a profile constant
  with exactly the standing of gemm v1's `K_LEAF_MIN`/`MAX_LEAVES`: changing
  it changes the output bits and is a v2, never a tuning knob. The
  CHUNK_SIZE_128 sabotage is its falsifier.
- **DEVIATION 784 — intra-chunk contractions route through gemm.fp32.v1.**
  Upstream's `tl.dot`/einsum/cuBLAS become the pinned GEMM's cells: S4's
  in_proj (`OP_NT`, k = d_model) and out_proj (`OP_NT`, k = d_inner); S12's
  G (k = N = 128, one serial leaf); S14's Y_diag (k = Q = 256, two leaves,
  one fold level); S16's chunk_states (k = Q = 256); S18's C·h (k = N =
  128). The numerical plan is `contract_leaf_size`'s pure function of k;
  the execution plan (batched-many-small-GEMM kernels over (b, c, h)) is
  the builders' and may not touch the arithmetic — gemm contract section
  6.1's batch-invariance clause is the reason this sentence exists.
- **DEVIATION 785 — serial inter-chunk state pass.** The NORMATIVE
  reference's step 3 builds a decay MATRIX over chunk boundaries and
  contracts it (ssd_minimal.py:64-69; `state_passing_ref`:327-350), which
  associates multi-chunk decays as `exp(sum)` where the recurrence
  associates them as products of exps — different bits from three chunks
  up, or from two with a nonzero initial state. The profile takes the
  SERIAL recurrence, which is what the shipped kernel actually runs
  (`_state_passing_fwd_kernel`:72-84), spelled with mamba1 S9's fused fma.
  STATEPASS_MATRIX and STATEPASS_UNFUSED are the falsifiers, and their
  fixtures must witness them (section 8f).
- **DEVIATION 786 — decode is prefill resumption.** Section 5. The carried
  decode state includes the intra-chunk buffer (bounded by Q tokens of xBC
  and dt per sequence); `selective_state_update_ref`'s recurrence is
  semantics-only and its rounding is the STEP_UPSTREAM_RECURRENCE arm. This
  is DEVIATION 721's principle at block scale: when the reference's own two
  paths cannot agree bitwise, the profile picks ONE spelling for both and
  keeps the other as the required-RED arm.
- **DEVIATION 787 — gated RMSNorm pinned spelling.** Gate before norm
  (norm_before_gate = False, rms_norm_ref:26-27), `identical_silu` for the
  gate, the mamba1 S1-S3 fold machinery for sumsq/mean/rstd/products, rstd
  as `identical_rsqrt` = 1/sqrt (rms_norm_ref:29 — HF's `torch.rsqrt` at
  :118 is the same disagreement mamba1 DEVIATION 741 settled, settled the
  same way), eps 1e-5. `group_size` = whole row at G = 1; grouped folds
  have no v1 site.
- **DEVIATION 788 — the dt seam's clamp, and `identical_clamp`.** Order
  bias -> softplus -> clamp per `_chunk_cumsum_fwd_kernel`:73-81 and HF
  :272-276. `identical_clamp(x, lo, hi) = ` the value-pick spelled from
  compares with defined zero-sign behavior, built once in
  `checks/numerics.mojo` (request to the identity lane, do not edit) — the
  ONE primitive this profile needs that does not exist. Present and pinned
  even at the inert default so an active dt_limit changes inputs, not code.
- **DEVIATION 789 — discretize-first pairing.** dt binds to x (`X_d`) and
  to A (`dA`) BEFORE any chunk contraction, per the normative composition
  (ssd_minimal.py:103) and HF (:288-289), on BOTH paths (786 makes decode
  follow). The fused chain instead carries dt into the chunk-state einsum
  (`chunk_state_ref`:1122) and the step pairs `(dt*B)*x`
  (`selective_state_update_ref`:277) — both refused. This is NOT a
  reopening of mamba1 S7/S8: that seam pinned the pairing inside
  `selective_scan_ref`'s per-token recurrence, an algorithm this profile
  does not run; the SSD factorization binds dt structurally earlier, both
  independent Mamba-2 prefill references agree, and the mamba1 pin stands
  untouched in its own profile. PAIR_DT_B is the falsifier.

## 10. Not claimed

- "Bit-identical AI inference" is not claimed; ONE Mamba-2 block is.
- Portability is inherited from Mojo, never claimed as novelty.
- **Nothing in this document has run.** Every gate is RUN OWED; every
  cross-vendor sentence is blank until an E-series leg prints it.
- No Mamba-1 claim is extended by this document, and no Mamba-2 claim is
  earned by Mamba-1's certificates; the profiles are siblings, not
  transitive.
- No BF16/FP16, no training, no backward, no varlen (deferred, section 5),
  no mem-eff fused surface, no multi-block model, no tokenizer.
- No performance number. The identity tier is priced elsewhere and IDENTITY
  IS NOT FREE on any vendor.
- **Mamba-3 exists at the pin** (`mamba_ssm/modules/mamba3.py` and the
  `ops/*/mamba3/` kernel trees at `e9594ce`) and is explicitly OUT OF
  SCOPE — recorded here only so the roadmap decision is made against a
  written sentence, the IDENTICAL_SSM_NOTES.md discipline.

## 11. Build order for the implementation round

Smallest first, one gate per phase, one compile at a time. Proposed slots
follow the lane's mirror convention; DERIVATION_MAP.tsv rows land WITH each
file, not after (the 2026-08-31 lesson).

1. **Phase 0 — primitives.** `identical_clamp` in `checks/numerics.mojo`
   (identity lane's file; REQUEST it, DEVIATION 788) with its check row.
   Gate: `pixi run check-portable-nn` extended. RUN OWED.
2. **Phase 1 — host oracle.** `mamba/checks/mamba2_oracle.mojo`: sections
   4-5 on the CPU through the shared seam functions, stage-carded. Corpus
   generator extension (mamba2 cases, section 8g). Gate:
   `pixi run check-mamba2-corpus` at the base cases. RUN OWED.
3. **Phase 2 — SSD core on device.**
   `mamba/impl/mamba_ssm/modules/ssd_minimal.mojo` (S10-S19 + h_last), no
   float crossing a thread boundary outside the pinned folds. Gates: (a),
   (b), (f)'s SSD arms at B=1 L=4. RUN OWED.
4. **Phase 3 — the block, prefill.**
   `mamba/impl/mamba_ssm/modules/mamba2.mojo` (S1-S9, S20-S22 composed
   around phase 2; projections through `core/gemm.mojo::gemm_nt`). Gates:
   (a), (b), (c), (e), remaining (f) arms. RUN OWED.
5. **Phase 4 — decode resumption.** Same file, section 5's three-piece
   state. Gates: (d), (d2), STEP_UPSTREAM_RECURRENCE. RUN OWED.
6. **Phase 5 — shapes and corpus.** L in {256, 257, 513, 770}, d_model 64,
   adversarial corpus cases, FAST-mode recording, `checks/kernel_matrix.mojo`
   rows for the new kernels. RUN OWED.
7. **Phase 6 — the E-series leg.** Three vendors, the `gemm/E1G_RUNBOOK.md`
   shape, reach measured not inferred (the leg-14 lesson), FAST negative
   control read on every column. RUN OWED — this is the phase the
   completion claim lives or dies on.

## RUN RECORD, 2026-09-01 evening (Apple M4, nice -19, one process at a time, with_identical_mode; orchestrator's box)

Everything below ran the day the contract and its implementation landed.
ONE COLUMN. Nothing here is a cross-vendor claim; phase 6 owns that.

- **Gate (a) + card, (b), (c): PASS on the first clean compile.** Every
  stage bit-identical to the oracle at the default shape; 8 repeated
  launches identical; row bits independent of launch companions
  (B in {1,2,3}) with the negative control moving 24,733 cells.
- **Gate (d): PASS** — decode == prefill bitwise per token, all token
  stages + final-token chunk/state stages, misalignment control differed.
  **Gate (d) cross: PASS** — prefill 250 + decode through 262 == prefill
  262 bitwise; the chain crossed Q = 256 and the resumption buffer
  emptied and refilled. **Gate (d2): PASS** — prefill 512 +
  initial_states = h_last continuation == prefill 520 bitwise.
- **Gate (e): PASS** — 28 nonfinite plants (quiet NaN + inf per each of
  14 names), each read back off the device first, each refused BY NAME
  with 0 stages recorded; the clean control recorded all 26.
- **Gate (g): PASS** — after the corpus generated (17 cases, 41.4 MB,
  sha c928478e7a28ab7400e1cc508bc54a514bd550b84e4bb79d3c7d615017bd7f62,
  generator's own six verify arms green, sha reproduced on a second run
  byte for byte), all 17 cases' input tensors byte-identical between the
  corpus files and the check's in-Mojo generator — two implementations
  of the hash spec agreeing. Two defects found and fixed on the way:
  the gate's text-mode read raised on binary bytes and a catch-all
  renamed every failure MISSING FIXTURE (now binary-safe with
  missing/unreadable/wrong-size/byte-mismatch each refusing by name).
- **All ELEVEN sabotage arms: RED AS REQUIRED**, each naming its own
  seam as the first-moved stage: SEGSUM_DESCENDING (22,785 cells,
  dacs.out), CHUNK_SIZE_128 (78,879, ydiag.out), STATEPASS_MATRIX +
  STATEPASS_UNFUSED (both witnessing fixtures), PAIR_DT_B (15,038,
  xd.out), FOLD_SERIAL_ZERO_SEED (80,110, ydiag.out), S6_BIAS_LAST
  (23,668, conv.out), S6_TAPS_REVERSED (22,970, conv.out),
  CLAMP_BEFORE_SOFTPLUS (115,218, dt.out — after its first form
  refused itself VACUOUS on a fixture that clamped zero cells and got
  a by-construction two-sided witness), GATE_NORM_BEFORE (1,544,
  gnorm.gate), STEP_UPSTREAM_RECURRENCE (58,774 on gate (d) — after
  its guard was re-gated to engage only at l == 1 so the armed build's
  prefill legs run clean).
- **STILL OWED:** the shape sweep (L in {256, 257, 513, 770} beyond the
  gates' own lengths at larger d_model), FAST recording, kernel-matrix
  rows' cross-column reads, check-portable-nn's identical_clamp arm,
  `pixi run check-mamba2-corpus` (the tool's mamba2 support is a
  separate lane item), and PHASE 6 — the three-vendor E-series leg the
  completion claim lives on.
