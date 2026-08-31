# The IDENTICAL FP32 transformer block contract

# PROFILE `mojolearn.identical.transformer.fp32.v1`

Written 2026-08-24, the transformer lane (DEVIATIONS 800-819). The shape of
this document is `mamba/IDENTICAL_MAMBA_CONTRACT.md`'s, which is
`gemm/IDENTICAL_FP32_CONTRACT.md`'s, on purpose. The code form of every
clause will be `transformer/original/transformer_oracle.mojo` (the host
oracle) and `transformer/derived/` (the device spelling), and the clauses
will cite both. **Neither file exists yet. Nothing in this lane has been
built, compiled or run.**

**THE PROFILE NAME IS PART OF THE CONTRACT.** Every card, gate and claim
under this document names `mojolearn.identical.transformer.fp32.v1`.
Changing any seam decision in section 4, any constant in section 3, the
softmax order of section 5 or the stage list in section 9 creates a v2. It
does not amend v1. FAST is unversioned and makes no identity claim.

The completion claim this contract exists to support is one sentence. One
Llama-shaped decoder block, FP32, is bit-identical across Apple, NVIDIA and
AMD GPUs at every stage, at every launch, at every batch composition, at
every sequence length, and between the prefill and decode paths, under the
declared profile. Not bit-identical inference of a model, not FlashAttention,
not training. The cross-vendor half of that sentence is earned ONLY by a
multi-vendor leg and is NOT earned by anything built on one machine.

---

## 0. What this lane does NOT rebuild

This section is first because it is the lane's founding instruction. A
transformer block and a Mamba block share most of their skeleton, the GEMM
lane already owns all the linear algebra, and the numerics lane owns every
transcendental. **The transformer lane's own new arithmetic is RoPE and
softmax and nothing else.** Everything below was read in the tree on
2026-08-24 and is cited by file and by symbol.

| piece | verdict | where it already is |
|---|---|---|
| RMSNorm, all four seams | **REUSED** | mamba contract seams S1-S4. Host form `mamba/original/mamba_oracle.mojo:249-262`. Device form `mamba/derived/transformers/models/mamba/modeling_mamba.mojo::mamba_rms_norm_kernel` (:529-575) and its launcher `::mamba_rms_norm` (:578-598). See the eps note below. |
| residual add | **REUSED** | mamba contract seam S16, `modeling_mamba.mojo::residual_add_kernel` (:988-1004). |
| SiLU | **REUSED** | `original/numerics.mojo::identical_silu` over `::portable_siluf`, DEVIATION 744, IDENTITY_PATHS row 53. |
| every linear projection (q, k, v, o, gate, up, down) | **REUSED** | profile `mojolearn.identical.gemm.fp32.v1`, entry `gemm/original/gemm_identical.mojo::identical_gemm(ctx, c, a, b, m, n, k, op)` (:1363), ops `OP_NN = 0`, `OP_NT = 1`, `OP_TN = 2` from `gemm/original/gemm_oracle.mojo:194-198`. Certified three-vendor at E3 round 11 (`144aa5b`), IDENTITY_PATHS row 40. |
| **the QK product** | **REUSED**, and this was not on the lane brief's list | it is a `gemm.fp32.v1` `OP_NT` cell with `k = head_dim` (section 4, seam S11). |
| exp, division, rsqrt, sqrt, log, cos, pow | **REUSED** | `original/numerics.mojo`. `portable_expf`, `identical_div`, `identical_rsqrt`, `portable_sqrtf`, `portable_cosf`, `portable_powf`. DEVIATIONS 258 and 740-746, IDENTITY_PATHS rows 10, 12, 49-54. |
| `pinned_mul`, the uncontractible multiply | **REUSED IN SPIRIT, COPIED IN FACT** | DEVIATION 720. It is NOT in `original/numerics.mojo`. Three identical copies exist, at `mamba/original/mamba_oracle.mojo:41`, `mamba/derived/transformers/models/mamba/modeling_mamba.mojo:207` and `mamba/derived/mamba_ssm/ops/selective_scan_interface.mojo:263`. This lane needs a fourth or an import. DEVIATION 816, section 12.3. |
| `identical_fmax`, the order-free maximum | **REUSED**, and it was built for this seam | `original/numerics.mojo::identical_fmax` over `::portable_fmaxf`, DEVIATION 825, landed 2026-08-24. It is the softmax row maximum (section 5.1) and it closes IDENTITY_PATHS row 13 at this site. |
| deterministic block folds | **BOTH REFUSED** | `core/pinned_reduce.mojo::pinned_block_sum` (:73), `::pinned_block_max` (:159), `::pinned_block_min` (:193). `pinned_block_sum` may not be the softmax denominator (section 5.3) and `pinned_block_max` may not be the softmax row max (section 5.1). Neither refusal is about determinism; both helpers are perfectly deterministic. They compute different answers. |
| the stage card and the differ | **REUSED** | `core/identity_trace.mojo` (`IdentityTrace.record_device` :313, `::record_host` :369), `tools/identity_trace_diff.py`. |
| the refusal of a nonfinite input | **REUSED** | `mamba/original/mamba_oracle.mojo::refuse_nonfinite` (:57), tested by BITS because Metal flushes compare operands (IDENTITY_PATHS row 49). |
| the independent reference corpus pattern | **REUSED** | `mamba/corpus/`, generator and `tools/mamba_corpus_check.py`. A transformer corpus is a later phase and does not exist. |
| **RoPE, the angle table and the rotation** | **NEW** | nothing in this repository computes a sine. Section 4 seams S6 through S10. |
| **softmax** | **NEW** | there is no device softmax anywhere in the tree. The four `softmax` hits under `gbdt/` and `original/` are CatBoost's HOST float64 multiclass probability, a different arithmetic in a different lane. Section 5. |
| **the attention-weighted value sum** | **NEW**, and deliberately NOT routed through the GEMM | section 4 seam S19 and section 7.2. This is the least obvious decision in the document. |
| `portable_sinf` and `identical_sin` | **REUSED**, and NOT this lane's to edit | DEVIATION 820, `original/numerics.mojo`, LANDED 2026-08-24 by the concurrent numerics lane while this document was being written. It shares `_cephes_sincosf_core` with `portable_cosf`, so RoPE's sin and cos of one angle come from one certified reduction. Its docstring addresses this lane by name about the domain; section 3 answers it. |
| `portable_tanhf`, `portable_erff`, both GELU forms | **LANDED AND NOT USED HERE** | DEVIATIONS 821-824. They are not on the Llama path and are not seams of this profile (section 11). |

Three corrections to the brief this lane was given, because a table that
agrees with a guess is worth less than one that is right.

1. `pinned_mul` is not in `original/numerics.mojo`. It is DEVIATION 720 and
   it is duplicated three times inside `mamba/`.
2. The RMSNorm kernel is reusable but not yet parameterized. `RMS_EPS` is a
   module constant `1e-5` at `mamba/original/mamba_fixture.mojo:44`, read
   directly by `mamba_rms_norm_kernel`. Llama's default is `1e-6`. Taking
   the kernel therefore requires lifting eps to an argument, which is
   DEVIATION 801 and a cross-lane edit that the mamba lane must agree to.
3. `core/pinned_reduce.mojo` also ships `pinned_block_max` and
   `pinned_block_min`, which the brief did not mention. They matter here
   because the softmax has a max reduction, and they turn out to be the
   WRONG helper for it (section 5.1).
4. The concurrent numerics lane's range is 820-825 and it is not four
   functions but seven. **DEVIATION 825 is `portable_fmaxf` and
   `identical_fmax`, an order-free maximum written specifically for
   softmax's row reduction**, and it decides section 5.1. It was not in the
   brief and it landed on 2026-08-24, after this lane started reading.

Line numbers into `original/numerics.mojo` are deliberately absent from
this document. That file is under concurrent edit by the numerics lane and
moved by about 200 lines during the writing of this one; symbol names and
DEVIATION numbers are stable and line numbers are not.

## 1. The reference, pinned

| what | where | pin |
|---|---|---|
| the block order | `src/transformers/models/llama/modeling_llama.py::LlamaDecoderLayer.forward` (:295-324) | huggingface/transformers `d56c55bf564ddb176759eb6ec199442682564916` |
| RMSNorm | `::LlamaRMSNorm.forward` (:62-67) | same |
| the rotary table | `::LlamaRotaryEmbedding.compute_default_rope_parameters` (:93-109) and `::LlamaRotaryEmbedding.forward` (:113-127) | same |
| the rotation | `::rotate_half` (:130-134) and `::apply_rotary_pos_emb` (:137-160) | same |
| attention, the EAGER path | `::eager_attention_forward` (:191-213), `::repeat_kv` (:179-188), `::LlamaAttention.forward` (:243-281) | same |
| the MLP | `::LlamaMLP.forward` (:174-176) | same |
| the additive causal mask's VALUE | `src/transformers/masking_utils.py:601-603`, `torch.finfo(dtype).min` where the mask is False and `0.0` where it is True | same |
| the config defaults | `src/transformers/models/llama/configuration_llama.py:64-89` | same |
| the device kernel SHAPE for reference attention | `max/kernels/src/nn/attention/gpu/mha.mojo::mha_gpu_naive` (:6328-6501) with `_bmm0_bs` (:6504-6640) and `_bmm1_bs` (:6643-6736). One thread owns one `(query, key)` score, the dot over `depth` is serial inside that thread, there is no cross-thread reduction, and the scores are materialized into a full `[batch*heads, q_len, kv_len]` buffer in an FP32 accumulator | modular/modular `10d978e3c783ef940d1d30d0a10852b69fe285c8` |
| the device kernel SHAPE for softmax | `max/kernels/src/nn/softmax.mojo::softmax_kernel` (:840-1015), one block per row, a max reduction then an exp-sum then a scale; `::_softmax_warp_kernel` (:1018-1095), one warp per row for short rows | same |
| the device SHAPE for RoPE | `max/kernels/src/nn/rope.mojo::_rope` (:36-45), `::apply_rope` (:65-115), `::rope_ragged` (:118-310). **The sin/cos table is PRECOMPUTED OUTSIDE the kernel** and arrives as `freqs_cis` of shape `[max_seq_len, rope_dim]`; the kernels only load rows from it. `::get_safetensors_idx` (:51-53) is the rotate-half pairing | same |
| the device SHAPE for RMSNorm | `max/kernels/src/nn/normalization.mojo::_rms_norm_gpu_block_subkernel` (:742-806), one block per row; `::rms_norm_gpu_warp_per_row` (:450-576) | same |
| the projections' arithmetic | profile `mojolearn.identical.gemm.fp32.v1`, IDENTITY_PATHS row 40, certified three-vendor at E3 round 11 (`144aa5b`) | this repository |

Checkouts live in `/Users/andrewhendel/CascadeProjects/upstream/`. There is
no PyTorch checkout there, so **ATen's own softmax kernel could not be read**
and section 5.4 records what that leaves unverified. `max/kernels/` is the
only part of the modular monorepo present in the checkout, so `max/graph`,
the serving layer and the `kv_cache` package are absent; where a MAX symbol
is cited from outside `max/kernels/` this document says it could not be read
rather than guessing a path.

MAX's FlashAttention family is cited in section 6 where its shape is the
reason for an exclusion, not as a thing to mirror.

## 2. What one block call is

    residual  = x
    h         = residual + self_attn(rmsnorm(x, w_norm1))         LDL:305-317
    residual2 = h
    out       = residual2 + mlp(rmsnorm(h, w_norm2))               LDL:320-323

with `self_attn`, in order (`LlamaAttention.forward` :243-281, eager path)

    q_proj, k_proj, v_proj  ->  reshape to heads  ->  RoPE on q and k
    ->  append k and v to the cache  ->  repeat_kv  ->  q . k^T
    ->  * scaling  ->  + causal mask  ->  softmax  ->  . v
    ->  reshape  ->  o_proj

and with `mlp` (`LlamaMLP.forward` :174-176)

    down_proj( silu(gate_proj(x)) * up_proj(x) )

Inference only. Every `nn.Linear` in the reference carries `bias=False` at
the config defaults (`attention_bias` and `mlp_bias`, both `False`,
configuration_llama.py:81 and :83), and this profile REFUSES a nonzero bias
rather than specifying where it would round. `attention_dropout` is `0.0`
and dropout is refused for the same reason. The decode step is the SAME
arithmetic read one token at a time with the KV cache carried (section 7).

## 3. Profile constants

Two kinds of constant live here and mixing them up is how a version number
stops naming an arithmetic. The FROZEN column says whether changing the
value is a v2 or merely a different model.

| constant | value | frozen? | source |
|---|---|---|---|
| dtype | Float32 for weights, activations, accumulators, the KV cache and the rotary table | **YES** | Andrew's order |
| rms eps | `1e-6` (`0x358637BD`) | **YES** | `LlamaConfig.rms_norm_eps`, configuration_llama.py:73 |
| rope theta | `10000.0` (`0x461C4000`) | **YES** | `RopeParameters` default base, modeling_rope_utils.py:177 |
| rope type | `default` only, so `attention_scaling` is exactly `1.0` and the multiply at modeling_llama.py:124-125 is bit-inert and is NOT spelled | **YES** | `compute_default_rope_parameters` :103-109 |
| activation | SiLU, the reference's one-division spelling | **YES** | `hidden_act` default `"silu"`, configuration_llama.py:70 |
| attention scale | `identical_rsqrt(Float32(head_dim))`, computed ONCE on the host, stored FP32 | **YES** | `LlamaAttention.__init__` :226, `self.scaling = self.head_dim ** -0.5` |
| mask fill | `-3.4028234663852886e+38` (`0xFF7FFFFF`), ADDED, not selected | **YES** | masking_utils.py:601-603 |
| unmasked fill | `+0.0`, ADDED, and the add may NOT be elided | **YES** | masking_utils.py:603 |
| max absolute position | strictly less than `8192` | **YES** | the Cody-Waite domain of `_cephes_sincosf_core`, shared by `portable_cosf` and `portable_sinf`. DEVIATION 820's docstring names this lane and requires the choice to be made here |
| biases | none, anywhere | **YES** | configuration_llama.py:81, :83 |
| d_model, n_heads, n_kv_heads, head_dim, intermediate_size | free, subject to the divisibility rules below | no | model shape, not arithmetic |
| B, L, cache length | free | no | launch shape, not arithmetic |

Divisibility, refused by name rather than silently truncated. `d_model ==
n_heads * head_dim`, `n_heads % n_kv_heads == 0`, `head_dim` even (RoPE
pairs halves), `d_model > 0`, `intermediate_size > 0`.

**GQA is ADMITTED in v1, and `n_kv_heads == n_heads` is not required.**
`repeat_kv` (modeling_llama.py:179-188) is `expand` followed by `reshape`,
which is a COPY. A copy moves bits untouched and is not a seam (section 4's
preamble), so admitting grouped-query attention adds no rounding and no
decision. Refusing it would have excluded the shape every deployed
Llama-3-class model actually uses, for nothing. DEVIATION 813. The cost is
gate coverage rather than arithmetic, and it is a real cost, because at
`n_rep == 1` a broken head-to-kv-head index map is invisible. **The gates
must therefore carry both `n_rep == 1` and `n_rep == 2`.**

Scale note, measured on the host 2026-08-24 with float32 round-trips and NOT
on a device. `head_dim ** -0.5` evaluated in float64 and rounded once to
FP32 equals `f32div(1, f32sqrt(head_dim))` at `head_dim` in
{8, 16, 32, 64, 80, 128, 256}. That is agreement at seven points and not a
proof. The profile pins the SPELLING (`identical_rsqrt`, DEVIATION 741) so
that the value does not depend on the host's libm, which is IDENTITY_PATHS
row 18's hazard applied to a constant. At `head_dim = 16` the scale is
`0.25` (`0x3E800000`) exactly and at `head_dim = 64` it is `0.125`
(`0x3E000000`) exactly, so the gate shape below cannot see a wrong scale
spelling and a fixture at a non-power-of-four `head_dim` is required.

**The gate shape.** `d_model = 32`, `n_heads = 2`, `head_dim = 16`,
`n_kv_heads` in {1, 2}, `intermediate_size` in {64, 300}, `B` in {1, 2, 3},
`L` in {1, 4, 16, 64, 257}, absolute positions in `[0, 512)`. The
arithmetic per cell does not read B, L, the cache length or the launch, so
shape coverage is about the gates and not about the profile.

`intermediate_size = 300` is not decoration. Every projection in a small
config has `k <= 128`, which is `P == 1` under the GEMM profile, so the
whole balanced fold tree sits unexercised inside the block exactly as it
does in the Mamba lane. At `intermediate_size = 300` the `down_proj` has
`k = 300`, giving `P = 3` with a ragged 44-element last leaf and one carry.
That is the GEMM contract's own clause-5 shape (its section 12.1) lifted
into this block on purpose, and without it the composition never runs the
tree at all. A second fixture at a non-power-of-four `head_dim`, say 24,
covers the inexact scale.

## 4. The seams, every one, with the fused-or-unfused decision

FMA contraction is PER SEAM. A seam marked FUSED is one rounding through
`identical_mul_add` (fma). A seam marked PRODUCT is one rounding through
`pinned_mul(a, b) = identical_mul_add(a, b, -0.0)`, the spelling no codegen
may contract into a neighboring add and which preserves a `-0.0` product
(a `+0.0` addend would launder it, the GEMM lane's F6a lesson). Every seam's
RESULT passes `ftz`, and every operand LOADED from a buffer passes `ftz`
(IDENTITY_PATHS row 10's checklist unit).

**Copies are NOT seams.** `repeat_kv`, the head reshape and transpose, the
concatenation of `freqs` with itself at modeling_llama.py:123, the KV cache
append, and the flattening before `o_proj` all move bits untouched.

| # | seam | reference spelling | pinned spelling | fused? |
|---|---|---|---|---|
| S1 | RMSNorm sum of squares | `hidden_states.pow(2).mean(-1)` (LRN:65), fold order torch's | serial ascending j from `+0.0`, `acc = ftz(fma(x_j, x_j, acc))`; one fold per row, no block fold. The mamba contract's S1 unchanged | FUSED |
| S2 | RMSNorm mean and rstd | `torch.rsqrt(variance + eps)` (LRN:66) | `mean = ftz(identical_div(sumsq, d_model))`; `rstd = ftz(identical_rsqrt(ftz(mean + eps)))`, never the hardware rsqrt intrinsic (DEVIATION 746). Mamba S2 with eps `1e-6` | n/a |
| S3 | `hidden_states * rstd` (LRN:66) | one product | `pinned_mul(x_j, rstd)`. Mamba S3 | PRODUCT |
| S4 | `weight * hidden_states` (LRN:67) | one product | `pinned_mul(w_j, S3)`. Mamba S4 | PRODUCT |
| S5 | the seven projections | `nn.Linear`, weight `[out, in]`, `y = x @ W^T`, cuBLAS | `gemm.fp32.v1` `OP_NT` cells. `q/k/v/gate/up` have `k = d_model`, `o_proj` has `k = n_heads*head_dim`, `down_proj` has `k = intermediate_size`. The GEMM refuses no shape | per gemm v1 |
| S6 | `inv_freq[i] = 1 / theta**(2i/head_dim)` | `base ** (arange(0, dim, 2).float() / dim)` then a reciprocal, all FP32 (LRE:108) | HOST, once per (theta, head_dim). `e = ftz(identical_div(Float32(2*i), Float32(head_dim)))`; `inv = ftz(identical_div(1.0, portable_powf(theta, e)))`. NOT the host libm `pow`, which is IDENTITY_PATHS row 18 (cross-vendor is cross-HOST for a constant). DEVIATION 809 | n/a |
| S7 | `freqs[t, i] = position_t * inv_freq[i]` | a `k = 1` matmul in FP32 with autocast explicitly disabled (LRE:114-122) | `pinned_mul(Float32(abs_pos_t), inv_freq[i])`. At `k = 1` the GEMM leaf is `ftz(fma(a, b, +0.0))`, bit-equal to the product here because both operands are non-negative, so the two spellings coincide and the cheaper one is pinned. **The position is the ABSOLUTE position, always** | PRODUCT |
| S8 | `cos(freqs)`, `sin(freqs)` | `emb.cos()`, `emb.sin()`, then `* attention_scaling` which is exactly `1.0` (LRE:106, :124-125) | `identical_cos` (DEVIATION 258) and `identical_sin` (DEVIATION 820), which share `_cephes_sincosf_core`, so both halves of one rotation come from ONE reduction rather than two functions wearing one name. The `* 1.0` is bit-inert on every finite input and both zero signs and is NOT spelled | n/a |
| S9 | `q * cos` and `rotate_half(q) * sin` | two separate products (LAR:158) | two `pinned_mul` calls. `rotate_half` pairs index `j` with `j + head_dim/2` and negates the upper half into the lower (LRH:130-134), and the table is `cat(freqs, freqs)` (LRE:123) so `cos[j]` and `cos[j + head_dim/2]` are the SAME value | PRODUCT, twice |
| S10 | `(q*cos) + (rotate_half(q)*sin)` (LAR:158) | one add of two already-rounded products | `ftz(ftz(S9a) + ftz(S9b))`. **UNFUSED**, because the reference rounds both products before adding. An fma here is one rounding where the reference has three, and it is the natural thing for a kernel to write. DEVIATION 811, sabotage `S10_ROPE_FUSED` | add |
| S11 | `q . k^T` | `torch.matmul(query, key_states.transpose(2, 3))` (EAF:204), cuBLAS batched | `gemm.fp32.v1` `OP_NT` with `k = head_dim`, one call per `(batch, head)`. `k = head_dim` is the SAME in prefill and decode, which is what makes this reuse decode-safe (section 7.2). DEVIATION 808 | per gemm v1 |
| S12 | `* scaling` (EAF:204) | one product, applied to the FINISHED dot, not folded into q | `pinned_mul(score, scale)`. Pre-scaling `q` instead is a different answer (one rounding per q element rather than one per score) and is sabotage `S12_SCALE_INTO_Q` | PRODUCT |
| S13 | `+ attention_mask` (EAF:206) | one add of `finfo(float32).min` where masked and of `+0.0` where not | `ftz(ftz(score) + mask_val)`. **Not a select.** Two clauses below | add |
| S14 | the softmax row max | `nn.functional.softmax` subtracts the row max | `identical_fmax` (DEVIATION 825) folded over the whole row in ANY order. Section 5.1 | n/a |
| S15 | `s - m` | subtraction | `ftz(ftz(s) - ftz(m))` | subtract |
| S16 | `exp(s - m)` | `torch.exp` inside softmax | `identical_exp` (IDENTITY_PATHS row 12's polynomial) | n/a |
| S17 | the denominator | a sum whose order is ATen's | section 5.3. Serial ascending over the ABSOLUTE key index from `+0.0`, plain adds | add |
| S18 | `e / denom` | see section 5.4 | `ftz(identical_div(e, denom))`, ONE division per weight, never a reciprocal multiplied in | n/a |
| S19 | the attention-weighted value sum | `torch.matmul(attn_weights, value_states)` (EAF:210), cuBLAS | serial ascending over the ABSOLUTE key index from `+0.0`, `acc = ftz(fma(w[j], v[j][d], acc))`. **Deliberately NOT a `gemm.fp32.v1` call**, and section 7.2 is the reason. DEVIATION 807 | FUSED |
| S20 | `silu(gate_proj(x))` (LMLP:175) | `F.silu`, which is ATen's `z / (1 + exp(-z))`, ONE division | `identical_silu` (DEVIATION 744). **Not** `x * sigmoid(x)`, which is two roundings and which is what MAX itself spells (`max/kernels/src/nn/activations.mojo:249`). Sabotage `S20_SILU_MUL_SIGMOID` | n/a |
| S21 | `silu(gate) * up` (LMLP:175) | one product | `pinned_mul(silu_out, up_out)` | PRODUCT |
| S22 | `residual + attn_out` (LDL:317) | one add | `ftz(ftz(residual) + ftz(o_proj_out))`. The mamba contract's S16 | add |
| S23 | `residual2 + mlp_out` (LDL:323) | one add | `ftz(ftz(residual2) + ftz(down_proj_out))`. Mamba S16 again | add |

Abbreviations in the reference column are `modeling_llama.py` symbols. LRN
is `LlamaRMSNorm`, LRE is `LlamaRotaryEmbedding`, LRH is `rotate_half`, LAR
is `apply_rotary_pos_emb`, LMLP is `LlamaMLP`, EAF is
`eager_attention_forward`, LDL is `LlamaDecoderLayer`.

### 4.1 Two clauses about S13, the mask, because a mask looks inert

**(a) The additive `+0.0` at an UNMASKED cell may not be elided.** IEEE
round-to-nearest gives `x + (+0.0) == x` for every finite x, every infinity
and every NaN EXCEPT `x = -0.0`, where it gives `+0.0`. That is
IDENTITY_PATHS row 39 in one line. An implementation that skips the add
where the mask is true therefore keeps a `-0.0` score that the reference
launders, and it will pass every fixture that does not PLANT a negative zero
in a score. A `-0.0` score is reachable only through `ftz` of a negative
subnormal GEMM accumulator (GEMM contract 9.2(a) says products alone can
never produce one), so the fixture has to plant it by bits. Sabotage
`S13_MASK_SELECT`.

**(b) `-FLT_MAX` and not `-inf`, and the choice is not cosmetic.** The
reference's mask value is `torch.finfo(dtype).min` (masking_utils.py:601),
which is `-3.4028234663852886e+38`. With `-inf` a fully masked row gives
`-inf - (-inf) = NaN`, and a computed NaN's payload is vendor-shaped
(IDENTITY_PATHS row 39 measured three payloads for one IEEE answer), so a
certified stage could not contain one. The causal mask never produces a
fully masked row, because position t always attends to itself, so the
difference is not reachable through the mask alone. It IS reachable through
an extreme score. `s + (-FLT_MAX)` equals `-FLT_MAX` exactly for
`|s| < 2^103` and overflows to `-inf` below roughly `-1e31`, so a planted
score of magnitude `1e35` separates the add from a select and separates
`-FLT_MAX` from `-inf`. Sabotage `S13_MASK_NEG_INF`. MAX's own reference
kernel makes the same choice in the other direction and is worth knowing
about, because it is inconsistent with itself: `_bmm0_bs` stamps
out-of-range scores with `min_or_neg_inf` (mha.mojo:6639) while
`softmax_kernel` seeds its max with `Scalar[dtype].MIN`, the finite one
(softmax.mojo:958), and `_softmax_warp_kernel` seeds with `min_or_neg_inf`
(softmax.mojo:1063). Two spellings of the same reduction in one file.

## 5. The softmax reduction order

This is where cross-vendor identity is won or lost, so it gets its own
section rather than four rows of a table.

### 5.1 The max (S14) is `identical_fmax`, and its fold shape is FREE

    m(row) = the fold of identical_fmax over EVERY element of the row,
             masked cells included, in ANY order, with no seed.

`identical_fmax` is DEVIATION 825, `original/numerics.mojo`, added by the
numerics lane on 2026-08-24 for this seam and for no other. Under IDENTICAL
it is `portable_fmaxf`, which canonicalizes a NaN operand to `0x7FC00000`
first, flushes both operands, and then selects on `_total_order_key`, the
DEVIATION 204 map under which `+0.0` keys at `0x80000000` and `-0.0` keys at
`0x7FFFFFFF`. There is no hardware `max` instruction and no float compare in
it. Under FAST it is the stdlib `max` and there is no contract.

**Because the result is commutative and associative over all of Float32
including both zeros and NaN, this profile does not pin a fold topology for
the max.** That is the only place in this document where an execution plan
may choose its own tree, and it may do so because the operation is exactly
associative rather than because the difference is thought to be small.

Why an unpinned float max would be a defect here, in one paragraph, because
this is the site IDENTITY_PATHS row 13 was written about. A float `max` is
exactly commutative and associative EXCEPT at `+0.0` against `-0.0`, which
compare equal, and there the answer is the implementation's. Row 39 measured
it on three columns on 2026-08-23. `max(+0.0, -0.0)` is `-0.0` on Apple, the
second operand, and `+0.0` on NVIDIA and AMD, IEEE-2019 `maximum`. **That is
a live three-vendor split and a row of attention scores reaches both zero
signs easily**, a masked lane one way and a flushed subnormal the other.

**`core/pinned_reduce.mojo::pinned_block_max` may NOT be used**, and this is
the trap, because it is the deterministic-looking helper already in the tree.
Its fold is a plain `other > red[tid]` compare (:159-190), which is precisely
the spelling row 13 closed everywhere else, and its own block comment states
that a caller whose inputs can carry `+-0.0` or NaN must say why before using
it. This caller cannot say why. A halving tree over `identical_fmax` is fine.
The shipped `pinned_block_max` is not, and the difference is one function
call.

Two honest notes about what the hazard costs, because it is smaller than it
looks and the clause is worth keeping anyway.

- **Downstream it costs nothing.** The only consumer of `m` is S15, and
  `s - (+0.0)` and `s - (-0.0)` agree bit for bit for every finite s except
  `s = -0.0`, where they give `-0.0` and `+0.0`; `exp` of either is exactly
  `1.0`. So a vendor split at S14 is laundered before it reaches an output.
- **On the card it costs the clause.** `attn.max` is a recorded stage
  (section 9), and clause (a) requires every stage to agree bitwise. A
  laundered divergence is still a divergence at the stage that produced it.
  This is the mamba lane's absorption finding pointed the other way. There,
  thirteen moved stages were absorbed by a residual add and an output-only
  gate called the sabotage inert. Here, an unpinned max is invisible in the
  output and visible on the card. **The card is the only instrument that can
  see this clause at all**, which is the argument for recording `attn.max`
  rather than treating it as an internal.

Sabotage `S14_MAX_PLAIN_COMPARE` replaces `identical_fmax` with
`a > b ? a : b` and folds one planted row in two orders. It must move
`attn.max` on a row carrying a planted `-0.0` beside a `+0.0`, and it must
NOT move it on an ordinary row, which is what makes it a reach proof rather
than a smoke test.

### 5.2 What the max is taken OVER

The masked cells, all of them, in the same ascending order as everything
else. Not "the unmasked prefix". This matters for section 7.

### 5.3 The denominator (S17) is a SERIAL ASCENDING CHAIN, and `pinned_block_sum` is REFUSED

    denom(row) = acc = +0.0
                 for j = 0 .. kv_len - 1                ASCENDING, ABSOLUTE
                     acc = ftz( ftz(acc) + ftz(e[j]) )
                 return ftz(acc)

Plain adds. There is nothing to fuse, because `e[j]` is not a product.

**`core/pinned_reduce.mojo::pinned_block_sum` may NOT be used for this**, and
saying so is the point of the clause. It is a halving tree, its own docstring
says a halving tree and CUB's warp-then-block shape combine different
partials, and a halving tree is not a serial ascending chain either. Reaching
for the deterministic block fold because it is the deterministic block fold
is the single most likely way to get this wrong, and it would be wrong in a
way that passes every launch-invariance gate, because the tree is perfectly
launch-invariant. It is simply a different sum. Sabotage
`S17_DENOM_HALVING_TREE`, which must move `attn.denom` at any kv length of
3 or more.

There are two reasons for the serial chain rather than the GEMM's
leaf-plus-balanced-tree, and the second is the load-bearing one.

1. It is the same spelling as S1 and as the mamba contract's S10, so the
   block has one fold shape rather than two.
2. **It is what makes decode equal prefill and makes the answer independent
   of sequence length.** Under the GEMM's topology, `P` is a pure function of
   the contraction length `k`, so a row folded over 257 keys and the same row
   folded over 5 keys have different trees and different bits. Under a serial
   ascending chain seeded `+0.0`, a tail of exactly-`+0.0` terms is bitwise
   inert, and section 7.1 shows the masked terms are exactly `+0.0`. Section
   7 turns that into a theorem.

The price is stated rather than hidden. A row's fold may not be split across
threads, so v1's kv length is bounded by what one thread will walk, and this
profile is reference quality and slow by construction. **The exit is not a
free choice.** A v2 that wants a tree must fix the fold LENGTH so that
prefill and decode fold the same number of terms, which means folding over
the allocated cache length rather than the used one, which makes `P` a
function of an allocation quantity and puts batch invariance back at risk.
That is a research question and it is named here so that whoever opens it
knows it is one.

### 5.4 The division (S18), and the one thing this lane could not read

`w[j] = ftz(identical_div(e[j], denom))`, one division per weight,
IDENTITY_PATHS row 49's `portable_divf`.

**A reciprocal multiplied in is the alternative and it is a different
answer.** `e * (1/denom)` rounds twice where `e / denom` rounds once, and
they differ in the last bit on ordinary inputs. **The reference's own
spelling could not be verified.** `nn.functional.softmax` dispatches into
ATen and there is no PyTorch checkout in
`/Users/andrewhendel/CascadeProjects/upstream/`, so whether ATen's CUDA
softmax divides or multiplies by a reciprocal is not known to this lane.
MAX's `softmax_kernel` multiplies by a reciprocal (softmax.mojo, the step-3
scale), which is evidence about MAX and not about the reference. The
profile pins the DIVISION because it is the spelling with one rounding and
because `identical_div` is already characterized per class on Apple. **This
is a stated gap, not a decision made on evidence**, and it is the first thing
to check when a PyTorch checkout lands. Sabotage `S18_RECIPROCAL_MUL`.

### 5.5 One axis, one direction, one origin

S17 and S19 walk the key axis ASCENDING, and every one of S11 through S19
indexes it BY ABSOLUTE POSITION, which is the index into the KV cache and
not an offset into whatever slice this launch happens to hold. S14 is the
single exception to the ascending clause and only to that clause, because
`identical_fmax` is associative and its fold shape is free; it still reads
the same elements, over the same absolute range, and section 5.2 says so.

An implementation that walks a per-launch local index gets the right answer
in prefill and a different one in decode, and it is section 7's clause that
catches it, not section 5's.

## 6. Why the EAGER path, and why FlashAttention and SDPA are out of scope

This is the most important scoping decision in the document.

`LlamaAttention.forward` chooses its attention implementation at
modeling_llama.py:264-266 through `ALL_ATTENTION_FUNCTIONS`, defaulting to
`eager_attention_forward`. **v1 pins the eager path and nothing else.**
FlashAttention, SDPA, paged attention and chunked prefill are all excluded.

FlashAttention is not a faster spelling of the same arithmetic. It is an
ONLINE softmax. It carries a running maximum and a running denominator per
query row, and every time a new key tile raises the maximum it RESCALES the
accumulated output and the accumulated denominator by `exp(old_max -
new_max)`. Read the shape in MAX's own kernel.
`max/kernels/src/nn/attention/gpu/mha.mojo::mha_single_batch` holds `rowmax`
and `rowsum` as per-thread register arrays (:3161-3166) and calls
`_online_softmax_iter_for_mma_output` per KV tile (:3487-3500);
`max/kernels/src/nn/softmax.mojo::_online_softmax_correction` (:2908-2940) is
the rescale itself.

Three consequences, each of which makes this a research question rather than
a porting question.

1. **The number of rescalings is the number of KV tiles, and the tile size
   is an execution-plan quantity.** Two vendors with different tile widths
   perform different numbers of rounded rescalings on the same row. That is
   IDENTITY_PATHS rows 3 and 7 exactly, *a block count is a summation order*,
   moved inside the softmax where the contract cannot reach it by pinning a
   fold topology.
2. **The decode path splits K further.** `mha_decoding_single_batch` spills
   a per-partition `exp_sum_ptr` and `qk_max_ptr` (mha.mojo:4916-4917) for
   `mha_splitk_reduce` (:6120-6325) to combine, and `num_partitions` is
   chosen from the shape and the machine. The combine is a second summation
   order chosen by a hardware number.
3. **The exponent base is not the reference's.** These kernels use `exp2`
   with the `1/sqrt(d)` scale folded into `scale_log2e` (mha.mojo:3401-3405),
   so even the per-element arithmetic differs from `exp(s - m)`.

A bit-identical online softmax is a real and interesting object. It would
need the rescale schedule to be a pure function of the kv length and a
profile constant, exactly as the GEMM's leaves are a pure function of `k`,
and it would need its own contract, its own oracle and its own separating
fixtures. **It is a v2 at the earliest and it is not in this lane's scope.**
Writing it as an optimization of v1 would be the single largest way to lose
the property this repository exists to hold.

SDPA is excluded for a smaller reason that is just as final. It dispatches
to whichever backend the runtime picks, which means the arithmetic is chosen
at run time, which is the same class of defect as a vendor BLAS dispatch
(IDENTITY_PATHS row 40's pathway column).

The materialized `[B, n_heads, L, S]` score buffer that the eager path
requires is a cost, and it is the same shape MAX's own reference kernel
allocates (`mha_gpu_naive`, mha.mojo:6328-6501). It is also what makes
`attn.scores`, `attn.masked`, `attn.max`, `attn.exp`, `attn.denom` and
`attn.weights` recordable as separate card stages, and a lane whose whole
instrument is the per-stage card should not begin by fusing the stages away.

## 7. State, decode, sequence length, and batch composition

### 7.1 The masked tail is exactly `+0.0`, and everything rests on it

Take query position t in a prefill of length L, with the causal mask. For
`j > t` the score is `s + (-FLT_MAX)`, which is `-FLT_MAX` or `-inf`
(section 4.1(b)). The row max `m` is at least the diagonal score `s_tt`,
which is finite, so `masked - m` is about `-3.4e38` and `identical_exp` of
it is exactly `+0.0` (`portable_expf` returns `+0.0` below `-87.33655`,
numerics.mojo:291). The denominator is at least `exp(0) = 1.0` because the
maximal element contributes exactly `1.0`, so `identical_div(+0.0, denom)`
is exactly `+0.0`. Therefore

> **every masked attention weight is exactly `+0.0`, and it is `+0.0` and
> not `-0.0`.**

Now the two folds. S17 seeds `+0.0` and adds nonnegative terms, so its
accumulator is never `-0.0`, and `x + (+0.0) == x` for every value it can
hold. S19 seeds `+0.0` and its masked terms contribute
`fma(+0.0, v, acc)`, which is `acc + (+-0.0)`, which is `acc` for every
`acc` except `acc = -0.0`.

**THE SENTENCE THAT STOOD HERE SAID "and the seed forbids that". IT IS FALSE
AND THE COUNTEREXAMPLE IS TWO ADDS LONG.** DEVIATION 1327, found by the
embedding lane 2026-08-25 and verified by direct computation. The `+0.0` seed
forbids reaching `-0.0` by ADDITION. It does not forbid reaching it through
`ftz` OF A NEGATIVE SUBNORMAL PARTIAL SUM, and `ftz` runs at every seam:

    seed            +0.0
    + 0x80C00000 -> 0x80C00000   (-1.7632e-38, an ordinary NORMAL)
    + 0x00800000 -> 0x80400000   (-5.8775e-39, SUBNORMAL)
    ftz(...)     -> 0x80000000   = -0.0

Both operands are ordinary normals. A third exactly-`+0.0` contributor then
moves the bit back to `0x00000000`, so the masked tail is NOT inert on that
accumulator.

**For S19 the hole is REACHABLE BY CONSTRUCTION**, because value contributions
carry both signs. It is unreachable on the fixtures this lane ships today,
which is why every gate has passed, and that is exactly the condition this
project calls `[[reached-but-inert]]` -- a clause that is true of the fixtures
and false of the contract.

So the masked tail is bitwise inert in both folds **for every accumulator that
is not `-0.0`, and an accumulator CAN be `-0.0`.** OWED: a fixture that plants
the subnormal-cancellation case (the embedding lane ships one as F-SUBACC),
and a decision about whether S19 seeds differently or the clause is narrowed.

### 7.2 Decode equals prefill, and it is structural

The recurrent state between calls is the KV cache, `k_cache` and `v_cache`,
each `[B, n_kv_heads, S, head_dim]`, appended to at
modeling_llama.py:261-262. One spelling serves both paths.

- **RoPE** reads the table at the ABSOLUTE position, so token t gets the same
  `cos` and `sin` rows in a decode step as in a prefill. An implementation
  that indexes the table from the start of the current slice is the
  `S07_ROPE_RELATIVE_POSITION` sabotage, and it must break clause (d) and
  nothing else.
- **S11**, the QK product, contracts over `head_dim`, which is the same in
  both paths. That is precisely why S11 may be a `gemm.fp32.v1` call and S19
  may not, and the asymmetry is worth stating twice. GEMM v1's per-cell
  arithmetic is a pure function of `k` and the profile, so a contraction axis
  whose LENGTH is the same in both paths is safe and one whose length differs
  is not.
- **S17 and S19** contract over the key axis, whose length is `t + 1` in
  decode and `L` in prefill. By 7.1 the extra `L - t - 1` terms are exactly
  `+0.0` and bitwise inert in a serial ascending chain seeded `+0.0`.
  **Under the GEMM's leaf-and-tree topology they would NOT be inert**, because
  `P = f(k)` and `k` differs, so the trees differ. DEVIATION 807 is that
  sentence, and sabotage `S19_VALUE_SUM_VIA_GEMM` routes S19 through
  `identical_gemm` and must break clause (d) at any position past the first
  128 keys while leaving clause (a) green at a fixed length, which is exactly
  the failure a single-path gate cannot see.

Gate (d) then verifies what the construction promises, and the negative
control matters as much as the gate, in the same way it did for the mamba
lane's clause (c). A gate that compares a decode step against a prefill
token must also show two DIFFERENT positions differing, or a broken index
map makes it pass for ever.

### 7.3 Sequence-length invariance is the same theorem

A row's bits must be identical whether the sequence it belongs to has length
4 or 257. That is 7.1 again with the tail longer. It is stated as its own
half of clause (c) because it is a different fixture, and because a lane that
only ever runs one L cannot tell the two apart.

### 7.4 Batch composition invariance, which the serving world calls batch invariance

A row's bits must be identical whether its sequence shares the launch with
zero, one or two others. Nothing in sections 4 through 7 reads B, so this is
true by construction and the gate exists to catch the construction being
violated by an execution plan. `IDENTICAL_GEMM_PLAN.md:86-93` is the
in-repo statement of why this is the same problem, and
`check_pinned_gemm_is_batch_invariant` in `core/gemm_identity_check.mojo` is
the property one layer down.

**Honesty about novelty.** Unlike the Mamba block, batch invariance for
attention has been pursued in public. `IDENTICAL_SSM_NOTES.md:40-47` records
that Thinking Machines' *Defeating Nondeterminism in LLM Inference* covers
RMSNorm, matmul and attention, and that vLLM's `VLLM_BATCH_INVARIANT=1`
raises only for its Mamba-class backends. **So batch invariance is not this
lane's novelty claim.** What is not published, as far as this repository's
own prior-art notes go, is CROSS-VENDOR bitwise identity of the block, which
is a different property and the one the profile name is about. That reading
of vLLM was not re-verified by this lane and the entry it rests on is dated.

## 8. NaN, infinity, signed zero, denormals

- A NaN or infinity in any input or weight is REFUSED BY NAME before any
  recorded stage (`refuse_nonfinite`), because NaN payloads are
  vendor-shaped (IDENTITY_PATHS row 39 measured three payloads for one IEEE
  answer, `0x7fc00000` on Apple, `0x7fffffff` on NVIDIA, `0xffc00000` on
  AMD) and a certified stage may not contain one. The test is by BITS, not
  by compares, because Metal flushes compare operands (row 49).
- **A nonfinite INTERMEDIATE is a stated gap.** The refusal covers inputs.
  A score can overflow to `-inf` through S13 (section 4.1(b)) and, at
  extreme weights, `exp` can saturate. Those are deterministic and the same
  on every vendor, so they do not break identity, but a computed NaN would,
  and this profile does not check for one at every stage. What catches it is
  the card, since a stage hash containing a vendor-shaped payload cannot
  match. DEVIATION 815.
- `-0.0` is an admitted value everywhere. `pinned_mul`'s `-0.0` addend keeps
  a negative zero product. The S1, S17 and S19 folds seed `+0.0`, so an
  all-zero row sums to `+0.0` on every vendor by IEEE `(+0) + (-0) = +0`.
  Two sites need naming rather than assuming. The mask add at S13 LAUNDERS a
  `-0.0` score to `+0.0` and that is the reference's behavior, not an
  accident (4.1(a)). The row max at S14 is the one selection in this profile
  and it is pinned in 5.1, where the vendor split is real and measured.
- Denormals. `ftz` at every seam, flush to signed zero under IDENTICAL and
  compiled away under FAST, IDENTITY_PATHS row 10's policy. The portable
  transcendentals flush unconditionally. The trig pair's asymmetry is
  DEVIATION 820's and is settled there rather than here. `portable_sinf`
  flushes its INPUT and `portable_cosf` does not, because near zero sin
  returns its argument and a subnormal survives to the output, while cos
  returns `1.0` for every subnormal on every column and a flush would be
  bit-inert. Same policy, different reachability.
- `portable_sinf(-0.0)` is `-0.0`, which is IEEE, torch and numpy, and is a
  knowing departure from Cephes, whose `if (xx < 0)` never applies the sign
  to a negative zero. **It is inert at this profile's only consumer**,
  because RoPE's angle is `position * inv_freq` with `position >= 0`, and it
  is recorded here so that nobody plants a fixture expecting Cephes's
  answer.

## 9. The stages, in card order

One record per stage per block call, tags prefixed by the driver
(`core/identity_trace.mojo` rules; tags carry no machine property).
`M = B * L` token-major, `S` is the kv length after the append.

    input.x          [M, d_model]              the block input, as given
    norm1.sumsq      [M]                       S1
    norm1.out        [M, d_model]              S2-S4
    q_proj.out       [M, n_heads*head_dim]     S5
    k_proj.out       [M, n_kv*head_dim]        S5
    v_proj.out       [M, n_kv*head_dim]        S5
    rope.inv_freq    [head_dim/2]              S6, host, once
    rope.cos         [P_max, head_dim/2]       S7, S8
    rope.sin         [P_max, head_dim/2]       S7, S8
    q_rope.out       [M, n_heads*head_dim]     S9, S10
    k_rope.out       [M, n_kv*head_dim]        S9, S10
    kv.k_cache       [B, n_kv, S, head_dim]    copies
    kv.v_cache       [B, n_kv, S, head_dim]    copies
    attn.scores      [B, n_heads, L, S]        S11, S12
    attn.masked      [B, n_heads, L, S]        S13
    attn.max         [B, n_heads, L]           S14
    attn.exp         [B, n_heads, L, S]        S15, S16
    attn.denom       [B, n_heads, L]           S17
    attn.weights     [B, n_heads, L, S]        S18
    attn.ctx         [M, n_heads*head_dim]     S19
    o_proj.out       [M, d_model]              S5
    residual1.out    [M, d_model]              S22
    norm2.sumsq      [M]                       S1
    norm2.out        [M, d_model]              S2-S4
    gate_proj.out    [M, intermediate]         S5
    up_proj.out      [M, intermediate]         S5
    silu.out         [M, intermediate]         S20
    mlp.gated        [M, intermediate]         S21
    down_proj.out    [M, d_model]              S5
    residual2.out    [M, d_model]              S23

Thirty stages. `rope.inv_freq`, `rope.cos` and `rope.sin` are recorded once
per configuration rather than per call, and they are on the card because a
table computed with the wrong `pow` is a silent divergence that no
activation stage localizes.

## 10. What "identical" is gated to mean

(a) device card equals host oracle card, bitwise, at every stage and every
shape; (b) the same bits on every one of 8 repeated launches; (c) BATCH
COMPOSITION invariance, a row's bits identical whether its sequence shares
the launch with 0, 1 or 2 others, AND SEQUENCE LENGTH invariance, a row's
bits identical at L in {4, 16, 64, 257}, each half with its own negative
control; (d) decode equals prefill bitwise at every position, with a KV
cache; (e) the row-39 audit of section 8; (f) every clause above falsifiable
by a NAMED sabotage that fails a gate.

`transformer/original/transformer_check.mojo` will be the gate file. FAST
arms of (a) are RECORDED, not asserted, where they are vendor-shaped (the
metrics lane's leg-11 lesson).

The sabotage set clause (f) requires, one per contested decision.

| sabotage | first stage it must move | what it falsifies |
|---|---|---|
| `S1_FOLD_DESCENDING` | `norm1.sumsq` | the RMSNorm fold order |
| `S10_ROPE_FUSED` | `q_rope.out` | the two-rounding RoPE add |
| `S09_ROPE_HALVES_SWAPPED` | `q_rope.out` | the rotate-half pairing |
| `S07_ROPE_RELATIVE_POSITION` | nothing in prefill, clause (d) in decode | absolute position indexing |
| `S12_SCALE_INTO_Q` | `attn.scores` | scaling the finished dot, not q |
| `S13_MASK_NEG_INF` | `attn.masked`, at a planted extreme score | the mask VALUE |
| `S13_MASK_SELECT` | `attn.masked`, at a planted `-0.0` score | the mask being an ADD |
| `S14_MAX_PLAIN_COMPARE` | `attn.max`, at a planted `-0.0` beside a `+0.0`, and NOT on an ordinary row | the order-free maximum |
| `S17_DENOM_HALVING_TREE` | `attn.denom`, at kv length 3 or more | the serial denominator |
| `S18_RECIPROCAL_MUL` | `attn.weights` | the division |
| `S19_VALUE_SUM_VIA_GEMM` | clause (d) past 128 keys, clause (a) green | the value sum's topology |
| `S20_SILU_MUL_SIGMOID` | `silu.out` | SiLU's one-division spelling |
| `S05_OP_NUMBERING` | `q_proj.out` | the GEMM op numbering trap |

Each must move the stage its OWN seam writes and no earlier one, which is
the discipline the mamba lane's six arms were held to.

Two of these are gates on the GATES rather than on the code, and they are
the ones most likely to be skipped. `S07_ROPE_RELATIVE_POSITION` and
`S19_VALUE_SUM_VIA_GEMM` pass clause (a) by construction. If the lane has no
clause (d) when they are written, they will look inert and be deleted.

## 11. Not claimed

- **One block, not a model.** No embedding, no `lm_head`, no logits, no
  tokens, no sampling, no argmax, no multi-layer residual stream. The
  `IDENTICAL_GEMM_PLAN.md` sketch's item 7 (deterministic argmax first,
  seeded sampling later) is outside this contract entirely.
- **Not bit-identical AI inference**, in those words, which the GEMM
  charter forbids.
- **Not agreement with HuggingFace, PyTorch or MAX.** This profile's fold
  orders, transcendentals and division are OURS. The claim is that our
  arithmetic gives the same bits on three vendors, plus agreement with a
  float64 reference to a stated tolerance once a corpus exists. A reader who
  takes "identical" to mean "equal to torch" has taken more than is offered.
- **Not FlashAttention, not SDPA, not paged attention, not chunked
  prefill.** Section 6.
- **Not BF16, FP16, FP8, TF32 or any quantization.** Not FP64 anywhere on
  device; Metal does not have it.
- **No training, no backward, no dropout, no autograd.**
- **Not GELU**, not `tanh` and not `erf`. Llama's activation is SiLU. The
  numerics lane's DEVIATIONS 821-824 landed both GELU forms, `portable_tanhf`
  and `portable_erff` on 2026-08-24 for whoever pins a GPT-shaped or
  BERT-shaped block. They are not seams of this profile and this document
  does not specify them.
- **Not `rope_scaling`.** Only `rope_type = "default"`. Linear, dynamic,
  YARN and Llama-3 rope all carry an `attention_scaling` that is not 1.0 and
  inverse frequencies computed a different way.
- **Not any mask but the causal one.** No sliding window, no prefix mask, no
  arbitrary additive mask, no attention sink.
- **No bias on any projection**, no attention dropout.
- **No performance number.** None has been taken and none will be quoted
  until it is, alternated inside one thermal window.
- **Nothing cross-vendor until a leg runs.** Everything in this document is
  CONSTRUCTION and, as of writing, not even that. The GEMM lane's own history
  is the standing reason to say so: Apple and AMD agreed bit for bit through
  302 stages while NVIDIA diverged at `tree001.winners.scores`, so two
  backends agreeing closes nothing.
- **Nothing here has been compiled.** Every line number and every symbol in
  this document came from reading source on 2026-08-24. No file in
  `transformer/` other than this one and the README has any content.

## 12. Where this departs from the plan's sketch, and where the code goes

### 12.1 The sketch

`IDENTICAL_GEMM_PLAN.md:153-177`, "After GEMM: the minimum transformer
path", already sketched this lane in seven steps. Six are followed. Three
departures.

| sketch item | this contract |
|---|---|
| 1. RMSNorm with a pinned sum-of-squares TREE | **DEPARTS.** A serial ascending chain per row, one thread per row, no cross-thread fold. It is the mamba contract's S1, it gives the block one fold shape instead of two, and it keeps the norm out of any launch-geometry argument. `pinned_block_sum` is the tree and this lane does not use it anywhere. |
| 2. RoPE with portable sin/cos, OR a fully specified table | **FOLLOWED, both halves.** Portable sin and cos are used to BUILD a fully specified table, precomputed once, indexed by absolute position. Section 4 seams S6 through S8. `identical_sin` landed as DEVIATION 820 on 2026-08-24 and shares one reduction with `identical_cos`. MAX does the same thing (`rope.mojo` takes `freqs_cis` as a tensor and computes no angle in any kernel). |
| 3. Q/K/V GEMMs on the profile | **FOLLOWED**, and extended. The sketch treats attention's own matmuls as new work. `q . k^T` is a `gemm.fp32.v1` `OP_NT` cell and is reused unchanged (S11). |
| 4. Reference attention with a deterministic max reduction and the `-0.0` hazard named | **FOLLOWED entirely**, section 5, and the hazard is measured on three columns rather than anticipated. |
| 5. Output GEMM and residual addition | **FOLLOWED.** S5 and S22, S23. |
| 6. MLP with a portable SiLU or GELU | **FOLLOWED with SiLU.** GELU excluded, section 11. |
| 7. Deterministic argmax first, seeded sampling later | **DEPARTS. Out of scope.** There is no logit in one block. |
| (not in the sketch) the attention-weighted value sum | **DEPARTS from the obvious reading.** The sketch's "pinned dot products" would naturally route both attention matmuls through the GEMM. S19 does not, and section 7.2 is the argument. |

The sketch's own precondition, that this lane should open only after the
three-vendor GEMM run closes (`IDENTICAL_SSM_NOTES.md:48-51`), is satisfied.
IDENTITY_PATHS row 40 closed on three vendors at `144aa5b` on 2026-08-23.

### 12.2 Where the code will go

    transformer/IDENTICAL_TRANSFORMER_CONTRACT.md   this file
    transformer/README.md                           the lane's status
    transformer/original/transformer_fixture.mojo  config, weights, planted cases
    transformer/original/transformer_oracle.mojo   the NORMATIVE host oracle
    transformer/original/transformer_check.mojo    the gates and the sabotages
    transformer/derived/transformers/models/llama/modeling_llama.mojo
                                                    the device spelling
    transformer/corpus/                             the independent torch reference
    transformer/DERIVATION_MAP.tsv, NOT_IMPLEMENTED.tsv        upstream to ours

The `derived/` path mirrors the upstream path exactly, as `mamba/derived/`
does. PORTED IS OURS; the derivative-work language belongs in `NOTICE`.

### 12.3 The deviation numbers

This lane owns 800 through 819.

| # | what |
|---|---|
| 800 | the block composition `transformer.fp32.v1`, the IDENTITY_PATHS row |
| 801 | RMSNorm reused from the mamba lane with eps lifted to an ARGUMENT (`RMS_EPS` is hardcoded at `mamba/original/mamba_fixture.mojo:44`), a cross-lane edit the mamba lane must agree to |
| 802 | the attention scale pinned as a host `identical_rsqrt(Float32(head_dim))` constant |
| 803 | the additive causal mask, its value and the clause that the `+0.0` add may not be elided |
| 804 | the softmax row max through `identical_fmax` (DEVIATION 825), the fold shape left free, and the refusal of `pinned_block_max` |
| 805 | the softmax denominator as a serial ascending chain, and the refusal of `pinned_block_sum` |
| 806 | the softmax division through `identical_div`, and the reciprocal alternative refused |
| 807 | the attention-weighted value sum pinned serial ascending over the ABSOLUTE key index and NOT routed through gemm v1 |
| 808 | the QK product routed through gemm v1 `OP_NT`, per (batch, head) |
| 809 | RoPE inverse frequencies through `portable_powf` and the refusal of host libm `pow` |
| 810 | the rotary table precomputed once and indexed by absolute position, angle in FP32 |
| 811 | the rotate-half pairing and the unfused two-product-one-add |
| 812 | the absolute-position ceiling at 8192, `portable_cosf`'s Cody-Waite domain |
| 813 | GQA admitted with `repeat_kv` as a declared COPY |
| 814 | the KV cache layout and the one spelling that serves prefill and decode |
| 815 | the block-level `refuse_nonfinite` and the nonfinite-intermediate gap |
| 816 | this lane's `pinned_mul`, the fourth copy or the import |
| 817 | the stage list and its card tags |
| 818 | the sabotage set |
| 819 | reserved |
