# The IDENTICAL FP32 Mamba-1 block contract

# PROFILE `mojolearn.identical.mamba1.fp32.v1`

Written 2026-08-23, the mamba lane (DEVIATIONS 720-739). The shape of this
document is `gemm/IDENTICAL_FP32_CONTRACT.md`'s, on purpose. The code form
of every clause is `mamba/checks/mamba_oracle.mojo` (the host oracle)
and `mamba/impl/` (the device spelling), and the clauses cite both.

**THE PROFILE NAME IS PART OF THE CONTRACT.** Every card, gate and claim
under this document names `mojolearn.identical.mamba1.fp32.v1`. Changing
any seam decision in section 4, any constant in section 3, or the stage
list in section 7 creates a v2. It does not amend v1. FAST is unversioned
and makes no identity claim.

The completion claim this contract exists to support is one sentence. One
Mamba-1 block, FP32, is bit-identical across Apple, NVIDIA and AMD GPUs at
every stage, at every launch, at every batch composition, and between the
prefill and decode paths, under the declared profile. Not bit-identical
inference of a model, not Mamba-2, not training. The cross-vendor half of
that sentence is earned ONLY by the E-series leg (phase 8 lane `mamba`,
judge section 7) and is NOT earned by anything built on one machine.

---

## 1. The reference, pinned

| what | where | pin |
|---|---|---|
| the scan's math | `mamba_ssm/ops/selective_scan_interface.py::selective_scan_ref` lines 127-193 | state-spaces/mamba `e9594ce` |
| the block order | `src/transformers/models/mamba/modeling_mamba.py::MambaMixer.forward` (:359-483), `::MambaRMSNorm` (:485-503), `::MambaBlock.forward` (:505-530), and its torch fallbacks `causal_conv1d_fn` (:81-100), `mamba_selective_scan` (:175-280) | huggingface/transformers `d56c55b` |
| the decode step's semantics | `mamba_ssm/modules/mamba_simple.py::Mamba.step` (:208-253), `::allocate_inference_cache` (:255-266) | state-spaces/mamba `e9594ce` |
| the device kernel SHAPE for the scan | `max/kernels/src/state_space/selective_scan.mojo::selective_scan_fwd_gpu` (:74+), one thread per (batch, dim), serial over L, N=16 in registers; `causal_conv1d.mojo` (:24, :185-212) for the conv shape | modular/modular `10d978e` |
| the projections' arithmetic | profile `mojolearn.identical.gemm.fp32.v1`, IDENTITY_PATHS row 40, certified three-vendor at E3 round 11 (`a32e304`) | this repository |

Checkouts live in `/Users/andrewhendel/CascadeProjects/upstream/`. The
CUDA kernel `csrc/selective_scan/selective_scan_fwd_kernel.cuh` is cited
where its rounding order DIFFERS from the reference's, because a reader
porting from the kernel would otherwise assume they agree. They do not
(section 4, seams S7-S8 and S11).

## 2. What one block call is

    hidden = residual + mixer(rmsnorm(residual))          MambaBlock.forward

with mixer, in order (MambaMixer.forward, torch fallback path)

    in_proj -> split(hidden | gate) -> causal depthwise conv1d (d_conv=4,
    padding 3, truncated to L) -> SiLU -> x_proj -> split(dt | B | C) ->
    dt_proj (bias NOT applied here) -> delta = softplus(dt + bias) ->
    selective scan with A = -exp(A_log) -> + D * u -> * silu(gate) ->
    out_proj

Inference only. The decode step is the SAME arithmetic read one token at a
time with the conv window and the SSM state carried (section 5). Weights
carry no bias except conv1d and dt_proj (`use_bias` False, `use_conv_bias`
True, the MambaConfig defaults).

## 3. Profile constants

| constant | value | source |
|---|---|---|
| d_state | 16 | MambaConfig `state_size` default; also `MAX_DSTATE` in both device kernels |
| d_conv | 4 | MambaConfig `conv_kernel` default |
| expand | 2 | MambaConfig default; d_inner = 2 * d_model |
| dt_rank | ceil(d_model / 16) | MambaConfig `time_step_rank` |
| rms eps | 1e-5 (`0x3727C5AC`) | MambaConfig `layer_norm_epsilon` |
| softplus threshold | 20.0, compared `<=` | `selective_scan_fwd_kernel.cuh:160`, torch `F.softplus` threshold |
| dtype | Float32 everywhere (weights, activations, state) | Andrew's order |

Changing any of these is a v2. Shapes covered by the gates are B in
{1, 2, 3}, L in {1, 4, 16, 64, 257}, d_model in {8, 16}; the arithmetic
per (b, d, l) cell does not read B, L or the launch, so shape coverage is
about the gates, not the profile.

## 4. The seams, every one, with the fused-or-unfused decision

FMA contraction is PER SEAM. A seam marked FUSED is one rounding through
`identical_mul_add` (fma). A seam marked PRODUCT is one rounding through
`pinned_mul(a, b) = identical_mul_add(a, b, -0.0)`, the spelling no codegen
may contract into a neighboring add and which preserves a `-0.0` product
(a `+0.0` addend would launder it; gemm fixture F6a's lesson). Every seam's
RESULT passes `ftz`, and every operand LOADED from a buffer passes `ftz`
(row 10's checklist unit). Copies (the split of x_proj's columns, the conv
window update, the state store) are NOT seams; a copy moves bits untouched.

| # | seam | reference spelling | pinned spelling | fused? |
|---|---|---|---|---|
| S1 | RMSNorm sum of squares | `hidden.pow(2).mean(-1)` (MM:497), fold order torch's | serial ascending j from +0.0, `acc = ftz(fma(x_j, x_j, acc))`; one fold per row, no block fold | FUSED |
| S2 | RMSNorm mean and rstd | `mean` then `torch.rsqrt(var + eps)` | `mean = ftz(identical_div(sumsq, d_model))`; `rstd = ftz(identical_rsqrt(ftz(mean + eps)))`; identical_rsqrt is `1 / sqrt` as the reference spells it (`layer_norm.py:120`, DEVIATION 741), never the hardware rsqrt intrinsic | n/a |
| S3 | `hidden * rstd` (MM:498) | one product | `pinned_mul(x_j, rstd)` | PRODUCT |
| S4 | `weight * hidden` (MM:499) | one product | `pinned_mul(w_j, S3)` | PRODUCT |
| S5 | `delta * A` inside exp (ref:162) | einsum product | `pinned_mul(delta, A[d,n])` | PRODUCT |
| S6 | `exp(delta * A)` | `torch.exp` | `identical_exp` (row 12's polynomial) | n/a |
| S7 | `delta * B` (ref:167 einsum, first pairing) | torch pairs left to right; HF fallback spells it `discrete_B = dt * B` (MM:188) | `pinned_mul(delta, B[t,n])` | PRODUCT |
| S8 | `(delta * B) * u` (ref:167, second pairing) | HF MM:189 | `pinned_mul(S7, u)`. THE CUDA KERNEL ROUNDS THE OTHER WAY, `B * (delta * u)` (cuh:162,222; MAX's kernel too). The reference's order is the profile's; fixture F5 separates the two | PRODUCT |
| S9 | `h = deltaA * h + deltaB_u` (ref:175) | torch rounds twice (mul, add) | `ftz(fma(deltaA, h, deltaB_u))`, ONE rounding. Pinned fused because only fusion has a portable spelling (gemm contract section 4); the CUDA scan op and MAX's kernel contract here as well. Fixture F2 separates fused from unfused | FUSED |
| S10 | `y = sum_n h[n] * C[n]` (ref:177-182) | einsum, fold order torch internal; CUDA walks state_idx ascending (cuh:185-266) | serial ascending n from +0.0, `acc = ftz(fma(C[t,n], h[n], acc))` | FUSED |
| S11 | `out = y + u * D` (ref:189) | both the reference and the CUDA kernel round `u * D` as its own product (cuh:163 seeds out_vals with `D * u`) | `p = ftz(pinned_mul(u, D[d]))`, then `ftz(y + p)`. UNFUSED because both upstreams round the product. The CUDA kernel ADDS the C·h terms onto `D * u` (D first); the reference adds `u * D` to the finished y (D last). The REFERENCE's order is the profile's; fixture F6 separates them | PRODUCT + add |
| S12 | `out * silu(z)` (ref:190-191) | `F.silu` is `z / (1 + exp(-z))`, ONE division (ATen's spelling, also cuh:298) | `pinned_mul(skip, identical_silu(z))`; identical_silu is the single-quotient spelling (DEVIATION 744), NOT `z * sigmoid(z)` | PRODUCT |
| S13 | conv tap chain | `F.conv1d(padding=3, groups=d_inner)` + bias | bias-SEEDED accumulator, taps k = 0..3 ascending (oldest first), `acc = ftz(fma(w[d,k], x[l-3+k], acc))`, a pre-sequence position reads the window (zeros on prefill). The shape is MAX `causal_conv1d.mojo:190-205` and the CUDA `causal_conv1d` kernel (both bias-seeded); fixtures F3 (tap order) and F4 (bias seed vs bias last) separate the alternatives | FUSED |
| S14 | `delta = softplus(dt + bias)` (ref:145-148) | `dt + delta_bias`, then `F.softplus` | `biased = ftz(dt + ftz(bias[d]))`; `identical_softplus(biased)` = `x <= 20 ? log1p(exp(x)) : x` (DEVIATION 745) | add |
| S15 | `A = -exp(A_log)` (MM:373) | negate after exp | `-ftz(identical_exp(ftz(A_log)))`; negation is exact | n/a |
| S16 | residual add (MM:527) | one add | `ftz(ftz(x) + out_proj)` | add |
| S17 | the five matmuls | in_proj, x_proj, dt_proj, out_proj are `nn.Linear` (cuBLAS); there is NO fifth | all four are `gemm.fp32.v1` `OP_NT` cells (k = d_model, d_inner, dt_rank, d_inner, all <= 128, so P = 1, the serial ascending chain of gemm section 7.1). The GEMM refuses no shape; its entry accepts every (m, n, k) | per gemm v1 |

Inherited clause, stated so nobody rediscovers it. At `k = dt_rank = 1`
the GEMM leaf is `ftz(fma(a, b, +0.0))` (gemm section 7.1's +0.0 seed), so
a `-0.0`-valued dt product reaches `dt_proj.out` as `+0.0` (gemm section
9.2(a)). That is v1 GEMM behavior and this profile inherits it unchanged.

Softplus guard note, measured while building fixture F7. In FP32 the
`x <= 20` boundary itself cannot move a bit, since `log1p(exp(x)) - x <
ulp(x)/2` for every x above roughly 15. The guard's distinguishing range
is delta in about [8, 14], where the two arms differ in the last bits, and
that is where F7 plants. A sabotage that moves the threshold to 10 fails
on that fixture; a fixture that only straddles 20 passes it vacuously (the
corpus's `adv_softplus_guard` case is kept anyway, as the boundary's
regression record).

## 5. State, decode, and why prefill == decode is structural

The recurrent state between calls is the conv WINDOW (last d_conv inputs,
oldest first, zeros before the first token) and the SSM state h (zeros
before the first token). `allocate_inference_cache` (mamba_simple.py:
258-266) is the zeros; the window is `conv_state` after mamba_simple.py:
216-217's roll.

One spelling serves both paths. The conv kernel reads position `l-3+k`
from the sequence when it is nonnegative and from the window otherwise;
at L = 1 that IS the decode step's chain, and on prefill a zero window IS
the zero padding. The scan kernel takes h in a buffer and runs any L.
DEVIATION 721 records the one departure from `Mamba.step`'s spelling. The
step (:218-220) sums the taps first and adds the bias AFTER, while the
prefill kernels (MAX, CUDA causal_conv1d) seed with the bias. The profile
uses the bias SEED in both paths, because two spellings would make gate D
(decode == prefill, bitwise, per token) false by construction. Gate D then
verifies what the construction promises.

## 6. NaN, infinity, signed zero, denormals

- A NaN or infinity in any input or weight is REFUSED BY NAME before any
  recorded stage (`refuse_nonfinite`), because NaN payloads are
  vendor-shaped (row 39 measured three payloads for one IEEE answer) and a
  certified stage may not contain one. The test is by BITS, not compares
  (Metal flushes compare operands, row 49).
- `-0.0` is an admitted value everywhere. The seams preserve it where the
  reference does. `pinned_mul`'s `-0.0` addend keeps a negative zero
  product; the S10 and S1 folds seed `+0.0` (so an all-zero row sums to
  `+0.0` on every vendor, IEEE `(+0) + (-0) = +0`); the conv's bias seed
  makes the sign of a zero conv output a pure function of the input bits.
  No float max, min, argmax or argmin appears anywhere in this lane, so
  row 13's selection hazard has no site here.
- Denormals. `ftz` at every seam (flush to signed zero under IDENTICAL,
  compiled away under FAST), row 10's policy. The portable transcendentals
  flush unconditionally. A subnormal INPUT bit pattern reaches every stage
  through at least one arithmetic op, so the compare-flush asymmetry row
  49 measured (a pass-through subnormal keeps its bits on Metal) has no
  reachable site in the stage list.

## 7. The stages, in card order

One record per stage per block call, tags prefixed by the driver
(`core/identity_trace.mojo` rules; tags carry no machine property).

    input.x        [M, d_model]      the block input, as given
    norm.sumsq     [M]               S1, per row
    norm.out       [M, d_model]      S2-S4
    in_proj.out    [M, 2*d_inner]    S17; columns [0,di) hidden, [di,2di) gate
    A.out          [d_inner, 16]     S15
    conv.out       [M, d_inner]      S13, bias included
    silu.out       [M, d_inner]      identical_silu of conv.out (this is u)
    conv.window    [B, d_inner, 4]   the window AFTER the call (copies)
    x_proj.out     [M, dt_rank+32]   S17; columns dt | B | C
    dt_proj.out    [M, d_inner]      S17, bias NOT added (MM:429)
    softplus.out   [M, d_inner]      S14 (this is delta)
    scan.y         [M, d_inner]      S5-S10, before D
    scan.h         [B, d_inner, 16]  the state after the last token
    skip.out       [M, d_inner]      S11 (the corpus's "scan.y")
    gate.out       [M, d_inner]      S12
    out_proj.out   [M, d_model]      S17
    residual.out   [M, d_model]      S16 (the corpus's "block.out")

Token-major `M = B * L` layouts. The corpus's channel-major views of the
same values are reindexed by the corpus gate, not recomputed.

## 8. What "identical" is gated to mean

(a) device card == host oracle card, bitwise, at every stage and shape;
(b) the same bits on every one of 8 repeated launches; (c) BATCH
COMPOSITION invariance, a row's bits identical whether its sequence shares
the launch with 0, 1 or 2 others (the clause vLLM's batch-invariant mode
cannot give Mamba, `supports_batch_invariance()` is False for its Mamba
backends); (d) decode == prefill bitwise at every position; (e) the row-39
audit of section 6; (f) every clause above falsifiable by a named sabotage
that fails a gate. `mamba/checks/mamba_check.mojo` is the gate file;
FAST-mode arms of (a) are RECORDED, not asserted, where they are
vendor-shaped (the metrics lane's leg-11 lesson).

## 9. Not claimed

No chunked or tree scan (MAX's serial-over-L shape only; the cub
BlockScan tree is the priced follow-on and out of scope). No Mamba-2, no
SSD. No BF16 or FP16. No training, no backward. No multi-block model, no
tokenizer, no cache manager beyond one block's state. No performance
number (`tools/lanes_price.sh` carries the wiring; the number is blank
until measured). Nothing cross-vendor until the leg runs.
