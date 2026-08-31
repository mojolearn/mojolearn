# The IDENTICAL FP32 cross-entropy loss contract

# PROFILE `mojolearn.identical.loss.ce.fp32.v1`

Written 2026-08-25, the training lane, first piece. DEVIATIONS 1150-1169 are
this lane's and nothing here uses a number outside that range. The shape of
this document is `transformer/IDENTICAL_TRANSFORMER_CONTRACT.md`'s, which is
`mamba/IDENTICAL_MAMBA_CONTRACT.md`'s, which is
`gemm/IDENTICAL_FP32_CONTRACT.md`'s, on purpose.

The code form of every clause is `training/mojo_only/loss_oracle.mojo` (the
NORMATIVE host oracle) and `training/mojo_only/loss.mojo` (the device
spelling, one source for three vendors).

**STATUS BANNER, CORRECTED 2026-08-31.** This paragraph used to read
"Neither file has ever been compiled or executed. Nothing in this lane has
been built or run, no device has evaluated a single stage of it, and every
number in this document was derived on paper or read out of source on
2026-08-25." **Commit `ecd1a436` falsified it**: `loss.mojo`,
`loss_oracle.mojo`, `loss_check.mojo` and `loss_fixture.mojo` all ran there
for the first time, six clauses passed, 24 cases and 61,925 cells matched
device against oracle BITWISE, and clause (f) measured a real defect in the
device entry point that `b90f52ab` then fixed (DEVIATION 1495). Numbers in
this document that are still labelled PREDICTED and were not covered by a
clause remain paper arithmetic. **THE RUN WAS ON ONE DEVICE. Nothing in this
lane has run on a second vendor, so every cross-vendor sentence below is
still construction.**

**THE PROFILE NAME IS PART OF THE CONTRACT.** Every card, gate and claim
under this document names `mojolearn.identical.loss.ce.fp32.v1`. Changing any
seam decision in section 4, any fold topology in section 5, any constant in
section 3 or the stage list in section 9 creates a v2. It does not amend v1.
FAST is unversioned and makes no identity claim.

The completion claim this contract exists to support is one sentence. Softmax
cross-entropy over logits, and its gradient with respect to those logits, in
FP32, is bit-identical on Apple, NVIDIA and AMD GPUs at every stage, at every
launch, at every batch composition and at every row chunking, under the
declared profile. **Not an identical training step**, not an identical model,
not an identical optimizer, not agreement with PyTorch. The cross-vendor half
of that sentence is earned ONLY by a multi-vendor leg and is not earned by
anything built on one machine.

---

## 0. What this lane does NOT rebuild, and the one thing it does

Read in the tree on 2026-08-25 and cited by file and by symbol.

| piece | verdict | where it already is |
|---|---|---|
| `exp` | **REUSED** | `mojo_only/numerics.mojo::identical_exp` over `::portable_expf`, IDENTITY_PATHS row 12 |
| `log` | **REUSED** | `::identical_log` over `::portable_logf`. Mojo's own `std.math.log` carries about `5e-8` absolute error and re-decides plateau ties, which is the whole reason row 12 exists. This lane never spells `std.math.log` under IDENTICAL |
| division | **REUSED** | `::identical_div` over `::portable_divf`, DEVIATION 740, row 49 |
| the order-free maximum | **REUSED**, and it was built for exactly this reduction | `::identical_fmax` over `::portable_fmaxf`, DEVIATION 825. Its own docstring says it was written for softmax's row maximum and for no other seam |
| the flush | **REUSED** | `::ftz`, IDENTITY_PATHS row 10 |
| the uncontractible product | **REUSED, AND IT IS NOW IN `numerics.mojo`** | `::identical_mul`, DEVIATION 826, `identical_mul_add(a, b, -0.0)`. The transformer contract's section 12.3 item 816 records this as a `pinned_mul` copy the transformer lane still owed; the numerics lane landed it centrally on 2026-08-24. **This lane makes no fifth copy** |
| the fused multiply-add | **REUSED** | `::identical_mul_add`, row 9 |
| **every fold in this document** | **REUSED, ROUTED, NOT REBUILT** | profile `mojolearn.identical.gemm.fp32.v1`, entry `gemm/mojo_only/gemm_oracle.mojo::gemm_oracle` on the host and `gemm/mojo_only/gemm_identical.mojo::identical_gemm_into` on the device. Section 5 is the argument |
| the reduction-as-a-GEMM technique | **REUSED** | `gemm/mojo_only/gemm_backward.mojo::identical_gemm_backward_bias_into`, DEVIATION 851. That file proved `db[j] = sum_i dC[i, j]` is `ones . dC` under gemm v1 and therefore needs no second fold shape. This lane applies the same finding to three more reductions |
| the stage card and the differ | **REUSED** | `core/identity_trace.mojo` (`IdentityTrace.record_device` :313, `::record_host` :369), `tools/identity_trace_diff.py` |
| the refusal of a nonfinite input | **PATTERN REUSED, CODE COPIED** | `mamba/mojo_only/mamba_oracle.mojo::refuse_nonfinite` (:57) and, since 2026-08-25, `training/mojo_only/optimizer_oracle.mojo:162`. It is not in `numerics.mojo`; this lane makes a THIRD copy and owes the lift. DEVIATION 1164, section 13 |
| **the log-sum-exp, the negation, the label-smoothing combine and the gradient cell** | **NEW** | sections 4 and 6. That is the entire new arithmetic of this lane |

**Three refusals, each of a thing already in the tree that looks like the
right helper and is not.**

1. `core/pinned_reduce.mojo::pinned_block_sum` (:73) may not be the
   denominator. Section 5.3.
2. `core/pinned_reduce.mojo::pinned_block_max` (:159) may not be the row
   maximum. Section 5.1. Its own block comment (:145-155) says a caller whose
   inputs can carry `+-0.0` or NaN must state why first, and this caller
   cannot.
3. `max.gpu.primitives.block.sum` may not be any fold here, for
   `pinned_reduce.mojo`'s own reason -- its cross-lane stage folds at the
   HARDWARE lane width, 32 on Apple and NVIDIA and 64 on AMD's CDNA
   wavefront.

---

## 1. The reference, pinned

| what | where | pin |
|---|---|---|
| the causal-LM loss wrapper | `src/transformers/loss/loss_utils.py::ForCausalLMLoss` (:49-71) | huggingface/transformers `d56c55bf564ddb176759eb6ec199442682564916` |
| the reduction and the `num_items_in_batch` divide | `::fixed_cross_entropy` (:32-46) | same |
| the label shift | `ForCausalLMLoss` :61-64, pad with `ignore_index` then drop the first | same |
| the flatten to `[-1, vocab_size]` | `ForCausalLMLoss` :67-68 | same |
| the upcast to FP32 before the loss | `ForCausalLMLoss` :58-59, `logits.float()` | same |
| the loss itself | `torch.nn.functional.cross_entropy`, dispatched into ATen | **NOT READ. There is no PyTorch checkout in `/Users/andrewhendel/CascadeProjects/upstream/`** |
| the device kernel SHAPE for a row softmax | `max/kernels/src/nn/softmax.mojo::softmax_kernel` (:840-1015), one block per row, a max reduction then an exp-sum then a scale | modular/modular `10d978e3c783ef940d1d30d0a10852b69fe285c8` |
| the device SHAPE for log-softmax | `::softmax_kernel` with `logsoftmax=True` (:1004-1005), which computes `log(exp(x - m) * (1/sum))` | same |
| every fold | profile `mojolearn.identical.gemm.fp32.v1`, IDENTITY_PATHS row 40, certified three-vendor at E3 round 11 (`144aa5b`) | this repository |

**The absent PyTorch checkout is the single largest evidential gap in this
document and it is the same gap the transformer lane recorded at its section
5.4.** Three decisions below (the log-sum-exp spelling of 4.2, the label
smoothing combine of 6, and the reciprocal-versus-division question that
transformer 806 already answered for attention) are pinned on the
one-rounding argument and on internal consistency, NOT on having read ATen.
Every one of them is marked `UNVERIFIED AGAINST THE REFERENCE` where it
appears. When a PyTorch checkout lands, those three are the first things to
re-read, and a disagreement is a finding rather than an embarrassment.

**MAX's logsoftmax is evidence about MAX and not about the reference**, and
it is worth recording because it is the spelling a kernel author reaches for.
`softmax.mojo:1004` computes `log(exp(x - m) * recip(sum))` -- an exp, a
reciprocal, a product and a log, four roundings -- where this profile
computes `(x - m) - log(sum)`, two roundings and no exp on the target path at
all. They are different answers. Section 4.2.

---

## 2. What one loss call is

    logits   [N, V]   Float32, row-major, N = batch * tokens
    targets  [N]      Int32, each either a class in [0, V) or ignore_index
    ->
    row      [N]      Float32, the per-row loss
    loss     [1]      Float32, the reduced loss           (SUM and MEAN only)
    dlogits  [N, V]   Float32, dLoss/dlogits              (the backward)

and the arithmetic of one row `i` with target `y = targets[i]`, in order

    m        = fold of identical_fmax over logits[i, 0 .. V)          L1
    shift[v] = ftz(ftz(logits[i, v]) - ftz(m))                        L2
    e[v]     = identical_exp(shift[v])                                L3
    denom    = gemm v1 fold of e over v                               L4
    logdenom = identical_log(denom)                                   L5
    lp_y     = ftz(ftz(shift[y]) - ftz(logdenom))                     L6
    nll      = neg_by_bits(lp_y)                                      L7

with, when and only when label smoothing is on,

    lp[v]    = ftz(ftz(shift[v]) - ftz(logdenom))                     L8
    lpsum    = gemm v1 fold of lp over v                              L9
    smooth   = neg_by_bits(ftz(identical_div(lpsum, Float32(V))))     L10
    row      = ftz( ftz(identical_mul(ONE_MINUS_EPS, nll))
                  + ftz(identical_mul(EPS, smooth)) )                 L11

and, when it is off,

    row      = nll                                                    L11

then

    total    = gemm v1 fold of row over i                             L12
    loss     = identical_div(total, divisor)                          L13

and the backward

    w[v]     = ftz(identical_div(e[v], denom))                        L14
    t[v]     = T_TARGET if v == y else T_OTHER                        L15
    dl[v]    = ftz(identical_div(ftz(ftz(w[v]) - t[v]), divisor))     L16

An IGNORED row (`targets[i] == ignore_index`) writes `row = +0.0` and a
gradient row of `V` copies of `+0.0`, both STORED and neither skipped. It
does not contribute to `count`. Section 7.3.

Inference-shaped weighting is refused. **There is no per-class `weight`
vector in v1** (section 11), so `count` is an integer and never a sum of
floats, and that removes an entire fold from the document.

---

## 3. Profile constants

The FROZEN column says whether changing the value creates a v2 or merely
describes a different call.

| constant | value | frozen? | source |
|---|---|---|---|
| dtype | Float32 for logits, every intermediate, every accumulator, the loss and the gradient | **YES** | Andrew's order, and `ForCausalLMLoss` :59 upcasts to float for the same reason |
| target dtype | Int32 | **YES** | integers do not flush and do not round |
| `CONTRACT_K_LEAF_MIN` | `128` | **YES**, inherited | `gemm/mojo_only/gemm_oracle.mojo` |
| `CONTRACT_MAX_LEAVES` | `1024` | **YES**, inherited | same |
| `V`, the vocabulary | free per model, **FROZEN FOR THE RUN** | see 3.1 | `LlamaConfig.vocab_size`; Llama-3-8B is `128256` |
| `ignore_index` | free, default `-100` | no | `fixed_cross_entropy` :36 |
| `label_smoothing`, `EPS` | Float32 in `[0, 1)`, **and `EPS == 0` takes a different code path** | **YES**, in the sense that the eps-zero branch is a clause | `nn.functional.cross_entropy`'s kwarg. Not passed by `fixed_cross_entropy`; admitted here anyway |
| `ONE_MINUS_EPS` | `ftz(Float32(1.0) - EPS)`, computed ONCE on the host | **YES** | this document |
| `T_OTHER` | `ftz(identical_div(EPS, Float32(V)))`, host, once | **YES** | 6.3 |
| `T_TARGET` | `ftz(ONE_MINUS_EPS + T_OTHER)`, host, once | **YES** | 6.3 |
| reduction | `NONE`, `SUM` or `MEAN` | no | `fixed_cross_entropy` :39 |
| `divisor` | `Float32(count)` for MEAN, `Float32(num_items)` or `+1.0` for SUM | **YES**, its PRODUCER is frozen | 5.5 |
| `N`, the row count | free, a launch shape | no | 7.1 |
| per-class weights | **none, refused** | **YES** | section 11 |

Refusals by name rather than silent truncation. `V >= 1`; `N >= 1`;
`count >= 1` when the reduction is MEAN; `num_items >= 1` when it is
supplied; every target either equal to `ignore_index` or in `[0, V)`; no NaN
and no infinity in `logits`; `count <= 16777216` so `Float32(count)` is exact;
`N <= 4000000` so the batch fold's `k` stays inside the range the GEMM lane's
own sweep has exercised (`gemm_device_check.mojo::check_device_matches_oracle`
case `k4M.P1024.cap`).

### 3.1 Why `V` is FROZEN FOR THE RUN, and why that is the load-bearing fact

Under gemm v1 the leaf size `L` and the leaf count `P` are a pure function of
the contraction length `k` and the two profile constants -- contract section
6, and `contract_leaf_size`'s own docstring. So a fold whose length changes
between calls changes its tree and changes its bits.

**The vocabulary axis does not change between calls.** `V` is a property of
the tokenizer, fixed when the model is built, identical in every step of a
training run, identical in prefill and decode, identical at every batch size
and every sequence length. It is the same KIND of axis as the transformer
contract's `head_dim` and NOT the same kind as its key axis, and the
transformer contract's section 7.2 already drew that exact line --

> GEMM v1's per-cell arithmetic is a pure function of `k` and the profile, so
> a contraction axis whose LENGTH is the same in both paths is safe and one
> whose length differs is not.

-- which is why it routed the QK product (S11, `k = head_dim`) through gemm
v1 and refused to route the attention-weighted value sum (S19, `k = kv_len`)
through it. **The vocabulary fold is an S11, not an S19.** Section 5.3 turns
that into the departure this lane makes from transformer 5.3.

The price is stated rather than hidden. **A per-row vocabulary is REFUSED.**
A pruned, masked or ragged vocabulary -- a per-row candidate set, a
speculative-decoding restricted head, a hierarchical softmax -- has a
per-row `k`, therefore a per-row `P`, therefore per-row bits, and it is not
this profile. Section 11.

### 3.2 What the shipped vocabulary actually does to the tree, computed

`V = 128256`. `ceil(128256 / 128) = 1002 <= 1024`, so `L = 128` and
`P = 1002`, and `128 * 1002 = 128256` **exactly**.

| level `d` | width `N_d` | carry into `d+1`? |
|---|---|---|
| 0 | 1002 | no (even) |
| 1 | 501 | **YES** (odd) |
| 2 | 251 | **YES** (odd) |
| 3 | 126 | no |
| 4 | 63 | **YES** (odd) |
| 5 | 32 | no |
| 6 | 16 | no |
| 7 | 8 | no |
| 8 | 4 | no |
| 9 | 2 | no |
| 10 | 1 | -- |

Two consequences, and the first is a coverage hole that a lane which only
ever ran the shipped shape would never see.

- **The RAGGED LEAF NEVER OCCURS at the shipped vocabulary.** `128256` is an
  exact multiple of `128`, so every one of the 1002 leaves is full and the
  short-last-leaf path of `leaf_end` is never taken. The gates must therefore
  run a `V` that is not a multiple of 128. **`V = 300` is the required
  shape** -- it is `P = 3` with a ragged 44-element last leaf and one carry,
  which is the GEMM contract's own clause-5 fixture (its section 12.1) lifted
  into this lane on purpose.
- **The CARRY does occur, three times.** So the odd-tail path is exercised at
  the shipped shape and the `L_DENOM_PAD_PLUS_ZERO` sabotage of section 10
  has somewhere to land there as well as at `V = 300`.

**The gate shape.** `V` in `{1, 2, 3, 129, 256, 300, 1024, 128256}`, `N` in
`{1, 2, 3, 4, 5, 129, 300, 512}`, `ignore_index` present and absent, `EPS` in
`{0.0, 0.1}`, reduction in `{NONE, SUM, MEAN}`. `V = 1` and `N = 1` are the
`P == 1` cases where the tree performs NO addition (gemm 7.3) and where the
row loss is provably `-0.0` (section 8). `V = 3` and `N = 3` are the smallest
odd widths. `V = 129` is `P = 2` with a one-element ragged leaf, the most
extreme raggedness the leaf rule can produce.

---

## 4. The seams, every one, with the fused-or-unfused decision

FMA contraction is PER SEAM. A seam marked FUSED is one rounding through
`identical_mul_add`. A seam marked PRODUCT is one rounding through
`identical_mul` (DEVIATION 826), the spelling no codegen may contract into a
neighboring add and which preserves a `-0.0` product. Every seam's RESULT
passes `ftz` and every operand LOADED from a buffer passes `ftz`
(IDENTITY_PATHS row 10's checklist unit).

**Copies and integer work are NOT seams.** The label shift, the flatten to
`[N, V]`, the target load, the `v == y` test, and `count` are all exact.

| # | seam | reference spelling | pinned spelling | fused? |
|---|---|---|---|---|
| L1 | the row maximum | `log_softmax` subtracts the row max | `identical_fmax` folded over EVERY element of the row in ANY order, seeded `-inf`. Section 5.1 | n/a |
| L2 | `x - m` | subtraction | `ftz(ftz(x_v) - ftz(m))` | subtract |
| L3 | `exp(x - m)` | `torch.exp` inside log_softmax | `identical_exp`, row 12's polynomial | n/a |
| L4 | the denominator | a sum whose order is ATen's | **gemm v1 `OP_NN` at `(N, 1, V)` against a ones vector.** Section 5.3 | per gemm v1 |
| L5 | `log(denom)` | `torch.log` | `identical_log`, row 12's polynomial. Its argument is provably in `[1, V]`, section 8.2 | n/a |
| L6 | the target log-probability | `log_softmax(x)[y]` | `ftz(ftz(shift[y]) - ftz(logdenom))`. **NOT** `(m + logdenom) - x_y`, and **NOT** `log(w[y])`. Section 4.2 | subtract |
| L7 | the negation | unary minus | `neg_by_bits`, an XOR of the sign bit. Section 4.3 | n/a |
| L8 | every log-probability | `log_softmax(x)` | `ftz(ftz(shift[v]) - ftz(logdenom))`, the same expression as L6 at every `v`. Smoothing only | subtract |
| L9 | the smoothing fold | `log_probs.sum(class_dim)` | **gemm v1 `OP_NN` at `(N, 1, V)` against the same ones vector.** Smoothing only | per gemm v1 |
| L10 | the smoothing average | `smooth_loss / n_classes` | `neg_by_bits(ftz(identical_div(lpsum, Float32(V))))`, ONE division by `V`, never a host-folded `EPS/V` product. Section 6.2 | n/a |
| L11 | the smoothing combine | `(1 - eps) * nll + eps * smooth` | `ftz(ftz(identical_mul(ONE_MINUS_EPS, nll)) + ftz(identical_mul(EPS, smooth)))`. **UNFUSED**, two rounded products then one add. Section 6.2 | PRODUCT, twice, then an add |
| L12 | the batch reduction | `reduction="sum"` or `"mean"` | **gemm v1 `OP_NN` at `(1, 1, N)` against the ones vector.** Section 5.4 | per gemm v1 |
| L13 | the reduction divide | `loss / num_items_in_batch` (:45), or the mean's own divide | `identical_div(total, divisor)`, ONE division, never a reciprocal multiplied in. Section 5.5 | n/a |
| L14 | the softmax weight | `softmax(x)` | `ftz(identical_div(e[v], denom))`, ONE division per weight. This is transformer DEVIATION 806 at a second site and the two must agree | n/a |
| L15 | the target vector | `onehot(y)`, or the smoothed target | two HOST constants, `T_TARGET` and `T_OTHER`, selected by an integer compare. Section 6.3 | n/a |
| L16 | the gradient cell | `(softmax - onehot) * grad_output` | `ftz(identical_div(ftz(ftz(w[v]) - t[v]), divisor))`, ONE division per cell. Section 6.4 | subtract, then a divide |

### 4.1 Where this AGREES with the transformer contract, and where it DEPARTS

The brief for this lane was to make the same decisions the transformer's
section 5 made, or to number and justify every departure. Here is that
audit, complete.

| transformer clause | this lane |
|---|---|
| 5.1, the max is `identical_fmax`, fold shape FREE, `pinned_block_max` refused | **SAME**, no departure. Section 5.1 |
| 5.2, the max is taken over the WHOLE row | **SAME**. There is no mask here, so the clause is easier, and it is stated anyway because "the max over the unmasked prefix" is the mistake it exists to forbid |
| S15/S16, `exp(ftz(s) - ftz(m))` through `identical_exp` | **SAME**, seams L2 and L3 |
| 5.3, the denominator is a SERIAL ASCENDING CHAIN and `pinned_block_sum` is refused | **DEPARTS on the first half, AGREES on the second.** `pinned_block_sum` is refused here too and for the same reason. The serial chain is replaced by gemm v1's leaf-and-balanced-tree. **DEVIATION 1152**, and section 5.3 is the whole argument |
| 5.4, the division is `identical_div`, one rounding, reciprocal refused | **SAME**, seams L13, L14 and L16. Consistency with `attn.weights` was an explicit requirement of this lane's brief and it is met at all three sites |
| 5.5, one axis, one direction, one origin | **SAME IN SPIRIT, DIFFERENT IN FORM.** The vocabulary axis is walked in ASCENDING absolute class index, and the leaf-and-tree topology is a pure function of `V` alone. There is no per-launch local index to get wrong because there is no slicing of the vocabulary axis (section 3.1 refuses it) |
| S13, the mask is an ADD and the `+0.0` may not be elided | **NOT APPLICABLE.** There is no mask. The corresponding clause here is 7.3, the ignored ROW, and it has the same shape -- an ignored row's `+0.0` is STORED and provably inert in the fold, exactly as the transformer's masked tail is |
| section 8, refuse nonfinite BY BITS, admit `-0.0`, name every zero site | **SAME**, section 8 |

**The departure is one clause and it is DEVIATION 1152.** Everything else
agrees.

### 4.2 L6, the log-sum-exp, and the three spellings that are not it

`lp_y = ftz(ftz(shift[y]) - ftz(logdenom))`, where `shift[y]` is the value L2
already computed and stored. **UNVERIFIED AGAINST THE REFERENCE**, section 1.

Three alternatives, each of which a plausible implementation reaches for and
each of which is a different answer.

**(a) `(m + logdenom) - x_y`.** Algebraically the same. Numerically it is
not -- it re-adds the row maximum, which is the quantity the shift existed to
remove, so on a row whose logits are large the addition `m + logdenom` loses
`logdenom`'s low bits entirely and the subsequent subtraction of a nearby
`x_y` cancels catastrophically. The pinned spelling never forms a quantity
larger than `max(|shift[y]|, logdenom)`. Sabotage `L_NLL_VIA_ADDBACK`, and
its separating fixture is a row with a large common offset -- add `1e6` to
every logit of a row and the pinned answer does not move a bit while (a)
moves many.

**(b) `-log(w[y])`, the weight route.** One more transcendental and one more
division where a subtraction suffices, and `w[y]` can underflow to `+0.0` on
a confident-wrong row, whose log is `-inf`, whose negation is `+inf`. The
pinned spelling gives a large finite loss on that same row. Sabotage
`L_NLL_VIA_LOG_W`, whose fixture is a row where `shift[y]` is below about
`-87.34` so `e[y]` is exactly `+0.0` -- (b) returns `+inf` and the pinned
spelling returns `logdenom - shift[y]`, an ordinary finite number near 87.

**(c) MAX's `log(exp(shift) * recip(denom))`** (`softmax.mojo:1004`). Four
roundings against two, and it carries the reciprocal that transformer 5.4
already refused. Sabotage `L_NLL_VIA_MAX_LOGSOFTMAX`.

**What would make L6 pass while gating nothing.** A fixture whose rows are
small and centered -- logits in `[-1, 1]` -- makes (a) agree with the pinned
spelling to the last bit at most cells, because there is nothing for the
add-back to lose. The clause is only gated by the large-common-offset row,
and if that row is not in the fixture the sabotage reads as inert and
somebody deletes it. Named here so that does not happen.

### 4.3 L7 and L10, the negation, spelled BY BITS

    neg_by_bits(x) = bitcast(bits(x) XOR 0x80000000)

DEVIATION 1154. Three spellings and only this one is right at every input.

- `0.0 - x` is WRONG at `x = +0.0`, where it gives `+0.0` and IEEE negation
  gives `-0.0`. Section 8.1 shows `lp_y == +0.0` is reachable, so this is not
  a hypothetical. Sabotage `L_NEG_VIA_ZERO_SUB`, which must move `ce.nll` on
  a `V = 1` row and must NOT move it on an ordinary row.
- `identical_mul(x, -1.0)` is CORRECT at every input, including both zeros
  (`fma(+0, -1, -0) = (-0) + (-0) = -0`, and `fma(-0, -1, -0) = (+0) + (-0) =
  +0`). It is refused only because it presents a floating-point operation
  where an exact bit operation will do, and because an XOR cannot flush,
  cannot round and cannot be contracted.
- `-x` in source is the compiler's choice of the two above. It is very
  probably a sign-bit XOR on all three backends and this profile does not
  rely on "very probably".

The operand is already flushed by the seam that produced it, so the XOR needs
no flush of its own and adding one would be bitwise redundant. **Integer ops
do not flush on Metal** (`_ftz_always`'s docstring, DEVIATION 746(i)), which
is the second reason the bit spelling is the safe one.

---

## 5. The reduction orders

This is where cross-vendor identity is won or lost, so it gets its own
section rather than four rows of a table.

### 5.1 L1, the row maximum, is `identical_fmax` and its fold shape is FREE

    m(row) = the fold of identical_fmax over EVERY element of the row,
             in ANY order, seeded -inf (0xFF800000).

This is transformer clause 5.1 verbatim and DEVIATION 804/825, at a second
site. `identical_fmax` canonicalizes a NaN operand to `0x7FC00000` first,
flushes both operands, and then selects on `_total_order_key`, under which
`+0.0` keys at `0x80000000` and `-0.0` keys at `0x7FFFFFFF`. There is no
hardware `max` instruction and no float compare in it, so the result is
commutative and associative over all of Float32 including both zeros and NaN,
and **this profile therefore does not pin a fold topology for the maximum.**
That is the only place in this document where an execution plan may choose
its own tree, and it may do so because the operation is exactly associative
rather than because the difference is thought to be small.

Why an unpinned float max would be a defect here. IDENTITY_PATHS row 39
MEASURED `max(+0.0, -0.0)` as `-0.0` on Apple (the second operand) and `+0.0`
on NVIDIA and AMD. A row of 128256 logits reaches both zero signs easily -- a
flushed subnormal one way and an exact zero the other -- so this is a live
three-vendor split at the largest reduction in the profile.

**The seed is `-inf` and not `+0.0` and not `-FLT_MAX`.** DEVIATION 1151.

- `+0.0` is WRONG. A row whose logits are all negative -- which is most rows
  of a trained language model's head after the first steps -- would have its
  maximum clamped to `+0.0`, every shift would be too negative, and the loss
  would be quietly wrong on every vendor identically. Bit-identity does not
  catch it. Sabotage `L_MAX_SEED_ZERO`, and it must move `ce.max` on an
  all-negative row and must NOT move it on a row containing a positive logit.
- `-FLT_MAX` (`Scalar[Float32].MIN`, which is what MAX's own
  `softmax_kernel:958` seeds with) is CORRECT under this profile, because
  section 8 refuses an infinite logit, so no real element can key below it.
  It is refused anyway because it is correct only by virtue of a refusal made
  somewhere else, and because MAX is inconsistent with itself about it --
  `_softmax_warp_kernel:1063` seeds with `min_or_neg_inf` in the same file.
  Under `identical_fmax` the total-order key of `-inf` is `0x007FFFFF`, which
  is strictly below the key of every finite value, so `-inf` is a true
  identity element and needs no refusal to hold it up.

**`pinned_block_max` may NOT be used.** Its fold is a plain `other > red[tid]`
compare (`core/pinned_reduce.mojo:159-190`), which is precisely the spelling
IDENTITY_PATHS row 13 closed everywhere else, and its own block comment says
a caller whose inputs can carry `+-0.0` must state why before using it. This
caller cannot. **The clean fix is to give that fold `identical_fmax` as its
combine step**, which is `portable_fmaxf`'s own suggestion and is that file's
owner's call, not this lane's. Until then `training/mojo_only/loss.mojo`
carries a local `pinned_block_fmax` and DEVIATION 1165 is the debt.

**What would make L1 pass while gating nothing.** A fixture with no `-0.0`
anywhere in a row makes the plain-compare sabotage bit-inert -- a plain
compare and a total-order selection agree at every other input. So
`L_MAX_PLAIN_COMPARE` is only a gate on a row that PLANTS a `-0.0` beside a
`+0.0`, and the check must ALSO fold an ordinary row and show the sabotage
inert there. A one-armed version of this gate is evidence of nothing.

### 5.2 What the maximum is taken OVER

Every element of the row, all `V` of them, including the target's own logit.
Not "the top-k", not "a sampled subset", not "the unmasked prefix". Stated
because a fast implementation that maintains a running maximum over a
candidate set is a different reduction over a different multiset.

### 5.3 L4, the denominator, is gemm v1's LEAF-AND-BALANCED-TREE, and this is DEVIATION 1152

    denom(row) = identical_gemm( E[N x V], ones[V x 1], OP_NN, N, 1, V )

with `ones` holding exactly `Float32(1.0)` in every entry. On the host it is
`gemm_oracle(row_exps, ones, OP_NN, 1, 1, V)`.

**There is no new arithmetic here and no new fold to certify.** The leaf loop
of gemm v1 computes `acc = ftz(fma(ftz(e_p), ftz(1.0), acc))`, and
`fma(e, 1, acc)` is ONE rounding of `e + acc` because `e * 1.0` is exact. So
the ones vector turns the reduction into the contract's own ascending flushed
chain inside a leaf and the contract's own balanced tree across leaves. That
sentence is not this lane's invention -- it is
`gemm/mojo_only/gemm_backward.mojo`'s module docstring, DEVIATION 851, where
the bias gradient's row sum was routed the same way for the same reason.

**Why this DEPARTS from transformer 5.3, which pinned a serial chain.**
The transformer's argument for the serial chain has two parts and the second
is the load-bearing one --

> It is what makes decode equal prefill and makes the answer independent of
> sequence length. Under the GEMM's topology, `P` is a pure function of the
> contraction length `k`, so a row folded over 257 keys and the same row
> folded over 5 keys have different trees and different bits.

**That hazard does not exist on the vocabulary axis, because `V` never
changes** (section 3.1). There is no decode path over the vocabulary, no
variable-length tail, no per-launch slice. So the reason the transformer paid
for a serial chain is absent here, and what is left is the price.

The price of a serial chain at `V = 128256`, stated in three numbers.

1. **A 128256-term dependent FP32 chain per row.** Every step depends on the
   previous one, so a row's denominator cannot be split across threads at
   all, and the whole reduction is one thread deep by 128256. The balanced
   tree is 1002 independent leaves of 128 and then 10 levels.
2. **Conditioning.** The terms are `exp(shift)` in `[0, 1]` with exactly one
   of them equal to `1.0`. A serial ascending chain in vocabulary-index order
   adds 128256 small terms to a growing accumulator in an order that has
   nothing to do with their magnitudes. The GEMM contract's own section 7.2.1
   makes exactly this argument for exactly this reason, and its cap
   justification (`CONTRACT_MAX_LEAVES`'s docstring) says in as many words
   that 31,250 partials folding in 15 levels is BETTER conditioned than 1,024
   leaves of 3,907, not worse.
3. **One fold shape for the whole lane.** L4, L9 and L12 are all the same
   call to the same certified entry point with a different `k`. A serial
   chain here would have introduced a second fold shape that L12 could not
   share, because L12's `k` is `N`, a launch quantity, and that is where the
   tree's alignment property (5.4) actually matters.

The consequences are consequences and they are stated, not buried.

- **A per-row vocabulary is refused** (3.1). This is the cost of the
  departure and it is the only one.
- **`E`, the `[N, V]` buffer of exponentials, is MATERIALIZED.** At
  `N = 4096` and `V = 128256` that is `2.1 GB` per buffer, and this profile
  wants three or four of them (`ce.shift`, `ce.exp`, `ce.weights`, and
  `ce.logp` when smoothing is on). Section 7.1 is the escape and it is a real
  one -- the per-row arithmetic reads nothing about `N`, so chunking over
  rows cannot move a bit and a caller may run 256 rows at a time.
- **This is a reference-quality profile and it is slow by construction**, the
  same sentence the transformer contract wrote about its own materialized
  score buffer, and for the same reason -- a lane whose only instrument is
  the per-stage card should not begin by fusing the stages away.

**ONE ones vector serves all three folds and its OPERAND SIDE differs between
them.** The vocabulary folds put it on the RIGHT (`OP_NN` at `(N, 1, V)`, so
it is the `k x n` operand) and the batch fold puts it on the LEFT (`OP_NN` at
`(1, 1, N)`, the `m x k` operand, which is
`identical_gemm_backward_bias_into`'s own shape). **Operand side is part of
the routing and not a convention that can be assumed** -- that is
`gemm_backward.mojo`'s `BWD_DC_LEFT` / `BWD_DC_RIGHT` lesson and
`SAB_BWD_OPERAND_ORDER` is what it grew from. The caller allocates
`max(V, N)` entries once per shape.

**`pinned_block_sum` is REFUSED here as firmly as the transformer refused
it**, and the refusal costs nothing now because nothing in this lane folds by
hand. It pairs by STRIDE (`red[t] += red[t + step]`), which the GEMM contract
7.2 clause 1 names as a DIFFERENT ANSWER from v1's adjacent pairing, and it
folds only within one block. Sabotage `L_DENOM_HALVING_TREE`.

**What would make L4 pass while gating nothing.** At `V <= 128` the leaf
count is 1 and the tree performs NO addition (gemm 7.3), so
`L_DENOM_SERIAL_CHAIN` and `L_DENOM_HALVING_TREE` are both bit-inert. A gate
whose largest `V` is 128 would report both sabotages as failures-to-fire and
somebody would conclude the pins are vacuous. **The `V >= 129` shapes are
mandatory**, and the check must PREDICT the inert set at `V <= 128` and
assert it as a mask, the way `IDENTICAL_BACKWARD_PLAN.md` section 5.0 does
for its four-of-six routes.

### 5.4 L12, the batch reduction, is the SAME call at `k = N`

    total = identical_gemm( ones[1 x N], row[N x 1], OP_NN, 1, 1, N )

The `SUM` and the `MEAN` are the same fold; only L13's divisor differs.

**`N` IS A LAUNCH SHAPE, and unlike `V` it moves.** So this fold has exactly
the property the transformer refused to accept on its key axis, and this lane
accepts it knowingly because there is no alternative -- a batch reduction's
length IS the batch. It is the same finding
`gemm/mojo_only/gemm_backward.mojo::gemm_backward_b_call` recorded about the
weight gradient, in the same words, and the consequence is the same one --
**the microbatch schedule is part of a training run's numerical
specification, not an execution detail.**

`IDENTICAL_BACKWARD_PLAN.md` section 5.2 sharpens that, and its sharpening
applies here unchanged. Under v1's leaf rule `L` is held at 128 for every `k`
up to 131,072 and the fold is a balanced tree over ADJACENT leaves, so a
split at a token boundary that is both a LEAF boundary and a SUBTREE boundary
reproduces the unsplit tree exactly, provided the accumulation across
microbatches is spelled as the fold's own flushed add. **That is a PREDICTION
made by another lane and it has not been measured there or here.** This
document repeats it as a prediction and does not build on it.

**What would make L12 pass while gating nothing.** Every `N <= 128` is
`P == 1`, no addition, all fold sabotages inert. `N >= 129` is mandatory, and
the same predicted-mask discipline applies.

### 5.5 L13, the divide, and the ONE producer of the divisor

`loss = identical_div(total, divisor)`, one division, never a reciprocal
multiplied in. `e * (1/d)` rounds twice where `e / d` rounds once and they
differ in the last bit on ordinary inputs. This is transformer 5.4's
paragraph and this lane makes the same call at the same evidential standing
-- **UNVERIFIED AGAINST THE REFERENCE.** Sabotage `L_MEAN_RECIPROCAL_MUL`.

`divisor` is

| reduction | divisor |
|---|---|
| `NONE` | not formed; there is no `ce.loss` stage |
| `SUM`, no `num_items_in_batch` | `Float32(1.0)`, and the division IS SPELLED |
| `SUM` with `num_items_in_batch` | `Float32(num_items)`, `fixed_cross_entropy` :45 |
| `MEAN` | `Float32(count)`, `count` being the number of rows whose target is not `ignore_index` |

The `SUM` divide by exactly `1.0` is bitwise inert at every value the
accumulator can hold (`portable_divf(x, 1.0)` returns `x` for every finite
`x` and both zero signs). It is spelled anyway so there is ONE code path and
no branch whose two arms have to be shown to agree.

**`count` IS AN INTEGER.** It is counted with integer arithmetic, so it is
exact, order-free and vendor-free, and it is the reason section 11 refuses
per-class weights -- a weighted mean's denominator is a SUM OF FLOATS and
would need a fold, a clause, a fixture and a sabotage of its own.
`Float32(count)` is exact for `count <= 16777216`, which is the refusal in
section 3.

**`count == 0` is REFUSED BY NAME.** Torch returns NaN there
(`0.0 / 0.0`). A computed NaN's payload is vendor-shaped -- IDENTITY_PATHS
row 39 measured `0x7fc00000` on Apple, `0x7fffffff` on NVIDIA and
`0xffc00000` on AMD for one IEEE answer -- so a certified stage may not
contain one. This is a KNOWING DEPARTURE from the reference and it is in
section 11. DEVIATION 1156.

**The divisor has exactly ONE PRODUCER**, `loss_oracle.mojo::ce_divisor`, a
pure host function of `(reduction, count, num_items)` that reads no device,
no buffer and no launch. The forward's L13 and the backward's L16 both call
it. That is `gemm_backward.mojo`'s "the backward shape has exactly two
producers" discipline applied to the one quantity in this lane that two
different passes have to agree about. Sabotage `L_GRAD_DIVISOR_IS_N`, which
substitutes `N` for `count` in the backward only -- **it is bit-inert when no
row is ignored**, and the check must predict that and assert the inertness,
or the sabotage will look broken on a fixture with no ignored rows.

---

## 6. Label smoothing, the second fold over the vocabulary

**UNVERIFIED AGAINST THE REFERENCE**, section 1. `nn.functional.cross_entropy`
takes `label_smoothing` and dispatches into ATen, which could not be read.
What follows is the spelling this profile pins and the reason, not a
transcription of ATen.

### 6.1 The definition

    row_i = (1 - eps) * nll_i  +  eps * ( -(sum over v of lp[i, v]) / V )

with `lp[i, v]` the log-probability of L8. The second term is the
cross-entropy against the uniform distribution, which is why it is a MEAN
over the vocabulary rather than a sum.

### 6.2 The spelling, and the two things it is not

    lpsum  = gemm v1 OP_NN (N, 1, V) against ones                     L9
    smooth = neg_by_bits( ftz( identical_div( lpsum, Float32(V) ) ) ) L10
    row    = ftz( ftz(identical_mul(ONE_MINUS_EPS, nll))
                + ftz(identical_mul(EPS, smooth)) )                   L11

**(a) The division by `V` comes FIRST and is a real division.** The
alternative is a host-precomputed `EPS_OVER_V = eps / V` folded into one
product, which saves a division per row and is a different answer -- it
rounds `eps/V` once on the host and then rounds one product, where the pinned
spelling rounds one quotient and then one product, and the two quotients are
not the same number. Sabotage `L_SMOOTH_FOLDED_CONSTANT`. `Float32(V)` is
exact for every `V` under the section-3 ceiling.

**(b) The combine is UNFUSED.** Two rounded products, then one add of two
already-rounded values. An `identical_mul_add(EPS, smooth, product1)` is one
rounding where this is three, and it is the natural thing for a kernel to
write. This is transformer S10's clause at a different seam and it has the
same sabotage shape, `L_SMOOTH_FUSED_COMBINE`.

**(c) `EPS == 0` TAKES A DIFFERENT PATH.** DEVIATION 1155, and it is the
subtle one. At `eps = 0` the smoothing arm is ALMOST bit-inert --
`identical_mul(0.0, smooth)` is `fma(+0, smooth, -0)`, which is `+0.0` for
positive `smooth`, and `ftz(nll + (+0.0))` is `nll` for every value except
`nll = -0.0`, where it gives `+0.0`. **Section 8.1 shows `nll == -0.0` is
reachable.** So spelling the smoothing arm unconditionally would make
"turning label smoothing off" move a bit on exactly one row shape, which is
the worst kind of defect -- invisible in every ordinary fixture and real.

    if EPS == 0:   row = nll,  and L9, L10 and L11 are NOT SPELLED
    else:          row = L11

The `EPS == 0` test is on a Float32 constant known on the host before any
launch, so it is a configuration branch and not a data-dependent one.
Sabotage `L_SMOOTH_ALWAYS_SPELLED`, which must move `ce.row` on a row whose
`nll` is `-0.0` and must NOT move it on any other row. **That is the strongest
reach-per-branch sabotage in this document** and it is the one most likely to
be deleted as inert by somebody whose fixture has no `V = 1` row.

**What would make section 6 pass while gating nothing.** All three sabotages
are inert at `EPS == 0` except `L_SMOOTH_ALWAYS_SPELLED`, and
`L_SMOOTH_ALWAYS_SPELLED` is inert at every `EPS != 0` and at every row whose
`nll` is not `-0.0`. The gate must therefore run BOTH eps settings and must
carry the `V = 1` row, and it must assert the predicted inert masks rather
than reporting "the sabotage did not fire".

### 6.3 The smoothed target vector, L15, is TWO HOST CONSTANTS

    T_OTHER  = ftz( identical_div( EPS, Float32(V) ) )
    T_TARGET = ftz( ONE_MINUS_EPS + T_OTHER )

computed once on the host and passed to the kernel as two floats.

This is the analytic derivative of 6.1 with respect to `x_v`, worked out --
`d/dx_v [ (1-e) * nll + e * smooth ] = w_v - (1-e) * onehot_v - e/V` -- and
it is spelled as a target VECTOR rather than as three terms so that the
gradient kernel performs one subtraction and one division per cell and
nothing else.

**At `EPS == 0` these are exactly `1.0` and `+0.0`**, so unlike the forward,
the BACKWARD's smoothing arm is bitwise inert at eps zero and needs no branch.
That asymmetry is named because it is a `[[reached-but-inert]]` hazard in the
other direction -- a sabotage that deletes a branch the gradient does not
have will move nothing, and that is not evidence about the gradient.

**A subtlety that is a real decision.** `T_TARGET` and `T_OTHER` are the
derivative of the MATHEMATICAL loss, not the derivative of the
FLOATING-POINT expression L11 computes. Nobody computes the latter, and this
profile does not pretend to. Section 11.

---

## 7. Row independence, batch composition, and the ignored row

### 7.1 Nothing per-row reads `N`

L1 through L11 and L14 through L16 read `logits[i, :]`, `targets[i]`, `V`,
and the profile constants. **Not `N`, not the row chunk, not the launch
geometry, not the block size, not the vendor.** Therefore

> a row's bits are identical whether it is computed alone, in a chunk of 256,
> or in a launch of 4,096,

which is this lane's batch-composition clause and it is true BY
CONSTRUCTION. The gate exists to catch the construction being violated by an
execution plan, not to establish it. `IDENTICAL_GEMM_PLAN.md:86-93` is the
in-repo statement of why this is the same problem one layer down, and
`core/gemm_identity_check.mojo::check_pinned_gemm_is_batch_invariant` is that
layer's own gate.

**The exception is L12, and only L12.** The batch fold's `k` is `N`. Section
5.4 is the honest treatment.

**The consequence is the escape hatch for section 5.3's memory cost.** A
caller may split `[N, V]` into row chunks of any size, run L1-L11 and L14-L16
per chunk, and concatenate, and the bits do not move. Only the L12 fold has
to see all `N` rows at once, and it folds `[N]` floats, not `[N, V]`.

### 7.2 Vocabulary-fold invariance

`L` and `P` for L4 and L9 come from `V` and the two profile constants and
from nothing else -- `contract_leaf_size(V)`, one producer, gemm contract
section 6. No block size, no occupancy, no vendor, no `N`. Sabotage
`L_VOCAB_FOLD_READS_LAUNCH`, which reaches gemm v1's own
`SAB_LEAF_READS_LAUNCH` through this lane's routed call and is the proof that
the routing lands on the contract's arithmetic rather than on some other path
that happens to agree.

### 7.3 The ignored row is `+0.0`, STORED, and provably inert

For a row with `targets[i] == ignore_index`

    ce.row[i]        = +0.0, STORED
    ce.dlogits[i, v] = +0.0 for every v, STORED
    count            not incremented

and neither store may be skipped. This is the gemm contract's section 8
discipline (`identical_gemm_backward_a_into`'s degenerate-shape note --
"section 8 requires the value to be WRITTEN, and it is") applied to a row
instead of to a shape.

**Why `+0.0` and not `-0.0`, and why it matters.** L12's leaf accumulator is
seeded `+0.0` and every term it adds is a row loss. `fma(x, 1.0, acc)` is
`acc + x`, and `acc + (+0.0) == acc` for every `acc` except `acc = -0.0`,
which the `+0.0` seed does NOT forbid.

**DEVIATION 1327, 2026-08-25.** The seed forbids reaching `-0.0` by ADDITION.
It does not forbid `ftz` of a negative subnormal partial sum, and `ftz` runs
at every seam. Verified by direct computation: from a `+0.0` seed,
`+ 0x80C00000` then `+ 0x00800000` gives `0x80400000`, a negative subnormal,
and `ftz` of that is `0x80000000`. Both operands are ordinary normals. The
same false sentence stood in `transformer/IDENTICAL_TRANSFORMER_CONTRACT.md`
7.1 and is corrected there.

Whether the hole is REACHABLE for `ce.row` is a separate question this lane
has not answered: it needs a row whose partial log-prob sum cancels into the
subnormal range. The clause below is therefore true of every fixture shipped
today and is NOT proven of the contract. So an ignored row is BITWISE INERT in the
batch fold. **It is not inert in `P`** -- the fold still has `N` terms and
`P = f(N)` -- and it is not inert on the card, because `ce.row` records it.
That is the transformer's 7.1 theorem with a row where it had a masked key,
and it is what makes the loss with ignored rows equal to the loss of the same
rows with the ignored ones present.

A `-0.0` ignored row would be invisible in `ce.total` (laundered by the seed)
and visible in `ce.row`. Sabotage `L_IGNORED_ROW_NEG_ZERO`, which **must move
`ce.row` and must NOT move `ce.total`.** A gate that only compared the final
loss would call it inert, and it is not -- it is a divergence at the stage
that produced it. This is transformer 5.1's "the card is the only instrument
that can see this clause at all" at a second site.

---

## 8. NaN, infinity, signed zero, denormals

- **A NaN or an infinity in `logits` is REFUSED BY NAME before any recorded
  stage**, because NaN payloads are vendor-shaped (row 39, three payloads for
  one IEEE answer) and a certified stage may not contain one. **The test is
  BY BITS and never by compares**, because Metal flushes compare operands
  (row 49). The test is `(bits & 0x7FFFFFFF) >= 0x7F800000`.
- **A target is REFUSED unless it equals `ignore_index` or lies in
  `[0, V)`.** By INTEGER compare. Integers do not flush.
- **`count == 0` under MEAN is REFUSED**, section 5.5.
- **`num_items <= 0` is REFUSED.**
- `-0.0` is an admitted value everywhere. Every named site is in 8.1.
- Denormals. `ftz` at every seam named in section 4, flush to signed zero
  under IDENTICAL and compiled away under FAST, row 10's policy. The portable
  transcendentals flush unconditionally, so `identical_exp`, `identical_log`
  and `identical_div` carry the policy regardless of the build mode.

### 8.1 Every site a signed zero is reachable, named

| site | how | what depends on it |
|---|---|---|
| `ce.max` | the row's maximum is a flushed subnormal, or an exact `+-0.0` | nothing downstream. `s - (+0.0)` and `s - (-0.0)` agree for every finite `s` except `s = -0.0`, and `exp` of either is exactly `1.0`. **Laundered downstream, visible on the card**, transformer 5.1's note |
| `ce.shift` at the argmax | `ftz(x_argmax) - m` is exactly `+0.0` | the proof in 8.2 that `denom >= 1.0` |
| `ce.exp` | `identical_exp(shift)` is exactly `+0.0` for every `shift < -87.33655` (`portable_expf`'s own underflow edge) | the exact-fixture family of section 12 |
| `ce.logp_target` | `+0.0` when the target IS the argmax and `denom` is exactly `1.0` | `ce.nll` |
| `ce.nll` | `neg_by_bits(+0.0)` is `-0.0`. **Reachable at `V = 1`, and at any `V` where every non-target exponential underflows to `+0.0`** | the eps-zero branch of 6.2(c) and the `L_NEG_VIA_ZERO_SUB` sabotage |
| `ce.row` for an ignored row | `+0.0` by clause 7.3, never `-0.0` | the inertness proof of 7.3 |
| `ce.total` | `+0.0` when every row is ignored or every row loss is `+-0.0`. **The `+0.0` leaf seed does NOT forbid `-0.0`** -- see DEVIATION 1327 below | -- |
| `ce.weights` | `identical_div(+0.0, denom)` is `+0.0` for `denom > 0` | -- |
| `ce.dlogits` | `ftz(w - t)` is `+0.0` when `w == t` exactly, by IEEE's `x - x = +0` in round-to-nearest | -- |

### 8.2 Two theorems that remove a whole class of hazards

**(a) The shift is provably non-positive, so `exp` cannot overflow.** `m` is
the total-order maximum of the flushed row, so `ftz(x_v)` never keys above
`m`, so `ftz(x_v) - m <= +0.0` for every `v`. Therefore `e[v]` is in
`[+0.0, 1.0]` and `identical_exp` never takes its `x > 88.722835` overflow
branch on any row. **No forward stage can be `+inf` through the exponential.**

**(b) The denominator is provably in `[1.0, Float32(V)]`, so `log` never sees
a pathological argument.** The argmax contributes `identical_exp(+0.0)`,
which is exactly `1.0` (the polynomial evaluates to `y = 0.0`, then
`y + 1.0 = 1.0`, then two multiplies by `2^0`), and every other term is
non-negative, and the leaf chain seeded `+0.0` cannot lose it. The upper
bound is `V` terms each at most `1.0`, so at `V = 128256` the denominator is
below `2^17` and cannot overflow. **`identical_log`'s argument is therefore
never zero, never negative, never subnormal and never infinite**, and its
result is in `[+0.0, 11.7621]`. Every one of `portable_logf`'s special-value
branches is unreachable from this profile, which is worth knowing and is NOT
an argument for deleting them.

**(c) The row loss is provably non-negative, so the batch fold cannot
cancel.** `nll = logdenom - shift[y]` with `logdenom >= +0.0` and
`shift[y] <= +0.0`. So `nll >= +0.0` -- or `-0.0`, its own case, 8.1. With
smoothing, `smooth` is a non-negative average of non-negative quantities by
the same argument, and both coefficients are non-negative, so `row >= 0` too.
**Therefore `ce.total` can never form `inf - inf` and can never produce a
NaN.**

**What (a) through (c) do NOT cover, and it is a stated gap.** `shift[y]` can
be `-inf` from FINITE logits -- `x_y = -3.4e38` against `m = +3.4e38`
overflows the subtraction -- and then `nll` is `+inf` and `ce.loss` is
`+inf`. That is deterministic, the same on every vendor, and admitted. What
is NOT reachable is a NaN, by (c). DEVIATION 1161, and it is the analogue of
transformer DEVIATION 815's nonfinite-intermediate gap with the gap made
smaller by a proof rather than left open.

---

## 9. The stages, in card order

One record per stage per loss call, tags prefixed by the driver
(`core/identity_trace.mojo` rules; tags carry no machine property).

    input.logits      [N, V]   f32   as given
    input.targets     [N]      i32   as given
    ce.max            [N]      f32   L1
    ce.shift          [N, V]   f32   L2
    ce.exp            [N, V]   f32   L3
    ce.denom          [N]      f32   L4    gemm v1 OP_NN (N, 1, V)
    ce.logdenom       [N]      f32   L5
    ce.logp_target    [N]      f32   L6
    ce.nll            [N]      f32   L7
    ce.logp           [N, V]   f32   L8    SMOOTHING ONLY
    ce.logp_sum       [N]      f32   L9    SMOOTHING ONLY, gemm v1 OP_NN (N, 1, V)
    ce.smooth         [N]      f32   L10   SMOOTHING ONLY
    ce.row            [N]      f32   L11
    ce.count          [1]      i32   the integer, exact
    ce.divisor        [1]      f32   ce_divisor's one output
    ce.total          [1]      f32   L12   gemm v1 OP_NN (1, 1, N)
    ce.loss           [1]      f32   L13   SUM and MEAN only
    ce.target_vec     [2]      f32   L15   T_TARGET then T_OTHER, host constants
    ce.weights        [N, V]   f32   L14   BACKWARD
    ce.dlogits        [N, V]   f32   L16   BACKWARD

Nineteen stages, seventeen of which exist on every call.

`ce.divisor` and `ce.target_vec` are HOST constants recorded on the card for
the reason the transformer put `rope.inv_freq` there -- a constant computed
with the wrong spelling is a silent divergence that no activation stage
localizes, and `ce.divisor` in particular is read by two different passes.

**`ce.shift`, `ce.exp` and `ce.weights` are THREE SEPARATE BUFFERS in v1.**
An implementation may compute `w` in place over `e` and save `N * V` floats;
that is an EXECUTION PLAN and it is admitted, but the card then cannot record
both stages, so a run that uses it produces a SHORTER card and may not be
diffed against a full one. Named because "the fused arm agrees with the fused
arm" is the shape of a gate that proves nothing. A run that takes the in-place
arm must be gated against the three-buffer arm at a small shape first.

---

## 10. What "identical" is gated to mean

(a) device card equals host oracle card, BITWISE, at every stage and every
shape; (b) the same bits on every one of 8 repeated launches; (c) BATCH
COMPOSITION invariance -- a row's bits identical whether it is computed alone
or in a chunk with 1, 2 or 255 others, with a negative control showing two
DIFFERENT rows differing; (d) VOCABULARY FOLD invariance -- `ce.denom`
identical under at least three unrelated GEMM execution plans, which reaches
`identical_gemm_with_plan`'s named plans through this lane's routing;
(e) GRADIENT CORRECTNESS, bitwise, against an independently written analytic
reference on an EXACTLY-REPRESENTABLE fixture (section 12); (f) the row-39
audit of section 8; (g) every clause above falsifiable by a NAMED sabotage
that fails a gate, with its predicted INERT set asserted as a mask.

`training/mojo_only/loss_check.mojo` is the gate file. ~~It does not exist.~~
**It exists and RAN at `ecd1a436`, correction made 2026-08-31**: clauses (a)
through (f) all passed on ONE DEVICE, and (f) measured a real defect in
`loss.mojo` that `b90f52ab` fixed. It has run on no second vendor.
FAST arms of (a) are RECORDED, not asserted, where they are vendor-shaped
(the metrics lane's leg-11 lesson). Clause (e) is asserted in BOTH modes,
because on an exactly representable fixture a contracted multiply-add and an
uncontracted one produce the same bits, so a FAST failure there is a real
routing defect -- `IDENTICAL_BACKWARD_PLAN.md`'s G2 makes the same argument.

### 10.1 The sabotage set, one per contested decision

| sabotage | first stage it must move | must be INERT on | what it falsifies |
|---|---|---|---|
| `L_MAX_PLAIN_COMPARE` | `ce.max` on a row planting `-0.0` beside `+0.0` | every ordinary row | the order-free maximum, 5.1 |
| `L_MAX_SEED_ZERO` | `ce.max` on an all-negative row | any row with a positive logit | the `-inf` seed, 5.1 |
| `L_MAX_TOPK_PREFIX` | `ce.max` on a row whose maximum is in the tail | a row whose maximum is at index 0 | 5.2, the max is over the whole row |
| `L_DENOM_SERIAL_CHAIN` | `ce.denom` at `V >= 129` | every `V <= 128` | 5.3, the leaf-and-tree fold |
| `L_DENOM_HALVING_TREE` | `ce.denom` at `V >= 129` | every `V <= 128` | 5.3, `pinned_block_sum`'s stride pairing |
| `L_DENOM_PAD_PLUS_ZERO` | `ce.denom` at `V = 300` and at `V = 128256` | `V` whose every level width is even | the carry, gemm F7 reached through the routing |
| `L_VOCAB_FOLD_READS_LAUNCH` | `ce.denom` | nothing | 7.2, `(L, P)` from `V` alone |
| `L_EXP_STDLIB` | `ce.exp` | nothing under IDENTICAL | row 12's exp |
| `L_LOG_STDLIB` | `ce.logdenom` | nothing under IDENTICAL | row 12's log |
| `L_NLL_VIA_ADDBACK` | `ce.logp_target` on a large-common-offset row | a centered row | 4.2(a) |
| `L_NLL_VIA_LOG_W` | `ce.logp_target` on a row where `e[y]` underflows | a row where `e[y]` is normal | 4.2(b) |
| `L_NLL_VIA_MAX_LOGSOFTMAX` | `ce.logp_target` | nothing | 4.2(c) |
| `L_NEG_VIA_ZERO_SUB` | `ce.nll` at `V = 1` | every row whose `lp_y` is not `+0.0` | 4.3 |
| `L_SMOOTH_FOLDED_CONSTANT` | `ce.smooth` at `EPS = 0.1` | `EPS = 0` | 6.2(a) |
| `L_SMOOTH_FUSED_COMBINE` | `ce.row` at `EPS = 0.1` | `EPS = 0` | 6.2(b) |
| `L_SMOOTH_ALWAYS_SPELLED` | `ce.row` at `EPS = 0` on a row whose `nll` is `-0.0` | every other row and every `EPS != 0` | 6.2(c) |
| `L_REDUCE_SERIAL` | `ce.total` at `N >= 129` | every `N <= 128` | 5.4 |
| `L_MEAN_RECIPROCAL_MUL` | `ce.loss` | `divisor` an exact power of two | 5.5 |
| `L_IGNORED_ROW_NEG_ZERO` | `ce.row` | **`ce.total`, which it must NOT move** | 7.3 |
| `L_IGNORED_ROW_SKIPPED` | `ce.row` (unwritten memory) | nothing | 7.3, the store is required |
| `L_W_VIA_EXP_LOGP` | `ce.weights` | nothing | L14's division |
| `L_GRAD_SIGN` | `ce.dlogits` | a row where `w == t` at every `v` | 6.4, `w - t` |
| `L_GRAD_TARGET_OFF_BY_ONE` | `ce.dlogits` | a uniform row at `V = 1` | the target column |
| `L_GRAD_DIVISOR_IS_N` | `ce.dlogits` when a row is ignored | a fixture with no ignored row | 5.5, the one producer |
| `L_GRAD_RECIPROCAL_MUL` | `ce.dlogits` | `divisor` an exact power of two | L16's division |

Each must move the stage its OWN clause writes and no earlier one, which is
the discipline the mamba lane's six arms were held to.

**Three of these pass by construction on the obvious fixture and are the ones
most likely to be deleted as inert.** `L_GRAD_DIVISOR_IS_N` needs an ignored
row. `L_SMOOTH_ALWAYS_SPELLED` needs a `-0.0` row loss. `L_IGNORED_ROW_NEG_ZERO`
needs the CARD rather than the loss. If the gate file is written without
clause (a)'s per-stage comparison, all three look inert and get removed. That
is the transformer contract's warning about `S07_ROPE_RELATIVE_POSITION` and
`S19_VALUE_SUM_VIA_GEMM`, repeated here because it is the same failure.

---

## 11. Not claimed

- **One loss, not a training step.** No optimizer, no weight update, no
  gradient accumulation across steps, no RNG, no dropout, no gradient
  clipping, no mixed precision, no loss scaling, no distributed all-reduce.
  `IDENTICAL_BACKWARD_PLAN.md` section 4 is the dependency list and this
  document is one item on it.
- **Not the gradient with respect to anything but the logits.** The chain
  back into the `lm_head` GEMM is `gemm_backward.mojo`'s and the chain back
  into the block is the transformer lane's backward, which does not exist.
- **Not `reduction = NONE`'s backward.** A per-row upstream vector adds a
  second product per cell whose placement -- before or after L16's division
  -- is a real decision with two different answers, and this lane has no
  caller for it. An unused wrapper is an ungated one.
- **Not a per-class `weight` vector.** It would make `count` a SUM OF FLOATS
  and therefore a fourth fold with its own clause, its own fixtures and its
  own sabotages, and it would make `ce.divisor` data-dependent. Section 5.5.
- **Not a per-row or masked vocabulary**, no hierarchical softmax, no
  sampled softmax, no adaptive softmax, no candidate sets, no
  speculative-decoding restricted head. Section 3.1 is the reason and it is a
  consequence of DEVIATION 1152, stated as a cost.
- **Not a fused loss.** No chunked or fused cross-entropy kernel that never
  materializes the logits, which is what a production Llama head actually
  wants. That kernel splits the vocabulary axis and rescales, which is
  section 6 of the transformer contract's online-softmax argument moved to a
  new axis. It is a v2 at the earliest.
- **Not soft targets**, no probability-valued target tensor, no distillation
  loss, no KL divergence.
- **Not BF16, FP16, FP8, TF32 or any quantization.** Not FP64 anywhere on
  device; Metal does not have it. `ForCausalLMLoss` :59's upcast to float is
  followed, not the storage dtype.
- **Not agreement with PyTorch, HuggingFace or MAX.** This profile's fold
  order, transcendentals, division and log-sum-exp spelling are OURS. The
  claim is that our arithmetic gives the same bits on three vendors, plus
  agreement with a float64 reference to a stated tolerance once a corpus
  exists. A reader who takes "identical" to mean "equal to torch" has taken
  more than is offered.
- **Two KNOWING DEPARTURES from the reference's behavior**, both because a
  vendor-shaped NaN may not enter a certified stage. Torch returns NaN for a
  MEAN over zero unignored rows; **we refuse**. Torch propagates a NaN logit
  through to a NaN loss; **we refuse the input**. Section 5.5 and section 8.
- **Not the derivative of the floating-point forward.** L15 and L16 are the
  analytic gradient of the real-valued loss, spelled. Section 6.3.
- **No performance number.** None has been taken and none will be quoted
  until it is, alternated inside one thermal window
  (`[[mojolearn-box-drifts]]`).
- **Nothing cross-vendor until a leg runs.** Everything here is CONSTRUCTION.
  The GEMM lane's own history is the standing reason to say so -- Apple and
  AMD agreed bit for bit through 302 stages while NVIDIA diverged at
  `tree001.winners.scores` -- so two backends agreeing closes nothing.
- ~~**Nothing here has been compiled.** No file in `training/` other than
  these three has any content, `training/mojo_only/loss_check.mojo` does not
  exist, and no gate in section 10 has ever been run.~~ **FALSE SINCE
  `ecd1a436`, corrected 2026-08-31.** `loss_check.mojo` and
  `loss_fixture.mojo` exist and ran, six clauses of section 10 passed on one
  device, and the sabotage arms were built and fired. `training/` has since
  grown the optimizer and composed-step files as well. **The cross-vendor
  bullet above still stands, unchanged: one device is not three.**

---

## 12. How the GRADIENT is gated BITWISE, not approximately

This section exists because the brief for this lane put it in these words --
a loss whose gradient is only checked against finite differences is checked
against a TOLERANCE, and this lane deals in BITS.

**Finite differences are DEMOTED to a reported diagnostic and are NOT a gate
in this profile.** DEVIATION 1163. `IDENTICAL_BACKWARD_PLAN.md` section 5.1
item 1 made the same demotion for the GEMM backward and the reasoning is
identical -- on a fixture chosen so the derivative is exactly representable,
the derivative can be WRITTEN DOWN and compared to, and comparing two exact
numbers is simpler and stronger than comparing two exact differences of exact
numbers. A central difference has a step size, and a step size is a tolerance
wearing a different hat.

### 12.1 The UNIFORM family, where every gradient cell is exact

Take a row whose `V` logits are all the SAME value `c`, with `V` a power of
two and `divisor` a power of two.

    m         = c exactly                       (every element equal)
    shift[v]  = +0.0 exactly                    (x - x in round-to-nearest)
    e[v]      = 1.0 exactly                     (portable_expf(+0.0), 8.2(b))
    denom     = Float32(V) exactly              (V ones, exact for V <= 2^24,
                                                 in EVERY fold order, so this
                                                 arm does not gate the fold)
    w[v]      = 1/V exactly                     (V a power of two)
    t         = 1.0 at y, +0.0 elsewhere        (EPS = 0)
    dl[y]     = (1/V - 1) / divisor  EXACT      (a dyadic rational)
    dl[v!=y]  = (1/V) / divisor      EXACT

At `V = 4` and `divisor = 2` those are `-0.375` and `+0.125`. **Every cell of
the gradient is a number a person can write down**, so the check asserts the
BIT PATTERN and there is no epsilon anywhere in it.

**What this family gates.** The formula, the sign, the target COLUMN, the
divisor's identity, the target-vector constants, and the routing of every
buffer. A transposed index, an off-by-one target, a flipped sign or a divisor
taken from the wrong producer all fail it.

**What this family CANNOT gate, and saying so is the point of the
subsection.** Every quantity in it is exact, so it separates NO spelling from
any other spelling. `identical_div(1.0, 4.0)` and `1.0 * (1/4)` agree. A
serial fold and a balanced tree of ones agree. `identical_exp` and
`std.math.exp` agree at `+0.0`. **Run alone, this family would pass every
sabotage in section 10.1 and gate nothing about the arithmetic.** It is the
CORRECTNESS half and it needs the SPELLING half beside it, which is clause
(a)'s per-stage bitwise comparison against the oracle on the ordinary
fixtures.

### 12.2 The SATURATING family, exact and NOT uniform

Take a row with `c` copies of a value `a` and `V - c` copies of `a - 200.0`,
with `c` a power of two.

    m         = a                               (a > a - 200)
    shift     = +0.0 for the c high cells, -200.0 for the rest
    e         = 1.0 for the c high cells, EXACTLY +0.0 for the rest
                (portable_expf returns +0.0 below -87.33655)
    denom     = Float32(c) exactly
    w         = 1/c for the high cells, EXACTLY +0.0 for the rest
    dl        = ((1/c) - onehot) / divisor at the high cells, exact
                (+0.0 - onehot) / divisor  at the low cells, exact

Again every cell is writable. This family adds four things the uniform one
lacks -- a genuine argmax structure, an exercised underflow edge, exact
`+0.0` weights whose signs must be `+`, and a case where the TARGET is a low
cell (`dl[y] = (-1)/divisor`, exactly). It still gates no spelling, for the
same reason.

### 12.3 The three arms, and what each is for

| arm | asserts | mode | gates |
|---|---|---|---|
| **A1 EXACT-ANALYTIC** | device and host both equal a hand-written closed form on 12.1 and 12.2, by bits | BOTH | correctness -- formula, sign, index, divisor |
| **A2 ORACLE-BITWISE** | device equals `loss_oracle.mojo` at every stage on the ordinary fixtures, by bits | IDENTICAL asserted, FAST recorded | spelling -- every clause of sections 4, 5 and 6 |
| **A3 FLOAT64 TOLERANCE** | the FP32 answer is near the real number, on a Float64 host reference | reported, never asserted | that A1 and A2 are not both consistently wrong |

**A3 is the only place a tolerance appears and it asserts nothing.** It is
the mamba oracle's Float64 arm at the bottom of `mamba_oracle.mojo` -- "a
TOLERANCE instrument, never a bitwise one" -- and it exists so a systematic
error that A1's algebra and A2's oracle share would be visible.

### 12.4 The vacuity guards, enforced rather than argued

Copied in shape from `IDENTICAL_BACKWARD_PLAN.md`'s G2, which enforces its
three and raises rather than adjusting the fixture.

1. **`V != N` and `V != divisor` in every A1 case**, or the check RAISES. A
   square-ish fixture lets a transposed index produce a plausible matrix of
   the right size.
2. **`V` and `c` powers of two and `divisor` a power of two in A1**, or the
   check RAISES, because the exactness argument depends on it and an
   unchecked exactness argument is how a gate comes to assert what the code
   does rather than what it should do.
3. **At least one A1 case with an ignored row and at least one without**, or
   `L_GRAD_DIVISOR_IS_N` cannot fire and its predicted inert mask is
   untested.
4. **A `check_the_exact_fixture_is_vacuous` arm**, which re-spells the
   reciprocal-multiply gradient LOCALLY on the A1 fixture and DEMONSTRATES
   that it agrees bit for bit, in a clean build, with a non-power-of-two
   control beside it showing it disagreeing there. Without that arm a reader
   is entitled to think A1 gates the spelling, and it does not. This is
   `check_the_square_fixture_is_vacuous`'s pattern (DEVIATION 1053) at a new
   site.

---

## 13. Where the code goes, and the deviation numbers

    training/IDENTICAL_LOSS_CONTRACT.md      this file
    training/mojo_only/loss_oracle.mojo      the NORMATIVE host oracle
    training/mojo_only/loss.mojo             the device spelling, one source
    training/mojo_only/loss_fixture.mojo     NOT WRITTEN -- config and planted cases
    training/mojo_only/loss_check.mojo       NOT WRITTEN -- the gates and the sabotages
    training/corpus/                         NOT WRITTEN -- the independent torch reference

`training/__init__.mojo` and `training/mojo_only/__init__.mojo` already exist
and were not touched by this lane.

### 13.1 The deviation block, 1150 through 1169

| # | what |
|---|---|
| 1150 | the composition `loss.ce.fp32.v1` and its IDENTITY_PATHS row |
| 1151 | the row maximum through `identical_fmax`, fold shape FREE, seeded `-inf`, and the refusal of `pinned_block_max` |
| 1152 | **all three folds routed through gemm v1 `OP_NN` against a ones vector**, the departure from transformer 5.3, and the fixed-`V` argument that licenses it |
| 1153 | the log-sum-exp spelled `shift[y] - logdenom`, and the three refused alternatives |
| 1154 | the negation spelled BY BITS |
| 1155 | label smoothing -- divide by `V` first, the unfused combine, and the `EPS == 0` BRANCH |
| 1156 | the MEAN divisor as an exact integer count, `count == 0` REFUSED, and the divisor's single producer shared by forward and backward |
| 1157 | the softmax weight `identical_div(e, denom)`, one division per cell, reciprocal refused -- transformer 806 at a second site |
| 1158 | the gradient cell `identical_div(ftz(w - t), divisor)` and the smoothed target as TWO host constants |
| 1159 | the ignored row -- `+0.0` STORED both in `ce.row` and in `ce.dlogits`, and the proof of bitwise inertness in the batch fold |
| 1160 | the input refusals -- nonfinite BY BITS, target range by integer compare, `count` and `num_items` by integer compare |
| 1161 | the three theorems of 8.2 and the one stated gap they leave (`shift[y]` overflowing to `-inf` from finite logits) |
| 1162 | the row-independence clause -- `N` is a launch shape for everything except L12 |
| 1163 | the EXACT-ANALYTIC gradient gate, the two exact fixture families, and the DEMOTION of finite differences |
| 1164 | the local `refuse_nonfinite`, its debt to `mojo_only/numerics.mojo` |
| 1165 | the local `pinned_block_fmax`, its debt to `core/pinned_reduce.mojo::pinned_block_max` |
| 1166 | the stage list and its card tags |
| 1167 | the sabotage set and its predicted INERT masks |
| 1168 | the block size resolved from `mojo_only/kernel_matrix.mojo` rather than written as a literal, and the argument that every scheduling row here IS scheduling |
| 1169 | reserved |

No number outside 1150-1169 is claimed by this lane. Numbers cited from
elsewhere (204, 258, 504, 522, 528, 720, 740-746, 800-819, 820-826, 850-852,
1051-1060) belong to other lanes and are cited, never redefined.

---

## OWED, AND WHY I DID NOT DO IT HERE

This lane was permitted to write exactly three files. Everything below needs
a file this lane may not touch, and each item names the file, the change and
the reason it is somebody else's edit.

1. **`mojo_only/numerics.mojo` should carry `refuse_nonfinite`.** It existed
   once, at `mamba/mojo_only/mamba_oracle.mojo:57`. The concurrent optimizer
   lane added a second at `training/mojo_only/optimizer_oracle.mojo:162` on
   2026-08-25 and its own header already counts them; this lane makes a third
   (DEVIATION 1164). **Three copies of a refusal are three chances to
   disagree about what a NaN is**, and two of them are now in one directory
   written the same day, which is the strongest possible argument for making
   the lift once. `numerics.mojo` is the file every other directory already
   imports and the natural home, and it is under concurrent edit by the
   numerics lane, so the lift is theirs to make. The copy in
   `loss_oracle.mojo` carries a pointer to this item.

2. **`mojo_only/numerics.mojo` should carry `neg_by_bits`.** DEVIATION 1154
   introduces it and it is one XOR. It belongs beside `identical_mul`, for
   the reason DEVIATION 826 gives about `pinned_mul` -- a reader looks for it
   there, and four copies of an arithmetic have four chances to drift. Not
   done here for the same concurrency reason.

3. **`core/pinned_reduce.mojo::pinned_block_max` should take `identical_fmax`
   as its combine step.** This is not this lane's opinion; it is
   `portable_fmaxf`'s own docstring -- *"The clean fix is to give that fold
   this function as its combine step, which is the block's owner's call and
   not this file's."* Doing it would delete `loss.mojo`'s local
   `pinned_block_fmax` (DEVIATION 1165) and would also fix the transformer
   lane's S14, which needs exactly the same helper. **Two lanes now want this
   one edit**, which is the argument for making it once rather than growing a
   third copy.

4. **`IDENTITY_PATHS.md` needs a row for this profile.** DEVIATION 1150. The
   row's pathway column is the loss and its status is CONSTRUCTION, NOT
   MEASURED. That file is the repository's ledger and is edited by whoever
   owns the ledger.

5. **`training/mojo_only/loss_fixture.mojo` and
   `training/mojo_only/loss_check.mojo` do not exist**, so **not one clause
   in this document has ever been falsified by a sabotage.** Every entry in
   section 10.1 is a specification for a gate, not a report of one. This is
   the largest single gap in the lane and it is deliberate -- the brief
   scoped this pass to the contract, the oracle and the device spelling.

6. **`training/corpus/` does not exist**, so clause 12.3's A3 arm has no
   Float64 reference to compare against and the "agreement with a float64
   reference to a stated tolerance" phrase in section 11 is a promise about a
   thing that has not been built. `mamba/corpus/` and
   `tools/mamba_corpus_check.py` are the pattern.

7. **`bench/` has no loss shape and `SUPPORT_MATRIX.md` has no loss row.** No
   performance number exists and section 11 forbids quoting one.

8. **The three UNVERIFIED decisions of section 1 need a PyTorch checkout.**
   `/Users/andrewhendel/CascadeProjects/upstream/` has no `pytorch`, so
   ATen's `log_softmax`, its `nll_loss` and its `cross_entropy_loss`'s
   label-smoothing branch could not be read. Adding a checkout is an
   environment change outside this lane's three files. The three decisions
   are L6 (4.2), L13/L14/L16's division (5.5) and the smoothing combine (6).

9. **A `[[mojolearn-shared-checkout-parents]]` commit is owed by whoever
   commits these three files.** This lane does not touch git.
