# The IDENTICAL FP32 cross-entropy loss contract

# PROFILE `mojolearn.identical.loss.ce.fp32.v1`

## STATUS

**COMPILED, RUN AND CARDED ON TWO COLUMNS, APPLE AND AMD. NO NVIDIA.**

Two separate runs, both real, and neither is three vendors.

- **Apple, first execution, commit `ecd1a436`.** `loss.mojo`,
  `loss_oracle.mojo`, `loss_check.mojo` and `loss_fixture.mojo` all ran for
  the first time, **six clauses passed, 24 cases and 61,925 cells matched
  device against oracle BITWISE**, the sabotage arms were built and fired,
  and **clause (f) MEASURED a real defect in the device entry point that
  `b90f52ab` then fixed (DEVIATION 1495)**. Clause (f) carries the measured
  finding that `loss.mojo` never called `ce_refuse_inputs` at all (DEVIATION
  1460), so a leg that skips (f) has not looked at the device-side refusal
  once.
- **The two-column legs, 2026-08-28.** `training-loss.identical.card`, 17
  records, md5 `a87615d9`, **byte-identical on an Apple M4 and an AMD
  MI325X**. **Clause (a) PASS on both, 24 cases, all 61,925 cells. Clause (e)
  PASS on both**, the exact-analytic gradient, 4 cases, 264 cell comparisons
  against a hand-written closed form, no epsilon anywhere. **Clauses (b), (c),
  (d) and (f) SKIPPED on both.**

What is NOT closed by any of that, from the run's own SCOPE line. NO NVIDIA
LEG. Clauses (b), (c), (d) and (f) on either column, and without (d) the four
fold arms reached through `gemm_identical.mojo` are gated only by clause (a),
which sees ONE execution plan, whichever `choose_gemm_plan` picked. An
INDEPENDENT reference, because `training/corpus/` does not exist, so every
clause is our device against our oracle and **three of the twenty stages are
literally the SAME HOST CODE on both sides** (`stage_is_host_constant`). The
SHIPPED vocabulary `V = 128256` unless `MOJOLEARN_LOSS_CHECK_SHIPPED_V` was
set. FAST mode. Clause 5.1(c), which has no switch at all. And there is no
`pixi.toml` task for this lane; the legs drove the gate by path.

**Two backends agreeing closes nothing.** Apple and AMD agreed bit for bit
through 302 GBDT stages while NVIDIA diverged at `tree001.winners.scores`.

DEVIATIONS 1150 through 1169 are this lane's and nothing here uses a number
outside that range.

---

**THE PROFILE NAME IS PART OF THE CONTRACT.** Changing any seam decision in
section 4, any fold topology in section 5, any constant in section 3 or the
stage list in section 9 creates a v2. FAST is unversioned and makes no identity
claim.

The completion claim this contract supports is one sentence. Softmax
cross-entropy over logits, and its gradient with respect to those logits, in
FP32, is bit-identical on Apple, NVIDIA and AMD GPUs at every stage, at every
launch, at every batch composition and at every row chunking, under the
declared profile. **Not an identical training step**, not an identical model,
not an identical optimizer, not agreement with PyTorch. The cross-vendor half
is earned only by a multi-vendor leg.

The code form of every clause is `training/checks/loss_oracle.mojo` (the
NORMATIVE host oracle) and `training/checks/loss.mojo` (the device spelling,
one source for three vendors).

---

## 0. What this lane does NOT rebuild, and the one thing it does

Read in the tree on 2026-08-25 and cited by file and symbol.

| piece | verdict | where it already is |
|---|---|---|
| `exp` | REUSED | `checks/numerics.mojo::identical_exp` over `::portable_expf`, row 12 |
| `log` | REUSED | `::identical_log` over `::portable_logf`. Mojo's own `std.math.log` carries about `5e-8` absolute error and re-decides plateau ties, which is why row 12 exists. **This lane never spells `std.math.log` under IDENTICAL** |
| division | REUSED | `::identical_div` over `::portable_divf`, DEVIATION 740, row 49 |
| the order-free maximum | REUSED, and built for exactly this reduction | `::identical_fmax` over `::portable_fmaxf`, DEVIATION 825 |
| the flush | REUSED | `::ftz`, row 10 |
| the uncontractible product | REUSED | `::identical_mul`, DEVIATION 826, `identical_mul_add(a, b, -0.0)`. **This lane makes no further copy** |
| the fused multiply-add | REUSED | `::identical_mul_add`, row 9 |
| **every fold in this document** | **ROUTED, NOT REBUILT** | `mojolearn.identical.gemm.fp32.v1`, `gemm_oracle` on the host and `identical_gemm_into` on the device. Section 4 is the argument |
| the reduction-as-a-GEMM technique | REUSED | `gemm/checks/gemm_backward.mojo`, DEVIATION 851, which proved `db[j] = sum_i dC[i,j]` is `ones . dC` under gemm v1 |
| the stage card and the differ | REUSED | `core/identity_trace.mojo`, `tools/identity_trace_diff.py` |
| refusing a nonfinite input | PATTERN REUSED, CODE COPIED | `mamba/checks/mamba_oracle.mojo:57` is the first and `training/checks/optimizer_oracle.mojo:162` the second; this is a THIRD copy and the lift to `numerics.mojo` is owed. DEVIATION 1164 |
| **the log-sum-exp, the negation, the label-smoothing combine and the gradient cell** | **NEW** | sections 4 and 6. That is the entire new arithmetic of this lane |

**Three refusals, each of a thing already in the tree that looks like the
right helper and is not.** `core/pinned_reduce.mojo::pinned_block_sum` (:73)
may not be the denominator, 4.3. `::pinned_block_max` (:159) may not be the
row maximum, 4.2; its own block comment says a caller whose inputs can carry
`±0.0` or NaN must state why first, and this caller cannot.
`max.gpu.primitives.block.sum` may not be any fold here, because its
cross-lane stage folds at the HARDWARE lane width, 32 on Apple and NVIDIA and
64 on AMD's CDNA wavefront.

---

## 1. The reference, pinned

| what | where | pin |
|---|---|---|
| the causal-LM loss wrapper | `src/transformers/loss/loss_utils.py::ForCausalLMLoss` (:49-71) | huggingface/transformers `d56c55bf564ddb176759eb6ec199442682564916` |
| the reduction and the `num_items_in_batch` divide | `::fixed_cross_entropy` (:32-46) | same |
| the label shift, pad with `ignore_index` then drop the first | `ForCausalLMLoss` :61-64 | same |
| the flatten to `[-1, vocab_size]` | :67-68 | same |
| the upcast to FP32 before the loss | :58-59, `logits.float()` | same |
| the loss itself | `torch.nn.functional.cross_entropy` into ATen | **NOT READ. There is no PyTorch checkout in `/Users/andrewhendel/CascadeProjects/upstream/`** |
| the device kernel SHAPE for a row softmax | `max/kernels/src/nn/softmax.mojo::softmax_kernel` (:840-1015) | modular/modular `10d978e3c783ef940d1d30d0a10852b69fe285c8` |
| every fold | `mojolearn.identical.gemm.fp32.v1`, row 40, three-vendor at E3 round 11 (`144aa5b`) | this repository |

**The absent PyTorch checkout is the single largest evidential gap here**, and
it is the same gap the transformer lane records at its 5.4. Three decisions
below are pinned on the one-rounding argument and on internal consistency, NOT
on having read ATen, and each is marked UNVERIFIED AGAINST THE REFERENCE where
it appears: the log-sum-exp spelling of 4.2, the label smoothing combine of
section 5, and the division of 4.5. When a checkout lands those are the first
three to re-read, and a disagreement is a finding rather than an
embarrassment.

**MAX's logsoftmax is evidence about MAX and not about the reference**, and it
is worth recording because it is the spelling a kernel author reaches for.
`softmax.mojo:1004` computes `log(exp(x - m) * recip(sum))`, an exp, a
reciprocal, a product and a log, four roundings, where this profile computes
`(x - m) - log(sum)`, two roundings and no exp on the target path at all.

---

## 2. What one loss call is

    logits   [N, V]   Float32, row-major, N = batch * tokens
    targets  [N]      Int32, each either a class in [0, V) or ignore_index
    ->
    row      [N]      Float32, the per-row loss
    loss     [1]      Float32, the reduced loss           (SUM and MEAN only)
    dlogits  [N, V]   Float32, dLoss/dlogits              (the backward)

One row `i` with target `y = targets[i]`, in order.

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

and, when it is off, `row = nll`. Then

    total    = gemm v1 fold of row over i                             L12
    loss     = identical_div(total, divisor)                          L13

and the backward

    w[v]     = ftz(identical_div(e[v], denom))                        L14
    t[v]     = T_TARGET if v == y else T_OTHER                        L15
    dl[v]    = ftz(identical_div(ftz(ftz(w[v]) - t[v]), divisor))     L16

An IGNORED row writes `row = +0.0` and a gradient row of `V` copies of
`+0.0`, both STORED and neither skipped, and does not contribute to `count`.
Section 7.3.

**Inference-shaped weighting is refused. There is no per-class `weight`
vector in v1**, so `count` is an integer and never a sum of floats, which
removes an entire fold from the document.

## 3. Profile constants
| constant | value | frozen? |
|---|---|---|
| dtype | Float32 for logits, every intermediate, every accumulator, the loss and the gradient | **YES** |
| target dtype | Int32 | **YES**, integers do not flush and do not round |
| `CONTRACT_K_LEAF_MIN`, `CONTRACT_MAX_LEAVES` | 128, 1024 | **YES**, inherited from gemm v1 |
| `V` | free per model, **FROZEN FOR THE RUN**, 2.1 | see 3.1 |
| `ignore_index` | free, default `-100` | no |
| `label_smoothing`, `EPS` | Float32 in `[0, 1)`, **and `EPS == 0` takes a different code path** | **YES**, the eps-zero branch is a clause |
| `ONE_MINUS_EPS` | `ftz(Float32(1.0) - EPS)`, host, once | **YES** |
| `T_OTHER` | `ftz(identical_div(EPS, Float32(V)))`, host, once | **YES** |
| `T_TARGET` | `ftz(ONE_MINUS_EPS + T_OTHER)`, host, once | **YES** |
| reduction | `NONE`, `SUM` or `MEAN` | no |
| `divisor` | `Float32(count)` for MEAN, `Float32(num_items)` or `+1.0` for SUM | **YES**, its PRODUCER is frozen, 4.5 |
| `N` | free, a launch shape | no |
| per-class weights | **none, refused** | **YES** |

Refusals by name rather than silent truncation. `V >= 1`; `N >= 1`;
`count >= 1` under MEAN; `num_items >= 1` when supplied; every target either
`ignore_index` or in `[0, V)`; no NaN and no infinity in `logits`;
`count <= 16777216` so `Float32(count)` is exact; `N <= 4000000` so the batch
fold's `k` stays inside the range the GEMM lane's own sweep exercised.

### 3.1 Why `V` is FROZEN FOR THE RUN, and why that is the load-bearing fact

Under gemm v1, `L` and `P` are a pure function of `k` and the two profile
constants, so a fold whose length changes between calls changes its tree and
its bits. **The vocabulary axis does not change between calls.** `V` is a
property of the tokenizer, fixed when the model is built, identical in every
step, in prefill and decode, at every batch size and every sequence length.
It is the same KIND of axis as the transformer contract's `head_dim` and NOT
the same kind as its key axis, and transformer 7.2 drew exactly that line when
it routed the QK product (`k = head_dim`) through gemm v1 and refused to route
the attention-weighted value sum (`k = kv_len`). **The vocabulary fold is an
S11, not an S19.**

**The price. A per-row vocabulary is REFUSED.** A pruned, masked or ragged
vocabulary, a per-row candidate set, a speculative-decoding restricted head, a
hierarchical softmax, has a per-row `k`, therefore a per-row `P`, therefore
per-row bits, and it is not this profile.

### 3.2 What the shipped vocabulary actually does to the tree, computed

**What the shipped vocabulary does to the tree, computed.** `V = 128256` gives
`ceil(128256/128) = 1002 <= 1024`, so `L = 128` and `P = 1002`, and
`128 * 1002 = 128256` **exactly**. Level widths are 1002, 501, 251, 126, 63,
32, 16, 8, 4, 2, 1, so levels 1, 2 and 4 CARRY.

- **The RAGGED LEAF NEVER OCCURS at the shipped vocabulary**, since 128256 is
  an exact multiple of 128, so the short-last-leaf path is never taken. A lane
  that only ran the shipped shape would never see that. **`V = 300` is
  therefore a required shape**, `P = 3` with a ragged 44-element last leaf and
  one carry, which is the GEMM contract's own clause-5 fixture lifted here on
  purpose.
- **The CARRY does occur, three times**, so the odd-tail path is exercised at
  the shipped shape too.

**The gate shape.** `V` in `{1, 2, 3, 129, 256, 300, 1024, 128256}`, `N` in
`{1, 2, 3, 4, 5, 129, 300, 512}`, `ignore_index` present and absent, `EPS` in
`{0.0, 0.1}`, reduction in `{NONE, SUM, MEAN}`. `V = 1` and `N = 1` are the
`P == 1` cases where the tree performs NO addition and where the row loss is
provably `-0.0`. `V = 129` is `P = 2` with a one-element ragged leaf, the most
extreme raggedness the leaf rule can produce.

---

## 4. The seams, every one, with the fused-or-unfused decision

FMA contraction is PER SEAM. FUSED is one rounding through
`identical_mul_add`; PRODUCT is one rounding through `identical_mul`, the
spelling no codegen may contract into a neighboring add and which preserves a
`-0.0` product. Every seam's RESULT passes `ftz` and every operand LOADED from
a buffer passes `ftz`. **Copies and integer work are NOT seams**: the label
shift, the flatten, the target load, the `v == y` test and `count` are exact.

| # | seam | pinned spelling | fused? |
|---|---|---|---|
| L1 | the row maximum | `identical_fmax` folded over EVERY element of the row in ANY order, seeded `-inf`. 4.2 | n/a |
| L2 | `x - m` | `ftz(ftz(x_v) - ftz(m))` | subtract |
| L3 | `exp(x - m)` | `identical_exp`, row 12's polynomial | n/a |
| L4 | the denominator | **gemm v1 `OP_NN` at `(N, 1, V)` against a ones vector.** 4.3 | per gemm v1 |
| L5 | `log(denom)` | `identical_log`. Its argument is provably in `[1, V]`, 6.4 | n/a |
| L6 | the target log-probability | `ftz(ftz(shift[y]) - ftz(logdenom))`. **NOT** `(m + logdenom) - x_y` and **NOT** `log(w[y])`. 3.2 | subtract |
| L7 | the negation | `neg_by_bits`, an XOR of the sign bit. 4.3 | n/a |
| L8 | every log-probability | the same expression as L6 at every `v`. Smoothing only | subtract |
| L9 | the smoothing fold | **gemm v1 `OP_NN` at `(N, 1, V)`** against the same ones vector | per gemm v1 |
| L10 | the smoothing average | `neg_by_bits(ftz(identical_div(lpsum, Float32(V))))`, ONE division by `V`, never a host-folded `EPS/V` product | n/a |
| L11 | the smoothing combine | `ftz(ftz(identical_mul(ONE_MINUS_EPS, nll)) + ftz(identical_mul(EPS, smooth)))`. **UNFUSED** | PRODUCT twice, then an add |
| L12 | the batch reduction | **gemm v1 `OP_NN` at `(1, 1, N)`.** 4.4 | per gemm v1 |
| L13 | the reduction divide | `identical_div(total, divisor)`, ONE division, never a reciprocal multiplied in | n/a |
| L14 | the softmax weight | `ftz(identical_div(e[v], denom))`, ONE division per weight. **This is transformer DEVIATION 806 at a second site and the two must agree** | n/a |
| L15 | the target vector | two HOST constants, `T_TARGET` and `T_OTHER`, selected by an integer compare. 5.2 | n/a |
| L16 | the gradient cell | `ftz(identical_div(ftz(ftz(w[v]) - t[v]), divisor))` | subtract, then divide |

### 4.1 Where this AGREES with the transformer contract, and where it DEPARTS

The brief was to make the same decisions transformer section 5 made or to
number every departure. **The departure is ONE clause and it is DEVIATION
1152.** 5.1 (the max is `identical_fmax`, fold shape FREE, `pinned_block_max`
refused), 5.2 (the max over the WHOLE row), S15/S16 (`exp(ftz(s) - ftz(m))`),
5.4 (the division is `identical_div`, one rounding, reciprocal refused, at all
three of L13, L14 and L16, which was an explicit requirement of this lane's
brief) and section 8 (refuse nonfinite BY BITS, admit `-0.0`, name every zero
site) are all the SAME. 5.5 is the same in spirit and different in form: the
vocabulary axis is walked in ASCENDING absolute class index and the
leaf-and-tree topology is a pure function of `V` alone, and there is no
per-launch local index to get wrong because there is no slicing of the
vocabulary axis. S13's mask clause is NOT APPLICABLE, since there is no mask;
the corresponding clause here is 6.2, the ignored ROW, which has the same
shape. **5.3, the serial ascending denominator, is the departure**, and 4.3 is
the argument.

### 4.2 L6, the log-sum-exp, and the three spellings that are not it

`lp_y = ftz(ftz(shift[y]) - ftz(logdenom))`, where `shift[y]` is what L2
already computed. **UNVERIFIED AGAINST THE REFERENCE.**

**(a) `(m + logdenom) - x_y`** re-adds the row maximum, the quantity the shift
existed to remove, so on a row whose logits are large the addition loses
`logdenom`'s low bits entirely and the subsequent subtraction cancels
catastrophically; the pinned spelling never forms a quantity larger than
`max(|shift[y]|, logdenom)`. Sabotage `L_NLL_VIA_ADDBACK`, whose separating
fixture is a row with a large common offset. **A fixture of small centered
logits makes it agree to the last bit, so that row is the only thing gating
this clause.**

**(b) `-log(w[y])`** costs a transcendental and a division where a subtraction
suffices, and `w[y]` underflows to `+0.0` on a confident-wrong row, giving
`+inf` where the pinned spelling gives a large finite loss. Sabotage
`L_NLL_VIA_LOG_W`, fixture a row where `shift[y] < -87.34`.

**(c) MAX's `log(exp(shift) * recip(denom))`** is four roundings against two
and carries the reciprocal transformer 5.4 refused. Sabotage
`L_NLL_VIA_MAX_LOGSOFTMAX`.

### 4.3 L7 and L10, the negation, spelled BY BITS

    neg_by_bits(x) = bitcast(bits(x) XOR 0x80000000)

DEVIATION 1154. `0.0 - x` is WRONG at `x = +0.0`, where it gives `+0.0` and
IEEE negation gives `-0.0`, and 8.1 shows `lp_y == +0.0` is reachable, so that
is not hypothetical. Sabotage `L_NEG_VIA_ZERO_SUB`, which must move `ce.nll`
on a `V = 1` row and must NOT move it on an ordinary row.
`identical_mul(x, -1.0)` is CORRECT at every input including both zeros, and
is refused only because it presents a floating-point operation where an exact
bit operation will do. `-x` in source is the compiler's choice of the two, and
this profile does not rely on "very probably a sign-bit XOR". The operand is
already flushed by the seam that produced it, and **integer ops do not flush
on Metal** (DEVIATION 746(i)), which is the second reason the bit spelling is
safe.

---

## 5. The reduction orders

### 5.1 L1, the row maximum, is `identical_fmax` and its fold shape is FREE

    m(row) = the fold of identical_fmax over EVERY element of the row,
             in ANY order, seeded -inf (0xFF800000).

Transformer clause 5.1 verbatim, DEVIATION 804/825, at a second site.
`identical_fmax` canonicalizes a NaN operand to `0x7FC00000`, flushes both
operands, then selects on `_total_order_key`, under which `+0.0` keys at
`0x80000000` and `-0.0` at `0x7FFFFFFF`. **There is no hardware `max` and no
float compare in it, so the result is commutative and associative over all of
Float32 including both zeros and NaN, and this profile therefore does not pin a
fold topology for the maximum.** That is the only place here an execution plan
may choose its own tree, and it may because the operation is exactly
associative, not because the difference is thought to be small.

**Why an unpinned float max would be a defect.** Row 39 MEASURED
`max(+0.0, -0.0)` as `-0.0` on Apple and `+0.0` on NVIDIA and AMD, and a row of
128256 logits reaches both zero signs easily, so this is a live three-vendor
split at the largest reduction in the profile.

**The seed is `-inf`, DEVIATION 1151.** `+0.0` is WRONG, because a row whose
logits are all negative, which is most rows of a trained head after the first
steps, would have its maximum clamped and the loss would be quietly wrong on
every vendor identically. **Bit identity does not catch it.** `-FLT_MAX` (what
MAX's `softmax_kernel:958` seeds with) is correct here only by virtue of a
refusal made elsewhere, and MAX is inconsistent with itself about it
(`_softmax_warp_kernel:1063` seeds `min_or_neg_inf` in the same file); under
`identical_fmax` the total-order key of `-inf` is `0x007FFFFF`, strictly below
every finite value, so it is a true identity element.

**`pinned_block_max` may NOT be used.** Its fold is a plain `other > red[tid]`
compare (`core/pinned_reduce.mojo:159-190`), the spelling row 13 closed
everywhere else. **The clean fix is to give that fold `identical_fmax` as its
combine step**, which is `portable_fmaxf`'s own suggestion and that file
owner's call; until then `loss.mojo` carries a local `pinned_block_fmax`,
DEVIATION 1165.

### 5.2 What the maximum is taken OVER

Every element of the row, all `V` of them, including the target's own logit.
Not the top-k, not a sampled subset, not an unmasked prefix. Stated because a
fast implementation maintaining a running maximum over a candidate set is a
different reduction over a different multiset. Sabotage `L_MAX_TOPK_PREFIX`.

### 5.3 L4, the denominator, is gemm v1's LEAF-AND-BALANCED-TREE, and this is DEVIATION 1152

    denom(row) = identical_gemm( E[N x V], ones[V x 1], OP_NN, N, 1, V )

with `ones` holding exactly `Float32(1.0)`; on the host,
`gemm_oracle(row_exps, ones, OP_NN, 1, 1, V)`.

**There is no new arithmetic here and no new fold to certify.** The leaf loop
computes `acc = ftz(fma(ftz(e_p), ftz(1.0), acc))`, and `fma(e, 1, acc)` is ONE
rounding of `e + acc` because `e * 1.0` is exact, so the ones vector turns the
reduction into the contract's own ascending flushed chain inside a leaf and its
own balanced tree across leaves. That is DEVIATION 851's argument.

**Why this DEPARTS from transformer 5.3**, which pinned a serial chain. Its
load-bearing reason was that `P` is a pure function of `k`, so a row folded over
257 keys and the same row folded over 5 have different trees and different bits.
**That hazard does not exist on the vocabulary axis, because `V` never
changes.** What is left is the price of a serial chain at `V = 128256`: a
128256-term dependent chain per row that cannot be split across threads at all,
against 1002 independent leaves and 10 levels; worse conditioning, since the
terms are `exp(shift)` in `[0, 1]` with exactly one equal to `1.0` and a serial
ascending walk in vocabulary-index order has nothing to do with their
magnitudes, which is gemm 7.2.1's own argument; and a second fold shape that
L12 could not share.

The consequences are stated, not buried. **A per-row vocabulary is refused**,
2.1, and that is the only cost. **`E`, the `[N, V]` buffer of exponentials, is
MATERIALIZED**, 2.1 GB per buffer at `N = 4096` and `V = 128256`, and this
profile wants three or four; 7.1 is the escape and it is a real one. **This is a
reference-quality profile and it is slow by construction**, and a lane whose
only instrument is the per-stage card should not begin by fusing the stages
away.

**ONE ones vector serves all three folds and its OPERAND SIDE differs between
them.** The vocabulary folds put it on the RIGHT (`OP_NN` at `(N, 1, V)`, the
`k x n` operand) and the batch fold on the LEFT (`OP_NN` at `(1, 1, N)`).
**Operand side is part of the routing and not a convention that can be
assumed**, which is `gemm_backward.mojo`'s `BWD_DC_LEFT` / `BWD_DC_RIGHT`
lesson. The caller allocates `max(V, N)` entries once per shape.

**`pinned_block_sum` is REFUSED** because it pairs by STRIDE, which gemm 7.2
clause 1 names a DIFFERENT ANSWER from adjacent pairing, and folds only within
one block. Sabotage `L_DENOM_HALVING_TREE`.

**At `V <= 128` the leaf count is 1 and the tree performs NO addition, so both
fold sabotages are bit-inert. The `V >= 129` shapes are mandatory** and the
check must PREDICT that inert set and assert it as a mask.

### 5.4 L12, the batch reduction, is the SAME call at `k = N`

    total = identical_gemm( ones[1 x N], row[N x 1], OP_NN, 1, 1, N )

SUM and MEAN are the same fold; only the divisor differs. **`N` IS A LAUNCH
SHAPE, and unlike `V` it moves.** So this fold has exactly the property the
transformer refused on its key axis, and this lane accepts it knowingly
because a batch reduction's length IS the batch. It is the same finding
`gemm_backward_b_call` recorded about the weight gradient, and the consequence
is the same: **the microbatch schedule is part of a training run's numerical
specification, not an execution detail.** `archive/plans/gemm/IDENTICAL_BACKWARD_PLAN.md`
2.2 sharpens it with the aligned-split rule and that sharpening applies here
unchanged. Sabotage `L_REDUCE_SERIAL`, inert at every `N <= 128`, so
`N >= 129` is mandatory under the same predicted-mask discipline.

### 5.5 L13, the divide, and the ONE producer of the divisor

`loss = identical_div(total, divisor)`, one division, never a reciprocal
multiplied in; `e * (1/d)` rounds twice where `e / d` rounds once. Transformer
5.4's paragraph at the same evidential standing, **UNVERIFIED AGAINST THE
REFERENCE.** Sabotage `L_MEAN_RECIPROCAL_MUL`.

| reduction | divisor |
|---|---|
| `NONE` | not formed; there is no `ce.loss` stage |
| `SUM`, no `num_items_in_batch` | `Float32(1.0)`, and the division IS SPELLED |
| `SUM` with `num_items_in_batch` | `Float32(num_items)` |
| `MEAN` | `Float32(count)`, the number of rows whose target is not `ignore_index` |

The SUM divide by exactly `1.0` is bitwise inert at every value the
accumulator can hold, and is spelled anyway so there is ONE code path and no
branch whose two arms have to be shown to agree.

**`count` IS AN INTEGER**, counted with integer arithmetic, so it is exact,
order-free and vendor-free, and it is the reason section 11 refuses per-class
weights: a weighted mean's denominator is a SUM OF FLOATS and would need a
fold, a clause, a fixture and a sabotage of its own.

**`count == 0` is REFUSED BY NAME.** Torch returns NaN there, and a computed
NaN's payload is vendor-shaped (row 39 measured `0x7fc00000` on Apple,
`0x7fffffff` on NVIDIA, `0xffc00000` on AMD for one IEEE answer). A KNOWING
DEPARTURE from the reference. DEVIATION 1156.

**The divisor has exactly ONE PRODUCER**, `loss_oracle.mojo::ce_divisor`, a
pure host function of `(reduction, count, num_items)` reading no device, no
buffer and no launch, called by the forward's L13 and the backward's L16. That
is `gemm_backward.mojo`'s two-producers discipline applied to the one quantity
two passes have to agree about. Sabotage `L_GRAD_DIVISOR_IS_N`, which is
**bit-inert when no row is ignored**, so the check must predict and assert
that.

---

## 6. Label smoothing, the second fold over the vocabulary

**UNVERIFIED AGAINST THE REFERENCE.**

    row_i = (1 - eps) * nll_i  +  eps * ( -(sum over v of lp[i, v]) / V )

The second term is the cross-entropy against the uniform distribution, which
is why it is a MEAN over the vocabulary rather than a sum.

### 6.1 The definition and the spelling

**(a) The division by `V` comes FIRST and is a real division.** The
alternative is a host-precomputed `EPS_OVER_V = eps / V` folded into one
product, which saves a division per row and is a different answer, since it
rounds `eps/V` once on the host then rounds one product where the pinned
spelling rounds one quotient then one product. Sabotage
`L_SMOOTH_FOLDED_CONSTANT`. `Float32(V)` is exact for every `V` under the
section-2 ceiling.

**(b) The combine is UNFUSED.** Two rounded products then one add of two
already-rounded values. An `identical_mul_add(EPS, smooth, product1)` is one
rounding where this is three, and it is the natural thing for a kernel to
write. Sabotage `L_SMOOTH_FUSED_COMBINE`.

**(c) `EPS == 0` TAKES A DIFFERENT PATH, DEVIATION 1155, and it is the subtle
one.** At `eps = 0` the smoothing arm is ALMOST bit-inert:
`identical_mul(0.0, smooth)` is `fma(+0, smooth, -0)`, which is `+0.0` for
positive `smooth`, and `ftz(nll + (+0.0))` is `nll` for every value except
`nll = -0.0`, where it gives `+0.0`. **8.1 shows `nll == -0.0` is reachable.**
So spelling the smoothing arm unconditionally would make "turning label
smoothing off" move a bit on exactly one row shape, which is the worst kind of
defect, invisible in every ordinary fixture and real.

    if EPS == 0:   row = nll,  and L9, L10 and L11 are NOT SPELLED
    else:          row = L11

The test is on a Float32 constant known on the host before any launch, so it
is a configuration branch and not a data-dependent one. Sabotage
`L_SMOOTH_ALWAYS_SPELLED`, which must move `ce.row` on a row whose `nll` is
`-0.0` and must NOT move it on any other row. **That is the strongest
reach-per-branch sabotage in this document and it is the one most likely to be
deleted as inert by somebody whose fixture has no `V = 1` row.** The gate must
run BOTH eps settings, carry the `V = 1` row, and assert the predicted inert
masks rather than reporting "the sabotage did not fire".

### 6.2 The three decisions inside it

They are (a), (b) and (c) of 6.1 above.

---
### 6.3 The smoothed target vector, L15, is TWO HOST CONSTANTS

    T_OTHER  = ftz( identical_div( EPS, Float32(V) ) )
    T_TARGET = ftz( ONE_MINUS_EPS + T_OTHER )

computed once on the host and passed to the kernel as two floats. This is the
analytic derivative of 5's definition,
`d/dx_v [ (1-e)*nll + e*smooth ] = w_v - (1-e)*onehot_v - e/V`, spelled as a
target VECTOR rather than three terms so the gradient kernel performs one
subtraction and one division per cell and nothing else.

**At `EPS == 0` these are exactly `1.0` and `+0.0`**, so unlike the forward
the BACKWARD's smoothing arm is bitwise inert at eps zero and needs no branch.
That asymmetry is named because it is a `[[reached-but-inert]]` hazard in the
other direction: a sabotage that deletes a branch the gradient does not have
will move nothing, and that is not evidence about the gradient.

`T_TARGET` and `T_OTHER` are the derivative of the MATHEMATICAL loss, not of
the FLOATING-POINT expression L11 computes. Nobody computes the latter and
this profile does not pretend to.

---

## 7. Row independence, batch composition, and the ignored row

### 7.1 Nothing per-row reads `N`

L1 through L11 and L14 through L16 read `logits[i, :]`, `targets[i]`, `V` and
the profile constants. **Not `N`, not the row chunk, not the launch geometry,
not the block size, not the vendor.** Therefore a row's bits are identical
whether it is computed alone, in a chunk of 256, or in a launch of 4,096, and
that is TRUE BY CONSTRUCTION; the gate exists to catch an execution plan
violating the construction, not to establish it.

**The exception is L12, and only L12**, whose `k` is `N`. 5.4.

**The consequence is the escape hatch for 4.3's memory cost.** A caller may
split `[N, V]` into row chunks of any size, run L1-L11 and L14-L16 per chunk,
and concatenate, and the bits do not move. Only L12 has to see all `N` rows at
once, and it folds `[N]` floats, not `[N, V]`.

**Vocabulary-fold invariance.** `L` and `P` for L4 and L9 come from `V` and
the two profile constants and nothing else, one producer,
`contract_leaf_size(V)`. Sabotage `L_VOCAB_FOLD_READS_LAUNCH`, which reaches
gemm v1's own `SAB_LEAF_READS_LAUNCH` through this lane's routed call and is
the proof the routing lands on the contract's arithmetic rather than on some
other path that happens to agree.

### 7.2 Vocabulary-fold invariance

Covered in 7.1's last paragraph, and its sabotage is
`L_VOCAB_FOLD_READS_LAUNCH`.

---
### 7.3 The ignored row is `+0.0`, STORED, and provably inert

For a row with `targets[i] == ignore_index`, `ce.row[i]` is `+0.0` STORED,
`ce.dlogits[i, v]` is `+0.0` for every `v` STORED, and `count` is not
incremented. **Neither store may be skipped**, which is the gemm contract's
section 8 discipline applied to a row instead of a shape. Sabotage
`L_IGNORED_ROW_SKIPPED`.

**Why `+0.0` and not `-0.0`.** L12's leaf accumulator is seeded `+0.0` and
every term it adds is a row loss, and `acc + (+0.0) == acc` for every `acc`
except `acc = -0.0`.

**DEVIATION 1327, 2026-08-25. The seed forbids reaching `-0.0` by ADDITION and
does NOT forbid `ftz` of a negative subnormal partial sum**, and `ftz` runs at
every seam. Verified by direct computation: from a `+0.0` seed,
`+ 0x80C00000` then `+ 0x00800000` gives `0x80400000`, a negative subnormal,
and `ftz` of that is `0x80000000`; both operands are ordinary normals. The
same false sentence stood in `transformer/IDENTICAL_TRANSFORMER_CONTRACT.md`
7.1 and is corrected there. **Whether the hole is REACHABLE for `ce.row` is a
separate question this lane has not answered**, since it needs a row whose
partial log-prob sum cancels into the subnormal range; 8.2(c) proves every row
loss is non-negative, which makes it hard to reach and does not make it
impossible. **So the inertness of an ignored row is true of every fixture
shipped today and is NOT proven of the contract.**

A `-0.0` ignored row would be invisible in `ce.total` (laundered by the seed)
and visible in `ce.row`. Sabotage `L_IGNORED_ROW_NEG_ZERO`, which **must move
`ce.row` and must NOT move `ce.total`.** A gate that compared only the final
loss would call it inert, and it is not; it is a divergence at the stage that
produced it. That is transformer 5.1's "the card is the only instrument that
can see this clause at all" at a second site.

## 8. NaN, infinity, signed zero, denormals

- **A NaN or infinity in `logits` is REFUSED BY NAME before any recorded
  stage**, because NaN payloads are vendor-shaped and a certified stage may
  not contain one. **The test is BY BITS and never by compares**, because
  Metal flushes compare operands (row 49). It is
  `(bits & 0x7FFFFFFF) >= 0x7F800000`.
- **A target is REFUSED unless it equals `ignore_index` or lies in `[0, V)`**,
  by INTEGER compare. **`count == 0` under MEAN is REFUSED**, 5.5.
  **`num_items <= 0` is REFUSED.**
- `-0.0` is an admitted value everywhere. `ftz` at every seam of section 3,
  flush to signed zero under IDENTICAL and compiled away under FAST. The
  portable transcendentals flush unconditionally, so `identical_exp`,
  `identical_log` and `identical_div` carry the policy in either mode.

### 8.1 Every site a signed zero is reachable, named

**Every site a signed zero is reachable.** `ce.max`, when the row maximum is a
flushed subnormal or an exact `±0.0`; nothing downstream depends on the sign,
so it is laundered downstream and visible on the card. `ce.shift` at the
argmax, exactly `+0.0`, which is what 8.2(b) rests on. `ce.exp`, exactly
`+0.0` for every `shift < -87.33655` (`portable_expf`'s own underflow edge).
`ce.logp_target`, `+0.0` when the target IS the argmax and `denom` is exactly
`1.0`. **`ce.nll`, `-0.0` by `neg_by_bits(+0.0)`, reachable at `V = 1` and at
any `V` where every non-target exponential underflows**, which is what 6.1(c)
and `L_NEG_VIA_ZERO_SUB` turn on. `ce.row` for an ignored row, `+0.0` by 7.3.
`ce.total`, `+0.0` when every row is ignored or every row loss is `±0.0`, and
see DEVIATION 1327. `ce.weights`, `+0.0` for `denom > 0`. `ce.dlogits`, `+0.0`
when `w == t` exactly, by IEEE's `x - x = +0` in round-to-nearest.

### 8.2 Theorems that remove a whole class of hazards

**(a) The shift is provably non-positive, so `exp` cannot overflow.** `m` is
the total-order maximum of the flushed row, so `ftz(x_v) - m <= +0.0` for
every `v`, so `e[v]` is in `[+0.0, 1.0]` and `identical_exp` never takes its
`x > 88.722835` overflow branch. **No forward stage can be `+inf` through the
exponential.**

**(b) The denominator is provably in `[1.0, Float32(V)]`.** The argmax
contributes `identical_exp(+0.0)`, which is exactly `1.0`, every other term is
non-negative, and the leaf chain seeded `+0.0` cannot lose it; the upper bound
is `V` terms each at most `1.0`, below `2^17` at the shipped `V`. **So
`identical_log`'s argument is never zero, never negative, never subnormal and
never infinite**, and its result is in `[+0.0, 11.7621]`. Every one of
`portable_logf`'s special-value branches is unreachable from this profile,
which is worth knowing and is NOT an argument for deleting them.

**(c) The row loss is provably non-negative, so the batch fold cannot
cancel.** `nll = logdenom - shift[y]` with `logdenom >= +0.0` and
`shift[y] <= +0.0`, so `nll >= +0.0` or `-0.0`. With smoothing, `smooth` is a
non-negative average of non-negative quantities and both coefficients are
non-negative. **Therefore `ce.total` can never form `inf - inf` and can never
produce a NaN.**

**The one stated gap, DEVIATION 1161.** `shift[y]` can be `-inf` from FINITE
logits, since `x_y = -3.4e38` against `m = +3.4e38` overflows the subtraction,
and then `nll` is `+inf` and `ce.loss` is `+inf`. That is deterministic, the
same on every vendor, and admitted. What is NOT reachable is a NaN, by (c).

---

## 9. The stages, in card order

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
    ce.logp_sum       [N]      f32   L9    SMOOTHING ONLY
    ce.smooth         [N]      f32   L10   SMOOTHING ONLY
    ce.row            [N]      f32   L11
    ce.count          [1]      i32   the integer, exact
    ce.divisor        [1]      f32   ce_divisor's one output
    ce.total          [1]      f32   L12   gemm v1 OP_NN (1, 1, N)
    ce.loss           [1]      f32   L13   SUM and MEAN only
    ce.target_vec     [2]      f32   L15   T_TARGET then T_OTHER
    ce.weights        [N, V]   f32   L14   BACKWARD
    ce.dlogits        [N, V]   f32   L16   BACKWARD

Twenty stages, seventeen of which exist on every call, which is why the
shipped card carries 17 records.

`ce.divisor` and `ce.target_vec` are HOST constants recorded for the reason
the transformer records `rope.inv_freq`: a constant computed with the wrong
spelling is a silent divergence no activation stage localizes, and
`ce.divisor` is read by two different passes.

**`ce.shift`, `ce.exp` and `ce.weights` are THREE SEPARATE BUFFERS in v1.** An
implementation may compute `w` in place over `e` and save `N * V` floats; that
is an EXECUTION PLAN and it is admitted, but the card then cannot record both
stages, so a run that uses it produces a SHORTER card and may not be diffed
against a full one. **"The fused arm agrees with the fused arm" is the shape
of a gate that proves nothing**, so a run taking the in-place arm must be
gated against the three-buffer arm at a small shape first.

---

## 10. What "identical" is gated to mean

(a) device card equals host oracle card, BITWISE, at every stage and every
shape. (b) the same bits on every one of 8 repeated launches. (c) BATCH
COMPOSITION invariance, a row's bits identical whether computed alone or in a
chunk with 1, 2 or 255 others, with a negative control showing two DIFFERENT
rows differing. (d) VOCABULARY FOLD invariance, `ce.denom` identical under at
least three unrelated GEMM execution plans, reaching
`identical_gemm_with_plan`'s named plans through this lane's routing.
(e) GRADIENT CORRECTNESS, bitwise, against an independently written analytic
reference on an EXACTLY-REPRESENTABLE fixture, section 12. (f) the row-39 audit
of 6.3. (g) every clause falsifiable by a NAMED sabotage that fails a gate,
**with its predicted INERT set asserted as a mask**.

`training/checks/loss_check.mojo` is the gate file and it exists and has run;
the STATUS block says which clauses on which columns. FAST arms of (a) are
RECORDED, not asserted, where they are vendor-shaped. **Clause (e) is asserted
in BOTH modes**, because on an exactly representable fixture a contracted
multiply-add and an uncontracted one produce the same bits, so a FAST failure
there is a real routing defect.

### 10.1 The sabotage set, one per contested decision

| sabotage | first stage it must move | must be INERT on | falsifies |
|---|---|---|---|
| `L_MAX_PLAIN_COMPARE` | `ce.max` on a row planting `-0.0` beside `+0.0` | every ordinary row | 5.1, the order-free maximum |
| `L_MAX_SEED_ZERO` | `ce.max` on an all-negative row | any row with a positive logit | 5.1, the `-inf` seed |
| `L_MAX_TOPK_PREFIX` | `ce.max` on a row whose maximum is in the tail | a row whose maximum is at index 0 | 5.2 |
| `L_DENOM_SERIAL_CHAIN` | `ce.denom` at `V >= 129` | every `V <= 128` | 5.3, the leaf-and-tree fold |
| `L_DENOM_HALVING_TREE` | `ce.denom` at `V >= 129` | every `V <= 128` | 5.3, the stride pairing |
| `L_DENOM_PAD_PLUS_ZERO` | `ce.denom` at `V = 300` and `V = 128256` | `V` whose every level width is even | the carry, gemm F7 through the routing |
| `L_VOCAB_FOLD_READS_LAUNCH` | `ce.denom` | nothing | 7.1, `(L, P)` from `V` alone |
| `L_EXP_STDLIB` | `ce.exp` | nothing under IDENTICAL | row 12's exp |
| `L_LOG_STDLIB` | `ce.logdenom` | nothing under IDENTICAL | row 12's log |
| `L_NLL_VIA_ADDBACK` | `ce.logp_target` on a large-common-offset row | a centered row | 4.2(a) |
| `L_NLL_VIA_LOG_W` | `ce.logp_target` where `e[y]` underflows | a row where `e[y]` is normal | 4.2(b) |
| `L_NLL_VIA_MAX_LOGSOFTMAX` | `ce.logp_target` | nothing | 4.2(c) |
| `L_NEG_VIA_ZERO_SUB` | `ce.nll` at `V = 1` | every row whose `lp_y` is not `+0.0` | 4.3 |
| `L_SMOOTH_FOLDED_CONSTANT` | `ce.smooth` at `EPS = 0.1` | `EPS = 0` | 6.1(a) |
| `L_SMOOTH_FUSED_COMBINE` | `ce.row` at `EPS = 0.1` | `EPS = 0` | 6.1(b) |
| `L_SMOOTH_ALWAYS_SPELLED` | `ce.row` at `EPS = 0` on a row whose `nll` is `-0.0` | every other row, every `EPS != 0` | 6.1(c) |
| `L_REDUCE_SERIAL` | `ce.total` at `N >= 129` | every `N <= 128` | 5.4 |
| `L_MEAN_RECIPROCAL_MUL` | `ce.loss` | `divisor` an exact power of two | 5.5 |
| `L_IGNORED_ROW_NEG_ZERO` | `ce.row` | **`ce.total`, which it must NOT move** | 7.3 |
| `L_IGNORED_ROW_SKIPPED` | `ce.row` (unwritten memory) | nothing | 7.3, the store is required |
| `L_W_VIA_EXP_LOGP` | `ce.weights` | nothing | L14's division |
| `L_GRAD_SIGN` | `ce.dlogits` | a row where `w == t` at every `v` | L16, `w - t` |
| `L_GRAD_TARGET_OFF_BY_ONE` | `ce.dlogits` | a uniform row at `V = 1` | the target column |
| `L_GRAD_DIVISOR_IS_N` | `ce.dlogits` when a row is ignored | a fixture with no ignored row | 5.5, the one producer |
| `L_GRAD_RECIPROCAL_MUL` | `ce.dlogits` | `divisor` an exact power of two | L16's division |

Each must move the stage its OWN clause writes and no earlier one.

**Three pass by construction on the obvious fixture and are the ones most
likely to be deleted as broken arms.** `L_GRAD_DIVISOR_IS_N` needs an ignored
row, `L_SMOOTH_ALWAYS_SPELLED` needs a `-0.0` row loss, and
`L_IGNORED_ROW_NEG_ZERO` needs the CARD rather than the loss.

---

## 11. Not claimed

- **One loss, not a training step.** No optimizer, no weight update, no
  accumulation across steps, no RNG, no dropout, no clipping, no mixed
  precision, no loss scaling, no distributed all-reduce.
- **Not the gradient with respect to anything but the logits.**
- **Not `reduction = NONE`'s backward.** A per-row upstream vector adds a
  second product per cell whose placement, before or after L16's division, is
  a real decision with two answers, and this lane has no caller for it.
- **Not a per-class `weight` vector**, which would make `count` a SUM OF
  FLOATS and therefore a fourth fold with its own clause, fixtures and
  sabotages, and would make `ce.divisor` data-dependent.
- **Not a per-row or masked vocabulary**, no hierarchical, sampled or adaptive
  softmax, no candidate sets, no speculative-decoding restricted head. 2.1 is
  the reason and it is a cost of DEVIATION 1152.
- **Not a fused loss.** No chunked kernel that never materializes the logits,
  which is what a production Llama head actually wants. That kernel splits the
  vocabulary axis and rescales, which is the online-softmax argument on a new
  axis. A v2 at the earliest.
- **Not soft targets**, no distillation loss, no KL divergence.
- **Not BF16, FP16, FP8, TF32 or any quantization.** Not FP64 on device.
- **Not agreement with PyTorch, HuggingFace or MAX.** The fold order,
  transcendentals, division and log-sum-exp spelling are OURS. A reader who
  takes "identical" to mean "equal to torch" has taken more than is offered.
- **Two KNOWING DEPARTURES from the reference's behavior**, both because a
  vendor-shaped NaN may not enter a certified stage. Torch returns NaN for a
  MEAN over zero unignored rows; we refuse. Torch propagates a NaN logit
  through to a NaN loss; we refuse the input.
- **Not the derivative of the floating-point forward.** L15 and L16 are the
  analytic gradient of the real-valued loss.
- **No performance number.** None taken, and none quoted until it is,
  alternated inside one thermal window.
- **Nothing cross-vendor.** Two columns is not three, and the missing one is
  NVIDIA, which has broken every other lane.

---

## 12. How the GRADIENT is gated BITWISE, not approximately

A loss whose gradient is only checked against finite differences is checked
against a TOLERANCE, and this lane deals in BITS. **Finite differences are
DEMOTED to a reported diagnostic and are NOT a gate, DEVIATION 1163**, the same
demotion `archive/plans/gemm/IDENTICAL_BACKWARD_PLAN.md` made for its G2: on a fixture chosen
so the derivative is exactly representable, the derivative can be WRITTEN DOWN,
and a step size is a tolerance wearing a different hat.

**The UNIFORM family.** A row whose `V` logits are all `c`, with `V` and
`divisor` powers of two, gives `m = c`, `shift[v] = +0.0`, `e[v] = 1.0`,
`denom = Float32(V)` **in EVERY fold order**, `w[v] = 1/V`, and at `EPS = 0`
`dl[y] = (1/V - 1)/divisor` and `dl[v != y] = (1/V)/divisor`, exact dyadic
rationals; at `V = 4`, `divisor = 2` those are `-0.375` and `+0.125`.

**The SATURATING family.** A row with `c` copies of `a` and `V - c` copies of
`a - 200.0`, `c` a power of two, gives `e` exactly `1.0` at the high cells and
exactly `+0.0` at the rest (`portable_expf` returns `+0.0` below `-87.33655`),
`denom = Float32(c)`, and every cell writable. It adds a genuine argmax
structure, an exercised underflow edge, exact `+0.0` weights whose signs must
be `+`, and a case where the TARGET is a low cell.

**What they gate**: the formula, the sign, the target COLUMN, the divisor's
identity, the target-vector constants and the routing of every buffer.
**What they CANNOT gate, and saying so is the point**: every quantity is exact,
so they separate NO spelling from any other. MEASURED, in the run's own words,
the reciprocal-multiply agrees with the true divide on 7 of 7 exact-family
values at a power-of-two divisor and DISAGREES on 1 of 7 at divisor 3.0, so
**the exact family gates CORRECTNESS and separates NO spelling**, and clause
(a) is what covers the spelling.

**Three arms.** A1 EXACT-ANALYTIC, both sides against a hand-written closed
form on both families, by bits, in BOTH modes, gating correctness. A2
ORACLE-BITWISE against `loss_oracle.mojo` at every stage on the ordinary
fixtures, IDENTICAL asserted and FAST recorded, gating the spelling. A3 FLOAT64
TOLERANCE, **reported and never asserted**, existing so a systematic error A1's
algebra and A2's oracle share would be visible; MEASURED on both columns,
`base_n4_v8`'s FP32 per-row loss differs from the Float64 reference by at most
`7.07445340905206e-08` at row 3, and it cannot see an error this repository's
own Float64 code shares.

**Four vacuity guards, ENFORCED rather than argued**, raising rather than
adjusting the fixture. (1) `V != N` and `V != divisor` in every A1 case, because
a square-ish fixture lets a transposed index produce a plausible matrix of the
right size. (2) `V`, `c` and `divisor` powers of two, because an unchecked
exactness argument is how a gate comes to assert what the code does rather than
what it should do. (3) At least one A1 case with an ignored row and one
without, or `L_GRAD_DIVISOR_IS_N` cannot fire. (4) A
`check_the_exact_fixture_is_vacuous` arm that re-spells the reciprocal-multiply
gradient LOCALLY and DEMONSTRATES it agrees bit for bit, with a
non-power-of-two control showing it disagreeing. **Without that arm a reader is
entitled to think A1 gates the spelling, and it does not.**

---

## 13. Deviations, where the code goes, and what is OWED

DEVIATIONS 1150 through 1169. No number outside that range is claimed by this
lane; numbers cited from elsewhere (204, 258, 504, 522, 528, 720, 740-746,
800-819, 820-826, 850-852, 1051-1060, 1327, 1460, 1495) belong to other lanes.

| # | what |
|---|---|
| 1150 | the composition `loss.ce.fp32.v1` and its IDENTITY_PATHS row |
| 1151 | the row maximum through `identical_fmax`, fold shape FREE, seeded `-inf`, `pinned_block_max` refused |
| 1152 | **all three folds routed through gemm v1 `OP_NN` against a ones vector**, the departure from transformer 5.3, and the fixed-`V` argument that licenses it |
| 1153 | the log-sum-exp as `shift[y] - logdenom`, and the three refused alternatives |
| 1154 | the negation spelled BY BITS |
| 1155 | label smoothing, divide by `V` first, the unfused combine, and the `EPS == 0` BRANCH |
| 1156 | the MEAN divisor as an exact integer count, `count == 0` REFUSED, and the divisor's single producer shared by forward and backward |
| 1157 | the softmax weight `identical_div(e, denom)`, reciprocal refused, transformer 806 at a second site |
| 1158 | the gradient cell and the smoothed target as TWO host constants |
| 1159 | the ignored row, `+0.0` STORED in both `ce.row` and `ce.dlogits` |
| 1160 | the input refusals, nonfinite BY BITS, target range and counts by integer compare |
| 1161 | the three theorems of 6.4 and the one gap they leave |
| 1162 | the row-independence clause, `N` is a launch shape for everything except L12 |
| 1163 | the EXACT-ANALYTIC gradient gate, the two exact families, and the DEMOTION of finite differences |
| 1164 | the local `refuse_nonfinite` and its debt to `checks/numerics.mojo` |
| 1165 | the local `pinned_block_fmax` and its debt to `core/pinned_reduce.mojo` |
| 1166 | the stage list and its card tags |
| 1167 | the sabotage set and its predicted INERT masks |
| 1168 | the block size resolved from `checks/kernel_matrix.mojo` rather than a literal |
| 1169 | reserved |

**OWED.**

1. **An NVIDIA leg.** The largest item. Two columns is not a cross-vendor
   claim and the missing column is the one that has broken every other lane.
2. **Clauses (b), (c), (d) and (f) on both existing columns.** Without (d)
   the four fold arms reached through `gemm_identical.mojo` are gated only by
   clause (a), which sees ONE execution plan. Without (f) the device-side
   refusal has not been looked at on either 2026-08-28 leg, and (f) is the
   clause that MEASURED DEVIATION 1460.
3. **A `pixi.toml` task**, `check-loss`, plus the sabotage arms as `-D` builds.
4. **`training/corpus/` does not exist**, so clause 12's A3 arm has no
   independent Float64 reference and **three of the twenty stages are the same
   host code on both sides of every comparison**. `mamba/corpus/` and
   `tools/mamba_corpus_check.py` are the pattern.
5. **`checks/numerics.mojo` should carry `refuse_nonfinite`** (a THIRD copy
   lives here, DEVIATION 1164, with `mamba_oracle.mojo:57` and
   `optimizer_oracle.mojo:162`; three copies are three chances to disagree
   about what a NaN is) **and `neg_by_bits`** (one XOR, DEVIATION 1154,
   belonging beside `identical_mul`). Both are that file owner's edit.
6. **`core/pinned_reduce.mojo::pinned_block_max` should take `identical_fmax`
   as its combine step.** That is `portable_fmaxf`'s own docstring, not this
   lane's opinion. It would delete `loss.mojo`'s local `pinned_block_fmax` and
   would also fix the transformer lane's S14. **Two lanes want this one edit.**
7. **`IDENTITY_PATHS.md` needs a row for this profile**, DEVIATION 1150,
   recording exactly what the two columns closed and what they did not.
8. **A PyTorch checkout** for the three UNVERIFIED decisions of section 1, L6
   (3.2), the divisions (5.5) and the smoothing combine (section 5).
9. **`bench/` has no loss shape and `SUPPORT_MATRIX.md` has no loss row.** No
   performance number exists and section 10 forbids quoting one.
