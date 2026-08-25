# Transformer block reference corpus

An independent reference corpus for the IDENTICAL FP32 transformer block lane
(`transformer/IDENTICAL_TRANSFORMER_CONTRACT.md`, profile
`mojolearn.identical.transformer.fp32.v1`). The lane's expected values must not
be solely our own tally, so this directory carries small hashed Llama-shaped
decoder-block cases whose per-stage reference outputs are computed by somebody
else's algorithm. The block order and every step is HuggingFace
`src/transformers/models/llama/modeling_llama.py`, re-implemented in pure torch
in `gen_corpus.py` with each step cited by symbol and line. The `transformers`
package is not imported.

**Read this before trusting anything here.** `ref64/` is a TOLERANCE reference,
not a bitwise certificate. The lane's bitwise oracle is its own pinned host
oracle; this corpus exists so that the lane's expected values have an outside
anchor and so that a WRONG ALGORITHM is detectable at all. `ref32/` is a plain
float32 CPU run of the same stages. It is informative only, it is never a
target and never a gate, and its whole job is to calibrate how tight a
tolerance is honest.

**Nothing in this directory has been executed by its author.** The generator,
the checker and every number quoted in a docstring are CONSTRUCTION and
PREDICTION. The first person to run `--verify` learns whether they are right.
Where this README states a count or a measurement without saying it was
measured, read it as an estimate the generator will confirm or refute.

## Why this lane exists

Every other gate in the transformer lane compares our code against our own
code. A host oracle and a device kernel agreeing proves that they agree; it
does not prove either is a transformer. The sibling `mamba/corpus/` closed the
same hole for the Mamba-1 block and immediately found a real disagreement, a
stage-naming mismatch about whether `dt_proj.out` includes its bias, which no
amount of internal agreement would ever have surfaced. This corpus is the only
instrument in this lane that can say we ported the wrong algorithm.

## Versions and commits

| component | version / commit |
|---|---|
| Python | 3.14 (the repository's `skgpu` pixi environment) |
| torch (CPU) | 2.13.0 |
| numpy | 2.5.2 |
| huggingface/transformers (READ, not imported) | `d56c55bf564ddb176759eb6ec199442682564916` |
| einops | not required; unlike the mamba corpus this generator needs no verbatim upstream copy |

The transformers checkout lives at
`/Users/andrewhendel/CascadeProjects/upstream/transformers/`, outside this
repository, and was read at the commit above. There is no `transformers`
package installed anywhere on this machine and none is needed. There is no
PyTorch checkout, so ATen's own softmax kernel could not be read, which is the
same gap contract section 5.4 records.

## How to run it

The generator needs torch, which lives in one pixi environment only.

```sh
/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/skgpu/bin/python \
    /Users/andrewhendel/CascadeProjects/mojolearn/transformer/corpus/gen_corpus.py --verify
```

`--verify` writes the corpus and then runs nine controls, each of which prints
what it proves. It exits nonzero if determinism fails, if a decode negative
control does not move, if an adversarial case does not reach its intent, or if
any float64 reference holds a nonfinite value.

```sh
/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/skgpu/bin/python \
    /Users/andrewhendel/CascadeProjects/mojolearn/transformer/corpus/gen_corpus.py --self-test
```

`--self-test` reads the committed corpus and prints the per-case per-stage
tolerance calibration. It needs no recomputation and no GPU.

The checker needs numpy alone, so it runs in any of the repository's
environments.

```sh
python tools/transformer_corpus_check.py transformer/corpus/<case> <dump_dir>
python tools/transformer_corpus_check.py transformer/corpus/<case> --self-test
python tools/transformer_corpus_check.py transformer/corpus/<case> --negative-control
```

`--negative-control` runs today, before the lane has written a single stage
dump, because its stand-in dump is the corpus's own `ref32`. It corrupts that
in named ways and demands each corruption be caught, and it prints every
corruption it does NOT catch as a thing the case cannot certify.

## The precision rule (DEVIATION 1042)

This is the single most important decision in the corpus and it is stated here
so that a reader can disagree with it in one place.

The corpus computes in float64 EXCEPT at five places where the profile pins an
FP32 value as a CONSTANT OF THE ARITHMETIC, and there it uses the FP32 bits
widened to float64.

1. The rotary inverse frequencies used to build the angle. Contract seam S6,
   `LlamaRotaryEmbedding.compute_default_rope_parameters` line 108, computed in
   FP32 by the reference.
2. The rotary ANGLE `position * inv_freq`. Contract seam S7,
   `LlamaRotaryEmbedding.forward` lines 114 to 122, an FP32 matmul with
   autocast explicitly disabled.
3. The attention scale `head_dim ** -0.5`. Contract section 3.
4. The RMSNorm epsilon `1e-6`. Contract section 3.
5. The additive mask fill `-3.4028234663852886e+38`. Contract section 3. Note
   that this is `finfo(float32).min` and NOT `finfo(dtype).min`, so the float64
   reference uses the float32 constant on purpose, because the profile pins the
   NUMBER and not the expression.

Items 1 and 2 are not stylistic. RoPE's angle at absolute position p is
`p * inv_freq`, so a one-ulp difference in `inv_freq` becomes a p-ulp
difference in the angle, and `d(cos)/d(angle) = 1`. At the profile's ceiling
(`p < 8192`, contract DEVIATION 812) that is an amplification of about 8000x,
which puts the disagreement between a float64 angle table and the reference's
FP32 angle table orders of magnitude above float32 epsilon. A corpus built on
the float64 angle would fail every implementation on earth including torch's
own FP32, and it would fail hardest at exactly the case worth the most.
`--verify` control 5 measures the amplification per case and prints it beside
float32 epsilon, so the argument is on the record as a number.

The consequence for a checker is stated as its own rule. `rope.cos` and
`rope.sin` are anchored to `ref32/rope.inv_freq.f32`, the reference's own FP32
spelling. `tools/transformer_corpus_check.py` verifies a dump's
`rope.inv_freq` against that anchor BITWISE before it looks at `rope.cos`, and
reports the cos and sin verdicts as CONDITIONAL when the anchor does not
match, because at the far position a one-ulp anchor difference dwarfs any
tolerance worth setting. A conditional verdict is a finding about `inv_freq`
and `portable_powf` (contract S6, DEVIATION 809), not about `identical_cos`.

Two other places where a reference's own precision could have been the
definition and is not. The SiLU spelling is ATen's one-division form
`x / (1 + exp(-x))`, which contract S20 pins; the two-rounding
`x * sigmoid(x)` differs by about 1e-16 relative in float64, which this corpus
cannot see and the lane's `S20_SILU_MUL_SIGMOID` sabotage must carry. The
softmax division is one division per weight rather than a reciprocal multiply
for the same reason and with the same limitation.

## The hash spec (`mojolearn.transformer.corpus.hash.v1`)

Every tensor element is a HASHED value, never a library RNG draw, so a Mojo
fixture can regenerate the identical float32 bits from this section alone and
the checker can compare inputs BITWISE before it looks at any output. It is the
same splitmix64 construction as `mojolearn.mamba.corpus.hash.v1` with a
different name, a different seed base and a different tensor-id table, so a
Mojo fixture that already implements the mamba spec implements this one by
changing three constants.

```c
uint64 splitmix64(uint64 z) {
    z += 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

/* per tensor: seed is the case seed, tid the tensor id from the table below */
uint64 key = splitmix64(seed ^ (tid << 32));

/* element at row-major flat index i (all arithmetic wraps mod 2^64) */
uint64 h   = splitmix64(key + (uint64)i);
double f   = (double)(h >> 40) * 0x1p-24;      /* top 24 bits, in [0, 1) */
double v64 = lo + (hi - lo) * f;               /* exact in float64, see below */
float  v   = (float)v64;                       /* ONE round, to nearest even */
```

Every `(lo, hi)` pair used in this corpus is a dyadic rational, so
`(hi - lo) * f` and the sum with `lo` are exact in float64 and the only
rounding is the final cast to float32. The generator asserts that exactness
against exact rationals for every element of every tensor; a failure there
means this spec is not implementable in Mojo as written, which would be a
defect in the spec and not in torch.

Case seeds follow `seed_k = 0x58666D72436F7270 + 0x1000 * k` with `k` the case
index in the top-level `manifest.json`. The two `comp_row*` cases and the four
`decode_*_step*` subcases reuse their parent's seed, and their manifests say
so.

| tensor | id | shape | default `[lo, hi]` |
|---|---|---|---|
| `x` | 1 | `[B, L, d_model]` | `[-2, 2]` |
| `norm1.weight` | 2 | `[d_model]` | `[0.5, 1.5]` |
| `norm2.weight` | 3 | `[d_model]` | `[0.5, 1.5]` |
| `q_proj.weight` | 4 | `[n_heads*head_dim, d_model]` | `[-0.5, 0.5]` |
| `k_proj.weight` | 5 | `[n_kv_heads*head_dim, d_model]` | `[-0.5, 0.5]` |
| `v_proj.weight` | 6 | `[n_kv_heads*head_dim, d_model]` | `[-0.5, 0.5]` |
| `o_proj.weight` | 7 | `[d_model, n_heads*head_dim]` | `[-s, s]`, `s = 0.5 / 2^ceil(log2(fan_in)/2)`, fan_in `n_heads*head_dim` |
| `gate_proj.weight` | 8 | `[intermediate_size, d_model]` | `[-0.25, 0.25]` |
| `up_proj.weight` | 9 | `[intermediate_size, d_model]` | `[-0.25, 0.25]` |
| `down_proj.weight` | 10 | `[d_model, intermediate_size]` | `[-s, s]`, fan_in `intermediate_size` |

There are no biases anywhere, because contract section 2 refuses a nonzero bias
rather than specifying where it would round.

Two cases plant values AFTER hashing, and both plantings are exact so the bits
stay reproducible.

- `adv_signed_zeros_b2_l4_d32_kv2` applies
  `x[b,t,:] = +0.0` for `t % 4 == 0`, `x[b,t,:] = -0.0` for `t % 4 == 2`, and
  of the remaining elements those with row-major flat index `i % 7 == 3`
  become `-0.0`. This is the mamba corpus's rule unchanged.
- `adv_softmax_saturation_b1_l8_d32_kv1` applies `x[b,t,:] *= 2^(t-3)`, an
  exact power of two, which introduces no rounding.

The `overrides` column of a case's `manifest.json` is authoritative for every
tensor whose range an adversarial case changed.

## The block, stage by stage

Thirty stages, in the card order of contract section 9. `M = B * L`,
token-major. `S` is the kv length after the append. Every stage's definition,
its shape and its calibrated tolerance are in the case's `manifest.json`; the
top-level `manifest.json` carries the definitions once. Two conventions are
worth stating here because a driver will get them wrong otherwise.

- **Token-major everywhere except attention and the cache.** `input.x`,
  `norm1.out`, every projection output, `q_rope.out`, `k_rope.out`, `attn.ctx`,
  `silu.out`, `mlp.gated` and both residuals are `[M, F]`. `norm1.sumsq` and
  `norm2.sumsq` are `[M]`. Only `kv.k_cache`, `kv.v_cache`, `attn.scores`,
  `attn.masked`, `attn.exp`, `attn.weights`, `attn.max` and `attn.denom` lead
  with `B`.
- **`norm1.sumsq` is the SUM and not the mean.** The reference spells
  `hidden_states.pow(2).mean(-1)` at `LlamaRMSNorm.forward` line 65, which is
  this divided by `d_model`. The card's stage is the sum, contract seam S1. A
  dump carrying the mean differs by exactly a factor `d_model`, and the checker
  NAMES that rather than calling it an arithmetic defect. This is deliberately
  the same class of trap the mamba corpus hit on `dt_proj.out`, planted here so
  the checker's explainer has something known to find.

`rope.cos` and `rope.sin` are the HALF tables, `[n_positions, head_dim/2]`,
before the `cat(freqs, freqs)` of `LlamaRotaryEmbedding.forward` line 123,
which is a copy and not a seam. The card's shape is `[P_max, head_dim/2]`; this
corpus writes only the positions a case uses (DEVIATION 1041), because a full
table at the far-position case would be 8192 rows of decoration. The checker
accepts a dump carrying a full table by selecting the case's rows by absolute
position, and it says that it did so.

## The stage-dump convention

The lane's driver writes, for one case, a directory of raw little-endian
float32 files named exactly `<stage>.f32`, row-major, in the shapes the case's
`manifest.json` records. It SHOULD also write the input tensors under their own
names (`x.f32`, `q_proj.weight.f32`, and so on) and `positions.i32` as
little-endian int32, because the checker verifies those BITWISE before it looks
at any output and refuses to attribute an output difference to arithmetic
without them. Stages the driver does not dump are skipped and listed, never
failed.

## Cases

Thirteen top-level cases plus four decode-step subcases. Every case's manifest
carries a `certifies` field, a `cannot_certify` field, and a MEASURED
`sensitivity` table that records which of seventeen deliberate wrong answers
the case can actually see. The `cannot_certify` prose is the author's claim;
the sensitivity table is the measurement, and where they disagree the
measurement wins.

| case | B | L | d_model | heads / kv | head_dim | inter | pos0 | what it is for |
|---|---|---|---|---|---|---|---|---|
| `base_b1_l1_d32_kv2` | 1 | 1 | 32 | 2 / 2 | 16 | 64 | 0 | smallest shape; also the decode step shape with an empty cache |
| `base_b2_l4_d32_kv2` | 2 | 4 | 32 | 2 / 2 | 16 | 64 | 0 | a real causal triangle, `n_rep == 1`, parent of the composition rows |
| `comp_row0_b1_l4_d32_kv2` | 1 | 4 | 32 | 2 / 2 | 16 | 64 | 0 | batch composition, row 0 of the parent alone |
| `comp_row1_b1_l4_d32_kv2` | 1 | 4 | 32 | 2 / 2 | 16 | 64 | 0 | batch composition, row 1 |
| `base_b1_l4_d32_kv1` | 1 | 4 | 32 | 2 / 1 | 16 | 64 | 0 | GQA at `n_rep == 2`, so `repeat_kv` actually copies |
| `base_b2_l16_d32_kv1_i300` | 2 | 16 | 32 | 2 / 1 | 16 | 300 | 0 | `down_proj` contracts over `k = 300`, the GEMM profile's `P = 3` ragged shape; reduced stage set |
| `base_b1_l4_d48_hd24_kv1` | 1 | 4 | 48 | 2 / 1 | 24 | 64 | 0 | `head_dim` 24, so the attention scale is INEXACT |
| `adv_mask_extreme_b1_l4_d32_kv2` | 1 | 4 | 32 | 2 / 2 | 16 | 64 | 0 | scores far above `2^103`, so the mask ADD is not the identity |
| `adv_subnormal_scores_b1_l4_d32_kv2` | 1 | 4 | 32 | 2 / 2 | 16 | 64 | 0 | scores in and below the float32 subnormal band |
| `adv_signed_zeros_b2_l4_d32_kv2` | 2 | 4 | 32 | 2 / 2 | 16 | 64 | 0 | planted `+0.0` and `-0.0`, compared BY SIGN BIT |
| `adv_rope_far_pos_b1_l4_d32_kv2` | 1 | 4 | 32 | 2 / 2 | 16 | 64 | 8188 | absolute positions 8188 to 8191, one below the ceiling |
| `adv_softmax_saturation_b1_l8_d32_kv1` | 1 | 8 | 32 | 2 / 1 | 16 | 64 | 0 | both arms of the denominator chain in one case |
| `decode_b1_l4_d32_kv1` (+ 4 `_step*`) | 1 | 4 | 32 | 2 / 1 | 16 | 64 | 0 | decode against prefill, contract 7.2 and clause (d) |

### What each case cannot certify

This section is the point of the corpus and it is written before any of it has
run, so that nobody quotes a case as evidence for a clause it cannot reach.

- **`base_b1_l1_d32_kv2` cannot certify ANY ordering along L or S.** At L=1 the
  causal mask is a single unmasked cell, the denominator is one term, and a
  token-major to head-major reindexing of a `[1, F]` stage is the same bytes.
  This is the mamba lane's `base_b1_l1_d8` lesson stated in advance rather than
  discovered again.
- **`base_b2_l4_d32_kv2` cannot certify `repeat_kv`'s own index arithmetic**,
  which is the identity at `n_rep == 1`. **`base_b1_l4_d32_kv1` cannot certify
  the head-to-kv-head assignment**, which is vacuous at `n_kv == 1` because
  every head reads the one kv head. Neither half is sufficient alone and the
  pair must be read together. Contract DEVIATION 813 requires both.
- **`base_b2_l16_d32_kv1_i300` cannot certify the GEMM's fold topology.** A
  different summation order moves a float64 result by about 1e-16 relative,
  four orders below any honest FP32 tolerance. What that case does is make the
  balanced tree RUN with a ragged last leaf, so a shape-dependent crash or a
  wrong leaf length is reachable. Correctness of the fold belongs to
  `mojolearn.identical.gemm.fp32.v1`.
- **`adv_subnormal_scores_b1_l4_d32_kv2` cannot certify the `-0.0` mask
  laundering, and that is the honest limit of the whole corpus.** Contract
  4.1(a) needs a score that is exactly `-0.0`, which is reachable only through
  OUR ftz of a negative subnormal accumulator. Torch has no ftz, so the
  reference holds a small negative number where we hold `-0.0`, and no
  tolerance and no sign-bit check turns that into a certificate.
  `S13_MASK_SELECT` is the lane's own oracle's job.
- **`adv_signed_zeros_b2_l4_d32_kv2` cannot certify zero signs downstream of
  the first GEMM.** Torch's matmul seeds its accumulator at `+0.0`, so a whole
  `-0.0` token projects to `+0.0` and the sign is gone by `q_proj.out` IN THE
  REFERENCE TOO. The corpus certifies zero signs at `input.x` and `norm1.out`
  and nowhere after them. This is the mamba corpus's finding repeated.
- **`adv_rope_far_pos_b1_l4_d32_kv2` certifies nothing about `rope.cos` if the
  lane's `inv_freq` bits differ from the reference's**, for the reason the
  precision rule gives. The checker marks those stages CONDITIONAL rather than
  passing or failing them.
- **`adv_softmax_saturation_b1_l8_d32_kv1` cannot certify the ORDER of the
  denominator fold.** It certifies WHICH TERMS are in the sum and that the max
  subtraction happened. `S17_DENOM_HALVING_TREE` is unreachable from here.
- **`decode_b1_l4_d32_kv1` certifies nothing about absolute position
  indexing**, because `pos0 == 0` makes relative and absolute the same numbers.
  A gate built on this case alone passes a driver whose position map is broken
  by a constant offset. `adv_rope_far_pos` is the only case in the corpus where
  `rope_relative_position` is not bit-inert, so it carries that whole clause.
- **No case can see a summation order anywhere.** Contract sections 5.3, 5.4
  and 7.2's order clauses, and seams S12 and S18's one-rounding-versus-two
  arguments, are all below the tolerance floor. `--verify` control 9 prints the
  full list of perturbations no case can see, and that list is the coverage
  statement.

### Cases the author PREDICTS the lane will fail

Standing rule, never build to datasets. No case here was picked, dropped,
deferred or tuned by whether it flatters us, and an adversarial case we fail is
worth more than one we pass. These are the predictions, made before anything
ran, and each is marked `expect_lane_failure` in its manifest.

- `adv_mask_extreme_b1_l4_d32_kv2`. The FP32 arm legitimately holds `-inf` in
  `attn.masked` where the float64 reference holds a finite value past the
  float32 maximum. The checker's explained-overflow class exists for exactly
  this, and if the lane's driver refuses a nonfinite INTERMEDIATE rather than
  computing it, this case will not produce a dump at all. Contract section 8's
  DEVIATION 815 names that gap.
- `adv_subnormal_scores_b1_l4_d32_kv2`. The profile flushes to zero at every
  seam and torch does not, so `attn.scores` and `attn.max` will hold signed
  zeros where the reference holds subnormals. Most of that is absorbed by
  `atol`; the parts that are not are a finding about ftz reachability, not
  about the block.
- `adv_rope_far_pos_b1_l4_d32_kv2`. `portable_powf` and the reference's FP32
  `pow` almost certainly do not agree in the last bit of every inverse
  frequency, and at position 8191 that is thousands of ulps of angle. A
  CONDITIONAL verdict here is the expected outcome and the useful one.
- `adv_softmax_saturation_b1_l8_d32_kv1`. `portable_expf` returns exactly
  `+0.0` below -87.33655 while float64 `exp` keeps going to about -745, so
  `attn.exp` will hold exact zeros against tiny nonzero references. `atol`
  absorbs it, but `attn.denom` and `attn.weights` inherit whatever the
  difference is worth, and the ratio of a very small denominator is where a
  tolerance argument goes wrong first.

## Tolerance calibration

Tolerance is CALIBRATED PER CASE PER STAGE from the corpus's own self test and
is never a fixed global number. The mamba lane paid for that lesson once. Its
`adv_gate_saturation` case failed at rtol 1e-7 where TORCH'S OWN FP32 also
failed against the same float64 reference, because the values reached 7.25e8
and 1e-7 is below float32 epsilon there. The tolerance was the defect, not the
block.

Each stage's manifest entry carries three numbers, all measured.

- `rtol_torch_fp32`, the smallest ladder rung at which a plain float32 CPU run
  of the SAME algorithm passes against the same float64 reference.
- `rtol_correctly_rounded`, the smallest rung at which the float64 reference
  ROUNDED ONCE to float32 passes. That is the floor no FP32 implementation can
  beat. When `rtol_torch_fp32` sits far above it, the REFERENCE arm is the
  limit at that stage, not the lane, and the checker says so.
- `rtol_gate`, the checker's default, exactly one ladder rung looser than
  `rtol_torch_fp32`. One rung and no more. Holding a dump to the reference
  arm's own rung makes the tolerance the defect the first time a stage's
  magnitudes move.

The ladder is `{1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1}` and the default atol
is `1e-6`. A stage whose `rtol_torch_fp32` is NONE could not be passed by the
reference arm at the loosest rung, which is a reportable finding about the
case, not a licence to widen the ladder.

The checker reports, per stage, the rtol OUR dump reaches beside the rtol the
reference arm reaches. **Ours being tighter than the reference arm is the
interesting result. Ours being looser than the gate is the finding.** Neither
is a bitwise claim.

Tolerance can never settle a signed zero. `isclose` treats `+0.0` and `-0.0` as
equal, so wherever the reference is exactly `+-0.0` the dump's SIGN BIT is
compared instead, always, and a sign-bit mismatch fails the stage while leaving
the tolerance verdict intact so the two are never confused.

## Negative controls

A checker with no negative control passes for ever. There are two sets and both
are meant to be run.

`gen_corpus.py --verify` runs nine.

1. **Determinism.** Regenerate into a temp directory and byte-compare. An
   unseeded RNG, a dict ordering dependency or a thread-count dependency shows
   up here and nowhere else.
2. **Batch composition.** The two `comp_row*` cases against their parent, in
   both dtypes. Proves the corpus is consistent with itself across B. It does
   NOT prove the lane's batch invariance, which is clause (c).
3. **Decode against prefill in torch's own float64.** Contract 7.1's
   masked-tail-is-inert theorem measured rather than argued. Each decode step
   is computed INDEPENDENTLY, not sliced, or this report would be a tautology.
4. **The decode NEGATIVE control.** Two different steps must DIFFER. Without
   it a broken position map passes clause (d) for ever, which is the mamba
   lane's clause (c) lesson.
5. **The rotary anchor.** The measured amplification, per case, beside float32
   epsilon. This is the precision rule's evidence.
6. **The attention scale, both spellings.** Contract section 3 measured
   agreement at seven `head_dim` values, none of them 24.
7. **Adversarial intent reached.** Counts, per adversarial case. A zero where a
   nonzero is described is a defect in the generator and it says so.
8. **No nonfinite value in any float64 reference.** A NaN in the reference
   would mean the generator produced it and every stage after it is
   meaningless.
9. **Sensitivity.** Seventeen deliberate wrong answers run against every case,
   classified per stage as SEEN, SUBTOL or bit-inert. A perturbation with NO
   seen stage anywhere is a clause this entire corpus cannot reach.

`tools/transformer_corpus_check.py --negative-control` runs seven more, on the
corpus's own `ref32` as a stand-in dump, so it works today with no lane code.
`perturb_one_element_beyond_gate` proves the check is not vacuous.
`scale_by_two` and `add_constant` must be caught AND named by the explainer.
`flip_all_zero_signs` changes nothing a tolerance can see and must be caught by
the sign-bit check alone. `transpose_last_two_axes`, `reverse_last_axis` and
`roll_token_axis` are caught where the case has enough shape and are reported
as INERT where it does not, which is the coverage statement for that case. It
also checks the two file-level classifications, that an absent file reads as
MISSING and a short file as WRONG SIZE, never as a value difference; `cmp -s a
b` reports "differ" when `b` does not exist and the mamba lane read that as a
total divergence once.

## Definition mismatches are reported, not fixed

When a stage fails by a clean explainable term the checker NAMES the term. It
looks for a constant offset, a constant factor (and names it when it is one of
the block's own quantities such as `d_model` or the attention scale), a
per-column additive vector (the shape of an included or omitted BIAS), a
per-row additive vector, a transposition of the last two axes, and the same
multiset of values in a different order.

That last case is why every input value is hashed. Uniform test data hides
permutation; hashed values make an index map visible as a reordering rather
than as noise.

The mamba corpus's `dt_proj.out` included a bias the contract excludes. The
right move was to report the naming disagreement with the elementwise
difference shown to BE exactly the bias, and NOT to fold the bias in to turn
the check green, because that would have broken the contract's stage split to
satisfy a name. This checker is built so that a stage differing by a clean term
is reportable AS THAT, and the fix for such a report is a decision about the
contract, never a quiet change to the corpus.

## File format

Per case directory `transformer/corpus/<case>/`.

- `<tensor>.f32` for `x` and every weight, under the module names in the hash
  spec table. Raw little-endian IEEE-754 float32, row-major, no header.
- `positions.i32` the case's ABSOLUTE positions, little-endian int32.
- `ref64/<stage>.f64` the float64 reference, raw little-endian, row-major.
- `ref32/<stage>.f32` the same stages from a plain float32 CPU run. Informative
  only, with three exceptions that are the REFERENCE'S OWN FP32 spellings and
  are load-bearing: `rope.inv_freq`, `rope.cos` and `rope.sin`.
- `manifest.json` everything a regenerator, a driver or the checker needs,
  including the per-stage calibration and the measured sensitivity table.

The top-level `manifest.json` lists all cases, seeds, tensor ids, stage
definitions, the perturbation list, the precision rule and the upstream pin.

## Deviations

This corpus owns 1040 through 1049. Contract section 12.3's 800 to 819 belong
to design decisions already made and are not reused here.

| # | what |
|---|---|
| 1040 | the corpus itself, `transformer-corpus-v1`, an independent float64 reference for the thirty card stages of `mojolearn.identical.transformer.fp32.v1` |
| 1041 | the rotary tables written over only the positions a case uses, rather than the card's `[P_max, head_dim/2]`, with the checker selecting rows by absolute position from a full dump |
| 1042 | THE PRECISION RULE. float64 everywhere except five FP32-pinned constants of the arithmetic, because RoPE's angle amplifies a one-ulp inverse-frequency difference by the absolute position |
| 1043 | the mask fill taken as the FP32 constant `-3.4028234663852886e+38` in the float64 reference, rather than `finfo(float64).min`, because the profile pins the number and not the expression |
| 1044 | the ROTARY ANCHOR check. `rope.inv_freq` verified bitwise against the reference's FP32 spelling before `rope.cos` and `rope.sin` are believed, and those stages reported CONDITIONAL when it does not match |
| 1045 | the DEFINITION EXPLAINERS. A failing stage that differs by a constant offset, a constant factor, a per-column vector, a per-row vector, a transposition or a permutation gets that term named instead of a bare FAIL |
| 1046 | per-case per-stage tolerance calibration written into the manifest, with the gate one ladder rung looser than the measured reference arm and never a global number |
| 1047 | the EXPLAINED FP32 OVERFLOW class. An infinity of the right sign where the float64 reference exceeds the float32 maximum is a pass, counted and reported |
| 1048 | the MEASURED SENSITIVITY TABLE. Seventeen named perturbations run against every case and classified per stage, so "what this case cannot certify" is a measurement |
| 1049 | reserved |

## Not claimed

- Not a bitwise certificate of anything. `ref64` is a tolerance reference.
- Not agreement with HuggingFace or PyTorch. The profile's fold orders,
  transcendentals and division are OURS, and contract section 11 says so. What
  this corpus offers is agreement with a float64 reference to a stated,
  measured tolerance.
- Nothing cross-vendor. This corpus runs on one CPU and says nothing about
  Apple, NVIDIA or AMD.
- No summation order, anywhere. See the coverage statement above.
- No `-0.0` produced by ftz, and therefore not contract 4.1(a).
- Not FlashAttention, not SDPA, not `rope_scaling`, not any mask but the causal
  one, not a bias on any projection, not a model. Contract sections 6 and 11.
- No performance number of any kind.
- **Nothing here has been run.** The generator, the checker and every count in
  this file are construction. The first `--verify` is the first measurement.
