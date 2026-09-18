# LANE STATUS: lane/attention-fallback-fix

Branch `lane/attention-fallback-fix`, off `origin/main` at `1863520e9`.
Brief: `docs/lanes/PROMPT_1_eager_attention_fallback.txt`.
Worktree `~/mojolearn-wt/attention-fallback-fix`. Evidence
`~/mojolearn-evidence/attention-fallback-fix/`.

The instrumentation in the first commit here is `lane/lm-attention-fallback`'s
(`576ee02bf`), cherry-picked rather than rewritten. It had never been run.

---

## What the 2.13x actually is, and what this lane can and cannot take off it

`lane/lm-step-memory-build` measured the step going 0.207 s -> 0.44 s as the
layers cross over. Decompose the step into the fused attention launch `A_f`,
the eager attention path `A_e`, and everything else `R`:

    before the transition:  A_f + R          = 0.207 s
    after the transition:   A_f + A_e + R    = 0.44  s

so `A_e = 0.233 s`. **`A_f` is NOT known from those two numbers**, and the
recoverable part of the 2.13x is exactly `A_f`, not the whole gap.

THE LATCH (DEVIATION 3110) REMOVES `A_f` ON A LATCHED LAYER AND NOTHING ELSE.
It cannot remove `A_e`, because the refusal means the eager path's bits are
the contract's bits. **The headline 2.13x is therefore NOT what this change
returns**, and the `eager-ref` arm exists to price `A_f` directly instead of
letting the ratio be assumed. Removing `A_e` needs the fused path to stop
refusing, which is a different change and is described at the end.

## Registered BEFORE the first run, with the falsifying outcome named

H100 80GB HBM3, IDENTICAL fp32, seed 20260917, pinned R2 enwik8
(100,000,000 bytes, SHA-256
`2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8`), 700
consecutive steps, `B1 L2048 DM768 H12 KV12 HD64 FF2048 layers12 V50257`
(162,147,840 parameters). Witness read EVERY step, not at the end.

**P1. THE TRIGGER IS `FUSED_CORNER` (status 2), NOT `FUSED_REFUSED_REGIME`,
and the BACKWARD refuses before and more often than the forward.** The
reasoning is not the brief's "regime bound is unreachable" alone; it is that
`ftz()` flushes a negative subnormal to `-0.0` WITH ITS SIGN, so the smallest
magnitudes in the step reach the corner condition first, and those are the
`dk`/`dv` accumulators. The backward's `dk`/`dv` corner also has NO
`j_hi < s - 1` guard (`fused_attention.mojo:3028-3030`) while the forward's
needs `-0.0` AND an incomplete row range (`:2031`), so the backward's
condition is strictly weaker.
FALSIFIED BY: any status 1 anywhere, or the forward refusing first on more
layers than the backward does.

**P2. THE REFUSAL IS STICKY IN THE DATA.** After a (layer, direction) first
refuses, it refuses on at least 95% of its remaining steps in the `before`
arm. FALSIFIED BY: under 50%, which would make the latch a pessimization and
would have to be reported as one.

**P3. NO BIT MOVES.** `after_vs_before` has an empty `loss_differ` over all
700 steps and an empty `hash_differ` over all 8 witnessed anchors
(loss, gradients, parameters, m, v, flags). So does `eager_vs_before`.
FALSIFIED BY: any differing step. `eager_vs_before` differing while
`after_vs_before` does not would mean the fused and eager paths are not
bit-equal on the steps the fused path ran, which is a defect in the lane's
premise and not in the latch; it would be reported that way.

**P4. TIMING.** `before` head median 0.18 to 0.26 s (step 0 EXCLUDED: it is
15+ s of session setup, and including it is the defect that made the other
lane's `slowed` flag unable to fire), tail median 0.40 to 0.50 s. `after`
tail median lands within 5% of the `eager-ref` tail median, because once
every layer has latched in both directions the two builds run the same
kernels. Recovered fraction `(before_tail - after_tail) / before_tail`
between 0.15 and 0.45.
FALSIFIED BY: `after` tail >= `before` tail, or `after` tail more than 5%
BELOW `eager-ref` tail (which would mean the arms are not what they claim).

**P5. THE DEVICE PEAK DOES NOT MOVE.** `after` peak equals `before` peak
within allocator granularity, and `eager_bytes` reaches 17,314,086,912 in
both. The latch removes a LAUNCH, not a BUFFER. FALSIFIED BY: a peak that
moves, which would mean the latch is doing something it was not written to do.

**P6. BATCH 4 STILL DIES.** It does not reach 2,000 steps, and it dies at
about the step at which its layers finish latching. FALSIFIED BY: 2,000
steps completing.

## The A/B has two arms, and three ways to tell them apart

1. `byte_lm_attn_sticky_fallback()` read from INSIDE the process that loaded
   the binding (`training/byte_lm.mojo`, through the binding, into every
   `result.json`). A `.so` digest cannot answer this: the define gates one
   runtime branch on a comptime constant, so the two builds are the same size
   and differ by about one byte.
2. `FUSED_SKIPPED_STICKY` (status 3) appears in the `after` arm and in
   NEITHER of the others. Both arms reading zero would mean one build ran
   twice, which reads exactly like a passed identity gate.
3. The controls at head_dim 64. At head_dim 8 the fused kernel is not
   instantiated at all (`fused_supported_head_dim` admits 16, 24, 64, 128)
   so BOTH arms fall back and the comparison is pass-shaped and empty.

Each check in `tools/lm_attention_fallback_verdict.py` is run against a
deliberately corrupted copy of its own input first and must reject it BY
NAME, printing the match rather than a count.

---

# MEASURED: H100 sm_90a, pod rhjqy941tjl5yw, commit 8a5212a78, 2026-09-18

700 consecutive steps per arm, `B1 L2048 DM768 H12 KV12 HD64 FF2048
layers12 V50257`, pinned R2 enwik8 (stage.log: 100,000,000 bytes, digest
equal to the pin, linked into the tree), witness read EVERY step.
Filed at `bench/results/e1g/2026-09-18_lm-attention-fallback-nvidia-h100/`.
Pod terminated and VERIFIED gone (HTTP 404), dead-man cancelled.

## 1. THE TRIGGER, NAMED

    forward_status  over 8,400 observations:  FUSED_RAN 8,400
    backward_status over 8,400 observations:  FUSED_RAN 4,703  FUSED_CORNER 3,697

**`FUSED_CORNER`, in the BACKWARD, and ONLY in the backward.** Zero
`FUSED_REFUSED_REGIME` anywhere, which retires the regime bound as a
candidate by measurement rather than by argument. The FORWARD never refused
once in 8,400 observations.

Per layer, first refusal step and the refusal rate over the steps AFTER it:

    L00 first 69   14.7%     L06 first 214  85.6%
    L01 first 456  56.2%     L07 first 245  93.9%
    L02 first 309  85.7%     L08 first 219  91.3%
    L03 first 301  73.2%     L09 first 201  89.6%
    L04 first 264  91.3%     L10 first 218  53.1%
    L05 first 205  80.8%     L11 first 299  14.2%

## 2. P2 IS FALSIFIED, AND P1 IS CONFIRMED MORE STRONGLY THAN IT WAS PUT

P1 said the backward would refuse "before and more often" than the forward.
The forward refuses NEVER. P2 said the refusal would be sticky at 95%; it runs
from 14.2% to 93.9%. **The latch is therefore the wrong shape for this data**
and the arm that was built to remove the double pay is a pessimization.

## 3. THE BRIEF'S ITEM 2 IS WORTH 0.5%, NOT 2.13x

                 head median    tail median
    before        0.19791 s      0.45737 s     today's behavior
    after         0.19785 s      0.47403 s     the latch
    eager-ref     0.45504 s      0.45500 s     the fused kernels never run

**THE DISCARDED FUSED LAUNCH IS WORTH 0.00237 s A STEP** -- the `before` tail
minus the `eager-ref` tail, 0.5% of the step. The brief and
`LANE_STATUS_lm-step-memory-build.md` both say the cost is 2.13x "rather than
the eager path's own speed". IT IS THE EAGER PATH'S OWN SPEED. `eager-ref`
reads 0.45504 s at the HEAD and 0.45500 s at the TAIL: the eager path costs
0.455 s a step from step 0 and never changes, and the whole 0.198 -> 0.457
transition is the fused path being replaced by it, term for term.

The latch's tail is 0.47403 against 0.45737, **3.6% WORSE**, for the reason
P2's falsification predicts: it forces the 0.257 s eager path onto steps that
would have paid only the 0.0024 s launch. It is also slower than `eager-ref`
because the forward never refuses, so the forward still runs FUSED and the
latched eager backward then calls `ensure_attention_materialized`, which
recomputes this layer's whole eager forward on top.

**The latch is therefore OFF BY DEFAULT** (`-D MOJOLEARN_ATTN_STICKY=1` turns
it on). It is kept because it is the arm that MEASURES the launch.

## 4. P3 AND P5 CONFIRMED

`after_vs_before`: 700 steps, 8 state anchors, **0 differing**. The latch
moves no bit. `eager_vs_before`: 700 steps, 8 anchors, **0 differing** -- the
fused and eager paths are bit-identical over a whole training run on real
data, which is the strongest statement of the IDENTICAL contract this shape
has. Device peak 31,537 MB against 31,281 MB, `eager_bytes` 17,314,086,912 in
both, twelve layers grown in both: the latch removes a launch, not a buffer.

Both arms told apart three ways: `byte_lm_attn_sticky_fallback()` read from
inside each process (False / True), `FUSED_SKIPPED_STICKY` observed 5,388
times in `after` and 0 times in `before` and `eager-ref`, and four head_dim-64
controls reading -1 forced eager and 0 forced fused. Every coverage and
comparison check was watched rejecting a corrupted copy of its own input.

## 5. A DEFECT IN THE INSTRUMENTATION, FOUND BY THE BATCH ARM

**batch 4 did NOT OOM and this leg does NOT answer the batch question.** It
aborted at about step 320 with

    ./training/byte_lm.mojo:796: index 11 is out of bounds, valid range is 0 to 10

`byte_attention_eager_cells` read `n_layers` entries out of `tr.forward` /
`tr.backward`, but the backward loop `pop`s each layer's stages out of those
lists and reinserts them (`byte_lm.mojo:1169-1170`), so a step that raises
inside a backward call leaves them SHORT and the report aborts the process
instead of reporting. Fixed here by bounding the loop by the list length and
reporting both lengths, so a short list is VISIBLE rather than fatal. It
reached ~320 steps at batch 4 before dying, which is past the step 210 the
brief gives for the batch-4 OOM, but the crash is the instrument's and no
claim is made from it.

---

# DEVIATION 3111: the guard the backward `dk`/`dv` corner never had

Section 1 says the guarded forward refused 0 times and the unguarded backward
refused 3,697 times AT THE SAME SHAPE, ON THE SAME DATA, IN THE SAME STEPS.
That is not a property of forward versus backward arithmetic. It is the
predicates:

    forward ctx   `... == NEG_ZERO_BITS and rr[1] < s - 1`      :2031
    backward dq   `... == NEG_ZERO_BITS and j_hi < s - 1`       :2871
    backward dk   `... == NEG_ZERO_BITS`                        (no guard)
    backward dv   `... == NEG_ZERO_BITS`                        (no guard)

A `-0.0` is a refusal only because the EAGER chain folds the masked cells too
and `fma(+0.0, x, -0.0)` is `+0.0` whenever `x`'s sign bit is clear. That
laundering needs a masked cell AFTER the last visible one. The `dk`/`dv`
chains run over the QUERY axis, and `_key_query_range` (`:1782-1796`) returns
`hi = l - 1` for EVERY key unless a sliding window is set. At this shape
`window == 0` and `n_rep == 1`, so **there is no masked tail at all and every
one of those 3,697 refusals is a false positive.**

The change adds `hi[u] < l - 1 or (hh + 1 < n_rep and lo[u] > 0)` to the
`dk`/`dv` test in `fused_bwd_dkdv_r2_kernel`, the kernel the column default
resolves to (`native_attention_arm` on every `result.json` reads
`stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32_bswz`).
`-D MOJOLEARN_ATTN_NO_KV_CORNER_GUARD=1` restores the unguarded test and
`byte_lm_attn_kv_corner_guard()` reads which build is loaded from inside the
process. `fused_bwd_dkdv_tiled_kernel` and `fused_bwd_kvfold_r2_kernel` carry
the same unguarded test and are NOT changed; they are trial arms, and that is
said rather than counted.

## Registered BEFORE the second run, with the falsifying outcome named

**G0. THE CONTROL REPRODUCES.** The unguarded arm shows 3,697 backward
corners within 15% and a tail within 10% of 0.45737 s. FALSIFIED BY anything
else, and then NOTHING ELSE IN THE LEG IS COMPARABLE and it is reported that
way rather than as a win.

**G1. THE REFUSALS GO TO ZERO.** Guarded backward `FUSED_CORNER` = 0 out of
8,400, matching the forward's 0 out of 8,400. FALSIFIED BY any refusal.

**G2. NOTHING GROWS.** Guarded `eager_bytes` reads 432 at step 699 and
`layers_grown` is 0 / 0. FALSIFIED BY growth.

**G3. THE 2.13x IS GONE.** Guarded tail median equals its head median within
5%, both in 0.19 to 0.21 s. FALSIFIED BY a tail above 0.25 s.

**G4. NO BIT MOVES.** Guarded against unguarded: 0 differing over 700 steps
and 8 state anchors. **FALSIFIED BY ANY DIFFERING STEP, AND THAT IS THE
IMPORTANT OUTCOME**: it would mean those refusals were REAL, the guard is a
DEFECT rather than an optimization, and it gets reverted. This is a change to
a bit-equality TEST, so it is the one prediction that can turn the whole
lane into a retraction, and it is written that way on purpose.

**G5. THE MEMORY COMES BACK.** Guarded device peak stays near the 15,151 MB
`before` starts at, rather than climbing to 31,537 MB.

**G6. BATCH 4 SURVIVES 2,000 STEPS.** FALSIFIED BY an OOM, which under G2
would mean the growth has a second source this lane has not found.

## What is NOT in this change: the forward corner is REPAIRABLE, exactly

`A_e` survives the latch because the refusal is honest. But the divergence
the corner guards against is ONLY EVER THE SIGN OF A ZERO, and the eager
answer for a cornered cell is decidable without running the eager path at
all. This is DERIVED FROM THE CODE, not from the brief's two candidates, and
it is NOT MEASURED HERE.

Both chains are the same chain:

  eager   `attn_context_kernel` (modeling_llama.mojo:3057-3125): one thread
          per output cell, `acc = +0.0`, then `acc = ftz(fma(w[j], v[j][d],
          acc))` SERIAL ASCENDING over the ABSOLUTE key index `j` in
          `[0, s)`. Contract DEVIATION 807 makes this deliberately NOT a
          gemm call, for exactly this reason.
  fused   `fused_attn_forward_regblocked_kernel`: `cacc` seeded `0.0`
          (`fused_attention.mojo:1889`), `_step(w, v, cacc)` -- the same
          `ftz(fma(...))` seam -- over the VISIBLE run `[rr[0], rr[1]]`
          only.

A masked cell's weight is EXACTLY `+0.0`: S13 adds `-FLT_MAX`, S15's
`exp(masked - amax)` is `+0.0`, and S18 divides by a positive denominator.
So the two chains differ only in the masked terms the eager one also folds:

  LEADING masked terms (`j < rr[0]`) are INERT. `acc` is still `+0.0` there
  and `fma(+0.0, v, +0.0)` is `(+-0.0) + (+0.0)` = `+0.0` for every `v`.
  That is why the corner predicate does not test `rr[0] > 0`.

  TRAILING masked terms (`j > rr[1]`) are inert too, for every `acc` EXCEPT
  `acc = -0.0`. `fma(+0.0, v, -0.0)` is `sign(v) * 0.0 + (-0.0)`, which is
  `+0.0` when `v`'s sign bit is CLEAR and `-0.0` when it is SET. Once it
  flips to `+0.0` it stays there.

  `acc` reaches `-0.0` at all because `_step` FLUSHES: `ftz` of a negative
  subnormal is `-0.0`, sign preserved. The kernel docstring's "the `+0.0`
  seed forbids that" is true of the SEED and not of the chain, which is why
  `FUSED_CORNER` is a runtime flag and not a proof.

So, exactly:

  **eager(cell) == fused(cell), EXCEPT when fused(cell) has the bits of
  `-0.0` and `rr[1] < s - 1`, and then eager(cell) is `+0.0` if ANY
  `v[b][h // n_rep][j][d]` for `j` in `(rr[1], s - 1]` has its sign bit
  clear, and `-0.0` otherwise.**

That is a suffix scan of SIGN BITS over `v_cache`, `[b, n_kv, s, head_dim]`,
1,572,864 entries at this shape -- 1.6 MB as bytes -- plus one patch kernel
over `ctxv`. It replaces a 620,756,992-byte eager forward set, five stage
kernels and a per-head gather/GEMM/scatter with two cheap kernels and no
quadratic buffer at all. That is the change that removes `A_e` from the
forward.

IT IS NOT BUILT AND NOT MEASURED. Two things stand between the derivation
and a merge:
  1. It must be measured against the eager path bit for bit at the target
     shape, because every line above is a claim about signed zeros and FTZ
     and every one of them is a place to be wrong by one bit.
  2. THE BACKWARD IS A SEPARATE DERIVATION AND IS NOT DONE. Its `dk`/`dv`
     corner (`fused_attention.mojo:3028-3030`) has NO `j_hi < s - 1` guard
     at all, so its predicate is not the forward's and neither is its
     repair. Nothing above should be read as covering it.
