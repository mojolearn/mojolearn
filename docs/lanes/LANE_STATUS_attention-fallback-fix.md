> Follow-up: the exact masked-tail replay now qualifies NVIDIA defaults,
> including native NVIDIA/AMD bit controls and B4 2,000-step endurance.
> See [the complete repair record](LANE_STATUS_lm-attention-fallback.md).
> The earlier disabled experiments below remain historical evidence.

# LANE STATUS: lane/attention-fallback-fix

**FOR A READER WITH NO CONTEXT. READ THIS BOX FIRST.**

- **THE TRIGGER IS NAMED AND REPRODUCED ON TWO BOXES.** `FUSED_CORNER`
  (status 2), in the BACKWARD ONLY. Over 8,400 observations (12 layers x 700
  steps) per direction: forward `FUSED_RAN` 8,400 and nothing else; backward
  `FUSED_RAN` 4,703, `FUSED_CORNER` 3,697. **Zero `FUSED_REFUSED_REGIME`
  anywhere**, which retires the regime bound by measurement. The second leg,
  a different pod and a different commit, read **3,697 again**.
- **EVERY RUN HERE IS 700 STEPS.** Nothing in this document comes from a
  short run. A 100 or 200 step run cannot see this fallback.
- **THE BRIEF'S ITEM 2 IS WORTH 0.5%, NOT 2.13x.** The discarded fused launch
  is 0.00237 s a step. The 2.13x IS the eager path's own speed: a forced-eager
  arm reads 0.45504 s at the HEAD and 0.45500 at the TAIL, i.e. it costs
  0.455 s from step 0 and never changes.
- **STEP TIME** (median, step 0 excluded): today's behavior head 0.19791 s,
  tail 0.45737 s. Device 15,151 -> 31,537 MB, `eager_bytes` 432 ->
  17,314,086,912, twelve layers grown. **The latch (3110) and the guard
  (3111) each move NO BIT and each leave those numbers where they are.**
- **BATCH 4, BOTH WAYS.** With the fallback: a real
  `CUDA_ERROR_OUT_OF_MEMORY` out of `byte_lm_session_step` at **step ~361**
  (not the brief's step 210; the OOM step is data dependent like everything
  else here). Without it: **step 1,323 and still running**, 0.666 s a step,
  loss 1.593, `forward_eager_cells` still 48, killed by THIS LEG'S OWN 900 s
  `timeout` (exit 124) and NOT by memory. **It did not reach 2,000 steps and
  no claim is made that it would**; what is measured is 1,323 against 361 and
  a footprint that never grew.
- **`attn_materialized` PLUMBING IS DONE AND RUNNING.** Per layer, per step,
  per direction: `forward_status`, `backward_status`, their named `*_counts`,
  and `attn_materialized`, out through `byte_lm_session_info` into
  `attention_stage_report()`. Statuses 0/1/2 plus 3 `FUSED_SKIPPED_STICKY`.
- **EVIDENCE**: `bench/results/e1g/2026-09-18_lm-attention-fallback-nvidia-h100/`
  and `bench/results/e1g/2026-09-18_lm-attention-guard-nvidia-h100/`, both
  committed. Local logs `~/mojolearn-evidence/attention-fallback-fix/`.
- **PODS. THIS LANE HAS RENTED EXACTLY THREE, EVER.** The
  `mojolearn-gemm-nvidia-` prefix is shared by every lane that calls
  `tools/gemm_remote_leg.sh`, so the stamp is the only way to tell them apart:

      rhjqy941tjl5yw  stamp 035439  leg 1  TERMINATED, VERIFIED gone (404)
      h6o7o98tid619l  stamp 043438  leg 2  TERMINATED, VERIFIED gone (404)
      5guu23hvyqj7tg  stamp 050110  leg 3  TERMINATED, VERIFIED gone (404)

  Never more than one at a time. `5guu23hvyqj7tg`
  is now also TERMINATED and VERIFIED gone (404). **THIS LANE HAS NO PODS
  OUT.**
  **Any other `mojolearn-gemm-nvidia-` pod is NOT this lane's** and must not
  be reaped on its account; this lane's ids are the three above and they are
  in `pod_id.txt` in each filed leg directory.
- **LEG 3 ANSWERED IT. THE FALLBACK IS REMOVABLE AND IT IS WORTH 2.298x AND
  16,384 MB.** With the backward's corner refusal removed
  (`-D MOJOLEARN_ATTN_NO_BWD_CORNER=1`), 700 steps of real training on the
  pinned corpus produce **bit-identical loss, gradients, parameters, m, v and
  flags at every one of 8 state anchors and every one of 700 steps' losses**;
  backward refusals go 3,449 -> **0**; the tail goes 0.45721 -> **0.19896 s**
  against a head of 0.19768, i.e. **the tail equals the head and the 2.13x is
  gone**; `eager_bytes` stays at **432** with **zero** layers grown; the
  device peak stays at **15,153 MB** instead of climbing to 31,537.
- **THAT DOES NOT MAKE "NEVER REFUSE" SHIPPABLE**, and 3112 is a measurement
  arm that stays off. The corner test exists because the two chains CAN
  differ; what leg 3 proves is that at this shape they do not, so the cost is
  being paid for a predicate that is wrong, not for a hazard that is real.
  The shippable change is a CORRECT predicate (or the signed-zero repair at
  the end of this document), not a deleted one. **This is the top remaining
  item in the lane and it is now worth a measured 2.298x, not an argued one.**
- **ONE HONEST LIMIT ON THAT CLAIM**: the comparison is at the STEP OUTPUTS
  (loss, gradients, parameters, m, v, flags), not at `d_q_rope`, `d_k_cache`
  and `d_v_cache` themselves. A signed zero that differs inside those buffers
  and is absorbed before the parameter gradient would not be caught here. The
  per-stage identity card records stages 22-24 and would be the check that
  closes it; it has NOT been run against this arm.
- **THE EXACT NEXT COMMAND**, if leg 3 was lost:

      cd ~/mojolearn-wt/attention-fallback-fix
      MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
      MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_STAGE_STRICT=1 \
      MOJOLEARN_STAGE_KEYS="corpus/enwik8/input.txt" \
      MOJOLEARN_GEMM_LEG_EXTRA=tools/lm_attention_norefuse_body.sh \
      sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent \
          --allow-concurrent --minutes 60 --gpu "NVIDIA H100 80GB HBM3"

  then read `<leg out>/remote/lm-attention-norefuse/verdict.log`. It answers
  THE remaining question: with `-D MOJOLEARN_ATTN_NO_BWD_CORNER=1` the
  backward keeps its own output on a corner instead of falling back, and the
  bit comparison against the refusing arm says whether those 3,449 remaining
  refusals are REAL. Equal means the 2.13x is recoverable; different means the
  fallback is earning its cost and only the signed-zero repair at the end of
  this document can remove it. Either answer is the result.

---

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

## 6. LEG 2: DEVIATION 3111 IS BIT-SAFE AND INSUFFICIENT

H100 sm_90a, pod h6o7o98tid619l, 2026-09-18, 700 steps per arm.
`bench/results/e1g/2026-09-18_lm-attention-guard-nvidia-h100/`.

THE CONTROL REPRODUCED TO THE OBSERVATION: 3,697 backward corners against the
first leg's 3,697; tail 0.45734 s against 0.45737; head 0.19814 against
0.19791; forward 8,400 `FUSED_RAN` against 8,400. Two boxes, two commits, the
same numbers. G0 CONFIRMED.

    G1 backward corners -> 0            FALSIFIED: 3,697 -> 3,449 (6.7% removed)
    G2 nothing grows                    FALSIFIED: still 17,314,086,912
    G3 tail == head                     FALSIFIED: tail 0.45707, 1.0006x
    G4 no bit moves                     CONFIRMED: 0 differing, 700 steps, 8 anchors
    G5 memory comes back                FALSIFIED: 31,537 MB both ways
    G6 batch 4 survives 2,000 steps     FALSIFIED: OOM at step ~361

**REPORTED INERT, not passed.** The guard is a correct, bit-safe tightening of
an over-conservative predicate and it buys NO time and NO memory. Four of six
predictions falsified, which is what registering them was for.

The remaining 3,449 are NOT dk/dv: after the guard, dk/dv cannot fire at
`window == 0` and `n_rep == 1`. They are the `zdot` or `dq` chains. `dq` is
nominally guarded (`j_hi < s - 1`, `:2871`) but at a causal mask `j_hi` is the
query row, so that guard is true for every row but the last and buys almost
nothing. The forward carries a structurally identical guard and still refuses
ZERO times, so the difference is arithmetic, not predicate: the backward's
gradients are the smallest magnitudes in the step and `ftz` of a negative
subnormal is `-0.0` with its sign kept. That is why leg 3 asks whether the
refusals are real rather than tightening another predicate.


---

# WHAT MERGES, AND WHY MAIN'S BEHAVIOR IS UNCHANGED

**Every behavior change in this lane is OFF BY DEFAULT.** What merges is the
instrumentation, the three measurement arms, the evidence and this document.
The 2.298x is proven to EXIST and is not proven to be SAFE TO SHIP, and those
are different claims.

    DEVIATION 3110  the latch                 OFF  -D MOJOLEARN_ATTN_STICKY=1
    DEVIATION 3111  the dk/dv corner guard    OFF  -D MOJOLEARN_ATTN_KV_CORNER_GUARD=1
    DEVIATION 3112  the backward never refuses OFF -D MOJOLEARN_ATTN_NO_BWD_CORNER=1

3110 is off because it is MEASURED 3.6% slower. 3112 is off because it deletes
a safety check rather than fixing it, and one shape's 700 steps is not a
licence to stop checking. 3111 is off for the asymmetry: it buys NOTHING
measurable (248 of 3,697 refusals, 1.0006x, 0 MB) while a guard that is wrong
suppresses a needed refusal and moves bits SILENTLY, and its general form has
only ever run at `window == 0` with `n_rep == 1` at scale.

**What DOES merge as a default change: nothing in the kernels.** The
instrumentation is host metadata (`len()` of buffers the trainer owns plus
saved launcher return codes), and the one real bug fix is the out-of-bounds in
`byte_attention_eager_cells` described in section 5.

# THE NEXT LANE'S JOB, NOW WORTH A MEASURED 2.298x

Make the backward's corner predicate RIGHT, so it refuses when the laundering
actually changes a bit and not otherwise. Leg 3 says the whole 2.13x and
16,384 MB are on the other side of that. Three things are known and should not
be rediscovered:

1. **It is not `dk`/`dv`.** DEVIATION 3111 already guards those and it only
   moved 248 of 3,697. The remaining refusals are the `zdot` or `dq` chains.
   **The cheapest next measurement is to give each of the three backward
   kernels its own corner flag word** so the report names which chain fires.
   That is a one-word change and one 700-step run, and it is where to start.
2. **The forward is the worked example.** Its predicate is `-0.0 AND
   rr[1] < s - 1` and it refused 0 times out of 8,400 on the same data in the
   same steps. The derivation of exactly when a `-0.0` matters is below.
3. **The repair below needs no quadratic buffer at all**: a suffix scan of
   SIGN BITS over the trailing operand, 1.6 MB at this shape, against the
   620,756,992-byte eager stage set the fallback allocates instead.

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
