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

## What is NOT in this change, and what it would take

`A_e` survives because the refusal is honest: at a corner the fused output
and the eager output differ, so the eager bits are the contract's bits.

The difference is only ever THE SIGN OF A ZERO. The fused kernel folds the
row's VISIBLE key range; the eager reference folds the full `[0, s-1]`, where
the masked cells carry weight exactly `+0.0`. Folding a run of signed zeros
onto `-0.0` gives `-0.0` only when every one of them is `-0.0`, and `+0.0`
otherwise. So the eager answer for a corner cell is decidable from a suffix
scan of the sign bits of `v` -- an `[b, n_kv, s, head_dim]` array, 6.3 MB at
this shape -- instead of from re-running the whole eager path and its 1,376
MiB of stages. That is the change that would remove `A_e`.

IT IS NOT BUILT AND NOT MEASURED HERE. It reproduces the eager path's exact
semantics from reasoning about signed zeros and FTZ, and every one of those
steps is a place to be wrong by one bit; it also has to be done a second time
for the backward's `dq`/`dk`/`dv`, whose corner conditions are not the
forward's. It is written down so the next session starts from the arithmetic
rather than from the brief's two candidates.
