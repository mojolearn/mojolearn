# LANE STATUS: lane/attention-corner-predicate

**FOR A READER WITH NO CONTEXT. READ THIS BOX FIRST.**

- **THIS LANE'S TASK WAS ALREADY DONE AND MERGED TO MAIN BEFORE IT STARTED.**
  The brief that opened this lane ("write a correct corner predicate and turn
  the 2.298x on") describes work that `lane/lm-attention-fallback` finished
  and landed on 2026-09-18. This lane RENTED NOTHING, ran no GPU job, and
  changed no source. It verified the merged result from source and from the
  raw evidence JSONs and recorded the per-kernel attribution the brief asked
  for, which the merged evidence already contained.
- **THIS LANE HAS NO PODS OUT AND NEVER DID.** It rented zero boxes.
- Branch `lane/attention-corner-predicate`, cut from `origin/main` at
  `c7442abed`. Worktree `~/mojolearn-wt/corner-predicate`.

---

## 1. WHAT THE CORNER PREDICATE ACTUALLY GUARDS (prose, from the source)

It is **not** an overflow bound, not a range assumption and not a shared
memory capacity. It is a **signed-zero laundering** condition.

The fused kernels earn their speed by SKIPPING the masked cells of a row.
A masked cell is `ftz(s + (-FLT_MAX))`, which is exactly `-FLT_MAX`, whose
`exp` is exactly `+0.0` and whose chain contribution is therefore a term of
magnitude zero. For any accumulator `acc`, `acc + (±0.0) == acc` — **except
when `acc` is `-0.0`**, because IEEE round-to-nearest gives
`-0.0 + (+0.0) == +0.0`. So a chain that is holding `-0.0` at the end of its
VISIBLE run can be *laundered to `+0.0`* by the masked tail the fused kernel
skipped. The eager path, which computes every cell, would produce `+0.0`;
the fused path would produce `-0.0`. That is a one-bit difference in the
sign bit of a zero, and under this profile that is a defect.

Hence the shipped predicate, spelled per chain inside each kernel:

    if bitcast[uint32](acc) == NEG_ZERO_BITS and j_hi < s - 1:  refuse

i.e. "the chain ended at `-0.0`" AND "an omitted trailing cell exists".
`transformer/impl/llama/fused_attention.mojo` lines 31-50 state this; the
declaration is at :144-149 and `NEG_ZERO_BITS` at :156.

**This is a different check from the regime bound.** `REGIME_BOUND = 2^100`
(:158) is the actual numeric-magnitude guard — it keeps every dot below
`2^102`, where `x + (-FLT_MAX)` is still exactly `-FLT_MAX`. That one is
`FUSED_REFUSED_REGIME` (value 1) and it **fired zero times** in every
700-step run on either corpus. The regime bound is retired by measurement.

## 2. STEP 1 ANSWERED: PER-KERNEL ATTRIBUTION OF THE 3,697

The merged evidence separates the kernels two ways, and both were recomputed
here directly from the raw result JSONs (not from prose):

    tail/legacy/result.json          backward status: RAN 4703, CORNER 3697
    tail/guarded/result.json         backward status: RAN 4951, CORNER 3449
    integrated/repaired/result.json  repair masks: 4951 none, 3449 dQ, 0 zdot
                                     (mask bit 1 = zdot, bit 2 = dQ)

`tail/guarded` is the dk/dv tail guard alone with replay OFF. So, of 8,400
backward layer-steps at the target shape on enwik8:

| kernel | corner refusals | verdict |
|---|---:|---|
| joint r2 dk/dv (`fused_bwd_dkdv_*`) | **248** (6.7%) | **CONSERVATIVE** |
| dQ (`fused_bwd_dq_tiled_pf_kernel`) | **3,449** (93.3%) | **REAL corner, wrong remedy** |
| zdot (`fused_bwd_zdot_*`) | **0** | never fired in training |

248 + 3,449 = 3,697 exactly, and 4,703 + 248 = 4,951 exactly. The counts
close. The per-kernel flag the brief asked to plumb already exists:
`backward_repair_sites`, a per-layer bitmask, documented at
`python/mojolearn/_byte_lm_impl.py:613` and read at :670-671.

## 3. VIOLATE OR CONSERVATIVE? BOTH — AND A THIRD ANSWER

The brief framed this as a binary. The measurement says it is neither alone:

- **The 248 dk/dv refusals were purely conservative.** That kernel flagged
  every terminal `-0.0`, *including after the last head where no masked
  suffix follows at all*. A terminal negative zero with no omitted tail is
  already the correct answer. DEVIATION 3111 tightened the predicate to
  `hi < l - 1 or (hh < n_rep - 1 and lo > 0)` — a masked suffix, or a next
  head whose skipped prefix could still touch it — and those 248 vanished
  with no arithmetic change and no bit moved
  (`fused_attention.mojo:4907-4921`). The initial skipped prefix seeds from
  `+0.0` and cannot change that seed, which is why only a *following* head's
  prefix counts.
- **The 3,449 dQ refusals genuinely satisfy the corner condition.** The
  chain really does end at `-0.0` with omitted terms remaining. The
  predicate was right; "never refuse" is not a correct fix and the native
  fixtures prove it — with replay skipped, fused reads `0x80000000` where
  eager reads `0x00000000`
  (`transformer/checks/attention_masked_tail_check.mojo`).
- **The correct remedy for those 3,449 is not a looser predicate, it is
  EXACT REPLAY.** Instead of throwing the whole layer to the eager path,
  the kernel replays only the omitted trailing terms, in the eager order,
  with the eager seams (`_masked_tail_dy`, :4412-4423, and the replay loops
  at :4555-4568 for dQ and :5767-5776 for zdot), and stops the moment the
  accumulator leaves `-0.0` — because once it is `+0.0`, no remaining finite
  signed-zero term can change it. The sign is never canonicalized. That is
  the shipped fix.

That leg 3 "remove the check entirely" arm read bit-identical at 700 steps
is therefore **not** evidence the predicate was wrong: it is evidence that
at this shape the differing `-0.0` inside `d_q_rope` was absorbed before the
parameter gradient. The native fixtures show the buffer bit does differ.

## 4. DOES THE NEW PREDICATE STILL REFUSE? YES — AND IT IS EXERCISED

A predicate that never refuses is not a predicate. The dk/dv kernel still
takes the eager path on a genuine laundering corner; only the dQ and zdot
paths replace refusal with exact replay. The deliberately-constructed cases
are the **preservation fixtures** in
`transformer/checks/attention_masked_tail_check.mojo`: a negative masked
`dcell` against a positive K, where the correct answer IS `-0.0`. The gate
requires fused and eager to BOTH read `0x80000000` (:85-87 zdot, :144-146
dQ) and fails otherwise. `MOJOLEARN_ATTN_TAIL_GUARD_SABOTAGE`
(:4925-4928) deliberately erases a negative zero the clean arm keeps, and
the gate catches it. 130 explicit repair/preservation sites pass on NVIDIA
and AMD with observed failing controls.

## 5. MEASURED RESULT AS MERGED (NVIDIA H100 80GB HBM3, 700 steps)

| corpus | legacy tail s | default tail s | speedup | device MiB | backward CORNER |
|---|---:|---:|---:|---:|---:|
| enwik8 | 0.455438880 | 0.197156515 | **2.3100x** | 31537 → 15153 | 3697 → 0 |
| Pile GitHub | 0.377133856 | 0.196830695 | 1.9160x | 30257 → 15153 | 1768 → 0 |

Two-corpus geomean **2.1038x**. Mechanical verdict FLIP NVIDIA
(`bench/results/lm_attention_fallback_2026-09-18/default/flip_verdict.log`),
whose deliberately doubled-time control returned NO FLIP first. Bitwise
gate: all 700 losses and all six state hashes at steps 0 and 699 match
legacy on both corpora. Eager storage 17,314,086,912 → 432 bytes.
**The 2.298x target is met on enwik8 (2.3100x); the geomean is 2.1038x
because Pile GitHub started with 1,768 refusals, not 3,697.**

## 6. WHAT IS TURNED ON, AND WHERE

`checks/kernel_matrix.mojo:1210 attn_masked_tail_replay_for` →
`return column == COLUMN_NVIDIA`. That row drives
`ATTN_REPAIR_MASKED_TAIL` (`fused_attention.mojo:150-152`), which in turn
forces `ATTN_EXACT_TAIL_GUARD` (:155) and `ATTN_BWD_KV_CORNER_GUARD` (:195).
`MOJOLEARN_ATTN_LEGACY_CORNER` restores the old behavior for A/Bs.

## 7. HONEST LIMITS — WHAT IS STILL OWED

1. **NVIDIA ONLY.** `attn_masked_tail_replay_for` is `COLUMN_NVIDIA`.
   AMD and Apple still take the legacy refuse-to-eager path and do NOT have
   this 2.1x. AMD has native correctness evidence (all 15 broad cases, 130
   sites, four observed negative controls) but **no 700-step training
   qualification**. Apple has compile and host-mock evidence only.
2. **The dk/dv refusal fired ZERO times in training** on either corpus after
   the tightened guard. Its ability to refuse is proved by native fixtures,
   not by a training run. Report that as INERT in training, not as absence
   of the hazard.
3. **No per-stage identity card against the replay arm.** A grep of the
   whole evidence directory for `d_q_rope` / `stages 22-24` returns nothing.
   The training comparison is at the STEP OUTPUTS (loss, gradients,
   parameters, m, v, flags) plus the 130 native buffer-cell sites; stages
   22-24 in a real training step have not been carded against this arm.
4. **zdot replay is INERT in training** (0 sites across 8,400 layer-steps on
   enwik8 and across 24,000 at B4). Its correctness rests on its native
   fixture alone.

## 8. WHAT THIS LANE RECOMMENDS

Nothing new to open. The remaining items above belong to
`lane/lm-attention-fallback`'s record, not to a new lane
(`[[no-more-new-lanes]]`). If AMD training qualification is wanted, it is a
continuation of that lane and a Hot Aisle / RunPod MI300X leg, not a
re-derivation of the predicate.
