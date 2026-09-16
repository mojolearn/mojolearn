# rf-score-weighted: the mechanism is SPLIT SELECTION over IDENTICAL histograms

**2026-09-16, leg 11. The divergence is introduced inside `find_best_splits_kernel`, in the
reduction and selection over histogram data that is bit-identical between the two runs.**
Not accumulation, not feature sampling, not the partition. Still not narrowed to stale-read
versus dropped-write-back; that is leg 12 and needs no code change.

This file is the mechanism evidence. `LANE_STATUS_lane-rf-score-weighted-nondeterminism.md`
carries the defect's history, the shipped-release confirmation and the method notes.

## How it was obtained, with no code change

`instr.trace.enabled` is a RUNTIME check inside `fit_forest`, which constructs a live
`FitInstruments()` whose `IdentityTrace()` reads `getenv("MOJOLEARN_IDENTITY_TRACE")`. So the
PUBLISHED 0.8.5 wheel emits an ordered stage trace from an env var alone. `IdentityTrace()` is
built per `fit_forest` call and re-reads the variable, so a distinct path per fit gives one
complete trace per fit.

The checkpoint that mattered already existed. `.cand` (`builder.mojo:2276`) records
`_read_splits(len(st.active_items))` -- the candidate splits read back from the device for a
sampling round -- field by field, 10 u32 lanes per split: `is_valid`, `colid`, `quesval`
lo/hi, `best_metric_val` lo/hi, `global_n_left` lo/hi, `local_n_left` lo/hi. The
`record_device` call I was authorized to add was unnecessary.

**Arm 1 was UNTRACED and is the positive control: 4/100 moved, verdict REPRODUCES.** Tracing
drains the queue per record and could have masked a 4% race; without the control a traced null
would have proved nothing. Traced arm: 6/150 moved, and
`identical_trace_but_model_moved = 0` -- every moved fit had a diverging record, so the
divergence is visible in a recorded stage rather than confined to the unchecked `split[]`.

## The evidence

Six divergences, always the same shape. First differing record:

| repeats | first differing record | dtype | count |
|---|---|---|---|
| 48, 108, 142 | `tree0.batch6.round0.cand` (seq 139) | u32 | 610 |
| 99, 127, 143 | `tree12.batch6.round0.cand` (seq 655) | u32 | 610 |

The reference hash is always identical (`be63059365b47eb7`, `c18560c86172c3ff`); every
divergent hash is distinct (`744cc0d8`, `63ae6777`, `3680d89c` / `2bf164ff`, `d338bda8`,
`9f90f11d`). A race, now at stage granularity.

**What precedes the first difference, and matches bit for bit:**

    seq=124  tree0.batch6.round0.colsamples   i32  976      MATCHED
    seq=125  tree0.batch6.round0.cols0.hist   u8   624640   MATCHED
    seq=126  tree0.batch6.round0.cols10.hist  u8   374784   MATCHED
    seq=139  tree0.batch6.round0.cand         u32  610      <-- FIRST DIFFERENCE

Both column-block histograms and the feature sample for that exact round are identical. The
split results computed FROM them differ. This ordering was CHECKED, not assumed: tree0's
batch6 histograms are at seq 125-126, genuinely before seq 139, because the K=4 pipeline
interleaves trees and tree1-3's batch6 records sit at 128-138.

**The whole differing set is 8 of 693 records, a causal chain confined to ONE tree:**

    seq=139  tree0.batch6.round0.cand        <-- the flip
    seq=143  tree0.batch6.splits             <-- that batch's final splits
    seq=145  tree0.batch7.round0.cols0.hist  <-- NEXT batch: different split -> different
    seq=146  tree0.batch7.round0.cols10.hist     partition -> different histograms
    seq=159  tree0.batch7.round0.cand
    seq=163  tree0.batch7.splits
    seq=164  tree0.nodes
    seq=165  tree0.leaves

tree1, tree2 and tree3 are untouched, matching leg 3's independent finding of exactly one
diverging tree. The batch7 histogram differences are DOWNSTREAM consequences of the batch6
split flip, not a second fault.

`batch6` also matches leg 3's finding that the diverging node sits at depth 6, reached from
model arrays rather than traces. 610 lanes / 10 per split = 61 splits; 640 / 10 = 64 -- the
node counts of depth 6.

## What this rules out

- **Histogram accumulation** -- identical at the diverging round, both column blocks.
- **Feature sampling** -- `.colsamples` identical at the diverging round.
- **The partition and row order** -- everything before seq 139 matches, across all prior
  batches of all four in-flight trees.

Combined with the earlier exclusions (integer atomics, the label scale, the zeroing extent,
the content-guarded H2D caches, the pinned 32-lane reduce, `n_streams`), the divergence is
inside `find_best_splits_kernel`: the block reduction plus the mutex-guarded cross-block merge
in `_publish_to_global`.

## LEG 12: which field moves (no code change)

`MOJOLEARN_IDENTITY_TRACE_DUMP=cand` writes a `.bin` sidecar per matching record. Compare the
two `.cand` dumps lane by lane, 10 lanes per split:

- **`best_metric_val` IDENTICAL, `colid` differs** -> the gains matched and the merge chose a
  different candidate among them. That is the tie/merge path, and `Split::update`'s
  higher-colid rule should have made it deterministic, so the arrival-order-dependent step is
  implicated directly.
- **`best_metric_val` DIFFERS** -> a gain computed from an identical histogram moved, which
  means a candidate was lost or a partial reduction was read, not a tie resolved differently.

Either answer names the mechanism. The same untraced positive control must run first.

## Caveat kept from the trace tool's own header

A matching hash proves two buffers held the same bits at that checkpoint, not that the
computation was identical, and anything not hashed is invisible. The claim here is narrower
than "the histograms are correct": it is that the histogram bytes at that round were the same
in both runs, which is what makes the selection step the place the difference enters.
