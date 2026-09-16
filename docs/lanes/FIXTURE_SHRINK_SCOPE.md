# FIXTURE SHRINK SCOPE (lane/identity-fixtures-light, 2026-09-16)

**Read this before recording evidence.** It names every identity_break lane
whose FIXTURE this branch may change. A lane whose fixture changes invalidates
every cell recorded at the old size, and a sabotage arm proven live at the old
size **can be inert at the new one**, so evidence for a changed lane has to be
retaken and its sabotage re-proven at the new size.

Branch `lane/identity-fixtures-light`, cut from main at bfb8f725a. Full
reasoning and measurements:
`docs/lanes/LANE_STATUS_lane-identity-fixtures-light.md`.

## STATUS RIGHT NOW: NOTHING HAS CHANGED YET

At this commit **no lane body has been edited**. Every fixture in
`tools/identity_break.py` is still at its 0.8.6 size. Evidence being recorded
against any lane at this moment is valid. This file is published ahead of the
edits so nobody has to guess.

## A. WILL CHANGE (at most these two, and only if they pass the reach test)

| lane | current fixture | why it is a candidate |
|---|---|---|
| `hdbscan` | `X[:6000, :4]` | 6000 rows through an n^2 brute-force k-NN and mutual reachability graph. Real arithmetic, 471 s on the Apple column, queue count flat, so the time is genuinely its own |
| `hdbscan-leaf` | `X[:6000, :4]` | the same fit under leaf selection, 439 s |

**These two are the ENTIRE blast radius of this branch.** Both are still
conditional: a reach sweep (in flight) is measuring whether a smaller row
count still reaches the structure the lane hashes (cluster, outlier,
Boruvka-round and condensed-cluster counts). If a smaller n collapses the
Boruvka rounds or wipes the clusters, the lane **stays at 6000** and this
bucket becomes empty.

If they do change, both get a `LANE_REVISIONS` bump, so older columns read
OWED to the next record rather than DIVERGENT, and the sabotage host build is
re-proven DIVERGENT **at the new size**, not inherited from the old.

## B. LEAVE BIG (decided, measured, will NOT be touched)

Record against these freely.

| lane(s) | why they stay |
|---|---|
| `samba-untied-dropout-accum` | load-bearing in BOTH dimensions, measured. `accumulation_is_aligned(256, 4)` is **False**, so halving rows deletes the A=4 claim the lane exists to make; and with `warmup_steps=2`, **step 3 is the first step that evaluates the cosine at all**, so cutting steps leaves the exact-rational cosine path unexercised |
| `byte-lm`, `byte-lm-resident` | no size to remove: 3 steps on three (2, 33) windows, 34,944 parameters. 5.2 s on a CPU against ~222 s per fixture on Metal, so the cost is device round trips, not arithmetic |
| `mamba1`, `mamba2`, `mamba3`, `mamba2-dtlimit` | a `(2, 16, 32)` slab, 1024 floats, at the smallest legal d_model |
| `transformer`, `transformer-window` | the same 1024-float slab |
| `samba` | `_ids(X, 6, 17)`, three steps, already minimal |
| the GBDT family, `cross-val`, `umap`, `gpc-multiclass` | their apparent cost is the **Metal command-queue leak**, not fixture size (queue counts 300 to 1809 during those groups against a flat 23 to 27 for the clean lanes). Shrinking them would be treating a runtime defect with a weaker test. Referred to lane/metal-queue-leak phase 2 |
| **every other lane of the 192** | not examined for shrinking, not touched by this branch |

## C. UNDECIDED

Empty. The two lanes in bucket A are the only conditional ones, and their
condition is stated there.

## D. NOT CHANGING: the global batch knob

`BATCH_ALONE` (16 rows evaluated alone per batch call) is the single biggest
cost lever in the harness: it is ~16 of the ~24 device calls per cell, and
lowering it changes **no recorded hash** (`_eval_batch_rows` folds only the
whole-batch bytes into the digest).

**It is deliberately NOT being changed on this branch**, because the shipped
negative control cannot judge the change. `MOJOLEARN_IDENTITY_BATCH_SABOTAGE`
perturbs the FIRST element of the whole-batch answer, which row 0 alone
already catches, so it fails identically at `alone=16` and `alone=4` and says
nothing about the rows that stopped being checked. Lowering it on that
evidence would be a verification that cannot fail. A discriminating probe
(perturb only row k > alone) is owed before anyone touches this knob.

## E. Separately: `par-*` leaves the RELEASE RECORD's scope

Not a fixture change; no `par-*` lane body is edited and no `par-*` cell
changes. It removes all 39 `par-*` lanes from what a release record RUNS, via
an explicit exclusion in `tools/identity_break.py` (today the record's scope
is implicitly "every lane the harness defines"). Two-device `par` legs and the
CPU gate both pass `--lanes` explicitly and are unaffected.

One consequence to know: all 11 CPU-covered `par-*` lanes are in
`host_surface.record_covered_lanes()`. Today `TRAINING_GPU_COLUMNS` points at
the 166-lane record, which carries them, so nothing moves. The day those
columns are repointed at a record taken under the new scope, those 11 lanes
lose their GPU columns unless they are dropped from the covered set or
admitted as OWED.

## Contact point

If you need a lane in bucket A, say so and it stays at 6000; two lanes are not
worth blocking three columns of evidence over. Buckets B and E are safe to
record against now.
