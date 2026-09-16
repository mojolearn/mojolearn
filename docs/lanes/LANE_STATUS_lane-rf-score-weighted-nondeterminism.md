# lane/rf-score-weighted-nondeterminism

**Status 2026-09-16: THIS IS A SHIPPED-PRODUCT DEFECT, MEASURED ON THE PUBLISHED 0.8.5
WHEEL FROM PyPI. It is not a 0.8.6 release gate.** Reproduced, localized to one node, three
of my hypotheses falsified by my own experiments. NO named cause. NO fix.

Ten MI300X legs, $1.55 total, every box verified deleted.

## HEADLINE: users are affected today

`pip install mojolearn==0.8.5` on one MI300X, the lane's own fit configuration:

| arm | setting | result |
|---|---|---|
| **cols16_SHIPPED_DEFAULT** | `max_features=1.0`, the default | **13/300 moved (4.3%)** |
| cols10_CONTROL | `max_features=0.625` | **0/300 stable** |

All 13 moved fits produced **distinct** models (`distinct=13`), which is a race, not two
attractors. The control is clean, so the probe is sensitive rather than quiet.

Provenance: `version 0.8.5`, `vendor hip`, `pip_install_0.8.5_exit=0`,
`rf_so_sha256 7ca786abdaabfebcfa91f8055c7f98700bda366fda94a527ce8354143a935c0e`.

**A user on MI300X calling `RandomForestRegressor(...).fit(X, y)` with >=11 features and no
special configuration gets a different model about 4% of the time.** `max_features=1.0` is the
shipped default, so >=11 features is enough on its own. Eleven features is not exotic.

### How far back it goes

- **The fit path is semantically unchanged since v0.8.5.** In the `v0.8.5..db9047b9f` diff,
  every changed line of `split.mojo` is COMMENT TEXT from the "retire the port framing" sweep
  (`MIRRORS` -> `Reference:`, `transcribed` -> `matches the reference`), and in `builder.mojo`
  the only line matching the blocking loop, the cap or `dimy` is likewise a comment.
- **Every tag v0.8.0 through v0.8.5** carries the same `N_BLKS_FOR_COLS`, `_publish_to_global`
  and `update`. (Re-run under bash with a control token that correctly reports 0. The FIRST
  sweep was a zsh `for`/`if` loop that reported "v0.8.5 does not have N_BLKS_FOR_COLS" while a
  direct `git show` of the same file at the same tag printed line 78. A silently empty result
  is indistinguishable from a pass.)
- The published 0.8.5 wheel ships `mojolearn/hip/gfx942/identical/_mojolearn_rf.so`.

## The defect

1. **The clf/reg asymmetry is NOT an output-shape artifact.** Over 80 fits the classifier's
   **model** is bit-stable in all five exported arrays. The defect is regressor-specific.
2. **The weighted metric path is exonerated.** `reg_unweighted` passes no weights and moves.

Exactly ONE node per occurrence, at depth 6, its depth-7 children differing as a downstream
cascade; the competing splits are on different FEATURES. `Split::update` gives an equal-gain
tie to the HIGHER `colid`, so flips in both directions are not a tie-break bug. Depth 6 is
VISIBILITY (~31 bootstrap rows per node), not location. Tree SHAPE can also move (node counts
6812 vs 6814), seen in the STOCK binary, so it belongs to the defect.

## WHAT I GOT WRONG, three times

| hypothesis | committed in | killed by |
|---|---|---|
| The trigger is a SECOND `find_best_splits` launch | `ce9e4ffd9`, `061f9162b` | `cap16/11` is ONE launch and moves |
| The variable is column count >= 11 | `241fa7a59` | leg 8 `cap16/16` 0/100 ... which was a FLUKE |
| A PARTIAL final column-block group | `2e70de849` | `cap8/16` is `[8,8]`, both FULL, moves 10/300 |

The second is the instructive one: I flagged leg 8's `cap16/16` 0/100 as ~7% likely by luck
**and retracted a hypothesis on it in the same message**. Leg 9 ran it at 300 and got 4/300.
**Do not conclude from a single underpowered zero, including when you labelled it underpowered.**

## What survives: EXPOSURE, not mechanism

Leg 9, 300 per arm, three provably distinct binaries:

| arm | cols | launches | moves |
|---|---|---|---|
| stock (cap 10) | 16 | 2 | 16/300 (5.3%) |
| cap8 | 16 | 2 | 10/300 (3.3%) |
| cap16 | 11 | 1 | 5/300 (1.7%) |
| cap16 | 16 | 1 | 4/300 (1.3%) |
| any cap | <=10 | 1 | 0/600 |

Launch count **modulates** the rate (1 launch 1.50%, 2 launches 4.33%, p = 0.003, ~2.9x) and
does not gate it. Column count and launch count both change **how many near-tied split
candidates get resolved per fit**: two exposure proxies for one thing, which is why each
looked like the answer until tested against the other.

**I can expose this race and dial its rate. I cannot say why it happens.**

## NEXT: the trace needs NO REBUILD

I said twice that this needed a rebuild, then that it did not. The chain settles it:
`fit_forest` does `var instr = FitInstruments()` (LIVE, not `disabled()`),
`FitInstruments.__init__` builds `IdentityTrace()`, and `IdentityTrace.__init__` sets
`self.path = getenv("MOJOLEARN_IDENTITY_TRACE")` with `enabled = path != ""`. So an RF fit
through the binding emits a stage trace from an ENV VAR alone.

- `_compute_split` already records `tree<id>.batch<n>.round<r>.cols<col>.hist` before the
  split launch, so histograms are already checkpointed.
- Records append-and-close per record, and `IdentityTrace()` is constructed per `fit_forest`
  call, so a DISTINCT path per fit from Python gives one clean trace per fit, no subprocesses.
- `tools/identity_trace_diff.py a.trace b.trace` names the FIRST diverging stage.

**The decisive first cut, with no code change at all:** run fits until one diverges, then diff
its trace against a stable one.
- Every `.hist` matches but the model differs -> histograms are identical and the divergence
  is in SPLIT SELECTION, i.e. the merge: stale read vs dropped write-back.
- A `.hist` differs first -> accumulation, upstream of the merge.

Only if that is inconclusive is the splits `record_device` needed, and it is now trivial:
`self.splits` is a `DeviceBuffer[DType.uint8]` exactly like `self.histograms`, so one call
after the split launch with a live count of `n * size_of[Split]` (NEVER the `max_batch_size`
capacity -- `record_device`'s own docstring warns that hashing the tail "differs run to run on
ONE machine").

CAUTION: `record_device` DRAINS the queue. A traced run changes concurrency and may mask the
race, so the traced build must be shown to still reproduce before any null is interpretable.

## Ruled out, each with a reason

- **Weighted metrics** - `reg_unweighted` moves.
- **Label scale** - serial Float64 host fold over the caller's own buffer after
  `ctx.synchronize()`; `choose_scale` snaps to a power of two, every fixture far from a boundary.
- **Float reduction ordering** - Int32 fixed point plus UInt32 counts, relaxed **integer**
  atomics; no dither on this path.
- **Histogram zeroing extent** - global zero is bytes scaled by the real bin type; shared zero
  uses typed slot indexing. Correct at 4 and 8 bytes.
- **Compare-then-skip H2D caches** - guard by CONTENT on an in-order queue; also excluded by
  leg 2's K=1 arm.
- **The pipeline** - K=1 moves 3/80 vs K=4's 4/80; `n_streams` makes slots, not streams.
- **Wavefront-64 grouping** - DEVIATION 404 pins the reduce to 32 lanes;
  `eval_best_split_pinned` read end to end and is sound.
- **Device-property launch shapes** - no `get_attribute`, occupancy or CU query on the path.
- **Cross-device peer copy** - single-device box.
- **My diagnostic define** - `cap16 @ 10 cols` 0/100 matching stock, and leg 9's three digests
  distinct, so the cap arms are valid evidence.

## Method notes for the next remote experiment

- **A lone zero proves nothing.** Leg 8's 0/100 and leg 3's 0/59 both misled.
- **The `.so` digest comparison is the authoritative guard** that a build define took effect.
  Grepping the build log is NOT: `build_rf.sh` does not echo its command line.
- **A stale R2 object at a reused key** served an old leg's numbers to a new leg's monitor.
  Detected only because arm labels carry the fixture and column count. Use a fresh key, and
  print a distinctive token so staleness is visible.
- **`cmd | tail -N || fallback` tests `tail`'s exit status, not the command's.** An R2 push
  that failed with "unknown key" silently skipped its fallback.
- **Verify a fix in the GENERATED artifact, not the template.** Legs 6 and 7 both died from
  changes that never reached what ran.
- **`odd/17` is measured INSENSITIVE** and was dropped rather than reported as a null.

## Evidence (outside the repo)

    ~/mojolearn-evidence/rf-score-weighted-blocker/leg-{1..9}/remote/identity/
    ~/mojolearn-evidence/rf-score-weighted-blocker/leg-10-shipped-085/remote/identity/

## Boxes and cost

| leg | VM | deleted | cost | outcome |
|---|---|---|---|---|
| 1 | `8796a71c` | 204 / 404 | $0.15 | reproduced, stage localized |
| 2 | `68ef24e1` | 204 / 404 | $0.05 | pipeline excluded |
| 3 | `f03ec215` | 204 / 404 | $0.05 | node named |
| 4 | `bcce7c94` | 204 / 404 | $0.05 | threshold found |
| 5 | `3df6a953` | 204 / 404 | $0.05 | boundary straddled |
| 6 | `aa2f3ccb` | 204 / 404 | $0.05 | LOST, wheel-only import |
| 7 | `4b164b21` | 204 / 404 | $0.15 | launch reading broken |
| 8 | `33266e44` | 204 / 404 | $0.05 | define proven sound |
| 9 | `c1640753` | 204 / 404 | $0.10 | exposure, not mechanism |
| 10 | `06c3af5a` | 204 / 404 | $0.05 | **SHIPPED 0.8.5 DEFECT CONFIRMED** |

Balance $32.78 to $31.23, **$1.55 total** of the $5.00 authorized. One box at a time.
Provider billing LAGS by roughly $0.10 per leg: each leg's own line is the delta visible
at its teardown, and the running total settles upward for a few minutes afterwards.
