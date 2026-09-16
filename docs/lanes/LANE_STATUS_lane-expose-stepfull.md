# lane/expose-stepfull, 2026-09-16

Evidence: `/Users/andrewhendel/mojolearn-evidence/expose-stepfull/`.

## The gap

`python/mojolearn/_verify_reference.py:57` read
`PARTS = ("train", "infer", "model", "batch")`. `stepfull` was not in it, so
`python -m mojolearn verify --all` could not compare the part and **a user
could not check the decode property at all**, although the project could and
had: `lane/stateful-cpu-decoding` and `lane/decode-columns` proved it on FOUR
columns for all eight decode lanes, every cell `IDENTICAL x4`, merged as
`dd438aa4a`. The evidence existed; the door did not.

The property is the one incremental decoding rests on. A sequence decoded ONE
TOKEN AT A TIME with a carried state must be, at every position, the bits the
same model answers when the whole sequence runs as one fresh-state forward
pass. It is also where bitwise determinism usually breaks, which is why
recording it is not the same as letting a stranger ask it themselves.

## THE DECISION, and how to reverse it

**Andrew, 2026-09-16: do it now.** The reason it had not been done is a
sequencing constraint, not a doubt: a part with no value in any table row
reads OWED everywhere, so the `PARTS` change only lands WITH a table
regeneration, and a regeneration is a release-sized act. `release/0.8.7`
already requires all four record columns to be retaken, so the regeneration
costs nothing extra today and costs a full re-record if deferred. The same
reasoning landed the UMAP fixes and the shrink reversals on the same day.

To reverse it: revert the three commits of this branch. `stepfull` leaves
`PARTS`, the table returns to the one `429a8da9a` generated, and
`PUBLIC_PENDING_LANES` returns to thirteen `stale reference` entries. Nothing
else in the tree depends on the part.

## The regeneration, measured rather than relayed

One command, no GPU, no binding:
`python -m mojolearn verify --all --emit-reference <path>`.

    {"bytes": 585190, "cells": 1656, "classes_agreeing": {"1": 687, "2": 705,
     "3": 3136, "4": 2105}, "conflicts": 0, "parts": 6633, "records": 86}

86 records, 1656 cells, 0 conflicts, as the earlier measurement said.

**The `PARTS` change moves nothing else.** Regenerating the SAME records twice,
once without `stepfull` in `PARTS` and once with it, the delta is: **added 9,
removed 0, changed 0**, and the record list is identical. The nine are the
eight decode lanes' `base` cells plus `mlp/base`, which correctly carries
`n/a:no-decode-state`. The eight hashes are the ones
`LANE_STATUS_lane-decode-columns.md` publishes
(`f582474b00117f8e` for mamba1 and so on).

**The regeneration itself, against the table that shipped**, is the 0.8.7 bill
and is independent of this lane: records 54 -> 86, 332 cell parts added, 396
removed, 190 references changed. Of the 190, **159 are `n/a` -> a real hash**
(the transductive clusterers gained `predict`, `bootstrap` gained a batch
part), 12 are one `n/a` reason replaced by a better-worded one, and **19 are
hash -> a different hash**. All 19 belong to lanes whose FIXTURE moved this
morning (`spectral`, `samba`, `samba-untied-dropout-accum`, `mamba2-dtlimit`),
so a different hash is the correct answer for different input bytes.
**Nothing went from a hash to an `n/a`**, so no cell lost a reference in
place. `stale_reference_lanes` goes from thirteen to zero.

## Proved from the failing side

Two arms in ONE Metal acquisition, on this M4, `mamba1,transformer,ols` on
the base fixture (`proof.sh`, `clean.log`, `corrupt.log`).

| arm | mamba1 stepfull | verdict | exit |
|---|---|---|---|
| clean | `IDENTICAL` `f582474b00117f8e` | VERIFIED | 0 |
| one recorded stepfull value corrupted to `deadbeefdeadbeef` | `DIVERGENT` | MISMATCH | 1 |

The DIVERGENT line prints both values, not a count:

    DIVERGENT (1):
      mamba1/base stepfull: this box f582474b00117f8e, reference deadbeefdeadbeef

The three things that must hold, all read off the same two runs:

* a lane WITH a stepfull cell COMPARES it and can read DIVERGENT: `mamba1`
  and `transformer` both read `IDENTICAL` clean and `mamba1` read DIVERGENT
  corrupted;
* a lane WITHOUT one reads a clean `N/A`: `ols` read `N/A
  n/a:no-decode-state` in both arms, never OWED and never a silent pass;
* `verify --all` can still return a NON-VERIFIED verdict, which is the shape
  `87085a5eb` fixed and which this lane must not reinstate. It exits 1 above,
  and `test_a_run_that_refused_is_not_reported_as_verified` still holds the
  44-IDENTICAL-288-REFUSED case.

The Apple column ran the GPU classes, so `f582474b00117f8e` on Metal against
the same sixteen digits in the CPU record is a genuine cross-column
agreement, not the recording read back.

**The table was restored from a byte copy, never `git checkout --`**, and the
restore was checked by hash: `3f1ac41f0012...4635` before and after.

## The counts, before and after

| | before | after |
|---|---|---|
| lanes a CPU-only `verify --all` SELECTS | 122 | 122 |
| of those, lanes it can actually COMPARE | 109 (13 dropped as stale) | **122** (none stale) |
| decode lanes exposing `stepfull` to a user, GPU install | 0 | **8** |
| decode lanes exposing `stepfull`, CPU-only install | 0 | **5** |

The five are `transformer`, `transformer-window`, `mamba1`, `mamba2` and
`mamba3`. The other three (`mamba2-dtlimit`, `samba`,
`samba-untied-dropout-accum`) are not in `public_reference_lanes()` yet; see
below. On a full CPU-only run the new part contributes 1098 rows: **5**
compared against a real hash, **1053** a declared `N/A`, and **40** OWED.

**The 40 OWED are honest and are what is owed.** The only committed record
carrying `stepfull` ran the BASE fixture only, so the five public decode lanes
have no reference at the other eight fixtures. OWED says exactly that ("no
committed record carries this cell part yet") and it is not a pass. Recording
them is a recording job, not a code job, and it falls due with the 0.8.7
columns.

## The thirteen holds the regeneration cleared

`PUBLIC_PENDING_LANES` is checked against its sources by
`test_public_reference_lanes_are_derived_and_every_pending_reason_is_true`,
and that rule is self-clearing by design (lane/inference-coverage-complete):
regenerating the table is what makes it fire. It fired, and the split is the
one it predicted.

* **NINE lost every cell**, because every record that carried them predates
  this morning's fixture shrink: `holtwinters`, `gbdt-nan-modes`,
  `gbdt-parametric-losses`, `gbdt-lossguide-newtoncosine`, `gbdt-pair-logit`,
  `hdbscan`, `hdbscan-leaf`, `byte-lm`, `byte-lm-resident`. Their reason is
  now `no reference` and what they owe is a RECORD.
* **FOUR kept cells at the current revision**: `spectral`, `mamba2-dtlimit`,
  `samba`, `samba-untied-dropout-accum`.
* **SIX `no reference` lanes gained their first cells** from records the
  regeneration admitted: `metrics-fowlkes-mallows`, `gbdt-yeti-rank`,
  `arima-exog`, `arima-exog-seasonal`, `gbdt-categorical-ctr-tables`,
  `gbdt-tensor-ctr-tables`.

Those ten now pass every STATIC condition and owe only the one thing the
promotion rule will not skip: a CPU-only `verify --all` watched to read
IDENTICAL for them. The vocabulary had no word for that state, so it gains
one, **`unwatched`**, checked against exactly what it claims (the table must
carry cells for the lane AT THE CURRENT REVISION). **No lane is promoted
here**; `public_reference_lanes()` is unchanged at 122, and this lane cannot
watch that column because the shared checkout carries five of the thirty-two
host bindings.

Seen to fail: marking `gbdt-adapter-score-weighted`, which has no cells, as
`unwatched` fails with "the shipped table carries NO cell for it, so its
reason is 'no reference' and what is owed is a record, not a run".

## Tests

Four new, each watched failing first, plus one strengthened.

* Against MAIN's shipped table, the unfixed side, three of the four fail, the
  first with "transformer: the shipped table carries no stepfull reference".
* The fourth asserts new behavior, so it was watched failing against a
  sabotaged `_probe_stepfull` that declared `n/a:no-stepfull-part` instead of
  refusing. **An absent part must REFUSE**: an `n/a` there would make
  `stepfull` unable to fail on any lane on any install, which is the exact
  shape this repository found six of on 2026-09-16.
* `test_shipped_verifier_hashes_like_the_harness` now runs the harness with
  `--step-full` and asserts a `stepfull` part was actually compared, so the
  verifier and the harness cannot drift apart on the newest part. Run on
  Metal over `mamba1,transformer,ols`: 1 passed in 27.16s (`drift.log`).
  Without `--step-full` the harness column carries no `stepfull` key at all,
  so the row lookup raises rather than passing quietly.

155 tests in `test_host_surface.py` and 54 in `test_verify_all.py` pass.

## Not done, deliberately

* `tools/verification_matrix.py` keeps its own four-part tuple. It reports on
  sabotage movement for the maintainer document and is not the user-facing
  door; widening it would rewrite `docs/VERIFICATION_MATRIX.md` in a lane
  that is about `verify --all`.
* `rlpair` and `ragged` are still outside `PARTS`, for the same reason
  `stepfull` was: no committed column carries them. They are the next two,
  and they cost a recording rather than a decision.
