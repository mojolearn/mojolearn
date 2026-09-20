# INCONCLUSIVE: the two-device par column timed out, twice

Pod `y086zpx5cd4oxg`, 2x RTX 4090 sm_89, 2026-09-20. DELETEd and verified gone
(204, then 404, then absent from the listing).

**Nothing here is creditable.** All four JSONs are `complete: false` and
`admit()` refuses every one of them with "incomplete identity_break
checkpoint". They are committed as a diagnosis, not as evidence, and the
directory name says so.

    ...par-two-device-new-since-088-2026-09-20.json        par_devices=0    48 cells
    ...par-two-device-new-since-088-2026-09-20.two-device.json  0,1         30 cells
    solo-one-rerun.json                                    0                 2 cells
    solo-two-rerun.json                                    0,1               1 cell

## What did work

**The build-ordering fix did what it was for.** `elapsed_after_column=1536`
and `elapsed_after_all_families=1537`: every family was built BEFORE the
columns, and the insurance pass fell through in one second because everything
was already in `DONE`. The previous attempt on pod `70i7hnr5avagda` spent
1130 s there and starved the deliverable. That is fixed.

**The selector fix fired.** The gate reports

    DISAGREEING LANES: par-forecast-holtwinters,par-gpc-fit,par-ivf
      -- re-running each arm SOLO before this is reported

The old selector matched `DIVERGENT|MOVED` and would have reported nothing.
These three were caught by the `ONE-COLUMN` term added today.

**AND THEY ARE NOT A DEFECT.** Read the rows:

    | par-forecast-holtwinters/base    | IDENTICAL x2 | 5e8ffa04d8c2fb3d | 5e8ffa04d8c2fb3d |
    | par-forecast-holtwinters/denormal| ONE-COLUMN   | 61084170fa566d6f | (not run)        |

`(not run)` is the two-device column never reaching that fixture before its
timeout. So the selector correctly flagged a real asymmetry and the asymmetry
is truncation, not arithmetic. A selector that fires on truncated input is
working; a conclusion drawn from it would not be.

## What the cells that DID meet say, and why it is not a claim

Where both columns reached the same fixture:

    summary:              IDENTICAL=30, ONE-COLUMN=18
    summary (infer/model): IDENTICAL=60, ONE-COLUMN=36
    summary (batch):       IDENTICAL=30, ONE-COLUMN=18

**Zero DIVERGENT anywhere.** Every cell both columns reached hashed equal,
which is what a `par-*` driver claims. It is still not evidence: `admit()`
refuses a truncated column, and a two-device column that answered two thirds
of the question is not the column the table reads.

## The on-box diff said this was fine

    par_diff_exit=0     summary: IDENTICAL=30, ONE-COLUMN=18

The body shipped commit `4caa892a9`, which carries the selector fix but
predates the `--diff` refusal (`641a8b7b2`). Run through the fixed diff at
home:

    REFUSING TO DIFF: column ... is an INCOMPLETE checkpoint (complete:false)
    carrying 48 cell(s).

Third artifact today where the pre-fix diff reported a clean summary over
truncated columns, and the second where a conclusion was available to be drawn
from it.

## Why it timed out, and what would fix it

Nine `par-*` lanes over nine fixtures at two repeats, run twice, is simply
more than a 60-minute lease holds alongside a full build. Leg 4's measurement
on the same hardware: six lanes over nine fixtures took 784 s for ONE column.
`--minutes` is capped at 60 in `gemm_remote_leg.sh`, so the lease cannot be
extended; the work has to shrink instead — fewer fixtures, fewer lanes per
leg, or a warmed binding cache so the build costs minutes instead of half the
budget.

Not attempted again here: the instruction was to commit what the legs give and
stop, and a third attempt at the same shape would have been a third way to
spend a lease rather than a new answer.
