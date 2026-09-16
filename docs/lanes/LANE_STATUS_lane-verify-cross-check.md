# Lane status: lane/verify-cross-check (2026-09-16)

`python -m mojolearn verify --cross-check`: compare **this machine's GPU
against its own CPU**, so a user can check our central claim without trusting
our recorded table, our hardware, or us.

Branched off main at `a809d92f2`, after the four proven pieces of
lane/expose-inference-surface had landed.

## Why this is the strongest of the three checks

| check | what it asks you to trust |
|---|---|
| `--cross-check` | **nobody**: you generated both sides on your own machine |
| `--all` | our recorded table, which is auditable because the raw columns are committed |
| `--self-test` | nothing, but it only shows the comparison *can* fail |

Comparing your machine against our columns requires believing we recorded
honestly. This requires believing nothing: the lane is fitted once on your GPU,
then the same fitted model answers the same held-out rows twice, from the GPU
estimator and from the saved model reloaded through the CPU host binding. It
also works on any GPU mojolearn supports, not only the three vendors we
recorded, which answers "I do not own an H100".

## The design property that matters

**There is one digest implementation and no second comparison path.** The lane
is fitted once; `fit.probe(gpu_est)` gives one answer and
`fit.probe(host_model(saved))` the other, both hashed by the harness's own
`_h`. The only difference between the arms is which binding answers, so the two
sides cannot drift into disagreeing for a reason that is not arithmetic.

The guard from `_probe_fit_host` is kept: the binding that answered must be the
one named. That check exists because a forest sabotage column once read
IDENTICAL on a RunPod pod when the binding loaded was not the one asked for —
the cross-check was comparing a thing against itself.

## Two axes, not one

- **`infer`** is cross-vendor identity: same input, different hardware, same
  bits.
- **`batch`** is batch invariance: same row, different batch neighbours, same
  bits. A serving system batches dynamically, so a user whose prediction
  changes with traffic has a real problem, and this is the one place they can
  test both at once on their own machine.

A lane declaring `n/a` for batch keeps it. `umap` does exactly this
(`n/a:batch-dependent-by-contract`), and inventing a comparison there would be
a check that cannot fail.

## Measured on the Apple M4

    RESULT: YOUR GPU AND YOUR CPU AGREE on 9 of 9 compared cell parts,
    across 5 of 24 lanes (19 SKIPPED, listed above) and 1 fixture(s), in 22.7s.

Per lane, streamed as it ran:

    elasticnet                    base  batch=agree infer=agree   1.71s
    elasticnet-l2end-no-intercept base  batch=agree infer=agree   1.25s
    knn                           base  batch=agree infer=agree   1.27s
    ols                           base  batch=agree infer=agree   0.27s
    umap                          base  batch=n/a   infer=agree   8.48s

Both axes are genuinely exercised: `batch` compared on four lanes, and `umap`
kept its declared `n/a`.

### 22.7s over 5 lanes is NOT the real default runtime

Nineteen of 24 lanes did not compare, and the reasons are **entirely
environmental to this box**, not defects in the cross-check. Stated plainly
because a 5-lane agreement should not be read as a 24-lane one:

| why | lanes | detail |
|---|---|---|
| host binding I never built | 5 | the `gp*` lanes want `_mojolearn_gp_infer_host.so` |
| identical-tier GPU binding absent | 9 | `mixture`, `embedding`, `hdbscan`, `ivf` were never built here |
| **stale Metal build** | 5 | binaries built Sep 13 20:41; `arima` source changed Sep 15 19:17 and `svm` Sep 15 13:38 |

The staleness is confirmed by timestamp, not inferred: a binary two days older
than its source is exactly what produces `takes 7 positional arguments but 8
were given` (arima) and `svc_fit: params must contain 8 values, got 10`. A
properly built install compares far more than five lanes, and rebuilding four
GPU bindings to chase a bigger number here would cost hours of the single Metal
device for a figure a normal install produces for free.

## Scope, and why the default is capped

The intersection is every lane with both a shipped GPU path and a shipped host
family: **all 79 declared inference lanes are reachable from a binding the
wheel already carries.** None falls outside.

- `quick` — one lane per family, base fixture. 10 lanes, seconds.
- default — up to **24** lanes, minutes.
- `all` — the whole 79, and on Apple **refused by design**.

The cap is not taste. `identity_break.refuse_routine_apple_column` refuses more
than `APPLE_COLUMN_LANE_LIMIT = 24` lanes in one Metal process, because a full
Apple column is a per-release artifact that blocks the machine's only GPU for
seven hours. The cross-check respects that rule rather than tripping over it.

## Three defects found in my own work, by running it

1. **`RESULT: MISMATCH. 0 of 0 cells differ.`** The first GPU run compared
   nothing (a host binding was absent) and reported a mismatch about hashes it
   never computed. Nothing compared is not a mismatch: it now reads NOTHING
   COMPARED and exits 4.
2. **`--cross-check` ran the entire 39-lane suite.** I wired it into
   `_wants_suite` but gave it no early-return branch in `cmd_verify_all`, so it
   fell through. It burned two minutes of CPU before I killed it.
3. **A confident headline over a 5-of-24 run.** The skipped count now rides in
   the RESULT line itself, with "a skipped lane was NOT checked", rather than
   sitting in a section above it.

## A reversal: `--all` does NOT take the GPU implicitly

I first folded a `quick` cross-check into every `--all` run, so one artifact
would carry all three checks. That was wrong. On a GPU box it makes a
documented command **acquire the GPU as a side effect**, and two runs at once —
or a run beside a gate — would contend for the single Metal device, which is
the concurrency that previously returned NaN, constant and zero outputs in two
lanes. `--all` now records that the cross-check was not run **and how to run
it**, which is not a pass, and `--cross-check` stays the explicit door.

## A latent interaction this lane's parent introduced

`test_shipped_verifier_hashes_like_the_harness` fits
`public_reference_lanes()` in one process. That was 9 lanes; the 2026-09-16
promotion made it **39**, over the 24-lane Apple cap, so the test became
unrunnable on any Mac with a GPU build. It still passed on CPU-only, which is
why it went unseen until a Metal tree ran it. Fixed by capping the list on an
Apple GPU; the parity it checks is per lane, so a subset proves the same thing.
Confirmed both ways: fails at 39 on Metal, passes capped (30s), and passes
uncapped on CPU-only (110s).

## Tests

Five new tests, all passing, each encoding a property I proved by hand rather
than leaving it as something I once observed:

- a mismatch is reported and **both** differing hashes are printed;
- `compared == 0` reads NOTHING COMPARED, never MISMATCH;
- a CPU-only install says so and does not print an agreement;
- a `batch` `n/a` is respected, not counted as agreement;
- the default scope stays inside `APPLE_LANE_CAP`.

## `verify --compare`: two strangers, with us out of the loop

The honest gap in every other check is that we published the reference table.
Nobody outside has rerun our lanes on their own hardware, and we cannot
manufacture that. But the evidence document makes it unnecessary: two people on
different hardware each run `verify --all --json-out mine.json`, swap files,
and run `verify --compare`. If the hashes match they have demonstrated the
claim **to each other**, with us absent. That is stronger than anything we can
publish about ourselves, and it cost one pure function over two JSON files —
no GPU, no bindings, no network.

**Three outcomes, not two.** A cell present in one document and missing from
the other is INCOMPARABLE, never an agreement. Counting absence as a match is
exactly how a comparer becomes unable to fail, and this one exists for
adversarial use, so that is the worst possible place for it. A part both sides
record as `n/a` is an absence they agreed on, counted separately.

**Measured, all four behaviours seen rather than assumed:**

    two vendors, same hashes   -> AGREE, exit 0, "two independent machines"
    one cell altered           -> MISMATCH, exit 1, names ols/base infer and BOTH values
    a cell in only one doc     -> INCOMPLETE, exit 4, "Absence is not agreement"
    both docs the same device  -> AGREE but WARNS it shows repeatability, not identity

It also runs with no numeric mode, no host bindings and no GPU set, which is
the point: a third party has none of ours. `--compare` dispatches before the
import and tier checks for that reason.

Five tests encode those properties.

## Owed

- [ ] A run on a **properly built GPU install**, where far more than 5 of 24
      lanes compare. This box cannot produce it without hours of Metal.
- [ ] The main checkout's Metal bindings are two days stale; anything using
      them for `arima`, `svm` or `gpc` will mislead until rebuilt.
- [ ] `--cross-check` is not yet in the evidence document as a live third
      check, only as a recorded "not run, and how to run it".
