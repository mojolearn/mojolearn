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
promotion made it 39 and lane/ship-cpu-host-families took it to **122**, far
over the 24-lane Apple cap, so the test became unrunnable on any Mac with a GPU
build. It still passed on CPU-only, which is
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
publish about ourselves, and it cost one pure function over two JSON files, no
GPU, no bindings, no network.

### The only interesting question is how it could agree wrongly

A comparer has one failure mode, and it is not reporting a mismatch that is not
there. It is reporting agreement where two machines never computed the same
bits. This command exists to be pointed at us, so that is the whole of the
work. Five such routes were found, and **each was watched producing `AGREE,
exit 0` on the shipped code before the code that catches it existed**
(`ARM0_the_unfixed_side.txt`, the comparer as it stood at `bb579ecb8`, run on
the same files the fixed one is run on):

| the forgery | pre-fix | post-fix |
|---|---|---|
| two real runs where 24 cells' `value` is null, because the probe RAISED on both boxes | `AGREE` exit 0, 73 agreements | `INCOMPLETE` exit 4, all 24 named |
| one document copied under a second name | `AGREE` exit 0 | `SAME DOCUMENT` exit 4 |
| the losing row appended a second time carrying the other party's value | `AGREE` exit 0 | `MALFORMED` exit 2, both rows named |
| both documents recording `MOVED` for a cell | `AGREE` exit 0 | `SELF-CONTRADICTED` exit 1 |
| both holding the same hash for a cell one of them judged `DIVERGENT` | `AGREE` exit 0 | `AGREED ON A DIVERGENT ANSWER` exit 1 |

The duplicated-row arm is a **working forgery** against the old code: the hash
in `dupB.json` was flipped, and appending the honest row after it was enough to
make the mismatch disappear before anything was compared, because the dict
build was last-wins.

The null arm is the one that matters most, because it needed no forgery at all.
Two ordinary `verify --quick` runs on this box produce it, and 24 of the 73
"agreements" were lanes that ran on neither side.

### Held against the CLI, not just the function

The exit code is the interface: two parties compare in a script and what the
script branches on is `$?`. `test_cli_compare_exit_codes_are_what_a_stranger_scripts_against`
runs six pairs through the real process and asserts the code and a distinctive
token in stdout, under **both** `MOJOLEARN_NUMERIC_MODE=identical` and `fast`,
and asserts `RESULT: AGREE` appears in none of the non-zero cases. `fast` is in
there because every other check refuses with exit 3 under it, measured side by
side in `arms_9_to_12.txt`; `--compare` dispatches before that gate because a
third party has none of our bindings.

Every path out prints exactly one `RESULT:` line and returns a documented code,
including a crash, which is caught so it cannot exit 1 and be read as MISMATCH.
A truncated file, a non-JSON file and JSON that is not an evidence document
each refuse by name.

### It inherits the verdict fix rather than reproducing its bug

`87085a5eb` fixed `verdict()` the same morning: it returned VERIFIED as soon as
ONE part read IDENTICAL, before it looked at REFUSED, so a CPU-only install
printed `VERIFIED, exit 0` over 44 identical and 288 refused parts. A comparer
reaches the same place through agreement, so two properties are now explicit:

* **`AGREE` is the last outcome tried.** No number of agreements outranks one
  problem.
* **The exit-1 outcomes are read before the exit-4 ones**, exactly as
  `verdict()` reads DIVERGENT before REFUSED: a wrong answer outranks an
  absent one.

Each document's own verdict and detail line are printed beside the comparison,
because two parties can agree while one of them checked a fraction of what a
reader assumes. `ARM3.txt` shows the whole chain on this box: each run's own
verdict is `INCOMPLETE (verified 48 of 80 cell parts ... 24 refused)`, and the
comparison on top of two of those is `INCOMPLETE` as well, not a pass.

### No second lane set

`compare_documents` takes its lanes from the two documents. It never
enumerates, greps or imports a lane list, so it cannot grow a second idea of
what the lane set is, which is how one afternoon produced four different lane
totals. Where a lane list **is** needed, `--cross-check` reads it from the
registry by import (`host_surface.FAMILIES` against `harness.LANES`), the same
way `tools/lane_select.py` and `tools/verification_matrix.py` read it. Asked
after the 2026-09-16 merges: registry 211 lanes, `public_reference_lanes()`
122, wheel bindings 32, cross-check intersection **79**. The 79 is unchanged by
the thirty-two-family merge because it counts declared *inference* lanes.

### Numbers printed, never counted

Every differing, self-contradicted, uncomputed, one-sided and differently-`n/a`
cell is printed by name with both values, and a truncated listing says how many
it hid rather than stopping quietly.

## What this box could not produce

**A genuinely independent pair.** Both documents in the proof come from this
one Mac, so every real `AGREE` here carries the `WARNING: both documents
describe the SAME device ... repeatability, not cross-hardware identity` that
the command prints for exactly this case. The cross-vendor path is exercised in
the unit tests and the mismatch arms, not on two real machines. Manufacturing a
second party would defeat the only thing this command is for, so it is left
owed rather than faked.

**A complete install.** `python/mojolearn/host` in this worktree is a SYMLINK
to `~/mojolearn-evidence/expose-inference/hostprod`, another lane's build of
**10 of the 32** host families. Nothing there is committed and nothing should
be. Its `_mojolearn_forecast_host.so` was checked against the `kpss_test` entry
`87085a5eb` moved into `bindings/kpss_host_test.mojo` that morning: it carries
it, so this copy is current for what main changed, not stale. The probe that
said otherwise was `nm -gU | grep -c kpss`, which returns nothing for Mojo
entries and would have put "stale" into the record; the import probe was
validated first against `holtwinters`, a name known to be present, before its
answer about `kpss_test` was believed.

So the 49 agreeing cell parts are a statement about the comparer, not about
mojolearn's coverage. The 24 refusals in the same run are the missing
`identical` GPU binding in this worktree.

## Owed

- [ ] Two documents from two genuinely different machines. Nothing on one Mac
      can stand in for it.
- [ ] A run on a **properly built GPU install**, where far more than 5 of 24
      lanes cross-check. This box cannot produce it without hours of Metal.
- [ ] `--cross-check` is not yet in the evidence document as a live third
      check, only as a recorded "not run, and how to run it".

## Evidence

`~/mojolearn-evidence/verify-cross-check/compare-proof-2026-09-16/`

| file | what it holds |
|---|---|
| `ARM0_the_unfixed_side.txt` | the five forgeries against `bb579ecb8`, all `AGREE` exit 0 |
| `ARMS_all.txt` | the same files through the fixed command, arms 1, 2, 4 to 8, 13 |
| `ARM3.txt` | two real runs, each `INCOMPLETE` itself, compared to `INCOMPLETE` |
| `arms_9_to_12.txt` | the `fast` tier contrast, truncated file, non-JSON, not an evidence document |
| `partyA.json`, `partyB.json` | two real `verify --quick` documents from this box |
| `cleanA/B`, `flippedB`, `shortB`, `dupB`, `movedA/B`, `divA/B`, `cleanA_copy` | the arms |
