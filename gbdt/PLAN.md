# gbdt symmetric lane -- PLAN

Created 2026-09-01 by the symmetric performance lane. There was no PLAN doc
under `gbdt/` before this; the lane's wider ledger lives in the root
`NEXT_TWO.md` (port rungs, DEV 134 embargo) and `PORTING.md`. This file
carries the lane's OWED MEASUREMENTS so the orchestrator has one place to
read them. [[subagents-no-local-tests]]: this lane runs nothing; every item
below is UNVERIFIED, RUN OWED, with the exact commands.

## Measurement plan, appended 2026-09-01: DEVIATIONS 2030 + 2031

Survey and rationale: `gbdt/UPSTREAM_SURVEY_2026-09.md`. Both candidates are
comptime `-D` flags, OFF BY DEFAULT, one source both arms. The A/B pattern
is the 2026-09-01 window's (`bench/results/ab_large_2026-09-01/RESULTS.md`):
ONE checkout, arms differ only in the define; `checks/estimation_bench.mojo`
with its comptime `N_ROWS` patched IDENTICALLY in both arms to 1000000 and
then 2000000 (harness-only edit, restored after the builds -- the
orchestrator owns `checks/` and the patch); every run `nice -n 19`, box
otherwise idle, arms interleaved inside each round, two rounds minimum,
medians per cell. All numbers stay INTERNAL under the DEV 134 embargo --
no public speed claim until the soak closes.

### Gates first (cheap, must precede any timing)

DEVIATION 2030 (changed arm = the Logloss estimation path; bit-identical
claim in ALL tiers, so the identical-mode gate is load-bearing):

    # default arm sanity (nothing may move with the flag absent):
    pixi run check-fit-pointwise && pixi run check-logloss-train
    # flag arm, FAST:
    pixi run mojo run -D MOJOLEARN_2030_FUSED_EST_MOVE=1 -I . checks/fit_pointwise_check.mojo
    pixi run mojo run -D MOJOLEARN_2030_FUSED_EST_MOVE=1 -I . checks/logloss_train_check.mojo
    # flag arm, IDENTICAL (the bit-identity claim's own tier):
    tools/with_identical_mode.sh pixi run mojo run -D MOJOLEARN_2030_FUSED_EST_MOVE=1 -I . checks/fit_pointwise_check.mojo
    # (exact check-file names are the pixi task targets for check-fit-pointwise /
    #  check-logloss-train; run the pixi tasks with the define if the runner
    #  forwards -D, otherwise the mojo run forms above)

REACH, not just output ([[mojotrees-verify-reach-not-output]]): a Logloss
fit built with the 2030 define and `MOJOLEARN_STAGE_TIMES=1` must show the
`est.move` row collapse to host-arithmetic scale while `est.approx` absorbs
the pass; an unchanged `est.move` means the flag is inert and the A/B is
vacuous ([[reached-but-inert]]).

DEVIATION 2031 (fast tier only by routing; expected model-identical):

    # default arm sanity:
    pixi run check-fit-pointwise
    # flag arm, FAST -- model equality is the claim, so fingerprint A vs B:
    pixi run mojo run -D MOJOLEARN_2031_SYM_RIDX_SPLITS=1 -I . checks/fit_pointwise_check.mojo
    pixi run mojo run -D MOJOLEARN_2031_SYM_RIDX_SPLITS=1 -I . checks/logloss_train_check.mojo
    pixi run mojo run -D MOJOLEARN_2031_SYM_RIDX_SPLITS=1 -I . checks/searcher_parity_covtype_check.mojo ~/.cache/mojolearn
    # (the fixture-dir argument was missing from this line as first written;
    #  the check itself said so -- "usage: ... <fixture_dir>" -- fixed on the
    #  first run, 2026-09-01)

GATE RECORD 2026-09-01 (orchestrator, Apple M4, serial niced, at `1790aea1`):
the correctness ladder RAN GREEN in every arm. Defaults: check-fit-pointwise
(W1-W3 pass) + check-logloss-train (learn/replay/holdout/knob/soft-target all
pass). DEV 2030: fit + logloss under the define FAST both green; the
IDENTICAL arm (the bit-identity claim's own tier) green -- W2's "20
iterations identical to the bit" held with the fused kernels live. DEV 2031:
fit + logloss under the define green; searcher parity at covtype green with
mse greedy == pointwise IDENTICAL both reps; identical+define INERT-BY-
ROUTING arm green. Still owed: the two MOJOLEARN_STAGE_TIMES reach probes
(est.move collapse for 2030; sym.split -2 launches/level + equal fingerprint
for 2031) and the 1M/2M timing windows.
    # IDENTICAL with the define must be INERT BY ROUTING (ridx_only_splits_for
    # returns False): build it once and confirm byte-identical behavior --
    # a differing result under identical+define is a routing defect, not a perf finding.

REACH for 2031: `MOJOLEARN_STAGE_TIMES=1` symmetric fit -- the `sym.split`
row must lose two launches per level relative to the default build (the
reorder pair the flag deletes); the identity-trace tree fingerprint must be
EQUAL between arms on the same fixture (house rule: a flag whose fingerprint
moves is not the claimed candidate; a flag whose stage times do not move is
inert).

### The timing window (after gates)

    # builds, one checkout, N_ROWS patched to 1000000 in checks/estimation_bench.mojo:
    pixi run mojo build -I . checks/estimation_bench.mojo -o build/estbench_a_1m
    pixi run mojo build -I . -D MOJOLEARN_2030_FUSED_EST_MOVE=1 checks/estimation_bench.mojo -o build/estbench_2030_1m
    pixi run mojo build -I . -D MOJOLEARN_2031_SYM_RIDX_SPLITS=1 checks/estimation_bench.mojo -o build/estbench_2031_1m
    # repeat with N_ROWS = 2000000 -> build/estbench_{a,2030,2031}_2m
    # window: nice -n 19, interleaved a/2030/2031 within each round, 2 rounds,
    # logs to bench/results/ab_2030_2031_<date>/; alternate INSIDE one window
    # ([[mojolearn-box-drifts]] -- the box drifts 1.7x in 20 minutes).

Predictions to hold the arms against (write them down BEFORE the window):

* 2030 moves ll10 and ll1 only; rmse cells are its no-regression control.
  Expected sign: ll10 > ll1 > 0; magnitude order: a few percent of the cell
  (the deleted pass is ~40% of move+eval, and est is ~30-40% of an ll10
  tree at 1M by the 2026-09-01 window's arithmetic).
* 2031 moves all six cells; expected larger at depth 8 than depth 6
  (per-level saving x more levels), and larger at 2M than 1M (traffic-bound).
* A regression anywhere, or a fingerprint inequality under 2031, is a
  NEGATIVE: record it as DEVIATION 2030/2031 NEGATIVE in the block and
  revert nothing (the default is already off) -- the flag simply never
  flips ([[mojotrees-switches-must-flip]] governs the flip, and only a
  measured bit-identical/model-identical win flips it, in the
  orchestrator's session, after the DEV 134 embargo allows quoting).

### Standing owed items this lane inherits (not new)

* DEV 134 soak pair + greedy positive control (root `NEXT_TWO.md`) -- the
  embargo gate over every symmetric speed claim.
* AMD/NVIDIA legs for anything the window certifies on Apple
  ([[one-box-verdict-is-not-three]] -- name the column, record unrun as OWED).
* The `use_pointwise_searcher` default decision (root `NEXT_TWO.md` open
  decision 2) -- untouched by this round; the bench's default arm is greedy.
