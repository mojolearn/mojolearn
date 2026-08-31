# E1 — the first cross-vendor identity run (Apple ↔ AMD)

The run `IDENTITY_PATHS.md` names as its exit criterion. Everything above
the Apple column is construction plus transcription until this executes;
after it, "bitwise identical across GPUs" is a measured sentence for every
path that matches and a named ledger row for every path that does not.

**The claim under test:** the same fixed-seed fit, at the same commit,
under `NUMERIC_IDENTICAL`, produces byte-identical models and predictions
on an M4 (Metal) and an MI300X (HIP/CDNA).

## Preconditions

- **Same commit on both sides.** Cards from different commits are not
  comparable; record `git rev-parse HEAD` into every artifact directory.
- AMD box: AMD Developer Cloud MI300X droplet (activation:
  developer.amd.com profile → GPU Droplets console → Create; the $100
  credit is ~50 MI300X hours). `pixi.toml` already locks `linux-64`.
- Both sides flip `original/numerics.mojo:74` to `NUMERIC_IDENTICAL`
  for the whole session and NEVER commit the flip (`git diff` must show
  only that line; revert before ending). On AMD this is not optional:
  the hist-2 shared-slice layouts carry `comptime assert LANE_WIDTH ==
  32`, so an AMD FAST build is a compile error by design, not a silent
  regroup.
- Artifacts land in `bench/results/e1/<date>/{mac,amd}/` with the commit
  hash and `record_environment.sh` output beside them.

## Phase 0 — AMD smoke (minutes)

    pixi install
    pixi run check-hardware-matrix

Confirms the toolchain, the GPU is detected, and the AMD column meets the
identity floor. First contact with a transcribed column WILL surface
gaps (a wrong shared-limit transcription, an import that only Metal has
exercised). Each is a finding, not a failure of the method: fix, note the
matrix row it corrects, continue.

## Phase 1 — characterize the vendor (row 10's precondition)

    pixi run check-ieee-arith

Run FIRST on any new backend column, per IDENTITY_PATHS row 10. This
measures the actual FTZ/denormal and contraction behavior of the AMD
compile. Expected: CDNA honors denormals where Metal flushes — exactly
the divergence `numerics.ftz` exists to align. If the ftz-model arm
reproduces AMD's behavior the way it reproduced all 53,041 Metal
divergences, rows 9/10's constructions are validated on the second
vendor. Record the output verbatim.

## Phase 2 — the gates, IDENTICAL build, both machines

    pixi run check-depthwise          # claim 6: one model across core counts
    pixi run check-lossguide-policy   # P1-P8, tie/NaN rules
    extratrees/tools/check.sh         # 29 checks, integer core
    # ensemble (RF): builder/split/objectives/quantiles/forest/philox checks
    pixi run check-random-strength    # R1/R3a on the pinned noise path
    pixi run check-linalg-identity    # rows 27-32: the matrix products,
                                      # PCA, tSVD and OLS, in BOTH modes
    pixi run check-linalg-column-invariance   # one card across three columns

Every gate that is green on the M4 must be green on the MI300X. A red
gate here is a within-machine defect on AMD (a launch bug, a transcribed
constant) and is fixed BEFORE any cross-machine diff — diffing against a
machine that fails its own gates teaches nothing.

## Phase 3 — the card diff (the actual E1)

**Mojo-only cards (E2).** Four training paths have no Python surface and so fall outside `tools/e2_matrix_fit.py`: depthwise growth, lossguide growth, `MultiClassOneVsAll`, and the feature-parallel searcher. `tools/e2_mojo_cards.sh <out_dir>` emits one card each (`gbdt_depthwise.card`, `gbdt_lossguide.card`, `gbdt_multiclass_ova.card`, `gbdt_feature_parallel.card`; one fit per file, hashed fixtures that are pure functions of constants, machine-independent tags) via `original/e2_growth_cards.mojo` (pixi task `e2-growth-cards`), runs the set a second time into `<out_dir>/control/` as the run-to-run control, and writes `<out_dir>/e2_mojo_cards.json` (name, record count, description, `control_match`, numeric mode). A card whose control does not match under FAST is read only under IDENTICAL. The feature-parallel card is OUTPUT-LEVEL (splits + docBins; that searcher has no in-searcher trace plumbing). Diff them exactly like the Python cards below.

**Linear-algebra cards (rows 27-32).** `core/gemm.mojo` has no Python
binding either, so the matrix products and OLS produce their own cards:
`tools/check_linalg_column_invariance.sh` emits `gram`, `nt` and `gemv` from
`bench/linalg_trace_main.mojo`, and `pixi run ols-card` emits the eleven-stage
OLS card plus a run-to-run control beside it. Diff them with the same differ.
Read the `nt` arm's three tags together: `nt.narrow`, `nt.wide` and
`nt.wider` are the SAME logical rows at three launch widths, so a divergence
among them on ONE machine is a batch-invariance failure and a divergence of
all three between machines is an ordinary vendor difference. The OLS card's
two INTEGER stages (`step3.info`, which carries the Jacobi sweep count, and
`step4.rank`) come before any float stage on the ladder for the usual reason.

Same fixture, same seed, both machines:

    MOJOLEARN_IDENTITY_TRACE=<out>.card  <the traced fixture run>
    # hold n_streams equal across arms; one fit per card (the differ
    # refuses multi-fit files)

    python3 tools/identity_trace_diff.py mac.card amd.card

Work the differ's ladder: integers before floats. If the integer stages
(borders, cindex, colids, partition) diverge, STOP — that is a code-path
or RNG difference, bigger than any numeric row, and must be resolved
first. Only then read the float stages. On the first float divergence,
re-run with `MOJOLEARN_IDENTITY_TRACE_DUMP=<tag>` on both sides for the
per-cell ULP/denormal classification.

## Phase 3u — the unsupervised cards (rows 19-26)

`tools/e1_traced_fit.py` drives the GBDT/ET/RF families through the Python
bindings, and `cluster/`, `neighbors/` and `dbscan/` have no bindings, so
until 2026-08-23 the unsupervised half of the ledger had no way to produce
a card at all. `bench/unsupervised_trace_main.mojo` is that way: one arm
per process (the differ refuses a card whose sequence numbers restart), a
fixture that is an integer-exact function of a constant seed, and an input
hash printed before the fit so both machines can be proven to have fitted
the same bytes before anything else is compared.

    sh tools/e1_unsupervised.sh

on EACH machine. It records the resolved column and lane width, runs
`check-ieee-arith` and `check-portable-translog` (row 10's precondition and
row 12's certificate line), runs the six local gates under IDENTICAL, then
writes `kmeans.card`, `knn.card`, `dbscan.card` and their console hashes
into `bench/results/e1u/<stamp>-<host>/`. Every card's mode is READ BACK
from the run rather than assumed from the flip (DEVIATION 514).

It is separate from `e1_bootstrap.sh` because it needs none of the five
binding builds -- there is no Python surface for `cluster/`, `neighbors/`
or `dbscan/`, which is why rows 19-26 had never been to E1 at all -- so it
runs in minutes on a box that has nothing but pixi.

Then diff each `<arm>.card` against the other machine's. Compare
`input.*` FIRST; a card diff against different inputs measures nothing.

**The Apple leg is already committed** at
`bench/results/e1u/2026-08-23_065611-MacBook-Air-1-terrabyte/`, so an AMD
box only has to produce its own directory and diff against that one.

Read the k-NN card knowing what its stages are. `knn.out_dist` /
`knn.out_idx` are PRE-SORT and record an arm's internal order, which the
two arms genuinely disagree about (the tiled selector does not sort);
`knn.sorted_dist` / `knn.sorted_idx` are what `kneighbors` returns and are
what a cross-vendor claim is a claim about. Two runs whose sorted
DISTANCES agree and whose sorted INDICES do not have chosen different
members of an equidistant class and have diverged in the selector and
nowhere else.

DBSCAN records three stages and no per-batch ones deliberately: at the
default budget the batch count comes from the device's FREE MEMORY, so
`batch03.core` exists on one machine and not the other. The batch count is
in the card HEADER (a comment, skipped by the differ) and the invariance
those records would have tested is gated directly by
`check_dbscan_batch_count_invariance`.

**Run `pixi run check-column-invariance` BEFORE renting anything.** It
compiles the same three arms against the APPLE, NVIDIA and AMD columns on
the local device and requires one card across all of them. Every defect it
catches is a defect the rented box would have found more slowly and more
expensively, and it caught two on the day it was written (a k-NN arm pinned
to a kernel that refuses on a 64-lane column, and a DBSCAN invariance check
that had never verified its own lever moved).

## The train-here-infer-there leg (models cross machines, not just hashes)

Phase 3's driver also saves each fitted model beside its card as
`<name>.model.npz` and records the file's sha256 under the fit's `model`
entry. The npz carries floats as raw bytes, never as decimal text, and
its file bytes are a pure function of the model, so a bit-identical fit
gives a bit-identical model file. Ship the out_dir to the other machine
and run

    PYTHONPATH=python python3 tools/e1_cross_infer.py <out_dir>

It rebuilds the E1 inputs from the seed with the driver's exact recipe,
loads each saved model, predicts, and writes `e1_infer.json`. A
`predictions` hash there equal to the fitting machine's `e1_fits.json`
entry is the one-line claim that fit on A plus load and predict on B
reproduce the prediction bits. `tools/check_serialization.py` is the
local gate for the round trip itself, fresh-process load plus a sabotage
arm proving predict consumes the loaded file.

## Phase 4 — end-to-end harness pair (optional, needs the dataset)

    bench/external/run_gbm_bench.sh year 100 <pair>

Diff the `hashes` section between the Mac JSON and the AMD JSON. A
matching `predictions` hash on the mojolearn arm is the one-line form of
the claim. (`test_target` must match first — it proves both sides scored
the same rows.)

## What to expect, honestly (state of 2026-08-22, 18-row ledger)

EXPECTED IDENTICAL under IDENTICAL:
- **ET fits** (Gini/MSE regimes; integer score core, rows 13/14/16 closed)
  — avoid the algo-L sampler regime (rides row 12) and note row 18's
  host-libm dispatch: macOS vs glibc `log` can flip the sampled-column
  COUNT at knife edges, which the `colsamples` trace tag will show as an
  integer-stage divergence, not a numeric one.
- **RF fits with Gini/MSE/InverseGaussian** (rows 16/17, DEVIATION 405).
  Entropy/Poisson/Gamma REFUSE by name under IDENTICAL — an error on both
  machines IS the correct result (first wired REFUSE).
- **GBDT depthwise/lossguide RMSE** (no transcendental in the loss; hist
  family, score path, ties all closed).

EXPECTED DIVERGENT, at named rows (a divergence HERE confirms the ledger;
one anywhere ELSE is a new row and the real finding of the run):
- Logloss/Poisson/Quantile GBDT fits — row 12 (device `exp`/`log`).
- The symmetric arm — rows 9/10 residue (`_add_leaf` seams, part-stats).
- Configs reaching `exact_estimation` or `partitions_reduce` — row 8's
  last two producer sites (ride the parked DEVIATION 250 patch).

## Rules of the run

1. One variable: the device. Same commit, same mode, same seeds, same
   fixture sizes.
2. Every divergence gets a ledger row before it gets a fix.
3. The mode flip is never committed; the artifacts always are.
