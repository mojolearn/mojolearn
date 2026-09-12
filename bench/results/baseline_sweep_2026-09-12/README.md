# The first internally consistent board: every classical and tree lane, one pod

**One commit, one pod, one driver, one set of library versions, arms
interleaved inside every cell.** That is the entire point of this run, and it
is the property no previous board in this repository has had.

## Why a consistency sweep was worth a lease

The board before this one was a patchwork of rows taken on different pods, on
different dates, at different commits, against different drivers. That
patchwork produced a wrong headline. The published CatBoost symmetric-taxi row
(709.0 ms, `bench/OPPONENT_REFERENCE.md:2109`) turned out to be a slow sample:
re-measured on another pod the same arm ran 623.8 ms, so a published "0.437x"
was really 0.525x. Opponent columns are measured to drift about 10% pod to pod
(`opponent-columns-drift-pod-to-pod`).

A cross-lane sentence -- "KDE is our worst gap", "we beat CatBoost and lose to
XGBoost" -- is only meaningful if every cell in it came off the same machine in
one heat window. Until this run, no such set of cells existed.

## Provenance: the tuple this board is valid for

| | |
|---|---|
| pod | RunPod `9imx21xh3yuqd8`, SECURE, $3.49/hr |
| GPU | NVIDIA H100 80GB HBM3 (81559 MiB) |
| driver | 580.126.09 |
| CUDA | runtime 12090, driver API 13000 |
| CPU | Intel Xeon Platinum 8480+, 224 visible cores |
| image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| commit shipped | `06341dbf` (branch `lane/baseline-sweep`) |
| Mojo | 1.0.0 (ed45d567) |
| our tier | IDENTICAL, witnessed per estimator (`mode=identical vendor=cuda`) |
| cuML | 26.08.00 (cupy 14.2.0) |
| CatBoost | 1.2.10 |
| XGBoost | 3.2.0 |
| LightGBM | 4.7.0, **CPU build only** -- no CUDA tree learner |
| scikit-learn | 1.9.1 |
| numpy / scipy | 2.4.6 / 1.17.1 |
| torch | 2.4.1+cu124 |
| pyarrow | 23.0.1 |

**The board was produced by `06341dbf` plus three repairs made by hand on the
box**, because they were discovered by running it: the pixi bootstrap, the
`/root/bins/all` population, and the harvester's log-path fix. Their committed
forms are `abf8e340` and `df380a57`. Stated plainly rather than implied away:
the tree that produced these numbers is `06341dbf` with those three patches
applied, not the branch tip.

## The lease, decided before renting and not at minute 61

Rentals here carry a **one-hour** dead man by default and this sweep needed
roughly 3.4 hours, so the bound was chosen deliberately up front:
`tools/trees_leg.sh rent --gpu "NVIDIA H100 80GB HBM3" --minutes 240`.

That path takes a longer bound (unlike `tools/gemm_remote_leg.sh`, which hard
refuses anything over 60 and is the reason the one-hour figure is in the air).
The watchdog is armed **before any work**, runs ON the pod, and terminates the
pod through the API at the deadline, so the box cannot outlive the work even if
this machine disappears. Recorded in the lease file:

    pod='9imx21xh3yuqd8'  armed_utc='2026-09-12T21:11:28Z'  minutes=240

240 minutes = the ~3.4 h of work plus margin for setup and the owed items. One
pod, one segment; no second pod, because a second pod would defeat the run.

## Method

* **Size: full, not the floor.** ENGINEERING_RULES section 9 sets 1,000,000
  rows as a FLOOR for trees, and almost every published tree row sits exactly
  on it while the classical lanes already run at 4,000,000 x 11 and
  2,043,304 x 220. This sweep runs `--rows full`.
* **The real shapes, which are not the requested ones.** `--rows full` asks for
  taxi 5,750,000, and that ceiling never binds: the taxi npz holds 5,750,086
  rows, so after the fixed 500,000-row test tail the regression task has
  5,250,086 train rows and the classification task -- card-paid trips only --
  has **4,110,786**. Every tree line carries the truth in its `shape=` tag
  (`taxi-4110786x16`, `istella-2043304x220`) and the board reports that, not
  the flag.
* **Rounds:** 5 timed, one warm-up excluded, medians with min..max.
* **Interleaved within each cell**, one round each in rotation, so a box that
  throttles mid-run moves every arm together and the ratio survives what an
  absolute number does not.
* **Never two timed arms at once**, and **no sharding** -- `--shard` is now
  refused by default, because splitting lanes across pods is precisely the
  patchwork this run exists to end.
* **Datasets from Cloudflare R2**, staged and verified against committed
  size+sha256 pins in **45.6 s** (taxi 419,757,252 B; Istella-S 2,248,281,826 B
  pre-decoded, skipping the ~18 min LETOR parse). No refetch, no re-decode.
* **Our IDENTICAL arm against the opponent's FAST arm**, always. The FAST tier
  is trees-only, ships for the Mac, and is **not** raced here.

### Arms per lane, and why some lanes have no opponent

| lane | opponent(s) asked for |
|---|---|
| gbdt-symmetric | catboost-gpu (CatBoost only -- no symmetric grower elsewhere) |
| gbdt-depthwise | catboost-gpu, xgboost-gpu |
| gbdt-lossguide | catboost-gpu, xgboost-gpu, lightgbm-cuda |
| rf | cuml-rf-gpu |
| et | sklearn-et-cpu, with `--devices cpu,gpu` |
| iforest | cuml-iforest-gpu |
| kmeans / pca / knn | cuml-gpu, torch-gpu |
| ols | cuml-gpu, torch-gpu, torch-gpu-eigh |
| kde / svc / dbscan | cuml-gpu |
| hdbscan | cuml-gpu **only** -- this library ships no HDBSCAN |

`et` is the one lane where an NVIDIA box legitimately admits a CPU arm: no GPU
ExtraTrees exists on any vendor, so scikit-learn on all cores is its only legal
opponent, and the arm name carries `-cpu` so the table cannot hide it.
Everywhere else the vendors' CPU arms are refused BY NAME under GPU-PATH-ONLY;
those refusals appear in the board as `REFUSED`, which is the point -- an
opponent that did not run must be visible as a refusal, never as a blank.

**DBSCAN's parameters are now in version control.** `_dbscan_params` raises
when `MOJOLEARN_CTD_DBSCAN_<DATASET>` is unset, and there is no default, so
every dbscan cell of every previous sweep died at `ready` unless an operator
happened to export it. The values the published rows were measured at lived
ONLY in a pod shell: taxi `0.177,10`, Istella-S `4.17,10` (eps is the p75
quantile of the 10th-nearest-neighbour distance on the standardized block).
They are committed in `tools/bench_all_ours.sh`.

## Twelve faults in a script that had never been executed

`tools/bench_all_ours.sh` was written and never run once. Faults 1-8 were found
by reading it; 9-11 were found only by putting it on a rented H100; 12 was
found by running the harvester against real logs instead of its own fixtures.
Each was reproduced against the unfixed side before being changed.

1. **The classical half could never run.** `race` opens `<block>-<dataset>.json`
   before anything else and nothing ever called `prep`; all 16 classical cells
   died with `FileNotFoundError`.
2. **hdbscan exits by name** under `--arms ours`: this library ships no HDBSCAN.
3. **dbscan has no scikit-learn arm**, so `--opponents` sending `sklearn-cpu`
   exits by name.
4. **Worst: `--opponents` sent `catboost-gpu,xgboost-gpu` to every tree lane.**
   Neither name is in the rf, et or iforest roster, so all three lanes refused
   every opponent, the opponent list went EMPTY, and the cell ran ours-only
   while the TSV still recorded three arms. A cell that reads as a race and is
   not one is exactly the failure this sweep exists to remove.
5. **Every dbscan cell died at `ready`** -- see the parameters above.
6. **An Istella-S DBSCAN round takes ~337 s against a 300 s default**, which
   killed the first published attempt at round 1. Per-cell round bounds now.
7. **Per-cell output directories** made aggregation impossible; one `ctd/` now.
8. **`--rows full` recorded a no-op ceiling as fact** (see The real shapes).
9. **The tree harness needs a named binary set and nothing built one.**
   `speed <set>` refuses unless `/root/bins/<set>` exists and copies the `.so`
   FROM it; twelve tree cells exited 2 in milliseconds.
10. **pixi is not on the image and nothing said so.** `pixi install` returned
    127, then all 18 `bindings/build_*.sh` returned 127 in zero seconds: a
    "successful" setup phase with not one `.so`, and a sweep that timed a
    library which was not there.
11. **`--arms` filters opponents only**, so naming `ours` there printed
    `FSPEED-REFUSED arm=ours` beside a perfectly good ours row.
12. **The tree logs are not under `--out`.** The harvester searched
    `<out>/speed` while `trees_identical_ab.sh` writes to its own
    `/root/trees_out/speed`, reporting all twelve tree cells as UNKNOWN while
    the logs sat one directory up, fully populated.

`tools/bench_all_summarize.py` is new. `sweep.tsv` records WALL SECONDS PER
CELL, which is a fact about the harness and not about the library; the
harvester turns the `FSPEED` and `CTD` lines into one row per
(lane, dataset, arm) with median, min..max, opponent ms, ratio, quality, output
hash and the fit-equivalence verdict. **A cell that did not run is UNKNOWN,
never blank and never zero**, and `UNAVAILABLE`/`UNKNOWN` are carried through
as themselves rather than upgraded.

## What this run does not claim

* **It moves no default and computes no flip verdict.** Verdicts stay with
  `tools/flip_verdict.py` and the section 9 rule.
* **No cost-of-determinism figure.** That quantity needs our deterministic arm
  against our own optimized non-deterministic arm, which does not exist; the
  claim is withdrawn project-wide.
* **Our FAST tier is not raced.** It is trees-only, aimed at the Mac, and is
  not an optimization target here.
* **Span asymmetry is named, not corrected.** cuML and torch arms are timed
  with inputs already device-resident while ours uploads and validates inside
  the call. That runs AGAINST us, it is reported on every classical
  `CTD-RATIO` line as `span_asymmetry=`, and no work was moved into or out of
  any clock to flatter either side.

## Results

`board.tsv` / `board.md` in this directory, harvested by
`tools/bench_all_summarize.py`. Cells that did not run are listed as UNKNOWN
with the reason, and are not rounded to green.
