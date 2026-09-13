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

### The GBDT half, full size, one pod, 5 rounds interleaved

Ratio is OURS median / OPPONENT median; below 1 means our median is lower.
`fit` is the total leaf count each arm actually BUILT, read back through that
library's own API after the last round.

taxi, shape `taxi-4110786x16`:

| policy | ours ms (min..max) | CatBoost GPU | ours/CB | XGBoost GPU | ours/XGB | leaves ours/CB/XGB | verdict |
|---|---|---|---:|---|---:|---|---|
| symmetric | 795.8 (791.3..812.5) | 1699.2 (1662.1..1723.6) | **0.468** | n/a (no oblivious grower) | - | 6400 / 6400 / - | COMPARABLE |
| depthwise | 1127.7 (1117.1..1136.4) | 1779.8 (1737.1..1817.3) | 0.634 | 1011.3 (976.1..1031.6) | 1.115 | 5903 / 5811 / **3883** | NOT-COMPARABLE (0.342) |
| lossguide | 1651.5 (1645.2..1660.2) | 2005.4 (1991.5..2046.8) | 0.824 | 1114.0 (1081.8..1169.2) | 1.482 | 6239 / 4438 / **3883** | NOT-COMPARABLE (0.378) |

Istella-S, shape `istella-2043304x220`:

| policy | ours ms (min..max) | CatBoost GPU | ours/CB | XGBoost GPU | ours/XGB | leaves ours/CB/XGB | verdict |
|---|---|---|---:|---|---:|---|---|
| symmetric | 2223.9 (2187.7..2325.7) | 2236.1 (2209.2..2269.7) | **0.995** | n/a | - | 6400 / 6400 / - | COMPARABLE |
| depthwise | 2931.1 (2867.9..2954.0) | 2446.7 (2415.4..2589.2) | 1.198 | 3122.9 (3102.6..3192.6) | **0.939** | 6345 / 6364 / 6297 | COMPARABLE (0.011) |
| lossguide | 3494.9 (3481.1..3521.6) | 3248.9 (3230.4..3453.5) | 1.076 | 3124.6 (3061.0..3357.0) | 1.118 | 6396 / 6397 / 6297 | COMPARABLE (0.016) |

Quality is within about a thousandth everywhere and ours is ahead of CatBoost
on every symmetric and depthwise cell (taxi symmetric logloss 0.524213 against
0.524485; Istella depthwise 0.126973 against 0.127425).

**Our output hash is byte-identical across all five rounds in every cell.
CatBoost's alternates between two values inside a single cell.** XGBoost's is
stable.

### What this board changes about the XGBoost claim

`bench/OPPONENT_REFERENCE.md:2150-2152` says, merged on main:

> XGBoost is FASTER THAN US wherever it competes: 1.196x and 1.039x on
> depthwise, and 1.995x on taxi lossguide, which is the largest single gap on
> the board.

All three of those numbers were measured at the 1,000,000-row FLOOR. At full
size, on one pod, every one of them moves toward us and one reverses:

| cell | published @1M | this board @full | |
|---|---:|---:|---|
| depthwise taxi | 1.196x | 1.115x | gap narrows |
| **depthwise Istella-S** | **1.039x** | **0.939x** | **REVERSES -- we are faster** |
| lossguide taxi | 1.995x | 1.482x | gap narrows sharply |
| lossguide Istella-S | 1.256x | 1.118x | gap narrows |

Our ratio against XGBoost improves with scale in ALL FOUR cells, which is the
signature of a fixed cost being amortized rather than a faster inner loop. So
the claim is true at the floor and not in general, and a board that only ever
ran at 1,000,000 rows could not have seen it.

**Two qualifications that the fit-equivalence readback forced, and neither is
visible in a timing column.**

1. **XGBoost's lossguide column is not an independent measurement.** It builds
   a model IDENTICAL to its own depthwise one on both datasets -- taxi 7,666
   nodes / 3,883 leaves in both lanes, Istella-S 12,494 / 6,297 in both -- with
   byte-identical quality to match. The lane pins `max_depth=6` and
   `max_leaves=64`, and 64 = 2**6, so the depth cap binds first and leaf-wise
   growth degenerates to level-wise. Reporting the two XGBoost cells as
   separate evidence would be counting one measurement twice.
2. **The taxi cells where XGBoost looks fastest are NOT LIKE-FOR-LIKE.** It
   built 3,883 leaves against our 5,903 and CatBoost's 5,811 -- 34% fewer --
   which trips the 10% leaf-spread tripwire in both taxi policies. A smaller
   ensemble is a smaller job. The honest ground for an ours-versus-XGBoost
   comparison is Istella-S, where all three arms agree to within 1.6%, and
   there we lead depthwise (0.939x) and trail lossguide (1.118x).

None of this is a defect in XGBoost and none of it is corrected here. It is
reported because a ratio whose two sides built different-sized models cannot be
read as a like-for-like result, and until this run nothing in the harness read
the models at all.

### Istella-S symmetric moved to parity, and that is unflattering

Published 0.830x at 1M; **0.995x here at 2,043,304 rows** -- parity, not a win,
on identical 6,400-leaf ensembles. It is recorded exactly as measured. A
consistency sweep that only confirmed the flattering rows would not have been
worth the lease.

### The forest half, full size, same pod, same window

| lane | dataset | ours ms (min..max) | opponent | opponent ms | ratio | leaves ours/theirs | verdict |
|---|---|---|---|---|---:|---|---|
| rf | taxi | 2242.1 (2222.5..2297.9) | cuml-rf-gpu | 4256.2 (4209.3..4435.2) | **0.527** | 1,718,168 / UNAVAILABLE | UNKNOWN |
| rf | Istella-S | 2249.5 (2242.9..2258.9) | cuml-rf-gpu | 5894.7 (5754.3..5992.9) | **0.382** | 1,550,974 / UNAVAILABLE | UNKNOWN |
| et | taxi | 5975.0 (5953.3..6039.5) | sklearn-et-cpu (224 cores) | 21323.5 (21209.4..21928.1) | **0.280** | 881,399 / 906,974 | COMPARABLE |
| et | Istella-S | 9290.9 (9278.4..9352.7) | sklearn-et-cpu (224 cores) | 35426.0 (34998.3..35500.7) | **0.262** | 1,029,236 / 1,006,311 | COMPARABLE |
| iforest | taxi | 147.3 (128.7..148.6) | cuml-iforest-gpu | 360.1 (332.1..1003.3) | **0.409** | UNAVAILABLE both | UNKNOWN |

**The `rf` verdicts are UNKNOWN and that is not a pass.** cuML's Python forest
exposes no node or leaf accessor and this build answered `source=none`, so
nothing could be compared: `FSPEED-FIT-VERDICT ... verdict=UNKNOWN reason=fewer
than two arms exposed a leaf count; an unread comparison is not a fair one`.
The leaf count was NOT backfilled from `n_estimators`, which is the config we
asked for and says nothing about the fit. `tools/cuml_forest_json_probe.py`
(owed item 1, run after the sweep) settles whether that is a missing accessor
on this build or a reader bug.

`iforest` reads UNAVAILABLE on BOTH arms for a documented reason: our
`IsolationForest` does not keep its forest -- `fit` builds it, scores one row
and discards it, and every later scoring call rebuilds (DEVIATION 874/1836) --
so there is no fitted ensemble to read back on either side.

`et` is the one lane whose opponent is a CPU arm, and it is labeled `-cpu`
throughout: no GPU ExtraTrees exists on any vendor, so `cuml-et-gpu` REFUSES BY
NAME ("cuML has no ExtraTrees estimator: its RandomForest searches quantile
splits, not the uniform-random thresholds that define ExtraTrees"). A
GPU-versus-CPU ratio is a different claim from the GPU-versus-GPU ones above
and must not be read as the same kind of number.

### Determinism, observed rather than asserted

Across every cell on this board, **our output hash is byte-identical on all
five rounds**. Two opponents are not:

* `catboost-gpu` alternates between two hash values INSIDE a single cell, on
  every GBDT cell.
* `sklearn-et-cpu` changes its prediction hash between rounds on both datasets,
  despite `random_state=7` being pinned on every arm.

This is reported as measured and is not a price-of-determinism figure: that
quantity needs our deterministic arm against our own OPTIMIZED
non-deterministic arm, which does not exist, and the claim is withdrawn
project-wide.

### A RATIO AGAINST A SOLVER THAT DID NOT SOLVE IS NOT A SPEED RESULT

The single most important thing the classical half produced, and it is a
validity finding rather than a timing one. **OLS on Istella-S, 2,043,304 x 220:**

| arm | median ms | R2 | finite | is the ratio meaningful? |
|---|---:|---:|---|---|
| ours | 1318.9 | **0.3319** | yes | -- |
| cuml-gpu | 85.3 | **-6473.68** | yes | **NO** |
| torch-gpu | 95.1 | **NaN** | **no** | **NO** |
| torch-gpu-eigh | 9.7 | 0.1516 | yes | yes, but a much worse fit |

`CTD-RATIO` prints `ours/cuml-gpu=15.4645` and `ours/torch-gpu=13.8699` for
this cell. **Neither is a speed result.** cuML returned a fit 6,473 times worse
than predicting the mean, and torch's `lstsq` (driver `gels`, QR without
pivoting, which assumes full rank) returned NaN outright. Istella-S has 45
near-constant columns; a solver that assumes full rank does not survive them.
Being fast at not solving the problem is not a number this board will quote as
a gap, and the ratios are printed only because the harness prints every ratio
it computes -- they are struck here, by name.

On taxi every arm solves and agrees to six digits (R2 ~ 0.90884 for all four),
so **taxi's `ours/cuml-gpu=7.7455` IS a real gap** and is the honest OLS number
on this board. The `torch-gpu-eigh` arm -- the same algorithm class as ours --
solves both datasets and is far faster than us on both (85.2x on taxi, 135.4x
on Istella-S), at a materially worse Istella fit (0.1516 against our 0.3319).
That is where our OLS work has room, and it is visible only because quality was
computed beside every cell.

### The classical half, and what is inside each clock

`CTD-RATIO` is OURS median / OPPONENT median. Every GPU opponent is timed with
its inputs ALREADY DEVICE-RESIDENT, while ours uploads and validates inside the
timed call, because that is what a caller of our public surface pays. The
harness names the gap on every line rather than equalizing it.

| lane | dataset | ours/cuml-gpu | ours/torch-gpu | opponent upload EXCLUDED from its clock |
|---|---|---:|---:|---|
| kmeans | taxi | 1.535 | 1.163 | cuml 128.3 ms, torch 19.6 ms |
| kmeans | Istella-S | 7.344 | 9.244 | cuml 1122.4 ms, torch 194.0 ms |
| pca | taxi | 1.318 | 19.321 | cuml 127.4 ms, torch 19.4 ms |
| pca | Istella-S | 4.669 | 42.982 | cuml 1431.8 ms, torch 617.4 ms |
| ols | taxi | 7.746 | 2.503 | cuml 140.4 ms, torch 21.6 ms |
| ols | Istella-S | *struck, see above* | *struck* | cuml 1046.7 ms, torch 194.6 ms |
| knn | taxi | 2.385 | **0.696** | cuml 19.6 ms + fit before its clock |
| knn | Istella-S | 2.304 | 2.331 | cuml 270.2 ms + fit before its clock |

**The asymmetry is not a rounding detail.** On kmeans/taxi cuML's excluded
upload (128.3 ms) is LARGER than its entire timed median (126.6 ms), and on
pca/Istella-S it excludes 1431.8 ms. cuML's kNN also fits before its clock
starts and the harness reports that magnitude as `unmeasured` rather than
printing a dash that could be read as zero. Every one of these runs AGAINST us:
our published classical gaps are worse than the code deserves. Nothing was
moved into or out of any clock to improve them.

**kNN taxi is a win: ours 22.5 ms against torch-gpu 32.3 ms (0.696x)**, at
equal tie-aware recall@10 (0.99915 against 0.999325, measured here against a
float64 NumPy brute force, not against an arm's own scorer). On Istella-S at
d=220 torch pulls ahead (2.331x) and its recall is better (0.93835 against our
0.923025), which is a real quality gap and not only a speed one.

### KDE and SVC: the widest gaps on the board, on identical answers

| lane | dataset | ours ms (min..max) | cuml-gpu ms | ratio | quality ours vs theirs |
|---|---|---|---|---:|---|
| kde | taxi | 29.0 (28.2..30.1) | 2.48 (2.36..2.60) | **11.72** | mean log-lik -9.582072 vs -9.582051 |
| kde | Istella-S | 63.8 (62.0..69.1) | 7.16 (6.31..7.49) | **8.91** | -212.117468 vs -212.117465 |
| svc | taxi | 774.3 (774.2..775.3) | 422.1 (420.0..450.3) | 1.83 | accuracy 0.7675 both; support 5527 vs 5586 |
| svc | Istella-S | 61.4 (60.8..67.9) | 20.0 (19.8..23.3) | 3.07 | accuracy 0.9222 both; support 2400 vs 2401 |

**KDE is our worst lane and the two arms compute the same density** -- the mean
log-likelihoods agree to eight significant figures on both datasets, with zero
rows lacking a density on either side. So this is a pure speed gap on an agreed
answer, which is the cleanest kind of gap to have and the kind worth working
on. SVC likewise: identical accuracy on both datasets and support-vector counts
within 1.1%, so its 1.83x and 3.07x are like-for-like.

**KDE/taxi is the starkest span asymmetry on the whole board.** cuML's timed
median is 2.48 ms while the upload it does NOT time is 7.876 ms -- the work
excluded from its clock is more than three times the work inside it -- and it
also fits before the clock starts, a magnitude the harness reports as
`unmeasured` rather than as a dash that could be read as zero. Ours uploads,
validates and fits inside its 29.0 ms. The 11.72x is therefore an upper bound
on the real gap, it runs against us, and it is published in that shape rather
than quietly adjusted.

Both arms are `digest_stable=True` in all four of these cells; the
non-determinism seen elsewhere on this board is specific to `catboost-gpu`,
`sklearn-et-cpu`, and cuML/torch on kmeans and pca.

### DBSCAN/taxi: the largest margin on the board, on the same clustering

1,000,000 x 11, standardized, `eps=0.177 min_samples=10` -- the parameters that
until today lived only in an operator's shell.

| arm | median ms | clusters | noise fraction | agreement with ours |
|---|---:|---:|---:|---|
| ours | **1142.6** | 2900 | 0.216119 | -- |
| cuml-gpu | 13619.4 | 2900 | 0.216119 | ARI **0.9999999991**, noise agreement **1.0** |

**`ours/cuml-gpu = 0.0839` -- about twelve times faster, and it is the same
answer**: identical cluster count, identical noise fraction, and an adjusted
Rand index of 0.9999999991 against our labels. The span asymmetry is present
and trivial at this scale (cuML excludes 37.2 ms of upload against its own
13,619 ms), and it still runs against us.

This is the widest gap on the board in either direction and the cleanest, since
a speed claim is only interesting when both sides agree about the answer -- and
here they agree to nine decimal places on the clustering itself, not merely on
a summary metric.

### DBSCAN REVERSES WITH DIMENSION, AND BOTH ENDS ARE ON THIS BOARD

The Istella-S DBSCAN cell was PENDING in the published table: the first attempt
lost its `ours` arm at round 1 to the 300 s default round bound (fault 6). With
the bound raised it ran, and it is the sharpest single result here.

| dataset | dims | ours | cuml-gpu | ratio | agreement |
|---|---:|---:|---:|---:|---|
| taxi | 11 | 1,142.6 ms | 13,619.4 ms | **0.084 (12x faster)** | ARI 0.9999999991 |
| Istella-S | 220 | **335,403.3 ms** | 54,791.4 ms | **6.121 (6x slower)** | ARI 0.9999993801 |

Same algorithm, same parameters, same agreed clustering on both -- identical
cluster counts (2900, 1501), identical noise fractions, noise agreement 1.0 --
and the ratio swings by a factor of **73** between d=11 and d=220. Our default
is the random ball cover, whose pruning degrades as dimension rises until the
cover stops excluding candidates; cuML's brute force has no such cliff. A board
that ran only taxi would have published a 12x win and missed that entirely.

Neither number moves a default here. What they jointly say is that the `rbc`
default is a low-dimension default, and that is a finding for the DBSCAN lane
to act on, not for this sweep to act on.

### The board in one page

68 ok, 22 REFUSED, **0 UNKNOWN**, 0 PARTIAL across 90 (lane, dataset, arm)
rows. Every refusal is POLICY, not failure: the vendors' `*-cpu` arms under
GPU-PATH-ONLY, `lightgbm-cuda` (the wheel has no CUDA tree learner), and
`cuml-et-gpu` (cuML ships no ExtraTrees). `hdbscan` has no ratio because this
library ships no HDBSCAN -- an absence by construction, recorded as such rather
than as a gap.

Ours against the best legal opponent in each cell, `ours/opponent`:

| we lead | | | we trail | | |
|---|---|---:|---|---|---:|
| dbscan | taxi | **0.084** | kmeans | taxi | 1.535 |
| iforest | Istella-S | **0.098** | kmeans | Istella-S | 7.344 |
| et | Istella-S | 0.262 | pca | taxi | 1.318 |
| et | taxi | 0.280 | pca | Istella-S | 4.669 |
| rf | Istella-S | 0.382 | ols | taxi | 7.745 |
| iforest | taxi | 0.409 | ols | Istella-S | *struck (opponents unsolved)* |
| gbdt-symmetric | taxi | 0.468 | knn | Istella-S | 2.304 |
| rf | taxi | 0.527 | svc | taxi | 1.834 |
| gbdt-depthwise | taxi (vs CatBoost) | 0.634 | svc | Istella-S | 3.072 |
| knn | taxi (vs torch) | 0.695 | kde | Istella-S | 8.905 |
| gbdt-lossguide | taxi (vs CatBoost) | 0.824 | kde | taxi | 11.719 |
| gbdt-depthwise | Istella-S (vs XGBoost) | 0.939 | dbscan | Istella-S | 6.121 |
| gbdt-symmetric | Istella-S | 0.995 (parity) | gbdt-lossguide | taxi (vs XGBoost) | 1.482 |

**The split is not random and it is the one this repository already predicted.**
We lead every tree and forest lane on both datasets, and we trail the families
whose inner loop IS a BLAS call. `ENGINEERING_RULES.md` 0b-iii says exactly
that about Apple silicon -- "k-means, k-NN, PCA, SVD, OLS, UMAP, GP and ARIMA
are precisely the families whose inner loop IS a BLAS call, so that margin is
spent against AMX and there is nothing left to win" -- and this board is the
NVIDIA evidence for the same claim, against cuBLAS and cuSOLVER instead of AMX.
Tree fitting is scatter-gather over integers and calls no BLAS, which is why
those lanes look completely different.

Every one of the classical ratios above is additionally an UPPER BOUND, because
the GPU opponents are timed with inputs already device-resident while ours
uploads and validates inside the call (see the span table). Two of them --
cuML's OLS and torch's OLS on Istella-S -- are struck entirely, because a
solver that returns R2 = -6473 or NaN is not a faster solver.

## Owed item 3: DEVIATION 2634 on criteo is INERT, so the old 2.1% was never 2634

`OPPONENT_REFERENCE.md:2274-2320` recorded a 2634 A/B that measured a clean
2.1% separation and refused to claim it, because nothing observed whether the
branch executed. The worry was specific: 2634 gates on
`len(dependent_configs) > 0 and ctr_prep_wanted`, and if `dependent_configs`
were EMPTY on criteo then neither build built the prep and the gap belonged to
something else. Re-run here with the marker in the log, 1,000,000 rows, 3
rounds, ours-vs-ours:

| build | `.so` sha256 | marker | median ms |
|---|---|---|---:|
| ctr_on (shipped) | `9e354d27…` | `dependent=3 cat_columns=26 ctr_prep_wanted=True prep=ran gate_2634=on` | 14,264.5 |
| ctr_off (`-D MOJOLEARN_2634_CTR_PREP_OFF=1`) | `17246d59…` | `dependent=3 cat_columns=26 ctr_prep_wanted=True prep=ran gate_2634=off` | 14,180.4 |

**`dependent=3`, not 0 -- so the feared explanation is ruled out. And `prep=ran`
on BOTH SIDES, which is the actual answer: 2634 is behaviorally INERT on
criteo.** The deviation skips the CTR target prep when NO column is
categorical; criteo declares 26, so the skip never fires and the two builds do
identical work. Everything downstream agrees: identical quality (logloss
0.128083, AUC 0.742472) and **the identical model hash `70e7ff4f27045344`
across both builds and all six rounds**, while the compiled binaries genuinely
differ. The 0.6% by which ctr_on is SLOWER is run-to-run noise, not a
mechanism.

So the published 2.1% cannot have been 2634 either -- not because the branch
never ran, but because the gate cannot change the outcome on the one dataset
chosen to exercise it. **2634 is measurable only where there are NO categorical
columns**, which is exactly where its 0.9603 flip verdict was taken (taxi and
Istella-S). criteo can price the CTR PATH; it cannot price this switch.

**A correction to this lane's own tooling.** `criteo_2634_ab.sh` prints a
legend enumerating two outcomes -- `prep=ran` on both, or `dependent=0` on both
-- and the case actually observed is a third one it did not name: the gate
compiles differently, `dependent>0`, `prep=ran` on both, and the behaviour is
nonetheless identical because the skip condition is false. The legend is
incomplete and is recorded here as incomplete rather than quietly reinterpreted
to fit.

Nothing here moves a default. criteo is not a section 9 gating dataset, both
arms are ours, and the finding is that a switch is untestable on this fixture
-- which is a statement about the fixture, not a verdict on the switch.
