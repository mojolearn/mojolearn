# Apple FAST trees pass: the GBDT lanes nobody has optimized (written 2026-10-02 ~22:30Z by the M3 manager)

Paste this whole file as the prompt for a new cloud session. Rules from `docs/apple-fast/NEXT_PASS.md` all apply (read it
first): you write code only and run nothing; base `origin/main`; one branch per family `lane/apple-fast-<family>`; every
change compiled only under `comptime if GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()` behind a
`-D MOJOLEARN_<NAME>` define; IDENTICAL compiles main's code unchanged (state the static proof in each commit); FAST is GPU
only (no host loops, no NumPy/sklearn in the Python layer, no device-to-host round trips inside an iteration, no
one-thread/one-block launches over runtime sizes, no env reads); quality held within FAST run-to-run spread; never
`--no-verify`. Request M3 runs with `docs/apple-fast/ab/<family>.txt` lines (light form, `tools/aft_ab.sh <binding>
<lane> <dataset> 2 "" "-D <DEFINE>"`, one dataset first). Results land on `lane/apple-fast-results`
(`docs/apple-fast/m3/results.txt`, `build-errors.txt`).

## Why these lanes

FAST work so far went to RF, ExtraTrees, iforest, Lossguide, Depthwise and YetiRank. The lanes below have had no FAST
pass at all. Their opponents were just scored on the M3 (CPU libraries; there is no Apple GPU build of CatBoost, XGBoost
or LightGBM). Our IDENTICAL Metal time is shown where the opponent fill measured it; our FAST times for symmetric,
symmetric-1000, categorical and ordered are being measured now (M3 job `afb3-trees-fast`, read it in results.txt before
you start: it is the baseline).

| lane | dataset | best opponent on the M3 | ours IDENTICAL Metal | notes |
|---|---|---|---|---|
| gbdt-symmetric | taxi / istella | CatBoost 28.2 s / 60.1 s | pending (afb3) | oblivious trees; `lane/apple-fast-trees-symmetric` has one A/B (`-D MOJOLEARN_SYM_DEVICE_LEVEL`, queued) |
| gbdt-symmetric-1000 | taxi / istella | CatBoost 55.3 s / 120.8 s | pending (afb3) | same code, 1000 iterations: per-iteration overhead dominates |
| gbdt-categorical | taxi | LightGBM 58.9 s | 84.4 s | CTR stages (gbdt/ctrs/) |
| gbdt-ordered | taxi / istella | CatBoost 99.3 s / > 300 s cap (uncapped rerun queued) | pending (afb3) | ordered boosting (gbdt/methods/ordered_boosting.mojo, gbdt/data/ordered_plan.mojo) |
| gbdt-multiclass | taxi / istella | XGBoost 45.5 s / capped (rerun queued) | 22.5 s (taxi) | already ahead on taxi; istella unknown |
| gbdt-rank-pairlogit | istellarank | XGBoost 8.4 s | not measured | pairwise ranking loss |
| dart / dart-reg | istella | LightGBM ~2x faster | board 0.8.34 | `lane/apple-fast-dart` exists (peer, pushed 2026-10-02): continue it, do not restart |
| gbdt-depthwise | taxi | XGBoost 10.4 s | FAST 15.1 s | `lane/apple-fast-depthwise` (fused chain, one wait per level) and `-trees-depthwise` are queued: next is one wait per tree |

Where the code is: `gbdt/methods/` (oblivious_tree_structure_searcher, oblivious_tree_bin_builder, ordered_boosting,
dynamic_boosting, pointwise_*, greedy_subsets_searcher/, leaves_estimation/, kernel/), `gbdt/ctrs/`, `gbdt/data/`,
`gbdt/estimator.mojo`, `gbdt/train`. The FAST patterns that won on the other GBDT lanes are the template: batched exact
split search widths (Lossguide width 64, `greedy_search_helper_depthwise.mojo`), one block kernel per stage instead of
per-feature launches, estimation reuse across iterations (YetiRank 55 s -> 5.6 s), and fewer host waits per tree.

## What to do, per lane (one branch each)

1. **Profile by reading, then cut launches and waits.** For each lane, list the kernels one boosting iteration launches
   and every host synchronization (`ctx.synchronize`, buffer readbacks, `enqueue_copy` to host) per iteration and per
   tree level. On Apple every launch plus wait costs ~0.2 ms and grows with live buffers; symmetric-1000 is 1000
   iterations, so per-iteration overhead is the first target. Fuse per-level work into one launch, keep split records,
   leaf values and sizes on the device across levels, and make one wait per tree (or per N trees) the goal.
2. **gbdt-symmetric / -1000** (`trees-symmetric`): oblivious split search over all features x bins for one level is
   one scoring pass; do it as one device kernel per level (score + argmax on device), apply the split to the partition
   index on device, compute leaf values on device. Coordinate with the queued `-D MOJOLEARN_SYM_DEVICE_LEVEL` A/B: build
   on it if it wins, do not duplicate it.
3. **gbdt-ordered** (`trees-ordered`): ordered boosting keeps per-permutation models; batch the permutations into one
   launch per stage instead of a loop of launches, and keep the ordered plan on the device.
4. **gbdt-categorical** (`trees-ctr`): CTR computation and CTR binarization per iteration; build CTR tables with one
   device pass per feature group (segmented sums), no host CTR loops, reuse binarized CTR borders across iterations where
   the algorithm allows. `trees-depthwise` touches CTR prep: merge it first.
5. **gbdt-rank-pairlogit** (`trees-pairlogit`): pair generation and the pairwise gradient on device in one pass per
   iteration (YetiRank's block kernel and estimation reuse are the model).
6. **gbdt-multiclass** (`trees-multiclass`): K classes per iteration as one batched launch over classes, not K launches.
7. **dart**: continue `lane/apple-fast-dart`; dropout bookkeeping and tree re-weighting on device.

Each change: default OFF behind its define, request lines for the deciding dataset first (the one with the largest gap),
then the other after a win. The manager measures on the M3, keeps winners (faster, quality within spread), makes them the
FAST default and merges them to main.
