# Lane apple-slow-paths: orchestrator checks

Lane code is not compiled or measured. Everything below is for the orchestrator. Keep the logs in files and read
only the summary lines (`grep -E '^FA |^GBDT |error'`).

## Bindings to build

| binding | why | arms |
|---|---|---|
| `_mojolearn_x_decomp` FAST Apple build (`bindings/build_x_decomp.sh`) | `x_decomp/fa_fast.mojo` | default; `MOJOLEARN_BUILD_DEFINES=MOJOLEARN_FA_SVD_GROUP_OFF`; `MOJOLEARN_BUILD_DEFINES=MOJOLEARN_FA_FAST_GRAM_SPREAD` |
| `_mojolearn_gbdt` (`bindings/build_gbdt.sh`), IDENTICAL and FAST, Apple + NVIDIA + AMD | `gbdt/methods/leaves_estimation/device_walker.mojo`, `gbdt/methods/doc_parallel_boosting.mojo`, `gbdt/methods/ordered_boosting.mojo` | default; `MOJOLEARN_GBDT_WALK_DEV_STEP_OFF`; `MOJOLEARN_GBDT_ORD_TREE_BUFS_OFF` |
| `_mojolearn_gbdt_host` (`bindings/build_gbdt_host.sh`) | imports the same modules | default |
| `_mojolearn_forest_host` (`bindings/build_forest_host.sh`) | imports `gbdt.train` | default |

## 1. GBDT: same bits (timing-free, required)

The two GBDT switches are bit-neutral. Model text and predictions must not move between arms, or between vendors.

```
# on each vendor, once per arm build:
python tools/apple_slow_paths_check.py gbdt --n-estimators 50 --mode identical > gbdt-<vendor>-<arm>.txt
# every GBDT line equal across arms and vendors:
cat gbdt-*.txt | sort | uniq -c
```

The identity lanes as usual (no boxes are up now, so nothing is queued):

```
lq add nv  ID lane/apple-slow-paths gbdt-ordered,gbdt-symmetric taxi,istella
lq add amd ID lane/apple-slow-paths gbdt-ordered,gbdt-symmetric taxi,istella
python -m mojolearn verify --all   # Apple, gbdt lanes: VERIFIED, 0 divergent
```

## 2. GBDT: the drains that went away (timing-free reading)

Per Ordered tree with Logloss defaults (10 Newton iterations, every fold and estimation task walking in lock step):

| drain | before | after |
|---|---|---|
| structure search (`pw.drain`) | 1 | 1 |
| partition settle | 1 | 1 |
| leaf walk, all tasks in lock step (`ordered_boosting.mojo` `while walking`) | 10 (one per line-search round) | 1 (unless no task accepted a step in 10 evaluations) |
| learn loss + leaves home | 1 | 1 |

Device allocations per Ordered tree off the FAST wide path: up to 9 before (sw, sg, bins, noise partials, noise
terms, draws, |planes|, magnitudes, ...), 0 after. Plain fits: `device_walk_estimate` and the per-permutation batch
go from `iterations` drains per estimation to 1 in the same way.

There is no launch or drain counter in the tree today; the table is read from the code
(`device_walker.mojo` `device_walk_enqueue_speculative`, `doc_parallel_boosting.mojo` `estimate_advance` phase 3).
Section 1 is the check that the schedule change kept the bits.

## 3. GBDT: speed (NVIDIA and AMD decide; Apple does not vote on IDENTICAL)

A/B with the `_OFF` define as arm B, one run per arm, on gbdt-ordered (taxi, istella), and on the small Apple case
for the record:

```
python tools/apple_slow_paths_check.py gbdt --n-estimators 1000   # wall time of the whole run, per arm
```

## 4. FactorAnalysis FAST (Apple)

First rule out a one-time cost. The live finding (fast 7016 ms, identical 316 ms) ran fast first in a fresh
process. Run both orders:

```
python tools/apple_slow_paths_check.py fa --order fast,identical > fa-fast-first.txt
python tools/apple_slow_paths_check.py fa --order identical,fast > fa-identical-first.txt
grep '^FA ' fa-*.txt
```

- If fast's `warm` fit is at or below identical's `warm` fit, the 7 s is a one-time cost of the FAST binding's
  first fit (import, pipeline creation), not the FA route.
- If fast's warm fit is still slower: A/B on the M3 (Apple FAST peer), arm B = `MOJOLEARN_FA_SVD_GROUP_OFF`,
  plus `MOJOLEARN_FA_FAST_GRAM_SPREAD` as a candidate, on factor-analysis istella and taxi and on the 20k x 64
  shape above. Quality: mean_log_likelihood must not drop against FAST main. If FAST still loses to identical,
  route FAST FactorAnalysis to the identical kernels: build with `-D MOJOLEARN_FA_ITER_DEVICE_OFF` (no FA entry is
  then registered and `FactorAnalysis.fit` takes the `x_decomp_fa_em_main` loop).
