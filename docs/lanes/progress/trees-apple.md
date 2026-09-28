# trees-apple: progress (Apple Metal speed, FAST and IDENTICAL)

Branch `lane/trees-apple` (worktree `~/mojolearn-wt/trees-apple`), off
origin/main 5b622763d, with `lane/algos-trees` merged in (a54fadbe3: the M2
hist_2 / need-weights fixes and that lane's unmerged host speed work, which
is owed its NVIDIA gate there). Not merged to main: the gate runners merge.
Evidence: `~/mojolearn-evidence/trees-apple/` (runs/*.stdout are the steward
speed stdouts).

Harness: `tools/trees_apple_speed.sh` is the timing command of one steward
speed request (`TAP_CELLS`): `gbdt:` = tools/gbdt_train_probe.py (fixed cost
+ per-tree slope from 10 and 100 trees, prediction digests), `rf:`/`et:` =
tools/forest_train_ab.py (100 trees, board config, model hash),
`xt:` = bench/speed/trees_apple_profile.py (the xtrees estimators at the
tools/bench_board_algos.py settings, prediction digest + quality), `xtprof:`
= cProfile of one fit, `rfclock:` = the RF launch clock (a drain per launch:
attribution by COUNT, never a timing). 1,000,000 training rows, taxi / taxireg
(16 columns) and Istella-S (220 columns), staged from R2.

## M2 Pro hist_2 fault (fixed at the root on lane/algos-trees, carried here)

Root cause: on the M2 Pro (no Dynamic Caching) a Metal pipeline's
`maxTotalThreadsPerThreadgroup` falls with register use and a dispatch above
it is DROPPED WITH NO ERROR. Two GBDT launches hit it: the hist_2 one-byte
shared-Int32 arm at block 512 (every cell 0.0, every tree depth 1) and
`compute_need_weights_kernel` at 1024 threads (the Exact-estimator lanes).
Fixes: 7355dd241 / 67c9b2df0 (Apple hist_2 shared-Int32 block 256; Int32
sums, no bit moves) and dc27fe36d (need-weights: 512 threads each running
logical lanes t and t+512, the 1024 halving tree's first step; no bit moves).

M2 proof (steward 1790576952559, commit dc27fe36d, m2pro): **CLEAN: all 33
gbdt lanes AGREE, Metal column vs CPU column cpu-apple-m2-pro** (before the
fix every gbdt tree on the M2 was empty). M3 Ultra and M4 Pro AGREE on the
same 33. The request reads FAIL only on SABOTAGE COVERAGE: the hist_2
sabotage it carried does not reach gbdt-feature-freq, gbdt-ordered,
gbdt-ordered-bayesian-noise, gbdt-ordered-rmse, gbdt-pointwise-l2-bayesian-eval
and gbdt-tensor-ctr-tables (the same six on every Mac): those six need a
CPU-column sabotage that reaches the ordered / feature-freq host paths
(gbdt pass-2 debt, not an M2 fault).

## Baseline, IDENTICAL, M3 Ultra (commit 69cbc0c23 = main + harness)

| algorithm | dataset | fit | note |
|---|---|---|---|
| GBDT symmetric | taxi | 719 ms @100 trees; 48 ms + 6.71 ms/tree | FAST 661 ms |
| GBDT depthwise | taxi | 1756 ms; 49 ms + 17.1 ms/tree | FAST 1612 ms |
| GBDT lossguide | taxi | 5781 ms; 57.8 ms/tree | FAST 5483 ms |
| GBDT symmetric | istella | 2037 ms; 316 ms + 17.2 ms/tree | |
| GBDT depthwise | istella | 2916 ms; 317 ms + 26.0 ms/tree | |
| GBDT lossguide | istella | 7186 ms; 328 ms + 68.6 ms/tree | FAST 6899 ms |
| RandomForest | taxi | 2525 ms | FAST 2326 ms, same hash |
| RandomForest | istellareg | 21586 ms | FAST 17207 ms |
| ExtraTrees | taxi | 1816 ms | FAST 1047 ms, same hash |
| ExtraTrees | istellareg | 27017 ms | FAST 7100 ms |
| DecisionTree | taxi / istellareg | 71 / 451 ms | FAST equal, same digest |
| AdaBoostClassifier | taxi | 12313 ms | FAST 12732 ms |
| AdaBoostRegressor | taxireg | 4346 ms, RMSE 6.18 | FAST 7729 ms, **RMSE 15.99 (quality defect, being diagnosed)** |
| Bagging (10 x depth 12) | taxi | 953 ms | FAST 987 ms |
| DART (100 trees) | taxi | 7466 ms | FAST 7837 ms |
| RandomTreesEmbedding | taxi | 265 ms | FAST 193 ms |
| IsolationForest | taxi | 74 ms | FAST 80 ms |

Where the time goes (profiles on the M3 Ultra):
- AdaBoostClassifier: 87% in `_trees_weighted_rows`' Python loop (0.48 s of
  each 0.54 s member): fixed by lane/algos-trees' native weight check
  (d8b0469cc, merged here).
- AdaBoostRegressor member: x_trees_weighted_sample 81 ms (serial draws),
  the DT fit 36 ms, two host gathers 33 ms, r2_step 15 ms.
- DART: 1.3 s of a 20-tree fit in the Python `math.fsum` of the labels at
  init; per tree the RF fit 42 ms + host apply 11 ms.
- RF istellareg: 726 histogram passes per tree (10 columns per pass on
  IDENTICAL; FAST on Apple already ran 40).
- ET istellareg: IDENTICAL ran ONE row per search thread on Apple (FAST 16,
  NVIDIA/AMD IDENTICAL 64): the H100 measured 28.6 s at one row.
- GBDT lossguide: 63 leaf splits per tree, each with two host waits
  (score read, split sizes): ~0.9 ms per split on Metal. Keeping the loop on
  the device is a driver restructure, not a small change; not attempted.

## Baseline, IDENTICAL, M4 Pro m4pro-a (the before/after Mac; commit a54fadbe3)

| algorithm | dataset | fit |
|---|---|---|
| RandomForest | taxi / istellareg | 3295 / 60792 ms |
| AdaBoostClassifier | taxi | 2994 ms (the algos-trees weight check is in) |
| AdaBoostRegressor | taxireg | 3274 ms |
| DART | taxi / taxireg | 6647 / 6229 ms |
| Bagging | taxi | 672 ms |
| DecisionTree | istellareg | 848 ms |

## IDENTICAL changes (all bit-inert by construction; proofs pending)

| commit | change |
|---|---|
| df6abd315 | RF N_BLKS_FOR_COLS 40 on Apple IDENTICAL (split merge is `Split.update`'s total order); AdaBoost.R2 weighted draws across the host pool (DEVIATION 5607); DART init exact native float32 sum (bit-equal to `_portable_math.fsum` on random, extreme, subnormal, signed-zero cases, host build) |
| f635ed48a | ET search rows per thread 16 on Apple IDENTICAL; RF row-major bins and SIMD-group histogram aggregation on Apple IDENTICAL |

| 8a61b296e | AdaBoostClassifier two-class decision_function / predict / predict_proba: the SAMME margin in one native pass (`x_trees_margin2`), bit-equal to the Python (host build) |
| f49cc154f | origin/lane/merged merged in (Andrew 2026-09-28); d83def7b6: weighted_sample on `host_parallelize` (lane/merged's pinned-FP split) |

Per Andrew 2026-09-28 this lane runs NO verification (no identity or
sabotage requests); the orchestrator checks all Apple branches once on
lane/apple-merged. The identity requests queued earlier were withdrawn.
IDENTICAL changes are shown bit-inert by the before/after digests.

Pending on the stewards: before/after on m4pro-a (before a54fadbe3, after
df6abd315 and f635ed48a; the same Mac); identity request 1790581995433
(RF, ET and every xtrees lane, sabotage steward_combo_cpu_only.patch) on
m2pro, m3ultra, m4-a, do-amd.
