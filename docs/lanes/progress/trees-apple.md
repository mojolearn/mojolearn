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
| AdaBoostRegressor | taxireg | 4346 ms, RMSE 6.18 | FAST 7729 ms, RMSE 15.99 (seed 7; NOT a FAST defect, see below) |
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

## AdaBoostRegressor FAST vs IDENTICAL quality (paired, 5 seeds x 2 datasets, M3 Ultra)

Seed 7 on taxireg read FAST RMSE 15.99 against IDENTICAL 6.18. Per-member
diagnosis (bench/speed/trees_adaboost_reg_diag.py): the members agree to
member 5 and drift slightly after (the FAST DT arithmetic); IDENTICAL then
stops at 25 members (a member's error reached 0.5) and FAST runs all 50.
Late AdaBoost.R2 members are poor in BOTH modes on the heavy-tailed taxi
fare (y up to 493), so where the ensemble stops decides the RMSE. The
paired table (test RMSE; members in parentheses):

| seed | taxireg FAST | taxireg IDENTICAL | istellareg FAST | istellareg IDENTICAL |
|---|---|---|---|---|
| 7 | 15.989 (50) | 6.178 (25) | 0.7844 (12) | 0.7848 (12) |
| 11 | 11.161 (50) | 12.344 (50) | 0.7936 (11) | 0.7915 (10) |
| 13 | 9.410 (45) | 6.769 (28) | 0.7846 (11) | 0.7872 (9) |
| 17 | 6.019 (25) | 18.259 (50) | 0.7820 (11) | 0.7871 (10) |
| 19 | 14.556 (50) | 9.061 (50) | 0.7856 (12) | 0.7832 (14) |
| mean | 11.43 | 10.52 | 0.7860 | 0.7868 |

FAST is not worse in a paired sense (taxireg means within the seed spread,
Istella-S equal); the taxireg spread is the algorithm's instability on this
target in both modes, not a mode defect. Nothing to revert.

## IDENTICAL before -> after (fit ms, median of 2; every digest/hash equal before and after)

M4 Pro m4pro-a: before a54fadbe3 (main + lane/algos-trees), after 19b23a297.

| algorithm | dataset | before | after | after/before |
|---|---|---|---|---|
| ExtraTrees | taxi | 3183 | 2136 | 0.67 |
| ExtraTrees | taxireg | 8160 | 7097 | 0.87 |
| ExtraTrees | istellareg | 83352 | 62747 | 0.75 |
| RandomForest | taxi | 3295 | 3172 | 0.96 |
| RandomForest | istellareg | 60792 | 52462 | 0.86 |
| AdaBoostRegressor | taxireg | 3274 | 1837 | 0.56 |
| AdaBoostClassifier | taxi | 2994 | 2983 | 1.00 |
| DART | taxi | 6647 | 5589 | 0.84 |
| DART | taxireg | 6229 | 5333 | 0.86 |
| Bagging | taxi | 672 | 645 | 0.96 |
| DecisionTree | istellareg | 848 | 900 | 1.06 |

M3 Ultra m3ultra: before 69cbc0c23 (main + harness; the AdaBoostClassifier
weight fix is NOT in it), after 19b23a297.

| algorithm | dataset | before | after | after/before |
|---|---|---|---|---|
| ExtraTrees | taxi | 1816 | 1049 | 0.58 |
| ExtraTrees | istellareg | 27017 | 15534 | 0.57 |
| RandomForest | taxi | 2525 | 2525 | 1.00 |
| RandomForest | istellareg | 21586 | 25648 | **1.19 (slower)** |
| DecisionTree | istellareg | 451 | 559 | **1.24 (slower)** |
| AdaBoostRegressor | taxireg | 4346 | 2047 | 0.47 |
| DART | taxi | 7466 | 6466 | 0.87 |

The RF / DT slowdown on the M3 Ultra (and DT's 1.06 on the M4 Pro) comes
from one of the three RF flags turned on for IDENTICAL (row-major bins,
SIMD-group histogram aggregation, sorted bootstrap rows): probe arms, one
flag off each, are queued on m3ultra (branch lane/trees-apple-probe, not
for merge). N_BLKS_FOR_COLS 40 was already reverted for IDENTICAL (RF
Istella-S 60.8 -> 91.0 s on the M4 Pro, same hash).
