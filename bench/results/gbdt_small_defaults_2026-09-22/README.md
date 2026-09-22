# GradientBoosting defaults on a small pool, Apple M4 (2026-09-22)

Triage, not a benchmark: one fit per cell, IDENTICAL tier, one Metal job at a
time on the shared M4. The pool is 320 x 10 float32 (synthetic, and the
smoke test's diabetes split), depth 6.

## Why the defaults were slow

`GradientBoosting()` resolves to 1000 trees of Ordered boosting below 50,000
rows. Below 500 rows `MinEstimationSize` is 1, so an Ordered tree runs 9 folds
x 3 learn permutations + 1 estimation permutation = 28 leaf estimation tasks.
Each task drained the queue several times (bins download, partition upload,
weight fold, one readback per walker evaluation, task tail), and on this
device every drain and every launch has a fixed price. Per tree cost was flat
in n (explicit Ordered, RMSE, 50 trees, 0.8.13 wheel): 115 ms at 320 rows,
94 ms at 3,200, 137 ms at 32,000; Plain was 6 to 7 ms.

`MOJOLEARN_STAGE_TIMES=1` put 84% of an Ordered fit in the fold estimation
tasks. Measured causes, in the order they were removed:

| cause | fix | Ordered RMSE ms/tree at 320 rows |
|---|---|---|
| before | | 130 |
| estimation pool keyed on row count, rebuilt on every task | one workspace per task | 86 |
| about four drains per task, 28 tasks | tasks enqueued together, one drain per walker round | 44 |
| every live allocation bound to every Metal launch (22 us a launch at 100 live buffers, 80 to 98 us at 1000) | batch buffers carved from one arena | 31 |
| pinned partition stats grid folds dozens of empty 512 thread blocks per leaf | empty blocks write their +0.0 without folding; bounded launch where the widest leaf is known | 17 |
| 1.26 ms device attribute query per tree, fused gathers, discarded cursor move | threaded SM count, fused stage in | 15 |

## After (same bits)

| fit | before | after |
|---|---|---|
| defaults, RMSE, 1000 trees (smoke diabetes split) | 60.6 s | 15.2 s |
| defaults, RMSE, 500 trees | 57 ms/tree | 14.9 ms/tree |
| defaults, Logloss, 500 trees | 202 ms/tree | 46 ms/tree |
| explicit Ordered Logloss, 50 trees | 60 ms/tree | 15.6 ms/tree |
| Plain RMSE, 50 trees | 7 ms/tree | 4 ms/tree |

Model text and prediction hashes are unchanged at 320, 3,200 and 32,000 rows
and on the defaults above. `python -m mojolearn verify --all` over the 31 gbdt
lanes, fixtures base, reads VERIFIED with 0 divergent; a build with the arena
bug described in `pointwise_oracle.mojo` (`OracleHostScratch`) reads
DIVERGENT on `gbdt-ordered`, so the lanes see this code.

## Reference libraries, same pool, CPU

CatBoost defaults 0.44 s (regression) and 0.44 s (classification); CatBoost
with Ordered forced 1.8 s and 2.5 s. LightGBM 1000 trees 1.35 s. XGBoost 1000
trees depth 6 0.34 s. The GPU Ordered fit is still sync and launch bound at
this size: what remains is about 15 launches per task and the structure
search's per tree rebuild of its fold state.
