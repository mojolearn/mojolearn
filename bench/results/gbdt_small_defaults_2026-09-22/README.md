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

## Round 2, the CPU host route (2026-09-22, perf/gbdt-small-round2-sep22)

### Why a 320 row host tree took 133 ms

`sample(1)` on a host Ordered fit put 90% of the time in
`_partition_stat` and `_halving_fold`. The host restatement of the pinned
`compute_partition_stats` launch ran all 32 (or 64) chunks of 512 threads and
every 512 lane halving tree for each (leaf, stat), about 5,000 calls and
170 million adds a tree at 320 rows, almost all of them folding +0.0.

| cause | fix | host Ordered RMSE ms/tree, 320 rows |
|---|---|---|
| before | | 133.6 |
| partition stats fold every chunk and lane | only the live chunks and lanes (`_pinned_partition_stat`, `_halving_fold_live`) | 8.4 |
| histogram cells converted and cleared whether reached or not; empty slots scanned; empty parents subtracted; candidate loop in kernel order | reached cells only; empty slots and parents skipped; (leaf, fold) outer, candidate inner; noise drawn once per feature | 3.7 |
| scalar dynamic cosine score | 8 candidates a step, same expressions lane by lane | 2.96 |
| an 11.8 MB histogram plane allocated and filled every tree | one plane per fit, the slots the last tree wrote zeroed | 1.22 |

The Plain host fit had the same partition stats cost and a scalar cosine
score with a noise draw per candidate; with the live lane fold and
`_cosine_gains` (8 candidates a step, the noise drawn once per feature) a
Plain tree at 320 rows went from 12.6 to 1.1 ms.

`_halving_fold_live` is the same sum. Every outer step at or above the least
power of two covering the live lanes adds a lane that only ever holds +0.0,
and `x + 0.0` is its own fixed point, so those steps are one `+ 0.0` pass
over the prefix and the rest of the tree reads only prefix lanes. The host
compiler contracts `x += a * b` into one fma for scalars and vectors alike
(checked in its assembly), and a temporary build that recomputed every
vector lane with the scalar loop and compared bits reported no mismatch.

### Host against device, same bits (M4, depth 6, 10 features, ms/tree)

| rows | Ordered RMSE host / device | Ordered Logloss host / device | Plain RMSE host / device | Plain Logloss host / device |
|---|---|---|---|---|
| 320 | 1.22 / 18.7 | 1.28 / 16.1 | 1.08 / 3.9 | 1.05 / 4.3 |
| 1,000 | 1.62 / 17.2 | 2.02 / 14.5 | 1.66 / 4.4 | 1.71 / 4.8 |
| 3,200 | 2.66 / 14.1 | 3.51 / 14.4 | 1.51 / 4.4 | 1.57 / 4.7 |
| 10,000 | 5.6 / 16.0 | 7.7 / 16.3 | 3.0 / 4.7 | 3.3 / 5.3 |
| 20,000 | 11.0 / 18.0 | 13.5 / 17.8 | 4.5 / 4.8 | 5.1 / 5.5 |
| 32,000 | 17.8 / 20.5 | 22.6 / 20.0 | 5.5 / 4.8 | 6.3 / 5.5 |
| 50,000 | 20.6 / 20.9 | 27.1 / 21.7 | 7.8 / 5.4 | 9.5 / 5.7 |

Explicit fits of 50 trees (30 at 10,000 rows and above), one fit per cell.
Plain at 1,000, 10,000, 20,000 and 50,000 rows was measured before
`_cosine_gains` and is about 0.4 ms a tree slower than the host is now.
The model text is the same on both routes in every cell. Crossover on this
machine at 10 features is 20,000 to 32,000 rows for Plain and 25,000
(Logloss) to 50,000 (RMSE) rows for Ordered.

### The small pool route

`GradientBoosting.fit` now trains an IDENTICAL fit of at most 200,000 cells
(rows x features) on the host binding when the configuration is one the
verifier's CPU column covers (SymmetricTree, RMSE or Logloss, unit weights,
numeric columns, no groups, no eval set). The host fit is the CPU column of
every gbdt training lane. A configuration the host binding refuses by name
trains on the device. `MOJOLEARN_GBDT_ROUTE=device` pins the device,
`host` forces the host, `auto` is the default, and `fit_route_` says which
ran. `tools/identity_break.py` pins `device`, so the Metal, NVIDIA and AMD
columns keep hashing the GPU fit.

A column with exactly one border (the BinaryFeatures histogram policy, for
example the sex column of the smoke test's diabetes split) is refused by
the host binding for Ordered and Plain alike, and no gbdt lane fixture has
one, so that fit stays on the device, and the smoke defaults still take
15.1 s. Restating BinaryFeatures on the host is the next step for that fit,
and it needs a lane fixture with a one border column before it can route.

| fit, 1000 trees unless stated | round 1 | round 2 | route |
|---|---|---|---|
| defaults RMSE, 320 x 10 synthetic (Ordered) | 14.4 s | 1.06 s | host |
| defaults Logloss, 320 x 10 synthetic (Ordered, 10 Newton steps) | 44.5 s | 2.62 s | host |
| defaults RMSE, 3,200 x 10 (Ordered) | 13.2 s | 2.17 s | host |
| defaults Logloss, 3,200 x 10 (Ordered) | 39.9 s | 7.74 s | host |
| Plain RMSE, 320 x 10 | 3.9 s | 1.1 s | host |
| Plain Logloss, 320 x 10 | 4.3 s | 1.1 s | host |
| smoke defaults, diabetes 320 x 10 (binary column) | 15.2 s | 15.1 s | device |

References on the same 320 x 10 pool (round 1, CPU) were CatBoost defaults 0.44 s
(regression and classification), CatBoost with Ordered forced 1.8 s
(regression) and 2.5 s (Logloss), LightGBM 1000 trees 1.35 s, XGBoost 1000
trees depth 6 0.34 s.

### CatBoost's default boosting type here

CatBoost 1.2.10 on the CPU resolves an unset `boosting_type` to Plain at
320, 3,200, 32,000 and 60,000 rows, at 100 and at 1000 iterations
(`get_all_params()`). Our unset default is Ordered below 50,000 rows at 500
or more iterations, which is CatBoost's GPU chain
(`catboost_options.cpp:802-807`, `defaults_helper.h:33-42`), not its CPU
default. So our defaults match CatBoost GPU, and against CatBoost CPU's
defaults they train the more expensive boosting type. The semantics are
unchanged here.

### Identity

Model text and prediction hashes are unchanged against main at 320, 3,200
and 32,000 rows for RMSE and Logloss, explicit Ordered, explicit Plain and
the defaults; the host route's model text, loss curve and predictions equal
the device fit's at every size above. `python -m mojolearn verify --all`
over the 31 gbdt lanes, fixtures base, reads VERIFIED with 0 divergent on
Metal and on the CPU column (all fixtures on the CPU column too, 1,053
parts). A host build with a deliberately wrong `_halving_fold_live` reads
DIVERGENT on 29 of the 31 lanes of the CPU column and VERIFIED on Metal
(the device pin holds), and the auto routed public fit then differs from
the device fit.

### The device Ordered path (goal 2, not done in this round)

The device fit is unchanged in this round. What it costs at 320 rows, 50
trees, is 16 to 19 ms a tree, of which the 28 leaf estimation tasks are
most (about 10 launches and copies each, one drain per walker round for
all tasks since round 1). A device Plain tree at this size is 3.9 ms, and an
Ordered tree runs at least the same structure search over 18 fold
partitions, so fusing the tasks' kernels into one launch per stage and
keeping the fold state resident would bring the device Ordered tree toward
that floor, not below 2 ms. Below 200,000 cells the host route now takes
these fits at 1.1 to 1.3 ms a tree with the same bits. The fused device
path matters for Ordered pools of 20,000 to 50,000 rows, where the device
is still at 18 to 22 ms a tree and the host at 11 to 27 ms.

## Round 3, one-border columns on the host route (2026-09-22, perf/gbdt-host-one-border-sep22)

The host binding refused a column with exactly one border because it did not
restate the BinaryFeatures histograms. It does now, on both searchers:

- Plain (`gbdt/host/gbdt_oracle.mojo::_binary_block`): `binary_hist_kernel`
  and its gather twin are the half-byte accumulator at `UNROLL` 2 (a warp
  takes 256 points, a striped iteration is two batches of eight turns),
  32 flags to a word, and the writeback sums, per flag, the eight stage-2
  cells of its nibble whose value has the flag's bit clear. The flush and
  the fixed-point bridge are the half-byte ones.
- Ordered (`gbdt/host/gbdt_oracle_ordered.mojo::_pw_binary_cells`):
  `compute_split_properties_b_kernel` is the pointwise half-byte
  accumulator with `pw_hb_binary_sum` as its writeback and no scan. The
  host walks only the live threads of each point run, which is what keeps
  the Ordered host tree near 1 ms.

Identity (model text, loss curve, predictions, host against device): the
smoke case below; synthetic pools with 1, 2 and 3 binary columns at 320 and
3,200 rows, RMSE and Logloss, Plain and Ordered (24 cells, 40 trees);
10,000 to 30,000 rows (multi-block flush); and the defaults at 1000 trees.
All equal.

| fit, defaults (1000 trees, Ordered) | before | after | route |
|---|---|---|---|
| smoke, diabetes 320 x 10 (one binary column), RMSE | 15.2 s | 0.93 s | host |
| 320 x 10, 3 binary columns, RMSE | 15.0 s | 0.97 s | host |
| 320 x 10, 3 binary columns, Logloss | 45.2 s | 2.7 s | host |
| 3,200 x 10, 3 binary columns, RMSE | | 2.2 s | host |
| 3,200 x 10, 3 binary columns, Logloss | | 8.3 s | host |

Coverage: the new lane `gbdt-binary-columns` (five flag columns on every
fixture, Plain and Ordered, Logloss and RMSE, the Ordered defaults' bootstrap
and noise, OrderedRMSE) has Apple M4 and CPU columns, equal on all nine
fixtures (`bench/results/identity_break/2026-09-22_gbdt-binary-columns/`),
and the shipped reference table admits its cells from those two witnesses.
Its NVIDIA and AMD columns are owed (the commands are in that README), so
`GradientBoosting` auto-routes a one-border pool to the host on Metal only;
on CUDA and HIP it trains on the device as before. `verify --all` over the
32 gbdt lanes, fixtures base, reads VERIFIED with 0 divergent on Metal and on
the CPU column.
