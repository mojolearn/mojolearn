# Is the GBDT speed advantage over CatBoost real? (lane gbdt-fairness, 2026-09-12)

Andrew did not believe the 0.437x taxi symmetric row and asked for it to be
BROKEN, not confirmed. This lane tried, on its own pod, with the leading
hypothesis tested first and hardest.

**The advantage is real, and the headline number is still wrong.** The honest
taxi symmetric ratio is about **0.51x, not 0.437x**, and the win is almost
entirely a FIXED-COST win rather than a faster boosting loop.

Pod `7yubb7pxvwzklo`, NVIDIA H100 80GB HBM3, driver 580.126.09, Xeon Platinum
8480+ (224 cores), CatBoost 1.2.10, NumPy 2.4.6, our IDENTICAL tier at commit
84638fce with `2550_borrow_x=1 2551_device_partition=1 2580_level_quant=0
2581_group_width=1` -- the same switch paths the claim's run reported. Both
npz fixtures came from R2 verified against `bench/results/dataset_store/manifest.tsv`.

## The claim reproduces, at a slightly worse ratio

Our arm reproduced the claim's model EXACTLY: hash `90c3558501933f47` in every
round, logloss 0.525735 / AUC 0.619460, equal to the September 12 row. What
did not reproduce is CatBoost's time: 709.0 ms there, 624-636 ms here.

| cell | claim | here (median of 5) | ours/CatBoost here |
|---|---|---|---|
| symmetric taxi | 0.437x (310.1 / 709.0) | 327.7 / 623.8 | **0.525x** |
| symmetric Istella-S | 0.830x | 1219.8 / 1504.0 | **0.811x** |
| depthwise taxi | 0.518x | 446.6 / 755.7 | **0.591x** |

The published 0.437x was taken against a CatBoost sample at the slow end.
Two of the three cells are materially less favourable to us than the board says.

## 1. The leading hypothesis: `_our_sync` is a no-op. DEAD.

A third arm, `ours-sync`, is our arm with a REAL drain (torch AND cudart
through ctypes; cupy was absent and said so), interleaved with the other two
in one process and one heat window.

| cell | ours | ours-sync | difference |
|---|---|---|---|
| symmetric taxi | 327.7 | 324.3 | sync is 1% FASTER (noise) |
| symmetric Istella-S | 1219.8 | 1291.3 | within round-to-round spread |
| depthwise taxi | 446.6 | 440.1 | sync is 1.5% FASTER |

Both arms returned the same hash and the same accuracy. Two independent
checks agree: back-to-back fits with no sync between them average 315.1 ms
against a 317.8 ms per-fit median (`average/median = 0.9915`), so nothing is
queued behind the clock; and the library's own stage stamps, which drain per
stage, account for 258.8 ms of a 294.7 ms fit. **Nothing is in flight when
`gbdt_fit` returns.** The docstring's reasoning is also sound for GBDT after
all: the binding returns a model TEXT that Python parses during the call, and
the text is built from leaf values already copied host-side
(`greedy_search_helper.mojo:1636`, `enqueue_copy` then `ctx.synchronize()`).

## 2. Are the models the same size? YES on taxi; CatBoost's is SMALLER on Istella-S.

The check that did not exist before this lane.

| dataset | our trees / leaves | CatBoost trees / leaves |
|---|---|---|
| taxi | 100 / 6400 (64 per tree) | 100 / 6400 (64 per tree) |
| Istella-S | 100 / 6400 (64 per tree) | 100 / 6280 (8..64 per tree) |

We are not solving an easier problem. On Istella-S CatBoost builds 120 FEWER
leaves than we do and is still slower. Quality is a thousandth apart
everywhere (taxi 0.525735 against 0.525904; Istella-S 0.138653 against
0.138982; depthwise 0.525086 against 0.525047).

## 3. Is CatBoost really on the GPU? YES.

`get_gpu_device_count()` returns 1, and the `flags` cell -- which fits
CatBoost and nothing else -- sampled a CUDA compute application in 58 of 66
`nvidia-smi` samples with utilisation running to 100%.

## 4. Do the pinned flags put CatBoost on a slow path? NO, at most 3%.

Released one at a time, at 100 trees on taxi:

| CatBoost config | ms | vs pinned |
|---|---|---|
| pinned (`bootstrap_type='No'`, `boosting_type='Plain'`, 254 borders) | 655.9 | 1.000 |
| bootstrap default (Bayesian) | 651.8 | 0.994 |
| boosting default | 652.9 | 0.995 |
| 128 borders (its GPU default) | 636.3 | 0.970 |
| all three released | 637.0 | 0.971 |

## 5. THE REAL MECHANISM: this is a fixed-cost win, not a boosting win.

Each arm timed at 1, 10 and 100 trees on the same rows:

| arm | fixed cost | per tree | at 1 tree | at 100 trees |
|---|---|---|---|---|
| ours | **86.1 ms** | 2.297 ms | 88.4 | 315.8 |
| CatBoost | **346.0 ms** | 2.809 ms | 348.8 | 626.9 |

We are **4.0x cheaper before the first tree** and only **1.22x cheaper per
tree**. At the pinned 100 trees that fixed cost is 55% of CatBoost's total and
27% of ours, which is where the headline ratio comes from. Extrapolating the
same two lines to CatBoost's OWN default of 1000 iterations gives ours 2383 ms
against 3155 ms, a ratio of about **0.75x** -- the advantage is real but much
smaller on a job whose length is typical rather than chosen.

`n_estimators=100` is therefore doing a lot of work in this comparison, and it
was picked to match `tools/nvidia_forest_bench.sh`, not because it is what a
CatBoost user runs.

## 6. Two things this lane did not expect

- **We are CHARGED for a cost we could delete.** `fit` transposes X inside the
  timer (DEVIATION 1840). With the Fortran copy prepared outside
  (`MOJOLEARN_SPEED_FORTRAN=1`) our arm runs 284.8 ms against 316.9. Our
  number is conservative by ~32 ms, not flattered.
- **IDENTICAL is FASTER than our own FAST tier**, 316.9 ms against 383.1
  (1.21x), at a different hash. Asked to check whether IDENTICAL costs
  anything, the answer is that it costs less than nothing on CUDA, which says
  the FAST tier is stale here rather than saying anything good about
  IDENTICAL. It is recorded as an anomaly to chase, not as a result.

## What would still change the verdict

Nothing here rules out that CatBoost's 346 ms floor is dominated by pool
construction that a `Pool` reused across fits would amortise. A user fitting
once pays it, which is what `fit` measures for both arms, so the row is fair
as stated -- but "we are 2x faster than CatBoost" should be read as "we start
4x faster and boost 1.2x faster".
