# lane xgboost-gap: where the taxi lossguide time actually goes

2026-09-12, RunPod pod `e7wyv32nqzfpz2`, NVIDIA H100 80GB HBM3, driver
580.126.09, Xeon Platinum 8480+ (224 cores). XGBoost 3.2.0, scikit-learn
1.9.1, NumPy 2.4.6 -- the same opponent versions as the published row. Mojo
1.0.0 (ed45d567), our IDENTICAL tier at commit `ba69c692`. Datasets staged
from Cloudflare R2 and verified against `bench/results/dataset_store/manifest.tsv`
(taxi 419,757,252 bytes, Istella-S 2,248,281,826 bytes, both sha256 matched);
nothing was re-fetched or re-decoded.

THE QUESTION. `bench/OPPONENT_REFERENCE.md` records XGBoost GPU at 476.2 ms
against our 950.0 ms on taxi lossguide, a ratio of 1.995x and the largest
single gap on the GBDT board. Quality is within a thousandth, so it is a pure
speed gap.

THE ANSWER, IN ONE LINE. **It is not a lossguide problem and it is not a
startup problem. We are ~2x slower per tree than XGBoost under BOTH growth
policies, and our fixed cost is ~0.7x theirs. Lossguide looks worst only
because its per-tree cost is the largest, so a constant per-tree deficit
dominates the 100-tree total there.**

## 1. The cell, reproduced on this box

Interleaved, one process per cell, 5 rounds, medians (min..max). 1,000,000
rows, 100 trees, depth 6, `max_leaves` 64. Ours IDENTICAL against their FAST
GPU arm.

| cell | ours ms | XGBoost GPU ms | ours/XGB | published |
|---|---:|---:|---:|---:|
| lossguide taxi | 921.1 (910.8..934.7) | 433.6 (409.0..458.6) | **2.124x** | 1.995x |
| depthwise taxi | 414.7 (412.8..423.1) | 315.3 (308.3..350.1) | **1.315x** | 1.196x |

Both arms run faster on this pod than on `u4elzj1eo486ps` and both ratios come
out slightly worse for us. The gap reproduces; it was not a bad sample.

Quality, same fits: lossguide ours logloss 0.525504 / AUC 0.619386 against
XGBoost 0.525227 / 0.620101; depthwise ours 0.525086 / 0.621421 against the
same XGBoost figures. Within a thousandth, as published.

## 2. Fixed cost against per-tree cost -- the lens that reframed CatBoost

Each arm timed at 1, 10 and 100 trees with a REAL device drain (torch and
cudart both armed), `tools/xgboost_gap_probe.py decompose`. The slope is the
per-tree cost, the intercept is what the arm pays before it boosts at all.

| lane | dataset | ours fixed ms | ours ms/tree | XGB fixed ms | XGB ms/tree | fixed ratio | **per-tree ratio** |
|---|---|---:|---:|---:|---:|---:|---:|
| lossguide | taxi | 91.9 | 8.770 | 133.8 | 4.359 | 0.687x | **2.012x** |
| lossguide | Istella-S | 773.1 | 14.624 | 1065.8 | 6.872 | 0.725x | **2.128x** |
| depthwise | taxi | 91.1 | 3.280 | 121.4 | 1.613 | 0.750x | **2.034x** |

Three independent cells, three per-tree ratios within 6% of each other. This
is the finding: **the deficit is a constant factor on the boosting loop, not a
property of leaf-wise growth.**

It is the mirror image of the CatBoost result. Against CatBoost we are 4.0x
cheaper before the first tree and only 1.22x cheaper per tree, so the headline
flatters us and decays with more iterations. Against XGBoost we are again
cheaper on fixed cost (0.69-0.75x) and 2x more expensive per tree, so the
headline UNDERSTATES the gap and it gets worse with more iterations: at
XGBoost's own defaults the lossguide ratio tends toward 8.770/4.359 = 2.01x
rather than the 1.995x measured at 100 trees.

Our own fixed cost measures 91.9 ms here against the 86.1 ms the gbdt-fairness
lane measured for symmetric on a different pod -- consistent, and it confirms
that startup is not where the lossguide gap lives.

**A caveat that must travel with this table.** The probe's absolute XGBoost
totals run ~30% above the interleaved control's (569.7 ms against 433.6 ms at
100 trees) while ours differ by ~5%. The per-arm method is identical within
the probe, so the intercept/slope SPLIT and the per-tree RATIOS are sound, but
the probe's absolute numbers are not quotable as opponent rows. The absolute
ratios in section 1 come from the interleaved harness and are the quotable
ones.

## 3. Why lossguide costs more than depthwise -- for BOTH libraries

Our lossguide grows to `leaves=64, iterations=63`: 63 sequential expansion
steps per tree, against depthwise's 6 levels. Our per-tree cost rises 3.280 ->
8.770 ms (2.67x) and theirs 1.613 -> 4.359 ms (2.70x). Both libraries pay
almost exactly the same multiple for leaf-wise growth, which is why the
per-tree ratio barely moves between policies.

**A fairness nuance worth recording.** XGBoost returned prediction digest
`44d0d2cc41991bb1` in BOTH the depthwise and lossguide cells -- the same
model. At depth 6 with `max_leaves=64` their `max_leaves` is inert and
`max_depth` bounds the tree (`src/tree/driver.h:40-45`, defaults
`src/tree/param.h:90-96`), so their lossguide arm converges on the same tree
their depthwise arm builds, and pays 1.37x for the privilege of building it
one node at a time. Ours returned DIFFERENT digests per policy
(`0dd8bcfc3c3a4a1d` lossguide, `40c1683b9e0eb151` depthwise) and different
logloss, i.e. ours genuinely grows leaf-wise. The 2x per-tree deficit is
present in the depthwise cell too, where both libraries grow level-wise, so
this nuance does not explain the gap away -- but the lossguide headline does
compare our genuine leaf-wise tree against their depth-bounded one.

## 4. Where our per-tree time goes (our own stage ledger)

`MOJOLEARN_STAGE_TIMES=1`, 10 lossguide trees on taxi, `leaves=64
iterations=63`. **A stage-timed run drains on both edges of every stage, so
this is a SPLIT of the fit and not a timing of it**; only the shares are
meaningful, and the total is not comparable to a clean round. Named stages
account for 86.9 ms of the 193.9 ms summed wall (44.8%) -- the remainder is
un-wrapped host bookkeeping, which is itself worth knowing.

| stage | ms over 10 trees | % of accounted |
|---|---:|---:|
| split.chain | 21.126 | 24.3% |
| hist.build | 15.001 | 17.3% |
| split.host | 9.770 | 11.2% |
| hist.scan | 6.426 | 7.4% |
| split.sizes | 5.630 | 6.5% |
| partstats | 4.975 | 5.7% |
| hist.subtract | 4.025 | 4.6% |
| score.kernel | 3.844 | 4.4% |
| score.read | 3.723 | 4.3% |
| hist.zero | 3.192 | 3.7% |
| est.pstats / est.approx / est.move / est.readback | 6.796 | 7.8% |
| host.plan | 1.408 | 1.6% |
| score.hostreduce / leaf.values / model.build | 0.943 | 1.1% |

The split chain and the histogram build are 41.6% of accounted time between
them. That is where a per-tree factor of two has to be found.

## 5. What XGBoost does that we do not (read against their source)

Pinned checkout `/Users/andrewhendel/CascadeProjects/upstream/xgboost`,
`6c3fde7aefa6abee7458f2ba23ac291a31b88204`.

The hypothesis that had to be discarded first: that they batch leaf-wise
expansions and we do not. **They do not.** `Driver::Pop` returns exactly one
entry under `kLossGuide` (`src/tree/driver.h:88-101`); the batching branch at
`driver.h:106` is reachable only for depthwise, and their own unit test pins
it (`tests/cpp/tree/gpu_hist/test_driver.cu:65-70`). Their lossguide is as
serial as ours. Nor is it a drain count: they pay two blocking syncs per
expansion step (`src/tree/gpu_hist/row_partitioner.cuh:400-404` and
`src/tree/updater_gpu_hist.cu:613`), and so do we (the winner readback at
`gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo:1807-1818`
and `RebuildLeavesSizes` at `:2348-2357`). The difference is what each step
costs.

1. **One segmented CUB scan repartitions, with a scattering output iterator,
   and no sort.** `src/tree/gpu_hist/row_partitioner.cuh:145-205`; left rows
   pack forward and right rows backward inside the scan itself
   (`WriteResultsFunctor`, `:111-140`), and the scan's temp storage is sized
   ONCE against the root and reused for the life of the fit (`:168-184`).
   Ours is a multi-launch chain -- flag+sequence, stable partition, reorder,
   histogram copy, update partitions (driver `:2245-2344`) -- and it is our
   single largest stage.
2. **Partition and histogram build share ONE pass over the data**
   (`src/tree/updater_gpu_hist.cu:405-424`, rationale at `:369-370`). We
   launch them separately.
3. **The host never reads a histogram; it reads one `uint32` count and two
   split structs per step.** Under IDENTICAL we copy back `2 * argmax_blocks`
   score/bin values per leaf and fold them on the HOST
   (`greedy_search_helper_depthwise.mojo:1809-1818`, host fold `:1853-1904`).
   The device-side fold exists but is FAST-only (DEVIATION 1904, `:1917-1937`),
   and FAST is not an optimization target.
4. **Choosing the smaller sibling costs them nothing.** `left_sum`/`right_sum`
   are already host-resident in the `GPUExpandEntry` returned by the previous
   evaluate (`updater_gpu_hist.cu:390-397`,
   `gpu_hist/expand_entry.cuh:14-31`). We need the leaf-size readback to make
   the same choice, and that readback is a hard serialization point.
5. **The histogram arena is zeroed once per boosting iteration and
   bump-allocated per node** (`gpu_hist/histogram.cuh:93-104`, `135-145`),
   with a 4096-node cache so subtraction never misses
   (`src/tree/hist/hist_param.h:23`). We zero a `block_hist` scratch sized by
   `max_leaves` inside the per-policy-block loop every iteration
   (`greedy_subsets_searcher/greedy_search_helper.mojo:2832`, sized `:3457`),
   so a two-leaf iteration still clears 64 leaves' worth.
6. **The deepest level is never partitioned in-core**
   (`updater_gpu_hist.cu:385`); row-to-leaf assignment is recovered by walking
   the tree in `FinalisePosition` (`:433-449`). At depth 6 that elides a
   full-width partition pass per tree.

## 6. What was NOT done, and why

No DEVIATION was written. The candidates that follow from section 5 -- fusing
the partition and histogram passes (their #2), or removing the leaf-size
readback by keeping child sums host-side the way their `GPUExpandEntry` does
(#4) -- are both multi-hour kernel changes with real identity risk, and
neither could be written, gated on both sides, built and measured inside this
lease. Landing an unmeasured comptime switch would have added exactly the
unchecked path ENGINEERING_RULES 8 forbids. The explanation is the
deliverable; the next lane starts from section 5 with the target quantified:
**a factor of two on the per-tree loop, worth ~490 ms of the 921 ms taxi
lossguide cell and ~100 ms of the 415 ms depthwise cell.**

## 7. Identity

Every cell was our IDENTICAL tier. No source file under `gbdt/` was modified
by this lane, so no bits could move; the run is an observation, not a change.
Our arm returned ONE prediction digest across all 5 rounds of each cell:

| cell | ours digest, rounds 1-5 |
|---|---|
| lossguide taxi | `0dd8bcfc3c3a4a1d` |
| depthwise taxi | `40c1683b9e0eb151` |

(`speed_gbdt_arm.hash_predictions` is sha256 over the prediction vector's
dtype, shape and bytes truncated to 16 hex -- a same-device witness, not a
cross-vendor one.)

## 8. Files

- `logs/cells/control_taxi.log`, `logs/cells/control_taxi_depthwise.log` --
  the interleaved 5-round cells of section 1.
- `logs/cells/decompose_taxi.log`, `decompose_istella.log`,
  `decompose_taxi_depthwise.log` -- section 2.
- `logs/cells/stages_taxi.log` -- the stage ledger of section 4.
- `logs/setup.txt`, `gpu.txt`, `box.txt`, `versions.txt`, `so_sha256.txt`,
  `progress.txt` -- the box, the builds and the opponent versions.

The first body run is in `progress.txt` above the second: its two `decompose`
cells died on `ImportError: sklearn needs to be installed` (XGBoost's
scikit-learn API), and its control cell recorded our arm with the XGBoost arm
REFUSED for the same reason. Those logs were deleted and the cells re-run
after installing scikit-learn; nothing from the failed pass is quoted here.
