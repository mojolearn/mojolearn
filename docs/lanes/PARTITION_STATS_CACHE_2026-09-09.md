# Non-symmetric partition-statistics cache

IDENTICAL Depthwise and Lossguide now retain the statistics of leaves that did
not split. The root is reduced once; each split invalidates both children.
The next scoring iteration recomputes those children using the existing
`compute_partition_stats` kernels. Final leaf values still come from the
unchanged full reduction. The cache is local to one tree-growth call, so no
state survives into another boosting tree or a reused training workspace.

This changes scheduling, not sums. Unsplit partitions retain the same offset,
length and statistic bytes. The existing reduction's x-grid/chunk count,
per-thread stripe, stat dimension and pinned folds are unchanged. Its y-grid
already accepts an explicit leaf-ID list, and its final output is indexed by
leaf ID, so selecting fewer leaves does not change a selected leaf's arithmetic.
Histogram-derived statistic propagation remains confined to the existing
FAST/DETERMINISTIC path; enabling it under IDENTICAL would change rounding.

CatBoost source inspected at commit `54a8143a`:
`catboost/cuda/methods/greedy_subsets_searcher/greedy_search_helper.cpp:397-443`
and `split_properties_helper.cpp:918-936`. Its split chain updates child
statistics. This optimization preserves mojolearn's pinned reductions rather
than replacing them with a different upstream numerical schedule. It is not a
claim of full CatBoost numerical or feature parity.

Reference and diagnostic defines:

- `MOJOLEARN_GBDT_FULL_PARTITION_STATS=1`: restore the previous all-leaf sweep.
- `MOJOLEARN_GBDT_PART_STATS_WORK=1`: report search-time leaf and row visits;
  excludes the unchanged final leaf-value reduction.
- `MOJOLEARN_GBDT_SAB_SKIP_RIGHT_PART_STATS=1`: negative control only; deliberately
  omit right-child invalidation, which must move model bits.

Run the mode matrix and the separate fit benchmark with:

```sh
python3 tools/gbdt_partition_cache_ab.py --out build/partition_cache_ab
python3 tools/gbdt_partition_cache_bench.py --out build/partition_cache_bench
```

The gate covers mixed binary/half-byte/one-byte histogram layouts, both growth
policies, depth boundaries, nonzero score noise, weighted RMSE/Logloss boosting,
zero sample weights, repeated fits, tree fields, leaf weights/values, prediction
bits, loss bits and quantization borders. Each mode compares to its own
full-sweep reference. The benchmark measures native `train()` (including
quantization/upload, excluding prediction/fingerprinting), with five warmups
per process and opposite-order reference/cached passes.

Initial Apple M4 IDENTICAL gate: all 14 model fingerprints match. Search-time
partition row visits across Lossguide cases fell from 1,942,836 to 485,661
(75.0% fewer); Depthwise visits fell from 541,032 to 540,130 on these mostly
balanced trees. Vendor-independent scheduling is supported by the kernel
argument above; cross-vendor execution remains to be validated separately.

Completed gate: 14 full-model fingerprints matched per mode in IDENTICAL,
DETERMINISTIC and FAST. FAST/DETERMINISTIC row visits stayed exactly 188,536
(the cache is inactive there). The deliberate missing-right-child invalidation
changed model fingerprints, proving the invalidation gate is sensitive.
No per-tree row-visit count increased under the IDENTICAL cache.

Locked Apple M4 IDENTICAL Lossguide fit timings, six samples per arm:

| Rows | Full sweep median | Cached median | Ratio |
|---|---:|---:|---:|
| 65,537 | 228.583 ms | 222.347 ms | 1.028x |
| 262,145 | 258.9325 ms | 251.7565 ms | 1.029x |

All eight repeat fingerprints per process matched, including across arms.
These are modest local synthetic-fit improvements (~2.8%); sample distributions
are retained and this is not a universal speedup claim. The cache remains
**enabled by default in IDENTICAL**, with the full-sweep reference switch.
FAST and DETERMINISTIC keep their existing propagation path.

Evidence: `bench/results/partition_stats_cache_2026-09-09/`, including complete
correctness and timing logs, summaries, compressed build logs, and binary
SHA-256 fingerprints. No remote GPU execution was performed for this change.
