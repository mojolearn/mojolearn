# Istella-S as a ranking dataset: our IDENTICAL arm against CatBoost, XGBoost and LightGBM

Lane `lane/istella-ranking-bench`, 2026-09-15, one RunPod H100 (pod
`z0jbxz113hwf3n`, see `provenance.md`). Istella-S LETOR at full size: train
2,043,304 rows in 19,245 queries, test 681,250 rows in 6,562 queries, 220
dense features, graded relevance 0..4, query ids carried as `group_id`. Every
number here is OUR IDENTICAL TIER against an opponent's ordinary GPU
configuration; our fast and deterministic tiers are not measured
(`identical-vs-their-fast-only`).

**All three of our ranking losses ran, all opponents that ran are reported,
and nothing is dropped.**

## What the timings say, in one sentence each

- Against CatBoost's fastest cell on this dataset (QueryRMSE, 2,355 ms), our
  fastest cell (QueryRMSE, 3,530 ms) is a ratio of **1.50x**: ours took longer.
- Against XGBoost's fastest cell (`rank:pairwise`, 3,934 ms), our fastest cell
  is **0.90x**, and against LightGBM's fastest cell (CUDA `lambdarank`,
  5,457 ms) it is **0.65x**: ours took less time in both.
- Loss for loss against CatBoost, which is the only opponent implementing the
  same three: QueryRMSE **1.50x**, PairLogit **1.19x**, YetiRank **4.05x**,
  ours over theirs in each case.

## Every cell

Three repeats per tree count, 100 trees, depth 6, learning rate 0.1, L2 1.0,
254 borders (255 bins), no row or column sampling, seed 7. The timed region is
construction plus the library's own input conversion plus fit plus a
`cudaDeviceSynchronize` through libcudart, after one untimed warm-up fit.
Dataset load (about 3.4 s) is outside every timer.

| cell | version | 100 trees ms, median (min..max) | fixed ms | per tree ms | NDCG@5 | NDCG@10 | peak GPU MiB | distinct prediction hashes |
|---|---|---|---|---|---|---|---|---|
| ours IDENTICAL QueryRMSE GPU | 0.8.5 | 3530 (3440..3749) | 2918 | 6.12 | 0.6415 | 0.7099 | 1299 | 1 of 3 |
| ours IDENTICAL PairLogit GPU | 0.8.5 | 6769 (6560..6792) | 5529 | 12.38 | 0.6398 | 0.7104 | 1811 | 1 of 3 |
| ours IDENTICAL YetiRank GPU | 0.8.5 | 11491 (11435..11502) | 3001 | 84.89 | 0.6151 | 0.6810 | 1299 | 1 of 3 |
| CatBoost QueryRMSE GPU | 1.2.10 | 2355 (2329..2367) | 1538 | 8.15 | 0.6412 | 0.7094 | 77055 | 1 of 3 |
| CatBoost PairLogit GPU | 1.2.10 | 5674 (5585..5741) | 2857 | 28.17 | 0.6552 | 0.7254 | 77059 | 3 of 3 |
| CatBoost YetiRank GPU | 1.2.10 | 2837 (2789..2856) | 1548 | 12.93 | 0.6148 | 0.6812 | 77059 | 1 of 3 |
| XGBoost rank:pairwise GPU | 3.2.0 | 3934 (3922..3957) | 2400 | 15.27 | 0.6695 | 0.7380 | 1831 | 1 of 3 |
| XGBoost rank:ndcg GPU | 3.2.0 | 4236 (4206..4283) | 2284 | 19.51 | 0.6638 | 0.7258 | 1831 | 1 of 3 |
| LightGBM lambdarank CUDA | 4.7.0 | 5457 (5300..5518) | 3366 | 20.89 | 0.6831 | 0.7470 | 1605 | 3 of 3 |
| LightGBM lambdarank CPU | 4.7.0 | 7840 (7827..7906) | 2655 | 51.79 | 0.6803 | 0.7415 | 527 | 3 of 3 |

LightGBM's CUDA tree learner is a source build made on the box
(`USE_CUDA=ON`), the same 4.7.0 as the CPU wheel; both arms are reported.

NDCG is computed HERE for every library by one function, from raw test scores:
gain `2^grade - 1`, discount `1/log2(rank + 1)`, a query with no relevant
document scores 1.0, mean over the 6,562 test queries, ties broken
pessimistically. Breaking ties by file order instead moves every cell by at
most 0.001 (`ndcg*_fileorder` in the JSONs), so no cell here is a tie artifact.

`peak GPU MiB` is the highest `nvidia-smi` reading during the process, over an
idle baseline of 0. CatBoost's 77 GB is its allocator taking most of the card,
not a working set it needs; it is not comparable with the others as a
requirement.

## Ratios, both directions stated

Our time over the opponent's time, so **above 1.0 means ours took longer**.

| our loss | opponent cell | 100 trees | fixed cost | per tree |
|---|---|---|---|---|
| QueryRMSE | CatBoost QueryRMSE | 1.50x | 1.90x | 0.75x |
| QueryRMSE | CatBoost YetiRank | 1.24x | 1.89x | 0.47x |
| QueryRMSE | CatBoost PairLogit | 0.62x | 1.02x | 0.22x |
| QueryRMSE | XGBoost rank:pairwise | 0.90x | 1.22x | 0.40x |
| QueryRMSE | XGBoost rank:ndcg | 0.83x | 1.28x | 0.31x |
| QueryRMSE | LightGBM lambdarank CUDA | 0.65x | 0.87x | 0.29x |
| QueryRMSE | LightGBM lambdarank CPU | 0.45x | 1.10x | 0.12x |
| PairLogit | CatBoost PairLogit | 1.19x | 1.93x | 0.44x |
| PairLogit | CatBoost QueryRMSE | 2.87x | 3.60x | 1.52x |
| PairLogit | CatBoost YetiRank | 2.39x | 3.57x | 0.96x |
| PairLogit | XGBoost rank:pairwise | 1.72x | 2.30x | 0.81x |
| PairLogit | XGBoost rank:ndcg | 1.60x | 2.42x | 0.63x |
| PairLogit | LightGBM lambdarank CUDA | 1.24x | 1.64x | 0.59x |
| PairLogit | LightGBM lambdarank CPU | 0.86x | 2.08x | 0.24x |
| YetiRank | CatBoost YetiRank | 4.05x | 1.94x | 6.57x |
| YetiRank | CatBoost QueryRMSE | 4.88x | 1.95x | 10.41x |
| YetiRank | CatBoost PairLogit | 2.03x | 1.05x | 3.01x |
| YetiRank | XGBoost rank:pairwise | 2.92x | 1.25x | 5.56x |
| YetiRank | XGBoost rank:ndcg | 2.71x | 1.31x | 4.35x |
| YetiRank | LightGBM lambdarank CUDA | 2.11x | 0.89x | 4.06x |
| YetiRank | LightGBM lambdarank CPU | 1.47x | 1.13x | 1.64x |

## Is the gap fixed cost? For two of the three losses, yes. For YetiRank, no.

Fixed cost is the intercept and per-tree cost the slope of a least-squares
line through the medians at 1, 10 and 100 trees.

- **QueryRMSE: the whole gap is fixed cost.** Ours pays 2,918 ms before the
  first tree against CatBoost's 1,538, and then 6.12 ms per tree against their
  8.15. The 1.50x at 100 trees is 83% intercept on our side. Extrapolating both
  lines to CatBoost's own default of 1,000 iterations gives 9.0 s against 9.7 s,
  about 0.93x. That is an extrapolation, not a measurement, and no 1,000-tree
  fit was run.
- **PairLogit: mostly fixed cost.** 5,529 ms against 2,857 before the first
  tree, 12.38 ms per tree against 28.17.
- **YetiRank: NOT fixed cost.** Intercepts are close (3,001 against 1,548,
  1.94x) but our per-tree cost is 84.89 ms against 12.93, a ratio of 6.57x, and
  that slope is what produces the 4.05x at 100 trees. This one gets worse with
  tree count, not better: the same lines at 1,000 iterations give about 88 s
  against 15 s. YetiRank is where the per-tree work is, and it is the cell to
  look at next.
- **A measured piece of our fixed cost is Python, not GPU work.** Turning
  `group_id` into run lengths takes 1,278 ms on its own (`ensemble.py`
  `_group_sizes` walks 2,043,304 ids in a Python loop), which is 44% of the
  QueryRMSE intercept and 23% of the PairLogit one. It is inside every one of
  our timed fits because a user pays it, and CatBoost does the same work in
  C++ inside `Pool`.

## Quality, side by side

On the two losses CatBoost and this implementation share the definition of,
NDCG@10 agrees to within a thousandth: QueryRMSE 0.7099 against 0.7094,
YetiRank 0.6810 against 0.6812. On PairLogit CatBoost scores higher, 0.7254
against 0.7104. The best NDCG@10 in the table is LightGBM's CUDA `lambdarank`
at 0.7470, then XGBoost `rank:pairwise` at 0.7380. Ranking quality on this
dataset is a property of the objective more than of the library: every
lambdarank-family cell scores above every QueryRMSE and YetiRank cell,
including CatBoost's own.

## Do CatBoost and XGBoost differ from each other here? Yes, and not in one direction

CatBoost's QueryRMSE (2,355 ms) is the fastest opponent cell on this dataset;
XGBoost's fastest (3,934 ms) is 1.67x that. But CatBoost's PairLogit (5,674 ms)
is 1.44x XGBoost's `rank:pairwise`. So which opponent takes less time depends
entirely on the loss, and a ratio quoted against one of them says nothing about
the other (`opponents-are-not-equally-fast`). XGBoost and LightGBM have no
QueryRMSE or YetiRank, and XGBoost has no oblivious grower, so several cells
exist for only one library by construction.

## Configuration mismatches, all of them

Matched where the libraries allow: 100 trees, depth 6, learning rate 0.1, L2
1.0, 254 borders / 255 bins, no bagging or column sampling, seed 7.

- Ours and CatBoost grow SYMMETRIC (oblivious) trees; our ranking losses fit
  on the symmetric searcher only. XGBoost grows depthwise and has no oblivious
  mode. LightGBM grows leaf-wise, capped here at depth 6 and 64 leaves.
- CatBoost's and our YetiRank default `l2_leaf_reg` is 0; both are pinned to
  1.0 so the cell matches the rest of the table.
- XGBoost pair construction (`lambdarank_pair_method`,
  `lambdarank_num_pair_per_sample`) is left at its own defaults, as is
  LightGBM's `lambdarank_truncation_level` (30). Matching pair-sampling policy
  across three different definitions is not possible; each library runs the
  policy its users get.
- Minimum leaf occupancy is each library's own: ours and CatBoost 1 row,
  XGBoost `min_child_weight` 1 (a hessian, not a row count), LightGBM 20 rows
  and `min_sum_hessian` 1e-3.
- XGBoost refuses a `qid` that is not sorted non-decreasing and Istella-S
  numbers its queries in neither file in ascending order, so the XGBoost arm
  relabels each query by its order of appearance. Same partition, same rows,
  same order; no row moves.
- LightGBM CPU is reported beside LightGBM CUDA and labeled; it ran on 224
  cores.

## Things worth recording, not claims

- Our arm returned ONE prediction hash across the three repeats in every loss.
  CatBoost's PairLogit and both LightGBM arms returned three distinct hashes of
  three. That is a record of what each run produced, not a statement about the
  cost or benefit of determinism.
- One repeat of ours at 100 trees is timed at 3,749 ms against a 3,530 ms
  median (QueryRMSE); the spread is in the table and nothing is smoothed.
- 100 trees is the harness's pinned size, not what a ranking user runs.
  CatBoost's own default is 1,000, and the slopes above say the QueryRMSE and
  YetiRank cells move in opposite directions there.

## Reproduce

    sh tools/dataset_store.sh stage "<ssh flags+target>" \
        gbm-bench/istella/istella_speed.npz gbm-bench/istella/istella_rank.npz
    RANK_PHASE=setup sh tools/istella_rank_leg.sh     # on the pod
    RANK_PHASE=smoke sh tools/istella_rank_leg.sh
    RANK_PHASE=cells RANK_REPEATS=3 sh tools/istella_rank_leg.sh
    RANK_PHASE=lgbm_cuda sh tools/istella_rank_leg.sh
    python3 tools/istella_rank_summary.py bench/results/istella_ranking_2026-09-15/cells

`cells/*.json` holds every fit, every repeat, both tie conventions and the
per-cell mismatch list; `logs/` holds each cell's console output and
`status.tsv` each step's exit and seconds.
