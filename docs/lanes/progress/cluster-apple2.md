# cluster-apple2: progress

Apple speed round 2 for the cluster family (brief: ~/mojolearn-evidence/apple2_speed_brief.md,
2026-09-28). Branch `lane/cluster-apple2` off lane/apple-merged 037daa353. No verification
runs in this lane; a later combined run verifies on m2pro, NVIDIA, AMD and CPU.

Leads from round 1 (docs/lanes/progress/cluster-apple.md, FINAL):
1. k-means (and the ANN IVF fits it dominates) about 7x slower on the M4 Pro than the M3 Ultra.
2. DBSCAN taxi 100k splits into two batches on the 48 GB M4 Pro and gives different labels.
3. GaussianMixture 1M reads about 3% slower on the M4 Pro after round 1.

## Tooling

- bench/kmeans_apple_probe.py: the IVF-shaped k-means fits (coarse 1M x 28 k = 1024; PQ
  codebook 1M x 2 k = 256; the board's 1M x 8 k = 8), each at max_iter N and 1.
- bench/dbscan_batch_probe.py: DBSCAN taxi 100k at several `max_mbytes_per_batch` budgets.

## DBSCAN batch count moved labels: root cause and fix (ab82a8c3b, DEVIATION 5130)

Root cause: per-batch `weak_cc` + `merge_labels` gave a BORDER point the smallest RAW label
among its core neighbours in each per-batch labelling; `reassign` resolves that raw label
through `R` afterwards, but the raw minimum can name a component whose resolved label is
not the smallest. One batch has no such step. Core labels are exact after the merges.
A NumPy model of weak_cc + merge (scratch, not committed) differs from one batch on
8 of 3,000 random 40-point fixtures; with the fix, 0 of 3,000, and the fix never changes a
one-batch result.

Fix: after the merges a batched fit recomputes every non-core label as the smallest FINAL
label among its core neighbours (`border_pull_kernel`), the one-batch answer by
construction. The last batch's CSR is still resident; every other batch's is rebuilt with
loop 2's count and fill. A one-batch fit skips the pass, so its bits cannot move. New check
`check_dbscan_batching_shared_border` (dbscan_main), IDENTITY_PATHS row 231.

### Evidence (m4pro-b, IDENTICAL)

DBSCAN taxi 100k (probe data standardized over its own 100k rows), `max_mbytes_per_batch`:

| commit | 1000000 (one batch) | 0 (default) | 38000 | 20000 | 8000 | 4000 |
|---|---|---|---|---|---|---|
| 71e3303df (before, 1790603188236) | - | e10f0627f89268ce | e10f... | e10f... | **83a2d5a9baa55458** | - |
| 44bd98824 (fix, 1790604377387) | e10f0627f89268ce | e10f... | e10f... | e10f... | e10f... | e10f... |

Board (`bench/x_cluster_speed.py`, 1790604377387): taxi dbscan on the M4 Pro now gives
**9c8ea257cb04e118**, the digest of the M3 Ultra GPU, the CPU column and the H100 (it gave
2c9624c0d0cf42d4 before); HIGGS unchanged 534fe4e04df01f06. Time 0.408 s (round-1 M4
record 0.322 s at 4928e9c4f: the border pass rebuilt batch 0's whole CSR). The RBC arm's
pass now queries only the non-core rows (5169bb4e4, A/B pending).

## k-means on the M4 Pro: where the time goes (1790604377387, MOJOLEARN_KMEANS_STAGES=1)

Standalone `KMeans` (bench/kmeans_apple_probe.py), HIGGS 1M:

| stage (ms) | coarse 1M x 28, k 1024, 10 it | PQ codebook 1M x 2, k 256, 20 it |
|---|---|---|
| k-means-parallel rounds (8) | 898 (16,573 candidates) | 97 (4,145) |
| step 7 (candidate weights) | 132 | 14 |
| k-means++ over the candidates | 196 | 47 |
| step-8 Lloyd (17 it) | 28 | 5 |
| Lloyd, per iteration | 67 to 70 | 6.5 |
| fit total | 2.12 s | 0.34 s |

The rounds and the Lloyd iterations are the fused distance/argmin kernel
(`fused_distance_nn_kernel`): 1M x 1024 x 28 multiply-adds per Lloyd iteration in 67 ms is
about 0.85 TFLOP/s, and every multiply-add carried two `ftz` (the X operand and the
accumulator). Standalone, the coarse fit is 2.1 s on the M4 Pro, not the 8.6 s the ann lane
timed inside the IVF build (x_ann binding); the ivfsq probe case times that path directly.
`MOJOLEARN_EXPERIMENTAL_KMEANS_BLOCK_ACC` (the NVIDIA row-block accumulator) on Apple:
no gain (coarse 2.115 -> 2.120 s, board k = 8 0.141 -> 0.165 s), same digests; not taken.

## A/B m4pro-a, IDENTICAL, 1790607145653: before = base sources (037daa353) -> after 5169bb4e4

Change under test: 5169bb4e4 fused distance kernel flushes operands once at staging
(`FUSED_STAGE_FTZ`, default on), plus the DBSCAN border pass.

| case | before s | after s | digest (both) |
|---|---|---|---|
| KMeans coarse 1M x 28, k 1024, 10 it | 2.7064 | 1.9784 | c34e005aeb912d5d |
| KMeans coarse, 1 it | 1.9136 | 1.4085 | 9611bacf2e88d427 |
| KMeans PQ codebook 1M x 2, k 256, 20 it | 0.4294 | 0.3511 | 794c582742428b14 |
| IVF-SQ fit 1M x 28, 1024 lists (x_ann) | 3.033 | 2.309 | - |
| board kmeans taxi 1M | 0.0946 | 0.0845 | c030387c2bece495 |
| board kmeans higgs 1M | 0.1702 | 0.1476 | 13908e245c273076 |
| board dbscan taxi 100k | 0.3216 (2c9624c0d0cf42d4) | 0.3305 (**9c8ea257cb04e118**) | fixed |
| board dbscan higgs 100k | 0.0890 | 0.1377 | 534fe4e04df01f06 |

The HIGGS DBSCAN slowdown was the border pass querying HIGGS's many noise rows; b2033132c
skips non-core rows still at MAX_LABEL after the merges (noise in every batching: a row's
own batch pulls from every core neighbour). The FCMP trial arm in this job did not build
(two define variables passed the same -D twice), so it measured nothing.

The IVF-SQ fit (coarse quantizer inside the x_ann binding) takes 3.0 s at the base on this
M4 Pro, not the 8.6 s the ann lane recorded at 5b622763d: lane/apple-merged's k-means
changes (incremental k-means|| init under IDENTICAL, the no-sync k-means++) already
removed most of it.

## m4pro-b, IDENTICAL, 1790608687723: before (037daa353 sources) / after b2033132c / trial arms

| case | before s | after s | trial | digests |
|---|---|---|---|---|
| KMeans coarse 1M x 28 k 1024 10 it | 2.1260 | 1.7323 | FCMP 1.5001; Policy4x4 2.3477 | c34e005aeb912d5d all |
| KMeans PQ codebook 1M x 2 k 256 20 it | 0.3388 | 0.3003 | | 794c582742428b14 |
| IVF-SQ fit (x_ann) | 2.448 | 2.041 | Policy4x4 2.678 | - |
| board kmeans taxi / higgs | 0.0800 / 0.1397 | 0.0761 / 0.1294 | FCMP 0.0744 / 0.1243 | equal |
| board gmm taxi / higgs | 2.9654 / 2.8338 | 2.9586 / 2.8222 | FCMP 2.8710 / 2.7267 | 301203207f510506 / cb2f51dc4f8bb1a7 |
| board bayesian-gmm taxi / higgs | 1.5501 / 2.0549 | 1.5477 / 2.0570 | FCMP 1.5190 / 2.0137 | c1ae8cc386159e19 / 6bc026b1942c517f |
| GMM stages (E / M / Chol ms, 22 it) | 1115 / 1441 / 130 | 1115 / 1440 / 130 | FCMP 1044 / 1435 / 129 | |

FCMP (`ftz` as an `|x| < FLT_MIN` select on Apple GPUs) is default on since 493ddcc37;
the Policy4x4 tile for 16 <= d < 32 on Apple was slower and is removed.

**The gathered-query border pass (5169bb4e4 + b2033132c) was WRONG on this Mac**: board
taxi dbscan and the default-budget probe raised "the fit's core mask holds a value other
than 0 or 1", budget 4000 returned 9,955 noise points instead of 5,907, HIGGS moved to
2c304d651e03c887. (On m4pro-a, 1790607145653, 5169bb4e4 alone had given the right labels
at every budget.) Root cause not found in the time left: the gathered query ran the RBC
count and fill over a compact matrix of non-core rows. a1674e9bb goes back to the proven
whole-batch rebuild (ab82a8c3b) and only skips a batch whose non-core rows are all still
MAX_LABEL after the merges. FLAG for the neighbors lane: an RBC eps query over a query
matrix that is not a slice of the indexed data may be unsafe.

## M3 Ultra (m3ultra-b), IDENTICAL, 1790610168662: before 037daa353 sources -> after 493ddcc37

Same job, same Mac, builds of every cluster binding per arm. **Every digest equal before
and after** (board labels + centers, probe labels + centers).

| case | rows | taxi before -> after (s) | higgs before -> after (s) |
|---|---|---|---|
| kmeans | 1M | 0.0576 -> 0.0545 | 0.0930 -> 0.0837 |
| minibatch-kmeans | 1M | 0.2280 -> 0.2290 | 0.3038 -> 0.3068 |
| bisecting-kmeans | 1M | 0.2838 -> 0.2778 | 0.2994 -> 0.2878 |
| gmm | 1M | 2.4470 -> 2.3872 | 2.3465 -> 2.2361 |
| bayesian-gmm | 100k | 1.7565 -> 1.7289 | 2.3282 -> 2.2988 |
| dbscan | 100k | 0.1591 -> 0.1581 | 0.0440 -> 0.0412 |
| hdbscan | 40k | 1.5081 -> 1.5071 | 1.5822 -> 1.5804 |
| agglomerative (single) | 10k | 0.0607 -> 0.0615 | 0.0591 -> 0.0584 |
| agglomerative-ward | 10k | 1.2045 -> 1.1602 | 1.2511 -> 1.2213 |
| spectral | 10k | 0.4868 -> 0.4935 | 0.0762 -> 0.0762 |
| meanshift | 10k | 0.2940 -> 0.2831 | 0.2727 -> 0.2646 |
| optics | 10k | 0.4178 -> 0.4172 | 0.4428 -> 0.4415 |
| affinity-prop | 5k | 1.7253 -> 1.7179 | 0.9271 -> 0.9253 |

| probe (HIGGS) | before s | after s |
|---|---|---|
| KMeans coarse 1M x 28, k 1024, 10 it | 1.1460 | 0.8362 |
| KMeans coarse, 1 it | 0.8487 | 0.6395 |
| KMeans PQ codebook 1M x 2, k 256, 20 it | 0.2150 | 0.1819 |
| IVF-SQ fit (x_ann), 2 reps | 1.546 / 1.515 | 1.475 / 1.200 |

GMM stages (taxi, 22 it): E-step 704 -> 650 ms, M-step 1220 -> 1239 ms, Cholesky 180 -> 179.

### Lead 1 answered: the "7x" M4 Pro vs M3 Ultra k-means

At the lane base the coarse fit (1M x 28, k 1024, 10 it) takes 2.126 s on m4pro-b and
1.146 s on m3ultra-b: 1.9x, below the 4x GPU core ratio (20 vs 80 cores), and the IVF-SQ fit
2.45 s vs 1.53 s. The 7x (8.6 s vs 1.28 s) the ann lane recorded was at 5b622763d; lane
/apple-merged's k-means changes (the no-sync k-means++, the incremental k-means|| init now
taken under IDENTICAL) removed it before this round. What remains is compute: the fused
distance/argmin kernel is the k-means|| rounds (898 of 2,126 ms on the M4 Pro) and every
Lloyd iteration, and it scales with GPU cores. This round's two changes to it (operands
flushed once at staging; the cheaper `ftz`) cut the coarse fit 27% on both Macs.

## M4 Pro (m4pro-b), IDENTICAL, 1790610090438: before 037daa353 sources -> after 493ddcc37, + GMM trials

Every digest equal before and after except DBSCAN taxi, which moves from the two-batch
2c9624c0d0cf42d4 to the one-batch 9c8ea257cb04e118 (the fix). DBSCAN probe (after): taxi
e10f0627f89268ce at every budget (before: 83a2d5a9baa55458 at 8000, 5a03789628c7946c at
4000); HIGGS all-noise 4ac689b5d20b3cea at every budget.

| case | rows | taxi before -> after (s) | higgs before -> after (s) |
|---|---|---|---|
| kmeans | 1M | 0.0804 -> 0.0744 | 0.1399 -> 0.1256 |
| minibatch-kmeans | 1M | 0.2223 -> 0.2236 | 0.2930 -> 0.2939 |
| bisecting-kmeans | 1M | 0.3494 -> 0.3386 | 0.3643 -> 0.3475 |
| gmm | 1M | 2.9510 -> 2.8651 | 2.8280 -> 2.7226 |
| bayesian-gmm | 100k | 1.5467 -> 1.5180 | 2.0486 -> 2.0241 |
| dbscan | 100k | 0.3039 -> 0.4002 (digest fixed) | 0.0864 -> 0.1046 |
| hdbscan | 40k | 2.7132 -> 2.7047 | 2.8642 -> 2.8492 |
| agglomerative (single) | 10k | 0.0792 -> 0.0788 | 0.0759 -> 0.0758 |
| agglomerative-ward | 10k | 0.9896 -> 1.0001 | 1.0335 -> 1.0407 |
| spectral | 10k | 0.3560 -> 0.3092 | 0.0593 -> 0.0589 |
| meanshift | 10k | 0.2709 -> 0.2656 | 0.2578 -> 0.2522 |
| optics | 10k | 0.3960 -> 0.4029 | 0.4219 -> 0.4193 |
| affinity-prop | 5k | 1.7099 -> 1.7155 | 0.8529 -> 0.8511 |

Probe: KMeans coarse 2.12 -> 1.5061 s, PQ codebook 0.3370 -> 0.2725 s, IVF-SQ fit 2.43 ->
1.82 s. GMM stages (taxi): E-step 1115 -> 1047 ms, M-step 1441 -> 1440.

DBSCAN taxi on this 48 GB Mac is 0.30 -> 0.40 s: the price of the correct labels (the border
pass rebuilds batch 0's CSR). The two batches come from cuML's worst-case memory estimate
(`neigh_per_row = n_rows`), which the RBC arm does not use; the batch count no longer moves a
bit, so a tighter estimate is now purely a speed question (not done here).

GMM trial arms (same job, mixture rebuilt with the define, digests equal):
- E-step stacked products: taxi 2.4336 s, HIGGS 2.3522 s (E-step 620 ms): **taken, default on
  in 4174d14d2**.
- M-step PLAN_SPLIT_16_1X1: 2.8591 / 2.7208 s, M-step unchanged: removed.

## M3 Ultra (m3ultra-b), FAST, 1790613748972: before 037daa353 sources -> after 493ddcc37

No FAST-only change in this round (`FUSED_STAGE_FTZ` and the `ftz` spelling are no-ops
under FAST, where `ftz` is the identity). **Every FAST digest equal before and after**; times
within run-to-run spread (kmeans taxi 0.0529 -> 0.0533, gmm taxi 0.3242 -> 0.3415 / HIGGS
0.3690 -> 0.3603, spectral taxi 0.1991 -> 0.1750, KMeans coarse 0.689 -> 0.702 s).

The DBSCAN fix reaches FAST: the probe at budgets 8000 and 4000 moves from 83a2d5a9baa55458
/ 5a03789628c7946c to e10f0627f89268ce, the one-batch labels (default budget and one batch
unchanged). A FAST batched fit now returns exactly the one-batch labels, so its quality is
the reference's by construction.

## M4 Pro (m4pro-a), IDENTICAL, 1790615622690: 493ddcc37 sources -> 4174d14d2 (stacked E-step default) + X-resident trial

| case | 493ddcc37 | 4174d14d2 | X-resident trial | digests |
|---|---|---|---|---|
| gmm taxi 1M | 3.0005 | **2.5452** | - | 301203207f510506 both |
| gmm higgs 1M | 2.8692 | **2.4597** | - | cb2f51dc4f8bb1a7 both |
| GMM E-step (22 it) | 1099 ms | 651 ms | - | |
| KMeans coarse | 1.7067 | 1.7092 | 1.8123 | c34e005aeb912d5d all |
| KMeans PQ codebook | 0.3276 | 0.3247 | 0.3093 | 794c582742428b14 all |
| IVF-SQ fit | 2.05 | 2.04 | 2.18 | - |
| board kmeans taxi / higgs | 0.0834 / 0.1423 | 0.0829 / 0.1434 | 0.0857 / 0.1483 | equal |

The X-resident trial (X row tile kept in threadgroup memory across the column sweep) is
slower on the coarse fit and was removed (a6f3... see git log).

Merged origin/lane/apple-merged at a3e8ed8ea (10 commits: M2 Pro fixes, one process-lifetime
DeviceContext for the SVM/GMM/Cholesky/GP bindings); no conflicts.
