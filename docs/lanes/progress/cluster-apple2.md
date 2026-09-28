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
