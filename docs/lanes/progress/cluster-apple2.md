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
