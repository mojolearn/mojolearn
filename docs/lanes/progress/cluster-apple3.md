# cluster-apple3: progress

Apple FAST speed round 3 for the cluster family (brief: ~/mojolearn-evidence/apple3_speed_brief.md,
2026-09-28). Branch `lane/cluster-apple3` off lane/apple3-merged 6856b5f8f. Speed machine
m3ultra-b. No identity, sabotage or lane-check runs in this lane.

## Targets, from the measurements already on file

FAST seconds on the M3 Ultra (cluster-apple.md, 1790591694704 after dc5a20a43; round 2 made no
FAST change, 1790613748972 confirms the times), largest first:

| case | rows | taxi s | higgs s | where the time is expected |
|---|---|---|---|---|
| affinity-prop | 5k | 1.3834 | 0.9313 | per iteration device kernels (a column per thread), n^2 host loops |
| agglomerative-ward | 10k | 1.1856 | 1.2353 | the sequential host merge loop over a 400 MB matrix |
| KMeans coarse 1M x 28, k 1024 (probe, IVF shape) | 1M | | 0.70 | k-means-parallel rounds, Lloyd (shared with ann) |
| bayesian-gmm | 100k | 0.5515 | 0.6459 | 75 iterations, 2 x 3.2 MB read per iteration, host bound |
| hdbscan | 40k | 0.4805 | 0.4456 | kNN core distances (neighbors), dense pairwise, MST |
| optics | 10k | 0.4050 | 0.4035 | 400 MB read, the sequential host ordering loop |
| gmm | 1M | 0.3417 | 0.4022 | E-step, M-step |
| bisecting-kmeans | 1M | 0.2829 | 0.2953 | |
| meanshift | 10k | 0.2562 | 0.2388 | |
| minibatch-kmeans | 1M | 0.2303 | 0.2961 | |
| spectral | 10k | 0.1825 | 0.0693 | |
| dbscan | 100k | 0.0649 | 0.0333 | |
| agglomerative (single) | 10k | 0.0584 | 0.0456 | |
| kmeans | 1M | 0.0559 | 0.0906 | |

Round 2 left nothing opt-in in the tree. Its unproven commits are the consolidation's.

## Tooling (this lane)

- `MOJOLEARN_XC_PHASES=1`: `x_cluster/device_ops.mojo` drains after every primitive and prints
  the wall time per primitive and the driver's own (`host`); `x_cluster/agglo.mojo` prints the
  merge loop's four parts. Diagnostic, off by default.
- bench/cluster_apple3_prof.py: cProfile of one fit per board case (Python against binding time).
- bench/cluster_apple3_quality.py: the paired quality check, FAST against the IDENTICAL
  reference in one process (board quality number of each, labels equal, adjusted Rand index;
  for the agglomerative cases tree equal and the largest relative merge value difference).
- bench/cluster_apple3_ab.sh: round 2's A/B driver plus `phases:`, `prof:`, `quality:` cases and
  `identical/<binding>` builds.
- bench/cluster_apple3_ward_model.py: NumPy model of the ward rounds against the greedy
  Lance-Williams loop (n = 300, run on one laptop core: 6 of 6 fixtures give EQUAL children,
  planted exact duplicates included; 18 to 27 rounds).

## Jobs

| steward id | Mac | mode | commit | what | state |
|---|---|---|---|---|---|
| 1790626681580 | m4pro-b | fast | 8241f61dc | base FAST board, probes, cProfile, the existing stage timers | running |

## Changes

### WIP, opt-in `-D MOJOLEARN_WARD_ROUNDS=1` (4d1740ff8): FAST ward by rounds of reciprocal nearest neighbours

FAST only, GPU binding only (`ops.fast_device()`), ward + euclidean + no connectivity. Each round
is one device kernel (`_ward_nn_kernel`, one block of 256 threads per live cluster, the ward
dissimilarity from centroids and sizes, an integer min of (float bits, index) keys), one read of
the nearest indices and values, and the Float64 centroid updates on the host. Ward is reducible,
so every reciprocal pair of a round is a merge of the greedy tree. The merges are sorted by
(value, sequence) and numbered by rank. A budget of 256 rounds or a round without a pair returns
the fit to the matrix loop. Unbuilt and unmeasured so far.

New `ClusterOps` methods `fast_device` and `ward_nn` (both columns implement them; the host
column answers False and keeps the matrix loop).
