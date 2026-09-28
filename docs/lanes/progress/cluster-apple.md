# cluster-apple: progress

Apple speed lane for the cluster family (brief: ~/mojolearn-evidence/apple_speed_brief.md,
2026-09-28). Branch `lane/cluster-apple` off origin/main 5b622763d; merges are the gate
runners' (NVIDIA + CPU), not this lane's. Home Mac for speed: m4pro-a (M4 Pro, 48 GB).
Board: `bench/x_cluster_speed.py` (taxi and HIGGS from R2, 8 standardized features;
min of 2 reps; digest = labels + centers).

## Base, IDENTICAL, m4pro-a, origin/main 5b622763d (steward 1790579942404)

| case | rows | taxi s | higgs s | H100 (lane algos-cluster, after round 1) taxi / higgs |
|---|---|---|---|---|
| kmeans | 1M | 0.0925 | 0.1721 | 0.030 / 0.069 |
| minibatch-kmeans | 1M | 0.2212 | 0.2908 | 0.353 / 0.419 |
| bisecting-kmeans | 1M | 0.3841 | 0.4054 | 0.464 / 0.461 |
| gmm | 1M | 3.1468 | 3.0335 | 1.769 / 1.681 |
| bayesian-gmm | 100k | 1.9616 | 2.7436 | 0.942 / 1.251 |
| dbscan | 100k | 0.4746 | 0.2366 | 0.062 / 0.028 |
| hdbscan | 40k | 5.4032 | 5.3290 | 0.209 / 0.198 |
| agglomerative (single) | 10k | 0.1452 | 0.1440 | 0.018 / 0.016 |
| agglomerative-ward | 10k | 1.3563 | 1.4522 | 3.987 / 4.308 |
| spectral | 10k | 0.3366 | 0.0751 | 0.340 / 0.060 |
| meanshift | 10k | 0.2876 | 0.2710 | 0.254 / 0.249 |
| optics | 10k | 0.3893 | 0.3860 | 0.646 / 0.733 |
| affinity-prop | 5k | 2.6157 | 1.3479 | 4.673 / 3.848 |

Every digest equals the H100 and M3 Ultra records EXCEPT dbscan taxi (below).

## FINDING: M4 Pro DBSCAN taxi 100k disagrees with every other column

- m4pro-a GPU: labels digest 2c9624c0d0cf42d4 (silhouette -0.1085).
- m3ultra GPU at the same commit: 9c8ea257cb04e118; m3ultra CPU (host binding,
  `--column cpu`): 9c8ea257cb04e118; H100: 9c8ea257cb04e118.
- HIGGS dbscan agrees on M4 Pro (534fe4e04df01f06). The dbscan identity lanes pass on
  M4 (small fixtures). A defect to root-cause (probe: bench/dbscan_m4_probe.py on
  lane/cluster-apple-prof, rbc and brute, GPU vs CPU, identity traces diffed).

## Stage profiles (M3 Ultra, diagnostic branch lane/cluster-apple-prof, MOJOLEARN_STAGE_PROF=1)

- HDBSCAN taxi 40k (2.23 s): kNN core distances 880 ms (neighbors' kNN), dense
  pairwise 670 ms, mutual reachability 243 ms, non-finite refusal 49 ms, Boruvka
  MST 310 ms (6 rounds, 34 ms per min-edge pass), condense 45, extract 12.
- BayesianGaussianMixture taxi 100k (1.79 s): 75 iterations, E-step 40 ms, M-step
  moments 1478 ms (83%), host/rest 185 ms.

## Changes on lane/cluster-apple

1. d5c6541f0 DeviceOps (x_cluster's GPU column): uploads and zeros enqueue without a
   synchronize (an upload's source is a copy held until the next synchronize; zeros is
   a device memset); batched reads `gets` / `get_if` in BGMM/GMM, MiniBatchKMeans and
   `nearest_all`. Scheduling only.
2. 28ab726b3 moments (DEVIATION 5121): the speculative chain. Plain adds with an
   integer flag, off the chain, on every sum whose exponent field is zero from a
   nonzero operand (the only sums `ftz` can change); a flagged tile is re-added through
   `chain_add` from its saved start. moments_check gains the flag case; arm
   5121_moments_tile regenerated; new arm 5121_moments_flag.

## Requests in flight

- identity 1790582025688-cluster-28ab726b31 (18 x-cluster lanes, --pass 2, sabotage
  ~/mojolearn-evidence/cluster-apple/combined_e2e.patch) on m2pro, m3ultra, m4pro-a, do-amd.
- speed after (IDENTICAL) 1790582041926 on m4pro-a; before/after pair on m3ultra
  (1790582043709 at 5b622763d, 1790582046724 at 28ab726b3).
