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

3. b97028944 AffinityPropagation (DEVIATION 5105): the responsibility update one
   block per row on the device; the row's max and second max, each at its lowest
   index, from an integer max of `(float order with -0.0 folded onto +0.0, -index)`
   keys (the row loop's picks for every block shape); the cell update is one spelling
   (`bodies.ap_r_update`). ap_check gains n = 600 with planted ties; arm
   5105_ap_rmax_tie; 5105_ap_damping and e2e_device_fold_reversed regenerated.
4. 6c7a51fbd dense graph index math: the self-loop pass one thread per row (was one
   per cell with a 64-bit division); 32-bit row/col in fill_indices2 and in the
   mutual reachability transform.
5. eb1299ed7 HDBSCAN: `build_sorted_mst[DENSE=True]` computes each column as
   `e % m` (`mst_kernels._edge_dst`), so the m x m index array (6.4 GB at 40k rows) is
   neither written nor read; the min-edge scan tests colors before the mst_edge byte
   (same predicate); mutual reachability in place over the distances (one m x m
   buffer instead of two). Every other caller keeps the reference path (defaults).
   Arm e2e_hdbscan_dense_dst proves the lanes reach it.
6. d744c7cb5 core/spec_chain.mojo: the speculative flushed chain as a shared helper
   (`ftz_chain_block[U]`, `suspect_sum`); GaussianMixture's nk (mstep) and mean
   log-likelihood (estep) chains use it; x_cluster moments take `suspect_sum` from it.
   core/spec_chain_check.mojo (device + host vs the plain flushed chain, planted
   cancellations / subnormal partial sums / -0.0 / subnormal addends) with arm
   spec_chain_fallback.patch; e2e_spec_chain_reach proves the gmm lanes reach it.

7. 0ea0b944e AgglomerativeClustering single linkage: the DENSE solver too.
8. 30c6e9638 x_cluster moments, FAST only: row slices summed per block, the partials
   added over the slices (the same addends and finals; FAST moves bits, quality by the
   paired check).
9. 381f85584 merge of origin/lane/merged (Andrew 2026-09-28). Fix found in the merge:
   lane/merged's `_ap_noise_kernel` took `m: Int`, which is not DevicePassable, so the
   x_cluster GPU binding did not instantiate on this toolchain; `m` is Int32 now.

## Policy (Andrew 2026-09-28, ~08:25Z)

No verification in this lane: no identity requests, no sabotage runs. The identity
request 1790583161949 and the DBSCAN probe were withdrawn before they ran. The
sabotage arms and checks added above are for the orchestrator's one check on
lane/apple-merged. Only speed measurements (+ digests for IDENTICAL, paired quality
for FAST) run here.

## Requests in flight (superseded; see the results section)

- identity 1790583161949-cluster-819df01ee2: 24 lanes (18 x-cluster, hdbscan,
  hdbscan-leaf, x-cluster-hdbscan-epsilon, agglomerative, gmm, gmm-random-init),
  --pass 2 (every cluster.checks arm), sabotage
  ~/mojolearn-evidence/cluster-apple/combined_e2e_v4.patch; m2pro, m3ultra-b, m4pro-b,
  do-amd.
- speed after 1790583183566 (m4pro-a) and 1790583186740 (m3ultra, + GMM stage times);
  m3ultra before 1790582043709 (5b622763d); HDBSCAN/BGMM stage profile 1790583192185
  (m3ultra, lane/cluster-apple-prof b1afa236c).
- DBSCAN M4 probe 1790582090404 (m4pro-a, lane/cluster-apple-prof 6a34eb370).

## Results: M3 Ultra (m3ultra), IDENTICAL, before 5b622763d (1790582043709) -> after 819df01ee (1790583186740)

819df01ee = changes 1-6 (not 7-9). Every digest equals the before / H100 records.

| case | rows | taxi before -> after (s) | higgs before -> after (s) |
|---|---|---|---|
| minibatch-kmeans | 1M | 0.2534 -> 0.2426 | 0.3313 -> 0.3099 |
| bisecting-kmeans | 1M | 0.2872 -> 0.2903 | 0.2999 -> 0.3012 |
| gmm | 1M | 2.585 (31adcffb1) -> 2.920 | 2.482 (31adcffb1) -> 2.711 |
| bayesian-gmm | 100k | 1.7649 -> 1.7562 | 2.3394 -> 2.3206 |
| affinity-prop | 5k | 2.6912 -> 2.2112 | 1.5904 -> 1.4382 |
| optics | 10k | 0.4146 -> 0.4857 | 0.4104 -> 0.4846 |
| meanshift | 10k | 0.2956 -> 0.2931 | 0.2775 -> 0.2750 |
| spectral | 10k | 0.5026 -> 0.5183 | 0.0887 -> 0.0896 |
| agglomerative-ward | 10k | 1.5637 -> 1.5501 | 1.6538 -> 1.6642 |
| agglomerative (single) | 10k | 0.0888 (31adcffb1) -> 0.0847 | 0.0845 (31adcffb1) -> 0.0800 |
| hdbscan | 40k | 2.2273 (main + prof) -> 1.6168 | 2.2169 (31adcffb1) -> 1.6051 |

Reading: HDBSCAN -27% (DENSE solver + index math), AffinityPropagation -18/-10%
(block-per-row responsibilities). The speculative chain did NOT pay on the M3 Ultra:
BGMM flat, and GaussianMixture read SLOWER than its 31adcffb1 records (+13%/+9%;
GMM_STAGE_TIMES at 819df01ee: 22 iterations, E-step 804 ms, M-step 1407 ms, Cholesky
317 ms). Those chains are latency-bound on their loads, not on the result flush; the
flag adds instructions to a one-thread (meanll) or eight-thread (nk) kernel. To be
settled on m4pro-a against its own before; if it holds there, the chain is reverted
in mixture (and in the moments kernel, where it bought nothing).
