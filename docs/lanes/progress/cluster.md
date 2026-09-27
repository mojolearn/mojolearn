# cluster: progress

Lane `cluster` of the algorithm expansion (lane/algos-cluster). Pass 1: build +
sklearn sanity + `tools/algos_lane_check.sh` AGREE (CPU == NVIDIA, H100 pod).

Design: `x_cluster/` is one generic driver per algorithm over the `ClusterOps`
trait (`x_cluster/ops.mojo`): `DeviceOps` (GPU binding `_mojolearn_x_cluster`)
and `HostOps` (CPU binding `_mojolearn_x_cluster_host`) run the SAME
per-element bodies (`x_cluster/bodies.mojo`); host control code is one source
compiled into both. One Python call `x_cluster_call(which, ...)`; entries in
`x_cluster/entries.mojo`; classes in `python/mojolearn/_expansion_cluster.py`.

| algorithm | lane | commit | pass-1 gate |
|---|---|---|---|
| MiniBatchKMeans | x-cluster-minibatch-kmeans | a665cf2ae | sanity ARI 1.0 vs sklearn, inertia rel 3.7e-4; AGREE: compared batch 9, infer 9, train 9 |
| BisectingKMeans | x-cluster-bisecting-kmeans | 314edcfcf | sanity ARI 1.0 vs sklearn, inertia rel 2e-7; AGREE: compared batch 9, infer 9, train 9 |
| MeanShift | x-cluster-meanshift | 7f7dee47e | sanity: bandwidth equal to sklearn's estimate to 1e-7 rel, 3 centers as sklearn, ARI 1.0; AGREE: compared batch 9, infer 9, train 9 |
| OPTICS | x-cluster-optics | f393b3673 | sanity: ordering_ equal to sklearn's on every row, reachability max diff 3.2e-7, ARI 1.0; AGREE: compared train 9 (transductive) |
| AffinityPropagation | x-cluster-affinity-propagation | c87bf418e | sanity: 5 exemplars as sklearn, ARI 1.0 (60 vs 67 iterations: a different noise stream); AGREE: compared batch 9, infer 9, train 9 |
| BayesianGaussianMixture | x-cluster-bgmm | f09870a3d | sanity: weights within 5e-4 of sklearn's, lower_bound rel 8e-7, ARI 1.0; AGREE: compared batch 9, infer 9, train 9; all six lanes re-checked together: AGREE |

Note (2026-09-27): the BisectingKMeans drift was a dangling pointer in OUR DeviceOps.kmeans (a local List freed at its last use), not a kmeans_fit defect; fixed by a keepalive.

Pass-1 list DONE (six of six). Next: PASS 2 and the CURRENT DIRECTIVES at the top of docs/lanes/ALGORITHM_EXPANSION_PLAN.md (option parity, per-seam proof, AMD box, Apple steward, speed).
## Pass 2

Proof (step 2), all six algorithms, NVIDIA H100 pod:

- Seams: DEVIATIONS 5100-5110 at the bodies in `x_cluster/bodies.mojo`,
  IDENTITY_PATHS rows 110-119, table in `x_cluster/README.md`.
- Host oracles `x_cluster/checks/oracles.mojo`; eight check drivers listed in
  `tools/identity_lanes/cluster.checks` (dist, nearest, kth, meanshift, ap,
  descend, gauss, moments); each first proves its fixture SEPARATES the
  pinned spelling (VACUOUS otherwise), then device == oracle and host ==
  oracle bit for bit; each records its stage on the card (IdentityTrace).
- Per-seam sabotage arms (source patches of `x_cluster/bodies.mojo`, kept in
  ~/mojolearn-evidence/algos-cluster/sabotage/): 5100, 5101, 5102, 5103,
  5104, 5105, 5106, 5107, 5108, 5109, 5110, every one BITES its check.
- End-to-end sabotage for the lane check (`e2e_device_fold_reversed.patch`:
  the device kernels of sqdist, nearest, meanshift and gauss_q take the
  reversed fold, the host does not).
- OWED: AMD column (dev_pod `--vendor amd`), M2 Pro steward PASS, then option
  parity and speed (CURRENT DIRECTIVES).
