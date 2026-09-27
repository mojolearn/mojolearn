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
| BayesianGaussianMixture | x-cluster-bgmm | (this commit) | sanity: weights within 5e-4 of sklearn's, lower_bound rel 8e-7, ARI 1.0; AGREE: compared batch 9, infer 9, train 9; all six lanes re-checked together: AGREE |

Note (2026-09-27): the BisectingKMeans drift was a dangling pointer in OUR DeviceOps.kmeans (a local List freed at its last use), not a kmeans_fit defect; fixed by a keepalive.

Pass-1 list DONE (six of six). Next: PASS 2 and the CURRENT DIRECTIVES at the top of docs/lanes/ALGORITHM_EXPANSION_PLAN.md (option parity, per-seam proof, AMD box, Apple steward, speed).