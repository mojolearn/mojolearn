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
| BisectingKMeans | x-cluster-bisecting-kmeans | (this commit) | sanity ARI 1.0 vs sklearn, inertia rel 2e-7; AGREE: compared batch 9, infer 9, train 9 |

Next: MeanShift, OPTICS, AffinityPropagation, BayesianGaussianMixture.
Then PASS 2 (docs/lanes/ALGORITHM_EXPANSION_PLAN.md, last section).
