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
- `.checks` now pairs every driver with its sabotage patch
  (x_cluster/checks/sabotage/); `algos_lane_check.sh --pass 2` over all 15
  x-cluster lanes with the e2e patch: every seam PASS / FAIL under its
  patch / PASS; every lane AGREE, DISAGREE, AGREE (H100, 2026-09-27).
- OWED: the AMD column (no MI300X stock on RunPod or Hot Aisle all
  afternoon; `dev_pod.sh up cluster 240 --vendor amd` retrying), the M2 Pro
  steward PASS, then speed.

- Directive 000 (2026-09-27, session 2): the seam bites above were recorded
  before the lane-check fix (3084ca09c, merged 02b63f107). Re-run ONCE on
  the H100 pod with the fixed tool (box `tools/algos_lane_check.py` md5 =
  origin/main's): `algos_lane_check.sh x-cluster-meanshift --pass 2`. All
  12 arms (5100-5111) BUILD, RUN and FAIL under their patch, PASS after
  reversal; no BROKEN arm; clean lane AGREE. Done; never repeat.

## Option parity (CURRENT DIRECTIVES item 2), the lane's own six first

Implemented (x_cluster/NOT_IMPLEMENTED.tsv rows removed or narrowed), each
with a verifier lane, existing lanes' bits unchanged:

- MiniBatchKMeans: init='random', sample_weight, partial_fit
  (x-cluster-minibatch-options, x-cluster-minibatch-partial)
- BisectingKMeans: sample_weight, 'largest_cluster' coverage
  (x-cluster-bisecting-options)
- MeanShift: bin_seeding / min_bin_freq (x-cluster-meanshift-binned)
- OPTICS: metrics minkowski(p), manhattan, chebyshev, cosine, precomputed
  (DEVIATION 5111, bodies.pdist_cell; x-cluster-optics-metrics)
- AffinityPropagation: the median preference of a precomputed matrix with
  positive entries (x-cluster-ap-precomputed)
- BayesianGaussianMixture: covariance_type tied/diag/spherical, init
  'k-means++'/'random_from_data', warm_start (x-cluster-bgmm-covtypes,
  x-cluster-bgmm-inits)

Still refused by name: callable init / callable metric, sparse input.
Next: the existing cluster family (KMeans, DBSCAN, HDBSCAN, GaussianMixture,
Agglomerative, Spectral) option rows, then speed.

## Fixed at the root (CURRENT DIRECTIVES item 3)

- KMeans with sample weights above one (DEVIATION 5112): the fixed-point
  centroid sums quantize `x * w * sum_scale`, but the scale bounded
  `sum |x|` alone, so weights above one wrapped Int32 (2,000 blob rows,
  weights `|x0| + 0.5`: 300 iterations, inertia 1.75e6 against sklearn's
  3.97e4, ARI 0.43). `cluster/impl/kmeans_params.mojo::weighted_sum_scale_cap`
  caps the scale by the weighted bound in `kmeans_fit` and in the host
  oracle; weight vectors that never outgrow the unweighted bound keep their
  scale and bits (the `kmeans-weighted` lane's weights are in [0.5, 1.5]).
- tools/test_lane_select.py's kmeans_oracle pin 47 -> 63, attributed.

## Where the lane stands (2026-09-27, evening)

- Merged on main at e81b76c38: pass 1 (six algorithms), pass-2 proof, option
  parity for the six and for GaussianMixture, the KMeans weighted fix.
- Steward identity request `1790530630176-cluster-e81b76c386` (all 15
  x-cluster lanes, `--pass 2`, the e2e patch, commit e81b76c38): **PASS on
  m2pro (Apple Metal == Arm CPU) and PASS on do-amd (MI300X == x86 CPU)**;
  m3ultra spooled until its GPT-3 segment ends. With the H100 pod run, the
  lane is proven on NVIDIA, AMD, Apple and both CPU columns.
- AMD box of my own: none (no MI300X stock on RunPod or Hot Aisle all
  afternoon, retry stopped); do-amd carried the AMD column.
- NVIDIA pod `cluster` (H100) held; heartbeat `tools/dev_pod.sh extend
  cluster 120`.

Next, in order (CURRENT DIRECTIVES item 1):
1. (done) steward verdicts; a later m3ultra FAIL comes back here.
2. Option parity for the EXISTING cluster family: DBSCAN metric cosine and
   precomputed (dbscan/NOT_IMPLEMENTED.tsv rows), HDBSCAN
   cluster_selection_epsilon (hdbscan tsv), GaussianMixture save/sample for
   the routed options, the callable init/metric refusals.
3. GPU speed (IDENTICAL and FAST, NVIDIA/AMD/Apple) at 1M+ rows from R2:
   the OPTICS ordering loop and AffinityPropagation iterations are host /
   per-iteration-sync bound; MeanShift is one thread per seed.
4. CPU speed last.
