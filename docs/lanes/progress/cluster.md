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

## Option parity, the EXISTING family (session 3, 2026-09-27)

Merged to main at f2b0b051e in one batch. Batched steward request
1790538504301-cluster-f2b0b051e0 (all 18 x-cluster lanes, --pass 1, the
combined sabotage ~/mojolearn-evidence/cluster_combined_f2b0b051e.patch =
e2e_device_fold_reversed + the kmeans-init nearest cut + dbscan-metrics +
hdbscan-epsilon + hdbscan-probabilities): a FAIL comes back as a fix.
Each item, each with a lane, a
sabotage that bites where the change is numeric, existing lanes unchanged
(`lane_select --changed-since origin/main`, H100: every non-par lane AGREE,
51 lanes; the par-* lanes have no CPU arm in the lane check):

- DBSCAN metric='cosine' (DEVIATION 5113: `core/cosine_rows.mojo` scales
  rows to unit length on the HOST, one source for both bindings; the L2
  kernel against Float32(2 * eps); predict too) and metric='precomputed'
  (DEVIATION 5114: `eps_precomputed_neigh_kernel`, adj = D <= eps).
  Lane x-cluster-dbscan-metrics, sabotage e2e_dbscan_metrics.patch: PASS
  (AGREE / DISAGREE / AGREE, H100).
- DBSCAN core_sample_indices_ and components_ on every fit (cuML's
  calc_core_sample_indices default, scikit-learn's attributes); predict and
  save still need prediction_data=True. No numeric change.
- HDBSCAN cluster_selection_epsilon (DEVIATION 5115:
  `hdbscan/impl/detail/utils.mojo::cluster_epsilon_search_host`, host code
  both routes call; the labelling's epsilon branch, extract.cuh:148-153).
  Lane x-cluster-hdbscan-epsilon, sabotage e2e_hdbscan_epsilon.patch: PASS.
- HDBSCAN probabilities_ (DEVIATION 5116: `extract.mojo::
  get_probabilities_host`, both bindings, a new trailing fit address).
  Sabotage e2e_hdbscan_probabilities.patch on x-cluster-hdbscan-epsilon:
  PASS.
- KMeans init as an array or a callable, MiniBatchKMeans init callable
  (`_expansion_cluster._callable_init`, called once on the host, then the
  array path). Lane x-cluster-kmeans-init, sabotage e2e_kmeans_init.patch
  (the device nearest skips the last center; e2e_device_fold_reversed did
  not bite it on do-amd). No Mojo change.
- GaussianMixture precisions_init: already routed to x_cluster/bgmm.mojo;
  the stale mixture tsv row fixed and x-cluster-gmm-options now hashes it.
- sklearn sanity (3 seeds, sklearn 1.9): cosine ARI 1.0 on all three;
  precomputed labels EXACT on all three; HDBSCAN labels ARI 0.98-1.0 at
  epsilon 0 / 0.3 / 3 (at 3.0 the paired blobs merge, 8 -> 4 clusters;
  scikit-learn 1.9's own traverse_upwards raises a TypeError there, its
  bug); probabilities_ within 0.019 of scikit-learn's once min_samples is
  matched (cuML counts min_samples without the point itself, scikit-learn
  with it; the residual is cuML's expanded float32 L2).
- test_host_surface: 196 passed; test_lane_select: kmeans_oracle pin
  63 -> 64 (x-cluster-kmeans-init), attributed; the rest pass (two git-reachability probes fail only on the pod's
  synced, uncommitted tree under MOJOLEARN_LANE_SELECT_TEST_FORCE; they passed
  in the unforced run).
- Steward requests at 6e54f880b: do-amd PASS for dbscan-metrics,
  hdbscan-epsilon, hdbscan probabilities; kmeans-init resubmitted with its
  own patch (e2e_kmeans_init.patch: PASS on H100). Apple/AMD verdicts are post-merge release gates (0000b).
- FIXED AT THE ROOT (TOP PRIORITY from the cpu lane): the x_cluster GPU
  binding hung on its SECOND call in a process (RTX 4090, futex wait): the
  DeviceContext was a per-call DeviceOps field declared before the call's
  buffers. Now one process-lifetime context (`device_ops.mojo::
  x_cluster_ctx`, the x_cnn `_Global` pattern). Regression test
  `python/mojolearn/tests/test_x_cluster_twice.py` (every entry point twice
  in one process, GPU and CPU, byte-equal): 2 passed on H100. All 18
  x-cluster lanes AGREE after the fix (CPU path untouched, so the GPU bits
  are unchanged).

Next, in order (LANE CHARTER at the top of ALGORITHM_EXPANSION_PLAN.md; one
phase per session). Phase 1 (verification) holds for the six new algorithms
(see Pass 2 above); the existing six (KMeans, DBSCAN, HDBSCAN, Agglomerative,
SpectralClustering, GaussianMixture) carry release-record lanes (kmeans*,
dbscan*, hdbscan*, agglomerative, spectral*, gmm*).
1. PHASE 2, option parity, CONTINUES (this session merged the batch above),
   what is left in the family:
   - AgglomerativeClustering: linkage 'ward' (scikit-learn's DEFAULT),
     'complete', 'average'; metric l1/cosine/precomputed; distance_threshold
     and compute_distances (the per-merge deltas build_dendrogram_host already
     produces, hierarchy/README.md); a connectivity matrix; connectivity='knn'
     (hierarchy tsv row 2, cuML's Python default).
   - SpectralClustering: affinity='rbf' (scikit-learn's DEFAULT),
     assign_labels 'discretize' / 'cluster_qr', affinity_matrix_.
   - MeanShift estimate_bandwidth(n_samples) subsampling; OPTICS remaining
     metrics (x_cluster tsv rows 8, 10); BisectingKMeans callable init
     (refused: called per bisection inside the Mojo loop).
2. Phase 3 FAST speed, then phase 4 IDENTICAL speed, NVIDIA/AMD/Apple at
   1M+ rows from R2 (timing via apple_steward.py submit --kind speed
   --target m3ultra|do-amd): the OPTICS ordering loop and AffinityPropagation
   iterations are host / per-iteration-sync bound; MeanShift is one thread
   per seed.
3. Phase 5 CPU speed last.

## Session 4 (2026-09-28): option parity continued, NOT MERGED (no pod)

RunPod's account balance went negative mid-session: every RunPod pod was
deleted (ours, hvr2m4dqzf75z3, mid lane check) and `dev_pod.sh up` is
refused ("balance too low"). Orchestrator: do not retry renting. Nothing
below has run on a GPU yet. Branch lane/algos-cluster, pushed.

Code on the branch (all committed):
- AgglomerativeClustering on the x_cluster route (`x_cluster/agglo.mojo`,
  `_hierarchy_impl._fit_x`, ENTRY_AGGLO = 11): linkage ward / complete /
  average / single, metric euclidean / l1 (manhattan, cityblock) / cosine /
  precomputed, a connectivity matrix (scikit-learn's component join),
  compute_full_tree, compute_distances, distance_threshold. DEVIATIONS
  5117 (Lance-Williams) and 5118 (merge order), seam check
  `x_cluster/checks/agglo_check.mojo` with arms 5117/5118; lanes
  x-cluster-agglo-linkages and x-cluster-agglo-connectivity.
  connectivity='knn' (cuML's graph + cross-component fix-up) stays refused.
- SpectralClustering: affinity 'rbf' (gamma, default 1.0) and
  'precomputed_nearest_neighbors', affinity_matrix_; assign_labels
  'discretize' and 'cluster_qr' (`x_cluster/spectral_assign.mojo`,
  ENTRY_SPECTRAL_ASSIGN = 12, host float64, one-sided Jacobi SVD, DEVIATION
  5119; seam check `x_cluster/checks/spectral_assign_check.mojo`, arm
  5119_svd_fold.patch). Lane x-cluster-spectral-affinities (e2e sabotage to
  use: `x_decomp/checks/sabotage/e2e_host_sqdist.patch`, the rbf matrix;
  the assign step is host code in both bindings).
- IDENTITY_PATHS row 199; x_cluster/README rows 5117-5119; hierarchy and
  spectral NOT_IMPLEMENTED rows; test_x_cluster_twice now also calls the
  agglo and spectral_assign entries.

Checked WITHOUT a pod (Mac, one core, host code only):
- spectral_assign_check: PASS under IDENTICAL (fixture separates, 230
  cells); arm 5119 BITES (u differs in 4 cells), PASS after reversal.
- scikit-learn 1.9 sanity of the Mojo host code (scratch drivers in
  ~/mojolearn-evidence/algos-cluster/sanity/): cluster_qr labels EXACT on
  5 blob embeddings from scikit-learn's spectral_embedding; discretize ARI
  1.0 against scikit-learn's on all 5. agglo_tree (HostOps) on 3 blob sets:
  every linkage x metric, with and without a kNN connectivity matrix,
  labels ARI 1.0, distances within 2e-7 relative (cosine within 1.7e-7
  absolute); children EXACT for ward/complete/average and precomputed;
  single linkage's pairs can print in the other orientation ((j, i) where
  scipy's MST walk visits j first; ours is always lower first), same tree.

OWED ON A POD, in order (NVIDIA merge gate, then merge + push):
1. `tools/algos_lane_check.sh x-cluster-agglo-linkages,x-cluster-agglo-connectivity
   --pass 2 --sabotage x_cluster/checks/sabotage/e2e_device_fold_reversed.patch`
   (arms 5117, 5118 plus the seam drivers; AGREE / DISAGREE / AGREE).
2. `tools/algos_lane_check.sh x-cluster-spectral-affinities --pass 2
   --sabotage x_decomp/checks/sabotage/e2e_host_sqdist.patch` (arm 5119).
3. `python3 tools/lane_select.py --changed-since origin/main`: every
   selected lane AGREE (existing bits unchanged: agglomerative*, spectral*
   and the x-cluster lanes).
4. `pytest python/mojolearn/tests/test_host_surface.py` and
   `python/mojolearn/tests/test_x_cluster_twice.py` (GPU + CPU);
   `tools/test_lane_select.py` (inputs changed: new lanes and a new Mojo
   file; the kmeans_oracle pin may move again, attribute it).
5. Merge to main and push in one command; ONE batched steward request
   (do-amd + Apple) for the three new lanes with the e2e patches.

Still open in phase B after that: MeanShift estimate_bandwidth(n_samples)
subsampling; OPTICS metrics beyond the tsv's carried set; BisectingKMeans
callable init (refused: per bisection inside the Mojo loop);
AgglomerativeClustering connectivity='knn'.
