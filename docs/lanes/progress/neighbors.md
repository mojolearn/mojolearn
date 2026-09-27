# neighbors: progress

Lane table (main), then Additions, then Graph and GP additions. One commit per
algorithm. Pass-1 gate: builds on the pod, `python/mojolearn/tests/test_x_neighbors_sanity.py`
passes against scikit-learn, `tools/algos_lane_check.sh <lane>` reads AGREE.

HOW THE LANE IS BUILT: every primitive is an item function in
`x_neighbors/items.mojo`; `x_neighbors/gen.py` generates the GPU driver
(`device_ops.mojo`), the host driver (`host_ops.mojo`) and both bindings from
its OPS table (run it after editing the table). Python (`_expansion_neighbors.py`)
only moves buffers and does exact integer bookkeeping.

Pod setup that is not in the repo: the default pixi env needs
`python -m pip install scikit-learn pytest networkx` for the sanity tests.

| algorithm | commit | lane | pod verdict |
|---|---|---|---|
| LocalOutlierFactor | df17471f9 | x-neighbors-lof | AGREE: compared batch 9, infer 9, train 9 (cuda H100 vs CPU Xeon 8470); sanity vs sklearn 1.9.1 PASS |
| NearestCentroid | 2d1278378 | x-neighbors-nearest-centroid | AGREE: compared batch 9, infer 9, train 9 (cuda H100 vs CPU Xeon 8470); sanity vs sklearn 1.9.1 PASS |
| OneClassSVM | 9662ec3e3 | x-neighbors-ocsvm | AGREE: compared batch 9, infer 9, train 9 (cuda H100 vs CPU Xeon 8470); sanity vs sklearn 1.9.1 PASS |
| KernelPCA | 8f88cf4b0 | x-neighbors-kpca | AGREE: compared batch 9, infer 9, train 9 (cuda H100 vs CPU Xeon 8470); sanity vs sklearn 1.9.1 PASS |
| PolynomialCountSketch | 34743f12e | x-neighbors-poly-sketch | AGREE: compared batch 9, infer 9, train 9 (cuda H100 vs CPU Xeon 8470); sanity vs sklearn 1.9.1 PASS |
| AdditiveChi2Sampler | 8d9b98f02 | x-neighbors-additive-chi2 | AGREE: compared batch 9, infer 9, train 9 (cuda H100 vs CPU Xeon 8470); sanity vs sklearn 1.9.1 PASS |
| SkewedChi2Sampler | a2e6df29a | x-neighbors-skewed-chi2 | AGREE: compared batch 9, infer 9, train 9 (cuda H100 vs CPU Xeon 8470); sanity vs sklearn 1.9.1 PASS |
| LabelPropagation | 02ae00a00 | x-neighbors-label-propagation | AGREE: compared batch 9, infer 9, train 9 (cuda H100 vs CPU Xeon 8470); sanity vs sklearn 1.9.1 PASS |
| LabelSpreading | 51524eb28 | x-neighbors-label-spreading | AGREE: compared batch 9, infer 9, train 9 (cuda H100 vs CPU Xeon 8470); sanity vs sklearn 1.9.1 PASS |
| KNNImputer | 906444f23 | x-neighbors-knn-imputer | AGREE: compared batch 9, infer 9, train 9 (cuda H100 vs CPU Xeon 8470); sanity vs sklearn 1.9.1 PASS |
| PageRank | 6b9167b6e | x-neighbors-pagerank | AGREE: compared infer 9, train 9 (cuda H100 vs CPU Xeon 8470); sanity PASS (networkx / scipy / closed-form GPflow bound) |
| connected_components | fec9a491c | x-neighbors-connected-components | AGREE: compared train 9 (no out-of-sample method) (cuda H100 vs CPU Xeon 8470); sanity PASS (networkx / scipy / closed-form GPflow bound) |
| Louvain | 325843987 | x-neighbors-louvain | AGREE: compared train 9 (no out-of-sample method) (cuda H100 vs CPU Xeon 8470); sanity PASS (networkx / scipy / closed-form GPflow bound) |
| SVGP | c6c2badf3 | x-neighbors-svgp | AGREE: compared batch 9, infer 9, train 9 (cuda H100 vs CPU Xeon 8470); sanity PASS (networkx / scipy / closed-form GPflow bound) |

Pass-1 list DONE (14 of 14, main 8af392436).

## Pass 2

Proof (step 2), all fourteen algorithms, NVIDIA H100 pod:

- Seams: DEVIATIONS 5200-5217 (5211 unused), IDENTITY_PATHS rows 120-129,
  table in `x_neighbors/README.md`.
- Host oracles `x_neighbors/checks/oracles.mojo` (restatements, plus the
  unpinned spelling of each seam); six check drivers listed with their arms
  in `tools/identity_lanes/neighbors.checks` (dist, fold, model, sketch,
  semi, graph). Each proves its fixture SEPARATES first (VACUOUS otherwise),
  then device == oracle and host == oracle bit for bit, and records its
  stages on the card (IdentityTrace).
- Per-seam sabotage arms, one patch of `x_neighbors/items.mojo` each, in
  `x_neighbors/checks/sabotage/`: 5200, 5201, 5202, 5203, 5204, 5205, 5206,
  5207, 5208, 5209, 5210, 5212, 5213, 5214, 5215, 5216, 5217.
- End-to-end device-only sabotage for the lane check / steward:
  `x_neighbors/checks/sabotage/e2e_device_only.patch` (a scalar of each
  device kernel moved; the host untouched).
- Lesson: a check driver must keep every List alive past the op that reads
  its address (`_ = l^`); Mojo frees at the last use and taking an address
  is not a use (the first dist_check read a freed buffer on the host arm).
- NVIDIA H100 pod, 2026-09-27: `tools/algos_lane_check.sh <all 14> --pass 2`
  seam arms 17/17 bite (each driver PASS, FAIL under its arm, PASS after
  reversal); every lane AGREE. `--sabotage e2e_device_only.patch` (seams not
  re-run): AGREE, DISAGREE on all 14 lanes, AGREE after reversal: PASS.
- Seam arms RE-PROVEN ONCE on the fixed lane check (CURRENT DIRECTIVES 000;
  merged tree 799d1f0d3, `seam_checks(..., pass 2)` alone): 17/17 FAIL under
  their arm (a real driver failure, not a build break), PASS after reversal.
  Done; never repeat it.
- Apple / AMD stewards: submitted 799d1f0d3 (all 14 lanes, `--sabotage
  x_neighbors/checks/sabotage/e2e_device_only.patch`), request
  1790533314520-neighbors-799d1f0d31, queued on m2pro and do-amd, spooled for
  m3ultra. Read `python3 tools/apple_steward.py status`; a FAIL comes back to
  this lane.
- AMD dev box: `tools/dev_pod.sh up neighbors 240 --vendor amd` found no
  RunPod MI300X stock and no free Hot Aisle slot (twice, 2026-09-27); a
  retry was left running (log ~/mojolearn-evidence/devpods/neighbors-amd-up.log).

- Apple / AMD steward verdict for 1790533314520-neighbors-799d1f0d31: m2pro
  PASS, do-amd PASS (m3ultra spooled, not gating). All 14 lanes proven on
  NVIDIA, AMD, Apple and CPU. Done; never repeat it.
- connected_components end-to-end arm STRENGTHENED (the old device arm froze
  node 1 and moved 1 fixture of 9): the device now reads edges forward only
  (a[t, j], never a[j, t]), which moves every directed weak graph, and the
  lane gained a directed CHAIN fixture (`_neighbors_chain`: each bin a path
  i -> next i of its bin) so the smallest label travels many rounds.

## Option parity (phase c)

Branch commit 9055f378e (OCSVM sample_weight + kernel='precomputed',
KernelPCA 'precomputed', NearestCentroid deviations_ / predict_log_proba /
score, PageRank dangling + nstart, KNNImputer numeric missing_values, sparse
input, RandomState instances, get_feature_names_out) plus new seam arms
5200_smo_bound_unweighted, 5209_log_softmax_fold_reversed,
5212_nc_deviations_unthresholded, 5216_pagerank_dangling_follows_p.
`x_neighbors/NOT_IMPLEMENTED.tsv` rows updated to IMPLEMENTED.

NVIDIA H100 pod, 2026-09-27, merged tree 82ee9be8d + worktree:
`tools/algos_lane_check.sh <the 14 lanes> --pass 2 --sabotage
x_neighbors/checks/sabotage/e2e_device_only.patch`: seam arms 21/21 bite
(PASS, FAIL under the arm, PASS after reversal); every lane AGREE, DISAGREE
under the e2e arm (connected_components now on 9 of 9 fixtures), AGREE after
reversal: RESULT PASS. Apple / AMD: request 1790537199548-neighbors-689ddc2561
(m2pro + do-amd).

### Existing family (neighbors/, kernel_methods/, svm/, gaussian_process/)

MERGED (NVIDIA H100 pod, 2026-09-27; lane check on the 74 non-par lanes
lane_select picks for the svm / kernel_methods / x_neighbors diff: AGREE;
check-svm 44/44; test_host_surface, x_neighbors sanity, svc_poly,
kernel_methods and svr surfaces, test_lane_select: pass):
- gamma='scale' for SVC, SVR and RBFSampler (DEVIATION 870 revised,
  python/mojolearn/_scale_gamma.py: the exact variance of the float32 cells,
  the reciprocal rounded once). Lane x-neighbors-gamma-scale; arm
  870_gamma_scale_device_column.patch: PASS (AGREE, DISAGREE, AGREE).
- SVC / SVR sample_weight and SVC class_weight (InitPenalty's weighted arm:
  per-row C * cw[y] * w, rounded once to float32 on the host; C_vec on the
  device, per-index bound in smo_oracle_fit). Lane x-neighbors-svm-weights;
  arm svm_weights_device_unweighted.patch: PASS.
- SVC kernel='sigmoid'; SVR kernel='poly' and 'sigmoid' (kernel_methods'
  TANH / DEVIATION 1663 epilogues). Lanes x-neighbors-svc-sigmoid,
  x-neighbors-svr-kernels; arm svc_sigmoid_device_gain.patch: PASS on both.
- x_neighbors ONE process-lifetime DeviceContext (gen.py `xn_ctx`, the
  second-GPU-call hang); test_x_neighbors_repeat.py (every entry point twice,
  GPU and CPU, equal bits): pass; the 14 lanes AGREE after it.
- test_lane_select pins: forest_host_predict 81, forest_inference 47
  (trees-dt-random); the generator test selects the lanes gen.py's outputs
  reach.
- Stewards: requests 1790543082631 / 1790543091585 / 1790543098464 (the
  three option lanes) and 1790537199548 (14 lanes); post-merge gates now.

OWED, phase 2 (option parity), in this order (each: AGREE + a sabotage +
existing bits unchanged):
  1. SVC multiclass (one-vs-one over the binary solver, host bookkeeping),
     decision_function_shape 'ovr' / 'ovo', break_ties.
  2. kernel='precomputed' for SVC / SVR / KernelRidge / Nystroem.
  3. KernelRidge sample_weight (sqrt(w) scaling of K and y, dual *= sqrt(w)).
  4. cosine / chi2 / additive_chi2 pairwise kernels (KernelRidge, Nystroem).
  5. GaussianProcessRegressor predict(return_cov=True); RationalQuadratic,
     ExpSineSquared, DotProduct kernels.
  6. Sparse input (densified exactly) for SVC, SVR, KernelRidge, Nystroem,
     RBFSampler, the k-NN classes.
  7. The distance metrics neighbors/NOT_IMPLEMENTED.tsv refuses by name.
  8. SVC probability (Platt with libsvm's internal CV: a seeded fold
     assignment).
  PHASE 1 AUDIT for the existing algorithms (LANE CHARTER): confirm each of
  NearestNeighbors, kNN, radius, RBC, KDE, SVC/SVR, KernelRidge, Nystroem,
  RBFSampler, GP has per-seam oracles + separating fixtures + biting arms and
  Apple/AMD AGREE; list gaps here.
- NOTE: par-* lanes cannot run in the lane check's CPU arm (the parallel
  pool refuses); exclude them from a lane-check list.
- NOTE: never kill a lane check mid-seam-arm: it leaves the arm's patch
  applied on the pod (resync, then `git apply -R --check` every patch).

## NEXT (a fresh session starts here)

Phase 2 OPTION PARITY is in progress. Continue the OWED list under
"Existing family" from item 1 (SVC multiclass). Batch one steward request per
hour for lanes merged since the last one.
