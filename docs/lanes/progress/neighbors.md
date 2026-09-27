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
- OWED: AMD column (RunPod MI300X out of stock and Hot Aisle full on
  2026-09-27 16:xx; `up --vendor amd` retrying), M2 Pro steward, then option
  parity and speed (CURRENT DIRECTIVES).
