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
| SVGP | (this commit) | x-neighbors-svgp | AGREE: compared batch 9, infer 9, train 9 (cuda H100 vs CPU Xeon 8470); sanity PASS (networkx / scipy / closed-form GPflow bound) |
