# neighbors-apple2: progress

Apple speed round 2 for the neighbors family (k-NN, radius, KDE, SVM / SVR,
GP / GPC, kernel approximation, KernelRidge, and the x_neighbors expansion:
LOF, NearestCentroid, OneClassSVM, KernelPCA, PolynomialCountSketch,
AdditiveChi2Sampler, SkewedChi2Sampler, LabelPropagation / LabelSpreading,
KNNImputer, PageRank, connected components, Louvain, SVGP).
Brief: ~/mojolearn-evidence/apple2_speed_brief.md. Branch lane/neighbors-apple2,
forked from lane/apple-merged 037daa353. Round one: docs/lanes/progress/neighbors-apple.md.

Board: `bench/x_neighbors_apple2_speed.py` (the expansion estimators and the
large-k k-NN shapes; taxi and HIGGS, first eight columns standardized).
Job script: `bench/x_neighbors_apple2_job.sh` (arms of build defines and
environment, forward then reverse, IDENTICAL and FAST in one job).

## Changes on the branch

| commit | change | modes | bits | default / arm |
|---|---|---|---|---|
| ddc2f96ad | IDENTICAL k-NN, k > 1024: the pinned radix selector in rounds of 1024 (each round above the previous round's last composite key) and a binary-search merge of index tiles (DEVIATION 6300). Root fix of SpectralEmbedding(knn) at 20k rows (default n_neighbors = 2,000) | IDENTICAL | k <= 1024 untouched; k > 1024 was a refusal | default |
| ddc2f96ad | identical radix kernel: barriers order device memory on Apple (`air.wg.barrier(3, 1)`) | IDENTICAL (Apple) | same by construction (a missing ordering can only ever have produced a wrong read) | default |
| eb1a0a257 | x_neighbors `knn_sq` item: sqdist fused into the strict-< insertion, no n x m matrix (LOF, label propagation / spreading) | both | same statements; host check 0 slots differ | default; `MOJOLEARN_XN_UNFUSED_KNN=1` arm |
| 6f24c1d8b | OneClassSVM: the one-class SMO over one threadgroup (was ONE GPU thread); item factored into shared helpers | both | host check: refactored item == old item word for word; GPU arm vs serial arm pending | default; `-D MOJOLEARN_XN_SERIAL_SMO` arm |
| df27366e9 | LabelPropagation / LabelSpreading fit: the iteration as one resident op `lp_iterate` (the n x n graph uploaded once, not per iteration) | both | same kernels, same order; host check: iterations and every word equal | default; `MOJOLEARN_XN_HOST_LOOPS=1` arm |
| 4c4e5978d | PageRank / connected_components: resident `pr_iterate` / `cc_iterate` | both | same kernels, same order; host check equal | default; same arm |
| b89ad2efa | KNNImputer.transform: one GPU thread per MISSING cell (`knn_impute_cells`, the same item per cell) | both | same statements per cell | default; `MOJOLEARN_XN_UNCOMPACT_IMPUTE=1` arm |

## Shared code touched (the integration run must cover it)

- `neighbors/checks/select_radix_identical.mojo` (the pinned radix select):
  reached by every IDENTICAL k-NN tiled selection (NearestNeighbors, kNN
  classifier / regressor, DBSCAN / HDBSCAN / UMAP / spectral kNN graphs) and
  by `ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo`. Device-scope barrier
  on Apple; the round kernel is new.
- `bindings/build_x_neighbors.sh` now passes MOJOLEARN_BUILD_EXTRA_DEFINES.
- `x_neighbors/gen.py`: BLOCK_OPS (threadgroup GPU form of a sequential
  item) and CUSTOM_OPS (hand-written resident drivers,
  `x_neighbors/iter_device.mojo` / `iter_host.mojo`); both bindings are
  regenerated.

## Speed requests

- 1790604321269 (m4pro-a, b89ad2efa): large-k validation (IDENTICAL + FAST),
  round-one k-NN digests, x_neighbors old / new arms (old = serial SMO +
  unfused kNN + host loops + per-cell imputer), a stage profile of the round-one family.
  Command: ~/mojolearn-evidence/neighbors-apple/r2_job1_cmd.txt. Raw:
  ~/mojolearn-evidence/neighbors-apple/<request>.txt
