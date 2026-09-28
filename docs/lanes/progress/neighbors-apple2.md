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
| eb1a0a257 | x_neighbors `knn_sq` item: sqdist fused into the strict-< insertion, no n x m matrix (LOF, label propagation / spreading) | both | same statements; host check 0 slots differ | default; `MOJOLEARN_XN_UNFUSED_KNN=1` arm |
| 6f24c1d8b | OneClassSVM: the one-class SMO over one threadgroup (was ONE GPU thread); item factored into shared helpers | both | host check: refactored item == old item word for word; GPU arm vs serial arm pending | default; `-D MOJOLEARN_XN_SERIAL_SMO` arm |
| df27366e9 | LabelPropagation / LabelSpreading fit: the iteration as one resident op `lp_iterate` (the n x n graph uploaded once, not per iteration) | both | same kernels, same order; host check: iterations and every word equal | default; `MOJOLEARN_XN_HOST_LOOPS=1` arm |
| 4c4e5978d | PageRank / connected_components: resident `pr_iterate` / `cc_iterate` | both | same kernels, same order; host check equal | default; same arm |
| d0a0aeeae | PolynomialCountSketch split per cell (`pcs_resident`), LabelSpreading degrees once (`col_degree` + `ls_laplacian_deg`), PageRank dangling rows on the device (`row_all_zero`), Louvain's item on the host in the GPU binding (gen.py HOST_RUN) | both | host checks 0 differ (PCS degrees 1 to 3, laplacian) | default; `MOJOLEARN_XN_OLD_ITEMS=1`, `-D MOJOLEARN_XN_LOUVAIN_GPU` arms |
| a898eb9c2 | Cholesky multi-RHS forward sweep: 256 threads, later rows four at a time (after DEVIATION 6150 moved the chains into b cells, GPC.predict_proba went 0.26 -> 1.05 s) | both (Apple) | same chains | default; `-D MOJOLEARN_CHOL_MR_ROWWISE` arm |
| 9866b507f | x_neighbors large downloads through 64 MB host staging, copied over the host cores | both | a copy | default; `-D MOJOLEARN_XN_PLAIN_DOWN` arm |
| b7bec306c | label propagation stopping sum folded on the host (same item); PCS convolution one row per block in threadgroup memory | both | same statements | default; `-D MOJOLEARN_XN_LP_DEVICE_FOLD`, `-D MOJOLEARN_XN_PCS_CELL` arms |
| a1bb804fb | label propagation / spreading: the graph product over G's nonzero entries (CSR built once on the host), exact by the fma-with-zero argument; dense kernel when an x is non-finite | both | host check (signed values, -0.0): 0 words differ | default; `-D MOJOLEARN_XN_LP_DENSE` arm |
| d7934f99f | x_neighbors k-NN with y rows staged per block in threadgroup memory (`knn_sq_tiled`; LOF, label propagation) | both | the item's statements, same candidate order | default; `MOJOLEARN_XN_OLD_ITEMS=1` arm |
| ff625ac0c | NearestCentroid group means / std, variance, SVGP solve, absdiff_sum: item loops on the host in the GPU binding (HOST_RUN) | both | the CPU column's statements | default; `-D MOJOLEARN_XN_SERIAL_GPU` arm |
| c905e9e47 | KNNImputer: fit rows staged per block (`knn_impute_tiled`); item tail factored into `knn_impute_finish` | both | host check of the refactor: 0 differ | default; `MOJOLEARN_XN_OLD_ITEMS=1` arm |
| da21bc669 | REVERTED the identical radix device barrier: `air.wg.barrier` failed to legalize in the IDENTICAL build.sh / build_metrics.sh (request 1790604321269) | - | - | - |
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

## Results

### Request 1790604321269 (m4pro-a, M4 Pro, b89ad2efa), FAST, one job

IDENTICAL did not build there (the radix barrier, reverted in da21bc669), so
this request is FAST only. old = `-D MOJOLEARN_XN_SERIAL_SMO` +
`MOJOLEARN_XN_UNFUSED_KNN=1 MOJOLEARN_XN_HOST_LOOPS=1
MOJOLEARN_XN_UNCOMPACT_IMPUTE=1`, new = default. Fit / predict seconds,
reps 1 after a load run, forward and reverse arms; digests equal old vs new
in EVERY row. Raw: ~/mojolearn-evidence/neighbors-apple/1790604321269-speed-neighbors-b89ad2efa9.txt

| case | taxi old | taxi new | HIGGS old | HIGGS new | digest (taxi / HIGGS) |
|---|---|---|---|---|---|
| OneClassSVM.fit 3k | 1.143 | 0.030 | 1.153 | 0.038 | d09b1f12.. / 8a7bab7c.. |
| LocalOutlierFactor.fit 20k | 0.928 | 0.448 | 0.953 | 0.239 | 428f08c2.. / 8dbff18c.. |
| LocalOutlierFactor.score_samples 5k | 0.240 | 0.124 | 0.234 | 0.106 | |
| LabelPropagation.fit 5k (knn) | 9.405 | 2.156 | 1.588 | 0.454 | 0cafc4a9.. / b02924bf.. |
| KNNImputer.transform 5k x 50k | 0.331 | 0.140 | 0.334 | 0.141 | 9950a0fe.. / 104fb09b.. |
| connected_components 5k | 0.187 | 0.065 | 0.084 | 0.033 | 173f8996.. / 4c6cd7e0.. |
| LabelSpreading.fit 5k | 2.233 | 2.161 | 2.244 | 2.155 | (degree refold: d0a0aeeae) |
| PageRank.fit 5k | 0.546 | 0.576 | 0.564 | 0.532 | (Python dangling scan: d0a0aeeae) |
| PolynomialCountSketch.transform 200k x 500 | 3.058 | 3.070 | 3.058 | 3.044 | (per-row item: d0a0aeeae) |
| Louvain.fit 1k | 2.584 | 2.491 | 3.973 | 4.129 | (one GPU thread: d0a0aeeae) |

Large k (FAST, same job): SpectralEmbedding(knn) 20k fits in 0.27 s;
NearestNeighbors k=2000: 20k x 2k 0.38 s, 100k x 500 (index tiled) 0.11 s;
every row ascending with distinct indices (quality column >= 0). FAST's
first 1,024 slots match its k = 1,024 answer on 0.999998 to 1.0 of slots
(FAST ties are not pinned).

Round-one family profile (MOJOLEARN_STAGE_TIMES, same job): SVC taxi fit
0.97 s = block_solve 0.41, select_ws 0.13, square_tile 0.10, full_tile
0.16-0.32 (201 outer iterations); KernelRidge 10k potrf 0.89 s taxi / 0.38
HIGGS, cho_solve 0.36 s; GPC.predict_proba 1.05-1.09 s in BOTH modes (0.26 s
at 395bc882a: the regression a898eb9c2 addresses); RBFSampler 1M x 500 copy
out 0.25 s.

### Request 1790605992789 (m4pro-a, M4 Pro, 9866b507f), IDENTICAL and FAST, one job

old = `-D MOJOLEARN_XN_SERIAL_SMO -D MOJOLEARN_XN_LOUVAIN_GPU
-D MOJOLEARN_XN_PLAIN_DOWN` + `MOJOLEARN_XN_UNFUSED_KNN=1
MOJOLEARN_XN_HOST_LOOPS=1 MOJOLEARN_XN_UNCOMPACT_IMPUTE=1
MOJOLEARN_XN_OLD_ITEMS=1` (every x_neighbors path before this lane); new =
default. Forward and reverse. DIGESTS EQUAL old vs new in every row of both
modes. Raw: ~/mojolearn-evidence/neighbors-apple/1790605992789-speed-neighbors-9866b507f5.txt

| case | mode | taxi old | taxi new | HIGGS old | HIGGS new |
|---|---|---|---|---|---|
| OneClassSVM.fit 3k | IDENTICAL | 1.155 | 0.026 | 1.179 | 0.026 |
| OneClassSVM.fit 3k | FAST | 1.069 | 0.025 | 1.234 | 0.027 |
| Louvain.fit 1k | IDENTICAL | 2.614 | 0.085 | 4.001 | 0.091 |
| Louvain.fit 1k | FAST | 2.491 | 0.044 | 4.137 | 0.052 |
| PageRank.fit 5k | IDENTICAL | 0.547 | 0.054 | 0.549 | 0.052 |
| PageRank.fit 5k | FAST | 0.539 | 0.052 | 0.564 | 0.051 |
| LabelSpreading.fit 5k | IDENTICAL | 2.466 | 0.163 | 2.451 | 0.154 |
| LabelSpreading.fit 5k | FAST | 2.236 | 0.143 | 2.251 | 0.146 |
| LabelPropagation.fit 5k | IDENTICAL | 9.696 | 2.316 | 1.555 | 0.445 |
| LabelPropagation.fit 5k | FAST | 9.188 | 2.076 | 1.567 | 0.418 |
| PolynomialCountSketch.transform 200k x 500 | IDENTICAL | 3.102 | 0.759 | 3.083 | 0.753 |
| PolynomialCountSketch.transform 200k x 500 | FAST | 3.065 | 0.754 | 3.076 | 0.739 |
| SkewedChi2Sampler.transform 1M x 500 | IDENTICAL | 1.090 | 0.569 | 1.086 | 0.571 |
| SkewedChi2Sampler.transform 1M x 500 | FAST | 1.081 | 0.552 | 1.068 | 0.552 |
| AdditiveChi2Sampler.transform 1M | IDENTICAL | 0.053 | 0.032 | 0.049 | 0.025 |
| LocalOutlierFactor.fit 20k | IDENTICAL | 0.960 | 0.527 | 0.938 | 0.280 |
| LocalOutlierFactor.fit 20k | FAST | 0.925 | 0.439 | 0.988 | 0.239 |
| LocalOutlierFactor.score_samples 5k | IDENTICAL | 0.240 | 0.140 | 0.230 | 0.123 |
| KNNImputer.transform 5k x 50k | IDENTICAL | 0.407 | 0.162 | 0.406 | 0.166 |
| KNNImputer.transform 5k x 50k | FAST | 0.332 | 0.148 | 0.335 | 0.152 |
| connected_components 5k | IDENTICAL | 0.185 | 0.066 | 0.084 | 0.033 |
| NearestCentroid.predict 50k | IDENTICAL | 0.026 | 0.009 | 0.027 | 0.009 |
| KernelPCA.fit 500 (host Jacobi) | IDENTICAL | 1.874 | 1.934 | 1.279 | 1.274 |
| SVGP.fit 100k (HIGGS; taxi refuses: not PD at these hyperparameters) | IDENTICAL | | | 0.233 | 0.251 |

GPC / GP / KRR sweep arms (same job; base vs `-D MOJOLEARN_CHOL_MR_ROWWISE`,
forward and reverse, digests equal):

| case | mode | taxi rowwise | taxi a898eb9c2 | HIGGS rowwise | HIGGS a898eb9c2 |
|---|---|---|---|---|---|
| GaussianProcessClassifier.predict_proba 3k x 3k | IDENTICAL | 0.972 / 0.949 | 0.259 / 0.257 | 0.960 / 1.038 | 0.260 / 0.261 |
| GaussianProcessClassifier.predict_proba 3k x 3k | FAST | 0.907 / 0.966 | 0.225 / 0.228 | 1.007 / 0.994 | 0.228 / 0.224 |
| KRR / GPR fit, GPC fit | both | unchanged | unchanged | unchanged | unchanged |

Large k, IDENTICAL (same job; did not run before this lane: refused):

| case | taxi | HIGGS | check |
|---|---|---|---|
| SpectralEmbedding(affinity=nearest_neighbors).fit 20k (n_neighbors 2,000) | 0.769 s | 0.784 s | forward == reverse digest |
| NearestNeighbors(k=2000).kneighbors 20k x 2k | 0.188 s | 0.190 s | rows ascending, distinct; first 1,024 slots == the k = 1,024 answer on 100% of slots (indices and distances) |
| NearestNeighbors(k=2000).kneighbors 100k x 500 (two index tiles, wide merge) | 0.072 s | 0.072 s | same, 100% |

Round-one IDENTICAL k-NN digests at this head (nn, nn-k20, nn-ties, taxi
and HIGGS) equal round one's record (9ff75469.., a29e4118.., 34a8923f..,
f8814dc5.., 52412ee0.., 575ce1bc..).

### Request 1790608308824 (m4pro-a, M4 Pro, d7934f99f), IDENTICAL and FAST, one job

Arms: old = `-D MOJOLEARN_XN_LP_DEVICE_FOLD -D MOJOLEARN_XN_PCS_CELL
-D MOJOLEARN_XN_LP_DENSE` + `MOJOLEARN_XN_OLD_ITEMS=1` (the head of the
previous request), dense = `-D MOJOLEARN_XN_LP_DENSE`, new = default; each
forward and reverse, reps 2. DIGESTS EQUAL across all arms in every row.
Raw: ~/mojolearn-evidence/neighbors-apple/1790608308824-speed-neighbors-d7934f99fe.txt

| case | mode | taxi old | taxi dense | taxi new | HIGGS old | HIGGS dense | HIGGS new |
|---|---|---|---|---|---|---|---|
| LabelPropagation.fit 5k | IDENTICAL | 2.303 | 1.662 | 0.308 | 0.432 | 0.309 | 0.113 |
| LabelPropagation.fit 5k | FAST | 2.068 | 1.355 | 0.225 | 0.393 | 0.275 | 0.102 |
| LabelSpreading.fit 5k (old = per-cell degrees) | IDENTICAL | 2.337 | 0.089 | 0.091 | 2.342 | 0.089 | 0.091 |
| LocalOutlierFactor.fit 20k (old = untiled knn_sq) | IDENTICAL | 0.534 | | 0.095 | 0.280 | | 0.088 |
| LocalOutlierFactor.fit 20k | FAST | 0.439 | | 0.082 | 0.241 | | 0.077 |
| LocalOutlierFactor.score_samples 5k | IDENTICAL | 0.141 | | 0.053 | 0.120 | | 0.041 |
| PolynomialCountSketch.transform 200k x 500 (old = per-row item) | IDENTICAL | 2.994 | | 0.511 | 2.992 | | 0.518 |
| PolynomialCountSketch.transform 200k x 500 | FAST | 2.917 | | 0.506 | 2.904 | | 0.510 |
| KNNImputer.transform | both | unchanged (0.15 to 0.16) | | | | | |

### Request 1790609418287 (m4pro-a, M4 Pro, c905e9e47), IDENTICAL and FAST, one job

old = `-D MOJOLEARN_XN_SERIAL_GPU` + `MOJOLEARN_XN_OLD_ITEMS=1`, new =
default; forward and reverse, reps 2; digests equal in every row. Raw:
~/mojolearn-evidence/neighbors-apple/1790609418287-speed-neighbors-c905e9e471.txt

| case | mode | taxi old | taxi new | HIGGS old | HIGGS new |
|---|---|---|---|---|---|
| NearestCentroid.fit 200k | IDENTICAL | 0.160 | 0.060 | 0.163 | 0.061 |
| NearestCentroid.fit 200k | FAST | 0.150 | 0.049 | 0.151 | 0.054 |
| SVGP.fit 100k, 64 inducing | IDENTICAL | | | 0.209 | 0.092 |
| SVGP.fit 100k, 64 inducing | FAST | | | 0.219 | 0.086 |
| KNNImputer.transform (tiled vs per-cell) | IDENTICAL | 0.160 | 0.150 | 0.160 | 0.147 |
| KernelPCA.fit 500, SkewedChi2Sampler | both | unchanged | | | |
