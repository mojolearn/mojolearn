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
| 0e1394f6a | KNNImputer: each missing cell's donor scan split over 8 threads, merged by (distance, index) (host emulation: 0 differ) | both | same k-list, same tail | default; `-D MOJOLEARN_XN_IMPUTE_NO_SPLIT` arm |
| 4a8539a34 | PCS row convolution: four components per thread per pass | both | same chains | default (request 1790611824803: 0.51 -> 0.33 s IDENTICAL, 0.50 -> 0.30 FAST, digests equal to the previous head) |
| 28dc8c19d | merge origin/lane/apple-merged (M2 fix final: CHOL_MR_NT 256 for the row-by-row kernel, the GP context cleanup) | - | - | - |
| f71bfda90 | kneighbors host order pass (the (distance, index) check / sort after readback) over the host cores on raw pointers: it was 145 ms of a 190 ms IDENTICAL kneighbors at k = 2,000 | both | the same per-row statements | default; `-D MOJOLEARN_KNN_SERIAL_ORDER` arm |
| da21bc669 | REVERTED the identical radix device barrier: `air.wg.barrier` failed to legalize in the IDENTICAL build.sh / build_metrics.sh (request 1790604321269) | - | - | - |
| b89ad2efa | KNNImputer.transform: one GPU thread per MISSING cell (`knn_impute_cells`, the same item per cell) | both | same statements per cell | default; `MOJOLEARN_XN_UNCOMPACT_IMPUTE=1` arm |

## Shared code touched (the integration run must cover it)

- `neighbors/checks/select_radix_identical.mojo` (the pinned radix select):
  reached by every IDENTICAL k-NN tiled selection (NearestNeighbors, kNN
  classifier / regressor, DBSCAN / HDBSCAN / UMAP / spectral kNN graphs) and
  by `ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo`. Device-scope barrier
  on Apple; the round kernel is new.
- `bindings/build_x_neighbors.sh` now passes MOJOLEARN_BUILD_EXTRA_DEFINES.
- `cholesky/checks/trsm.mojo` (a898eb9c2): every Apple multi-RHS forward
  sweep (GP, GPC, KernelRidge with matrix y, the cholesky lane). It is a
  256-thread kernel holding a 4 x 8 float register tile per thread: m2pro
  must confirm it dispatches (the M2 dropped the 1024-thread sweeps).
- `neighbors/estimator.mojo` (f71bfda90): the k-NN host order pass, every
  k-NN result on every column.
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

### SUMMARY: lane base vs head, request 1790609896241 (m4pro-a, M4 Pro, 0e1394f6a), one job

old = every path this lane replaced, by its arms (`-D MOJOLEARN_XN_SERIAL_SMO
-D MOJOLEARN_XN_LOUVAIN_GPU -D MOJOLEARN_XN_PLAIN_DOWN -D MOJOLEARN_XN_SERIAL_GPU
-D MOJOLEARN_XN_LP_DEVICE_FOLD -D MOJOLEARN_XN_PCS_CELL -D MOJOLEARN_XN_LP_DENSE
-D MOJOLEARN_XN_IMPUTE_NO_SPLIT` + `MOJOLEARN_XN_UNFUSED_KNN=1
MOJOLEARN_XN_HOST_LOOPS=1 MOJOLEARN_XN_UNCOMPACT_IMPUTE=1
MOJOLEARN_XN_OLD_ITEMS=1`; GPC: `-D MOJOLEARN_CHOL_MR_ROWWISE`); new = the
default at 0e1394f6a. Forward and reverse in one job, minimum of each arm.
Every IDENTICAL digest is equal between old and new (1 distinct digest per
row); FAST digests equal between old and new too, except FAST k = 2,000 k-NN
rows, whose digests vary run to run INSIDE each arm (FAST's selector does not
pin ties; the code is the same in both arms). Seconds (fit, or predict /
transform where marked). Raw:
~/mojolearn-evidence/neighbors-apple/1790609896241-speed-neighbors-0e1394f6a9.txt

| algorithm | mode | taxi before | taxi after | HIGGS before | HIGGS after |
|---|---|---|---|---|---|
| LabelPropagation.fit 5k (knn) | IDENTICAL | 9.663 | 0.308 | 1.553 | 0.112 |
| LabelPropagation.fit 5k (knn) | FAST | 9.404 | 0.225 | 1.582 | 0.102 |
| LabelSpreading.fit 5k (knn) | IDENTICAL | 2.458 | 0.092 | 2.463 | 0.089 |
| LabelSpreading.fit 5k (knn) | FAST | 2.230 | 0.086 | 2.227 | 0.088 |
| Louvain.fit 1k | IDENTICAL | 2.522 | 0.084 | 4.001 | 0.091 |
| Louvain.fit 1k | FAST | 2.492 | 0.043 | 4.128 | 0.052 |
| OneClassSVM.fit 3k | IDENTICAL | 1.089 | 0.021 | 1.178 | 0.023 |
| OneClassSVM.fit 3k | FAST | 1.069 | 0.020 | 1.234 | 0.023 |
| LocalOutlierFactor.fit 20k | IDENTICAL | 0.927 | 0.095 | 0.937 | 0.089 |
| LocalOutlierFactor.fit 20k | FAST | 0.924 | 0.084 | 0.927 | 0.079 |
| LocalOutlierFactor.score_samples 5k | IDENTICAL | 0.234 | 0.053 | 0.230 | 0.040 |
| PolynomialCountSketch.transform 200k x 500 | IDENTICAL | 3.043 | 0.513 | 3.017 | 0.516 |
| PolynomialCountSketch.transform 200k x 500 | FAST | 3.025 | 0.501 | 3.037 | 0.505 |
| PageRank.fit 5k | IDENTICAL | 0.543 | 0.055 | 0.541 | 0.052 |
| PageRank.fit 5k | FAST | 0.548 | 0.052 | 0.550 | 0.052 |
| KNNImputer.transform 5k x 50k | IDENTICAL | 0.406 | 0.065 | 0.406 | 0.064 |
| KNNImputer.transform 5k x 50k | FAST | 0.333 | 0.056 | 0.331 | 0.055 |
| SkewedChi2Sampler.transform 1M x 500 | IDENTICAL | 1.083 | 0.572 | 1.088 | 0.572 |
| SkewedChi2Sampler.transform 1M x 500 | FAST | 1.080 | 0.548 | 1.068 | 0.544 |
| AdditiveChi2Sampler.transform 1M | IDENTICAL | 0.052 | 0.027 | 0.049 | 0.025 |
| connected_components 5k | IDENTICAL | 0.184 | 0.065 | 0.083 | 0.033 |
| NearestCentroid.fit 200k | IDENTICAL | 0.160 | 0.074 | 0.159 | 0.071 |
| NearestCentroid.fit 200k | FAST | 0.150 | 0.054 | 0.152 | 0.063 |
| SVGP.fit 100k (64 inducing) | IDENTICAL | | | 0.233 | 0.083 |
| SVGP.fit 100k (64 inducing) | FAST | | | 0.254 | 0.077 |
| KernelPCA.fit 500 (host Jacobi, untouched) | IDENTICAL | 1.874 | 1.883 | 1.281 | 1.277 |
| KernelPCA.transform 10k | IDENTICAL | 0.025 | 0.017 | 0.025 | 0.017 |
| GaussianProcessClassifier.predict_proba 3k x 3k | IDENTICAL | 1.130 to 1.255 | 0.259 to 0.265 | 1.147 to 1.215 | 0.256 to 0.259 |
| GaussianProcessClassifier.predict_proba 3k x 3k | FAST | 1.038 to 1.040 | 0.225 to 0.227 | 0.991 to 1.040 | 0.228 to 0.230 |
| SpectralEmbedding(knn).fit 20k | IDENTICAL | REFUSED (k > 1024) | 0.769 | REFUSED | 0.784 |
| NearestNeighbors(k=2000) 20k x 2k | IDENTICAL | REFUSED | 0.186 | REFUSED | 0.189 |
| NearestNeighbors(k=2000) 100k x 500 | IDENTICAL | REFUSED | 0.071 | REFUSED | 0.071 |

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

### Request 1790613586332 (m4pro-a, M4 Pro, f71bfda90): the k-NN host order pass

serial = `-D MOJOLEARN_KNN_SERIAL_ORDER` (one task), new = default; forward
and reverse; IDENTICAL digests equal in every row (FAST k = 2,000 taxi rows
vary run to run inside each arm, as before). KNN_PHASE_TIMERS sort_ms at
2,000 x 2,000, k = 2,000: 139 to 145 ms serial, 32 to 38 ms new (FAST: 55 to
60 -> 10). Raw: ~/mojolearn-evidence/neighbors-apple/1790613586332-speed-neighbors-f71bfda90b.txt

| case | mode | taxi serial | taxi new | HIGGS serial | HIGGS new |
|---|---|---|---|---|---|
| NearestNeighbors(k=2000).kneighbors 20k x 2k | IDENTICAL | 0.189 | 0.105 | 0.190 | 0.108 |
| NearestNeighbors(k=2000).kneighbors 20k x 2k | FAST | 0.265 | 0.092 | 0.266 | 0.093 |
| NearestNeighbors(k=2000).kneighbors 100k x 500 | IDENTICAL | 0.074 | 0.050 | 0.074 | 0.050 |
| NearestNeighbors(k=2000).kneighbors 100k x 500 | FAST | 0.084 | 0.037 | 0.083 | 0.037 |
| kneighbors / kNN classifier / regressor, k = 10, 20 (round-one board) | IDENTICAL | unchanged (0.05 to 0.13 s), digests = round one's | | | |
| SpectralEmbedding(knn).fit 20k | both | unchanged (0.77 IDENTICAL, 0.26 FAST) | | | |

### M3 ULTRA: lane base vs head, request 1790614983391 (m3ultra-b, 4a4415b41), one job

Same arms as the M4 Pro summary plus `-D MOJOLEARN_KNN_SERIAL_ORDER` in old.
Forward and reverse; every IDENTICAL digest equal old vs new; FAST equal
except the k = 2,000 k-NN / spectral rows whose FAST digests vary run to run
inside each arm (unpinned FAST ties, same code both arms). Raw:
~/mojolearn-evidence/neighbors-apple/1790614983391-speed-neighbors-4a4415b41c.txt

| algorithm | mode | taxi before | taxi after | HIGGS before | HIGGS after |
|---|---|---|---|---|---|
| LabelPropagation.fit 5k (knn) | IDENTICAL | 11.134 | 0.386 | 1.801 | 0.125 |
| LabelPropagation.fit 5k (knn) | FAST | 10.672 | 0.323 | 1.768 | 0.114 |
| LabelSpreading.fit 5k (knn) | IDENTICAL | 1.119 | 0.091 | 1.114 | 0.091 |
| Louvain.fit 1k | IDENTICAL | 2.746 | 0.092 | 4.351 | 0.100 |
| Louvain.fit 1k | FAST | 2.707 | 0.046 | 4.318 | 0.053 |
| OneClassSVM.fit 3k | IDENTICAL | 1.231 | 0.020 | 1.327 | 0.022 |
| OneClassSVM.fit 3k | FAST | 1.224 | 0.020 | 1.312 | 0.022 |
| LocalOutlierFactor.fit 20k | IDENTICAL | 1.010 | 0.060 | 1.005 | 0.049 |
| LocalOutlierFactor.fit 20k | FAST | 1.010 | 0.054 | 1.010 | 0.043 |
| LocalOutlierFactor.score_samples 5k | IDENTICAL | 0.265 | 0.057 | 0.258 | 0.043 |
| PolynomialCountSketch.transform 200k x 500 | IDENTICAL | 1.156 | 0.175 | 1.156 | 0.171 |
| PolynomialCountSketch.transform 200k x 500 | FAST | 1.163 | 0.162 | 1.168 | 0.159 |
| SkewedChi2Sampler.transform 1M x 500 | IDENTICAL | 1.149 | 0.380 | 1.160 | 0.389 |
| SkewedChi2Sampler.transform 1M x 500 | FAST | 1.138 | 0.373 | 1.149 | 0.381 |
| AdditiveChi2Sampler.transform 1M | IDENTICAL | 0.056 | 0.023 | 0.056 | 0.022 |
| PageRank.fit 5k | IDENTICAL | 0.580 | 0.056 | 0.576 | 0.055 |
| KNNImputer.transform 5k x 50k | IDENTICAL | 0.204 | 0.033 | 0.209 | 0.033 |
| KNNImputer.transform 5k x 50k | FAST | 0.171 | 0.028 | 0.174 | 0.030 |
| connected_components 5k | IDENTICAL | 0.207 | 0.065 | 0.098 | 0.034 |
| NearestCentroid.fit 200k | IDENTICAL | 0.198 | 0.084 | 0.194 | 0.075 |
| SVGP.fit 100k | IDENTICAL | | | 0.244 | 0.086 |
| KernelPCA.fit 500 (host Jacobi, untouched) | IDENTICAL | 2.371 | 2.373 | 1.609 | 1.613 |
| NearestNeighbors(k=2000) 20k x 2k | IDENTICAL | 0.180 (order pass serial; refused before ddc2f96ad) | 0.102 | 0.180 | 0.102 |
| NearestNeighbors(k=2000) 20k x 2k | FAST | 0.264 | 0.062 | 0.266 | 0.060 |
| NearestNeighbors(k=2000) 100k x 500 | IDENTICAL | 0.082 | 0.066 | 0.083 | 0.067 |
| SpectralEmbedding(knn).fit 20k | IDENTICAL | refused before ddc2f96ad | 0.315 | refused | 0.367 |

GPC.predict_proba 3k x 3k on the M3 Ultra (same job), a898eb9c2's 4-row
kernel vs `-D MOJOLEARN_CHOL_MR_ROWWISE` (after the merge the row-by-row
kernel runs at 256 threads too): IDENTICAL 0.210 / 0.208 vs 0.197 / 0.192;
FAST 0.174 / 0.174 vs 0.160 / 0.160 (taxi / HIGGS). On the M3 Ultra the
256-thread row-by-row kernel is 7% FASTER than the 4-row kernel; the M4 Pro
comparison at the merged head is request 1790618304592.

## FINAL (draft; updated when the M3 Ultra request lands)

### Default vs opt-in

Default ON, every one with an A/B arm that restores the old path:

| commit | default path | arm restoring the old path |
|---|---|---|
| ddc2f96ad | IDENTICAL k-NN k > 1024 in rounds + wide merge (was a refusal) | none (the old path refused) |
| eb1a0a257, d7934f99f | x_neighbors fused, tiled k-NN (`knn_sq_tiled`) | `MOJOLEARN_XN_UNFUSED_KNN=1`, `MOJOLEARN_XN_OLD_ITEMS=1` |
| 6f24c1d8b | OneClassSVM SMO over a threadgroup | `-D MOJOLEARN_XN_SERIAL_SMO` |
| df27366e9, 4c4e5978d, b7bec306c, a1bb804fb | resident label propagation / spreading (host stopping fold, sparse product), PageRank, connected components | `MOJOLEARN_XN_HOST_LOOPS=1`, `-D MOJOLEARN_XN_LP_DEVICE_FOLD`, `-D MOJOLEARN_XN_LP_DENSE` |
| b89ad2efa, c905e9e47, 0e1394f6a | KNNImputer: compact missing cells, staged fit rows, split donor scan | `MOJOLEARN_XN_UNCOMPACT_IMPUTE=1`, `MOJOLEARN_XN_OLD_ITEMS=1`, `-D MOJOLEARN_XN_IMPUTE_NO_SPLIT` |
| d0a0aeeae, b7bec306c, 4a8539a34 | PCS per-cell / per-row convolution, LabelSpreading degrees once, PageRank dangling on device, Louvain on the host | `MOJOLEARN_XN_OLD_ITEMS=1`, `-D MOJOLEARN_XN_PCS_CELL`, `-D MOJOLEARN_XN_LOUVAIN_GPU` |
| ff625ac0c | NearestCentroid / variance / SVGP / absdiff folds on the host | `-D MOJOLEARN_XN_SERIAL_GPU` |
| 9866b507f | x_neighbors large downloads staged over the host cores | `-D MOJOLEARN_XN_PLAIN_DOWN` |
| a898eb9c2 | Cholesky multi-RHS sweep, 4-row groups at 256 threads | `-D MOJOLEARN_CHOL_MR_ROWWISE` |
| f71bfda90 | k-NN host order pass over the host cores | `-D MOJOLEARN_KNN_SERIAL_ORDER` |

Nothing is left opt-in only; nothing is half done. Reverted: the device
barrier in the identical radix select (da21bc669: `air.wg.barrier` does not
legalize in the `_mojolearn` / metrics builds), so that kernel's
device-memory-across-`barrier()` hazard on Apple (the x_linear team one) is
OPEN, as it was before this lane.

### Unproven: needs the integration identity check (m2pro, NVIDIA, AMD, CPU)

Every change has one-Mac evidence (M4 Pro; M3 Ultra pending) with digests
equal old vs new in both modes, and host checks where stated, but NO identity
lane ran on this branch. The integration run must cover:

- ddc2f96ad IDENTICAL k-NN k > 1024 (rounds, wide merge): a NEW identical
  path; cross-vendor equality is by construction only (compares of the
  composite key). knn lanes, spectral lanes with knn affinity at k > 1024.
- eb1a0a257, d7934f99f x_neighbors fused / tiled k-NN (x-neighbors-lof*, label
  propagation lanes).
- 6f24c1d8b OneClassSVM threadgroup SMO (x-neighbors ocsvm lanes); uses
  `air.wg.barrier(3, 1)` (builds and runs on the M4 Pro in both modes; m2pro
  must confirm the 256-thread dispatch).
- df27366e9, 4c4e5978d, b7bec306c, a1bb804fb resident loops, host stopping
  fold, sparse product (label propagation / spreading, PageRank, cc lanes).
- b89ad2efa, c905e9e47, 0e1394f6a KNNImputer (x-neighbors knn-imputer lanes).
- d0a0aeeae, 4a8539a34 PCS, LabelSpreading laplacian, PageRank dangling,
  Louvain on host (their lanes).
- ff625ac0c host-run folds (NearestCentroid, SVGP, OneClassSVM gamma='scale').
- 9866b507f staged downloads (every x_neighbors op with a >= 16 MB output).
- a898eb9c2 Cholesky multi-RHS sweep (gp*, gpc*, kernel-ridge*, cholesky
  lanes; m2pro dispatch).
- f71bfda90 k-NN host order pass (every knn lane, every column).
