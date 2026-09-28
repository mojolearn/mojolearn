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
