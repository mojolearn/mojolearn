# Flagship gaps plan (2026-10-09): the algorithms where the boards still show 5-100x against the GPU opponents

Read-only analyses of main (42d1e42c6) by six analysts; their per-family code reads with file:line evidence are in
`docs/plans/flagship-gaps-20261009/read_*.md`. Nothing here was measured by the analysts; every number is either a board
row (release 0.8.25 boards, Sept 29 to Oct 1), an Oct 8 re-race of main (`~/mojolearn-evidence/p1-grid-flips/*.txt`,
switch grid ge123e6f9) or a cost model from the code.

## 0. Which "gaps" are real on main today

| lane | 0.8.25 board row | main today (Oct 8 re-race) | verdict |
|---|---|---|---|
| gbdt-ordered taxi / istella | 244 s (12x CatBoost) / 143 s (3.8x) | 21.2 s NV, 22.0 AMD (1.04x) / 36.6 s (0.96x), AUC equal | gone; only the ordered-over-plain overhead remains |
| ivf taxi / istella (AMD) | 20.4 s / 265 s | 232 ms / 1,556 ms | gone to ~1-5x; the PQ codebook stage is the AMD pathology |
| ols taxi (AMD) | 171 ms | 8.0 ms | gone on taxi; istella still open |
| ridge taxi / istella (AMD) | 54.8 / 1,160 ms | 3.1 / 1,076 ms | taxi gone; istella open |
| kmeans taxi / istella (AMD) | 396 / 736 ms | 210 / 413 ms | improved |
| dbscan taxi, hdbscan both | FAILED | run on main (stale refusals of the old wheel) | race them; dbscan istella 542 s is real |
| pca, tsvd, randomized-svd, knn (AMD), gaussian-nb, tsne, logreg, lasso, elasticnet, sgd-clf, kernel-ridge | see read files | no re-race yet | real until a race says otherwise |

A rolling main board (lane main-board) will replace this table with measured cells.

## 1. Dominant causes found (one per family, details in the read files)

- **PCA family** (`read_pca.md`): the Jacobi eigensolver launches 441 kernels per sweep with a host flag sync (2,600-4,400
  launches on 220 columns); a dead "restore shift" pass re-reads and re-writes X; tsvd uploads X twice; randomized-svd copies X
  twice on the host (`tobytes` + `frombytes`) and orthonormalizes on 64 blocks x 32 threads. Pageable upload is ~94 of 231 ms on
  pca istella (1.80 GB) and all of the taxi/tsvd cells.
- **kNN on AMD** (`read_knn_nb.md`): every AMD difference is a `checks/kernel_matrix.mojo` row choosing the NVIDIA schedule only
  on NVIDIA: query tile 4096 vs 256, resident derived cache off (352 MB layout, norms and admission recomputed every call),
  radix scratch reallocated per call, generic-k selector, software FTZ, 4 register rows. 112 tile iterations vs 7. Expected
  AMD time ~15-25 ms; 1,180 ms is a pathology (register-list spill at k=64 or 620 MB of hipMalloc per call).
- **Gaussian NB** (`read_knn_nb.md`): kernels are ~5 ms; the fit is a pageable host-to-device copy at ~0.8 GB/s plus two
  ~880 MB device allocations and a D2D hop per fit, plus ~300 ms size-independent glue (host label sort, allocations).
- **IVF-PQ on AMD** (`read_ivf.md`): the PQ codebook stage runs 11-55 serial k-means fits (n=400k, k=256, d=1|4) whose
  accumulator has ~1,563 threads with 256-step dependent RMW chains; 500-1,300x slower on CDNA3 than on the L40S. The batched
  device codebook loop already exists as an Apple FAST path. IVF-flat: k-means|| seeding with ~14 syncs, a 352 MB permutation
  by scatter, fused-distance grid sized from a static core count (108/110) that under-fills both GPUs.
- **t-SNE** (`read_tsne_dbscan.md`): exact N^2 repulsion, 2,000 launches, ~5x off peak; the kNN top-91 selection is a serial
  insertion in global memory (the whole istella-taxi difference). cuML is FIT-SNE O(n) per iteration.
- **DBSCAN** (`read_tsne_dbscan.md`): one warp per query, one candidate per lane, full 220-term distance from global memory
  with no reuse, run twice; RBC prunes nothing at eps=3 in 220 standardized dims; 134 batches x ~6 syncs.
- **Ordered GBDT** (`read_gbdt_ordered.md`): structure already matches CatBoost; the fold arm costs 35 ms/tree over the plain
  tree against CatBoost's 25: latency-bound histogram grid over tiny folds, ~540 estimation launches per tree, a learn
  permutation that is estimated but never searched on.
- **Linear models**: `read_linear.md` (pending at the time of writing; its lane launches when it lands).

## 2. Lanes (Opus 5.5, code only: no compile, no verify, no measure; one integration worktree compiles once, then main measures)

| lane | ideas (define names in the read files) | bits |
|---|---|---|
| fg-pca | P1 one-block Jacobi sweep (+P1b fused cs/update), P2 drop the restore pass and lean scratch, P3 centered TN tile load, P5 device truncate; T1 tsvd one upload, T2 fused column variance; R1 rsvd direct upload (no host copies), R4 staged GEMM | none except R4-plans (not in this round) |
| fg-knn-nb | K1 AMD takes the NVIDIA schedule rows (behind one define for the A/B), K2 wide-k selector for 64-lane columns, K3 block top-k to k=64 on AMD, K4 scratch pool on the resident handle, K5 exact chain for the AMD register tile; G1 pinned staged upload ring in dev_put, G2 arena/slot pool across calls, G3 stages read X from the slot (no D2D hop), G5 device label codes | none |
| fg-ivf | A1 device PQ codebooks promoted to IDENTICAL with a host twin (deletes the anomalous stage), A3 one upload + resident coarse, A4 PQ_LUT_TILED as the identical default, A5 fused score+select, A7 resident handle from fit; B1 strided initial centroids, B2 cap the k-means-parallel recluster, B3 layout as histogram+scan+scatter, B5 grid from the live core count | A1, B1 change bits (host column follows); rest none |
| fg-tsne-dbscan | T1 shared-memory top-k, T2 register-tiled repulsion; D1 register-tiled eps count/fill (same c-ascending chain), D2 adjacency bitmap, D4 sampled-degree batching, D5 hook-and-jump CC; H2 MST seeded from the kNN list, H3-H5 sync plumbing | none (D1 keeps the predicate and order) |
| fg-gbdt-ordered | A ORD_HIST_FOLD_SKIP, C IDN_ORD_CAT_PLANES, F ORD_SKIP_UNSEARCHED_PERM, E IDN_ORD_INDEX_SHARED | none |
| fg-linear | from read_linear.md | tbd |

Rules for every lane: each idea is one `-D MOJOLEARN_<AREA>_<NAME>` switch; a pure waste removal (dead pass, double upload,
double host copy, per-call reallocation) becomes the default with a `_OFF` define so the grid can measure the old path; anything
that changes an algorithm or bits is default off. Cost reasoning in a comment, never a board dimension. Host column changes with
every bit change. GPU-only parallel paths; no Python compute. Every switch gets a grid_controls entry (schema
mojolearn.grid-controls/1) so the switch grid measures it with the rest.

## 3. Measurement after the merge (orchestrator)

`lq add --front nv|amd RACE main <lanes> <datasets> ARMS=ours BUILDS=<all device builds>` for the default arms and one line per
B arm via `MOJOLEARN_BUILD_DEFINES`, plus first-look probes the analysts asked for: `MOJOLEARN_KNN_PHASE_TIMERS` on AMD knn,
`MOJOLEARN_ANN_STAGES=1 MOJOLEARN_KMEANS_STAGES=1` on AMD ivf-pq taxi, `MOJOLEARN_XPREP_PROFILE=1` on gaussian-nb, and
`hdbscan istella,taxi` on both vendors (no number exists). Identity = equal digests on NVIDIA and AMD per cell. Results feed the
main board through `tools/main_board_ingest.py` when it lands.
