# IDENTICAL trees and classical ML speed roadmap (NVIDIA + AMD), 2026-10-04

Companion to `IDENTICAL_NEURAL_ROADMAP.md`. Same status and rules:
**a proposal from a read-only code review. Nothing here was compiled, run or measured.**
Every gain is inferred from code structure, or is an Apple M3 A/B quoted from a code comment
(marked "M3"). Every file:line refers to `integration/identical-all-20261004` as read on
2026-10-04; lines will drift.

Items already in the fam/fam2 reports, the toggle inventory, the candidate audit
(`~/mojolearn-evidence/candidate-audit-2026-10-04.md`) or the live `fix-*` lanes are not
repeated; "overlap" names the nearest known item. "Bits: same" means the per-cell fold or
order statistic is preserved; "changes" means every vendor and the host column move together
in one version.

## 0. Pattern across families

Many candidates are forms the code already contains and documents as same-bits, but compiles
only under FAST + Apple, or leaves as untracked opt-ins. These need no new arithmetic: they
need an IDENTICAL gate, a CUDA/HIP compile, the usual cross-vendor ID check and one measured
ON/OFF sample. They are marked **EXISTS**.

## 1. Trees

Checked, nothing new: the symmetric fit already drains once per tree with exact fixed-point
subtraction; the fold searcher subtracts too (`gbdt/methods/histograms_helper.mojo:17-19`).
DART, permutation SHAP, OOB and bootstrap seeds are covered by fam2 and the R1/G1 lanes.

`DW` = `gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo`,
`OB` = `gbdt/methods/ordered_boosting.mojo`.

| # | Item | Evidence | Bits | Notes |
|---|---|---|---|---|
| T1 | **EXISTS** Lossguide: exact batched best-first under IDENTICAL (`LG_EXACT_ID`, opt-in). The default takes two host waits per split leaf, about 2 x (max_leaves - 1) per tree. | DW:1162-1166, DW:1150-1161, DW:2910, DW:3079, DW:3642 | same | rounds drop to about log2(L) + L/32; the Apple FAST twin measured 78 to 17 s. Untracked: absent from inventory, audit and reports; no recorded A/B. Leaf capacity doubles |
| T2 | **EXISTS** Depthwise/Lossguide: estimator inherits the searcher's partition (`GBDT_NS_INHERIT_ID`). | DW:1216-1224 | same | removes a tree walk per row, a radix sort and a host wait per tree. One-permutation fits only. Untracked like T1 |
| T3 | Ordered: fold-order compressed index with the dither keyed on the document id. The note in the code puts `pw.hist` at 797 of 2,098 ms on the L40S, from four uncoalesced words per document per level. | `gbdt/methods/oblivious_tree_doc_parallel_structure_searcher.mojo:147-172, 727-745` | same with the doc-id key; changes (four columns plus `gbdt_oracle_ordered`) with the position key | the position-key variant dropped taxi AUC once in FAST; prefer the doc-id key. Overlap: Apple `ORD_ALL` (FAST only) |
| T4 | Ordered: one-step leaf estimation on the device for every task. `_estimate_prepare` never takes the `one_step_device` path. | OB:2563-2619, OB:1415-1425, OB:2536; `doc_parallel_boosting.mojo:1705, 1199-1216` | same if it reuses those statements | removes the settle drain, the walk drain and one upload per task per tree. Extends the G1 lane's one-step work |
| T5 | Ordered: score-noise std and scale kept on the device. | OB:2363, OB:2458, OB:2485, OB:2345 | same for the scale; the std needs `sf64_sqrt` on the device | with T4, an Ordered tree drains twice |
| T6 | Ordered apply: read the leaf already in permutation order instead of `bins[perm[i]]`. | `gbdt/methods/dynamic_boosting.mojo:86-108`; OB:665-676 | same | `d_leaf` covers `[0, need)`; must extend to the largest apply size |
| T7 | Ordered: per-leaf sums from the existing chunk x leaf matrix instead of sort and gather. | OB:679-730, OB:1324-1375, OB:741-790 | changes | design work for a pinned in-chunk order |
| T8 | RF/DT: sibling histogram subtraction when every column is sampled. No subtraction exists in `ensemble/decisiontree/batched_levelalgo/`. | `kernels/builder_kernels_impl.mojo:1141-1497` | same if the histograms are integer (per the fam reports; not confirmed in code) | about half the histogram row work per level; needs a memory-budget rule. Not with per-node column sampling |
| T9 | RF: one merged frontier across trees, or more trees in flight (ExtraTrees already merges all trees). | `ensemble/randomforest.mojo:3220-3232, 3454-3471`; `extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:4321, 4556` | same | workspace memory per tree. Partial overlap: `RF_DEVICE_LOOP` |
| T10 | **EXISTS** Non-symmetric per-group bit width (`2661_NONSYM_GROUP_WIDTH`). | DW:224 | same | wide data only. Rejected once on the H100 (Sep 21); no IDENTICAL L40S or AMD result |
| T11 | Depthwise: one wait per level under IDENTICAL without the row-index-only schedule. | DW:157, DW:2910, DW:3079 | none claimed; not checked for the stat-moving schedule | parked on the `GBDT_ID_RIDX` A/B |
| T12 | TreeSHAP: linear unwind per leaf for paths wider than 8; resident table and forest. | `xtrees/shap_device.mojo:285, 320-329, 274-299` | changes for the path form; same for the resident table | deep forests only; high design cost |
| T13 | IsolationForest: about 20 allocation drains per fit. | `isolation_forest/impl/isolation_forest.mojo:612-640, 893` | same | small |
| T14 | Forest predict: fold the finiteness scan into the predict launch. | `core/forest_inference.mojo:553-574, 660` | same | small |

## 2. Linear models

Already good: QN two-loop on the device with one sync per iteration; the CV path is one block
per (fold, l1_ratio) in one launch; SMO has a 1024 working set with a device radix sort;
isotonic is already a parallel sort, pointer-jump and merge. Naive Bayes was not traced.

`XD` = `x_linear/device.mojo`, `SG` = `x_linear/sgd.mojo`.

| # | Item | Evidence | Bits | Notes |
|---|---|---|---|---|
| L1 | SGD at batch 4096: one launch per batch (step, rows, then own partials through shared memory). | XD:1686-1726; SG:744 | same | launches per batch 2 to 1, X read once. Extends `MB_FUSE` |
| L2 | SGD: blocked dot and 32-row partials at every batch size (above batch 256 each row is one d-long chain and each partial a 256-long chain). | SG:765-771, SG:914 | changes | chains about 8x shorter. pass135 kept the long chains to protect old bits |
| L3 | One-vs-rest classes in parallel at batch 4096 (grid = classes x blocks). | XD:1192 | same | per-class early stop and witness restore. Extends `OVR_PAR` |
| L4 | LassoCV / ElasticNetCV Gram: one X pass for all folds. Per-fold tile sums centered on the global mean; each training Gram is the fixed-order sum of the other folds plus a mean-shift correction. | `x_linear/cd.mojo:461-482`; `x_linear/cd_grid.mojo:110-143, 670` | changes | about (folds + 1)x fewer X passes. Never raw moments; check on `mse_path_` |
| L5 | LARS: stop refactoring Cholesky (the same matrix is built and factored twice per non-drop step); then append one row. | `x_linear/lars.mojo:305-309, 322-326` | same for the reuse | |
| L6 | LARS: team argmax and incremental correlations (the whole path is one block). | `lars.mojo:272-276, 280-285`; XD:4951-4955 | changes for the update; argmax same | |
| L7 | Bayesian ridge: several iterations per sync (**EXISTS** as `C1_BATCH`, FAST + Apple). | XD:5108-5140; `x_linear/cls1_fast.mojo:58` | same | |
| L8 | Bayesian ridge: Gram sse with the reference-row guard (**EXISTS**, FAST + Apple). | XD:4975-4986 | changes | fp32 cancellation in the sse; owner's call |
| L9 | Poisson/Gamma/Tweedie Newton: one guarded unit per iteration (4+ waits become 1). | XD:3228-3273 | same | |
| L10 | GLM Newton solve in one block (today about 3m tiny launches per iteration). | XD:3073-3078 | same if the element order is kept | |
| L11 | GLM line search: price t = 1, 1/2, 1/4, 1/8 in one pass, each from its own trial theta. | XD:3283-3287, XD:3112-3132 | same | helps only when backtracking happens |
| L12 | SMO: 2048 working set in IDENTICAL (**EXISTS**, FAST + Apple). | `svm/impl/workingset.mojo:96-100` | changes | gate fixtures assume 1024 |
| L13 | Isotonic: 8-bit radix digits. | XD:3807, XD:4421-4436 | same | shared-memory page size |
| L14 | SGD block-local averaging: B blocks each run K sequential small batches on fixed shards, weights averaged in a fixed tree. The only route to a whole epoch in few launches. | - | changes; a different algorithm | owner decision. No SGD quality gate was found under `x_linear/checks`; a gate must exist first |

Notes: the stored sgd-reg time (165,682 ms on the L40S) implies either many epochs or about
145 us per launch; compare the epoch count with the opponent's before ranking L1-L3 against
L14. Solver CD row sweeps beyond the Gram cap are still 2-3 launches per coordinate plus a
sync per sweep (`solver/impl/cd.mojo:1722-1772`).

## 3. Dense linear algebra, kernels, GP, GMM

Already answered in code: LU is blocked at 32 columns with a tree pivot; the SVD Jacobi
already runs on the small R^T with one launch per round; a round-robin eigh round rotates
every row and column, so a pair-only update cannot apply. Block Jacobi eigh stays quarantined.

`DD` = `x_decomp/device.mojo`.

| # | Item | Evidence | Bits | Notes |
|---|---|---|---|---|
| X1 | Move kit `orth`, kit `svd` and `qr_r` onto TSQR; they still call the column-at-a-time `qr_factor`. | DD:2047, DD:3032, DD:3465, DD:2039-2057 | changes (via `tsqr_host.mojo`) | users: randomized SVD/PCA range finders, FactorAnalysis, LLE. `TS_MAX_N = 512` cap |
| X2 | LU trailing update in 256-column strips, left-looking (the Cholesky strip schedule). | DD:2160-2167; `cholesky/checks/potrf_strip.mojo:21-28` | same (inferred: each cell keeps its ascending chain) | about 8x less trailing traffic at n = 8192. Swaps must reach deferred columns in order |
| X3 | TSQR combine tree at higher arity using the leaf panel kernel (16 R tiles fit a leaf: about 8 levels become 2). | `x_decomp/tsqr_device.mojo:375-430, 655-668` | changes | Householder throughout |
| X4 | Skip structural zeros when the right-hand side is the identity (GP optimizer `cho_solve`, GMM `trsm_lower`). | `gaussian_process/gp_optim.mojo:460`; `mixture/checks/mstep.mojo:1720`; `cholesky/checks/trsm.mojo:326-390` | same (skipped steps are `fma(-l, +0, +0)`; inferred) | forward solve n^3/2 to n^3/6 per GP evaluation |
| X5 | **EXISTS** SVGP blocked float-float Cholesky in IDENTICAL. | `x_neighbors/iter_device.mojo:1613-1632` | same if contraction is pinned | 1,024 dependent launches become 64 at m = 512 |
| X6 | **EXISTS** SVGP tiled RBF in IDENTICAL. | `iter_device.mojo:1373-1385` | same (inferred) | |
| X7 | Single right-hand-side triangular solve across blocks. | `cholesky/checks/trsm.mojo:245, 806-811` | same (inferred) | AMD wave mapping. Helps KernelRidge and GP alpha |
| X8 | **EXISTS** Tridiagonal eigh for IDENTICAL at large n. | `x_decomp/eigh_tridiag.mojo:3-43, 67`; DD:2447-2452 | changes (all columns plus a new host replay) | refuses clustered spectra and falls back to Jacobi. Listed "worth doing" before; the code now exists |
| X9 | eigh: one launch per round with ping-pong buffers. | DD:2550-2558 | same (inferred) | halves launches at mid n; pointless at n = 4096 |
| X10 | **EXISTS** Kernel PCA Lanczos on the device. | `x_decomp/lanczos_dev.mojo:3-24` | changes | the Ritz residual test stays the gate |
| X11 | **EXISTS** GMM fused Cholesky, inverse and log-determinant for all components; fused covariance. | `mixture/checks/mstep.mojo:895-904, 1602-1620, 1634-1650, 723-731` | changes unless the kernel restates potrf's chains at d <= 32 | |
| X12 | **EXISTS** Same-bits glue routes still Apple-only: `IPCA_FAST_DEV`, RBFSampler one-call fit_transform. | `lanczos_dev.mojo:132-146`; `kernel_methods/rbf_resident.mojo:3-30` | same | |
| X13 | SVGP row-slice split of the float-float B and b chains. | `iter_device.mojo:1763-1773` | changes | about 1e-14 relative re-association |
| X14 | LU tournament pivoting for the panel. | DD:581-587 | changes (pivots) | weaker growth bound than partial pivoting; owner decision |

## 4. Clustering, neighbors, prep, resample, time series

Read, nothing new: the k-means Lloyd loop and init, the UMAP optimizer, silhouette,
Holt-Winters. Not re-read beyond the reports: DBSCAN, HDBSCAN, IVF, ARIMA. Not read: KDE,
trustworthiness.

| # | Item | Evidence | Bits | Notes |
|---|---|---|---|---|
| K1 | **EXISTS** Radix sort for `sort_cols` under IDENTICAL (bitonic today, about 41 passes at 1M rows against 4 counting passes). | `x_prep/dradix.mojo:38`; `x_prep/device.mojo:514-518`; `x_prep/dsort.mojo:191` | same | feeds RobustScaler, QuantileTransformer, KBins, SimpleImputer median, encoders, SplineTransformer. First CUDA/HIP compile |
| K2 | **EXISTS** Radix select for median/quantiles instead of a full sort (`QSELECT`). | `x_prep/fastprep2.mojo:36-43` | same | depends on K1's keys |
| K3 | **EXISTS** Theta: `THETA_SNAP` and `THETA_SPEC` under IDENTICAL (Nelder-Mead candidates side by side in one warp). | `sequence/theta.mojo:38, 62`; `sequence/theta_spec.mojo:3-43`; `sequence/exec_device.mojo:758` | same, if `THETA_REG`'s same-order claim holds without `SEQ_FAST_FMA` | M3: 218 to 20 ms. `shuffle_idx` 32-lane groups on AMD's 64-lane wave |
| K4 | ETS: stop fitting each series on one thread. (a) speculative candidates applied to `op_ets`; (b) **EXISTS** block-per-series affine-prefix likelihood with a host twin. | `sequence/ets_team.mojo:3-36, 50`; `exec_device.mojo:218` | (a) same, (b) changes | M3 (b): 612 to 48 ms |
| K5 | **EXISTS** STL as one thread per output point (`TSA2_STL`). | `sequence/stl_grid.mojo:3-36`; `sequence/ops.mojo:60-76` | changes (window sums) | M3: 359 to 7 ms. Jump 1 only |
| K6 | MeanShift on a (seed, row-chunk) grid with fixed-point sums (the k-means integer plan makes the fold order-free). | `x_cluster/bodies.mojo:117-150`; `x_cluster/device_ops.mojo:169, 439`; `cluster/impl/sum_scale_plan.mojo` | changes | stored AMD ratio 30.7x. Scale-plan overflow bounds |
| K7 | CAGRA IDENTICAL build from an IVF candidate graph instead of the exact all-pairs graph. | `x_ann/fast_env.mojo:36-74`; `x_ann/knn_device.mojo:276-311` | changes (the graph) | recall risk; needs a recall gate and the owner's decision |
| K8 | Certified candidates plus pinned rescoring for brute-force kNN on NVIDIA/AMD (**EXISTS**, Apple only). | `neighbors/impl/detail/certified_mma_knn.mojo:3-40`; `knn_brute_force.mojo:206-212` | same (uncertified queries fall back) | certificate failure rate on duplicate-heavy data |
| K9 | **EXISTS** AMD low-d kNN: time `EXPERIMENTAL_KNN_FUSED_SELECT` across several d below 32 (today AMD writes the distance matrix). | `checks/kernel_matrix.mojo:1665-1679, 1594-1602` | same | the stored 53.8x row. A range rule only |
| K10 | Agglomerative (ward/complete/average): merge reciprocal nearest pairs in rounds (**EXISTS** as `WARD_ROUNDS`, FAST opt-in, "unproven"). | `x_cluster/device_ops.mojo:2511-2524`; `x_cluster/agglo.mojo:153-160` | may change on exact ties | reducible linkages only |
| K11 | Segmented radix sort: 8-bit digits instead of 32 one-bit passes. | `core/segmented_sort.mojo:306, 353-420` | same | bootstrap quantiles and forest bin tables |
| K12 | Bootstrap median/quantile without materializing replicates. | `resample/estimator.mojo:956-996` | same (inferred) | |
| K13 | Partial-distance pruning in the x_ann kNN graph. | `x_ann/knn_device.mojo:175-198` | same | Overlap: `IDN_XN_KNN_PRUNE` (x_neighbors only) |
| K14 | x_prep one-thread units with same-word parallel forms (`ii_conv` max tree, 32-thread eigh block). | `x_prep/fastprep2.mojo:11-30` | same for those two | |

## 5. Proposed order

1. The owed measured batch first (see the neural roadmap, section 5).
2. The **EXISTS** same-bits set, because it needs gates and a compile, not new arithmetic:
   T1, T2, K1, K2, K3, K9, X5, X6, X12, L7. Register each in the toggle inventory and recipes.
3. Same-bits new code with the widest reach: X2, X4, L1, L3, T4-T6, K11.
4. Ordered boosting (the worst tree row, 2.03x): T3 with the doc-id key, then T4-T6.
5. The bits-changing set as one version step with the neural set: X1, X3, X8, L2, L4, K4(b),
   K5, K6, T7.
6. Owner decisions before any work: L14 (SGD block-local averaging), K7 (CAGRA candidate
   graph), X14 (tournament pivoting), L8 (Gram sse).

## 6. Limits

- Read-only. No build, no run, no timing. M3 figures are Apple FAST A/Bs from code comments
  and do not transfer to NVIDIA or AMD.
- Stored ratios predate the integration wave.
- "Same bits" labels marked "inferred" rest on reading the fold, not on a digest.
- Not read: Naive Bayes, KDE, trustworthiness; DBSCAN, HDBSCAN, IVF and ARIMA only through
  the reports.
