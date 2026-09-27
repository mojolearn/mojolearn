# decomp: progress

Design (pass 1): every float operation is a cell in `x_decomp/cells.mojo`, run
by `x_decomp/device.mojo` (GPU binding `_mojolearn_x_decomp`, one thread per
output) or `x_decomp/host.mojo` (host binding, same cell in a loop); the entry
points are written once in `x_decomp/api.mojo`. Python
(`python/mojolearn/_expansion_decomp.py`, `_Kit`) holds control flow and data
movement only. eigh reuses `decomposition/` (device_eigh / host_eigh).

| algorithm | lane | commit | pod check |
|---|---|---|---|
| IncrementalPCA | x-decomp-ipca | 9135f7adb | AGREE: compared batch 9, infer 9, train 9 (A40 cuda vs x86 CPU) |
| GaussianRandomProjection / SparseRandomProjection | x-decomp-grp, x-decomp-srp | (see git log: "decomp lane: GaussianRandomProjection / SparseRandomProjection") | AGREE batch 9, infer 9, train 9 each |
| SpectralEmbedding | -- | already public | the existing _spectral_impl.py class (cuML route); its missing affinity='rbf' is added in the next row |
| NMF | x-decomp-nmf | (see git log: "decomp lane: NMF") | AGREE infer 9, train 9 (NMF has no batch part: transform's stopping test is over the whole batch); ipca/grp/srp re-AGREE after the new cells |
| FastICA | x-decomp-fastica | (see git log: "decomp lane: FastICA") | AGREE batch 9, infer 9, train 9 |
| FactorAnalysis | x-decomp-factor-analysis | (see git log: "decomp lane: FactorAnalysis") | AGREE batch 9, infer 9, train 9. Sanity: fixed-iteration trajectories match sklearn (psi within 4e-5); at tol=1e-2 the stopping test sits under the float32 resolution of the log-likelihood (noise ~0.03 at n=1000) so the stop iteration can differ from sklearn's float64 run |
| SpectralEmbedding affinity='rbf' | x-decomp-spectral-rbf | (see git log: "decomp lane: SpectralEmbedding affinity='rbf'") | AGREE train 9; --sabotage (host sqdist features descending) PASS: AGREE, DISAGREE 9, AGREE; verify spectral-embedding/spectral/spectral-precomputed VERIFIED before and after (67 parts, 0 divergent) |
| LU solve (lu_factor, lu_solve, solve) | x-decomp-lu | (see git log: "decomp lane: LU solve (lu_factor, lu_solve, solve)") | AGREE train 9; pivots equal scipy's including exact ties |
| lstsq / randomized_svd | x-decomp-lstsq-rsvd | (see git log: "decomp lane: lstsq / randomized_svd") | AGREE train 9; all 9 decomp lanes re-AGREE after the new MGS2 orth cell |
| PLSRegression / PLSCanonical / CCA | x-decomp-pls | (see git log: "decomp lane: PLSRegression / PLSCanonical / CCA") | AGREE batch 9, infer 9, train 9; sklearn match 2e-6 (CCA mode B needs the new QR + one-sided Jacobi svd cell; _thin_svd and FactorAnalysis moved to it, FA now stops at 60 vs sklearn 62 iterations); fa/lstsq-rsvd/nmf/ipca/fastica re-AGREE |
| SparsePCA / MiniBatchSparsePCA / DictionaryLearning (+ MiniBatchDictionaryLearning, sparse_encode) | x-decomp-sparse-pca, x-decomp-dict-learning | (see git log: "decomp lane: LatentDirichletAllocation ...", same commit) | AGREE batch 9, infer 9, train 9 each; sklearn match 1e-6 (DictionaryLearning, SparsePCA, MiniBatch* with shuffle=False) |
| LatentDirichletAllocation | x-decomp-lda | (same commit) | AGREE batch 9, infer 9, train 9; given one model, transform and score equal sklearn's to 1e-7; fit reaches sklearn's optimum on 3 of 6 seeds in 20 iterations (the Philox Gamma draws start it elsewhere) |
| LatentDirichletAllocation, Isomap / MDS / ClassicalMDS / LocallyLinearEmbedding, SparsePCA / MiniBatchSparsePCA / DictionaryLearning | x-decomp-manifold (Isomap, MDS, ClassicalMDS, LLE) | (see git log: "decomp lane: LatentDirichletAllocation, Isomap / MDS / ClassicalMDS / LocallyLinearEmbedding, SparsePCA / MiniBatchSparsePCA / DictionaryLearning") | AGREE batch 9, infer 9, train 9; sklearn match 1e-6 (Isomap, ClassicalMDS, MDS), 1e-4 (LLE through the SVD of I - W); test_lane_select OK |
| EllipticEnvelope / MinCovDet | x-decomp-robust-cov | (same commit as ALS) | AGREE batch 9, infer 9, train 9; sklearn match 4e-7 at n <= 500 (the n > 500 path's random subsets are Philox, so it can land elsewhere) |
| ALS (AlternatingLeastSquares) and EllipticEnvelope / MinCovDet | x-decomp-als | (see git log: "decomp lane: ALS (AlternatingLeastSquares) and EllipticEnvelope / MinCovDet") | AGREE train 9; matches a numpy restatement of implicit's exact least_squares to 7e-6 from the same start; test_lane_select's two structural tests PASS |

Next: PASS 2: AMD box (requested, RunPod out of stock, retrying), per-seam checks, option parity, speed

## Pass 2 (started 2026-09-27)

- Per-seam proof: `x_decomp/checks/{fold_ew,dense,rows,graph}_check.mojo` hold
  the device (DevExec) and CPU (HostExec) columns to independent host oracles
  (`x_decomp/checks/xd_oracles.mojo`) bit for bit, each fixture first shown to
  separate the pinned spelling; card stages through IdentityTrace.
  `tools/identity_lanes/decomp.checks` pairs each of the 17 seams (DEVIATIONS
  5300-5316, annotated in `x_decomp/cells.mojo`) with a sabotage patch under
  `x_decomp/checks/sabotage/`. IDENTITY_PATHS rows 130-139. End-to-end
  sabotage for the steward: `x_decomp/checks/sabotage/e2e_host_sqdist.patch`.
- NVIDIA A40 + x86 CPU, `tools/algos_lane_check.sh <all 17 x-decomp lanes> --pass 2`:
  RESULT PASS: every seam driver PASS, FAIL under its patch, PASS after reversal
  (17 of 17 arms bite), and every lane AGREE.
- Options added (each AGREE, sklearn sanity 1e-6): NMF shuffle and beta_loss
  KL/IS; PCA svd_solver='randomized' and float n_components, TruncatedSVD
  algorithm='randomized' (lane x-decomp-pca-randomized); PLSCanonical
  algorithm='svd'; SparseCoder; LocallyLinearEmbedding method='ltsa';
  MinCovDet one feature. Remaining: `x_decomp/NOT_IMPLEMENTED.tsv`.
- Host manifest: test_host_surface.py reads parametrized def_function
  registrations; MOJOLEARN_HOST_SABOTAGE moves HostExec.gemm; 196 passed.
- AMD: no box (RunPod MI300X and Hot Aisle both out of stock all session);
  AMD identity goes through the do-amd steward (`apple_steward.py submit`).
- Directive 000 (DONE, never repeat): the 18 seam arms (5300-5317) re-run on
  the fixed lane check (box `tools/algos_lane_check.py` md5 = origin/main's,
  A40, /root/pass2b.log 18:15-18:43Z, tree with the speed commits): every arm
  BUILDS, RUNS and FAILS under its patch, PASSES after reversal; no BROKEN arm.
- Stewards: request 1790531301235 at 16ae279cc FAILED on m2pro and do-amd only
  because the end-to-end patch was stale.
- Session 3 (2026-09-27), NVIDIA A40 + x86 CPU, after merging origin/main:
  - 27 existing lanes the selector names (arima*, bootstrap, cholesky,
    embedding*, holtwinters*, ivf*, kpss, monte-carlo, pca*, permutation-test,
    select-d, sequence-autoarima, spectral*, tsvd, umap): all AGREE
    (/root/s3o.log). They reach x_decomp only through the package import
    (spectral's affinity='rbf' is x-decomp-spectral-rbf's own path); par-*
    lanes need two GPUs and were not run.
  - The speed commits' two-pass A R^-1 orth was NOT orthonormal on a
    rank-deficient input (|Q^T Q - I| up to 20) and randomized_svd then
    REFUSED on the denormal fixture (the one-sided Jacobi SVD of Q^T M rotated
    a noise column forever). Fixed at the root: `orth_rank_guard` (DEVIATION
    5318, IDENTITY_PATHS row 134): R[j, j]^2 <= 2^-32 of the column's sum of
    squares zeroes R[j, j], so that Q column is 0; `_rsvd_core` drops zero
    range columns before the small SVD and returns 0 components past the
    numerical rank. Oracle (`oracle_orth` alt 2), separating fixture and arm
    5318 in dense_check. randomized_svd on all 9 fixtures x both lane calls:
    converges, top singular values within 6e-3 (n_iter 4, flat spectrum) and
    2e-6 elsewhere of the float64 SVD.
  - `tools/algos_lane_check.sh <17 x-decomp lanes> --pass 2 --sabotage
    x_decomp/checks/sabotage/e2e_host_ulp.patch` (host gemm and LU by one ulp),
    s7: all 19 seam arms (5300-5318) BUILD, RUN, FAIL under their patch, PASS
    after reversal; 17 lanes AGREE clean; 16 DISAGREE under the patch (every
    lane but x-decomp-spectral-rbf, which calls neither gemm nor LU on the
    host), 17 AGREE restored. x-decomp-spectral-rbf's end-to-end arm is
    `e2e_host_sqdist.patch` (s8).
  - The first ulp patch also flipped every host `ew` result: Isomap's host
    eigh then refused on 8 of 9 fixtures (60 sweeps did not help). A one-ulp
    perturbation of the INPUT converges everywhere (Isomap, ClassicalMDS, LLE,
    LTSA on both columns), so that was the patch's artifact; ew is not in it.
  - test_lane_select: OK (0 failures) at ac05911b5, whose Mojo imports are
    this branch's. test_host_surface: 196 passed after merging origin/main
    (3af84e8a0) into the branch.
  - Stewards: 1790542293471-decomp-3c73fcee93 (16 lanes, e2e_host_ulp) and
    the spectral-rbf request (e2e_host_sqdist) queued on m2pro and do-amd.
    PHASE 1 IS DONE when both PASS on both (`python3 tools/apple_steward.py
    status | grep decomp`). A FAIL: fix, re-prove on the pod, resubmit only the
    affected lanes.
  - NOT MERGED TO MAIN YET: directive 0a makes a merge wait for m2pro PASS and
    do-amd PASS, and the steward queue was ~18 requests deep. lane/algos-decomp
    is pushed and holds everything (the speed commits and the orth fix; main
    never had the orth defect). On both PASS: `git fetch origin && git merge
    origin/main && git push origin HEAD:main`, test_host_surface first.

## IDENTICAL GPU speed, NVIDIA only so far (phase 4 work done early; NVIDIA A40, higgs 1M x 28 from R2)

| algorithm | before | after | what |
|---|---|---|---|
| PCA randomized / randomized_svd | 108 s (200k rows) | 1.9 s (1M) | orth: one-thread MGS2 -> two passes of the Householder R + a row-parallel solve (DEV 5309) |
| FactorAnalysis (20 it) | 7.8 s | 0.86 s | one QR of Xc, then the d x d SVD of R D / sqrt(n) per iteration |
| lstsq | 3.6 s | 0.9 s | sign flips through absmax_sign_cell (DEV 5317); strided data movement |
| NMF mu (20 it) | 5.6 s | 2.2 s | finite/negative scans through the cells; strided take_cols/hstack |
| FastICA (20 it) | 2.8 s | 1.3 s | row sums past 4096 two-stage (DEV 5301) |
| gemm / colsum / rowsum | one thread per output over 1M terms | FOLD_BLOCK = 4096 two-stage folds (DEV 5300/5301) | |

Still owed in phase 4: AMD and Apple timings (`apple_steward.py submit --kind
speed`), device-resident matrices (every kit call still uploads and downloads;
`ew` over 1M x 28 is ~12 ms of copies), the Jacobi eigh's fixed cost (~60 ms a
call).

## PHASE 1: MERGED to main 025c7a921 (2026-09-27, directive 0000b: merge on
NVIDIA + CPU; the Apple / AMD steward verdicts of 1790542293471 and
1790542727482 are post-merge release gates, a FAIL comes back as a fix).

## PHASE 2 (option parity), session 4 (2026-09-27)

Done on lane/algos-decomp (each: sklearn/scipy sanity on the A40, lane AGREE
CUDA == CPU, a sabotage that bites, old part hashes unchanged):

| option | route | proof |
|---|---|---|
| Isomap radius (n_neighbors=None), path_method 'FW' (DIVERGENT: Dijkstra) | closed radius on the float32 distance; transform min over in-radius rows | sklearn 8e-6 |
| Isomap / MDS / ClassicalMDS metric: manhattan, chebyshev, minkowski p, cosine | NEW cell `pdist_cell` (DEVIATION 5319, oracle `oracle_pdist`, fixture separates, arm 5319_pdist_order bites) through x_decomp_sqdist's optional kind/p | pairwise_distances 2e-7; Isomap 2e-5 |
| MDS metric_mds=False (Kruskal non-metric) | linear lane's IsotonicRegression (out_of_bounds='clip') inside `_single` | sklearn 5e-6, same stress/n_iter |
| LocallyLinearEmbedding 'hessian', 'modified' | stacked factor B (M = B^T B), SVD null space; complement / null bases DIVERGENT (tsv) | sklearn 2e-5 |
| sparse_encode / SparseCoder / DictionaryLearning transform 'lars' | linear lane's Lars(fit_intercept=False) per row | sklearn 2e-6 |
| lu_solve trans=1/2 | getrs 'T' in lu_solve_serial; oracle_lu_solve_t; 5308 arm extended | scipy 2e-7 |
| EllipticEnvelope.score | accuracy (weighted), IEEE double | exact |
| AlternatingLeastSquares calculate_training_loss | `training_loss_` per iteration through the cells | numpy restatement 1e-8 |
| IncrementalPCA / LDA / ALS / every x_decomp input: scipy.sparse | densified exactly (IPCA per batch) | bitwise == dense |
| PCA svd_solver='arpack', TruncatedSVD algorithm='arpack' (DIVERGENT: exact arms), tol, copy | 'full' arm / Gram arm, n_components < min(shape) | sklearn 8e-7 |
| PCA n_components='mle' | `_pca_mle_rank` (logs, gammaln in the cells) | sklearn rank equal, 4 seeds |
| TruncatedSVD explained_variance_ / _ratio_ (Gram / arpack arms) | `_tsvd_explained` through the cells (IDENTICAL cells in FAST too) | sklearn 9e-7 |
| PCA / TruncatedSVD scipy.sparse input | densified exactly | |
| SpectralEmbedding affinity='precomputed_nearest_neighbors'; eigen_solver arpack/lobpcg/amg (DIVERGENT: Lanczos); eigen_tol='auto', n_jobs, verbose accepted | Python data movement, then the precomputed route | sklearn affinity exact, embedding subspace cos 1 - 1e-11 |
| linalg.eigh UPLO | chosen triangle mirrored (strided copies) | |
| directive (x_* context): x_decomp keeps ONE process-lifetime DeviceContext (`xd_ctx`); decomposition's device_qr_r / device_eigh take a caller's context | python/mojolearn/tests/test_x_decomp_repeat.py (every entry twice, GPU and host, bytes equal) | |

Pod runs (A40 + x86 CPU, /root/mojolearn-evidence/lane-check/p2d-*):
- p2d-sab: x-decomp-manifold, -dict-learning, -als, -robust-cov,
  -pca-randomized, -spectral-rbf `--pass 2 --sabotage
  x_decomp/checks/sabotage/e2e_p2_options.patch` (the new option paths moved on
  the CPU column only): RESULT PASS (AGREE, DISAGREE, AGREE); all 20 seam arms
  (5300-5319) build, run, bite. Old part hashes vs s7-x: 918 compared, 0 moved.

## NEXT: PHASE 2 REMAINDER (start here; the rows above are done, never re-run)

Each item gets the gate: lane AGREE on the pod (`tools/algos_lane_check.sh`,
only the lanes `lane_select.py --changed-since origin/main` names), a
sabotage for a numeric change, old part hashes unchanged
(`/root/oldbits.py <new out> <old out>` on the pod), test_host_surface;
merge each as it passes (directive 0000b), one batched steward request per
hour.

1. UMAP option parity (umap-learn + cuML; `python/mojolearn/_umap_impl.py`
   refuses them in `_parameters`): init 'random' / 'pca' / an array (route: a
   `umap_fit_transform` variant that takes the initial embedding; 'random' is
   umap-learn's uniform(-10, 10) on a Philox stream, 'pca' the x_decomp PCA
   scaled to 10 plus noise); metric (the pdist_cell kinds, DEVIATION 5319,
   into umap/graph.mojo's kNN); local_connectivity != 1 (the rho
   interpolation in smooth_knn_dist); n_components > 3 (the optimizer is
   2D/3D only); supervised y (target_metric / target_weight); a, b given
   directly; densmap (refuse by name if not written).
2. linalg Q: numpy.linalg.qr mode 'reduced' / 'complete' / 'raw' and
   numpy.linalg.svd (U, S, Vt): a Householder QR that keeps its reflectors
   (geqrf's (h, tau) = 'raw') and an orgqr; svd's U = Q U_R. Wide inputs
   (LQ of the transpose). Rows in decomposition/NOT_IMPLEMENTED.tsv.
3. SpectralEmbedding eigen_tol float: one more params entry into
   spectral_embedding_graph / _dataset (bindings/_mojolearn_metrics.mojo),
   the Lanczos config SpectralClustering already exposes.
4. AlternatingLeastSquares use_cg=True: implicit's 3 CG steps per row from the
   previous factors, a row cell beside als_row (new DEVIATION + arm).
5. Then PHASE 3 (FAST speed on NVIDIA / AMD / Apple), per the LANE CHARTER.
