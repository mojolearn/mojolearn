# decomp: progress

> **AWAITING ANDREW: the `linalg.qr(a)` default changed from `'r'` (R alone) to numpy's
> `'reduced'` ((Q, R)). This is a breaking change to a public call. It is kept on the lane and
> recorded as BREAKING in CHANGELOG.md "Changed"; revert it if Andrew says no.**

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
- p2d-rest: the other 11 x-decomp lanes AGREE; old part hashes vs s7-x: 1692
  compared, 0 moved.
- p2d-old: pca, pca-full-whiten, pca-whiten, tsvd, spectral,
  spectral-embedding, spectral-precomputed, linalg-qr, linalg-eigh,
  linalg-svdvals AGREE; vs s3-o 324 parts compared, 0 moved.
- pytest test_x_decomp_repeat (GPU ran, not skipped), test_spectral_embedding,
  test_linalg_decompositions: 33 passed. test_lane_select (box worktree at
  13ffd2f1): OK, 0 failures.

MERGED to main ca7782942 (the options, the context fix, row 130) and
3266b66bb (x_decomp/checks/sabotage/e2e_host_all.patch: the host ulp and the
reversed host sqdist in one patch, so ONE steward request covers all 17
x-decomp lanes). Steward request 1790542293472-decomp-3266b66bb0 (17 lanes,
e2e_host_all) queued on m2pro, m3ultra, m4pro-a, do-amd: a post-merge release
gate; a FAIL comes back as a fix at the root.

## PHASE 2 REMAINDER, session 5 (2026-09-27): CODE DONE, HOST-PROVEN, POD GATE OWED

The RunPod account balance went negative: every pod was deleted (the decomp
A40 4phrsbddlgcd5a included) and `dev_pod.sh up` is refused ("balance too
low"). Do NOT rent until Andrew tops up. Everything below is committed and
pushed on lane/algos-decomp (NOT merged: the NVIDIA + CPU gate has not run).

Done in code (commits 386898fe0, a6fc1ea56, 3bc50cd85, b27e071d7, cb64532b6,
114b1eacb and after):

| item | route |
|---|---|
| merge review: solver names | PCA svd_solver='arpack', TruncatedSVD algorithm='arpack', a nonzero tol, SpectralEmbedding eigen_solver other than None, Isomap/LLE eigen_solver='arpack' REFUSED BY NAME (3bc50cd85); none is aliased to another algorithm |
| merge review: TruncatedSVD's x_decomp dependency | ROOT FIX: explained_variance_ / _ratio_ in TruncatedSVD's own binding, `tsvd_explained` (decomposition/estimator.mojo `tsvd_explained_host`: gemm_nt, column_mean_kernel, shift, pinned square; host twin pca_oracle.mojo `host_tsvd_explained`; IDENTITY_PATHS 139d). The default fit no longer loads _mojolearn_x_decomp (only algorithm='randomized', an x_decomp algorithm, does). New part `explained` on the `tsvd` lane |
| merge review: linalg.eigh one triangle | numpy's semantics kept (reads the UPLO triangle); CHANGELOG "Changed" entry + test (3bc50cd85); symmetric inputs keep their bits |
| linalg.qr every numpy mode, linalg.svd (U, S, Vh) | geqrf + orgqr cells (DEVIATION 5320); qr(a) now defaults to 'reduced' (numpy) -- CHANGELOG; mode='r' keeps its TSQR bits (its rows may differ in sign from qr(a)[1], documented); svd's U = the Householder re-orthonormalization of A v / s (orthonormal to float32 at any condition); QRResult/SVDResult private as in numpy |
| AlternatingLeastSquares use_cg | als_cg_row (DEVIATION 5321) |
| SpectralEmbedding eigen_tol float | e2e_eigen_tol.patch (a6fc1ea56) |
| UMAP option parity | n_components 1-32 (run-time-dimension kernel + host twin, DEVIATION 5322; 2/3 keep the comptime kernel), local_connectivity (DEVIATION 5323), metrics sqeuclidean/cosine/manhattan/chebyshev/minkowski p (the k-NN lane's arms), init random/pca/array, a/b, supervised categorical + l2 targets (DEVIATION 5324); densmap, output_metric, other metrics/target metrics REFUSED BY NAME. New entry `umap_fit_transform_ex` (GPU + host); transform takes the metric/a/b; save/load carries them. Lane `x-decomp-umap-options` |

Host-only proof (Mac, one core, tools/mac_slot.py; ~/mojolearn-evidence/algos-decomp/hostbuild):
the metrics, estimators, x_decomp, core, linalg HOST bindings build clean and
also build under `x_decomp/checks/sabotage/e2e_p2b_options.patch`; TruncatedSVD
explained vs sklearn 3e-7; qr/svd vs numpy 4e-7 / 2e-6; every UMAP option
fits finite (sanity_p2b.py); the p2b sabotage moves every new part on the host
(tsvd_ev, umap_c5, umap_cat, umap_man, qr, als_cg); test_host_surface,
test_linalg_decompositions, test_umap_options, test_x_decomp_repeat (now
with geqrf/orgqr/als_cg_rows), test_spectral_embedding, test_pca_full_surface:
all pass on the host. The GPU bindings have NOT been compiled.

## NEXT (start here, on a pod once RunPod is funded)

1. `tools/dev_pod.sh up decomp`, `MOJOLEARN_DEVPOD_ALLOW_SELF=1 tools/dev_pod.sh sync decomp <worktree>`.
   Build every GPU binding the lanes run (metrics, estimators, x_decomp,
   linalg): the GPU side of umap_identical_epoch_kernel_rt, tsvd_explained_host
   and the 5320/5321 cells has never compiled. Fix what fails.
2. Seam arms: `tools/algos_lane_check.sh x-decomp-lu --pass 2` runs every
   decomp.checks driver; 5320_householder_order and 5321_als_cg_order must
   BUILD, RUN, FAIL, PASS after reversal (never proven yet).
3. Lane checks (lane_select names all 481 because the x_decomp binding and
   host_surface.py changed; the lanes that carry the new code are):
   `x-decomp-umap-options,x-decomp-als,x-decomp-pca-randomized,x-decomp-lstsq-rsvd,tsvd,linalg-qr,umap,spectral-embedding`
   with `--pass 2 --sabotage x_decomp/checks/sabotage/e2e_p2b_options.patch`
   (AGREE, DISAGREE, AGREE), then the other x-decomp lanes clean.
4. Old part hashes unchanged (`/root/oldbits.py <new> <old>` against the
   p2d-* outputs): every old part of every lane above; EXPECTED to move: none
   (tsvd's `explained` and x-decomp-pca-randomized's `ta` are the explained
   variance through a new summation order; `ta` was never on a release record).
5. Sanity on the pod: UMAP options vs umap-learn (trustworthiness / the
   supervised graph's rho, sigmas, weights for lc=1.5, categorical, l2);
   ALS use_cg vs implicit's _least_squares_cg.
6. test_host_surface + test_lane_select, merge to main and push in one
   command, ONE batched steward request (e2e_p2b_options + e2e_host_all over
   the decomp lanes), progress file. Then PHASE 3 (FAST speed).

## SESSION 7 (2026-09-28 ~05Z): RunPod out of money again; no pod

State on lane/algos-decomp (pushed, NOT merged to main):
- Session 6 (a pod, RTX 4090 ukollon8nsl8oz, down 2026-09-28 00:00Z) ran part
  of the phase-2-remainder gate; what it proved is in IDENTITY_PATHS rows
  139a-139e (x-decomp-umap-options and tsvd `explained` AGREE on the 4090,
  DISAGREE under e2e_p2b_options.patch, AGREE restored; UMAP option sanity vs
  umap-learn). Its IDENTICAL speed commits (orth on the device across passes,
  geqrf/orgqr and getrf on the device in parallel steps, absmax in FOLD_BLOCK
  slices DEVIATION 5317 + arms 5317/5317b, one-copy input, 5307 arm vs
  lu_pivot) have NO recorded NVIDIA + CPU gate: treat them as unproven.
- Steward 1790542727482-decomp-3c73fcee93 (x-decomp-spectral-rbf,
  e2e_host_sqdist): do-amd FAIL = timeout (exit 124) in the wide/train cell;
  taken as a do-amd hang (neural's same-minute hang did not reproduce on Hot
  Aisle). RESUBMITTED to do-amd only as 1790571228173-decomp-3c73fcee93
  (Apple already PASS on m4pro-a / m2pro via 3266b66bb0).
- Steward 1790542293472-decomp-3266b66bb0: m4pro-a FAIL was NOT our numerics:
  every x-decomp-dict-learning cell REFUSED because _mojolearn_x_linear.so was
  never built on a clean Mac (DictionaryLearning's lars transform and MDS
  non-metric run the linear lane's Lars / IsotonicRegression). Root cause in
  tools/lane_select.py: a lane's own seed that every lane also reaches
  (`_expansion_decomp.py`, via `_linalg_impl.py`) was entered and not
  followed, so its imports were lost, and a class imported by name leaves its
  binding narrow (never built by the lane check). FIX (this session):
  `_own_walk(..., follow=seeds - sinks)` follows a lane's own non-registry
  seeds; `_expansion_decomp.py` imports `_expansion_linear` as a module.
  Measured: declared bindings change for 7 unrelated lanes (bootstrap,
  byte-lm*, metrics-classification: bindings they already load) and every
  x-decomp lane + cholesky gains x_linear(+host); source sets grow for the
  x-linear lanes (57 -> 133 files: their own door's imports now count),
  gemm/linalg/lowbit lanes (90 -> 139), sequence and cnn lanes (+3 to +5).
- AMD central box: our IDENTICAL "before" remainder (bench/decomp_speed.py,
  /root/ev-decomp/speed_before_identical_rest.log) has held slot 0 since
  02:21Z; linalg.qr / svd (geqrf 217 s, orgqr 15 s at 1M x 28: the one-thread
  cells, before 3eb5dd554), solve(512) 7.1 s (lu one thread, before
  5cc491ca8), ALS 10.4 s (als_rows 9.7 s), ALS cg 2.7 s are recorded; it has
  sat in Isomap(10nn) at N3=10000 since 02:38Z (python 100% CPU, GPU0 100%).
  Not cancelled (owed run rule). Isomap at 10k rows is the next speed target
  (dijkstra_rows); the bench's N3 must drop (or Isomap be fixed) before the
  "after" run.
- Apple speed 1790562095893 / 1790562097205 (m4pro-b / m4pro-a) run commit
  32bf8cbb80, whose bench fits MinCovDet on 1M rows (the fix 3638c5c29 came
  after): they will likely hit the steward's timing timeout like
  1790558492260 / 1790558501645 did. Resubmit at a commit with the fixed bench
  and ONLY= lists, BEFORE the Macs go (m2pro/m3ultra ~12:35Z, the rest
  ~21:15Z Sep 28).

## NEXT (start here)

1. Pod (once RunPod is funded): gate every commit since 069bf7678 on NVIDIA
   + CPU: `tools/algos_lane_check.sh <every x-decomp lane> --pass 2 --sabotage
   x_decomp/checks/sabotage/e2e_host_all.patch` (all seam arms incl. 5317b,
   5320, 5321 BUILD/RUN/FAIL/PASS), old part hashes unchanged vs the p2d-*
   outputs (oldbits.py), test_host_surface, test_lane_select (its inputs
   changed). Then merge to main + push in one command and ONE batched steward
   request (e2e_p2b_options + e2e_host_all over the decomp lanes).
2. Until then, the central AMD box can carry the same lane check for gfx942
   (tools/amd_central.sh run decomp ...), but the merge gate is NVIDIA + CPU.
3. Speed (phase 1 IDENTICAL, then FAST): Isomap dijkstra_rows, ALS als_rows,
   the per-call uploads (device-resident matrices), Jacobi eigh fixed cost;
   AMD and Apple before/after tables per algorithm.

## OWED (orchestrator, 2026-09-28, from Andrew)

- Isomap hangs/stalls at 10k rows on AMD (MI300X), bench/decomp_speed.py;
  find and fix. The "before" speed bench on the central AMD box sat in
  Isomap(10nn) at N3=10000 from 02:38Z, 100% CPU for 4.5 h with no output,
  and was killed on Andrew's order. It is a BUG to fix at the root, not a
  slow run. Before-numbers come from records already taken: never re-measure
  old code, never run Isomap at that size, never resubmit that bench.
