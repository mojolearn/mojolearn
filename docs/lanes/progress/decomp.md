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
  because the end-to-end patch was stale. The old sqdist-only patch could not
  reach every lane (LU, pca never call sqdist), so it is replaced by
  `x_decomp/checks/sabotage/e2e_host_ulp.patch` (host gemm, ew, sqdist and LU
  results each move by one ulp).

## GPU speed (started 2026-09-27; NVIDIA A40, higgs 1M x 28 from R2, IDENTICAL)

| algorithm | before | after | what |
|---|---|---|---|
| PCA randomized / randomized_svd | 108 s (200k rows) | 1.9 s (1M) | orth: one-thread MGS2 -> two passes of the Householder R + a row-parallel solve (DEV 5309) |
| FactorAnalysis (20 it) | 7.8 s | 0.86 s | one QR of Xc, then the d x d SVD of R D / sqrt(n) per iteration |
| lstsq | 3.6 s | 0.9 s | sign flips through absmax_sign_cell (DEV 5317); strided data movement |
| NMF mu (20 it) | 5.6 s | 2.2 s | finite/negative scans through the cells; strided take_cols/hstack |
| FastICA (20 it) | 2.8 s | 1.3 s | row sums past 4096 two-stage (DEV 5301) |
| gemm / colsum / rowsum | one thread per output over 1M terms | FOLD_BLOCK = 4096 two-stage folds (DEV 5300/5301) | |

Next speed targets: device-resident matrices (every kit call still uploads and
downloads; `ew` over 1M x 28 is ~12 ms of copies), the Jacobi eigh's fixed
cost (~60 ms a call), FAST-mode schedules (tiled gemm) with the quality rule;
then CPU speed (last).
