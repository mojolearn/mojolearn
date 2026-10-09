# Linear models: where the fit time goes (read-only analysis, main 42d1e42c6, 2026-10-09)

Lanes: ols, ridge, lasso, elasticnet, logreg, kernel-ridge, sgd-clf. Board rows are the 0.8.25 board (2026-09-30); the
switch grid (freeze-20261007, `~/mojolearn-evidence/grid-lq/{nv,amd}-grid-logs.txt` GRIDB lines, `{nv,amd}-results.txt`
ALGOS lines) has newer numbers for ols, ridge, logreg and sgd-clf; lasso, elasticnet and kernel-ridge were NOT re-raced.

## 0. Shapes the board actually fits (fp32, C-order, one timed round, fit clock includes the X upload)

| lane | rows x cols (bytes of X) | source |
|---|---|---|
| ols taxi | 4,000,000 x 11 (176 MB), unscaled, 1 target | `tools/classical_two_datasets.py:218,625-640`; `tools/speed_gbdt_arm.py:998` (TAXI_NUMERIC) |
| ols istella | 2,043,304 x 220 (1.80 GB) = the whole train split (cap 4M not reached) | `tools/speed_gbdt_arm.py:843-855` |
| ridge, lasso, elasticnet, logreg | 1,000,000 fit rows, standardized; taxi 16 cols (64 MB), istella 220 cols (880 MB) | `tools/bench_board_more.py:131-132,288` |
| kernel-ridge | 10,000 x d (taxi 16 / istella 220); K is 10k x 10k = 400 MB | `tools/bench_board_more.py:146` |
| sgd-clf | 1,000,000 x d, hinge, batch 4096, 100 epochs, no tol | `tools/bench_board_algos.py:219-231` |

Transfer floor (L40S, PCIe 4 x16): pinned ~25 GB/s, pageable NumPy ~10-12 GB/s. 64 MB = 3-6 ms, 176 MB = 7-15 ms,
880 MB = 35-90 ms, 1.8 GB = 72-180 ms. Every IDENTICAL entry below uploads from the pageable NumPy buffer.

## 1. Current numbers (grid, ms) vs the 0.8.25 board row and the best opponent

| lane / ds | 0.8.25 row | grid NV (tag) | grid AMD | best opp (NV) | note |
|---|---|---|---|---|---|
| ridge taxi | 77 | 48.9 (P333) -> **4.8** (P335) | 16.5 -> **3.0** | cuML 9.4 | Gram route taken: WIN |
| ridge istella | 800 | 1,341-1,455 (all tags) | 970-1,037 | cuML 48.2 | Gram gate REJECTS -> incumbent eig route |
| ols taxi | 260 | 39.9 (P338) -> 17.5 (P343) | 15.3 -> 8.1 | torch eigh 1.95 (kernel-only) | upload-bound |
| ols istella | 911 | 734-1,008 | 432-966 | torch eigh 19.8 | gate rejects -> TSQR + one-block Jacobi SVD |
| logreg taxi | 91 | 22.1-23.0 | 23.0 | cuML 9.7 | gradient kernel 16 blocks |
| logreg istella | 6,280 | 1,802-1,887 | 2,217 | cuML 1,850 | parity now |
| sgd-clf taxi | 128,000 | 1,040 | 2,271-2,314 | cuML 1,540 | board row = old per-sample route |
| sgd-clf istella | 615,000 | 1,580-1,692 | 2,915-3,956 | cuML 4,840 | ahead; AMD 2x NV |
| lasso / enet / kernel-ridge | 5,730 / 2,320 / 4,910 | not raced | not raced | cuML 935 / 316 / 324 | main has new routes since 0.8.25 (sec 5, 6) |

## 2. OLS and Ridge: stage tables

### 2a. Shared first stage: the resident Gram fit (MOJOLEARN_CLASSICAL_LINEAR_GRAM_SOLVE, on in IDENTICAL)
Route: `linear_model.py:490-514 _linear_gram_fit` -> `bindings/_mojolearn_estimators.mojo:802-835` -> `glm/impl/gram_solve.mojo:184-289`.

| stage | launches / shape | bytes | host sync | cost istella (1M x 220) |
|---|---|---|---|---|
| upload X, y (pageable) | 2 copies | 880 MB | - | 35-90 ms |
| exact column sums `col_sums_pair_buf` (`center_device.mojo:299-320`; `cs_partial_kernel` :47-65: one thread per (256-row block, column), 9-limb exact adds) | 3 launches, ~860k threads | reads X once | 1 sync (:319) | 3-8 ms |
| means `col_means_buf` + D2H of mu, ybar | 1 launch + 2 copies | 1 KB | 1 sync (`gram_solve.mojo:223`) | ~0.1 ms |
| centered Gram `bm_leaf_gram_kernel` (`core/blocked_moments.mojo:95-190`): grid (1024 leaves, 105 tile pairs), 256 threads, 16x16 cells, smem-staged 32-row tiles, scalar FMA | 1 launch + 10-level fold (`bm_sum_fold`) | reads X ~105/14 = 7.5x (each 16-col panel read by 14 pair-blocks) = 6.6 GB from L2/DRAM | 0 | 10-25 ms |
| centered cross `bm_leaf_cross_kernel` (:554-578): 1024 blocks x 256 threads striding columns | 1 + fold | reads X once | 0 (fold syncs inside `bm_sum_fold`) | 2-4 ms |
| equilibrate (DEVIATION 2620) + Cholesky `gs_chol_step_kernel` one launch per column, thread i does a serial dot over p<j | 2 + d launches (220) | d^2 | 0 | 1-3 ms |
| trust gate `gs_trust_kernel` (1 thread) + D2H flag | 1 launch, 1 copy | 4 B | 1 sync (:257) | 0.1 ms |
| two triangular solves, one launch per column; unscale; D2H coef | 2d+1 launches | d | 1 sync (:272) | 1-3 ms |

Gate (`glm/impl/gram_solve_cells.mojo:40-43`, GS_TRUST_GATE 2^-12 :14): every equilibrated pivot must satisfy
l_jj^2 >= 2^-12 a_jj, i.e. cond(S(G+alpha I)S) <~ 4,096. Istella's 220 LTR features are strongly collinear (several are
near-duplicates / constants), and alpha = 1 is negligible against G_jj ~ 1e6 (1M standardized rows), so the gate fails for
BOTH ridge and OLS on istella: status 1, nothing written, the whole pass above (~60-130 ms) is wasted, and the caller runs
the incumbent route. That is why the ridge istella cell did not move under the flip while taxi went 48.9 -> 4.8.

### 2b. OLS incumbent (gate rejected or weights): TSQR + SVD of R
Route: `linear_model.py:733-740 _ols_tsqr_centered` -> `x_decomp_ols_tsqr_r` (resident center + TSQR, X up once) ->
`_expansion_decomp.py:3056-3104 _tsqr_lstsq_core` -> `_linalg_impl.py:1469-1486 _svd_tall` (QR + one-sided Jacobi) -> ~10 Kit ops.

| stage | launches | istella cost (2.04M x 220) |
|---|---|---|
| upload X (1.8 GB pageable) | 1 | 72-180 ms |
| exact col sums + center (resident) | ~5 | 10-20 ms (reads/writes 1.8 GB twice) |
| TSQR `ts_factor_device` (`x_decomp/tsqr_device.mojo:1041-1155`): leaves of 2048 rows (TS_ROWS, C24), 14 panels of 16 as 7 WY pairs; per pair 4-5 launches over ~1,000 leaf blocks + update over trailing columns; tree combine log2(1000) levels; ONE sync at the end | ~60 | 2mn^2 = 200 GFLOP scalar + ~14 GB traffic: 40-90 ms |
| SVD of the 221 x 221 R: **one-block cyclic Jacobi** (`decomposition/checks/jacobi_eigh_device.mojo:324-325, 459-520`): 24,090 rotations and 96,360 barriers per sweep, 12 sweeps on istella, grid (1,1,1) | 1 launch, ~1.2M barriers | **0.45-0.7 s** (the dominant stage) |
| min-norm solve: `k.word` (D2H), `k.count_gt` (D2H), select/recip/mm/mm (small) | ~10 launches, 2-3 syncs | ~1 ms |

OLS taxi (4M x 11) takes the Gram route (gate passes): 17.5 ms NV = 176 MB upload (12-15 ms pageable) + ~1 ms kernels +
3 syncs + a 176 MB `enqueue_create_buffer`. It is upload-bound; torch-gpu-eigh's 1.95 ms is kernel-only (inputs on device).

### 2c. Ridge incumbent (gate rejected): the Python four-crossing eig route
`linear_model.py:950-968`: `_column_means` (uploads X, `lm_col_sums`), `_center` (uploads X, downloads centered X),
`ridge_fit` (`glm/estimator.mojo:419-450`: uploads centered X again) -> `ridge_eig_traced` (`glm/impl/ridge.mojo:317-352`):
`gemm_tn` Gram (`glm/impl/linalg/detail/svd.mojo:147`), **one-block Jacobi** (:161-168, grid (1,1,1), 15 sweeps max),
`U = A V` (writes an m x d U: RIDGE_NO_U is FAST+Apple only, `ridge.mojo:112-116`), `S = U^T b` gemv, `w = V S`.
The resident one-upload form `ridge_fit_resident_host` (`glm/estimator.mojo:455+`) exists but its flag
`RIDGE_FAST_RESIDENT` is FAST+Apple only (`bindings/_mojolearn_estimators.mojo:60-64`).

| stage | istella cost |
|---|---|
| wasted Gram pass (gate reject) | 60-130 ms |
| 4 PCIe crossings of 880 MB (3 up, 1 down) + `_helper_output` allocs | 150-350 ms |
| exact col sums, center (2 full passes, each its own sync) | 5-10 ms |
| gemm_tn 220^2 x 1M (scalar tiled) | 10-25 ms |
| one-block Jacobi on the 220 x 220 Gram | 0.45-0.7 s |
| U = A V (880 MB write) + gemv | 3-5 ms |
Sum ~0.7-1.2 s: matches 1,341-1,455 NV / 970-1,037 AMD (AMD's 6 TB/s hides the crossings less than its faster block).

### 2d. Ideas: OLS / Ridge

| # | idea (define) | what changes | expected | bits | identity risk | effort | Opus? |
|---|---|---|---|---|---|---|---|
| L1 | **Parallel-ordered Jacobi** (`MOJOLEARN_IDN_JACOBI_ROUND_ROBIN`) | replace the one-block cyclic sweep with the round-robin (chess-tournament) ordering: d-1 rounds per sweep, each round d/2 independent rotations applied in a fixed order, one launch (or one block set) per round; the rotation arithmetic, tolerance and sweep test stay (`jacobi_eigh_device.mojo:459-520`); host column (`decomposition/host/pca_oracle.mojo:204-205`) runs the same schedule | istella: 0.45-0.7 s -> ~20-40 ms (12 sweeps x 219 rounds x ~8-15 us). ridge istella 1,400 -> ~500 before L2/L3; ols istella 800 -> ~250. Also PCA / lstsq / every eigh user | rotation order changes -> bits change (fold change with host column, allowed within a version) | low: schedule is a pure function of d | M-L | yes, from this text + the kernel file |
| L2 | **Resident ridge fallback** (`MOJOLEARN_IDN_RIDGE_RESIDENT`) | when the Gram gate rejects, do NOT return to Python: run the eig route on the already-resident X and the Gram already formed (`gram_solve.mojo:231-246` has G, c, mu): Jacobi on the equilibrated A, w = S V S^-1 V^T b (same math as `ridge_solve_no_u_traced`, `ridge.mojo:211-270`), no U, no crossings | removes 3-4 crossings (150-350 ms) and the duplicated sums/center/Gram (~80 ms); ridge istella -> ~0.6 s alone, ~0.15 s with L1 | fold of the Gram differs from `gemm_tn`'s -> bits change w/ host column; or keep `gemm_tn` on the resident centered copy for no bit change (then only the crossings go) | low | S-M | yes |
| L3 | **Float-float Gram solve when fp32 gate fails** (`MOJOLEARN_IDN_GRAM_FF_FALLBACK`) | second pass: centered Gram + cross accumulated in float-float (`x_linear/ff.mojo` ff_cross/ff_cholesky already exist for ridge-clf's RIDGE_FF_ALWAYS arm), ff Cholesky on d x d, solve; skip Jacobi entirely for alpha > 0 | ridge istella -> upload + 2 passes over X + tiny solve ~ 120-180 ms (vs cuML 48 whole-clock incl. its copy); quality: ff ~ fp64 accuracy >= fp32 eig | bits change vs eig route (host column `glm/host/gram_solve_host.mojo` gets the same ff cells) | low | M | yes |
| L4 | **Pinned upload** (`MOJOLEARN_IDN_PINNED_UPLOAD`) | stage the NumPy buffer through a pinned host buffer (`enqueue_create_host_buffer` pool, chunked) before `enqueue_copy`; OLS taxi is 70-85% upload today | 176 MB: 12-15 -> ~7 ms; 1.8 GB: ~150 -> ~75 ms; 880 MB: ~70 -> ~35 ms; helps every classical lane | none | none | S-M | yes |
| L5 | Both clocks on the OLS/ridge cells | `upload_ms_separate` is recorded since freeze-20261007; make sure the ols/ridge cells show kernel/kernel vs torch-gpu-eigh (CLAUDE.md OPPONENT CLOCKS) | OLS taxi kernel-only ~3-5 ms vs 1.95: ratio 9x -> ~2x on the shown clock | none | none | S | yes |
| L6 | OLS: SVD of R only when R is ill-conditioned (`MOJOLEARN_IDN_OLS_R_TRISOLVE`) | after TSQR, estimate cond(R) from the diagonal of the equilibrated R (power-of-two scales as DEVIATION 2620); if min/max |r_jj| > cutoff, solve R w = Q^T y by one triangular solve (d launches) and skip the Jacobi; else the SVD as today | ols taxi (already Gram) no change; ols istella depends on cond(R): if it passes -> ~250 ms; if not L1 is needed anyway | different solve when the trisolve is taken -> bits change w/ host column | low | M | yes |

Already in main / measured (do not re-propose): Gram solve (ridge taxi 50 -> 4.8), TSQR leaves 2048 (C24), c02_linear_pair
(neutral), IDN_OLS_ONE_ENTRY (center + TSQR one entry, `classical-fixes.json`), C05_OLS_PHASE (unreached), RIDGE_FF_ALWAYS
(x_linear ridge-clf/ridge-cv only), FAST-Apple RIDGE_RESIDENT / RIDGE_NO_U / OLS_FAST_NORMAL_EQ (EXPERIMENTS.md:231,237,238).

## 3. Logistic regression (L-BFGS, cuML qn's form)

Route: `linear_model.py:1236-1290` -> `qn_fit` -> `glm/estimator.mojo:639-718 qn_fit_host` (X, y up, 1 sync; w down, 1 sync)
-> `qn_solvers.mojo:123-302 min_lbfgs`. Per iteration: `ls_backtrack` (`qn_linesearch.mojo:133-176`; default
`ls_backtrack_sequential` :177-222; C17_LS_TRIALS = 2 speculative trials, kept, A/B arm) -> `f.evaluate`
(`glm_base.mojo:2003-2060`): forward `gemv_n` (coalesced, 1 read of X), loss/dZ kernel, backward `linear_bwd`
(`glm_base.mojo:1702-1800`): `xty_kernel` (`core/column_stats.mojo:207-231`): **grid = d blocks x 256 threads, each thread
strides rows, reads x[r*d + j]** (stride-d, 1 useful float per 32 B sector), then epilogue; loss + penalty words come home
behind ONE synchronize per evaluate. Direction: `lbfgs_search_dir_enqueue` one small launch, no sync. `QN_FAST_XTDZ`
(rows across blocks, X read once) and the fused one-pass `QN_FAST_FUSED` (d <= 32) are FAST+Apple only
(`glm_base.mojo:100-104, 523-538`). MOJOLEARN_QN_IDN_DCONV (device Armijo + convergence loop) was tried 2026-10-08:
neutral/slower on NV/AMD, deleted (`qn_solvers.mojo:274-276`): do not repeat.

| stage per evaluate | istella (1M x 220) | taxi (1M x 16) |
|---|---|---|
| gemv_n forward | 880 MB coalesced: ~1.2 ms | 64 MB: ~0.1 ms |
| loss/dZ, tile sums | ~0.1 ms | ~0.1 ms |
| xty_kernel gradient | 220 blocks, 8x sector amplification on 880 MB (L2 helps partly): 3-9 ms | **16 blocks on 142 SMs**, 3,900 serial strided loads per thread: 2-4 ms latency-bound |
| epilogue + sync | ~50-100 us | ~50-100 us |
With ~150-250 evaluates (max_iter 1000, tol 1e-4) that is ~1.8 s istella (matches) and ~20 ms taxi (matches 22-23).

| # | idea (define) | what changes | expected | bits | risk | effort | Opus? |
|---|---|---|---|---|---|---|---|
| G1 | **Row-split X^T dZ for IDENTICAL** (`MOJOLEARN_IDN_QN_XTDZ_ROWS`) | the gradient as a leaf fold over row blocks (as `bm_leaf_cross_kernel`: leaves of `contract_leaf_size(n)` rows, adjacent threads adjacent columns, binary-counter fold), replacing the d-block strided kernel on NVIDIA/AMD; host column the same fold | istella evaluate 5-10 -> ~2 ms: logreg istella 1,850 -> ~700-900 ms (2x ahead of cuML); taxi evaluate 3 -> 0.3 ms: 22 -> ~8-10 ms (parity) | fold change -> bits change w/ host column | low (same pattern as the Gram leaves) | M | yes |
| G2 | **One-pass fused evaluate for d <= 32, IDENTICAL** (`MOJOLEARN_IDN_QN_FUSED_SMALL_D`) | an IDENTICAL twin of QNF (`glm_base.mojo:2190-2260`): per 16-row tile forward, loss, dZ and gradient partials in one kernel, one fold; applies by width (d <= QNF_MAX_D), the host column folding the same tiles | taxi: X read once per evaluate, 2 launches instead of 5: ~5-6 ms total vs cuML 9.7 | bits change w/ host column | low | M | yes, after G1 |
| G3 | Fewer evaluates per iteration | C17_LS_TRIALS (2 trials, one sync) is in the grid; read its verdict before more line-search work | - | - | - | - | - |

## 4. SGD classifier (minibatch, hinge, 100 epochs)

Route: `_expansion_linear.py:434-446 _sgd_fit` -> `x_linear/device.mojo:5127-5132` -> `_sgd_mb_grid` (:1827-2095). Per epoch:
`sgd_perm_kernel`, optional contiguous gather (SGD_EPOCH_CONTIGUOUS, classical-structural switch), then per batch THREE
launches (rows: one thread a row; parts: (d+2) x sub-blocks; step: d+2 threads), no sync inside the epoch; epoch end: one
witness check (`wit.ok`: a readback) + one-block `sgd_end` kernel + `ctx.synchronize()` (:2092). taxi/istella: 245
batches/epoch -> 735 launches/epoch -> 73,500 launches + ~200 syncs per fit. The chain is inherent (batch b+1 reads the
weights batch b wrote), so the floor is 73.5k dependent launches ~ 0.4-0.7 s: that IS the 1,040 ms taxi cell (istella adds
the 880 MB upload + wider rows). The 0.8.25 rows (128-615 s) were the per-sample one-block route (`_sgd_ps_grid`, grid 1,
sample by sample: 100M dependent steps) before 3b7aea8c7 (2026-10-01). C19_SGD_CHUNK and C17_OVR are unreached on the board.

| # | idea (define) | what changes | expected | bits | risk | effort | Opus? |
|---|---|---|---|---|---|---|---|
| S1 | **Device-sequenced epoch for small d** (`MOJOLEARN_IDN_SGD_EPOCH_KERNEL`) | one launch runs K consecutive batches (one block of 1024 threads per launch, batches in order; the batch's dot products, fold and step inside with barriers); rule by work per batch (bs x d FMAs <= ~1e5, i.e. a block finishes a batch in ~10-20 us); wide rows keep the grid form | taxi: 73.5k launches -> ~250-1,000: ~1,040 -> ~400 ms (2.5x+ below cuML 1,540); istella unchanged | in-batch combine order must be the same fixed order as the parts/step kernels -> can be bit-identical if the fold tree is reproduced; else fold change w/ host | medium (ordering) | L | yes with a precise fold spec |
| S2 | **Fuse rows+parts** (`MOJOLEARN_IDN_SGD_ROWS_PARTS_FUSED`) | the rows' loss derivatives and the gradient partials in one kernel (block per sub-block: compute dloss for its rows, then the column partials from registers/smem) | 3 -> 2 launches per batch: ~-30% of launch time on both datasets | none if the per-sub-block partial order is kept | low | M | yes |
| S3 | AMD 2x NV | same launch count; AMD launch latency is higher: S1/S2 pay twice there | - | - | - | - | - |

## 5. Lasso / ElasticNet (coordinate descent)

Route: `_solver_impl.py:338-400` (C-order X crosses as is, `row_major=1`) -> `solver/estimator.mojo:81-180 cd_fit_host`
(upload, device `transpose_kernel`, sync) -> `cd.mojo:1278+ cd_fit_traced`. On NVIDIA/AMD IDENTICAL both board shapes
pass `cd_idn_gram_shape` (`cd_gram_rule.mojo:40-45,71-77`: n_cols <= 256, n >= 4d, d(n+128) <= 256n) so the fit takes
CD_IDN_GRAM (`cd.mojo:1023-1031, 1656-1700`): G = X^T X and q = X^T y via `identical_gemm_into` (1M x 220^2, scalar
tiled + leaf fold), then `cd_idn_gram_sweep_kernel`: **one block**, coordinates serial (inherent to cyclic CD), q[k] moves
in parallel (thread k), 16 epochs per launch, 4-word readback per launch (<= 63 syncs for max_iter 1000). The row form
(`cd_fused_step_kernel`, one launch per coordinate per epoch + a per-epoch sync, `cd.mojo:1700-1790`) is the fallback.
Cost model: istella Gram 10-25 ms + sweeps ~0.8 ms/epoch (220 coords x 3-4 barriers) -> 50-800 ms depending on epochs
to tol; taxi ~0.05 ms/epoch -> 5-60 ms + 64 MB upload. The 0.8.25 rows (5,730 / 63.5) predate this route.

| # | idea | expected | effort |
|---|---|---|---|
| C1 | **Re-race lasso and elasticnet first** (not in the grid); record `n_iter_` (epochs) on the cell | tells whether the sweep or the Gram dominates | S |
| C2 | Epochs per launch 16 -> 64 for small d (`MOJOLEARN_IDN_CD_GRAM_EPOCHS_64`): the readback is ~50 us; at d = 16 an epoch is ~50 us, so the sync is half the time | taxi: ~2x on the sweep phase | S |
| C3 | Gram by `bm_leaf_gram_kernel` instead of `identical_gemm_into` (same leaf rule `contract_leaf_size`): the blocked-moments kernel reads X 7.5x from L2 but needs no d x n transpose pass; drops `transpose_kernel` + its sync | istella: -1 pass over 880 MB and one sync (~5 ms) | M (bits: same leaf fold if the per-cell chain is the same; verify with the host cell) |

## 6. Kernel ridge (RBF, n = 10,000)

Route: `kernel_methods.py:258-300` -> `kernel_methods/estimator.mojo:620-780 kernel_ridge_fit_host`: upload X (10k x d, 1 sync),
`km_kernel_matrix` (`kernel_methods/checks/kernel_matrix.mojo:792+`: `km_rbf_cell_kernel` one thread per cell walking d when
d <= KM_RBF_CELL_MAX_D, else dot via `identical_gemm_into` + rbf epilogue; sync), ridge the diagonal (sync),
`kernel_ridge_solve` (`kernel_methods/impl/kernel_ridge/kernel_ridge.mojo:251-340`) -> `potrf_lower`
(`cholesky/checks/potrf.mojo:2012-2300`): nb = 32 pinned in IDENTICAL (`:402`; 128 under MOJOLEARN_IDN_CHOL_NB128,
`potrf_blocked.mojo:54-60`, the gap-linalg grid switch), per panel: one-block panel factor, trsm panel, pack,
`identical_gemm_into` trailing update (n_trail^2 x 32), and **a synchronize per panel to read info** (`:2045`, DEVIATION 1634)
-> 313 panels x (~5 launches + 1-2 syncs); then `cho_solve` (`cholesky/checks/trsm.mojo:918-954`): one block of 1024
threads sweeping 10k columns (forward + back, ~20-50 ms).
Cost model today: K 20-50 ms, Cholesky n^3/3 = 333 GFLOP scalar-tiled 60-150 ms + 313 syncs ~20 ms, solve ~30 ms:
~150-300 ms total vs cuML 324 (cuSOLVER potrf). The 0.8.25 row (4,910 ms, both datasets identical) is consistent with the
host/one-thread route of that build (as the LU rows in `docs/plans/gaps-2026-10-08.md:9-14`), not with this path.

| # | idea | expected | effort |
|---|---|---|---|
| K1 | **Re-race kernel-ridge** with MOJOLEARN_STAGE_TIMES=1 (`kernel_ridge.mojo:312-335` prints factor/solve walls) | confirms the row is stale | S |
| K2 | Take the gap-linalg verdict on MOJOLEARN_IDN_CHOL_NB128 (`_potrf_lower_blocked`, no per-panel waits, `potrf.mojo:2098-2107`) | 313 -> 79 panels, 0 per-panel syncs: Cholesky ~2x | already a switch |
| K3 | K build by GEMM + epilogue for every d (the cell kernel reads d floats per thread from two rows: 2 x 10k x 10k x 220 x 4 B = 176 GB of L1/L2 traffic at d = 220) | istella K: 30-50 -> ~10 ms | S-M (bits change w/ host cell `km_oracle.mojo:171`) |

## 7. Order of work (by expected board movement per effort)
1. L1 (parallel Jacobi: ols istella, ridge istella, and every eigh user) and L2 (resident ridge fallback) together: ridge istella ~1,400 -> ~150-250, ols istella ~800 -> ~250.
2. G1 (row-split gradient): logreg istella -> ~0.8 s, taxi -> parity. 3. L4 + L5 (pinned upload, both clocks): ols taxi.
4. C1 / K1 re-races, then C2, K2/K3. 5. S2 then S1 for sgd-clf (already ahead of cuML; AMD is the weaker column).
Nothing above depends on Modular shipping anything; none keys on a board dimension (rules are by width/work, stated above).
