# lane/apple-fast-kernel: kernel methods, GP and Bayesian linear lanes under FAST on Apple

Written without a Mojo toolchain (cloud peer); the first M3 build is the compile check.
Every switch is compiled under FAST + Apple only and defaults OFF; IDENTICAL compiles the old code.
Bindings touched: x_linear (bayes.mojo, bayes_fast.mojo, device.mojo, tops.mojo), gp (estimator.mojo),
kernel_methods (estimator.mojo). Nothing in gemm/, core/gemm.mojo, decomposition/impl/linalg/detail/pca.mojo.

| switch | kind | site | what it changes under FAST on Apple |
|---|---|---|---|
| `MOJOLEARN_KERNEL_FAST_BAYES_STATS=1` | env, read in `x_linear/device.mojo` fit_device | `x_linear/bayes_fast.mojo` (new); the grid driver's setup unit in `fit_device` (BayesianRidge, unweighted, on main's `bayes_grid` path only; ARD keeps main's moments grid) | the column means, the centered Gram, X'y, |y - ym|^2 and var(y) of all rows on the whole grid (8192-row chunks of 32 x 32 tiles, chunk partials folded ascending: enetcv_fast's pattern on [X, y]) in place of `xg_means_kernel` + `xg_gram_kernel` (one SERIAL million-row float32 chain per Gram cell, `chain_cfmad`) and the team's own one-block passes (`t_col_means`, `t_mean`, the X'y chains, `_var` on the lead thread) |
| `MOJOLEARN_KERNEL_FAST_BAYES_JACOBI=1` | env, same site (ip[7]) | `x_linear/tops.mojo` t_jacobi_eig (new `skip_tol` parameter, default the old 1e-9), through `bayes_eig_prep`'s `jtol` from `bayes_prep` (team) and `bayes_eig_kernel` (grid driver) | the team Jacobi skips a rotation at |a_pq| <= 1e-7 sqrt(a_pp a_qq) instead of 1e-9; float32 roundoff leaves every |a_pq| near 6e-8 x scale so the 1e-9 test never fires and the 220 x 220 solve ran all 60 sweeps (about 5x what the diagonal needs), 24,090 rotations x 3 barriers each on one block |
| `MOJOLEARN_KERNEL_FAST_GPR_RESIDENT=1` | env | `gaussian_process/estimator.mojo` gpr_fit_host | K stays on the device for add_jitter, potrf_lower, chol_logdet and cho_solve; L and the dual come down once. The shipped path downloads K, runs `chol_validate_matrix` on the host (a symmetry pass over n^2 cells), uploads K again (`cholesky_factor_host`), downloads L, uploads L and y (`cholesky_solve_host`), downloads the dual: five n^2 host round trips around one 3,000-row factorization. Not taken with an identity card or a sabotage arm |
| `MOJOLEARN_KERNEL_FAST_NYS_RR_EIGH=1` | env | `kernel_methods/estimator.mojo` _nystroem_device_eigh (new `_nystroem_rr_eigh`) | the q x q basis kernel's eigenproblem by x_decomp/jacobi_par.mojo's round-robin Jacobi (q / 2 disjoint rotations a launch, two launches a round, the cyclic test on the host once a sweep, 30-sweep budget, fallback to the cyclic kernel on the untouched input) instead of `jacobi_eigh_kernel`: ONE block of 256 threads running the 32,640 rotations of a sweep (q = 256) serially behind two barriers each |

## Causes, per lane (file:line at head f5f61bde)

- **bayesian-ridge / ard (quality, Istella FAST r2 -4.2e4 vs sklearn -890).** Not a bug in FAST's update order:
  `x_linear/bayes.mojo` bayes_ridge_fit matches scikit-learn's `BayesianRidge.fit` statement for statement
  (coef = V diag(1/(ev + lambda/alpha)) V'X'y; gamma, lambda, alpha updates; `sum|coef_old - coef| < tol` from
  the second iteration; the final coefficient update with the updated hyperparameters, also on the max_iter
  exit; alpha_init = 1/(var(y) + eps), lambda_init = 1). The difference is the eigenproblem's input: theirs is
  the float64 SVD of X; ours is the float32 centered Gram X'X, every cell one serial fmad chain over the
  million rows (`x_linear/device.mojo:128` xg_gram_kernel / `x_linear/tops.mojo:59` t_centered_gram,
  `chain_cfmad` with one accumulator). A serial float32 sum of 1e6 terms carries an error of order n x eps
  (6e-5 relative typical, 6e-2 worst) on diagonal cells of 1e6, so the Gram's small eigenvalues (Istella's
  near-collinear columns) are roundoff, their V'X'y components are noise, and coef = vty_k / (ev_k + lambda/alpha)
  puts mass on those directions; the held-out rows with large values there blow the R2 up. IDENTICAL takes the
  same chains, so the row is expected to be bad in both tiers (the board's IDENTICAL column is not in this tree).
  The STATS arm cuts the error to order log(n) x eps (8192-term partials, 123-term chunk fold); the paired
  quality of `kernel-bayes-stats-ist` against `-` is the test of this diagnosis. `_t_sse` / GRAM_SSE untouched.
- **bayesian-ridge / ard (speed).** After GRAM_SSE: the one-block Jacobi at 60 sweeps (`x_linear/tops.mojo:507`
  t_jacobi_eig, called `bayes.mojo:361` with max_sweeps 60 and the 1e-9 test) and the one-block row passes
  (`bayes.mojo` t_col_means / t_mean / X'y chains at the fit's start, `_var` on the lead) are what remain.
- **huber (board taxi 90x, team form).** Verified at head: `x_linear/device.mojo:1163` routes every Huber fit
  to `x_linear/huber_grid.mojo` huber_fit_grid unless `MOJOLEARN_X_LINEAR_HUBER_GRID=0`. Its objective is
  three grid kernels (a thread per row `hg_map_kernel`, a thread per (4096-row block, task) `hg_part_kernel`,
  a thread per task folding the blocks `hg_fold_kernel`): no one-block or one-thread step remains. What remains
  is the driver: `huber_fit` runs on a HOST team of one (`huber_grid.mojo:181`, `solo(...)`), so each L-BFGS
  objective evaluation is a theta upload, three launches, two readbacks, a witness check and a synchronize
  (`huber_grid.mojo:98-150`), with the two-loop recursion and the line search on the host. No switch written;
  `kernel-huber-fast-taxi` measures the grid form on the M3 (the board's 168 s was the team form).
- **gpr (3,000 rows).** The host round trips above; also `_download`'s element-by-element appends
  (`gaussian_process/estimator.mojo:551`, bulk only under `-D MOJOLEARN_GP_BULK_DOWNLOAD`) on the n^2 K and L
  lists, which both arms pay once for L.
- **gpc (3,000 rows).** `gaussian_process/classifier.mojo:255` _gpc_fit_binary_device: K is built on the device,
  downloaded and uploaded again (`_gpc_kernel_self` + `_upload`) unless `-D MOJOLEARN_GPC_RESIDENT_K` (an
  existing opt-in define); per Newton iteration the B matrix, potrf_lower, chol_logdet and cho_solve run on the
  device with O(n) host vectors in between (`gpc_weights`, `gpc_newton_rhs`, `gpc_scale`, `gpc_a_vector`,
  `gpc_lml`) and two matvecs that each upload a vector and download one (`_gpc_matvec_dev`): about six
  synchronizes an iteration, all O(n) work; the per-iteration n^3/3 factorization is the cost. No switch;
  baseline only.
- **svr (10,000 fit rows, n_train 20,000).** `svm/impl/smosolver.mojo:760-1000` solve: FAST on Apple already
  runs FAST_SMO_SYNCS (one readback per outer iteration, `host_return_buff`, needed for the stopping test),
  the fused gradient update and the 2048 working set. No per-iteration host step beyond that readback; the
  tier family's `MOJOLEARN_APPLE_FAST_GEMM_PINNED=1` covers the kernel tile GEMMs. Baseline only.
- **kernel-ridge (10,000 rows).** `kernel_methods/estimator.mojo:377` kernel_ridge_fit_host: kernel matrix
  (GEMM + epilogue), add_ridge_diag, potrf_lower (CHOL_FAST_APPLE left-looking) and cho_solve with one
  right-hand side on the device; one upload and one n-vector download. Nothing one-block; the GEMM arm is the
  tier family's. Baseline only.
- **kernel-pca (10,000 rows, 8 components, Lanczos).** `python/mojolearn/_expansion_neighbors.py:716-740`:
  the kernel, colsum, scale_div and kpca_center ops are grid kernels; the top-k solve is `_lanczos_top`
  (`python/mojolearn/_expansion_decomp.py:3008-3064`), a Python loop of per-step kit calls: `QT.extend(q.s)`
  downloads q every step, `_M(QT[:], j + 1, n)` re-uploads the whole j x n basis every step, and the A q,
  Qj w and Qj' c products are N = 1 GEMMs through the kit (M = 10,000, K = 10,000), each a synchronous call.
  A device-resident basis needs a kit primitive the kit does not have (write a row of a resident matrix),
  so no switch here; baselines only. Transform is the fused chain (`x_neighbors/iter_device.mojo:1078`).
- **nystroem (100,000 rows, 256 components).** Fit: the one-block Jacobi above (`kernel_methods/estimator.mojo:746`),
  then the order and clip on the host (q scalars) and the normalization GEMM. Transform: kernel GEMM + epilogue
  and the embedding GEMM, every row on the grid, `components_` and `normalization_` uploaded per call
  (`:1175`, 256 x d and 256 x 256 floats; not the cost at 100,000 rows). The GEMM arm is the tier family's.

## Keep rule

A switch becomes the FAST default when its arm is faster on the M3 and held-out quality stays within FAST's
run-to-run spread (for `BAYES_STATS` the Istella R2 must also move toward scikit-learn's: that is the point of
it); then the env read goes and the arm is the code. `BAYES_STATS` and `BAYES_JACOBI` are independent and the
`-both-` rows measure them together.

## After the 2026-10-02 merge of origin/main

Main rebuilt BayesianRidge on grid kernels (`bayes_yparts_kernel`, `bayes_yvar_parts_kernel`, `bayes_xty_kernel`,
`bayes_eig_kernel`, the yy / step / finish kernels) with the Gram from the moments grid
(`x_linear/moments_grid.mojo`: a block per 16-column tile pair, one serial chain per cell, IDENTICAL bits) and
factored the team fit into `bayes_prep` / `bayes_eig_prep` / `bayes_step` / `bayes_finish`. Main's path wins:
`x_linear/bayes.mojo` is main's plus the `jtol` parameter; the ip[6] team flag is gone.
- `BAYES_STATS` is re-expressed on top: when set (BayesianRidge, unweighted, `bayes_grid` on) the setup unit
  runs `bayes_fast_stats` (chunked 32 x 32 tiles, pairwise partials) for xm, G and X'y instead of the moments
  grid and `bayes_xty_kernel`; the y partials and the eig unit are main's. It still adds the pairwise Gram
  (the Istella quality diagnosis above) and row parallelism at small d; ARD is no longer covered (main's
  ARD Istella is 0.86 s).
- `BAYES_JACOBI` is unchanged in effect; it reaches both the team Jacobi and the grid driver's eig block.
- Light A/B form: `1 2`, no -ident lines, one dataset per switch (istella; taxi after a win). The huber,
  kernel-pca, gpc, svr and kernel-ridge baseline-only lines were dropped (no switch of this lane; the
  board holds their times).
