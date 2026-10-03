# lane/apple-fast-gram: the shared FAST grid Gram (PLAN.md item 4; PLAN-classical.md 4 and 9)

Written without a Mojo toolchain (cloud peer); the first M3 build is the compile check
(bindings x_linear and x_prep, FAST). Every switch defaults OFF; IDENTICAL compiles the old code.

New file `x_linear/fast_gram.mojo`: enetcv_fast's `ef_sums_kernel` / `ef_gram_kernel` /
`ef_gram_red_kernel` generalised to one row range (no folds, so no parallel-axis move): column sums
per chunk of 8192 rows (2048 below 8 tile pairs, so taxi's 16 features still fill hundreds of
blocks), the means (zeros without an intercept), the Gram of [X, Y] centered at them as 32 x 32
tiles per chunk through shared memory, a sum over the chunks into G (both triangles) and X'Y.
Options: uncentered, a class mask (rows whose label is not `cls` contribute nothing), a divisor.
Entries: `fast_gram_into` (x_linear: xm, ym, G, X'Y into the fit's own fw words; waits, so its
scratch dies inside) and `fast_sym_gram_into` (x_prep: one symmetric block, caller's scratch,
enqueue only). No witness (as `xg_gram_kernel`); enetcv_fast.mojo is untouched.

| switch | kind | site | what it changes under FAST on Apple |
|---|---|---|---|
| `-D MOJOLEARN_X_LINEAR_LARS_FAST_GRAM` | define (`XL_LARS_FAST_GRAM`, x_linear binding) | `x_linear/device.mojo` fit_device (ip[4] = 2, main's `pre_all` flag); `x_linear/lars.mojo` lars_fit reads it | Lars / LassoLars: means, centered Gram, X'y and the y mean from `fast_gram_into` into fw before `fit_kernel` (the words main's moments grid `mg_means_kernel` / `mg_cross_kernel` fills); the team skips `t_col_means`, `t_mean`, `t_centered_gram`, `t_centered_xty`; neither the moments grid nor the sliced `xg_gram_kernel` is launched |
| `-D MOJOLEARN_X_LINEAR_RIDGE_FAST_GRAM` | define (`XL_RIDGE_FAST_GRAM`, x_linear binding) | `x_linear/device.mojo` (ridge ip[4] = 1, main's `pre` flag, `ridge_pre` off; the k-fold unit A); `x_linear/ridge.mojo` `_ridge_fit_team` reads it (main's code) | RidgeClassifier (and LOO RidgeCV): xm, ym (T targets), G, X'Y from the grid, the team skips its cell chains (unweighted fits only). k-fold RidgeCV (the board's cv=5): each fold's means, Gram and X'y from `fast_gram_into` instead of `kf_means_kernel` + `kf_cells_kernel` |
| `-D MOJOLEARN_X_PREP_CLASS_COV_GRID` | define (x_prep binding) | `x_prep/device.mojo` (ops 40 `qda_cov` and 13 `matmul` when Gram-shaped, `_matmul_is_gram`) | QDA: each class's covariance (divisor CNT[k]) as the masked grid Gram, one launch pair a class, into the unit's COV words. LDA (solver svd): the Gram `matmul` Z2'Z2 as the uncentered grid Gram into C. The other stages are the units' |

## Causes (what was slow)
- lars / lasso-lars taxi 5.3x: `xg_gram_kernel` (`x_linear/device.mojo`) is one thread per upper cell
  walking every row: d(d+1)/2 = 136 threads, ONE block at 16 features; then the team's own means and
  X'y passes (`x_linear/lars.mojo` lars_fit, `x_linear/tops.mojo` t_col_means / t_centered_xty) on one block.
- ridge-clf taxi 5.3x: `x_linear/ridge.mojo` `_ridge_fit_team` builds means, `t_centered_gram` and
  the X'Y chains one thread per cell on ONE block of 256 threads. ridge-cv (cv=5): `kf_cells_kernel`
  is the same one-thread-per-cell chain per fold on the grid (136 threads at taxi).
- lda-clf Istella 5.1x: `x_prep/prims.mojo` `matmul_unit` as Z2'Z2 (python `_expansion_prep.py`
  LinearDiscriminantAnalysis.fit, the `matmul` stage): 48,400 threads each walking a million rows.
- qda Istella 2.6x: `naive_bayes/da.mojo` `qda_cov_unit`: K d^2 threads each walking every row with a
  class test (two column loads a row a thread; the tile shares them through shared memory).

## Keep rule
A switch becomes the FAST default when its arm is faster on the M3 and held-out quality stays within
FAST's run-to-run spread (light A/Bs: old FAST vs new FAST, `1 2`, no -ident lines; the IDENTICAL
hash check is the manager's); then the define goes (the arm is the code), the old team passes stay as the IDENTICAL /
other-vendor code. One dataset per switch first (taxi for the x_linear switches, where 16 features
make main's moments grid ONE block per launch; istella for x_prep's); the second after a win.

## After the 2026-10-02 merge of origin/main
Main's lane/neural-pass120 moments grid (`x_linear/moments_grid.mojo`: a block per 16-column tile pair
of [X | Y], one thread per cell folding every row, IDENTICAL bits) now fills the same fw words for
Lars (ip[4] == 2) and Ridge (ip[4] == 1) by default. The FAST grid Gram differs by splitting the rows
into chunks (hundreds of blocks at taxi's 16 features against the moments grid's one) and summing the
partials; its launches run inside main's witnessed setup unit (x_linear/witness.mojo) but are not
themselves witnessed (`fast_gram_into` waits for them). lars.mojo and ridge.mojo carry no change of
this lane any more (main's flags serve).

## Not done here (notes for the owner)
- `x_prep` `eigh` (op 18, `x_prep/eigh.mojo`): one thread per matrix (cyclic Jacobi at 220 x 220 for
  LDA, K of them for QDA) remains after the Gram; it is PLAN.md item 7's shape (PCA one-block Jacobi).
- Weighted ridge (`sample_weight`) and LOO RidgeCV's per-row solves stay on the team.
- The grid Gram carries no completion witness (x_linear/witness.mojo); add it if the M3 shows cut
  launches under contention, as enetcv_fast does.
- Compile risks to watch on the first M3 build: `fast_gram_into` / `fast_sym_gram_into` take
  `mut ctx: DeviceContext` (as `enetcv_fast`); device pointers cross as
  `FP(unsafe_from_address=Int(buf.unsafe_ptr()))`; x_prep imports `x_linear.fast_gram` (the build's
  `-I .` resolves it, as `naive_bayes.da` does).

Switches are build-time defines (COMMON.md: no env read on the fit path); the A/B lines use
`tools/afc_ab_def.sh` (copied from lane/apple-fast-tier) with the x_linear and x_prep bindings.
