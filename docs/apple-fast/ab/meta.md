# meta: calibrated and multioutput-reg (lane/apple-fast-meta)

Both switches are FAST + Apple only, default OFF, `-D` defines; IDENTICAL compiles main's code unchanged.

## MOJOLEARN_CALIB_GNB_FOLDS (calibrated, taxi 2.4x)

CalibratedClassifierCV(GaussianNB, method="sigmoid", cv=int, ensemble=True), no priors, no sample_weight.
Reference route: Python-list StratifiedKFold over every row, a host gather of each fold's rows, one GaussianNB
program per fold (X up per fold) plus one per held-out score, Platt's sigmoid on the host in float64, and at
predict one program per member (X up per member).
Switch: one x_prep program per fit and one per predict (x_prep/calib.mojo ops 136-150): device fold assignment
(per-class ranks, sklearn `_make_test_folds`'s allocation), per (fold, class) `class_stats` with pseudo-classes
fold*K+class, leave-one-fold-out Chan merge for each fold's train statistics, per-fold epsilon / priors /
constants, every row scored by the model that left it out, Platt per (fold, column) as 40 unrolled Newton
iterations (blocked partials over the fold's rows, 16 step lengths evaluated in one pass, Armijo's first
accepted; a stopped problem's stages return at once), and at predict every member's scores, sigmoids and the
average in one program. The members are GaussianNB objects filled from the program's words.
Files: x_prep/calib.mojo (new), x_prep/units.mojo (N_OPS 151 under CALIB_FOLDS, else 136),
bindings/_mojolearn_x_prep.mojo (x_prep_calib_folds under the guard), python/mojolearn/_expansion_prep.py (op ids),
python/mojolearn/_expansion_trees.py (`_fit_fast`, `_predict_proba_fast`).
Binding for afc_ab_def.sh: x_prep. Bits: float32 Platt (the host is float64); quality to be checked against the
reference arm's log loss / accuracy.
Risky compile sites: x_prep/calib.mojo `abs(Float32)` in cal_platt_step_unit; `is_defined` / `has_apple_gpu_accelerator`
imports; the `comptime if CALIB_FOLDS:` block nested in run_unit; `Python.list()` + append in the binding.

## MOJOLEARN_MULTIOUT_RIDGE (multioutput-reg, taxi 2.1x)

MultiOutputRegressor(Ridge) (plain Ridge, no normalize, no sample_weight).
Reference route: one Ridge.fit per target (X centered and uploaded per target, the A^T A eigendecomposition
redone per target, a host gather of each Y column), one predict per target plus a host scatter.
Switch: glm/impl/ridge_multi.mojo `ridge_fit_multi_host` (X up once, svd_eig once, ridgeSolve's S and V
transforms once, per target one `xty_launch` and one `gemv_n`; Y's column means and centering as one-thread-per-row
kernels) and `ridge_predict_multi_host` (one kernel, one thread per row, every target, row-major (n, m) float32).
`estimators_` holds m Ridge objects filled from the words (coef_, intercept_), so the per-target API reads as before.
Files: glm/impl/ridge_multi.mojo (new), bindings/_mojolearn_estimators.mojo (entries under MULTIOUT_RIDGE),
python/mojolearn/_expansion_trees.py (`_fit_ridge_multi`, predict).
Binding for afc_ab_def.sh: estimators. Bits: predict's dot product is a plain float32 chain per row (the reference
`gemv_n` has its own fold); predictions come back float32 where the reference returns float64.
Risky compile sites: ridge_multi.mojo's kernel pointer arguments (`buf.unsafe_ptr()` passed to
`MutPointer[Float32, MutAnyOrigin]` parameters, the idiom core/xtdz_coalesced.mojo uses for column_mean_kernel);
`IdentityTrace.disabled()` as a `mut trace` argument.
