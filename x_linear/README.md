# x_linear: the linear expansion lane

SGDClassifier, SGDRegressor, Perceptron, PassiveAggressiveClassifier,
PassiveAggressiveRegressor, SGDOneClassSVM, PoissonRegressor, GammaRegressor,
TweedieRegressor, HuberRegressor, BayesianRidge, ARDRegression, Lars,
LassoLars, QuantileRegressor, RidgeClassifier, RidgeCV, LassoCV,
ElasticNetCV, LogisticRegressionCV and IsotonicRegression
(`python/mojolearn/_expansion_linear.py`). Each module's header names its
scikit-learn reference and every difference from it; `NOT_IMPLEMENTED.tsv`
lists what is refused or replaced.

## One source, two columns

Every fit is one plain function over pointers (`dispatch.mojo::fit_dispatch`).
The CPU host binding (`bindings/_mojolearn_x_linear_host.mojo`) calls it
directly; the GPU binding (`bindings/_mojolearn_x_linear.mojo`) calls the same
function from a one-thread kernel (`device.mojo`). Scoring
(`dispatch.mojo::decision_one`) runs one device thread per (row, output).
IDENTICAL is bought by the sequence of operations alone; the parallel fit
schedules are the speed work that follows.

## The seams (IDENTITY_PATHS rows 100-109)

| DEVIATION | seam | move |
|---|---|---|
| 5000 | every reduction (`ops.dot`, `row_dot`, the Gram and X'y folds) | PIN: j ascending, one `identical_mul_add` per term |
| 5001 | every operand and result | PIN: `ftz` (a subnormal is its signed zero) |
| 5002 | Cholesky (`ops.cholesky`, `chol_solve`) | PIN: column j ascending, inner sums k ascending, products rounded alone |
| 5003 | the symmetric eigensolver (`ops.jacobi_eig`, BayesianRidge) | REPLACE their SVD: cyclic Jacobi, p then q ascending, Rutishauser's rotation |
| 5004 | the SGD shuffle (`ops.shuffle`) | REPLACE numpy's MT19937: splitmix64, Fisher-Yates, j = draw mod (i + 1) |
| 5005 | every argmax/argmin (LARS's feature, the class argmax, the CV choices) | PIN: the lowest index wins an exact tie |
| 5006 | isotonic out-of-bounds 'nan' | PIN: the constant word 0x7FC00000, never a computed NaN (Clause B) |
| 5007 | a score (`decision_one`) | PIN: the fold of x.w first, the intercept added last |
| 5008 | exp and log in the fits (`ops.fexp`, `flog`) | PIN: the portable spellings of checks/numerics.mojo |
| 5009 | the CV alpha grid (`cd.alpha_grid_value`) | PIN: alpha_max * exp(frac * log eps) |

Check: `tools/with_identical_mode.sh pixi run mojo run -I . x_linear/checks/seams_check.mojo`
(oracle `x_linear/checks/seams_oracle.mojo`; each seam's fixture must separate
its pinned spelling from the unpinned one first). Sabotage arms:
`x_linear/checks/sabotage/seam_50NN_*.patch`, listed in
`tools/identity_lanes/linear.checks` and run by
`tools/algos_lane_check.sh <lanes> --pass 2`. Card:
`$MOJOLEARN_XLINEAR_CARD`, compared across boxes with
`tools/identity_trace_diff.py`.
