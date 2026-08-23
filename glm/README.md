# glm: ordinary least squares, ridge and logistic regression, from cuML and RAFT

Sixth section. **COPY, DO NOT IMPROVE.** Three estimators: `LinearRegression`
(RAFT `lstsqEig`, below), `Ridge` (cuML `ridgeFit`, the `eig` arm; DEVIATION
545) and `LogisticRegression` (cuML `qnFit`, the L-BFGS sigmoid arm;
DEVIATIONS 546-549). The second and third are at the end of this file.

## Status: launched and passing

    check_ols_exact OK: all 8 coefficients recovered within 1% from a
      noiseless planted model
    check_ols_scale_invariant OK: y x5 scaled every coefficient by exactly 5
    check_ols_beats_truth_on_noise OK: fitted residual 85.00028895726304
      against the true model's 85.21995181198805

The third one is the assertion that catches a merely plausible solver. Least
squares minimizes the residual **on the sample in front of it**, so its
residual is at most the true model's, on any sample, always.

## And none of those three is an identity check (DEVIATION 527)

All three are TOLERANCE checks -- 1%, 1%, 1.0001 -- so every one of them
passes on a build whose summation order is a different number on every
vendor. A tolerance cannot see a summation order. `glm/mojo_only/
ols_check.mojo`'s second half is the bitwise and discrete set:

    check_ols_arms_are_pinned          step 1 enters the PINNED Gram kernel
                                       at OLS's own shapes, resolved from
                                       COLUMN_BIT_IDENTICAL and not from
                                       the device's column
    check_ols_refuses_over_capacity    n_cols past the pinned kernel's
                                       capacity RAISES under IDENTICAL,
                                       still fits under FAST
    check_ols_is_launch_invariant      THE HEADLINE: the fitted coefficient
                                       BYTES do not move across elem_tpb
                                       256/64, 0/37 floats of buffer
                                       padding, or two scratch poisons
    check_ols_host_surface_takes_the_guard
                                       the Python-facing entry goes through
                                       olsFit's dispatch (see below)
    check_ols_rank_guard_is_absolute   the same design at two scales has two
                                       RANKS -- measured, not argued
    check_ols_card_hashes_raw_bytes    390 of 200,000 adjacent Float32 pairs
                                       print the same decimal text, so a
                                       card built on String() is blind to a
                                       one-ulp move
    check_ols_card_is_emitted          the certificate still emits its 11
                                       named stages and matches its control

Run both arms: `pixi run check-linalg-identity`, or

    tools/with_build_lock.sh     pixi run mojo run -I . glm/ols_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . glm/ols_main.mojo

Every printed line carries the mode the binary COMPILED IN.

## The certificate

    MOJOLEARN_IDENTITY_TRACE=/tmp/mac.card \
    MOJOLEARN_OLS_CARD_CONTROL=/tmp/mac.control.card \
        pixi run mojo run -I . glm/ols_trace_main.mojo

    python3 tools/identity_trace_diff.py /tmp/mac.card /tmp/amd.card

Eleven stages -- the two inputs, the six steps, the eigensolver's `info`
and the RANK -- so a cross-vendor difference has an address instead of a
verdict. Two of the eleven are not floats and are read first: `info` slot 2
is the Jacobi SWEEP COUNT and `ols.step4.rank` is how many directions the
pseudo-inverse keeps. A card that first differs at either of those says the
two machines disagreed about how much work to do, or about how many
directions the model HAS, which is a bigger thing than a rounding.

The fixture is assembled FROM BITS and performs no floating-point operation
at all, because a host `target += v * w[k]` chain is a contraction decision
exactly like a device one -- IDENTITY_PATHS row 18's class, where
cross-vendor is cross-HOST. See `glm/mojo_only/ols_trace.mojo`.

## Two defects DEVIATION 527's audit found

**The host surface bypassed the dispatch guard.** `glm/estimator.mojo::
ols_fit_host` -- the entry `bindings/_mojolearn_estimators.mojo` and
therefore `mojolearn.LinearRegression` reach -- called `lstsq_eig` directly,
around `ols.cuh:112-113`. `glm/ported/glm/ols.mojo` exists precisely because
that is not safe, and its docstring recorded the bypass as closed; it was
closed for the Mojo callers only. A Python user handing in a wide design or
a single column got a plausible vector out of a singular `A^T A` and no
error. Fixed, and gated by `check_ols_host_surface_takes_the_guard`.
The shipped `_mojolearn_estimators.so` was rebuilt with the fix on
2026-08-23 (the DEVIATION 517 commit, which also routes `ols_fit_host`
through `ols_fit_traced` so `mojolearn.LinearRegression().fit` leaves the
same `ols.step*` card the Mojo driver does; `tools/e2u_matrix_fit.py`
reads it).

**`DivideByNonZero`'s threshold is absolute.** `1e-10` is compared against
an eigenvalue of `A^T A`, which scales with the SQUARE of the data and with
`n_rows`. Measured: the same 4096 x 8 design has rank 8 at scale 1 and rank
**0** at scale 2^-24 -- the same design in different units returns the zero
model. It is the identical defect `decomposition/mojo_only/
jacobi_eigh_device.mojo` DEVIATION BLOCK 1 fixed one step upstream, where
the convergence test was also absolute on a quantity that scales with the
square of the data, and where the fix was to make it relative. NOT changed
here: `glm/ported/` is COPY-DO-NOT-IMPROVE and moving the threshold moves
shipped `NUMERIC_FAST` bits on every design near the boundary. Recorded,
gated by `check_ols_rank_guard_is_absolute`, and handed up.

## It cost almost nothing, and the reason is structural

`lstsqEig` is six steps and only the first two touch rows:

    covA = A^T A     -> core/column_stats.mojo covariance_kernel, scale 1
    Ab   = A^T b     -> core/column_stats.mojo xty_kernel            NEW
    Q S Q* = eig     -> decomposition/mojo_only jacobi_eigh_kernel
    QS   = Q invS    -> divide_columns_by_nonzero_kernel             NEW
    covA = QS Q^T    -> core/gemm.mojo
    w    = covA Ab   -> core/gemm.mojo with n = 1

Two new kernels. Everything else was built for PCA.

## We port their NON-DEFAULT solver, deliberately

`olsFit` defaults to `algo = 0`, a one-sided Jacobi SVD. We port `algo = 1`.

Forming `A^T A` SQUARES the condition number, so this route loses roughly
twice the digits an SVD route would on an ill-conditioned design.
`DivideByNonZero` is the guard: a direction the data barely constrains
appears as a near-zero eigenvalue and is DROPPED rather than divided by,
making the inverse a pseudo-inverse.

scikit-learn's `LinearRegression` uses LAPACK `gelsd`, also an SVD route, so
this difference is real and belongs in any accuracy comparison rather than
being discovered during one.

## Ridge (DEVIATION 545): cuML's `eig` solver is an SVD, not "OLS plus alpha"

`cuml/cpp/src/glm/ridge.cuh::ridgeFit` -> `glm/ported/glm/ridge.mojo`, with
`raft/linalg/detail/svd.cuh::svdEig` -> `glm/ported/linalg/detail/svd.mojo`
and the RAFT matrix primitives it calls -> `glm/ported/matrix/math.mojo`.
`glm/UNPORTED.tsv` used to say ridge "is lstsqEig with alpha added to the
eigenvalues before inverting. Cheap once wanted." That sentence is deleted:
their `ridgeEig` is

    svdEig(A): covA = A^T A; eig -> V, S (descending); S = sqrt(S);
               U = A V; U /= S per column (|S| < 1e-10 left AS IS)
    ridgeSolve: setSmallValuesZero(S, 1e-10); S_nnz = S^2 + alpha;
                S = S / S_nnz (|S_nnz| < 1e-10 -> 0); V *= S per column;
                S_nnz = U^T b; w = V S_nnz

which is algebraically `(A^T A + alpha I)^-1 A^T b` and is a different
PROGRAM from that closed form, with an extra `A V` pass and two absolute
thresholds on a SINGULAR VALUE (the `OLS_NONZERO_THRESH` deviation again,
carried and gated). The Gram, the Jacobi and `U^T b` are rows 27, 31 and 29
unchanged; `U = A V` and `w = V S_nnz` are row 28's pinned products; the
elementwise RAFT ops are one thread per cell.

    pixi run mojo run -I . glm/mojo_only/ridge_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . glm/mojo_only/ridge_check.mojo

    check_ridge_matches_closed_form    device == float64 Gaussian elimination
                                       at alpha 0/1/100 to 1e-3
    check_ridge_alpha_reaches          ||w|| decreases 0 -> 1 -> 100, bits move
    check_ridge_dispatch_guard         n_cols==1, algo 0, fit_intercept,
                                       normalize, alpha<0 RAISE by name
    check_ridge_device_equals_host     IDENTICAL: U (8192 cells) and w bit for
                                       bit against the host replay of every
                                       kernel this lane added; FAST: a report
    check_ridge_run_twice_identical    IDENTICAL asserts, FAST reports
    check_ridge_card_is_emitted        14 stages (`ridge.input.A/b/alpha`,
                                       `ridge.svd.covA/eigvals/info/order/S/V/U`,
                                       `ridge.solve.S_over/nnz/Utb`, `ridge.coef`;
                                       `order` and `nnz` are the INTEGER
                                       stages), run-to-run control

The Python surface `mojolearn.Ridge(alpha, solver='auto'|'eig',
fit_intercept, normalize=False)` centers on the host exactly as
`LinearRegression` does (DEVIATION 517's note applies verbatim); `'svd'`,
`'cd'`, `normalize`, `sample_weight` and one column are refused by name.
Four sabotages are recorded in `glm/mojo_only/ridge_check.mojo`'s header.

## Logistic regression (DEVIATIONS 546-549): cuML's QN solver, the L-BFGS sigmoid arm

cuML's `cpp/src/glm/qn/` is an L-BFGS / OWL-QN solver with a line search and
an objective class hierarchy, and the section's earlier note -- that porting
it "is a project, not a follow-on" and that its GPU upside is the weakest in
the repository (each iteration is `X w` and `X^T r`, matrix-VECTOR,
bandwidth-bound, strictly serial with a host line search between) -- stands
as written and is why it was ported LAST. It is ported now, one file per
theirs:

    cuml/cpp/src/glm/qn/qn.cuh             -> glm/ported/glm/qn/qn.mojo
    cuml/cpp/src/glm/qn/glm_base.cuh       -> glm/ported/glm/qn/glm_base.mojo
    cuml/cpp/src/glm/qn/glm_logistic.cuh   -> glm/ported/glm/qn/glm_logistic.mojo
    cuml/cpp/src/glm/qn/glm_regularizer.cuh-> glm/ported/glm/qn/glm_regularizer.mojo
    cuml/cpp/src/glm/qn/qn_solvers.cuh     -> glm/ported/glm/qn/qn_solvers.mojo
    cuml/cpp/src/glm/qn/qn_linesearch.cuh  -> glm/ported/glm/qn/qn_linesearch.mojo
    cuml/cpp/src/glm/qn/qn_util.cuh        -> glm/ported/glm/qn/qn_util.mojo
    cuml/cpp/src/glm/qn/simple_mat/dense.hpp -> glm/ported/glm/qn/simple_mat/dense.mojo
    cuml/cpp/include/cuml/linear_model/qn.h  -> glm/ported/linear_model/qn.mojo

NOT PORTED, each refused by name at the layer it would enter:
`glm_softmax.cuh` (multinomial; > 2 classes), `min_owlqn` (penalty 'l1' /
'elasticnet'), `add_sample_weights` (sample_weight, class_weight),
`glm_linear.cuh` / `glm_svm.cuh` (the other losses), `qnFitSparse`.

**THEIR REDUCTIONS ARE FLOAT ATOMICS, AND THAT IS THE IDENTITY STORY
(DEVIATION 547).** `dot`, `nrm2`, the loss sum and the regularizer sum all go
through `raft::linalg::mapThenSumReduce`, whose kernel folds each block with
CUB and then `atomicAdd`s the block partials (`map_then_reduce.cuh:33-38`).
The number the Armijo test compares, the `ys <= eps * yy` skipping test, the
convergence test -- every branch in the solver -- is a branch on a float
whose last bit depends on which block arrived first. cuML ships one backend
and accepts that. Here every one of them is REPLACED by one block of
`STATS_TPB` strided partials through `pinned_block_sum`; the gradient
`X^T dZ` is row 29's `xty_kernel`; `X w` is row 28's gemv; `exp`/`log` in
the loss go through `identical_exp`/`identical_log` (row 12); the line
search's one host multiply-add `fx_init + step * dg_test` is an fma; and the
bias gradient's `mean` is `sum * (1/N)` as `raft::stats::mean` spells it
(a multiply, which is why `column_mean_kernel`'s `s / n` is not reused).
Under IDENTICAL the accepted steps, the L-BFGS history, the ITERATION COUNT
and the coefficients are a function of the inputs alone, and the card records
them: `qn.init.loss/grad`, per iteration `qn.iterNNNN.loss/grad/ls` (`ls` =
the line-search return code and its evaluation count, two integers),
`qn.coef`, `qn.n_iter`, `qn.retcode`. Measured on this Mac: FAST and
IDENTICAL take **32 vs 35 iterations** on the E2U `logreg_c100` cell -- the
fold shape moves the count, which is exactly the thing the certificate has to
carry.

    pixi run mojo run -I . glm/mojo_only/logistic_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . glm/mojo_only/logistic_check.mojo

    check_logistic_planted_separable   planted signs recovered, accuracy > 0.97
    check_logistic_is_a_minimizer      float64 host gradient Linf at the device
                                       solution <= tol * max(fx, tol) (the
                                       solver's own rule), below 8 perturbed
                                       objectives -- the oracle, no sklearn
    check_logistic_c_reaches           C 1 / 100 / none: ||w|| grows, bits move;
                                       fit_intercept=False has D parameters
    check_logistic_refuses_by_name     l1, softmax, 3 classes, sample_weight,
                                       an unknown loss
    check_logistic_device_equals_host  ONE objective evaluation: loss and every
                                       gradient entry bit for bit (IDENTICAL)
    check_logistic_run_twice_identical coefficients AND n_iter (IDENTICAL)
    check_logistic_card_is_emitted     2 + 3 n_iter + 3 stages, control

Three sabotages are recorded in `glm/mojo_only/logistic_check.mojo`'s header;
the first -- the gradient SIGN flipped -- is the one to read: the solver
declares SUCCESS at the zero model in 10 iterations, and only the oracle sees
it.

The Python surface is `mojolearn.LogisticRegression(penalty='l2'|None, C,
tol, fit_intercept, max_iter, linesearch_max_iter, solver='qn')` with
`coef_ (1, D)`, `intercept_ (1,)`, `classes_`, `n_iter_`, `objective_`,
`retcode_`, `predict`, `predict_proba` (float64, the sigmoid through
`identical_exp64` on the host -- DEVIATION 549; cuML's cupy computes it in
float32 on the device), `decision_function`. Its docstring carries the
HONORED / REFUSED table.
