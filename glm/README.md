# glm: ordinary least squares, from RAFT

Sixth section. **COPY, DO NOT IMPROVE.**

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

## Logistic regression: why it is not here, and why its GPU case is weaker

cuML does logistic and softmax regression in `cpp/src/glm/qn/`, 2737 lines.
It is not a linear-algebra routine, it is an **LBFGS / OWL-QN solver** with a
line search, an objective/regularizer class hierarchy, and its own dense and
sparse matrix wrappers. Porting it is a project, not a follow-on.

**And the GPU upside is smaller than for anything else in this repository,**
which is worth saying before it gets built rather than after:

- Least squares is dominated by `A^T A`, which is `O(n d^2)` and
  arithmetic-dense. That is matrix-MATRIX work and it is where a GPU wins.
- Each LBFGS iteration is `X w` and `X^T r`, both `O(n d)`. That is
  matrix-VECTOR work, which is bandwidth-bound: it reads the whole dataset to
  do one multiply-add per element.
- And the iterations are strictly serial, with a host-side line search
  between them, so the control-plane cost per iteration lands on top.

That is the same profile as gradient boosting, which this repository already
measured as the weakest GPU case it has. Worth porting for completeness;
not worth porting first if the goal is a number that makes the case.
