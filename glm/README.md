# glm: ordinary least squares, from RAFT

Sixth section. **COPY, DO NOT IMPROVE.**

## Status: launched and passing

    check_ols_exact OK: all 8 coefficients recovered within 1% from a
      noiseless planted model
    check_ols_scale_invariant OK: y x5 scaled every coefficient by exactly 5
    check_ols_beats_truth_on_noise OK: fitted residual 85.0003 against the
      true model's 85.2200

The third one is the assertion that catches a merely plausible solver. Least
squares minimizes the residual **on the sample in front of it**, so its
residual is at most the true model's, on any sample, always.

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
