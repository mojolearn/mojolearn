# decomposition: PCA, from RAFT

Fourth section. Same rule: **COPY, DO NOT IMPROVE.**

cuML's `cpp/src/pca/pca.cu` is a thin dispatch layer over
`raft::linalg::pca`, so the algorithm is RAFT's and that is what is mirrored.

## RAFT's PCA is not randomized SVD, and that is why it suits a GPU

I expected randomized SVD and found covariance eigendecomposition.
`pca_fit` is six steps, and only four of them touch rows at all:

    1  column means                       O(rows)
    2  center the input IN PLACE          O(rows)
    3  covariance Xc^T Xc / (n_rows - 1)  O(rows * cols^2)
    4  eigendecompose, take top k         O(cols^3)
    5  singular_vals = sqrt(var * (n-1))  O(k)
    6  RESTORE the input, add the mean    O(rows)

Everything after step 3 works on an `n_cols x n_cols` matrix. **That is the
structural reason this algorithm is a good GPU fit**: one bandwidth-bound
pass, one big arithmetic-dense product, then a small dense problem that does
not care where it runs.

Step 4 runs on the host here (`mojo_only/jacobi_eigh.mojo`). That is inside
`HOST_AND_DEVICE.md`'s rule, not an exception to it: the rule forbids host
work that is O(rows), and this is `n_cols^3` on a matrix that for a
million-row, fifty-feature fit is 50x50. **Jacobi is also THEIR algorithm**,
not our substitute: `cal_eig` branches on `prms.algorithm` and
`solver::COV_EIG_JACOBI` selects exactly it.

Step 6 is the one that is easy to drop and invisible when dropped, so it has
its own check.

## Status

**Launched and passing.**

    check_pca_fit OK: 4/4 eigenvalues within 6% of planted [100, 50, 20, 5],
      4/4 components aligned with the planted rotation and orthogonal to the
      others, sign convention applied, ratios sum to 1.0
    check_pca_invariants OK: x3 scaling moved variances by exactly 9 and moved
      no direction; a +1000 column shift moved nothing at all
    check_input_restored OK: worst element moved 9.5e-07 after a full fit

Never benchmarked.

## The checks are INVARIANTS, which is stronger than a fixture

The fixture has an analytic answer: `X = Z A` with independent latent columns
of known variance and a known orthogonal `A`, so `cov(X) = A^T diag(v) A` and
both the eigenvalues and the eigenvectors are known in closed form. Alignment
is checked as a full dot-product matrix so a permutation cannot pass.

On top of that, two invariants that hold for correct PCA and fail for most
broken ones:

- **Scale by 3.** Variances must move by exactly 9; directions must not move
  at all. Fails if the `n_rows - 1` normalization is wrong.
- **Shift a column by 1000.** Nothing may change, because PCA centers. This
  is the reach evidence for the centering path: if `column_mean_kernel` were
  a no-op then `mu` is zero, if `shift_columns_kernel` were a no-op the
  centering never happens, and either way a column carrying a 1000 offset
  dominates the covariance and the first component swings onto it.

## Three failures this section paid for

**1. `** 0.5` instead of `sqrt`.** Replaced everywhere; see `PORTING.md 22`.

**2. Transposed tile indices in the covariance kernel.** The symptom was not
a wrong answer. It was the Jacobi solver **never converging**, because a
transposed contraction produces a plausible but NON-SYMMETRIC matrix and
Jacobi only converges on symmetric input. `PORTING.md 23`.

**3. The fixture's latent columns were not independent.** They were
`(row * C + k * D) % prime`, an arithmetic progression differing only by a
constant offset per column, so the covariance was not `diag(v)` and the
planted eigenvalues were not the true ones. The check failed at 145 against a
planted 100 and the port was right. A fixture needs a real mixer.
