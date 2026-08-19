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

Step 4 runs **on the device** (`mojo_only/jacobi_eigh_device.mojo`), because
cuSOLVER's `syevj` does, and the standing rule is to mirror their host/device
split rather than to make our own. The first version of this port put it on
the host; that was inside `HOST_AND_DEVICE.md`'s O(rows) rule but was not a
mirror of theirs, and the rule about mirroring wins. The host version
survives as the reference the device one is checked against.

**Jacobi is also THEIR algorithm**, not our substitute: `cal_eig` branches on
`prms.algorithm` and `solver::COV_EIG_JACOBI` selects exactly it.

**Honest caveat on that kernel.** It is device-RESIDENT but its `(p, q)` pair
loop is serial, with the O(n) updates inside each rotation parallel. cuSOLVER
is the round-robin tournament version, which does `n/2` disjoint rotations at
once and is what actually leverages the GPU. Ours is the right shape for
being diffed against the host reference rotation for rotation, and it is the
wrong shape for speed. At `n_cols = 4` this is irrelevant; at `n_cols = 32`
it is 496 serial pairs and it will show. Named here rather than discovered
later.

Step 6 is the one that is easy to drop and invisible when dropped, so it has
its own check.

## Truncated SVD comes almost free

`tsvd_fit` is PCA without the centering and without the `n_rows - 1`
division: same product, same `cal_eig`, same truncation. Their two files
share `cal_eig` and so do ours, which is why `tsvd.mojo` is thirty lines.

The user-visible difference gets its own check: truncated SVD is NOT
translation invariant and PCA is. Shift a column by 1000 and tsvd's first
component swings onto it while PCA's does not move. Asserting both halves
together is the point, because either alone would pass for an implementation
that centered in the wrong place.

## Status

**Launched and passing.**

    check_pca_fit OK: 4/4 eigenvalues within 6% of planted [100, 50, 20, 5],
      4/4 components aligned with the planted rotation and orthogonal to the
      others, sign convention applied, ratios sum to 1.0
    check_pca_invariants OK: x3 scaling moved variances by exactly 9 and moved
      no direction; a +1000 column shift moved nothing at all
    check_input_restored OK: worst element moved 9.5e-07 after a full fit
    check_tsvd_against_pca OK: identical directions on centered data, and a
      +1000 shift moved tsvd's first component to |dot| = 0.83 while PCA's
      was unmoved

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
