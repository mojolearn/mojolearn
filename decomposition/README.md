# decomposition: PCA, from RAFT

Fourth section. Same rule: **COPY, DO NOT IMPROVE.**

cuML's `cpp/src/pca/pca.cu` is a thin dispatch layer over
`cuml/cpp/src/pca/pca.cuh`, which is where the algorithm lives. It calls RAFT
only for `stats::mean`, `stats::cov`, `matrix::*` and `linalg::eig*`.

**CORRECTION, 2026-08-19.** Until this round every file in this section cited
`raft/linalg/detail/pca.cuh` and `raft/linalg/detail/tsvd.cuh`. **Neither has
ever existed** -- `git log --all` on those paths in the RAFT checkout returns
nothing -- so every line number quoted from them was invented, along with a
`sign_flip_components` function and a `flip_signs_based_on_U` switch that
appear nowhere in RAFT or cuML. The section was written from a recollection of
their code. All citations below are now from the files on disk at cuML
`00094f7` and RAFT `661a3b8`.

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

**Jacobi is NOT the arm their dispatch takes, and calling it "their algorithm"
was wrong.** `calEig` (`cuml/cpp/src/tsvd/tsvd.cuh:99`) branches on
`prms.algorithm`, and that field **defaults to `solver::COV_EIG_DQ`**
(`cuml/cpp/include/cuml/decomposition/params.hpp:53`) -- divide and conquer,
`raft::linalg::eigDC`, cuSOLVER `syevd`. cuML's Python layer maps both
`svd_solver='auto'` and `'full'` onto it and reaches `COV_EIG_JACOBI` only for
`svd_solver='jacobi'` (`pca.pyx:392-404`). So we ship their OPT-IN arm.

That is a substitution and it is recorded as one in `UNPORTED.tsv`. The reason
it is defensible: both arms end inside cuSOLVER, which is closed and has no
source to transliterate, and MAX ships no symmetric eigensolver or SVD at all
(`linalg` is `matmul`, `bmm`, `gemv`, `transpose`, `qr_factorization`,
`matrix_band_part` and nothing else). Something had to be written, and between
their two named algorithms Jacobi is the one that can be diffed against a host
reference rotation by rotation.

**Honest caveat on that kernel.** It is device-RESIDENT but its `(p, q)` pair
loop is serial, with the O(n) updates inside each rotation parallel. cuSOLVER
is the round-robin tournament version, which does `n/2` disjoint rotations at
once and is what actually leverages the GPU. Ours is the right shape for
being diffed against the host reference rotation for rotation, and it is the
wrong shape for speed. At `n_cols = 4` this is irrelevant; at 128 it is 8128
serial pairs per sweep and at 256 it is 32640, and it will show. Named here
rather than discovered later.

## THE 32-FEATURE CAP, AND WHY THIS SECTION NOW HAS A WIDE CHECK

The device eigensolver held the matrix AND the accumulated basis in
THREADGROUP memory, which forced `JACOBI_MAX_N = 32`. A PCA, truncated SVD or
OLS at 33 or more features returned, with no error, something that was not an
eigendecomposition of anything. That is not a property of Jacobi and not a
property of Metal; it was a choice to keep the whole problem on chip, and it
costs 8 KB at 32 features, all of Metal's 32 KB at 64 and an impossible 128 KB
at 128.

**Nothing in this section could see it, because every check ran at
`PCA_COLS = 4`.** A check suite whose widest case is four columns does not
test a PCA, it tests a 4x4 PCA. Both arrays now live in global memory,
`JACOBI_MAX_N` is gone, and there are three new checks whose whole job is to
cross where it sat:

- `check_jacobi_device_sizes` -- `A V = V diag(lambda)` per CELL and
  `V^T V = I` at n = 16, 32, **33**, 64, 128, 256, each cross-checked against
  the host Float64 solver. 33 is the load-bearing one: it is the first size
  the old cap silently broke.
- `check_jacobi_reaches_past_32` -- a +1000 spike planted at index 63 of a
  64x64 matrix must move the largest eigenvalue to ~1000, which it cannot do
  if the kernel never reads index 63.
- `check_pca_wide` / `check_pca_truncation` -- the real fit at 64 and 128
  features, with `decomposition/pca_wide_sklearn.py` re-deriving the same
  fixture under `sklearn.decomposition.PCA`.

The size checks have REACH, proved by sabotage rather than by a passing
digest: re-imposing the cap inside the kernel leaves n = 16 and n = 32 at
byte-identical numbers and makes n = 33 fail at `||V^T V - I|| = 0.61`.

Step 6 is the one that is easy to drop and invisible when dropped, so it has
its own check.

## Truncated SVD comes almost free

`tsvd_fit` is PCA without the centering and without the `n_rows - 1`
division: same product, same `calEig`, same truncation. Their two files
share `calEig` and so do ours, which is why `tsvd.mojo` is thirty lines.

The user-visible difference gets its own check: truncated SVD is NOT
translation invariant and PCA is. Shift a column by 1000 and tsvd's first
component swings onto it while PCA's does not move. Asserting both halves
together is the point, because either alone would pass for an implementation
that centered in the wrong place.

## Status

**Launched and passing, at 4, 64 and 128 features.**

    check_jacobi_device_sizes OK: 16, 32, 33, 64, 128, 256 all satisfy
      A V = V diag(lambda) and V^T V = I. The 32-feature cap is gone.
      worst cell of A V - V diag(lam) / ||A||_F: 3.9e-07 at n=16 rising to
      2.2e-06 at n=256; ||V^T V - I||_max 2.2e-06 rising to 6.2e-05;
      device Float32 vs host Float64 spectrum <= 4.3e-06 through n=128
    check_jacobi_reaches_past_32 OK: 9.598 -> 1001.327 for a +1000 spike at
      index 63, past the old cap
    check_jacobi_scale_invariance OK: A and 1000 A both converge in 7 sweeps
    check_pca_wide OK at 64 and 128 features
    check_pca_truncation OK at 128 features: noise_var 0.98947 = mean of the
      118 discarded eigenvalues, and 0 when nothing is discarded

    decomposition/pca_wide_sklearn.py, against scikit-learn 1.9.0:
      n_cols= 64  explained_variance 1.597e-06  ratio 3.606e-07
                  worst |1-|dot|| over separated components 3.596e-06  OK
      n_cols=128  explained_variance 1.674e-06  ratio 5.480e-07
                  worst |1-|dot|| over separated components 5.993e-06  OK

and at four features:

    check_pca_fit OK: 4/4 eigenvalues within 6% of planted [100, 50, 20, 5],
      4/4 components aligned with the planted rotation and orthogonal to the
      others, sign convention applied, ratios sum to 1.0
    check_pca_invariants OK: x3 scaling moved variances by exactly 9 and moved
      no direction; a +1000 column shift moved nothing at all
    check_input_restored OK: worst element moved 9.5e-07 after a full fit
    check_tsvd_against_pca OK: identical directions on centered data, and a
      +1000 shift moved tsvd's first component to |dot| = 0.83 while PCA's
      was unmoved

Never benchmarked; the orchestrator owns interleaved timing arms.

## The eigensolver's tolerance is now RELATIVE, and non-convergence is reported

Two defects of the same class as the size cap, both fixed:

- the convergence test was `off <= 1e-10` on an ABSOLUTE sum of squares, and
  `off` scales with the square of the data, so the same matrix in different
  units was a different problem. It is now
  `||offdiag||_F <= tol * ||A||_F`, which is cuSOLVER's quantity, with
  `tol = 1e-7` and 15 sweeps -- `raft::linalg::eigJacobi`'s own defaults
  (`raft/linalg/eig.cuh:108-109`) and cuML's Python defaults, in place of a
  hardcoded 80 and 1e-10 that came from nowhere. Measured: n = 16, 32, 33,
  64, 128, 256 converge in 5, 6, 6, 8, 8, 9 sweeps.
- hitting the sweep limit was SILENT. RAFT's Jacobi arm is silent too --
  `detail/eig.cuh:310` fetches `executed_sweeps` and never reads it, and it
  never checks `dev_info` -- but their DEFAULT arm `eigDC` aborts with
  "eigensolver couldn't converge to a solution" (`detail/eig.cuh:149-151`;
  the identical ASSERT at `:79-81` is `eigDC_legacy`'s, which nothing calls), and
  that is the behaviour we follow. The kernel writes a three-slot info buffer
  and `eig_and_truncate` raises on it.

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

**0. A 32-feature cap that no check could see.** Above. The lesson is not
about threadgroup memory, it is that a check suite whose widest case is four
columns cannot see a cap at 32.

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

**4. A predicted eigenvalue that was wrong, not a solver that was.** The first
version of `check_pca_wide` asserted the eight signal eigenvalues equalled
`sd[k]^2 * n_cols / 3` and failed at 787 against a predicted 1194. That
formula holds only if the eight loading vectors are ORTHOGONAL and they are
hashed, so they are not. The check now asserts the SHAPE of the spectrum -- a
rank-8 cliff over a noise floor of 1 -- which is what the fixture actually
plants, and sklearn checks the numbers.
