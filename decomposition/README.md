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

Step 4 runs **on the device** (`checks/jacobi_eigh_device.mojo`), because
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

That is a substitution and it is recorded as one in `NOT_IMPLEMENTED.tsv`. The reason
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

    decomposition/pca_wide_sklearn.py, against scikit-learn 1.9.0
    (RE-RUN 2026-08-23; the four numbers that used to sit here were from an
    earlier round and had drifted, so they are replaced rather than kept):
      n_cols= 64  explained_variance 1.340e-06  ratio 8.756e-08
                  singular_values 3.165e-06
                  worst |1-|dot|| over separated components 3.420e-06  OK
      n_cols=128  explained_variance 1.429e-06  ratio 4.644e-07
                  singular_values 3.410e-06
                  worst |1-|dot|| over separated components 5.714e-06  OK

    and the sign half, which the oracle does not ask because it compares
    |dot|: all 16 separated components agree with sklearn in SIGNED dot, and
    all 192 of sklearn's own components satisfy our stated rule.

    check_sign_flip_rule_and_ties OK: ties broken by LOWEST INDEX and
      separated four ways (across lanes, within one lane, sign-swapped,
      three-way); all-zero and all--0.0 components bit-identical after; an
      all-NaN component left alone; a flipped component's zeros pinned to
      -0.0; the rule is idempotent, bitwise
    check_sign_flip_matches_host_rule OK: device == a fold-free host scan,
      BITWISE, on 7186 cells at n = 4/31/32/33/64, 56 tied columns, at
      least 89 components actually flipped
    check_sign_flip_reaches_the_fit OK: all 48 components of a real pca_fit
      at 48 features satisfy the convention

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

## The SIGN of a component is pinned, and the pin has a tie-break

An eigenvector is defined up to sign, so an implementation that does not PICK
one returns a model whose every component sign is the eigensolver's rounding.
cuML's `pcaFit` is that implementation: it calls no flip at all, and
`PCA.fit()` is the entry a Python user reaches (`pca.pyx:547`). We diverge
deliberately, and `sign_flip_kernel` runs in `eig_and_truncate`, which means
PCA and truncated SVD carry ONE rule rather than two copies of one.

The rule (DEVIATION 525) is three clauses, and the second is the one that is
usually left out:

1. take the entry of largest ABSOLUTE value in the component;
2. **on a tie for that magnitude, the LOWEST INDEX wins**;
3. negate the component iff that entry is `< 0.0`.

Clause 2 is not an invention. `cub::ArgMax` keeps the lower index, cuML's own
thrust loop keeps the first strict improvement (`tsvd.cuh:160`), and
`np.argmax` -- and therefore scikit-learn's `svd_flip` -- returns the first
occurrence. Without it a component holding both `+x` and `-x` at the largest
magnitude has no defined sign and a schedule decides it, which is exactly the
class `IDENTITY_PATHS.md` exists to close.

Clause 3 is `< 0.0` and not a sign-bit test, deliberately: `-0.0 < 0.0` is
FALSE, so a component whose largest-magnitude entry is a zero of either sign
is never flipped and an all-zero component comes back BIT FOR BIT as it
arrived. That is `IDENTITY_PATHS` row 13's hazard -- `-0.0` and `+0.0` compare
equal, so a fold picks between them -- and it cannot decide anything here,
because the magnitude fold's result is consumed only by an `==` that cannot
tell the two zeros apart. An all-NaN component has no maximum at all (a NaN
loses every compare) and is left alone rather than flipped on a sentinel.

**There is no `NUMERIC_IDENTICAL` arm and that is the answer, not an
omission.** Two block collectives compute the rule, and `IDENTITY_PATHS` row
20 records why that shape is dangerous: `block.sum` folds at the hardware
lane width, 32 on Apple and NVIDIA and 64 on AMD. But a fold shape cannot
move a MIN or a MAX -- a sum is non-associative in floating point, a max is a
selection over a total order -- and the two escapes from that order
(`-0.0`/`+0.0`, NaN) are neutralised inside the rule. The flip has also been
unconditional since `b495627`, so gating it to IDENTICAL would move SHIPPED
`NUMERIC_FAST` bits and put the default build's component signs back in the
eigensolver's hands.

Three checks, and they fail differently:

- `check_sign_flip_rule_and_ties` -- planted matrices where the tie EXISTS by
  construction (across lanes at indices 3 and 4, within one lane at 3 and 35,
  sign-swapped, three-way), plus the all-zero, all-`-0.0`, all-NaN and
  single-nonzero-at-63 columns, asserted on the BITS. It refuses to pass
  unless the tie multiplicity is what the fixture claims.
- `check_sign_flip_matches_host_rule` -- the device must equal a fold-free
  host scan BITWISE over 7,186 cells at n = 4/31/32/33/64, with 56 tied
  columns and at least 89 components actually flipped.
- `check_sign_flip_reaches_the_fit` -- the rule holds on a real `pca_fit` at
  48 features. **48 and not 4 for a reason**: each component's sign is
  decided independently, so a 4-component fixture agrees with a no-flip build
  one run in sixteen. Measured: with the flip removed, the 4-column
  `check_pca_fit` still passes and 24 of these 48 components fail.

Reach is by SABOTAGE, four of them, each caught by a different assertion:
inverting the tie-break to highest-index gives `0xc0000000` where
`0x40000000` is required; removing the flip launch fails 24 of 48 components
on the fit path; making clause 3 sensitive to the sign bit turns the
all-`-0.0` column into `0x00000000`; and replacing the index selection with
"whichever lane wrote last" fails the same tie the same way -- identically on
all three runs, which is worth noticing, because a race that is stable on ONE
device is precisely what stays invisible until a second vendor runs.

sklearn is the corroboration and not the design source. Its `_fit_full` calls
`svd_flip(U, Vt, u_based_decision=False)`, which is clauses 1-3 with
`np.argmax`'s first-occurrence tie rule. Measured on the wide fixture: all 16
separated components agree with sklearn in SIGNED dot, not merely in |dot|,
and all 192 of sklearn's own components satisfy our stated rule. One real
difference, in our favour: `np.sign(0.0)` is 0, so sklearn MULTIPLIES a
component whose maximum is zero by zero; ours leaves it untouched.

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
  a no-op then `mu` is zero, if the centered read were a no-op -- the fused
  `x - mu` tile load on the split-K arm (DEVIATION 42), `shift_columns_kernel`
  on the fallback arm -- the centering never happens, and either way a column
  carrying a 1000 offset dominates the covariance and the first component
  swings onto it.

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

## whiten=True is implemented (2026-09-01), and one binding line is OWED

`PCA(whiten=True)` used to raise. It now rescales, in all three numeric
modes, and the arithmetic is pinned at the seam.

**What it is.** Component `c` carries variance `explained_var[c] =
s_c^2 / (n_fit_rows - 1)`, so dividing score column `c` by
`sqrt(explained_var[c])` gives it unit variance. cuML folds that into a COPY
of the components before the projection rather than dividing the output
(`pca.cuh:292-302` forward, `:232-243` inverse; the same two calls at the
26.08 pin are `raft/linalg/detail/pca.cuh:233-243` and `:301-311`), and this
port takes their shape: it is the small matrix, so the pass is
`O(n_components * n_cols)` where scikit-learn's is `O(n_rows * n_components)`,
and the projection kernel does not change at all.

**Five deviations, and the first one is a bug of theirs.** cuML's DENSE
transform scales by `sqrt(n_rows - 1)` where `n_rows` is the row count of the
matrix handed to `transform`, not of the matrix the model was fitted on
(`pca.pyx:770` sets `params.n_rows = _n_rows`, `pca.cuh:293` square-roots it).
So `pca.transform(X[:100])` does not agree with `pca.transform(X)[:100]`.
Their own SPARSE path uses `self.n_samples_` (`pca.pyx:716-718`), and so does
scikit-learn, whose `explained_variance_` is a fit-time quantity. Two of the
three agree and the third disagrees with itself, so DEVIATION 580 takes the
fit's count. 581 gives both directions one skip threshold, where theirs skips
below `1e-10` forward and only at exactly zero inverse and is therefore not
round-trip exact. 582 keeps cuML's skip rather than scikit-learn's clamp to
`eps`, which amplifies a null direction instead of leaving it alone. 583 puts
the scalar on the host in float64, rounded once, which is what
`math_t(sqrt(prms.n_rows - 1))` does in C++. 584 pins the arithmetic:
`identical_mul` for the scalar multiply so no codegen fuses it into the
divide, `identical_div` for the divide, `ftz` on the intermediate and on the
store. **A division is not a tier cost here** -- `identical_div` is bit-inert
on a column that flushes in hardware and exists to align one that does not --
so whitening runs identically in FAST, DETERMINISTIC and IDENTICAL.

**Four gates, and they catch four different things.** `check_whiten_unit_variance`
is the property and is independent of how we spelled the rescale.
`check_whiten_round_trip` is the only one that reaches the inverse direction.
`check_whiten_row_subset_agrees` is DEVIATION 580's gate and its sabotage arm
is cuML's own dense behavior. `check_whiten_edges` drives the kernel on
planted bit patterns -- both signed zeros, both signed subnormals, the
smallest normal, NaN, and a singular value one ulp either side of the
threshold -- against a fold-free host scan. Seven sabotage arms live in
`decomposition/checks/pca_sabotage.mojo` (DEVIATION 585); two of them
(`STD_DIV`, `NO_FTZ`) are RECORDED rather than asserted because they are
expected inert on a column that flushes in hardware, which is the honest
statement rather than an assertion that would pass for the wrong reason.

**RUN OWED.** Nothing here has been compiled or run; this lane does not run
anything. The gate is:

    pixi run mojo run -I . decomposition/checks/pca_check.mojo
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . \
        decomposition/checks/pca_check.mojo
    bash tools/check_linalg_identity.sh

**ONE LINE OWED IN A FILE THIS LANE DOES NOT OWN.**
`decomposition/estimator.mojo` exports `pca_whiten_transform_host` and
`pca_whiten_inverse_transform_host` beside the unwhitened pair, deliberately
ADDITIVE so no existing signature moves. Until
`bindings/_mojolearn_estimators.mojo` wraps and registers them and
`bindings/build_estimators.sh` reruns, `PCA(whiten=True)` raises a
`NotImplementedError` that names exactly that and nothing else; the Python
side tests for it with `hasattr`, so it starts working the moment the binding
is rebuilt and no Python edit is needed then.

## svd_solver='full' and algorithm='randomized': the triage, 2026-09-01

Both stay unimplemented, and the reason is NOT the one that used to be given.

**The old reason was wrong on the facts.** The Python surface said `'full'`
is "a different algorithm" in cuML. It is not: `pca.pyx:394-395` maps BOTH
`'auto'` AND `'full'` to `Solver.COV_EIG_DQ`, so for cuML the two names are
one solver on the covariance matrix. The name means a dense SVD of the data
matrix only in scikit-learn (`_pca.py:539-540` sends it to `_fit_full`, which
calls `scipy.linalg.svd` on the centered `X`), and scikit-learn's is the
signature this class copies, so `'full'` must mean the dense route or mean
nothing.

**The other old reason is no longer admissible.** "cuSOLVER `gesvd` is
closed" is a reason to write it ourselves, not a reason to refuse; this tree
already hand-wrote `potrf` and `trsm` for exactly that reason
(`cholesky/DERIVATION_MAP.tsv` rows 18-19). Both routes were re-assessed
against the pinned upstream on that basis and both are REACHABLE:

- `'full'` is R-SVD, which is what LAPACK does for a tall matrix anyway:
  Householder QR of the centered `X`, then a one-sided Jacobi SVD of the
  `n_cols x n_cols` `R`. The second half is a small in-block kernel close in
  shape to `checks/jacobi_eigh_device.mojo`. Its value over the shipped arm
  is real and worth stating: forming `X^T X` squares the condition number and
  the dense route does not, which is why scikit-learn keeps `'full'` beside
  `'covariance_eigh'`.
- `'randomized'` is easier than it looks, because RAFT's own `randomized_svd`
  has two tails and the second calls NO dense SVD:
  `raft/linalg/detail/rsvd.cuh:243` picks between "QR of B^T" and
  "eigendecompose BB^T", and the second reaches `raft::linalg::eigJacobi`
  (`:364`) -- the solver this lane already substitutes for -- plus
  `raft::matrix::sqrt`. The Gaussian sketch (`raft::random::normal`, `:180`)
  has `core/philox.mojo` and the portable `log`/`sqrt`/`sin` in
  `checks/numerics.mojo` behind it.

**So the honest blocker is one missing primitive, named:** a tall-skinny
Householder QR, bit-identical across three vendors (`raft::linalg::qrGetQ`,
`rsvd.cuh:198, :218, :241`, which is cuSOLVER `geqrf` + `orgqr`). That is a
lane the size of `cholesky/`'s `potrf`, not an afternoon, and it is the same
primitive for both names. Until it exists, accepting either name and running
the covariance arm would be a silent substitution, which is what the refusal
is now for. The message says so and points here.
