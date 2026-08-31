# gaussian_process: cross-vendor bit-identical exact dense GP regression

Opened 2026-08-25. The first CONSUMER of the Cholesky lane, and the reason
that lane names Gaussian processes in its own opening sentence. **DEVIATIONS
1750-1771 are this lane's**; 1772-1779 are reserved and unspent.

**The profile is `mojolearn.identical.gp.fp32.v1`.** It CONTAINS
`mojolearn.identical.cholesky.fp32.v1` and
`mojolearn.identical.gemm.fp32.v1` rather than restating them. Changing any
pinned constant here, the postfix evaluation order, the feature-axis order
or the predictive-variance fold creates a v2; so does a v2 of either
contained profile, because a GP fitted under one is not comparable with a GP
fitted under the other.

## Status

**BUILT AND GATED ON ONE APPLE M4, BOTH MODES, 2026-08-25. NO SECOND VENDOR
HAS RUN THIS.** `pixi run check-gaussian-process` is green in FAST and green
under `tools/with_identical_mode.sh`, eleven checks each, nine sabotage arms
driven at run time with no source edited.

    check_kernels_vs_oracle OK [IDENTICAL]: 55 kernel-case-by-fixture
      combinations, 5324 cells, bit-equal at every cell
    check_posterior_recovers_training OK: 16 planted observations
      recovered, worst |mean - y| 5.96e-08 against the 2^-14 bound
    check_launch_invariance OK: 5 fixtures byte-identical across elem_tpb
      256/64/32, solve_tpb 256/8, three padding-and-poison combinations,
      and every test point predicted ALONE against the same point in batch

Under FAST `check_kernels_vs_oracle` REPORTS divergence on 43 of 55 cases,
because the vendor `exp`, `sqrt` and `div` are free to differ from the host
replay. That contrast is what makes the IDENTICAL result evidence.

**No performance number and none is claimed.** See WHAT THIS WILL COST.

### The float32 ridge finding, confirmed by bits

`check_duplicate_inputs_need_the_ridge` establishes that **scikit-learn's
default `alpha = 1e-10` is a NO-OP in float32**: `1.0f + 1e-10f` is exactly
`1.0f`, and a kernel matrix has a unit diagonal. A user porting a float64
script gets an unridged factorization and a pivot failure with no
explanation. With no ridge the duplicate fixture refuses at `info = 2`, and
the pivot at column 1 is exactly `1 - 1 = 0` by hand.

### THREE COVERAGE GAPS FOUND BY RUNNING, not predicted

**The variance clamp never fires, so its counting path is UNVERIFIED.** On
all five fixtures and 32 test points the clamp fired zero times, and the
closest any variance came to zero was EXACTLY 0.0 at `planted[0]`. The
fixture was designed to drive this and its design argument was sound: it
predicts at its own training points with no ridge and no white noise, so
`k** - v^T v` is zero in exact arithmetic and float32 error should straddle
it. It did not straddle, it landed. Consequently `GP_SAB_NO_CLAMP` removes a
clamp that never runs and `GP_SAB_CLAMP_UNCOUNTED` corrupts a flag nothing
ever sets, so **both arms are unreachable and neither is evidence**. Owed: a
fixture whose true predictive variance is small but not exactly zero.

**"log|K| moved so the lml must move" is not a valid inference in float32.**
The lml carries `-0.5 log|K|` against terms of larger magnitude, so a
one-ulp move enters as half of itself and rounds away. Measured on
`planted`: log|K| moved by 1.59e-12, half of that is 7.96e-13, one ulp of
the lml is 1.91e-06. The clause is now quantitative and RECORDS below the
resolution, raises at or above it.

**`GP_SAB_LOGDET_RECOMPUTED` is inert on one fixture and live on another**,
so the driver sweeps fixtures and names which moved, rather than trusting
the one its author picked. Same lesson the Cholesky lane learned.

### One defect the gate found in itself

`check_kernel_algebra` multiplied the two leaf matrices with a bare `*`
while `gp_combine_kernel` computes `ftz(identical_mul(ftz(a), ftz(b)))`. At
cell 3 of the planted fixture the node returned `0x00000000` and the hand
value was `0x000c00d2`, a subnormal. The claim "the Product node equals the
leaves multiplied" is only meaningful if "multiplied" is the same operation,
so the hand path now uses the same seam, and the residual FAST divergence is
demoted to a REPORT because `ftz` compiles away there while Metal flushes in
hardware regardless.


**CONSTRUCTION PLUS WRITTEN GATES, 2026-08-25. NOTHING HAS BEEN RUN.**

Not one file in this directory has been compiled or executed on any device.
There is no performance number, no accuracy number, no card, no cross-vendor
claim, and no claim of any kind that rests on a measurement. Every table
below is a PREDICTION until an orchestrator runs it. The three predictions
that could be wrong in an interesting way are named in
`checks/gp_check.mojo`'s header rather than smoothed over, and the
strongest of them is the first.

1. **Whether the predictive-variance CLAMP fires at all.** The argument that
   it must is in `gp_fixture.mojo::gp_fixture_alpha` and it is an argument
   about float32 noise around an exactly-zero quantity, not a measurement.
   If it fires nowhere, `check_variance_is_nonnegative_and_clamps_are_counted`
   RAISES and prints the closest-to-zero variance it saw. That is the
   correct outcome. **Do not weaken that gate and do not add a ridge to the
   planted fixture**, which would move the true variance to `alpha` and
   guarantee the clamp never fires again.
2. **Whether the un-ridged duplicate fixture stops at `info = 2`.** The
   derivation is in `gp_fixture.mojo::gp_duplicate_expected_info` and every
   step of it is exact, so this one should hold. It is still a derivation.
3. **Whether `GP_SAB_ALGEBRA_REASSOCIATE` moves any bit.** Predicted
   LARGELY INERT, driven anyway, RECORDED either way.

The commands, once the orchestrator has wired them.

    pixi run check-gaussian-process                                             # FAST
    tools/with_build_lock.sh     pixi run mojo run -I . gaussian_process/checks/gp_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . gaussian_process/checks/gp_check.mojo

    MOJOLEARN_IDENTITY_TRACE=/tmp/mac.gp.identical.card \
        tools/with_identical_mode.sh pixi run mojo run -I . gaussian_process/gp_main.mojo

## THERE IS NO UPSTREAM GAUSSIAN PROCESS

Verified against the pinned checkouts on 2026-08-25, and the command is in
`DERIVATION_MAP.tsv` so it can be re-run.

    grep -ril "gaussian_process\|GaussianProcessRegressor\|gaussian process\
    \|marginal_likelihood\|MaternKernel" \
      upstream/cuml-v26.08.00 upstream/cuvs-v26.08.00 upstream/raft-v26.08.00

returns exactly ONE file, `cuml-v26.08.00/python/cuml/cuml_accel_tests/
integration/test_kernel_ridge.py`, and its only hit is the line
`from sklearn.gaussian_process.kernels import ExpSineSquared`, an import of
scikit-learn inside a KERNEL RIDGE test. `grep -ril matern` over the same
three trees returns nothing at all. cuML is pinned at `265b9da`, cuVS at
`6ba2ce2`, RAFT at `ebf9268`.

**`PORTING_RULES.md`'s COPY DO NOT IMPROVE does not apply where there is
nothing to copy.** Rule 0b's charter is to take an incumbent library's code
and its algorithms and port them; for a Gaussian process there is no
incumbent GPU library in this repository's pin set, so there is no file to
transliterate and no dispatch to follow. Every file here is `checks/` by
rule 0b-ii's own definition, "something they never needed", and none of them
cites a line number it cannot have.

**scikit-learn 1.9.0 (`77def0e`) is the SEMANTICS reference and the ORACLE,
and it is never the design source.** It is CPU, LAPACK-shaped and float64.
What it settles is what our parameters MEAN, and every one of those meanings
is cited by file and line in the source rather than recalled.

| what it settles | their line |
|---|---|
| `fit` computes `K = k(X)`, adds `alpha` to the diagonal, factors, solves | `_gpr.py:349-367` |
| the marginal likelihood is three terms in that order, with `sum(log diag L)` and not `log|K|` | `_gpr.py:613-617` |
| `predict` builds `K_trans = kernel(X_test, X_train)` and the mean is `K_trans @ alpha_` | `_gpr.py:446-447` |
| the predictive variance is `kernel.diag(X) - einsum(V, V)` and negatives are CLAMPED with a warning | `_gpr.py:480-491` |
| `return_std` returns the ROOT of that | `_gpr.py:500` |
| RBF is `exp(-0.5 * sqeuclidean(X/l, Y/l))` and the distance is UNEXPANDED | `kernels.py:1568-1570` |
| Matern's three closed forms, in their order, over the EUCLIDEAN distance | `kernels.py:1720-1730` |
| WhiteKernel's diagonal is a STRUCTURAL test (`Y is None`), not a coordinate test | `kernels.py:1406-1419` |
| `Sum` and `Product` are elementwise and are NOT distributed | `kernels.py:871`, `:971` |
| an isotropic length scale is one entry and ARD is `n_features` entries | `kernels.py:34-48` |
| the RBF and Matern diagonals are ASSIGNED `1` rather than computed | `kernels.py:1564`, `:1741` |

That last row is worth its own sentence. sklearn forces the unit diagonal;
here it falls out of the arithmetic, because the unexpanded distance from a
row to itself is a chain of `fma(+0.0, +0.0, +0.0)` and `portable_expf(-0.0)`
is exactly `1.0`. `check_kernels_vs_oracle` asserts it by bits rather than
assuming it, because every jitter argument downstream rests on it.

## Two findings from the Cholesky lane, carried and not re-derived

1. **Every correlation-shaped kernel matrix has a unit diagonal**, because
   `k(x, x) = exp(0) = 1`. So on exactly the shapes this lane feeds the
   factorization, its absolute jitter policy and a relative one COINCIDE.
   `cholesky/README.md`'s first correction is the record of that, found when
   `CHOL_SAB_JITTER_RELATIVE` turned out to be reached and provably inert on
   an RBF fixture. **This lane adds no second jitter knob** (DEVIATION 1751)
   and does not re-derive the argument.
2. **An RBF Gram matrix in float32 at 64 points is numerically close to
   singular.** The ridge is not optional and its size is part of the
   numerical profile. If a marginal likelihood here needs a larger ridge
   than `cholesky_profile_jitter()` returns, **that is a finding to write
   down and not a number to quietly raise**, and
   `check_duplicate_inputs_need_the_ridge`'s failure message says so in
   those words.

That second finding shaped the fixtures rather than being noted beside them.
`GP_FIX_PLANTED` puts its points four length scales apart, so its largest
off-diagonal is `exp(-8) = 3.4e-4` and its matrix is numerically
near-identity, because a fixture meant to test posterior RECOVERY must not
also be testing conditioning. `GP_FIX_ARD` and `GP_FIX_SIGNED_ZERO` use
length scales BELOW one for the same reason, so that dividing spreads the
hashed coordinates instead of compressing them.

## A third finding, this lane's own

**scikit-learn's default ridge is a no-op in float32.** `alpha = 1e-10` on a
matrix whose diagonal is `1.0` adds nothing at all, because the float32 gap
above `1.0` is `2^-23 = 1.19e-7`, three orders of magnitude larger, and
`1.0f + 1e-10f` is exactly `1.0f`. A user porting a working float64
scikit-learn script would get an unridged factorization and a pivot refusal
with no idea why. DEVIATION 1752, refused by name under IDENTICAL with that
explanation in the message, and asserted BY BITS on the host in
`check_duplicate_inputs_need_the_ridge` so that the message is only in the
source while the claim inside it is true.

The pinned ridge `2^-20 = 9.54e-7` IS above that gap, and is the smallest
ridge on this column that can do anything at all to a correlation-shaped
matrix.

## WHAT THIS LANE REUSES RATHER THAN REWRITES

Nothing in the list below was re-implemented here, and each was checked for
before a line was written.

| what | the file it lives in | why not a second copy |
|---|---|---|
| **The Cholesky factorization**, `L L^T = K + alpha I` | `cholesky/estimator.mojo::cholesky_factor_host` | gated green on an Apple M4 in both modes at `1339da7`, twelve checks and ten runtime sabotage arms. Profile `mojolearn.identical.cholesky.fp32.v1`. A second factorization here would have to earn that certificate again |
| **The two triangular solves**, `alpha_ = K^-1 y` | `cholesky/estimator.mojo::cholesky_solve_host`, which is `cho_solve` | same certificate |
| **The single triangular solve**, `v = L^-1 k_star` | `cholesky/checks/trsm.mojo::trsm_lower`, called DIRECTLY on device buffers so `predict` stays asynchronous | its right-hand-side layout, `n x nrhs` row-major, is exactly the orientation DEVIATION 1758 stores the cross-covariance in, so nothing is transposed between the two |
| **The log-determinant** | `cholesky/checks/potrf.mojo::chol_logdet`, reached through `cholesky_logdet_host` | DEVIATION 1639 made it an ENTRY POINT precisely so a GP, a kernel-ridge solver and a Gaussian mixture cannot each invent a fold order and a `log` for the same quantity. This lane taking it rather than computing its own is the entire reason it exists. DEVIATION 1757, and `GP_SAB_LOGDET_RECOMPUTED` is the arm that proves the gate would see a recomputation |
| **The ridge** | `cholesky_factor_host`'s `jitter` argument and `chol_validate_jitter` | DEVIATION 1751. `alpha` IS that jitter, passed through unchanged. There is no second knob, no second validator and no second pinned value |
| **The matrix product** in the posterior mean | `gemm/checks/gemm_identical.mojo::identical_gemm_into` at `OP_TN`, sized by `identical_gemm_workspace_max_floats` | profile `mojolearn.identical.gemm.fp32.v1`, gated at 62 shapes across eight execution plans with six sabotages. `linalg.matmul` is refused under IDENTICAL and is not called |
| **The normative answer for that product**, in the oracle | `gemm/checks/gemm_oracle.mojo::gemm_oracle` at `OP_TN` | `contract_partition`'s docstring is explicit that a second spelling of the leaf rule is a second thing that can be wrong, and records that the shape table already shipped one such re-spelling and got it wrong |
| **The per-feature step of the squared distance** | `kde/impl/distance/distance_ops.mojo::l2_unexp_core`, ported from cuVS `distance_ops/l2_unexp.cuh:62-63` | DEVIATION 1754. What is NOT reusable is the surrounding kernel, because that file's `pairwise_unexpanded_kernel` applies cuVS's `L2SqrtUnexpanded` epilog unconditionally and an RBF needs the squared distance with no root. So the ARITHMETIC is imported and only the loop is written here; the right merge is named under WHAT IS OWED |
| **The float64 factorization, solve and log-determinant** in the reference | `cholesky/checks/cholesky_oracle.mojo::reference_potrf_lower_f64`, `reference_solve_f64`, `reference_logdet_f64`, plus `oracle_add_jitter` | a textbook Cholesky written a second time here, even in float64 and even in an oracle, is exactly the duplication this section exists to refuse |
| **Transcendentals and the arithmetic pins**: `identical_exp`, `identical_sqrt`, `identical_log`, `identical_div`, `identical_mul`, `identical_mul_add`, `ftz` | `checks/numerics.mojo` | rows 9, 10, 12 and 49 of `IDENTITY_PATHS.md`. Nothing in this directory calls `std.math` on a numeric path; the only `std.math` here is in the FLOAT64 host reference and inside the `GP_SAB_STD_EXP` sabotage arm, both by design |
| **Stage hashing and the differ** | `core/identity_trace.mojo` | one hash function per repository |
| **The fixture hash and its value constructors** | `cholesky/checks/cholesky_fixture.mojo::chol_mix64`, `bits_value`, `exact_offdiag` | that lane's README records that four copies of one splitmix64 already exist in this tree and that a fifth would be a debt. This lane adds NONE |
| **The RBF Gram fixture itself** | `cholesky/checks/cholesky_fixture.mojo::rbf_gram` | its docstring says it lives there rather than here "so that the callers cannot disagree about the fixture". `check_kernels_vs_oracle`'s cross-lane arm is what stops them disagreeing: our `RBF([1.0])` must equal it BIT FOR BIT, and if it does not, one of the two lanes is wrong about what an RBF is |
| **The pinned block fold**, considered and NOT used | `core/pinned_reduce.mojo` | named because a reviewer will look for it. No kernel in this lane folds across threads at all, so there is no fold shape to pin and importing one would suggest there is |
| **Fixed-point accumulation**, considered and NOT used | `checks/fixed_point.mojo` | it exists to REPLACE float atomics and this lane has none. There is no float atomic, no integer atomic, no `Atomic.fetch_add`, no warp shuffle, ballot or vote, and no `block.sum` anywhere in `gaussian_process/` |
| **The EXPANDED RBF**, deliberately NOT used | `svm/impl/distance/kernel_matrices.mojo::kernel_op` | DEVIATION 1755. It is correct for an SVM and wrong for a GP, and the argument is below |

### Why the SVM lane's RBF is not reused

`svm/impl/distance/kernel_matrices.mojo` already ships an RBF kernel
matrix, ported from cuVS, and it is the natural thing to reach for. It
computes the squared distance by the EXPANDED identity, `||x||^2 + ||y||^2 -
2 x.y`, because that is what cuVS does and because it turns the whole matrix
into one GEMM. That is the right design for a support vector machine and the
wrong arithmetic for a Gaussian process, for two reasons and only the second
is the load-bearing one.

The obvious objection is the diagonal, and it is weaker than it looks. When
the norms and the dot product are all ascending serial chains over the same
values, they coincide exactly and the diagonal is exactly `1` after all. But
their dot product comes from a GEMM whose fold is the contract's balanced
tree, and at `k > CONTRACT_K_LEAF_MIN` that is not the norm's ascending
chain, so `k(x, x)` stops being exactly `1` at high feature counts. The unit
diagonal that finding 1 rests on would then be a function of `d`.

The real reason is CANCELLATION. For two nearby points the expanded form
subtracts two large, nearly equal quantities to obtain a small one, and a
Gram matrix is made of nearby points. In float32 that can return a NEGATIVE
squared distance, hence a kernel value above `1`, hence a matrix that is not
positive semi-definite and that no ridge of the pinned size can repair.
scikit-learn's `cdist(..., 'sqeuclidean')` is unexpanded and this lane
follows it. `GP_SAB_EXPANDED_RBF` drives the substitution so the gate can be
shown to see it, and the arm's docstring states exactly what it does and
does not demonstrate.

## The identity table (row text for `IDENTITY_PATHS.md`)

| n | path | what is vendor-dependent in the ordinary spelling | what we did | status |
|---|---|---|---|---|
| 67 | **`gaussian_process/` -- the COVARIANCE FUNCTION** (`checks/kernels.mojo::gp_rbf_kernel`, `gp_matern_kernel`) | every cell of a Gram matrix goes through a device `exp`, and Matern additionally through a `sqrt` and a divide. A device `exp` is a VENDOR CHOICE in its last bit (row 12), NVIDIA's PTX `sqrt` is off by one ulp on 180,714 of 2^20 patterns (DEVIATION 258), and the feature-axis accumulation is a contraction a codegen may or may not fuse (row 9). Nothing about a GP is small enough for those to stay invisible: the factorization amplifies them and the marginal likelihood is a log of a determinant of the result | **PIN EVERY SEAM.** `identical_exp`, `identical_sqrt`, `identical_div`, `identical_mul`, `identical_mul_add`, and `ftz` at every stored intermediate. ONE THREAD OWNS ONE CELL and walks the feature axis ASCENDING in its own registers, so no float crosses a thread boundary and the summation order is a pure function of `d`. The per-feature step is IMPORTED from the KDE lane rather than re-spelled. DEVIATIONS 1753, 1754, 1767 | **CONSTRUCTION 2026-08-25, NOTHING RUN.** `check_kernels_vs_oracle` compares 11 kernel cases on 5 point sets, per cell, bit for bit under IDENTICAL, and additionally asserts the unit diagonal, exact two-argument symmetry, exact ARD irrelevance, row-39 zero-sign blindness and bit equality with the Cholesky lane's own `rbf_gram` |
| 68 | **`gaussian_process/` -- the KERNEL ALGEBRA** (`gp_kernel_matrix`'s postfix walk) | a sum of products is an EXPRESSION, and an implementation is free to distribute, factor or reassociate it. `(A + B) * C` and `A*C + B*C` are the same mathematics and two different float32 answers, and a library that normalizes its kernel expressions before evaluating them would give two answers for one user-visible kernel | **PIN THE EVALUATION ORDER.** The kernel is a POSTFIX node list evaluated once, in order, over a device operand stack, and nothing is distributed, factored or reassociated. `Sum` and `Product` are elementwise into the LEFT operand's slot, which is scikit-learn's own two lines. The node count and stack depth are pinned CAPACITIES and are refused by name rather than grown. DEVIATION 1756 | **CONSTRUCTION, NOTHING RUN.** `check_kernel_algebra` compares a sum, a product and a NESTED `(Const * RBF) + White` against the hand-composed matrices, and requires `ConstantKernel(1.0) * RBF` to equal the bare `RBF` bit for bit in BOTH modes |
| 69 | **`gaussian_process/` -- the PREDICTIVE VARIANCE and its CLAMP** (`gp_variance_kernel`) | `k** - v^T v` is a Schur complement, non-negative in exact arithmetic and NOT in float32 -- at a test point coinciding with a training point the true value is exactly zero and the computed one is a few ulps either side. So the clamp fires in ordinary use. scikit-learn clamps on `y_var < 0`, which is FALSE for `-0.0`, so a negative zero survives into the output where no tolerance comparison can see it; and a `max(v, 0.0)` spelling would return `+0` on NVIDIA and AMD and `-0` on Apple (row 39, measured) | **PIN THE COMPARISON AND RECORD THE CLAMP.** Spelled `not (v > 0.0)`, so NaN and both zeros take the same branch on every vendor and no `max` is involved. The fold is ONE THREAD PER TEST POINT, ascending, never a GEMM. The clamp is recorded as a per-test-point FLAG VECTOR set from a BIT COMPARISON of stored against computed, hashed as the card stage `gp.clamped` and summed on the host -- never a count, never an atomic. DEVIATIONS 1759, 1760 | **CONSTRUCTION, NOTHING RUN.** `check_variance_is_nonnegative_and_clamps_are_counted` asserts no negative and no `-0.0` reaches the output, compares the FLAGS cell by cell against the oracle, REPORTS the count per fixture whether or not it is zero, and RAISES if it is zero everywhere |
| 70 | **`gaussian_process/` -- the LOG MARGINAL LIKELIHOOD** (`gp_log_marginal_likelihood_value`) | three terms, two of them folds and one of them a transcendental constant. A caller that recomputes `log|K|` from the factor instead of taking the one the factorization already produced introduces a SECOND fold order and a SECOND `log` for one quantity, which is rows 21 and 12 at once; and the three terms may be assembled in any order a codegen likes unless they are named | **TAKE IT FROM THE FACTOR AND PIN THE ASSEMBLY.** `log|K|` comes from `cholesky_logdet_host` and is never recomputed; `y^T alpha` is a HOST ascending fold over two vectors that are already on the host (the contrast with `chol_logdet`, which is on the device because ITS input is, is stated in the source); `log(2 pi)` is a pinned float32 constant written as BITS. The three terms are two named partials and one final add, in scikit-learn's order. DEVIATIONS 1757, 1767 | **CONSTRUCTION, NOTHING RUN.** `check_log_marginal_likelihood` asserts `dual_coef` and `y^T alpha` BIT FOR BIT against a hand derivation, compares `log|K|` and the likelihood to float64, and DRIVES `GP_SAB_LOGDET_RECOMPUTED` requiring the number to move |
| 71 | **`gaussian_process/` -- the RIDGE, and the absence of a second one** | a GP library that owns its own ridge owns a numerical parameter nobody writes down, and three lanes calling one factorization would then apply three | **THERE IS NO SECOND KNOB.** `alpha` IS `cholesky_factor_host`'s `jitter`, passed through unchanged, and under IDENTICAL the only accepted values are the Cholesky profile's two. scikit-learn's default `1e-10` is refused by name AND is a float32 no-op on a unit diagonal, which is asserted by bits. DEVIATIONS 1751, 1752 | **CONSTRUCTION, NOTHING RUN.** `check_duplicate_inputs_need_the_ridge`, which is the gate that proves the ridge is load bearing, and `check_gp_refusals` |
| 72 | **`gaussian_process/` -- DATA-DEPENDENT ITERATION**, refused rather than pinned | a hyperparameter optimizer's step count is a function of the data, so the CONVERGENCE TEST is part of the arithmetic. Two vendors agreeing bit for bit on every L-BFGS step still return two different models if one takes 41 steps and the other 42, and no amount of per-step agreement prevents that. The same shape appears in a GP classifier's Laplace Newton loop | **REFUSE, WITH THE CLOSURE CONDITION NAMED.** `optimizer` accepts only `"none"`; `n_restarts_optimizer` must be 0 (it additionally draws from an RNG); classification raises by name. IDENTITY_PATHS' third move applied to a control-flow question rather than to an arithmetic one. DEVIATIONS 1761, 1766 | **CONSTRUCTION, NOTHING RUN.** `check_gp_refusals` drives four optimizer values, the restart count and the classifier entry point, with an ACCEPTED fit beside them so the refusals are known not to be always firing |
| 73 | **`gaussian_process/` -- row 39 in a covariance function**: signed zeros in a coordinate and in the variance | a `-0.0` coordinate, a `-0.0` predictive variance and a `-0.0` in the mean are all invisible to every tolerance comparison, and none of them arises from ordinary data, so no fixture that does not PLANT one can see the question at all | the covariance function is blind to the sign of a zero coordinate BY ARITHMETIC (`-0.0 - (+0.0)` is `-0.0`, and `fma(-0.0, -0.0, +0.0)` is `+0.0`) and that is asserted rather than assumed; the variance clamp turns a `-0.0` into `+0.0` and COUNTS it, where scikit-learn's `y_var < 0` lets it through | **CONSTRUCTION, NOTHING RUN.** `check_kernels_vs_oracle`'s row-39 arm, on a fixture that PLANTS both zero signs and which RAISES if the plant is not actually there, because an agreement between two matrices neither of which contains a negative zero proves nothing |

Row numbers 67-73 are proposed, not claimed. `IDENTITY_PATHS.md` is not this
lane's file and the orchestrator assigns them.

## The deviations

| # | what |
|---|---|
| **1750** | **THERE IS NO UPSTREAM GAUSSIAN PROCESS**, verified by grep over the three pinned checkouts, so COPY DO NOT IMPROVE does not apply where there is nothing to copy. scikit-learn is SEMANTICS and ORACLE, never design |
| **1751** | **`alpha` IS the Cholesky lane's jitter, passed through unchanged.** There is no second jitter knob. Under IDENTICAL the two pinned values are the only ones accepted |
| **1752** | scikit-learn's default `alpha = 1e-10` is a NO-OP in float32 on a unit-diagonal kernel matrix, because `1.0f + 1e-10f` is exactly `1.0f`. Refused by name under IDENTICAL with that explanation, and the claim itself is asserted by bits |
| **1753** | the ARD length scale is applied by SCALING THE COORDINATES, scikit-learn's own `X / length_scale`, and it is FUSED into the feature loop rather than materialized. Bit-equal because each quotient is rounded to float32 before the subtraction either way, and the oracle materializes it so the equality is gated |
| **1754** | the per-feature step of the squared distance is `l2_unexp_core`, IMPORTED from `kde/impl/distance/distance_ops.mojo` rather than re-spelled. Only the loop is written here, because that file's kernel applies an unconditional `sqrt` epilog |
| **1755** | the EXPANDED RBF (`svm/impl/distance/kernel_matrices.mojo`) is NOT used. Cancellation between two large nearly-equal quantities is what a Gram matrix is made of, and the unit diagonal would additionally become a function of `d` |
| **1756** | the kernel expression is POSTFIX and is NEVER distributed, factored or reassociated. The node count and stack depth are pinned CAPACITIES, refused by name rather than grown silently |
| **1757** | `log|K|` comes from `cholesky_logdet_host` and is NEVER recomputed. scikit-learn's `sum(log(diag L))` is spelled `0.5 * logdet` |
| **1758** | the cross-covariance is stored `n_train x n_star`, which is `K_trans^T`, and NOTHING is ever transposed. That one orientation feeds both `trsm_lower` (as `n x nrhs`) and the mean (as `OP_TN`'s left operand), and it is legitimate because every kernel here is exactly symmetric in its two arguments BY BITS |
| **1759** | the predictive variance's `v^T v` is a per-test-point serial ascending fold in ONE thread, not a GEMM. Only the diagonal of `V^T V` is wanted, and scikit-learn makes the same choice for the same reason |
| **1760** | the negative-variance CLAMP is recorded as a per-test-point FLAG VECTOR, hashed as a card stage and summed on the host. Never a count, never an atomic, and the flag is set from a BIT COMPARISON so an exactly `+0.0` variance is not miscounted |
| **1761** | hyperparameter optimization is NOT implemented; `optimizer` accepts only `"none"`. The iteration count is data dependent, so the convergence test is part of the arithmetic |
| **1762** | `WhiteKernel` is a STRUCTURAL flag (`Y is None`), not a coordinate test, exactly as scikit-learn spells it. Two identical training rows therefore do NOT get the white noise on their off-diagonal cell |
| **1763** | `y` is a SINGLE target; multi-output refused by name |
| **1764** | `normalize_y` is not ported and is refused by name. It is two folds and a division through which every prediction passes |
| **1765** | the general Matern (the Bessel branch) and `nu = inf` are refused by name. `kv` is a new transcendental that would need its own row-12 certificate |
| **1766** | classification and `sample_y` are refused by name, as ENTRY POINTS that raise rather than as absent symbols |
| **1767** | `sqrt(3)`, `sqrt(5)` and `log(2 pi)` are written as float32 BITS, and `check_gp_constants` asserts each against its float64 value so a hex transcription error cannot ship |
| **1768** | non-finite `X`, `y`, length scales and `alpha` are refused by name on the HOST before any launch |
| **1769** | the sabotage arms are runtime-selectable through a `sabotage` argument, in a separate file, never reached by any driver |
| **1770** | `kernel.diag(X)` is ONE HOST SCALAR for this kernel set, because no kernel here has a coordinate-dependent diagonal. Porting `DotProduct` turns it into a vector and turns the `gp.kss` stage into an array |
| **1771** | the self-kernel `k(X, X)` uploads `X` TWICE, because Mojo cannot pass one `DeviceBuffer` as two `mut` arguments of one call. `PORTING_RULES` rule 4's shape: it changes HOW the call is spelled and not WHAT is computed |
| 1772-1779 | RESERVED, unspent |

## WHAT THE ORCHESTRATOR MUST WIRE

Nothing outside `gaussian_process/` was edited by this lane. These lines are
wanted in `pixi.toml`, in the file's existing format, beside the other
classical lanes' tasks (`check-cholesky` is at line 983 today).

    check-gaussian-process = "mojo run -I . gaussian_process/checks/gp_check.mojo"
    gaussian-process-main = "mojo run -I . gaussian_process/gp_main.mojo"

The IDENTICAL pass is the injector, exactly as for every other gate.
`tools/with_identical_mode.sh pixi run check-gaussian-process`. No
`*-identity` task is wanted.

Also owed by the orchestrator, and none of it is this lane's to do.

1. **`IDENTITY_PATHS.md` rows 67-73**, from the identity table above.
   Numbers proposed, not claimed.
2. **`PORTING.md` and `ROADMAP.md`**. `gaussian_process/` is a new section
   and appears in neither.
3. **`UNWIRED.md`**. `gaussian_process/estimator.mojo` has no caller in
   `bindings/_mojolearn_estimators.mojo` or `python/mojolearn/`. So do
   `gpr_classify_host` and `gpr_sample_y_host`, deliberately, because they
   exist only to raise.
4. **The Python surface**, when a binding is wanted. `gpr_fit_host`,
   `gpr_predict_host`, `gpr_log_marginal_likelihood`, `gp_profile_alpha`,
   and the six kernel constructors in
   `gaussian_process/checks/kernels.mojo`. Refuse `float64` inputs, any
   `optimizer` other than `None`, `normalize_y=True` and a two-dimensional
   `y`, by name, in Python.
5. **A file this lane's file list did not name.**
   `gaussian_process/checks/gp_sabotage.mojo` is not in the brief's list
   of files, and it exists because the brief ALSO requires the sabotages to
   be runtime-selectable "the way `cholesky/checks/chol_sabotage.mojo`
   does it", and that construction is a separate file by design. Recorded
   here rather than slipped in.

## SABOTAGES TO PERFORM

All eleven are selected at RUN TIME through the `sabotage` argument
(`gaussian_process/checks/gp_sabotage.mojo`), copying
`cholesky/checks/chol_sabotage.mojo`'s construction and how
`cholesky_check.mojo` drives it. **No source edit and no rebuild is required
for any of them**; `check_gp_sabotages` drives all eleven in one run and
prints a line per arm. Every arm below is a PREDICTION until it has run.

**Each arm SWEEPS the fixtures and never names one.** An arm inert on the
fixture its author happened to pick is indistinguishable from an arm that is
unreached, and the Cholesky lane shipped one of each before this discipline
existed. The print names WHICH fixture the arm moved on and how many earlier
ones were INERT to it.

| check it targets | sabotage | exactly what it corrupts | what must move |
|---|---|---|---|
| `check_kernels_vs_oracle` | `GP_SAB_DIST_DESCENDING` | the feature axis of the squared distance walks `f` DESCENDING. Same multiset, different order | the kernel matrix's bits. MUST FAIL. **Structurally INERT on any one-feature fixture**, because reversing a loop of length one is the same loop, so `planted` and `handworked` cannot see it and the sweep is what finds `duplicate`, `ard` or `signed_zero` |
| `check_kernels_vs_oracle` | `GP_SAB_STD_EXP` | `std.math.exp` instead of `identical_exp` at the kernel value. EVERY cell of a Gram matrix goes through one, so this is the widest seam in the lane | the kernel matrix's bits. MUST FAIL under IDENTICAL (where `identical_exp` is the Cephes polynomial and `std.math.exp` is the vendor's). Expected INERT under FAST, where the two are the same function, and the check's verdict table says so |
| `check_kernels_vs_oracle` | `GP_SAB_EXPANDED_RBF` | the squared distance becomes `||x||^2 + ||y||^2 - 2 x.y`, the SVM lane's correct-for-an-SVM arithmetic | the kernel matrix's bits OFF THE DIAGONAL. MUST FAIL. Inert on the diagonal, because with all three sums ascending serial chains they coincide exactly there, and inert on a fixture whose kernel has no RBF or Matern leaf. Both statements are in the arm's docstring so the arm is not credited with more than it shows |
| `check_kernels_vs_oracle` | `GP_SAB_NO_FTZ_KERNEL` | `ftz` dropped at every seam of the distance and the exponent | APPLE-INERT, and this is the arm that matters most for a second vendor. Metal flushes in hardware so no Apple bit is expected to move; on NVIDIA and AMD a subnormal coordinate or partial sum survives where IDENTICAL flushes it. RECORDED, never claimed |
| `check_kernel_algebra` | `GP_SAB_ALGEBRA_REASSOCIATE` | a Sum or Product node combines `b op a` instead of `a op b` | REPORT. Predicted LARGELY INERT, because float `+` and `*` are exactly commutative away from NaN payloads and zero signs. It is driven so the answer is recorded instead of assumed, and a reader who expects it to fail learns something either way |
| `check_variance_is_nonnegative_and_clamps_are_counted` | `GP_SAB_VDOTV_PAIRWISE` | the predictive variance's `v^T v` folds PAIRWISE in a register stack instead of ascending serial | the variance's bits. MUST FAIL. Inert at `n_train = 2`, where the two bracketings coincide |
| same | `GP_SAB_NO_CLAMP` | the clamp is dropped, so a negative variance is reported and its root is NaN | the variance's bits, on any fixture where the clamp fires. MUST FAIL, and it is inert exactly where the clamp count is zero, which makes the two arms one gate |
| same | `GP_SAB_CLAMP_UNCOUNTED` | the clamp still fires and the per-test-point FLAG is never written | `gp.clamped`'s bits and `n_clamped`, with the variance UNCHANGED. MUST FAIL. This is the arm that proves DEVIATION 1760 is gated rather than decorative, because it is the only one whose corruption is invisible in the output |
| `check_launch_invariance`, `check_posterior_recovers_training` | `GP_SAB_MEAN_DESCENDING` | the posterior mean is computed by a hand kernel folding the training axis DESCENDING, so the mean leaves the gemm profile entirely | the mean's bits. MUST FAIL. Inert at `n_train = 2`, where `a + b` and `b + a` are the same float |
| `check_log_marginal_likelihood` | `GP_SAB_LOGDET_RECOMPUTED` | `log|K|` is recomputed as `log(prod_j L_jj^2)` instead of being taken from the factor | `gp.logdet` AND `gp.lml`. MUST FAIL, and the value it moves TO is printed, because it is instructive: on any correlation-shaped factor the product underflows toward zero and the answer collapses, which is why the sum-of-logs form exists |
| `check_log_marginal_likelihood` | `GP_SAB_YALPHA_DESCENDING` | `y^T alpha` folds DESCENDING | `gp.ydotalpha` and `gp.lml`. MUST FAIL past `n = 2` |

**One arm a reader will look for and will not find, and it is worth saying
why.** There is no `linalg.matmul` swap here. That arm is
`CHOL_SAB_VENDOR_MATMUL` in the Cholesky lane and its siblings in the gemm
lane, and re-driving it would be a second opinion about a profile whose
certificate this lane inherits rather than re-earns. What this lane gates is
that the mean's product is ON that profile at all, and
`GP_SAB_MEAN_DESCENDING` is the arm that shows the gate can see it leave.

**And one whole class of arm that has no object here.** There is no block
reduction anywhere in this lane, so there is nothing to swap two fold levels
in and no block-reduction width to change. Every kernel keeps every sum
inside one thread; the only cross-thread combination in the whole lane lives
inside `identical_gemm_into`, where the gemm lane's own six sabotages
already cover it. That absence is the reason launch invariance is a property
of shape here rather than a property a check happens to observe.

## What the checks are expected to establish

| check | what it would establish |
|---|---|
| `check_gp_constants` | the three pinned float32 constants are the correctly rounded float32 of the float64 values scikit-learn uses, so a hex transcription error is caught by a named message rather than by a wrong answer four stages later |
| `check_gp_refusals` | the general Matern and `nu = inf`, classification, six unported kernel names, four optimizer values, restarts, `normalize_y`, four bad `alpha` values (including scikit-learn's own default under IDENTICAL), non-finite `X` and `y`, a multi-output `y`, a zero length scale, a mis-sized ARD vector, `sample_y`, and a failed fit's marginal likelihood and prediction each RAISE BY NAME -- and the three accepted `nu` values, four accepted kernel names and both pinned ridges are ACCEPTED on the same code paths, so the refusals are not simply always firing |
| `check_kernels_vs_oracle` | 11 kernel cases on 5 point sets, per cell, bit for bit against the float32 host replay under IDENTICAL (a REPORT under FAST), within a printed tolerance of the float64 reference, PLUS the unit diagonal, exact two-argument symmetry, exact ARD irrelevance, row-39 zero-sign blindness, and bit equality with the Cholesky lane's own `rbf_gram` |
| `check_kernel_algebra` | a sum, a product and a NESTED `(Const * RBF) + White` equal the hand-composed matrices, `ConstantKernel(1.0) * RBF` equals the bare `RBF` bit for bit in both modes, and `kernel.diag` follows the same algebra |
| `check_posterior_recovers_training` | 16 planted observations recovered by the posterior mean to within `2^-14`, with the residual PRINTED, and the dual coefficients within a printed distance of the float64 reference |
| `check_log_marginal_likelihood` | a two-point hand derivation gives `dual_coef = (0.5, -0.5)` and `y^T alpha = 5` BIT FOR BIT and `log|K| = log(400)` and the likelihood to a printed tolerance; every fixture that fits agrees with the float64 reference; and recomputing `log|K|` a different way MOVES it, which is what makes DEVIATION 1757 a gated claim rather than a comment |
| `check_variance_is_nonnegative_and_clamps_are_counted` | no reported variance is negative and none is `-0.0`; the clamp FLAGS match the oracle cell by cell rather than only in total; and the count is PRINTED per fixture whether it is zero or not, with the whole check RAISING if it is zero everywhere |
| `check_duplicate_inputs_need_the_ridge` | **the gate that proves the ridge is load bearing.** Two bit-identical rows make the leading 2x2 block exactly `[[1, 1], [1, 1]]`, so the pivot at column 1 is exactly `1 - 1 = 0` and the factorization refuses at `info = 2`; with the pinned ridge it succeeds and predicts. And, on the host by bits, that scikit-learn's default `1e-10` is a float32 no-op while the pinned `2^-20` is not |
| `check_launch_invariance` | **the headline.** Nothing moves across `elem_tpb` 256/64/32, `solve_tpb` 256/8, three padding-and-poison combinations, the same run twice, or one test point predicted ALONE against the same point inside its batch. The last arm is the one a GP can fail and the others cannot, because the variance folds a COLUMN of `v` and `v` comes out of a solve whose right-hand-side count IS the batch size |
| `check_card_is_emitted` | the sixteen stages, in order, with the right record count, and two runs of one fixture producing an identical card |
| `check_gp_sabotages` | all eleven arms driven at run time with no source edited, each classified MUST FAIL, APPLE-INERT or REPORT in advance, each swept across the fixtures with the inert ones named |

## The card

`gaussian_process/gp_main.mojo` emits sixteen stages.

    gp.x_train, gp.y_train,
    gp.kernel, gp.ridged, gp.factor, gp.dual_coef,
    gp.logdet, gp.ydotalpha, gp.lml,
    gp.kss, gp.kcross, gp.mean, gp.v, gp.var, gp.clamped, gp.std

The ORDER is the product, not the length. A card that diverges has an
address and the address is the diagnosis; `gp_main.mojo`'s header is the
full map, stage by stage, and its most useful entries are the ones that hand
the problem to another lane. `gp.factor` moving with `gp.ridged` identical is
the Cholesky lane's certificate; `gp.mean` moving with `gp.kcross` and
`gp.dual_coef` identical is the gemm lane's; `gp.v` moving with `gp.kcross`
and `gp.factor` identical is the Cholesky lane's again. When the environment
variable is set, `cholesky_factor_host` and `cholesky_solve_host` write
their own `chol.*` stages into the same file, so a divergence inside the
factorization has a per-panel address without this lane doing anything.

**No card has been emitted.** The stage list above is what the source
records, not a transcript.

## WHAT THIS WILL COST

**Under IDENTICAL this lane is expected to be SLOWER than scikit-learn on
the CPU, and it will stay that way until the GEMM kernel gap closes.** That
is not a hedge and it is not a guess about this lane; it is what the
repository has already measured about the primitive this lane is built on.

`bench/results/SPEED_LANE_2026-08-25.md` section 3.4 measured this
repository's identical FP32 GEMM at

| column | ours | vendor library, strict FP32 | ratio | ours as share of FP32 peak |
|---|---|---|---|---|
| Apple M4 | 0.078 TF/s | 2.33 (MPS) | ~30x | ~2% |
| H100 80GB | 1.95 - 3.51 | 44.4 (cuBLAS) | 12.7x - 22.8x | 3 - 5% |
| MI325X | 4.31 | 77.1 (hipBLASLt) | 17.9x | ~5% |

and that report's own conclusion is that the consistency is the result worth
reporting, because sitting at 2 to 5 percent of FP32 peak on all three
vendors says the gap is structural to the kernel rather than an accident of
one backend.

**A Gaussian process is Cholesky and GEMM bound.** `fit` is `O(n^3)` in the
factorization and `O(n^2 d)` in the kernel matrix; `predict` is `O(n^2 m)`
in the triangular solve and `O(n m)` in the mean. Every one of those runs on
this repository's identical kernels, and two of them, the trailing update
inside `potrf_lower` and the posterior mean, are literally the GEMM the
table above measures. The Cholesky lane's own WHAT IS OWED additionally
records that its `trsm` reads `i` floats of `L` per row per column with no
reuse, which is `O(n^2 nrhs)` global loads, and that a blocked solve is
unwritten. This lane's kernel matrix reads `d` floats of each operand per
output cell with no reuse, where a Contractions tile would read each row
once per tile. None of those three has been measured and none of them has
been optimized.

So, plainly.

- **IDENTICAL is a REPRODUCIBILITY MODE WITH A PRICED COST**, not a fast
  mode. What it buys is that the same fit gives the same model on Apple,
  NVIDIA and AMD, bit for bit, which no incumbent offers at all. What it
  costs has a number and the number is above.
- **Any speed claim this lane ever makes lives in FAST**, and no FAST number
  exists yet either. FAST makes no cross-vendor claim, by construction.
- **Do not read a win into this directory.** There is no benchmark here, no
  timing harness, no comparison against scikit-learn, and no arm that could
  produce one. A reader who infers a performance result from the presence of
  a GPU kernel is inferring something the GEMM number says is not there.
- **`IDENTITY IS NOT FREE`, and this lane does not claim otherwise.**
  Conforming costs on every vendor. The Apple column's 1.33-1.51x figure
  elsewhere in this tree compares two LAYOUTS and not FAST against
  IDENTICAL, and nothing in this directory is entitled to lean on it.

The one honest comparative statement available today is a REACHABILITY one
rather than a speed one, and it belongs to the thesis rather than to this
lane. scikit-learn's `GaussianProcessRegressor` is SciPy and LAPACK and
cannot reach an Apple GPU at all. That is a statement about where the code
runs, not about how fast it runs, and it stays that way until somebody
measures.

## WHAT IS OWED

1. **THE FIRST VENDOR LEG HAS NOT RUN.** Not the second and third, the
   FIRST. Nothing in this directory has been compiled or executed. The Apple
   M4 pass, in both modes, with `check-gaussian-process` green and one card
   emitted, is what turns this from source into construction.
2. **The second and third vendor legs.** An NVIDIA H100 and an AMD MI325X
   run of `gp_main.mojo` under `tools/with_identical_mode.sh`, and
   `tools/identity_trace_diff.py` over the three cards. Until then there is
   no cross-vendor claim of any kind. Two arms are expected to behave
   differently there and are recorded that way rather than as passes.
   `GP_SAB_STD_EXP` (a vendor `exp` at every cell) and
   `GP_SAB_NO_FTZ_KERNEL` (NVIDIA and AMD keep subnormals).
3. **Whether the clamp fires, whether the duplicate fixture stops at
   `info = 2`, and whether the reassociation arm is inert.** The three
   predictions in the Status section. The orchestrator records what it got.
4. **A `DIST_L2_SQ_UNEXPANDED` metric value in
   `kde/impl/distance/distance_ops.mojo`,** and the deletion of this
   lane's `gp_scaled_sqdist` loop in favour of it. DEVIATION 1754. That is
   the KDE lane's file, so it is named here rather than done.
5. **A single-sided forward-solve oracle in
   `cholesky/checks/cholesky_oracle.mojo`.** This lane's
   `gp_oracle_forward_solve` is a second spelling of `trsm_lower_kernel`
   that exists only because that lane's oracle exposes `oracle_cho_solve`
   (both substitutions) and not `oracle_trsm_lower` at an arbitrary `nrhs`.
   One export there deletes it here.
6. **A per-node card stage for the kernel expression.** `gp_kernel_matrix`
   records ONE stage over the final matrix, because a per-node tag would
   make the stage list a function of the kernel expression and two runs with
   two kernels would produce cards a differ cannot align. The fix is a tag
   derived from the node index, the way `chol_panel_tag` derives one from
   the panel index, plus a rule that the card is only comparable across runs
   of the SAME kernel. Not attempted, because that rule is a change to what
   a card means.
7. **One sequence-number stream across `fit` and `predict`.** Each
   constructs its own `IdentityTrace`, so the `seq` column restarts at the
   boundary. The differ aligns on TAGS so nothing depends on it, and
   `cholesky_factor_host` and `cholesky_solve_host` behave the same way, but
   a reader of a raw card should not have to discover it.
8. **A blocked kernel matrix.** `gp_scaled_sqdist` reads `d` floats of each
   operand per output cell with no reuse. A Contractions tile
   (`dbscan/impl/neighbors/epsilon_neighborhood.mojo` transcribes one)
   would read each row once per tile. It is a speed idea, it would introduce
   a staging shape to pin, and this lane has measured nothing.
9. **An `n == 1` row in the gemm shape table.** The posterior mean is an
   `m x 1 x k` product at `OP_TN`, and `bench/gemm_shapes.mojo`'s 62 shapes
   should be checked for an `n == 1` entry with one added if there is none.
   `core/gemm.mojo` already records that MAX's `linalg.matmul` does not
   write its output at `n == 1`; that is a different code path from
   `identical_gemm`, and this lane's `check_launch_invariance` exercises
   `m` from 1 to `n_star` at `n == 1`, but the gemm lane's own table is
   where the shape belongs. That is the gemm lane's call.
10. **`return_cov`, and `sample_y` on top of it.** `NOT_IMPLEMENTED.tsv` carries
    both rows. `return_cov` is one more kernel matrix and one more `OP_TN`
    GEMM; `sample_y` additionally needs a second Cholesky and an RNG stream
    inside a reproducibility claim.
11. **Rung 2, in the order the cost of each suggests.** `RationalQuadratic`
    (one `identical_pow` seam, already certified), then `DotProduct` (which
    turns `gp_kernel_diag` into a vector, DEVIATION 1770), then
    `ExpSineSquared` (which needs `identical_sin` at an argument that grows
    with the data, where a range reduction is exactly where vendors
    diverge), then multi-output, then `normalize_y`, then classification.
12. **Larger `n`.** The largest fixture is 16 training points. Nothing here
    says what happens at 4096, where the kernel matrix is 64 MB, the
    Cholesky walks 128 panels with a host round trip each, and the
    conditioning question finding 2 raises stops being hypothetical.
13. **A comparison against scikit-learn.** There is none, in either mode,
    for either accuracy or speed. WHAT THIS WILL COST says what to expect of
    the speed half before anybody runs it.
