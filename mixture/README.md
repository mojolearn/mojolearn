# mixture: a cross-vendor bit-identical full-covariance Gaussian mixture model

Opened 2026-08-25. The first lane in this repository whose upstream **does
not exist**, and the first whose most consequential output is an INTEGER:
the number of EM iterations. **DEVIATIONS 1720-1747 are this lane's**;
1748-1749 are reserved and unspent.

**The profile is `mojolearn.identical.gmm.full.fp32.v1`.** It CONTAINS two
other profiles: `mojolearn.identical.gemm.fp32.v1` (four matrix products) and
`mojolearn.identical.cholesky.fp32.v1` (one factorization, one triangular
solve and one log determinant per component per iteration). A v2 of either of
those is a v2 of this one.

## Status

**BUILT AND GATED ON ONE APPLE M4, BOTH MODES, 2026-08-25. NO SECOND VENDOR
HAS RUN THIS UNDER IDENTICAL, so there is no identity card outside the M4.**
An NVIDIA H100 compiled and ran this lane under FAST on 2026-08-26
(`bench/results/fast_speed/2026-08-26_040100-nvidia-classical.md`), which is
a speed leg and not an identity one.
`pixi run check-mixture` is green in FAST and green under
`tools/with_identical_mode.sh`, ten checks each, 13 sabotage arms driven at
run time with no source edited.

    check_iteration_count_is_identical OK: 5 fixtures, two fits each from
      one seed, agreeing on the ITERATION COUNT, the convergence flag, the
      lower bound's BITS and every parameter cell
    check_estep_vs_oracle OK [IDENTICAL]: 765 cells bit-equal to the serial
      oracle; the naive log-sum-exp underflowed to -inf where the shifted
      one did not; 2 planted mixed-zero rows x 3 block sizes keep the
      LOWER-INDEX zero's bits on device and oracle
    check_mstep_vs_oracle OK [IDENTICAL]: 732 cells bit-equal, BOTH
      scikit-learn divisors driven
    check_collapse_is_identical OK: refuses on device and oracle at the
      same component, same LAPACK info, same iteration, identical cards

**One FAST timing exists.** On an NVIDIA H100 on 2026-08-26, fixture
`SEPARATED.24x2K3`, our median was 2.390 ms with NO OPPONENT ON THAT BOX,
because RAPIDS ships no `GaussianMixture` and the scikit-learn CPU arm is
refused there under GPU-PATH-ONLY
(`bench/results/fast_speed/2026-08-26_040100-nvidia-classical.md`). So there
is a cost figure and still no comparison. See WHAT THIS WILL COST.

## THE COVERAGE STORY, WHICH IS WORSE THAN THE PASS LINE SUGGESTS

Five of the 13 sabotage arms could not fire. Four are explained and one is
not, and none of them is tuned away.

**THE GENERAL RULE, which three lanes have now paid for separately: an order
sabotage is provably inert whenever the axis it reverses has length two or
less.** Float addition is commutative to the bit even though it is not
associative, so reversing or rotating a fold over two terms is `b + a`
against `a + b`. `resample/` met it as reversal invariance of a balanced
halving fold, `cholesky/` met it as a relative ridge on a unit diagonal, and
this lane meets it three times in one driver. **Check the axis length before
believing an order arm.**

| arm | why it could not fire | owed |
|---|---|---|
| `LSE_DESCENDING` | four fixtures run K = 2; the K = 3 one is separated enough that non-maximal shifted terms underflow to exactly zero | a K >= 3 OVERLAPPING fixture |
| `LSE_ROTATE` | same, one axis over; the launch DOES reach the rotation | same |
| `MAHAL_DESCENDING` | every fixture is d = 1 or d = 2 | a d >= 3 fixture |
| `LOGDET_FROM_DIAG` | at d = 1 it is `log(1/L00)` against `-log(L00)`, same bits here; at d = 2 the sum is two terms | a d >= 3 fixture with a wide spread of diagonal magnitudes |
| `TOL_ULP` | **the important one, below** | a fixture whose `tol` is an observed likelihood delta |
| `MEANLL_PAIRWISE` | **UNRESOLVED, below** | a direct probe of the two folds |

**`TOL_ULP` is the arm that was supposed to prove the convergence test is
load bearing, and it is unreachable here.** Every fixture converges in TWO
iterations, so the mean log likelihood's change collapses far below `tol` at
once and a one-ulp perturbation cannot flip a comparison that is not close.
The arm is reached; the BOUNDARY is not.

What that costs, stated rather than glossed:
`check_iteration_count_is_identical` still holds and still means something,
since two runs from one seed agree on the count, the flag, the lower bound's
bits and every parameter cell. **It does NOT establish that the convergence
TEST is load bearing**, because nothing here ever stops near the threshold.
This lane's headline identity claim is gated in the converges-immediately
regime only. The closure is cheap: read a mean-log-likelihood delta off the
card and add a fixture whose `tol` IS that delta.

**`MEANLL_PAIRWISE` is UNRESOLVED and is labelled that way on purpose.** It
folds the mean log likelihood pairwise instead of serially over n rows, and
n is in the tens, so unlike the K = 2 and d = 2 arms there is NO
commutativity argument that makes it inert. `_models_agree` does compare
`lower_bound` in bits, so the corrupted quantity is being looked at. It still
moved nothing. Either the arm is not wired into the reported `lower_bound`
or the two folds coincide at these magnitudes, and **neither has been
established**. DEVIATION 1725's pinned fold is NOT gated by this arm.


**BUILT AND GATED ON ONE APPLE M4 IN BOTH MODES, 2026-08-25, AND RUN ON AN
NVIDIA H100 UNDER FAST ON 2026-08-26.**

What does NOT exist is a cross-vendor identity claim. The NVIDIA leg was a
FAST speed leg
(`bench/results/fast_speed/2026-08-26_040100-nvidia-classical.md`), so no
card outside the M4 has ever been compared against another. Where a table
below is an expectation rather than a transcript it is marked as one, and
where an expectation could be wrong in an interesting way it is named rather
than smoothed over. There are three of those:

1. whether `GMM_SAB_TOL_ULP` moves the ITERATION COUNT on any fixture in this
   lane, or only the parameters. It is the arm that demonstrates hazard 1 and
   **it can legitimately come back INERT on the count** if no fixture here
   ever stops within one ulp of `tol`. That would itself be a finding: the
   gate is not exercising the boundary it claims to, and the fixture set
   needs a case that does.
2. whether `GMM_SAB_NO_FTZ_RESP` is inert on Apple (predicted yes, because
   Metal flushes subnormals in hardware, and a responsibility is an
   exponential of a large negative number, so subnormals are exactly what it
   manufactures).
3. whether `GMM_SAB_VENDOR_MATMUL` moves a bit at these shapes (not
   predicted; if it moves none, that is RECORDED and is **not** evidence the
   swap is safe).

<!-- The orchestrator runs these. -->

    pixi run check-mixture                                                    # FAST
    tools/with_build_lock.sh     pixi run mojo run -I . mixture/checks/gmm_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . mixture/checks/gmm_check.mojo

    MOJOLEARN_IDENTITY_TRACE=/tmp/mac.gmm.identical.card \
        tools/with_identical_mode.sh pixi run mojo run -I . mixture/mixture_main.mojo

## THERE IS NO UPSTREAM, AND THAT CHANGES WHICH RULE GOVERNS

**cuML has no Gaussian mixture model.** Verified against the pinned checkout
at `/Users/andrewhendel/CascadeProjects/upstream/cuml-v26.08.00` (commit
`265b9da`, dated 2026-08-05) on 2026-08-25, and this is what was actually
found:

| what was checked | what was found |
|---|---|
| `find . -type d -iname '*gmm*' -o -type d -iname '*mixture*'` | nothing |
| `find . -iname '*gmm*' -o -iname '*mixture*' -o -iname '*gaussian*'` | nothing |
| `ls python/cuml/cuml/` | 33 modules; `cluster`, `decomposition`, `kernel_ridge`, `mixture` is NOT among them |
| `grep -rli 'gaussianmixture\|gaussian_mixture' .` | ONE file: `python/cuml/cuml_accel_tests/upstream/scikit-learn/xfail-list.yaml`, lines 574 and 596, marking `sklearn.mixture.tests.test_gaussian_mixture::test_gaussian_mixture_precisions_init_diag` FLAKY UNDER `cuml.accel`. **That is a record that cuML does not accelerate the estimator and falls through to scikit-learn.** |
| `grep -rli 'expectation.maximization' .` | `cluster/kmeans.pyx` and a notebook. Nothing else |
| cuVS `6ba2ce2`, RAFT `ebf9268` | no GMM either; cuVS's only `gmm`/`mixture` hits are in notebooks |

`ROADMAP.md` records that cuML issue #2034 has been open since April 2020
with no branches and no PRs and that `branch-0.11` never had a `gmm`
directory. **That is consistent with what the checkout shows and it is NOT
independently verified here** -- this lane read the pinned tree and did not
query GitHub. What the tree supports is the weaker and sufficient statement:
at `265b9da` there is no Gaussian mixture model in cuML, cuVS or RAFT.

**So there is nothing to transliterate, and `PORTING_RULES.md`'s COPY DO NOT
IMPROVE does not apply.** No file in this directory is a port, none cites an
upstream line range as a transliteration, and `DERIVATION_MAP.tsv`'s upstream
column says `none` on every row.

What governs instead:

- **`sklearn/mixture/_gaussian_mixture.py` and `_base.py` define the
  SEMANTICS and are the ORACLE. They are never the design source.** Their
  `_estimate_log_gaussian_prob` is written for BLAS and a CPU cache; the GPU
  shape is different, and where it is different this lane says so with a
  number.
- **Every parameter accepted here means exactly what scikit-learn's
  parameter of that name means, or is named differently.** The mapping is the
  table below and it is the contract.
- scikit-learn's implementation is GEMM shaped already,
  `y = (X @ prec_chol) - (mu @ prec_chol)`, and that is followed **because it
  is the arithmetic, not because it is their design**. DEVIATION 1743 keeps
  their spelling over the numerically better `(X - mu) @ P`, and says why.

The upstream pin table, for the record:

| upstream | checkout | commit | what it contributed |
|---|---|---|---|
| cuML | `upstream/cuml-v26.08.00` | `265b9da` | READ; contributes NO code and NO design. Has no mixture model |
| cuVS | `upstream/cuvs-v26.08.00` | `6ba2ce2` | READ; contributes nothing. Same |
| RAFT | `upstream/raft-v26.08.00` | `ebf9268` | READ; contributes nothing directly. Its Cholesky rank-one update reached this lane THROUGH `cholesky/`, which is where it is credited |
| scikit-learn | `upstream/scikit-learn` | `77def0e` | **THE SEMANTICS ORACLE.** `sklearn/mixture/_gaussian_mixture.py` and `_base.py`, cited by line throughout |

## The scikit-learn semantics mapping

Every parameter, and whether it means the same thing. **MEANING-COMPATIBLE**
means the same quantity computed to a different last bit;
**BIT-DIFFERENT** flags where a caller comparing against scikit-learn should
expect different numbers and why.

| scikit-learn | here | same meaning | notes |
|---|---|---|---|
| `n_components` | `GmmParams.n_components` | YES | additionally refused when it exceeds `n_samples`, where they let it through and fail later inside k-means. DEVIATION 1738 |
| `covariance_type="full"` | `COV_FULL` | YES | the only one implemented |
| `covariance_type` in `{tied, diag, spherical}` | REFUSED BY NAME | n/a | `NOT_IMPLEMENTED.tsv` has a row each with what it would cost |
| `tol` | `GmmParams.tol` | YES | float32 rather than a Python float. `abs(lower_bound - prev) < tol` on the mean log likelihood, `_base.py:276` |
| `reg_covar` | `GmmParams.reg_covar` | YES | added to the covariance DIAGONAL after the division by `nk`, their order (`:195-196`). DEVIATION 1736. It is the ONLY ridge: the Cholesky lane's own jitter is passed as `+0.0` (DEVIATION 1737) so `reg_covar = 0` still means what theirs means |
| `max_iter` | `GmmParams.max_iter` | YES | `0` is legal and means initialization only, their behavior. DEVIATION 1745 |
| `init_params="kmeans"` | `INIT_KMEANS` | **MEANING-COMPATIBLE, BIT-DIFFERENT** | theirs runs `sklearn.cluster.KMeans(n_init=1)`, ours runs `cluster/estimator.mojo::kmeans_fit` at the same defaults (`max_iter=300`, `tol=1e-4`, `n_init=1`, k-means++ init). Both one-hot the labels. Two different k-means implementations give two different labelings on ambiguous data |
| `init_params="random"` | `INIT_RANDOM` | **MEANING-COMPATIBLE, BIT-DIFFERENT** | theirs draws from a numpy `RandomState` STREAM; ours is POSITION-MAPPED Philox, because a stream is an order and an order is what three vendors cannot agree on for free. DEVIATION 1733 |
| `init_params` in `{k-means++, random_from_data}` | REFUSED BY NAME | n/a | `NOT_IMPLEMENTED.tsv` |
| `random_state` | `GmmParams.random_state` | **BIT-DIFFERENT** | a `UInt64` seed into k-means or into Philox, not a numpy `RandomState`. The same integer does not produce the same draws |
| `n_init` | ABSENT | n/a | DEVIATION 1734. Their default is 1, so the shipped behavior matches the shipped default |
| `warm_start`, `means_init`, `weights_init`, `precisions_init` | ABSENT | n/a | DEVIATION 1734 |
| `verbose`, `verbose_interval` | ABSENT | n/a | the card is the instrument |
| `weights_`, `means_`, `covariances_`, `precisions_cholesky_` | `GaussianMixtureModel` fields | YES | same shapes, same row-major layout, `covariances_` includes `reg_covar` exactly as theirs does |
| `precisions_` | ABSENT | n/a | `precisions_cholesky_ @ precisions_cholesky_.T` is one line for a caller and a second `d x d` product per component here |
| `n_iter_`, `converged_`, `lower_bound_` | `n_iter`, `converged`, `lower_bound` | YES | **and `n_iter` is on the identity card**, which theirs is not |
| `lower_bounds_` (the per-iteration list) | ABSENT as a field | n/a | it IS on the card, as `gmm.iterNNN.meanll` |
| `fit`, `fit_predict` | `gaussian_mixture_fit` | YES | `fit_predict`'s final extra E-step is not folded in; a caller wanting labels calls `gaussian_mixture_predict` against the returned model, which is the same answer |
| `score_samples`, `score`, `predict`, `predict_proba`, `bic`, `aic` | `gaussian_mixture_*` | YES | `bic`/`aic` use integer parameter counting where theirs divides by `2.0` in float and casts; `d (d+1)` is always even so the two agree at every `d` |
| `sample` | ABSENT | n/a | `NOT_IMPLEMENTED.tsv` |
| `ConvergenceWarning` | `converged` field plus a printed line | **BEHAVIOR-COMPATIBLE** | Mojo has no warning channel. Never a raise; their fit returns a usable model and so does ours. DEVIATION 1746 |
| float64 input | REFUSED | n/a | no float64 on the Apple device column. DEVIATION 1724 |

## WHAT THIS LANE REUSES RATHER THAN REWRITES

Nothing in the list below was re-implemented here, and each was checked for
before a line was written.

| what | the file and entry point it lives in | why not a second copy |
|---|---|---|
| **The Cholesky factorization** of every component's covariance | `cholesky/checks/potrf.mojo::potrf_lower`, sized by `::chol_workspace_floats`, at `CHOL_NB_PINNED` | profile `mojolearn.identical.cholesky.fp32.v1`, gated green on an Apple M4 in both modes at commit `1339da7`. **Its DEVIATION 1634 IS THIS LANE'S COLLAPSE DECISION**: the pivot value is `ftz`-flushed and folded ascending, and the test is `not (s > 0.0)` so NaN fails, both zeros fail and a flushed subnormal fails on every column. A second pivot comparison in this repository would be a second thing that can disagree across vendors, and it would be the thing that decides whether a fit succeeds at all |
| **The triangular solve** producing `L^{-1}` | `cholesky/checks/trsm.mojo::trsm_lower`, against an identity right-hand side | one thread per right-hand-side column, `k` ascending, every divide an `identical_div` and never a reciprocal-times. Already gated by `check_cho_solve_residual` and `check_launch_invariance` there |
| **The log determinant** | `cholesky/checks/potrf.mojo::chol_logdet` | its own docstring says it exists so that "a GP, a KRR and a GMM cannot each invent one", and **this is the GMM**. DEVIATION 1726 takes `-0.5 *` its answer rather than scikit-learn's `sum log(P_jj)`, which is the same number and not the same bits |
| **The Cholesky ridge policy** | `cholesky/checks/potrf.mojo::add_jitter`, called at `+0.0` | DEVIATION 1737. `reg_covar` is this lane's ridge; `add_jitter` is still called for its `ftz` on the diagonal, so a subnormal diagonal cannot be kept on one column and flushed on another |
| **All four matrix products** -- `X . P`, `mu . P`, `resp^T . X`, `scaled^T . diff` | `gemm/checks/gemm_identical.mojo::identical_gemm_into`, sized by `::identical_gemm_workspace_max_floats`, at `OP_NN` and `OP_TN` from `gemm/checks/gemm_oracle.mojo` | profile `mojolearn.identical.gemm.fp32.v1`, **CLOSED on three vendors 2026-08-23** (IDENTITY_PATHS row 40): 62 shapes, eight execution plans, six sabotages, and a 60-stage card bit-identical Apple M4 / H100 / MI325X. The covariance accumulation's k-axis is the SAMPLE axis, which makes it the largest summation order in the lane; it is the gemm lane's fold and not one this lane invents. `linalg.matmul` is REFUSED |
| **The normative answer for those products**, in the oracle | `gemm/checks/gemm_oracle.mojo::gemm_oracle` | `gemm_identical.mojo::contract_partition` is explicit that a second spelling of the leaf rule is a second thing that can be wrong, and records that the shape table already shipped one such re-spelling and got it wrong |
| **THE LOGSUMEXP**, including the shift, the underflow argument, the positional row max and the all-`-inf` guard | `kde/impl/neighbors/kernel_density.mojo::logsumexp_kernel`, and its oracle `kde/checks/kde_oracle.mojo::oracle_logsumexp_row` | **the E-step needs exactly this operation and that lane has already solved both hard parts of it.** The shift against underflow (`check_kde_logsumexp_beats_naive`: naive `-inf`, shifted `-1708.7214` against a float64 `-1708.7213588720942`). The SIGNED-ZERO ROW MAX: a strict `>` from index 0, so the LOWER index survives a tie -- not a hardware `max`, whose `(+0.0, -0.0)` is `-0.0` on Apple and `+0.0` on NVIDIA and AMD, all three measured (row 39). Their DEVIATION 603 (an all-`-inf` row is `-inf`, never `exp(-inf - -inf) = NaN`) is carried over as this lane's DEVIATION 1727. `estep.mojo::logsumexp_kernel` reproduces that construction line for line with `n_train` replaced by `n_components`, and its docstring cites the KDE kernel |
| **The k-means initialization** | `cluster/estimator.mojo::kmeans_fit`, at `n_init=1`, `init=INIT_KMEANS_PLUS_PLUS`, `metric=METRIC_L2_EXPANDED`, `max_iter=300`, `tol=1e-4` | identity certified in its own lane (`pixi run check-kmeans-identity`, `cluster/checks/kmeans_identity_check.mojo`). **The k-means is not this lane's claim.** `gmm.init.resp0` is a card stage precisely so that a divergence in the initialization lands at stage 1 rather than propagating silently, and the card's diagnosis list says to run that lane's gate first |
| **The random initialization's generator** | `core/philox.mojo::philox4x32_10` | cuRAND's `Philox4x32_10`, already ported and checked. This lane supplies only the POSITION MAPPING (counter from `(i, k)`, key from the seed, word 0) and an exact `2^-24` scale |
| **Transcendentals and the arithmetic pins**: `identical_exp`, `identical_log`, `identical_mul`, `identical_mul_add`, `identical_div`, `ftz` | `checks/numerics.mojo` | IDENTITY_PATHS rows 9, 10 and 12. Nothing on a numeric path in this directory calls `std.math`; the only `std.math` here is `log`, `exp` and `sqrt` inside the FLOAT64 host reference and `max` inside the `GMM_SAB_ROWMAX_HARDWARE` sabotage arm, both by design |
| **Stage hashing and the differ** | `core/identity_trace.mojo`: `IdentityTrace`, `record_device`, `record_host`, `record_list_f32`, `record_list_i32`, `first_divergence`, `read_trace_lines` | one hash function per repository. Its tag-uniqueness invariant is what forced the three-digit iteration and component numbering, and what forced `gmm.init.resp0` to name itself apart from `gmm.init.resp` |
| **Pinned block folds**, considered and NOT used | `core/pinned_reduce.mojo` | named here because a reviewer will look for it. **No kernel in this lane folds across threads at all.** The Mahalanobis fold, the logsumexp, the `nk` fold, the `sum(nk)` fold and the mean-log-likelihood fold each keep every sum inside ONE thread, so there is no fold shape to pin and importing one would suggest there is. The only cross-thread combination in the whole lane lives inside `identical_gemm_into`, where the gemm lane's own six sabotages already cover it |
| **Fixed-point accumulation**, considered and NOT used | `checks/fixed_point.mojo` | same. It exists to REPLACE float atomics, and this lane has none. There is no float atomic, no `Atomic.fetch_add`, no warp shuffle, ballot or vote, and no `block.sum` anywhere in `mixture/` |
| **Identity tracing, top-k, distance kernels, sorting** | `neighbors/`, `core/segmented_sort.mojo` | a mixture model needs none of them |

**One duplication is taken and it is named rather than hidden.**
`gmm_mix64` in `gmm_fixture.mojo` is the same three lines as
`kde/checks/kde_fixture.mojo:18`,
`cholesky/checks/cholesky_fixture.mojo:88`,
`holtwinters/checks/hw_fixture.mojo:30` and
`isolation_forest/checks/if_fixture.mojo:32`, with the same splitmix64
constants. That is the established per-lane convention in this tree; the
alternative is a cross-lane import of another lane's fixture file, which
`core/pinned_reduce.mojo` argues against at length for hot files. Five copies
of one hash is a debt and this sentence is the record of it. `_tokey` in
`gmm_check.mojo` is a second copy of `numerics.mojo::_total_order_key`, which
is private to that file; same class of debt, same sentence.

## The identity table (row text for `IDENTITY_PATHS.md`)

Row numbers 67-72 are PROPOSED, not claimed. `IDENTITY_PATHS.md` is not this
lane's file, the orchestrator assigns the numbers, and `cholesky/README.md`
has 60-66 outstanding and unassigned.

| n | path | what is vendor-dependent in the ordinary spelling | what we did | status |
|---|---|---|---|---|
| **67** | **`mixture/` -- THE CONVERGENCE TEST AND THE ITERATION COUNT** (`estep.mojo::meanll_kernel`, `::gmm_convergence_change`, `::gmm_converged`) | **THIS IS THE HAZARD THE LANE EXISTS AROUND.** EM stops when the mean log likelihood moves less than `tol`. That is a DATA-DEPENDENT ITERATION COUNT decided by a float comparison on a folded quantity. If two vendors' mean log likelihood differs by ONE BIT at iteration 40 and the change is sitting within one ulp of `tol`, one stops and the other runs a 41st -- and after that the parameters, the responsibilities and every stage are incomparable. A bitwise gate on the OUTPUT reports a difference whose cause is three stages and one control-flow decision upstream, and no amount of tolerance tuning recovers it | **PIN THE QUANTITY, PIN THE TEST, AND PUT THE COUNT ON THE CARD.** The mean log likelihood is folded in ONE THREAD, ascending, `ftz` at every seam, divided by `n` with one `identical_div` -- never a block fold, a warp primitive or an atomic (DEVIATION 1732). The test itself has ONE SPELLING shared by the device driver and the oracle, which is a deliberate exception to this lane's two-spellings rule and is argued in `estep.mojo`'s banner. The first iteration's change is `+inf` and not a computed NaN (DEVIATION 1747), because a computed NaN carries the vendor's payload and `gmm.iterNNN.change` is a certified stage. `gmm.niter` = `[n_iter, converged, max_iter]` is the LAST record on every card | **CONSTRUCTION 2026-08-25, NOTHING RUN.** `check_iteration_count_is_identical` runs THIRD, before any parameter is compared, and asserts the count, the flag, the lower bound's BITS and every parameter cell across two fits from one seed, then reads the card back and requires a `gmm.niter` record. `GMM_SAB_TOL_ULP` moves the compared value by exactly one ulp and requires the count to move |
| **68** | **`mixture/` -- A COMPONENT CAN COLLAPSE, AND THE DECISION TO FAIL IS A FLOAT COMPARISON** (`mstep.mojo::gmm_precision_cholesky`) | when a component's responsibility mass goes to zero its covariance is singular, `potrf_lower` returns `info != 0`, and scikit-learn raises. **A fit that succeeds on one vendor and fails on another is the worst outcome this lane can produce**: the two runs do not have the same stages, so no bitwise gate of any kind is available between them. The decision rests on a pivot -- a float sum compared against zero -- and a subnormal pivot is positive on a column that keeps subnormals and zero on one that flushes | **MIRROR THE REFUSAL, NEVER RESET, AND INHERIT THE PINNED PIVOT.** DEVIATION 1723: `info != 0` RAISES BY NAME with scikit-learn's own wording plus the ITERATION, the COMPONENT and LAPACK's `info`. The component is not reset, not re-seeded and not given a fallback covariance. The decision itself is `cholesky/`'s DEVIATION 1634, pinned in both halves (the value `ftz`-flushed and folded ascending; the test `not (s > 0.0)`), inherited rather than restated. Components are attempted ASCENDING and the FIRST failure stops the step, so WHICH component failed is a function of the data alone | **CONSTRUCTION, NOTHING RUN.** `check_collapse_is_identical` runs SECOND. `FIX_COLLAPSE` has six bit-identical copies of one point, so at `reg_covar = +0.0` that component's maximum-likelihood covariance is EXACTLY the zero matrix and the first pivot is exactly `+0.0` -- a collapse by arithmetic, not by luck. Two device runs must refuse with the same message; the float32 ORACLE must refuse at the same component with the same `info`; the two partial cards must be identical AND the same length. `GMM_SAB_COLLAPSE_RESET` is the arm that removes the refusal, and it is measured in OUTCOME rather than in bits |
| **69** | **`mixture/` -- THE RESPONSIBILITIES ARE A SOFTMAX AND THE M-STEP SUMS OVER THEM** (`estep.mojo::logsumexp_kernel`, `mstep.mojo::nk_kernel` and the covariance accumulation) | three summation orders in one step. (a) the logsumexp's ROW MAX sees `+0.0` and `-0.0` together, and a hardware `max` answers that differently on Apple than on NVIDIA and AMD (row 39, all three measured). (b) the logsumexp's SUM over components. (c) the WEIGHTED COVARIANCE ACCUMULATION, whose k-axis is the SAMPLE axis -- `n` terms per cell, the largest fold in the lane | **PIN ALL THREE AND SAY WHICH FOLD.** (a) the row max is the POSITIONAL strict `>` from `k = 0`, REUSED from `kde/impl/neighbors/kernel_density.mojo::logsumexp_kernel`, so the LOWER index survives a tie on every vendor; and a row of all `-inf` is `-inf`, never a computed NaN (DEVIATION 1727, KDE's 603). (b) the sum is ASCENDING in `k`, one thread per sample row, so the order is a pure function of `n_components` (DEVIATION 1727). (c) the accumulation is `identical_gemm_into` at `OP_TN`, profile `mojolearn.identical.gemm.fp32.v1`, and `linalg.matmul` is REFUSED (DEVIATION 1729). `nk`'s own fold is one thread per component, ascending in `i` (DEVIATION 1731) | **CONSTRUCTION, NOTHING RUN.** `check_estep_vs_oracle` compares every cell of `mahal`, `wlp`, `rowmax`, `lse`, `logresp` and `meanll` against the serial oracle, and PLANTS mixed-zero rows in both orders at three block sizes because no legal mixture data produces two components with exactly equal weighted log probabilities of opposite-signed zero. `check_mstep_vs_oracle` compares `resp`, `nk`, `weights`, `means` and `covariances` per cell at BOTH scikit-learn divisors. Arms `LSE_DESCENDING`, `LSE_ROTATE`, `ROWMAX_GE`, `ROWMAX_HARDWARE`, `NK_DESCENDING`, `COV_PRESCALE` |
| 70 | **`mixture/` -- the MAHALANOBIS fold** (`estep.mojo::mahal_kernel`) | `sum_j y[i][j]^2` over the feature axis, `n * K` of them per iteration, and it is the inner loop the lane's arithmetic intensity comes from | **PIN.** One thread per sample row, feature axis ASCENDING, one `identical_mul_add` per term, `ftz` at every seam, and NO float crosses a thread boundary -- so launch and batch invariance are properties of the kernel's SHAPE rather than properties a check happens to observe. DEVIATION 1728 | **CONSTRUCTION, NOTHING RUN.** `check_estep_vs_oracle` per cell; `check_launch_invariance` across two threads-per-block choices and 37 floats of poisoned padding; `GMM_SAB_MAHAL_DESCENDING` |
| 71 | **`mixture/` -- the LOG DETERMINANT and the two constants in the answer** (`mstep.mojo::gmm_precision_cholesky`, `estep.mojo`'s `GMM_LOG_2PI_BITS` and `GMM_TEN_EPS_BITS`) | `log|Sigma_k|` is a fold over a device `log` (row 12) and it has TWO legal spellings that differ in bits: `sum_j log(P_jj)` (scikit-learn's) and `-0.5 * 2 sum_j log(L_jj)` (the Cholesky lane's). `d * log(2 pi)` and `10 * finfo(float32).eps` are HOST libm and HOST float constants in scikit-learn (row 18's class) and they are NUMBERS IN THE ANSWER, not guards | **PIN AND CENTRALIZE.** DEVIATION 1726: `-0.5 * chol_logdet(L_k)`, the Cholesky lane's one pinned fold, so this lane does not add a second `log` fold to the tree. DEVIATION 1730: both constants pinned by their float32 BITS (`0x3FEB3F8E`, `0x35A00000`) and bitcast, never written as decimals, because `String(Float32)` does not round trip and a profile constant a log line cannot reproduce is one nobody can check | **CONSTRUCTION, NOTHING RUN.** `check_log_likelihood_by_hand` derives three one-dimensional closed forms from the mathematics and asserts them BIT FOR BIT; `GMM_SAB_LOGDET_FROM_DIAG` drives scikit-learn's spelling and must move bits, which is what makes 1726 a decision rather than a preference |
| 72 | **`mixture/` -- row 39 in a mixture**: signed zeros in the data, subnormal responsibilities, and the one NaN a legal fit could compute | a `-0.0` coordinate is invisible to every tolerance comparison; a responsibility is an exponential of a large negative number, so **subnormals are exactly what the M-step's input is made of**; and `exp(-inf - (-inf))` is the NaN a point far from every component produces | signed zeros are carried in a PLANTED fixture (`FIX_SIGNED_ZERO`) and the mixed-zero ROW is planted directly into the kernel because no legal data reaches it; the `+0.0` seed of every fold and the `+0.0` off-diagonal of the identity right-hand side are stated conventions; `resp = exp(log_resp)` is flushed through `ftz` (DEVIATION 1742) and `GMM_SAB_NO_FTZ_RESP` is the arm; the all-`-inf` row is `-inf` (DEVIATION 1727) and the first iteration's change is `+inf` (DEVIATION 1747), so **no computed NaN can reach a certified stage**; non-finite input is refused by name with the CELL and the BITS before any upload (DEVIATION 1738) | **CONSTRUCTION, NOTHING RUN.** `check_estep_vs_oracle`'s arms (c) and (d); `check_gmm_refusals`' three non-finite cases; `check_mstep_vs_oracle`'s per-cell `resp` comparison |

## The deviations

| # | what |
|---|---|
| **1720** | **THERE IS NO UPSTREAM.** cuML, cuVS and RAFT have no Gaussian mixture model at their pins, verified against the checkout. `PORTING_RULES.md`'s COPY DO NOT IMPROVE does not apply; scikit-learn is the SEMANTICS ORACLE and never the design source |
| 1721 | `covariance_type="full"` only. `tied`, `diag` and `spherical` refused by name, with what each would cost |
| 1722 | *(unspent; the convergence work is numbered 1732 and 1747)* |
| **1723** | **A COLLAPSED COMPONENT RAISES BY NAME AND IS NEVER RESET**, and the decision to fail is `cholesky/`'s pinned pivot (its DEVIATION 1634) inherited rather than restated. The message carries the iteration, the component and LAPACK's `info` |
| 1724 | float32 on the device, float64 only in the host reference. scikit-learn defaults to float64 and its own error message recommends it |
| 1725 | the precision Cholesky is `trsm_lower` against the identity plus a transpose, scikit-learn's `solve_triangular(L, I).T`, and never a `getri`-shaped inverse |
| **1726** | `log_det_chol[k] = -0.5 * chol_logdet(L_k)`, not scikit-learn's `sum_j log(P_jj)`. Same number, different bits, and the Cholesky lane's fold already has an owner |
| **1727** | the logsumexp is KDE's: the shift against underflow, the POSITIONAL strict `>` row max (row 39), the ascending sum, and an all-`-inf` row yielding `-inf` rather than a computed NaN (their DEVIATION 603) |
| 1728 | the Mahalanobis fold is one thread per sample, ascending in the feature axis, `identical_mul_add`, no float across a thread boundary |
| **1729** | the weighted covariance accumulation is `identical_gemm_into` at `OP_TN`; `linalg.matmul` is refused. Its k-axis is the SAMPLE axis, the largest summation order in the lane |
| 1730 | `log(2 pi)` and `10 * finfo(float32).eps` pinned by their float32 BITS and bitcast, never written as decimals |
| 1731 | `nk` is one thread per component, ascending in the sample axis, and `10 eps` is added AFTER the sum rather than seeded before it |
| **1732** | **the mean log likelihood is folded in ONE THREAD, ascending, and divided by `n` with one `identical_div`.** It is the convergence quantity |
| 1733 | `init_params="kmeans"` runs `cluster/`'s identity-certified k-means; `"random"` is POSITION-MAPPED Philox rather than a stream draw; `"k-means++"` and `"random_from_data"` refused by name |
| 1734 | `n_init`, `warm_start`, `means_init`, `weights_init` and `precisions_init` refused BY ABSENCE. `n_init > 1` selects a restart by a float comparison of lower bounds, a second data-dependent branch |
| 1735 | the component MATCHING order in the recovery check is the IEEE totalOrder of the mean vector, lexicographic, ties broken by index -- never slot order |
| 1736 | `reg_covar` is added to the DIAGONAL, AFTER the division by `nk`, scikit-learn's order. Adding it before would scale it by `1/nk` and it would stop meaning their parameter |
| **1737** | the Cholesky lane's own jitter is passed as `+0.0`: `reg_covar` IS this lane's ridge. `add_jitter` is still CALLED, for its `ftz` on the diagonal |
| 1738 | non-finite input, `n_components` out of range at both ends, a negative or NaN `tol` or `reg_covar`, and an infinite `tol` refused by name on the HOST before any launch, naming the cell and the bits |
| 1739 | one point scored alone equals the same point inside a batch, bit for bit. There is no per-batch normalization anywhere in the scoring path and there cannot be one |
| **1740** | the sabotage arms are runtime-selectable through a `sabotage` argument, in a separate file, never reached by any driver. Same construction as `cholesky/checks/chol_sabotage.mojo` |
| 1741 | *(folded into 1734; unspent)* |
| 1742 | `resp = exp(log_resp)` is recomputed on the DEVICE from `log_resp` and recorded as a card stage, rather than kept from the E-step, which never forms it |
| 1743 | `y = (X @ P) - (mu @ P)` is scikit-learn's two-GEMM spelling, KEPT over the numerically better `(X - mu) @ P`, because an answer that differs from the oracle for a reason unrelated to the GPU is the one thing this lane cannot afford |
| 1744 | card tags carry a THREE-DIGIT iteration prefix and a THREE-DIGIT component prefix (`IdentityTrace`'s tag-uniqueness invariant), and `gmm.init.resp0` names itself apart from `gmm.init.resp` |
| 1745 | `max_iter = 0` is legal and means initialization only, scikit-learn's behavior; the card is still emitted |
| 1746 | `ConvergenceWarning` becomes a FIELD and a printed line, never a warning and never a raise |
| **1747** | the convergence change at iteration 1 is `+inf`, not the `-inf - (-inf) = NaN` numpy computes. Same behavior (`abs(change) < tol` is False either way), and a value that can sit in a certified stage |
| 1748-1749 | RESERVED, unspent |

## WHAT THE ORCHESTRATOR MUST WIRE

Nothing outside `mixture/` was edited by this lane. These lines are wanted in
`pixi.toml`, in the file's existing format, beside the other classical lanes'
tasks (they belong next to `check-cholesky` and `cholesky-main`, around line
983):

    check-mixture = "mojo run -I . mixture/checks/gmm_check.mojo"
    mixture-main = "mojo run -I . mixture/mixture_main.mojo"

The IDENTICAL pass is the injector, exactly as for every other gate:
`tools/with_identical_mode.sh pixi run check-mixture`. No `*-identity` task is
wanted.

Also owed by the orchestrator, and none of it is this lane's to do:

1. **`IDENTITY_PATHS.md` rows 67-72**, from the identity table above. They
   follow `cholesky/README.md`'s proposed 60-66, which are also unassigned.
2. **`PORTING.md` / `ROADMAP.md`**: `mixture/` is a new section and appears in
   neither. `ROADMAP.md`'s Gaussian Mixture Models row still reads "demoted
   2026-08-20" with an argument that has since been overturned twice in its
   own cell; it needs a line saying the lane is open.
3. **`UNWIRED.md`**: `mixture/estimator.mojo` has no caller in
   `bindings/_mojolearn_estimators.mojo` or `python/mojolearn/`.
4. **The Python surface**, when a binding is wanted:
   `gaussian_mixture_fit`, `gaussian_mixture_score_samples`,
   `gaussian_mixture_score`, `gaussian_mixture_predict`,
   `gaussian_mixture_predict_proba`, `gaussian_mixture_bic`,
   `gaussian_mixture_aic`, plus `covariance_type_from_name` and
   `init_params_from_name` so the refusals reach Python BY NAME. Refuse
   `float64` inputs, `n_init`, `warm_start` and the three `*_init` keywords
   in Python, by name.
5. **A `tag_prefix` argument on `cholesky/checks/potrf.mojo::potrf_lower`
   and `::chol_logdet`.** THIS LANE DID NOT MAKE IT and the change is named
   rather than done, because `cholesky/` is not this lane's directory. Those
   two hardcode the tags `chol.panelNNN.*`, `chol.factor`, `chol.nb`,
   `chol.diag` and `chol.logdet`, so calling them `n_components` times per
   iteration violates `IdentityTrace`'s tag-uniqueness invariant and raises on
   the second component. `gmm_precision_cholesky` therefore calls them with a
   DISABLED trace and records their OUTPUTS under its own tags -- which is
   arguably the right card anyway, since a `chol.panelNNN.trailing` moving is
   the gemm lane's certificate and a `chol.panelNNN.factored` moving is the
   Cholesky lane's. A `tag_prefix` would let a caller nest their stages inside
   a per-component card when it is a Cholesky bug being chased.
   `trsm_lower` already takes a `tag` and needs nothing.

## SABOTAGES TO PERFORM

All thirteen are selected at RUN TIME through the `sabotage` argument
(`mixture/checks/gmm_sabotage.mojo`), copying
`cholesky/checks/chol_sabotage.mojo`'s construction. **No source edit and
no rebuild is required for any of them**; `check_gmm_sabotages` drives all
thirteen in one run and prints a line per arm. These arms were driven on the
Apple M4 on 2026-08-25 in both modes (Status). The classifications below are
what each arm is expected to do rather than a transcript of what it did.

The arms are driven in three GROUPS and the reason is not tidiness. Group (A)
sweeps every fixture, because `cholesky/README.md` records that lane shipping
two arms that were reached and provably INERT on the fixture its author
picked, and an inert arm is indistinguishable from an unreached one. Group
(B) cannot be reached from any legal fixture at all and is driven against a
PLANTED matrix through a direct kernel launch; sweeping fixtures for those
three would report "inert everywhere" and prove nothing. Group (C) is not
measured in bits.

| group | check it targets | sabotage | exactly what it corrupts | what must move |
|---|---|---|---|---|
| A | `check_iteration_count_is_identical` | **`GMM_SAB_TOL_ULP`** | the value compared against `tol` is moved ONE ULP toward zero, on the host. No parameter and no probability changes; only whether the loop stops | **THE ITERATION COUNT.** MUST MOVE something on at least one fixture. If it moves the parameters but NOT the count on any fixture, that is recorded as the weaker result and hazard 1 is not yet demonstrated on this hardware -- because it means no fixture here stops within one ulp of `tol`, which is a gap in the FIXTURE SET rather than in the gate |
| A | `check_iteration_count_is_identical`, `check_estep_vs_oracle` | `GMM_SAB_MEANLL_PAIRWISE` | the mean log likelihood folds PAIRWISE (an equal-rank merge) instead of ascending serial. Same multiset, different bracketing | the convergence quantity's bits, and through them the iteration count. MUST MOVE |
| A | `check_estep_vs_oracle` | `GMM_SAB_LSE_DESCENDING` | the logsumexp's sum walks the components DESCENDING | the `lse` bits on any fixture with `K > 1` and unequal terms. MUST FAIL |
| A | `check_estep_vs_oracle`, `check_launch_invariance` | `GMM_SAB_MAHAL_DESCENDING` | the Mahalanobis fold walks the feature axis DESCENDING | the `mahal` bits on any fixture with `d > 1`. MUST FAIL. **Note it is expected INERT on `FIX_ONE_D`**, which has one feature and therefore one term -- exactly the kind of cell the fixture sweep exists to skip past rather than be defeated by |
| A | `check_mstep_vs_oracle` | `GMM_SAB_NK_DESCENDING` | the `nk` fold walks the sample axis DESCENDING. `nk` divides the mean and the covariance | every M-step output. MUST FAIL |
| A | `check_mstep_vs_oracle` | `GMM_SAB_COV_PRESCALE` | the division by `nk` moves from AFTER the outer-product accumulation to INTO the responsibility before it. One rounding moves from after the sum to before it, on every term | the `covariances` bits. MUST FAIL |
| A | `check_estep_vs_oracle`, `check_log_likelihood_by_hand` | `GMM_SAB_LOGDET_FROM_DIAG` | `log_det_chol` is computed as scikit-learn spells it, `sum_j log(P_jj)`, instead of `-0.5 * chol_logdet(L)` | the `wlp` bits. MUST FAIL, and that is what makes DEVIATION 1726 a decision rather than a preference: two legal spellings of one number, and a lane that used one while its oracle used the other would disagree for a reason nobody would find |
| A | `check_mstep_vs_oracle` | `GMM_SAB_NO_FTZ_RESP` | `ftz` dropped on `resp = exp(log_resp)` | **APPLE-INERT, predicted.** Metal flushes subnormals in hardware. On NVIDIA and AMD, both of which KEEP subnormals (row 39, measured), the unflushed responsibilities survive into `nk` and into the covariance. **This is the lane's denormal site**: a responsibility is an exponential of a large negative number, so subnormals are what the M-step's input is made of. RECORDED, never claimed |
| A | `check_estep_vs_oracle` | `GMM_SAB_VENDOR_MATMUL` | the E-step's `X . P` calls `core/gemm.mojo::gemm_nt`, i.e. MAX `linalg.matmul`. It needs no transpose of its own: `P^T` is exactly the `L^{-1}` the driver still holds, so the arm differs from the production path in the LIBRARY and in nothing else | REPORT. If it moves NO bit, that is RECORDED and is **not** evidence the swap is safe: it means the vendor happened to pick this profile's order at these shapes on this device, which is the thing no vendor guarantees |
| B | `check_estep_vs_oracle` (row 39) | `GMM_SAB_ROWMAX_GE` | the row max becomes `>=`, so the HIGHER index wins a tie | the `rowmax` bits on the PLANTED mixed-zero rows. MUST FAIL, and it fails on every vendor because it is a compare and not a hardware `max` |
| B | `check_estep_vs_oracle` (row 39) | `GMM_SAB_ROWMAX_HARDWARE` | the row max becomes the hardware `max(acc, v)` | **THE VENDOR-SPLIT ARM.** `max(+0.0, -0.0)` is `-0.0` on Apple (the second operand) and `+0.0` on NVIDIA and AMD (IEEE-2019 maximum), all three MEASURED. So this line is EXPECTED TO READ DIFFERENTLY on the three vendors, and that is the finding rather than a failure. REPORT |
| B | `check_launch_invariance` | `GMM_SAB_LSE_ROTATE` | the logsumexp's sum starts at `k0 = block_idx.x % K` and wraps, so the order is a function of LAUNCH GEOMETRY | MUST FAIL at `row_tpb = 8` on a 24-row matrix (three blocks) and be INERT at `row_tpb = 128` (one block). **Both halves are reported in one line**, because "inert" and "fails" are the same arm at two launches and printing only one of them is how a launch-dependent order hides from a gate |
| C | `check_gmm_refusals`, `check_collapse_is_identical` | **`GMM_SAB_COLLAPSE_RESET`** | a component whose Cholesky returns `info != 0` is silently RESET to the identity covariance and the fit continues | **THE OUTCOME.** The fit must go from REFUSED to SUCCEEDED. Not measured in bits at all, because that is the point: a fit that succeeds on one vendor and refuses on another produces two runs with different stages, and no bitwise gate can see it |

**Two arms the brief's usual list asks for have no object here and it is
worth saying why: "swap the two fold levels" and "change the block
reduction's width". There is no block reduction anywhere in this lane.** The
Mahalanobis fold, the logsumexp, the `nk` fold, the `sum(nk)` fold and the
mean-log-likelihood fold all keep every sum inside one thread; the only
cross-thread combination in the whole lane lives inside `identical_gemm_into`,
where the gemm lane's own six sabotages already cover it. That absence is the
reason launch invariance is a property of shape here rather than a property a
check happens to observe.

## What the checks are expected to establish

Ten checks, and the ORDER is an argument: the iteration count and the collapse
outcome are gated BEFORE any parameter, because two runs that took different
numbers of iterations, or that disagreed about whether the fit failed at all,
have no parameters worth comparing.

| check | what it would establish |
|---|---|
| `check_gmm_refusals` | thirteen refusals BY NAME -- three covariance types, three `init_params`, three non-finite inputs (naming the CELL), `n_components` at both ends, `tol`, `reg_covar`, and a collapsed component citing DEVIATION 1723 with its component and its `info`; **and the acceptances**, so the refusals are not simply always firing: `full`, `kmeans`, `random`, and the same collapse fixture at `reg_covar = 1e-6` fitting normally |
| `check_collapse_is_identical` | **runs SECOND.** `FIX_COLLAPSE` at `reg_covar = +0.0` refuses on the device and on the float32 oracle, at the same component with the same `info`; two runs' messages are identical; the two partial cards are identical AND the same length |
| `check_iteration_count_is_identical` | **runs THIRD, and it is the headline.** Two fits from one seed agree on the iteration count, the convergence flag, the lower bound's BITS and every parameter cell, on every fixture that fits; and the card carries a `gmm.niter` record |
| `check_estep_vs_oracle` | the log probabilities and the responsibilities PER CELL, bit for bit under IDENTICAL (a REPORT under FAST); the naive log-sum-exp underflowing to `-inf` where the shifted one does not; an all-`-inf` row giving `0xFF800000` on device AND oracle; and PLANTED mixed-zero rows in both orders at three block sizes keeping the LOWER-INDEX zero's bits on both |
| `check_mstep_vs_oracle` | `resp`, `nk`, `weights`, `means` and `covariances` per cell, at BOTH scikit-learn divisors (`_initialize`'s `n_samples` and `_m_step`'s `sum(nk)`), because a non-default path is an unchecked path |
| `check_log_likelihood_by_hand` | three one-dimensional closed forms derived from the mathematics and asserted BIT FOR BIT -- `x` at the mean, `x` one away, and the same distance reached through the MEAN rather than the point, so the subtraction and the point are exercised separately and required to agree -- plus a fourth with unequal weights (so the row max is component 1 and the shift is not zero) against the float64 reference |
| `check_recovers_planted_parameters` | the SEPARATED fixture's fitted parameters equal the PLANT, matched by the IEEE totalOrder of the mean vector and never by slot; `predict` giving one label per planted cluster and `predict_proba` being exactly one-hot (which is only assertable because the cross-cluster exponentials underflow float32 to exactly zero at this separation); `score`, `bic` and `aic` finite with `bic > aic` |
| `check_launch_invariance` | whole fits at `elem/row/comp/solve/panel` tpb `256/128/32/256/128` against `64/32/8/8/32`, with the iteration count and every parameter cell compared; E-steps at 0 against 37 floats of POISONED padding; **every row scored ALONE equal to the same row inside its batch**; and the logsumexp equal at one block and three |
| `check_card_is_emitted` | the record count matching the documented layout `1 + 1 + 5 + 2K + 1 + n_iter (13 + 2K) + 1`, `gmm.input` first, `gmm.niter` LAST, and two runs byte-identical |
| `check_gmm_sabotages` | all thirteen arms driven at run time, each classified MUST FAIL, MUST MOVE, OUTCOME, APPLE-INERT or REPORT in advance, with the fixture each moved on and how many were inert to it |

## The card

`mixture/mixture_main.mojo` emits, for `K` components and `n_iter`
iterations, `8 + 2K + n_iter (13 + 2K) + 1` stages. The full list and the
per-stage diagnosis are in that file's header. The short version:

    gmm.input, gmm.init.resp0,
    gmm.init.{resp,nk,weights,means,covariances},
    gmm.init.comp000.{cholesky,precchol} ... gmm.init.logdet,
    gmm.iter001.{mahal,wlp,rowmax,lse,logresp,meanll},
    gmm.iter001.{resp,nk,weights,means,covariances},
    gmm.iter001.comp000.{cholesky,precchol} ... gmm.iter001.logdet,
    gmm.iter001.change,
    ... one block per iteration ...,
    gmm.niter

**COMPARE `gmm.niter` FIRST.** Two cards that disagree there have no
comparable parameters below the first differing iteration, and
`core/identity_trace.mojo::first_divergence` will correctly report a
STRUCTURAL divergence and send you to `tools/identity_trace_diff.py` rather
than diffing by position.

**No card has been emitted.** The list above is what the source records, not
a transcript.

## WHAT THIS WILL COST

**Under IDENTICAL this lane is expected to be SLOWER than scikit-learn on the
CPU, and the reason is measured and is not this lane's arithmetic.**

`bench/results/SPEED_LANE_2026-08-25.md` section 3.4 measured this
repository's identical FP32 GEMM at **2% of FP32 peak on Apple: 78 GF/s
against MPS at 2,332 GF/s**, and at 3 to 5% of peak on an H100 and an MI325X.
The consistency across three vendors is the result worth reading: the gap is
structural to that kernel rather than an accident of one backend.

The full-covariance E-step is GEMM shaped -- `X . P` is `n x d` by `d x d`
per component per iteration, and the M-step's covariance accumulation is
`d x d` with an `n`-long k-axis -- so **this lane inherits that 2% directly**,
on the arithmetic that dominates it. There is no version of this in which a
kernel at 2% of peak beats a well-tuned CPU BLAS on a small problem.

Three things follow and all three are statements about what is claimed, not
excuses:

1. **Any speed claim lives in FAST.** FAST calls MAX's `linalg.matmul`, which
   on the Apple M4 measured 39.8 ms against our 771.5 ms at
   `llama8b.mlp_up.t512`. That arm is where a benchmark against scikit-learn
   belongs, and it has not been run either.
2. **IDENTICAL is a reproducibility mode with a priced cost, and the price is
   real on every vendor.** The standing rule
   `[[identity-is-not-free]]` is explicit: conforming COSTS ON EVERY VENDOR,
   and "identity is free on Apple" is a banned sentence. The gemm lane's own
   measurement (legs 16 and 17) is that the fold pin alone costs **1.52x on
   the M4, 2.31x on the H100 and 2.06x on the MI325X** against its own
   unpinned kernel, before any comparison with a vendor library. This lane
   adds its own IDENTICAL-only costs on top -- the `ftz` at every seam, the
   one-thread folds, the per-component Cholesky's two host drains, and the
   one drain per iteration for the convergence test -- and none of them has
   been measured.
3. **`ROADMAP.md` says this has the best arithmetic intensity of anything
   remaining, and that is not in conflict with any of the above.**
   `d^2` FLOPs per point per component against k-means' `~d` is a statement
   about where the ROOM is. `SPEED_LANE_2026-08-25.md` is a statement about
   how much of that room a particular kernel has taken. `ROADMAP.md`'s own
   filter section says exactly this after being wrong twice: "arithmetic
   intensity says which algorithms have room, not how much of it a given
   implementation has already taken. A 0.45x is evidence about our kernel,
   not about the hardware ceiling." **The intensity argument is why the lane
   is worth opening. The 2% is why nothing here is a speed claim yet.**

One further cost that is this lane's alone and is not the GEMM's:
`gmm_precision_cholesky` factors `n_components` matrices ONE AT A TIME, with
two host drains per component per iteration (`potrf_lower` reads `info` back
per panel; `chol_logdet` reads its scalar back). At `K = 8` and 40 iterations
that is 640 drains. A batched per-component factorization would collapse it to
two drains per iteration. It is named under WHAT IS OWED rather than
attempted, because the batched factorization has never been measured against
the present one and `PORTING_RULES.md` rule 2's corollary is about exactly
this shape of debt.

## WHAT IS OWED

1. **THE IDENTICAL-MODE LEG ON A SECOND VENDOR.** This lane has been
   compiled and executed. It is built and gated on one Apple M4 in both
   modes (Status, 2026-08-25), and an NVIDIA H100 compiled and ran it under
   FAST on 2026-08-26 at fixture `SEPARATED.24x2K3`, median 2.390 ms with no
   opponent on that box
   (`bench/results/fast_speed/2026-08-26_040100-nvidia-classical.md`). A
   FAST leg is a speed leg and buys no identity; there is still no
   IDENTICAL-mode card from any vendor except the M4, which is item 2.
2. **The second and third vendor legs.** An NVIDIA H100 and an AMD MI325X run
   of `mixture_main.mojo` under `tools/with_identical_mode.sh`, and
   `tools/identity_trace_diff.py` over the three cards, with `gmm.niter`
   compared FIRST. Until then there is no cross-vendor claim of any kind. Two
   arms are expected to behave differently there and are recorded that way
   rather than as passes: `GMM_SAB_NO_FTZ_RESP` (NVIDIA and AMD keep
   subnormals) and `GMM_SAB_ROWMAX_HARDWARE` (the `max(+0, -0)` split).
3. **Whether any fixture here stops within one ulp of `tol`.** Deliberately
   not predicted. `GMM_SAB_TOL_ULP` is the arm that demonstrates hazard 1 and
   it can legitimately come back moving the parameters but not the count. The
   check prints which happened; if it is the weaker result, the fixture set
   owes a case that sits on the boundary, and that case has to be constructed
   rather than searched for.
4. **A LATE collapse.** `FIX_COLLAPSE` fails at the INITIALIZATION, which is
   deterministic and hand-derivable and is stated as such. A component that
   collapses at EM iteration 7 is the more interesting case and it is not
   constructible without running the fit to see where it goes.
   `check_collapse_is_identical` asserts "the same iteration and the same
   partial state" whatever iteration that is, so it would cover a late
   collapse if one were planted; none is.
5. **A batched per-component Cholesky**, and the drain count above.
6. **Larger `n`, larger `d`, larger `K`.** The largest fixture is 24 points in
   two dimensions at three components. Nothing here says what happens at
   `n = 100,000`, `d = 64`, `K = 32`, where the covariance GEMM's `k`-axis is
   100,000 and `identical_gemm_into` will pick a SPLITK plan that no fixture
   in this lane reaches. `check_launch_invariance` at a shape that crosses
   that dispatch boundary is owed.
7. **A benchmark, in FAST, against a GPU baseline.** The NVIDIA H100 row of
   2026-08-26 has no opponent because RAPIDS ships no `GaussianMixture`, so
   the comparison is owed on Apple, following
   `[[gpu-baseline-if-it-exists]]`: scikit-learn's own Array API path reaches
   MPS under `init_params="random"`, so **the GPU baseline exists and must be
   the comparison** -- with the caveat that
   `bench/results/SKLEARN_GPU_BASELINE_2026-08-20.md` measured that path 1.22x
   SLOWER than torch CPU, so both arms are owed. `[[large-data-runs-default]]`
   applies: the benchmark uses the shipped size and a smaller run is flagged
   beside the number.
8. **`covariance_type` in `{tied, diag, spherical}`**, each refused by name
   today with a row in `NOT_IMPLEMENTED.tsv` saying what it would cost.
9. **`BayesianGaussianMixture`**, which shares `_base.py`'s loop and therefore
   hazard 1 exactly. Whoever opens it should read this file's hazards first.
