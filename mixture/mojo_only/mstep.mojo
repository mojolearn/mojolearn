"""The M-step: the weighted moments, and the precision Cholesky that can fail.

NOT A PORT. See `mixture/mojo_only/estep.mojo`'s header for the upstream
finding: cuML, cuVS and RAFT have no Gaussian mixture model at their pins, so
COPY DO NOT IMPROVE has nothing to copy and
`sklearn/mixture/_gaussian_mixture.py` is the SEMANTICS ORACLE only.

WHAT SCIKIT-LEARN DOES, AND IN WHAT ORDER
------------------------------------------
`_estimate_gaussian_parameters` (`:282-320`) followed by
`_compute_precision_cholesky` (`:323-385`), which `_m_step` (`:883-901`) and
`_initialize` (`:849-881`) both call:

    nk    = sum(resp, axis=0) + 10 * finfo(dtype).eps        # `:316`
    means = (resp.T @ X) / nk[:, newaxis]                     # `:317`
    diff  = X - means[k]                                      # `:194`
    cov_k = ((resp[:, k] * diff.T) @ diff) / nk[k]            # `:195`
    cov_k[j][j] += reg_covar                                  # `:196`
    weights /= n_samples        (initialize)                  # `:864`
    weights /= sum(weights)     (m_step)                      # `:898`
    L_k   = cholesky(cov_k, lower=True)   -> ValueError on failure
    P_k   = solve_triangular(L_k, I, lower=True).T            # `:368-370`

Every one of those lines is reproduced below in that order. Every place this
file departs from it carries a DEVIATION number, and the reason is never "we
think this is better".

THE FOUR PINS IN THIS FILE
----------------------------
**(a) `nk`'s fold** (`nk_kernel`). One thread per component, sample axis
ASCENDING, `ftz` at every seam. `nk` divides both the mean and the
covariance, so its last bit is in every parameter. DEVIATION 1731.

**(b) The weighted covariance accumulation.** `sum_i resp[i][k] * diff[i]
outer diff[i]` is a matrix product and its k-axis is the SAMPLE axis, which
makes it the largest summation order in the lane -- `n` terms per cell,
against `d` in the Mahalanobis fold and `K` in the logsumexp. It goes through
`identical_gemm_into` at `OP_TN`, profile `mojolearn.identical.gemm.fp32.v1`,
and `linalg.matmul` is REFUSED. The fold shape is therefore the gemm lane's,
already gated at 62 shapes across eight execution plans with six sabotages,
and this lane inherits that certificate rather than re-earning it.
DEVIATION 1729.

**(c) `reg_covar` is added AFTER the division, on the diagonal only.**
scikit-learn's `:195` then `:196`. Adding it before, or adding it to every
cell, or scaling it by anything, is a different matrix. DEVIATION 1736.

**(d) The precision Cholesky routes through the `cholesky/` lane, entirely.**
`potrf_lower` for the factorization, `trsm_lower` for the inverse,
`chol_logdet` for the log determinant. Nothing is re-implemented here. See
`mixture/README.md`'s WHAT THIS LANE REUSES RATHER THAN REWRITES.

============ DEVIATION 1723 (2026-08-25): A COLLAPSED COMPONENT RAISES BY
============ NAME, AND THE DECISION TO FAIL IS ITSELF PINNED ==============
THEIRS (`_compute_precision_cholesky:363-367`): `np.linalg.LinAlgError` from
`scipy.linalg.cholesky` is caught and re-raised as a `ValueError` reading
"Fitting the mixture model failed because some components have ill-defined
empirical covariance (for instance caused by singleton or collapsed samples).
Try to decrease the number of components, increase reg_covar, or scale the
input data."

OURS: the same refusal, by name, carrying the ITERATION, the COMPONENT and
LAPACK's `info`. The component is NOT silently reset, NOT re-seeded, and NOT
given a fallback covariance. `GMM_SAB_COLLAPSE_RESET` is the arm that does
reset it, and it exists so the gate can be shown to see the difference.

WHY: **a fit that succeeds on one vendor and fails on another is the worst
outcome this lane can produce.** The failure is decided by a float
comparison against a pivot -- `cholesky/mojo_only/potrf.mojo`'s DEVIATION
1634 -- and that comparison is pinned in both halves there: the value is
`ftz`-flushed and folded ascending through `identical_mul_add`, and the test
is `not (s > 0.0)`, so NaN fails, both zeros fail, and a flushed subnormal
fails on every column. This lane inherits that pin rather than restating it,
which is the whole reason `potrf_lower` was written before this file existed.
A silent reset would take a DECISION that is currently identical on three
vendors and turn it into a VALUE that is not.
==========================================================================

============ DEVIATION 1726 (2026-08-25): log_det_chol COMES FROM
============ chol_logdet, NOT FROM THE PRECISION DIAGONAL =================
THEIRS (`_compute_log_det_cholesky:470-476`): `sum_j log(P[k][j][j])` over
the precision Cholesky's own diagonal, which for `covariance_type="full"`
they extract with a stride-`(d + 1)` slice of the flattened matrix.
OURS: `-0.5 * chol_logdet(L_k)`, where `chol_logdet` is the Cholesky lane's
pinned single-thread ascending fold through `identical_log`.
WHY: the two are the same quantity -- `P[j][j] = 1 / L[j][j]`, so
`sum log(1/L_jj) = -sum log(L_jj) = -0.5 * (2 sum log L_jj)` -- and they are
NOT the same float32, because `log(1/x)` and `-log(x)` round differently.
Given two spellings of one number, the rule in this repository is to use the
one that already has an owner: `chol_logdet`'s own docstring says it exists
so that "a GP, a KRR and a GMM cannot each invent one", and this is the GMM.
Taking the diagonal spelling instead would put a second `log` fold in the
tree (IDENTITY_PATHS rows 12 and 21) for a quantity that already has a pinned
one. `GMM_SAB_LOGDET_FROM_DIAG` drives their spelling so the check can show
that the choice moves bits and is therefore a decision rather than a
preference.
==========================================================================
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from cholesky.mojo_only.potrf import (
    CHOL_ELEM_TPB,
    CHOL_NB_PINNED,
    CHOL_PANEL_TPB,
    add_jitter,
    chol_logdet,
    chol_workspace_floats,
    potrf_lower,
)
from cholesky.mojo_only.trsm import CHOL_SOLVE_TPB, trsm_lower
from core.identity_trace import IdentityTrace
from gemm.mojo_only.gemm_identical import (
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.mojo_only.gemm_oracle import OP_TN
from mixture.mojo_only.estep import (
    GMM_COMP_TPB,
    GMM_ELEM_TPB,
    gmm_comp_tag,
    gmm_ten_eps,
    set_identity_kernel,
    transpose_square_kernel,
)
from mixture.mojo_only.gmm_sabotage import (
    GMM_SAB_COLLAPSE_RESET,
    GMM_SAB_COV_PRESCALE,
    GMM_SAB_LOGDET_FROM_DIAG,
    GMM_SAB_NK_DESCENDING,
    GMM_SAB_NONE,
    GMM_SAB_NO_FTZ_RESP,
    sabotage_center_scale_kernel,
    sabotage_logdet_from_diag_kernel,
    sabotage_nk_kernel,
    sabotage_resp_exp_kernel,
)
from mojo_only.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_log,
    identical_mul,
)


#: **NUMERIC. DEVIATION 1737.** The ridge handed to the Cholesky lane, which
#: is `+0.0`, because `reg_covar` IS this lane's ridge and it has already
#: been added to the covariance's diagonal by `cov_finish_kernel` exactly
#: where scikit-learn adds it.
#:
#: Adding `cholesky/`'s own `CHOL_JITTER_PINNED` on top would be a SECOND
#: ridge that the caller never asked for and that no scikit-learn parameter
#: names, so `reg_covar = 0` would stop meaning what scikit-learn's
#: `reg_covar = 0` means -- which `mixture/README.md`'s semantics table
#: forbids. `add_jitter` is still CALLED with this value rather than skipped,
#: because it also flushes the diagonal through `ftz`, and a diagonal that
#: one column's hardware would keep as a subnormal and another's would flush
#: is the divergent-outcome failure one step before the pivot test.
comptime GMM_CHOL_JITTER = Float32(0.0)


@fieldwise_init
struct GmmMStepRun(Copyable, Movable):
    """What one M-step actually did, READ BACK FROM THE RUN.

    `info` and `failed_component` are not diagnostics. They are the
    DATA-DEPENDENT decision DEVIATION 1723 is about, and
    `check_collapse_is_identical` compares them between two runs and against
    the oracle before it compares any parameter.
    """

    var info: Int
    """LAPACK's `info` from the FIRST component whose Cholesky failed, or 0.
    `k > 0` means the leading minor of order `k` of that component's
    covariance was not positive definite."""

    var failed_component: Int
    """Which component that was, or `-1`. Components are attempted in
    ASCENDING order and the first failure stops the step, so this is a
    function of the data and of nothing else -- not of a launch, not of a
    schedule, not of which component happened to be scanned first."""


# ===========================================================================
# THE KERNELS
# ===========================================================================


def resp_exp_kernel(
    logresp: MutPointer[Float32, MutAnyOrigin],
    resp: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    ncomp_in: Int32,
):
    """`resp = exp(log_resp)`, one thread per cell. `_m_step:896`'s
    `xp.exp(log_resp)`.

    **DEVIATION 1742: this is RECOMPUTED on the device from `log_resp` and
    recorded as a card stage, rather than being kept from the E-step.** The
    E-step never forms it -- `_estimate_log_prob_resp` returns logs -- so
    there is nothing to keep, and materializing it here means the M-step's
    actual input is a hashed stage. A card whose first divergent stage is
    `resp` and whose `logresp` matched localizes the difference to this one
    `exp`, which is IDENTITY_PATHS row 12's site and is exactly the kind of
    one-ulp vendor difference the instrument exists to find.

    `ftz` on the result matters more here than almost anywhere else in the
    lane: **a responsibility is an exponential of a large negative number, so
    this is where subnormals are manufactured.** A component a point does not
    belong to gets a `log_resp` of a few hundred negative, and `exp` of that
    lands in the subnormal range before it reaches zero. On a
    denormal-honoring column those values survive into `nk` and into the
    covariance; on a flushing one they are zero. `GMM_SAB_NO_FTZ_RESP` is the
    arm, and it is expected INERT on Apple and to move bits on NVIDIA and
    AMD, which is recorded rather than claimed.
    """
    var n = Int(n_in)
    var ncomp = Int(ncomp_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= n * ncomp:
        return
    resp.unsafe_store(idx, ftz(identical_exp(ftz(logresp.unsafe_load(idx)))))


def nk_kernel(
    resp: MutPointer[Float32, MutAnyOrigin],
    nk: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    ncomp_in: Int32,
    ten_eps: Float32,
):
    """`nk[k] = sum_i resp[i][k] + 10 eps`, ONE THREAD PER COMPONENT.

    `_estimate_gaussian_parameters:316`. **DEVIATION 1731: the sample axis
    walks ASCENDING and the whole sum lives in one thread's register.** No
    block fold, no warp primitive, no atomic, so the order is a pure function
    of `n` and of nothing about the launch.

    `10 * finfo(float32).eps` is added AFTER the sum, not seeded before it,
    which is scikit-learn's order and is not the same number: seeding would
    put the constant through every one of the `n` roundings. It is a NUMBER
    IN THE ANSWER rather than a guard -- it divides the mean and the
    covariance of every component, including components with real mass -- so
    it is pinned by bits as `GMM_TEN_EPS_BITS` (DEVIATION 1730).

    ONE THREAD PER COMPONENT rather than one per sample plus a reduction, for
    the same reason `meanll_kernel` is one thread: `K` is small, the work is
    `n` adds, and a fold shape is a summation order that would then have to
    be pinned, checked and sabotaged. A lane that needs zero fold shapes
    should have zero.
    """
    var n = Int(n_in)
    var ncomp = Int(ncomp_in)
    var k = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if k >= ncomp:
        return
    var acc = Float32(0.0)
    for i in range(n):
        acc = ftz(acc + ftz(resp.unsafe_load(i * ncomp + k)))
    nk.unsafe_store(k, ftz(acc + ten_eps))


def nk_total_kernel(
    nk: MutPointer[Float32, MutAnyOrigin],
    out_scalar: MutPointer[Float32, MutAnyOrigin],
    ncomp_in: Int32,
):
    """`sum_k nk[k]`, in ONE THREAD, ascending. `_m_step:898`'s
    `xp.sum(self.weights_)`.

    **This is NOT `n_samples`, and the difference is deliberate.**
    `sum_k nk[k]` is `n + K * 10 * eps` in exact arithmetic and something
    within a rounding of it in float32, and scikit-learn's `_m_step` divides
    by THIS while `_initialize` divides by `n_samples` (`:864`). Two
    divisors, two code paths, one estimator. Following that exactly is what
    `mixture/README.md`'s semantics table promises, and computing it here
    rather than assuming `n` is what keeps the promise.
    """
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    var ncomp = Int(ncomp_in)
    var acc = Float32(0.0)
    for k in range(ncomp):
        acc = ftz(acc + ftz(nk.unsafe_load(k)))
    out_scalar.unsafe_store(0, acc)


def weights_kernel(
    nk: MutPointer[Float32, MutAnyOrigin],
    denom: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    log_weights: MutPointer[Float32, MutAnyOrigin],
    ncomp_in: Int32,
):
    """`weights = nk / denom` and `log_weights = log(weights)`, one thread
    per component.

    `denom` is a DEVICE SCALAR rather than a kernel argument so that the two
    call sites -- `n_samples` at initialization, `sum(nk)` in the loop -- go
    through one kernel and one spelling. A host float passed as an argument
    would have been simpler and would have meant the loop's divisor made a
    round trip to the host and back every iteration for no reason.

    `identical_div` and never a multiply by a reciprocal: two roundings where
    a divide is one, the same argument `cholesky/mojo_only/trsm.mojo`'s
    DEVIATION 1643 makes.

    `log(weights)` is computed HERE rather than in the E-step, because it is
    a per-component quantity that the E-step would otherwise recompute for
    every one of `n * K` cells. `identical_log` at IDENTITY_PATHS row 12.
    A weight is strictly positive -- `nk` carries `10 eps` and `denom` is a
    sum of such -- so `identical_log`'s `-inf` and NaN branches are
    unreachable from a state this kernel can be called in.
    """
    var ncomp = Int(ncomp_in)
    var k = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if k >= ncomp:
        return
    var w = ftz(identical_div(ftz(nk.unsafe_load(k)), ftz(denom.unsafe_load(0))))
    weights.unsafe_store(k, w)
    log_weights.unsafe_store(k, ftz(identical_log(w)))


def means_divide_kernel(
    raw: MutPointer[Float32, MutAnyOrigin],
    nk: MutPointer[Float32, MutAnyOrigin],
    means: MutPointer[Float32, MutAnyOrigin],
    ncomp_in: Int32,
    d_in: Int32,
):
    """`means[k][j] = (resp^T X)[k][j] / nk[k]`, one thread per cell.
    `_estimate_gaussian_parameters:317`, second half."""
    var ncomp = Int(ncomp_in)
    var d = Int(d_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= ncomp * d:
        return
    var k = idx // d
    means.unsafe_store(
        idx, ftz(identical_div(ftz(raw.unsafe_load(idx)), ftz(nk.unsafe_load(k))))
    )


def center_scale_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    means: MutPointer[Float32, MutAnyOrigin],
    resp: MutPointer[Float32, MutAnyOrigin],
    diff: MutPointer[Float32, MutAnyOrigin],
    scaled: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    d_in: Int32,
    kcomp_in: Int32,
    ncomp_in: Int32,
):
    """`diff = X - means[k]` and `scaled = resp[:, k] * diff`, one thread per
    cell. `_estimate_gaussian_covariances_full:194-195`, the two operands of
    their product.

    Their `(resp[:, k] * diff.T) @ diff` scales `diff.T`, which is `d x n`,
    so the broadcast runs along the SAMPLE axis: cell `(a, i)` of the scaled
    operand is `resp[i][k] * diff[i][a]`. Written row-major here, that is
    `scaled[i][a]`, which is why this kernel produces an `n x d` array and
    the product below is `OP_TN` rather than `OP_NN`. The transposition is an
    ADDRESSING difference and reaches no arithmetic.

    `identical_mul` rather than a bare `*`: DEVIATION 826's argument, a
    multiply no codegen may contract into the neighbouring add of the
    accumulation that follows.
    """
    var d = Int(d_in)
    var kc = Int(kcomp_in)
    var ncomp = Int(ncomp_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n_in) * d:
        return
    var i = idx // d
    var j = idx % d
    var dv = ftz(ftz(x.unsafe_load(idx)) - ftz(means.unsafe_load(kc * d + j)))
    diff.unsafe_store(idx, dv)
    var r = ftz(resp.unsafe_load(i * ncomp + kc))
    scaled.unsafe_store(idx, ftz(identical_mul(r, dv)))


def cov_finish_kernel(
    raw: MutPointer[Float32, MutAnyOrigin],
    nk: MutPointer[Float32, MutAnyOrigin],
    cov: MutPointer[Float32, MutAnyOrigin],
    d_in: Int32,
    kcomp_in: Int32,
    reg_covar: Float32,
    divide_in: Int32,
):
    """`cov_k = raw / nk[k]`, then `+ reg_covar` on the DIAGONAL ONLY.
    `_estimate_gaussian_covariances_full:195-196`, in their order.

    **DEVIATION 1736: the order is theirs and it is not interchangeable.**
    Adding `reg_covar` before the division would scale it by `1/nk`, so the
    same nominal `reg_covar` would mean a different ridge in a component with
    a lot of mass than in one with little -- and `reg_covar` would stop
    meaning what scikit-learn's `reg_covar` means. Adding it to every cell
    rather than the diagonal is a different matrix entirely.

    `divide_in` is 0 only for `GMM_SAB_COV_PRESCALE`, which moves the
    division into the responsibility before the accumulation. It is a kernel
    ARGUMENT rather than a second kernel because the arm changes WHICH
    roundings happen and not which cells are written, and a second copy of
    this kernel would have to be kept in step with this one by hand.
    """
    var d = Int(d_in)
    var kc = Int(kcomp_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= d * d:
        return
    var a = idx // d
    var b = idx % d
    var v = ftz(raw.unsafe_load(idx))
    if Int(divide_in) != 0:
        v = ftz(identical_div(v, ftz(nk.unsafe_load(kc))))
    if a == b:
        v = ftz(v + reg_covar)
    cov.unsafe_store(kc * d * d + idx, v)


def cov_reset_identity_kernel(
    cov: MutPointer[Float32, MutAnyOrigin],
    d_in: Int32,
    kcomp_in: Int32,
):
    """`GMM_SAB_COLLAPSE_RESET`'s kernel: overwrite a collapsed component's
    covariance with the identity and carry on.

    There is NO production counterpart, on purpose. This is the thing a
    library does when it wants its fit never to fail, and DEVIATION 1723 is
    the decision not to. It lives here so the check can drive it and show
    that a fit which refuses on one set of pivots and succeeds on another is
    a fit whose OUTCOME is data-dependent in a way no downstream bitwise gate
    would ever see -- because the two runs would not have the same stages to
    compare.
    """
    var d = Int(d_in)
    var kc = Int(kcomp_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= d * d:
        return
    var i = idx // d
    var j = idx % d
    cov.unsafe_store(
        kc * d * d + idx, Float32(1.0) if i == j else Float32(0.0)
    )


# ===========================================================================
# THE WORKSPACES
# ===========================================================================


def gmm_mstep_scratch_floats(n: Int, d: Int, ncomp: Int) -> Int:
    """Floats the M-step needs beside its named buffers.

        [0, n*d)                        `diff`  = X - means[k]
        [n*d, 2*n*d)                    `scaled` = resp[:, k] * diff
        [2*n*d, 2*n*d + max(ncomp*d, d*d))
                                        the raw product, which is `K x d` for
                                        the means and `d x d` for a
                                        covariance -- one region, used by one
                                        at a time
        + 1                             `sum_k nk`, the loop's divisor
    """
    var raw = ncomp * d
    if d * d > raw:
        raw = d * d
    return 2 * n * d + raw + 1


def gmm_mstep_gemm_workspace_floats(n: Int, d: Int, ncomp: Int) -> Int:
    """Floats `identical_gemm_into` may need for the M-step's two products:
    `resp^T . X` (`K x d`, k-axis `n`) and `scaled^T . diff` (`d x d`, k-axis
    `n`). The MAX of the two, never less than one, for the reason
    `identical_gemm_into`'s docstring gives about a workspace with slack."""
    var a = identical_gemm_workspace_max_floats(ncomp, d, n)
    var b = identical_gemm_workspace_max_floats(d, d, n)
    var w = a if a > b else b
    if w < 1:
        return 1
    return w


def gmm_chol_workspace_floats(d: Int) -> Int:
    """Floats the per-component Cholesky needs, from the Cholesky lane's own
    helper at this lane's pinned block size.

    `chol_workspace_floats(d, CHOL_NB_PINNED)` and not a guess: the Cholesky
    lane's `NB` is a NUMERIC parameter pinned under IDENTICAL (its DEVIATION
    1630), a hint asking for anything else RAISES BY NAME, and the workspace
    a smaller `nb` would need is BIGGER rather than smaller. Sizing this by
    hand is the out-of-bounds write both that lane and the gemm lane record
    having been bitten by.
    """
    var w = chol_workspace_floats(d, CHOL_NB_PINNED)
    if w < 1:
        return 1
    return w


def gmm_chol_dwork_floats(d: Int) -> Int:
    """Floats `gmm_precision_cholesky` needs for its own three scratch
    regions, laid out in `dwork`:

        [0, d*d)                   the identity, then `L^{-1}` over it
        [d*d, d*d + d + 1)         `chol_logdet`'s diagonal plus its scalar
        [d*d + d + 1, 2*d*d + d + 1)
                                   the working copy the factorization
                                   overwrites in place

    The working copy exists because `potrf_lower` factors IN PLACE and
    `covariances_` is an OUTPUT of the fit that a caller reads. Factoring the
    covariance itself would leave the estimator returning a triangular factor
    where scikit-learn returns a covariance.
    """
    return 2 * d * d + d + 1


# ===========================================================================
# THE PRECISION CHOLESKY: THE ONE PLACE THIS LANE CAN FAIL
# ===========================================================================


def gmm_precision_cholesky(
    ctx: DeviceContext,
    mut cov: DeviceBuffer[DType.float32],
    mut chol_l: DeviceBuffer[DType.float32],
    mut linv: DeviceBuffer[DType.float32],
    mut prec: DeviceBuffer[DType.float32],
    mut log_det_chol: DeviceBuffer[DType.float32],
    mut cws: DeviceBuffer[DType.float32],
    mut dwork: DeviceBuffer[DType.float32],
    d: Int,
    ncomp: Int,
    mut trace: IdentityTrace,
    tag: String,
    elem_tpb: Int = GMM_ELEM_TPB,
    solve_tpb: Int = CHOL_SOLVE_TPB,
    panel_tpb: Int = CHOL_PANEL_TPB,
    chol_elem_tpb: Int = CHOL_ELEM_TPB,
    sabotage: Int = GMM_SAB_NONE,
) raises -> GmmMStepRun:
    """`_compute_precision_cholesky` for `covariance_type="full"`
    (`_gaussian_mixture.py:353-371`), component by component in ASCENDING
    order.

    For each `k`, in this order:

      1. copy `covariances[k]` into a working `d x d` block, because
         `potrf_lower` factors IN PLACE and the covariance is an output of
         the fit that a caller reads;
      2. `add_jitter(..., GMM_CHOL_JITTER)` -- a ridge of `+0.0`
         (DEVIATION 1737), called for its `ftz` on the diagonal rather than
         for a ridge;
      3. `potrf_lower` -> `L_k`, and **`info != 0` STOPS EVERYTHING**
         (DEVIATION 1723);
      4. `chol_logdet(L_k)` -> `2 sum log L_jj`, and
         `log_det_chol[k] = -0.5 *` that (DEVIATION 1726);
      5. `trsm_lower(L_k, I)` -> `L_k^{-1}`;
      6. transpose -> `P_k = (L_k^{-1})^T`, scikit-learn's `.T` at `:370`.

    **THE CHOLESKY LANE'S OWN CARD STAGES ARE NOT EMITTED, AND THAT IS
    DELIBERATE.** `potrf_lower` and `chol_logdet` record `chol.panelNNN.*`,
    `chol.factor`, `chol.nb`, `chol.diag` and `chol.logdet` under tags they
    hardcode, so calling them `K` times per iteration would violate
    `IdentityTrace`'s tag-uniqueness invariant and raise on the second
    component. They are therefore called with a DISABLED trace, and this
    function records the OUTPUTS -- `<tag>.compKKK.cholesky`,
    `.compKKK.precchol`, `<tag>.logdet` -- under its own tags instead.

    That is also the right card. A `chol.panelNNN.trailing` moving is the
    GEMM lane's certificate, and a `chol.panelNNN.factored` moving is the
    Cholesky lane's; neither is this lane's to certify, and both are already
    gated by `pixi run check-cholesky`. What this lane owes is that the
    FACTOR it fed forward is the same factor. A `tag_prefix` argument on
    `potrf_lower` and `chol_logdet` would let a caller nest their stages
    inside a per-component card, and `mixture/README.md`'s WHAT THE
    ORCHESTRATOR MUST WIRE names it as a change the Cholesky lane could make.
    It was NOT made here.

    SYNCHRONIZES, once per component, twice: `potrf_lower` reads `info` back
    per panel and `chol_logdet` reads its scalar back. At `d <= 32` that is
    one panel, so it is two drains per component per iteration. Named rather
    than hidden; `mixture/README.md`'s WHAT IS OWED carries it.
    """
    var dd = d * d
    var need_dwork = gmm_chol_dwork_floats(d)
    if len(dwork) < need_dwork:
        raise Error(
            "gmm_precision_cholesky: dwork holds "
            + String(len(dwork))
            + " floats, d="
            + String(d)
            + " needs "
            + String(need_dwork)
            + "; use gmm_chol_dwork_floats, not a guess"
        )
    if len(cws) < gmm_chol_workspace_floats(d):
        raise Error(
            "gmm_precision_cholesky: the Cholesky workspace holds "
            + String(len(cws))
            + " floats, d="
            + String(d)
            + " at the pinned block size needs "
            + String(gmm_chol_workspace_floats(d))
            + "; use gmm_chol_workspace_floats, not a guess"
        )
    var identity = dwork.create_sub_buffer[DType.float32](0, dd)
    var scal = dwork.create_sub_buffer[DType.float32](dd, d + 1)
    var work = dwork.create_sub_buffer[DType.float32](dd + d + 1, dd)
    var quiet = IdentityTrace.disabled()
    var logdet_host = List[Float32]()

    var grid_dd = (dd + elem_tpb - 1) // elem_tpb

    for kc in range(ncomp):
        # (1) the working copy
        var src = cov.create_sub_buffer[DType.float32](kc * dd, dd)
        ctx.enqueue_copy(dst_buf=work, src_buf=src)
        ctx.synchronize()

        # (2) the profile's diagonal flush, with no ridge
        add_jitter(ctx, work, d, GMM_CHOL_JITTER, chol_elem_tpb)

        # (3) the factorization, and the only failure this lane has
        var run = potrf_lower(
            ctx,
            work,
            cws,
            d,
            quiet,
            CHOL_NB_PINNED,
            panel_tpb,
            chol_elem_tpb,
        )
        if run.info != 0:
            if sabotage == GMM_SAB_COLLAPSE_RESET:
                # ARM. See `cov_reset_identity_kernel`. The component's
                # covariance is silently replaced and the loop continues, so
                # the fit SUCCEEDS where DEVIATION 1723 refuses.
                ctx.enqueue_function[cov_reset_identity_kernel](
                    cov.unsafe_ptr(),
                    Int32(d),
                    Int32(kc),
                    grid_dim=(grid_dd, 1, 1),
                    block_dim=(elem_tpb, 1, 1),
                )
                ctx.synchronize()
                var src2 = cov.create_sub_buffer[DType.float32](kc * dd, dd)
                ctx.enqueue_copy(dst_buf=work, src_buf=src2)
                ctx.synchronize()
                add_jitter(ctx, work, d, GMM_CHOL_JITTER, chol_elem_tpb)
                run = potrf_lower(
                    ctx,
                    work,
                    cws,
                    d,
                    quiet,
                    CHOL_NB_PINNED,
                    panel_tpb,
                    chol_elem_tpb,
                )
                _ = src2^
                if run.info != 0:
                    # The identity matrix failed to factor, which is
                    # arithmetically impossible. Reported rather than looped
                    # on, so a broken Cholesky cannot be mistaken for a
                    # collapsed component even on the sabotage path.
                    _ = src^
                    _ = identity^
                    _ = scal^
                    _ = work^
                    return GmmMStepRun(run.info, kc)
            else:
                _ = src^
                _ = identity^
                _ = scal^
                _ = work^
                return GmmMStepRun(run.info, kc)

        # (4) log|Sigma_k| through the ONE pinned fold, halved and negated.
        var ld = chol_logdet(ctx, work, scal, d, quiet, chol_elem_tpb)
        logdet_host.append(ftz(identical_mul(Float32(-0.5), ld)))

        # the factor, kept for the card and for the VENDOR_MATMUL arm
        var ldst = chol_l.create_sub_buffer[DType.float32](kc * dd, dd)
        ctx.enqueue_copy(dst_buf=ldst, src_buf=work)
        ctx.synchronize()
        trace.record_device(
            ctx, gmm_comp_tag(tag, kc, "cholesky"), ldst, dd
        )

        # (5) L^{-1} by forward substitution against the identity
        ctx.enqueue_function[set_identity_kernel](
            identity.unsafe_ptr(),
            Int32(d),
            grid_dim=(grid_dd, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
        trsm_lower(
            ctx, work, identity, d, d, quiet, "gmm.precision.trsm", solve_tpb
        )
        var linv_k = linv.create_sub_buffer[DType.float32](kc * dd, dd)
        ctx.enqueue_copy(dst_buf=linv_k, src_buf=identity)
        ctx.synchronize()

        # (6) P_k = (L_k^{-1})^T
        var prec_k = prec.create_sub_buffer[DType.float32](kc * dd, dd)
        ctx.enqueue_function[transpose_square_kernel](
            linv_k.unsafe_ptr(),
            prec_k.unsafe_ptr(),
            Int32(d),
            grid_dim=(grid_dd, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
        ctx.synchronize()
        trace.record_device(
            ctx, gmm_comp_tag(tag, kc, "precchol"), prec_k, dd
        )

        if sabotage == GMM_SAB_LOGDET_FROM_DIAG:
            # ARM: scikit-learn's own spelling, from the precision
            # diagonal. Overwrites the value computed at (4).
            ctx.enqueue_function[sabotage_logdet_from_diag_kernel](
                prec_k.unsafe_ptr(),
                log_det_chol.unsafe_ptr(),
                Int32(d),
                Int32(kc),
                grid_dim=(1, 1, 1),
                block_dim=(1, 1, 1),
            )
            ctx.synchronize()

        _ = src^
        _ = ldst^
        _ = linv_k^
        _ = prec_k^

    if sabotage != GMM_SAB_LOGDET_FROM_DIAG:
        var h = ctx.enqueue_create_host_buffer[DType.float32](ncomp)
        for k in range(ncomp):
            h.unsafe_ptr().unsafe_store(k, logdet_host[k])
        ctx.enqueue_copy(dst_buf=log_det_chol, src_ptr=h.unsafe_ptr())
        ctx.synchronize()
        _ = h^
    trace.record_device(ctx, tag + ".logdet", log_det_chol, ncomp)

    _ = identity^
    _ = scal^
    _ = work^
    return GmmMStepRun(0, -1)


# ===========================================================================
# THE MOMENTS
# ===========================================================================


def gmm_m_step(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut logresp: DeviceBuffer[DType.float32],
    mut resp: DeviceBuffer[DType.float32],
    mut nk: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    mut log_weights: DeviceBuffer[DType.float32],
    mut means: DeviceBuffer[DType.float32],
    mut cov: DeviceBuffer[DType.float32],
    mut scratch: DeviceBuffer[DType.float32],
    mut gws: DeviceBuffer[DType.float32],
    n: Int,
    d: Int,
    ncomp: Int,
    reg_covar: Float32,
    divide_weights_by_n: Bool,
    mut trace: IdentityTrace,
    tag: String,
    elem_tpb: Int = GMM_ELEM_TPB,
    comp_tpb: Int = GMM_COMP_TPB,
    sabotage: Int = GMM_SAB_NONE,
) raises:
    """**THE M-STEP'S MOMENTS.** `_estimate_gaussian_parameters` plus the
    weight normalization, in scikit-learn's order.

    `divide_weights_by_n` selects which of their two divisors to use:
    `True` is `_initialize:864`'s `weights /= n_samples`, `False` is
    `_m_step:898`'s `weights /= sum(weights)`. It is a required argument
    rather than a defaulted one, because a default would let the choice be
    made by not making it, and the two are different numbers.

    **THE PRECISION CHOLESKY IS NOT HERE.** scikit-learn calls it at the end
    of `_m_step`, and the driver in `mixture/estimator.mojo` calls
    `gmm_precision_cholesky` right after this for the same reason -- but it
    is a separate function because it is THE ONLY PLACE THIS LANE CAN FAIL,
    and a step that can fail and a step that cannot should not be one call.

    ASYNCHRONOUS except for the trace records. Stages, in order:

        <tag>.resp         exp(log_resp), n x K
        <tag>.nk           the component masses, K
        <tag>.weights      the mixing weights, K
        <tag>.means        K x d
        <tag>.covariances  K x d x d
    """
    if n <= 0 or d <= 0 or ncomp <= 0:
        raise Error(
            "gmm_m_step: n, d and n_components must all be positive, got n="
            + String(n)
            + " d="
            + String(d)
            + " n_components="
            + String(ncomp)
        )
    var need = gmm_mstep_scratch_floats(n, d, ncomp)
    if len(scratch) < need:
        raise Error(
            "gmm_m_step: the scratch buffer holds "
            + String(len(scratch))
            + " floats, this shape needs "
            + String(need)
            + "; use gmm_mstep_scratch_floats, not a guess"
        )

    var raw_max = ncomp * d
    if d * d > raw_max:
        raw_max = d * d
    var diff = scratch.create_sub_buffer[DType.float32](0, n * d)
    var scaled = scratch.create_sub_buffer[DType.float32](n * d, n * d)
    var raw = scratch.create_sub_buffer[DType.float32](2 * n * d, raw_max)
    var denom = scratch.create_sub_buffer[DType.float32](
        2 * n * d + raw_max, 1
    )

    var grid_cells = (n * ncomp + elem_tpb - 1) // elem_tpb
    if sabotage == GMM_SAB_NO_FTZ_RESP:
        ctx.enqueue_function[sabotage_resp_exp_kernel](
            logresp.unsafe_ptr(),
            resp.unsafe_ptr(),
            Int32(n),
            Int32(ncomp),
            Int32(sabotage),
            grid_dim=(grid_cells, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
    else:
        ctx.enqueue_function[resp_exp_kernel](
            logresp.unsafe_ptr(),
            resp.unsafe_ptr(),
            Int32(n),
            Int32(ncomp),
            grid_dim=(grid_cells, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
    trace.record_device(ctx, tag + ".resp", resp, n * ncomp)

    var grid_comp = (ncomp + comp_tpb - 1) // comp_tpb
    if sabotage == GMM_SAB_NK_DESCENDING:
        ctx.enqueue_function[sabotage_nk_kernel](
            resp.unsafe_ptr(),
            nk.unsafe_ptr(),
            Int32(n),
            Int32(ncomp),
            gmm_ten_eps(),
            Int32(sabotage),
            grid_dim=(grid_comp, 1, 1),
            block_dim=(comp_tpb, 1, 1),
        )
    else:
        ctx.enqueue_function[nk_kernel](
            resp.unsafe_ptr(),
            nk.unsafe_ptr(),
            Int32(n),
            Int32(ncomp),
            gmm_ten_eps(),
            grid_dim=(grid_comp, 1, 1),
            block_dim=(comp_tpb, 1, 1),
        )
    trace.record_device(ctx, tag + ".nk", nk, ncomp)

    # THE DIVISOR. `_initialize:864` uses n_samples; `_m_step:898` uses
    # sum(nk). One kernel, one spelling, two sources.
    if divide_weights_by_n:
        var hd = ctx.enqueue_create_host_buffer[DType.float32](1)
        hd.unsafe_ptr().unsafe_store(0, Float32(n))
        ctx.enqueue_copy(dst_buf=denom, src_ptr=hd.unsafe_ptr())
        ctx.synchronize()
        _ = hd^
    else:
        ctx.enqueue_function[nk_total_kernel](
            nk.unsafe_ptr(),
            denom.unsafe_ptr(),
            Int32(ncomp),
            grid_dim=(1, 1, 1),
            block_dim=(1, 1, 1),
        )
    ctx.enqueue_function[weights_kernel](
        nk.unsafe_ptr(),
        denom.unsafe_ptr(),
        weights.unsafe_ptr(),
        log_weights.unsafe_ptr(),
        Int32(ncomp),
        grid_dim=(grid_comp, 1, 1),
        block_dim=(comp_tpb, 1, 1),
    )
    trace.record_device(ctx, tag + ".weights", weights, ncomp)

    # means = (resp^T . X) / nk.  OP_TN: A is `k x m` = `n x K`, B is
    # `k x n_cols` = `n x d`, C is `K x d`. `linalg.matmul` is REFUSED here;
    # DEVIATION 1729.
    identical_gemm_into(ctx, raw, resp, x, gws, ncomp, d, n, OP_TN)
    var grid_md = (ncomp * d + elem_tpb - 1) // elem_tpb
    ctx.enqueue_function[means_divide_kernel](
        raw.unsafe_ptr(),
        nk.unsafe_ptr(),
        means.unsafe_ptr(),
        Int32(ncomp),
        Int32(d),
        grid_dim=(grid_md, 1, 1),
        block_dim=(elem_tpb, 1, 1),
    )
    trace.record_device(ctx, tag + ".means", means, ncomp * d)

    var grid_nd = (n * d + elem_tpb - 1) // elem_tpb
    var grid_dd = (d * d + elem_tpb - 1) // elem_tpb
    var divide_after = Int32(0) if sabotage == GMM_SAB_COV_PRESCALE else Int32(1)
    for kc in range(ncomp):
        if sabotage == GMM_SAB_COV_PRESCALE:
            ctx.enqueue_function[sabotage_center_scale_kernel](
                x.unsafe_ptr(),
                means.unsafe_ptr(),
                resp.unsafe_ptr(),
                nk.unsafe_ptr(),
                diff.unsafe_ptr(),
                scaled.unsafe_ptr(),
                Int32(n),
                Int32(d),
                Int32(kc),
                Int32(ncomp),
                Int32(sabotage),
                grid_dim=(grid_nd, 1, 1),
                block_dim=(elem_tpb, 1, 1),
            )
        else:
            ctx.enqueue_function[center_scale_kernel](
                x.unsafe_ptr(),
                means.unsafe_ptr(),
                resp.unsafe_ptr(),
                diff.unsafe_ptr(),
                scaled.unsafe_ptr(),
                Int32(n),
                Int32(d),
                Int32(kc),
                Int32(ncomp),
                grid_dim=(grid_nd, 1, 1),
                block_dim=(elem_tpb, 1, 1),
            )
        # cov_raw = scaled^T . diff, the `d x d` weighted second moment.
        # OP_TN, k-axis = n: THE LARGEST SUMMATION ORDER IN THE LANE, and it
        # is the gemm profile's rather than one this file invents.
        identical_gemm_into(ctx, raw, scaled, diff, gws, d, d, n, OP_TN)
        ctx.enqueue_function[cov_finish_kernel](
            raw.unsafe_ptr(),
            nk.unsafe_ptr(),
            cov.unsafe_ptr(),
            Int32(d),
            Int32(kc),
            reg_covar,
            divide_after,
            grid_dim=(grid_dd, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
    trace.record_device(ctx, tag + ".covariances", cov, ncomp * d * d)

    _ = diff^
    _ = scaled^
    _ = raw^
    _ = denom^
