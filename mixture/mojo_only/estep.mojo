# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The E-step: the log Gaussian probability, and the responsibilities.

NOT A PORT, AND THERE IS NOTHING TO PORT. cuML has no Gaussian mixture model
at `upstream/cuml-v26.08.00` (`265b9da`): no `gmm` directory, no `mixture`
module, no `GaussianMixture` symbol outside a scikit-learn test xfail list.
cuVS (`6ba2ce2`) and RAFT (`ebf9268`) have none either. `PORTING_RULES.md`'s
COPY DO NOT IMPROVE therefore **does not apply to this lane**, because there
is no upstream file to copy. What applies instead is
`mixture/README.md`'s rule: `sklearn/mixture/_gaussian_mixture.py` and
`_base.py` define the SEMANTICS and are the ORACLE, and they are never the
design source.

WHAT IS TAKEN FROM SCIKIT-LEARN, AND WHY IT IS TAKEN
------------------------------------------------------
`_estimate_log_gaussian_prob` (`_gaussian_mixture.py:490-548`) computes, for
`covariance_type="full"`:

    y = (X @ prec_chol) - (mu @ prec_chol)      # `:533`
    log_prob[:, k] = sum(square(y), axis=1)     # `:534`
    return -0.5 * (d * log(2 pi) + log_prob) + log_det_chol   # `:548`

That IS the arithmetic this file computes, and it is followed because it is
the ORACLE'S ARITHMETIC, not because it is their design. Two consequences are
worth stating rather than discovering:

1. **`(X @ P) - (mu @ P)` is kept, not rewritten to `(X - mu) @ P`.** The
   second form is numerically better -- it subtracts before the products
   instead of after them, so it does not lose the leading digits of a point
   far from the origin. It is also a DIFFERENT ANSWER, and an answer that
   differs from scikit-learn's for a reason unrelated to the GPU is the one
   thing a lane whose oracle is scikit-learn cannot afford. DEVIATION 1743.
2. **`X @ P` is a GEMM, `mu @ P` is a GEMM, and both go through
   `identical_gemm_into`.** `linalg.matmul` is refused (DEVIATION 1729 in
   `mstep.mojo` covers the covariance product; the same refusal is here).
   This is what makes the E-step the `d^2`-FLOPs-per-point-per-component
   shape that justifies the lane at all.

THE THREE PINS IN THIS FILE
-----------------------------
**(a) The Mahalanobis fold** (`mahal_kernel`). `sum_j y[i][j]^2` is a
summation order. One thread per sample row, feature axis ASCENDING,
`identical_mul_add` per term, `ftz` at every seam, and NO float crosses a
thread boundary -- so launch and batch invariance are properties of the
kernel's shape rather than properties a check happens to observe.
DEVIATION 1728.

**(b) The row max in the logsumexp** (`logsumexp_kernel`). IDENTITY_PATHS row
39: a hardware `max` answers `(+0.0, -0.0)` differently on Apple than on
NVIDIA and AMD, all three measured. So the max here is a POSITIONAL strict
`>` walking `k` ascending from `k = 0`, exactly as
`kde/ported/neighbors/kernel_density.mojo::logsumexp_kernel` does it, so
among equal values the LOWEST INDEX survives on every vendor and a NaN
candidate can never displace a non-NaN seed. **This construction is REUSED
from the KDE lane rather than re-derived**, including its DEVIATION 603 guard
(a row whose every entry is `-inf` yields `-inf`, never `exp(-inf - -inf) =
NaN`), because that lane already solved both the underflow and the
signed-zero problem and a second opinion about either is a second thing that
can be wrong. DEVIATION 1727.

**(c) The mean log likelihood** (`meanll_kernel`). This is THE CONVERGENCE
QUANTITY. It is folded in ONE THREAD, ascending, and divided by `n` with one
`identical_div`. Never a block reduction, never a warp primitive, never an
atomic. See `mixture/README.md`'s hazard 1: a one-bit difference here at
iteration 40 makes one vendor stop and another run a 41st, after which
nothing is comparable at all. DEVIATION 1732.

WHAT THIS FILE DOES NOT COMPUTE
---------------------------------
The precision Cholesky. scikit-learn computes it at the end of `_m_step`
(`:899-901`) and in `_initialize` (`:875-877`), never in the E-step, and
`mixture/mojo_only/mstep.mojo::gmm_precision_cholesky` is where it lives
here for the same reason. It is also the COLLAPSE SITE (DEVIATION 1723), so
keeping it out of the E-step keeps the E-step total: given parameters, it
cannot fail.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext

from core.gemm import gemm_nt
from core.identity_trace import IdentityTrace
from gemm.mojo_only.gemm_identical import (
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.mojo_only.gemm_oracle import OP_NN
from mixture.mojo_only.gmm_sabotage import (
    GMM_SAB_LSE_DESCENDING,
    GMM_SAB_LSE_ROTATE,
    GMM_SAB_MAHAL_DESCENDING,
    GMM_SAB_NONE,
    GMM_SAB_ROWMAX_GE,
    GMM_SAB_ROWMAX_HARDWARE,
    GMM_SAB_VENDOR_MATMUL,
    sabotage_logsumexp_kernel,
    sabotage_mahal_kernel,
)
from mojo_only.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_log,
    identical_mul,
    identical_mul_add,
)


#: The profile's name. Changing the Mahalanobis fold's direction, the row
#: max's rule, the logsumexp's order, the mean-log-likelihood fold, the
#: convergence spelling, `GMM_LOG_2PI_BITS`, `GMM_TEN_EPS_BITS` or the
#: `(X @ P) - (mu @ P)` spelling creates a v2; it does not amend v1. Same
#: discipline as `mojolearn.identical.gemm.fp32.v1` and
#: `mojolearn.identical.cholesky.fp32.v1`, both of which this profile
#: CONTAINS: a v2 of either of them is a v2 of this one.
comptime GMM_PROFILE = "mojolearn.identical.gmm.full.fp32.v1"

#: SCHEDULING. Threads per block for the elementwise kernels. Free in both
#: modes.
comptime GMM_ELEM_TPB = 256

#: SCHEDULING. Threads per block for the one-thread-per-sample-row kernels
#: (the Mahalanobis fold and the logsumexp). Free.
comptime GMM_ROW_TPB = 128

#: SCHEDULING. Threads per block for the one-thread-per-component kernels.
#: Free.
comptime GMM_COMP_TPB = 32

#: **NUMERIC. DEVIATION 1730.** `log(2 pi)` as float32 BITS.
#: `1.8378770664093453` correctly rounded to float32 is `0x3FEB3F8E`.
#:
#: Written as bits and bitcast, never as a decimal string:
#: `[[mojo-string-float-roundtrip]]` says `String(Float32)` does not round
#: trip in this toolchain, and a profile constant a log line cannot reproduce
#: is a profile constant nobody can check. scikit-learn computes
#: `math.log(2 * math.pi)` in host float64 and multiplies by `n_features`
#: there (`_gaussian_mixture.py:548`); a host libm is IDENTITY_PATHS row 18's
#: class, so under IDENTICAL the constant is pinned and the multiply by `d`
#: is one `identical_mul`.
comptime GMM_LOG_2PI_BITS: UInt32 = 0x3FEB3F8E

#: **NUMERIC. DEVIATION 1730.** `10 * finfo(float32).eps` as float32 bits.
#: `finfo(float32).eps` is `2^-23`; ten times it is `1.1920929e-06`, whose
#: float32 bits are `0x35A00000`. scikit-learn adds exactly this to every
#: `nk` (`_gaussian_mixture.py:317`) so that a component with no mass still
#: has a positive divisor. Pinned here because it is a NUMBER IN THE ANSWER,
#: not a guard: it shifts every mean and every covariance.
comptime GMM_TEN_EPS_BITS: UInt32 = 0x35A00000

#: `-inf` as float32 bits. The seed of the convergence test's previous lower
#: bound and the value `logsumexp_kernel` returns for an all-`-inf` row
#: (DEVIATION 1727, inherited from KDE's DEVIATION 603).
comptime GMM_NEG_INF_BITS: UInt32 = 0xFF800000

#: `+inf` as float32 bits. The convergence change at iteration 1, where the
#: previous lower bound is `-inf`. DEVIATION 1747.
comptime GMM_POS_INF_BITS: UInt32 = 0x7F800000


def gmm_log_2pi() -> Float32:
    """`GMM_LOG_2PI_BITS` as a value. A function rather than a `comptime`
    binding because the constant is defined by its BITS and the bitcast is
    the definition."""
    return bitcast[DType.float32](GMM_LOG_2PI_BITS)


def gmm_ten_eps() -> Float32:
    """`GMM_TEN_EPS_BITS` as a value. Same reason."""
    return bitcast[DType.float32](GMM_TEN_EPS_BITS)


def gmm_neg_inf() -> Float32:
    """`-inf` as a value, by its bits."""
    return bitcast[DType.float32](GMM_NEG_INF_BITS)


def gmm_pos_inf() -> Float32:
    """`+inf` as a value, by its bits."""
    return bitcast[DType.float32](GMM_POS_INF_BITS)


def gmm_iter_tag(prefix: String, it: Int, leaf: String) -> String:
    """`gmm.iter003.mahal` and its siblings. THREE digits, zero padded.

    `core/identity_trace.mojo` rule 1 makes tag uniqueness an INVARIANT and
    the differ aligns two traces by their tag SEQUENCES, so every stage a
    loop emits has to carry its iteration index. Rule 2 makes the tag name a
    POSITION IN THE ALGORITHM and never a property of the machine: an
    iteration index is a position, and the ITERATION COUNT is an output of
    the algorithm rather than of the box -- which is exactly the claim
    `check_iteration_count_is_identical` gates. The padding is what stops a
    reader from mistaking `iter10` for a sibling of `iter1`.
    """
    var s = String(it)
    while s.byte_length() < 3:
        s = String("0") + s
    return prefix + ".iter" + s + "." + leaf


def gmm_iter_prefix(prefix: String, it: Int) -> String:
    """`gmm.iter003`, with no leaf. What a whole step is handed as its `tag`
    argument, so the step names its own stages under it.

    A separate function rather than `gmm_iter_tag(prefix, it, "")`, which
    would produce a trailing separator that every caller would then have to
    strip. Mojo `String` has no slice syntax, so "then strip it" is not the
    one-line fix it looks like.
    """
    var s = String(it)
    while s.byte_length() < 3:
        s = String("0") + s
    return prefix + ".iter" + s


def gmm_comp_tag(prefix: String, k: Int, leaf: String) -> String:
    """`gmm.iter003.comp001.cholesky`. Three digits, same argument."""
    var s = String(k)
    while s.byte_length() < 3:
        s = String("0") + s
    return prefix + ".comp" + s + "." + leaf


# ===========================================================================
# THE KERNELS
# ===========================================================================


def set_identity_kernel(
    b: MutPointer[Float32, MutAnyOrigin],
    d_in: Int32,
):
    """`B = I_d`, `d x d` row-major. `+1.0` on the diagonal and `+0.0`
    everywhere else.

    `+0.0` and not `-0.0`, stated because the sign of a written zero is a bit
    the card hashes and because this matrix is the right-hand side of the
    triangular solve that produces `L^{-1}`: a `-0.0` off-diagonal entry
    would propagate a signed zero into the precision Cholesky, where nothing
    downstream distinguishes it and every tolerance comparison would agree
    while the card would not. The convention is one value on every column.
    """
    var d = Int(d_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= d * d:
        return
    var i = idx // d
    var j = idx % d
    b.unsafe_store(idx, Float32(1.0) if i == j else Float32(0.0))


def transpose_square_kernel(
    src: MutPointer[Float32, MutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
    d_in: Int32,
):
    """`dst = src^T`, `d x d` row-major, out of place.

    This is scikit-learn's `.T` at `_gaussian_mixture.py:368-370`: they solve
    `L X = I` for `X = L^{-1}` and then TRANSPOSE it, so
    `precisions_chol[k] = (L^{-1})^T`, which is upper triangular. A transpose
    moves no float through any arithmetic -- every output bit is an input bit
    -- so there is nothing here to pin, and it is written out of place so no
    cell is read after it has been overwritten.
    """
    var d = Int(d_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= d * d:
        return
    var i = idx // d
    var j = idx % d
    dst.unsafe_store(i * d + j, src.unsafe_load(j * d + i))


def mahal_kernel(
    y: MutPointer[Float32, MutAnyOrigin],
    murow: MutPointer[Float32, MutAnyOrigin],
    mahal: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    d_in: Int32,
    kcomp_in: Int32,
    ncomp_in: Int32,
):
    """`mahal[i][k] = sum_j (y[i][j] - murow[j])^2`, ONE THREAD PER SAMPLE.

    `y` is `X . P_k` (`n x d`) and `murow` is `mu_k . P_k` (`d`), so the
    subtraction inside the fold is scikit-learn's `:533` and the fold is
    their `xp.sum(xp.square(y), axis=1)` at `:534`. Materializing the
    difference into a second `n x d` array first and folding it after would
    give the same bits (each `y[i][j] - murow[j]` is one rounding either way)
    and would cost an `n x d` write and read; it is fused here and that is a
    SCHEDULING choice, stated so nobody has to work out whether it is a
    numeric one.

    **DEVIATION 1728. THE FOLD IS PINNED AND NO FLOAT CROSSES A THREAD.**
    The feature axis walks ASCENDING, every term is one `identical_mul_add`
    (row 9's contraction pin) and every seam is flushed (row 10). The whole
    sum lives in one thread's register, so the answer is a pure function of
    `d` -- not of the block size, the grid, the lane width, or which other
    samples share the launch. There is no block fold and no warp primitive to
    pin because there is none here.

    `mahal` is `n x ncomp` row-major and this kernel writes column `kcomp`,
    which is why `ncomp` is a parameter of a kernel that otherwise never uses
    it: the column stride is the only place it appears.
    """
    var n = Int(n_in)
    var d = Int(d_in)
    var kc = Int(kcomp_in)
    var ncomp = Int(ncomp_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    var acc = Float32(0.0)
    for j in range(d):
        var t = ftz(ftz(y.unsafe_load(i * d + j)) - ftz(murow.unsafe_load(j)))
        acc = ftz(identical_mul_add(t, t, acc))
    mahal.unsafe_store(i * ncomp + kc, acc)


def weighted_log_prob_kernel(
    mahal: MutPointer[Float32, MutAnyOrigin],
    log_det_chol: MutPointer[Float32, MutAnyOrigin],
    log_weights: MutPointer[Float32, MutAnyOrigin],
    wlp: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    ncomp_in: Int32,
    d_log_2pi: Float32,
):
    """`log P(x_i | k) + log pi_k`, one thread per cell.

    scikit-learn's `:548` then `_base.py:513-524`, in their order and with
    their bracketing:

        log_prob = -0.5 * (d * log(2 pi) + mahal) + log_det_chol
        wlp      = log_prob + log(weights)

    `d_log_2pi` is `identical_mul(Float32(d), GMM_LOG_2PI)`, computed once on
    the host (DEVIATION 1730) rather than per cell, because it is the same
    number for every cell and a per-cell recomputation is `n * K` chances for
    a codegen to contract it into the neighbouring add differently.

    Every intermediate is stored through `ftz`, which is row 10's checklist
    requirement for a pinned expression: on a denormal-honoring column an
    intermediate that is not written out cannot be reached by the flush, so
    the expression is spelled with one local per step.
    """
    var n = Int(n_in)
    var ncomp = Int(ncomp_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= n * ncomp:
        return
    var k = idx % ncomp
    var m = ftz(mahal.unsafe_load(idx))
    var inner = ftz(d_log_2pi + m)
    var half = ftz(identical_mul(Float32(-0.5), inner))
    var lp = ftz(half + ftz(log_det_chol.unsafe_load(k)))
    wlp.unsafe_store(idx, ftz(lp + ftz(log_weights.unsafe_load(k))))


def logsumexp_kernel(
    wlp: MutPointer[Float32, MutAnyOrigin],
    lse: MutPointer[Float32, MutAnyOrigin],
    rowmax: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    ncomp_in: Int32,
):
    """`log sum_k exp(wlp[i][k])`, ONE THREAD PER SAMPLE ROW.

    **THIS IS KDE'S LOGSUMEXP, REUSED RATHER THAN REWRITTEN.**
    `kde/ported/neighbors/kernel_density.mojo::logsumexp_kernel` already
    solved both hard parts of this operation for this repository, and this
    file reproduces its construction line for line with `n_train` replaced by
    `n_components`. What is inherited, and what each part is for:

    **THE SHIFT, against underflow.** `sum_k exp(wlp_k)` computed naively is
    `exp` of a large negative number summed, which underflows to exactly zero
    and makes the answer `log(0) = -inf` on data where the true answer is an
    ordinary number. Subtracting the row max first makes the largest term
    exactly `exp(0) = 1`, so the sum is at least one and the logarithm is
    finite. `kde/mojo_only/kde_check.mojo::check_kde_logsumexp_beats_naive`
    is the measurement (naive `-inf`, shifted `-1708.7214` against a float64
    `-1708.7213588720942`), and `check_estep_vs_oracle` re-runs the same
    demonstration on a mixture row.

    **THE ROW MAX, against IDENTITY_PATHS row 39.** The max is a strict `>`
    walking `k` ASCENDING from `k = 0`, so among EQUAL values the FIRST in
    ascending `k` survives. It is NOT a hardware `max`, whose answer on
    `(+0.0, -0.0)` is the vendor's: `-0.0` on Apple (the second operand) and
    `+0.0` on NVIDIA and AMD (IEEE-2019 maximum), all three MEASURED
    (IDENTITY_PATHS row 39). A positional rule is the same answer on every
    vendor. A NaN candidate never displaces a non-NaN seed either, because
    `>` is false in both directions against a NaN.

    Reachability, stated rather than assumed: a mixed-zero row is not
    reachable from legal data here any more than it is in KDE. Two components
    would have to produce weighted log probabilities that are both zero and
    of opposite sign, and a weighted log probability is
    `-0.5 (d log 2pi + mahal) + log_det + log_weight` -- a sum whose terms
    are not zero and whose value is a zero only by exact cancellation of four
    unrelated quantities. So `check_estep_vs_oracle` PLANTS the mixed row
    directly into this kernel, in both orders, exactly as
    `check_kde_row39_signed_zero_rowmax` does, and asserts the lower-index
    zero's bits on the device and on the oracle.

    **THE ALL-`-inf` ROW, inherited DEVIATION 603.** If every component's
    weighted log probability is `-inf` -- reachable with a legal finite
    input, a point far from every component against a tiny covariance -- then
    `exp(-inf - (-inf))` is `exp(NaN)` is NaN, and the score is NaN.
    scikit-learn folds the same row with `logsumexp`, whose `(-inf, -inf)` is
    `-inf`. **A COMPUTED NaN CARRIES THE VENDOR'S PAYLOAD** (Apple
    `0x7fc00000`, NVIDIA `0x7fffffff`, AMD `0xffc00000`, all measured) and can
    never sit in a certified stage, and `gmm.iterNNN.lse` is a certified
    stage. So the guard is here, the value is `-inf`, and it is the value the
    mathematics and the oracle both give. DEVIATION 1727.

    ONE THREAD PER ROW, so the order is a pure function of `n_components`:
    not of the block size, the grid, the lane width, or which other rows
    share the launch. No block fold, no warp primitive, no atomic.
    """
    var n = Int(n_in)
    var ncomp = Int(ncomp_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    var base = i * ncomp

    var max_exp = wlp.unsafe_load(base)
    for k in range(1, ncomp):
        var v = wlp.unsafe_load(base + k)
        # ROW 39: strict `>`, lower index wins a tie; NOT a hardware max.
        if v > max_exp:
            max_exp = v
    rowmax.unsafe_store(i, max_exp)

    # DEVIATION 1727 (KDE's 603): every component -inf -> the log-sum-exp is
    # -inf, never a computed NaN.
    if max_exp == bitcast[DType.float32](GMM_NEG_INF_BITS):
        lse.unsafe_store(i, max_exp)
        return

    var s = Float32(0.0)
    for k in range(ncomp):
        s = ftz(
            s + ftz(identical_exp(ftz(wlp.unsafe_load(base + k) - max_exp)))
        )
    lse.unsafe_store(i, ftz(identical_log(s) + max_exp))


def log_resp_kernel(
    wlp: MutPointer[Float32, MutAnyOrigin],
    lse: MutPointer[Float32, MutAnyOrigin],
    logresp: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    ncomp_in: Int32,
):
    """`log_resp = wlp - log_prob_norm[:, newaxis]`, one thread per cell.

    scikit-learn's `_base.py:578-580`, including the fact that they wrap it
    in `np.errstate(under="ignore")`: the difference underflows for a
    component a point does not belong to, and the answer there is a large
    negative number rather than an error. Nothing here needs the errstate
    because nothing here raises on underflow; the `ftz` is row 10's flush and
    is a numeric decision, not an error-handling one.
    """
    var n = Int(n_in)
    var ncomp = Int(ncomp_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= n * ncomp:
        return
    var i = idx // ncomp
    logresp.unsafe_store(
        idx, ftz(ftz(wlp.unsafe_load(idx)) - ftz(lse.unsafe_load(i)))
    )


def meanll_kernel(
    lse: MutPointer[Float32, MutAnyOrigin],
    out_scalar: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`mean(log_prob_norm)`, in ONE THREAD, ascending. **DEVIATION 1732.**

    `_base.py:331` is `xp.mean(log_prob_norm)` and `_gaussian_mixture.py`'s
    `_compute_lower_bound` returns it unchanged, so this scalar IS the value
    the convergence test compares. That makes it the single most
    consequential number in the lane, and the reason is not its magnitude:

    **A ONE-BIT DIFFERENCE HERE CHANGES THE ITERATION COUNT.** The loop stops
    when `|lower_bound - prev| < tol`. If two vendors' means differ in the
    last bit at iteration 40 and the change is sitting within one ulp of
    `tol`, one stops and the other runs a 41st iteration -- after which the
    parameters, the responsibilities and every subsequent stage are
    incomparable, and a bitwise gate on the OUTPUT reports a difference whose
    cause is three stages and one control-flow decision upstream.
    `mixture/README.md`'s hazard 1 is this paragraph.

    So: ONE THREAD, ASCENDING, `ftz` at every seam, and the division by `n`
    is one `identical_div` and never a multiply by a reciprocal (two
    roundings where a divide is one, the same argument as
    `cholesky/mojo_only/trsm.mojo`'s DEVIATION 1643). No block fold, no warp
    shuffle, no atomic, no `pinned_block_sum`. `n` values is not enough work
    to be worth pinning a tree for, and a tree here would be a second fold
    shape in a lane that needs zero of them.

    `GMM_SAB_MEANLL_PAIRWISE` is the arm that folds this pairwise instead,
    and the check requires it to move the ITERATION COUNT on at least one
    fixture, not merely the bits.
    """
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    var n = Int(n_in)
    var acc = Float32(0.0)
    for i in range(n):
        acc = ftz(acc + ftz(lse.unsafe_load(i)))
    out_scalar.unsafe_store(0, ftz(identical_div(acc, Float32(n))))


def argmax_kernel(
    wlp: MutPointer[Float32, MutAnyOrigin],
    labels: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
    ncomp_in: Int32,
):
    """`labels[i] = argmax_k wlp[i][k]`, one thread per row.

    `_base.py:412` (`predict`) and `:311` (`fit_predict`'s final assignment)
    are `xp.argmax(..., axis=1)`, whose documented tie rule is THE FIRST
    OCCURRENCE. The strict `>` below reproduces it, and it is the same
    positional rule `logsumexp_kernel`'s row max uses, so the two cannot
    disagree about which component a tied row belongs to.

    **EVERY TIE BREAKS ON A TOTAL ORDER INCLUDING AN INDEX.** Here the index
    IS the order: equal values leave `best` untouched, so the surviving
    candidate is the lowest `k`. On exact duplicate points (`FIX_DUPLICATES`)
    every row of the pair produces the same `wlp` row and therefore the same
    label, which is what makes duplicates a fixture rather than a hazard.

    `predict` uses `wlp` and not `log_resp` on purpose, mirroring
    `_base.py:412`: the two differ by a per-row constant, so the argmax is
    the same, and using the weighted log probability means `predict` does not
    have to run the logsumexp at all.
    """
    var n = Int(n_in)
    var ncomp = Int(ncomp_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    var base = i * ncomp
    var best = wlp.unsafe_load(base)
    var best_k = 0
    for k in range(1, ncomp):
        var v = wlp.unsafe_load(base + k)
        if v > best:
            best = v
            best_k = k
    labels.unsafe_store(i, Int32(best_k))


# ===========================================================================
# THE WORKSPACE AND THE DRIVER
# ===========================================================================


def gmm_estep_gemm_workspace_floats(n: Int, d: Int) -> Int:
    """Floats `identical_gemm_into` may need for the E-step's two products at
    this shape: `X . P` (`n x d` by `d x d`) and `mu . P` (`1 x d` by
    `d x d`).

    **SIZE FOR THE LARGEST PLAN EITHER SHAPE COULD REACH, not for one of
    them.** `identical_gemm_into`'s own docstring records that sizing a
    workspace for one plan and letting the dispatcher pick another is an
    out-of-bounds write a small shape will not show you, and that it cost the
    gemm lane a run. The max of the two helper answers is the only safe
    number, and it is never less than one so the buffer is always
    constructible.
    """
    var a = identical_gemm_workspace_max_floats(n, d, d)
    var b = identical_gemm_workspace_max_floats(1, d, d)
    var w = a if a > b else b
    if w < 1:
        return 1
    return w


def gmm_estep_scratch_floats(n: Int, d: Int) -> Int:
    """Floats the E-step needs beside its named buffers.

        [0, n*d)          `y`, the product `X . P_k`
        [n*d, n*d + d)    `murow`, the product `mu_k . P_k`

    One component at a time, reused across `k`. Holding all `K` of them would
    be `K n d` floats for no arithmetic benefit: the Mahalanobis fold reads
    `y` once and never again.
    """
    return n * d + d


def gmm_e_step(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut means: DeviceBuffer[DType.float32],
    mut prec: DeviceBuffer[DType.float32],
    mut linv: DeviceBuffer[DType.float32],
    mut log_det_chol: DeviceBuffer[DType.float32],
    mut log_weights: DeviceBuffer[DType.float32],
    mut scratch: DeviceBuffer[DType.float32],
    mut gws: DeviceBuffer[DType.float32],
    mut mahal: DeviceBuffer[DType.float32],
    mut wlp: DeviceBuffer[DType.float32],
    mut rowmax: DeviceBuffer[DType.float32],
    mut lse: DeviceBuffer[DType.float32],
    mut logresp: DeviceBuffer[DType.float32],
    mut meanll: DeviceBuffer[DType.float32],
    n: Int,
    d: Int,
    ncomp: Int,
    mut trace: IdentityTrace,
    tag: String,
    elem_tpb: Int = GMM_ELEM_TPB,
    row_tpb: Int = GMM_ROW_TPB,
    sabotage: Int = GMM_SAB_NONE,
) raises:
    """**THE E-STEP.** Given the parameters, produce `mahal`, `wlp`,
    `rowmax`, `lse`, `logresp` and the mean log likelihood.

    `_base.py::_e_step` (`:314-333`) composed with
    `_estimate_log_prob_resp` (`:552-582`),
    `_estimate_weighted_log_prob` (`:513-524`) and
    `_estimate_log_gaussian_prob` (`_gaussian_mixture.py:490-548`), in their
    order.

    **THIS STEP CANNOT FAIL.** Given finite parameters it is total: no
    division by a value that could be zero, no square root, no logarithm of a
    quantity that could be non-positive (`identical_log`'s argument is the
    shifted exponential sum, which is at least `exp(0) = 1` by construction),
    and the one NaN a legal input could compute is guarded by DEVIATION 1727.
    Every failure this lane can have lives in `mstep.mojo`'s Cholesky, which
    is where scikit-learn puts it too. That is a design property worth naming
    because it is what lets `predict`, `predict_proba` and `score_samples`
    be simple calls against stored parameters.

    ASYNCHRONOUS except for the trace records, which drain by construction
    (`core/identity_trace.mojo` rule 4). Every buffer must outlive the
    caller's own `ctx.synchronize()`.

    `tag` is this step's card prefix, for example `gmm.iter003`. It must be
    unique within a trace: `IdentityTrace._emit` raises on a repeat,
    deliberately, so two E-steps in one fit have to name themselves apart.

    THE STAGES, in the order a divergence would first show up in:

        <tag>.mahal    the Mahalanobis distances, n x K
        <tag>.wlp      log P(x | k) + log pi_k, n x K
        <tag>.rowmax   the logsumexp's row maxima, n   (row 39's site)
        <tag>.lse      log p(x), n
        <tag>.logresp  the log responsibilities, n x K
        <tag>.meanll   the mean log likelihood, 1      (the convergence
                                                        quantity)
    """
    if n <= 0 or d <= 0 or ncomp <= 0:
        raise Error(
            "gmm_e_step: n, d and n_components must all be positive, got n="
            + String(n)
            + " d="
            + String(d)
            + " n_components="
            + String(ncomp)
        )
    var need_scratch = gmm_estep_scratch_floats(n, d)
    if len(scratch) < need_scratch:
        raise Error(
            "gmm_e_step: the scratch buffer holds "
            + String(len(scratch))
            + " floats, this shape needs "
            + String(need_scratch)
            + "; use gmm_estep_scratch_floats, not a guess"
        )
    var need_gws = gmm_estep_gemm_workspace_floats(n, d)
    if len(gws) < need_gws:
        raise Error(
            "gmm_e_step: the gemm workspace holds "
            + String(len(gws))
            + " floats, this shape needs "
            + String(need_gws)
            + "; use gmm_estep_gemm_workspace_floats, not a guess"
        )

    var y = scratch.create_sub_buffer[DType.float32](0, n * d)
    var murow = scratch.create_sub_buffer[DType.float32](n * d, d)

    for kc in range(ncomp):
        var pk = prec.create_sub_buffer[DType.float32](kc * d * d, d * d)
        var muk = means.create_sub_buffer[DType.float32](kc * d, d)

        # y = X . P_k    (`_gaussian_mixture.py:533`, the first half)
        if sabotage == GMM_SAB_VENDOR_MATMUL:
            # ARM: MAX `linalg.matmul`, a CLOSED library whose k-split is a
            # per-vendor summation order that nothing in this repository can
            # pin, read or check. `gemm_nt` computes `x[m x k] . y[n x k]^T`,
            # and `P^T` is exactly `L^{-1}`, which the driver still holds --
            # so the arm needs no transpose of its own and differs from the
            # production path in the LIBRARY and in nothing else.
            var lk = linv.create_sub_buffer[DType.float32](
                kc * d * d, d * d
            )
            gemm_nt(ctx, y, x, lk, n, d, d)
            _ = lk^
        else:
            identical_gemm_into(ctx, y, x, pk, gws, n, d, d, OP_NN)

        # murow = mu_k . P_k    (`:533`, the second half). A separate GEMM
        # rather than a broadcast subtraction before the product, because
        # that is scikit-learn's spelling and DEVIATION 1743 keeps it.
        identical_gemm_into(ctx, murow, muk, pk, gws, 1, d, d, OP_NN)

        var grid_rows = (n + row_tpb - 1) // row_tpb
        if sabotage == GMM_SAB_MAHAL_DESCENDING:
            ctx.enqueue_function[sabotage_mahal_kernel](
                y.unsafe_ptr(),
                murow.unsafe_ptr(),
                mahal.unsafe_ptr(),
                Int32(n),
                Int32(d),
                Int32(kc),
                Int32(ncomp),
                Int32(sabotage),
                grid_dim=(grid_rows, 1, 1),
                block_dim=(row_tpb, 1, 1),
            )
        else:
            ctx.enqueue_function[mahal_kernel](
                y.unsafe_ptr(),
                murow.unsafe_ptr(),
                mahal.unsafe_ptr(),
                Int32(n),
                Int32(d),
                Int32(kc),
                Int32(ncomp),
                grid_dim=(grid_rows, 1, 1),
                block_dim=(row_tpb, 1, 1),
            )
        _ = pk^
        _ = muk^

    trace.record_device(ctx, tag + ".mahal", mahal, n * ncomp)

    var d_log_2pi = ftz(identical_mul(Float32(d), gmm_log_2pi()))
    var grid_cells = (n * ncomp + elem_tpb - 1) // elem_tpb
    ctx.enqueue_function[weighted_log_prob_kernel](
        mahal.unsafe_ptr(),
        log_det_chol.unsafe_ptr(),
        log_weights.unsafe_ptr(),
        wlp.unsafe_ptr(),
        Int32(n),
        Int32(ncomp),
        d_log_2pi,
        grid_dim=(grid_cells, 1, 1),
        block_dim=(elem_tpb, 1, 1),
    )
    trace.record_device(ctx, tag + ".wlp", wlp, n * ncomp)

    var grid_rows2 = (n + row_tpb - 1) // row_tpb
    if (
        sabotage == GMM_SAB_LSE_DESCENDING
        or sabotage == GMM_SAB_LSE_ROTATE
        or sabotage == GMM_SAB_ROWMAX_GE
        or sabotage == GMM_SAB_ROWMAX_HARDWARE
    ):
        ctx.enqueue_function[sabotage_logsumexp_kernel](
            wlp.unsafe_ptr(),
            lse.unsafe_ptr(),
            rowmax.unsafe_ptr(),
            Int32(n),
            Int32(ncomp),
            Int32(sabotage),
            grid_dim=(grid_rows2, 1, 1),
            block_dim=(row_tpb, 1, 1),
        )
    else:
        ctx.enqueue_function[logsumexp_kernel](
            wlp.unsafe_ptr(),
            lse.unsafe_ptr(),
            rowmax.unsafe_ptr(),
            Int32(n),
            Int32(ncomp),
            grid_dim=(grid_rows2, 1, 1),
            block_dim=(row_tpb, 1, 1),
        )
    trace.record_device(ctx, tag + ".rowmax", rowmax, n)
    trace.record_device(ctx, tag + ".lse", lse, n)

    ctx.enqueue_function[log_resp_kernel](
        wlp.unsafe_ptr(),
        lse.unsafe_ptr(),
        logresp.unsafe_ptr(),
        Int32(n),
        Int32(ncomp),
        grid_dim=(grid_cells, 1, 1),
        block_dim=(elem_tpb, 1, 1),
    )
    trace.record_device(ctx, tag + ".logresp", logresp, n * ncomp)

    # ONE BLOCK, ONE THREAD. The launch is written out rather than defaulted
    # so that a reader of the driver sees the fold's shape without opening
    # the kernel, and so no future edit can widen it by changing a default.
    ctx.enqueue_function[meanll_kernel](
        lse.unsafe_ptr(),
        meanll.unsafe_ptr(),
        Int32(n),
        grid_dim=(1, 1, 1),
        block_dim=(1, 1, 1),
    )
    trace.record_device(ctx, tag + ".meanll", meanll, 1)
    _ = y^
    _ = murow^


def gmm_predict_labels(
    ctx: DeviceContext,
    mut wlp: DeviceBuffer[DType.float32],
    mut labels: DeviceBuffer[DType.int32],
    n: Int,
    ncomp: Int,
    mut trace: IdentityTrace,
    tag: String,
    row_tpb: Int = GMM_ROW_TPB,
) raises:
    """`argmax` over the weighted log probabilities, plus the card stage.

    Split out from `gmm_e_step` because `predict` needs it and does not need
    the logsumexp, exactly as `_base.py:395-412` does not call
    `_estimate_log_prob_resp` at all.
    """
    var grid = (n + row_tpb - 1) // row_tpb
    ctx.enqueue_function[argmax_kernel](
        wlp.unsafe_ptr(),
        labels.unsafe_ptr(),
        Int32(n),
        Int32(ncomp),
        grid_dim=(grid, 1, 1),
        block_dim=(row_tpb, 1, 1),
    )
    var h = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=labels)
    ctx.synchronize()
    var lst = List[Int32]()
    for i in range(n):
        lst.append(h.unsafe_ptr().unsafe_load(i))
    trace.record_list_i32(tag + ".labels", lst)
    _ = h^


# ===========================================================================
# THE CONVERGENCE TEST
#
# **ONE SPELLING, SHARED BY THE DEVICE DRIVER AND THE ORACLE, AND THAT IS A
# DELIBERATE EXCEPTION TO THIS LANE'S TWO-SPELLINGS RULE.** Everywhere else
# `mixture/mojo_only/gmm_oracle.mojo` re-spells the arithmetic a second time
# so the gate compares two spellings of one operation rather than a function
# against itself. Not here, and the reason is `mixture/README.md`'s hazard 1:
# the thing that must be identical across vendors is the VALUE the test
# compares (`meanll`, DEVIATION 1732), and the test itself is HOST control
# flow that runs once per iteration on both arms. Two spellings of it would
# create a way for the driver and the oracle to take different numbers of
# iterations for a reason that has nothing to do with the GPU -- which is
# exactly the failure the hazard is about, manufactured by the instrument
# meant to detect it.
#
# The device driver in `mixture/estimator.mojo` and `oracle_fit` in
# `gmm_oracle.mojo` both call these two functions and neither has a copy.
# ===========================================================================


def gmm_convergence_change(
    lower_bound: Float32, prev: Float32
) -> Float32:
    """`change = lower_bound - prev_lower_bound`, WITH DEVIATION 1747's
    first-iteration rule.

    `_base.py:273` computes `lower_bound - prev_lower_bound` with `prev`
    initialized to `-inf` (`:257`), so at iteration 1 numpy evaluates
    `-inf - (-inf) = NaN` and emits a runtime warning, and the subsequent
    `abs(change) < tol` is False for every NaN. That is the right BEHAVIOR
    and the wrong VALUE to compute: a COMPUTED NaN carries the vendor's
    payload -- Apple `0x7fc00000`, NVIDIA `0x7fffffff`, AMD `0xffc00000`, all
    three measured (IDENTITY_PATHS row 39) -- and `gmm.iterNNN.change` is a
    RECORDED CARD STAGE, so a NaN there would make three vendors' cards
    disagree at iteration 1 of every fit while every number in them matched.

    So the change at iteration 1 is `+inf`: the value the mathematics gives
    for the improvement over an empty model, the same bits on every vendor,
    and a value for which `abs(change) < tol` is False exactly as it is for
    the NaN. The behavior is scikit-learn's; the bits are ours and are
    stated. DEVIATION 1747.
    """
    if prev == gmm_neg_inf():
        return gmm_pos_inf()
    return ftz(lower_bound - prev)


def gmm_converged(change: Float32, tol: Float32) -> Bool:
    """`abs(change) < tol`, `_base.py:276`.

    `abs` is spelled as a comparison against zero and a negation rather than
    through a library call, so it has one behavior on `-0.0`: `-0.0` is not
    less than `+0.0`, so it is returned unchanged, and `-0.0 < tol` is True
    for any positive `tol` -- the right answer, since a change of zero has
    converged. NaN cannot reach here: DEVIATION 1747 removed the only one a
    legal fit could compute.

    **THIS IS THE WHOLE BALLGAME AND IT IS FOUR LINES LONG.** Every number
    this lane produces after iteration `k` depends on whether this returned
    True at iteration `k`.
    """
    var a = change
    if a < Float32(0.0):
        a = -a
    return a < tol


def gmm_n_parameters(d: Int, ncomp: Int) -> Int:
    """`GaussianMixture._n_parameters` for `covariance_type="full"`
    (`_gaussian_mixture.py:945-957`).

        cov_params  = K * d * (d + 1) / 2
        mean_params = d * K
        return int(cov_params + mean_params + K - 1)

    Integer arithmetic here where theirs is a float division by `2.0`
    followed by an `int()`: `d * (d + 1)` is always even, so the two agree at
    every `d`, and an integer count computed in float is one place a large
    model could round.
    """
    var cov_params = ncomp * d * (d + 1) // 2
    var mean_params = d * ncomp
    return cov_params + mean_params + ncomp - 1
