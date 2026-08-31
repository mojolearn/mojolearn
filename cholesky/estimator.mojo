# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host-pointer surface for the Cholesky lane: what three lanes will call.

**NOT YET WIRED** into `bindings/_mojolearn_estimators.mojo` or
`python/mojolearn/` -- those directories are not this lane's. The README's
WHAT THE ORCHESTRATOR MUST WIRE names the tasks; this file is the entry a
binding, a Gaussian process, a kernel-ridge solver or a Gaussian mixture
should reach, shaped like `kde/estimator.mojo::kde_score_samples_host` and
`glm/estimator.mojo::ols_fit_host`.

THE SURFACE IS DESIGNED FOR THREE CALLERS THAT DO NOT EXIST YET, so its shape
is an argument rather than a convenience:

1. **`CholeskyFactor` carries the log-determinant.** A GP needs
   `log|K|` for its marginal likelihood, a GMM needs it per component, and
   kernel ridge needs it for its evidence. If each computes it from
   `factor.l` itself, there are three fold orders and three `log`s in the
   tree (IDENTITY_PATHS rows 21 and 12) and the identity claim splits three
   ways. So it is computed once, here, on the device, and handed over
   already done. DEVIATION 1639.
2. **`CholeskyFactor` carries `info`, `nb` and `jitter`.** Not for
   diagnostics: `info` is DATA-DEPENDENT (DEVIATION 1634), and a caller that
   drops it will happily solve against a partial factor and return numbers.
   `nb` and `jitter` are the two numeric parameters of the profile
   (DEVIATIONS 1630 and 1637), and a factor that does not carry them cannot
   be compared with another factor.
3. **Nothing here takes a block size.** `cholesky_factor_host` has no `nb`
   argument at all, so the ordinary caller cannot express the question
   DEVIATION 1630 refuses. The device-level `potrf_lower` does take a hint,
   because the check has to drive both sides of the pin and because
   NUMERIC_FAST is a real mode.

A caller that keeps its matrix ON THE DEVICE across many operations -- which
a GP fitting hyperparameters will -- should call
`cholesky/original/potrf.mojo::potrf_lower` and `trsm.mojo::cho_solve`
directly and keep its own `DeviceBuffer`s, exactly as cuML's `fit` keeps `X`
on the device and `score_samples` reuses it. This entry is the one-shot form,
which is what the gates and the card use.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from cholesky.original.potrf import (
    CHOL_ELEM_TPB,
    CHOL_NB_PINNED,
    CHOL_PANEL_TPB,
    add_jitter,
    chol_jitter_pinned,
    chol_logdet,
    chol_nb_for,
    chol_validate_jitter,
    chol_validate_matrix,
    chol_workspace_floats,
    potrf_lower,
)
from cholesky.original.trsm import CHOL_SOLVE_TPB, cho_solve
from cholesky.derived.linalg.cholesky_r1_update import (
    chol_rank1_update,
    chol_rank1_update_workspace_floats,
)


@fieldwise_init
struct CholeskyFactor(Movable):
    """`A = L L^T`, plus everything a caller must not recompute for itself."""

    var l: List[Float32]
    """`n x n` row-major; lower triangle is `L`, strict upper is `+0.0`."""

    var n: Int

    var info: Int
    """LAPACK's `info`. **CHECK IT.** 0 means `l` is a factor; `k > 0` means
    the leading minor of order `k` was not positive definite and `l` holds a
    partial result. DEVIATION 1634."""

    var logdet: Float32
    """`log |A|` (of the JITTERED `A`), computed on the device by the one
    pinned fold. Meaningless when `info != 0`, and set to `+0.0` there."""

    var nb: Int
    """The panel width that ran. Part of the profile, not a tuning record."""

    var jitter: Float32
    """The ridge that was added, by value. Part of the profile."""


def cholesky_profile_jitter() -> Float32:
    """The profile's ridge, re-exported so a downstream lane never has to
    reach into `cholesky/original/` for it -- and so that when it appears in
    a Gaussian process's source it appears as a NAME rather than as a
    literal somebody will later change. DEVIATION 1637."""
    return chol_jitter_pinned()


def _upload(
    ctx: DeviceContext, values: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    var n = len(values)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    for i in range(n):
        host.unsafe_ptr().unsafe_store(i, values[i])
    ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return buf^


def _download(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    return out^


def cholesky_factor_host(
    a: List[Float32],
    n: Int,
    jitter: Float32,
    panel_tpb: Int = CHOL_PANEL_TPB,
    elem_tpb: Int = CHOL_ELEM_TPB,
) raises -> CholeskyFactor:
    """`A = L L^T`, host in and host out, one shot.

    Validates on the host and refuses by name -- non-finite, non-symmetric,
    a bad dimension (DEVIATION 1638) and an unpinned jitter (DEVIATION 1637)
    -- BEFORE any upload. Then jitters, factors, and computes the
    log-determinant if the factorization succeeded.

    `jitter` is not defaulted, on purpose. Every caller of this in the three
    downstream lanes needs a ridge and every one of them needs to have
    decided about it; a default would let the decision be made by not making
    it. Pass `chol_jitter_pinned()` for the profile's ridge or
    `Float32(0.0)` for none.

    NO `nb` ARGUMENT. See this file's header, point 3.
    """
    chol_validate_matrix(a, n, "the matrix")
    chol_validate_jitter(jitter)
    var nb = chol_nb_for(n, CHOL_NB_PINNED)

    var ctx = DeviceContext()
    var da = _upload(ctx, a)
    var ws = ctx.enqueue_create_buffer[DType.float32](
        chol_workspace_floats(n, nb)
    )
    var dwork = ctx.enqueue_create_buffer[DType.float32](n + 1)
    ctx.synchronize()

    var trace = IdentityTrace()
    trace.header(
        "cholesky: n="
        + String(n)
        + " nb="
        + String(nb)
        + " jitter_bits=see CHOL_JITTER_BITS"
    )
    add_jitter(ctx, da, n, jitter, elem_tpb)
    trace.record_device(ctx, "chol.jittered", da, n * n)
    var run = potrf_lower(
        ctx, da, ws, n, trace, CHOL_NB_PINNED, panel_tpb, elem_tpb
    )
    var logdet = Float32(0.0)
    if run.info == 0:
        logdet = chol_logdet(ctx, da, dwork, n, trace, elem_tpb)
    var l = _download(ctx, da, n * n)
    _ = da^
    _ = ws^
    _ = dwork^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return CholeskyFactor(l^, n, run.info, logdet, run.nb, jitter)


def cholesky_solve_host(
    factor: CholeskyFactor,
    b: List[Float32],
    nrhs: Int,
    solve_tpb: Int = CHOL_SOLVE_TPB,
) raises -> List[Float32]:
    """`A X = B` given the factor. `b` is `n x nrhs` row-major; the return is
    `X` in the same shape. cuSOLVER's `potrs`.

    **REFUSES A FAILED FACTOR BY NAME.** A factor with `info != 0` has
    unfinished columns whose diagonal is whatever the trailing update last
    wrote, and dividing by those returns infinities and NaNs that look like
    numbers. cuSOLVER's `potrs` would run; this does not. The device-level
    `cho_solve` still trusts its caller, because that is the form a lane
    keeping buffers on the device calls in a loop.
    """
    if factor.info != 0:
        raise Error(
            "cholesky_solve_host: refusing to solve against a FAILED"
            " factorization (info="
            + String(factor.info)
            + "). The leading minor of order "
            + String(factor.info)
            + " was not positive definite, so columns "
            + String(factor.info - 1)
            + " onward of the factor are unfinished and solving against"
            " them returns infinities that look like numbers. Add a ridge"
            " (DEVIATION 1637) or fix the matrix. DEVIATION 1634"
        )
    var n = factor.n
    if nrhs <= 0:
        raise Error(
            "cholesky_solve_host: nrhs must be positive, got " + String(nrhs)
        )
    if len(b) != n * nrhs:
        raise Error(
            "cholesky_solve_host: the right-hand side holds "
            + String(len(b))
            + " floats, "
            + String(n)
            + " x "
            + String(nrhs)
            + " needs "
            + String(n * nrhs)
        )
    for i in range(len(b)):
        var v = b[i]
        if v != v:
            raise Error(
                "cholesky_solve_host: the right-hand side contains NaN at"
                " flat index "
                + String(i)
                + "; refused by name (DEVIATION 1638)"
            )

    var ctx = DeviceContext()
    var dl = _upload(ctx, factor.l)
    var db = _upload(ctx, b)
    ctx.synchronize()
    var trace = IdentityTrace()
    trace.header(
        "cholesky solve: n=" + String(n) + " nrhs=" + String(nrhs)
    )
    cho_solve(ctx, dl, db, n, nrhs, trace, solve_tpb)
    ctx.synchronize()
    var x = _download(ctx, db, n * nrhs)
    _ = dl^
    _ = db^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return x^


def cholesky_logdet_host(factor: CholeskyFactor) raises -> Float32:
    """`log |A|` of the jittered matrix, from the factor that already holds
    it. A function rather than a bare field read so that a caller who reaches
    for `logdet` on a failed factor is refused rather than handed `+0.0`."""
    if factor.info != 0:
        raise Error(
            "cholesky_logdet_host: the factorization failed (info="
            + String(factor.info)
            + "), so there is no determinant to report. DEVIATION 1634"
        )
    return factor.logdet


def cholesky_rank1_update_host(
    l: List[Float32], n: Int, ld: Int, eps: Float32
) raises -> List[Float32]:
    """`raft::linalg::choleskyRank1Update`, host in and host out.

    On entry `l` is `ld x ld` row-major holding the factor of the leading
    `(n-1) x (n-1)` block, with the new row of `A` in row `n-1`. On exit row
    `n-1` holds the new row of `L`. Pass a negative `eps` for RAFT's default
    "refuse rather than clamp"; DEVIATION 1633 governs the rest.
    """
    if n <= 0 or ld < n:
        raise Error(
            "cholesky_rank1_update_host: need 0 < n <= ld, got n="
            + String(n)
            + " ld="
            + String(ld)
        )
    if len(l) < ld * ld:
        raise Error(
            "cholesky_rank1_update_host: the factor holds "
            + String(len(l))
            + " floats, ld = "
            + String(ld)
            + " needs "
            + String(ld * ld)
        )
    var ctx = DeviceContext()
    var dl = _upload(ctx, l)
    var dws = ctx.enqueue_create_buffer[DType.float32](
        chol_rank1_update_workspace_floats(n)
    )
    ctx.synchronize()
    var trace = IdentityTrace()
    trace.header(
        "cholesky rank1: n=" + String(n) + " ld=" + String(ld)
    )
    chol_rank1_update(ctx, dl, dws, n, ld, eps, trace)
    ctx.synchronize()
    var out = _download(ctx, dl, ld * ld)
    _ = dl^
    _ = dws^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return out^
