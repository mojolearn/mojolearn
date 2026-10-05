# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host-pointer surface for the Cholesky lane: what three lanes will call.

Wired: `bindings/_mojolearn_gp.mojo` (built by `bindings/build_gp.sh`)
calls this file, and `python/mojolearn/_cholesky_impl.py` exposes it as
`mojolearn.Cholesky`; the Gaussian process, kernel ridge and Gaussian
mixture lanes reach it too. This file is the entry they reach, shaped like `kde/estimator.mojo::kde_score_samples_host` and
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
`cholesky/checks/potrf.mojo::potrf_lower` and `trsm.mojo::cho_solve`
directly and keep its own `DeviceBuffer`s, exactly as cuML's `fit` keeps `X`
on the device and `score_samples` reuses it. This entry is the one-shot form,
which is what the gates and the card use.
"""

# DEVIATION 2486: bulk host staging; stream/lifetime boundaries unchanged.
from bindings.hostptr import copy_f32
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from std.gpu import block_dim, block_idx, thread_idx
from checks.numerics import NUMERIC_FAST
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.ffi import _Global
from core.device_pool import pool_give, pool_take
from core.neural_context import neural_ctx
from checks.numerics import GLOBAL_NUMERIC_MODE as _CTX_MODE, NUMERIC_IDENTICAL as _CTX_IDENTICAL

#: ONE process-lifetime DeviceContext for every device entry of this binding
#: (`core/neural_context.mojo`). A context per call ran the M2 Pro out of Metal
#: command queues: later calls in a process REFUSED or returned output the
#: device never wrote (x-neighbors-svc-multiclass BATCH_MOVED, gp-optimize,
#: steward 1790601762837). Same kernels, same order, each entry still
#: synchronizes before it returns, so no bit moves.
comptime _CTX_NAME = "MojoCholeskyContextIdentical" if _CTX_MODE == _CTX_IDENTICAL else "MojoCholeskyContextFast"


def _binding_ctx() raises -> DeviceContext:
    return neural_ctx[_CTX_NAME]()

from core.identity_trace import IdentityTrace
from cholesky.checks.potrf import (
    chol_default_nb_hint,
    CHOL_ELEM_TPB,
    CHOL_NB_PINNED,
    CHOL_PANEL_TPB,
    add_jitter,
    chol_jitter_pinned,
    chol_logdet,
    chol_nb_for,
    CHOL_FAST_NOSYNC,
    chol_sym_rel_tol,
    chol_validate_jitter,
    chol_validate_matrix,
    chol_workspace_floats,
    potrf_lower,
)
from cholesky.checks.trsm import CHOL_SOLVE_TPB, cho_solve
from cholesky.impl.linalg.cholesky_r1_update import (
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
    reach into `cholesky/checks/` for it -- and so that when it appears in
    a Gaussian process's source it appears as a NAME rather than as a
    literal somebody will later change. DEVIATION 1637."""
    return chol_jitter_pinned()


def _upload(
    ctx: DeviceContext, values: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    var n = len(values)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    copy_f32(values.unsafe_ptr(), host.unsafe_ptr(), n)
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
    # one copy into a list of the final length (lane/neural-pass113): the
    # element-by-element append regrew the list and cost 0.43 s of a 1.1 s
    # 8192 factorization on the M4
    var out = List[Float32](length=n, fill=Float32(0))
    copy_f32(h.unsafe_ptr(), out.unsafe_ptr(), n)
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

    var ctx = _binding_ctx()
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
        ctx, da, ws, n, trace, chol_default_nb_hint(), panel_tpb, elem_tpb
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


# ---- lane/apple-fast-gap-linalg2 (2026-10-03): CHOL_FAST_DEVIO (default; _OFF reverts) ----
#: FAST + Apple: `cholesky_factor_binding` without the host Lists and the
#: host scans. Main's one-shot form copies the caller's n x n matrix into a
#: List (read_f32), scans it twice on ONE host thread (finite, then the
#: tiled symmetry test), copies the List into a staging buffer, and on the
#: way out copies the device result into another staging buffer, then a
#: List, then the caller's array: at n = 8192 that is five 256 MB host
#: copies and two serial 64 M-cell scans around a ~200 ms factorization.
#: Here the caller's matrix is copied once into the staging buffer, the
#: finite and symmetry predicates run on the device (one thread per cell,
#: the same predicate as `chol_validate_matrix`; a hit re-runs the host
#: validator on the caller's matrix for the by-name error), and the factor
#: comes back through one staging buffer straight into the caller's array.
#: The factorization itself is `potrf_lower` unchanged: the same words.
#: The FAST + Apple default since 2026-10-03 (M3 A/B, one run per arm:
#: cholesky synthetic 435 -> 290 ms, residual the same 1.659e-07; tag
#: gl2-chol-devio-synthetic). -D MOJOLEARN_CHOL_FAST_DEVIO_OFF restores
#: main's host-List route (the A/B arm).
#: lane/idn-gates (2026-10-04): also the IDENTICAL default on every vendor
#: (the same predicate on the device, `potrf_lower` unchanged, so the same
#: words; `defer_ok` is honoured under CHOL_FAST_NOSYNC only, which stays
#: FAST + Apple). -D MOJOLEARN_IDN_GATES_OFF (or the _OFF above) restores
#: the host-List route in IDENTICAL.
comptime CHOL_FAST_DEVIO = (
    (
        (_CTX_MODE == NUMERIC_FAST and has_apple_gpu_accelerator())
        or (_CTX_MODE == _CTX_IDENTICAL and not (is_defined["MOJOLEARN_IDN_GATES_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()))
    )
    and not is_defined["MOJOLEARN_CHOL_FAST_DEVIO_OFF"]()
)


def chol_devio_check_kernel(a: MutPointer[Float32, MutAnyOrigin], bad: MutPointer[Int32, MutAnyOrigin], n: Int32, tol: Float32):
    """Cell (i, j): non-finite, or (j < i) the relative symmetry test failed
    -> bad[0] = 1."""
    var nn = Int(n)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= nn * nn:
        return
    var i = t // nn
    var j = t - i * nn
    var v = a[t]
    var fail = (v != v) or v > Float32(3.4028234663852886e38) or v < Float32(-3.4028234663852886e38)
    if j < i and not fail:
        var up = a[j * nn + i]
        var d = abs(v - up)
        var m = max(abs(v), abs(up))
        fail = d > tol * m
    if fail:
        bad[0] = Int32(1)


# ---- lane/apple-fast-s-linalg (2026-10-04): CHOL_FAST_POOLIO ----
#: FAST + Apple only (rollback `-D MOJOLEARN_CHOL_FAST_POOLIO_OFF`), on top of
#: CHOL_FAST_DEVIO. What: DEVIO still makes, every fit at n = 8192, a FRESH
#: 256 MB pinned host buffer (page faults), copies the caller's matrix into
#: it on one host thread (`copy_f32`) and DMAs it up, and allocates the
#: 256 MB device matrix and the ~256 MB potrf workspace fresh (GPU first-touch
#: page faults). Here the caller's matrix goes up by the raw host-pointer
#: copy (measured 1.6-2.4 ms per 64 MB on the M4, faster than the staged
#: upload; x_decomp/device.mojo `_up_into`), the device matrix and workspace
#: come from a named device pool (core/device_pool.mojo) and the download
#: stage is one pinned host buffer kept between fits (grown when n grows).
#: The NOSYNC redo re-uploads from the caller's matrix. `potrf_lower`
#: unchanged: the same words. Expect: cholesky synthetic 261 -> ~200-235 ms.
#: CHOL_FAST_POOLIO OUTCOME (M3 afc_ab_def, full board size, 1 run per arm,
#: 2026-10-05, rab15-cholpoolio): cholesky 262.4 -> 228.1 ms, residual
#: identical, digest same. KEEP: the FAST + Apple default; rollback
#: -D MOJOLEARN_CHOL_FAST_POOLIO_OFF.
comptime CHOL_FAST_POOLIO = CHOL_FAST_DEVIO and not is_defined["MOJOLEARN_CHOL_FAST_POOLIO_OFF"]()
comptime _CHOL_POOL = "MojoCholFastPoolIO"


struct _CholStage(Defaultable, Movable):
    var buf: Optional[HostBuffer[DType.float32]]
    var n: Int

    def __init__(out self):
        self.buf = Optional[HostBuffer[DType.float32]]()
        self.n = 0


comptime CHOL_STAGE = _Global[StorageType=_CholStage, name="MojoCholFastPoolIOStage", init_fn=_CholStage.__init__]


def _chol_stage_ptr(ctx: DeviceContext, cells: Int) raises -> MutPointer[Float32, MutAnyOrigin]:
    """The kept pinned download stage, at least `cells` floats."""
    var slot = CHOL_STAGE.get_or_create_ptr()
    if not slot[].buf or slot[].n < cells:
        slot[].buf = Optional[HostBuffer[DType.float32]]()
        slot[].buf = ctx.enqueue_create_host_buffer[DType.float32](cells)
        slot[].n = cells
        ctx.synchronize()
    return MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(slot[].buf.value().unsafe_ptr()))


def cholesky_factor_poolio(
    ap: MutPointer[Float32, MutUntrackedOrigin],
    lp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
    n: Int,
    jitter: Float32,
) raises -> Int:
    """`cholesky_factor_devio` with CHOL_FAST_POOLIO's IO (see above)."""
    if n <= 0:
        raise Error("cholesky: the matrix must have a positive dimension, got n=" + String(n))
    chol_validate_jitter(jitter)
    var nb = chol_nb_for(n, CHOL_NB_PINNED)
    var cells = n * n
    var ctx = _binding_ctx()
    var da = pool_take[_CHOL_POOL](ctx, cells)
    var ws = pool_take[_CHOL_POOL](ctx, chol_workspace_floats(n, nb))
    var dbad = ctx.enqueue_create_buffer[DType.int32](1)
    var hbad = ctx.enqueue_create_host_buffer[DType.int32](1)
    var dwork = ctx.enqueue_create_buffer[DType.float32](n + 1)
    ctx.synchronize()
    hbad.unsafe_ptr()[0] = Int32(0)
    ctx.enqueue_copy(dst_buf=dbad, src_ptr=hbad.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=da, src_ptr=ap)
    ctx.enqueue_function[chol_devio_check_kernel](
        da.unsafe_ptr(), dbad.unsafe_ptr(), Int32(n), chol_sym_rel_tol(),
        grid_dim=((cells + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_copy(dst_ptr=hbad.unsafe_ptr(), src_buf=dbad)
    ctx.synchronize()
    if hbad.unsafe_ptr()[0] != Int32(0):
        # the by-name refusal: the host validator on the caller's matrix
        # (as DEVIO: a pass falls through to the factor)
        var a = List[Float32](unsafe_uninit_length=cells)
        copy_f32(ap, a.unsafe_ptr(), cells)
        chol_validate_matrix(a, n, "the matrix")
    var trace = IdentityTrace()
    add_jitter(ctx, da, n, jitter, CHOL_ELEM_TPB)
    var run = potrf_lower(
        ctx, da, ws, n, trace, chol_default_nb_hint(), CHOL_PANEL_TPB, CHOL_ELEM_TPB, defer_ok=True
    )
    comptime if CHOL_FAST_NOSYNC:
        if run.info != 0:
            # the deferred route left a full (not partial) sweep: redo it
            # with the per-panel reads for LAPACK's partial factor
            ctx.enqueue_copy(dst_buf=da, src_ptr=ap)
            add_jitter(ctx, da, n, jitter, CHOL_ELEM_TPB)
            run = potrf_lower(ctx, da, ws, n, trace, chol_default_nb_hint(), CHOL_PANEL_TPB, CHOL_ELEM_TPB)
    var logdet = Float32(0.0)
    if run.info == 0:
        logdet = chol_logdet(ctx, da, dwork, n, trace, CHOL_ELEM_TPB)
    var stage = _chol_stage_ptr(ctx, cells)
    ctx.enqueue_copy(dst_ptr=stage, src_buf=da)
    ctx.synchronize()
    copy_f32(stage, lp, cells)
    sp.unsafe_store(0, Float64(run.info))
    sp.unsafe_store(1, Float64(run.nb))
    sp.unsafe_store(2, Float64(logdet))
    sp.unsafe_store(3, Float64(jitter))
    pool_give[_CHOL_POOL](da^)
    pool_give[_CHOL_POOL](ws^)
    _ = dbad^
    _ = hbad^
    _ = dwork^
    _ = ctx^
    return run.info


def cholesky_factor_devio(
    ap: MutPointer[Float32, MutUntrackedOrigin],
    lp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
    n: Int,
    jitter: Float32,
) raises -> Int:
    """`cholesky_factor_host` + the binding's copies, FAST IO (see above):
    writes L into lp and info, nb, logdet, jitter into sp; returns info."""
    comptime if CHOL_FAST_POOLIO:
        return cholesky_factor_poolio(ap, lp, sp, n, jitter)
    if n <= 0:
        raise Error("cholesky: the matrix must have a positive dimension, got n=" + String(n))
    chol_validate_jitter(jitter)
    var nb = chol_nb_for(n, CHOL_NB_PINNED)
    var cells = n * n
    var ctx = _binding_ctx()
    var host = ctx.enqueue_create_host_buffer[DType.float32](cells)
    var da = ctx.enqueue_create_buffer[DType.float32](cells)
    var dbad = ctx.enqueue_create_buffer[DType.int32](1)
    var hbad = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.synchronize()
    copy_f32(ap, host.unsafe_ptr(), cells)
    hbad.unsafe_ptr()[0] = Int32(0)
    ctx.enqueue_copy(dst_buf=dbad, src_ptr=hbad.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=da, src_ptr=host.unsafe_ptr())
    ctx.enqueue_function[chol_devio_check_kernel](
        da.unsafe_ptr(), dbad.unsafe_ptr(), Int32(n), chol_sym_rel_tol(),
        grid_dim=((cells + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_copy(dst_ptr=hbad.unsafe_ptr(), src_buf=dbad)
    var ws = ctx.enqueue_create_buffer[DType.float32](chol_workspace_floats(n, nb))
    var dwork = ctx.enqueue_create_buffer[DType.float32](n + 1)
    ctx.synchronize()
    if hbad.unsafe_ptr()[0] != Int32(0):
        # the by-name refusal: the host validator on the caller's matrix
        var a = List[Float32](unsafe_uninit_length=cells)
        copy_f32(ap, a.unsafe_ptr(), cells)
        chol_validate_matrix(a, n, "the matrix")
    var trace = IdentityTrace()
    # the host route's trace records (no-ops unless the trace is enabled)
    trace.header("cholesky: n=" + String(n) + " nb=" + String(nb) + " jitter_bits=see CHOL_JITTER_BITS")
    add_jitter(ctx, da, n, jitter, CHOL_ELEM_TPB)
    trace.record_device(ctx, "chol.jittered", da, n * n)
    var run = potrf_lower(
        ctx, da, ws, n, trace, chol_default_nb_hint(), CHOL_PANEL_TPB, CHOL_ELEM_TPB, defer_ok=True
    )
    comptime if CHOL_FAST_NOSYNC:
        if run.info != 0:
            # the deferred route left a full (not partial) sweep: redo it
            # with the per-panel reads for LAPACK's partial factor
            ctx.enqueue_copy(dst_buf=da, src_ptr=host.unsafe_ptr())
            add_jitter(ctx, da, n, jitter, CHOL_ELEM_TPB)
            run = potrf_lower(ctx, da, ws, n, trace, chol_default_nb_hint(), CHOL_PANEL_TPB, CHOL_ELEM_TPB)
    var logdet = Float32(0.0)
    if run.info == 0:
        logdet = chol_logdet(ctx, da, dwork, n, trace, CHOL_ELEM_TPB)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=da)
    ctx.synchronize()
    copy_f32(host.unsafe_ptr(), lp, cells)
    sp.unsafe_store(0, Float64(run.info))
    sp.unsafe_store(1, Float64(run.nb))
    sp.unsafe_store(2, Float64(logdet))
    sp.unsafe_store(3, Float64(jitter))
    _ = host^
    _ = da^
    _ = dbad^
    _ = hbad^
    _ = ws^
    _ = dwork^
    _ = ctx^
    return run.info


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

    var ctx = _binding_ctx()
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
    var ctx = _binding_ctx()
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
