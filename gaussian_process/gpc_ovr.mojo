# ===----------------------------------------------------------------------=== #
# Gaussian process classification: the whole one-vs-rest fit and the whole
# one-vs-rest prediction, each in ONE device session (lane fam2-kernel-gp,
# 2026-10-04).
# ===----------------------------------------------------------------------=== #
"""`GaussianProcessClassifier.fit` / `predict` / `predict_proba` with every
class in one call.

Before, `_gpc_impl.py` called `gpc_fit` once per class and `gpc_predict` once
per class. Every one of those calls copied X into an owned host list, walked
it on one host thread for NaN / infinity, uploaded it, computed the SAME
kernel matrix again (K(X, X) in fit, K(X, X_star) in predict: neither reads
the class), and read its results back through host lists. The probability
columns then crossed to the host and back for the one-vs-rest combine, and a
binary model's `[1 - p, p]` rows and `mean > 0` codes were host loops.

Here:
  - X goes to the device once, from the caller's memory, and is scanned for
    NaN / infinity there (`core/device_scan.mojo`);
  - K(X, X) (fit) and K(X, X_star) (predict) are computed ONCE for all classes;
  - the class-k targets `code == k` are a device kernel;
  - the Newton loop's log-determinant stays on the device
    (`GPC_IDN_NEWTON_LOGDET`): `chol_logdet`'s read-back and its per-iteration
    scratch allocation are gone, the likelihood kernel reads the scalar where
    it lies;
  - the probability columns stay on the device for the combine, the binary
    pairs and codes are device kernels, and only the output asked for
    (float64 probabilities or int64 codes) is read back.

BITS. The kernels, their launch order and every fold are the ones the
per-class route runs (`classifier.mojo`), so every output keeps its bits: the
host column (`gaussian_process/host/gpc_*.mojo`) is untouched.
`1 - p` is `sf64_sub(1, p)`, the correctly rounded binary64 subtraction
Python's float gives.

Switches (IDENTICAL only, ON by default, all off under
`MOJOLEARN_IDN_ALL_OFF`):
  `-D MOJOLEARN_IDN_GPC_OVR_OFF`            `gp_idn_caps` answers 0 and the
                                            Python glue takes the per-class
                                            binding calls
  `-D MOJOLEARN_IDN_GPC_NEWTON_LOGDET_OFF`  the Newton loop calls
                                            `chol_logdet` (a read-back per
                                            iteration) as before
"""

from std.atomic import Atomic
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.host.dim import Dim

from checks.numerics import GLOBAL_NUMERIC_MODE as _CTX_MODE, NUMERIC_IDENTICAL as _CTX_IDENTICAL
from checks.soft_f64 import SF64_ONE, sf64_sub
from cholesky.checks.potrf import (
    CHOL_ELEM_TPB,
    CHOL_NB_PINNED,
    CHOL_PANEL_TPB,
    add_jitter,
    chol_default_nb_hint,
    chol_logdet,
    chol_nb_for,
    chol_workspace_floats,
    enqueue_logdet,
    potrf_lower,
)
from cholesky.checks.trsm import CHOL_SOLVE_TPB, cho_solve, trsm_lower
from cholesky.impl.matrix.detail.matrix import copy_vector_from_matrix_diagonal_kernel
from cholesky.logdet_fold import logdet_blocks
from core.device_scan import device_classify_nonfinite, device_first_nonfinite
from core.device_zero import enqueue_fill
from core.identity_trace import IdentityTrace
from gaussian_process.checks.gp_sabotage import GP_SAB_NONE
from gaussian_process.checks.kernels import (
    GP_ELEM_TPB,
    GPKernelSpec,
    gp_kernel_diag,
    gp_kernel_matrix,
    gp_kernel_stack_floats,
    gp_validate_kernel,
)
from gaussian_process.classifier import (
    GPC_B_TPB,
    GPC_STEP_TPB,
    _family_ctx,
    gpc_a_kernel,
    gpc_b_matrix_kernel,
    gpc_lml_fin_kernel,
    gpc_lml_part_kernel,
    gpc_ovr_combine_kernel,
    gpc_proba_kernel,
    gpc_residual_kernel,
    gpc_rhs_kernel,
    gpc_scale_nan_kernel,
    gpc_w_kernel,
)
from gaussian_process.estimator import _length_scale_table, _upload
from gaussian_process.gpc_common import gpc_neg_inf32, gpc_stop, gpc_validate_max_iter
from gaussian_process.gpc_device_var import (
    GPC_VAR_TPB,
    gpc_latent_var_launch,
    gpc_scale_rows_kernel,
)
from gaussian_process.gpc_items import gpc_fold_blocks, gpc_lml_fin
from gemm.checks.gemm_identical import (
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.contract import OP_TN
from std.gpu import block_idx, thread_idx

#: fam2-kernel-gp (2026-10-04), IDENTICAL, ON by default
#: (`-D MOJOLEARN_IDN_GPC_OVR_OFF` restores the per-class binding calls).
comptime GPC_IDN_OVR = _CTX_MODE == _CTX_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_GPC_OVR_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)

#: fam2-kernel-gp (2026-10-04), IDENTICAL, ON by default
#: (`-D MOJOLEARN_IDN_GPC_NEWTON_LOGDET_OFF` restores `chol_logdet`): log|B|
#: stays on the device inside the Newton loop of this file. The same copy of
#: the diagonal and the same fold (`enqueue_logdet`): no bit moves.
comptime GPC_IDN_NEWTON_LOGDET = _CTX_MODE == _CTX_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_GPC_NEWTON_LOGDET_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)

#: `gpc_predict_all_device` output kinds (the binding's `out_kind`)
comptime GPC_ALL_CODES = 1
comptime GPC_ALL_PROBA = 2

comptime _GP = MutPointer[Float32, MutAnyOrigin]
comptime _GI = MutPointer[Int32, MutAnyOrigin]
comptime _GL = MutPointer[Int64, MutAnyOrigin]
comptime _GU = MutPointer[UInt64, MutAnyOrigin]


@always_inline
def _tid() -> Int:
    return Int(block_idx.x) * GPC_STEP_TPB + Int(thread_idx.x)


@always_inline
def _blocks(count: Int) -> Int:
    return (count + GPC_STEP_TPB - 1) // GPC_STEP_TPB if count > 0 else 1


@always_inline
def _fp(buf: DeviceBuffer[DType.float32]) -> _GP:
    return _GP(unsafe_from_address=Int(buf.unsafe_ptr()))


@always_inline
def _ip(buf: DeviceBuffer[DType.int32]) -> _GI:
    return _GI(unsafe_from_address=Int(buf.unsafe_ptr()))


@always_inline
def _lp(buf: DeviceBuffer[DType.int64]) -> _GL:
    return _GL(unsafe_from_address=Int(buf.unsafe_ptr()))


@always_inline
def _up(buf: DeviceBuffer[DType.uint64]) -> _GU:
    return _GU(unsafe_from_address=Int(buf.unsafe_ptr()))


# ===========================================================================
# KERNELS (one thread per row; nothing here folds)
# ===========================================================================


def gpc_targets_kernel(codes: _GI, y: _GP, seen: _GI, k: Int32, n: Int32):
    """The one-vs-rest targets of class k: 1 where the class code is k, 0
    elsewhere (`unnorm.mojo::gpc_ovr_targets`, a host loop). `seen[1]` /
    `seen[0]` take the lowest row carrying a 1 / a 0 by an atomic min (they
    start at n), so the caller can refuse a column with one class."""
    var i = _tid()
    if i < Int(n):
        var hit = codes.unsafe_load(i) == k
        y.unsafe_store(i, Float32(1.0) if hit else Float32(0.0))
        if hit:
            _ = Atomic[DType.int32].min(seen + 1, Int32(i))
        else:
            _ = Atomic[DType.int32].min(seen, Int32(i))


def gpc_lml_fin_dev_kernel(pdot: _GP, pt2: _GP, nb: Int32, logdet: _GP, dst: _GP):
    """`gpc_lml_fin_kernel` with log|B| read from the device scalar
    `enqueue_logdet` wrote, not passed through the host."""
    if _tid() == 0:
        dst.unsafe_store(0, gpc_lml_fin(pdot, pt2, Int(nb), logdet.unsafe_load(0)))


def gpc_mean_codes_kernel(mean: _GP, dst: _GL, n: Int32):
    """The binary `predict`: 1 where the float32 latent mean is > 0
    (`_gpc.py:287-290`; a NaN takes 0), as int64
    (`unnorm.mojo::gpc_binary_out` kind 1, a host loop)."""
    var t = _tid()
    if t < Int(n):
        dst.unsafe_store(t, Int64(1) if mean.unsafe_load(t) > Float32(0.0) else Int64(0))


def gpc_pairs_kernel(p: _GU, dst: _GU, n: Int32):
    """The binary `predict_proba` rows `[1 - p, p]` in software binary64
    (`unnorm.mojo::gpc_binary_out` kind 2, a host loop with one float64
    subtraction a row)."""
    var t = _tid()
    if t < Int(n):
        var v = p.unsafe_load(t)
        dst.unsafe_store(2 * t, sf64_sub(SF64_ONE, v))
        dst.unsafe_store(2 * t + 1, v)


def gpc_codes_widen_kernel(src: _GI, dst: _GL, n: Int32):
    """The one-vs-rest argmax codes widened to int64 (it was `astype` on the
    host)."""
    var t = _tid()
    if t < Int(n):
        dst.unsafe_store(t, Int64(src.unsafe_load(t)))


# ===========================================================================
# INPUT
# ===========================================================================


def _gpc_upload_checked(
    ctx: DeviceContext, addr: Int, rows: Int, cols: Int, what: String
) raises -> DeviceBuffer[DType.float32]:
    """`rows x cols` float32 from the caller's memory to the device, then
    `gp_validate_data`'s refusals (DEVIATION 1768) from a device scan: the
    row and the feature of the first NaN or infinity."""
    if rows <= 0:
        raise Error("gpr: " + what + " must have at least one row, got " + String(rows))
    if cols <= 0:
        raise Error("gpr: " + what + " must have at least one feature, got " + String(cols))
    if addr == 0:
        raise Error("gpc: " + what + " has a null address")
    var n = rows * cols
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_buf=buf, src_ptr=_GP(unsafe_from_address=addr))
    var bad = device_first_nonfinite(ctx, buf, n)
    if bad >= 0:
        var is_nan = device_classify_nonfinite(ctx, buf, bad)
        raise Error(
            "gpr: "
            + what
            + (" contains NaN at row " if is_nan else " contains infinity at row ")
            + String(bad // cols)
            + ", feature "
            + String(bad % cols)
            + "; refused by name before any model launch (DEVIATION 1768)"
        )
    return buf^


def _copy_in(ctx: DeviceContext, mut dst: DeviceBuffer[DType.float32], addr: Int, what: String) raises:
    """One host-to-device copy from the caller's memory, enqueued (the
    caller's array outlives the binding call)."""
    if addr == 0:
        raise Error("gpc: " + what + " has a null address")
    ctx.enqueue_copy(dst_buf=dst, src_ptr=_GP(unsafe_from_address=addr))


# ===========================================================================
# FIT
# ===========================================================================


@fieldwise_init
struct GPCNewtonRun(Movable):
    """What the Newton loop leaves besides the resident factor, pi and W_sr:
    the likelihood kept (DEVIATION 2830), the iterations and the Cholesky
    panel width that ran."""

    var lml: Float32
    var n_iter: Int
    var nb: Int


def gpc_newton_device(
    ctx: DeviceContext,
    mut dk: DeviceBuffer[DType.float32],
    mut dy: DeviceBuffer[DType.float32],
    mut db: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    mut dwork: DeviceBuffer[DType.float32],
    mut dgw: DeviceBuffer[DType.float32],
    mut dpi: DeviceBuffer[DType.float32],
    mut dwsr: DeviceBuffer[DType.float32],
    n: Int,
    max_iter_predict: Int,
) raises -> GPCNewtonRun:
    """`classifier.mojo::_gpc_fit_binary_device`'s Newton loop against a
    resident K and resident targets: the same launches in the same order.
    On return `db` holds the last factor, `dpi` and `dwsr` the last pi and
    W_sr (the caller reads them back). One word pair crosses back per
    iteration for the stop test: the likelihood and the NaN guard's index."""
    var df = ctx.enqueue_create_buffer[DType.float32](n)
    var dw = ctx.enqueue_create_buffer[DType.float32](n)
    var dbv = ctx.enqueue_create_buffer[DType.float32](n)
    var dkb = ctx.enqueue_create_buffer[DType.float32](n)
    var dc = ctx.enqueue_create_buffer[DType.float32](n)
    var da = ctx.enqueue_create_buffer[DType.float32](n)
    var nbf = gpc_fold_blocks(n)
    var dpd = ctx.enqueue_create_buffer[DType.float32](max(nbf, 1))
    var dpt = ctx.enqueue_create_buffer[DType.float32](max(nbf, 1))
    var dlml = ctx.enqueue_create_buffer[DType.float32](1)
    var dnan = ctx.enqueue_create_buffer[DType.int32](1)
    var dlparts = ctx.enqueue_create_buffer[DType.float32](2 * logdet_blocks(n))
    var diag = dwork.create_sub_buffer[DType.float32](0, n)
    var scalar = dwork.create_sub_buffer[DType.float32](n, 1)
    var hlml = ctx.enqueue_create_host_buffer[DType.float32](1)
    var hnan = ctx.enqueue_create_host_buffer[DType.int32](1)
    enqueue_fill(ctx, df, Float32(0.0))
    var trace = IdentityTrace()
    var previous = gpc_neg_inf32()
    var n_iter = 0
    var nb = 0
    # Resolved once, launched by handle (classifier.mojo's form for this kernel).
    var b_kernel = ctx.compile_function[gpc_b_matrix_kernel]()
    for it in range(max_iter_predict):
        ctx.enqueue_function[gpc_w_kernel](
            _fp(df), _fp(dpi), _fp(dw), _fp(dwsr), Int32(n), grid_dim=_blocks(n), block_dim=GPC_STEP_TPB
        )
        ctx.enqueue_function(
            b_kernel, db.unsafe_ptr(), dk.unsafe_ptr(), dwsr.unsafe_ptr(), Int32(n),
            grid_dim=Dim((n * n + GPC_B_TPB - 1) // GPC_B_TPB, 1, 1),
            block_dim=Dim(GPC_B_TPB, 1, 1),
        )
        add_jitter(ctx, db, n, Float32(0.0), CHOL_ELEM_TPB)
        var run = potrf_lower(
            ctx, db, ws, n, trace, chol_default_nb_hint(), CHOL_PANEL_TPB, CHOL_ELEM_TPB
        )
        if run.info != 0:
            raise Error(
                "gpc_fit_host: the factorization of B = I + W_sr K W_sr failed"
                " at Newton iteration "
                + String(it + 1)
                + " (info="
                + String(run.info)
                + "). B's eigenvalues are at least 1 for any finite kernel"
                " matrix, so this means a non-finite latent value"
            )
        var logdet_host = Float32(0.0)
        comptime if GPC_IDN_NEWTON_LOGDET:
            # `chol_logdet`'s two launches, its scalar left where it lies
            ctx.enqueue_function[copy_vector_from_matrix_diagonal_kernel](
                diag.unsafe_ptr(), db.unsafe_ptr(), Int32(n), Int32(n),
                grid_dim=((n + CHOL_ELEM_TPB - 1) // CHOL_ELEM_TPB, 1, 1), block_dim=(CHOL_ELEM_TPB, 1, 1),
            )
            enqueue_logdet(ctx, _fp(diag), _fp(dlparts), _fp(scalar), n)
        else:
            logdet_host = chol_logdet(ctx, db, dwork, n, trace, CHOL_ELEM_TPB)
        ctx.enqueue_function[gpc_rhs_kernel](
            _fp(dw), _fp(df), _fp(dy), _fp(dpi), _fp(dbv), Int32(n), grid_dim=_blocks(n), block_dim=GPC_STEP_TPB
        )
        identical_gemm_into(ctx, dkb, dk, dbv, dgw, n, 1, n, OP_TN)
        enqueue_fill(ctx, dnan, Int32(n))
        ctx.enqueue_function[gpc_scale_nan_kernel](
            _fp(dwsr), _fp(dkb), _fp(dc), _ip(dnan), Int32(n),
            grid_dim=_blocks(n), block_dim=GPC_STEP_TPB,
        )
        cho_solve(ctx, db, dc, n, 1, trace, CHOL_SOLVE_TPB)
        ctx.enqueue_function[gpc_a_kernel](
            _fp(dbv), _fp(dwsr), _fp(dc), _fp(da), Int32(n), grid_dim=_blocks(n), block_dim=GPC_STEP_TPB
        )
        identical_gemm_into(ctx, df, dk, da, dgw, n, 1, n, OP_TN)
        if nbf > 0:
            ctx.enqueue_function[gpc_lml_part_kernel](
                _fp(da), _fp(df), _fp(dy), Int32(n), _fp(dpd), _fp(dpt), Int32(nbf),
                grid_dim=_blocks(nbf), block_dim=GPC_STEP_TPB,
            )
        comptime if GPC_IDN_NEWTON_LOGDET:
            ctx.enqueue_function[gpc_lml_fin_dev_kernel](
                _fp(dpd), _fp(dpt), Int32(nbf), _fp(scalar), _fp(dlml), grid_dim=1, block_dim=1
            )
        else:
            ctx.enqueue_function[gpc_lml_fin_kernel](
                _fp(dpd), _fp(dpt), Int32(nbf), logdet_host, _fp(dlml), grid_dim=1, block_dim=1
            )
        ctx.enqueue_copy(dst_ptr=hlml.unsafe_ptr(), src_buf=dlml)
        ctx.enqueue_copy(dst_ptr=hnan.unsafe_ptr(), src_buf=dnan)
        ctx.synchronize()
        var nan_at = Int(hnan.unsafe_ptr().unsafe_load(0))
        if nan_at < n:
            raise Error(
                "cholesky_solve_host: the right-hand side contains NaN at"
                " flat index "
                + String(nan_at)
                + "; refused by name (DEVIATION 1638)"
            )
        var lml = hlml.unsafe_ptr().unsafe_load(0)
        n_iter = it + 1
        nb = run.nb
        if gpc_stop(lml, previous):
            break
        previous = lml
    ctx.synchronize()
    _ = diag^
    _ = scalar^
    _ = df^
    _ = dw^
    _ = dbv^
    _ = dkb^
    _ = dc^
    _ = da^
    _ = dpd^
    _ = dpt^
    _ = dlml^
    _ = dnan^
    _ = dlparts^
    _ = hlml^
    _ = hnan^
    return GPCNewtonRun(previous, n_iter, nb)


def gpc_fit_all_device(
    x_addr: Int,
    codes_addr: Int,
    n_train: Int,
    n_features: Int,
    kernel: GPKernelSpec,
    max_iter_predict: Int,
    class_ks: List[Int],
    y_addrs: List[Int],
    l_addrs: List[Int],
    pi_addrs: List[Int],
    wsr_addrs: List[Int],
    scalar_addrs: List[Int],
) raises:
    """Every binary Laplace fit of one classifier (`class_ks` holds the
    class of each: `[1]` for two classes, `0 .. K-1` past two) against ONE
    upload of X and ONE kernel matrix.

    Fit j writes, into the caller's memory: `y_addrs[j]` the n_train float32
    targets `code == class_ks[j]`, `l_addrs[j]` the n_train^2 float32 factor,
    `pi_addrs[j]` and `wsr_addrs[j]` n_train float32 each, and
    `scalar_addrs[j]` three float64 (lml, n_iter, nb), `gpc_fit`'s outputs."""
    var n_fits = len(class_ks)
    if n_fits <= 0:
        raise Error("gpc_fit_all: needs at least one binary fit, got " + String(n_fits))
    if (
        len(y_addrs) != n_fits
        or len(l_addrs) != n_fits
        or len(pi_addrs) != n_fits
        or len(wsr_addrs) != n_fits
        or len(scalar_addrs) != n_fits
    ):
        raise Error("gpc_fit_all: every output list must hold one address per binary fit")
    gp_validate_kernel(kernel, n_features)
    gpc_validate_max_iter(max_iter_predict)
    if codes_addr == 0:
        raise Error("gpc_fit_all: the class codes have a null address")
    var n = n_train
    var ctx = _family_ctx()
    var dx = _gpc_upload_checked(ctx, x_addr, n_train, n_features, String("X"))
    var dls = _upload(ctx, _length_scale_table(kernel))
    var dk = ctx.enqueue_create_buffer[DType.float32](n * n)
    var dstack = ctx.enqueue_create_buffer[DType.float32](gp_kernel_stack_floats(n, n))
    var ktrace = IdentityTrace()
    ctx.synchronize()
    gp_kernel_matrix(
        ctx, dk, dx, dx, dls, dstack, n, n, n_features, kernel, True, ktrace,
        "gpc.kernel", GP_ELEM_TPB, GP_SAB_NONE,
    )
    ctx.synchronize()
    _ = dx^
    _ = dls^
    _ = dstack^

    var dcodes = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_buf=dcodes, src_ptr=_GI(unsafe_from_address=codes_addr))
    var dy = ctx.enqueue_create_buffer[DType.float32](n)
    var dseen = ctx.enqueue_create_buffer[DType.int32](2)
    var hseen = ctx.enqueue_create_host_buffer[DType.int32](2)
    var db = ctx.enqueue_create_buffer[DType.float32](n * n)
    var nb_pin = chol_nb_for(n, CHOL_NB_PINNED)
    var ws = ctx.enqueue_create_buffer[DType.float32](chol_workspace_floats(n, nb_pin))
    var dwork = ctx.enqueue_create_buffer[DType.float32](n + 1)
    var dgw = ctx.enqueue_create_buffer[DType.float32](identical_gemm_workspace_max_floats(n, 1, n))
    var dpi = ctx.enqueue_create_buffer[DType.float32](n)
    var dwsr = ctx.enqueue_create_buffer[DType.float32](n)
    for j in range(n_fits):
        if y_addrs[j] == 0 or l_addrs[j] == 0 or pi_addrs[j] == 0 or wsr_addrs[j] == 0 or scalar_addrs[j] == 0:
            raise Error("gpc_fit_all: binary fit " + String(j) + " has a null output address")
        enqueue_fill(ctx, dseen, Int32(n))
        ctx.enqueue_function[gpc_targets_kernel](
            _ip(dcodes), _fp(dy), _ip(dseen), Int32(class_ks[j]), Int32(n),
            grid_dim=_blocks(n), block_dim=GPC_STEP_TPB,
        )
        ctx.enqueue_copy(dst_ptr=hseen.unsafe_ptr(), src_buf=dseen)
        ctx.enqueue_copy(dst_ptr=_GP(unsafe_from_address=y_addrs[j]), src_buf=dy)
        ctx.synchronize()
        if Int(hseen.unsafe_ptr().unsafe_load(0)) >= n or Int(hseen.unsafe_ptr().unsafe_load(1)) >= n:
            raise Error(
                "gpc_fit_host: a binary Laplace fit requires 2 classes; got 1"
                " (one-vs-rest class "
                + String(class_ks[j])
                + ")"
            )
        var run = gpc_newton_device(ctx, dk, dy, db, ws, dwork, dgw, dpi, dwsr, n, max_iter_predict)
        ctx.enqueue_copy(dst_ptr=_GP(unsafe_from_address=l_addrs[j]), src_buf=db)
        ctx.enqueue_copy(dst_ptr=_GP(unsafe_from_address=pi_addrs[j]), src_buf=dpi)
        ctx.enqueue_copy(dst_ptr=_GP(unsafe_from_address=wsr_addrs[j]), src_buf=dwsr)
        ctx.synchronize()
        # lml, n_iter, nb, in that order; each widens to float64 exactly.
        var sp = MutPointer[Float64, MutAnyOrigin](unsafe_from_address=scalar_addrs[j])
        sp.unsafe_store(0, Float64(run.lml))
        sp.unsafe_store(1, Float64(run.n_iter))
        sp.unsafe_store(2, Float64(run.nb))
    ctx.synchronize()
    _ = dk^
    _ = dcodes^
    _ = dy^
    _ = dseen^
    _ = hseen^
    _ = db^
    _ = ws^
    _ = dwork^
    _ = dgw^
    _ = dpi^
    _ = dwsr^
    # DEVIATION 1946: the context dies LAST.
    _ = ctx^


# ===========================================================================
# PREDICT
# ===========================================================================


def gpc_predict_all_device(
    xt_addr: Int,
    n_train: Int,
    n_features: Int,
    kernel: GPKernelSpec,
    xs_addr: Int,
    n_star: Int,
    y_addrs: List[Int],
    pi_addrs: List[Int],
    wsr_addrs: List[Int],
    l_addrs: List[Int],
    out_kind: Int,
    out_addr: Int,
) raises:
    """`predict` (`out_kind` 1: n_star int64 class codes at `out_addr`) or
    `predict_proba` (`out_kind` 2: n_star x max(k, 2) float64, row-major)
    of a fitted classifier with k = len(y_addrs) binary fits, against ONE
    upload of X_train and X_star and ONE cross kernel.

    One binary fit: the codes are `mean > 0` (no variance is computed) and
    the probabilities `[1 - p, p]`. Past one: DEVIATION 2833's combine of
    the k class-1 probability columns, which never leave the device."""
    var k = len(y_addrs)
    if k <= 0:
        raise Error("gpc_predict_all: needs at least one binary fit, got " + String(k))
    if len(pi_addrs) != k or len(wsr_addrs) != k or len(l_addrs) != k:
        raise Error("gpc_predict_all: every model list must hold one address per binary fit")
    if out_kind != GPC_ALL_CODES and out_kind != GPC_ALL_PROBA:
        raise Error("gpc_predict_all: unknown output kind " + String(out_kind))
    if n_star <= 0:
        raise Error("gpc_predict_host: n_star must be positive, got " + String(n_star))
    if out_addr == 0:
        raise Error("gpc_predict_all: the output has a null address")
    gp_validate_kernel(kernel, n_features)
    var kss = gp_kernel_diag(kernel)
    var binary = k == 1
    var want_proba = not (binary and out_kind == GPC_ALL_CODES)

    var trace = IdentityTrace()
    var ctx = _family_ctx()
    var dx = _gpc_upload_checked(ctx, xt_addr, n_train, n_features, String("X_train"))
    var dxs = _gpc_upload_checked(ctx, xs_addr, n_star, n_features, String("X_star"))
    var dls = _upload(ctx, _length_scale_table(kernel))
    var dkc = ctx.enqueue_create_buffer[DType.float32](n_train * n_star)
    var dstack = ctx.enqueue_create_buffer[DType.float32](gp_kernel_stack_floats(n_train, n_star))
    ctx.synchronize()
    gp_kernel_matrix(
        ctx, dkc, dx, dxs, dls, dstack, n_train, n_star, n_features, kernel, False, trace,
        "gpc.kcross", GP_ELEM_TPB, GP_SAB_NONE,
    )
    ctx.synchronize()
    _ = dx^
    _ = dxs^
    _ = dls^
    _ = dstack^

    var dyt = ctx.enqueue_create_buffer[DType.float32](n_train)
    var dpit = ctx.enqueue_create_buffer[DType.float32](n_train)
    var dr = ctx.enqueue_create_buffer[DType.float32](n_train)
    var dmean = ctx.enqueue_create_buffer[DType.float32](n_star)
    var dws = ctx.enqueue_create_buffer[DType.float32](
        identical_gemm_workspace_max_floats(n_star, 1, n_train)
    )
    # the variance side, sized 1 when the binary codes alone are asked for
    var cells = n_train * n_star if want_proba else 1
    var dwv = ctx.enqueue_create_buffer[DType.float32](n_train)
    var dl2 = ctx.enqueue_create_buffer[DType.float32](n_train * n_train if want_proba else 1)
    var dv2 = ctx.enqueue_create_buffer[DType.float32](cells)
    var dvar = ctx.enqueue_create_buffer[DType.float32](n_star)
    var dcols = ctx.enqueue_create_buffer[DType.uint64](k * n_star)
    for c in range(k):
        _copy_in(ctx, dyt, y_addrs[c], String("y"))
        _copy_in(ctx, dpit, pi_addrs[c], String("pi"))
        ctx.enqueue_function[gpc_residual_kernel](
            _fp(dyt), _fp(dpit), _fp(dr), Int32(n_train), grid_dim=_blocks(n_train), block_dim=GPC_STEP_TPB
        )
        identical_gemm_into(ctx, dmean, dkc, dr, dws, n_star, 1, n_train, OP_TN)
        if want_proba:
            _copy_in(ctx, dwv, wsr_addrs[c], String("W_sr"))
            _copy_in(ctx, dl2, l_addrs[c], String("L"))
            ctx.enqueue_function[gpc_scale_rows_kernel](
                dv2.unsafe_ptr(), dkc.unsafe_ptr(), dwv.unsafe_ptr(), Int32(n_train), Int32(n_star),
                grid_dim=((n_train * n_star + GPC_VAR_TPB - 1) // GPC_VAR_TPB, 1, 1),
                block_dim=(GPC_VAR_TPB, 1, 1),
            )
            trsm_lower(ctx, dl2, dv2, n_train, n_star, trace, "gpc.v", CHOL_SOLVE_TPB)
            gpc_latent_var_launch(ctx, dvar, dv2, n_train, n_star, kss)
            var col = dcols.create_sub_buffer[DType.uint64](c * n_star, n_star)
            ctx.enqueue_function[gpc_proba_kernel](
                _fp(dmean), _fp(dvar), _up(col), Int32(n_star),
                grid_dim=_blocks(n_star), block_dim=GPC_STEP_TPB,
            )
            # the next class overwrites dmean / dvar / dv2: finish this one
            ctx.synchronize()
            _ = col^

    var width = 2 if binary else k
    var dout = ctx.enqueue_create_buffer[DType.uint64](n_star * width)
    var dcodes = ctx.enqueue_create_buffer[DType.int32](n_star)
    var dcodes64 = ctx.enqueue_create_buffer[DType.int64](n_star)
    if binary:
        if out_kind == GPC_ALL_CODES:
            ctx.enqueue_function[gpc_mean_codes_kernel](
                _fp(dmean), _lp(dcodes64), Int32(n_star), grid_dim=_blocks(n_star), block_dim=GPC_STEP_TPB
            )
        else:
            ctx.enqueue_function[gpc_pairs_kernel](
                _up(dcols), _up(dout), Int32(n_star), grid_dim=_blocks(n_star), block_dim=GPC_STEP_TPB
            )
    else:
        ctx.enqueue_function[gpc_ovr_combine_kernel](
            _up(dcols), _up(dout), _ip(dcodes), Int32(n_star), Int32(k),
            grid_dim=_blocks(n_star), block_dim=GPC_STEP_TPB,
        )
        if out_kind == GPC_ALL_CODES:
            ctx.enqueue_function[gpc_codes_widen_kernel](
                _ip(dcodes), _lp(dcodes64), Int32(n_star), grid_dim=_blocks(n_star), block_dim=GPC_STEP_TPB
            )
    if out_kind == GPC_ALL_CODES:
        ctx.enqueue_copy(dst_ptr=_GL(unsafe_from_address=out_addr), src_buf=dcodes64)
    else:
        ctx.enqueue_copy(dst_ptr=_GU(unsafe_from_address=out_addr), src_buf=dout)
    ctx.synchronize()
    _ = dkc^
    _ = dyt^
    _ = dpit^
    _ = dr^
    _ = dmean^
    _ = dws^
    _ = dwv^
    _ = dl2^
    _ = dv2^
    _ = dvar^
    _ = dcols^
    _ = dout^
    _ = dcodes^
    _ = dcodes64^
    # DEVIATION 1946: the context dies LAST.
    _ = ctx^
