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
  - the Newton loop is one bounded device loop (gaussian_process/
    gpc_newton.mojo, lane n8-gp-krr-smo): the factor's info, log|B| and the
    stop test stay on the device, read once per GPC_NEWTON_CHECK iterations;
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
"""

from std.atomic import Atomic
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE as _CTX_MODE, NUMERIC_IDENTICAL as _CTX_IDENTICAL
from checks.soft_f64 import SF64_ONE, sf64_sub
from cholesky.checks.trsm import CHOL_SOLVE_TPB, trsm_lower
from core.device_scan import device_classify_nonfinite, device_first_nonfinite
from core.device_zero import enqueue_fill
from core.device_query_x import DeviceQueryX, QueryFold, query_context
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
    GPC_STEP_TPB,
    _family_ctx,
    gpc_ovr_combine_kernel,
    gpc_proba_kernel,
    gpc_residual_kernel,
)
from gaussian_process.estimator import _length_scale_table, _upload
from gaussian_process.gpc_common import gpc_validate_max_iter
from gaussian_process.gpc_device_var import (
    GPC_VAR_TPB,
    gpc_latent_var_launch,
    gpc_scale_rows_kernel,
)
from gaussian_process.gpc_newton import gpc_newton_device_batch
from gaussian_process.gp_resident_store import gp_resident_store
from core.fitted_resident_store import fitted_resident_view
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

# TOMBSTONE: MOJOLEARN_IDN_GPC_NEWTON_LOGDET_OFF (superseded by the device Newton loop) deleted 2026-11-09 by lane/n8-gp-krr-smo; code recoverable at f59d913b0b.
# Restore: git apply experiments/removed/N8_GPC_NEWTON_ONE_LOOP.patch

#: `gpc_predict_all_device` output kinds (the binding's `out_kind`)
#: The device bytes of B factors one batched one-vs-rest Newton loop may
#: hold (each class keeps its n x n float32 factor): 1 GiB, a memory bound.
comptime GPC_OVR_BATCH_BYTES = 1 << 30

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


def gpc_one_class_kernel(seen: _GI, g: Int32, n: Int32, bad: _GI):
    """One thread: the lowest class c of the group whose targets hold one
    value only (`seen[2c]` or `seen[2c + 1]` still n), else -1."""
    if _tid() != 0:
        return
    for c in range(Int(g)):
        if seen.unsafe_load(2 * c) >= n or seen.unsafe_load(2 * c + 1) >= n:
            bad.unsafe_store(0, Int32(c))
            return
    bad.unsafe_store(0, Int32(-1))


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
    _gpc_check_finite(ctx, buf, rows, cols, what)
    return buf^


def _gpc_check_finite(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], rows: Int, cols: Int, what: String
) raises:
    """`gp_validate_data`'s refusals (DEVIATION 1768) over `rows x cols`
    uploaded float32: the row and the feature of the first NaN or infinity."""
    var n = rows * cols
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


def _gpc_addr_check(addr: Int, what: String) raises:
    """`_copy_in`'s null-address refusal, for the resident reads."""
    if addr == 0:
        raise Error("gpc: " + what + " has a null address")


def _copy_in(ctx: DeviceContext, mut dst: DeviceBuffer[DType.float32], addr: Int, what: String) raises:
    """One host-to-device copy from the caller's memory, enqueued (the
    caller's array outlives the binding call)."""
    if addr == 0:
        raise Error("gpc: " + what + " has a null address")
    ctx.enqueue_copy(dst_buf=dst, src_ptr=_GP(unsafe_from_address=addr))


# ===========================================================================
# FIT
# ===========================================================================


# TOMBSTONE: MOJOLEARN_IDN_GPC_NEWTON_LOGDET_OFF (superseded: the Newton loop is one device loop, log|B| always stays on the device) deleted 2026-11-09 by lane/n8-gp-krr-smo; code recoverable at f59d913b0b.
# Restore: git apply experiments/removed/N8_GPC_NEWTON_ONE_LOOP.patch
# The Newton loop (`gpc_newton_device`, `GPCNewtonRun`) moved to
# gaussian_process/gpc_newton.mojo, where classifier.mojo's copy of it now
# lands too (lane n8-gp-krr-smo).


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
    if n_train <= 0 or n_features <= 0:
        raise Error("gpc_fit_all: X must have at least one row and one feature, got " + String(n_train) + " x "
                    + String(n_features))
    var n = n_train
    var ctx = _family_ctx()
    # TOMBSTONE: the synchronising first-non-finite scan of the fit X (_gpc_upload_checked; DEVIATION 1768 text) removed 2026-11-09 by lane a8a-device-x-fit-svm-gp-krr: the public door admits X once and refuses a non-finite X in scikit-learn's wording before this entry; recoverable at 0ce88411d3 (docs/TOMBSTONES.md).
    var dx = ctx.enqueue_create_buffer[DType.float32](n_train * n_features)
    _copy_in(ctx, dx, x_addr, String("X"))
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
    var dgw = ctx.enqueue_create_buffer[DType.float32](identical_gemm_workspace_max_floats(n, 1, n))
    for j in range(n_fits):  # driver-loop: over the one-vs-rest classes (address checks)
        if y_addrs[j] == 0 or l_addrs[j] == 0 or pi_addrs[j] == 0 or wsr_addrs[j] == 0 or scalar_addrs[j] == 0:
            raise Error("gpc_fit_all: binary fit " + String(j) + " has a null output address")
    # Lane m3-linalg-and-housekeeping (2026-11-09): the classes' Newton
    # loops run as ONE batched loop (`gpc_newton_device_batch`: one register
    # read per GPC_NEWTON_CHECK iterations for the whole group, the targets
    # and their class check in one read), where each class ran its own loop
    # and reads after the previous one. Each class keeps its own B factor
    # (n^2 floats), so a group holds as many classes as GPC_OVR_BATCH_BYTES
    # of factors allows (a memory bound, the same rule at every n and k).
    var group = max(1, GPC_OVR_BATCH_BYTES // max(1, n * n * 4))
    var j0 = 0
    while j0 < n_fits:  # driver-loop: over groups of one-vs-rest classes, sized by device memory
        var g = min(group, n_fits - j0)
        var dys = List[DeviceBuffer[DType.float32]](capacity=g)
        var dbs = List[DeviceBuffer[DType.float32]](capacity=g)
        var dpis = List[DeviceBuffer[DType.float32]](capacity=g)
        var dwsrs = List[DeviceBuffer[DType.float32]](capacity=g)
        var dseen = ctx.enqueue_create_buffer[DType.int32](2 * g)
        var dbad = ctx.enqueue_create_buffer[DType.int32](1)
        var hbad = ctx.enqueue_create_host_buffer[DType.int32](1)
        enqueue_fill(ctx, dseen, Int32(n))
        for c in range(g):  # driver-loop: over the group's classes (model structure)
            var dy = ctx.enqueue_create_buffer[DType.float32](n)
            ctx.enqueue_function[gpc_targets_kernel](
                _ip(dcodes), _fp(dy), _ip(dseen) + 2 * c, Int32(class_ks[j0 + c]), Int32(n),
                grid_dim=_blocks(n), block_dim=GPC_STEP_TPB,
            )
            ctx.enqueue_copy(dst_ptr=_GP(unsafe_from_address=y_addrs[j0 + c]), src_buf=dy)
            dys.append(dy^)
            dbs.append(ctx.enqueue_create_buffer[DType.float32](n * n))
            dpis.append(ctx.enqueue_create_buffer[DType.float32](n))
            dwsrs.append(ctx.enqueue_create_buffer[DType.float32](n))
        ctx.enqueue_function[gpc_one_class_kernel](  # small-launch(g: the group's classes): one thread finds the first one-class column
            _ip(dseen), Int32(g), Int32(n), _ip(dbad), grid_dim=1, block_dim=1,
        )
        ctx.enqueue_copy(dst_ptr=hbad.unsafe_ptr(), src_buf=dbad)
        ctx.synchronize()
        var bad = Int(hbad.unsafe_ptr().unsafe_load(0))
        if bad >= 0:
            raise Error(
                "gpc_fit_host: a binary Laplace fit requires 2 classes; got 1"
                " (one-vs-rest class "
                + String(class_ks[j0 + bad])
                + ")"
            )
        var runs = gpc_newton_device_batch(ctx, dk, dys, dbs, dgw, dpis, dwsrs, n, max_iter_predict)
        for c in range(g):  # driver-loop: over the group's classes (model structure)
            var j = j0 + c
            ctx.enqueue_copy(dst_ptr=_GP(unsafe_from_address=l_addrs[j]), src_buf=dbs[c])
            ctx.enqueue_copy(dst_ptr=_GP(unsafe_from_address=pi_addrs[j]), src_buf=dpis[c])
            ctx.enqueue_copy(dst_ptr=_GP(unsafe_from_address=wsr_addrs[j]), src_buf=dwsrs[c])
        ctx.synchronize()
        for c in range(g):  # driver-loop: over the group's classes (model structure)
            # lml, n_iter, nb, in that order; each widens to float64 exactly.
            var sp = MutPointer[Float64, MutAnyOrigin](unsafe_from_address=scalar_addrs[j0 + c])
            sp.unsafe_store(0, Float64(runs[c].lml))
            sp.unsafe_store(1, Float64(runs[c].n_iter))
            sp.unsafe_store(2, Float64(runs[c].nb))
        _ = dys^
        _ = dbs^
        _ = dpis^
        _ = dwsrs^
        _ = dseen^
        _ = dbad^
        _ = hbad^
        j0 += g
    ctx.synchronize()
    _ = dk^
    _ = dcodes^
    _ = dgw^
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
    lease: Int = -1,
    fold: QueryFold = QueryFold.none(),
) raises -> Tuple[Int32, Int32]:
    """`predict` (`out_kind` 1: n_star int64 class codes at `out_addr`) or
    `predict_proba` (`out_kind` 2: n_star x max(k, 2) float64, row-major)
    of a fitted classifier with k = len(y_addrs) binary fits, against ONE
    upload of X_train and X_star and ONE cross kernel.

    One binary fit: the codes are `mean > 0` (no variance is computed) and
    the probabilities `[1 - p, p]`. Past one: DEVIATION 2833's combine of
    the k class-1 probability columns, which never leave the device.

    `lease` (lane ne1-a-gp-svm): -1 copies every fitted array in per call,
    as before; otherwise X_train and each binary fit's arrays are read from
    the GP store's copies for that lease (gaussian_process/gp_resident_store.mojo;
    0 uploads them for this call only): slot 0 X_train, slot 1 + 2c the
    fit's y and pi, slot 2 + 2c its W_sr and L (only when the variance is
    needed). The same bytes reach the same launches: the bits are the same.

    The query (lane a7a-device-x-svm-gp-krr) is a `DeviceQueryX`: host
    words at `xs_addr`, or the cross-validation fold `fold` gathered on the
    device (then the entry runs on the fold store's context and a resident
    lease reads as 0). Returns the query's (finite word, range word); the
    door refuses a flagged query and discards `out_addr`'s words. The
    synchronising `_gpc_upload_checked` scan of X_star (DEVIATION 1768
    text) is gone from this entry."""
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
    var ctx = query_context(fold, _family_ctx())
    var resident = lease >= 0
    # A fold runs on the fold store's context: its fresh model is uploaded
    # for this call (lease 0), never kept under the family context's lease.
    var use_lease = 0 if (resident and fold.active()) else lease
    var store = gp_resident_store()
    var dx: DeviceBuffer[DType.float32]
    if resident:
        if n_train <= 0:
            raise Error("gpr: X_train must have at least one row, got " + String(n_train))
        if n_features <= 0:
            raise Error("gpr: X_train must have at least one feature, got " + String(n_features))
        if xt_addr == 0:
            raise Error("gpc: X_train has a null address")
        var held = fitted_resident_view(store, ctx, use_lease, 0, [xt_addr], [n_train * n_features])
        dx = held.bufs[0].copy()
        if held.fresh:
            # The refusal `_gpc_upload_checked` makes, once per upload.
            _gpc_check_finite(ctx, dx, n_train, n_features, String("X_train"))
        _ = held^
    else:
        dx = _gpc_upload_checked(ctx, xt_addr, n_train, n_features, String("X_train"))
    # TOMBSTONE: _gpc_upload_checked on X_star (synchronising scan, DEVIATION 1768 text) removed 2026-11-09 by lane a7a-device-x-svm-gp-krr; recoverable at 5b5490960d (docs/TOMBSTONES.md).
    var qx = DeviceQueryX(ctx, xs_addr, n_star, n_features, fold)
    var dls = _upload(ctx, _length_scale_table(kernel))
    var dkc = ctx.enqueue_create_buffer[DType.float32](n_train * n_star)
    var dstack = ctx.enqueue_create_buffer[DType.float32](gp_kernel_stack_floats(n_train, n_star))
    ctx.synchronize()
    gp_kernel_matrix(
        ctx, dkc, dx, qx.x, dls, dstack, n_train, n_star, n_features, kernel, False, trace,
        "gpc.kcross", GP_ELEM_TPB, GP_SAB_NONE,
    )
    ctx.synchronize()
    # The query's status words were copied before the synchronisation above.
    var words = (qx.finite_word(), qx.range_word())
    _ = dx^
    _ = qx^
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
    # The resident reads replace these per class (lease >= 0): no n x n
    # staging buffer is made for them.
    var dl2 = ctx.enqueue_create_buffer[DType.float32](n_train * n_train if want_proba and not resident else 1)
    var dv2 = ctx.enqueue_create_buffer[DType.float32](cells)
    var dvar = ctx.enqueue_create_buffer[DType.float32](n_star)
    var dcols = ctx.enqueue_create_buffer[DType.uint64](k * n_star)
    for c in range(k):
        if resident:
            _gpc_addr_check(y_addrs[c], String("y"))
            _gpc_addr_check(pi_addrs[c], String("pi"))
            var fit = fitted_resident_view(store, ctx, use_lease, 1 + 2 * c, [y_addrs[c], pi_addrs[c]],
                                           [n_train, n_train])
            dyt = fit.bufs[0].copy()
            dpit = fit.bufs[1].copy()
            _ = fit^
        else:
            _copy_in(ctx, dyt, y_addrs[c], String("y"))
            _copy_in(ctx, dpit, pi_addrs[c], String("pi"))
        ctx.enqueue_function[gpc_residual_kernel](
            _fp(dyt), _fp(dpit), _fp(dr), Int32(n_train), grid_dim=_blocks(n_train), block_dim=GPC_STEP_TPB
        )
        identical_gemm_into(ctx, dmean, dkc, dr, dws, n_star, 1, n_train, OP_TN)
        if want_proba:
            if resident:
                _gpc_addr_check(wsr_addrs[c], String("W_sr"))
                _gpc_addr_check(l_addrs[c], String("L"))
                var var_fit = fitted_resident_view(store, ctx, use_lease, 2 + 2 * c, [wsr_addrs[c], l_addrs[c]],
                                                   [n_train, n_train * n_train])
                dwv = var_fit.bufs[0].copy()
                dl2 = var_fit.bufs[1].copy()
                _ = var_fit^
            else:
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
    return words
