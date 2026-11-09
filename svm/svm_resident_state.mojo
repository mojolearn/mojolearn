# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Device-resident fitted state of the kernel SVMs (lane ne1-a-gp-svm,
workstream A; row N376).

`svc_predict_binding` and `svr_predict_binding` copied the query, the dual
coefficients and the whole support matrix into host lists under the GIL on
every call, `svc_predict` uploaded all three again, and the result came back
as a list the binding copied into the output. Here, for a lease the public
door holds (bindings/svc_kernel_native.mojo through
bindings/fitted_resident_door.mojo), each fitted machine's support rows and
dual coefficients are uploaded once into the SVM binding's store
(`core/fitted_resident_store.mojo`), slot = the machine's index (the
one-vs-one pairs of a multiclass SVC are slots 0 .. P-1); the query crosses
from the caller's memory straight to the device and the decisions are
copied straight into the caller's output. Lease 0 uploads the model for the
call only, from the caller's memory (no host list either way).

Not resident: a model with no support vectors (its decision is the bias; no
upload exists) and kernel='precomputed' (its query is the cross-kernel the
caller gathered, flushed on the host as before): both keep the host-list
route `svc_predict_host`.

WHAT DOES NOT CHANGE. The decisions are `svc_predict_dev`
(svm/impl/svc_impl.mojo), the launches `svc_predict` runs after its own
uploads, over the same bytes. The bits are the same.

THE QUERY (lane a7a-device-x-svm-gp-krr, 2026-11-09). The query is a
`DeviceQueryX` (core/device_query_x.mojo): host words copied once, or a
cross-validation fold's rows gathered on the device out of model
selection's resident base (no host gather, no upload), with the finite and
range words read back with the decisions. The entries return the two words
and the door refuses in scikit-learn's wording (bindings/svc_kernel_native.mojo
`_svm_query_refuse`); the synchronising `check_finite_device` scan and its
DEVIATION 636 text are gone from this route. A fold runs on the fold
store's context with lease 0 (its fresh fitted model is uploaded for the
call, as a lease-0 predict always was).
"""
from std.ffi import _Global
from std.memory import memcpy
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.device_plan import device_plan_current
from core.identity_trace import IdentityTrace
from core.hostptr import read_f32
from core.fitted_resident_store import (
    FittedResidentStore,
    fitted_lease_open,
    fitted_lease_close,
    fitted_resident_view,
    fitted_resident_stats,
)
# TOMBSTONE: check_finite_device on the resident query (the synchronising first-non-finite scan, DEVIATION 636 text) removed 2026-11-09 by lane a7a-device-x-svm-gp-krr: the query's finite word rides back with the decisions (core/device_query_x.mojo). Never reached main: recoverable at 5b5490960d.
from core.device_query_x import DeviceQueryX, QueryFold, query_context
from svm.estimator import _family_ctx, _kernel_params, svc_predict_host, svr_predict_host
from svm.impl.svc_impl import svc_predict_batch_rows, svc_predict_dev
from svm.impl.svm_parameter import KERNEL_PRECOMPUTED, SvmModel


# One store per numeric tier, next to that tier's SVM context.
comptime SVM_RESIDENT_STORE = _Global[
    StorageType=FittedResidentStore,
    name=(
        "MojoSvmResidentFittedIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
        else "MojoSvmResidentFittedFast"
    ),
    init_fn=FittedResidentStore.__init__,
]


def svm_resident_store() raises -> UnsafePointer[FittedResidentStore, MutAnyOrigin]:
    return SVM_RESIDENT_STORE.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()


struct SvmResidentLease(Movable, Writable):
    """The native owner of one fitted state's lease: dropping it frees the
    device copies of every machine (a refit, a replaced fitted array, the
    model's death)."""

    var lease: Int

    def __init__(out self, lease: Int):
        self.lease = lease

    def write_to(self, mut writer: Some[Writer]):
        writer.write("SvmResidentLease(open=", self.lease != 0, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def __deinit__(deinit self):
        if self.lease != 0:
            try:
                fitted_lease_close(svm_resident_store(), self.lease)
            except:
                pass


def svm_resident_lease_open() raises -> Int:
    return fitted_lease_open(svm_resident_store())


def svm_resident_stats_words() raises -> List[Int]:
    return fitted_resident_stats(svm_resident_store())


def _svm_resident_decide(
    lease: Int,
    slot: Int,
    x_addr: Int,
    n_rows: Int,
    n_cols: Int,
    sv_addr: Int,
    dual_addr: Int,
    n_support: Int,
    model: SvmModel,
    kernel: Int,
    gamma: Float64,
    predict_class: Bool,
    buffer_size_mib: Float64,
    degree: Int,
    coef0: Float64,
    out_addr: Int,
    header: String,
    fold: QueryFold,
) raises -> Tuple[Int32, Int32]:
    """The resident decision of one machine, written as n_rows float32 at
    `out_addr`. `model` carries the scalars only (n_support, n_cols, b, the
    label pair). The query is host words at `x_addr`, or the fold `fold`
    gathered on the device. Returns (finite word, range word): both 0 for a
    finite, in-range query; otherwise the door refuses and discards
    `out_addr`'s rows."""
    var kp = _kernel_params(kernel, gamma, degree, coef0)
    var ctx = query_context(fold, _family_ctx())
    var card = IdentityTrace()
    card.header(header)
    var plan = device_plan_current("MOJOLEARN_SVM_DEVICE_COUNT")
    var qx = DeviceQueryX(ctx, x_addr, n_rows, n_cols, fold)
    var n_batch = svc_predict_batch_rows(n_rows, n_support, buffer_size_mib)
    # A fold runs on the fold store's context: its fresh model is uploaded
    # for this call (lease 0), never kept under the family context's lease.
    var held = fitted_resident_view(svm_resident_store(), ctx, 0 if fold.active() else lease, slot,
                                    [sv_addr, dual_addr], [n_support * n_cols, n_support])
    var sv = held.bufs[0].copy()
    var dual = held.bufs[1].copy()
    var preds = List[Float32]()
    try:
        # svc_predict_dev synchronises before it returns its host list, so
        # the status words copied after the query are valid below.
        preds = svc_predict_dev(
            ctx, model, qx.x, sv, dual, n_rows, n_cols, n_batch, kp, predict_class, card, plan,
        )
    except error:
        # A non-finite query makes a NaN decision (DEVIATION 637's refusal):
        # the input is at fault, so the door's refusal wins.
        if qx.flagged(ctx):
            return (qx.finite_word(), qx.range_word())
        raise error^
    var words = (qx.finite_word(), qx.range_word())
    if words[0] == 0 and words[1] == 0:
        memcpy(
            dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=out_addr),
            src=preds.unsafe_ptr(),
            count=n_rows,
        )
    _ = preds^
    _ = qx^
    _ = sv^
    _ = dual^
    _ = held^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return words


def _svm_predict_checks(who: String, n_rows: Int, n_cols: Int, n_support: Int, kernel: Int,
                        buffer_size_mib: Float64, x_addr: Int, out_addr: Int, fold: QueryFold) raises:
    """The `svc_predict_host` / `svr_predict_host` refusals that do not read
    the arrays, in their order, plus the null addresses (a fold has no
    query address)."""
    if n_rows <= 0:
        raise Error(who + ": n_rows must be at least one")
    if n_cols <= 0:
        raise Error(who + ": n_cols must be at least one")
    if n_support < 0:
        raise Error(who + ": n_support cannot be negative")
    if not (buffer_size_mib > 0.0):
        raise Error(
            who + ": the predict buffer (cache_size) must be a"
            " positive number of MiB, got " + String(buffer_size_mib)
        )
    if (x_addr == 0 and not fold.active()) or out_addr == 0:
        raise Error(who + ": a query or output address is null")


def _svm_host_route_fold(who: String, fold: QueryFold) raises:
    """The host-list route (no support vectors, or kernel='precomputed')
    reads host words: the door never hands it a fold."""
    if fold.active():
        raise Error(who + ": a device fold needs support rows and a computed kernel")


def svc_predict_resident_host(
    lease: Int,
    slot: Int,
    x_addr: Int,
    n_rows: Int,
    n_cols: Int,
    sv_addr: Int,
    dual_addr: Int,
    n_support: Int,
    b: Float32,
    label0: Float32,
    label1: Float32,
    kernel: Int,
    gamma: Float64,
    predict_class: Bool,
    buffer_size_mib: Float64,
    degree: Int,
    coef0: Float64,
    out_addr: Int,
    fold: QueryFold = QueryFold.none(),
) raises -> Tuple[Int32, Int32]:
    """`svc_predict_host` over the caller's memory (or a device fold), the
    machine's support rows and duals read from slot `slot` of `lease`;
    writes n_rows float32 at `out_addr`. Returns the query's (finite,
    range) words."""
    if n_support == 0 or kernel == KERNEL_PRECOMPUTED:
        # The host-list route, unchanged (see the module note).
        _svm_host_route_fold(String("svc_predict_host"), fold)
        var xl = read_f32(x_addr, n_rows * n_cols if n_rows > 0 and n_cols > 0 else 0)
        var dl = List[Float32]()
        if n_support > 0:
            dl = read_f32(dual_addr, n_support)
        var decided = svc_predict_host(
            xl, n_rows, n_cols, List[Float32](), dl, n_support, b, label0, label1,
            kernel, gamma, predict_class, buffer_size_mib, degree, coef0,
        )
        memcpy(
            dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=out_addr),
            src=decided.unsafe_ptr(),
            count=len(decided),
        )
        return (Int32(0), Int32(0))
    _svm_predict_checks(String("svc_predict_host"), n_rows, n_cols, n_support, kernel, buffer_size_mib,
                        x_addr, out_addr, fold)
    if sv_addr == 0 or dual_addr == 0:
        raise Error("svc_predict_host: a support or dual address is null")
    var model = SvmModel()
    model.n_support = n_support
    model.n_cols = n_cols
    model.b = b
    model.support_idx = List[Int32]()
    model.n_classes = 2
    model.unique_labels = List[Float32]()
    model.unique_labels.append(label0)
    model.unique_labels.append(label1)
    model.n_iter = 0
    var words = _svm_resident_decide(
        lease, slot, x_addr, n_rows, n_cols, sv_addr, dual_addr, n_support, model,
        kernel, gamma, predict_class, buffer_size_mib, degree, coef0, out_addr,
        "svc_predict_host: n_rows=" + String(n_rows) + " n_cols=" + String(n_cols)
        + " n_support=" + String(n_support) + " kernel=" + String(kernel)
        + " gamma=" + String(gamma)
        + " predict_class=" + String(predict_class)
        + " buffer_mib=" + String(buffer_size_mib),
        fold,
    )
    _ = model^
    return words


def svr_predict_resident_host(
    lease: Int,
    slot: Int,
    x_addr: Int,
    n_rows: Int,
    n_cols: Int,
    sv_addr: Int,
    dual_addr: Int,
    n_support: Int,
    b: Float32,
    kernel: Int,
    gamma: Float64,
    buffer_size_mib: Float64,
    degree: Int,
    coef0: Float64,
    out_addr: Int,
    fold: QueryFold = QueryFold.none(),
) raises -> Tuple[Int32, Int32]:
    """`svr_predict_host` over the caller's memory (or a device fold), the
    support rows and duals read from slot `slot` of `lease`; writes n_rows
    float32 at `out_addr`. Returns the query's (finite, range) words."""
    if n_support == 0 or kernel == KERNEL_PRECOMPUTED:
        _svm_host_route_fold(String("svr_predict_host"), fold)
        var xl = read_f32(x_addr, n_rows * n_cols if n_rows > 0 and n_cols > 0 else 0)
        var dl = List[Float32]()
        if n_support > 0:
            dl = read_f32(dual_addr, n_support)
        var decided = svr_predict_host(
            xl, n_rows, n_cols, List[Float32](), dl, n_support, b, kernel, gamma,
            buffer_size_mib, degree, coef0,
        )
        memcpy(
            dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=out_addr),
            src=decided.unsafe_ptr(),
            count=len(decided),
        )
        return (Int32(0), Int32(0))
    _svm_predict_checks(String("svr_predict_host"), n_rows, n_cols, n_support, kernel, buffer_size_mib,
                        x_addr, out_addr, fold)
    if sv_addr == 0 or dual_addr == 0:
        raise Error("svr_predict_host: a support or dual address is null")
    var model = SvmModel()
    model.n_support = n_support
    model.n_cols = n_cols
    model.b = b
    model.support_idx = List[Int32]()
    model.n_classes = 0
    var dummy_labels: List[Float32] = [Float32(-1.0), Float32(1.0)]
    model.unique_labels = dummy_labels^
    model.n_iter = 0
    var words = _svm_resident_decide(
        lease, slot, x_addr, n_rows, n_cols, sv_addr, dual_addr, n_support, model,
        kernel, gamma, False, buffer_size_mib, degree, coef0, out_addr,
        "svr_predict_host: n_rows=" + String(n_rows) + " n_cols=" + String(n_cols)
        + " n_support=" + String(n_support) + " kernel=" + String(kernel)
        + " gamma=" + String(gamma)
        + " buffer_mib=" + String(buffer_size_mib),
        fold,
    )
    _ = model^
    return words
