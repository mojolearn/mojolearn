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
uploads, over the same bytes; the query is checked by
`check_finite_device` (the same first flat index and message as the host
`check_finite_list`). The bits are the same.
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
from svm.checks.device_select import check_finite_device
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
) raises:
    """The resident decision of one machine, written as n_rows float32 at
    `out_addr`. `model` carries the scalars only (n_support, n_cols, b, the
    label pair)."""
    var kp = _kernel_params(kernel, gamma, degree, coef0)
    var ctx = _family_ctx()
    var card = IdentityTrace()
    card.header(header)
    var plan = device_plan_current("MOJOLEARN_SVM_DEVICE_COUNT")
    var n = n_rows * n_cols
    var x = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_buf=x, src_ptr=MutPointer[Float32, MutAnyOrigin](unsafe_from_address=x_addr))
    # DEVIATION 636, as `check_finite_list` raised it, before any model copy.
    check_finite_device(ctx, x, n, "predict X")
    var n_batch = svc_predict_batch_rows(n_rows, n_support, buffer_size_mib)
    var held = fitted_resident_view(svm_resident_store(), ctx, lease, slot,
                                    [sv_addr, dual_addr], [n_support * n_cols, n_support])
    var sv = held.bufs[0].copy()
    var dual = held.bufs[1].copy()
    var preds = svc_predict_dev(
        ctx, model, x, sv, dual, n_rows, n_cols, n_batch, kp, predict_class, card, plan,
    )
    memcpy(
        dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=out_addr),
        src=preds.unsafe_ptr(),
        count=n_rows,
    )
    _ = preds^
    _ = x^
    _ = sv^
    _ = dual^
    _ = held^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^


def _svm_predict_checks(who: String, n_rows: Int, n_cols: Int, n_support: Int, kernel: Int,
                        buffer_size_mib: Float64, x_addr: Int, out_addr: Int) raises:
    """The `svc_predict_host` / `svr_predict_host` refusals that do not read
    the arrays, in their order, plus the null addresses."""
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
    if x_addr == 0 or out_addr == 0:
        raise Error(who + ": a query or output address is null")


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
) raises:
    """`svc_predict_host` over the caller's memory, the machine's support
    rows and duals read from slot `slot` of `lease`; writes n_rows float32
    at `out_addr`."""
    if n_support == 0 or kernel == KERNEL_PRECOMPUTED:
        # The host-list route, unchanged (see the module note).
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
        return
    _svm_predict_checks(String("svc_predict_host"), n_rows, n_cols, n_support, kernel, buffer_size_mib,
                        x_addr, out_addr)
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
    _svm_resident_decide(
        lease, slot, x_addr, n_rows, n_cols, sv_addr, dual_addr, n_support, model,
        kernel, gamma, predict_class, buffer_size_mib, degree, coef0, out_addr,
        "svc_predict_host: n_rows=" + String(n_rows) + " n_cols=" + String(n_cols)
        + " n_support=" + String(n_support) + " kernel=" + String(kernel)
        + " gamma=" + String(gamma)
        + " predict_class=" + String(predict_class)
        + " buffer_mib=" + String(buffer_size_mib),
    )
    _ = model^


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
) raises:
    """`svr_predict_host` over the caller's memory, the support rows and
    duals read from slot `slot` of `lease`; writes n_rows float32 at
    `out_addr`."""
    if n_support == 0 or kernel == KERNEL_PRECOMPUTED:
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
        return
    _svm_predict_checks(String("svr_predict_host"), n_rows, n_cols, n_support, kernel, buffer_size_mib,
                        x_addr, out_addr)
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
    _svm_resident_decide(
        lease, slot, x_addr, n_rows, n_cols, sv_addr, dual_addr, n_support, model,
        kernel, gamma, False, buffer_size_mib, degree, coef0, out_addr,
        "svr_predict_host: n_rows=" + String(n_rows) + " n_cols=" + String(n_cols)
        + " n_support=" + String(n_support) + " kernel=" + String(kernel)
        + " gamma=" + String(gamma)
        + " buffer_mib=" + String(buffer_size_mib),
    )
    _ = model^
