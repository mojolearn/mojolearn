# SPDX-License-Identifier: Apache-2.0
"""The SVC device table: the device-X solve and the resident query as typed
entries over the existing owners (`svc_fit_host_borrowed`,
`svc_predict_resident_host`). The bodies are those of the by-name bindings
`svc_fit` (its 15-value form) and `svc_predict_resident`
(bindings/_mojolearn_svm.mojo), with the Python argument unpacking removed:
same checks, same owner calls, same info and status words."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import f32_ptr, f64_ptr, i32_ptr, read_f32
from core.device_query_x import QueryFold
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.svc_native_api import (
    SVC_DEVICE_ABI, SVC_DEVICE_FIT, SVC_DEVICE_RESIDENT, SVC_DEVICE_FIT_ADDRESSES, SVC_DEVICE_FIT_INTS,
    SVC_DEVICE_FIT_FLOATS, SVC_DEVICE_RESIDENT_ADDRESSES, SVC_DEVICE_RESIDENT_INTS, SVC_DEVICE_RESIDENT_FLOATS,
)
from svm.estimator import svc_fit_host_borrowed
from svm.svm_resident_state import svc_predict_resident_host
from svm.svm_binding_run import svm_c_rows_run, svc_fit_store_run


def _svc_device_fit(a: Pointer[Int, ImmUntrackedOrigin], p: Pointer[Int, ImmUntrackedOrigin],
                    f: Pointer[Float64, ImmUntrackedOrigin]) raises -> Int:
    # a: x, labels, dual, support, vectors, info (8 float64), bounds (0: unweighted)
    # p: n_rows, n_features, kernel, max_iter, nochange_steps, degree, fold base id, fold base rows,
    #    fold index id, scale_gamma;  f: gamma, C, tol, coef0
    var fold = QueryFold(p[6], p[7], p[8])
    var scale_gamma = p[9] != 0
    var x_word = 0 if fold.active() else a[0]
    var yp = f32_ptr(a[1])
    var dp = f32_ptr(a[2])
    var sip = i32_ptr(a[3])
    var smp = f32_ptr(a[4])
    var ip = f64_ptr(a[5])
    var n_rows = p[0]
    var n_cols = p[1]
    if n_rows <= 0 or n_cols <= 0:
        raise Error("svc_fit: n_rows and n_features must both be positive")
    var y = read_f32(Int(yp), max(0, n_rows))
    var c_rows = svm_c_rows_run(a[6], n_rows)
    # DEVIATION 2665: X is not copied into a host List; the fit stages the
    # caller's buffer (or gathers the fold) on the device once.
    var res = svc_fit_host_borrowed(x_word, fold, y, n_rows, n_cols, p[2], f[0], f[1], f[2], p[3],
                                    p[4], p[5], f[3], c_rows, scale_gamma)
    return svc_fit_store_run(res, dp, sip, smp, ip, n_cols, True)


def _svc_device_resident(a: Pointer[Int, ImmUntrackedOrigin], p: Pointer[Int, ImmUntrackedOrigin],
                         f: Pointer[Float64, ImmUntrackedOrigin]) raises -> Int:
    # a: x (0 with a fold), dual, vectors, dst, status (two int32: finite word, range word)
    # p: n_rows, n_features, n_support, kernel, predict_class, degree, lease, slot, fold base id,
    #    fold base rows, fold index id;  f: b, label0, label1, gamma, cache_size_mib, coef0
    var n_rows = p[0]
    var n_cols = p[1]
    var n_support = p[2]
    var kernel = p[3]
    if n_rows <= 0 or n_cols <= 0:
        raise Error("svc_predict: n_rows and n_features must both be positive")
    if n_support < 0:
        raise Error("svc_predict: n_support cannot be negative")
    var dual = a[1] if n_support > 0 else 0
    var support = a[2] if n_support > 0 and kernel != 4 else 0
    var words = svc_predict_resident_host(p[6], p[7], a[0], n_rows, n_cols, support, dual, n_support,
        Float32(f[0]), Float32(f[1]), Float32(f[2]), kernel, f[3], p[4] != 0, f[4], p[5], f[5], a[3],
        QueryFold(p[8], p[9], p[10]))
    if a[4] != 0:
        var status = i32_ptr(a[4])
        status[0] = words[0]
        status[1] = words[1]
    return n_rows


def svc_device_native_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if not result or (operation != SVC_DEVICE_FIT and operation != SVC_DEVICE_RESIDENT):
            raise Error("SVC device ABI: invalid operation/result")
        var addresses = SVC_DEVICE_FIT_ADDRESSES if operation == SVC_DEVICE_FIT else SVC_DEVICE_RESIDENT_ADDRESSES
        var integers = SVC_DEVICE_FIT_INTS if operation == SVC_DEVICE_FIT else SVC_DEVICE_RESIDENT_INTS
        var floats = SVC_DEVICE_FIT_FLOATS if operation == SVC_DEVICE_FIT else SVC_DEVICE_RESIDENT_FLOATS
        if na != addresses or np != integers or nf != floats:
            raise Error("SVC device ABI: invalid request spans")
        if not aa or not pp or not ff:
            raise Error("SVC device ABI: null request spans")
        var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
        var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
        var f = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
        var value = 0
        if operation == SVC_DEVICE_FIT:
            value = _svc_device_fit(a, p, f)
        else:
            value = _svc_device_resident(a, p, f)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct SvcDeviceAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(SVC_DEVICE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(svc_device_native_v1)


comptime _SLOT = "MojolearnSvcDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnSvcDeviceFastV1"
comptime SVC_DEVICE_API = SharedSlot[SvcDeviceAPI, _SLOT]
