# SPDX-License-Identifier: Apache-2.0
# CPU SVC/SVR through the same common provider ABI as the GPU family.
# Numerical calls and checks are retained from _mojolearn_svm_host.
from std.ffi import _Global
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.hostptr import f32_ptr, f64_ptr, i32_ptr, read_f32
from checks.numerics import GLOBAL_NUMERIC_MODE
from svm.operation_metadata import SVC_ABI_NAME
from svm.native_descriptor_schema import svc_binary_spans
from svm.host.native_epilogue_dispatch import svc_host_epilogue, svc_host_platt, svc_host_splitmix
from svm.host.native_auxiliary_dispatch import svc_host_auxiliary
from svm.host.svm_host_impl import svc_fit_host, svc_predict_host, svr_fit_host, svr_predict_host


def svc_host_operation(op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if op >= 8:
        return svc_host_auxiliary(op, a, p)
    if op == 2:
        return svc_host_epilogue(a[0], a[1], a[2], a[3], p, f)
    if op == 3:
        return svc_host_platt(a[0], a[1], a[2], p[0])
    if op == 4:
        return svc_host_splitmix(a[0], p[0], p[1], p[2])
    if op == 5:
        return GLOBAL_NUMERIC_MODE
    var n = p[0]
    var d = p[1]
    if op == 0 or op == 6:
        var xp = f32_ptr(a[0])
        var y_address = a[1]
        var dp = f32_ptr(a[2])
        var sip = i32_ptr(a[3])
        var smp = f32_ptr(a[4])
        var ip = f64_ptr(a[5])
        if n <= 0 or d <= 0:
            raise Error(("svc_fit" if op == 0 else "svr_fit") + ": n_rows and n_features must both be positive")
        var labels = read_f32(y_address, max(0, n))
        var bounds_address = a[6] if op == 0 else (p[9] if len(p) > 9 else 0)
        var bounds = List[Float32]()
        if bounds_address != 0:
            bounds = read_f32(bounds_address, n)
        if op == 0:
            return svc_fit_host(xp, labels, bounds, dp, sip, smp, ip, n, d, p[2], f[0], f[1],
                                f[2], p[3], p[4], p[5], f[3])
        var degree = p[10] if len(p) == 12 else 3
        var coef0 = f[11] if len(p) == 12 else 0.0
        return svr_fit_host(xp, labels, bounds, dp, sip, smp, ip, n, d, p[2], f[3], f[4],
                            f[5], f[6], p[7], p[8], degree, coef0)
    var output = f32_ptr(a[3])
    var count = p[2]
    if n <= 0 or d <= 0:
        raise Error(("svc_predict" if op == 1 else "svr_predict") + ": n_rows and n_features must both be positive")
    if count < 0:
        raise Error(("svc_predict" if op == 1 else "svr_predict") + ": n_support cannot be negative")
    var dual = a[1] if count > 0 else 0
    var support = a[2] if count > 0 else 0
    if op == 1:
        svc_predict_host(a[0], output, n, d, count, Float32(f[0]), Float32(f[1]), Float32(f[2]),
                         p[3], f[3], p[4] != 0, f[4], p[5], f[5], dual, support)
    else:
        var degree = p[7] if len(p) == 9 else 3
        var coef0 = f[8] if len(p) == 9 else 0.0
        svr_predict_host(a[0], output, n, d, count, Float32(f[3]), p[4], f[5], f[6],
                         degree, coef0, dual, support)
    return n


def svc_host_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                  result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        svc_binary_spans(op, na, np, nf)
        if result == 0 or (nf > 0 and ff == 0):
            raise Error("SVC native ABI: null result or scalar span")
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        var fp = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        var f = List[Float64]()
        for i in range(nf):  # small-loop(named operation's four or six scalar words)
            f.append(fp[i])
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = svc_host_operation(op, a, p, f)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct SvcHostNativeAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(SVC_ABI_NAME)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(svc_host_v1)


comptime SVC_HOST_NATIVE_API = _Global[StorageType=SvcHostNativeAPI,
    name="MojoSvcHostNativeAPIv1", init_fn=SvcHostNativeAPI.__init__]
