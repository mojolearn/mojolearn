# SPDX-License-Identifier: Apache-2.0
# Common private descriptor ABI over existing SVC solver/predict owners.
# No estimator-specific provider, Python dependency or new numerical route.
from std.ffi import _Global
from std.memory import memcpy
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.hostptr import f32_ptr, f64_ptr, i32_ptr, read_f32, copy_f32
from svm.operation_metadata import SVC_ABI_NAME
from svm.estimator import svc_fit_host_borrowed, svc_predict_host


def svc_binary_operation(op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    var n = p[0]
    var d = p[1]
    if op == 0:
        # Same pointer admission/refusal order as the existing Python door.
        var xp = f32_ptr(a[0])
        var yp = f32_ptr(a[1])
        var dp = f32_ptr(a[2])
        var sip = i32_ptr(a[3])
        var smp = f32_ptr(a[4])
        var ip = f64_ptr(a[5])
        if n <= 0 or d <= 0:
            raise Error("svc_fit: n_rows and n_features must both be positive")
        var y = read_f32(Int(yp), n)
        var bounds = List[Float32]()
        if a[6] != 0:
            bounds = read_f32(a[6], n)
        var result = svc_fit_host_borrowed(xp, y, n, d,
            p[2], f[0], f[1], f[2], p[3], p[4], p[5], f[3], bounds)
        copy_f32(result.dual_coefs.unsafe_ptr(), dp, result.n_support)
        memcpy(dest=sip, src=result.support_idx.unsafe_ptr(), count=result.n_support)
        copy_f32(result.support_matrix.unsafe_ptr(), smp, result.n_support * d)
        var info = ip
        info[0] = Float64(result.b)
        info[1] = Float64(result.n_support)
        info[2] = Float64(result.n_iter)
        info[3] = Float64(result.label0)
        info[4] = Float64(result.label1)
        return result.n_support
    var xp = f32_ptr(a[0])
    var optr = f32_ptr(a[3])
    if n <= 0 or d <= 0:
        raise Error("svc_predict: n_rows and n_features must both be positive")
    var count = p[2]
    if count < 0:
        raise Error("svc_predict: n_support cannot be negative")
    var x = read_f32(Int(xp), n * d)
    var dual = List[Float32]()
    var support = List[Float32]()
    if count > 0:
        var dp = f32_ptr(a[1])
        var smp = f32_ptr(a[2])
        dual = read_f32(Int(dp), count)
        if p[3] != 4:
            support = read_f32(Int(smp), count * d)
    var output = svc_predict_host(x, n, d, support, dual, count,
        Float32(f[0]), Float32(f[1]), Float32(f[2]), p[3], f[3], p[4] != 0,
        f[4], p[5], f[5])
    copy_f32(output.unsafe_ptr(), optr, n)
    return n


def svc_binary_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                  result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if op < 0 or op > 1:
            raise Error("SVC native ABI: unknown operation")
        if na != (7 if op == 0 else 4) or np != 6 or nf != (4 if op == 0 else 6):
            raise Error("SVC native ABI: descriptor span differs from the named operation")
        if result == 0 or ff == 0:
            raise Error("SVC native ABI: null result or scalar span")
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        var fp = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        var f = List[Float64]()
        for i in range(nf):  # small-loop(named operation's four or six scalar words)
            f.append(fp[i])
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = svc_binary_operation(op, a, p, f)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct SvcNativeAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(SVC_ABI_NAME)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(svc_binary_v1)


comptime SVC_NATIVE_API = _Global[StorageType=SvcNativeAPI,
    name="MojoSvcNativeAPIv1", init_fn=SvcNativeAPI.__init__]
