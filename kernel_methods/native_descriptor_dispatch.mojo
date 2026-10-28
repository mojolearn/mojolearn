# SPDX-License-Identifier: Apache-2.0
# Native descriptor transport over the existing kernel-method owners. Bodies
# below preserve the packed binding's refusals, routes, conversions and calls.
# No Python/provider dependency or new numerical implementation belongs here.
from std.ffi import _Global
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.hostptr import f32_ptr as _f32_ptr, f64_ptr as _f64_ptr, i32_ptr as _i32_ptr, read_f32, read_i32
from core.identity_trace import IdentityTrace
from svm.impl.svm_parameter import KernelParams
from kernel_methods.estimator import (KernelRidgeModel, NystroemModel, RBFSamplerModel,
    KRR_IDN_PTR_IN, NYS_IDN_FIT_PTR_IN, KM_FAST_PTR_IN,
    kernel_ridge_fit_ptr_into, kernel_ridge_predict_ptr_into,
    nystroem_transform_ptr_into, rbf_sampler_transform_ptr_into)
from kernel_methods.km_binding_impl import (
    km_kernel_ridge_fit_run as _kernel_ridge_fit_run,
    km_kernel_ridge_predict_run as _kernel_ridge_predict_run,
    km_nystroem_fit_run as _nystroem_fit_run,
    km_nystroem_transform_run as _nystroem_transform_run,
    km_rbf_sampler_fit_run as _rbf_sampler_fit_run,
    km_rbf_sampler_transform_run as _rbf_sampler_transform_run,
    km_nystroem_fit_ptr_run, km_nystroem_fit_transform_ptr_run, km_sqrt_weights)
from kernel_methods.krr_resident import (
    kernel_ridge_resident_prepare, kernel_ridge_resident_release,
    kernel_ridge_predict_resident)
from core.py2mojo_linear import py2mojo_linear_flags
from svm.impl.scale_gamma_device import scale_gamma_limbs_device
from kernel_methods.estimator import _family_ctx

from kernel_methods.descriptor_schema import KERNEL_METHOD_ABI, KERNEL_METHOD_OPERATIONS

def km_native_kernel_ridge_fit(a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if len(a) != 4 and len(a) != 5:
        raise Error(
            "kernel_ridge_fit: addrs must contain 4 addresses (x, y,"
            " dual_out, scalars_out) and optionally sw, got "
            + String(len(a))
        )
    if len(p) != 8:
        raise Error(
            "kernel_ridge_fit: params must contain 8 values (n, d, t, kernel,"
            " degree, gamma, coef0, alpha), got "
            + String(len(p))
        )
    var xp = _f32_ptr(a[0])
    var yp = _f32_ptr(a[1])
    var dp = _f32_ptr(a[2])
    var sp = _f64_ptr(a[3])
    var n = p[0]
    var d = p[1]
    var t = p[2]
    var kp = KernelParams(
        p[3],
        p[4],
        f[5],
        f[6],
    )
    var alpha = Float32(f[7])
    var x = read_f32(Int(xp), max(0, n * d))
    var y = read_f32(Int(yp), max(0, n * t))
    var sw = List[Float32]()
    if len(a) == 5:
        # the per-row factors sqrt(sample_weight): the binary64 square root
        # of each weight rounded once to float32 (formerly Python's loop)
        sw = km_sqrt_weights(_f64_ptr(a[4]), n)
    var info = 0
    info = _kernel_ridge_fit_run(x, y, n, d, t, kp, alpha, dp, sp, sw)
    return info


def km_native_kernel_ridge_predict(a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if len(a) != 4:
        raise Error(
            "kernel_ridge_predict: addrs must contain 4 addresses (x_fit,"
            " dual, x_new, out), got "
            + String(len(a))
        )
    if len(p) != 10:
        raise Error(
            "kernel_ridge_predict: params must contain 10 values (n, d, t,"
            " kernel, degree, gamma, coef0, alpha, info, q), got "
            + String(len(p))
        )
    var xfp = _f32_ptr(a[0])
    var dp = _f32_ptr(a[1])
    var xnp = _f32_ptr(a[2])
    var op = _f32_ptr(a[3])
    var n = p[0]
    var d = p[1]
    var t = p[2]
    var kernel = p[3]
    var degree = p[4]
    var gamma = f[5]
    var coef0 = f[6]
    var alpha = Float32(f[7])
    var info = p[8]
    var q = p[9]
    var x_fit = read_f32(Int(xfp), max(0, n * d))
    var dual = read_f32(Int(dp), max(0, n * t))
    var x_new = read_f32(Int(xnp), max(0, q * d))
    var model = KernelRidgeModel(
        dual^, x_fit^, n, d, t, kernel, degree, gamma, coef0, alpha, info
    )
    _kernel_ridge_predict_run(model, x_new, q, op)
    return 0


def km_native_kernel_ridge_fit_ptr(a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if len(a) != 4 and len(a) != 5:
        raise Error(
            "kernel_ridge_fit: addrs must contain 4 addresses (x, y,"
            " dual_out, scalars_out) and optionally sw, got "
            + String(len(a))
        )
    if len(p) != 8:
        raise Error(
            "kernel_ridge_fit: params must contain 8 values (n, d, t, kernel,"
            " degree, gamma, coef0, alpha), got "
            + String(len(p))
        )
    var xp = _f32_ptr(a[0])
    var yp = _f32_ptr(a[1])
    var dp = _f32_ptr(a[2])
    var sp = _f64_ptr(a[3])
    var n = p[0]
    var d = p[1]
    var t = p[2]
    var kp = KernelParams(
        p[3],
        p[4],
        f[5],
        f[6],
    )
    var alpha = Float32(f[7])
    var sw = List[Float32]()
    var waddr = 0
    if len(a) == 5:
        # cpu2-l6-bindings: the per-row factors sqrt(sample_weight) are
        # formed and refused on the device from the caller's float64
        # weights (`_krr_sqrt_weights_dev`), not in a host loop
        waddr = Int(_f64_ptr(a[4]))
    var xaddr = Int(xp)
    var yaddr = Int(yp)
    var info = 0
    var trace = IdentityTrace()
    info = kernel_ridge_fit_ptr_into(
        xaddr, yaddr, n, d, t, kp, alpha, sw, dp, trace, waddr=waddr
    )
    sp.unsafe_store(0, Float64(info))
    return info


def km_native_kernel_ridge_predict_ptr(a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if len(a) != 4:
        raise Error(
            "kernel_ridge_predict: addrs must contain 4 addresses (x_fit,"
            " dual, x_new, out), got "
            + String(len(a))
        )
    if len(p) != 10:
        raise Error(
            "kernel_ridge_predict: params must contain 10 values (n, d, t,"
            " kernel, degree, gamma, coef0, alpha, info, q), got "
            + String(len(p))
        )
    var xfaddr = Int(_f32_ptr(a[0]))
    var daddr = Int(_f32_ptr(a[1]))
    var xnaddr = Int(_f32_ptr(a[2]))
    var op = _f32_ptr(a[3])
    var n = p[0]
    var d = p[1]
    var t = p[2]
    var kp = KernelParams(
        p[3],
        p[4],
        f[5],
        f[6],
    )
    var q = p[9]
    var trace = IdentityTrace()
    kernel_ridge_predict_ptr_into(xfaddr, daddr, xnaddr, n, d, t, kp, q, op, trace)
    return 0


def km_native_kernel_ridge_resident_prepare(a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if len(a) != 2 or len(p) != 3:
        raise Error("kernel_ridge_resident_prepare: addrs must hold x_fit, dual and params n, d, t")
    var xfaddr = Int(_f32_ptr(a[0]))
    var daddr = Int(_f32_ptr(a[1]))
    var n = p[0]
    var d = p[1]
    var t = p[2]
    var handle: Int
    handle = kernel_ridge_resident_prepare(xfaddr, daddr, n, d, t)
    return handle


def km_native_kernel_ridge_resident_release(a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    kernel_ridge_resident_release(a[0])
    return 0


def km_native_kernel_ridge_predict_resident(original: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    var handle = original[0]
    var a: List[Int] = [original[1], original[2]]
    if len(a) != 2:
        raise Error(
            "kernel_ridge_predict_resident: addrs must contain 2 addresses (x_new, out), got "
            + String(len(a))
        )
    if len(p) != 10:
        raise Error(
            "kernel_ridge_predict: params must contain 10 values (n, d, t,"
            " kernel, degree, gamma, coef0, alpha, info, q), got "
            + String(len(p))
        )
    var xnaddr = Int(_f32_ptr(a[0]))
    var op = _f32_ptr(a[1])
    var n = p[0]
    var d = p[1]
    var t = p[2]
    var kp = KernelParams(
        p[3],
        p[4],
        f[5],
        f[6],
    )
    var q = p[9]
    var h = handle
    var trace = IdentityTrace()
    kernel_ridge_predict_resident(h, xnaddr, n, d, t, kp, q, op, trace)
    return 0


def km_native_nystroem_fit_transform_ptr(a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if len(a) != 8:
        raise Error(
            "nystroem_fit_transform: addrs must contain 8 addresses (x, components_out,"
            " indices_out, normalization_out, eigenvalues_out,"
            " eigenvectors_out, scalars_out, transform_out), got "
            + String(len(a))
        )
    if len(p) != 8:
        raise Error(
            "nystroem_fit: params must contain 8 values (n, d, kernel, degree,"
            " gamma, coef0, q, seed), got "
            + String(len(p))
        )
    var xaddr = Int(_f32_ptr(a[0]))
    var cp = _f32_ptr(a[1])
    var ip = _i32_ptr(a[2])
    var np_ = _f32_ptr(a[3])
    var evp = _f32_ptr(a[4])
    var ecp = _f32_ptr(a[5])
    var sp = _f64_ptr(a[6])
    var op = _f32_ptr(a[7])
    var n = p[0]
    var d = p[1]
    var kp = KernelParams(
        p[2],
        p[3],
        f[4],
        f[5],
    )
    var q = p[6]
    var seed = UInt64(p[7])
    var sweeps = 0
    sweeps = km_nystroem_fit_transform_ptr_run(xaddr, n, d, kp, q, seed, cp, ip, np_, evp, ecp, sp, op)
    return sweeps


def km_native_nystroem_fit(a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if len(a) != 7:
        raise Error(
            "nystroem_fit: addrs must contain 7 addresses (x, components_out,"
            " indices_out, normalization_out, eigenvalues_out,"
            " eigenvectors_out, scalars_out), got "
            + String(len(a))
        )
    if len(p) != 8:
        raise Error(
            "nystroem_fit: params must contain 8 values (n, d, kernel, degree,"
            " gamma, coef0, q, seed), got "
            + String(len(p))
        )
    var xp = _f32_ptr(a[0])
    var cp = _f32_ptr(a[1])
    var ip = _i32_ptr(a[2])
    var np_ = _f32_ptr(a[3])
    var evp = _f32_ptr(a[4])
    var ecp = _f32_ptr(a[5])
    var sp = _f64_ptr(a[6])
    var n = p[0]
    var d = p[1]
    var kp = KernelParams(
        p[2],
        p[3],
        f[4],
        f[5],
    )
    var q = p[6]
    var seed = UInt64(p[7])
    var x = read_f32(Int(xp), max(0, n * d))
    var sweeps = 0
    sweeps = _nystroem_fit_run(
        x, n, d, kp, q, seed, cp, ip, np_, evp, ecp, sp
    )
    return sweeps


def km_native_nystroem_fit_ptr(a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if len(a) != 7:
        raise Error(
            "nystroem_fit: addrs must contain 7 addresses (x, components_out,"
            " indices_out, normalization_out, eigenvalues_out,"
            " eigenvectors_out, scalars_out), got "
            + String(len(a))
        )
    if len(p) != 8:
        raise Error(
            "nystroem_fit: params must contain 8 values (n, d, kernel, degree,"
            " gamma, coef0, q, seed), got "
            + String(len(p))
        )
    var xaddr = Int(_f32_ptr(a[0]))
    var cp = _f32_ptr(a[1])
    var ip = _i32_ptr(a[2])
    var np_ = _f32_ptr(a[3])
    var evp = _f32_ptr(a[4])
    var ecp = _f32_ptr(a[5])
    var sp = _f64_ptr(a[6])
    var n = p[0]
    var d = p[1]
    var kp = KernelParams(
        p[2],
        p[3],
        f[4],
        f[5],
    )
    var q = p[6]
    var seed = UInt64(p[7])
    var sweeps = 0
    sweeps = km_nystroem_fit_ptr_run(xaddr, n, d, kp, q, seed, cp, ip, np_, evp, ecp, sp)
    return sweeps


def km_native_nystroem_transform(a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if len(a) != 7:
        raise Error(
            "nystroem_transform: addrs must contain 7 addresses (components,"
            " indices, normalization, eigenvalues, eigenvectors, x, out),"
            " got "
            + String(len(a))
        )
    if len(p) != 9:
        raise Error(
            "nystroem_transform: params must contain 9 values (q, d, kernel,"
            " degree, gamma, coef0, seed, sweeps, m), got "
            + String(len(p))
        )
    var q = p[0]
    var d = p[1]
    var kernel = p[2]
    var degree = p[3]
    var gamma = f[4]
    var coef0 = f[5]
    var seed = UInt64(p[6])
    var sweeps = p[7]
    var m = p[8]
    var components = read_f32(a[0], max(0, q * d))
    var indices = read_i32(a[1], max(0, q))
    var normalization = read_f32(a[2], max(0, q * q))
    var eigenvalues = read_f32(a[3], max(0, q))
    var eigenvectors = read_f32(a[4], max(0, q * q))
    var xaddr = a[5]
    var op = _f32_ptr(a[6])
    var model = NystroemModel(
        components^,
        indices^,
        normalization^,
        eigenvalues^,
        eigenvectors^,
        q,
        d,
        kernel,
        degree,
        gamma,
        coef0,
        seed,
        sweeps,
    )
    comptime if KM_FAST_PTR_IN:
        if xaddr == 0:
            raise Error("nystroem_transform: null X address")
        var trace = IdentityTrace()
        nystroem_transform_ptr_into(model, xaddr, m, op, trace)
    else:
        var x = read_f32(xaddr, max(0, m * d))
        _nystroem_transform_run(model, x, m, op)
    return 0


def km_native_rbf_sampler_fit(a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if len(a) != 3:
        raise Error(
            "rbf_sampler_fit: addrs must contain 3 addresses (weights_out,"
            " offset_out, scalars_out), got "
            + String(len(a))
        )
    if len(p) != 4:
        raise Error(
            "rbf_sampler_fit: params must contain 4 values (d, q, gamma,"
            " seed), got "
            + String(len(p))
        )
    var wp = _f32_ptr(a[0])
    var bp = _f32_ptr(a[1])
    var sp = _f64_ptr(a[2])
    var d = p[0]
    var q = p[1]
    var gamma = Float32(f[2])
    var seed = UInt64(p[3])
    _rbf_sampler_fit_run(d, q, gamma, seed, wp, bp, sp)
    return 0


def km_native_rbf_sampler_transform(a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if len(a) != 4:
        raise Error(
            "rbf_sampler_transform: addrs must contain 4 addresses (weights,"
            " offset, x, out), got "
            + String(len(a))
        )
    if len(p) != 7:
        raise Error(
            "rbf_sampler_transform: params must contain 7 values (d, q,"
            " gamma, seed, sigma, scale, m), got "
            + String(len(p))
        )
    var d = p[0]
    var q = p[1]
    var gamma = Float32(f[2])
    var seed = UInt64(p[3])
    var sigma = Float32(f[4])
    var scale = Float32(f[5])
    var m = p[6]
    var weights = read_f32(a[0], max(0, d * q))
    var offset = read_f32(a[1], max(0, q))
    var xaddr = a[2]
    var op = _f32_ptr(a[3])
    var model = RBFSamplerModel(
        weights^, offset^, d, q, gamma, seed, sigma, scale
    )
    comptime if KM_FAST_PTR_IN:
        if xaddr == 0:
            raise Error("rbf_sampler_transform: null X address")
        var trace = IdentityTrace()
        rbf_sampler_transform_ptr_into(model, xaddr, m, op, trace)
    else:
        var x = read_f32(xaddr, max(0, m * d))
        _rbf_sampler_transform_run(model, x, m, op)
    return 0

def km_native_operation(operation: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if operation == 0:
        return km_native_kernel_ridge_fit(a, p, f)
    if operation == 1:
        return km_native_kernel_ridge_predict(a, p, f)
    if operation == 2:
        return km_native_kernel_ridge_fit_ptr(a, p, f)
    if operation == 3:
        return km_native_kernel_ridge_predict_ptr(a, p, f)
    if operation == 4:
        return km_native_kernel_ridge_resident_prepare(a, p, f)
    if operation == 5:
        return km_native_kernel_ridge_resident_release(a, p, f)
    if operation == 6:
        return km_native_kernel_ridge_predict_resident(a, p, f)
    if operation == 7:
        return km_native_nystroem_fit_transform_ptr(a, p, f)
    if operation == 8:
        return km_native_nystroem_fit(a, p, f)
    if operation == 9:
        return km_native_nystroem_fit_ptr(a, p, f)
    if operation == 10:
        return km_native_nystroem_transform(a, p, f)
    if operation == 11:
        return km_native_rbf_sampler_fit(a, p, f)
    if operation == 12:
        return km_native_rbf_sampler_transform(a, p, f)
    if operation == 13:
        return GLOBAL_NUMERIC_MODE
    if operation == 14:
        return Int(KRR_IDN_PTR_IN) | (Int(NYS_IDN_FIT_PTR_IN) << 1) | (Int(KM_FAST_PTR_IN) << 2)
    if operation == 15:
        return py2mojo_linear_flags()
    if operation == 16:
        var n = p[0]
        if n < 0 or a[1] == 0 or (n > 0 and a[0] == 0):
            raise Error("scale_gamma_limbs: null buffer or negative count")
        var ctx = _family_ctx()
        scale_gamma_limbs_device(ctx, a[0], n, a[1])
        return n
    raise Error("kernel methods provider: unknown operation")


def km_native_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                 result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if operation < 0 or operation >= KERNEL_METHOD_OPERATIONS or na < 0 or na > 8 or np < 0 or np > 10 or nf != np:
            raise Error("kernel methods provider: invalid operation or descriptor span")
        if result == 0 or (nf > 0 and ff == 0):
            raise Error("kernel methods provider: null result or scalar span")
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        var f = List[Float64]()
        var fp = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        for i in range(nf):  # small-loop(at most ten admitted scalar metadata words)
            f.append(fp[i])
        if operation >= 13 and operation <= 15:
            if na != 0 or np != 0:
                raise Error("kernel methods provider: metadata takes no operands")
        if operation == 16 and (na != 2 or np != 1):
            raise Error("kernel methods provider: gamma requires two buffers and count")
        if operation == 5 and (na != 1 or np != 0):
            raise Error("kernel methods provider: release requires one native handle")
        if operation == 6 and na != 3:
            raise Error("kernel methods provider: resident predict requires handle and two buffers")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = km_native_operation(operation, a, p, f)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct KernelMethodNativeAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]
    def __init__(out self):
        self.name = String(KERNEL_METHOD_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(km_native_v1)

comptime KERNEL_METHOD_NATIVE_API = _Global[StorageType=KernelMethodNativeAPI,
    name="MojoKernelMethodNativeAPIv1", init_fn=KernelMethodNativeAPI.__init__]
