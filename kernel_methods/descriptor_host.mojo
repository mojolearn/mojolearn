# SPDX-License-Identifier: Apache-2.0
# CPU implementation of the existing kernel-method descriptor ABI. These are
# the same typed host owners used by _mojolearn_kernel_methods_host; only the
# argument transport changes. No Python entry, device route or arithmetic copy.
from std.ffi import _Global
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.hostptr import f32_ptr, f64_ptr, i32_ptr, read_f32
from core.py2mojo_linear import py2mojo_linear_flags
from kernel_methods.descriptor_schema import KERNEL_METHOD_ABI, KERNEL_METHOD_OPERATIONS, kernel_method_scalar_mask
from kernel_methods.host.km_host_impl import (
    kmh_kernel_ridge_fit_run, kmh_kernel_ridge_predict_into,
    kmh_nystroem_fit_run, kmh_nystroem_transform_into,
    kmh_rbf_sampler_fit_into, kmh_rbf_sampler_transform_into, kmh_sqrt_weights,
)
from svm.host.scale_gamma_host import scale_gamma_limbs_host


def km_host_operation(operation: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    # The host has no resident/pointer-only device entries. Capability word 14
    # is zero, so ordinary callers choose the supported host fit/predict and
    # the separate fit+transform sequence. Direct unsupported calls fail here.
    if operation == 13 or operation == 14 or operation == 15:
        if len(a) != 0 or len(p) != 0:
            raise Error("kernel methods host provider: metadata takes no operands")
        if operation == 13:
            return GLOBAL_NUMERIC_MODE
        if operation == 14:
            return 0
        return py2mojo_linear_flags()
    if operation == 16:
        if len(a) != 2 or len(p) != 1:
            raise Error("kernel methods host provider: gamma requires two buffers and count")
        var n = p[0]
        if n < 0 or a[1] == 0 or (n > 0 and a[0] == 0):
            raise Error("scale_gamma_limbs: null buffer or negative count")
        scale_gamma_limbs_host(
            MutPointer[UInt32, MutAnyOrigin](unsafe_from_address=a[0]), n,
            MutPointer[Int64, MutAnyOrigin](unsafe_from_address=a[1]),
        )
        return n
    if operation != 0 and operation != 1 and operation != 8 and operation != 10 and operation != 11 and operation != 12:
        raise Error("kernel methods host provider: operation " + String(operation) + " is unavailable on CPU")
    var schema = kernel_method_scalar_mask(operation)
    var addresses = 7 if operation == 8 or operation == 10 else (3 if operation == 11 else 4)
    if len(p) != schema[0] or (len(a) != addresses and not (operation == 0 and len(a) == 5)):
        raise Error("kernel methods host provider: packed admission does not match operation")
    if operation == 0:
        var n = p[0]
        var d = p[1]
        var t = p[2]
        var x = read_f32(a[0], max(0, n * d))
        var y = read_f32(a[1], max(0, n * t))
        var sw = List[Float32]()
        if len(a) == 5:
            sw = kmh_sqrt_weights(f64_ptr(a[4]), n)
        var info = kmh_kernel_ridge_fit_run(x, y, n, d, t, p[3], p[4], f[5], f[6],
                                           Float32(f[7]), f32_ptr(a[2]), f64_ptr(a[3]), sw)
        _ = x^
        _ = y^
        _ = sw^
        return info
    if operation == 1:
        var n = p[0]
        var d = p[1]
        var t = p[2]
        var q = p[9]
        var x_fit = read_f32(a[0], max(0, n * d))
        var dual = read_f32(a[1], max(0, n * t))
        var x_new = read_f32(a[2], max(0, q * d))
        kmh_kernel_ridge_predict_into(x_fit, dual, n, d, t, p[3], p[4], f[5], f[6], x_new, q, f32_ptr(a[3]))
        _ = x_fit^
        _ = dual^
        _ = x_new^
        return 0
    if operation == 8:
        var n = p[0]
        var d = p[1]
        var x = read_f32(a[0], max(0, n * d))
        var sweeps = kmh_nystroem_fit_run(x, n, d, p[2], p[3], f[4], f[5], p[6], UInt64(p[7]),
                                         f32_ptr(a[1]), i32_ptr(a[2]), f32_ptr(a[3]), f32_ptr(a[4]),
                                         f32_ptr(a[5]), f64_ptr(a[6]))
        _ = x^
        return sweeps
    if operation == 10:
        var q = p[0]
        var d = p[1]
        var m = p[8]
        var components = read_f32(a[0], max(0, q * d))
        var normalization = read_f32(a[2], max(0, q * q))
        var x = read_f32(a[5], max(0, m * d))
        kmh_nystroem_transform_into(components, normalization, q, d, p[2], p[3], f[4], f[5], x, m, f32_ptr(a[6]))
        _ = components^
        _ = normalization^
        _ = x^
        return 0
    if operation == 11:
        kmh_rbf_sampler_fit_into(p[0], p[1], Float32(f[2]), UInt64(p[3]), f32_ptr(a[0]), f32_ptr(a[1]), f64_ptr(a[2]))
        return 0
    var d = p[0]
    var q = p[1]
    var m = p[6]
    var weights = read_f32(a[0], max(0, d * q))
    var offset = read_f32(a[1], max(0, q))
    var x = read_f32(a[2], max(0, m * d))
    kmh_rbf_sampler_transform_into(weights, offset, d, q, Float32(f[5]), x, m, f32_ptr(a[3]))
    _ = weights^
    _ = offset^
    _ = x^
    return 0


def km_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
               result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        comptime assert GLOBAL_NUMERIC_MODE == 1, "kernel methods host descriptor requires identical mode"
        if operation < 0 or operation >= KERNEL_METHOD_OPERATIONS or na < 0 or na > 8 or np < 0 or np > 10 or nf != np:
            raise Error("kernel methods host provider: invalid operation or descriptor span")
        if result == 0 or (nf > 0 and ff == 0):
            raise Error("kernel methods host provider: null result or scalar span")
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        var f = List[Float64]()
        var fp = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        for i in range(nf):  # small-loop(at most ten admitted scalar metadata words)
            f.append(fp[i])
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = km_host_operation(operation, a, p, f)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct KernelMethodHostAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]
    def __init__(out self):
        self.name = String(KERNEL_METHOD_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(km_host_v1)


comptime KERNEL_METHOD_HOST_API = _Global[StorageType=KernelMethodHostAPI,
    name="MojoKernelMethodHostAPIv1", init_fn=KernelMethodHostAPI.__init__]
