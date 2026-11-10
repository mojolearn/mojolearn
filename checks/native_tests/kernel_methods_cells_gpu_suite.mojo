# SPDX-License-Identifier: Apache-2.0
"""Actual GPU matrix/Nystroem paths, ordered on their existing device context.

A device stream is serial by contract; fixture submissions are ordered.
Independent input decoding, gathers and output checks use host_parallelize.
No tuned kernel, dispatch choice, fusion or numerical setting is replaced.
"""
from core.host_parallel import host_parallelize
from core.identity_trace import IdentityTrace
from svm.impl.svm_parameter import KernelParams
from kernel_methods.estimator import _family_ctx, nystroem_fit_host, nystroem_transform_host
from kernel_methods.checks.kernel_matrix import km_kernel_matrix, km_kernel_workspace_floats
from checks.native_tests.kernel_methods_cells_cases import (
    KernelMethodsCellCalls, KERNEL_METHODS_CELLS_GPU_FIXTURES, fixture, require,
)

comptime CELL_OUTPUT_GUARD = Float32(27182.5)


struct DeviceKernelCells(KernelMethodsCellCalls):
    @staticmethod
    def matrix(a: List[Float32], b: List[Float32], m: Int, n: Int, d: Int,
               kernel: Int, gamma: Float64) raises -> List[Float32]:
        var ctx = _family_ctx()
        var da = ctx.enqueue_create_buffer[DType.float32](max(1, len(a)))
        var db = ctx.enqueue_create_buffer[DType.float32](max(1, len(b)))
        var dst = ctx.enqueue_create_buffer[DType.float32](m * n + 2)
        var norm_a = ctx.enqueue_create_buffer[DType.float32](max(1, m))
        var norm_b = ctx.enqueue_create_buffer[DType.float32](max(1, n))
        var workspace = ctx.enqueue_create_buffer[DType.float32](max(1, km_kernel_workspace_floats(m, n, d)))
        if len(a):
            ctx.enqueue_copy(dst_buf=da, src_ptr=a.unsafe_ptr())
        if len(b):
            ctx.enqueue_copy(dst_buf=db, src_ptr=b.unsafe_ptr())
        ctx.enqueue_memset(dst, CELL_OUTPUT_GUARD)
        var parameters = KernelParams(kernel, 3, gamma, 1.0)
        km_kernel_matrix(ctx, parameters, dst, da, db, m, n, d, norm_a, norm_b, workspace)
        var downloaded = List[Float32](length=m * n + 2, fill=Float32(0))
        ctx.enqueue_copy(dst_ptr=downloaded.unsafe_ptr(), src_buf=dst)
        ctx.synchronize()
        require(downloaded[m * n] == CELL_OUTPUT_GUARD and downloaded[m * n + 1] == CELL_OUTPUT_GUARD,
            "GPU matrix wrote beyond its requested cells or touched empty output")
        var result = List[Float32](length=m * n, fill=Float32(0))
        var sp = downloaded.unsafe_ptr()
        var rp = result.unsafe_ptr()
        def copy(at: Int) {imm sp, imm rp}:
            rp.unsafe_store(at, sp[at])
        host_parallelize(copy, m * n)
        _ = downloaded^
        _ = da^
        _ = db^
        _ = dst^
        _ = norm_a^
        _ = norm_b^
        _ = workspace^
        _ = a
        _ = b
        _ = ctx^
        return result^

    @staticmethod
    def basis(x: List[Float32], kernel: Int) raises -> Tuple[List[Int32], List[Float32]]:
        var trace = IdentityTrace.disabled()
        var gamma = Float64(1) / Float64(6) if kernel == 6 else Float64(1)
        var parameters = KernelParams(kernel, 3, gamma, 1.0)
        var model = nystroem_fit_host(x, 80, 6, parameters, 12, UInt64(1), trace)
        require(len(model.component_indices) == 12, "Nystroem device basis index extent")
        for i in range(12):
            require(model.component_indices[i] >= 0 and model.component_indices[i] < 80, "Nystroem device basis index outside input")
        var basis = List[Float32](length=72, fill=Float32(0))
        var ip = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=Int(model.component_indices.unsafe_ptr()))
        var xp = x.unsafe_ptr()
        var bp = basis.unsafe_ptr()
        def gather(at: Int) {imm ip, imm xp, imm bp}:
            bp.unsafe_store(at, xp[Int(ip[at // 6]) * 6 + at % 6])
        host_parallelize(gather, 72)
        var embedded = nystroem_transform_host(model, basis, 12, trace)
        var indices = model.component_indices.copy()
        _ = model^
        _ = trace^
        _ = basis^
        _ = x
        return (indices^, embedded^)


def run_checks() raises:
    var failed = 0
    for index in range(KERNEL_METHODS_CELLS_GPU_FIXTURES):
        try:
            fixture[DeviceKernelCells](index)
        except error:
            failed += 1
            print("FAIL kernel_methods_cells_gpu fixture", index, String(error))
    require(failed == 0, "kernel_methods_cells_gpu failed " + String(failed) + " fixtures")
    print("PASS kernel_methods_cells_gpu:", KERNEL_METHODS_CELLS_GPU_FIXTURES, "native matrix/basis fixtures")
