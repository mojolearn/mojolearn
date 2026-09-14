# SPDX-License-Identifier: Apache-2.0
"""Cloud-only covariance cells, including structural WhiteKernel diagonals."""
from std.os import getenv, setenv
from std.memory import bitcast
from max.gpu.host import DeviceContext
from core.identity_trace import IdentityTrace
from metrics.checks.device_io import upload_f32, download_f32
from gaussian_process.checks.kernels import gp_kernel_matrix, gp_kernel_stack_floats, gp_kernel_rbf, gp_kernel_white, gp_kernel_sum, gp_kernel_matern


def main() raises:
    if String(getenv("RUNPOD_POD_ID")) == "":
        raise Error("cloud host required")
    var ctx = DeviceContext()
    var sizes: List[Int] = [5, 17]
    for m in sizes:
        var values = List[Float32]()
        for i in range(m * 3):
            values.append(Float32((i * 17) % 71) / Float32(64))
        var x = upload_f32(ctx, values)
        var one = ctx.enqueue_create_buffer[DType.float32](m * m)
        var many = ctx.enqueue_create_buffer[DType.float32](m * m)
        var stack = ctx.enqueue_create_buffer[DType.float32](gp_kernel_stack_floats(m, m))
        for kind in range(4):
            var ls: List[Float32] = [0.7, 1.0, 1.3]
            var leaf = gp_kernel_rbf(ls)
            if kind > 0:
                leaf = gp_kernel_matern(ls, Float32(kind) - Float32(0.5))
            var white = gp_kernel_white(Float32(0.2))
            var spec = gp_kernel_sum(leaf, white)
            var dls = upload_f32(ctx, spec.length_scales)
            for self_case in range(2):
                var trace = IdentityTrace.disabled()
                if not setenv("MOJOLEARN_GP_DEVICE_COUNT", "1", True):
                    raise Error("setenv failed")
                gp_kernel_matrix(ctx, one, x, x, dls, stack, m, m, 3, spec, self_case == 1, trace, "one")
                ctx.synchronize()
                if not setenv("MOJOLEARN_GP_DEVICE_COUNT", "2", True):
                    raise Error("setenv failed")
                gp_kernel_matrix(ctx, many, x, x, dls, stack, m, m, 3, spec, self_case == 1, trace, "many")
                ctx.synchronize()
                var a = download_f32(ctx, one, m * m)
                var b = download_f32(ctx, many, m * m)
                for i in range(m * m):
                    if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
                        raise Error("GP cell differs " + String(m) + "/" + String(kind) + "/" + String(self_case) + "/" + String(i))
                print("PASS GP covariance cells", m, kind, self_case)
            _ = dls^
        _ = x^
        _ = one^
        _ = many^
        _ = stack^
    ctx.synchronize()
