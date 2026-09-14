# SPDX-License-Identifier: Apache-2.0
"""Cloud-only cellwise identity for linear/RBF output-row partitions."""
from std.os import getenv, setenv
from std.memory import bitcast
from max.gpu.host import DeviceContext
from metrics.checks.device_io import upload_f32, download_f32
from svm.impl.distance.kernel_matrices import kernel_op, kernel_workspace_floats, row_norms_l2sq
from svm.impl.svm_parameter import KernelParams


def main() raises:
    if String(getenv("RUNPOD_POD_ID")) == "":
        raise Error("cloud host required")
    var ctx = DeviceContext()
    var rows: List[Int] = [1, 3, 17]
    var features: List[Int] = [7, 129, 257]
    for m in rows:
        for k in features:
            var n = 19
            var av = List[Float32]()
            var bv = List[Float32]()
            for i in range(m * k):
                var value = Float32((i * 37) % 127 - 63) / Float32(64)
                if i % 19 == 0:
                    value = Float32(1e-40)
                elif i % 23 == 0:
                    value = Float32(-0.0)
                av.append(value)
            for i in range(n * k):
                bv.append(Float32((i * 17) % 113 - 56) / Float32(64))
            var a = upload_f32(ctx, av)
            var b = upload_f32(ctx, bv)
            var na = ctx.enqueue_create_buffer[DType.float32](m)
            var nb = ctx.enqueue_create_buffer[DType.float32](n)
            row_norms_l2sq(ctx, na, a, m, k)
            row_norms_l2sq(ctx, nb, b, n, k)
            var one = ctx.enqueue_create_buffer[DType.float32](m * n)
            var many = ctx.enqueue_create_buffer[DType.float32](m * n)
            var ws = ctx.enqueue_create_buffer[DType.float32](kernel_workspace_floats(m, n, k))
            for kind in range(2):
                var kp = KernelParams.linear()
                if kind == 1:
                    kp = KernelParams.rbf(0.01)
                if not setenv("MOJOLEARN_SVM_DEVICE_COUNT", "1", True):
                    raise Error("setenv failed")
                kernel_op(ctx, kp, one, a, b, m, n, k, na, nb, ws)
                ctx.synchronize()
                if not setenv("MOJOLEARN_SVM_DEVICE_COUNT", "2", True):
                    raise Error("setenv failed")
                kernel_op(ctx, kp, many, a, b, m, n, k, na, nb, ws)
                ctx.synchronize()
                var x = download_f32(ctx, one, m * n)
                var y = download_f32(ctx, many, m * n)
                for i in range(m * n):
                    if bitcast[DType.uint32](x[i]) != bitcast[DType.uint32](y[i]):
                        raise Error("SVM kernel mismatch " + String(m) + "/" + String(k) + "/" + String(kind) + "/" + String(i))
                print("PASS SVM kernel cells", m, k, kind)
            _ = a^
            _ = b^
            _ = na^
            _ = nb^
            _ = one^
            _ = many^
            _ = ws^
    ctx.synchronize()
