# SPDX-License-Identifier: Apache-2.0
"""Cloud-only comparison of distributed dot leaves with FP32-v1 oracle/plans."""
from std.os import getenv, setenv
from std.memory import bitcast
from max.gpu.host import DeviceContext
from solver.checks.profile_dot import profile_dot_into, profile_dot_workspace_floats, profile_dot_host
from gemm.checks.gemm_identical import GEMM_PLAN_COUNT
from metrics.checks.device_io import upload_f32, download_f32


def main() raises:
    if String(getenv("RUNPOD_POD_ID")) == "":
        raise Error("Run this gate on the cloud host, never locally")
    var ctx = DeviceContext()
    var rows: List[Int] = [1, 127, 128, 129, 257, 1031, 131073]
    for k in rows:
        var av = List[Float32](length=k, fill=Float32(0))
        var bv = List[Float32](length=k, fill=Float32(0))
        for i in range(k):
            av[i] = Float32((i * 37) % 257 - 128) / Float32(64)
            bv[i] = Float32((i * 17) % 127 - 63) / Float32(32)
            if i % 19 == 0:
                av[i] = Float32(1e-40)
            elif i % 23 == 0:
                av[i] = Float32(-0.0)
        var expected = profile_dot_host(av, bv, k)
        var a = upload_f32(ctx, av)
        var b = upload_f32(ctx, bv)
        var c = ctx.enqueue_create_buffer[DType.float32](1)
        var ws = ctx.enqueue_create_buffer[DType.float32](profile_dot_workspace_floats(k))
        for count in range(1, 3):
            if not setenv("MOJOLEARN_SOLVER_DEVICE_COUNT", String(count), True):
                raise Error("cannot set device count")
            # Automatic selects distributed leaves; explicit plans remain probes
            # of their original launches, even inside the cooperative worker.
            for plan in range(-1, GEMM_PLAN_COUNT):
                profile_dot_into(ctx, c, a, b, ws, k, plan)
                ctx.synchronize()
                var actual = download_f32(ctx, c, 1)
                if bitcast[DType.uint32](actual[0]) != bitcast[DType.uint32](expected):
                    raise Error("dot mismatch " + String(k) + "/" + String(count) + "/" + String(plan))
            print("PASS dot oracle and plans", k, count)
        _ = a^
        _ = b^
        _ = c^
        _ = ws^
    ctx.synchronize()
