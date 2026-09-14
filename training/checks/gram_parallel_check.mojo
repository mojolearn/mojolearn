# SPDX-License-Identifier: Apache-2.0
"""Cloud-only bit checks of every pinned Gram partial and the final fold."""
from std.os import getenv, setenv
from std.memory import bitcast
from max.gpu.host import DeviceContext
from core.gram_splitk import _splitk_launch, _splitk_launch_centered, gram_splitk_chunk_count
from metrics.checks.device_io import upload_f32, download_f32


def main() raises:
    if String(getenv("RUNPOD_POD_ID")) == "":
        raise Error("Run this gate on the cloud host, never locally")
    var ctx = DeviceContext()
    var widths: List[Int] = [7, 16, 33, 65, 128]
    var rows: List[Int] = [1, 127, 259, 1031]
    for m in widths:
        for k in rows:
            var values = List[Float32](length=k * m, fill=Float32(0))
            for i in range(k * m):
                values[i] = Float32((i * 37) % 257 - 128) / Float32(64)
                if i % 19 == 0:
                    values[i] = Float32(1e-40)
                elif i % 23 == 0:
                    values[i] = Float32(-0.0)
            var means = List[Float32](length=m, fill=Float32(0.25))
            var x = upload_f32(ctx, values)
            var mu = upload_f32(ctx, means)
            var cells = m * m
            var pcells = gram_splitk_chunk_count() * cells
            var one = ctx.enqueue_create_buffer[DType.float32](cells)
            var many = ctx.enqueue_create_buffer[DType.float32](cells)
            var p_one = ctx.enqueue_create_buffer[DType.float32](pcells)
            var p_many = ctx.enqueue_create_buffer[DType.float32](pcells)
            for centered in range(2):
                if not setenv("MOJOLEARN_GRAM_DEVICE_COUNT", "1", True):
                    raise Error("cannot set reference device count")
                if centered == 0:
                    _splitk_launch(ctx, one, x, p_one, m, k)
                else:
                    _splitk_launch_centered(ctx, one, x, mu, p_one, m, k)
                ctx.synchronize()
                if not setenv("MOJOLEARN_GRAM_DEVICE_COUNT", "2", True):
                    raise Error("cannot set distributed device count")
                if centered == 0:
                    _splitk_launch(ctx, many, x, p_many, m, k)
                else:
                    _splitk_launch_centered(ctx, many, x, mu, p_many, m, k)
                ctx.synchronize()
                var a = download_f32(ctx, one, cells)
                var b = download_f32(ctx, many, cells)
                var pa = download_f32(ctx, p_one, pcells)
                var pb = download_f32(ctx, p_many, pcells)
                for i in range(cells):
                    if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
                        raise Error("Gram result differs at " + String(m) + "/" + String(k) + "/" + String(centered) + "/" + String(i))
                for i in range(pcells):
                    if bitcast[DType.uint32](pa[i]) != bitcast[DType.uint32](pb[i]):
                        raise Error("Gram partial differs at " + String(m) + "/" + String(k) + "/" + String(centered) + "/" + String(i))
                print("PASS Gram partials and output", m, k, centered)
            _ = x^
            _ = mu^
            _ = one^
            _ = many^
            _ = p_one^
            _ = p_many^
    ctx.synchronize()
