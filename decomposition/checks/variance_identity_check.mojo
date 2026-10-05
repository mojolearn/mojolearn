# SPDX-License-Identifier: Apache-2.0
"""Untimed device/host variance-fold identity, including the OFF control.

Build in IDENTICAL mode; repeat with MOJOLEARN_IDN_DECOMP_MEAN_LAUNCH_OFF.
Both sides call the shipped variance implementations. Run only on GPU boxes.
"""
from std.memory import bitcast
from max.gpu.host import DeviceContext
from decomposition.estimator import _column_variance
from decomposition.host.pca_oracle import host_column_variance
from decomposition.mean_switch import IDN_DECOMP_MEAN_LAUNCH
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


def check(ctx: DeviceContext, rows: Int, cols: Int, mut digest: UInt64) raises:
    var values = List[Float32](length=rows * cols, fill=0.0)
    for i in range(rows * cols):
        values[i] = Float32((Int64(i) * 2654435761) % 1000003 - 500001) / Float32(977.0) * Float32(1 + (i % cols) % 11)
    var expected = host_column_variance(values, rows, cols)
    var matrix = ctx.enqueue_create_buffer[DType.float32](rows * cols)
    var means = ctx.enqueue_create_buffer[DType.float32](cols)
    var output = ctx.enqueue_create_buffer[DType.float32](cols)
    var actual = List[Float32](length=cols, fill=0.0)
    ctx.enqueue_copy(dst_buf=matrix, src_ptr=values.unsafe_ptr())
    _column_variance(ctx, matrix, means, output, rows, cols)
    ctx.enqueue_copy(dst_ptr=actual.unsafe_ptr(), src_buf=output)
    ctx.synchronize()
    for i in range(cols):
        var got = bitcast[DType.uint32](actual[i])
        var want = bitcast[DType.uint32](expected[i])
        if got != want:
            print("VARIANCE_IDENTITY status=DIFFER rows=", rows, " cols=", cols,
                  " index=", i, " device_bits=", got, " host_bits=", want)
            raise Error("variance mean dispatch differs between device and host")
        digest = (digest ^ UInt64(got)) * UInt64(1099511628211)
    _ = matrix^
    _ = means^
    _ = output^
    ctx.synchronize()


def main() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, "IDENTICAL required"
    var ctx = DeviceContext()
    var digest = UInt64(14695981039346656037)
    check(ctx, 257, 3, digest)
    check(ctx, 900, 7, digest)
    check(ctx, 1500, 33, digest)
    check(ctx, 4096, 220, digest)
    print("VARIANCE_IDENTITY status=PASS cases=4 words=263 mean_launch=",
          IDN_DECOMP_MEAN_LAUNCH, " digest=", digest)
