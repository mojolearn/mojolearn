# SPDX-License-Identifier: Apache-2.0
"""Device refusal vs original host refusal, including exact error text."""
from std.memory import bitcast
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from mamba.impl.modules.mamba3_refusal import (
    M3_REFUSAL_NONE, M3_DEVICE_REFUSAL,
    m3_first_nonfinite_code, m3_refuse_nonfinite_named,
)
from mamba.impl.modules.mamba3_transfer import m3_upload, m3_download
from mamba.impl.modeling.modeling_mamba import _refuse_nonfinite_named


def run_case(
    ctx: DeviceContext, values: List[Float32], n: Int, expected: Int64,
) raises:
    var dev = m3_upload(ctx, values)
    var code = m3_first_nonfinite_code(ctx, dev, n)
    if code != expected:
        raise Error("first bad code mismatch: " + String(code) + " vs " + String(expected))
    var prefix = List[Float32]()
    for i in range(n):
        prefix.append(values[i])
    var host_error = String("")
    var device_error = String("")
    try:
        _refuse_nonfinite_named("state.buf_qrot", prefix)
    except e:
        host_error = String(e)
    try:
        m3_refuse_nonfinite_named(ctx, "state.buf_qrot", dev, n)
    except e:
        device_error = String(e)
    if host_error != device_error:
        raise Error("refusal text changed: " + host_error + " vs " + device_error)
    var after = m3_download(ctx, dev, len(values))
    for i in range(len(values)):
        if bitcast[DType.uint32](after[i]) != bitcast[DType.uint32](values[i]):
            raise Error("refusal mutated operand bits")
    _ = dev^


def main() raises:
    if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL or not M3_DEVICE_REFUSAL:
        raise Error("mamba3_refusal_check requires the IDENTICAL device refusal")
    var ctx = DeviceContext()
    var finite: List[UInt32] = [0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x3F800000, 0xBF800000, 0x7F7FFFFF, 0xFF7FFFFF]
    var lengths: List[Int] = [0, 1, 17, 255, 256, 257, 1003, 32769, 70001]
    var total = 0
    for n in lengths:
        # One unscanned bad suffix proves the extent is honored.
        var values = List[Float32]()
        for i in range(n):
            values.append(bitcast[DType.float32](finite[i % len(finite)]))
        values.append(bitcast[DType.float32](UInt32(0x7FC00042)))
        run_case(ctx, values, n, M3_REFUSAL_NONE)
        total += 1
        if n == 0:
            continue
        var bad_bits: List[UInt32] = [0x7F800000, 0xFF800000, 0x7FC00001, 0xFFC12345, 0x7F800001, 0xFF800001]
        for bits in bad_bits:
            var positions: List[Int] = [0, n // 2, n - 1]
            for where in positions:
                var bad = values.copy()
                # Put another kind at the tail to test index precedence.
                bad[n - 1] = bitcast[DType.float32](UInt32(0x7FC01234))
                bad[where] = bitcast[DType.float32](bits)
                var kind = Int64(0)
                if (bits & UInt32(0x7FFFFFFF)) == UInt32(0x7F800000):
                    kind = 1
                run_case(ctx, bad, n, Int64(where) * 2 + kind)
                total += 1
    # Named buffers keep caller order, even if the second has index zero.
    var first: List[Float32] = [0.0, 1.0, bitcast[DType.float32](UInt32(0xFF800000))]
    var second: List[Float32] = [bitcast[DType.float32](UInt32(0x7FC00001))]
    var da = m3_upload(ctx, first)
    var db = m3_upload(ctx, second)
    var got = String("")
    try:
        m3_refuse_nonfinite_named(ctx, "first", da, 3)
        m3_refuse_nonfinite_named(ctx, "second", db, 1)
    except e:
        got = String(e)
    if got != "mamba: infinity in first at flat index 2 REFUSED (row 39)":
        raise Error("named buffer order or exact infinity text changed: " + got)
    _ = da^
    _ = db^
    print("Mamba3 refusal PASS:", total, "cases, exact errors, operand bits unchanged, named order")
