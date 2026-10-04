# SPDX-License-Identifier: Apache-2.0
"""GPU-only callpath transport/adapter gate; run only on an authorized GPU box.

Raw-bit comparisons use unchanged production transforms on the same device.
The printed digest permits NVIDIA/AMD comparison; this does not prove Apple
or host identity, and does not measure performance.
"""
from std.memory import bitcast
from max.gpu.host import DeviceContext
from core.identical_callpath import IdenticalCallSession
from preprocessing.minmax import minmax_transform
from preprocessing.standard import standard_transform
from experiments.identical_callpath.minmax import ResidentIdenticalMinMax
from experiments.identical_callpath.standard import ResidentIdenticalStandard


def baseline(ctx: DeviceContext, values: List[Float32], scales: List[Float32], offsets: List[Float32],
             rows: Int, cols: Int, inverse: Int, flag_a: Int, flag_b: Int,
             standard: Bool) raises -> List[Float32]:
    var x = ctx.enqueue_create_buffer[DType.float32](rows * cols)
    var scale = ctx.enqueue_create_buffer[DType.float32](cols)
    var offset = ctx.enqueue_create_buffer[DType.float32](cols)
    ctx.enqueue_copy(dst_buf=x, src_ptr=values.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=scale, src_ptr=scales.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=offset, src_ptr=offsets.unsafe_ptr())
    var result = List[Float32]()
    if standard:
        result = standard_transform(ctx, x, offset, scale, rows, cols,
                                    inverse, flag_a, flag_b)
    else:
        result = minmax_transform(ctx, x, scale, offset, rows, cols,
                                  inverse, flag_a, -1.0, 1.0)
    ctx.synchronize()
    _ = x^
    _ = scale^
    _ = offset^
    ctx.synchronize()
    return result^


def compare(actual: List[Float32], expected: List[Float32], mut digest: UInt64) raises:
    if len(actual) != len(expected):
        raise Error("identity result shape mismatch")
    for i in range(len(actual)):
        var a = bitcast[DType.uint32](actual[i])
        var b = bitcast[DType.uint32](expected[i])
        if a != b:
            print("CALLPATH_DIFFER index=", i, " actual=", a, " expected=", b)
            raise Error("callpath changed result bits")
        digest = (digest ^ UInt64(a)) * UInt64(1099511628211)


def main() raises:
    var baseline_ctx = DeviceContext()
    var rows = 257
    var cols = 3
    var scales: List[Float32] = [1.0, 0.5, 2.0]
    var offsets: List[Float32] = [0.0, -1.0, 1.0]
    var inputs = List[List[Float32]]()
    var results = List[List[Float32]]()
    for slot in range(2):
        var values = List[Float32](length=rows * cols, fill=0.0)
        for i in range(rows * cols):
            values[i] = Float32((i + slot * 11) % 37 - 18) * 0.125
        values[0] = bitcast[DType.float32](UInt32(0x80000000))
        values[1] = bitcast[DType.float32](UInt32(1))
        values[2] = bitcast[DType.float32](UInt32(0x80000001))
        inputs.append(values^)
        results.append(List[Float32](length=rows * cols, fill=0.0))
    var minmax = ResidentIdenticalMinMax(rows, cols, 2, scales, offsets)
    var standard = ResidentIdenticalStandard(rows, cols, 2, offsets, scales)
    var digest = UInt64(14695981039346656037)
    var comparisons = 0
    # The same persistent slots are repeatedly dirtied across operations/modes.
    for grouped in range(2):
        for inverse in range(2):
            for first in range(2):
                minmax.transform_batch_into(inputs, results, inverse, first,
                                            -1.0, 1.0, Bool(grouped))
                for slot in range(2):
                    var expected = baseline(baseline_ctx, inputs[slot], scales, offsets,
                                            rows, cols, inverse, first, 0, False)
                    compare(results[slot], expected, digest)
                    comparisons += 1
                for second in range(2):
                    standard.transform_batch_into(inputs, results, inverse,
                                                  first, second, Bool(grouped))
                    for slot in range(2):
                        var expected = baseline(baseline_ctx, inputs[slot], scales, offsets,
                                                rows, cols, inverse, first, second, True)
                        compare(results[slot], expected, digest)
                        comparisons += 1
    if comparisons != 48:
        raise Error("incomplete scaler gate coverage")

    var session = IdenticalCallSession()
    var slot = session.reserve_u32(2)
    var payload: List[UInt32] = [0x80000000, 0xffffffff]
    var output = List[UInt32](length=2, fill=0)
    for turn in range(2):
        payload[0] += UInt32(turn)
        session.stage_u32(slot, payload)
        session.begin()
        session.upload_u32(slot)
        session.readback_u32(slot)
        session.finish()
        session.collect_u32(slot, output)
        if output[0] != payload[0] or output[1] != payload[1]:
            raise Error("typed bank stale or altered data")
    session.begin()
    var refused_stale = False
    try:
        session.collect_u32(slot, output)
    except:
        refused_stale = True
    session.finish()
    if not refused_stale:
        raise Error("active session exposed stale readback")
    print("CALLPATH_GATE status=PASS comparisons=", comparisons,
          " typed_roundtrips=2 stale_readback_refused=1 digest=", digest)
