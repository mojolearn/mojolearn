# SPDX-License-Identifier: Apache-2.0
"""Batched scans: complete host-oracle equality, reuse and wait/allocation counts."""
from std.memory import bitcast
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from core.device_scan import DeviceNonfiniteBatch, SCAN_BLOCKS, SCAN_TPB
from core.device_scan_check import upload, clean_values, host_first_nonfinite
from core.step_phase import step_counts_now


def main() raises:
    comptime assert is_defined["MOJOLEARN_STEP_PHASE_TIMERS"](), "counters required"
    var ctx = DeviceContext()
    var cap = SCAN_BLOCKS * SCAN_TPB
    var lengths: List[Int] = [0, 1, 255, 256, 257, cap - 1, cap, cap + 1, 3 * cap + 7]
    var before = step_counts_now()
    var batch = DeviceNonfiniteBatch(ctx, lengths)
    var after = step_counts_now()
    if after.device_allocs - before.device_allocs != 1 or after.host_allocs - before.host_allocs != 1:
        raise Error("batch must allocate exactly one device and one host scratch buffer")
    if after.syncs != before.syncs:
        raise Error("batch allocation must not wait")
    var checked = 0
    for pattern in range(5):
        var buffers = List[DeviceBuffer[DType.float32]]()
        var expected = List[Int]()
        for slot in range(len(lengths)):
            var n = lengths[slot]
            # The zero-length scan borrows a real one-element buffer; no
            # backend-dependent zero-sized allocation is needed for this test.
            var values = clean_values(max(n, 1))
            if n > 0:
                values[0] = bitcast[DType.float32](UInt32(0x80000000))
                if n > 1:
                    values[1] = bitcast[DType.float32](UInt32(1))
                if pattern > 0:
                    var bits = UInt32(0x7FC01234)
                    if pattern == 2:
                        bits = UInt32(0x7F800000)
                    elif pattern == 3:
                        bits = UInt32(0xFF800000)
                    values[n - 1] = bitcast[DType.float32](bits)
                    if pattern == 4:
                        values[n // 3] = bitcast[DType.float32](UInt32(0xFF800000))
            expected.append(-1 if n == 0 else host_first_nonfinite(values))
            buffers.append(upload(ctx, values))
        before = step_counts_now()
        for slot in range(len(lengths)):
            batch.enqueue(ctx, slot, buffers[slot])
        var got = batch.finish(ctx)
        after = step_counts_now()
        if after.syncs - before.syncs != 1 or after.d2h - before.d2h != 1:
            raise Error("batch must use exactly one wait and one result copy")
        if after.launches - before.launches != 8:
            raise Error("one unchanged kernel per nonempty scan required")
        if after.device_allocs != before.device_allocs or after.host_allocs != before.host_allocs:
            raise Error("batch reuse must not allocate")
        for slot in range(len(lengths)):
            if got[slot] != expected[slot]:
                raise Error("first-index mismatch in batch slot " + String(slot))
            checked += 1
        _ = buffers^
    var missing = DeviceNonfiniteBatch(ctx, [1, 1])
    var source = upload(ctx, [Float32(1)])
    missing.enqueue(ctx, 0, source)
    var refused = False
    try:
        _ = missing.finish(ctx)
    except e:
        refused = String(e) == "nonfinite batch: missing slot"
    if not refused:
        raise Error("incomplete batch was not refused")
    _ = missing^
    _ = source^
    _ = batch^
    _ = ctx^
    print("PASS batched scans", checked, "host-oracle cases; reuse: 0 allocations, 1 wait, 1 copy; incomplete batch refused")
