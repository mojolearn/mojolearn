# SPDX-License-Identifier: Apache-2.0
"""Parallel private checkpoint conversion; optimizer update arithmetic is unchanged."""
from std.memory import bitcast
from std.math import isfinite
from core.classical_cast import classical_f32_bits
from core.classical_pandas_impl import _pd_double, pd_raw_at
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk


def optimizer_state_cast(addr: Int, stride: Int, size: Int, item: Int, kind: Int, native: Bool,
                         dest: MutPointer[UInt32, MutUntrackedOrigin], integer: Bool) raises:
    if size < 0 or size > 9223372036854775807 // 8 or (item != 1 and item != 2 and item != 4 and item != 8):
        raise Error("ValueError: optimizer checkpoint span is out of range")
    if size == 0:
        return
    if addr == 0 or Int(dest) == 0:
        raise Error("ValueError: optimizer checkpoint requires nonnull storage")
    var tasks = host_predict_task_count_work(size, size * 8)
    var chunk = host_predict_chunk(size, tasks)
    var failed = List[Int](length=tasks if integer else 0, fill=-1)
    var reasons = List[Int](length=tasks if integer else 0, fill=0)
    var indices = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(failed.unsafe_ptr()))
    var codes = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(reasons.unsafe_ptr()))
    def _convert(task: Int) {imm addr, imm stride, imm size, imm item, imm kind, imm native,
        imm dest, imm integer, imm chunk, imm indices, imm codes}:
        var first = task * chunk
        var last = min(size, first + chunk)
        for i in range(first, last):  # parallel-work: disjoint private checkpoint scratch spans
            var raw = pd_raw_at(addr, i * stride, item, native)
            if integer:
                var value = _pd_double(raw, kind, item)
                if value != value:
                    indices[task] = i
                    codes[task] = 1
                    break
                if not isfinite(value) or value <= -2147483649.0 or value >= 2147483648.0:
                    indices[task] = i
                    codes[task] = 2
                    break
                dest[i] = bitcast[DType.uint32](Int32(value))
            else:
                dest[i] = classical_f32_bits(raw, kind, item)
    if tasks <= 1:
        _convert(0)
    else:
        host_parallelize(_convert, tasks)
    # The public boundary publishes only successful scratch. On error retain
    # the original earliest-index exception independently of worker scheduling.
    var earliest = size
    var reason = 0
    if integer:
        for task in range(tasks):  # driver-loop: native worker refusal records
            if failed[task] >= 0 and failed[task] < earliest:
                earliest = failed[task]
                reason = reasons[task]
    if reason == 1:
        raise Error("ValueError: cannot convert float NaN to integer")
    if reason == 2:
        raise Error("OverflowError: buf_initialized value does not fit int32")
