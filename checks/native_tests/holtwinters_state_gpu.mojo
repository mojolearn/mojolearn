# SPDX-License-Identifier: Apache-2.0
"""Direct GPU per-call/resident fitted-state queries and owner lifetime.

Resident-owner submissions and changing queries are ordered for the existing
GPU context. Forecast kernels, native data preparation and oracle cells run
in parallel. No concurrent GPU-query claim is made here.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr
from holtwinters.estimator import holtwinters_predict_ptr, holtwinters_forecast_ptr
from holtwinters.resident_state import holtwinters_resident_prepare, holtwinters_resident_release, holtwinters_resident_count, holtwinters_query_resident
from checks.native_tests.holtwinters_state_cases import (
    HW_STATE_OWNERS, HW_STATE_QUERIES, HW_SENTINEL, require, hw_shape, hw_span,
    hw_components, hw_output, hw_check,
)


def fixture(owner_id: Int) raises:
    var shape = hw_shape(owner_id)
    var n = shape[0]
    var batch = shape[1]
    var frequency = shape[2]
    var seasonal = String("add" if shape[3] else "mul")
    var packed = hw_components(owner_id)
    var preserved = packed.copy()
    var before = holtwinters_resident_count()
    var handle = holtwinters_resident_prepare(f32_ptr(Int(packed.unsafe_ptr())), n, batch, frequency)
    try:
        require(holtwinters_resident_count() == before + 1, "resident component count after prepare")
        # A resident model owns an uploaded snapshot, not the caller's span.
        var pp = packed.unsafe_ptr()
        def mutate(index: Int) {imm pp}:
            pp.unsafe_store(index, Float32(-99))
        host_parallelize(mutate, len(packed))
        for query_id in range(HW_STATE_QUERIES):  # successive queries on one retained native owner
            var span = hw_span(n, frequency, query_id)
            var start = span[0]
            var end = span[1]
            var actual = hw_output(owner_id, query_id)
            var pointer_result = hw_output(owner_id, query_id)
            var count = (end - start) * batch
            require(holtwinters_query_resident(handle, f32_ptr(Int(actual.unsafe_ptr()) + 4), seasonal, start, end) == count, "resident query extent")
            hw_check(actual, owner_id, query_id)
            var inside = max(0, min(end, n) - start)
            if inside:
                require(holtwinters_predict_ptr(f32_ptr(Int(preserved.unsafe_ptr())), f32_ptr(Int(pointer_result.unsafe_ptr()) + 4), n, batch, frequency, seasonal, start, min(end, n)) == inside * batch, "per-call inside extent")
            if end > n:
                var future = List[Float32](length=(end - n) * batch, fill=HW_SENTINEL)
                require(holtwinters_forecast_ptr(f32_ptr(Int(preserved.unsafe_ptr())), f32_ptr(Int(future.unsafe_ptr())), n, batch, frequency, seasonal, end - n) == len(future), "per-call forecast extent")
                var src = future.unsafe_ptr()
                var dst = pointer_result.unsafe_ptr()
                var skip = max(0, start - n) * batch
                var offset = inside * batch + 1
                def copy_future(index: Int) {imm src, imm dst, imm skip, imm offset}:
                    dst.unsafe_store(offset + index, src.unsafe_load(skip + index))
                host_parallelize(copy_future, count - inside * batch)
                _ = future
            hw_check(pointer_result, owner_id, query_id)
            for index in range(count + 2):
                require(bitcast[DType.uint32](actual[index]) == bitcast[DType.uint32](pointer_result[index]), "resident/per-call prediction words differ")
            require(holtwinters_resident_count() == before + 1, "new query duplicated resident owner")
    except error:
        holtwinters_resident_release(handle)
        raise error
    holtwinters_resident_release(handle)
    holtwinters_resident_release(handle)
    require(holtwinters_resident_count() == before, "resident owner release count")
    var refused_output = List[Float32](length=batch, fill=HW_SENTINEL)
    var refused = False
    try:
        _ = holtwinters_query_resident(handle, f32_ptr(Int(refused_output.unsafe_ptr())), seasonal, n, n + 1)
    except:
        refused = True
    require(refused, "released resident handle accepted")
    for value in refused_output:
        require(value == HW_SENTINEL, "stale handle wrote output")
    var original = hw_components(owner_id)
    for index in range(len(preserved)):
        require(preserved[index] == original[index], "per-call query mutated source components")
    _ = packed
    _ = preserved
    _ = refused_output


def main() raises:
    var failed = 0
    for owner_id in range(HW_STATE_OWNERS):
        try:
            fixture(owner_id)
        except error:
            print("FAIL holtwinters_state_gpu owner", owner_id, String(error))
            failed += 1
    require(failed == 0, "holtwinters_state_gpu failed " + String(failed) + " owners")
    print("PASS holtwinters_state_gpu:", HW_STATE_OWNERS, "owners x", HW_STATE_QUERIES, "per-call/resident query intervals; shared-context ordered")
