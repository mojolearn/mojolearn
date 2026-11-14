# SPDX-License-Identifier: Apache-2.0
"""The device widening: one launch over all 65,536 halves (one element per
thread), then the comparison chunk by chunk across host cores."""
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from decomposition.linalg_door_impl import from_f16_device
from checks.native_tests.byte_lm_models_loader_f16_cases import (
    F16_HALVES, F16_CHUNKS, all_halves, check_chunk, require,
)


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var halves = all_halves()
    var widened = List[Float32](length=F16_HALVES, fill=Float32(0))
    var wp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(widened.unsafe_ptr()))
    var ctx = DeviceContext()
    from_f16_device(ctx, wp, MutPointer[UInt16, MutUntrackedOrigin](unsafe_from_address=Int(halves.unsafe_ptr())),
                    F16_HALVES)
    _ = halves  # read through its address by the synchronous call above
    var failures = List[Int](length=F16_CHUNKS, fill=-1)
    var fp = failures.unsafe_ptr()
    def task(chunk: Int) {imm fp, imm wp}:
        fp.unsafe_store(chunk, check_chunk(wp, chunk))
    host_parallelize(task, F16_CHUNKS)
    _ = widened
    var failed = 0
    for chunk in range(F16_CHUNKS):
        if failures[chunk] >= 0:
            failed += 1
            print("FAIL byte_lm_models_loader_f16_gpu: half", hex(failures[chunk]), "widened wrong")
    require(failed == 0, "byte_lm_models_loader_f16_gpu failed " + String(failed) + " chunks")
    print("PASS byte_lm_models_loader_f16_gpu:", F16_HALVES, "halves against the value oracle")
