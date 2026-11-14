# SPDX-License-Identifier: Apache-2.0
"""The host widening, chunk by chunk across host cores: each task widens
its own 1,024 halves through `from_f16_host_run` and checks them."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from decomposition.linalg_host_impl import from_f16_host_run
from checks.native_tests.byte_lm_models_loader_f16_cases import (
    F16_HALVES, F16_CHUNKS, F16_CHUNK, all_halves, check_chunk, require,
)


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var halves = all_halves()
    var widened = List[Float32](length=F16_HALVES, fill=Float32(0))
    var failures = List[Int](length=F16_CHUNKS, fill=-1)
    var fp = failures.unsafe_ptr()
    var hp = Int(halves.unsafe_ptr())
    var wp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(widened.unsafe_ptr()))
    def task(chunk: Int) {imm fp, imm hp, imm wp}:
        try:
            from_f16_host_run(wp + chunk * F16_CHUNK, hp + 2 * chunk * F16_CHUNK, F16_CHUNK)
            fp.unsafe_store(chunk, check_chunk(wp, chunk))
        except error:
            fp.unsafe_store(chunk, chunk * F16_CHUNK)
            print("FAIL byte_lm_models_loader_f16_cpu chunk", chunk, String(error))
    host_parallelize(task, F16_CHUNKS)
    _ = halves  # read through its address by every task above
    _ = widened
    var failed = 0
    for chunk in range(F16_CHUNKS):
        if failures[chunk] >= 0:
            failed += 1
            print("FAIL byte_lm_models_loader_f16_cpu: half", hex(failures[chunk]), "widened wrong")
    require(failed == 0, "byte_lm_models_loader_f16_cpu failed " + String(failed) + " chunks")
    print("PASS byte_lm_models_loader_f16_cpu:", F16_HALVES, "halves against the value oracle, parallel")
