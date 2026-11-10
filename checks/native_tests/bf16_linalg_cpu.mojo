# SPDX-License-Identifier: Apache-2.0
"""Independent host BF16 GEMM fixtures, using typed buffers and native calls."""
from core.host_parallel import host_parallelize
from checks.native_tests.bf16_linalg_cases import BF16_LINALG_FIXTURES, require, fixture_operands, host_results


def main() raises:
    var failures = List[Int](length=BF16_LINALG_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            var operands = fixture_operands(fixture_id)
            _ = host_results(operands[0], operands[1], operands[2], fixture_id)
        except error:
            print("FAIL bf16_linalg_cpu fixture", fixture_id, String(error))
            fp.unsafe_store(fixture_id, 1)
    host_parallelize(task, BF16_LINALG_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "bf16_linalg_cpu failed " + String(failed) + " fixtures")
    print("PASS bf16_linalg_cpu:", BF16_LINALG_FIXTURES, "native mixed/both-BF16 GEMM fixtures, parallel")
