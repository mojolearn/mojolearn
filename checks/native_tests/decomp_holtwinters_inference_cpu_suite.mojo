# SPDX-License-Identifier: Apache-2.0
"""Holt-Winters fitted level and in-sample prediction, on the host fit body.

The two seasonal fixtures are independent and run across host cores; each
fixture's per-cell relations use the host pool as well. The fit is the host
reference body only (the GPU fit is checked against it in hw_check), so
there is no GPU column.
"""
from core.host_parallel import host_parallelize
from checks.native_tests.decomp_holtwinters_inference_cases import HW_INFERENCE_FIXTURES, fixture, require


def run_checks() raises:
    var failures = List[Int](length=HW_INFERENCE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL decomp_holtwinters_inference_cpu fixture", unit, String(error))
    host_parallelize(task, HW_INFERENCE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "decomp_holtwinters_inference_cpu failed " + String(failed) + " fixtures")
    print("PASS decomp_holtwinters_inference_cpu:", HW_INFERENCE_FIXTURES, "fitted level/in-sample fixtures")
