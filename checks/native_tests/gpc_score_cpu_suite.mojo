# SPDX-License-Identifier: Apache-2.0
"""Independent GPC score cases run across the full shared host pool."""
from core.host_parallel import host_parallelize
from core.gp_native_api import GPDriver
from core.gp_native_provider import gp_native_v1
from core.gp_native_host import GPHostKernels
from x_metrics.descriptor_host import metrics_program_host_v1
from checks.native_tests.gp_state_suite import require
from checks.native_tests.gpc_score_cases import fixture as numerical, empty_case, GPC_SCORE_NATIVE_CASES

comptime GPC_SCORE_CPU_CASES = 1 + GPC_SCORE_NATIVE_CASES


def run_checks() raises:
    var failures = List[Int](length=GPC_SCORE_CPU_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def run(unit: Int) {imm flags}:
        try:
            if unit == 0:
                empty_case()
            else:
                numerical(GPDriver(gp_native_v1[GPHostKernels]), metrics_program_host_v1,
                    unit - 1)
        except problem:
            flags[unit] = 1
            print("FAIL gpc_score_cpu fixture", unit, String(problem))
    host_parallelize(run, GPC_SCORE_CPU_CASES)
    var failed = 0
    for flag in failures:  # metadata: independent fixture statuses
        failed += flag
    require(failed == 0, "GPC score CPU failures: " + String(failed))
    print("PASS gpc_score_cpu:", GPC_SCORE_CPU_CASES, "fixtures")
    _ = failures
