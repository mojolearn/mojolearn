# SPDX-License-Identifier: Apache-2.0
"""Independent theta policy fixtures execute across host cores.

No product operation is executed by the reference generator.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.gp_native_api import GPDriver
from core.gp_native_provider import gp_native_v1
from core.gp_native_host import GPHostKernels
from checks.native_tests.gp_theta_cases import GP_THETA_FIXTURES, theta_fixture
from checks.native_tests.gp_state_suite import require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failures = List[Int](length=GP_THETA_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            theta_fixture(GPDriver(gp_native_v1[GPHostKernels]), unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL gp_theta_cpu fixture", unit, String(error))
    host_parallelize(task, GP_THETA_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "gp_theta_cpu failed " + String(failed) + " fixtures")
    print("PASS gp_theta_cpu:", GP_THETA_FIXTURES, "original precision/FTZ policy fixtures")
