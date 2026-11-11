# SPDX-License-Identifier: Apache-2.0
"""Independent actual GP host graphs execute across host cores."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.gp_native_api import GPDriver
from core.gp_native_provider import gp_native_v1
from core.gp_native_host import GPHostKernels
from core.gp_native_host_infer import GPHostInferenceKernels
from checks.native_tests.gp_native_cases import GP_NATIVE_CPU_FIXTURES, fixture
from checks.native_tests.gp_native_learning_cases import GP_LEARNING_FIXTURES, learning_fixture
from checks.native_tests.gp_native_classifier_cases import GP_CLASSIFIER_CPU_FIXTURES, classifier_fixture
from checks.native_tests.gp_state_suite import require

comptime GP_NATIVE_CPU_TOTAL = GP_NATIVE_CPU_FIXTURES + GP_LEARNING_FIXTURES + GP_CLASSIFIER_CPU_FIXTURES


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failures = List[Int](length=GP_NATIVE_CPU_TOTAL, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            if unit < GP_NATIVE_CPU_FIXTURES:
                var query = GPDriver(gp_native_v1[GPHostInferenceKernels]) if unit == 6 else GPDriver(gp_native_v1[GPHostKernels])
                fixture(GPDriver(gp_native_v1[GPHostKernels]), query, unit, True)
            elif unit < GP_NATIVE_CPU_FIXTURES + GP_LEARNING_FIXTURES:
                learning_fixture(GPDriver(gp_native_v1[GPHostKernels]), unit - GP_NATIVE_CPU_FIXTURES, True)
            else:
                var at = unit - GP_NATIVE_CPU_FIXTURES - GP_LEARNING_FIXTURES
                var query = GPDriver(gp_native_v1[GPHostInferenceKernels]) if at >= 2 else GPDriver(gp_native_v1[GPHostKernels])
                classifier_fixture(GPDriver(gp_native_v1[GPHostKernels]), query, at, True)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL gp_native_cpu fixture", unit, String(error))
    host_parallelize(task, GP_NATIVE_CPU_TOTAL)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "gp_native_cpu failed " + String(failed) + " fixtures")
    print("PASS gp_native_cpu:", GP_NATIVE_CPU_TOTAL, "actual full-host/inference GP graphs")
