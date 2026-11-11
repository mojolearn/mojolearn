# SPDX-License-Identifier: Apache-2.0
"""Independent normalized GP graphs execute across host cores."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.gp_native_api import GPDriver
from core.gp_native_provider import gp_native_v1
from core.gp_native_host import GPHostKernels
from core.scaler_native_api import ScalerDriver, PrepProgramDriver, scaler_native_v1, prep_program_v1
from core.scaler_native_host import ScalerHostKernels
from core.prep_program_native_host import PrepProgramHostKernels
from checks.native_tests.gp_normalization_cases import GP_NORMALIZATION_FIXTURES, normalization_fixture
from checks.native_tests.gp_state_suite import require

comptime GP_NORMALIZATION_CPU_FIXTURES = GP_NORMALIZATION_FIXTURES


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failures = List[Int](length=GP_NORMALIZATION_CPU_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            normalization_fixture(GPDriver(gp_native_v1[GPHostKernels]),
                ScalerDriver(scaler_native_v1[ScalerHostKernels]),
                PrepProgramDriver(prep_program_v1[PrepProgramHostKernels]),
                unit, False)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL gp_normalization_cpu fixture", unit, String(error))
    host_parallelize(task, GP_NORMALIZATION_CPU_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "gp_normalization_cpu failed " + String(failed) + " fixtures")
    print("PASS gp_normalization_cpu:", GP_NORMALIZATION_CPU_FIXTURES, "actual normalization/GP graphs")
