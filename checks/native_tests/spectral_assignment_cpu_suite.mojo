# SPDX-License-Identifier: Apache-2.0
"""Independent original discretize/cluster-QR graphs run across host cores."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.spectral_native_api import SpectralDriver
from core.spectral_native_provider import spectral_native_v1
from core.spectral_native_host import SpectralHostKernels
from core.spectral_assignment_api import SpectralAssignmentDriver, spectral_assign_native_v1
from core.spectral_assignment_host import SpectralAssignmentHost
from checks.native_tests.spectral_assignment_cases import SPECTRAL_ASSIGNMENT_FIXTURES, fixture, require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failures = List[Int](length=SPECTRAL_ASSIGNMENT_FIXTURES,fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            var fitter = SpectralDriver(spectral_native_v1[SpectralHostKernels])
            var assignment = SpectralAssignmentDriver(spectral_assign_native_v1[SpectralAssignmentHost])
            fixture(fitter,assignment,unit)
        except error:
            fp.unsafe_store(unit,1)
            print("FAIL spectral_assignment_cpu fixture",unit,String(error))
    host_parallelize(task,SPECTRAL_ASSIGNMENT_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0,"spectral_assignment_cpu failed " + String(failed) + " fixtures")
    print("PASS spectral_assignment_cpu:",SPECTRAL_ASSIGNMENT_FIXTURES,"original alternative-assignment graphs")
