# SPDX-License-Identifier: Apache-2.0
"""One complete assignment graph at a time on the physical GPU.

The tuned fit/assignment kernels remain parallel. This schedule does not fix
the separately recorded shared-context concurrent-submission failure.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.spectral_native_api import SpectralDriver
from core.spectral_native_provider import spectral_native_v1
from core.spectral_native_device import SpectralDeviceKernels
from core.spectral_assignment_api import SpectralAssignmentDriver, spectral_assign_native_v1
from core.spectral_assignment_device import SpectralAssignmentDevice
from checks.native_tests.spectral_assignment_cases import SPECTRAL_ASSIGNMENT_FIXTURES, fixture, require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var fitter = SpectralDriver(spectral_native_v1[SpectralDeviceKernels])
    var assignment = SpectralAssignmentDriver(spectral_assign_native_v1[SpectralAssignmentDevice])
    var failed = 0
    for unit in range(SPECTRAL_ASSIGNMENT_FIXTURES):
        try:
            fixture(fitter,assignment,unit)
        except error:
            failed += 1
            print("FAIL spectral_assignment_gpu fixture",unit,String(error))
    require(failed == 0,"spectral_assignment_gpu failed " + String(failed) + " fixtures")
    print("PASS spectral_assignment_gpu:",SPECTRAL_ASSIGNMENT_FIXTURES,"original ordered alternative-assignment graphs")
