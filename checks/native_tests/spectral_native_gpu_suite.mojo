# SPDX-License-Identifier: Apache-2.0
"""One graph at a time on the physical GPU; existing kernels remain parallel.

This numerical schedule does not resolve or relabel the separately reproduced
shared-DeviceContext concurrent-submission failure. COO and dense provider
routes both retain their original graph/solver semantics.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.spectral_native_api import SpectralDriver
from core.spectral_native_device import SpectralDeviceKernels
from core.spectral_native_provider import spectral_native_v1
from checks.native_tests.spectral_native_cases import SPECTRAL_NATIVE_COMMON_FIXTURES, SPECTRAL_NATIVE_DENSE_FIXTURES, fixture, require

comptime SPECTRAL_NATIVE_GPU_FIXTURES = SPECTRAL_NATIVE_COMMON_FIXTURES + SPECTRAL_NATIVE_DENSE_FIXTURES


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var driver = SpectralDriver(spectral_native_v1[SpectralDeviceKernels])
    var failed = 0
    for unit in range(SPECTRAL_NATIVE_GPU_FIXTURES):
        try:
            fixture(driver, unit)
        except error:
            failed += 1
            print("FAIL spectral_native_gpu fixture", unit, String(error))
    require(failed == 0, "spectral_native_gpu failed " + String(failed) + " fixtures")
    print("PASS spectral_native_gpu:", SPECTRAL_NATIVE_GPU_FIXTURES, "original ordered fit/embedding graphs")
