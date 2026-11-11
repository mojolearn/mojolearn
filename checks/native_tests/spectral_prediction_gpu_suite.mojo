# SPDX-License-Identifier: Apache-2.0
"""One query graph at a time on the physical GPU; kernels stay parallel.

Selected single-row queries use the same ordered device schedule. The recorded
shared-DeviceContext concurrent-submission failure remains separate and open.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.spectral_native_api import SpectralDriver
from core.spectral_native_device import SpectralDeviceKernels
from core.spectral_native_provider import spectral_native_v1
from checks.native_tests.spectral_prediction_cases import SPECTRAL_PREDICTION_FIXTURES, fixture, require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var driver = SpectralDriver(spectral_native_v1[SpectralDeviceKernels])
    var failed = 0
    for unit in range(SPECTRAL_PREDICTION_FIXTURES):
        try:
            fixture[True](driver,unit)
        except error:
            failed += 1
            print("FAIL spectral_prediction_gpu fixture",unit,String(error))
    require(failed == 0,"spectral_prediction_gpu failed " + String(failed) + " fixtures")
    print("PASS spectral_prediction_gpu:",SPECTRAL_PREDICTION_FIXTURES,"original ordered fit/state/query graphs")
