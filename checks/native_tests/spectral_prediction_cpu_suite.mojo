# SPDX-License-Identifier: Apache-2.0
"""Independent full Spectral prediction graphs run across host cores."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.spectral_native_api import SpectralDriver
from core.spectral_native_host import SpectralHostKernels
from core.spectral_native_provider import spectral_native_v1
from checks.native_tests.spectral_prediction_cases import SPECTRAL_PREDICTION_FIXTURES, fixture, require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failures = List[Int](length=SPECTRAL_PREDICTION_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            var driver = SpectralDriver(spectral_native_v1[SpectralHostKernels])
            fixture[False](driver,unit)
        except error:
            fp.unsafe_store(unit,1)
            print("FAIL spectral_prediction_cpu fixture",unit,String(error))
    host_parallelize(task,SPECTRAL_PREDICTION_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0,"spectral_prediction_cpu failed " + String(failed) + " fixtures")
    print("PASS spectral_prediction_cpu:",SPECTRAL_PREDICTION_FIXTURES,"original fit/state/query graphs")
