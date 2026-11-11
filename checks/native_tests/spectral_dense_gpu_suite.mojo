# SPDX-License-Identifier: Apache-2.0
"""One affinity graph at a time on the physical GPU; kernels stay parallel.

The separately recorded shared-DeviceContext concurrency failure stays open.
COO scans shared by both public images are host metadata operations and are
covered once by spectral_dense_cpu, not counted again as GPU execution.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.spectral_preparation_api import SpectralPreparationDriver, SpectralKNNRequest, spectral_preparation_native_v1
from core.spectral_preparation_device import SpectralPreparationDevice
from checks.native_tests.spectral_dense_cases import SPECTRAL_AFFINITY_FIXTURES, SpectralDenseCalls, affinity_case, require


struct DeviceSpectralDense(SpectralDenseCalls):
    @staticmethod
    def affinity(source: Int, rows: Int, neighbors: Int, dst: Int, status: Int) raises:
        var driver = SpectralPreparationDriver(spectral_preparation_native_v1[SpectralPreparationDevice])
        driver.knn(SpectralKNNRequest(source, 0, 0, 0, dst, status, rows, neighbors, 0, False))


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failed = 0
    for unit in range(SPECTRAL_AFFINITY_FIXTURES):
        try:
            affinity_case[DeviceSpectralDense](unit)
        except error:
            failed += 1
            print("FAIL spectral_dense_gpu fixture", unit, String(error))
    require(failed == 0, "spectral_dense_gpu failed " + String(failed) + " fixtures")
    print("PASS spectral_dense_gpu:", SPECTRAL_AFFINITY_FIXTURES, "original ordered affinity fixtures")
