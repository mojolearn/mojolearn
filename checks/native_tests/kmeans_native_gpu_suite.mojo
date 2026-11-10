# SPDX-License-Identifier: Apache-2.0
"""Native GPU KMeans fixtures; one shared device context is exercised in order."""
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from core.kmeans_native_api import KMeansNativeDriver
from core.kmeans_native_device import kmeans_device_native_v1
from checks.native_tests.kmeans_kernel_cases import fixture, require

comptime KMEANS_GPU_FIXTURES = 9


def run_checks() raises:
    comptime assert TARGET_COLUMN != COLUMN_CPU
    var failed = 0
    for unit in range(KMEANS_GPU_FIXTURES):
        try:
            fixture(KMeansNativeDriver(kmeans_device_native_v1), unit)
        except error:
            failed += 1
            print("FAIL kmeans_native_gpu fixture", unit, String(error))
    require(failed == 0, "KMeans GPU native fixtures failed " + String(failed))
    print("PASS kmeans_native_gpu:", KMEANS_GPU_FIXTURES, "native fit/query/resident fixtures")
