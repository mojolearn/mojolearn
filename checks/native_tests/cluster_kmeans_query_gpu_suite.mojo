# SPDX-License-Identifier: Apache-2.0
"""KMeans predict/transform GPU fixtures.

The fixtures run in order on the one physical device (the shared device
context of the KMeans provider); each fit and query keeps its own kernel
parallelism.
"""
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from core.kmeans_native_api import KMeansNativeDriver
from core.kmeans_native_device import kmeans_device_native_v1
from checks.native_tests.cluster_kmeans_query_cases import KMEANS_QUERY_FIXTURES, fixture, require


def run_checks() raises:
    comptime assert TARGET_COLUMN != COLUMN_CPU
    var failed = 0
    for unit in range(KMEANS_QUERY_FIXTURES):
        try:
            fixture(KMeansNativeDriver(kmeans_device_native_v1), unit)
        except error:
            failed += 1
            print("FAIL cluster_kmeans_query_gpu fixture", unit, String(error))
    require(failed == 0, "cluster_kmeans_query_gpu failed " + String(failed) + " fixtures")
    print("PASS cluster_kmeans_query_gpu:", KMEANS_QUERY_FIXTURES, "predict/transform fixtures")
