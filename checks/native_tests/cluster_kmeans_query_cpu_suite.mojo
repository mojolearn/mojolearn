# SPDX-License-Identifier: Apache-2.0
"""KMeans predict/transform CPU fixtures; independent fixtures run across host workers."""
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from core.host_parallel import host_parallelize
from core.kmeans_native_api import KMeansNativeDriver
from core.kmeans_native_host import kmeans_host_native_v1
from checks.native_tests.cluster_kmeans_query_cases import KMEANS_QUERY_FIXTURES, fixture, require


def run_checks() raises:
    comptime assert TARGET_COLUMN == COLUMN_CPU
    var failures = List[Int](length=KMEANS_QUERY_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(KMeansNativeDriver(kmeans_host_native_v1), unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL cluster_kmeans_query_cpu fixture", unit, String(error))
    host_parallelize(task, KMEANS_QUERY_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "cluster_kmeans_query_cpu failed " + String(failed) + " fixtures")
    print("PASS cluster_kmeans_query_cpu:", KMEANS_QUERY_FIXTURES, "predict/transform fixtures")
