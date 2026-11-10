# SPDX-License-Identifier: Apache-2.0
"""Native KMeans CPU fixtures; independent owners run across host workers."""
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from core.host_parallel import host_parallelize
from core.kmeans_native_api import KMeansNativeDriver
from core.kmeans_native_host import kmeans_host_native_v1
from checks.native_tests.kmeans_kernel_cases import fixture, require

comptime KMEANS_CPU_FIXTURES = 6


def run_checks() raises:
    comptime assert TARGET_COLUMN == COLUMN_CPU
    var failures = List[Int](length=KMEANS_CPU_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(KMeansNativeDriver(kmeans_host_native_v1), unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL kmeans_native_cpu fixture", unit, String(error))
    host_parallelize(task, KMEANS_CPU_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "KMeans CPU native fixtures failed " + String(failed))
    print("PASS kmeans_native_cpu:", KMEANS_CPU_FIXTURES, "native fit/query fixtures")
