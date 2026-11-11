# SPDX-License-Identifier: Apache-2.0
"""Actual device numerical fixtures, ordered within one GPU process.

The shared DeviceContext concurrent-submit reproducer remains parked. This
wrapper exercises existing numerical routes only; product operations remain
unchanged and parallel. CPU fixture units use their parallel suite runner.
"""
from core.kmeans_variants_api import KMeansVariantsDriver
from x_cluster.variants_device import kmeans_variants_device_v1
from checks.native_tests.kmeans_variants_native_cpu_suite import fixture, KMEANS_VARIANTS_NATIVE_CPU_FIXTURES


def main() raises:
    var driver = KMeansVariantsDriver(kmeans_variants_device_v1)
    var failed = 0
    for unit in range(KMEANS_VARIANTS_NATIVE_CPU_FIXTURES):  # one shared device context; no concurrency qualification
        try:
            fixture(driver, unit)
        except problem:
            failed += 1
            print("FAIL kmeans_variants_native_gpu fixture", unit, String(problem))
    if failed:
        raise Error("native KMeans variants GPU failed " + String(failed) + " fixtures")
    print("PASS kmeans_variants_native_gpu:", KMEANS_VARIANTS_NATIVE_CPU_FIXTURES, "ordered actual device numerical fixtures")
