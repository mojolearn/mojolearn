# SPDX-License-Identifier: Apache-2.0
"""Ordered numerical GPU coverage; shared-context concurrency remains parked."""
from core.cluster_centroid_fit_api import CentroidFitDriver
from x_cluster.centroid_fit_device import cluster_centroid_fit_device_v1
from checks.native_tests.centroid_fit_native_cpu_suite import fixture, CENTROID_FIT_NATIVE_FIXTURES


def run_checks() raises:
    var failed = 0
    for unit in range(CENTROID_FIT_NATIVE_FIXTURES):  # numerical-only shared device context; no concurrency qualification
        try:
            fixture(CentroidFitDriver(cluster_centroid_fit_device_v1), unit)
        except problem:
            failed += 1
            print("FAIL centroid_fit_native_gpu fixture", unit, String(problem))
    if failed:
        raise Error("centroid fit GPU failed " + String(failed) + " fixtures")
    print("PASS centroid_fit_native_gpu:", CENTROID_FIT_NATIVE_FIXTURES, "ordered actual provider fixtures")
