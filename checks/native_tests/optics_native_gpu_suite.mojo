# SPDX-License-Identifier: Apache-2.0
"""Ordered numerical OPTICS cases; shared-DeviceContext concurrency is parked."""
from core.optics_fit_api import OpticsFitDriver
from x_cluster.optics_fit_device import optics_fit_device_v1
from checks.native_tests.optics_native_cpu_suite import fixture, OPTICS_NATIVE_CASES


def run_checks() raises:
    var failed = 0
    for unit in range(OPTICS_NATIVE_CASES):  # shared context numerical coverage; no concurrency qualification
        try:
            fixture(OpticsFitDriver(optics_fit_device_v1), unit)
        except problem:
            failed += 1
            print("FAIL optics_native_gpu fixture", unit, String(problem))
    if failed:
        raise Error("OPTICS GPU failed " + String(failed) + " fixtures")
    print("PASS optics_native_gpu:", OPTICS_NATIVE_CASES, "ordered actual provider fixtures")
