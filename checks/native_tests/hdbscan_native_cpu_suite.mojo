# SPDX-License-Identifier: Apache-2.0
from core.host_parallel import host_parallelize
from core.hdbscan_fit_api import HdbscanFitDriver
from core.hdbscan_query_api import HdbscanQueryDriver
from hdbscan.native_fit_host import hdbscan_fit_host_v1
from hdbscan.native_query_host import hdbscan_query_host_v1
from checks.native_tests.hdbscan_native_cases import fixture, require, HDBSCAN_NATIVE_CASES


def run_checks() raises:
    var failures = List[Int](length=HDBSCAN_NATIVE_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def run(unit: Int) {imm flags}:
        try:
            fixture(HdbscanFitDriver(hdbscan_fit_host_v1), HdbscanQueryDriver(hdbscan_query_host_v1), unit)
        except problem:
            flags[unit] = 1
            print("FAIL hdbscan_native_cpu fixture", unit, String(problem))
    host_parallelize(run, HDBSCAN_NATIVE_CASES)
    var failed = 0
    for value in failures:  # metadata: independent fixture statuses
        failed += value
    require(failed == 0, "HDBSCAN CPU failed " + String(failed) + " fixtures")
    print("PASS hdbscan_native_cpu:", HDBSCAN_NATIVE_CASES, "actual provider fixtures")
