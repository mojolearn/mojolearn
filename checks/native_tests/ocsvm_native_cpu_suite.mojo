# SPDX-License-Identifier: Apache-2.0
from core.host_parallel import host_parallelize
from core.neighbors_native_driver import XnDriver
from core.ocsvm_resident_api import OcResidentAPI, OcResidentDriver
from x_neighbors.native_descriptor_host import xn_native_v1
from x_neighbors.ocsvm_resident_native_host import OcHostResident
from checks.native_tests.ocsvm_native_cases import fixture, require, OCSVM_NATIVE_CASES


def run_checks() raises:
    var failures = List[Int](length=OCSVM_NATIVE_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def run(unit: Int) {imm flags}:
        try:
            fixture(XnDriver(xn_native_v1), OcResidentDriver(OcResidentAPI[OcHostResident].entry), unit)
        except problem:
            flags[unit] = 1
            print("FAIL ocsvm_native_cpu fixture", unit, String(problem))
    host_parallelize(run, OCSVM_NATIVE_CASES)
    var failed = 0
    for flag in failures:  # independent fixture statuses
        failed += flag
    require(failed == 0, "OneClassSVM CPU failed " + String(failed) + " cases")
    print("PASS ocsvm_native_cpu:", OCSVM_NATIVE_CASES)
