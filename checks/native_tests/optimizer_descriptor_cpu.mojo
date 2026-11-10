# SPDX-License-Identifier: Apache-2.0
"""Parallel host descriptor contracts; each call has private request/error owners."""
from core.host_parallel import host_parallelize
from training.host.optimizer_native import optimizer_host_v1
from training.optimizer_native_api import OptimizerNativeDriver
from checks.native_tests.optimizer_descriptor_cases import (
    DESCRIPTOR_COMMON_FIXTURES, require, run_descriptor_fixture, host_io_refusals,
)


def main() raises:
    comptime COUNT = DESCRIPTOR_COMMON_FIXTURES + 1
    var failures = List[Int](length=COUNT, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            var driver = OptimizerNativeDriver(optimizer_host_v1, True)
            if fixture_id == DESCRIPTOR_COMMON_FIXTURES:
                host_io_refusals(driver)
            else:
                run_descriptor_fixture(driver, fixture_id)
        except error:
            print("FAIL optimizer_descriptor_cpu fixture", fixture_id, String(error))
            fp.unsafe_store(fixture_id, 1)
    host_parallelize(task, COUNT)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "optimizer_descriptor_cpu failed " + String(failed) + " fixtures")
    print("PASS optimizer_descriptor_cpu:", COUNT, "parallel native ABI/owner fixtures")
