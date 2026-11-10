# SPDX-License-Identifier: Apache-2.0
"""Native CPU device-capability refusals and pure storage/descriptor contracts."""
from core.host_parallel import host_parallelize
from training.host.optimizer_native import optimizer_host_v1
from training.optimizer_native_api import OptimizerNativeDriver
from checks.native_tests.optimizer_device_cases import DEVICE_COMMON, DEVICE_UNAVAILABLE, RawWords, common
from checks.native_tests.optimizer_descriptor_cases import require


def unsupported(driver: OptimizerNativeDriver, fixture_id: Int) raises:
    var source = RawWords(4, True)
    var refused = False
    try:
        if fixture_id == 0:
            _ = driver.allocate_device(4)
        elif fixture_id == 1:
            _ = driver.upload_device(0, source.address(), 4)
        else:
            _ = driver.download_device(0, source.address(), 4)
    except error:
        refused = True
        require(String(error) == DEVICE_UNAVAILABLE, "CPU device-unavailable refusal differs")
    require(refused, "CPU accepted an unsupported training device operation")
    source.expect(4)
    _ = source


def run_checks() raises:
    comptime COUNT = DEVICE_COMMON + 3
    var failures = List[Int](length=COUNT, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            var driver = OptimizerNativeDriver(optimizer_host_v1, True)
            if fixture_id < DEVICE_COMMON:
                common(driver, fixture_id)
            else:
                unsupported(driver, fixture_id - DEVICE_COMMON)
        except error:
            fp.unsafe_store(fixture_id, 1)
            print("FAIL optimizer_device_cpu fixture", fixture_id, String(error))
    host_parallelize(task, COUNT)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "optimizer_device_cpu failed " + String(failed) + " fixtures")
    print("PASS optimizer_device_cpu:", COUNT, "parallel native descriptor/storage fixtures")
