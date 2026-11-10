# SPDX-License-Identifier: Apache-2.0
"""Native device allocation/transfer/lease checks; unchanged training kernels."""
from core.host_parallel import host_parallelize
from training.optimizer_native_gpu import optimizer_gpu_v1
from training.optimizer_native_api import OptimizerNativeDriver
from training.optimizer_device_handle import OptimizerDeviceHandle
from checks.native_tests.optimizer_device_cases import (
    DEVICE_COMMON, RawWords, WORD_GUARD, common, transfer_refusal,
)
from checks.native_tests.optimizer_descriptor_cases import require

comptime DEVICE_GPU_EXTRA = 8


def closed_error(handle: Int) -> String:
    return "training device array: handle " + String(handle) + " is not open"


def roundtrip(driver: OptimizerNativeDriver, fixture_id: Int) raises:
    var count = 1 if fixture_id == 0 else (13 if fixture_id == 1 else 4097)
    var source = RawWords(count, True)
    var result = RawWords(count)
    var prefix = RawWords(count)
    var handle = driver.allocate_device(count)
    var owner = OptimizerDeviceHandle(driver, handle)
    require(driver.upload_device(handle, source.address(), count) == count, "device upload count")
    source.expect(count)
    # The upload is synchronous and owns its device copy. Destroying/replacing
    # caller source storage after it returns cannot change later downloads.
    source = RawWords(count)
    require(driver.download_device(handle, result.address(), count) == count, "device download count")
    result.expect(count)
    source.expect(0)
    var short_count = max(count - 1, 1)
    require(driver.download_device(handle, prefix.address(), short_count) == short_count, "prefix download count")
    prefix.expect(short_count)
    owner.release()
    transfer_refusal(driver, False, handle, result.address(), count, closed_error(handle))
    result.expect(count)
    _ = source
    _ = result
    _ = prefix
    _ = owner


def refused_calls(driver: OptimizerNativeDriver) raises:
    for count in [-1, 0]:  # Two related invalid allocation contracts, no allocation succeeds.
        var refused = False
        try:
            _ = driver.allocate_device(count)
        except error:
            refused = True
            require(String(error) == "train_dev_alloc: n must be >= 1, got " + String(count), "allocation refusal text")
        require(refused, "invalid device allocation succeeded")
    var source = RawWords(4, True)
    var target = RawWords(4)
    var handle = driver.allocate_device(4)
    var owner = OptimizerDeviceHandle(driver, handle)
    require(driver.upload_device(handle, source.address(), 4) == 4, "refusal fixture setup upload")
    # Both directions must refuse before touching source/destination/device bytes.
    for direction in range(2):
        var upload = direction == 0
        var address = source.address() if upload else target.address()
        var name = String("the upload") if upload else String("the download")
        transfer_refusal(driver, upload, handle, address, -1,
            "ValueError: training device transfer count is out of range")
        transfer_refusal(driver, upload, handle, address, 2305843009213693952,
            "ValueError: training device transfer count is out of range")
        transfer_refusal(driver, upload, handle, 0, 4,
            "ValueError: training device transfer requires complete native storage")
        transfer_refusal(driver, upload, handle, -1, 4,
            "ValueError: training device transfer requires complete native storage")
        transfer_refusal(driver, upload, handle, 9223372036854775807, 4,
            "ValueError: training device transfer requires complete native storage")
        transfer_refusal(driver, upload, handle, address, 0,
            "training device array: " + name + " needs 0 floats, handle " + String(handle) + " holds 4")
        transfer_refusal(driver, upload, handle, address, 5,
            "training device array: " + name + " needs 5 floats, handle " + String(handle) + " holds 4")
        transfer_refusal(driver, upload, -1, address, 4, closed_error(-1))
    source.expect(4)
    target.expect(0)
    require(driver.download_device(handle, target.address(), 4) == 4, "device content after refused calls")
    target.expect(4)
    owner.release()
    for direction in range(2):
        transfer_refusal(driver, direction == 0, handle, source.address(), 4, closed_error(handle))
    source.expect(4)
    target.expect(4)
    _ = source
    _ = target
    _ = owner


def release_scope(driver: OptimizerNativeDriver, handle: Int, fail: Bool) raises:
    var owner = OptimizerDeviceHandle(driver, handle)
    if fail:
        raise Error("native publication fixture refused")
    _ = owner


def detached_scope(driver: OptimizerNativeDriver, handle: Int) -> Int:
    var owner = OptimizerDeviceHandle(driver, handle)
    var published = owner.detach()
    _ = owner
    return published


def reused_scope(driver: OptimizerNativeDriver, handle: Int) raises -> Int:
    var owner = OptimizerDeviceHandle(driver, handle)
    owner.release()
    owner.release()  # An already released owner must not call free twice.
    var replacement = driver.allocate_device(4)
    _ = owner
    return replacement


def lease(driver: OptimizerNativeDriver, fixture_id: Int) raises:
    var source = RawWords(4, True)
    var result = RawWords(4)
    var handle = driver.allocate_device(4)
    var setup_owner = OptimizerDeviceHandle(driver, handle)
    require(driver.upload_device(handle, source.address(), 4) == 4, "lease fixture setup upload")
    _ = setup_owner.detach()  # The scope under test takes responsibility below.
    if fixture_id < 2:
        var refused = False
        try:
            release_scope(driver, handle, fixture_id == 1)
        except error:
            refused = True
            require(String(error) == "native publication fixture refused", "lease masked original refusal")
        require(refused == (fixture_id == 1), "lease scope error contract")
        transfer_refusal(driver, False, handle, result.address(), 4, closed_error(handle))
        result.expect(0)
    elif fixture_id == 2:
        var published = detached_scope(driver, handle)
        var owner = OptimizerDeviceHandle(driver, published)
        require(published == handle, "detach changed the published handle")
        require(driver.download_device(published, result.address(), 4) == 4, "detached handle died with lease")
        result.expect(4)
        owner.release()
        transfer_refusal(driver, False, published, result.address(), 4, closed_error(published))
        _ = owner
    else:
        var replacement = reused_scope(driver, handle)
        var owner = OptimizerDeviceHandle(driver, replacement)
        require(driver.upload_device(replacement, source.address(), 4) == 4, "released owner freed replacement allocation")
        require(driver.download_device(replacement, result.address(), 4) == 4, "replacement handle read")
        result.expect(4)
        owner.release()
        _ = owner
    source.expect(4)
    _ = source
    _ = result
    _ = setup_owner


def run_checks() raises:
    # Malformed envelopes and pure plans never enter the device pool, so those
    # independent fixtures dispatch across host workers. GPU resource operations
    # below remain ordered for the existing shared context/handle pool.
    var failures = List[Int](length=DEVICE_COMMON, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            common(OptimizerNativeDriver(optimizer_gpu_v1, False), fixture_id)
        except error:
            fp.unsafe_store(fixture_id, 1)
            print("FAIL optimizer_device_gpu common", fixture_id, String(error))
    host_parallelize(task, DEVICE_COMMON)
    var failed = 0
    for value in failures:
        failed += value
    for fixture_id in range(DEVICE_GPU_EXTRA):
        try:
            var driver = OptimizerNativeDriver(optimizer_gpu_v1, False)
            if fixture_id < 3:
                roundtrip(driver, fixture_id)
            elif fixture_id == 3:
                refused_calls(driver)
            else:
                lease(driver, fixture_id - 4)
        except error:
            failed += 1
            print("FAIL optimizer_device_gpu resource", fixture_id, String(error))
    require(failed == 0, "optimizer_device_gpu failed " + String(failed) + " fixtures")
    print("PASS optimizer_device_gpu:", DEVICE_COMMON + DEVICE_GPU_EXTRA, "native descriptor/storage/lease fixtures")
