# SPDX-License-Identifier: Apache-2.0
"""GPU descriptor contracts and unpublished device allocation release.

Kernel submissions are ordered for the producer's existing shared neural
context/optimizer pool. Math runs on parallel GPU kernels. This is no claim
of concurrent GPU-session safety, identity or speed.
"""
from core.hostptr import f32_ptr
from core.neural_context import neural_ctx
from training.optimizer_kernels_gpu import _NEURAL_CTX
from training.optimizer_native_gpu import optimizer_gpu_v1
from training.optimizer_native_api import OptimizerNativeDriver
from training.dev_tensors import train_dev_get, train_dev_size
from checks.native_tests.optimizer_descriptor_cases import (
    DESCRIPTOR_COMMON_FIXTURES, FourCells, StepOwners, require, run_descriptor_fixture,
)


def io_lifetime(driver: OptimizerNativeDriver) raises:
    var owners = StepOwners()
    var downloaded = FourCells(17, 17, 17, 17)
    var req = owners.request(-1, 1)
    var queried = driver.io(req^, 0, 0)
    var handle = queried[0]
    var device_handle = -1
    require(handle >= 0 and queried[1] == 1,
        "default identical GPU fixture requires resident parameters; unsupported configuration is not a pass")
    try:
        # IO sources are explicit host spans, borrowed only through sync return.
        req = owners.request(handle, 1)
        req.g_addr = [owners.parameters.address()]
        var moved = driver.io(req^, 1, 0)
        require(moved[0] == handle and moved[1] == 4, "parameter upload result differs")
        owners.parameters.fill(88)
        req = owners.request(handle, 1)
        req.g_addr = [owners.parameters.address()]
        moved = driver.io(req^, 2, 0)
        require(moved[0] == handle and moved[1] == 4, "parameter download result differs")
        owners.parameters.expect(1, -2, 0.5, -0.25)
        req = owners.request(handle, 1)
        moved = driver.io(req^, 3, 0)
        require(moved[0] == handle and moved[1] == 4, "gradient upload result differs")
        owners.gradients.fill(89)
        req = owners.request(handle, 1)
        moved = driver.io(req^, 4, 0)
        require(moved[0] == handle and moved[1] == 4, "gradient download result differs")
        owners.gradients.expect(1, 2, -1, -2)
        req = owners.request(handle, 1)
        moved = driver.io(req^, 5, 0)
        device_handle = moved[1]
        require(moved[0] == handle and train_dev_size(device_handle) == 4,
            "parameter-device IO did not publish an independent four-cell native owner")
        var ctx = neural_ctx[_NEURAL_CTX]()
        train_dev_get(ctx, device_handle, f32_ptr(downloaded.address()), 4)
        downloaded.expect(1, -2, 0.5, -0.25)
        # Change optimizer-owned device parameters; the extracted tensor is a
        # copy with an independent lifetime, not a borrowed slice of that owner.
        owners.parameters.fill(77)
        req = owners.request(handle, 1)
        req.g_addr = [owners.parameters.address()]
        _ = driver.io(req^, 1, 0)
        train_dev_get(ctx, device_handle, f32_ptr(downloaded.address()), 4)
        downloaded.expect(1, -2, 0.5, -0.25)
        var released = device_handle
        require(driver.free_device(device_handle) == 0, "unpublished device free result differs")
        device_handle = -1
        var refused = False
        try:
            train_dev_get(ctx, released, f32_ptr(downloaded.address()), 4)
        except error:
            refused = True
            require(String(error) == "training device array: handle " + String(released) + " is not open",
                "released device owner refusal differs")
        require(refused, "released unpublished device owner remained readable")
        downloaded.expect(1, -2, 0.5, -0.25)
        refused = False
        try:
            _ = driver.free_device(released)
        except error:
            refused = True
            require(String(error) == "training device array: handle " + String(released) + " is not open",
                "double device release error was lost across descriptor")
        require(refused, "double device release was accepted")
        owners.parameters.expect(77, 77, 77, 77)
        owners.gradients.expect(1, 2, -1, -2)
    finally:
        if device_handle >= 0:
            _ = driver.free_device(device_handle)
        _ = driver.close(handle)
    _ = downloaded
    _ = owners


def main() raises:
    comptime COUNT = DESCRIPTOR_COMMON_FIXTURES + 1
    var failed = 0
    for fixture_id in range(COUNT):
        try:
            var driver = OptimizerNativeDriver(optimizer_gpu_v1, False)
            if fixture_id == DESCRIPTOR_COMMON_FIXTURES:
                io_lifetime(driver)
            else:
                run_descriptor_fixture(driver, fixture_id)
        except error:
            failed += 1
            print("FAIL optimizer_descriptor_gpu fixture", fixture_id, String(error))
    require(failed == 0, "optimizer_descriptor_gpu failed " + String(failed) + " fixtures")
    print("PASS optimizer_descriptor_gpu:", COUNT, "native ABI/owner fixtures")
