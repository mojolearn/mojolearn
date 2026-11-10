# SPDX-License-Identifier: Apache-2.0
"""Native training-device handle ownership; boundary scopes manage the GIL."""
from training.optimizer_native_api import OptimizerNativeDriver


struct OptimizerDeviceHandle(Movable):
    var driver: OptimizerNativeDriver
    var handle: Int
    var owned: Bool

    def __init__(out self, driver: OptimizerNativeDriver, handle: Int):
        self.driver = driver.copy()
        self.handle = handle
        self.owned = True

    def release(mut self) raises:
        if self.owned:
            _ = self.driver.free_device(self.handle)
            self.owned = False

    def detach(mut self) -> Int:
        self.owned = False
        return self.handle

    def __del__(deinit self):
        if self.owned:
            try:
                _ = self.driver.free_device(self.handle)
            except:
                pass
