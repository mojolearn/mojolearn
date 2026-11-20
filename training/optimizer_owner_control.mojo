# SPDX-License-Identifier: Apache-2.0
"""Native lifetime/control state for one public optimizer; no Python values."""
from training.optimizer_native_api import OptimizerNativeDriver


def optimizer_restored_pristine(pristine: Bool, first_present: Bool, second_present: Bool) -> Bool:
    # Materialized public moments are authoritative in a handle-free copy.
    # A live slot can still carry its initial pristine flag after many steps.
    return pristine and not first_present and not second_present


@fieldwise_init
struct OptimizerCloseRequest(Copyable, Movable):
    var driver: OptimizerNativeDriver
    var handle: Int

    def run(self) raises:
        _ = self.driver.close(self.handle)


struct OptimizerOwnerControl(Movable):
    var identity: Int
    var kind: Int
    var handle: Int
    var driver: Optional[OptimizerNativeDriver]
    var host_fresh: Bool
    var pristine: Bool
    var parameters_device: Bool
    var gradients_device: Bool
    var busy: Bool

    def __init__(out self, identity: Int, kind: Int, pristine: Bool):
        self.identity = identity
        self.kind = kind
        self.handle = -1
        self.driver = Optional[OptimizerNativeDriver]()
        self.host_fresh = True
        self.pristine = pristine
        self.parameters_device = False
        self.gradients_device = False
        self.busy = False

    def name(self) -> String:
        return String("SGD") if self.kind == 0 else (String("Adam") if self.kind == 1 else String("AdamW"))

    def begin(mut self) raises:
        if self.busy:
            raise Error("RuntimeError: optimizer owner is already in use by another native operation")
        self.busy = True

    def finish(mut self):
        self.busy = False

    def attach(mut self, handle: Int, driver: OptimizerNativeDriver):
        if self.handle < 0 and handle >= 0:
            self.handle = handle
            self.driver = Optional[OptimizerNativeDriver](driver.copy())

    def on_host(self) -> Bool:
        return not self.driver or self.driver.value().host

    def resident(self) -> Bool:
        return self.handle >= 0 and not self.on_host()

    def serialized_pristine(self, first_present: Bool, second_present: Bool) -> Bool:
        return optimizer_restored_pristine(self.pristine, first_present, second_present)

    def close_request(self) -> Optional[OptimizerCloseRequest]:
        if self.handle < 0:
            return Optional[OptimizerCloseRequest]()
        return Optional[OptimizerCloseRequest](OptimizerCloseRequest(self.driver.value().copy(), self.handle))

    def finish_close(mut self):
        self.handle = -1
        self.driver = Optional[OptimizerNativeDriver]()
        self.host_fresh = True
        self.parameters_device = False
        self.gradients_device = False

    def close(mut self) raises:
        var request = self.close_request()
        if request:
            request.value().run()
            self.finish_close()
