# SPDX-License-Identifier: Apache-2.0
"""Native optimizer control transitions and actual descriptor-owned resources."""
from training.optimizer_owner_control import OptimizerOwnerControl, optimizer_restored_pristine
from training.optimizer_native_api import OptimizerNativeDriver
from checks.native_tests.optimizer_descriptor_cases import FourCells, require

comptime OWNER_PURE = 18
comptime OWNER_RESOURCE = 4


def pure(driver: OptimizerNativeDriver, fixture_id: Int) raises:
    var kind = fixture_id % 3
    var owner = OptimizerOwnerControl(1234567, kind, True)
    if fixture_id < 3:
        var names: List[String] = ["SGD", "Adam", "AdamW"]
        require(owner.identity == 1234567 and owner.kind == kind and owner.name() == names[kind], "native owner identity/kind")
        require(not owner.resident() and owner.on_host() and not owner.close_request(), "new owner acquired a resource")
        require(owner.host_fresh and owner.pristine and not owner.parameters_device and not owner.gradients_device,
            "new owner freshness contract")
    elif fixture_id < 6:
        var other = OptimizerOwnerControl(7654321, kind, False)
        owner.begin()
        other.begin()  # Independent owners never share an operation lock.
        var refused = False
        try:
            owner.begin()
        except error:
            refused = True
            require(String(error) == "RuntimeError: optimizer owner is already in use by another native operation",
                "busy owner refusal changed")
        require(refused and owner.busy and other.busy, "busy refusal changed active owners")
        owner.finish()
        owner.begin()
        require(owner.busy and other.busy, "finish did not permit later operation")
        owner.finish()
        other.finish()
        require(not owner.busy and not other.busy, "operation lease failed to release busy state")
    elif fixture_id < 14:
        var bits = fixture_id - 6
        var pristine = (bits & 4) != 0
        var first = (bits & 2) != 0
        var second = (bits & 1) != 0
        var restored = OptimizerOwnerControl(99, 1, pristine)
        var expected = bits == 4  # Only pristine with neither materialized moment.
        require(optimizer_restored_pristine(pristine, first, second) == expected,
            "materialized moments lost restore authority")
        require(restored.serialized_pristine(first, second) == expected, "owner export pristine decision differs")
    elif fixture_id < 17:
        owner.attach(-1, driver)
        require(not owner.close_request() and not owner.resident() and owner.on_host(), "negative handle attached a driver")
        owner.close()
        owner.close()
        require(owner.handle == -1 and not owner.driver, "empty close manufactured state")
    else:
        owner.begin()
        owner.host_fresh = False
        owner.parameters_device = True
        owner.gradients_device = True
        owner.finish_close()
        require(owner.handle == -1 and not owner.driver and owner.host_fresh
            and not owner.parameters_device and not owner.gradients_device, "published close did not reset storage facts")
        require(owner.busy and owner.pristine and owner.identity == 1234567 and owner.kind == kind,
            "close publication changed identity, pristine or active lease")
        owner.finish()


def closed_error(handle: Int) -> String:
    return "optimizer_resident: handle " + String(handle) + " is not open"


def closed(driver: OptimizerNativeDriver, handle: Int) raises:
    var first = FourCells(17, 17, 17, 17)
    var second = FourCells(19, 19, 19, 19)
    var refused = False
    try:
        _ = driver.download(handle, first.address(), second.address(), 4)
    except error:
        refused = True
        require(String(error) == closed_error(handle), "closed owner descriptor refusal changed")
    require(refused, "closed optimizer owner still exposes native storage")
    first.expect(17, 17, 17, 17)
    second.expect(19, 19, 19, 19)
    _ = first
    _ = second


def alive(driver: OptimizerNativeDriver, handle: Int) raises:
    var first = FourCells(1, -2, 0.5, 4)
    var second = FourCells(3, 1, -0.5, -2)
    var target = FourCells()
    require(driver.upload(handle, first.address(), second.address(), 4) == 4, "owner state upload count")
    require(driver.download(handle, target.address(), 0, 4) == 4, "owner state read count")
    target.expect(1, -2, 0.5, 4)
    first.expect(1, -2, 0.5, 4)
    second.expect(3, 1, -0.5, -2)
    _ = first
    _ = second
    _ = target


def resource(driver: OptimizerNativeDriver, fixture_id: Int) raises:
    var owner = OptimizerOwnerControl(101, fixture_id % 3, True)
    var handle = driver.open(4)
    var attached = True
    owner.attach(handle, driver)
    try:
        require(owner.handle == handle and owner.on_host() == driver.host and owner.resident() == (not driver.host),
            "typed owner attached wrong target/resource")
        alive(driver, handle)
        owner.host_fresh = False
        owner.parameters_device = True
        owner.gradients_device = True
        if fixture_id == 0:
            owner.close()
            attached = False
            closed(driver, handle)
            require(not owner.close_request() and not owner.resident() and owner.host_fresh,
                "successful close retained resident state")
            # A second close must not reach the descriptor, including when its
            # former pool slot has already been allocated for another owner.
            var replacement = driver.open(4)
            try:
                owner.close()
                alive(driver, replacement)
            finally:
                _ = driver.close(replacement)
        elif fixture_id == 1:
            owner.begin()
            var request = owner.close_request()
            if not request:
                raise Error("live owner did not produce a native close request")
            require(request.value().handle == handle and request.value().driver.host == driver.host,
                "close request lost handle/driver")
            request.value().run()
            attached = False
            closed(driver, handle)
            require(owner.handle == handle and owner.driver and owner.busy and not owner.host_fresh,
                "RUN prematurely published control changes")
            owner.finish_close()
            require(owner.handle == -1 and not owner.driver and owner.host_fresh and owner.busy,
                "PACK failed to publish close under the active lease")
            owner.finish()
        elif fixture_id == 2:
            # A real descriptor failure: an external owner has already closed
            # the native slot. No mock entry or fabricated Python callback.
            _ = driver.close(handle)
            attached = False
            for attempt in range(2):  # Dependent explicit retry on the same retained record.
                var refused = False
                try:
                    owner.close()
                except error:
                    refused = True
                    require(String(error) == closed_error(handle), "failed close lost native error")
                require(refused and owner.handle == handle and owner.driver
                    and owner.driver.value().host == driver.host, "failed close discarded its retry record")
                require(not owner.host_fresh and owner.parameters_device and owner.gradients_device,
                    "failed close partially reset storage facts")
                require(owner.close_request().value().handle == handle, "failed close lost retry handle")
            # The fixture knows the resource was externally released.
            owner.finish_close()
        else:
            var other = driver.open(4)
            try:
                owner.attach(other, driver)
                require(owner.handle == handle, "second attach replaced a live native resource")
                owner.close()
                attached = False
                closed(driver, handle)
                alive(driver, other)
                owner.attach(other, driver)
                require(owner.handle == other and owner.resident() == (not driver.host), "closed owner could not attach again")
                owner.close()
                other = -1
            finally:
                if other >= 0:
                    _ = driver.close(other)
        require(owner.identity == 101 and owner.pristine, "resource lifecycle changed identity/pristine")
    finally:
        if attached:
            _ = driver.close(handle)
    _ = owner
