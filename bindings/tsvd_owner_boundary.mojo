# SPDX-License-Identifier: Apache-2.0
"""Boundary image/storage pins around native TSVD resident-component leases."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.tsvd_owner_state import TSVDLeaseState
from bindings.native_loaded_module import NativeModules
from bindings.native_prepared_owner import prepared_get, prepared_put, prepared_drop
from bindings.native_model_records import model_generation
from bindings.p3nc_fitted_freeze import fitted_is_freezable_any, fitted_freeze_any
from bindings.tsvd_descriptor_boundary import tsvd_provider, tsvd_driver

comptime TSVD_RESIDENT_SLOT = "tsvd_resident_native_v2"


@fieldwise_init
struct TSVDResidentOwner(Movable, Writable):
    var state: TSVDLeaseState
    var provider_pin: PythonObject
    var components_pin: PythonObject

    def write_to(self, mut writer: Some[Writer]):
        writer.write("TSVDResidentOwner(open=", self.state.lease != 0, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def __del__(deinit self):
        var lease = self.state.take_lease()
        if lease != 0:
            try:
                with GILReleased(Python()):
                    self.state.driver.close(lease)
            except:
                pass
        _ = self.provider_pin
        _ = self.components_pin


@fieldwise_init
struct TSVDLeasePin(Copyable, Movable):
    var owner: PythonObject
    var lease: Int


def tsvd_owner_drop(model: PythonObject) raises:
    var mods = NativeModules()
    prepared_drop(mods, TSVD_RESIDENT_SLOT, model)


def tsvd_owner_lease(model: PythonObject, connection: Int, components: PythonObject) raises -> TSVDLeasePin:
    var mods = NativeModules()
    var generation = model_generation(model)
    var held = prepared_get(mods, TSVD_RESIDENT_SLOT, model)
    if held is not None:
        var owner = held.downcast_value_ptr[TSVDResidentOwner]()
        if owner[].state.matches(connection, generation, Int(components._obj_ptr)):
            var lease = owner[].state.lease
            return TSVDLeasePin(held, lease)
    # The old resident door only freezes exact Arrays which own their
    # storage. Views, writable ndarrays and conversion copies stay per-call.
    if not fitted_is_freezable_any(components) or components._base is not None:
        if held is not None:
            prepared_drop(mods, TSVD_RESIDENT_SLOT, model)
        return TSVDLeasePin(PythonObject(None), 0)
    _ = fitted_freeze_any(components)
    var provider = tsvd_provider(connection)
    var driver = tsvd_driver(connection)
    var lease = 0
    try:
        with GILReleased(Python()):
            lease = driver.open()
        var state = TSVDLeaseState(connection, generation, Int(components._obj_ptr), lease, driver.copy())
        var owner = TSVDResidentOwner(state^, provider, components)
        var opened_lease = lease
        lease = 0  # native owner now owns cleanup, including Python allocation failure
        var object = PythonObject(alloc=owner^)
        prepared_put(mods, TSVD_RESIDENT_SLOT, model, object)
        var retained = prepared_get(mods, TSVD_RESIDENT_SLOT, model)
        if retained is not object:
            # No weak model record: drop this lease before query, as the old
            # temporary opener object did. The resident kernel still sees its
            # retired lease ID and uses the same per-call storage path.
            return TSVDLeasePin(PythonObject(None), opened_lease)
        return TSVDLeasePin(object, opened_lease)
    except error:
        if lease != 0:
            try:
                with GILReleased(Python()):
                    driver.close(lease)
            except:
                pass
        raise error^
    finally:
        _ = provider
