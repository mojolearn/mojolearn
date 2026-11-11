# SPDX-License-Identifier: Apache-2.0
"""Prepared GP lease owner: native state plus mandatory storage/image pins."""
from std.python import Python, PythonObject
from bindings.gp_native_boundary import GPConnection
from bindings.native_loaded_module import NativeModules, native_interpreter_finalizing
from bindings.native_prepared_owner import prepared_get, prepared_put, prepared_drop
from bindings.native_model_records import model_generation
from bindings.p3nc_fitted_freeze import fitted_owns_store_any, fitted_freeze_any
from core.gp_native_api import GPDriver
from core.gp_native_state import GPLeaseState


struct GPPreparedLease(Movable, Writable):
    # This owner is only in the weak-keyed prepared registry, never a model
    # field. The provider and mirrored buffers survive native lease cleanup.
    var state: GPLeaseState
    var driver: GPDriver
    var provider: PythonObject
    var mirrors: List[PythonObject]

    def __init__(out self, var state: GPLeaseState, driver: GPDriver, provider: PythonObject,
                 var mirrors: List[PythonObject]):
        self.state = state^
        self.driver = driver.copy()
        self.provider = provider
        self.mirrors = mirrors^

    def __del__(deinit self):
        var handle = self.state.take()
        if handle > 0:
            try:
                if not native_interpreter_finalizing():
                    self.driver.lease_close(handle)
            except:
                pass
        _ = self.provider
        _ = self.mirrors

    def write_to(self, mut writer: Some[Writer]):
        writer.write("GPPreparedLease(", self.state.handle, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def gp_resident_lease(model: PythonObject, slot: String, connection: GPConnection,
                      mirrors: List[PythonObject]) raises -> Int:
    var mods = NativeModules()
    var generation = model_generation(model)
    var record = prepared_get(mods, slot, model)
    if record is not None:
        try:
            var held = record.downcast_value_ptr[GPPreparedLease]()
            var same = held[].state.matches_header(Int(connection.provider._obj_ptr), generation, len(mirrors))
            if same:
                for i in range(len(mirrors)):  # driver-loop: fitted buffer identities in the original slot order
                    if not held[].state.matches_owner(i, Int(mirrors[i]._obj_ptr)):
                        same = False
                        break
            if same:
                var handle = held[].state.handle
                _ = record
                return handle
        except:
            # A record made by the former tuple envelope is replaced once.
            pass
    for i in range(len(mirrors)):  # driver-loop: keep the existing owning/read-only admission policy
        if not fitted_owns_store_any(mirrors[i]):
            if record is not None:
                prepared_drop(mods, slot, model)
            return 0
    for i in range(len(mirrors)):  # driver-loop: freeze the admitted owning arrays before their first upload
        _ = fitted_freeze_any(mirrors[i])
    var identities = List[Int]()
    var retained = List[PythonObject]()
    for i in range(len(mirrors)):  # driver-loop: capture resident storage lifetime and native identity once
        identities.append(Int(mirrors[i]._obj_ptr))
        retained.append(mirrors[i])
    # Lease creation only mutates native registry metadata; keep the GIL as
    # the previous opener did, with no extra release/reacquire boundary.
    var handle = connection.driver.lease_open()
    var state = GPLeaseState(handle, Int(connection.provider._obj_ptr), generation, identities^)
    var owner = GPPreparedLease(state^, connection.driver, connection.provider, retained^)
    var packed: PythonObject
    try:
        packed = PythonObject(alloc=owner^)
    except error:
        # Closing is idempotent if allocation unwinding already dropped owner.
        try:
            connection.driver.lease_close(handle)
        except:
            pass
        raise error
    prepared_put(mods, slot, model, packed)
    # A non-weakref model does not retain this owner. Its lease closes here
    # and the provider keeps the existing per-call upload behavior.
    _ = connection
    _ = mirrors
    return handle
