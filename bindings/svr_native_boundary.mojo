# SPDX-License-Identifier: Apache-2.0
"""Exact SVR provider connection and boundary storage/image pins."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.svr_native_api import SVR_NATIVE_ABI, SVR_NATIVE_COUNT, SvrNativeDriver, SvrFitRequest, SvrFitPlan, SvrQueryRequest
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_entry, neural_descriptor_connect_factory_once


@fieldwise_init
struct SvrConnection(Movable):
    var provider: PythonObject
    var driver: SvrNativeDriver


def svr_connection(binding: PythonObject) raises -> SvrConnection:
    var factory = neural_descriptor_entry(binding, "svr_native_api_v1")
    var provider = neural_descriptor_connect_factory_once(factory, String(SVR_NATIVE_ABI), SVR_NATIVE_COUNT)
    return SvrConnection(provider, SvrNativeDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry))


def svr_run_fit(connection: SvrConnection, request: SvrFitRequest, mut plan: SvrFitPlan) raises:
    try:
        with GILReleased(Python()):
            plan.count = connection.driver.fit(request)
    finally:
        _ = connection
        _ = plan


def svr_run_query(connection: SvrConnection, request: SvrQueryRequest, resident: Bool) raises:
    try:
        with GILReleased(Python()):
            _ = connection.driver.query(request, resident)
    finally:
        _ = connection


from core.gp_native_state import GPLeaseState
from bindings.native_loaded_module import NativeModules, native_interpreter_finalizing
from bindings.native_prepared_owner import prepared_get, prepared_put, prepared_drop
from bindings.native_model_records import model_generation
from bindings.p3nc_fitted_freeze import fitted_owns_store_any, fitted_freeze_any


@fieldwise_init
struct SvrResidentPin(Movable):
    var owner: PythonObject
    var handle: Int


struct SvrPreparedLease(Movable, Writable):
    # Weak prepared-registry state only, never an estimator field. Public
    # mirrors and both image/binding identities outlive final native release.
    var state: GPLeaseState
    var driver: SvrNativeDriver
    var provider: PythonObject
    var binding: PythonObject
    var mirrors: List[PythonObject]

    def __init__(out self, var state: GPLeaseState, driver: SvrNativeDriver,
        provider: PythonObject, binding: PythonObject, var mirrors: List[PythonObject]):
        self.state = state^
        self.driver = driver.copy()
        self.provider = provider
        self.binding = binding
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
        _ = self.binding
        _ = self.mirrors

    def write_to(self, mut writer: Some[Writer]):
        writer.write("SvrPreparedLease(", self.state.handle, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def svr_resident_pin(model: PythonObject, slot: String, binding: PythonObject,
    connection: SvrConnection, mirrors: List[PythonObject]) raises -> SvrResidentPin:
    var mods = NativeModules()
    var generation = model_generation(model)
    var record = prepared_get(mods, slot, model)
    if record is not None:
        try:
            var held = record.downcast_value_ptr[SvrPreparedLease]()
            var same = held[].state.matches_header(Int(binding._obj_ptr), generation, len(mirrors))
            if same:
                for i in range(len(mirrors)):  # driver-loop: original fitted owner identity policy
                    if not held[].state.matches_owner(i, Int(mirrors[i]._obj_ptr)):
                        same = False
                        break
            if same:
                return SvrResidentPin(record, held[].state.handle)
        except:
            # The former tuple record is replaced once by the native owner.
            pass
    # The former optional opener still controls admission/availability. Its
    # compiled body is represented by the pinned native driver, never called
    # through a Python envelope.
    var opener = neural_descriptor_entry(binding, "svm_resident_lease", False)
    if opener is None:
        if record is not None:
            prepared_drop(mods, slot, model)
        return SvrResidentPin(PythonObject(None), 0)
    for i in range(len(mirrors)):  # driver-loop: exact owning Array admission before any freeze
        if not fitted_owns_store_any(mirrors[i]):
            if record is not None:
                prepared_drop(mods, slot, model)
            return SvrResidentPin(PythonObject(None), 0)
    for i in range(len(mirrors)):  # driver-loop: preserve original read-only publication order
        _ = fitted_freeze_any(mirrors[i])
    var identities = List[Int]()
    var retained = List[PythonObject]()
    for i in range(len(mirrors)):  # driver-loop: boundary pins and native identity words, no tensor copy
        identities.append(Int(mirrors[i]._obj_ptr))
        retained.append(mirrors[i])
    var handle = connection.driver.lease_open()
    var state = GPLeaseState(handle, Int(binding._obj_ptr), generation, identities^)
    var owner = SvrPreparedLease(state^, connection.driver, connection.provider, binding, retained^)
    var packed: PythonObject
    try:
        packed = PythonObject(alloc=owner^)
    except error:
        try:
            connection.driver.lease_close(handle)
        except:
            pass
        raise error
    prepared_put(mods, slot, model, packed)
    var saved = prepared_get(mods, slot, model)
    if saved is not packed:
        # As before, a non-weakref model cannot retain a lease. Close before
        # query and use the same provider's existing per-call upload route.
        return SvrResidentPin(PythonObject(None), 0)
    _ = connection
    _ = mirrors
    return SvrResidentPin(packed, handle)
