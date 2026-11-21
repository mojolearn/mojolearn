# SPDX-License-Identifier: Apache-2.0
# DBSCAN boundary, UNPACK side: the provider connection, the native owners
# of the resident core rows (lifetime pins: provider, references, keys,
# labels) and their holder, the holder's admission and release, the typed
# provider calls entered with the GIL released, and the resident reference
# owner (prepared once per model, provider and key). PACK is
# bindings/dbscan_native_boundary_pack.mojo. (The two were
# bindings/dbscan_native_boundary_door.mojo.)
from std.python import Python, PythonObject
from bindings.native_owner_registry_pack import borrowed_native_owner
from std.python._cpython import GILReleased
from bindings.py_balanced import py_fields, py_get, py_setitem
from bindings.native_loaded_module_door import native_interpreter_finalizing
from bindings.public_lifecycle_owner_door import lifecycle_admit_fields
from bindings.native_owner_lifetime_door import native_release_admit
from bindings.classical_arrays_door import classical_address, classical_wrap_layout
from bindings.native_array_storage_door import native_array_zeros
from bindings.p3nc_fitted_freeze_unpack import fitted_is_freezable, fitted_freeze
from providers.python.neural_descriptor_pyapi_door import NeuralDescriptorProvider, neural_descriptor_connect_once
from core.native_array_layout import NativeArrayLayout, native_array_code
from core.dbscan_native_api import (
    DBSCAN_NATIVE_ABI, DBSCAN_NATIVE_COUNT, DBSCAN_CAPABILITIES, DBSCAN_CAP_RESIDENT, DBSCANNativeDriver,
    DBSCANFitRequest, DBSCANCoreRequest, DBSCANGatherRequest, DBSCANReferenceRequest, DBSCANQueryRequest, DBSCANResidentQuery,
)
from core.dbscan_native_state import DBSCANReferenceKey, DBSCANReferenceState
from bindings.py_optional import py_present


struct DBSCANConnection(Movable):
    var provider: PythonObject
    var driver: DBSCANNativeDriver
    var capabilities: Int

    def __init__(out self, binding: PythonObject) raises:
        self.provider = neural_descriptor_connect_once(binding, "dbscan_native_api_v1", String(DBSCAN_NATIVE_ABI), DBSCAN_NATIVE_COUNT)
        self.driver = DBSCANNativeDriver(self.provider.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry)
        self.capabilities = self.driver.metadata(DBSCAN_CAPABILITIES)


struct DBSCANReferenceOwner(Movable, Writable):
    var state: DBSCANReferenceState
    var provider: PythonObject
    var refs: PythonObject
    var keys: PythonObject
    var labels: PythonObject

    def __init__(out self, var state: DBSCANReferenceState, provider: PythonObject,
                 refs: PythonObject, keys: PythonObject, labels: PythonObject):
        self.state = state^
        self.provider = provider
        self.refs = refs
        self.keys = keys
        self.labels = labels

    def __del__(deinit self):
        var handle = self.state.take_handle()
        if handle >= 0:
            try:
                if not native_interpreter_finalizing():
                    with GILReleased(Python()):
                        _ = self.state.driver.release(handle)
            except:
                pass
        _ = self.provider
        _ = self.refs
        _ = self.keys
        _ = self.labels

    def write_to(self, mut writer: Some[Writer]):
        writer.write("DBSCANReferenceOwner(", self.state.key.rows, ", ", self.state.key.features, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


struct DBSCANHolder(Movable, Writable):
    # A boundary lifetime pin to the immutable native owner. The separate
    # owner's native state stays alive while a query pins it, even if this
    # holder is replaced. Changing this pin never changes model.__dict__.
    var resident: PythonObject

    def __init__(out self):
        self.resident = PythonObject(None)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("DBSCANHolder")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def dbscan_reference_owner_pins(value: Int, index: Int) abi("C") -> Int:
    if value == 0 or index < -1 or index >= 4:
        return -1
    try:
        var owner = borrowed_native_owner(value)
        var native = owner.downcast_value_ptr[DBSCANReferenceOwner]()
        var result = 4
        if index == 0:
            result = Int(native[].provider._obj_ptr)
        elif index == 1:
            result = Int(native[].refs._obj_ptr)
        elif index == 2:
            result = Int(native[].keys._obj_ptr)
        elif index == 3:
            result = Int(native[].labels._obj_ptr)
        _ = owner
        return result
    except:
        return -1


def dbscan_holder_pins(value: Int, index: Int) abi("C") -> Int:
    if value == 0 or index < -1 or index >= 1:
        return -1
    try:
        var owner = borrowed_native_owner(value)
        var native = owner.downcast_value_ptr[DBSCANHolder]()
        var result = 1 if index == -1 else Int(native[].resident._obj_ptr)
        _ = owner
        return result
    except:
        return -1


def dbscan_holder_empty() raises -> PythonObject:
    var holder = DBSCANHolder()
    return PythonObject(alloc=holder^)


def dbscan_holder_admit(value: PythonObject) raises:
    if value is not None:
        try:
            _ = value.downcast_value_ptr[DBSCANHolder]()
        except:
            raise Error("TypeError: DBSCAN residency requires its native holder")


def dbscan_release(model: PythonObject) raises:
    var fields = py_fields(model)
    lifecycle_admit_fields(fields)
    var holder = py_get(fields, "_resident_holder")
    dbscan_holder_admit(holder)
    if holder is not None:
        holder.downcast_value_ptr[DBSCANHolder]()[].resident = PythonObject(None)
    _ = holder


def dbscan_run_fit(connection: DBSCANConnection, request: DBSCANFitRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return connection.driver.fit(request)
    finally:
        _ = connection


def dbscan_run_count(connection: DBSCANConnection, request: DBSCANCoreRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return connection.driver.core_count(request)
    finally:
        _ = connection


def dbscan_run_gather(connection: DBSCANConnection, request: DBSCANGatherRequest) raises:
    try:
        with GILReleased(Python()):
            _ = connection.driver.core_gather(request)
    finally:
        _ = connection


def _dbscan_prepare(model: PythonObject, connection: DBSCANConnection, key: DBSCANReferenceKey,
                    refs: PythonObject, keys: PythonObject, labels: PythonObject) raises -> PythonObject:
    var request = DBSCANReferenceRequest(key.refs, key.keys, key.labels, key.rows, key.features, key.metric)
    var handle = -1
    try:
        with GILReleased(Python()):
            handle = connection.driver.prepare(request)
        var state = DBSCANReferenceState(Int(model._obj_ptr), Int(connection.provider._obj_ptr), key.copy(), handle,
                                         connection.driver.copy())
        var owner = DBSCANReferenceOwner(state^, connection.provider, refs, keys, labels)
        handle = -1
        return PythonObject(alloc=owner^)
    finally:
        if handle >= 0:
            with GILReleased(Python()):
                _ = connection.driver.release(handle)
        _ = connection
        _ = refs
        _ = keys
        _ = labels


def dbscan_reference_owner(model: PythonObject, connection: DBSCANConnection, refs: PythonObject,
    keys: PythonObject, labels: PythonObject, rows: Int, features: Int, metric: Int, context: PythonObject) raises -> PythonObject:
    if (connection.capabilities & DBSCAN_CAP_RESIDENT) == 0:
        return PythonObject(None)
    var fields = py_fields(model)
    lifecycle_admit_fields(fields)
    var holder = py_get(fields, "_resident_holder")
    dbscan_holder_admit(holder)
    if holder is None:
        holder = dbscan_holder_empty()
        py_setitem(fields, "_resident_holder", holder)
    var key = DBSCANReferenceKey(Int(refs._obj_ptr), Int(keys._obj_ptr), Int(labels._obj_ptr),
        Int(py=classical_address(refs)), Int(py=classical_address(keys)), Int(py=classical_address(labels)), rows, features, metric)
    var held = holder.downcast_value_ptr[DBSCANHolder]()[].resident
    if held is not None:
        if held.downcast_value_ptr[DBSCANReferenceOwner]()[].state.matches(Int(model._obj_ptr), Int(connection.provider._obj_ptr), key):
            return held
    holder.downcast_value_ptr[DBSCANHolder]()[].resident = PythonObject(None)
    held = PythonObject(None)
    # Public ndarray owners remain the original direct query route, even
    # when read-only. Only the existing freezable Array owners become resident.
    if (not fitted_is_freezable(refs, context) or not fitted_is_freezable(keys, context)
            or not fitted_is_freezable(labels, context)):
        return PythonObject(None)
    _ = fitted_freeze(refs, context)
    _ = fitted_freeze(keys, context)
    _ = fitted_freeze(labels, context)
    var owner = _dbscan_prepare(model, connection, key, refs, keys, labels)
    holder.downcast_value_ptr[DBSCANHolder]()[].resident = owner
    return owner
