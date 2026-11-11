# SPDX-License-Identifier: Apache-2.0
"""Agglomerative provider, buffer and retained-reference unpack/pack boundary."""
from std.python import Python, PythonObject
from bindings.native_owner_registry import borrowed_native_owner
from std.python._cpython import GILReleased, PyObjectPtr, ExternalFunction
from bindings.py_balanced import py_get, py_setitem, py_type
from bindings.native_loaded_module import native_interpreter_finalizing
from bindings.public_lifecycle_owner import lifecycle_admit_fields
from bindings.native_owner_lifetime import native_release_admit
from bindings.cpython_table import cpython_table
from bindings.optimizer_public_words import stash_python_error
from core.shared_slot import SharedSlot
from bindings.classical_arrays import classical_address, classical_wrap_layout
from bindings.native_array_storage import native_array_zeros
from bindings.p3nc_fitted_freeze import fitted_is_freezable, fitted_freeze
from bindings.dbscan_native_boundary import DBSCANConnection
from providers.python.compiled_binding_snapshot_pyapi import CompiledBindingSnapshot
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_once
from providers.python.hierarchy_solver_descriptor_pyapi import hierarchy_solver_admit
from providers.python.hierarchy_cluster_descriptor_pyapi import hierarchy_cluster_admit
from providers.python.x_neighbors_admission_pyapi import xn_is_module, xn_module
from core.native_array_layout import NativeArrayLayout, native_array_code
from core.neighbors_native_driver import XnDriver
from core.dbscan_native_api import DBSCAN_CAP_RESIDENT
from core.agglo_native_api import (
    AggloSolverRequest, AggloClusterRequest, AggloZeroRowsRequest,
    AggloSolverDriver, AggloClusterDriver, AggloZeroRowsDriver,
    agglo_reference_request, agglo_query_request, agglo_resident_query,
)
from core.agglo_native_state import AggloReferenceKey, AggloReferenceState
from x_neighbors.descriptor_schema import XN_ABI_NAME, XN_OPERATION_COUNT


@fieldwise_init
struct AggloBindings(Movable):
    var estimators: PythonObject
    var solver: PythonObject
    var cluster: PythonObject
    var neighbors: PythonObject


struct AggloSolverConnection(Movable):
    var provider: PythonObject
    var driver: AggloSolverDriver

    def __init__(out self, binding: PythonObject) raises:
        self.provider = hierarchy_solver_admit(binding)
        var native = self.provider.downcast_value_ptr[CompiledBindingSnapshot]()[].provider
        self.driver = AggloSolverDriver(native.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)


struct AggloClusterConnection(Movable):
    var provider: PythonObject
    var driver: AggloClusterDriver

    def __init__(out self, binding: PythonObject) raises:
        self.provider = hierarchy_cluster_admit(binding)
        var native = self.provider.downcast_value_ptr[CompiledBindingSnapshot]()[].provider
        self.driver = AggloClusterDriver(native.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)


struct AggloZeroRowsConnection(Movable):
    var provider: PythonObject
    var driver: AggloZeroRowsDriver

    def __init__(out self, binding: PythonObject) raises:
        var module = xn_module(binding)
        var provider: PythonObject
        if xn_is_module(binding):
            provider = neural_descriptor_connect_once(module, "x_neighbors_native_api_v1",
                String(XN_ABI_NAME), XN_OPERATION_COUNT)
        else:
            provider = binding.downcast_value_ptr[CompiledBindingSnapshot]()[].provider
        self.driver = AggloZeroRowsDriver(XnDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry))
        self.provider = provider
        _ = module
        _ = binding


def agglo_solver_run(connection: AggloSolverConnection, request: AggloSolverRequest) raises:
    try:
        with GILReleased(Python()):
            _ = connection.driver.fit(request)
    finally:
        _ = connection


def agglo_cluster_run(connection: AggloClusterConnection, request: AggloClusterRequest) raises:
    try:
        with GILReleased(Python()):
            connection.driver.fit(request)
    finally:
        _ = connection


def agglo_zero_rows_run(connection: AggloZeroRowsConnection, request: AggloZeroRowsRequest) raises:
    try:
        with GILReleased(Python()):
            connection.driver.run(request)
    finally:
        _ = connection


def agglo_array(shape: List[Int], typestr: String, context: PythonObject) raises -> PythonObject:
    var layout = NativeArrayLayout(shape.copy(), 4)
    var storage = native_array_zeros(native_array_code(typestr), layout.count)
    return classical_wrap_layout(storage, layout, typestr, context)


comptime _AggloHash = ExternalFunction["PyObject_Hash", def(PyObjectPtr) thin abi("C") -> Int]
comptime _AggloRepr = ExternalFunction["PyObject_Repr", def(PyObjectPtr) thin abi("C") -> PyObjectPtr]


struct AggloScalarAPI(Defaultable, Movable):
    var hash_value: _AggloHash.type
    var repr_value: _AggloRepr.type

    def __init__(out self):
        ref cpy = Python().cpython()
        self.hash_value = _AggloHash.load(cpy.lib.borrow())
        self.repr_value = _AggloRepr.load(cpy.lib.borrow())


comptime AGGLO_SCALAR_API = SharedSlot[AggloScalarAPI, "MojolearnAggloScalarAPIV1"]


def agglo_hash_admitted(value: PythonObject) raises:
    # Preserve the former exact-dictionary unhashable-option refusal without
    # looking up a replaceable Python callable in builtins.
    var result = AGGLO_SCALAR_API.get_or_create_ptr()[].hash_value(value._obj_ptr)
    _ = value
    if result == -1:
        var failure = stash_python_error()
        raise failure^


def agglo_integer_admitted(value: PythonObject) raises -> Int:
    var pointer = cpython_table()[].PyNumber_Long(value._obj_ptr)
    if not pointer:
        var failure = stash_python_error()
        raise failure^
    var converted = PythonObject(from_owned=pointer)
    var result = Int(py=converted)
    _ = value
    return result


def agglo_float_admitted(value: PythonObject) raises -> Float64:
    var pointer = cpython_table()[].PyNumber_Float(value._obj_ptr)
    if not pointer:
        var failure = stash_python_error()
        raise failure^
    var converted = PythonObject(from_owned=pointer)
    var result = Float64(py=converted)
    _ = value
    return result


def agglo_scalar_repr(value: PythonObject, context: PythonObject) raises -> String:
    # hierarchy_settings already admits exact single-item numeric arrays for
    # truth/comparison. NumPy array repr can invoke global Python formatters.
    if py_type(value) is py_get(context, "ndarray"):
        return "<native scalar array>"
    var pointer = AGGLO_SCALAR_API.get_or_create_ptr()[].repr_value(value._obj_ptr)
    if not pointer:
        var failure = stash_python_error()
        raise failure^
    var rendered = PythonObject(from_owned=pointer)
    var result = String(py=rendered)
    _ = value
    return result


def agglo_publish(fields: PythonObject, name: String, value: PythonObject, context: PythonObject) raises:
    lifecycle_admit_fields(fields)
    var old = py_get(fields, name)
    if old is not value:
        native_release_admit(old, context)
    py_setitem(fields, name, value)


struct AggloReferenceOwner(Movable, Writable):
    var state: AggloReferenceState
    var provider: PythonObject
    var refs: PythonObject
    var labels: PythonObject

    def __init__(out self, var state: AggloReferenceState, provider: PythonObject,
                 refs: PythonObject, labels: PythonObject):
        self.state = state^
        self.provider = provider
        self.refs = refs
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
        _ = self.labels

    def write_to(self, mut writer: Some[Writer]):
        writer.write("AggloReferenceOwner(", self.state.key.rows, ", ", self.state.key.features, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


struct AggloHolder(Movable, Writable):
    var resident: PythonObject

    def __init__(out self):
        self.resident = PythonObject(None)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("AggloHolder")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def agglo_reference_owner_pins(value: Int, index: Int) abi("C") -> Int:
    if value == 0 or index < -1 or index >= 3:
        return -1
    try:
        var owner = borrowed_native_owner(value)
        var native = owner.downcast_value_ptr[AggloReferenceOwner]()
        var result = 3
        if index == 0:
            result = Int(native[].provider._obj_ptr)
        elif index == 1:
            result = Int(native[].refs._obj_ptr)
        elif index == 2:
            result = Int(native[].labels._obj_ptr)
        _ = owner
        return result
    except:
        return -1


def agglo_holder_pins(value: Int, index: Int) abi("C") -> Int:
    if value == 0 or index < -1 or index >= 1:
        return -1
    try:
        var owner = borrowed_native_owner(value)
        var native = owner.downcast_value_ptr[AggloHolder]()
        var result = 1 if index == -1 else Int(native[].resident._obj_ptr)
        _ = owner
        return result
    except:
        return -1


def agglo_holder_empty() raises -> PythonObject:
    var holder = AggloHolder()
    return PythonObject(alloc=holder^)


def agglo_holder_admit(value: PythonObject) raises:
    if value is not None:
        try:
            _ = value.downcast_value_ptr[AggloHolder]()
        except:
            raise Error("TypeError: hierarchy residency requires its native holder")


def agglo_holder(fields: PythonObject) raises -> PythonObject:
    lifecycle_admit_fields(fields)
    var holder = py_get(fields, "_resident_holder")
    agglo_holder_admit(holder)
    if holder is None:
        holder = agglo_holder_empty()
        py_setitem(fields, "_resident_holder", holder)
    return holder


def agglo_release(fields: PythonObject) raises:
    var holder = agglo_holder(fields)
    holder.downcast_value_ptr[AggloHolder]()[].resident = PythonObject(None)
    _ = holder


def agglo_fresh_holder(fields: PythonObject) raises:
    agglo_release(fields)
    lifecycle_admit_fields(fields)
    agglo_holder_admit(py_get(fields, "_resident_holder"))
    py_setitem(fields, "_resident_holder", agglo_holder_empty())


def _agglo_prepare(fields: PythonObject, connection: DBSCANConnection, key: AggloReferenceKey,
    refs: PythonObject, labels: PythonObject, refs_address: Int, labels_address: Int) raises -> PythonObject:
    var request = agglo_reference_request(refs_address, labels_address, key.rows, key.features)
    var handle = -1
    try:
        with GILReleased(Python()):
            handle = connection.driver.prepare(request)
        var state = AggloReferenceState(Int(fields._obj_ptr), Int(connection.provider._obj_ptr),
            key.copy(), handle, connection.driver.copy())
        var owner = AggloReferenceOwner(state^, connection.provider, refs, labels)
        handle = -1
        return PythonObject(alloc=owner^)
    finally:
        if handle >= 0:
            with GILReleased(Python()):
                _ = connection.driver.release(handle)
        _ = connection
        _ = refs
        _ = labels


def agglo_reference_owner(fields: PythonObject, connection: DBSCANConnection, refs: PythonObject,
    labels: PythonObject, refs_address: Int, labels_address: Int, rows: Int, features: Int,
    context: PythonObject) raises -> PythonObject:
    # The existing hierarchy path creates its holder even on host-only images.
    var holder = agglo_holder(fields)
    if (connection.capabilities & DBSCAN_CAP_RESIDENT) == 0:
        return PythonObject(None)
    var key = AggloReferenceKey(Int(refs._obj_ptr), Int(labels._obj_ptr), rows, features)
    var held = holder.downcast_value_ptr[AggloHolder]()[].resident
    if held is not None:
        if held.downcast_value_ptr[AggloReferenceOwner]()[].state.matches(
            Int(fields._obj_ptr), Int(connection.provider._obj_ptr), key):
            return held
    holder.downcast_value_ptr[AggloHolder]()[].resident = PythonObject(None)
    held = PythonObject(None)
    if not fitted_is_freezable(refs, context) or not fitted_is_freezable(labels, context):
        return PythonObject(None)
    _ = fitted_freeze(refs, context)
    _ = fitted_freeze(labels, context)
    var owner = _agglo_prepare(fields, connection, key, refs, labels, refs_address, labels_address)
    holder.downcast_value_ptr[AggloHolder]()[].resident = owner
    return owner


def agglo_mint(fields: PythonObject, binding: PythonObject, refs: PythonObject, labels: PythonObject,
    refs_address: Int, labels_address: Int, rows: Int, features: Int, context: PythonObject) raises:
    _ = agglo_holder(fields)
    try:
        var connection = DBSCANConnection(binding)
        _ = agglo_reference_owner(fields, connection, refs, labels, refs_address, labels_address, rows, features, context)
    except:
        agglo_release(fields)


def agglo_reference_labels(fields: PythonObject, binding: PythonObject, refs: PythonObject,
    labels: PythonObject, query: PythonObject, refs_address: Int, labels_address: Int, query_address: Int,
    rows: Int, queries: Int, features: Int, context: PythonObject, resident: Bool = True) raises -> PythonObject:
    var connection = DBSCANConnection(binding)
    var owner = PythonObject(None)
    if resident:
        owner = agglo_reference_owner(fields, connection, refs, labels, refs_address, labels_address, rows, features, context)
    var dst = agglo_array(List[Int](length=1, fill=queries), "<i4", context)
    var dst_address = Int(py=classical_address(dst))
    try:
        if owner is not None:
            var driver = owner.downcast_value_ptr[AggloReferenceOwner]()[].state.driver.copy()
            var request = agglo_resident_query(query_address, dst_address,
                owner.downcast_value_ptr[AggloReferenceOwner]()[].state.handle, queries, features)
            with GILReleased(Python()):
                _ = driver.resident(request)
        else:
            var chosen = List[Int32](length=queries, fill=0)
            var request = agglo_query_request(refs_address, labels_address, query_address, dst_address,
                Int(chosen.unsafe_ptr()), rows, queries, features)
            with GILReleased(Python()):
                _ = connection.driver.query(request)
            _ = chosen^
    finally:
        _ = owner
        _ = connection
        _ = refs
        _ = labels
        _ = query
        _ = dst
    return dst
