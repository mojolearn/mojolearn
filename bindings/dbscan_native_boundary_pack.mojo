# SPDX-License-Identifier: Apache-2.0
# DBSCAN boundary, PACK side: public output arrays, fitted-field
# publication (with the admission of a replaced owner), the fresh resident
# holder of a fit, the mint of the resident core rows, and predict's labels.
# UNPACK is bindings/dbscan_native_boundary_unpack.mojo. (The two were
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
from bindings.dbscan_native_boundary_unpack import (
    DBSCANConnection, DBSCANReferenceOwner, dbscan_holder_admit, dbscan_holder_empty, dbscan_release,
    dbscan_reference_owner,
)


def dbscan_fresh_holder(model: PythonObject) raises:
    dbscan_release(model)
    var fields = py_fields(model)
    lifecycle_admit_fields(fields)
    # A release can let another Python thread replace the namespace/holder.
    dbscan_holder_admit(py_get(fields, "_resident_holder"))
    py_setitem(fields, "_resident_holder", dbscan_holder_empty())


def dbscan_publish(fields: PythonObject, name: String, value: PythonObject, context: PythonObject) raises:
    # RUN can release the GIL. Admit a newly rebound old owner before its
    # last reference is dropped by a public field publication.
    lifecycle_admit_fields(fields)
    var old = py_get(fields, name)
    if old is not value:
        native_release_admit(old, context)
    py_setitem(fields, name, value)


def dbscan_array(shape: List[Int], typestr: String, context: PythonObject) raises -> PythonObject:
    var width = 1 if typestr == "|u1" else 4
    var layout = NativeArrayLayout(shape.copy(), width)
    var storage = native_array_zeros(native_array_code(typestr), layout.count)
    return classical_wrap_layout(storage, layout, typestr, context)


def dbscan_mint(model: PythonObject, connection: DBSCANConnection, refs: PythonObject, keys: PythonObject,
                 labels: PythonObject, rows: Int, features: Int, metric: Int, context: PythonObject) raises:
    if rows < 1:
        return
    try:
        _ = dbscan_reference_owner(model, connection, refs, keys, labels, rows, features, metric, context)
    except:
        dbscan_release(model)


def dbscan_reference_labels(model: PythonObject, binding: PythonObject, refs: PythonObject, keys: PythonObject,
    labels: PythonObject, query: PythonObject, rows: Int, features: Int, metric: Int, eps: Float64,
    context: PythonObject, resident: Bool = True) raises -> PythonObject:
    var connection = DBSCANConnection(binding)
    # Capture admitted query dimensions before a resident prepare/release
    # allows another Python thread to alter caller-visible metadata.
    var queries = Int(py=query.shape[0])
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var owner = Optional[PythonObject]()
    if resident:
        owner = py_present(dbscan_reference_owner(model, connection, refs, keys, labels, rows, features, metric, context))
    var dst = dbscan_array(List[Int](length=1, fill=queries), "<i4", context)
    try:
        if Bool(owner):
            var driver = owner.value().downcast_value_ptr[DBSCANReferenceOwner]()[].state.driver.copy()
            var request = DBSCANResidentQuery(Int(py=classical_address(query)), Int(py=classical_address(dst)),
                owner.value().downcast_value_ptr[DBSCANReferenceOwner]()[].state.handle, queries, features, metric, True, eps)
            with GILReleased(Python()):
                _ = driver.resident(request)
        else:
            # The old unused chosen-reference output was a private Python
            # Array. Native scratch keeps its original size and zero fill.
            var chosen = List[Int32](length=queries, fill=0)
            var request = DBSCANQueryRequest(Int(py=classical_address(refs)), Int(py=classical_address(keys)),
                Int(py=classical_address(labels)), Int(py=classical_address(query)), Int(py=classical_address(dst)),
                Int(chosen.unsafe_ptr()), rows, queries, features, metric, True, eps)
            with GILReleased(Python()):
                _ = connection.driver.query(request)
            _ = chosen^
    finally:
        _ = owner
        _ = connection
        _ = refs
        _ = keys
        _ = labels
        _ = query
        _ = dst
    return dst
