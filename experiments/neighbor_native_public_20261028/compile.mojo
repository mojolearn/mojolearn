# Focused CPU compilation only: do not run this fixture.
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from bindings.full_reset_constructor_native import full_reset_constructor_c
from bindings.full_reset_params_native import full_reset_params_c
from bindings.neighbor_query_native import neighbor_query_c, neighbor_query_attribute_c
from bindings.neighbors_knn_native import neighbors_knn_operation
from bindings.neighbors_local_native import neighbors_local_operation
from bindings.neighbors_ann_native import neighbors_ann_operation

@export
def compile_neighbor_constructor(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return full_reset_constructor_c(py_self, args)

@export
def compile_neighbor_setter(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return full_reset_params_c(py_self, args)

@export
def compile_neighbor_query(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return neighbor_query_c(py_self, args)

@export
def compile_neighbor_attribute(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return neighbor_query_attribute_c(py_self, args)

@export
def compile_neighbor_knn(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return neighbors_knn_operation(model, operation, args, context)

@export
def compile_neighbor_local(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return neighbors_local_operation(model, operation, args, context)

@export
def compile_neighbor_ann(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return neighbors_ann_operation(model, operation, args, context)

from bindings.host_archive_native import host_archive_build

@export
def compile_neighbor_archive(path: PythonObject, context: PythonObject, cls: PythonObject) raises -> PythonObject:
    return host_archive_build(path, context, cls)
