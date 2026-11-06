# SPDX-License-Identifier: Apache-2.0
# The spectral owner's sparse COO reader (bindings/cfb_spectral.mojo): the
# (row, col, data, shape) a SciPy matrix's `tocoo()` returns, read from the
# matrix's own stored buffers instead of calling into SciPy. An exact COO
# container hands back its stored coordinates (its `tocoo()` returns itself);
# an exact CSR/CSC container's pointer is expanded natively
# (core/cfb_sparse_read_impl.mojo), entries in stored order with duplicates
# kept, as `tocoo()` keeps them. Any other input, or a container whose
# storage is not well formed, returns None and the caller takes `tocoo()`,
# so SciPy's own refusals and the empty-matrix refusal are unchanged.
from std.python import Python, PythonObject
from bindings.py_balanced import py_type, py_contains
from bindings.native_loaded_module import native_loaded_module
from bindings.host_sparse_native import host_sparse_format, _sparse_fields, _sparse_indices
from bindings.classical_arrays import classical_address
from bindings.family_common import family_array
from core.cfb_sparse_read_impl import cfb_sparse_expand


def cfb_sparse_triples(source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`(row, col, data, shape)` of an exact SciPy COO/CSR/CSC input (row and
    col int32 for CSR/CSC, the stored arrays for COO; data the stored array;
    shape a tuple), or None when the caller must use the input's `tocoo()`."""
    var format = host_sparse_format(source)
    if format != "coo" and format != "csr" and format != "csc":
        return PythonObject(None)
    var b = native_loaded_module("builtins")
    var fields = _sparse_fields(source)
    if not py_contains(fields, PythonObject("_shape")) or not py_contains(fields, PythonObject("data")):
        return PythonObject(None)
    var shape = fields["_shape"]
    if py_type(shape) is not b.tuple or len(shape) != 2 or py_type(shape[0]) is not b.int or py_type(shape[1]) is not b.int:
        return PythonObject(None)
    var data = fields["data"]
    if format == "coo":
        if not py_contains(fields, PythonObject("coords")):
            return PythonObject(None)
        var coords = fields["coords"]
        if py_type(coords) is not b.tuple or len(coords) != 2:
            return PythonObject(None)
        return Python.tuple(coords[0], coords[1], data, shape)
    if not py_contains(fields, PythonObject("indptr")) or not py_contains(fields, PythonObject("indices")):
        return PythonObject(None)
    if py_type(data) is not context["ndarray"] or Int(py=data.ndim) != 1:
        return PythonObject(None)
    var csc = format == "csc"
    var rows = Int(py=shape[0])
    var columns = Int(py=shape[1])
    var major = columns if csc else rows
    var minor = rows if csc else columns
    var pointers = _sparse_indices(fields["indptr"], context)
    var indices = _sparse_indices(fields["indices"], context)
    var count = indices.count
    # An empty matrix keeps `tocoo()`: its refusal names the empty rows.
    if count < 1 or Int(py=data.size) != count or pointers.count != major + 1:
        return PythonObject(None)
    var major_out = family_array(Python.tuple(count), "<i4", context)
    var minor_out = family_array(Python.tuple(count), "<i4", context)
    var status = cfb_sparse_expand(pointers.address, pointers.strides[0], pointers.width, pointers.native,
                                   indices.address, indices.strides[0], indices.width, indices.native,
                                   major, minor, count,
                                   Int(py=classical_address(major_out)), Int(py=classical_address(minor_out)))
    _ = major_out  # keep the Python object alive through the native call above
    _ = minor_out  # keep the Python object alive through the native call above
    _ = pointers^
    _ = indices^
    _ = fields^
    if status != 0:
        return PythonObject(None)
    if csc:
        return Python.tuple(minor_out, major_out, data, shape)
    return Python.tuple(major_out, minor_out, data, shape)
