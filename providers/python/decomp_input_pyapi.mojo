# SPDX-License-Identifier: Apache-2.0
# Operation-local decomposition input admission. Keep the original native
# storage owner and the captured CastIn metadata through the complete operation.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call
from bindings.classical_validate import classical_view
from bindings.classical_arrays import classical_address
from bindings.family_common import family_array
from bindings.host_sparse_native import host_sparse_format, host_sparse_dense
from bindings.native_loaded_module import native_loaded_module
from providers.python.array_metadata_pyapi import ArrayMetadata, array_shape_once, array_metadata_once
from core.family_cast_impl import family_cast_rows, family_all_finite_f32

comptime _F32 = MutPointer[Float32, MutUntrackedOrigin]
comptime _RESHAPE = ". Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample."


def decomp_input_snapshot(source: PythonObject, name: String, context: PythonObject,
                          ndim: Int = 2, public_name: String = "",
                          public_target: Bool = False, sparse_allowed: Bool = True) raises -> ArrayMetadata:
    var value = source
    var format = host_sparse_format(source)
    if format.byte_length() > 0:
        if not sparse_allowed:
            raise Error("TypeError: Sparse data was passed for X, but dense data is required. Use '.toarray()' to convert to a dense numpy array. (" + public_name + " takes dense X only.)")
        value = host_sparse_dense(source, format, context)
    # classical_view admits the exact input type before any input protocol.
    var view = classical_view(value, context, name)
    var shape = array_shape_once(view, context)
    var nd = len(shape)
    var words = Python.list()
    var count = 1
    for dimension in shape:  # driver-loop: over captured native dimensions
        _ = py_call(words.append, dimension)
        count *= dimension
    var tuple_shape = py_call(native_loaded_module("builtins").tuple, words)
    var shown = String(py_call(native_loaded_module("builtins").repr, tuple_shape))
    var complex_error = "TypeError: mojolearn: " + name + " has a non-numeric dtype"
    if public_name != "":
        complex_error = "Complex data not supported: " + public_name + " got a complex " + (String("y") if public_target else String("X")) + "."
    # Public complex refusal precedes rank/empty checks. Retain all other
    # capture errors until after those shape checks, without rereading dtype.
    var captured = Optional[ArrayMetadata]()
    var failure = String("")
    var rows = shape[0] if nd > 0 else 1
    try:
        captured = array_metadata_once(view, shape^, name, context, complex_error=complex_error)
    except error:
        failure = String(error)
        if failure.startswith("Complex data not supported:"):
            raise error^
    if public_name != "" and not public_target and nd < 2:
        raise Error("Expected 2D array, got " + (String("scalar") if nd == 0 else String("1D"))
                    + " array instead (shape=" + shown + ")" + String(_RESHAPE))
    if public_name != "" and not public_target and nd == 2 and count == 0:
        var what = String("sample(s)") if rows == 0 else String("feature(s)")
        raise Error("Found array with 0 " + what + " (shape=" + shown + ") while a minimum of 1 is required by " + public_name + ".")
    if (ndim >= 0 and nd != ndim) or nd == 0 or nd > 2:
        var expected = ndim if ndim >= 0 else 2
        raise Error("mojolearn: " + name + " must be " + String(expected) + "-D, got " + String(nd) + "-D shape " + shown
                    + (String(_RESHAPE) if expected == 2 and nd < 2 else String("")))
    if count == 0:
        var what = String("feature(s)") if nd == 2 and rows > 0 else String("sample(s)")
        raise Error("mojolearn: " + name + " is empty, shape " + shown + ": Found array with 0 " + what
                    + " (shape=" + shown + ") while a minimum of 1 is required.")
    if failure != "":
        if failure.startswith(name + ": dtype number"):
            raise Error("TypeError: mojolearn: " + name + " has a non-numeric dtype")
        raise Error(failure)
    var snapshot = captured.value().copy()
    var original_owner = Python.tuple(source, snapshot.owner)
    if not snapshot.f32_c():
        var result = family_array(tuple_shape, "<f4", context)
        var address = Int(py=classical_address(result))
        var src = snapshot.source
        family_cast_rows(src.addr, src.s0, src.s1, src.rows, src.cols, src.itemsize, src.kind,
                         _F32(unsafe_from_address=address))
        snapshot.replace_f32(result, address)
    # Pin the captured source storage even when a converted output is owned.
    snapshot.owner = Python.tuple(original_owner, snapshot.owner)
    if not family_all_finite_f32(_F32(unsafe_from_address=snapshot.source.addr), snapshot.count):
        if public_target and public_name != "":
            raise Error("Input y contains NaN or infinity (" + public_name + ").")
        raise Error(name + ": input must be finite; NaN/inf are unsupported")
    return snapshot^
