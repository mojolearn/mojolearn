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
from core.admitted_input import admitted_cast_f32
from core.host_finite_scan import host_all_finite_f32

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
    var count = 1
    # The shape text names refusals only: it is written natively (Python's
    # tuple repr, "(n,)" for one axis), and no Python list or tuple is built
    # on the success path unless a cast needs the shape (p3 rank 420).
    var shown = String("(")
    for axis in range(nd):  # driver-loop: over captured native dimensions
        count *= shape[axis]
        if axis > 0:
            shown += ", "
        shown += String(shape[axis])
    shown += ",)" if nd == 1 else ")"
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
    # The cast and the finiteness scan are one pass (N352): a converted X
    # is scanned while it is written; an X already C float32 is scanned once.
    var finite = True
    if not snapshot.f32_c():
        var tuple_shape = Python.tuple(snapshot.shape[0]) if nd == 1 else Python.tuple(snapshot.shape[0], snapshot.shape[1])
        var result = family_array(tuple_shape, "<f4", context)
        var address = Int(py=classical_address(result))
        var src = snapshot.source
        finite = admitted_cast_f32(src.addr, src.s0, src.s1, src.rows, src.cols, src.itemsize, src.kind,
                                         _F32(unsafe_from_address=address))
        snapshot.replace_f32(result, address)
    else:
        finite = host_all_finite_f32(_F32(unsafe_from_address=snapshot.source.addr), snapshot.count)
    # Pin the captured source storage even when a converted output is owned.
    snapshot.owner = Python.tuple(original_owner, snapshot.owner)
    if not finite:
        if public_target and public_name != "":
            raise Error("Input y contains NaN or infinity (" + public_name + ").")
        raise Error(name + ": input must be finite; NaN/inf are unsupported")
    return snapshot^
