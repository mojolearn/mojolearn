# SPDX-License-Identifier: Apache-2.0
# PCA query admission owns the original error order and one buffer metadata
# snapshot. Unsupported descriptors are never passed to numerical routines.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_contains
from bindings.classical_validate import classical_view, RESHAPE_HINT
from bindings.host_sparse_native import host_sparse_format, host_sparse_dense
from bindings.native_loaded_module import native_loaded_module
from bindings.native_buffer_owner import native_buffer_view
from bindings.sequential_arrays import sequential_array, sequential_address
from providers.python.array_metadata_pyapi import ArrayMetadata, array_shape_once, array_metadata_once
from core.cast_in import cast_in_kind
from core.family_cast_impl import family_cast_rows, family_all_finite_f32
from core.pca_exact_impl import pca_check_state


def pca_input_snapshot(source: PythonObject, context: PythonObject) raises -> ArrayMetadata:
    var format = host_sparse_format(source)
    var sparse = format.byte_length() > 0
    var value = host_sparse_dense(source, format, context) if sparse else source
    var view = classical_view(value, context, "X")
    var shape = array_shape_once(view, context)
    # Even unsupported native dtype/rank metadata is captured once. This
    # opt-in allows PCA's existing complex/rank/type refusals to run in their
    # established order before a CastIn is admitted for execution.
    var captured = array_metadata_once(view, shape^, "X", context,
                                      allow_non_native=True, allow_unsupported=True)
    var nd = len(captured.shape)
    var code = captured.dtype_number
    var numeric = (0 <= code and code <= 13) or code == 23
    var shown = String(py_call(native_loaded_module("builtins").repr, captured.python_shape()))
    if sparse:
        # The former sparse branch returned classical_host_f32 directly,
        # retaining its simpler rank/empty/finite messages.
        if nd != 2:
            raise Error("X must be 2-D")
        if captured.shape[0] < 1 or captured.shape[1] < 1:
            raise Error("X is empty")
    else:
        if code == 14 or code == 15 or code == 16:
            raise Error("Complex data not supported: PCA got a complex X.")
        if numeric:
            if nd < 2:
                raise Error("Expected 2D array, got " + (String("scalar") if nd == 0 else String("1D"))
                            + " array instead (shape=" + shown + "). " + String(RESHAPE_HINT))
            if nd == 2 and captured.shape[0] == 0:
                raise Error("Found array with 0 sample(s) (shape=" + shown + ") while a minimum of 1 is required by PCA.")
            if nd == 2 and captured.shape[1] == 0:
                raise Error("Found array with 0 feature(s) (shape=" + shown + ") while a minimum of 1 is required by PCA.")
        if nd != 2:
            raise Error("mojolearn: X must be 2-D, got " + String(nd) + "-D shape " + shown)
        if not numeric:
            # Only the failing path asks the admitted native exporter for
            # its PEP3118 spelling; it never executes or converts its values.
            var buffer = native_buffer_view(view)
            var spelling = String(py_call(native_loaded_module("builtins").repr, buffer.format))
            raise Error("TypeError: mojolearn: X has buffer format " + spelling + ", which is not a numeric array")
    if captured.shape[0] < 1 or captured.shape[1] < 1:
        # Preserve classical_host_f32's final guard for a mutable Array's
        # invalid negative dimensions as well as the sparse empty case.
        raise Error("X is empty")
    if not captured.native_byte_order:
        raise Error("TypeError: X must be in native byte order")
    if not captured.castable:
        # Recreate the original CastIn refusal from captured scalar metadata,
        # never rereading dtype or executing the placeholder descriptor.
        _ = cast_in_kind(code, captured.source.itemsize, "X")
        raise Error("PCA requires an executable native matrix descriptor")
    var owners = Python.tuple(source, value, captured.owner)
    if not captured.f32_c():
        var result = sequential_array(captured.python_shape(), context["float32"], context)
        var address = Int(py=sequential_address(result))
        var input = captured.source
        family_cast_rows(input.addr, input.s0, input.s1, input.rows, input.cols,
                         input.itemsize, input.kind,
                         MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address))
        captured.replace_f32(result, address)
    captured.owner = Python.tuple(owners, captured.owner)
    if not family_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=captured.source.addr), captured.count):
        raise Error("Input X contains NaN or infinity" + (String("") if sparse else String(" (PCA).")))
    return captured^


def pca_state_snapshot(fields: PythonObject, name: String, expected: List[Int],
                       context: PythonObject, finite: Bool = False,
                       nonnegative: Bool = False) raises -> ArrayMetadata:
    if not py_contains(fields, name):
        raise Error("PCA fitted state is missing " + name)
    var source = fields[name]
    var view = classical_view(source, context, name)
    var shape = array_shape_once(view, context)
    var captured = array_metadata_once(view, shape^, name, context,
                                      allow_non_native=True, allow_unsupported=True)
    var same_shape = len(captured.shape) == len(expected)
    if same_shape:
        for axis in range(len(expected)):  # small-loop(fitted vector/matrix dimensions)
            same_shape = same_shape and captured.shape[axis] == expected[axis]
    # No cast or contiguity repair is permitted for fitted state. The public
    # contract requires the supplied fitted arrays themselves to be C f32.
    if not same_shape or not captured.castable or not captured.native_byte_order or not captured.f32_c():
        raise Error("PCA requires C-order float32 " + name + " with the fitted shape")
    captured.owner = Python.tuple(source, captured.owner)
    if finite or nonnegative:
        pca_check_state(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=captured.source.addr),
                        captured.count, name, finite, nonnegative)
    return captured^
