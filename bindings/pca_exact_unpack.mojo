# SPDX-License-Identifier: Apache-2.0
# PCA helpers shared by the native PCA owner (bindings/pca_fit_native.mojo,
# bindings/pca_query_native.mojo) and the decomposition doors: the
# input_copied_ rule, the compiled-entry refusal and too-small refusal text.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_get, py_type
from bindings.sequential_inputs import sequential_input
from bindings.classical_arrays_door import classical_dtype_is

# TOMBSTONE: pca_exact_fit, pca_input, pca_keep_fit, pca_prefix, pca_solver_checks and PCADimensions (the pre-owner PCA exact door; zero callers since bindings/pca_fit_native.mojo owns fit) deleted 2026-11-08 by lane a3-admit-decomp-prep-metrics; code recoverable at f7d10d2b09.
# Restore: experiments/removed/A3_ADMISSION_HELPERS.patch.


def pca_copied(source: PythonObject, context: PythonObject) raises -> Bool:
    var b = native_loaded_module("builtins")
    # NumPy's keys are in the context only once NumPy is loaded
    var ndarray = py_get(context, "ndarray")
    if (ndarray is not None and py_type(source) is ndarray) or py_type(source) is context["array_class"]:
        # An Array already exposes its native metadata. sequential_input is
        # the ndarray adapter and requires NumPy's connected types.
        var view = source if ndarray is None else sequential_input(source, context)
        var f32 = py_get(context, "float32")
        var is_f32 = classical_dtype_is(view.dtype, f32, "<f4") if f32 is not None else (String(py=py_call(b.str, view.dtype)) == "<f4")
        # an ndarray's flags object, or an Array's flags dict
        var flags = view.flags
        var contiguous = (Bool(py=flags["C_CONTIGUOUS"]) if py_type(flags) is b.dict
                          else Bool(py=flags.c_contiguous))
        return not (is_f32 and contiguous)
    if py_type(source) is b.memoryview or py_type(source) is context["array_type"] or py_type(source) is b.bytearray:
        var view = py_call(b.memoryview, source)
        return not (Int(py=view.itemsize) == 4 and String(view.format) == "f" and Bool(py=view.c_contiguous))
    # Nested native sequences are materialized. Pandas/Torch publication and
    # feature-name semantics remain outside this private dense-buffer slice.
    return True
