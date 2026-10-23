# SPDX-License-Identifier: Apache-2.0
# The decomposition owners' input conversion (bindings/tsvd_native.mojo,
# bindings/ipca_native.mojo): X as C-order float32 with scikit-learn's input
# refusals, an exact SciPy sparse container densified from its storage, as
# the Python shells densified it before converting. Metadata reads and the
# shared typed CastIn conversion only.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call
from bindings.classical_arrays import classical_host_f32, classical_kind, classical_is_f32
from bindings.classical_validate import classical_view, _cast_x, RESHAPE_HINT
from bindings.host_sparse_native import host_sparse_format, host_sparse_dense
from bindings.pca_exact import pca_copied


def decomp_input(source: PythonObject, est: String, context: PythonObject) raises -> PythonObject:
    """X as C-order float32 (the caller's buffer when it already is one);
    refusals name `est`."""
    var format = host_sparse_format(source)
    if format.byte_length() > 0:
        var dense = host_sparse_dense(source, format, context)
        return classical_host_f32(dense, 2, "X", context)
    var view = classical_view(source, context, "X")
    var kind = classical_kind(view, context)
    if kind == "c":
        raise Error("Complex data not supported: " + est + " got a complex X.")
    if kind == "b" or kind == "i" or kind == "u" or kind == "f":
        var nd = Int(py=view.ndim)
        var shape = String(py_call(Python.import_module("builtins").repr, view.shape))
        if nd < 2:
            raise Error("Expected 2D array, got " + ("scalar" if nd == 0 else "1D") + " array instead (shape=" + shape + "). " + RESHAPE_HINT)
        if nd == 2 and Int(py=view.shape[0]) == 0:
            raise Error("Found array with 0 sample(s) (shape=" + shape + ") while a minimum of 1 is required by " + est + ".")
        if nd == 2 and Int(py=view.shape[1]) == 0:
            raise Error("Found array with 0 feature(s) (shape=" + shape + ") while a minimum of 1 is required by " + est + ".")
    var x = _cast_x(view, est, context)
    _ = view^
    return x


def decomp_copied(source: PythonObject, context: PythonObject) raises -> Bool:
    """Whether X had to be copied to C-order float32 (a densified sparse X
    counts as the dense array its storage holds)."""
    var format = host_sparse_format(source)
    if format.byte_length() > 0:
        return not classical_is_f32(host_sparse_dense(source, format, context), context)
    return pca_copied(source, context)
