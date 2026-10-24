# SPDX-License-Identifier: Apache-2.0
# scikit-learn's input checks for the native LinearRegression/Ridge owner, in
# the order and with the wording the public estimators have always raised:
# sparse, complex, rank and empty X; a missing, complex, column or non-finite
# y; inconsistent lengths; the sample_weight shape. Only array metadata is read
# here; values are read by the typed CastIn conversion.
from bindings.native_loaded_module import native_loaded_module
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type
from bindings.classical_arrays import classical_array, classical_host_f32, classical_input, classical_kind, classical_address
from bindings.classical_pandas import native_pandas_exact, native_pandas_f32, classical_admit_value
from bindings.sequential_admission import _ordinary_torch
from bindings.native_torch_admission import native_torch_numpy
from core.classical_arrays_impl import classical_check_weights

comptime RESHAPE_HINT = "Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample."


def _repr(value: PythonObject) raises -> String:
    return String(py_call(native_loaded_module("builtins").repr, value))


def classical_estimator_name(model: PythonObject) raises -> String:
    var b = native_loaded_module("builtins")
    var cls = model if py_type(model) is b.type else py_type(model)
    return String(cls.__name__)


def classical_view(source: PythonObject, context: PythonObject, name: String = "input") raises -> PythonObject:
    """An array view of an admitted input (pandas and Torch storage read
    natively), before any cast: an ndarray with NumPy, an Array without."""
    classical_admit_value(source, context)
    var prepared = source
    if native_pandas_exact(source):
        prepared = native_pandas_f32(source, context)
    elif _ordinary_torch(py_type(source), context):
        prepared = native_torch_numpy(source)
    return classical_input(prepared, context, name)


def classical_refuse_sparse(source: PythonObject, est: String) raises:
    # scipy.sparse containers are recognized by their defining module, so no
    # attribute or property of the container runs.
    var module = py_type(source).__module__
    if py_type(module) is native_loaded_module("builtins").str and Bool(py=py_call(module.startswith, "scipy.sparse")):
        raise Error("TypeError: Sparse data was passed for X, but dense data is required. Use '.toarray()' to convert to a dense numpy array. (" + est + " takes dense X only.)")


def _numeric(view: PythonObject, context: PythonObject) raises -> Bool:
    var kind = classical_kind(view, context)
    return kind == "b" or kind == "i" or kind == "u" or kind == "f"


def _cast_x(view: PythonObject, est: String, context: PythonObject) raises -> PythonObject:
    var nd = Int(py=view.ndim)
    if nd != 2:
        raise Error("mojolearn: X must be 2-D, got " + String(nd) + "-D shape " + _repr(view.shape))
    if not _numeric(view, context):
        raise Error("TypeError: mojolearn: X has buffer format " + _repr(py_call(native_loaded_module("builtins").memoryview, view).format) + ", which is not a numeric array")
    try:
        return classical_host_f32(view, 2, "X", context)
    except error:
        if String(error) == "Input X contains NaN or infinity":
            raise Error("Input X contains NaN or infinity (" + est + ").")
        raise Error(String(error))


def classical_check_x(source: PythonObject, est: String, context: PythonObject) raises -> PythonObject:
    """X for fit, predict or score: refused with scikit-learn's wording, then
    cast to C-ordered float32 (the input itself when it already is one)."""
    classical_refuse_sparse(source, est)
    var view = classical_view(source, context, "X")
    var kind = classical_kind(view, context)
    if kind == "c":
        raise Error("Complex data not supported: " + est + " got a complex X.")
    if _numeric(view, context):
        var nd = Int(py=view.ndim)
        if nd < 2:
            raise Error("Expected 2D array, got " + ("scalar" if nd == 0 else "1D") + " array instead (shape=" + _repr(view.shape) + "). " + RESHAPE_HINT)
        if nd == 2 and Int(py=view.shape[0]) == 0:
            raise Error("Found array with 0 sample(s) (shape=" + _repr(view.shape) + ") while a minimum of 1 is required by " + est + ".")
        if nd == 2 and Int(py=view.shape[1]) == 0:
            raise Error("Found array with 0 feature(s) (shape=" + _repr(view.shape) + ") while a minimum of 1 is required by " + est + ".")
    var x = _cast_x(view, est, context)
    _ = view^
    return x


def classical_check_y(source: PythonObject, rows: Int, est: String, context: PythonObject) raises -> PythonObject:
    """The fit target: present, real, one column (a column vector warns and is
    raveled), finite and as long as X."""
    if source is None:
        raise Error(est + " requires y to be passed, but the target y is None.")
    var b = native_loaded_module("builtins")
    var view = classical_view(source, context, "y")
    if classical_kind(view, context) == "c":
        raise Error("Complex data not supported: " + est + " got a complex y.")
    var nd = Int(py=view.ndim)
    var column = nd == 2 and Int(py=view.shape[1]) == 1 and py_type(source) is not b.list and py_type(source) is not b.tuple and py_type(source) is not b.memoryview
    var y = PythonObject(None)
    if nd == 1 or column:
        try:
            y = classical_host_f32(view, 1, "y", context, column_target=column)
        except error:
            if String(error) == "Input y contains NaN or infinity":
                raise Error("Input y contains NaN or infinity (" + est + ").")
            raise Error(String(error))
    var length = 1 if nd == 0 else Int(py=view.shape[0])
    if length != rows:
        raise Error("Found input variables with inconsistent numbers of samples: [" + String(rows) + ", " + String(length) + "]")
    if y is None:
        raise Error("mojolearn " + est + " currently requires one target")
    _ = view^
    return y


def classical_weight_refusal(source: PythonObject, error: Error) raises -> Error:
    """The refusal of a sample_weight the shared admission refused. That
    refusal names no accepted type; say what a weight may be. An object that
    only defines __array__ (scikit-learn's _NotAnArray) is never converted:
    no Python protocol runs in a fit."""
    if String(error).startswith("TypeError: Sequential requires native input types"):
        return Error("TypeError: sample_weight must be None, a number, a NumPy array, a buffer "
                     + "(memoryview, array.array, bytearray), a list or tuple of numbers, a pandas "
                     + "Series or a Torch tensor; got " + String(py_type(source).__name__)
                     + ", which is not read because it only converts through Python-defined "
                     + "methods such as __array__")
    return Error(String(error))


def classical_check_weights_input(source: PythonObject, rows: Int, context: PythonObject) raises -> PythonObject:
    """sample_weight as rows float32 words: None stays None, a scalar fills
    every row; scikit-learn's shape, finiteness, sign and all-zero refusals."""
    if source is None:
        return PythonObject(None)
    var view = PythonObject(None)
    try:
        view = classical_view(source, context, "sample_weight")
    except error:
        var refusal = classical_weight_refusal(source, error)
        raise refusal
    var nd = Int(py=view.ndim)
    if nd > 1:
        raise Error("Sample weights must be 1D array or scalar")
    var scalar = nd == 0
    if not scalar and Int(py=view.shape[0]) != rows:
        raise Error("sample_weight.shape == (" + String(Int(py=view.shape[0])) + ",), expected (" + String(rows) + ",)!")
    var value = PythonObject(None)
    try:
        value = classical_host_f32(view, 0 if scalar else 1, "sample_weight", context)
    except error:
        if String(error) == "Input sample_weight contains NaN or infinity":
            raise Error("Input sample_weight contains NaN or infinity.")
        raise Error(String(error))
    var out = classical_array(rows, 0, context) if scalar else value
    var input = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(value)))
    var output = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out)))
    var any_positive = classical_check_weights(input, output, rows, scalar)
    if not any_positive:
        raise Error("Sample weights must contain at least one non-zero number.")
    _ = value^
    _ = view^
    return out


def classical_check_width(cols: Int, fitted: Int, est: String) raises:
    if cols != fitted:
        raise Error("X has " + String(cols) + " features, but " + est + " is expecting " + String(fitted) + " features as input.")
