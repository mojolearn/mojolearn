# SPDX-License-Identifier: Apache-2.0
# Complete row-resample boundary: mode, admission, frozen metadata, kernels,
# and finished dtype-preserving rows. Arithmetic stays in existing kernels.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get
from bindings.native_loaded_module import native_loaded_module
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_pandas import native_pandas_exact
from bindings.native_torch_admission import native_torch_exact, native_torch_numpy
from bindings.host_inference_scalars import host_inference_scalar
from providers.python.estimator_binding_pyapi import estimator_binding_mode
from providers.python.resample_rows_pyapi import ResampleRows, resample_rows_indices, resample_rows_gpu, resample_rows_narrow
from providers.python.resample_pandas_pyapi import resample_pandas_array


def resample_rows_operation(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    classical_connect_numpy(context)
    # Public calls reserve slot zero; the shared import-connected provider
    # admits the requested mode and reads native compile-time mode once.
    var binding = estimator_binding_mode(args[7], "_mojolearn_resample", context)
    if args[5] is not None:
        raise Error("mojolearn resample: stratify= is refused by name (resample/NOT_IMPLEMENTED.tsv): "
                    + "scikit-learn's per-class allocation (_approximate_mode) breaks ties with its RNG stream")
    if args[6] is not None:
        raise Error("mojolearn resample: sample_weight= is refused by name (resample/NOT_IMPLEMENTED.tsv): "
                    + "a weighted draw is an inverse-CDF lookup over a float cumulative sum not yet pinned")
    var arrays = args[1]
    if py_type(arrays) is not b.tuple:
        raise Error("TypeError: resample arguments must be the native argument tuple")
    if len(arrays) == 0:
        return PythonObject(None)
    var rows = List[ResampleRows]()
    var lengths = Python.list()
    for i in range(len(arrays)):  # driver-loop: capture each original input and metadata once, retaining its owners
        var original = arrays[i]
        var value = original
        var gpu = py_type(original) is py_get(context, "ndarray")
        if native_pandas_exact(original):
            value = resample_pandas_array(original, context)
        elif native_torch_exact(py_type(original)):
            value = native_torch_numpy(original)
        rows.append(ResampleRows(value, context, gpu))
        py_call(lengths.append, rows[len(rows) - 1].n)
    var n = rows[0].n
    for i in range(1, len(rows)):  # driver-loop: all row lengths are compared from retained native metadata
        if rows[i].n != n:
            raise Error("mojolearn resample: Found input variables with inconsistent numbers of samples: " + String(py=py_call(b.repr, lengths)))
    if args[3] is not None and py_type(args[3]) is not b.int:
        raise Error("TypeError: mojolearn resample: n_samples must be an int")
    if py_type(args[4]) is not b.int:
        raise Error("TypeError: mojolearn resample: random_state must be an int")
    var count = n if args[3] is None else Int(py=args[3])
    var seed = args[4]
    var replace_type = py_type(args[2])
    if replace_type is not b.list and replace_type is not b.tuple and replace_type is not b.dict:
        host_inference_scalar(args[2], context, True)
    var replace = Bool(py=args[2])
    if count < 0 or (not replace and count > n):
        # Preserve the numerical owner's existing size refusal before gather
        # output allocation, as the former public operation did.
        _ = resample_rows_indices(binding, n, count, replace, seed, context)
    var result = resample_rows_gpu(rows, binding, n, count, replace, seed, context)
    if result is None and replace:
        result = resample_rows_narrow(rows, binding, n, count, seed, context)
    if result is None:
        var indices = resample_rows_indices(binding, n, count, replace, seed, context)
        result = Python.list()
        for i in range(len(rows)):  # driver-loop: return finished rows in the caller's argument order
            py_call(result.append, rows[i].take(indices, context))
        _ = indices^
    _ = rows^
    return result[0] if len(arrays) == 1 else result
