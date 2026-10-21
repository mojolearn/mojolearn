# SPDX-License-Identifier: Apache-2.0
# Native classical ABI preparation. Only admitted native buffers are read;
# dtype conversion is the shared integer-defined CastIn conversion.
from std.python import Python, PythonObject
from core.cast_in import cast_in_kind
from core.shell_impl.classical_arrays_impl import classical_cast_rows, classical_check_weights
from bindings.classical_warnings import classical_warn_column
from bindings.sequential_admission import sequential_admit_value, _ordinary_torch
from bindings.sequential_arrays import sequential_array, sequential_address
from bindings.sequential_inputs import sequential_input
from bindings.native_torch_admission import native_torch_numpy
from bindings.classical_pandas import native_pandas_exact, native_pandas_f32, classical_admit_value


def classical_array(rows: Int, cols: Int, context: PythonObject, wide: Bool = False) raises -> PythonObject:
    return sequential_array(Python.tuple(rows) if cols == 0 else Python.tuple(rows, cols),
                            context["float64"] if wide else context["float32"], context)


def classical_host_f32(source: PythonObject, ndim: Int, name: String, context: PythonObject, column_target: Bool = False) raises -> PythonObject:
    classical_admit_value(source, context)
    var b = Python.import_module("builtins")
    var prepared = source
    if native_pandas_exact(source):
        prepared = native_pandas_f32(source, context)
    elif _ordinary_torch(b.type(source), context):
        prepared = native_torch_numpy(source)
    var value = sequential_input(prepared, context)
    var nd = Int(py=value.ndim)
    var column = column_target and ndim == 1 and nd == 2 and Int(py=value.shape[1]) == 1 and b.type(source) is not b.list and b.type(source) is not b.tuple and b.type(source) is not b.memoryview
    if column:
        classical_warn_column(context)
    if nd != ndim and not column:
        raise Error(name + " must be " + String(ndim) + "-D")
    var rows = 1 if nd == 0 else Int(py=value.shape[0])
    var cols = 1 if nd < 2 else Int(py=value.shape[1])
    if rows < 1 or cols < 1:
        raise Error(name + " is empty")
    if not Bool(py=value.dtype.isnative):
        raise Error("TypeError: " + name + " must be in native byte order")
    var item = Int(py=value.dtype.itemsize)
    var kind = cast_in_kind(Int(py=value.dtype.num), item, name)
    var s0 = 0 if nd == 0 else Int(py=value.strides[0])
    var s1 = 0 if nd < 2 else Int(py=value.strides[1])
    var addr = Int(py=sequential_address(value))
    var borrowed = not column and value.dtype is context["float32"] and Bool(py=value.flags.c_contiguous)
    var result = value if borrowed else classical_array(rows, cols if ndim == 2 else 0, context)
    var out = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(result)))
    classical_cast_rows(addr, s0, s1, rows, cols, item, kind, name, borrowed, out)
    _ = value^
    _ = prepared^
    return result


def classical_word(value: PythonObject, index: Int = 0) raises -> Float64:
    var address = Int(py=sequential_address(value))
    if Int(py=value.dtype.itemsize) == 8:
        return MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=address)[index]
    return Float64(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)[index])


def classical_host_weights(source: PythonObject, rows: Int, context: PythonObject) raises -> PythonObject:
    if source is None:
        return PythonObject(None)
    classical_admit_value(source, context)
    var prepared = native_pandas_f32(source, context) if native_pandas_exact(source) else source
    var raw = sequential_input(prepared, context)
    var scalar = Int(py=raw.ndim) == 0
    var value = classical_host_f32(source, 0 if scalar else 1, "sample_weight", context)
    if not scalar and Int(py=value.size) != rows:
        raise Error("sample_weight length differs from X")
    var out = classical_array(rows, 0, context) if scalar else value
    var input = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(value)))
    var output = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(out)))
    var any_positive = classical_check_weights(input, output, rows, scalar)
    if not any_positive:
        raise Error("Sample weights must contain at least one non-zero number.")
    _ = value^
    return out
