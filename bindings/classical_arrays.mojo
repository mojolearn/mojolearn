# SPDX-License-Identifier: Apache-2.0
# Native classical ABI preparation. Only admitted native buffers are read;
# dtype conversion is the shared integer-defined CastIn conversion.
from std.memory import bitcast
from std.python import Python, PythonObject
from core.cast_in import cast_in_kind, cast_f32_bits
from bindings.sequential_admission import sequential_admit_value, _ordinary_torch
from bindings.sequential_arrays import sequential_array, sequential_address
from bindings.sequential_inputs import sequential_input


def classical_array(rows: Int, cols: Int, context: PythonObject, wide: Bool = False) raises -> PythonObject:
    return sequential_array(Python.tuple(rows) if cols == 0 else Python.tuple(rows, cols),
                            context["float64"] if wide else context["float32"], context)


def classical_host_f32(source: PythonObject, ndim: Int, name: String, context: PythonObject) raises -> PythonObject:
    sequential_admit_value(source, context)
    var b = Python.import_module("builtins")
    var prepared = source
    if _ordinary_torch(b.type(source), context):
        if String(source.device.type) != "cpu":
            raise Error("TypeError: " + name + " must be a host buffer; copy device tensors to CPU explicitly")
        # Exact standard Tensor/Parameter methods are C entry points. User
        # tensor subclasses were refused by the admission check above.
        prepared = source.detach().numpy()
    var value = sequential_input(prepared, context)
    var nd = Int(py=value.ndim)
    if nd != ndim:
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
    var borrowed = value.dtype is context["float32"] and Bool(py=value.flags.c_contiguous)
    var result = value if borrowed else classical_array(rows, cols if ndim == 2 else 0, context)
    var out = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(result)))
    for r in range(rows):  # native input marshaling: input rows, never a model or training loop
        for c in range(cols):  # native input marshaling: numeric buffer items preserve exact cast bits
            var bytes = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addr + r * s0 + c * s1)
            var raw = UInt64(0)
            for k in range(item):
                raw |= UInt64(bytes[k]) << UInt64(k * 8)
            var bits = cast_f32_bits(raw, kind, item)
            if (bits & UInt32(0x7F800000)) == UInt32(0x7F800000):
                raise Error("Input " + name + " contains NaN or infinity")
            if not borrowed:
                out[r * cols + c] = bitcast[DType.float32](bits)
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
    sequential_admit_value(source, context)
    var raw = sequential_input(source, context)
    var scalar = Int(py=raw.ndim) == 0
    var value = classical_host_f32(source, 0 if scalar else 1, "sample_weight", context)
    if not scalar and Int(py=value.size) != rows:
        raise Error("sample_weight length differs from X")
    var out = classical_array(rows, 0, context) if scalar else value
    var input = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(value)))
    var output = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(out)))
    var any_positive = False
    for i in range(rows):  # native input marshaling: weight validation and scalar expansion
        var v = input[0 if scalar else i]
        if v < 0:
            raise Error("Negative values in data passed to `sample_weight`")
        any_positive = any_positive or v > 0
        if scalar:
            output[i] = v
    if not any_positive:
        raise Error("Sample weights must contain at least one non-zero number.")
    _ = value^
    return out
