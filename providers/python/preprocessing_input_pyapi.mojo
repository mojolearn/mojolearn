# SPDX-License-Identifier: Apache-2.0
# One operation-local input snapshot for StandardScaler and MinMaxScaler.
# Only this edge reads the user's mutable array metadata. The fit/transform
# stages share a frozen Array view and retain its original buffer owners.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_setattr, py_setitem
from bindings.native_loaded_module import native_loaded_module
from bindings.native_buffer_owner import native_buffer_view, native_storage_allocate
from bindings.classical_arrays import classical_input, classical_is_array, classical_is_f32, classical_address, classical_typed
from bindings.classical_pandas import classical_admit_value, native_pandas_exact, native_pandas_f32
from bindings.native_torch_admission import native_torch_numpy
from bindings.sequential_admission import _ordinary_torch, sequential_admit_value
from providers.python.array_metadata_pyapi import ArrayMetadata, array_shape_once, array_metadata_once, array_slot_once
from core.scaler_fit_impl import scaler_cast

alias F32P = MutPointer[Float32, MutUntrackedOrigin]


struct ScalerInputSnapshot(Movable, Writable):
    var original: PythonObject
    var metadata: ArrayMetadata
    var values: PythonObject
    var write_view: PythonObject

    def __init__(out self, original: PythonObject, var metadata: ArrayMetadata,
                 values: PythonObject, write_view: PythonObject):
        self.original = original
        self.metadata = metadata^
        self.values = values
        self.write_view = write_view

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ScalerInputSnapshot")


def _frozen_values(metadata: ArrayMetadata, view: PythonObject, context: PythonObject) raises -> PythonObject:
    # This view owns its pinned storage independently of the caller's Array
    # slots. Its shape, dtype and address are supplied from the snapshot.
    var result = native_storage_allocate(context["array_class"])
    var n = metadata.source.rows
    var d = metadata.source.cols
    py_setattr(result, "_store", PythonObject(None))
    py_setattr(result, "_base", metadata.keep)
    py_setattr(result, "_pin", metadata.owner)
    py_setattr(result, "_mv", view)
    py_setattr(result, "_addr", PythonObject(metadata.source.addr))
    py_setattr(result, "_readonly", PythonObject(metadata.readonly))
    py_setattr(result, "shape", Python.tuple(n, d))
    py_setattr(result, "dtype", PythonObject("<f4"))
    py_setattr(result, "ndim", PythonObject(2))
    py_setattr(result, "size", PythonObject(n * d))
    py_setattr(result, "nbytes", PythonObject(n * d * 4))
    py_setattr(result, "itemsize", PythonObject(4))
    py_setattr(result, "strides", Python.tuple(d * 4, 4))
    py_setattr(result, "order", PythonObject("C"))
    return result


def scaler_input_snapshot(source: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_admit_value(source, context)
    var b = native_loaded_module("builtins")
    var pandas = native_pandas_exact(source)
    var prepared = source
    if pandas:
        prepared = native_pandas_f32(source, context, True, True)
    elif _ordinary_torch(py_type(source), context):
        prepared = native_torch_numpy(source)
    # Keep an Array in its native representation: converting it to ndarray
    # here would read the caller's mutable slots before the snapshot.
    var value = prepared if classical_is_array(prepared, context) else classical_input(prepared, context)
    var sequence = py_type(source) is b.list or py_type(source) is b.tuple or py_type(source) is b.range
    var shape = array_shape_once(value, context)
    if len(shape) != 2:
        # Preserve the dtype-before-rank refusal on an invalid input. A
        # successful operation only enters array_metadata_once below.
        if not sequence and not classical_is_f32(value, context):
            if classical_is_array(value, context) or String(value.dtype.kind) != "f" or Int(py=value.itemsize) != 4:
                raise Error("TypeError: Scaler input must have dtype float32")
        raise Error("Scaler requires a nonempty two-dimensional input")
    var metadata = array_metadata_once(value, shape^, "input", context, True,
        f32_error="" if sequence else "TypeError: Scaler input must have dtype float32")
    var n = metadata.source.rows
    var d = metadata.source.cols
    if n < 1 or d < 1:
        raise Error("Scaler requires a nonempty two-dimensional input")
    if n > 2147483647 // d:
        raise Error("Scaler exceeds the native Int32 indexing bound")
    var original_owners = metadata.owner
    var view = original_owners[3] if metadata.is_array else native_buffer_view(value)
    if py_type(view) is not b.memoryview:
        raise Error("TypeError: scaler input storage requires an exact native memoryview")
    # Use the established classical exporter table, which includes mmap
    # buffers used by preprocessing's mapped arenas as well as NumPy/ctypes.
    sequential_admit_value(view, context)
    var write_view = PythonObject(None)
    if not pandas and not sequence:
        # Without NumPy, classical_input may have copied a memoryview to a
        # native Array. Copy-back still targets the original exported view.
        var destination = view if classical_is_array(source, context) else native_buffer_view(prepared)
        if py_type(destination) is not b.memoryview:
            raise Error("TypeError: scaler copy-back requires an exact native memoryview")
        sequential_admit_value(destination, context)
        var format = String(destination.format)
        if (not Bool(py=destination.readonly) and Bool(py=destination.c_contiguous)
                and Int(py=destination.itemsize) == 4
                and (format == "f" or format == "<f" or format == "=f" or format == "@f")
                and (not classical_is_array(source, context) or (metadata.c_order and not metadata.readonly))):
            write_view = destination
    if not metadata.f32_c() or not metadata.native_byte_order:
        var output = classical_typed(Python.tuple(n, d), "<f4", context["output_context"])
        var address = Int(py=classical_address(output))
        scaler_cast(metadata.source.addr, metadata.source.s0, metadata.source.s1, n, d,
                    metadata.source.itemsize, metadata.source.kind, metadata.native_byte_order,
                    F32P(unsafe_from_address=address))
        metadata.replace_f32(output, address)
        view = array_slot_once(output, "_mv", context)
    # C-order float32 bytes are a stable, one-dimensional native buffer view.
    view = py_call(py_call(view.cast, "B").cast, "f")
    var values = _frozen_values(metadata, view, context)
    # Preserve converted/prepared storage as well as the immutable original
    # owner tuple; neither depends on mutable caller Array slots afterwards.
    py_setattr(values, "_pin", Python.tuple(original_owners, metadata.owner, prepared))
    return PythonObject(alloc=ScalerInputSnapshot(source, metadata^, values, write_view))


def scaler_snapshot_write_back(snapshot: PythonObject, output: PythonObject, copy: Bool) raises -> PythonObject:
    var captured = snapshot.downcast_value_ptr[ScalerInputSnapshot]()
    if copy or captured[].write_view is None:
        return output
    var destination = py_call(captured[].write_view.cast, "B")
    var payload = py_call(output._mv.cast, "B")
    py_setitem(destination, py_call(native_loaded_module("builtins").slice, PythonObject(None)), payload)
    # write_view pins the original storage and its metadata for this whole
    # operation; never reacquire source._mv or source.shape after execution.
    return captured[].original


def scaler_operation_input(source: PythonObject, context: PythonObject,
                           cache: PythonObject = PythonObject(None)) raises -> PythonObject:
    # cache is an operation-local native list, never part of model state or
    # exposed to callers. Lazy capture preserves validation/refit ordering.
    if cache is None:
        return scaler_input_snapshot(source, context)
    if len(cache) == 0:
        _ = py_call(cache.append, scaler_input_snapshot(source, context))
    return cache[0]
