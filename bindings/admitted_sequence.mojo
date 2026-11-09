# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE ONE ADMISSION PATH, rank-N members: the sequence, time-series and
# neural doors (TSA, forecast, embedding, xcnn, xseq_neural, the byte-LM
# trainers). Plan docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md section
# 3.3 row A4 (lane a4-admit-sequence-tsa-neural, 2026-11-08); the rank 0-2
# X / y / sample_weight doors are bindings/admitted_input.mojo (lane A0).
#
#   admitted_ndarray_f32(shape, count, context) -> ndarray
#                     the uninitialised float32 ndarray a cast fills
#   admit_nd_f32(value, name, copy, context, double_ints) -> ndarray
#                     `np.array(value, dtype=float32, order="C", copy=copy)`
#                     of an admitted ndarray: the array itself when it is
#                     C-order float32 and no copy is asked for (borrowed),
#                     otherwise ONE parallel cast into uninitialised storage
#                     (the former xseq_f32)
#   admit_nd_owned_f32(source, dims, name, context) -> Array
#                     the model's own C-order float32 copy of an input of
#                     exactly `dims`, refused unless finite, in one pass (the
#                     former training_f32_cast_owned + xseq_all_finite_f32)
#
# Every cast is core/admitted_input_nd.mojo `admitted_cast_nd_f32` (row tasks
# across the host's cores, a typed load per item, the finite test in the same
# pass). Numerics: none; the same float32 words reach the same kernels.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call
from bindings.classical_pandas import classical_admit_value
from bindings.classical_arrays import classical_input, classical_address, classical_cast_kind
from bindings.sequential_arrays import sequential_address
from bindings.native_array_storage import native_scratch_bytes, native_array_zeros
from bindings.arena_materialize_native import _owned
from core.cast_in import cast_in_kind
from core.admitted_input_nd import admitted_cast_nd_f32

comptime _AS_F32P = MutPointer[Float32, MutUntrackedOrigin]


def admitted_ndarray_f32(shape: PythonObject, count: Int, context: PythonObject) raises -> PythonObject:
    """A C-order float32 ndarray of `shape` (`count` words) whose contents
    are NOT initialised: the destination of one cast that writes every word
    before anything reads it. It is the same ndarray-over-bytearray object
    `sequential_array` makes, without the zero fill."""
    # REGRESSIONS F3 / C3: the zero fill of a destination the cast then
    # overwrote in full was a second pass over X.
    var storage = native_scratch_bytes(count * 4)
    return py_call(context["ndarray"], shape, dtype=context["float32"], buffer=storage)


def admit_nd_f32(value: PythonObject, name: String, copy: Bool, context: PythonObject,
                 double_ints: Bool = False) raises -> PythonObject:
    """`np.array(value, dtype=np.float32, order="C", copy=copy)` of an
    admitted ndarray (`sequential_input`): the array itself when it is
    already C-ordered float32 and no copy is asked for, otherwise one
    C-ordered float32 copy with the shared integer-defined conversion
    (`cast_f32_bits`: `numpy.astype(float32)`'s bits; `double_ints` takes
    `Array.astype`'s integer -> double -> float32 route,
    `classical_f32_bits`). Non-finite words are copied as they are."""
    var dtype = value.dtype
    var kind = String(py=dtype.kind)
    if kind != "b" and kind != "i" and kind != "u" and kind != "f":
        raise Error("TypeError: " + name + " must be a numeric array")
    if not Bool(py=dtype.isnative):
        raise Error("TypeError: " + name + " must be in native byte order")
    var rank = Int(py=value.ndim)
    var c_order = Bool(py=value.flags.c_contiguous)
    if not copy and c_order and Int(py=dtype.num) == 11:
        return value
    var shape = value.shape
    var item = Int(py=dtype.itemsize)
    var cast = cast_in_kind(Int(py=dtype.num), item, name)
    var dims = List[Int]()
    var steps = List[Int]()
    var strides = value.strides  # one attribute read, not one per axis
    var count = 1
    for k in range(rank):  # driver-loop: over array rank metadata
        dims.append(Int(py=shape[k]))
        steps.append(Int(py=strides[k]))
        count *= dims[k]
    var out = admitted_ndarray_f32(shape, count, context)
    _ = admitted_cast_nd_f32(Int(py=sequential_address(value)), dims, steps, item, cast, double_ints,
                             _AS_F32P(unsafe_from_address=Int(py=sequential_address(out))))
    _ = value  # keep the source alive through the native cast above
    return out


def admit_nd_owned_f32(source: PythonObject, dims: List[Int], name: String,
                       context: PythonObject) raises -> PythonObject:
    """The model's own C-order float32 Array of `source` (any admitted
    numeric exporter or nested sequence), which must have exactly the shape
    `dims`; refused ("<name> must be finite") when any word is NaN or an
    infinity. Integers take `Array.astype`'s integer -> double -> float32
    route."""
    classical_admit_value(source, context)
    var value = classical_input(source, context, name)
    var shape = value.shape
    var strides = value.strides
    var item = Int(py=value.itemsize)
    var kind = classical_cast_kind(value, item, name, context)
    if len(shape) != len(dims):
        raise Error(name + " has an unexpected shape")
    var count = 1
    var native_strides = List[Int]()
    var output_shape = Python.list()
    for axis in range(len(dims)):  # driver-loop: over admitted array rank metadata
        if Int(py=shape[axis]) != dims[axis]:
            raise Error(name + " has an unexpected shape")
        count *= dims[axis]
        native_strides.append(Int(py=strides[axis]))
        _ = py_call(output_shape.append, dims[axis])
    output_shape = py_call(native_loaded_module("builtins").tuple, output_shape)
    if count < 0 or count > 9223372036854775807 // 4:
        raise Error("ValueError: " + name + " exceeds native indexing bounds")
    # A retained Array: its array.array store has no uninitialised
    # constructor, so it keeps the zero-filled store (as admitted_storage_f32
    # does without NumPy).
    var result = _owned(native_array_zeros("f", count), output_shape, "<f4", context)
    var pointer = _AS_F32P(unsafe_from_address=Int(py=classical_address(result)))
    # One pass: the cast's own finite flag replaces the second scan.
    if not admitted_cast_nd_f32(Int(py=classical_address(value)), dims, native_strides, item, kind, True, pointer):
        raise Error(name + " must be finite")
    _ = value^
    return result
