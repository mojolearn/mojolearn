# SPDX-License-Identifier: Apache-2.0
"""CPython container allocation used by the native generation owner.

Only C builtins are invoked. Array.__init__, reshape, __buffer__, and package
helpers never run. A foreign buffer is packed inside the operation's boundary.

Every call with an argument, attribute assignment and type query goes through
bindings/py_balanced.mojo: Mojo 1.0's PythonObject call and `__setattr__` keep
one extra reference on each argument or value (docs/MODULAR_ASKS.md), which
kept every array built here, and the caller's buffers, alive for ever.
"""
from std.ffi import _Global
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_setattr, py_type
from core.generation_buffers_impl import generation_ids_into
from std.memory import memcpy


struct GenerationBoundary(Defaultable, Movable):
    var bi: PythonObject
    var arrays: PythonObject
    var array_cls: PythonObject
    var state_cls: PythonObject
    var layer_classes: PythonObject
    var stream_cls: PythonObject

    def __init__(out self):
        self.bi = PythonObject(None)
        self.arrays = PythonObject(None)
        self.array_cls = PythonObject(None)
        self.state_cls = PythonObject(None)
        self.layer_classes = PythonObject(None)
        self.stream_cls = PythonObject(None)


comptime GENERATION_BOUNDARY = _Global[
    StorageType=GenerationBoundary, name="MojoCausalGenerationBoundary",
    init_fn=GenerationBoundary.__init__,
]


def generation_initialize_binding(classes: PythonObject) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    g[].bi = Python.import_module("builtins")
    g[].arrays = Python.import_module("array").array
    g[].array_cls = classes[0]
    g[].state_cls = classes[1]
    g[].layer_classes = classes[2]
    g[].stream_cls = classes[3]
    return PythonObject(None)


def py_ints(values: List[Int]) raises -> PythonObject:
    var out = GENERATION_BOUNDARY.get_or_create_ptr()[].bi.list()
    var append = out.append
    for i in range(len(values)):  # small-loop(argument words handed to a binding entry)
        _ = py_call(append, values[i])
    return out


def generation_array(dims: List[Int], code: String = "f") raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var n = 1
    var itemsize = 1 if code == "b" else 4
    for i in range(len(dims)):  # small-loop(array shape dimension words)
        if dims[i] < 0 or (dims[i] != 0 and n > 2147483647 // dims[i]):
            raise Error("generation: invalid shape or signed 32-bit kernel indexing overflow")
        n *= dims[i]
    var store = py_call(g[].arrays, code)
    _ = py_call(store.frombytes, py_call(g[].bi.bytes, n * itemsize))
    var result = py_call(g[].bi.object.__new__, g[].array_cls)
    py_setattr(result, "_store", store)
    py_setattr(result, "_base", PythonObject(None))
    py_setattr(result, "_pin", PythonObject(None))
    py_setattr(result, "_mv", py_call(g[].bi.memoryview, store))
    py_setattr(result, "_addr", store.buffer_info()[0])
    py_setattr(result, "_readonly", PythonObject(False))
    py_setattr(result, "shape", py_call(g[].bi.tuple, py_ints(dims)))
    py_setattr(result, "dtype", PythonObject("<f4" if code == "f" else ("<i4" if code == "i" else "<i1")))
    py_setattr(result, "ndim", PythonObject(len(dims)))
    py_setattr(result, "size", PythonObject(n))
    py_setattr(result, "nbytes", PythonObject(n * itemsize))
    py_setattr(result, "itemsize", PythonObject(itemsize))
    py_setattr(result, "order", PythonObject("C"))
    var strides = g[].bi.list()
    var insert = strides.insert
    var stride = itemsize
    for i in range(len(dims)):  # small-loop(array shape dimension words)
        var axis = len(dims) - i - 1
        _ = py_call(insert, 0, stride)
        stride *= dims[axis]
    py_setattr(result, "strides", py_call(g[].bi.tuple, strides))
    return result


def generation_address(value: PythonObject, mut keep: List[PythonObject]) raises -> Int:
    if value is None:
        return 0
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    if py_type(value) is g[].array_cls:
        if String(value.order) != "C":
            raise Error("generation: internal model and state arrays must be C-contiguous")
        keep.append(value)
        return Int(py=value._addr)
    var view = py_call(g[].bi.memoryview, value)
    if not Bool(py=view.c_contiguous):
        raise Error("generation: internal model and state buffers must be C-contiguous")
    # Copy read-only model weights to a native owner. Mutable state is created
    # as Array by this owner; supplied state arrays are validated separately.
    var format = String(view.format)
    if format != "f" and format != "i" and format != "b":
        raise Error("generation: unsupported native model buffer format " + format)
    var packed = py_call(g[].arrays, format)
    _ = py_call(packed.frombytes, view.tobytes())
    keep.append(packed)
    return Int(py=packed.buffer_info()[0])


def generation_ids(value: PythonObject, vocab: Int, step: Bool = False) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var shape = PythonObject(None)
    var view = PythonObject(None)
    var column_major = False
    if py_type(value) is g[].array_cls:
        column_major = String(value.order) == "F"
        shape = value.shape
        view = value._mv
    else:
        view = py_call(g[].bi.memoryview, value)
        shape = view.shape
    var rank = len(shape)
    if (not step and rank != 2) or (step and rank != 1 and rank != 2):
        raise Error("generation: ids must have shape (B, L), or (B,) for step")
    var b = Int(py=shape[0])
    var l = 1 if rank == 1 else Int(py=shape[1])
    if b < 1 or l < 1 or (step and l != 1):
        raise Error("generation: ids must be nonempty, with one token per row for step")
    var fmt = String(view.format)
    if fmt != "i" and fmt != "q" and fmt != "l" and fmt != "h" and fmt != "b" and fmt != "I" and fmt != "Q" and fmt != "L" and fmt != "H" and fmt != "B":
        raise Error("TypeError: generation ids must be integer (a native-endian integer buffer)")
    # Pack strided buffer inputs with the CPython buffer implementation; all
    # validation and conversion below is a Mojo loop over native storage.
    var packed = py_call(g[].arrays, fmt)
    _ = py_call(packed.frombytes, py_call(view.tobytes, "C"))
    var out = generation_array([b, l], "i")
    var dst = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=out._addr))
    var signed = fmt == "b" or fmt == "h" or fmt == "i" or fmt == "l" or fmt == "q"
    generation_ids_into(Int(py=packed.buffer_info()[0]), Int(py=packed.itemsize), signed, b, l, column_major, vocab, dst)
    _ = packed^
    return out


def generation_copy(value: PythonObject) raises -> PythonObject:
    var dims = List[Int]()
    for i in range(len(value.shape)):  # small-loop(array shape dimension words)
        dims.append(Int(py=value.shape[i]))
    var code = "f" if String(value.dtype) == "<f4" else "i"
    var out = generation_array(dims, code)
    var dst = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=out._addr))
    var src = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=value._addr))
    memcpy(dest=dst, src=src, count=Int(py=value.nbytes))
    return out


def generation_mutable(
    value: PythonObject, expected: List[Int], mut keep: List[PythonObject], mut copies: List[PythonObject],
    name: String = "",
) raises -> Int:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var label = String("state buffer")
    if name != "":
        label = "state buffer " + name
    var view = PythonObject(None)
    var shape = PythonObject(None)
    if value is None:
        raise Error("CausalLM: " + label + " is missing; allocate_state() makes a complete state")
    if py_type(value) is g[].array_cls:
        if String(value.dtype) != "<f4" or String(value.order) != "C" or Bool(py=value._readonly):
            raise Error("CausalLM: " + label + " must be writable C-contiguous float32 (it is updated in place)")
        view = value._mv
        shape = value.shape
    else:
        view = py_call(g[].bi.memoryview, value)
        shape = view.shape
        if Bool(py=view.readonly) or not Bool(py=view.c_contiguous) or String(view.format) != "f":
            raise Error("CausalLM: " + label + " must be writable C-contiguous float32 (it is updated in place)")
    var want = String("(")
    for i in range(len(expected)):  # small-loop(state buffer shape dims)
        if i > 0:
            want += ", "
        want += String(expected[i])
    want += ")"
    if len(shape) != len(expected):
        raise Error("CausalLM: " + label + " has shape " + String(shape) + ", want " + want)
    var size = 1
    for i in range(len(expected)):  # small-loop(state buffer shape dims)
        if Int(py=shape[i]) != expected[i]:
            raise Error("CausalLM: " + label + " has shape " + String(shape) + ", want " + want)
        size *= expected[i]
    if Int(py=view.nbytes) != size * 4:
        raise Error("CausalLM: " + label + " span differs from its shape")
    var address = generation_address(value, keep)
    if py_type(value) is not g[].array_cls:
        var pair = g[].bi.list()
        _ = py_call(pair.append, view)
        _ = py_call(pair.append, keep[len(keep) - 1])
        copies.append(pair)
    return address
