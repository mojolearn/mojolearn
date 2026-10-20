# SPDX-License-Identifier: Apache-2.0
"""CPython container allocation used by the native generation owner.

Only C builtins are invoked. Array.__init__, reshape, __buffer__, and package
helpers never run. A foreign buffer is packed inside the operation's boundary.
"""
from std.ffi import _Global
from std.python import Python, PythonObject
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
    for i in range(len(values)):
        out.append(values[i])
    return out


def generation_array(dims: List[Int], code: String = "f") raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var n = 1
    var itemsize = 1 if code == "b" else 4
    for i in range(len(dims)):
        if dims[i] < 0 or (dims[i] != 0 and n > 2147483647 // dims[i]):
            raise Error("generation: invalid shape or signed 32-bit kernel indexing overflow")
        n *= dims[i]
    var store = g[].arrays(code)
    store.frombytes(g[].bi.bytes(n * itemsize))
    var result = g[].bi.object.__new__(g[].array_cls)
    result._store = store
    result._base = PythonObject(None)
    result._pin = PythonObject(None)
    result._mv = g[].bi.memoryview(store)
    result._addr = store.buffer_info()[0]
    result._readonly = PythonObject(False)
    result.shape = g[].bi.tuple(py_ints(dims))
    result.dtype = PythonObject("<f4" if code == "f" else ("<i4" if code == "i" else "<i1"))
    result.ndim = PythonObject(len(dims))
    result.size = PythonObject(n)
    result.nbytes = PythonObject(n * itemsize)
    result.itemsize = PythonObject(itemsize)
    result.order = PythonObject("C")
    var strides = g[].bi.list()
    var stride = itemsize
    for i in range(len(dims)):
        var axis = len(dims) - i - 1
        strides.insert(0, stride)
        stride *= dims[axis]
    result.strides = g[].bi.tuple(strides)
    return result


def generation_address(value: PythonObject, mut keep: List[PythonObject]) raises -> Int:
    if value is None:
        return 0
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    if g[].bi.type(value) is g[].array_cls:
        if String(value.order) != "C":
            raise Error("generation: internal model and state arrays must be C-contiguous")
        keep.append(value)
        return Int(py=value._addr)
    var view = g[].bi.memoryview(value)
    if not Bool(py=view.c_contiguous):
        raise Error("generation: internal model and state buffers must be C-contiguous")
    # Copy read-only model weights to a native owner. Mutable state is created
    # as Array by this owner; supplied state arrays are validated separately.
    var format = String(view.format)
    if format != "f" and format != "i" and format != "b":
        raise Error("generation: unsupported native model buffer format " + format)
    var packed = g[].arrays(format)
    packed.frombytes(view.tobytes())
    keep.append(packed)
    return Int(py=packed.buffer_info()[0])


def generation_ids(value: PythonObject, vocab: Int, step: Bool = False) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var shape = PythonObject(None)
    var view = PythonObject(None)
    if g[].bi.type(value) is g[].array_cls:
        if String(value.order) != "C":
            raise Error("generation: Array ids must be C-contiguous")
        shape = value.shape
        view = value._mv
    else:
        view = g[].bi.memoryview(value)
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
        raise Error("generation: ids must be a native-endian integer buffer")
    # Pack strided buffer inputs with the CPython buffer implementation; all
    # validation and conversion below is a Mojo loop over native storage.
    var packed = g[].arrays(fmt)
    packed.frombytes(view.tobytes("C"))
    var out = generation_array([b, l], "i")
    var dst = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=out._addr))
    for i in range(b * l):
        var id = Int(py=packed[i])
        if id < 0 or id >= vocab:
            raise Error("generation: id outside vocabulary at flat index " + String(i))
        dst[i] = Int32(id)
    return out


def generation_copy(value: PythonObject) raises -> PythonObject:
    var dims = List[Int]()
    for i in range(len(value.shape)):
        dims.append(Int(py=value.shape[i]))
    var code = "f" if String(value.dtype) == "<f4" else "i"
    var out = generation_array(dims, code)
    var dst = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=out._addr))
    var src = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=value._addr))
    memcpy(dest=dst, src=src, count=Int(py=value.nbytes))
    return out
