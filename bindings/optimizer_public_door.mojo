# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python door of the public optimizers (`mojolearn.SGD`, `Adam`,
`AdamW`, `clip_grad_norm_`, `accumulate_grads`; lane optimizers-thin-glue,
2026-10-21). Each public method is ONE native call: the Python method passes
the optimizer object and the caller's arguments as they are, and this door
reads them, refuses by name what the Python glue used to refuse, calls ONE
typed kernel entry and writes the results back onto the optimizer.

What the door does is argument conversion only:

  * the caller's parameter and gradient tensors are read through the
    buffer protocol (`PyObject_GetBuffer`, released before the kernel
    runs, exactly as `_bufcheck.probe` did) or, for a `mojolearn.Array`,
    off its own storage fields; a strided read-only input is made
    C-contiguous by `PyBuffer_ToContiguous` into memory this door owns;
  * dtype, emptiness, layout, count and shape are refused with the
    messages the Python glue raised;
  * built-in schedules supply admitted native configuration words; Python
    learning-rate callbacks are refused before lookup or invocation;
  * the optimizer state's host mirrors (`exp_avg`, `exp_avg_sq`) are
    `mojolearn.Array`s assembled from native storage and slots.

The kernels are training/optimizer_public.mojo (the GPU binding),
training/host/optimizer_public_host.mojo (a CPU-only install) and
training/optimizer_public_common.mojo (shared, no device code); `OptKernels`
names the entries a binding provides. Ownership (docs/MODULAR_ASKS.md):
every Python call and attribute or item assignment goes through
bindings/py_balanced.mojo, so no caller object is retained."""
from std.ffi import _Global
from std.memory import memcpy
from std.math import isfinite
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr

from bindings.optimizer_public_words import optimizer_python_error, stash_python_error
from bindings.py_balanced import py_call, py_setattr, py_setitem, py_type, py_get, py_fields
from bindings.optimizer_admission import opt_guards, opt_owner, opt_value, opt_scalar, opt_schedule
from sequence.lr_table import LrSpec
from training.optimizer_public_common import (
    GRADS_DEVICE, GRADS_HOST, GRADS_RESIDENT, I32P, OptHyper, PublicStep, PublicStepOut, eager_moments,
    registry_offsets, zero_tensors,
)

comptime IP = MutPointer[Int, MutUntrackedOrigin]
comptime BP = MutPointer[UInt8, MutUntrackedOrigin]
comptime PYBUF_FULL_RO = 0x11C

#: `optimizer_step`'s operations
comptime OP_STEP = 0
comptime OP_STEP_DEVICE = 1
comptime OP_STEP_ACCUMULATED = 2

#: `optimizer_state`'s readings
comptime STATE_DICT = 0
comptime STATE_EXP_AVG = 1
comptime STATE_EXP_AVG_SQ = 2

comptime _SRC = " (python/mojolearn/_training_impl.py)"


# ---------------------------------------------------------------------------
# the kernels a binding provides


trait OptKernels:
    """The typed kernel entries behind the door, one binding's set."""

    @staticmethod
    def step(req: PublicStep) raises -> PublicStepOut:
        ...

    @staticmethod
    def io(req: PublicStep, op_code: Int, tensor: Int) raises -> Tuple[Int, Int]:
        ...

    @staticmethod
    def clip(addrs: List[Int], nbytes: List[Int], max_norm: Float32) raises -> Tuple[Float32, Float32]:
        ...

    @staticmethod
    def accumulate(
        in_addrs: List[Int], nbytes: List[Int], out_addrs: List[Int], a_count: Int, tokens: Int
    ) raises -> Int:
        ...

    @staticmethod
    def download(handle: Int, m_addr: Int, v_addr: Int, n: Int) raises -> Int:
        ...

    @staticmethod
    def upload(handle: Int, m_addr: Int, v_addr: Int, n: Int) raises -> Int:
        ...


# ---------------------------------------------------------------------------
# the classes and tables the door reads, resolved once


struct OptDoor(Defaultable, Movable):
    var context: PythonObject
    var ready: Bool
    var bi: PythonObject
    var array_cls: PythonObject
    var dev_cls: PythonObject
    var arrays: PythonObject
    var not_covered: PythonObject
    var refused: PythonObject

    def __init__(out self):
        self.context = PythonObject(None)
        self.ready = False
        self.bi = PythonObject(None)
        self.array_cls = PythonObject(None)
        self.dev_cls = PythonObject(None)
        self.arrays = PythonObject(None)
        self.not_covered = PythonObject(None)
        self.refused = PythonObject(None)


comptime OPT_DOOR = _Global[StorageType=OptDoor, name="MojoOptimizerPublicDoor", init_fn=OptDoor.__init__]


def door_ready() raises:
    """Resolve the builtins, `mojolearn.Array`, `TrainDeviceArray`, the
    `array` type and the refusal tables once per process."""
    var d = OPT_DOOR.get_or_create_ptr()
    if d[].ready:
        opt_guards(d[].context)
        return
    d[].bi = Python.import_module("builtins")
    d[].array_cls = Python.import_module("mojolearn._array").Array
    var training = Python.import_module("mojolearn._training_impl")
    d[].context = py_fields(training)["_OPT_CONTEXT"]
    opt_guards(d[].context)
    d[].dev_cls = training.TrainDeviceArray
    d[].not_covered = training._NOT_COVERED
    d[].refused = training._REFUSED_KWARGS
    d[].arrays = Python.import_module("array").array
    d[].ready = True


def _bi() raises -> PythonObject:
    return OPT_DOOR.get_or_create_ptr()[].bi


def getattr_or(obj: PythonObject, name: String, default: PythonObject) raises -> PythonObject:
    return py_get(py_fields(obj), PythonObject(name), default)


def as_float(value: PythonObject) raises -> Float64:
    """`float(value)`."""
    opt_scalar(value, OPT_DOOR.get_or_create_ptr()[].context)
    return Float64(py=py_call(_bi().float, value))


def truth(value: PythonObject) raises -> Bool:
    opt_scalar(value, OPT_DOOR.get_or_create_ptr()[].context)
    return Bool(py=py_call(_bi().bool, value))


def py_repr(value: PythonObject) raises -> String:
    return String(py=py_call(_bi().repr, value))


def type_name(value: PythonObject) raises -> String:
    return String(py=py_type(value).__name__)


def is_instance(value: PythonObject, cls: PythonObject) raises -> Bool:
    return Bool(py=py_call(_bi().isinstance, value, cls))


def shape_text(dims: List[Int]) -> String:
    """A shape as Python prints a tuple: `(3, 4)`, `(5,)`, `()`."""
    var text = String("(")
    for i in range(len(dims)):  # driver-loop: over shape words of a refusal message
        if i > 0:
            text += ", "
        text += String(dims[i])
    if len(dims) == 1:
        text += ","
    return text + ")"


def same_shape(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):  # driver-loop: over shape words
        if a[i] != b[i]:
            return False
    return True


def py_ints(values: List[Int]) raises -> PythonObject:
    """A Python tuple of ints (a shape argument)."""
    var out = Python.list()
    for i in range(len(values)):  # driver-loop: over shape words
        _ = py_call(out.append, PythonObject(values[i]))
    return py_call(_bi().tuple, out)


def new_array(dims: List[Int], dtype: String) raises -> PythonObject:
    """Assemble an Array from native storage and slots; never call its Python constructor."""
    var d = OPT_DOOR.get_or_create_ptr()
    var b = d[].bi
    var size = 1
    var strides = List[Int](length=len(dims), fill=0)
    var step = 4
    for i in range(len(dims) - 1, -1, -1):  # driver-loop: over one native shape/stride word per dimension
        strides[i] = step
        step *= dims[i]
        size *= dims[i]
    var store = py_call(d[].arrays, PythonObject("f" if dtype == "<f4" else "i"), py_call(b.bytes, size * 4))
    var result = py_call(b.object.__new__, d[].array_cls)
    py_setattr(result, "_store", store)
    py_setattr(result, "_base", PythonObject(None))
    py_setattr(result, "_pin", PythonObject(None))
    # Acquire the native store's buffer through CPython's public API. The
    # mutable builtins.memoryview module field is not a runtime constructor.
    ref cpy = Python().cpython()
    var view_ptr = cpy.lib.get_function[PyObjectPtr]("PyMemoryView_FromObject")(store._obj_ptr)
    if not view_ptr:
        var failure = stash_python_error()
        raise failure^
    var view = PythonObject(from_owned=view_ptr)
    py_setattr(result, "_mv", view)
    py_setattr(result, "_addr", py_call(store.buffer_info)[0])
    py_setattr(result, "_readonly", PythonObject(False))
    py_setattr(result, "shape", py_ints(dims))
    py_setattr(result, "dtype", PythonObject(dtype))
    py_setattr(result, "ndim", PythonObject(len(dims)))
    py_setattr(result, "size", PythonObject(size))
    py_setattr(result, "nbytes", PythonObject(size * 4))
    py_setattr(result, "itemsize", PythonObject(4))
    py_setattr(result, "strides", py_ints(strides))
    py_setattr(result, "order", PythonObject("C"))
    return result


def set_field(opt: PythonObject, name: String, value: PythonObject) raises:
    py_setitem(py_fields(opt), PythonObject(name), value)


def owner_kind(opt: PythonObject) raises -> Int:
    var classes = OPT_DOOR.get_or_create_ptr()[].context["owners"]
    for i in range(len(classes)):  # driver-loop: over three native optimizer kinds
        if py_type(opt) is classes[i]:
            return i
    raise Error("TypeError: optimizer requires an exact native optimizer owner")


# ---------------------------------------------------------------------------
# one tensor, read through the buffer protocol


struct TensorView(Copyable, Movable):
    """The facts `_bufcheck.probe` read, the address `_buffer.addr` read, and
    the dtype name a refusal prints."""
    var addr: Int
    var nbytes: Int
    var itemsize: Int
    var readonly: Bool
    var c_contig: Bool
    var fmt: String
    var dtype: String
    var shape: List[Int]
    var strides: List[Int]
    var has_strides: Bool
    var is_array: Bool

    def __init__(out self):
        self.addr = 0
        self.nbytes = 0
        self.itemsize = 0
        self.readonly = False
        self.c_contig = True
        self.fmt = String("B")
        self.dtype = String("")
        self.shape = List[Int]()
        self.strides = List[Int]()
        self.has_strides = False
        self.is_array = False

    def count(self) -> Int:
        var n = 1
        for i in range(len(self.shape)):  # driver-loop: over shape words
            n *= self.shape[i]
        return n


def _contiguous(shape: List[Int], strides: List[Int], has_strides: Bool, itemsize: Int) -> Bool:
    """`_buffer._contiguity`'s C half (NumPy's rule)."""
    if len(shape) == 0:
        return True
    var n = 1
    for i in range(len(shape)):  # driver-loop: over shape words
        n *= shape[i]
    if n <= 1 or not has_strides:
        return True
    var expect = itemsize
    for k in range(len(shape) - 1, -1, -1):  # driver-loop: over one stride word per axis
        if shape[k] != 1 and strides[k] != expect:
            return False
        expect *= shape[k]
    return True


def base_format(fmt: String) -> String:
    """`_bufcheck.base_format`: a byte-order mark naming the host's order
    removed (this library's hosts are little-endian)."""
    if fmt.byte_length() > 1 and (fmt.startswith("@") or fmt.startswith("=") or fmt.startswith("<")):
        return String(fmt[byte=1:])
    return fmt


def _format_name(code: String, itemsize: Int, fmt: String) -> String:
    if code == "f":
        return "float32"
    if code == "d":
        return "float64"
    if code == "e":
        return "float16"
    if code == "g":
        return "longdouble"
    if code == "?":
        return "bool"
    if code == "b":
        return "int8"
    if code == "B":
        return "uint8"
    if code == "h":
        return "int16"
    if code == "H":
        return "uint16"
    if code == "i":
        return "int32"
    if code == "I":
        return "uint32"
    if code == "q":
        return "int64"
    if code == "Q":
        return "uint64"
    if code == "n":
        return "intp"
    if code == "N":
        return "uintp"
    if code == "c":
        return "bytes"
    if code == "l":
        return "int" + String(8 * itemsize)
    if code == "L":
        return "uint" + String(8 * itemsize)
    return "'" + fmt + "'"


def _typestr_name(t: String) -> String:
    if t == "<f4" or t == ">f4":
        return "float32"
    if t == "<f8" or t == ">f8":
        return "float64"
    if t == "<f2":
        return "float16"
    if t == "<i4" or t == ">i4":
        return "int32"
    if t == "<i8" or t == ">i8":
        return "int64"
    if t == "<u4" or t == ">u4":
        return "uint32"
    if t == "<u1" or t == "|u1":
        return "uint8"
    if t == "<i2":
        return "int16"
    if t == "<u2":
        return "uint16"
    if t == "|i1":
        return "int8"
    return ""


def _array_code(t: String) -> String:
    """`_array._CODE`: an Array dtype's struct code."""
    if t == "<f4":
        return "f"
    if t == "<f8":
        return "d"
    if t == "<i4":
        return "i"
    if t == "<i8":
        return "q"
    if t == "<u4":
        return "I"
    if t == "<u1":
        return "B"
    if t == "<f2" or t == "<u2":
        return "H"
    if t == "<i1":
        return "b"
    return "B"


def dtype_name(obj: PythonObject, fmt: String, itemsize: Int) raises -> String:
    # Buffer format is native metadata; querying arbitrary dtype properties is
    # not needed and could call a Python descriptor.
    return _format_name(base_format(fmt), itemsize, fmt)


def _get_buffer(obj: PythonObject, mut view: List[Int]) raises -> Bool:
    """`PyObject_GetBuffer(obj, view, PyBUF_FULL_RO)`; False (indicator
    cleared) when `obj` does not export a buffer."""
    ref cpy = Python().cpython()
    var p = view.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var rc = cpy.lib.get_function[Int32]("PyObject_GetBuffer")(obj._obj_ptr, p, Int32(PYBUF_FULL_RO))
    if rc != 0:
        cpy.lib.get_function[NoneType]("PyErr_Clear")()
        return False
    return True


def _release_buffer(mut view: List[Int]) raises:
    ref cpy = Python().cpython()
    var p = view.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    cpy.lib.get_function[NoneType]("PyBuffer_Release")(p)


def tensor_view(obj: PythonObject, mut v: TensorView) raises -> Bool:
    """Fill `v` with `obj`'s buffer facts; False when `obj` is not a buffer
    (`_training_impl._is_buffer`). The export is released before this
    returns, as `probe` released it."""
    var d = OPT_DOOR.get_or_create_ptr()
    v = TensorView()
    opt_value(obj, d[].context)
    if is_instance(obj, d[].array_cls):
        # an Array is read off its own storage fields (`_buffer.Buf._from_array`)
        v.is_array = True
        v.addr = Int(py=obj._addr)
        v.nbytes = Int(py=obj.nbytes)
        v.itemsize = Int(py=obj.itemsize)
        v.readonly = truth(obj._readonly)
        var t = String(py=obj.dtype)
        v.fmt = _array_code(t)
        var shape = obj.shape
        for i in range(len(shape)):  # driver-loop: over shape words
            v.shape.append(Int(py=shape[i]))
        var strides = obj.strides
        for i in range(len(strides)):  # driver-loop: over stride words
            v.strides.append(Int(py=strides[i]))
        v.has_strides = True
        v.c_contig = _contiguous(v.shape, v.strides, True, v.itemsize)
        v.dtype = dtype_name(obj, v.fmt, v.itemsize)
        return True
    if py_type(obj) is d[].dev_cls:
        return False
    var view = List[Int](length=10, fill=0)
    if not _get_buffer(obj, view):
        _ = view^
        return False
    var p = view.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var words32 = p.bitcast[Int32]()
    var indirect = False
    v.addr = p[0]
    v.nbytes = p[2]
    v.itemsize = p[3]
    v.readonly = words32[8] != 0
    var ndim = Int(words32[9])
    if p[5] != 0:
        var fp = BP(unsafe_from_address=p[5])
        var text = String("")
        for k in range(64):  # driver-loop: over format characters of one buffer
            if fp[k] == 0:
                break
            text += chr(Int(fp[k]))
        v.fmt = text
    if ndim > 0:
        if p[6] != 0:
            var sp = IP(unsafe_from_address=p[6])
            for k in range(ndim):  # driver-loop: over shape words of one buffer
                v.shape.append(sp[k])
        else:
            v.shape.append(v.nbytes // max(v.itemsize, 1))
        if p[7] != 0:
            var tp = IP(unsafe_from_address=p[7])
            for k in range(ndim):  # driver-loop: over stride words of one buffer
                v.strides.append(tp[k])
            v.has_strides = True
        if p[8] != 0:
            var op = IP(unsafe_from_address=p[8])
            for k in range(ndim):  # driver-loop: over suboffset words of one buffer
                if op[k] >= 0:
                    indirect = True
    _release_buffer(view)
    _ = view^
    if indirect:
        # `Buf._fill` refuses an indirect buffer with a TypeError, which
        # `_is_buffer` reads as "not a buffer"
        return False
    v.c_contig = _contiguous(v.shape, v.strides, v.has_strides, v.itemsize)
    v.dtype = dtype_name(obj, v.fmt, v.itemsize)
    return True


def contiguous_addr(obj: PythonObject, v: TensorView, mut keep: List[List[Float32]]) raises -> Int:
    """The address of `obj`'s values in C order (`_buffer.as_f32_c` of a
    float32 tensor): its own memory when it is C-contiguous, else a copy
    `PyBuffer_ToContiguous` makes into `keep`."""
    if v.c_contig:
        return v.addr
    ref cpy = Python().cpython()
    var n = v.count()
    var store = List[Float32](length=max(n, 1), fill=Float32(0.0))
    var dst = store.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var view = List[Int](length=10, fill=0)
    var p = view.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var shape = v.shape.copy()
    var strides = v.strides.copy()
    var held = False
    if v.is_array:
        # an F-order Array: its storage described with its own strides
        p[0] = v.addr
        p[2] = v.nbytes
        p[3] = v.itemsize
        p.bitcast[Int32]()[8] = Int32(1)
        p.bitcast[Int32]()[9] = Int32(len(shape))
        p[6] = Int(shape.unsafe_ptr())
        p[7] = Int(strides.unsafe_ptr())
    else:
        if not _get_buffer(obj, view):
            raise Error("TypeError: mojolearn: tensor does not support the buffer protocol")
        held = True
    var rc = cpy.lib.get_function[Int32]("PyBuffer_ToContiguous")(dst, p, n * 4, Int8(67))
    if held:
        _release_buffer(view)
    _ = view^
    _ = shape^
    _ = strides^
    if rc != 0:
        raise cpy.unsafe_get_error()
    keep.append(store^)
    var last = len(keep) - 1
    return Int(keep[last].unsafe_ptr())


# ---------------------------------------------------------------------------
# the argument checks of the Python glue, message for message


def _not_covered(topic: String, where: String) raises -> Error:
    var d = OPT_DOOR.get_or_create_ptr()
    return Error("NotImplementedError: mojolearn." + where + ": " + String(py=d[].not_covered[topic]))


def as_seq(x: PythonObject, name: String, where: String) raises -> PythonObject:
    """`_as_seq`: one buffer, or an iterable of them, as a Python list."""
    var v = TensorView()
    if tensor_view(x, v):
        var one = Python.list()
        _ = py_call(one.append, x)
        return one
    var out: PythonObject
    try:
        out = py_call(_bi().list, x)
    except:
        raise Error(
            "TypeError: mojolearn." + where + ": " + name + " must be a float32 array (a numpy array, an "
            + "array.array or a mojolearn.Array -- anything supporting the buffer protocol) or an iterable "
            + "of them, got " + type_name(x)
        )
    if len(out) == 0:
        raise Error(
            "ValueError: mojolearn." + where + ": " + name + " is empty; there is nothing to do and an "
            + "empty parameter registry is refused rather than answered (training/estimator.mojo)"
        )
    for i in range(len(out)):  # driver-loop: over parameter buffers
        if not tensor_view(out[i], v):
            raise Error(
                "TypeError: mojolearn." + where + ": " + name + "[" + String(i) + "] is "
                + type_name(out[i]) + ", not an array (it does not support the buffer protocol)"
            )
    return out


def check_tensors(seq: PythonObject, name: String, where: String, inplace: Bool) raises -> List[TensorView]:
    """`_check_dtype`: float32 or refused by name, never cast; non-empty;
    with `inplace`, C-contiguous and writable. Returns every tensor's view."""
    var views = List[TensorView]()
    for i in range(len(seq)):  # driver-loop: over parameter buffers
        var v = TensorView()
        if not tensor_view(seq[i], v):
            raise Error("TypeError: mojolearn: object does not support the buffer protocol")
        var code = base_format(v.fmt)
        if code == "e" or v.dtype == "float16" or v.dtype == "bfloat16":
            var refusal = _not_covered("mixed precision", where)
            raise refusal^
        if code != "f":
            raise Error(
                "TypeError: mojolearn." + where + ": " + name + "[" + String(i) + "] has dtype " + v.dtype
                + "; this surface is float32 only (loss contract section 1, optimizer contract section "
                + "1) and refuses rather than casts, because a cast changes which arithmetic answered "
                + "and leaves the label unchanged" + _SRC
            )
        if v.count() < 1:
            raise Error(
                "ValueError: mojolearn." + where + ": " + name + "[" + String(i) + "] is empty, shape "
                + shape_text(v.shape) + "; an empty tensor is REFUSED and not skipped -- the clip would "
                + "fold a zero-length GEMM, which no gate in this tree has run (training/estimator.mojo)"
            )
        if inplace and (not v.c_contig or v.readonly):
            raise Error(
                "ValueError: mojolearn." + where + ": " + name + "[" + String(i) + "] must be "
                + "C-contiguous and writable; it is written back IN PLACE (the torch.optim contract, the "
                + "trailing underscore on clip_grad_norm_) and a strided or read-only buffer cannot be. "
                + "Pass np.ascontiguousarray(...) of it, or a writable mojolearn.Array "
                + "(python/mojolearn/_training_impl.py, DEVIATION 2403)"
            )
        views.append(v^)
    return views^


def write_addr(obj: PythonObject, name: String) raises -> Int:
    """`_buffer.addr`: the address of a buffer the kernel writes."""
    var v = TensorView()
    if not tensor_view(obj, v):
        raise Error("TypeError: mojolearn: " + name + " does not support the buffer protocol")
    if v.readonly:
        raise Error("ValueError: mojolearn: " + name + " is read-only, refusing to write to it")
    return v.addr


def read_addr(obj: PythonObject, name: String) raises -> Int:
    """`_buffer.addr_ro`."""
    var v = TensorView()
    if not tensor_view(obj, v):
        raise Error("TypeError: mojolearn: " + name + " does not support the buffer protocol")
    return v.addr


def _addrs_of(views: List[TensorView], mut addrs: List[Int], mut nbytes: List[Int]):
    for j in range(len(views)):  # driver-loop: over parameter buffers
        addrs.append(views[j].addr)
        nbytes.append(views[j].nbytes)


def _check_grads(params: PythonObject, gs: PythonObject, views: List[TensorView], owhere: String, where: String) raises:
    """`_Optimizer._check_grads`: grads match params in count and shape."""
    if len(gs) != len(params):
        raise Error(
            "ValueError: mojolearn." + owhere + "." + where + ": " + String(len(gs)) + " gradients for "
            + String(len(params)) + " parameter tensors"
        )
    for j in range(len(params)):  # driver-loop: over parameter buffers
        var pv = TensorView()
        if not tensor_view(params[j], pv):
            raise Error("TypeError: mojolearn: object does not support the buffer protocol")
        if not same_shape(pv.shape, views[j].shape):
            raise Error(
                "ValueError: mojolearn." + owhere + "." + where + ": grads[" + String(j) + "] has shape "
                + shape_text(views[j].shape) + ", params[" + String(j) + "] has " + shape_text(pv.shape)
            )


def refuse_kwargs(kwargs: PythonObject, where: String) raises:
    """`_refuse_unknown`: every keyword this surface does not have, refused
    by name."""
    var d = OPT_DOOR.get_or_create_ptr()
    opt_value(kwargs, d[].context)
    var keys = py_call(d[].bi.list, kwargs)
    for i in range(len(keys)):  # driver-loop: over keyword argument names
        var k = String(py=keys[i])
        if k == "dropout":
            var refusal = _not_covered("dropout", where)
            raise refusal^
        if k == "distributed" or k == "world_size" or k == "rank":
            var refusal = _not_covered("distributed", where)
            raise refusal^
        var why = py_call(d[].refused.get, keys[i])
        if truth(why):
            raise Error(
                "TypeError: mojolearn." + where + ": " + k + "= is REFUSED BY NAME -- " + String(py=why) + _SRC
            )
        raise Error(
            "TypeError: mojolearn." + where + ": unexpected keyword " + k + "=. This surface takes only "
            + "the parameters in its WHAT IS HONORED table; an unknown keyword is refused rather than "
            + "ignored, because a keyword that is accepted and does nothing is a caller who believes "
            + "something about their run that is not true" + _SRC
        )


# ---------------------------------------------------------------------------
# the optimizer object's words


def _resident_word(value: PythonObject) raises -> Int:
    """`_RESIDENT_WORD.get(resident, 1)`: None -1, False 0, True 1."""
    if value is None:
        return -1
    var bi = _bi()
    if is_instance(value, bi.bool) or is_instance(value, bi.int):
        return 0 if Int(py=value) == 0 else 1
    return 1


def _state_into(opt: PythonObject, mut req: PublicStep) raises:
    """`_state_words`: handle, resident, pristine and the mirrors' addresses."""
    var res = py_fields(opt)["_res"]
    req.handle = -1 if res is None else Int(py=res)
    req.resident_wanted = _resident_word(py_fields(opt)["_resident_wanted"])
    req.pristine = truth(py_fields(opt)["_pristine"])
    var m = py_fields(opt)["_m"]
    var v = py_fields(opt)["_v"]
    req.m_addr = 0 if m is None else write_addr(m, "exp_avg")
    req.v_addr = 0 if v is None else write_addr(v, "exp_avg_sq")


def host_moments(opt: PythonObject) raises:
    """`_host_moments`: the host mirrors, allocated (zero filled) on first need."""
    var n = Int(py=py_fields(opt)["n_total"])
    if py_fields(opt)["_m"] is None:
        set_field(opt, "_m", new_array([n], "<f4"))
    if py_fields(opt)["_v"] is None:
        set_field(opt, "_v", new_array([n], "<f4"))


def keep_slot(opt: PythonObject, handle: Int, module: PythonObject, on_host: Bool) raises:
    """`_keep_slot`: record a slot the native call opened, with the extension
    module that owns it (its finalizer closes the slot through it)."""
    if py_fields(opt)["_res"] is None and handle >= 0:
        set_field(opt, "_res", PythonObject(handle))
        set_field(opt, "_res_binding", module)
        set_field(opt, "_res_host", PythonObject(on_host))


def _hyper(opt: PythonObject, maximize: Bool, max_norm: Float64) raises -> OptHyper:
    """`_config()` of SGD (kind 0) and Adam / AdamW, as the step's floats."""
    var kind = owner_kind(opt)
    if kind == 0:
        return OptHyper(
            kind, 1 if truth(py_fields(opt)["nesterov"]) else 0, maximize, as_float(py_fields(opt)["lr"]), 0.9, 0.999, 1e-8,
            as_float(py_fields(opt)["weight_decay"]), as_float(py_fields(opt)["momentum"]), as_float(py_fields(opt)["dampening"]), max_norm,
        )
    var betas = py_fields(opt)["betas"]
    return OptHyper(
        kind, 0, maximize, as_float(py_fields(opt)["lr"]), as_float(betas[0]), as_float(betas[1]), as_float(py_fields(opt)["eps"]),
        as_float(py_fields(opt)["weight_decay"]), 0.0, 0.0, max_norm,
    )


def schedule_spec(schedule: PythonObject) raises -> LrSpec:
    var context = OPT_DOOR.get_or_create_ptr()[].context
    opt_schedule(schedule, context)
    var fields = py_fields(schedule)
    var kind = 0
    var classes = context["schedules"]
    for i in range(len(classes)):  # driver-loop: over three built-in schedule configurations
        if py_type(schedule) is classes[i]:
            kind = i
    if kind < 3:
        var total = fields["total_steps"]
        return LrSpec(kind, Int(py=fields["warmup_steps"]), -1 if total is None else Int(py=total), 0,
                      Float64(py=fields["peak_lr"]), Float64(py=fields["min_lr"]), 0.0, 0.0)
    if kind < 5:
        var base = Float64(py=fields["base_lr"])
        var gamma = Float64(py=fields["gamma"])
        if not isfinite(base) or not isfinite(gamma):
            raise Error("ValueError: base_lr and gamma must be finite")
        return LrSpec(20, Int(py=fields["_step"]) if kind == 3 else 1, 0, 0, base, gamma, 0.0, 0.0)
    return LrSpec(10, 1 if String(py=fields["anneal_strategy"]) == "linear" else 0,
                  1 if Bool(py=fields["three_phase"]) else 0, Int(py=fields["total_steps"]),
                  Float64(py=fields["max_lr"]), Float64(py=fields["div_factor"]),
                  Float64(py=fields["final_div_factor"]), Float64(py=fields["pct_start"]))


# ---------------------------------------------------------------------------
# the public methods


def door_init(opt: PythonObject, params: PythonObject) raises -> PythonObject:
    """The registry half of `_Optimizer.__init__`: `params` as the tensor
    list, its offsets (filled natively), `n_total`, the eager mirrors under
    MOJOLEARN_OPT_LAZY_MOMENTS=0 and SGD's `buf_initialized`. Returns
    n_total."""
    door_ready()
    opt_owner(opt, OPT_DOOR.get_or_create_ptr()[].context)
    var where = String(py=py_fields(opt)["_where"])
    var seq = as_seq(params, "params", where)
    set_field(opt, "params", seq)
    var views = check_tensors(seq, "params", where, True)
    var j = len(seq)
    var nbytes = List[Int]()
    for k in range(j):  # driver-loop: over parameter buffers
        nbytes.append(views[k].nbytes)
    var offsets = new_array([j + 1], "<i4")
    var n_total = registry_offsets(nbytes, I32P(unsafe_from_address=Int(py=offsets._addr)))
    set_field(opt, "offsets", offsets)
    set_field(opt, "n_total", PythonObject(n_total))
    set_field(opt, "_m", PythonObject(None))
    set_field(opt, "_v", PythonObject(None))
    set_field(opt, "_pristine", PythonObject(True))
    if eager_moments():
        # MOJOLEARN_OPT_LAZY_MOMENTS=0: the eager pair (the A/B)
        host_moments(opt)
        set_field(opt, "_pristine", PythonObject(False))
    set_field(opt, "buf_initialized", new_array([j], "<i4"))
    return PythonObject(n_total)


def door_step[K: OptKernels](
    module: PythonObject, opt: PythonObject, grads: PythonObject, max_norm: PythonObject, op: Int,
    tokens: PythonObject,
) raises -> PythonObject:
    """`step` (op 0), `step_device` (op 1) and `step_accumulated` (op 2;
    `grads` is the microbatch gradient lists): the argument checks of the
    Python glue, the ONE kernel call, the assignments of its result.
    Returns the pre-clip total norm, or None."""
    door_ready()
    opt_owner(opt, OPT_DOOR.get_or_create_ptr()[].context)
    var d = OPT_DOOR.get_or_create_ptr()
    var bi = d[].bi
    var owhere = String(py=py_fields(opt)["_where"])
    var where = String("step") if op == OP_STEP else (
        String("step_device") if op == OP_STEP_DEVICE else String("step_accumulated"))
    var p_dev = truth(getattr_or(opt, "_p_dev", PythonObject(False)))
    var parts = PythonObject(None)
    var t_tokens = -1
    if op == OP_STEP_DEVICE and not p_dev:
        raise Error("RuntimeError: mojolearn." + owhere + ".step_device: call params_to_device() first")
    opt_value(grads, d[].context)
    opt_scalar(tokens, d[].context)
    if op == OP_STEP_ACCUMULATED:
        parts = py_call(bi.list, grads)
        var want = Int(py=py_fields(opt)["accumulation_steps"])
        if len(parts) != want:
            raise Error(
                "ValueError: mojolearn." + owhere + ".step_accumulated: " + String(len(parts))
                + " microbatch gradient lists for accumulation_steps=" + String(want) + "; the count is "
                + "part of the run's numerical specification (optimizer contract clause 9.2)" + _SRC
            )
        if not (tokens is None):
            t_tokens = Int(py=py_call(bi.int, tokens))
        if len(parts) > 1 and not (tokens is None) and t_tokens < 1:
            raise Error("ValueError: mojolearn.accumulate_grads: tokens must be >= 1")
    var mn = 0.0
    if not (max_norm is None):
        mn = as_float(max_norm)
        if not (mn > 0.0):
            raise Error(
                "ValueError: mojolearn." + owhere + "." + where + ": max_norm must be > 0 or None, got "
                + py_repr(max_norm) + "; None is how 'no clipping' is spelled" + _SRC
            )
    var params = py_fields(opt)["params"]
    var j = len(params)
    var n_total = Int(py=py_fields(opt)["n_total"])
    var req = PublicStep()
    var keep = List[List[Float32]]()
    req.g_mode = GRADS_HOST
    req.a_count = 1
    if op == OP_STEP_ACCUMULATED:
        req.a_count = len(parts)
        for i in range(len(parts)):  # driver-loop: over microbatch arguments
            var label = "microbatch_grads[" + String(i) + "]"
            var gs = as_seq(parts[i], label, owhere)
            var views = check_tensors(gs, label, owhere, req.a_count == 1)
            _check_grads(params, gs, views, owhere, where)
            for k in range(len(views)):  # driver-loop: over parameter buffers
                if req.a_count > 1:
                    # read-only inputs: strided buffers are made contiguous
                    req.g_addr.append(contiguous_addr(gs[k], views[k], keep))
                else:
                    if views[k].readonly:
                        raise Error("ValueError: mojolearn: tensor is read-only, refusing to write to it")
                    req.g_addr.append(views[k].addr)
                req.g_bytes.append(views[k].nbytes)
    elif p_dev and not (grads is None) and (
        is_instance(grads, d[].dev_cls) or (
            (is_instance(grads, bi.list) or is_instance(grads, bi.tuple)) and len(grads) > 0
            and is_instance(grads[0], d[].dev_cls))):
        var gs = grads
        if is_instance(grads, d[].dev_cls):
            gs = Python.list()
            _ = py_call(gs.append, grads)
        if len(gs) != j:
            raise Error(
                "ValueError: mojolearn." + owhere + "." + where + ": " + String(len(gs)) + " gradients for "
                + String(j) + " parameter tensors"
            )
        for k in range(len(gs)):  # driver-loop: over parameter buffers
            if not is_instance(gs[k], d[].dev_cls):
                raise Error("ValueError: mojolearn." + owhere + "." + where + ": grads[" + String(k)
                            + "] must be a device array")
        req.g_mode = GRADS_DEVICE
        for k in range(len(gs)):  # driver-loop: over parameter buffers: one device handle each
            req.g_addr.append(Int(py=py_fields(gs[k])["h"]))
            req.g_bytes.append(Int(py=py_fields(gs[k])["size"]))
    elif p_dev and grads is None:
        req.g_mode = GRADS_RESIDENT
    else:
        var gs = as_seq(grads, "grads", owhere)
        var views = check_tensors(gs, "grads", owhere, True)
        if p_dev and len(gs) != j:
            raise Error(
                "ValueError: mojolearn." + owhere + ": grads does not match the optimizer's registry ("
                + String(j) + " tensors, " + String(n_total) + " floats)"
            )
        if not p_dev:
            _check_grads(params, gs, views, owhere, where)
        _addrs_of(views, req.g_addr, req.g_bytes)
    if not p_dev:
        # Re-read every parameter buffer's layout on EVERY step: a caller
        # can reshape or resize their own array between steps.
        var pviews = check_tensors(params, "params", owhere, True)
        _addrs_of(pviews, req.p_addr, req.p_bytes)
    var schedule = py_fields(opt)["lr_schedule"]
    req.has_sched = False
    if not (schedule is None):
        req.has_sched = True
        req.sched = schedule_spec(schedule)
    var resident = py_fields(opt)["_resident_wanted"]
    if is_instance(resident, bi.bool) and not truth(resident):
        # `resident=False`: the state lives in these host arrays (no slot)
        host_moments(opt)
    var maximize = truth(getattr_or(opt, "maximize", PythonObject(False)))
    req.j = j
    req.t = Int(py=py_fields(opt)["t"])
    req.hyper = _hyper(opt, maximize, mn)
    req.n_total = n_total
    req.offsets = read_addr(py_fields(opt)["offsets"], "offsets")
    req.init = write_addr(py_fields(opt)["buf_initialized"], "buf_initialized")
    req.tokens = t_tokens
    _state_into(opt, req)
    req.p_dev = p_dev
    req.g_dev = truth(getattr_or(opt, "_g_dev", PythonObject(False)))
    req.device_op = op == OP_STEP_DEVICE
    var out = K.step(req)
    _ = keep^
    # the assignments of the native result
    set_field(opt, "t", PythonObject(out.t))
    keep_slot(opt, out.handle, module, not out.on_device)
    set_field(opt, "_host_fresh", PythonObject(py_fields(opt)["_res"] is None))
    var total = PythonObject(None)
    if out.clip_ran:
        total = PythonObject(Float64(out.total_norm))
        set_field(opt, "total_norm_", total)
        set_field(opt, "clip_coef_", PythonObject(Float64(out.coef)))
    else:
        set_field(opt, "total_norm_", PythonObject(None))
        set_field(opt, "clip_coef_", PythonObject(None))
    var lr = PythonObject(Float64(out.lr))
    set_field(opt, "lr_", lr)
    if req.has_sched:
        set_field(opt, "lr", lr)
    if out.packed >= 0:
        set_field(opt, "packed_", PythonObject(out.packed != 0))
    set_field(opt, "_g_dev", PythonObject(out.g_dev))
    return total


def door_io[K: OptKernels](
    module: PythonObject, opt: PythonObject, op_code: Int, tensors: PythonObject, name: String, tensor: Int
) raises -> PythonObject:
    """`_io`: the device transfers of `params_to_device`, `params_to_host`,
    `grads_to_device`, `grads_to_host`, `param_device` and the query.
    Returns the result word."""
    door_ready()
    opt_owner(opt, OPT_DOOR.get_or_create_ptr()[].context)
    var owhere = String(py=py_fields(opt)["_where"])
    var j = len(py_fields(opt)["params"])
    var n_total = Int(py=py_fields(opt)["n_total"])
    var req = PublicStep()
    if not (tensors is None):
        var seq = as_seq(tensors, name, owhere)
        var views = check_tensors(seq, name, owhere, True)
        if len(seq) != j:
            raise Error(
                "ValueError: mojolearn." + owhere + ": " + name + " does not match the optimizer's registry ("
                + String(j) + " tensors, " + String(n_total) + " floats)"
            )
        _addrs_of(views, req.g_addr, req.g_bytes)
    req.j = j
    req.n_total = n_total
    req.offsets = read_addr(py_fields(opt)["offsets"], "offsets")
    _state_into(opt, req)
    var r = K.io(req, op_code, tensor)
    try:
        keep_slot(opt, r[0], module, False)
        # A device allocation's handle stays an integer through publication.
        # The caller has prepared its owner before asking for this allocation.
        return PythonObject(r[1]) if op_code == 5 else PythonObject(Float64(r[1]))
    except error:
        if op_code == 5:
            try:
                _ = py_call(resident_entry(module, "train_dev_free"), r[1])
            except:
                pass
        raise error^


def door_zero(opt: PythonObject, grads: PythonObject) raises -> PythonObject:
    """`zero_grad`: +0.0 into every gradient buffer."""
    door_ready()
    opt_owner(opt, OPT_DOOR.get_or_create_ptr()[].context)
    var owhere = String(py=py_fields(opt)["_where"])
    var gs = as_seq(grads, "grads", owhere)
    var views = check_tensors(gs, "grads", owhere, True)
    var addrs = List[Int]()
    var nbytes = List[Int]()
    _addrs_of(views, addrs, nbytes)
    zero_tensors(addrs, nbytes)
    return PythonObject(None)


def resident_entry(owner: PythonObject, name: String) raises -> PythonObject:
    var context = OPT_DOOR.get_or_create_ptr()[].context
    if py_type(owner) is not context["module_type"]:
        raise Error("TypeError: optimizer resident state must retain its owning native extension")
    var entry = py_get(py_fields(owner), PythonObject(name))
    if py_type(entry) is not py_type(_bi().len):
        raise Error("TypeError: optimizer resident state requires a compiled native entry")
    return entry


def _sync_host[K: OptKernels](module: PythonObject, opt: PythonObject) raises:
    """`_download`: the slot's state into the mirrors, when they are stale,
    through the module that owns the slot."""
    if py_fields(opt)["_res"] is None or truth(py_fields(opt)["_host_fresh"]):
        return
    host_moments(opt)
    var h = Int(py=py_fields(opt)["_res"])
    var m = write_addr(py_fields(opt)["_m"], "exp_avg")
    var v = write_addr(py_fields(opt)["_v"], "exp_avg_sq")
    var n = Int(py=py_fields(opt)["n_total"])
    var owner = py_fields(opt)["_res_binding"]
    if owner is module:
        _ = K.download(h, m, v, n)
    else:
        _ = py_call(resident_entry(owner, "optimizer_resident_download"), PythonObject(h), PythonObject(m), PythonObject(v), PythonObject(n))
    set_field(opt, "_host_fresh", PythonObject(True))


def _upload[K: OptKernels](module: PythonObject, opt: PythonObject, m: Int, v: Int) raises:
    """`_upload_one`: assigned moments into the slot (0 leaves that moment)."""
    if py_fields(opt)["_res"] is None:
        return
    var h = Int(py=py_fields(opt)["_res"])
    var n = Int(py=py_fields(opt)["n_total"])
    var owner = py_fields(opt)["_res_binding"]
    if owner is module:
        _ = K.upload(h, m, v, n)
    else:
        _ = py_call(resident_entry(owner, "optimizer_resident_upload"), PythonObject(h), PythonObject(m), PythonObject(v), PythonObject(n))


def copy_array(source: PythonObject) raises -> PythonObject:
    var v = TensorView()
    if not tensor_view(source, v) or not v.c_contig or v.itemsize != 4:
        raise Error("TypeError: optimizer state copy requires a contiguous float32/int32 buffer")
    var result = new_array(v.shape, "<f4" if base_format(v.fmt) == "f" else "<i4")
    memcpy(dest=BP(unsafe_from_address=Int(py=result._addr)), src=BP(unsafe_from_address=v.addr), count=v.nbytes)
    return result


def door_state[K: OptKernels](module: PythonObject, opt: PythonObject, which: Int) raises -> PythonObject:
    """`state_dict()` (which 0) and the `exp_avg` / `exp_avg_sq` readings
    (1 / 2): the slot's state downloaded once per stale read."""
    door_ready()
    opt_owner(opt, OPT_DOOR.get_or_create_ptr()[].context)
    _sync_host[K](module, opt)
    set_field(opt, "_pristine", PythonObject(False))
    host_moments(opt)
    if which == STATE_EXP_AVG:
        return py_fields(opt)["_m"]
    if which == STATE_EXP_AVG_SQ:
        return py_fields(opt)["_v"]
    var out = Python.dict()
    py_setitem(out, PythonObject("t"), PythonObject(Int(py=py_fields(opt)["t"])))
    py_setitem(out, PythonObject("exp_avg"), copy_array(py_fields(opt)["_m"]))
    py_setitem(out, PythonObject("exp_avg_sq"), copy_array(py_fields(opt)["_v"]))
    py_setitem(out, PythonObject("buf_initialized"), copy_array(py_fields(opt)["buf_initialized"]))
    return out


def door_state_set[K: OptKernels](module: PythonObject, opt: PythonObject, which: Int, value: PythonObject) raises -> PythonObject:
    """`exp_avg = value` (which 1) and `exp_avg_sq = value` (2)."""
    door_ready()
    opt_owner(opt, OPT_DOOR.get_or_create_ptr()[].context)
    opt_value(value, OPT_DOOR.get_or_create_ptr()[].context)
    var name = String("exp_avg") if which == STATE_EXP_AVG else String("exp_avg_sq")
    set_field(opt, String("_m") if which == STATE_EXP_AVG else String("_v"), value)
    set_field(opt, "_pristine", PythonObject(False))
    var a = 0 if value is None else write_addr(value, name)
    if which == STATE_EXP_AVG:
        _upload[K](module, opt, a, 0)
    else:
        _upload[K](module, opt, 0, a)
    return PythonObject(None)


def door_load_state[K: OptKernels](
    module: PythonObject, opt: PythonObject, t: PythonObject, ea: PythonObject, c1: PythonObject,
    es: PythonObject, c2: PythonObject, bi: PythonObject, c3: PythonObject,
) raises -> PythonObject:
    """`load_state_dict` after the conversion of the state's arrays
    (`_buffer.as_f32_c` / `as_i32_c`, which may copy): the optimizer keeps
    its own copies, the sizes are checked, the state is uploaded."""
    door_ready()
    opt_owner(opt, OPT_DOOR.get_or_create_ptr()[].context)
    opt_scalar(t, OPT_DOOR.get_or_create_ptr()[].context)
    opt_value(ea, OPT_DOOR.get_or_create_ptr()[].context)
    opt_value(es, OPT_DOOR.get_or_create_ptr()[].context)
    opt_value(bi, OPT_DOOR.get_or_create_ptr()[].context)
    var m = ea if truth(c1) else copy_array(ea)
    var v = es if truth(c2) else copy_array(es)
    var n_total = Int(py=py_fields(opt)["n_total"])
    var owhere = String(py=py_fields(opt)["_where"])
    if Int(py=m.size) != n_total:
        raise Error("ValueError: mojolearn." + owhere + ".load_state_dict: exp_avg holds " + String(Int(py=m.size))
                    + " floats, this optimizer's registry is " + String(n_total))
    if Int(py=v.size) != n_total:
        raise Error("ValueError: mojolearn." + owhere + ".load_state_dict: exp_avg_sq holds " + String(Int(py=v.size))
                    + " floats, this optimizer's registry is " + String(n_total))
    var init_view = TensorView()
    if not tensor_view(bi, init_view) or init_view.count() != len(py_fields(opt)["params"]):
        raise Error("ValueError: buf_initialized does not match the optimizer parameter registry")
    set_field(opt, "_m", m)
    set_field(opt, "_v", v)
    set_field(opt, "_pristine", PythonObject(False))
    _upload[K](module, opt, write_addr(m, "exp_avg"), write_addr(v, "exp_avg_sq"))
    set_field(opt, "_host_fresh", PythonObject(True))
    set_field(opt, "buf_initialized", bi if truth(c3) else copy_array(bi))
    set_field(opt, "t", t)
    return PythonObject(None)


comptime _NORM_TYPE_REFUSAL = (
    " is REFUSED BY NAME. The profile has ONE norm, the two-level L2 of optimizer contract 3.1; an L-inf or "
    "L1 clip is a different reduction with its own clause, its own fixture and its own sabotage, and none "
    "of them exist (python/mojolearn/_training_impl.py)"
)
comptime _NONFINITE_REFUSAL = (
    "mojolearn.clip_grad_norm_: error_if_nonfinite=False is REFUSED BY NAME. The profile ALWAYS refuses a "
    "non-finite total norm (optimizer contract 8a, refuse_nonfinite_scalar in "
    "training/checks/optimizer_oracle.mojo) and there is no build in which it does not, so accepting False "
    "would be accepting a promise this surface cannot keep. Skipping a step whose gradient norm is "
    "non-finite is a useful thing to do and the contract says where it belongs: OUTSIDE the pinned region, "
    "as an explicit recorded branch in your own loop (python/mojolearn/_training_impl.py)"
)


def door_clip[K: OptKernels](
    grads: PythonObject, max_norm: PythonObject, norm_type: PythonObject, error_if_nonfinite: PythonObject,
    kwargs: PythonObject,
) raises -> PythonObject:
    """`clip_grad_norm_`: the refusals, the tensors, the in-place scale.
    Returns the pre-clip total norm."""
    door_ready()
    var bi = _bi()
    refuse_kwargs(kwargs, "clip_grad_norm_")
    if as_float(norm_type) != 2.0:
        raise Error("NotImplementedError: mojolearn.clip_grad_norm_: norm_type=" + py_repr(norm_type)
                    + _NORM_TYPE_REFUSAL)
    if not truth(error_if_nonfinite):
        raise Error("NotImplementedError: " + _NONFINITE_REFUSAL)
    var gs = as_seq(grads, "grads", "clip_grad_norm_")
    var views = check_tensors(gs, "grads", "clip_grad_norm_", True)
    var mn = as_float(max_norm)
    if not (mn > 0.0):
        raise Error(
            "ValueError: mojolearn.clip_grad_norm_: max_norm must be > 0, got " + py_repr(max_norm)
            + "; reaching this function IS the clip running, and 'no clipping' is spelled by not calling "
            + "it (training/estimator.mojo)"
        )
    var addrs = List[Int]()
    var nbytes = List[Int]()
    _addrs_of(views, addrs, nbytes)
    var r = K.clip(addrs, nbytes, Float32(mn))
    return PythonObject(Float64(r[0]))


def door_accumulate[K: OptKernels](microbatch_grads: PythonObject, tokens: PythonObject) raises -> PythonObject:
    """`accumulate_grads`: the microbatches checked, made contiguous where
    strided, combined by the clause 9.2 tree into new arrays."""
    door_ready()
    var bi = _bi()
    opt_value(microbatch_grads, OPT_DOOR.get_or_create_ptr()[].context)
    opt_scalar(tokens, OPT_DOOR.get_or_create_ptr()[].context)
    var parts = py_call(bi.list, microbatch_grads)
    if len(parts) == 0:
        raise Error("ValueError: mojolearn.accumulate_grads: no microbatches")
    var first = TensorView()
    var single = tensor_view(parts[0], first)
    var seqs = Python.list()
    for i in range(len(parts)):  # driver-loop: over microbatch arguments
        _ = py_call(seqs.append, as_seq(parts[i], "microbatch_grads[" + String(i) + "]", "accumulate_grads"))
    var probes = List[List[TensorView]]()
    for i in range(len(parts)):  # driver-loop: over microbatch arguments
        probes.append(check_tensors(seqs[i], "microbatch_grads[" + String(i) + "]", "accumulate_grads", False))
    var j0 = len(seqs[0])
    for i in range(len(parts)):  # driver-loop: over microbatch arguments
        var ji = len(seqs[i])
        if ji != j0:
            raise Error("ValueError: mojolearn.accumulate_grads: microbatch " + String(i) + " has " + String(ji)
                        + " tensors, microbatch 0 has " + String(j0))
        for k in range(ji):  # driver-loop: over parameter buffers
            if not same_shape(probes[i][k].shape, probes[0][k].shape):
                raise Error("ValueError: mojolearn.accumulate_grads: microbatch " + String(i) + " tensor "
                            + String(k) + " has shape " + shape_text(probes[i][k].shape)
                            + ", microbatch 0 has " + shape_text(probes[0][k].shape))
    var t_tokens = -1
    if not (tokens is None):
        t_tokens = Int(py=py_call(bi.int, tokens))
        if t_tokens < 1:
            raise Error("ValueError: mojolearn.accumulate_grads: tokens must be >= 1")
    var keep = List[List[Float32]]()
    var in_addrs = List[Int]()
    for i in range(len(parts)):  # driver-loop: over microbatch arguments
        var sq = seqs[i]
        for k in range(j0):  # driver-loop: over parameter buffers
            in_addrs.append(contiguous_addr(sq[k], probes[i][k], keep))
    var nbytes = List[Int]()
    var out_addrs = List[Int]()
    var result = Python.list()
    for k in range(j0):  # driver-loop: over parameter buffers: one result array each
        var a = new_array(probes[0][k].shape, "<f4")
        _ = py_call(result.append, a)
        out_addrs.append(Int(py=a._addr))
        nbytes.append(probes[0][k].nbytes)
    _ = K.accumulate(in_addrs, nbytes, out_addrs, len(parts), t_tokens)
    _ = keep^
    if single:
        return result[0]
    return result


# ---------------------------------------------------------------------------
# the C boundary of each entry (shared by the GPU and the CPU binding)


def entry_args(packed: PyObjectPtr, count: Int, name: String) raises -> PythonObject:
    var args = PythonObject(from_borrowed=packed)
    if len(args) != count:
        raise Error("TypeError: " + name + " expects " + String(count) + " arguments")
    return args


def entry_step[K: OptKernels](py_self: PyObjectPtr, packed: PyObjectPtr) -> PyObjectPtr:
    """optimizer_step(opt, grads, max_norm, op, tokens)."""
    try:
        var a = entry_args(packed, 5, "optimizer_step_public")
        return door_step[K](PythonObject(from_borrowed=py_self), a[0], a[1], a[2], Int(py=a[3]), a[4]).steal_data()
    except error:
        return optimizer_python_error(error)


def entry_io[K: OptKernels](py_self: PyObjectPtr, packed: PyObjectPtr) -> PyObjectPtr:
    """optimizer_io(opt, op, tensors, name, tensor)."""
    try:
        var a = entry_args(packed, 5, "optimizer_io")
        return door_io[K](PythonObject(from_borrowed=py_self), a[0], Int(py=a[1]), a[2], String(py=a[3]),
                          Int(py=a[4])).steal_data()
    except error:
        return optimizer_python_error(error)


def entry_zero(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_zero(opt, grads)."""
    try:
        var a = entry_args(packed, 2, "optimizer_zero")
        return door_zero(a[0], a[1]).steal_data()
    except error:
        return optimizer_python_error(error)


def entry_state[K: OptKernels](py_self: PyObjectPtr, packed: PyObjectPtr) -> PyObjectPtr:
    """optimizer_state(opt, which)."""
    try:
        var a = entry_args(packed, 2, "optimizer_state")
        return door_state[K](PythonObject(from_borrowed=py_self), a[0], Int(py=a[1])).steal_data()
    except error:
        return optimizer_python_error(error)


def entry_state_set[K: OptKernels](py_self: PyObjectPtr, packed: PyObjectPtr) -> PyObjectPtr:
    """optimizer_state_set(opt, which, value)."""
    try:
        var a = entry_args(packed, 3, "optimizer_state_set")
        return door_state_set[K](PythonObject(from_borrowed=py_self), a[0], Int(py=a[1]), a[2]).steal_data()
    except error:
        return optimizer_python_error(error)


def entry_load_state[K: OptKernels](py_self: PyObjectPtr, packed: PyObjectPtr) -> PyObjectPtr:
    """optimizer_load_state(opt, t, exp_avg, copied, exp_avg_sq, copied, buf_initialized, copied)."""
    try:
        var a = entry_args(packed, 8, "optimizer_load_state")
        return door_load_state[K](PythonObject(from_borrowed=py_self), a[0], a[1], a[2], a[3], a[4], a[5], a[6],
                                  a[7]).steal_data()
    except error:
        return optimizer_python_error(error)


def entry_clip[K: OptKernels](py_self: PyObjectPtr, packed: PyObjectPtr) -> PyObjectPtr:
    """optimizer_clip(grads, max_norm, norm_type, error_if_nonfinite, kwargs)."""
    try:
        var a = entry_args(packed, 5, "optimizer_clip")
        return door_clip[K](a[0], a[1], a[2], a[3], a[4]).steal_data()
    except error:
        return optimizer_python_error(error)


def entry_accumulate[K: OptKernels](py_self: PyObjectPtr, packed: PyObjectPtr) -> PyObjectPtr:
    """optimizer_accumulate(microbatch_grads, tokens)."""
    try:
        var a = entry_args(packed, 2, "optimizer_accumulate")
        return door_accumulate[K](a[0], a[1]).steal_data()
    except error:
        return optimizer_python_error(error)


def entry_init(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_init(opt, params)."""
    try:
        var a = entry_args(packed, 2, "optimizer_init")
        return door_init(a[0], a[1]).steal_data()
    except error:
        return optimizer_python_error(error)
