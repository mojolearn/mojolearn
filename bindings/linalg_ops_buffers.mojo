# SPDX-License-Identifier: Apache-2.0
# Buffer readings of the native linalg owner (bindings/linalg_ops_native.mojo):
# `_bufcheck.probe`'s facts and `dtype_name`'s refusal name, `_buffer.empty`'s
# zero-filled Arrays of any public dtype, and `_buffer.as_*_c`'s layout copy or
# element conversion (Array.astype's rules: float to float32 round to nearest
# even, integers range checked, float to integer truncated toward zero), so the
# kernels read the same element values the Python shells handed them.
from std.math import isinf
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_setattr, py_type
from bindings.native_buffer_owner import native_storage_allocate
from bindings.classical_pandas import classical_admit_value
from bindings.classical_arrays import classical_is_array
from providers.python.borrowed_buffer_pyapi import AdmittedBorrowedBuffer
from providers.python.array_metadata_pyapi import array_owner_once
from bindings.classical_arrays import classical_wrap_array, _store_code, classical_address
from bindings.family_common import family_isinstance, family_view
from bindings.optimizer_public_door import (
    TensorView, _get_buffer, _release_buffer, _contiguous, base_format, _format_name, _typestr_name,
    _array_code, shape_text,
)
from core.cast_in import cast_f32_bits, f16_to_f32_bits, CAST_KIND_UNSIGNED, CAST_KIND_SIGNED, CAST_KIND_FLOAT, CAST_KIND_BOOL

comptime LIN_IP = MutPointer[Int, MutUntrackedOrigin]
comptime LIN_BP = MutPointer[UInt8, MutUntrackedOrigin]


struct LinBuf(Movable):
    """A C-ordered buffer the kernels read: the object that owns it (kept
    alive across the call), its address and its shape."""
    var keep: PythonObject
    var addr: Int
    var shape: List[Int]
    var pin: Optional[AdmittedBorrowedBuffer]
    var owner: PythonObject

    def __init__(out self, keep: PythonObject, addr: Int, var shape: List[Int],
                 var pin: Optional[AdmittedBorrowedBuffer] = None, owner: PythonObject = PythonObject(None)):
        self.keep = keep
        self.addr = addr
        self.shape = shape^
        self.pin = pin^
        self.owner = owner

    def dim(self, k: Int) -> Int:
        return self.shape[k]

    def rank(self) -> Int:
        return len(self.shape)

    def count(self) -> Int:
        var n = 1
        for i in range(len(self.shape)):  # driver-loop: over shape words
            n *= self.shape[i]
        return n

    def text(self) -> String:
        return shape_text(self.shape)


def lin_truth(value: PythonObject) raises -> Bool:
    """`bool(value)`."""
    return Bool(py=py_call(Python.import_module("builtins").bool, value))


def lin_view(obj: PythonObject, context: PythonObject, mut v: TensorView) raises -> Bool:
    """`_bufcheck.probe(obj)`'s facts in `v` (the export released before
    this returns); False when `obj` does not export a buffer. A mojolearn
    Array is read off its own slots (`_buffer.Buf._from_array`); `v.dtype`
    then holds its typestr."""
    v = TensorView()
    if family_isinstance(obj, context["array_class"]):
        v.is_array = True
        v.addr = Int(py=obj._addr)
        v.nbytes = Int(py=obj.nbytes)
        v.itemsize = Int(py=obj.itemsize)
        v.readonly = lin_truth(obj._readonly)
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
        v.dtype = t
        return True
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
        var fp = LIN_BP(unsafe_from_address=p[5])
        var text = String("")
        for k in range(64):  # driver-loop: over format characters of one buffer
            if fp[k] == 0:
                break
            text += chr(Int(fp[k]))
        v.fmt = text
    if ndim > 0:
        if p[6] != 0:
            var sp = LIN_IP(unsafe_from_address=p[6])
            for k in range(ndim):  # driver-loop: over shape words of one buffer
                v.shape.append(sp[k])
        else:
            v.shape.append(v.nbytes // max(v.itemsize, 1))
        if p[7] != 0:
            var tp = LIN_IP(unsafe_from_address=p[7])
            for k in range(ndim):  # driver-loop: over stride words of one buffer
                v.strides.append(tp[k])
            v.has_strides = True
        if p[8] != 0:
            var op = LIN_IP(unsafe_from_address=p[8])
            for k in range(ndim):  # driver-loop: over suboffset words of one buffer
                if op[k] >= 0:
                    indirect = True
    _release_buffer(view)
    _ = view^
    if indirect:
        # `Buf._fill` refuses an indirect buffer with a TypeError
        return False
    v.c_contig = _contiguous(v.shape, v.strides, v.has_strides, v.itemsize)
    v.dtype = String("")
    return True


def lin_is_f32(v: TensorView) -> Bool:
    """`_bufcheck.is_native_f32(pb.format)`."""
    return base_format(v.fmt) == "f"


def lin_dtype_name(obj: PythonObject, v: TensorView) raises -> String:
    """`_bufcheck.dtype_name(obj, pb)`: NumPy's `dtype.name` when the object
    carries one, an Array typestr's name, else the struct code's."""
    var b = Python.import_module("builtins")
    var dt = py_call(b.getattr, obj, "dtype", PythonObject(None))
    var name = py_call(b.getattr, dt, "name", PythonObject(None))
    if family_isinstance(name, b.str):
        return String(py=name)
    if family_isinstance(dt, b.str):
        var t = _typestr_name(String(py=dt))
        if t != "":
            return t
    return _format_name(base_format(v.fmt), v.itemsize, v.fmt)


def lin_width(typestr: String) raises -> Int:
    if typestr == "<f4" or typestr == "<i4" or typestr == "<u4":
        return 4
    if typestr == "<f8" or typestr == "<i8":
        return 8
    if typestr == "<u2" or typestr == "<f2":
        return 2
    if typestr == "<i1" or typestr == "<u1":
        return 1
    raise Error("TypeError: " + typestr + " is not an Array dtype")


def lin_tuple(shape: List[Int]) raises -> PythonObject:
    """A shape as a Python tuple."""
    var dims = Python.list()
    for i in range(len(shape)):  # driver-loop: over shape words
        _ = py_call(dims.append, PythonObject(shape[i]))
    return py_call(Python.import_module("builtins").tuple, dims)


def lin_array(shape: List[Int], typestr: String, context: PythonObject) raises -> PythonObject:
    """`_buffer.empty(shape, typestr)`: a new zero-filled C-order Array."""
    var count = 1
    for i in range(len(shape)):  # driver-loop: over shape words
        count *= shape[i]
    var store = py_call(context["array_type"], PythonObject(_store_code(typestr)),
                        py_call(context["native_bytes_type"], count * lin_width(typestr)))
    return classical_wrap_array(store, lin_tuple(shape), typestr, "C", context)


def lin_new(shape: List[Int], typestr: String, context: PythonObject) raises -> LinBuf:
    """A new zero-filled C-order Array as a LinBuf."""
    var out = lin_array(shape, typestr, context)
    var addr = Int(py=classical_address(out))
    _ = out  # keep the Python object alive through the native call above
    return LinBuf(out, addr, shape.copy())


def _lin_kind(v: TensorView, mut item: Int) raises -> Int:
    """The CastIn element kind of a viewed buffer (its item size in `item`);
    -1 for a format no Array dtype holds."""
    item = v.itemsize
    if v.is_array:
        var t = v.dtype
        if t == "<f4" or t == "<f8" or t == "<f2":
            return CAST_KIND_FLOAT
        if t == "<i1" or t == "<i4" or t == "<i8":
            return CAST_KIND_SIGNED
        return CAST_KIND_UNSIGNED
    var code = base_format(v.fmt)
    if code == "f" or code == "d" or code == "e":
        return CAST_KIND_FLOAT
    if code == "?":
        return CAST_KIND_BOOL
    if code == "b" or code == "h" or code == "i" or code == "l" or code == "q" or code == "n":
        return CAST_KIND_SIGNED
    if code == "B" or code == "H" or code == "I" or code == "L" or code == "Q" or code == "N":
        return CAST_KIND_UNSIGNED
    return -1


def _lin_target(typestr: String, mut item: Int) raises -> Int:
    item = lin_width(typestr)
    if typestr == "<f4" or typestr == "<f8" or typestr == "<f2":
        return CAST_KIND_FLOAT
    if typestr == "<i1" or typestr == "<i4" or typestr == "<i8":
        return CAST_KIND_SIGNED
    return CAST_KIND_UNSIGNED


def _lin_ctype(typestr: String) -> String:
    """The C type `array.array` names in its range refusal."""
    if typestr == "<i1":
        return "signed char"
    if typestr == "<i4":
        return "signed integer"
    if typestr == "<u2":
        return "unsigned short"
    if typestr == "<u1":
        return "unsigned byte integer"
    if typestr == "<u4":
        return "unsigned int"
    return "signed long long"


def _lin_bounds(typestr: String, mut low: Int, mut high: Int):
    if typestr == "<i1":
        low = -128
        high = 127
    elif typestr == "<u1":
        low = 0
        high = 255
    elif typestr == "<u2":
        low = 0
        high = 65535
    elif typestr == "<i4":
        low = -(1 << 31)
        high = (1 << 31) - 1
    elif typestr == "<u4":
        low = 0
        high = (1 << 32) - 1
    else:
        low = Int.MIN
        high = Int.MAX


def _lin_raw(base: Int, off: Int, item: Int) -> UInt64:
    """The item at byte `off` from `base`, zero-extended (little-endian)."""
    var p = LIN_BP(unsafe_from_address=base + off)
    var raw = UInt64(0)
    for j in range(item):  # small-loop(bytes of one item): at most 8
        raw |= UInt64(p[j]) << UInt64(8 * j)
    return raw


def _lin_store(dst: Int, t: Int, item: Int, raw: UInt64):
    var p = LIN_BP(unsafe_from_address=dst + t * item)
    for j in range(item):  # small-loop(bytes of one item): at most 8
        p[j] = UInt8((raw >> UInt64(8 * j)) & UInt64(0xFF))


def _lin_float(raw: UInt64, item: Int) -> Float64:
    if item == 2:
        return bitcast[DType.float32](f16_to_f32_bits(raw)).cast[DType.float64]()
    if item == 4:
        return bitcast[DType.float32](UInt32(raw)).cast[DType.float64]()
    return bitcast[DType.float64](raw)


def _lin_integer(raw: UInt64, kind: Int, item: Int, typestr: String) raises -> Int:
    """One item as an exact integer for an integer target, refused as
    Array.astype refuses (a NaN, an infinity, a value outside the target)."""
    var ctype = _lin_ctype(typestr)
    var value: Int
    if kind == CAST_KIND_BOOL:
        value = 1 if raw != 0 else 0
    elif kind == CAST_KIND_SIGNED:
        var bits = raw
        if item < 8 and (raw >> UInt64(8 * item - 1)) & UInt64(1) != 0:
            bits = raw | (~UInt64(0) << UInt64(8 * item))
        value = Int(bitcast[DType.int64](bits))
    elif kind == CAST_KIND_UNSIGNED:
        if raw > UInt64(Int.MAX):
            raise Error("OverflowError: mojolearn: value does not fit " + typestr + ": " + ctype + " is greater than maximum")
        value = Int(raw)
    else:
        var x = _lin_float(raw, item)
        if x != x:
            raise Error("cannot convert float NaN to integer")
        if isinf(x):
            raise Error("OverflowError: cannot convert float infinity to integer")
        if x >= 9223372036854775808.0 or x < -9223372036854775808.0:
            raise Error("OverflowError: mojolearn: value does not fit " + typestr)
        value = Int(x)  # truncation toward zero
    var low = 0
    var high = 0
    _lin_bounds(typestr, low, high)
    if value > high:
        raise Error("OverflowError: mojolearn: value does not fit " + typestr + ": " + ctype + " is greater than maximum")
    if value < low:
        raise Error("OverflowError: mojolearn: value does not fit " + typestr + ": " + ctype + " is less than minimum")
    return value


def lin_convert(v: TensorView, typestr: String, dst: Int, name: String) raises:
    """Every element of the viewed buffer, in C order, into the C-order
    `typestr` store at `dst`: a byte copy for the same dtype, else one
    conversion per element by Array.astype's rules."""
    var item = 0
    var kind = _lin_kind(v, item)
    if kind < 0:
        raise Error("TypeError: mojolearn: " + name + " has buffer format '" + v.fmt + "', which is not a numeric array")
    var titem = 0
    var tkind = _lin_target(typestr, titem)
    var same = kind == tkind and item == titem
    var nd = len(v.shape)
    var strides = List[Int](length=nd, fill=0)
    if v.has_strides:
        for k in range(nd):  # driver-loop: over stride words
            strides[k] = v.strides[k]
    else:
        var step = v.itemsize
        for k in range(nd - 1, -1, -1):  # driver-loop: over stride words
            strides[k] = step
            step *= v.shape[k]
    var count = v.count()
    var idx = List[Int](length=nd, fill=0)
    var off = 0
    # cpu-route: the host input conversion `_buffer.as_*_c` made (the base
    # binding's cast_elements / layout copy), before the kernel call
    for t in range(count):  # cpu-route: host input conversion, element by element
        var raw = _lin_raw(v.addr, off, item)
        if same:
            _lin_store(dst, t, titem, raw)
        elif tkind == CAST_KIND_FLOAT and titem == 4:
            _lin_store(dst, t, 4, UInt64(cast_f32_bits(raw, kind, item)))
        elif tkind == CAST_KIND_FLOAT and titem == 8:
            var x = Float64(0.0)
            if kind == CAST_KIND_FLOAT:
                x = _lin_float(raw, item)
            else:
                x = Float64(_lin_integer(raw, kind, item, "<i8"))
            _lin_store(dst, t, 8, bitcast[DType.uint64](x))
        else:
            var value = _lin_integer(raw, kind, item, typestr)
            _lin_store(dst, t, titem, UInt64(bitcast[DType.uint64](Int64(value))))
        var k = nd - 1
        while k >= 0:  # driver-loop: over the axes of one index step
            idx[k] += 1
            off += strides[k]
            if idx[k] < v.shape[k]:
                break
            off -= strides[k] * v.shape[k]
            idx[k] = 0
            k -= 1


def lin_c_order(obj: PythonObject, v: TensorView, typestr: String, name: String,
                context: PythonObject) raises -> LinBuf:
    """`obj` (already the dtype `typestr`, or converted to it) in C order:
    the buffer itself when it already is, else one converted copy."""
    var item = 0
    var kind = _lin_kind(v, item)
    var titem = 0
    var tkind = _lin_target(typestr, titem)
    if kind == tkind and item == titem and v.c_contig:
        return LinBuf(obj, v.addr, v.shape.copy())
    var out = lin_new(v.shape, typestr, context)
    lin_convert(v, typestr, out.addr, name)
    _ = out  # keep the Python object alive through the native call above
    return out^


def lin_typed(source: PythonObject, typestr: String, ndim: Int, name: String,
              context: PythonObject) raises -> LinBuf:
    """`_buffer.as_*_c(source, ndim=ndim, name=name)` for `typestr` ("<f4",
    "<u2", "<i1", "<i4"): the admitted input with `_as_typed`'s rank and
    emptiness refusals (`ndim` < 0 accepts any rank), then the C-order
    buffer of that dtype, borrowed when it already is one."""
    var view = family_view(source, ndim, name, context)
    var v = TensorView()
    if not lin_view(view, context, v):
        raise Error("TypeError: mojolearn: " + name + " does not support the buffer protocol")
    var result = lin_c_order(view, v, typestr, name, context)
    _ = view^
    return result^


struct LinAdmission(Movable):
    # One operation-local metadata capture. Exact native exporter admission
    # precedes PyObject_GetBuffer; the pin lives through the numerical call.
    var keep: PythonObject
    var view: TensorView
    var pin: Optional[AdmittedBorrowedBuffer]
    var owner: PythonObject

    def __init__(out self, value: PythonObject, context: PythonObject) raises:
        classical_admit_value(value, context)
        self.keep = value
        self.view = TensorView()
        self.pin = None
        self.owner = PythonObject(None)
        if classical_is_array(value, context):
            var slots = context["array_fields"]
            var cls = context["array_class"]
            # Keep the admitted storage owners themselves, so replacement of
            # the input Array's slots while native code releases the GIL cannot
            # free the captured address. Each slot is read once through its
            # import-pinned native descriptor rather than a mutable class hook.
            self.owner = array_owner_once(value, context)
            var address = py_call(slots["_addr"], value, cls)
            var item = py_call(slots["itemsize"], value, cls)
            var readonly = py_call(slots["_readonly"], value, cls)
            var dtype = py_call(slots["dtype"], value, cls)
            var shape = py_call(slots["shape"], value, cls)
            var strides = py_call(slots["strides"], value, cls)
            if (py_type(address) is not py_type(PythonObject(0)) or py_type(item) is not py_type(PythonObject(0))
                or py_type(readonly) is not py_type(PythonObject(True)) or py_type(dtype) is not py_type(PythonObject(""))
                or py_type(shape) is not py_type(Python.tuple()) or py_type(strides) is not py_type(Python.tuple())):
                raise Error("TypeError: Array metadata must contain native integer, boolean, string and tuple values")
            if len(shape) != len(strides):
                raise Error("TypeError: Array shape and stride ranks differ")
            self.view.is_array = True
            self.view.addr = Int(py=address)
            self.view.itemsize = Int(py=item)
            self.view.readonly = Bool(py=readonly)
            self.view.dtype = String(py=dtype)
            self.view.fmt = _array_code(self.view.dtype)
            for axis in range(len(shape)):  # small-loop(Array rank metadata admitted once)
                if py_type(shape[axis]) is not py_type(PythonObject(0)) or py_type(strides[axis]) is not py_type(PythonObject(0)):
                    raise Error("TypeError: Array dimensions and strides must be native integers")
                self.view.shape.append(Int(py=shape[axis]))
                self.view.strides.append(Int(py=strides[axis]))
            self.view.nbytes = self.view.count() * self.view.itemsize
            self.view.has_strides = True
            self.view.c_contig = _contiguous(self.view.shape, self.view.strides, True, self.view.itemsize)
            return
        self.pin = AdmittedBorrowedBuffer(value)
        ref layout = self.pin.value().layout.value()
        self.view.addr = layout.address
        self.view.itemsize = layout.itemsize
        self.view.readonly = layout.readonly
        self.view.nbytes = layout.count * layout.itemsize
        self.view.shape = layout.shape.copy()
        self.view.strides = layout.strides.copy()
        self.view.has_strides = True
        self.view.fmt = self.pin.value().format
        if self.view.fmt.startswith(">") or self.view.fmt.startswith("!"):
            raise Error("TypeError: mojolearn: big-endian buffers are not supported")
        self.view.dtype = String("")
        self.view.c_contig = _contiguous(self.view.shape, self.view.strides, True, layout.itemsize)
        for axis in range(len(layout.suboffsets)):  # small-loop(buffer rank)
            if layout.suboffsets[axis] >= 0:
                raise Error("TypeError: mojolearn: indirect buffers are not supported")



struct LinBufferPin(Movable, Writable):
    var pin: Optional[AdmittedBorrowedBuffer]

    def __init__(out self, var pin: Optional[AdmittedBorrowedBuffer]):
        self.pin = pin^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("LinBufferPin")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def lin_borrow_array(var input: LinAdmission, typestr: String, context: PythonObject) raises -> LinBuf:
    # Array.from_buffer's zero-copy ownership semantics from metadata already
    # admitted once. The public CPython raw-memory view does not ask the exporter
    # for its metadata again; the owned pin keeps storage valid for its lifetime.
    var v = input.view
    var array = native_storage_allocate(context["array_class"])
    var pin = PythonObject(alloc=LinBufferPin(input.pin^))
    ref cpy = Python().cpython()
    var raw = cpy.lib.get_function[PyObjectPtr]("PyMemoryView_FromMemory")(
        MutPointer[Int8, MutUntrackedOrigin](unsafe_from_address=v.addr), v.nbytes,
        Int32(0x100 if v.readonly else 0x200))
    if not raw:
        raise cpy.unsafe_get_error()
    var bytes_view = PythonObject(from_owned=raw)
    var view = py_call(bytes_view.cast, _store_code(typestr))
    py_setattr(array, "_store", input.keep)
    py_setattr(array, "_base", PythonObject(None))
    py_setattr(array, "_pin", pin)
    py_setattr(array, "_mv", view)
    py_setattr(array, "_addr", PythonObject(v.addr))
    py_setattr(array, "_readonly", PythonObject(v.readonly))
    py_setattr(array, "shape", lin_tuple(v.shape))
    var canonical = List[Int](length=len(v.shape), fill=0)
    var step = v.itemsize
    for axis in range(len(v.shape) - 1, -1, -1):  # small-loop(known output view shape)
        canonical[axis] = step
        step *= v.shape[axis]
    py_setattr(array, "strides", lin_tuple(canonical))
    py_setattr(array, "dtype", PythonObject(typestr))
    py_setattr(array, "ndim", PythonObject(len(v.shape)))
    py_setattr(array, "size", PythonObject(v.count()))
    py_setattr(array, "itemsize", PythonObject(v.itemsize))
    py_setattr(array, "nbytes", PythonObject(v.nbytes))
    py_setattr(array, "order", PythonObject("C"))
    return LinBuf(array, v.addr, v.shape.copy())

def lin_admitted_c(var input: LinAdmission, typestr: String, name: String,
                   context: PythonObject, force_copy: Bool = False) raises -> LinBuf:
    var item = 0
    var kind = _lin_kind(input.view, item)
    var target_item = 0
    var target_kind = _lin_target(typestr, target_item)
    if not force_copy and kind == target_kind and item == target_item and input.view.c_contig:
        if input.view.is_array:
            return LinBuf(input.keep, input.view.addr, input.view.shape.copy(), input.pin^, input.owner)
        return lin_borrow_array(input^, typestr, context)
    var result = lin_new(input.view.shape, typestr, context)
    lin_convert(input.view, typestr, result.addr, name)
    _ = input^
    return result^
