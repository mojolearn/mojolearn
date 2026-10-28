# SPDX-License-Identifier: Apache-2.0
# Native zero-copy Array construction. Admission precedes buffer acquisition;
# only retained native slots publish the completed object, without callbacks.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module import native_loaded_module
from bindings.native_numpy_metadata import native_numpy_dtype
from bindings.native_buffer_owner import native_storage_allocate
from bindings.py_typed_call import py_call_typed, typed_python_error
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from providers.python.borrowed_buffer_pyapi import admit_native_buffer_exporter, AdmittedBorrowedBuffer


def _namespace(cls: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var fields = py_call(b.type.__dict__["__dict__"].__get__, cls)
    var names = py_call(b.list, fields.keys())
    for i in range(len(names)):  # driver-loop: native class metadata keys before lookup
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: Array storage classes require native string keys")
    return fields


def _is_array_class(cls: PythonObject, context: PythonObject) raises -> Bool:
    var b = native_loaded_module("builtins")
    if py_type(cls) is not b.type:
        return False
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
    for i in range(len(ancestry)):  # driver-loop: native Array class ancestry
        if ancestry[i] is context["array_class"]:
            return True
    return False


def _slots(cls: PythonObject, context: PythonObject) raises -> PythonObject:
    if not _is_array_class(cls, context):
        raise Error("TypeError: Array.from_buffer builds a mojolearn Array class")
    var b = native_loaded_module("builtins")
    var slots = context["from_buffer_slots"]
    var names = py_call(b.list, slots.keys())
    var base = context["array_class"]
    var base_fields = _namespace(base)
    for i in range(len(names)):  # small-loop(retained Array member descriptors)
        if py_get(base_fields, names[i]) is not slots[names[i]]:
            raise Error("TypeError: Array native storage descriptors changed")
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
    for i in range(len(ancestry)):  # driver-loop: admit passive subclass storage and lifetime
        var fields = _namespace(ancestry[i])
        if py_contains(fields, "__getattr__") or py_contains(fields, "__del__"):
            raise Error("TypeError: Array.from_buffer does not invoke Python metadata or lifetime hooks")
        for name in ["__getattribute__", "__setattr__", "__new__"]:  # small-loop(native storage access protocols)
            var member = py_get(fields, PythonObject(name))
            if member is not None and member is not b.object.__dict__[name]:
                raise Error("TypeError: Array.from_buffer requires native allocation and storage access")
        if ancestry[i] is base:
            continue
        for j in range(len(names)):  # small-loop(refuse subclass field overrides before slot access)
            if py_contains(fields, names[j]):
                raise Error("TypeError: Array.from_buffer requires its inherited native storage slots")
    return slots


def _tuple(values: List[Int]) raises -> PythonObject:
    var result = Python.list()
    for i in range(len(values)):  # small-loop(captured buffer shape or stride words)
        _ = py_call(result.append, PythonObject(values[i]))
    return py_call(native_loaded_module("builtins").tuple, result)


def _dtype(original: String, itemsize: Int, context: PythonObject) raises -> String:
    var fmt = original
    if fmt.startswith("@") or fmt.startswith("=") or fmt.startswith("<") or fmt.startswith(">") or fmt.startswith("!") or fmt.startswith("|"):
        if (fmt.startswith(">") or fmt.startswith("!")) and itemsize > 1:
            raise Error("TypeError: mojolearn: big-endian buffer format is not supported")
        var bare = String(fmt[byte=1:])
        fmt = bare^
    var dtype = String("")
    if fmt == "f":
        dtype = "<f4"
    elif fmt == "d":
        dtype = "<f8"
    elif fmt == "e":
        dtype = "<f2"
    elif fmt == "b" or fmt == "h" or fmt == "i" or fmt == "l" or fmt == "q" or fmt == "n":
        dtype = "<i" + String(itemsize)
    elif fmt == "B" or fmt == "H" or fmt == "I" or fmt == "L" or fmt == "Q" or fmt == "N" or fmt == "P":
        dtype = "<u" + String(itemsize)
    elif fmt.startswith("Z"):
        raise Error("ValueError: Complex data not supported: buffer format is complex")
    if dtype == "" or not py_contains(context["codes"], PythonObject(dtype)):
        raise Error("TypeError: mojolearn: buffer format " + original + " is not a supported Array dtype")
    if itemsize != Int(py=context["itemsizes"][PythonObject(dtype)]):
        raise Error("TypeError: mojolearn: buffer format and itemsize disagree")
    return dtype


def _admit_buffer(value: PythonObject, context: PythonObject, depth: Int = 0) raises:
    if depth > 64:
        raise Error("TypeError: native buffer exporter nesting exceeds the supported limit")
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is b.memoryview:
        # The typed native descriptor also preserves a released view's ValueError.
        var owner = py_call_typed(b.memoryview.__dict__["obj"].__get__, value)
        if owner is not None:
            _admit_buffer(owner, context, depth + 1)
        return
    if cls is context["from_buffer_mmap"]:
        # Array setup retains the canonical mmap extension independently of the
        # optional corpus API; exact native mmap remains a supported exporter.
        var fields = _namespace(cls)
        var hooks = context["from_buffer_mmap_hooks"]
        for name in ["__buffer__", "__release_buffer__"]:  # small-loop(retained native mmap buffer protocols)
            if py_get(fields, PythonObject(name)) is not hooks[PythonObject(name)]:
                raise Error("TypeError: mmap buffer protocols changed after native Array connection")
        return
    try:
        admit_native_buffer_exporter(value)
        return
    except:
        pass  # scalar exporters below are admitted by native dtype identities
    ref cpy = Python().cpython()
    var flags = cpy.lib.get_function[UInt64]("PyType_GetFlags")(cls._obj_ptr)
    if (flags & UInt64(512)) == 0 and py_get(context["modules"], "numpy") is not None:
        # NumPy scalar buffers are zero-dimensional native exporters too.
        # Derive exact scalar identities from the checked C dtype constructor,
        # not mutable NumPy scalar-name exports or Python isinstance hooks.
        var dtype_class = native_numpy_dtype(context)
        var scalar_types = py_get(context, "from_buffer_scalar_types")
        if scalar_types is None:
            scalar_types = Python.list()
            for code in ["<f4", "<f8", "<i4", "<i8", "<u4", "<u1", "<f2", "<u2", "<i1", "<c8", "<c16"]:  # small-loop(supported native scalar exporter metadata and complex refusal)
                var dtype = py_call(dtype_class, PythonObject(code))
                _ = py_call(scalar_types.append, dtype.type)
            scalar_types = py_call(b.tuple, scalar_types)
            py_setitem(context, "from_buffer_scalar_types", scalar_types)
        for i in range(len(scalar_types)):  # small-loop(exact supported native scalar exporters)
            if cls is scalar_types[i]:
                return
    raise Error("TypeError: Array.from_buffer requires native buffer exporters without Python protocols")


def array_from_buffer(cls: PythonObject, obj: PythonObject, context: PythonObject) raises -> PythonObject:
    var slots = _slots(cls, context)
    var b = native_loaded_module("builtins")
    var store = obj
    var base = PythonObject(None)
    var captured = Python.dict()
    var is_array = _is_array_class(py_type(obj), context)
    if is_array:
        var source_slots = _slots(py_type(obj), context)
        var names = py_call(b.list, source_slots.keys())
        for i in range(len(names)):  # small-loop(capture each native Array field exactly once)
            py_setitem(captured, names[i], py_call(source_slots[names[i]].__get__, obj, py_type(obj)))
        store = captured["_store"]
        if store is None:
            store = captured["_mv"]
            if py_type(store) is not b.memoryview or store.obj is None:
                raise Error("TypeError: Array.from_buffer needs retained native backing storage")
        base = obj
    _admit_buffer(store, context)
    # The first native memoryview captures exporter metadata and pins its owner.
    # All subsequent layout reads use this private view, never the source again.
    ref cpy = Python().cpython()
    var raw_pin = cpy.lib.get_function[PyObjectPtr]("PyMemoryView_FromObject")(store._obj_ptr)
    if not raw_pin:
        raise typed_python_error()
    var pin = PythonObject(from_owned=raw_pin)
    var borrowed = AdmittedBorrowedBuffer(pin)
    var layout = borrowed.layout.value().copy()
    var c_order = layout.count == 0 or layout.contiguous()
    var f_order = True
    var stride = layout.itemsize
    for i in range(len(layout.shape)):  # small-loop(admitted Fortran storage strides)
        f_order = f_order and layout.suboffsets[i] < 0 and (layout.count == 0 or layout.shape[i] <= 1 or layout.strides[i] == stride)
        stride *= layout.shape[i]
    if not c_order and not f_order:
        raise Error("ValueError: mojolearn: from_buffer needs a contiguous buffer")
    var shape = layout.shape.copy()
    var dtype = _dtype(borrowed.format, layout.itemsize, context) if not is_array else String("")
    var order = String("C") if c_order else String("F")
    var address = layout.address
    var readonly = layout.readonly
    if is_array:
        if (py_type(captured["shape"]) is not b.tuple or py_type(captured["strides"]) is not b.tuple
                or py_type(captured["dtype"]) is not b.str or py_type(captured["order"]) is not b.str
                or py_type(captured["_readonly"]) is not b.bool):
            raise Error("TypeError: Array.from_buffer requires native shape, dtype and layout metadata")
        for name in ["_addr", "size", "nbytes", "ndim", "itemsize"]:  # small-loop(native Array size/address schema)
            if py_type(captured[name]) is not b.int:
                raise Error("TypeError: Array.from_buffer requires native integer size metadata")
        dtype = String(captured["dtype"])
        order = String(captured["order"])
        if not py_contains(context["codes"], PythonObject(dtype)) or (order != "C" and order != "F"):
            raise Error("TypeError: Array.from_buffer requires supported contiguous Array storage")
        shape = List[Int]()
        var dims = captured["shape"]
        for i in range(len(dims)):  # small-loop(admit captured Array dimensions)
            if py_type(dims[i]) is not b.int or Int(py=dims[i]) < 0:
                raise Error("TypeError: Array.from_buffer requires nonnegative integer dimensions")
            shape.append(Int(py=dims[i]))
        address = Int(py=captured["_addr"])
        readonly = Bool(py=captured["_readonly"])
        if layout.readonly and not readonly:
            raise Error("TypeError: Array metadata cannot make readonly backing storage writable")
    var itemsize = Int(py=context["itemsizes"][PythonObject(dtype)])
    var count = 1
    for i in range(len(shape)):  # small-loop(admitted output shape product)
        if shape[i] != 0 and count > 9223372036854775807 // shape[i]:
            raise Error("ValueError: Array shape exceeds native indexing")
        count *= shape[i]
    if count > 9223372036854775807 // itemsize:
        raise Error("ValueError: Array byte size exceeds native indexing")
    var nbytes = count * itemsize
    var strides = List[Int](length=len(shape), fill=0)
    stride = itemsize
    for i in range(len(shape)):  # small-loop(publish canonical contiguous strides)
        var axis = len(shape) - i - 1 if order == "C" else i
        strides[axis] = stride
        stride *= shape[axis]
    if is_array:
        var old_strides = captured["strides"]
        if (len(old_strides) != len(shape) or Int(py=captured["ndim"]) != len(shape)
                or Int(py=captured["size"]) != count or Int(py=captured["nbytes"]) != nbytes
                or Int(py=captured["itemsize"]) != itemsize):
            raise Error("ValueError: Array storage metadata is inconsistent")
        for i in range(len(shape)):  # small-loop(validate source layout against its captured order)
            if py_type(old_strides[i]) is not b.int or (count != 0 and shape[i] > 1 and Int(py=old_strides[i]) != strides[i]):
                raise Error("ValueError: Array strides differ from its declared contiguous order")
    var available = layout.count * layout.itemsize
    var offset = address - layout.address if nbytes != 0 else 0
    if offset < 0 or offset > available or nbytes > available - offset:
        raise Error("ValueError: Array view lies outside its retained native storage")
    var raw_address = address if nbytes != 0 else Int(py=context["empty_address"])
    var raw = cpy.lib.get_function[PyObjectPtr]("PyMemoryView_FromMemory")(raw_address, nbytes, Int32(0x100 if readonly else 0x200))
    if not raw:
        raise typed_python_error()
    var bytes_view = PythonObject(from_owned=raw)
    var view = py_call(bytes_view.cast, context["codes"][PythonObject(dtype)])
    var result = native_storage_allocate(cls)
    var values = Python.dict()
    py_setitem(values, "_store", store)
    py_setitem(values, "_base", base)
    py_setitem(values, "_pin", pin)
    py_setitem(values, "_mv", view)
    py_setitem(values, "_addr", PythonObject(address))
    py_setitem(values, "_readonly", PythonObject(readonly))
    py_setitem(values, "shape", _tuple(shape))
    py_setitem(values, "dtype", PythonObject(dtype))
    py_setitem(values, "order", PythonObject(order))
    py_setitem(values, "strides", _tuple(strides))
    py_setitem(values, "ndim", PythonObject(len(shape)))
    py_setitem(values, "size", PythonObject(count))
    py_setitem(values, "nbytes", PythonObject(nbytes))
    py_setitem(values, "itemsize", PythonObject(itemsize))
    var names = py_call(b.list, values.keys())
    for i in range(len(names)):  # small-loop(publish native member slots; no Python constructors or properties)
        _ = py_call(slots[names[i]].__set__, result, values[names[i]])
    _ = borrowed^
    return result
