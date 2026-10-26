# SPDX-License-Identifier: Apache-2.0
# The public operations of `mojolearn.Array` (python/mojolearn/_array.py) as
# ONE native entry, `array_native_call(op, array, context, *args)`.
#
# Each operation is the former Python method, statement for statement, with
# the same CPython C calls it made (memoryview casts and tolist, array.array
# construction, struct pack/unpack, slice.indices, int(), getattr) and the
# same core helpers (cast_elements, reduce_stat, strided_copy_bytes, the
# functions `_buffer._native` resolves, captured once at connection). Results
# are byte-identical to the Python definitions, which stay in the package only
# as test oracles (`_array._ArrayReference`, test_native_array_operations.py).
#
# A caller's own object can still run its own code where the Python method
# did: an `__index__`, `__int__` or dtype attribute of an index, a shape or a
# dtype argument. The library itself executes no Python here.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_loaded_module import native_loaded_module
from bindings.native_buffer_owner import native_buffer_view, native_storage_allocate
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_setattr
from bindings.py_typed_call import py_call_typed
from bindings.optimizer_public_words import stash_python_error, optimizer_python_error


# ------------------------------------------------------------------ calls


def _call(callable: PythonObject, *args: PythonObject) raises -> PythonObject:
    """`callable(*args)`; an exception it raises reaches the caller unchanged
    (type, message and traceback), as the Python method let it through."""
    ref cpy = Python().cpython()
    var envelope = cpy.PyTuple_New(len(args))
    if not envelope:
        var failure = stash_python_error()
        raise failure^
    for i in range(len(args)):  # small-loop(args: positional arguments of one call): bounded by the call site
        _ = cpy.PyTuple_SetItem(envelope, i, cpy.Py_NewRef(args[i]._obj_ptr))
    var result = cpy.PyObject_Call(callable._obj_ptr, envelope, PyObjectPtr())
    cpy.Py_DecRef(envelope)
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


def _call_tuple(callable: PythonObject, arguments: PythonObject) raises -> PythonObject:
    """`callable(*arguments)` for an argument tuple built here."""
    ref cpy = Python().cpython()
    var result = cpy.PyObject_Call(callable._obj_ptr, arguments._obj_ptr, PyObjectPtr())
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


def _truth(value: PythonObject) raises -> Bool:
    return Bool(py=_call(native_loaded_module("builtins").bool, value))


def _text(value: PythonObject) raises -> String:
    """`str(value)`, as an f-string field formats it."""
    return String(_call(native_loaded_module("builtins").str, value))


def _repr(value: PythonObject) raises -> String:
    return String(_call(native_loaded_module("builtins").repr, value))


def _rebuild(key: String) -> Error:
    return Error("RuntimeError: mojolearn: this binary has no `" + key
                 + "` helper (an older build); rebuild it")


def _helper(key: String, size: Int, context: PythonObject) raises -> PythonObject:
    """The connected core helper `key` for a non-empty block, else None."""
    if size <= 0:
        return PythonObject(None)
    return py_get(context["helpers"], PythonObject(key))


# --------------------------------------------------------------- metadata


def _ints(values: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    for i in range(len(values)):  # driver-loop: over array rank metadata
        out.append(Int(py=values[i]))
    return out^


def _tuple(values: List[Int]) raises -> PythonObject:
    var items = Python.list()
    for i in range(len(values)):  # driver-loop: over array rank metadata
        _ = py_call(items.append, PythonObject(values[i]))
    return py_call(native_loaded_module("builtins").tuple, items)


def _prod(values: List[Int]) -> Int:
    var n = 1
    for i in range(len(values)):  # driver-loop: over array rank metadata
        n *= values[i]
    return n


def _c_strides(shape: List[Int], itemsize: Int) -> List[Int]:
    var out = List[Int](length=len(shape), fill=0)
    var acc = itemsize
    for k in range(len(shape)):  # driver-loop: over array rank metadata
        var axis = len(shape) - 1 - k
        out[axis] = acc
        acc *= shape[axis]
    return out^


def _f_strides(shape: List[Int], itemsize: Int) -> List[Int]:
    var out = List[Int](length=len(shape), fill=0)
    var acc = itemsize
    for axis in range(len(shape)):  # driver-loop: over array rank metadata
        out[axis] = acc
        acc *= shape[axis]
    return out^


def _both_orders(shape: List[Int]) -> Bool:
    var big = 0
    for i in range(len(shape)):  # driver-loop: over array rank metadata
        if shape[i] != 1:
            big += 1
    return big <= 1


def _code(dtype: String, context: PythonObject) raises -> PythonObject:
    return py_get(context["codes"], PythonObject(dtype))


def _is_int(dtype: String) -> Bool:
    return (dtype == "<i4" or dtype == "<i8" or dtype == "<u4" or dtype == "<u1"
            or dtype == "<u2" or dtype == "<i1")


def _is_float(dtype: String) -> Bool:
    return dtype == "<f4" or dtype == "<f8" or dtype == "<f2"


def _set_meta(array: PythonObject, shape: PythonObject, dtype: String, order: String,
              view: PythonObject, context: PythonObject) raises:
    """`Array._set_meta`: the shape, dtype and layout slots, then the size check."""
    var dims = _ints(shape)
    var itemsize = Int(py=py_get(context["itemsizes"], PythonObject(dtype)))
    var size = _prod(dims)
    py_setattr(array, "shape", shape)
    py_setattr(array, "dtype", PythonObject(dtype))
    py_setattr(array, "ndim", PythonObject(len(dims)))
    py_setattr(array, "size", PythonObject(size))
    py_setattr(array, "itemsize", PythonObject(itemsize))
    py_setattr(array, "nbytes", PythonObject(size * itemsize))
    py_setattr(array, "order", PythonObject(order))
    if order == "C":
        py_setattr(array, "strides", _tuple(_c_strides(dims, itemsize)))
    else:
        py_setattr(array, "strides", _tuple(_f_strides(dims, itemsize)))
    if len(view) != size:
        raise Error("ValueError: mojolearn: buffer holds " + String(len(view)) + " elements, shape "
                    + _text(shape) + " needs " + String(size))


def _init_owned(array: PythonObject, store: PythonObject, shape: PythonObject, dtype: String,
                order: String, context: PythonObject) raises -> PythonObject:
    """`Array._init_owned`: `array` owns `store`, an `array.array`."""
    var code = _code(dtype, context)
    var view = native_buffer_view(store)
    if String(view.format) != String(code):
        view = _call(_call(view.cast, PythonObject("B")).cast, code)
    py_setattr(array, "_store", store)
    py_setattr(array, "_base", PythonObject(None))
    py_setattr(array, "_pin", PythonObject(None))
    py_setattr(array, "_mv", view)
    py_setattr(array, "_addr", _call(store.buffer_info)[0])
    py_setattr(array, "_readonly", PythonObject(False))
    _set_meta(array, shape, dtype, order, view, context)
    return array


def _owned(store: PythonObject, shape: PythonObject, dtype: String, order: String,
           context: PythonObject) raises -> PythonObject:
    return _init_owned(native_storage_allocate(context["array_class"]), store, shape, dtype, order, context)


def _owned_as(cls: PythonObject, store: PythonObject, shape: PythonObject, dtype: String, order: String,
              context: PythonObject) raises -> PythonObject:
    """`cls._owned(...)` for the class a classmethod constructor was called on."""
    return _init_owned(native_storage_allocate(cls), store, shape, dtype, order, context)


def _view_of(other: PythonObject, shape: PythonObject, order: String, context: PythonObject) raises -> PythonObject:
    """`Array._view_of`: another shape over the same memory, `other` kept alive."""
    var view = native_storage_allocate(context["array_class"])
    py_setattr(view, "_store", other._store)
    py_setattr(view, "_base", other)
    py_setattr(view, "_pin", other._pin)
    py_setattr(view, "_mv", other._mv)
    py_setattr(view, "_addr", other._addr)
    py_setattr(view, "_readonly", other._readonly)
    _set_meta(view, shape, String(other.dtype), order, other._mv, context)
    return view


def _new_store(code: PythonObject, size: Int, context: PythonObject) raises -> PythonObject:
    """`_new_store`: a zero-filled `array.array` of `size` elements."""
    if size == 0:
        return _call(context["array_type"], code)
    var seed = _call(context["array_type"], code, Python.list(PythonObject(0)))
    return _call(seed.__mul__, PythonObject(size))


def _values(view: PythonObject, dtype: String, context: PythonObject) raises -> PythonObject:
    """`_to_py_values`: Python scalars of a flat store memoryview."""
    var b = native_loaded_module("builtins")
    if dtype == "<f2":
        var n = len(view)
        if n == 0:
            return Python.list()
        return _call(b.list, _call(context["struct_unpack"], PythonObject("<" + String(n) + "e"), _call(view.tobytes)))
    return _call(view.tolist)


def _strided_copy(src_addr: PythonObject, dst_addr: PythonObject, itemsize: Int, shape: List[Int],
                  src_strides: List[Int], dst_strides: List[Int], src_base: Int, dst_base: Int,
                  context: PythonObject) raises:
    """`_strided_copy`: one call of the core helper `strided_copy_bytes`."""
    var words = Python.list()
    for i in range(len(shape)):  # driver-loop: over array rank metadata
        _ = py_call(words.append, PythonObject(shape[i]))
    for i in range(len(src_strides)):  # driver-loop: over array rank metadata
        _ = py_call(words.append, PythonObject(src_strides[i]))
    for i in range(len(dst_strides)):  # driver-loop: over array rank metadata
        _ = py_call(words.append, PythonObject(dst_strides[i]))
    _ = py_call(words.append, PythonObject(src_base))
    _ = py_call(words.append, PythonObject(dst_base))
    var dims = _call(context["array_type"], PythonObject("q"), words)
    var copy = py_get(context["helpers"], PythonObject("strided_copy_bytes"))
    if copy is None:
        raise _rebuild("strided_copy_bytes")
    _ = _call(copy, src_addr, dst_addr, _call(dims.buffer_info)[0], PythonObject(len(shape)), PythonObject(itemsize))
    _ = dims  # the dimension words stay alive through the native copy


def _reorder(a: PythonObject, src_order: String, out: PythonObject, context: PythonObject) raises:
    """`_reorder`: the elements of `a` into `out` in the other layout."""
    var shape = _ints(a.shape)
    if len(shape) <= 1 or Int(py=a.size) == 0:
        var b = native_loaded_module("builtins")
        _ = _call(native_buffer_view(out).__setitem__, _call(b.slice, PythonObject(None)), a._mv)
        return
    var cs = _c_strides(shape, 1)
    var fs = _f_strides(shape, 1)
    if src_order == "C":
        _strided_copy(a._addr, _call(out.buffer_info)[0], Int(py=a.itemsize), shape, cs, fs, 0, 0, context)
    else:
        _strided_copy(a._addr, _call(out.buffer_info)[0], Int(py=a.itemsize), shape, fs, cs, 0, 0, context)


def _as_c(a: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Array._as_c`: self if C-order, a relabeled view when both layouts
    coincide, else a C-order copy."""
    if String(a.order) == "C":
        return a
    if _both_orders(_ints(a.shape)):
        return _view_of(a, a.shape, "C", context)
    var dtype = String(a.dtype)
    var out = _new_store(_code(dtype, context), Int(py=a.size), context)
    _reorder(a, "F", out, context)
    return _owned(out, a.shape, dtype, "C", context)


def _admit(a: PythonObject, context: PythonObject) raises:
    var b = native_loaded_module("builtins")
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, py_type(a))
    for i in range(len(ancestry)):  # driver-loop: the connected Array class in the instance's ancestry
        if ancestry[i] is context["array_class"]:
            return
    raise Error("TypeError: native Array operations require a mojolearn Array")


# ------------------------------------------------------------ operations


def _length(a: PythonObject) raises -> PythonObject:
    if Int(py=a.ndim) == 0:
        raise Error("TypeError: len() of a 0-d Array")
    return a.shape[0]


def _representation(a: PythonObject) raises -> PythonObject:
    return PythonObject("Array(shape=" + _text(a.shape) + ", dtype=" + _repr(a.dtype) + ")")


def _flags(a: PythonObject) raises -> PythonObject:
    var both = _both_orders(_ints(a.shape))
    var order = String(a.order)
    var out = Python.dict()
    py_setitem(out, PythonObject("C_CONTIGUOUS"), PythonObject(order == "C" or both))
    py_setitem(out, PythonObject("F_CONTIGUOUS"), PythonObject(order == "F" or both))
    py_setitem(out, PythonObject("WRITEABLE"), PythonObject(not _truth(a._readonly)))
    return out


def _interface(a: PythonObject, context: PythonObject) raises -> PythonObject:
    """Version 3; `strides` None means C-contiguous."""
    var address = a._addr
    if not _truth(a.size) and not _truth(address):
        address = context["empty_address"]
    var out = Python.dict()
    py_setitem(out, PythonObject("shape"), a.shape)
    py_setitem(out, PythonObject("typestr"), a.dtype)
    py_setitem(out, PythonObject("data"), Python.tuple(address, PythonObject(_truth(a._readonly))))
    if String(a.order) == "C":
        py_setitem(out, PythonObject("strides"), PythonObject(None))
    else:
        py_setitem(out, PythonObject("strides"), a.strides)
    py_setitem(out, PythonObject("version"), PythonObject(3))
    return out


def _bool(a: PythonObject, context: PythonObject) raises -> PythonObject:
    if Int(py=a.size) != 1:
        raise Error("ValueError: mojolearn: the truth value of an Array with more than one element is ambiguous")
    return PythonObject(_truth(_values(a._mv, String(a.dtype), context)[0]))


def _copy(a: PythonObject, context: PythonObject) raises -> PythonObject:
    var dtype = String(a.dtype)
    var store = _call(context["array_type"], _code(dtype, context))
    _ = _call(store.frombytes, _call(a._mv.cast, PythonObject("B")))  # one memcpy
    return _owned(store, a.shape, dtype, String(a.order), context)


def _empty_nest(shape: List[Int], depth: Int) raises -> PythonObject:
    """The nested lists of an empty block, up to its first zero dimension."""
    var out = Python.list()
    if depth >= len(shape) or shape[depth] == 0:
        return out
    for _ in range(shape[depth]):  # driver-loop: list structure of an empty block, no elements
        _ = py_call(out.append, _empty_nest(shape, depth + 1))
    return out


def _tolist(a: PythonObject, context: PythonObject) raises -> PythonObject:
    var c = _as_c(a, context)
    var dtype = String(c.dtype)
    if Int(py=a.ndim) == 0:
        return _values(c._mv, dtype, context)[0]
    if Int(py=c.size) == 0:
        return _empty_nest(_ints(c.shape), 0)
    if dtype == "<f2":
        var flat = _call(context["array_type"], PythonObject("d"), _values(c._mv, dtype, context))
        return _call(_call(_call(native_buffer_view(flat).cast, PythonObject("B")).cast, PythonObject("d"), c.shape).tolist)
    return _call(_call(_call(c._mv.cast, PythonObject("B")).cast, _code(dtype, context), c.shape).tolist)


def _row(c: PythonObject, shape: PythonObject, step: Int, i: Int, context: PythonObject) raises -> PythonObject:
    """`Array._row_view`: row `i` of a C-order Array, zero-copy."""
    var b = native_loaded_module("builtins")
    var view = native_storage_allocate(context["array_class"])
    var offset = i * step
    py_setattr(view, "_store", c._store)
    py_setattr(view, "_base", c)
    py_setattr(view, "_pin", c._pin)
    var part = _call(c._mv.__getitem__, _call(b.slice, PythonObject(offset), PythonObject(offset + step)))
    py_setattr(view, "_mv", part)
    py_setattr(view, "_addr", PythonObject(Int(py=c._addr) + offset * Int(py=c.itemsize)))
    py_setattr(view, "_readonly", c._readonly)
    _set_meta(view, shape, String(c.dtype), "C", part, context)
    return view


def _iterate(a: PythonObject, context: PythonObject) raises -> PythonObject:
    """1-D: an iterator over the Python values. Rank 2+: a lazy iterator of
    zero-copy row views, `map(partial(entry, "row", c, context, shape, step),
    range(rows))`, made of C objects only."""
    var b = native_loaded_module("builtins")
    var ndim = Int(py=a.ndim)
    if ndim == 0:
        raise Error("TypeError: iteration over a 0-d Array")
    if ndim == 1:
        return _call(b.iter, _values(a._mv, String(a.dtype), context))
    var c = _as_c(a, context)
    var dims = _ints(c.shape)
    var rest = List[Int]()
    for i in range(1, len(dims)):  # driver-loop: over array rank metadata
        rest.append(dims[i])
    var row = _call(context["partial"], context["entry"], PythonObject("row"), c, context,
                    _tuple(rest), PythonObject(_prod(rest)))
    return _call(b.map, row, _call(b.range, PythonObject(dims[0])))


def _reshape(a: PythonObject, requested: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var shape_arg = requested
    if _truth(_call(b.isinstance, requested, b.int)):
        shape_arg = Python.tuple(requested)
    var dims = _ints(_call(b.list, _call(b.map, b.int, shape_arg)))  # int(s) per entry, as the method did
    var size = Int(py=a.size)
    var unknown = 0
    var at = -1
    for i in range(len(dims)):  # driver-loop: over the requested rank
        if dims[i] == -1:
            unknown += 1
            if at < 0:
                at = i
    if unknown > 1:
        raise Error("ValueError: mojolearn: reshape allows one -1")
    if at >= 0:
        var known = 1
        for i in range(len(dims)):  # driver-loop: over the requested rank
            if dims[i] != -1:
                known *= dims[i]
        if known == 0 or size % known != 0:
            raise Error("ValueError: mojolearn: cannot reshape size " + String(size) + " into " + _text(_tuple(dims)))
        dims[at] = size // known
    var shape = _tuple(dims)
    if _prod(dims) != size:
        raise Error("ValueError: mojolearn: cannot reshape size " + String(size) + " into " + _text(shape))
    return _view_of(_as_c(a, context), shape, "C", context)


# ---------------------------------------------------------------- indexing


def _same_dtype_store(raw: PythonObject, dtype: String, context: PythonObject) raises -> PythonObject:
    """`_buffer._same_dtype_store`: a byte copy, a float32 one quieting a
    signaling NaN through `cast_elements` (None without that helper)."""
    var store = _call(context["array_type"], _code(dtype, context))
    _ = _call(store.frombytes, raw)
    if dtype != "<f4" or len(store) == 0:
        return store
    var cast = py_get(context["helpers"], PythonObject("cast_elements"))
    if cast is None:
        return PythonObject(None)
    var at = _call(store.buffer_info)[0]
    _ = _call(cast, at, PythonObject(0), at, PythonObject(0), PythonObject(len(store)))
    return store


def _getitem(a: PythonObject, key: PythonObject, context: PythonObject) raises -> PythonObject:
    """Int / slice / tuple of them. Full integer indexing returns a Python
    scalar; anything else a C-order copy."""
    var b = native_loaded_module("builtins")
    var operator = context["operator"]
    var c = _as_c(a, context)
    var shape = _ints(c.shape)
    var ndim = len(shape)
    var keys = Python.tuple(key)
    if _truth(_call(b.isinstance, key, b.tuple)):
        keys = _call(b.tuple, key)
    if len(keys) > ndim:
        raise Error("IndexError: mojolearn: too many indices (" + String(len(keys)) + ") for shape " + _text(c.shape))
    # One entry per axis: (is_slice, start or index, step, count).
    var sliced = List[Bool]()
    var first = List[Int]()
    var steps = List[Int]()
    var counts = List[Int]()
    for axis in range(len(keys)):  # driver-loop: one index entry per array axis
        var k = keys[axis]
        var n = shape[axis]
        if py_type(k) is b.slice:
            var bounds = _call(k.indices, PythonObject(n))
            var start = Int(py=bounds[0])
            var stop = Int(py=bounds[1])
            var step = Int(py=bounds[2])
            var count = 0
            if step > 0 and stop > start:
                count = (stop - start + step - 1) // step
            elif step < 0 and start > stop:
                count = (start - stop - step - 1) // (-step)
            sliced.append(True)
            first.append(start)
            steps.append(step)
            counts.append(count)
        elif _truth(_call(b.isinstance, k, b.int)) or _truth(_call(b.hasattr, k, PythonObject("__index__"))):
            var index = _call(k.__index__)
            if (_truth(_call(operator.lt, index, PythonObject(-n)))
                    or _truth(_call(operator.ge, index, PythonObject(n)))):
                raise Error("IndexError: mojolearn: index " + _text(index) + " out of range for axis "
                            + String(axis) + " of size " + String(n))
            var value = Int(py=index)
            if n != 0:
                value = value % n
                if value < 0:
                    value += n
            sliced.append(False)
            first.append(value)
            steps.append(0)
            counts.append(0)
        else:
            raise Error("TypeError: mojolearn: Array indices are ints, slices or tuples of them, not "
                        + String(py_type(k).__name__))
    for axis in range(len(keys), ndim):  # driver-loop: one entry per remaining array axis
        sliced.append(True)
        first.append(0)
        steps.append(1)
        counts.append(shape[axis])
    var cs = _c_strides(shape, 1)
    var base = 0
    var out_shape = List[Int]()
    var out_strides = List[Int]()
    for axis in range(ndim):  # driver-loop: one entry per array axis
        base += first[axis] * cs[axis]
        if sliced[axis]:
            out_shape.append(counts[axis])
            out_strides.append(steps[axis] * cs[axis])
    var dtype = String(c.dtype)
    if len(out_shape) == 0:
        var one = _call(c._mv.__getitem__, _call(b.slice, PythonObject(base), PythonObject(base + 1)))
        return _values(one, dtype, context)[0]
    # `_contiguous_block`: leading integer indices, at most one step-1
    # slice, then only whole axes is ONE run; `_block_store` copies a run of
    # at least 256 elements in one move.
    var store = PythonObject(None)
    var axis = 0
    var offset = 0
    while axis < ndim and not sliced[axis]:  # driver-loop: leading integer indices
        offset += first[axis] * cs[axis]
        axis += 1
    var contiguous = axis < ndim and steps[axis] == 1
    if contiguous:
        for later in range(axis + 1, ndim):  # driver-loop: one check per trailing array axis
            if not sliced[later] or first[later] != 0 or steps[later] != 1 or counts[later] != shape[later]:
                contiguous = False
    if contiguous:
        var run_start = offset + first[axis] * cs[axis]
        var run_length = counts[axis] * cs[axis]
        if run_length >= Int(py=context["native_min"]):
            var raw = _call(_call(c._mv.__getitem__, _call(b.slice, PythonObject(run_start),
                                                          PythonObject(run_start + run_length))).cast, PythonObject("B"))
            store = _same_dtype_store(raw, dtype, context)
    if store is None:
        var total = _prod(out_shape)
        store = _new_store(_code(dtype, context), total, context)
        if total > 0:
            _strided_copy(c._addr, _call(store.buffer_info)[0], Int(py=c.itemsize), out_shape,
                          out_strides, _c_strides(out_shape, 1), base, 0, context)
            if dtype == "<f4":
                # A strided float32 selection is quieted like the block copy,
                # so a copy's bits do not depend on the selection's layout.
                var cast = py_get(context["helpers"], PythonObject("cast_elements"))
                if cast is not None:
                    var at = _call(store.buffer_info)[0]
                    _ = _call(cast, at, PythonObject(0), at, PythonObject(0), PythonObject(total))
    return _owned(store, _tuple(out_shape), dtype, "C", context)


# ------------------------------------------------------------ conversion


def _normalize(dtype: PythonObject, context: PythonObject) raises -> String:
    """`normalize_dtype`: a supported typestr, or the method's ValueError."""
    var b = native_loaded_module("builtins")
    var aliases = context["aliases"]
    var codes = context["codes"]
    if _truth(_call(b.isinstance, dtype, b.str)):
        var ts = _call(aliases.get, dtype, dtype)
        if py_contains(codes, ts):
            return String(ts)
        raise Error("ValueError: mojolearn: unsupported dtype " + _repr(dtype) + "; supported: "
                    + String(_call(PythonObject(", ").join, context["supported"])))
    if dtype is b.float:
        return "<f8"
    if dtype is b.int:
        return "<i8"
    if dtype is b.bool:
        return "<u1"
    var s = _call(b.getattr, dtype, PythonObject("str"), PythonObject(None))
    if _truth(_call(b.isinstance, s, b.str)):
        return _normalize(s, context)
    var n = _call(b.getattr, dtype, PythonObject("__name__"), PythonObject(None))
    if _truth(_call(b.isinstance, n, b.str)) and py_contains(aliases, n):
        return String(py_get(aliases, n))
    raise Error("ValueError: mojolearn: unsupported dtype " + _repr(dtype))


def _from_flat(flat: PythonObject, shape: PythonObject, dtype: String, context: PythonObject) raises -> PythonObject:
    return _from_flat_as(context["array_class"], flat, shape, dtype, context)


def _from_flat_as(cls: PythonObject, flat: PythonObject, shape: PythonObject, dtype: String,
                  context: PythonObject) raises -> PythonObject:
    """`Array._from_flat`: a C-order Array of Python values."""
    var b = native_loaded_module("builtins")
    var code = _code(dtype, context)
    var store = PythonObject(None)
    if dtype == "<f2":
        store = _call(context["array_type"], PythonObject("H"))
        if len(flat) > 0:
            var arguments = Python.list(PythonObject("<" + String(len(flat)) + "e"))
            _ = _call(arguments.extend, flat)
            _ = _call(store.frombytes, _call_tuple(context["struct_pack"], _call(b.tuple, arguments)))
    else:
        try:
            store = py_call_typed(context["array_type"], code, flat)
        except error:
            var message = String(error)
            if message.startswith("TypeError: ") or message.startswith("OverflowError: "):
                var detail = message[byte=message.find(": ") + 2:]
                raise Error("TypeError: mojolearn: cannot build a " + dtype + " Array from these values: "
                            + String(detail))
            raise error^
    return _owned_as(cls, store, shape, dtype, "C", context)


def _flatten_fast(nested: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_flatten_fast`: `(shape, flat)` for a flat list of Python scalars or
    a list of equal-length rows of them, by C builtins; None otherwise."""
    var b = native_loaded_module("builtins")
    var kind = py_type(nested)
    if (kind is not b.list and kind is not b.tuple) or len(nested) == 0:
        return PythonObject(None)
    var kinds = _call(b.set, _call(b.map, b.type, nested))
    if _truth(_call(kinds.issubset, context["scalar_type_set"])):
        return Python.tuple(Python.tuple(PythonObject(len(nested))), _call(b.list, nested))
    if not _truth(_call(kinds.issubset, context["row_type_set"])):
        return PythonObject(None)
    var widths = _call(b.set, _call(b.map, b.len, nested))
    if len(widths) != 1:
        return PythonObject(None)
    var flat = _call(b.list, _call(context["chain_from_iterable"], nested))
    if not _truth(_call(_call(b.set, _call(b.map, b.type, flat)).issubset, context["scalar_type_set"])):
        return PythonObject(None)
    return Python.tuple(Python.tuple(PythonObject(len(nested)), _call(widths.pop)), flat)


def _from_list(cls: PythonObject, nested: PythonObject, requested: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Array.from_list`: a new C-order Array from a (nested) list of numbers."""
    var b = native_loaded_module("builtins")
    var dtype = _normalize(requested, context)
    var shape = PythonObject(None)
    var flat = PythonObject(None)
    if _truth(_call(b.isinstance, nested, context["array_class"])):
        var c = _as_c(nested, context)
        shape = nested.shape
        flat = _values(c._mv, String(c.dtype), context)
    else:
        var fast = _flatten_fast(nested, context)
        if fast is not None:
            shape = fast[0]
            flat = fast[1]
        else:
            # The general walk (a ragged or mixed nested list, Arrays inside
            # it) is the user-object reader's native "nested" read.
            var got = _call(context["user_objects"], PythonObject("nested"), nested, context["array_class"])
            if py_type(got) is b.tuple and len(got) == 3 and _truth(_call(context["operator"].eq, got[0], PythonObject("__refuse__"))):
                raise Error(String(got[1]) + ": " + String(got[2]))
            shape = _call(b.tuple, got[0])
            flat = got[1]
    return _from_flat_as(cls, flat, shape, dtype, context)


def _astype_reference(a: PythonObject, dtype: String, context: PythonObject) raises -> PythonObject:
    """The conversion `cast_elements` declined: the same C calls the Python
    definition makes, so the same result or the same refusal words."""
    var b = native_loaded_module("builtins")
    var src = String(a.dtype)
    var code = _code(dtype, context)
    var out = PythonObject(None)
    if src == "<f2" or dtype == "<f2":
        var values = _values(a._mv, src, context)
        if _is_int(dtype):
            values = _call(b.list, _call(b.map, b.int, values))
        out = _from_flat(values, a.shape, dtype, context)
    elif _is_float(dtype):
        out = _owned(_call(context["array_type"], code, a._mv), a.shape, dtype, "C", context)
    elif _is_int(src):
        var store = PythonObject(None)
        try:
            store = py_call_typed(context["array_type"], code, a._mv)
        except error:
            var message = String(error)
            if message.startswith("OverflowError: "):
                raise Error("OverflowError: mojolearn: value does not fit " + dtype + ": "
                            + String(message[byte=15:]))
            raise error^
        out = _owned(store, a.shape, dtype, "C", context)
    else:
        out = _from_flat(_call(b.list, _call(b.map, b.int, a._mv)), a.shape, dtype, context)
    var order = String(a.order)
    py_setattr(out, "order", PythonObject(order))
    _set_meta(out, a.shape, dtype, order, out._mv, context)
    return out


def _astype(a: PythonObject, requested: PythonObject, context: PythonObject) raises -> PythonObject:
    """A converted copy (always a copy, even for the same dtype)."""
    var dtype = _normalize(requested, context)
    var src = String(a.dtype)
    if dtype == src:
        return _copy(a, context)
    var code = _code(dtype, context)
    var size = Int(py=a.size)
    var order = String(a.order)
    if size == 0:
        return _owned(_new_store(code, 0, context), a.shape, dtype, order, context)
    var sc = py_get(context["native_codes"], PythonObject(src))
    var dc = py_get(context["native_codes"], PythonObject(dtype))
    var cast = _helper("cast_elements", size, context)
    if sc is not None and dc is not None and cast is not None:
        var store = _new_store(code, size, context)
        var status = _call(cast, a._addr, sc, _call(store.buffer_info)[0], dc, PythonObject(size))
        if Int(py=status) == 0:
            return _owned(store, a.shape, dtype, order, context)
    if cast is None:
        raise _rebuild("cast_elements")
    # The helper met an element the conversion refuses: raise its words.
    return _astype_reference(a, dtype, context)


# ------------------------------------------------------------ reductions


def _reduce(a: PythonObject, what: Int, name: String, context: PythonObject) raises -> PythonObject:
    """min (0), max (1), float sum (2) and argmax (3) in storage order."""
    var size = Int(py=a.size)
    var code = py_get(context["native_codes"], a.dtype)
    var stat = _helper("reduce_stat", size, context)
    if code is not None and stat is not None:
        return _call(stat, a._addr, code, PythonObject(size), PythonObject(what))
    if size == 0:
        raise Error("ValueError: mojolearn: " + name + " of an empty Array")
    raise _rebuild("reduce_stat")


def _sum(a: PythonObject, context: PythonObject) raises -> PythonObject:
    var dtype = String(a.dtype)
    var size = Int(py=a.size)
    if size == 0:
        if _is_int(dtype):
            return PythonObject(0)
        return PythonObject(0.0)
    var stat = _helper("reduce_stat", size, context)
    var code = py_get(context["native_codes"], a.dtype)
    if stat is None or code is None:
        raise _rebuild("reduce_stat")
    if _is_float(dtype):
        return _call(stat, a._addr, code, PythonObject(size), PythonObject(2))
    # The exact integer sum: (hi << 64) + (mid << 32) + low of a 128-bit total.
    var parts = _call(stat, a._addr, code, PythonObject(size), PythonObject(5))
    var operator = context["operator"]
    var b = native_loaded_module("builtins")
    var hi = _call(b.int, parts[0])
    var mid = _call(b.int, parts[1])
    var low = _call(b.int, parts[2])
    var total = _call(operator.add, _call(operator.lshift, hi, PythonObject(64)), _call(operator.lshift, mid, PythonObject(32)))
    return _call(operator.add, total, low)


# -------------------------------------------------------------- equality


def _exact_in(given: PythonObject, dtype: String, context: PythonObject) raises -> PythonObject:
    """`_exact_in`: `given` (a Python int, float or bool) as an element of
    `dtype` that compares equal to it exactly under `==`, or None."""
    var b = native_loaded_module("builtins")
    var operator = context["operator"]
    var value = given
    if _truth(_call(b.isinstance, value, b.bool)):
        value = _call(b.int, value)
    var bounds = py_get(context["int_bounds"], PythonObject(dtype))
    if bounds is not None:
        if _truth(_call(b.isinstance, value, b.float)):
            if (_truth(_call(operator.ne, value, value))
                    or _truth(_call(operator.eq, value, context["infinity"]))
                    or _truth(_call(operator.eq, value, context["minus_infinity"]))
                    or _truth(_call(operator.ne, value, _call(b.int, value)))):
                return PythonObject(None)
            value = _call(b.int, value)
        if _truth(_call(operator.le, bounds[0], value)) and _truth(_call(operator.le, value, bounds[1])):
            return value
        return PythonObject(None)
    if _truth(_call(b.isinstance, value, b.int)):
        var widened = PythonObject(None)
        try:
            widened = py_call_typed(b.float, value)
        except error:
            if String(error).startswith("OverflowError: "):
                return PythonObject(None)
            raise error^
        if _truth(_call(operator.ne, widened, value)):
            return PythonObject(None)  # an int no double holds: no float element equals it
        value = widened
    if _truth(_call(operator.ne, value, value)):
        return PythonObject(None)  # NaN equals nothing
    if dtype == "<f4" or dtype == "<f2":
        var fmt = PythonObject("<f") if dtype == "<f4" else PythonObject("<e")
        var packed = PythonObject(None)
        try:
            packed = py_call_typed(context["struct_pack"], fmt, value)
        except error:
            if String(error).startswith("OverflowError: "):
                return PythonObject(None)
            raise error^
        var g = _call(context["struct_unpack"], fmt, packed)[0]
        if _truth(_call(operator.ne, g, value)):
            return PythonObject(None)
        return g
    return value


def _zero_flags(size: Int, context: PythonObject) raises -> PythonObject:
    """`array.array("B", bytes(size))`."""
    var b = native_loaded_module("builtins")
    return _call(context["array_type"], PythonObject("B"), _call(b.bytes, PythonObject(size)))


def _equal(a: PythonObject, other: PythonObject, context: PythonObject) raises -> PythonObject:
    """Elementwise equality against an Array of the same shape or a scalar,
    as a `'<u1'` Array of 0/1; NotImplemented for any other operand."""
    var b = native_loaded_module("builtins")
    var operator = context["operator"]
    var dtype = String(a.dtype)
    var size = Int(py=a.size)
    var is_array = _truth(_call(b.isinstance, other, context["array_class"]))
    var same_shape = is_array and _truth(_call(operator.eq, other.shape, a.shape))
    var scalar = _truth(_call(b.isinstance, other, context["scalar_types"]))
    var equal = _helper("equal_elements", size, context)
    var code = py_get(context["native_codes"], a.dtype)
    if same_shape:
        if code is not None and String(other.dtype) == dtype and equal is not None:
            var x = _as_c(a, context)
            var y = _as_c(other, context)
            var store = _zero_flags(size, context)
            _ = _call(equal, x._addr, y._addr, code, PythonObject(size), _call(store.buffer_info)[0])
            return _owned(store, a.shape, "<u1", "C", context)
    elif scalar:
        if size == 0:
            return _owned(_call(context["array_type"], PythonObject("B")), a.shape, "<u1", "C", context)
        if code is not None and equal is not None:
            var v = _exact_in(other, dtype, context)
            var store = _zero_flags(size, context)
            if v is not None:
                var x = _as_c(a, context)
                var fill = _call(_call(context["array_type"], _code(dtype, context), Python.list(v)).__mul__,
                                 PythonObject(size))
                _ = _call(equal, x._addr, _call(fill.buffer_info)[0], code, PythonObject(size),
                          _call(store.buffer_info)[0])
            return _owned(store, a.shape, "<u1", "C", context)
        raise _rebuild("equal_elements")
    if same_shape:
        if size == 0:
            return _owned(_call(context["array_type"], PythonObject("B")), a.shape, "<u1", "C", context)
        var theirs = String(other.dtype)
        if theirs == dtype:
            raise _rebuild("equal_elements")
        # Two dtypes: both widened EXACTLY to one dtype, then one native
        # comparison; int64 against a float compares with the helper's exact
        # int == float rule.
        var a_int = _is_int(dtype)
        var b_int = _is_int(theirs)
        if a_int and b_int:
            return _equal(_astype(a, PythonObject("<i8"), context), _astype(other, PythonObject("<i8"), context), context)
        if dtype != "<i8" and theirs != "<i8":
            return _equal(_astype(a, PythonObject("<f8"), context), _astype(other, PythonObject("<f8"), context), context)
        if equal is None:
            raise _rebuild("equal_elements")
        var ints = a if a_int else other
        var floats = other if a_int else a
        var ia = _as_c(ints, context)
        if String(ints.dtype) != "<i8":
            ia = _as_c(_astype(ints, PythonObject("<i8"), context), context)
        var fa = _as_c(floats, context)
        if String(floats.dtype) != "<f8":
            fa = _as_c(_astype(floats, PythonObject("<f8"), context), context)
        var store = _zero_flags(size, context)
        _ = _call(equal, ia._addr, fa._addr, context["eq_i64_f64"], PythonObject(size), _call(store.buffer_info)[0])
        return _owned(store, a.shape, "<u1", "C", context)
    if is_array:
        raise Error("ValueError: mojolearn: shapes " + _text(a.shape) + " and " + _text(other.shape) + " differ")
    return context["not_implemented"]


def _not_equal(a: PythonObject, other: PythonObject, context: PythonObject) raises -> PythonObject:
    var eq = _equal(a, other, context)
    if eq is context["not_implemented"]:
        return eq
    # `bytes.translate` flips 0 and 1 in C
    var flipped = _call(_call(eq._mv.tobytes).translate, context["not_table"])
    return _owned(_call(context["array_type"], PythonObject("B"), flipped), a.shape, "<u1", "C", context)


# --------------------------------------------------------------- interop


def _array_protocol(a: PythonObject, dtype: PythonObject, copy: PythonObject, context: PythonObject) raises -> PythonObject:
    """`numpy.asarray(a)` through `__array__`; NumPy is read from
    `sys.modules`, never imported."""
    var b = native_loaded_module("builtins")
    var np = py_get(context["sys_modules"], PythonObject("numpy"))
    if np is None:
        raise Error("TypeError: mojolearn: Array.__array__ needs NumPy loaded")
    var carrier = native_storage_allocate(context["interface_class"])
    py_setattr(carrier, "_owner", a)
    py_setattr(carrier, "__array_interface__", _interface(a, context))
    var out = _call(np.asarray, carrier)
    if dtype is not None and _truth(_call(context["operator"].ne, out.dtype, _call(np.dtype, dtype))):
        if copy is not None and py_type(copy) is b.bool and not _truth(copy):
            raise Error("ValueError: mojolearn: a dtype change needs a copy")
        return _call(out.astype, dtype)
    if _truth(copy):
        return _call(out.copy)
    return out


def _buffer(a: PythonObject, context: PythonObject) raises -> PythonObject:
    """Python 3.12+ buffer protocol: a memoryview with this Array's shape."""
    var dtype = String(a.dtype)
    var ndim = Int(py=a.ndim)
    if dtype == "<f2":
        raise Error("BufferError: mojolearn: a float16 Array has no memoryview format; use "
                    + "numpy.asarray (zero-copy) or tolist()")
    if ndim == 0:
        return a._mv
    if String(a.order) != "C" and ndim > 1:
        raise Error("BufferError: mojolearn: an F-order Array cannot be exported as a memoryview; "
                    + "use numpy.asarray (zero-copy), tobytes() or reshape() for a C-order copy")
    var code = _code(dtype, context)
    if Int(py=a.size) == 0:
        if ndim == 1:
            return _call(_call(a._mv.cast, PythonObject("B")).cast, code)
        raise Error("BufferError: mojolearn: an empty matrix has no memoryview shape; use "
                    + "numpy.asarray (zero-copy), tobytes() or reshape(-1)")
    return _call(_call(a._mv.cast, PythonObject("B")).cast, code, a.shape)


def _reduce_protocol(a: PythonObject, context: PythonObject) raises -> PythonObject:
    """`__reduce__`: values and layout, never a borrowed pointer."""
    return Python.tuple(context["restore"],
                        Python.tuple(_call(a._mv.tobytes), a.shape, a.dtype, a.order, a._readonly))


def _initialize(a: PythonObject, shape: PythonObject, requested: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Array(shape, dtype)`: a zero-filled C-order block."""
    var b = native_loaded_module("builtins")
    var dtype = _normalize(requested, context)
    var shape_arg = shape
    if _truth(_call(b.isinstance, shape, b.int)):
        shape_arg = Python.tuple(shape)
    var checked = _call(b.tuple, _call(b.map, b.int, shape_arg))
    var dims = _ints(checked)
    for i in range(len(dims)):  # driver-loop: over the requested rank
        if dims[i] < 0:
            raise Error("ValueError: mojolearn: negative dimension in shape " + _text(checked))
    var store = _new_store(_code(dtype, context), _prod(dims), context)
    _ = _init_owned(a, store, checked, dtype, "C", context)
    return PythonObject(None)


# ------------------------------------------------------------------ door


def _dispatch(op: String, a: PythonObject, context: PythonObject, args: PythonObject) raises -> PythonObject:
    if op == "init":
        return _initialize(a, args[0], args[1], context)
    if op == "from_list":
        # `a` is the class the classmethod was called on.
        var b = native_loaded_module("builtins")
        if py_type(a) is not b.type or not _truth(_call(b.issubclass, a, context["array_class"])):
            raise Error("TypeError: Array.from_list builds a mojolearn Array class")
        return _from_list(a, args[0], args[1], context)
    _admit(a, context)
    if op == "len":
        return _length(a)
    if op == "repr":
        return _representation(a)
    if op == "flags":
        return _flags(a)
    if op == "interface":
        return _interface(a, context)
    if op == "bool":
        return _bool(a, context)
    if op == "tobytes":
        return _call(a._mv.tobytes)
    if op == "copy":
        return _copy(a, context)
    if op == "tolist":
        return _tolist(a, context)
    if op == "iter":
        return _iterate(a, context)
    if op == "row":
        return _row(a, args[0], Int(py=args[1]), Int(py=args[2]), context)
    if op == "reshape":
        return _reshape(a, args[0], context)
    if op == "ravel":
        return _reshape(a, Python.tuple(PythonObject(-1)), context)
    if op == "getitem":
        return _getitem(a, args[0], context)
    if op == "astype":
        return _astype(a, args[0], context)
    if op == "min":
        return _reduce(a, 0, "min", context)
    if op == "max":
        return _reduce(a, 1, "max", context)
    if op == "sum":
        return _sum(a, context)
    if op == "argmax":
        return _reduce(_as_c(a, context), 3, "argmax", context)
    if op == "eq":
        return _equal(a, args[0], context)
    if op == "ne":
        return _not_equal(a, args[0], context)
    if op == "array":
        return _array_protocol(a, args[0], args[1], context)
    if op == "buffer":
        return _buffer(a, context)
    if op == "reduce":
        return _reduce_protocol(a, context)
    raise Error("TypeError: unknown native Array operation " + op)


def array_native_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        if len(args) < 3:
            raise Error("TypeError: native Array operations expect an operation, an Array and a context")
        var b = native_loaded_module("builtins")
        if py_type(args[0]) is not b.str:
            raise Error("TypeError: native Array operation names are exact strings")
        var rest = Python.list()
        for i in range(3, len(args)):  # small-loop(args: operation arguments): at most three per operation
            _ = py_call(rest.append, args[i])
        var result = _dispatch(String(args[0]), args[1], args[2], rest)
        return result.steal_data()
    except error:
        var message = String(error)
        if message.startswith("IndexError: "):
            return raise_python_exception(Error(message[byte=12:]), ExceptionType("PyExc_IndexError"))
        if message.startswith("BufferError: "):
            return raise_python_exception(Error(message[byte=13:]), ExceptionType("PyExc_BufferError"))
        return optimizer_python_error(error)
