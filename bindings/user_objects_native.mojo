# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# The user-object reader: a Python object that cannot cross into Mojo as a
# buffer (a list of label objects, a ragged nested list, a user's index
# pairs or group ids) is handed to ONE native call, which reads it once
# through the C API and returns typed stores (array.array words) or the
# objects placed where they go. No Python bytecode runs per element: the
# per-element loops below are the read itself, each tagged with the object
# it reads. Lane purge-user-objects (2026-10-22) moved these loops here from
# model_selection.py, ensemble.py, _umap_impl.py, _metrics_impl.py,
# linear_model.py, resample.py, _array.py, _expansion_trees.py and
# _expansion_prep.py (python/mojolearn/_user_objects.py is the Python door).
#
#   ("take", values, address, count, width)  -> [values[i] for i in idx]
#   ("first_seen", values)                    -> (classes, int32 codes) by dict equality
#   ("lookup", classes, values)               -> (int32 codes, -1 unseen; distinct unseen values)
#   ("kinds", values)                         -> counts per kind (KIND_* below)
#   ("ints", values) / ("floats", values)     -> int64 / float64 words of int(v) / float(v)
#   ("strs", values)                          -> [str(v) for v in values]
#   ("column", rows, j) / ("column_f32", rows, j) -> [r[j] for r in rows] / its float32 words
#   ("transpose", cols)                       -> [list(r) for r in zip(*cols)]
#   ("scatter", rows, address, count, vals)   -> rows[idx[k]] = vals[k]
#   ("flatten", obj)                          -> leaves of admitted native containers/array storage
#   ("nested", obj, array_type)               -> (shape, leaves) of a rectangular nested list
#   ("group_runs", ids)                       -> int64 run lengths of a ranking pool's group ids
#   ("pairs", rows, weights, n_rows)          -> (int64 winner/loser words, float64 weights)
#   ("sorted_codes", values, unknown)         -> float32 codes over sorted((str(type(v)), v))
#   ("f32_exact", values)                     -> [float(v)] when float32 holds every real label, else None
#   ("indicator", rows)                       -> (status, float32 words, width) of a 0/1 label matrix
#   ("group_rows", codes, k)                  -> k int64 stores: the rows of each code
#   ("group_permute", values, rows, perms)    -> values permuted within each group's rows
#
# A refusal comes back as ("__refuse__", exception name, message); the door
# raises it, so the exception type and wording are the ones the Python had.
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.py_balanced import py_call, py_type, py_setitem, py_get
from bindings.boundary_read import admitted_buffer_view
from core.boundary_layout import gather_i64, gather_f64
from bindings.py_attr import py_attr
from core.user_objects import group_run_sizes
from bindings.user_object_flatten import user_object_flatten
from bindings.cpython_table import cpython_table
from bindings.cpython_borrowed import borrowed_type_word, borrowed_item
from bindings.user_objects_native_run import _refusal, _index_at, _float_is_integer

comptime KIND_STR = 0         # isinstance(v, str)
comptime KIND_BOOL = 1        # bool or NumPy bool_ (_labels.is_bool)
comptime KIND_EXACT_INT = 2   # type(v) is int
comptime KIND_INTEGRAL = 3    # isinstance(v, numbers.Integral)
comptime KIND_REAL = 4        # isinstance(v, numbers.Real)
comptime KIND_INTVAL = 5      # Integral, or a finite integer-valued Real
comptime KIND_SEQ = 6         # isinstance(v, (list, tuple))
comptime KIND_INT_OR_BOOL = 7 # Integral or bool / bool_
comptime KIND_PLAIN_INT = 8   # Integral and not bool / bool_
comptime KIND_REAL_NONBOOL = 9  # isinstance(v, numbers.Real) and not isinstance(v, bool)
comptime N_KINDS = 10


struct _Env(Movable):
    var b: PythonObject
    var op: PythonObject
    var integral: PythonObject
    var real: PythonObject
    var seq: PythonObject
    var array: PythonObject
    # Callables read once per operation, not one attribute lookup per element.
    var isinstance_fn: PythonObject
    var eq_fn: PythonObject
    var lt_fn: PythonObject
    # Class words of the exact builtins, read once per operation: an item's
    # class is compared by pointer (lane list-input-loops, 2026-11-12).
    var list_word: Int
    var tuple_word: Int
    var int_word: Int
    var float_word: Int
    var bool_word: Int
    var str_word: Int
    var bytes_word: Int

    def __init__(out self) raises:
        var b = native_loaded_module("builtins")
        # The import-time connection (sys.modules read and stamp), not the
        # import machinery on every call.
        var numbers = native_loaded_module("numbers")
        self.seq = Python.tuple(b.list, b.tuple)
        self.integral = numbers.Integral
        self.real = numbers.Real
        self.op = native_loaded_module("operator")
        self.array = native_loaded_module("array").array
        self.isinstance_fn = b.isinstance
        self.eq_fn = self.op.eq
        self.lt_fn = self.op.lt
        self.list_word = Int(b.list._obj_ptr)
        self.tuple_word = Int(b.tuple._obj_ptr)
        self.int_word = Int(b.int._obj_ptr)
        self.float_word = Int(b.float._obj_ptr)
        self.bool_word = Int(b.bool._obj_ptr)
        self.str_word = Int(b.str._obj_ptr)
        self.bytes_word = Int(b.bytes._obj_ptr)
        self.b = b

    def isinstance(self, v: PythonObject, cls: PythonObject) raises -> Bool:
        return Bool(py=py_call(self.isinstance_fn, v, cls))

    def eq(self, a: PythonObject, b: PythonObject) raises -> Bool:
        return Bool(py=py_call(self.eq_fn, a, b))

    def lt(self, a: PythonObject, b: PythonObject) raises -> Bool:
        return Bool(py=py_call(self.lt_fn, a, b))

    def is_bool(self, v: PythonObject) raises -> Bool:
        if self.isinstance(v, self.b.bool):
            return True
        return String(py=py_type(v).__name__) == "bool_"

    def text(self, v: PythonObject) raises -> String:
        return String(py=py_call(self.b.str, v))

    def py_repr(self, v: PythonObject) raises -> String:
        return String(py=py_call(self.b.repr, v))

    def store(self, code: String, n: Int, itemsize: Int) raises -> PythonObject:
        return py_call(self.array, code, py_call(self.b.bytes, itemsize * max(n, 0)))


# ------------------------------------------------------------ item reads
# Lane list-input-loops (2026-11-12): the list-input loops below read the
# items of an exact list or tuple borrowed from its C storage
# (PyList_GetItem / PyTuple_GetItem: no boxed index, no PyObject_GetItem, no
# new object per item) and convert an exact int / float by its class word
# with one C call. Any other container is read through its own __getitem__
# and any other item class keeps the builtin's protocol (PyNumber_Long /
# PyNumber_Float are exactly what int(v) / float(v) call), so the values,
# refusals and their texts are the former ones.

comptime SEQ_OTHER = 0   # read through the container's own __getitem__
comptime SEQ_LIST = 1    # exact list: borrowed PyList_GetItem
comptime SEQ_TUPLE = 2   # exact tuple: borrowed PyTuple_GetItem


def _seq_kind(env: _Env, values: PythonObject) raises -> Int:
    var word = borrowed_type_word(values._obj_ptr)
    if word == env.list_word:
        return SEQ_LIST
    if word == env.tuple_word:
        return SEQ_TUPLE
    return SEQ_OTHER


def _item_at(values: PythonObject, kind: Int, i: Int) raises -> PythonObject:
    """`values[i]`: a new reference to the stored item of an exact list or
    tuple (one incref; an index past the end raises the same IndexError),
    else (or for a negative index) the container's own subscript."""
    if kind == SEQ_OTHER or i < 0:
        return values[i]
    return PythonObject(from_borrowed=borrowed_item(values, i, kind == SEQ_TUPLE))


def _int_word(env: _Env, v: PythonObject, mut overflow: List[Int32]) raises -> Int64:
    """`int(v)` as an int64 word, refused with the OverflowError the former
    `int(v)` range check raised outside int64. An exact int is one
    PyLong_AsLongLongAndOverflow; any other class one PyNumber_Long first.
    `overflow` is the caller's one-word flag, allocated once per loop."""
    var t = cpython_table()
    ref cpy = Python().cpython()
    var number = v
    if borrowed_type_word(v._obj_ptr) != env.int_word:
        var converted = t[].PyNumber_Long(v._obj_ptr)
        if not converted:
            var pending = cpy.unsafe_get_error()
            raise pending^
        number = PythonObject(from_owned=converted)
    overflow[0] = 0
    var word = t[].PyLong_AsLongLongAndOverflow(number._obj_ptr, Int(overflow.unsafe_ptr()))
    if overflow[0] != 0:
        var refused = _refusal("OverflowError", "Python int too large to convert to C long")
        raise refused^
    if word == -1 and t[].PyErr_Occurred():
        var pending = cpy.unsafe_get_error()
        raise pending^
    _ = number
    return word


def _float_word(env: _Env, v: PythonObject) raises -> Float64:
    """`float(v)`: an exact float or int is one PyFloat_AsDouble (an int too
    large raises the OverflowError float() raises); any other class one
    PyNumber_Float first."""
    var t = cpython_table()
    ref cpy = Python().cpython()
    var word = borrowed_type_word(v._obj_ptr)
    if word == env.float_word or word == env.int_word:
        var x = t[].PyFloat_AsDouble(v._obj_ptr)
        if x == -1.0 and t[].PyErr_Occurred():
            var pending = cpy.unsafe_get_error()
            raise pending^
        return x
    var converted = t[].PyNumber_Float(v._obj_ptr)
    if not converted:
        var pending = cpy.unsafe_get_error()
        raise pending^
    var owned = PythonObject(from_owned=converted)
    var y = t[].PyFloat_AsDouble(owned._obj_ptr)
    if y == -1.0 and t[].PyErr_Occurred():
        var pending = cpy.unsafe_get_error()
        raise pending^
    _ = owned
    return y


def _is_real_nonbool(env: _Env, v: PythonObject) raises -> Bool:
    """`isinstance(v, numbers.Real) and not isinstance(v, bool)`: an exact
    int or float by class word, any other class through isinstance."""
    var word = borrowed_type_word(v._obj_ptr)
    if word == env.int_word or word == env.float_word:
        return True
    if word == env.bool_word or word == env.str_word:
        return False
    return not env.isinstance(v, env.b.bool) and env.isinstance(v, env.real)


def _new_list(n: Int) raises -> PythonObject:
    """A presized list of n NULL slots that `_list_put` fills, every slot
    exactly once."""
    ref cpy = Python().cpython()
    var raw = cpython_table()[].PyList_New(n)
    if not raw:
        var pending = cpy.unsafe_get_error()
        raise pending^
    return PythonObject(from_owned=raw)


def _list_put(dst: PythonObject, i: Int, v: PythonObject) raises:
    """`dst[i] = v` into a `_new_list` slot (PyList_SetItem steals the new
    reference taken here)."""
    ref cpy = Python().cpython()
    _ = cpython_table()[].PyList_SetItem(dst._obj_ptr, i, cpy.Py_NewRef(v._obj_ptr))


def _address(store: PythonObject) raises -> Int:
    return Int(py=store.buffer_info()[0])

def _take(env: _Env, values: PythonObject, address: Int, count: Int, width: Int) raises -> PythonObject:
    var n = len(values)
    var kind = _seq_kind(env, values)
    var result = _new_list(max(count, 0))
    for k in range(count):  # list-input: Python list of label objects read once by user_objects_c (take): borrowed item, presized result
        var i = _index_at(address, k, width)
        var j = i + n if i < 0 else i
        if j < 0 or j >= n:
            raise _refusal("IndexError", "list index out of range")
        _list_put(result, k, _item_at(values, kind, j))
    return result


def _first_seen(env: _Env, values: PythonObject) raises -> PythonObject:
    # {v: i for i, v in enumerate(dict.fromkeys(values))} and each value's code:
    # dict equality and hashing, the object's own protocol, read from Mojo
    var n = len(values)
    var index = Python.dict()
    var classes = Python.list()
    var codes = env.store("i", n, 4)
    var out = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=_address(codes))
    _ = codes  # keep the Python object alive through the native call above
    var kind = _seq_kind(env, values)
    for i in range(n):  # api-bound-loop: dict.setdefault of each user label object (its own __hash__ / __eq__); items read borrowed
        var v = _item_at(values, kind, i)
        var k = len(classes)
        var got = py_call(index.setdefault, v, PythonObject(k))
        var code = Int(py=got)
        if code == k:
            _ = py_call(classes.append, v)
        out[i] = Int32(code)
        _ = codes  # keep the Python object alive through the native call above
    return Python.tuple(classes, codes)


def _lookup(env: _Env, classes: PythonObject, values: PythonObject) raises -> PythonObject:
    # {c: i for i, c in enumerate(classes)}.get(v, -1) per value (a repeated
    # class keeps its last index, as the comprehension does), and the distinct
    # unseen values in first-seen order
    var index = Python.dict()
    var class_kind = _seq_kind(env, classes)
    for i in range(len(classes)):  # api-bound-loop: dict store of each user class object (its own __hash__ / __eq__); items read borrowed
        py_setitem(index, _item_at(classes, class_kind, i), PythonObject(i))
    var n = len(values)
    var codes = env.store("i", n, 4)
    var out = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=_address(codes))
    _ = codes  # keep the Python object alive through the native call above
    var unseen = Python.dict()
    var missing = Python.list()
    var none = PythonObject(None)
    var kind = _seq_kind(env, values)
    var get_fn = index.get
    for i in range(n):  # api-bound-loop: dict lookup of each user label object (its own __hash__ / __eq__); items read borrowed
        var v = _item_at(values, kind, i)
        var got = py_call(get_fn, v, none)
        if got is none:
            out[i] = Int32(-1)
            _ = codes  # keep the Python object alive through the native call above
            var before = len(unseen)
            _ = py_call(unseen.setdefault, v, none)
            if len(unseen) != before:
                _ = py_call(missing.append, v)
        else:
            out[i] = Int32(Int(py=got))
            _ = codes  # keep the Python object alive through the native call above
    return Python.tuple(codes, missing)

def _is_intval(env: _Env, v: PythonObject) raises -> Bool:
    # math.isfinite(v) and float(v).is_integer(), for a Real
    return _float_is_integer(Float64(py=py_call(env.b.float, v)))


def _kinds(env: _Env, values: PythonObject) raises -> PythonObject:
    var counts = List[Int](length=N_KINDS, fill=0)
    var n = len(values)
    var kind = _seq_kind(env, values)
    var t = cpython_table()
    for i in range(n):  # list-input: Python list of label objects read once by user_objects_c (kinds): borrowed item, exact str/int/float/bool by class word
        var v = _item_at(values, kind, i)
        # The exact builtin classes by class word, with the counts the ABC
        # checks below give them; any other class takes the checks.
        var word = borrowed_type_word(v._obj_ptr)
        if word == env.str_word:
            counts[KIND_STR] += 1
            continue
        if word == env.int_word:
            counts[KIND_EXACT_INT] += 1
            counts[KIND_INTEGRAL] += 1
            counts[KIND_INTVAL] += 1
            counts[KIND_REAL] += 1
            counts[KIND_INT_OR_BOOL] += 1
            counts[KIND_PLAIN_INT] += 1
            counts[KIND_REAL_NONBOOL] += 1
            continue
        if word == env.float_word:
            counts[KIND_REAL] += 1
            counts[KIND_REAL_NONBOOL] += 1
            if _float_is_integer(t[].PyFloat_AsDouble(v._obj_ptr)):
                counts[KIND_INTVAL] += 1
            continue
        if word == env.bool_word:
            counts[KIND_BOOL] += 1
            counts[KIND_INTEGRAL] += 1
            counts[KIND_INTVAL] += 1
            counts[KIND_REAL] += 1
            counts[KIND_INT_OR_BOOL] += 1
            continue
        var is_b = env.is_bool(v)
        var integral = env.isinstance(v, env.integral)
        var real = integral or env.isinstance(v, env.real)
        if env.isinstance(v, env.b.str):
            counts[KIND_STR] += 1
        if is_b:
            counts[KIND_BOOL] += 1
        if py_type(v) is env.b.int:
            counts[KIND_EXACT_INT] += 1
        if integral:
            counts[KIND_INTEGRAL] += 1
            counts[KIND_INTVAL] += 1
        elif real and _is_intval(env, v):
            counts[KIND_INTVAL] += 1
        if real:
            counts[KIND_REAL] += 1
        if env.isinstance(v, env.seq):
            counts[KIND_SEQ] += 1
        if integral or is_b:
            counts[KIND_INT_OR_BOOL] += 1
        if integral and not is_b:
            counts[KIND_PLAIN_INT] += 1
        if real and not env.isinstance(v, env.b.bool):
            counts[KIND_REAL_NONBOOL] += 1
    var out = Python.list()
    for k in range(N_KINDS):  # driver-loop: over the kind counters
        _ = py_call(out.append, PythonObject(counts[k]))
    return out


def _as_list(env: _Env, values: PythonObject) raises -> PythonObject:
    if py_type(values) is env.b.list:
        return values
    return py_call(env.b.list, values)


def _ndarray_vector(values: PythonObject, integers_only: Bool) raises -> Bool:
    """True for an exact, little-endian or native, one-dimensional NumPy
    array of integer or bool storage (and float storage unless
    `integers_only`; uint64 only then), whose `int(v)` / `float(v)` per
    entry is exactly the native widening of its words."""
    var numpy = py_get(native_loaded_module("sys").modules, "numpy")
    if numpy is None or not (py_type(values) is numpy.ndarray):
        return False
    var dtype = values.dtype
    if Int(py=values.ndim) != 1 or String(py=dtype.byteorder) == ">":
        return False
    var kind = String(py=dtype.kind)
    if kind == "i" or kind == "b":
        return True
    if kind == "u":
        return not integers_only or Int(py=dtype.itemsize) < 8
    return kind == "f" and not integers_only


def _ints(env: _Env, values: PythonObject) raises -> PythonObject:
    if _ndarray_vector(values, True):
        # ONE buffer read (bindings/boundary_read.mojo), the int64 words
        # copied natively; list(values) made one NumPy scalar per entry.
        # PERF FIX (perf-audit-3, 2026-11-11): gathered straight into the
        # store (one memcpy for C-order int64, one native widening pass
        # otherwise); `as_i64_c` + memcpy copied a non-int64 input twice.
        var view = admitted_buffer_view(values)
        var count = view.count()
        var packed = env.store("q", count, 8)
        gather_i64(view.layout, view.kind, _address(packed))
        view.release()
        return packed
    var vals = _as_list(env, values)
    var n = len(vals)
    var words = env.store("q", n, 8)
    var out = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=_address(words))
    _ = words  # keep the Python object alive through the native call above
    var overflow = List[Int32](length=1, fill=0)
    for i in range(n):  # list-input: Python int objects read once by user_objects_c (ints): borrowed item, one PyLong_AsLongLongAndOverflow
        out[i] = _int_word(env, _item_at(vals, SEQ_LIST, i), overflow)
        _ = words  # keep the Python object alive through the native call above
    _ = overflow
    return words


def _floats(env: _Env, values: PythonObject) raises -> PythonObject:
    if _ndarray_vector(values, False):
        # ONE buffer read, the float64 words widened natively.
        # PERF FIX (perf-audit-3): gathered straight into the store, as `_ints`.
        var view = admitted_buffer_view(values)
        var count = view.count()
        var packed = env.store("d", count, 8)
        gather_f64(view.layout, view.kind, _address(packed))
        view.release()
        return packed
    var vals = _as_list(env, values)
    var n = len(vals)
    var words = env.store("d", n, 8)
    var out = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=_address(words))
    _ = words  # keep the Python object alive through the native call above
    for i in range(n):  # list-input: Python number objects read once by user_objects_c (floats): borrowed item, one PyFloat_AsDouble
        out[i] = _float_word(env, _item_at(vals, SEQ_LIST, i))
        _ = words  # keep the Python object alive through the native call above
    return words


def _strs(env: _Env, values: PythonObject) raises -> PythonObject:
    var n = len(values)
    var kind = _seq_kind(env, values)
    var str_fn = env.b.str
    var result = _new_list(n)
    for i in range(n):  # api-bound-loop: str() of each user label object (its own __str__); items read borrowed, the result presized
        _list_put(result, i, py_call(str_fn, _item_at(values, kind, i)))
    return result


def _column(env: _Env, rows: PythonObject, j: Int) raises -> PythonObject:
    var n = len(rows)
    var kind = _seq_kind(env, rows)
    var result = _new_list(n)
    for i in range(n):  # list-input: Python label rows read once by user_objects_c (column): borrowed row and item, presized result
        var row = _item_at(rows, kind, i)
        _list_put(result, i, _item_at(row, _seq_kind(env, row), j))
    return result


def _column_f32(env: _Env, rows: PythonObject, j: Int) raises -> PythonObject:
    var n = len(rows)
    var words = env.store("f", n, 4)
    var out = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=_address(words))
    _ = words  # keep the Python object alive through the native call above
    var kind = _seq_kind(env, rows)
    for i in range(n):  # list-input: Python probability rows read once by user_objects_c (column_f32): borrowed row and item, one PyFloat_AsDouble
        var row = _item_at(rows, kind, i)
        out[i] = Float32(_float_word(env, _item_at(row, _seq_kind(env, row), j)))
        _ = words  # keep the Python object alive through the native call above
    return words


def _transpose(env: _Env, cols: PythonObject) raises -> PythonObject:
    var m = len(cols)
    if m == 0:
        return Python.list()
    # Each column and its container kind read once, not once per cell.
    var columns = List[PythonObject](capacity=m)
    var kinds = List[Int](capacity=m)
    for c in range(m):  # driver-loop: over the output columns
        columns.append(cols[c])
        kinds.append(_seq_kind(env, columns[c]))
    var n = len(columns[0])
    for c in range(1, m):  # driver-loop: over the output columns
        n = min(n, len(columns[c]))
    var result = _new_list(n)
    for i in range(n):  # list-input: Python label columns read once by user_objects_c (transpose): borrowed items, presized rows
        var row = _new_list(m)
        for c in range(m):  # driver-loop: over the output columns
            _list_put(row, c, _item_at(columns[c], kinds[c], i))
        _list_put(result, i, row)
    return result


def _scatter(env: _Env, rows: PythonObject, address: Int, count: Int, vals: PythonObject) raises -> PythonObject:
    var n = min(count, len(vals))
    var kind = _seq_kind(env, vals)
    for k in range(n):  # api-bound-loop: each prediction stored into the caller's own container through its __setitem__; items read borrowed
        py_setitem(rows, PythonObject(_index_at(address, k, 8)), _item_at(vals, kind, k))
    return PythonObject(None)


def _nested_into(env: _Env, obj: PythonObject, array_type: PythonObject, out_buf: PythonObject) raises -> PythonObject:
    # _array._flatten: an Array contributes its shape and values; a str,
    # bytes or object without __len__ is a leaf; a list is rectangular
    if env.isinstance(obj, array_type):
        _ = py_call(out_buf.extend, py_call(py_call(obj._as_c)._values))
        return obj.shape
    if env.isinstance(obj, Python.tuple(env.b.str, env.b.bytes)) or not Bool(py=py_call(env.b.hasattr, obj, "__len__")):
        _ = py_call(out_buf.append, obj)
        return Python.tuple()
    var seq = py_call(env.b.list, obj)
    var n = len(seq)
    if n == 0:
        return Python.tuple(PythonObject(0))
    var first = _nested_into(env, _item_at(seq, SEQ_LIST, 0), array_type, out_buf)
    for i in range(1, n):  # api-bound-loop: a ragged user list walked through its items' own __len__ protocol (recursive leaves)
        var shape = _nested_into(env, _item_at(seq, SEQ_LIST, i), array_type, out_buf)
        if not env.eq(shape, first):
            raise _refusal("ValueError", "mojolearn: nested sequence is not rectangular (" + env.text(shape)
                           + " vs " + env.text(first) + ")")
    return py_call(env.op.add, Python.tuple(PythonObject(n)), first)


def _key_bytes(env: _Env, raw: PythonObject, mut keys: List[UInt8]) raises -> Int:
    # the bytes of a bytes object, appended to `keys` (one memcpy through array('B'))
    var arr = py_call(env.array, "B", raw)
    var n = len(arr)
    var at = len(keys)
    for _ in range(n):  # driver-loop: over one key's bytes (room for the memcpy)
        keys.append(UInt8(0))
    if n > 0:
        unsafe_memcpy(dest=keys.unsafe_ptr() + at,
               src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=_address(arr)), count=n)
        _ = arr  # keep the Python object alive through the native call above
    _ = arr^
    return n


def _group_runs(env: _Env, ids: PythonObject) raises -> PythonObject:
    # ensemble._group_id_key per id (bytes as is, str as UTF-8, an integer as
    # its decimal spelling, anything else refused), then the runs natively
    var n = len(ids)
    var keys = List[UInt8]()
    var starts = List[Int](capacity=n)
    var lengths = List[Int](capacity=n)
    var kind = _seq_kind(env, ids)
    for i in range(n):  # list-input: group_id objects (int/str/bytes) read once by user_objects_c (group_runs): borrowed item, exact classes by class word
        var v = _item_at(ids, kind, i)
        var word = borrowed_type_word(v._obj_ptr)
        var exact = word == env.bytes_word or word == env.str_word or word == env.int_word
        var raw: PythonObject
        if word == env.bytes_word or (not exact and env.isinstance(v, env.b.bytes)):
            raw = v
        elif word == env.str_word or (not exact and env.isinstance(v, env.b.str)):
            raw = py_call(v.encode, "utf-8")
        elif word == env.int_word or (env.isinstance(v, env.integral) and not env.isinstance(v, env.b.bool)):
            raw = py_call(py_call(env.b.str, py_call(env.b.int, v)).encode, "ascii")
        else:
            raise _refusal("ValueError", "mojolearn: group_id[" + String(i) + "] object (" + env.py_repr(v)
                           + ") is unsuitable (should be string or integral type)")
        starts.append(len(keys))
        lengths.append(_key_bytes(env, raw, keys))
    var sizes = List[Int]()
    var bad = group_run_sizes(keys, starts, lengths, sizes)
    if bad >= 0:
        raise _refusal("ValueError", "mojolearn: group Ids are not consecutive: the rows of group "
                       + env.py_repr(ids[bad]) + " are split into more than one run (row " + String(bad)
                       + "); CatBoost requires every group's rows to be contiguous")
    var words = env.store("q", len(sizes), 8)
    var out = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=_address(words))
    _ = words  # keep the Python object alive through the native call above
    for k in range(len(sizes)):  # driver-loop: over the groups
        out[k] = Int64(sizes[k])
        _ = words  # keep the Python object alive through the native call above
    return words


def _pairs(env: _Env, rows: PythonObject, weights: PythonObject, n_rows: Int) raises -> PythonObject:
    # ensemble._pairs_arrays: each pair checked in their Pool's order, then
    # flattened winner, loser; one finite nonnegative weight per pair
    var n = len(rows)
    var flat = env.store("q", 2 * n, 8)
    var fw = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=_address(flat))
    _ = flat  # keep the Python object alive through the native call above
    var kind = _seq_kind(env, rows)
    var t = cpython_table()
    ref cpy = Python().cpython()
    var overflow = List[Int32](length=1, fill=0)
    for i in range(n):  # list-input: user pairs (Python index pairs) read once by user_objects_c (pairs): borrowed items, exact int by class word, one PyLong_AsLongLongAndOverflow
        var pair = _item_at(rows, kind, i)
        if len(pair) != 2:
            raise _refusal("ValueError", "mojolearn: Length of pairs[" + String(i) + "] isn't equal to 2.")
        var pair_kind = _seq_kind(env, pair)
        for j in range(2):  # driver-loop: over the two rows of a pair
            var index = _item_at(pair, pair_kind, j)
            var exact_int = borrowed_type_word(index._obj_ptr) == env.int_word
            if not exact_int and (not env.isinstance(index, env.integral) or env.isinstance(index, env.b.bool)):
                raise _refusal("ValueError", "mojolearn: Invalid pairs[" + String(i) + "][" + String(j) + "] = '"
                               + env.text(index) + "' value type=" + env.text(py_type(index))
                               + ": must be an integer.")
            var x = index if exact_int else py_call(env.b.int, index)
            overflow[0] = 0
            var word = t[].PyLong_AsLongLongAndOverflow(x._obj_ptr, Int(overflow.unsafe_ptr()))
            if overflow[0] == 0 and word == -1 and t[].PyErr_Occurred():
                var pending = cpy.unsafe_get_error()
                raise pending^
            # An int beyond int64 is outside the rows too (the former `<` said so).
            if overflow[0] != 0 or word < 0 or Int(word) >= n_rows:
                raise _refusal("ValueError", "mojolearn: pairs[" + String(i) + "][" + String(j) + "] = "
                               + env.text(x) + " is outside the " + String(n_rows) + " rows")
            fw[2 * i + j] = word
            _ = flat  # keep the Python object alive through the native call above
        if fw[2 * i] == fw[2 * i + 1]:
            raise _refusal("ValueError", "mojolearn: pairs[" + String(i) + "] pairs row " + String(fw[2 * i])
                           + " with itself")
            _ = flat  # keep the Python object alive through the native call above
        _ = flat  # keep the Python object alive through the native calls above
    var wout = env.store("d", n, 8)
    var ww = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=_address(wout))
    _ = wout  # keep the Python object alive through the native call above
    if weights is PythonObject(None):
        for i in range(n):  # driver-loop: over the pairs (unit weights)
            ww[i] = 1.0
            _ = wout  # keep the Python object alive through the native call above
        return Python.tuple(flat, wout)
    if len(weights) != n:
        raise _refusal("ValueError", "mojolearn: len(pairs_weight) = " + String(len(weights))
                       + " is not equal to len(pairs) = " + String(n) + " ")
    _ = overflow
    var weight_kind = _seq_kind(env, weights)
    for i in range(n):  # list-input: user pairs_weight objects read once by user_objects_c (pairs): borrowed item, exact float/int/bool by class word, one PyFloat_AsDouble
        var w = _item_at(weights, weight_kind, i)
        var wword = borrowed_type_word(w._obj_ptr)
        var ok = wword == env.float_word or wword == env.int_word or wword == env.bool_word or (
            wword != env.str_word and env.isinstance(w, env.real))
        var x = 0.0
        if ok:
            x = _float_word(env, w)
            ok = x - x == 0.0 and x >= 0.0
        if not ok:
            raise _refusal("ValueError", "mojolearn: pairs_weight[" + String(i)
                           + "] must be a finite nonnegative number")
        ww[i] = x
        _ = wout  # keep the Python object alive through the native call above
    return Python.tuple(flat, wout)


def _sorted_codes(env: _Env, values: PythonObject, unknown: PythonObject) raises -> PythonObject:
    # UMAP's categorical target: sorted({v for v in vals if v != unknown},
    # key=lambda v: (str(type(v)), v)) as float codes, `unknown` kept as -1
    var n = len(values)
    var seen = Python.dict()
    var keys = Python.list()
    var none = PythonObject(None)
    var kind = _seq_kind(env, values)
    for i in range(n):  # api-bound-loop: equality and hashing of each user label object (its own __eq__ / __hash__); items read borrowed
        var v = _item_at(values, kind, i)
        if not env.eq(v, unknown):
            var before = len(seen)
            _ = py_call(seen.setdefault, v, none)
            if len(seen) != before:
                _ = py_call(keys.append, Python.tuple(py_call(env.b.str, py_type(v)), v))
    # a stable merge sort of the distinct labels by their keys' `<`
    var m = len(keys)
    var order = List[Int](capacity=m)
    for i in range(m):  # driver-loop: over the distinct classes
        order.append(i)
    var buf = List[Int](length=m, fill=0)
    var width = 1
    while width < m:  # driver-loop: over merge widths
        var lo = 0
        while lo < m:  # driver-loop: over merged runs
            var mid = min(lo + width, m)
            var hi = min(lo + 2 * width, m)
            var a = lo
            var b = mid
            var k = lo
            while a < mid and b < hi:  # driver-loop: over one merge of classes
                if env.lt(keys[order[b]], keys[order[a]]):
                    buf[k] = order[b]
                    b += 1
                else:
                    buf[k] = order[a]
                    a += 1
                k += 1
            while a < mid:  # driver-loop: over one merge's tail
                buf[k] = order[a]
                a += 1
                k += 1
            while b < hi:  # driver-loop: over one merge's tail
                buf[k] = order[b]
                b += 1
                k += 1
            lo = hi
        for t in range(m):  # driver-loop: over the distinct classes
            order[t] = buf[t]
        width *= 2
    var code = Python.dict()
    for r in range(m):  # driver-loop: over the distinct classes
        py_setitem(code, keys[order[r]][1], PythonObject(r))
    var words = env.store("f", n, 4)
    var out = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=_address(words))
    _ = words  # keep the Python object alive through the native call above
    for i in range(n):  # api-bound-loop: equality and dict lookup of each user label object (its own __eq__ / __hash__); items read borrowed
        var v = _item_at(values, kind, i)
        if env.eq(v, unknown):
            out[i] = Float32(-1.0)
            _ = words  # keep the Python object alive through the native call above
        else:
            out[i] = Float32(Int(py=code[v]))
            _ = words  # keep the Python object alive through the native call above
    return words


def _group_rows(env: _Env, codes: PythonObject, k: Int) raises -> PythonObject:
    # the rows of each code 0..k-1, increasing, one int64 store per code
    var n = len(codes)
    var cp = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=_address(codes))
    var counts = List[Int](length=k, fill=0)
    for i in range(n):  # native data loop (not a driver loop): counts over the caller's int32 code store (no CPython call)
        counts[Int(cp[i])] += 1
    var stores = Python.list()
    var at = List[Int](length=k, fill=0)
    var ptrs = List[Int](capacity=k)
    for c in range(k):  # driver-loop: over the groups
        var st = env.store("q", counts[c], 8)
        ptrs.append(_address(st) if counts[c] > 0 else 0)
        _ = st  # keep the Python object alive through the native call above
        _ = py_call(stores.append, st)
    for i in range(n):  # native data loop (not a driver loop): row scatter over the caller's int32 code store (no CPython call)
        var c = Int(cp[i])
        MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=ptrs[c])[at[c]] = Int64(i)
        at[c] += 1
    return stores


def _group_permute(env: _Env, values: PythonObject, rows: PythonObject, perms: PythonObject) raises -> PythonObject:
    # permutation_test_score's groups route: yp = list(values), then per
    # group, yp[rows[a]] = values[rows[perm[a]]]
    var out = py_call(env.b.list, values)
    var n_out = len(out)
    var kind = _seq_kind(env, values)
    for g in range(len(rows)):  # driver-loop: over the groups
        var r = rows[g]
        var p = perms[g]
        var m = min(len(r), len(p))
        if m == 0:
            continue
        var rp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=_address(r))
        _ = r  # keep the Python object alive through the native call above
        var pp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=_address(p))
        _ = p  # keep the Python object alive through the native call above
        for a in range(m):  # list-input: Python label objects read once by user_objects_c (group_permute): borrowed source item, C store into the copy
            var dst_row = Int(rp[a])
            var item = _item_at(values, kind, Int(rp[Int(pp[a])]))
            if dst_row >= 0 and dst_row < n_out:
                _list_put(out, dst_row, item)
            else:
                py_setitem(out, PythonObject(dst_row), item)
            _ = p  # keep the Python object alive through the native call above
            _ = r  # keep the Python object alive through the native call above
    return out


def _f32_exact(env: _Env, values: PythonObject) raises -> PythonObject:
    # the labels as Python floats when every one is a real number (not a
    # bool) that float32 holds exactly and none is NaN, else None
    var n = len(values)
    var kind = _seq_kind(env, values)
    var result = _new_list(n)
    for i in range(n):  # list-input: Python label objects read once by user_objects_c (f32_exact): borrowed item, exact float/int by class word, presized result
        var v = _item_at(values, kind, i)
        if not _is_real_nonbool(env, v):
            return PythonObject(None)
        var x = _float_word(env, v)
        if x != x or Float64(Float32(x)) != x:
            return PythonObject(None)
        _list_put(result, i, PythonObject(x))
    return result


def _indicator(env: _Env, rows: PythonObject) raises -> PythonObject:
    # _expansion_prep._multilabel_indicator of list rows: (0, float32 words,
    # width), (1, None, 0) when the rows' lengths differ, (2, None, 0) when a
    # value is not an integral real or more than two distinct values occur
    var n = len(rows)
    var kind = _seq_kind(env, rows)
    var w = len(_item_at(rows, kind, 0))
    for i in range(n):  # list-input: Python list-of-lists y read once by user_objects_c (indicator): borrowed rows
        if len(_item_at(rows, kind, i)) != w:
            return Python.tuple(PythonObject(1), PythonObject(None), PythonObject(0))
    var words = env.store("f", n * w, 4)
    var out = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=_address(words))
    _ = words  # keep the Python object alive through the native call above
    var d0 = 0.0
    var d1 = 0.0
    var nd = 0
    for i in range(n):  # list-input: Python list-of-lists y read once by user_objects_c (indicator): borrowed row and item, exact float/int by class word
        var r = _item_at(rows, kind, i)
        var row_kind = _seq_kind(env, r)
        for j in range(w):  # driver-loop: over one row's columns
            var v = _item_at(r, row_kind, j)
            if not _is_real_nonbool(env, v):
                return Python.tuple(PythonObject(2), PythonObject(None), PythonObject(0))
            var x = _float_word(env, v)
            if not _float_is_integer(x):
                return Python.tuple(PythonObject(2), PythonObject(None), PythonObject(0))
            if nd == 0 or (x != d0 and (nd == 1 or x != d1)):
                if nd == 0:
                    d0 = x
                elif nd == 1:
                    d1 = x
                nd += 1
                if nd > 2:
                    return Python.tuple(PythonObject(2), PythonObject(None), PythonObject(0))
            out[i * w + j] = Float32(x)
            _ = words  # keep the Python object alive through the native call above
    return Python.tuple(PythonObject(0), words, PythonObject(w))


def user_objects_call(call: PythonObject) raises -> PythonObject:
    var operation = String(py=call[0])
    if operation == "flatten":
        # The prepared reader does not load numbers/ABC conversion machinery.
        return user_object_flatten(call[1])
    var env = _Env()
    if operation == "take":
        return _take(env, call[1], Int(py=call[2]), Int(py=call[3]), Int(py=call[4]))
    if operation == "first_seen":
        return _first_seen(env, call[1])
    if operation == "lookup":
        return _lookup(env, call[1], call[2])
    if operation == "kinds":
        return _kinds(env, call[1])
    if operation == "ints":
        return _ints(env, call[1])
    if operation == "floats":
        return _floats(env, call[1])
    if operation == "strs":
        return _strs(env, call[1])
    if operation == "column":
        return _column(env, call[1], Int(py=call[2]))
    if operation == "column_f32":
        return _column_f32(env, call[1], Int(py=call[2]))
    if operation == "transpose":
        return _transpose(env, call[1])
    if operation == "scatter":
        return _scatter(env, call[1], Int(py=call[2]), Int(py=call[3]), call[4])
    if operation == "nested":
        var flat = Python.list()
        var shape = _nested_into(env, call[1], call[2], flat)
        return Python.tuple(shape, flat)
    if operation == "group_runs":
        return _group_runs(env, call[1])
    if operation == "pairs":
        return _pairs(env, call[1], call[2], Int(py=call[3]))
    if operation == "sorted_codes":
        return _sorted_codes(env, call[1], call[2])
    if operation == "f32_exact":
        return _f32_exact(env, call[1])
    if operation == "indicator":
        return _indicator(env, call[1])
    if operation == "group_rows":
        return _group_rows(env, call[1], Int(py=call[2]))
    if operation == "group_permute":
        return _group_permute(env, call[1], call[2], call[3])
    raise Error("TypeError: unknown user-object operation " + operation)


def user_objects_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        return user_objects_call(call).steal_data()
    except error:
        var message = String(error)
        if message.startswith("__refuse__"):
            # ("__refuse__", kind, message): the door raises it as that kind
            var rest = String(message[byte=10:])
            var colon = rest.find(":")
            try:
                var refusal = Python.tuple(PythonObject("__refuse__"), PythonObject(String(rest[byte=:colon])),
                                           PythonObject(String(rest[byte=colon + 1:])))
                return refusal.steal_data()
            except tuple_error:
                return raise_python_exception(tuple_error^, ExceptionType.ValueError)
        if message.startswith("TypeError: "):
            return raise_python_exception(Error(String(message[byte=11:])), ExceptionType.TypeError)
        return raise_python_exception(error^, ExceptionType.ValueError)
