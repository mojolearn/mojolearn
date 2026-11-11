# SPDX-License-Identifier: Apache-2.0
# Shared native model-selection storage, label, fold and random helpers.
# Binding entries are connected at import. Input admission precedes metadata,
# indexing and conversion, and unsupported Python protocols never run.
from std.memory import unsafe_memcpy
from core.admitted_convert import admitted_convert_vector
from bindings.native_loaded_module import native_loaded_module
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.cpython_table import cpython_table
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_setattr, py_contains, py_fields
from bindings.py_attr import py_attr
from bindings.optimizer_public_words import stash_python_error, PYTHON_ERROR_PENDING, _restore_python_error
from bindings.classical_arrays import classical_address, classical_wrap_array
from bindings.family_common import (
    FAMILY_I64P, FAMILY_I32P, FAMILY_F64P, family_array, family_call, family_call_kw, family_isinstance,
    family_is_str, family_repr, family_str, family_type_name,
)
from bindings.metrics_reg_native import _Prog, _reg_run, _arena_base
from bindings.native_buffer_owner import native_storage_allocate
from bindings.arena_materialize_native import _prepared_info, _slot
from providers.python.borrowed_buffer_pyapi import AdmittedBorrowedBuffer
from bindings.msel_inputs import msel_admit, msel_input, msel_input_count
from bindings.sequential_admission import _one_of
from bindings.classical_labels import _label_list, classical_encode_labels
from bindings.label_input_sources import labels_list_scalar_kind, labels_text_storage, LABELS_LIST_OTHER
from bindings.native_typed_context import native_typed_context
from bindings.classical_attributes import classical_connect_numpy
from bindings.metrics_native_support import metric_warn
from bindings.native_torch_admission import native_torch_exact

comptime MSEL_NONE = -1
comptime MSEL_U8P = MutPointer[UInt8, MutUntrackedOrigin]
comptime MSEL_U32P = MutPointer[UInt32, MutUntrackedOrigin]
#: `_expansion_metrics._FOLD_ROWS_BOUND`: the largest K-fold row table.
comptime MSEL_FOLD_ROWS_BOUND = 1 << 27
comptime MSEL_GOLDEN: UInt64 = 0x9E3779B97F4A7C15


# ---------------------------------------------------------------- calls


def msel_b() raises -> PythonObject:
    return native_loaded_module("builtins")


def msel_native(name: String, context: PythonObject) raises -> PythonObject:
    """The identical helper resolved once during package connection."""
    var entry = py_get(context["family_msel_entries"], PythonObject(name))
    if entry is None:
        raise Error("ImportError: model selection has no connected native helper " + name)
    if py_type(entry) is py_type(PythonObject("")):
        raise Error(String(entry))
    if py_type(entry) is not context["family_builtin_type"]:
        raise Error("TypeError: model selection requires its connected compiled helper " + name)
    return entry


def msel_entry(name: String, context: PythonObject) raises -> PythonObject:
    """A preconnected compiled helper; never call a Python loader."""
    return msel_native(name, context)


def msel_entry_ready(name: String, context: PythonObject) raises -> Bool:
    """Whether the compiled helper `name` was connected at import (an older
    core binding without it keeps its refusal there): the caller then takes
    the helper it replaces."""
    var entry = py_get(context["family_msel_entries"], PythonObject(name))
    return entry is not None and py_type(entry) is context["family_builtin_type"]


def msel_try(callable: PythonObject, args: PythonObject, classes: PythonObject,
             context: PythonObject) raises -> PythonObject:
    """`callable(*args)`; an exception of `classes` returns the miss
    sentinel (`context["family_msel_miss"]`), any other is restored at the
    door unchanged."""
    ref cpy = Python().cpython()
    var result = cpy.PyObject_Call(callable._obj_ptr, args._obj_ptr, PyObjectPtr())
    if result:
        return PythonObject(from_owned=result)
    if cpython_table()[].PyErr_ExceptionMatches(classes._obj_ptr) != 0:
        cpython_table()[].PyErr_Clear()
        return context["family_msel_miss"]
    var failure = stash_python_error()
    raise failure^


def msel_apply(callable: PythonObject, args: PythonObject) raises -> PythonObject:
    """`callable(*args)` for an argument tuple; its exception is restored at
    the door unchanged."""
    ref cpy = Python().cpython()
    var result = cpy.PyObject_Call(callable._obj_ptr, args._obj_ptr, PyObjectPtr())
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


def msel_missed(value: PythonObject, context: PythonObject) raises -> Bool:
    return value is context["family_msel_miss"]


def _error_class(message: String) raises -> PythonObject:
    """The Python class the door (bindings/classical_native.mojo) raises
    for an owner Error: its mapped prefixes, else ValueError."""
    var b = msel_b()
    var names: List[String] = ["TypeError", "OverflowError", "NotImplementedError", "ImportError", "IndexError",
                               "AttributeError", "RuntimeError"]
    for i in range(len(names)):  # small-loop(error classes: the door's prefixes)
        if message.startswith(names[i] + ": "):
            return py_call(b.getattr, b, PythonObject(names[i]))
    return b.ValueError


def msel_catch(error: Error, classes: PythonObject) raises -> Bool:
    """Whether `error` (an owner Error or a stashed Python exception) is an
    instance of `classes`, as the Python `except classes:` tested. A caught
    stashed exception is cleared; an uncaught one stays stashed, so the
    caller re-raises `error` unchanged."""
    var message = String(error)
    if message == PYTHON_ERROR_PENDING:
        _ = _restore_python_error()
        ref cpy = Python().cpython()
        if cpython_table()[].PyErr_ExceptionMatches(classes._obj_ptr) != 0:
            cpython_table()[].PyErr_Clear()
            return True
        _ = stash_python_error()
        return False
    return Bool(py=py_call(msel_b().issubclass, _error_class(message), classes))


def msel_warn(message: String, category: PythonObject, context: PythonObject) raises:
    """`warnings.warn(message, category)`, attributed to the public call's
    caller."""
    metric_warn(message, category, context, 2)


# ---------------------------------------------------------------- values


def msel_is_bool(value: PythonObject, context: PythonObject) raises -> Bool:
    """`_labels.is_bool`: Python's bool or NumPy's bool_."""
    classical_connect_numpy(context)
    return _one_of(py_type(value), py_get(context, "bool_types", Python.tuple(py_type(PythonObject(True)))))


def msel_is_integral(value: PythonObject, context: PythonObject) raises -> Bool:
    # Exact native scalar identities avoid numbers.Integral's Python ABC hooks.
    classical_connect_numpy(context)
    return py_type(value) is py_type(PythonObject(True)) or _one_of(py_type(value), context["integer_types"])


def msel_is_int_arg(value: PythonObject, context: PythonObject) raises -> Bool:
    """`isinstance(v, numbers.Integral) and not is_bool(v)`."""
    return msel_is_integral(value, context) and not msel_is_bool(value, context)


def msel_is_real(value: PythonObject, context: PythonObject) raises -> Bool:
    classical_connect_numpy(context)
    var cls = py_type(value)
    return (cls is py_type(PythonObject(True)) or cls is py_type(PythonObject(0))
            or cls is py_type(PythonObject(0.0))
            or (_one_of(cls, context["scalar_types"]) and not msel_is_bool(value, context)))


def _msel_exact_type(value: PythonObject, sample: PythonObject) -> Bool:
    """`type(value) is type(sample)` by object-header pointer (no call)."""
    ref cpy = Python().cpython()
    var t = cpy.PyObject_Type(value._obj_ptr)
    var s = cpy.PyObject_Type(sample._obj_ptr)
    var same = t == s
    cpy.Py_DecRef(t)
    cpy.Py_DecRef(s)
    return same


def msel_int(value: PythonObject) raises -> Int:
    # pass 3 (rank 202): an exact int is `int(value)` itself; read it
    # natively instead of admitting builtins and calling int() per value.
    if _msel_exact_type(value, PythonObject(0)):
        return Int(py=value)
    return Int(py=family_call(msel_b().int, value))


def msel_op(name: String) raises -> PythonObject:
    return py_call(msel_b().getattr, native_loaded_module("operator"), PythonObject(name))


def _msel_cmp_ints(name: String, x: Int, y: Int) -> Bool:
    if name == "eq":
        return x == y
    if name == "ne":
        return x != y
    if name == "lt":
        return x < y
    if name == "le":
        return x <= y
    if name == "gt":
        return x > y
    return x >= y


def msel_cmp(name: String, a: PythonObject, b: PythonObject) raises -> Bool:
    """`operator.<name>(a, b)` as a truth value (the Python comparison)."""
    # pass 3 (rank 202): two exact ints that fit a machine word, and eq/ne of
    # two exact strs, compare natively with Python's result; every other
    # pair (bools, floats, subclasses, rich-comparison hooks) keeps the
    # Python comparison.
    var known = name == "eq" or name == "ne" or name == "lt" or name == "le" or name == "gt" or name == "ge"
    if known and _msel_exact_type(a, PythonObject(0)) and _msel_exact_type(b, PythonObject(0)):
        var fits = True
        var x = 0
        var y = 0
        try:
            x = Int(py=a)
            y = Int(py=b)
        except:
            fits = False
        if fits:
            return _msel_cmp_ints(name, x, y)
    if ((name == "eq" or name == "ne") and _msel_exact_type(a, PythonObject("")) and _msel_exact_type(b, PythonObject(""))):
        var decoded = True
        var same = False
        try:
            same = String(py=a) == String(py=b)
        except:
            decoded = False  # a str UTF-8 cannot hold (lone surrogate): Python compares it
        if decoded:
            return same if name == "eq" else not same
    return Bool(py=family_call(msel_b().bool, family_call(msel_op(name), a, b)))


def msel_materialize(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """Admitted native storage, in C order, with the original dtype promotion."""
    return msel_input(value, name, context)


def msel_try_materialize(value: PythonObject, name: String, classes: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    """`_materialize(value, name)[0]`, or the miss sentinel on `classes`."""
    try:
        return msel_materialize(value, name, context)
    except error:
        if msel_catch(error, classes):
            return context["family_msel_miss"]
        raise error^


def msel_flatten(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_labels.flatten_labels(value)` (the label input step)."""
    msel_admit(value, context)
    var source = value
    if py_type(value) is msel_b().range:
        source = py_call(msel_b().list, value)
    elif py_type(value) is context["array_class"] or native_torch_exact(py_type(value)):
        # Validate retained slots and owners before the native label reader
        # reads Array._mv; its Python convenience methods are never called.
        source = msel_materialize(value, "labels", context)
    var b = msel_b()
    var cls = py_type(source)
    if cls is not b.list and cls is not b.tuple and cls is not b.str and cls is not b.bytes:
        source = _label_list(source, context)
        # An exact object ndarray's native tolist preserves arbitrary leaves.
        # Admit those owners before any classification or flattening protocol.
        msel_admit(source, context)
        cls = py_type(source)
    if cls is b.str or cls is b.bytes:
        raise Error("mojolearn: y must be a sequence of labels")
    var flat = Python.list()
    _msel_flatten_into(source, flat)
    return flat


def _msel_flatten_into(item: PythonObject, out_buf: PythonObject) raises:
    # Exact native containers: isinstance could consult a leaf's __class__.
    var cls = py_type(item)
    if cls is py_type(Python.list()) or cls is py_type(Python.tuple()):
        for i in range(len(item)):  # driver-loop: flatten already admitted native label containers
            _msel_flatten_into(item[i], out_buf)
    else:
        _ = py_call(out_buf.append, item)


def _msel_admit_labels(values: PythonObject, context: PythonObject) raises:
    var b = msel_b()
    for i in range(len(values)):  # driver-loop: label scalar identities before hashing, equality or error display
        var item = values[i]
        var cls = py_type(item)
        if cls is b.str or msel_is_real(item, context) or msel_is_bool(item, context):
            continue
        # Native NumPy str_ is admitted by its exact type before this native
        # subclass check. Unknown label types never reach their repr hook.
        if Bool(py=py_call(b.issubclass, cls, b.str)):
            continue
        raise Error("TypeError: model-selection labels must be native numbers or strings")


def _msel_exact_str_items(value: PythonObject) raises -> Bool:
    """True for an exact nonempty list or tuple whose every item is exactly a
    str: one pass over the container's C storage comparing item classes by
    pointer (no Python call per label), stopping at the first other item."""
    var b = msel_b()
    var cls = py_type(value)
    var is_tuple = cls is b.tuple
    if cls is not b.list and not is_tuple:
        return False
    var n = len(value)
    if n == 0:
        return False
    ref cpy = Python().cpython()
    # N066: the item reads through the one CPython table (resolved once per
    # process), not a by-name lookup per call.
    var table = cpython_table()
    var str_type = b.str
    var exact = True
    for i in range(n):  # driver-loop: exact list/tuple item classes by pointer, one C read per label
        var raw = table[].PyTuple_GetItem(value._obj_ptr, i) if is_tuple else table[].PyList_GetItem(value._obj_ptr, i)
        if not raw:
            raise cpy.unsafe_get_error()
        var item_type = cpy.PyObject_Type(raw)
        var same = item_type == str_type._obj_ptr
        cpy.Py_DecRef(item_type)
        if not same:
            exact = False
            break
    _ = value  # keep the container alive through the borrowed item reads above
    _ = str_type
    return exact


def _msel_direct_typestr(typestr: String) -> Bool:
    """The numeric label dtypes the shared encoder reads from their buffer
    with the same classes the flattened Python numbers encode to."""
    return (typestr == "<i4" or typestr == "<i8" or typestr == "<u4"
            or typestr == "<f4" or typestr == "<f8")


def msel_numeric_vector(value: PythonObject, column: Bool, context: PythonObject) raises -> Bool:
    """A non-empty exact ndarray (C-contiguous) or native Array label vector
    of a dtype `_msel_direct_typestr` accepts: labels the shared encoder
    reads from the buffer, with no Python leaf to admit. `column` also
    accepts an (n, 1) Array."""
    var ndarray = py_get(context, "ndarray")
    if ndarray is not None and py_type(value) is ndarray:
        msel_admit(value, context)
        var typestr = String(value.dtype.str)
        return (Int(py=value.ndim) == 1 and Int(py=value.size) > 0 and Bool(py=value.flags.c_contiguous)
                and _msel_direct_typestr(typestr))
    var array_class = py_get(context, "array_class")
    if array_class is not None and py_type(value) is array_class:
        msel_admit(value, context)
        var typestr = String(py=value.dtype)
        var ndim = Int(py=value.ndim)
        var vector = ndim == 1 or (column and ndim == 2 and Int(py=value.shape[1]) == 1)
        return vector and Int(py=value.size) > 0 and _msel_direct_typestr(typestr)
    return False


def msel_encode(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_labels.encode_labels(value)`: (classes, int32 codes)."""
    # A C-contiguous int32/int64/uint32 ndarray vector is encoded straight from
    # its buffer (classical_encode_labels' one prepare_targets call). Flattening
    # it first boxed and admitted every label as a Python int, which only
    # rebuilt the same int64 vector (_list_labels) for the same encoder: same
    # Python-int classes and codes, so no bits change. Every other input keeps
    # the label input step below.
    #
    # Perf (lane perf-svm-and-msel): the same holds for float32/float64
    # vectors (the flattened Python floats went through _list_labels' float64
    # encode: the same values, the same prepare_targets order and NaN
    # refusal) and for the native Array msel_try_materialize returns
    # (msel_discrete_codes: default folds of cross_validate, check_cv and
    # the stratified splitters), whose C-order storage is its logical order,
    # a column vector included.
    var direct = msel_numeric_vector(value, True, context)
    # perf3-nbr-msel-trees: an exact flat list or tuple of exact ints, exact
    # floats or exact bools, and an exact '<U' ndarray, go to the shared
    # encoder as given: its one-pass readers (classical_labels
    # `_flat_list_labels`, `_text_buffer_labels`) give the classes and codes
    # the flatten + per-label admission + encode gave (the flatten of such a
    # list is the same items in the same order, and every item passes
    # `_msel_admit_labels`), without those two per-label walks.
    if not direct:
        if labels_list_scalar_kind(value) != LABELS_LIST_OTHER:
            direct = True
        elif _msel_exact_str_items(value):
            # pass 3 (rank 37): an exact flat list or tuple of exact str. Its
            # flatten is the same items in the same order and every item
            # passes `_msel_admit_labels` (exact str), so the shared encoder
            # gets what the two per-label Python walks gave it.
            direct = True
        elif labels_text_storage(value, context):
            msel_admit(value, context)
            direct = True
    var flat = value
    if not direct:
        flat = msel_flatten(value, context)
        _msel_admit_labels(flat, context)
    # The labels context (context plus the concrete real/integral type
    # tuples) is built once per connected context and cached there, not
    # copied and rebuilt on every encode (bindings/native_typed_context.mojo).
    var labels_context = native_typed_context(context, True, False)
    var encoded = classical_encode_labels(flat, labels_context)
    return Python.tuple(encoded.classes, encoded.codes)


def msel_labels_list(y: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    if y is None:
        raise Error("The '" + name + "' parameter should not be None.")
    var flat = msel_flatten(y, context)
    _msel_admit_labels(flat, context)
    return flat


def msel_n_samples(x: PythonObject, context: PythonObject) raises -> Int:
    """Read sample count only from admitted native container metadata."""
    return msel_input_count(x, context)


def msel_is_array(value: PythonObject, context: PythonObject) raises -> Bool:
    return py_type(value) is context["array_class"]


# ---------------------------------------------------------------- arrays


def msel_addr(array: PythonObject) raises -> Int:
    return Int(py=classical_address(array))


def msel_p64(array: PythonObject) raises -> FAMILY_I64P:
    return FAMILY_I64P(unsafe_from_address=msel_addr(array))


def msel_p32(array: PythonObject) raises -> FAMILY_I32P:
    return FAMILY_I32P(unsafe_from_address=msel_addr(array))


def msel_pf64(array: PythonObject) raises -> FAMILY_F64P:
    return FAMILY_F64P(unsafe_from_address=msel_addr(array))


def msel_new(n: Int, typestr: String, context: PythonObject) raises -> PythonObject:
    """`empty((n,), typestr)` (zero filled)."""
    return family_array(Python.tuple(n), typestr, context)


def msel_new2(rows: Int, cols: Int, typestr: String, context: PythonObject) raises -> PythonObject:
    return family_array(Python.tuple(rows, cols), typestr, context)


def msel_i64_list(values: List[Int], context: PythonObject) raises -> PythonObject:
    """`Array.from_list(values, '<i8')`."""
    var out = msel_new(len(values), "<i8", context)
    var p = msel_p64(out)
    _ = out  # keep the Python object alive through the native call above
    for i in range(len(values)):  # driver-loop: over a short result list (sizes, folds)
        p[i] = Int64(values[i])
        _ = out  # keep the Python object alive through the native call above
    return out


def msel_i32_list(values: List[Int], context: PythonObject) raises -> PythonObject:
    var out = msel_new(len(values), "<i4", context)
    var p = msel_p32(out)
    _ = out  # keep the Python object alive through the native call above
    for i in range(len(values)):  # driver-loop: over a short per-group or per-class table
        p[i] = Int32(values[i])
        _ = out  # keep the Python object alive through the native call above
    return out


def msel_f64_list(values: List[Float64], context: PythonObject) raises -> PythonObject:
    """`Array.from_list(values, '<f8')`."""
    var out = msel_new(len(values), "<f8", context)
    var p = msel_pf64(out)
    _ = out  # keep the Python object alive through the native call above
    for i in range(len(values)):  # driver-loop: over one score per fold or candidate
        p[i] = values[i]
        _ = out  # keep the Python object alive through the native call above
    return out


def msel_f64_block(values: List[Float64], rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
    """`Array.from_list(flat, '<f8').reshape((rows, cols))`."""
    var out = msel_new2(rows, cols, "<f8", context)
    var p = msel_pf64(out)
    _ = out  # keep the Python object alive through the native call above
    for i in range(len(values)):  # driver-loop: over the score block (sizes or candidates x folds)
        p[i] = values[i]
        _ = out  # keep the Python object alive through the native call above
    return out


def msel_read_i64(array: PythonObject) raises -> List[Int]:
    """The values of a short int64 Array (fold sizes, class counts)."""
    var out = List[Int]()
    var p = msel_p64(array)
    for i in range(Int(py=array.size)):  # driver-loop: over a short table (classes, folds)
        out.append(Int(p[i]))
    return out^


def msel_slice_i64(array: PythonObject, start: Int, count: Int, context: PythonObject) raises -> PythonObject:
    """`array[start:start + count]` of a C int64 Array, copied."""
    var out = msel_new(count, "<i8", context)
    if count > 0:
        unsafe_memcpy(dest=msel_p64(out), src=msel_p64(array) + start, count=count)
        _ = out  # keep the Python object alive through the native call above
    return out


def msel_index_i64(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """A 1-D integer index Array as C int64 (`value` itself when it is one,
    else one typed parallel widening, the `astype('<i8')`)."""
    var dtype = String(py=value.dtype)
    if dtype == "<i8":
        return value
    var codes = context["family_msel_codes"]
    if not py_contains(codes, PythonObject(dtype)):
        # a narrow integer dtype the cast does not name: `array('q', view)`
        # widens it in C
        var store = family_call(native_loaded_module("array").array, PythonObject("q"), value._mv)
        return classical_wrap_array(store, Python.tuple(len(store)), "<i8", "C", context)
    var out = msel_new(Int(py=value.size), "<i8", context)
    if Int(py=value.size) > 0:
        # The typed parallel conversion (core/admitted_convert.mojo), not a
        # py_call into the base binding's cast_elements.
        _ = admitted_convert_vector(msel_addr(value), dtype, msel_addr(out), "<i8", Int(py=value.size))
        _ = out  # keep the Python object alive through the native call above
    return out


def msel_arange(start: Int, n: Int, context: PythonObject) raises -> PythonObject:
    """`_arange_rows(start, n)`: int64 rows start .. start + n - 1."""
    var out = msel_new(n, "<i8", context)
    if n > 0:
        _ = family_call(msel_entry("arange_i64", context), PythonObject(msel_addr(out)), PythonObject(start), PythonObject(n))
        _ = out  # keep the Python object alive through the native call above
    return out


def msel_range_index(lo: Int, hi: Int, context: PythonObject) raises -> PythonObject:
    """`_as_index(range(lo, hi))`."""
    return msel_arange(lo, max(hi - lo, 0), context)


def msel_split_mask(mask: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`_split_mask`: (train, test), the ascending rows whose mask byte is 0
    / nonzero, by `count_mask_u8` and `select_mask_u8_i64`."""
    var maddr = msel_addr(mask)
    var n_test = msel_int(family_call(msel_entry("count_mask_u8", context), PythonObject(maddr), PythonObject(n)))
    var test = msel_new(n_test, "<i8", context)
    var train = msel_new(n - n_test, "<i8", context)
    _ = family_call(msel_entry("select_mask_u8_i64", context), PythonObject(maddr), PythonObject(n),
                    PythonObject(msel_addr(test) if n_test else 0), PythonObject(msel_addr(train) if n - n_test else 0))
    _ = train  # keep the Python object alive through the native call above
    _ = test  # keep the Python object alive through the native call above
    return Python.tuple(train, test)


def msel_split_indices(iaddr: Int, k: Int, n: Int, context: PythonObject) raises -> PythonObject:
    """`_split_indices`: (train, test) from k int64 test rows at `iaddr`."""
    var mask = msel_new(n, "<u1", context)
    _ = family_call(msel_entry("mask_from_indices_u8", context), PythonObject(iaddr if k else 0), PythonObject(k),
                    PythonObject(n), PythonObject(msel_addr(mask)))
    _ = mask  # keep the Python object alive through the native call above
    return msel_split_mask(mask, n, context)


def msel_index_copy(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_index_copy(value)`: an int64 / int32 1-D buffer widened to an
    owned int64 Array; anything else `_as_index(flatten_labels(value))`."""
    var b = msel_b()
    msel_admit(value, context)
    var cls = py_type(value)
    if cls is not b.list and cls is not b.tuple and cls is not b.range:
        var a = msel_try_materialize(value, "cv indices", Python.tuple(b.TypeError, b.ValueError), context)
        if not msel_missed(a, context) and Int(py=a.ndim) == 1:
            var dtype = String(py=a.dtype)
            if dtype == "<i8":
                return msel_slice_i64(a, 0, Int(py=a.size), context)
            if dtype == "<i4":
                return msel_index_i64(a, context)
    return msel_as_index(msel_flatten(value, context), context)


def msel_as_index(values: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_as_index(values)` of a Python sequence of ints: `array('q', values)`
    (ints and bools as they are, anything else through int()) as an owned
    int64 Array."""
    var b = msel_b()
    msel_admit(values, context)
    var array_mod = native_loaded_module("array")
    var store = msel_try(array_mod.array, Python.tuple(PythonObject("q"), values), b.TypeError, context)
    if msel_missed(store, context):
        store = family_call(array_mod.array, PythonObject("q"), py_call(b.map, b.int, values))
    return classical_wrap_array(store, Python.tuple(len(store)), "<i8", "C", context)


# ---------------------------------------------------------------- groups


struct MselGroups(Movable):
    """`_GroupCodes`: groups (or labels) under the order rule as int32 codes
    with their per-code row counts."""
    var codes: PythonObject
    var classes: PythonObject
    var m: Int
    var n: Int
    var counts: List[Int]
    var scratch: PythonObject
    # This group plan belongs to one native public operation. Keep the
    # constructor's admitted entries and private storage addresses through
    # all group/fold calls; a later operation constructs a fresh plan.
    var select_entry: PythonObject
    var gather_entry: PythonObject
    var codes_address: Int
    var scratch_address: Int

    def __init__(out self, values: PythonObject, n: Int, name: String, context: PythonObject) raises:
        if values is None:
            _ = msel_labels_list(values, name, context)
        _ = msel_entry("fold_ids", context)
        var select_entry = msel_entry("select_fold_i64", context)
        var gather_entry = msel_entry("gather_i32", context)
        var got = msel_encode(values, context)
        var classes = got[0]
        var codes = got[1]
        if Int(py=codes.ndim) != 1 or Int(py=codes.size) != n:
            raise Error(name + " has " + String(Int(py=codes.size)) + " labels for " + String(n) + " samples")
        if String(py=codes.dtype) != "<i4":
            var narrow = msel_new(n, "<i4", context)
            # The label codes narrowed to int32 by the typed parallel
            # conversion (core/admitted_convert.mojo), not a py_call into the
            # base binding's cast_elements.
            if not admitted_convert_vector(msel_addr(codes), String(py=codes.dtype), msel_addr(narrow), "<i4", n):
                raise Error("TypeError: " + name + " label codes have dtype " + String(py=codes.dtype) + ", which no int32 conversion reads")
            _ = codes  # keep the Python object alive through the native call above
            _ = narrow  # keep the Python object alive through the native call above
            codes = narrow
        var m = len(classes)
        var counts = List[Int](length=m, fill=0)
        if m > 0:
            var class_counts = msel_new(m, "<i8", context)
            var fold_store = msel_new(n, "<i4", context)
            var fold_counts = msel_new(2, "<i8", context)
            _ = family_call(msel_entry("fold_ids", context), PythonObject(msel_addr(codes)), PythonObject(n), PythonObject(m),
                            PythonObject(2), PythonObject(msel_addr(class_counts)), PythonObject(msel_addr(fold_store)),
                            PythonObject(msel_addr(fold_counts)))
            _ = codes  # keep the Python object alive through the native call above
            _ = fold_store  # keep the Python object alive through the native call above
            _ = class_counts  # keep the Python object alive through the native call above
            _ = fold_counts  # keep the Python object alive through the native call above
            counts = msel_read_i64(class_counts)
        self.counts = counts^
        self.codes = codes
        self.classes = classes
        self.m = m
        self.n = n
        self.scratch = PythonObject(None)
        self.select_entry = select_entry
        self.gather_entry = gather_entry
        self.codes_address = msel_addr(codes)
        self.scratch_address = 0

    def mapped(self, table: List[Int], context: PythonObject) raises -> PythonObject:
        """Each row's `table[code]` as int32 words."""
        var tbl = msel_i32_list(table, context)
        var _keep_ret_tbl_r412 = self.mapped_words(tbl, context)
        _ = tbl  # keep the Python object alive through the native call above
        var _keep_ret_tbl = _keep_ret_tbl_r412^
        _ = tbl  # keep the Python object alive through the native call above
        return _keep_ret_tbl^

    def mapped_words(self, tbl: PythonObject, context: PythonObject) raises -> PythonObject:
        var dst = msel_new(self.n, "<i4", context)
        _ = family_call(self.gather_entry, PythonObject(msel_addr(tbl)), tbl.size,
                        PythonObject(self.codes_address), PythonObject(self.n), PythonObject(msel_addr(dst)))
        _ = self.codes  # keep the captured private code storage through the gather
        _ = tbl
        _ = dst  # keep the Python object alive through the native call above
        return dst

    def rows(self, words: PythonObject, want: Int, size: Int, context: PythonObject) raises -> PythonObject:
        """(train, test): the ascending rows whose word is `want` (test) and
        every other row (train)."""
        var address = self.codes_address if words is self.codes else msel_addr(words)
        var result = self._rows_at(address, want, size, context)
        _ = words
        return result

    def _rows_at(self, words_address: Int, want: Int, size: Int, context: PythonObject) raises -> PythonObject:
        # Caller retains the words owner through the complete selection.
        var test = msel_new(size, "<i8", context)
        var train = msel_new(self.n - size, "<i8", context)
        var got = msel_int(family_call(self.select_entry, PythonObject(words_address),
                                       PythonObject(self.n), PythonObject(want), PythonObject(msel_addr(test)),
                                       PythonObject(msel_addr(train))))
        _ = test  # keep the Python object alive through the native call above
        _ = train  # keep the Python object alive through the native call above
        if got != size:
            raise Error("RuntimeError: mojolearn: select_fold_i64 disagrees with the group counts")
        _ = train  # keep the Python object alive through the native calls above
        _ = test  # keep the Python object alive through the native calls above
        return Python.tuple(train, test)

    def only(mut self, words: PythonObject, want: Int, size: Int, context: PythonObject) raises -> PythonObject:
        """The ascending rows whose word is `want` (the rest discarded)."""
        if self.scratch is None:
            var scratch = msel_new(self.n, "<i8", context)
            var address = msel_addr(scratch)
            # Publish the owner/address pair only after capture succeeds.
            self.scratch = scratch
            self.scratch_address = address
        var out = msel_new(size, "<i8", context)
        var address = self.codes_address if words is self.codes else msel_addr(words)
        var got = msel_int(family_call(self.select_entry, PythonObject(address),
                                       PythonObject(self.n), PythonObject(want), PythonObject(msel_addr(out)),
                                       PythonObject(self.scratch_address)))
        _ = words
        _ = self.scratch
        _ = out  # keep the Python object alive through the native call above
        if got != size:
            raise Error("RuntimeError: mojolearn: select_fold_i64 disagrees with the group counts")
        _ = out  # keep the Python object alive through the native calls above
        return out

    def split_by_fold(self, to_fold: PythonObject, k: Int, sizes: PythonObject,
                      context: PythonObject) raises -> PythonObject:
        """[(train, test)] per fold from each group's fold (int32 words) and
        each fold's row count (an int64 Array)."""
        var words = self.mapped_words(to_fold, context)
        var out = Python.list()
        var sz = msel_p64(sizes)
        var words_address = msel_addr(words)
        if msel_entry_ready("select_folds_i64", context):
            # One split plan for every fold (N051).
            var planned = msel_select_folds_plan(words, sizes, self.n, k, "the group counts", context)
            _ = words
            _ = sizes
            return planned
        for f in range(k):  # driver-loop: over folds
            _ = py_call(out.append, self._rows_at(words_address, f, Int(sz[f]), context))
        _ = words
        _ = sizes
        return out


def msel_select_folds_plan(fold_store: PythonObject, fold_counts: PythonObject, n: Int, k: Int, counted_by: String,
                           context: PythonObject) raises -> PythonObject:
    """Every fold's (train, test) in ONE `select_folds_i64` call (lane
    ne1-b-door, N051: one split plan per CV call; the fold ids go up once and
    every fold's rows come back behind one synchronize). Same arrays, same
    integers, same order as the per-fold `select_fold_i64` loop. `fold_store`
    holds the n int32 fold words, `fold_counts` each fold's int64 row count;
    `counted_by` names the counts' source in the disagreement refusal."""
    var counts = msel_p64(fold_counts)
    var tests = msel_new(k, "<i8", context)
    var trains = msel_new(k, "<i8", context)
    var tp = msel_p64(tests)
    var rp = msel_p64(trains)
    var out = Python.list()
    # An empty side shares one private scratch allocation, as in the loop.
    var scratch = PythonObject(None)
    var scratch_address = 0
    for fold in range(k):  # driver-loop: over folds (output allocation only; the selection is one call)
        var size = Int(counts[fold])
        var test = msel_new(size, "<i8", context)
        var train = msel_new(n - size, "<i8", context)
        if size == 0 or n - size == 0:
            if scratch is None:
                scratch = msel_new(n, "<i8", context)
                scratch_address = msel_addr(scratch)
        tp.unsafe_store(fold, Int64(msel_addr(test) if size else scratch_address))
        rp.unsafe_store(fold, Int64(msel_addr(train) if n - size else scratch_address))
        _ = py_call(out.append, Python.tuple(train, test))
    var bad = msel_int(family_call(msel_entry("select_folds_i64", context), PythonObject(msel_addr(fold_store)),
                                   PythonObject(n), PythonObject(k), PythonObject(msel_addr(fold_counts)),
                                   PythonObject(msel_addr(tests)), PythonObject(msel_addr(trains))))
    _ = out  # keep every fold array alive through the native call above
    _ = scratch
    _ = tests
    _ = trains
    _ = fold_store
    _ = fold_counts
    if bad >= 0:
        raise Error("RuntimeError: mojolearn: select_fold_i64 disagrees with " + counted_by)
    return out


# ---------------------------------------------------------------- random draws


def msel_mix64(var x: UInt64) -> UInt64:
    """splitmix64's finalizer (`_expansion_metrics._mix64`), in wrapping
    64-bit arithmetic: the exact Python integers modulo 2**64."""
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB
    return x ^ (x >> 31)


def msel_u64(value: PythonObject) raises -> UInt64:
    """`int(value) & (2**64 - 1)` of a Python int."""
    var b = msel_b()
    var whole = family_call(b.int, value)
    var lo = Int(py=py_call(msel_op("and_"), whole, PythonObject(0xFFFFFFFF)))
    var hi = Int(py=py_call(msel_op("and_"), py_call(msel_op("rshift"), whole, PythonObject(32)), PythonObject(0xFFFFFFFF)))
    return (UInt64(hi) << 32) | UInt64(lo)


def msel_signed32(v: UInt64) -> Int:
    """A 32-bit word as the signed parameter `_Prog.stage` records."""
    var w = Int(v & 0xFFFFFFFF)
    return w - (1 << 32) if w >= (1 << 31) else w


struct MselRng(Movable):
    """`CounterRng`: draw k of seed s is a device permutation keyed by
    `_mix64(s * GOLDEN + k + 1)`."""
    var seed: UInt64
    var draws: UInt64

    def __init__(out self, random_state: PythonObject, context: PythonObject) raises:
        var state = random_state
        if state is None:
            state = family_call(msel_b().int.from_bytes, py_call(native_loaded_module("os").urandom, PythonObject(8)),
                                PythonObject("little"))
        if (msel_is_bool(state, context) or not msel_is_integral(state, context)
                or msel_cmp("lt", state, PythonObject(0))):
            raise Error("mojolearn model_selection: random_state must be None or a non-negative int; "
                        + "numpy RandomState / Generator objects are NOT IMPLEMENTED (their stream "
                        + "is numpy's, x_metrics/split.mojo DEVIATION 6108)")
        self.seed = msel_u64(state)
        self.draws = 0

    def salt(mut self) -> UInt64:
        var s = msel_mix64(self.seed * MSEL_GOLDEN + self.draws + 1)
        self.draws += 1
        return s

    def permute_stage(mut self, mut prog: _Prog, n: Int) raises -> Int:
        """The next draw as a `permute` stage of `prog`; returns its slot."""
        var s = self.salt()
        var out = prog.alloc(max(n, 1))
        prog.stage("permute", 1, [n, out, msel_signed32(s), msel_signed32(s >> 32)])
        return out

    def permutation_rows(mut self, sizes: List[Int], context: PythonObject) raises -> PythonObject:
        """One draw per size, in order, as int64 row Arrays (`permutation_rows`)."""
        var prog = _Prog(context["family_metrics_ops"])
        var outs = List[Int]()
        for i in range(len(sizes)):  # driver-loop: over the requested draws
            var o = self.permute_stage(prog, sizes[i])
            var slot = prog.alloc(2 * sizes[i])
            var w = prog.want(slot, 2 * sizes[i])
            if sizes[i] > 0:
                prog.stage("rows64", sizes[i], [o, w])
            outs.append(w)
        var result = Python.list()
        if len(outs) == 0:
            return result
        var arena = _reg_run(prog, PythonObject(None), context)
        var base = _arena_base(arena)
        for i in range(len(sizes)):  # driver-loop: over the requested draws
            var rows = msel_new(sizes[i], "<i8", context)
            if sizes[i] > 0:
                unsafe_memcpy(dest=msel_p64(rows), src=FAMILY_I64P(unsafe_from_address=base + 4 * outs[i]), count=sizes[i])
                _ = rows  # keep the Python object alive through the native call above
            _ = py_call(result.append, rows)
        # The copied offsets are raw addresses into arena. Retain its storage
        # through every row allocation/copy, not just through _arena_base.
        _ = arena
        _ = prog^
        return result

    def permutation(mut self, size: Int, context: PythonObject) raises -> PythonObject:
        """`permutation(size)`: one draw read back as a list of ints (the
        arena words as `array('i')`, as `_Prog.ints` read them)."""
        var prog = _Prog(context["family_metrics_ops"])
        var slot = self.permute_stage(prog, size)
        var o = prog.want(slot, size)
        var arena = _reg_run(prog, PythonObject(None), context)
        _ = prog^
        var store = py_call(native_loaded_module("array").array, PythonObject("i"))
        _ = py_call(store.frombytes, py_call(py_call(py_attr(arena, "__getitem__"), py_call(msel_b().slice, PythonObject(o), PythonObject(o + size))).tobytes))
        return py_call(msel_b().list, store)

    def approximate_modes(mut self, counts: List[Int], n_train: Int, n_test: Int,
                          context: PythonObject) raises -> List[Int]:
        """StratifiedShuffleSplit's per-class train then test counts (two
        `approx_mode` stages of one program; the draws they report are
        consumed). Returns n_i followed by t_i."""
        var k = len(counts)
        var prog = _Prog(context["family_metrics_ops"])
        var c = prog.put_i32(msel_i32_list(counts, context))
        var out_slot = prog.alloc(2 * k)
        var out = prog.want(out_slot, 2 * k)
        var used_slot = prog.alloc(2)
        var used = prog.want(used_slot, 2)
        var w0 = msel_signed32(self.seed)
        var w1 = msel_signed32(self.seed >> 32)
        var w2 = msel_signed32(self.draws)
        var w3 = msel_signed32(self.draws >> 32)
        prog.stage("approx_mode", k, [k, c, MSEL_NONE, n_train, out, MSEL_NONE, used, w0, w1, w2, w3])
        prog.stage("approx_mode", k, [k, c, out, n_test, out + k, used, used + 1, w0, w1, w2, w3])
        var arena = _reg_run(prog, PythonObject(None), context)
        # Completed arena metadata is invariant across the class-count loop.
        var words = FAMILY_I32P(unsafe_from_address=_arena_base(arena))
        self.draws += UInt64(Int(words[used]) + Int(words[used + 1]))
        var both = List[Int]()
        for i in range(2 * k):  # driver-loop: over classes (train and test counts)
            both.append(Int(words[out + i]))
        _ = arena
        _ = prog^
        return both^


def msel_mix_seed(seed: UInt64, r: Int) -> Int:
    """`_mix_seed(seed, r)`: repeat r's seed."""
    return Int(msel_mix64(seed * MSEL_GOLDEN + UInt64(0x632BE59BD9B4E019) * UInt64(r + 1)) >> 1)


# ---------------------------------------------------------------- shuffle sizes


def _size_kind(v: PythonObject, context: PythonObject) raises -> String:
    if v is None:
        return "n"
    if msel_is_bool(v, context):
        return "x"
    if msel_is_integral(v, context):
        return "i"
    if msel_is_real(v, context):
        return "f"
    return "x"


def msel_shuffle_sizes(n: Int, test_size_in: PythonObject, train_size: PythonObject,
                       default_test: PythonObject, context: PythonObject) raises -> Tuple[Int, Int]:
    """`_validate_shuffle_split`: (n_train, n_test) with its refusals."""
    var test_size = test_size_in
    if test_size is None and train_size is None:
        test_size = default_test
    var tk = _size_kind(test_size, context)
    var rk = _size_kind(train_size, context)
    var pn = PythonObject(n)
    var zero = PythonObject(0)
    var one = PythonObject(1)
    if ((tk == "i" and (msel_cmp("ge", test_size, pn) or msel_cmp("le", test_size, zero)))
            or (tk == "f" and not (msel_cmp("lt", zero, test_size) and msel_cmp("lt", test_size, one)))):
        raise Error("test_size=" + family_str(test_size) + " should be either positive and smaller than the number of "
                    + "samples " + String(n) + " or a float in the (0, 1) range")
    if ((rk == "i" and (msel_cmp("ge", train_size, pn) or msel_cmp("le", train_size, zero)))
            or (rk == "f" and not (msel_cmp("lt", zero, train_size) and msel_cmp("lt", train_size, one)))):
        raise Error("train_size=" + family_str(train_size) + " should be either positive and smaller than the number "
                    + "of samples " + String(n) + " or a float in the (0, 1) range")
    if rk == "x":
        raise Error("Invalid value for train_size: expected a native numeric size")
    if tk == "x":
        raise Error("Invalid value for test_size: expected a native numeric size")
    var add = msel_op("add")
    var mul = msel_op("mul")
    if rk == "f" and tk == "f" and msel_cmp("gt", family_call(add, train_size, test_size), one):
        raise Error("The sum of test_size and train_size = " + family_str(family_call(add, train_size, test_size))
                    + ", should be in the (0, 1) range. Reduce test_size and/or train_size.")
    var math = native_loaded_module("math")
    var n_test = PythonObject(None)
    var n_train = PythonObject(None)
    if tk == "f":
        n_test = family_call(math.ceil, family_call(mul, test_size, pn))
    else:
        n_test = test_size
    if rk == "f":
        n_train = family_call(math.floor, family_call(mul, train_size, pn))
    else:
        n_train = train_size
    if train_size is None:
        n_train = family_call(msel_op("sub"), pn, n_test)
    elif test_size is None:
        n_test = family_call(msel_op("sub"), pn, n_train)
    var total = family_call(add, n_train, n_test)
    if msel_cmp("gt", total, pn):
        raise Error("The sum of train_size and test_size = " + family_str(total) + ", should be smaller than "
                    + "the number of samples " + String(n) + ". Reduce test_size and/or train_size.")
    var tr = msel_int(n_train)
    var te = msel_int(n_test)
    if tr == 0:
        raise Error("With n_samples=" + String(n) + ", test_size=" + family_str(test_size) + " and train_size="
                    + family_str(train_size) + ", the resulting train set will be empty. Adjust any of the "
                    + "aforementioned parameters.")
    return (tr, te)


# ---------------------------------------------------------------- resident rows


def msel_resident_op(res: PythonObject, op: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """One `_Resident` lifecycle operation through its compiled door
    (bindings/arena_resident_native.mojo), as the Python methods call it."""
    return family_call(context["family_msel_resident_call"], res, PythonObject(op), args,
                       context["family_msel_resident_context"])


def msel_resident_new(context: PythonObject) raises -> PythonObject:
    """`_Resident()`: the instance allocated and initialized by its door."""
    var res = native_storage_allocate(context["family_msel_resident_class"])
    _ = msel_resident_op(res, "msel_resident_init", Python.tuple(), context)
    return res


def msel_resident_take(res: PythonObject, values: PythonObject, indices: PythonObject,
                       context: PythonObject) raises -> PythonObject:
    return msel_resident_op(res, "msel_resident_take", Python.tuple(values, indices), context)


def msel_resident_release(res: PythonObject, values: PythonObject, context: PythonObject) raises:
    _ = msel_resident_op(res, "msel_resident_release", Python.tuple(values), context)


def msel_resident_close(res: PythonObject, context: PythonObject) raises:
    _ = msel_resident_op(res, "msel_resident_close", Python.tuple(), context)


def msel_take_labels(values: PythonObject, indices: PythonObject, context: PythonObject) raises -> PythonObject:
    """Gather admitted list/tuple references without boxed index iteration."""
    # Public input admission already rejects list/tuple subclasses (including
    # ordinary subclasses); do not bypass that policy with the C list APIs.
    var b = msel_b()
    var kind = py_type(values)
    var tuple_input = kind is b.tuple
    if kind is not b.list and not tuple_input:
        raise Error("TypeError: model-selection label rows require an exact native list or tuple")
    var row_context = context["family_msel_resident_context"]
    # Generated folds and msel_indices produce prepared C int64 Arrays. Recheck
    # mutable Array metadata and backing storage at each gather, never cache an
    # address across fits. _prepared_info admits the exact native view/exporter.
    var metadata = _prepared_info(indices, row_context)
    if String(metadata["dtype"]) != "<i8" or Int(py=metadata["ndim"]) != 1:
        raise Error("TypeError: model-selection label rows require one-dimensional int64 indices")
    var view = _slot(indices, "_mv", row_context)
    var pin = AdmittedBorrowedBuffer(view)
    var out = PythonObject(None)
    try:
        var layout = pin.layout.value().copy()
        if (len(layout.shape) != 1 or layout.itemsize != 8 or not layout.contiguous()
                or layout.count != Int(py=metadata["size"]) or pin.format != "q"):
            raise Error("ValueError: model-selection label index storage disagrees with its metadata")
        ref cpy = Python().cpython()
        # N066: every C entry through the one CPython table (resolved once
        # per process), not four by-name lookups per fold side.
        var table = cpython_table()
        var count = table[].PyTuple_Size(values._obj_ptr) if tuple_input else table[].PyList_Size(values._obj_ptr)
        var raw = table[].PyList_New(layout.count)
        if not raw:
            var failure = stash_python_error()
            raise failure^
        out = PythonObject(from_owned=raw)
        var words = FAMILY_I64P(unsafe_from_address=layout.address)
        for i in range(layout.count):  # driver-loop: native indices copy existing label references into sized output
            var index = Int(words[i])
            if index < 0:
                index += count
            # GetItem reports the original list/tuple IndexError for either
            # bound. Py_NewRef supplies the owned reference SetItem steals,
            # including on failure; out owns and cleans up any partial list.
            var label = (table[].PyTuple_GetItem(values._obj_ptr, index) if tuple_input
                         else table[].PyList_GetItem(values._obj_ptr, index))
            if not label:
                var failure = stash_python_error()
                raise failure^
            if table[].PyList_SetItem(out._obj_ptr, i, cpy.Py_NewRef(label)) != 0:
                var failure = stash_python_error()
                raise failure^
        _ = values  # borrowed labels remain owned through the final incref
    finally:
        pin.release()  # retain actual buffer export through the final index read
    return out


def msel_take_any(values: PythonObject, indices: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_take_any` / `_take_rows`: a list or tuple taken by index (a list),
    else one device gather out of a `_Resident` used once."""
    var b = msel_b()
    msel_admit(values, context)
    if py_type(values) is b.list or py_type(values) is b.tuple:
        return msel_take_labels(values, indices, context)
    var prepared = msel_materialize(values, "split data", context)
    var res = msel_resident_new(context)
    var out = PythonObject(None)
    try:
        out = msel_resident_take(res, prepared, indices, context)
    finally:
        msel_resident_close(res, context)
    return out


def msel_lazy(model: PythonObject, op: String, items: PythonObject, context: PythonObject) raises -> PythonObject:
    """A lazy iterator of `door(model, op, item, context)` over `items` (each
    an argument tuple): `map` over the compiled door, so a splitter with
    one split per row or per combination makes each split on demand, as the
    Python generator did, with no Python frame."""
    var itertools = native_loaded_module("itertools")
    return py_call(msel_b().map, context["family_msel_door"], py_call(itertools.repeat, model),
                   py_call(itertools.repeat, PythonObject(op)), items, py_call(itertools.repeat, context))


# ---------------------------------------------------------------- scorers
from bindings.msel_estimator_native import msel_estimator_call
from bindings.msel_scorer_native import msel_scorer_fields, msel_scorer_call, msel_scorer_value, MselMetricPrepared
from bindings.metrics_native_support import metric_float


def _msel_attr(model: PythonObject, name: String) raises -> PythonObject:
    """`model.<name>` with its own AttributeError."""
    return family_call(native_loaded_module("builtins").getattr, model, PythonObject(name))

def msel_get_scorer(var scoring: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if scoring is None:
        return scoring
    if py_type(scoring) is b.str:
        var table = context["family_msel_scorers"]
        if not py_contains(table, scoring):
            raise Error(family_repr(scoring) + " is not a valid scoring value. Use "
                        + "sklearn.metrics.get_scorer_names() to get valid options.")
        scoring = table[scoring]
    _ = msel_scorer_fields(scoring, context)
    return scoring


def msel_score(estimator: PythonObject, x: PythonObject, y: PythonObject, scorer: PythonObject,
               context: PythonObject) raises -> PythonObject:
    # Construct the no-memo sentinel during execution: a PythonObject default
    # requires compile-time evaluation in callers that omit the argument.
    return msel_score(estimator, x, y, scorer, context, PythonObject(None))


def msel_score(estimator: PythonObject, x: PythonObject, y: PythonObject, scorer: PythonObject,
               context: PythonObject, memo: PythonObject) raises -> PythonObject:
    """`model_selection._score(estimator, X, y, scorer)`: the estimator's own
    `score` without a scorer; a real scalar, as a float."""
    var prepared = MselMetricPrepared()
    return PythonObject(msel_score_value(estimator, x, y, scorer, context, memo, prepared))


def msel_score_value(estimator: PythonObject, x: PythonObject, y: PythonObject, scorer: PythonObject,
                     context: PythonObject, memo: PythonObject,
                     mut prepared: MselMetricPrepared) raises -> Float64:
    # Internal consumers already own native Float64 score tables. Preserve
    # public wrappers while avoiding float boxing and real-type rechecks here.
    if scorer is not None:
        return msel_scorer_value(scorer, estimator, x, y, PythonObject(None), memo, context, prepared)
    var value = msel_estimator_call(estimator, "score", Python.tuple(x, y), context)
    if not msel_is_real(value, context):
        raise Error("TypeError: scoring must return a real scalar")
    return metric_float(value, context)
