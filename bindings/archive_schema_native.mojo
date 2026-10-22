# SPDX-License-Identifier: Apache-2.0
# Schema-driven `save` for the public estimator classes.
#
# Each public class connects one fixed tuple of save steps at class
# definition (python/mojolearn/_native_archive_bridge.py `register`). This
# owner reads the fitted fields from the instance dictionary, applies the
# steps' refusals, packs the metadata vectors and writes the deterministic
# NPZ through `classical_archive_write`: the same member names, dtypes,
# shapes and bytes the 0.8.55 Python writer (`_serialize.write_npz`) wrote.
# No Python save code, constructor, property or callback runs.
#
# A step is a tuple whose first item names its kind:
#
#   ("need", field, exc, msg)          raise exc(msg) unless the instance holds field
#   ("need_set", field, exc, msg)      ... unless it holds field and field is not None
#   ("need_true", field, exc, msg)     ... unless it holds field and field is true
#   ("need_str", field, exc, msg)      ... unless field is a str
#   ("refuse_set", field, exc, msg)    raise exc(msg) when field is held and not None
#   ("refuse_in", field, values, exc, msg)
#                                      raise exc(msg) when field equals one of values
#   ("format", member, text)           a constant text member
#   ("estimator", member)              the class name
#   ("mode", member, prefix)           the numeric mode a predict would use now
#   ("str", member, field)             str(field) for a builtin scalar field
#   ("lower", member, field)           str(field).lower()
#   ("init_name", member, field, alt)  field when it is a str, else alt
#   ("array", member, field)           the field's array, as stored
#   ("flat", member, field)            the field's array in C order, shape (size,)
#   ("head", member, field, count)     the first max(count, 1) elements of a 1-D array
#   ("i8" | "i4" | "f8" | "f4", member, items)
#                                      one packed vector, an item per element
#   ("when", field, value, steps)      run steps when field == value
#   ("when_in", field, values, steps)  run steps when field is held and equals one of values
#   ("when_set", field, steps)         run steps when field is held and not None
#   ("when_none", field, steps)        run steps when field is absent or None
#   ("classes", member, field)         `_labels.classes_member(field)`: int64 for
#                                      integer (or all-bool) labels, float64 for real
#                                      labels, a list of str (one `<U` member) for str
#                                      labels; anything else is refused
#   ("rows", member, dtype, field, items)
#                                      one packed vector of dtype ("<i8", "<i4",
#                                      "<f8", "<f4"): for each element of the list
#                                      field, its items in order; an item's field
#                                      paths start at "_", the element
#   ("concat", member, dtype, field, key)
#                                      the C-order elements of element[key] for each
#                                      element of the list field, in order, as one flat
#                                      dtype vector (4-byte dtypes)
#
# An item is ("int", field), ("bool", field), ("const", number),
# ("dim", field, axis), ("size", field), ("len", field), ("none", field, sentinel),
# ("noneflag", field), ("float", field), ("float_or", field, default),
# ("int_or", field, default), ("bool_or", field, default), ("code", field, table),
# ("if_eq", field, value, item, item) or ("sum", field, key): the int sum of
# element[key] over the list field. A field is an instance-dictionary
# name, or a tuple path: a name, then dictionary keys and sequence indices.
# Messages may name the class as {cls}.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains, py_fields
from bindings.classical_arrays import classical_address, classical_wrap_array, classical_is_array, classical_input, classical_dtype_str, classical_c_bytes, _store_code
from bindings.classical_pandas import classical_admit_value
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_archive import classical_archive_write, _archive_array
from bindings.classical_archive_read import _archive_builtin_repr


def archive_schema_lookup(cls: PythonObject, table: PythonObject) raises -> PythonObject:
    """The entry of `table` for `cls` or its nearest connected ancestor, or None."""
    var b = Python.import_module("builtins")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    for i in range(len(mro)):  # driver-loop: over the loading class's ancestry metadata
        if py_contains(table, mro[i]):
            return table[mro[i]]
    return PythonObject(None)


def _schema_fail(kind: PythonObject, text: PythonObject, name: String) raises:
    raise Error(String(kind) + ": " + String(text).replace("{cls}", name))


def _schema_field(fields: PythonObject, key: PythonObject, name: String) raises -> PythonObject:
    if not py_contains(fields, key):
        raise Error("AttributeError: '" + name + "' object has no attribute '" + String(key) + "'")
    return fields[key]


def _schema_has(fields: PythonObject, path: PythonObject) raises -> Bool:
    var b = Python.import_module("builtins")
    var key = path[0] if py_type(path) is b.tuple else path
    return py_contains(fields, key)


def _schema_value(fields: PythonObject, path: PythonObject, name: String) raises -> PythonObject:
    """The field a path names: an instance field, then keys and indices."""
    var b = Python.import_module("builtins")
    if py_type(path) is not b.tuple:
        return _schema_field(fields, path, name)
    var value = _schema_field(fields, path[0], name)
    for i in range(1, len(path)):  # driver-loop: over the fixed steps of one schema field path
        var container = py_type(value)
        if container is not b.dict and container is not b.tuple and container is not b.list:
            raise Error("TypeError: saved field path " + String(path[0]) + " requires native containers")
        value = value[path[i]]
    return value


def _schema_list(value: PythonObject, label: String) raises -> PythonObject:
    """A list or tuple field, refused otherwise (no Python iteration protocol runs)."""
    var b = Python.import_module("builtins")
    if py_type(value) is not b.list and py_type(value) is not b.tuple:
        raise Error("TypeError: saved field " + label + " must be a list")
    return value


def _schema_int(value: PythonObject, label: String) raises -> Int:
    """int(value) for the builtin numbers a fitted field holds (a float
    truncates toward zero, as int() does)."""
    var b = Python.import_module("builtins")
    if value is None:
        raise Error("TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType' (" + label + ")")
    if py_type(value) is b.float:
        return Int(Float64(py=value))
    return Int(py=value)


def _schema_truth(value: PythonObject) raises -> Bool:
    return Bool(py=value)


def _schema_label(path: PythonObject) raises -> String:
    var b = Python.import_module("builtins")
    return String(path[0]) if py_type(path) is b.tuple else String(path)


def _schema_real(fields: PythonObject, item: PythonObject, name: String,
                 context: PythonObject) raises -> Float64:
    var kind = String(item[0])
    if kind == "float":
        return Float64(py=_schema_value(fields, item[1], name))
    if kind == "float_or":
        if not _schema_has(fields, item[1]):
            return Float64(py=item[2])
        return Float64(py=_schema_value(fields, item[1], name))
    if kind == "const":
        return Float64(py=item[1])
    if kind == "if_eq":
        var chosen = item[3] if Bool(py=_schema_value(fields, item[1], name) == item[2]) else item[4]
        return _schema_real(fields, chosen, name, context)
    return Float64(_schema_whole(fields, item, name, context))


def _schema_whole(fields: PythonObject, item: PythonObject, name: String,
                  context: PythonObject) raises -> Int:
    var kind = String(item[0])
    if kind == "int":
        return _schema_int(_schema_value(fields, item[1], name), _schema_label(item[1]))
    if kind == "bool":
        return 1 if _schema_truth(_schema_value(fields, item[1], name)) else 0
    if kind == "const":
        return _schema_int(item[1], "constant")
    if kind == "dim":
        var shape = _schema_value(fields, item[1], name).shape
        return Int(py=shape[item[2]])
    if kind == "size":
        return Int(py=_schema_value(fields, item[1], name).size)
    if kind == "len":
        return len(_schema_value(fields, item[1], name))
    if kind == "none":
        var value = _schema_value(fields, item[1], name)
        return _schema_int(item[2], "sentinel") if value is None else _schema_int(value, _schema_label(item[1]))
    if kind == "noneflag":
        return 1 if _schema_value(fields, item[1], name) is None else 0
    if kind == "int_or":
        if not _schema_has(fields, item[1]):
            return _schema_int(item[2], "default")
        return _schema_int(_schema_value(fields, item[1], name), _schema_label(item[1]))
    if kind == "bool_or":
        if not _schema_has(fields, item[1]):
            return 1 if Bool(py=item[2]) else 0
        return 1 if _schema_truth(_schema_value(fields, item[1], name)) else 0
    if kind == "code":
        var key = _schema_value(fields, item[1], name)
        if not py_contains(item[2], key):
            raise Error("ValueError: mojolearn " + name + ": " + _schema_label(item[1]) + " has no saved code")
        return _schema_int(item[2][key], "code")
    if kind == "if_eq":
        var chosen = item[3] if Bool(py=_schema_value(fields, item[1], name) == item[2]) else item[4]
        return _schema_whole(fields, chosen, name, context)
    if kind == "sum":
        var elements = _schema_list(_schema_value(fields, item[1], name), _schema_label(item[1]))
        var total = 0
        for i in range(len(elements)):  # driver-loop: over the fitted sub-models of one list field
            total += _schema_int(elements[i][item[2]], String(item[2]))
        return total
    raise Error("RuntimeError: unknown saved item kind " + kind)


def _schema_vector(fields: PythonObject, dtype: String, items: PythonObject, name: String,
                   context: PythonObject) raises -> PythonObject:
    """One packed metadata vector, an Array of `dtype` as Array.from_list made it."""
    var count = len(items)
    var vector = _schema_zeros(dtype, count, context)
    var address = Int(py=classical_address(vector))
    for i in range(count):  # driver-loop: over the fixed elements of one saved metadata vector
        _schema_put(address, i, dtype, fields, items[i], name, context)
    return vector


def _schema_zeros(dtype: String, count: Int, context: PythonObject) raises -> PythonObject:
    """A zeroed C-order Array of `count` elements of `dtype`."""
    var store = py_call(context["array_type"], _store_code(dtype))
    var raw = py_call(Python.import_module("builtins").bytes, count * Int(py=store.itemsize))
    _ = py_call(store.frombytes, raw)
    return classical_wrap_array(store, Python.tuple(count), dtype, "C", context)


def _schema_put(address: Int, i: Int, dtype: String, fields: PythonObject, item: PythonObject,
                name: String, context: PythonObject) raises:
    """Element `i` of the packed vector at `address`: one item, as
    Array.from_list(..., dtype) stores it."""
    if dtype == "<i8":
        var i64 = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=address)
        i64[i] = Int64(_schema_whole(fields, item, name, context))
    elif dtype == "<i4":
        var i32 = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)
        var whole = _schema_whole(fields, item, name, context)
        if whole < -2147483648 or whole > 2147483647:
            raise Error("OverflowError: saved int32 field is out of range")
        i32[i] = Int32(whole)
    elif dtype == "<f8":
        var f64 = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=address)
        f64[i] = _schema_real(fields, item, name, context)
    elif dtype == "<f4":
        # Array.from_list(..., "<f4") stores through the C (float) cast:
        # one round-to-nearest of the binary64 value.
        var f32 = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)
        f32[i] = Float32(_schema_real(fields, item, name, context))
    else:
        raise Error("RuntimeError: unknown saved vector dtype " + dtype)


def _schema_rows(fields: PythonObject, dtype: String, path: PythonObject, items: PythonObject,
                 name: String, context: PythonObject) raises -> PythonObject:
    """The items of every element of a list field, element by element, as
    one packed vector (a "rows" step)."""
    var elements = _schema_list(_schema_value(fields, path, name), _schema_label(path))
    var width = len(items)
    var vector = _schema_zeros(dtype, len(elements) * width, context)
    var address = Int(py=classical_address(vector))
    for e in range(len(elements)):  # driver-loop: over the fitted sub-models of one list field
        var row = Python.dict()
        py_setitem(row, "_", elements[e])
        for k in range(width):  # driver-loop: over the fixed items of one saved row
            _schema_put(address, e * width + k, dtype, row, items[k], name, context)
    return vector


def _schema_concat(fields: PythonObject, dtype: String, path: PythonObject, key: PythonObject,
                   name: String, context: PythonObject) raises -> PythonObject:
    """element[key] of every element of a list field, C-order elements
    concatenated into one flat `dtype` vector (a "concat" step)."""
    var elements = _schema_list(_schema_value(fields, path, name), _schema_label(path))
    var joined = py_call(context["archive_bytearray_type"])
    for e in range(len(elements)):  # driver-loop: over the fitted sub-models of one list field
        var element = elements[e]
        var block = _schema_field(element, key, name) if py_type(element) is Python.import_module("builtins").dict else element[key]
        var parts = _schema_bytes(block, context)
        if String(parts[0]) != dtype:
            raise Error("TypeError: saved block " + String(key) + " holds " + String(parts[0]) + ", the format stores " + dtype)
        _ = py_call(joined.extend, parts[1])
    var width = Int(py=py_call(context["array_type"], _store_code(dtype)).itemsize)
    return _schema_copy(joined, dtype, 0, len(joined) // width, context)


def _schema_classes(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_labels.classes_member(classes)`: int64 when every label is an
    integer (bools excluded) or every label is a bool, float64 when every
    label is a real number, the list of str when every label is a str."""
    var b = Python.import_module("builtins")
    var numbers = Python.import_module("numbers")
    var labels = _schema_list(value, "classes_")
    var count = len(labels)
    var integral = True
    var real = True
    var flags = True
    var text = True
    for i in range(count):  # driver-loop: over the fitted distinct class labels
        var label = labels[i]
        var cls = py_type(label)
        var flag = cls is b.bool or String(py=cls.__name__) == "bool_"
        flags = flags and flag
        integral = integral and not flag and Bool(py=py_call(b.isinstance, label, numbers.Integral))
        real = real and not flag and Bool(py=py_call(b.isinstance, label, numbers.Real))
        text = text and Bool(py=py_call(b.isinstance, label, b.str))
    var whole = py_call(b.list)
    if integral or flags:
        for i in range(count):  # driver-loop: over the fitted distinct class labels
            _ = py_call(whole.append, py_call(b.int, labels[i]))
        return _schema_packed(whole, "<i8", context)
    if real:
        for i in range(count):  # driver-loop: over the fitted distinct class labels
            _ = py_call(whole.append, py_call(b.float, labels[i]))
        return _schema_packed(whole, "<f8", context)
    if text:
        return py_call(b.list, labels)
    raise Error("mojolearn: classes_ holds labels no model file can carry")


def _schema_packed(values: PythonObject, dtype: String, context: PythonObject) raises -> PythonObject:
    """A list of builtin numbers as Array.from_list(values, dtype) stores
    them (the array module's C conversion and range checks)."""
    var store = py_call(context["array_type"], _store_code(dtype), values)
    return classical_wrap_array(store, Python.tuple(len(values)), dtype, "C", context)


def _schema_bytes(value: PythonObject, context: PythonObject) raises -> Tuple[String, PythonObject]:
    """(typestr, C-order bytes) of a saved array field."""
    classical_admit_value(value, context)
    var array = value if classical_is_array(value, context) or py_type(value) is py_get(context, "ndarray") else classical_input(value, context)
    var typestr: String
    var raw: PythonObject
    if classical_is_array(array, context):
        var parts = _archive_array(array, context)
        typestr = String(parts[0])
        raw = parts[2]
    else:
        typestr = classical_dtype_str(array, context)
        raw = py_call(context["archive_bytes_type"]) if Int(py=array.nbytes) == 0 else classical_c_bytes(array, "model field", context)
    if typestr == "|u1":
        typestr = "<u1"
    return (typestr, raw)


def _schema_copy(raw: PythonObject, dtype: String, first: Int, count: Int, context: PythonObject) raises -> PythonObject:
    """A C-order Array of `count` elements of `raw` from element `first`."""
    var store = py_call(context["array_type"], _store_code(dtype))
    var width = Int(py=store.itemsize)
    var view = py_call(Python.import_module("builtins").memoryview, raw)
    var part = py_call(view.__getitem__, py_call(Python.import_module("builtins").slice, first * width, (first + count) * width))
    _ = py_call(store.frombytes, part)
    return classical_wrap_array(store, Python.tuple(count), dtype, "C", context)


def _schema_text_member(value: PythonObject, lower: Bool) raises -> PythonObject:
    """str(value) for a builtin scalar; refuses a value whose str would run
    Python code."""
    var b = Python.import_module("builtins")
    var cls = py_type(value)
    var text = PythonObject(None)
    if cls is b.str:
        text = value
    elif value is None or cls is b.bool or cls is b.int or cls is b.float:
        text = PythonObject(_archive_builtin_repr(value))
    else:
        raise Error("TypeError: a saved name field must be a str")
    return py_call(text.lower) if lower else text


def _schema_mode(fields: PythonObject, prefix: PythonObject, context: PythonObject) raises -> PythonObject:
    """The tier `predict` would run on now, persisted so a loaded model
    never silently changes tier (`numeric_mode` or the process default)."""
    var b = Python.import_module("builtins")
    var mode = py_get(fields, "numeric_mode")
    if mode is None or not Bool(py=mode):
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not b.str:
        var shown = _archive_builtin_repr(mode) if mode is None or py_type(mode) is b.bool or py_type(mode) is b.int or py_type(mode) is b.float else String("<" + String(py_type(mode).__name__) + ">")
        raise Error("ValueError: " + String(prefix) + "cannot save invalid numeric_mode " + shown)
    var folded = py_call(py_call(mode.strip).lower)
    if String(folded) != "fast" and String(folded) != "identical":
        raise Error("ValueError: " + String(prefix) + "cannot save invalid numeric_mode " + _archive_builtin_repr(mode))
    return folded


def _schema_run(model: PythonObject, fields: PythonObject, steps: PythonObject, values: PythonObject,
                name: String, context: PythonObject) raises:
    var b = Python.import_module("builtins")
    for k in range(len(steps)):  # driver-loop: over the fixed save steps of one class
        var step = steps[k]
        var kind = String(step[0])
        if kind == "need":
            if not _schema_has(fields, step[1]):
                _schema_fail(step[2], step[3], name)
        elif kind == "need_set":
            if not _schema_has(fields, step[1]) or _schema_value(fields, step[1], name) is None:
                _schema_fail(step[2], step[3], name)
        elif kind == "need_true":
            if not _schema_has(fields, step[1]) or not _schema_truth(_schema_value(fields, step[1], name)):
                _schema_fail(step[2], step[3], name)
        elif kind == "need_str":
            if not _schema_has(fields, step[1]) or py_type(_schema_value(fields, step[1], name)) is not b.str:
                _schema_fail(step[2], step[3], name)
        elif kind == "refuse_set":
            if _schema_has(fields, step[1]) and _schema_value(fields, step[1], name) is not None:
                _schema_fail(step[2], step[3], name)
        elif kind == "refuse_in":
            if _schema_has(fields, step[1]) and py_contains(step[2], _schema_value(fields, step[1], name)):
                _schema_fail(step[3], step[4], name)
        elif kind == "format":
            py_setitem(values, step[1], step[2])
        elif kind == "estimator":
            py_setitem(values, step[1], PythonObject(name))
        elif kind == "mode":
            py_setitem(values, step[1], _schema_mode(fields, step[2], context))
        elif kind == "str":
            py_setitem(values, step[1], _schema_text_member(_schema_value(fields, step[2], name), False))
        elif kind == "lower":
            py_setitem(values, step[1], _schema_text_member(_schema_value(fields, step[2], name), True))
        elif kind == "init_name":
            var value = _schema_value(fields, step[2], name)
            py_setitem(values, step[1], value if py_type(value) is b.str else step[3])
        elif kind == "array":
            py_setitem(values, step[1], _schema_value(fields, step[2], name))
        elif kind == "flat":
            var parts = _schema_bytes(_schema_value(fields, step[2], name), context)
            var width = Int(py=py_call(context["array_type"], _store_code(parts[0])).itemsize)
            py_setitem(values, step[1], _schema_copy(parts[1], parts[0], 0, len(parts[1]) // width, context))
        elif kind == "head":
            var parts = _schema_bytes(_schema_value(fields, step[2], name), context)
            var width = Int(py=py_call(context["array_type"], _store_code(parts[0])).itemsize)
            var total = len(parts[1]) // width
            var count = max(_schema_whole(fields, Python.tuple("int", step[3]), name, context), 1)
            py_setitem(values, step[1], _schema_copy(parts[1], parts[0], 0, min(count, total), context))
        elif kind == "i8" or kind == "i4" or kind == "f8" or kind == "f4":
            var dtype = String("<") + kind
            py_setitem(values, step[1], _schema_vector(fields, dtype, step[2], name, context))
        elif kind == "when":
            if _schema_has(fields, step[1]) and Bool(py=_schema_value(fields, step[1], name) == step[2]):
                _schema_run(model, fields, step[3], values, name, context)
        elif kind == "when_in":
            if _schema_has(fields, step[1]) and py_contains(step[2], _schema_value(fields, step[1], name)):
                _schema_run(model, fields, step[3], values, name, context)
        elif kind == "when_set":
            if _schema_has(fields, step[1]) and _schema_value(fields, step[1], name) is not None:
                _schema_run(model, fields, step[2], values, name, context)
        elif kind == "when_none":
            if not _schema_has(fields, step[1]) or _schema_value(fields, step[1], name) is None:
                _schema_run(model, fields, step[2], values, name, context)
        elif kind == "classes":
            py_setitem(values, step[1], _schema_classes(_schema_value(fields, step[2], name), context))
        elif kind == "rows":
            py_setitem(values, step[1], _schema_rows(fields, String(step[2]), step[3], step[4], name, context))
        elif kind == "concat":
            py_setitem(values, step[1], _schema_concat(fields, String(step[2]), step[3], step[4], name, context))
        else:
            raise Error("RuntimeError: unknown save step " + kind)


def archive_schema_save(model: PythonObject, path: PythonObject, context: PythonObject) raises -> PythonObject:
    """`model.save(path)` for a class with a connected save schema."""
    var cls = py_type(model)
    var steps = archive_schema_lookup(cls, context["archive_save_schemas"])
    if steps is None:
        raise Error("TypeError: " + String(cls.__name__) + " has no connected save schema")
    # A fitted field may be an ndarray when the caller loaded NumPy.
    classical_connect_numpy(context)
    var fields = py_fields(model)
    var values = Python.dict()
    _schema_run(model, fields, steps, values, String(cls.__name__), context)
    return classical_archive_write(path, values, context)
