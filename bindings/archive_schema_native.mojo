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
#   ("need_eq", field, value, exc, msg, shown)
#                                      raise exc(msg) unless field == value
#   ("need_in", field, values, exc, msg, shown)
#                                      raise exc(msg) unless field is in values
#                                      (msg's {0}, {1}, ... are repr() of the
#                                      fields named in the tuple `shown`)
#   ("choice", field, key, names, saved, fallback, exc, name_msg, code_msg)
#                                      a (name, code) pair under the derived
#                                      field `key` (read later through the
#                                      path (key, 0) or (key, 1)): a str field
#                                      is its own name, refused by name_msg
#                                      unless it is in the table `names`
#                                      (name -> code); with a fallback (name,
#                                      code) pair, a callable, a bool or a
#                                      non-int field takes the fallback;
#                                      otherwise code = int(field), refused by
#                                      code_msg unless it is in the table
#                                      `saved` (code -> name). Messages carry
#                                      {value} (repr of the field) and {code}.
#   ("check_array", field, dtype, items, exc, msg)
#                                      raise exc(msg) unless the field's array has
#                                      typestr dtype and the shape the int items
#                                      give; msg may name {want} (that shape),
#                                      {dtype} and {shape} (the field's own)
#   ("format", member, text)           a constant text member
#   ("estimator", member)              the class name
#   ("mode", member, prefix)           the numeric mode a predict would use now
#   ("str", member, field)             str(field) for a builtin scalar field
#   ("lower", member, field)           str(field).lower()
#   ("init_name", member, field, alt)  field when it is a str, else alt
#   ("array", member, field)           the field's array, as stored
#   ("flat", member, field)            the field's array in C order, shape (size,)
#   ("head", member, field, count)     the first max(count, 1) elements of a 1-D array
#   ("reshape", member, field, items)  the field's array in C order with the shape
#                                      the items give, one per dimension (whole
#                                      items); a size mismatch raises Array.reshape's
#                                      ValueError
#   ("i8" | "i4" | "f8" | "f4", member, items)
#                                      one packed vector, an item per element
#   ("when", field, value, steps)      run steps when field == value
#   ("when_set", field, steps)         run steps when the instance holds field and
#                                      field is not None (no == on an array)
#   ("unless", field, value, steps)    run steps when field is held and field != value
#   ("classes", member, field)         when the instance holds field: the class
#                                      labels as `_labels.classes_member` stored
#                                      them (int64 Array for integer or bool
#                                      labels, float64 Array for real labels,
#                                      the list of str for str labels)
#   ("utf8", member, field)            str(field) for a builtin scalar field,
#                                      as its UTF-8 bytes in a `<u1` Array
#   ("forest_engine", format)          the 0.8.55 `ForestProtocol._archive_inference_metadata`:
#                                      a model predicting with the
#                                      'parallel_groves' engine writes
#                                      format + "-parallel-groves-1" and its
#                                      effective numeric_mode
#
# An item is ("int", field), ("bool", field), ("const", number),
# ("dim", field, axis), ("size", field), ("len", field), ("none", field, sentinel),
# ("noneflag", field), ("setflag", field), ("float", field), ("float_or", field, default),
# ("none_real", field, sentinel) (sentinel when field is None, else float(field)),
# ("int_or", field, default), ("bool_or", field, default), ("code", field, table)
# ("if_eq", field, value, item, item), ("strict", field, exc, msg),
# ("none_or", field, sentinel, item), ("at_least", item, low, exc, msg) or
# ("mod", item, modulus). "strict" is the field as an int, refused with
# exc(msg) unless it is an int and not a bool ({type} in msg names the held
# type); "none_or" is sentinel when field is None, else item; "at_least" is
# item, refused with exc(msg) below low; "mod" is item % modulus (floored,
# as Python's %).
#   ("class_table", member, field, integral, real)
#                                      a classifier's class table as
#                                      `_labels.classes_member` wrote it: <i8
#                                      for integer (or all-bool) labels, <f8
#                                      for real labels, a list of str (a <U
#                                      member) for str labels; `integral` and
#                                      `real` are numbers.Integral and
#                                      numbers.Real, fixed at register
#   ("kernel", field, table, dtype, kinds, params, ls_len, ls)
#                                      a Gaussian process kernel's postfix node
#                                      list (`Kernel._nodes()`) as four members:
#                                      kinds <i4, params `dtype`, ls_len <i4 and
#                                      the length-scale table `dtype` (one 1.0
#                                      when empty). `table` maps each kernel
#                                      class to its node layout: ("leaf", kind,
#                                      param field or None, scales field or
#                                      None), ("pair", left, right, op field) or
#                                      ("nodes", saved node list field)
#   ("stack", member, field, attr, dtype, item)
#                                      the storage bytes of `attr` of every
#                                      element of the list `field`, joined as a
#                                      (len, item) array of `dtype`
#   ("each", member, field, attr, dtype)
#                                      `attr` of every element of the list
#                                      `field`, as an <i8 or <f8 vector
#
# Two more items: ("mul", item, item) and ("kernel_ls", field, table), the
# length-scale count of a "kernel" step's kernel. A field is an instance-dictionary
#   ("when_in", field, values, steps)  run steps when field is held and equals one of values
#   ("when_none", field, steps)        run steps when field is absent or None
#   ("rows", member, dtype, field, items)
#                                      one packed vector of dtype ("<i8", "<i4",
#                                      "<f8", "<f4"): for each element of the list
#                                      field, its items in order; an item's field
#                                      paths start at "_", the element
#   ("concat", member, dtype, field, key)
#                                      the C-order elements of element[key] for each
#                                      element of the list field, in order, as one flat
#                                      dtype vector (4-byte dtypes)
# One more item: ("sum", field, key), the int sum of element[key] over the
# list field.
# name, or a tuple path: a name, then dictionary keys and sequence indices.
# ("noneflag", field) is 1 when field is None, ("setflag", field) 1 when it
# is not. Messages may name the class as {cls}.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains, py_fields
from bindings.classical_arrays import classical_address, classical_wrap_array, classical_is_array, classical_input, classical_dtype_str, classical_c_bytes, _store_code
from bindings.classical_pandas import classical_admit_value
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_archive import classical_archive_write, _archive_array
from bindings.classical_archive_read import _archive_builtin_repr
from bindings.trees_public_iforest import iforest_validate


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


def _schema_shown(fields: PythonObject, msg: PythonObject, shown: PythonObject, name: String) raises -> PythonObject:
    """`msg` with {0}, {1}, ... replaced by repr() of the fields `shown` names."""
    var b = Python.import_module("builtins")
    var text = String(msg)
    for i in range(len(shown)):  # driver-loop: over the fixed fields one refusal message names
        var shown_value = _schema_value(fields, shown[i], name)
        text = text.replace("{" + String(i) + "}", String(py_call(b.repr, shown_value)))
    return PythonObject(text)


def _schema_choice(fields: PythonObject, step: PythonObject, name: String) raises:
    """The ("choice", ...) step: stores (name, code) under the derived field."""
    var b = Python.import_module("builtins")
    var value = _schema_value(fields, step[1], name)
    var names = step[3]
    var saved = step[4]
    var fallback = step[5]
    var shown = String(py_call(b.repr, value))
    if py_type(value) is b.str:
        if not py_contains(names, value):
            _schema_fail(step[6], PythonObject(String(step[7]).replace("{value}", shown)), name)
        py_setitem(fields, step[2], Python.tuple(value, names[value]))
        return
    if fallback is not None:
        if Bool(py=py_call(b.callable, value)) or py_type(value) is b.bool or not Bool(py=py_call(b.isinstance, value, b.int)):
            py_setitem(fields, step[2], fallback)
            return
    var code = py_call(b.int, value)
    if not py_contains(saved, code):
        var text = String(step[8]).replace("{value}", shown).replace("{code}", String(py_call(b.repr, code)))
        _schema_fail(step[6], PythonObject(text), name)
    py_setitem(fields, step[2], Python.tuple(saved[code], code))
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
    if kind == "none_real":
        var value = _schema_value(fields, item[1], name)
        return Float64(py=item[2]) if value is None else Float64(py=value)
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
    if kind == "setflag":
        return 0 if _schema_value(fields, item[1], name) is None else 1
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
    if kind == "strict":
        # An int and not a bool, as `isinstance(v, int)` after the bool
        # refusal (bool has no subclasses, so the type test is exact).
        var b = Python.import_module("builtins")
        var value = _schema_value(fields, item[1], name)
        if py_type(value) is b.bool or not Bool(py=py_call(b.isinstance, value, b.int)):
            var held = String(py_type(value).__name__)
            raise Error(String(item[2]) + ": " + String(item[3]).replace("{cls}", name).replace("{type}", held))
        return Int(py=value)
    if kind == "none_or":
        if _schema_value(fields, item[1], name) is None:
            return _schema_int(item[2], "sentinel")
        return _schema_whole(fields, item[3], name, context)
    if kind == "at_least":
        var whole = _schema_whole(fields, item[1], name, context)
        if whole < _schema_int(item[2], "bound"):
            raise Error(String(item[3]) + ": " + String(item[4]).replace("{cls}", name))
        return whole
    if kind == "mod":
        var whole = _schema_whole(fields, item[1], name, context)
        var modulus = _schema_int(item[2], "modulus")
        if modulus <= 0:
            raise Error("RuntimeError: a saved modulus must be positive")
        var rest = whole % modulus
        if rest < 0:
            rest += modulus
        return rest
    if kind == "mul":
        return _schema_whole(fields, item[1], name, context) * _schema_whole(fields, item[2], name, context)
    if kind == "kernel_ls":
        var nodes = Python.list()
        _schema_kernel_nodes(_schema_value(fields, item[1], name), item[2], nodes)
        var total = 0
        for i in range(len(nodes)):  # driver-loop: over the kernel's postfix nodes
            total += len(nodes[i][2])
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
    var builtins = Python.import_module("builtins")
    var getitem = py_call(builtins.getattr, view, PythonObject("__getitem__"))
    var part = py_call(getitem, py_call(builtins.slice, first * width, (first + count) * width))
    _ = py_call(store.frombytes, part)
    return classical_wrap_array(store, Python.tuple(count), dtype, "C", context)


def _schema_shaped(raw: PythonObject, dtype: String, shape: PythonObject, context: PythonObject) raises -> PythonObject:
    """A C-order Array of `shape` holding all of `raw`."""
    var store = py_call(context["array_type"], _store_code(dtype))
    _ = py_call(store.frombytes, raw)
    return classical_wrap_array(store, shape, dtype, "C", context)


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


def _schema_repr_shown(value: PythonObject) raises -> String:
    """repr(value) for a builtin scalar; `<type>` for anything whose repr
    would run Python code."""
    var b = Python.import_module("builtins")
    var cls = py_type(value)
    if value is None or cls is b.str or cls is b.bool or cls is b.int or cls is b.float:
        return _archive_builtin_repr(value)
    return "<" + String(cls.__name__) + ">"


def _schema_is_bool(value: PythonObject) raises -> Bool:
    """`_labels.is_bool`: Python's bool or NumPy's bool_."""
    var cls = py_type(value)
    return cls is Python.import_module("builtins").bool or String(cls.__name__) == "bool_"


def _schema_classes(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_labels.classes_member(value)`: the saved form of `classes_`."""
    var b = Python.import_module("builtins")
    var numbers = Python.import_module("numbers")
    var count = len(value)
    var labels = Python.list()
    var all_int = True
    var all_real = True
    var all_bool = True
    var all_str = True
    for i in range(count):  # driver-loop: over the distinct class labels of one fitted model
        var label = value[i]
        _ = py_call(labels.append, label)
        var boolean = _schema_is_bool(label)
        all_bool = all_bool and boolean
        all_int = all_int and not boolean and Bool(py=py_call(b.isinstance, label, numbers.Integral))
        all_real = all_real and not boolean and Bool(py=py_call(b.isinstance, label, numbers.Real))
        all_str = all_str and Bool(py=py_call(b.isinstance, label, b.str))
    if all_int or all_bool or all_real:
        var items = Python.list()
        for i in range(count):  # driver-loop: over the distinct class labels of one fitted model
            _ = py_call(items.append, Python.tuple("const", labels[i]))
        return _schema_vector(Python.dict(), String("<i8") if all_int or all_bool else String("<f8"), items, name, context)
    if all_str:
        return labels
    raise Error("ValueError: mojolearn: classes_ holds labels no model file can carry")


def _schema_valid_mode(mode: PythonObject) raises -> PythonObject:
    """`ForestProtocol._validated_mode`."""
    var b = Python.import_module("builtins")
    if mode is None:
        return mode
    if not Bool(py=py_call(b.isinstance, mode, b.str)):
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    var folded = py_call(b.str.lower, py_call(b.str.strip, mode))
    if String(py=folded) != "fast" and String(py=folded) != "identical":
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    return folded


def _schema_forest_mode(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """`ForestProtocol._effective_mode`: the mode captured at fit, else the
    requested one, else the process default."""
    var requested = _schema_valid_mode(py_get(fields, "numeric_mode"))
    var captured = py_get(fields, "_fit_numeric_mode")
    if captured is not None:
        if requested is not None and not Bool(py=requested == captured):
            raise Error("ValueError: This forest was fitted with numeric_mode=" + _schema_repr_shown(captured)
                        + "; refit or use set_params before changing numeric_mode")
        return captured
    if requested is not None:
        return requested
    return context["backend_fields"]["_DEFAULT_MODE"]


def _schema_forest_engine(fields: PythonObject, context: PythonObject) raises -> String:
    """`ForestProtocol._prediction_engine`."""
    var b = Python.import_module("builtins")
    var engine = py_get(fields, "inference_engine", PythonObject("sequential"))
    var valid = Bool(py=py_call(b.isinstance, engine, b.str))
    # str.__str__ (not str()) so a str subclass's own __str__ cannot change the text.
    var str_str = py_call(b.getattr, b.str, PythonObject("__str__"))
    var text = String(py=py_call(str_str, engine)) if valid else String("")
    if not valid or (text != "auto" and text != "sequential" and text != "parallel_groves"):
        raise Error("ValueError: inference_engine must be 'auto', 'sequential' or 'parallel_groves'")
    if text == "auto":
        return String("parallel_groves") if Bool(py=_schema_forest_mode(fields, context) == PythonObject("fast")) else String("sequential")
    return text
def _schema_label_kind(label: PythonObject, integral: PythonObject, real: PythonObject,
                       context: PythonObject) raises -> Int:
    """0 bool (`_labels.is_bool`), 1 integer, 2 real, 3 str, 4 another label.
    Builtin and NumPy scalars are told by type; any other label by the
    numbers ABCs, as `_labels.label_kind` did."""
    var b = Python.import_module("builtins")
    var cls = py_type(label)
    if cls is b.bool:
        return 0
    if cls is b.int:
        return 1
    if cls is b.float:
        return 2
    if cls is b.str:
        return 3
    if String(cls.__name__) == "bool_":
        return 0
    var scalars = py_get(context, "scalar_types")
    if scalars is not None and py_contains(scalars, cls):
        var kind = String(label.dtype.kind)
        if kind == "i" or kind == "u":
            return 1
        if kind == "f":
            return 2
    if Bool(py=py_call(b.isinstance, label, b.str)):
        return 3
    if Bool(py=py_call(b.isinstance, label, integral)):
        return 1
    if Bool(py=py_call(b.isinstance, label, real)):
        return 2
    return 4


def _schema_label_store(values: PythonObject, code: String, dtype: String, context: PythonObject) raises -> PythonObject:
    """`Array.from_list(values, dtype)` for a flat list of builtin numbers,
    with its refusal of a value the dtype cannot hold."""
    var store = PythonObject(None)
    try:
        store = py_call(context["array_type"], code, values)
    except error:
        raise Error("TypeError: mojolearn: cannot build a " + dtype + " Array from these values: " + String(error))
    return classical_wrap_array(store, Python.tuple(len(values)), dtype, "C", context)


def _schema_class_table(value: PythonObject, integral: PythonObject, real: PythonObject,
                    context: PythonObject) raises -> PythonObject:
    """`_labels.classes_member(value)`: int64 when every label is an integer
    (bools excluded) or every label is a bool, float64 when every label is a
    real number (bools excluded), the list of str when every label is a str."""
    var b = Python.import_module("builtins")
    var labels = py_call(b.list, value)
    var count = len(labels)
    var integers = True
    var reals = True
    var bools = True
    var texts = True
    for i in range(count):  # driver-loop: over the distinct class labels' kinds
        var kind = _schema_label_kind(labels[i], integral, real, context)
        bools = bools and kind == 0
        integers = integers and kind == 1
        reals = reals and (kind == 1 or kind == 2)
        texts = texts and kind == 3
    var packed = Python.list()
    if integers or bools:
        for i in range(count):  # driver-loop: over the distinct class labels
            _ = py_call(packed.append, py_call(b.int, labels[i]))
        return _schema_label_store(packed, "q", "<i8", context)
    if reals:
        for i in range(count):  # driver-loop: over the distinct class labels
            _ = py_call(packed.append, py_call(b.float, labels[i]))
        return _schema_label_store(packed, "d", "<f8", context)
    if texts:
        # str.__str__ gives the exact str of a str subclass without its hooks.
        var exact = b.str.__dict__["__str__"]
        for i in range(count):  # driver-loop: over the distinct class labels
            _ = py_call(packed.append, py_call(exact, labels[i]))
        return packed
    raise Error("ValueError: mojolearn: classes_ holds labels no model file can carry")


def _schema_kernel_layout(kernel: PythonObject, table: PythonObject) raises -> PythonObject:
    """The node layout of `kernel`'s class or its nearest connected ancestor;
    a class that defines its own `_nodes` first has none."""
    var b = Python.import_module("builtins")
    var cls = py_type(kernel)
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    for i in range(len(mro)):  # driver-loop: over the kernel class ancestry metadata
        if py_contains(table, mro[i]):
            return table[mro[i]]
        if py_contains(py_call(b.type.__dict__["__dict__"].__get__, mro[i]), "_nodes"):
            break
    raise Error("TypeError: mojolearn: a " + String(cls.__name__) + " kernel has no saved node layout")


def _schema_kernel_nodes(kernel: PythonObject, table: PythonObject, nodes: PythonObject) raises:
    """Append `kernel._nodes()`: postfix (kind, param, length scales) tuples,
    read from the kernels' instance fields."""
    var b = Python.import_module("builtins")
    var layout = _schema_kernel_layout(kernel, table)
    var fields = py_fields(kernel)
    var name = String(py_type(kernel).__name__)
    var how = String(layout[0])
    if how == "pair":
        _schema_kernel_nodes(_schema_field(fields, layout[1], name), table, nodes)
        _schema_kernel_nodes(_schema_field(fields, layout[2], name), table, nodes)
        _ = py_call(nodes.append, Python.tuple(_schema_field(fields, layout[3], name), PythonObject(Float64(0.0)), Python.list()))
    elif how == "leaf":
        var param = PythonObject(Float64(0.0)) if layout[2] is None else _schema_field(fields, layout[2], name)
        var scales = Python.list() if layout[3] is None else py_call(b.list, _schema_field(fields, layout[3], name))
        _ = py_call(nodes.append, Python.tuple(layout[1], param, scales))
    elif how == "nodes":
        var saved = _schema_field(fields, layout[1], name)
        for i in range(len(saved)):  # driver-loop: over a saved kernel's postfix nodes
            var node = saved[i]
            _ = py_call(nodes.append, Python.tuple(node[0], node[1], py_call(b.list, node[2])))
    else:
        raise Error("RuntimeError: unknown kernel node layout " + how)


def _schema_kernel(model_fields: PythonObject, step: PythonObject, values: PythonObject, name: String,
                   context: PythonObject) raises:
    """The four kernel members of a "kernel" step, as `_kernel_arrays` and
    GaussianProcessRegressor.save packed them with Array.from_list."""
    var b = Python.import_module("builtins")
    var nodes = Python.list()
    _schema_kernel_nodes(_schema_value(model_fields, step[1], name), step[2], nodes)
    var dtype = String(step[3])
    var code = _store_code(dtype)
    var kinds = Python.list()
    var params = Python.list()
    var lengths = Python.list()
    var scales = Python.list()
    for i in range(len(nodes)):  # driver-loop: over the kernel's postfix nodes
        var node = nodes[i]
        _ = py_call(kinds.append, py_call(b.int, node[0]))
        _ = py_call(params.append, py_call(b.float, node[1]))
        _ = py_call(lengths.append, len(node[2]))
        for j in range(len(node[2])):  # driver-loop: over one node's length scales
            _ = py_call(scales.append, py_call(b.float, node[2][j]))
    if len(scales) == 0:
        _ = py_call(scales.append, PythonObject(Float64(1.0)))
    py_setitem(values, step[4], _schema_label_store(kinds, "i", "<i4", context))
    py_setitem(values, step[5], _schema_label_store(params, code, dtype, context))
    py_setitem(values, step[6], _schema_label_store(lengths, "i", "<i4", context))
    py_setitem(values, step[7], _schema_label_store(scales, code, dtype, context))


def _schema_elements(fields: PythonObject, field: PythonObject, attr: PythonObject,
                     name: String) raises -> PythonObject:
    """`attr` of every element of the list field `field`, from each
    element's instance dictionary."""
    var b = Python.import_module("builtins")
    var items = py_call(b.list, _schema_value(fields, field, name))
    var result = Python.list()
    for i in range(len(items)):  # driver-loop: over a fitted model's sub-model list
        var own = py_fields(items[i])
        _ = py_call(result.append, _schema_field(own, attr, String(py_type(items[i]).__name__)))
    return result


def _schema_stack(fields: PythonObject, step: PythonObject, name: String,
                  context: PythonObject) raises -> PythonObject:
    """`_gpc_impl._stacked`: the elements' storage bytes joined and read as
    one C-order (count, width) array of `dtype`."""
    var b = Python.import_module("builtins")
    var parts = _schema_elements(fields, step[2], step[3], name)
    var dtype = String(step[4])
    var width = _schema_whole(fields, step[5], name, context)
    var joined = py_call(b.bytearray)
    for i in range(len(parts)):  # driver-loop: over a fitted model's sub-model list
        var part = parts[i]
        if classical_is_array(part, context):
            var view = py_call(context["array_fields"]["_mv"], part, context["array_class"])
            _ = py_call(joined.extend, py_call(view.cast, "B"))
        else:
            _ = py_call(joined.extend, _schema_bytes(part, context)[1])
    var store = py_call(context["array_type"], _store_code(dtype))
    var shape = Python.tuple(len(parts), width)
    var need = len(parts) * width * Int(py=store.itemsize)
    if len(joined) != need:
        raise Error("ValueError: mojolearn: " + String(len(joined)) + " bytes do not fill shape "
                    + _archive_builtin_repr(shape) + " of " + dtype + " (" + String(need) + " bytes)")
    _ = py_call(store.frombytes, joined)
    return classical_wrap_array(store, shape, dtype, "C", context)


def _schema_each(fields: PythonObject, step: PythonObject, name: String,
                 context: PythonObject) raises -> PythonObject:
    """`Array.from_list([int|float(e.attr) for e in field], dtype)`."""
    var b = Python.import_module("builtins")
    var parts = _schema_elements(fields, step[2], step[3], name)
    var dtype = String(step[4])
    var packed = Python.list()
    for i in range(len(parts)):  # driver-loop: over a fitted model's sub-model list
        _ = py_call(packed.append, py_call(b.float if dtype == "<f8" else b.int, parts[i]))
    return _schema_label_store(packed, _store_code(dtype), dtype, context)


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
        elif kind == "need_eq":
            if not Bool(py=_schema_value(fields, step[1], name) == step[2]):
                _schema_fail(step[3], _schema_shown(fields, step[4], step[5], name), name)
        elif kind == "need_in":
            if not py_contains(step[2], _schema_value(fields, step[1], name)):
                _schema_fail(step[3], _schema_shown(fields, step[4], step[5], name), name)
        elif kind == "choice":
            _schema_choice(fields, step, name)
        elif kind == "check_array":
            var value = _schema_value(fields, step[1], name)
            var parts = _schema_bytes(value, context)
            var want = Python.list()
            for j in range(len(step[3])):  # driver-loop: over the fixed dimensions of one saved array check
                _ = py_call(want.append, _schema_whole(fields, step[3][j], name, context))
            var wanted = py_call(b.tuple, want)
            var got = py_call(b.tuple, value.shape)
            if parts[0] != String(step[2]) or not Bool(py=got == wanted):
                var text = String(step[5]).replace("{want}", _archive_builtin_repr(wanted))
                text = text.replace("{dtype}", parts[0]).replace("{shape}", _archive_builtin_repr(got))
                _schema_fail(step[4], PythonObject(text), name)
        elif kind == "iforest_knobs":
            # IsolationForest.save on an instance assembled without fit or
            # load: `_validate_params()` resolves the knobs first.
            if not _schema_has(fields, "_seed"):
                iforest_validate(model, fields, context)
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
        elif kind == "reshape":
            var parts = _schema_bytes(_schema_value(fields, step[2], name), context)
            var width = Int(py=py_call(context["array_type"], _store_code(parts[0])).itemsize)
            var total = len(parts[1]) // width
            var dims = Python.list()
            var count = 1
            for i in range(len(step[3])):  # driver-loop: over the fixed dimension items of one saved shape
                var dim = _schema_whole(fields, step[3][i], name, context)
                _ = py_call(dims.append, dim)
                count *= dim
            var shape = py_call(b.tuple, dims)
            if count != total:
                raise Error("ValueError: mojolearn: cannot reshape size " + String(total) + " into " + _archive_builtin_repr(shape))
            py_setitem(values, step[1], _schema_shaped(parts[1], parts[0], shape, context))
        elif kind == "i8" or kind == "i4" or kind == "f8" or kind == "f4":
            var dtype = String("<") + kind
            py_setitem(values, step[1], _schema_vector(fields, dtype, step[2], name, context))
        elif kind == "classes":
            if _schema_has(fields, step[2]):
                py_setitem(values, step[1], _schema_classes(_schema_value(fields, step[2], name), name, context))
        elif kind == "utf8":
            var text = _schema_text_member(_schema_value(fields, step[2], name), False)
            var raw = py_call(b.str.encode, text, "utf-8")
            py_setitem(values, step[1], _schema_copy(raw, "<u1", 0, len(raw), context))
        elif kind == "forest_engine":
            if _schema_forest_engine(fields, context) == "parallel_groves":
                py_setitem(values, "format", PythonObject(String(step[1]) + "-parallel-groves-1"))
                py_setitem(values, "numeric_mode", _schema_forest_mode(fields, context))
        elif kind == "class_table":
            py_setitem(values, step[1], _schema_class_table(_schema_value(fields, step[2], name), step[3], step[4], context))
        elif kind == "kernel":
            _schema_kernel(fields, step, values, name, context)
        elif kind == "stack":
            py_setitem(values, step[1], _schema_stack(fields, step, name, context))
        elif kind == "each":
            py_setitem(values, step[1], _schema_each(fields, step, name, context))
        elif kind == "when":
            if _schema_has(fields, step[1]) and Bool(py=_schema_value(fields, step[1], name) == step[2]):
                _schema_run(model, fields, step[3], values, name, context)
        elif kind == "when_set":
            if _schema_has(fields, step[1]) and _schema_value(fields, step[1], name) is not None:
                _schema_run(model, fields, step[2], values, name, context)
        elif kind == "unless":
            if _schema_has(fields, step[1]) and not Bool(py=_schema_value(fields, step[1], name) == step[2]):
                _schema_run(model, fields, step[3], values, name, context)
        elif kind == "when_in":
            if _schema_has(fields, step[1]) and py_contains(step[2], _schema_value(fields, step[1], name)):
                _schema_run(model, fields, step[3], values, name, context)
        elif kind == "when_none":
            if not _schema_has(fields, step[1]) or _schema_value(fields, step[1], name) is None:
                _schema_run(model, fields, step[2], values, name, context)
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
    # A shallow copy: "choice" steps add derived fields, never to the model.
    var fields = py_call(Python.import_module("builtins").dict, py_fields(model))
    var values = Python.dict()
    _schema_run(model, fields, steps, values, String(cls.__name__), context)
    return classical_archive_write(path, values, context)
