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
#   ("classes", member, field, integral, real)
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
# An item is ("int", field), ("bool", field), ("const", number),
# ("dim", field, axis), ("size", field), ("len", field), ("none", field, sentinel),
# ("noneflag", field), ("float", field), ("float_or", field, default),
# ("int_or", field, default), ("bool_or", field, default), ("code", field, table),
# ("if_eq", field, value, item, item), ("mul", item, item) or
# ("kernel_ls", field, table), the length-scale count of a "kernel" step's kernel. A field is an instance-dictionary
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
    if kind == "mul":
        return _schema_whole(fields, item[1], name, context) * _schema_whole(fields, item[2], name, context)
    if kind == "kernel_ls":
        var nodes = Python.list()
        _schema_kernel_nodes(_schema_value(fields, item[1], name), item[2], nodes)
        var total = 0
        for i in range(len(nodes)):  # driver-loop: over the kernel's postfix nodes
            total += len(nodes[i][2])
        return total
    raise Error("RuntimeError: unknown saved item kind " + kind)


def _schema_vector(fields: PythonObject, dtype: String, items: PythonObject, name: String,
                   context: PythonObject) raises -> PythonObject:
    """One packed metadata vector, an Array of `dtype` as Array.from_list made it."""
    var count = len(items)
    var store = py_call(context["array_type"], _store_code(dtype))
    var raw = py_call(Python.import_module("builtins").bytes, count * Int(py=store.itemsize))
    _ = py_call(store.frombytes, raw)
    var vector = classical_wrap_array(store, Python.tuple(count), dtype, "C", context)
    var address = Int(py=classical_address(vector))
    var i64 = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=address)
    var i32 = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)
    var f64 = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=address)
    var f32 = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(count):  # driver-loop: over the fixed elements of one saved metadata vector
        var item = items[i]
        if dtype == "<i8":
            i64[i] = Int64(_schema_whole(fields, item, name, context))
        elif dtype == "<i4":
            var whole = _schema_whole(fields, item, name, context)
            if whole < -2147483648 or whole > 2147483647:
                raise Error("OverflowError: saved int32 field is out of range")
            i32[i] = Int32(whole)
        elif dtype == "<f8":
            f64[i] = _schema_real(fields, item, name, context)
        elif dtype == "<f4":
            # Array.from_list(..., "<f4") stores through the C (float) cast:
            # one round-to-nearest of the binary64 value.
            f32[i] = Float32(_schema_real(fields, item, name, context))
        else:
            raise Error("RuntimeError: unknown saved vector dtype " + dtype)
    _ = store^
    return vector


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


def _schema_classes(value: PythonObject, integral: PythonObject, real: PythonObject,
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
        elif kind == "classes":
            py_setitem(values, step[1], _schema_classes(_schema_value(fields, step[2], name), step[3], step[4], context))
        elif kind == "kernel":
            _schema_kernel(fields, step, values, name, context)
        elif kind == "stack":
            py_setitem(values, step[1], _schema_stack(fields, step, name, context))
        elif kind == "each":
            py_setitem(values, step[1], _schema_each(fields, step, name, context))
        elif kind == "when":
            if _schema_has(fields, step[1]) and Bool(py=_schema_value(fields, step[1], name) == step[2]):
                _schema_run(model, fields, step[3], values, name, context)
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
