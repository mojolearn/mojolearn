# SPDX-License-Identifier: Apache-2.0
# Normalize dtype metadata without NumPy's Python _commastring/_usefields.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_fields, py_get, py_setitem, py_contains
from bindings.native_numpy_metadata import native_numpy_dtype


def sequence_dtype_classes(context: PythonObject) raises -> PythonObject:
    var dtype = native_numpy_dtype(context)
    var stored = py_get(context, "sequence_dtype_classes")
    if stored is not None:
        return stored
    var result = Python.list()
    for code in ["?", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f2", "f4", "f8", "g", "c8", "c16", "G", "O", "S", "U", "V", "M8", "m8"]:  # driver-loop: connect fixed native dtype metadata
        var item = py_call(dtype, code)
        py_call(result.append, Python.tuple(py_type(item), item.type))
    py_setitem(context, "sequence_dtype_classes", result)
    return result


def sequence_dtype_instance(value: PythonObject, classes: PythonObject) raises -> Bool:
    for i in range(len(classes)):  # driver-loop: compare connected native dtype classes
        if py_type(value) is classes[i][0]:
            return True
    return False


def sequence_dtype_integer(value: PythonObject, classes: PythonObject, context: PythonObject) raises:
    var b = Python.import_module("builtins")
    if py_type(value) is b.int or py_type(value) is b.bool:
        return
    for i in range(len(classes)):  # driver-loop: admit native integer metadata before index conversion
        if py_type(value) is classes[i][1]:
            var dtype = py_call(native_numpy_dtype(context), classes[i][1])
            var kind = String(dtype.kind)
            if kind == "i" or kind == "u":
                return
    raise Error("TypeError: dtype dimensions and offsets require native integers")


def sequence_dtype_shape(value: PythonObject, classes: PythonObject, context: PythonObject) raises:
    var b = Python.import_module("builtins")
    if py_type(value) is b.tuple or py_type(value) is b.list:
        for i in range(len(value)):  # driver-loop: validate dtype subarray dimensions
            sequence_dtype_integer(value[i], classes, context)
    else:
        sequence_dtype_integer(value, classes, context)


def sequence_dtype_title(value: PythonObject, depth: Int = 0) raises:
    var b = Python.import_module("builtins")
    var kind = py_type(value)
    if value is None or kind is b.str or kind is b.bytes or kind is b.int or kind is b.bool or kind is b.float or kind is b.complex:
        return
    if kind is b.tuple and depth < 64:
        for i in range(len(value)):  # driver-loop: validate native compound field titles before hashing
            sequence_dtype_title(value[i], depth + 1)
        return
    raise Error("TypeError: dtype field titles require native scalar metadata; Python hooks cannot execute")


def sequence_dtype_text(value: PythonObject, classes: PythonObject, context: PythonObject, depth: Int, aligned: Bool) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var dtype = native_numpy_dtype(context)
    var text = value if py_type(value) is b.str else py_call(b.bytes.decode, value, "ascii", "strict")
    var aliases = py_fields(context["modules"]["numpy"])["sctypeDict"]
    if py_type(aliases) is not b.dict:
        raise Error("TypeError: dtype aliases require native dictionary metadata")
    var keys = py_call(b.list, aliases)
    for i in range(len(keys)):  # driver-loop: validate alias keys before native dictionary lookup
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: dtype aliases require native string keys")
    var mapped = py_get(aliases, text)
    if mapped is not None:
        return sequence_dtype_normalize(mapped, classes, context, depth + 1, aligned)
    if py_call(context["sequence_dtype_fullmatch"], context["sequence_dtype_pattern"], text) is not None:
        return py_call(dtype, text, align=PythonObject(aligned))

    var spelling = String(text)
    var first = 0
    if spelling.startswith("<") or spelling.startswith(">") or spelling.startswith("=") or spelling.startswith("|"):
        first = 1
    if spelling.find(",") < 0:
        if first >= spelling.byte_length():
            raise Error("TypeError: dtype text is not understood")
        var initial = UInt8(spelling.unsafe_ptr()[first])
        if initial != 40 and (initial < 48 or initial > 57):
            raise Error("TypeError: dtype text is not understood")

    var position = 0
    var fields = Python.list()
    var formats = Python.list()
    var record = False
    while position < len(text):  # driver-loop: parse each compact dtype field in Mojo
        var matched = py_call(context["sequence_dtype_match"], context["sequence_dtype_compact"], text, position)
        if matched is None or Int(py=py_call(matched.end)) <= position:
            raise Error("ValueError: dtype format is not recognized")
        var groups = py_call(matched.groups)
        position = Int(py=py_call(matched.end))
        if position < len(text):
            if py_call(context["sequence_dtype_match"], context["sequence_dtype_space"], text, position) is not None:
                position = len(text)
            else:
                var separator = py_call(context["sequence_dtype_match"], context["sequence_dtype_separator"], text, position)
                if separator is None:
                    raise Error("ValueError: dtype format is not recognized")
                position = Int(py=py_call(separator.end))
                record = True
        var order1 = String(groups[0])
        var order2 = String(groups[2])
        var order = order1 if order2 == "" else order2
        if order1 != "" and order2 != "":
            if order1 == "=":
                order1 = "<"
            if order2 == "=":
                order2 = "<"
            if order1 != order2:
                raise Error("ValueError: inconsistent dtype byte-order specification")
        if order == "|" or order == "=" or order == "<":
            order = ""
        var base = sequence_dtype_normalize(PythonObject(order + String(groups[3])), classes, context, depth + 1, aligned)
        var repeats = String(groups[1]).strip()
        if repeats != "":
            var tuple_shape = repeats.find(",") >= 0
            if repeats.startswith("(") and repeats.endswith(")"):
                var body = String(repeats[byte=1:repeats.byte_length() - 1]).strip()
                repeats = body^
                if repeats != "" and not tuple_shape:
                    raise Error("TypeError: parenthesized dtype dimensions require a tuple comma")
                tuple_shape = True
            var shape = PythonObject(0)
            if tuple_shape:
                var dimensions = Python.list()
                var parts = py_call(b.str.split, PythonObject(repeats), ",")
                for i in range(len(parts)):  # driver-loop: parse native integer shape metadata
                    var part = py_call(b.str.strip, parts[i])
                    if len(part) > 0:
                        py_call(dimensions.append, py_call(b.int, part))
                    elif i + 1 < len(parts):
                        raise Error("ValueError: dtype shape is not recognized")
                shape = py_call(b.tuple, dimensions)
            else:
                shape = py_call(b.int, PythonObject(repeats))
            base = py_call(dtype, Python.tuple(base, shape), align=PythonObject(aligned))
        py_call(formats.append, base)
        py_call(fields.append, Python.tuple("f" + String(len(fields)), base))
    if len(formats) == 0:
        raise Error("TypeError: dtype text is not understood")
    if record:
        return py_call(dtype, fields, align=PythonObject(aligned))
    return formats[0]


def sequence_dtype_dict(value: PythonObject, classes: PythonObject, context: PythonObject, depth: Int, aligned: Bool) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var keys = py_call(b.list, value)
    for i in range(len(keys)):  # driver-loop: validate keys before any schema lookup
        if py_type(keys[i]) is not b.str and not (py_type(keys[i]) is b.int and Int(py=keys[i]) == -1):
            raise Error("TypeError: dtype schema keys require native field names")
    var result = Python.dict()
    var names = Python.list()
    var formats = Python.list()
    var offsets = Python.list()
    var titles = Python.list()
    var ordered = py_get(value, PythonObject(-1))
    var native_fields = py_contains(value, "names") and py_contains(value, "formats")
    var alignment = aligned
    if native_fields:
        var requested_align = py_get(value, "aligned")
        if requested_align is not None:
            if py_type(requested_align) is not b.bool and py_type(requested_align) is not b.int:
                raise Error("TypeError: dtype alignment requires a native boolean")
            alignment = Bool(py=requested_align)
        ordered = value["names"]
    if ordered is None:
        ordered = keys
    if py_type(ordered) is not b.list and py_type(ordered) is not b.tuple:
        raise Error("TypeError: dtype names require a native sequence")
    if native_fields:
        var specs = value["formats"]
        if (py_type(specs) is not b.list and py_type(specs) is not b.tuple) or len(specs) != len(ordered):
            raise Error("ValueError: dtype names and formats must have equal lengths")
        for field in ["offsets", "titles"]:  # driver-loop: validate optional native field metadata lengths
            if py_contains(value, PythonObject(field)):
                var items = value[PythonObject(field)]
                if (py_type(items) is not b.list and py_type(items) is not b.tuple) or len(items) != len(ordered):
                    raise Error("ValueError: dtype field metadata must have equal lengths")
    for i in range(len(ordered)):  # driver-loop: normalize explicit field layouts without _usefields
        var name = ordered[i]
        if py_type(name) is not b.str:
            raise Error("TypeError: dtype field names require native strings")
        var format = PythonObject(None)
        var offset = PythonObject(None)
        var title = PythonObject(None)
        if native_fields:
            var specs = value["formats"]
            if py_type(specs) is not b.list and py_type(specs) is not b.tuple:
                raise Error("TypeError: dtype formats require a native sequence")
            format = specs[i]
            if py_contains(value, "offsets"):
                var explicit_offsets = value["offsets"]
                if py_type(explicit_offsets) is not b.list and py_type(explicit_offsets) is not b.tuple:
                    raise Error("TypeError: dtype offsets require a native sequence")
                offset = explicit_offsets[i]
            if py_contains(value, "titles"):
                var explicit_titles = value["titles"]
                if py_type(explicit_titles) is not b.list and py_type(explicit_titles) is not b.tuple:
                    raise Error("TypeError: dtype titles require a native sequence")
                title = explicit_titles[i]
        else:
            var spec = value[name]
            if py_type(spec) is not b.tuple or (len(spec) != 2 and len(spec) != 3):
                raise Error("ValueError: dtype field requires a 2- or 3-tuple")
            format = spec[0]
            offset = spec[1]
            if len(spec) == 3:
                title = spec[2]
        sequence_dtype_title(title)
        if not native_fields and title == name:
            continue
        if offset is not None:
            sequence_dtype_integer(offset, classes, context)
        var position = len(names)
        if not native_fields and py_get(value, PythonObject(-1)) is None:
            while position > 0 and offset < offsets[position - 1]:  # driver-loop: stable field ordering by native offsets
                position -= 1
        py_call(names.insert, position, name)
        py_call(formats.insert, position, sequence_dtype_normalize(format, classes, context, depth + 1, alignment))
        py_call(offsets.insert, position, offset)
        py_call(titles.insert, position, title)
    py_setitem(result, "names", names)
    py_setitem(result, "formats", formats)
    py_setitem(result, "titles", titles)
    if not native_fields or py_contains(value, "offsets"):
        py_setitem(result, "offsets", offsets)
    if native_fields and py_contains(value, "itemsize"):
        sequence_dtype_integer(value["itemsize"], classes, context)
        py_setitem(result, "itemsize", value["itemsize"])
    return py_call(native_numpy_dtype(context), result, align=PythonObject(alignment))


def sequence_dtype_normalize(value: PythonObject, classes: PythonObject, context: PythonObject, depth: Int = 0, aligned: Bool = False) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if depth > 64:
        raise Error("TypeError: dtype schema nesting exceeds the native limit")
    var dtype = native_numpy_dtype(context)
    var kind = py_type(value)
    if sequence_dtype_instance(value, classes):
        return value
    if kind is b.str or kind is b.bytes:
        return sequence_dtype_text(value, classes, context, depth, aligned)
    if value is None or value is b.float or value is b.int or value is b.bool or value is b.complex or value is b.str or value is b.bytes or value is b.object:
        return py_call(dtype, value, align=PythonObject(aligned))
    for i in range(len(classes)):  # driver-loop: admit exact native scalar classes
        if value is classes[i][1]:
            return py_call(dtype, value, align=PythonObject(aligned))
    if kind is b.tuple and len(value) == 2:
        var base = sequence_dtype_normalize(value[0], classes, context, depth + 1, aligned)
        var shape = value[1]
        if py_type(shape) is b.dict or sequence_dtype_instance(shape, classes):
            var normalized_shape = sequence_dtype_normalize(shape, classes, context, depth + 1, aligned)
            shape = normalized_shape^
        else:
            sequence_dtype_shape(shape, classes, context)
        return py_call(dtype, Python.tuple(base, shape), align=PythonObject(aligned))
    if kind is b.list:
        var fields = Python.list()
        for i in range(len(value)):  # driver-loop: normalize explicit field records and their subarrays
            var field = value[i]
            if py_type(field) is not b.tuple or (len(field) != 2 and len(field) != 3):
                raise Error("TypeError: dtype fields require native 2- or 3-tuples")
            var name = field[0]
            if py_type(name) is b.tuple and len(name) == 2:
                sequence_dtype_title(name[0])
                if py_type(name[1]) is not b.str:
                    raise Error("TypeError: dtype field names require native strings")
            elif py_type(name) is not b.str:
                raise Error("TypeError: dtype field names require native strings")
            var format = sequence_dtype_normalize(field[1], classes, context, depth + 1, aligned)
            if len(field) == 3:
                sequence_dtype_shape(field[2], classes, context)
                py_call(fields.append, Python.tuple(name, format, field[2]))
            else:
                py_call(fields.append, Python.tuple(name, format))
        return py_call(dtype, fields, align=PythonObject(aligned))
    if kind is b.dict:
        return sequence_dtype_dict(value, classes, context, depth, aligned)
    raise Error("TypeError: dtype requires a built-in native dtype/schema; Python dtype protocols cannot execute")
