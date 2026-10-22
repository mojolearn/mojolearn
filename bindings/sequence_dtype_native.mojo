# SPDX-License-Identifier: Apache-2.0
# Normalize dtype metadata without NumPy's Python _commastring/_usefields.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_fields, py_get, py_setitem, py_contains
from bindings.native_numpy_metadata import native_numpy_dtype
from bindings.sequence_string_storage import sequence_string_class, sequence_string_admit
from bindings.sequence_array_warnings import sequence_array_warn, sequence_array_category
from bindings.optimizer_public_words import stash_python_error


def sequence_dtype_create(value: PythonObject, context: PythonObject, aligned: Bool,
                          metadata: PythonObject) raises -> PythonObject:
    # The verified native dtype constructor reports specific public exception
    # classes. Preserve its CPython exception triple across the Mojo door;
    # converting it to a plain Mojo Error loses ValueError/OverflowError.
    ref cpy = Python().cpython()
    var constructor = native_numpy_dtype(context)
    var arguments = Python.tuple(value)
    var keywords = Python.dict(align=PythonObject(aligned))
    if metadata is not None:
        py_setitem(keywords, "metadata", metadata)
    var result = cpy.PyObject_Call(constructor._obj_ptr, arguments._obj_ptr, keywords._obj_ptr)
    _ = arguments^
    _ = keywords^
    _ = constructor^
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


def sequence_dtype_create(value: PythonObject, context: PythonObject, aligned: Bool = False) raises -> PythonObject:
    return sequence_dtype_create(value, context, aligned, PythonObject(None))


def sequence_dtype_syntax(repeats: PythonObject) raises:
    # Empty compact-shape components have the same native SyntaxError
    # contract as NumPy's literal parser, without invoking its Python AST
    # wrapper. Preserve the source information carried by that exception.
    ref cpy = Python().cpython()
    var category = sequence_array_category("SyntaxError")
    var text = String(repeats)
    var offset = 1
    var empty = True
    for i in range(text.byte_length()):  # driver-loop: locate an empty compact-shape component in native parser text
        var character = UInt8(text.unsafe_ptr()[i])
        if character == 32 or character == 40:
            continue
        if character == 44:
            if empty:
                offset = i + 1
                break
            empty = True
        else:
            empty = False
    var details = Python.tuple("<unknown>", 1, offset, repeats, 1, offset + 1)
    var exception = py_call(category, "invalid syntax", details)
    cpy.lib.get_function[NoneType]("PyErr_SetObject")(category._obj_ptr, exception._obj_ptr)
    var failure = stash_python_error()
    raise failure^


def sequence_dtype_classes(context: PythonObject) raises -> PythonObject:
    var dtype = native_numpy_dtype(context)
    var stored = py_get(context, "sequence_dtype_classes")
    if stored is not None:
        return stored
    var result = Python.list()
    for code in ["?", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f2", "f4", "f8", "g", "c8", "c16", "G", "O", "S", "U", "V", "M8", "m8"]:  # driver-loop: connect fixed native dtype metadata
        var item = sequence_dtype_create(PythonObject(code), context)
        py_call(result.append, Python.tuple(py_type(item), item.type))
    var strings = sequence_string_class(context)
    if strings is not None:
        var item = sequence_dtype_create(PythonObject("T"), context)
        py_call(result.append, Python.tuple(strings, item.type))
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
            var dtype = sequence_dtype_create(classes[i][1], context)
            var kind = String(dtype.kind)
            if kind == "i" or kind == "u":
                return
    raise Error("TypeError: dtype dimensions and offsets require native integers")


def sequence_dtype_dimension(value: PythonObject, classes: PythonObject, context: PythonObject) raises:
    var b = Python.import_module("builtins")
    var kind = py_type(value)
    if value is None or kind is b.str or kind is b.bytes or kind is b.float or kind is b.complex or kind is b.list or kind is b.tuple or kind is b.dict:
        raise Error("ValueError: invalid shape in fixed-type tuple.")
    for i in range(len(classes)):  # driver-loop: reject native non-integer scalar dimensions without protocol calls
        if kind is classes[i][1]:
            var dtype = sequence_dtype_create(classes[i][1], context)
            var code = String(dtype.kind)
            if code != "i" and code != "u":
                raise Error("ValueError: invalid shape in fixed-type tuple.")
            return
    sequence_dtype_integer(value, classes, context)


def sequence_dtype_shape(value: PythonObject, classes: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if py_type(value) is b.tuple or py_type(value) is b.list:
        for i in range(len(value)):  # driver-loop: validate dtype subarray dimensions
            sequence_dtype_dimension(value[i], classes, context)
        return py_call(b.tuple, value)
    else:
        sequence_dtype_dimension(value, classes, context)
        return Python.tuple(value)


def sequence_dtype_pair(base: PythonObject, shape: PythonObject, classes: PythonObject,
                        context: PythonObject, aligned: Bool) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var dtype = native_numpy_dtype(context)
    var scalar = py_type(shape) is not b.list and py_type(shape) is not b.tuple
    var dimensions = sequence_dtype_shape(shape, classes, context)
    if scalar:
        var kind = String(base.kind)
        if Int(py=base.itemsize) == 0 and base.fields is None and base.subdtype is None and (kind == "S" or kind == "U" or kind == "V"):
            # Unsized flexible dtypes interpret a scalar as itemsize, not
            # subarray shape. Construct the same native descriptor directly;
            # passing a scalar in a dtype tuple invokes npy_ctypes_check.
            var limit = 2147483647 // 4 if kind == "U" else 2147483647
            if py_type(shape) is b.bool or shape < PythonObject(0) or shape > PythonObject(limit):
                raise Error("ValueError: invalid itemsize in generic type tuple")
            var spelling = String(base.byteorder) + kind + String(Int(py=shape))
            var metadata = base.metadata
            if metadata is not None:
                return sequence_dtype_create(PythonObject(spelling), context, aligned, py_call(b.dict, metadata))
            return sequence_dtype_create(PythonObject(spelling), context, aligned)
        if py_type(shape) is not b.bool and shape == PythonObject(1):
            var module = py_get(context["modules"], "numpy._core._multiarray_umath")
            if module is None:
                module = py_get(context["modules"], "numpy.core._multiarray_umath")
            if module is None or py_type(module) is not context["module_type"]:
                raise Error("TypeError: dtype compatibility requires NumPy's native version metadata")
            var entry = py_get(py_fields(module), "_get_ndarray_c_version")
            var warning = context["sequence_array_warnings"]
            if entry is None or py_type(entry) is not warning["warning_native_function"] or String(entry.__name__) != "_get_ndarray_c_version" or entry.__self__ is not module:
                raise Error("TypeError: dtype compatibility requires NumPy's native version entry")
            if Int(py=py_call(entry)) < 0x02000000:
                # NumPy 1.26 preserves its deprecated scalar-size-one alias;
                # explicit tuple (1,) already means a one-element subarray.
                sequence_array_warn("Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.", warning, "DeprecationWarning")
                return base
    return sequence_dtype_create(Python.tuple(base, dimensions), context, aligned)


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
        # NumPy 1.26's registry includes integer type numbers beside names.
        # Exact builtin int/str hashing and equality cannot invoke user hooks;
        # subclasses or other keys must still refuse before looking up text.
        var key_type = py_type(keys[i])
        if key_type is not b.str and key_type is not b.int:
            raise Error("TypeError: dtype aliases require native string or integer keys")
    var mapped = py_get(aliases, text)
    if mapped is not None:
        return sequence_dtype_normalize(mapped, classes, context, depth + 1, aligned)
    if py_call(context["sequence_dtype_fullmatch"], context["sequence_dtype_pattern"], text) is not None:
        return sequence_dtype_create(text, context, aligned)

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
        var raw_repeats = String(groups[1])
        var repeats = String(raw_repeats.strip())
        if repeats != "":
            var tuple_shape = repeats.find(",") >= 0
            if repeats.startswith("(") and repeats.endswith(")"):
                var body = String(repeats[byte=1:repeats.byte_length() - 1])
                var trimmed_body = String(body.strip())
                repeats = trimmed_body^
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
                        sequence_dtype_syntax(PythonObject(String(raw_repeats.strip())))
                shape = py_call(b.tuple, dimensions)
            else:
                shape = py_call(b.int, PythonObject(repeats))
            var shaped = sequence_dtype_pair(base, shape, classes, context, aligned)
            base = shaped^
        py_call(formats.append, base)
        py_call(fields.append, Python.tuple("f" + String(len(fields)), base))
    if len(formats) == 0:
        raise Error("TypeError: dtype text is not understood")
    if record:
        return sequence_dtype_create(fields, context, aligned)
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
    return sequence_dtype_create(result, context, alignment)


def sequence_dtype_normalize(value: PythonObject, classes: PythonObject, context: PythonObject, depth: Int = 0, aligned: Bool = False) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if depth > 64:
        raise Error("TypeError: dtype schema nesting exceeds the native limit")
    var dtype = native_numpy_dtype(context)
    var kind = py_type(value)
    if sequence_dtype_instance(value, classes):
        if String(value.kind) == "T":
            sequence_string_admit(value, classes, context)
        return value
    if kind is b.str or kind is b.bytes:
        return sequence_dtype_text(value, classes, context, depth, aligned)
    if value is None or value is b.float or value is b.int or value is b.bool or value is b.complex or value is b.str or value is b.bytes or value is b.object:
        return sequence_dtype_create(value, context, aligned)
    for i in range(len(classes)):  # driver-loop: admit exact native scalar classes
        if value is classes[i][1]:
            return sequence_dtype_create(value, context, aligned)
        if value is classes[i][0]:
            # astype accepts a concrete DType class as its top-level target;
            # np.dtype(class), including inside a field schema, instead
            # interprets that class as object. Parametric legacy classes
            # cannot all be instantiated with class(), so use their native
            # scalar's default descriptor; StringDType's scalar is str and
            # needs its distinct managed-string descriptor.
            if depth == 0:
                if value is sequence_string_class(context):
                    return sequence_dtype_create(PythonObject("T"), context, aligned)
                return sequence_dtype_create(classes[i][1], context, aligned)
            return sequence_dtype_create(b.object, context, aligned)
    if kind is b.tuple and len(value) == 2:
        var base = sequence_dtype_normalize(value[0], classes, context, depth + 1, aligned)
        var shape = value[1]
        if py_type(shape) is b.dict or sequence_dtype_instance(shape, classes):
            var normalized_shape = sequence_dtype_normalize(shape, classes, context, depth + 1, aligned)
            shape = normalized_shape^
        else:
            return sequence_dtype_pair(base, shape, classes, context, aligned)
        return sequence_dtype_create(Python.tuple(base, shape), context, aligned)
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
                var shaped = sequence_dtype_pair(format, field[2], classes, context, aligned)
                py_call(fields.append, Python.tuple(name, shaped))
            else:
                py_call(fields.append, Python.tuple(name, format))
        return sequence_dtype_create(fields, context, aligned)
    if kind is b.dict:
        return sequence_dtype_dict(value, classes, context, depth, aligned)
    raise Error("TypeError: dtype requires a built-in native dtype/schema; Python dtype protocols cannot execute")
