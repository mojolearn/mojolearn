# SPDX-License-Identifier: Apache-2.0
# Fixed native CV wire records. No Python class/function, pickle reducer or
# import name from the request is executed. Import-time maps select all owners.
from bindings.native_array_storage import native_array_empty
from bindings.native_loaded_module import native_loaded_module
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.native_buffer_owner import native_storage_allocate, native_buffer_view
from bindings.native_numpy_metadata import native_numpy_dtype, native_numpy_type
from bindings.classical_attributes import classical_connect_numpy
from bindings.msel_estimator_native import msel_native_clone, msel_estimator_plan, msel_clone_parameter
from bindings.msel_scorer_native import msel_scorer_fields, msel_metric_plan
from bindings.family_common import family_call

comptime MSEL_WIRE_VERSION = "native-parallel-cv-v1"


def _depth(depth: Int) raises:
    if depth > 64:
        raise Error("ValueError: native CV configuration exceeds 64 nested levels")


def _record_size(record: PythonObject, size: Int) raises:
    if len(record) != size:
        raise Error("ValueError: malformed native CV value record length")


def _sequence(record: PythonObject) raises:
    var b = native_loaded_module("builtins")
    if py_type(record) is not b.list and py_type(record) is not b.tuple:
        raise Error("TypeError: native CV wire sequences require exact list/tuple metadata")


def _dtype_pack(dtype: PythonObject, context: PythonObject, depth: Int = 0) raises -> PythonObject:
    _depth(depth)
    var b = native_loaded_module("builtins")
    if Bool(py=dtype.hasobject):
        raise Error("TypeError: native CV parameters cannot contain Python object references")
    var metadata = dtype.metadata
    var encoded_metadata = Python.tuple("atom", PythonObject(None))
    if metadata is not None:
        # Iterate the native dictionary's item view without rehashing its
        # keys. Admission/encoding must precede any new dictionary insertion.
        var pairs = py_call(b.list, py_call(metadata.items))
        var encoded = Python.list()
        for i in range(len(pairs)):  # driver-loop: native dtype metadata without arbitrary mapping hooks
            _ = py_call(encoded.append, Python.tuple(msel_wire_pack(pairs[i][0], context, depth + 1),
                                                     msel_wire_pack(pairs[i][1], context, depth + 1)))
        encoded_metadata = Python.tuple("dict", encoded)
    var payload: PythonObject
    var fields = dtype.fields
    var sub = dtype.subdtype
    if fields is not None:
        var names = dtype.names
        var formats = Python.list()
        var offsets = Python.list()
        var titles = Python.list()
        for i in range(len(names)):  # driver-loop: native dtype field metadata, preserving explicit offsets
            var field = fields[names[i]]
            _ = py_call(formats.append, _dtype_pack(field[0], context, depth + 1))
            _ = py_call(offsets.append, field[1])
            _ = py_call(titles.append, msel_wire_pack(field[2] if len(field) == 3 else PythonObject(None), context, depth + 1))
        payload = Python.tuple("fields", names, formats, offsets, titles, dtype.itemsize)
    elif sub is not None:
        payload = Python.tuple("subarray", _dtype_pack(sub[0], context, depth + 1), sub[1])
    else:
        payload = Python.tuple("scalar", dtype.str)
    # dtype.descr/name/str(dtype) call NumPy Python helpers; raw native fields,
    # offsets and the primitive typestr getter do not. Explicit names/formats
    # avoid the dtype converter's legacy Python _usefields dictionary path.
    return Python.tuple(payload, dtype.isalignedstruct, encoded_metadata)


def _dtype_unpack(record: PythonObject, context: PythonObject, depth: Int = 0) raises -> PythonObject:
    _depth(depth)
    var b = native_loaded_module("builtins")
    _sequence(record)
    _record_size(record, 3)
    if py_type(record[1]) is not b.bool:
        raise Error("TypeError: native dtype alignment must be a bool")
    var payload = record[0]
    _sequence(payload)
    if len(payload) < 2 or py_type(payload[0]) is not b.str:
        raise Error("TypeError: malformed native dtype record")
    var tag = String(payload[0])
    var descriptor: PythonObject
    if tag == "scalar":
        _record_size(payload, 2)
        if py_type(payload[1]) is not b.str or len(payload[1]) == 0:
            raise Error("TypeError: native dtype typestr requires an exact string")
        var text = String(payload[1])
        if "," in text or "(" in text or ")" in text:
            raise Error("TypeError: dtype wire requires primitive typestrs, not comma syntax")
        descriptor = payload[1]
    elif tag == "subarray":
        _record_size(payload, 3)
        _sequence(payload[2])
        for i in range(len(payload[2])):  # driver-loop: native subarray dimensions
            if py_type(payload[2][i]) is not b.int:
                raise Error("TypeError: native dtype shape requires integers")
        descriptor = Python.tuple(_dtype_unpack(payload[1], context, depth + 1), payload[2])
    elif tag == "fields":
        _record_size(payload, 6)
        for i in range(1, 5):  # small-loop(four native dtype field metadata sequences)
            _sequence(payload[i])
            if len(payload[i]) != len(payload[1]):
                raise Error("ValueError: native dtype field metadata lengths differ")
        if py_type(payload[5]) is not b.int:
            raise Error("TypeError: native dtype itemsize must be an integer")
        var formats = Python.list()
        var titles = Python.list()
        for i in range(len(payload[1])):  # driver-loop: restore field dtypes without Python descriptor conversion
            if py_type(payload[1][i]) is not b.str or py_type(payload[3][i]) is not b.int:
                raise Error("TypeError: dtype field names/offsets require strings/integers")
            _ = py_call(titles.append, msel_wire_unpack(payload[4][i], context, depth + 1))
            _ = py_call(formats.append, _dtype_unpack(payload[2][i], context, depth + 1))
        descriptor = Python.dict()
        py_setitem(descriptor, "names", payload[1])
        py_setitem(descriptor, "formats", formats)
        py_setitem(descriptor, "offsets", payload[3])
        py_setitem(descriptor, "titles", titles)
        py_setitem(descriptor, "itemsize", payload[5])
    else:
        raise Error("TypeError: unknown native dtype record")
    var dtype = py_call(native_numpy_dtype(context), descriptor, align=record[1])
    var metadata = msel_wire_unpack(record[2], context, depth + 1)
    if metadata is not None:
        if py_type(metadata) is not b.dict:
            raise Error("TypeError: NumPy wire dtype metadata must be a dictionary")
        dtype = py_call(native_numpy_dtype(context), dtype, metadata=metadata)
    return dtype


def _numpy_buffer(value: PythonObject, scalar: Bool, context: PythonObject, depth: Int) raises -> PythonObject:
    var dtype = value.dtype
    var saved_dtype = _dtype_pack(dtype, context, depth + 1)
    var view = native_buffer_view(value)
    var order = PythonObject("F" if not scalar and Bool(py=view.f_contiguous) and not Bool(py=view.c_contiguous) else "C")
    var raw = py_call(view.tobytes, order)
    _ = view  # keep the acquired native buffer alive through the byte copy
    return Python.tuple("numpy_scalar" if scalar else "ndarray", saved_dtype, view.shape, order, raw)


def msel_wire_pack(value: PythonObject, context: PythonObject, depth: Int = 0) raises -> PythonObject:
    _depth(depth)
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if value is None or cls is b.bool or cls is b.int or cls is b.float or cls is b.str or cls is b.bytes:
        return Python.tuple("atom", value)
    if cls is b.complex:
        return Python.tuple("complex", value.real, value.imag)
    if cls is b.range:
        return Python.tuple("range", value.start, value.stop, value.step)
    if cls is b.bytearray:
        return Python.tuple("bytearray", py_call(b.bytes, value))
    if cls is b.list or cls is b.tuple or cls is b.set or cls is b.frozenset:
        var items = py_call(b.list, value)
        var result = Python.list()
        for i in range(len(items)):  # driver-loop: admitted constructor container values
            _ = py_call(result.append, msel_wire_pack(items[i], context, depth + 1))
        var tag = "list" if cls is b.list else ("tuple" if cls is b.tuple else ("set" if cls is b.set else "frozenset"))
        return Python.tuple(tag, result)
    if cls is b.dict:
        var pairs = py_call(b.list, value.items())
        var result = Python.list()
        for i in range(len(pairs)):  # driver-loop: structural keys and values preserve insertion order
            _ = py_call(result.append, Python.tuple(msel_wire_pack(pairs[i][0], context, depth + 1),
                                                    msel_wire_pack(pairs[i][1], context, depth + 1)))
        return Python.tuple("dict", result)
    if cls is context["array_class"]:
        return Python.tuple("Array", msel_clone_parameter(value, context))
    if cls is context["array_type"]:
        var view = native_buffer_view(value)
        var raw = py_call(view.tobytes)
        _ = view
        return Python.tuple("array", value.typecode, raw)
    classical_connect_numpy(context)
    if cls is py_get(context, "ndarray"):
        return _numpy_buffer(msel_clone_parameter(value, context), False, context, depth + 1)
    var scalars = context["scalar_types"]
    for i in range(len(scalars)):  # driver-loop: connected immutable native NumPy scalar types
        if cls is scalars[i]:
            return _numpy_buffer(value, True, context, depth + 1)
    if py_type(cls) is b.type and py_contains(context["family_msel_estimators"], cls):
        return Python.tuple("estimator", msel_parallel_model_pack(value, context, depth + 1))
    # Reuse the serial boundary's admission/error before any external protocol.
    _ = msel_clone_parameter(value, context)
    raise Error("TypeError: this constructor value has no native parallel CV wire representation")


def msel_wire_needs_numpy(value: PythonObject) raises -> Bool:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is b.tuple or cls is b.list:
        if len(value) and py_type(value[0]) is b.str:
            if String(value[0]) == "numpy_scalar" or String(value[0]) == "ndarray":
                return True
        for i in range(len(value)):  # driver-loop: generated wire metadata, never application iterators
            if msel_wire_needs_numpy(value[i]):
                return True
    elif cls is b.dict:
        var values = py_call(b.list, value.values())
        for i in range(len(values)):  # driver-loop: generated configuration record fields
            if msel_wire_needs_numpy(values[i]):
                return True
    return False


def msel_wire_unpack(record: PythonObject, context: PythonObject, depth: Int = 0) raises -> PythonObject:
    _depth(depth)
    var b = native_loaded_module("builtins")
    if py_type(record) is not b.tuple or len(record) < 2 or py_type(record[0]) is not b.str:
        raise Error("ValueError: malformed native CV value record")
    var tag = String(record[0])
    if tag == "atom":
        _record_size(record, 2)
        var cls = py_type(record[1])
        if record[1] is not None and cls is not b.bool and cls is not b.int and cls is not b.float and cls is not b.str and cls is not b.bytes:
            raise Error("TypeError: invalid native CV scalar record")
        return record[1]
    if tag == "complex":
        _record_size(record, 3)
        if py_type(record[1]) is not b.float or py_type(record[2]) is not b.float:
            raise Error("TypeError: native complex components must be floats")
        return py_call(b.complex, record[1], record[2])
    if tag == "range":
        _record_size(record, 4)
        for i in range(1, 4):  # small-loop(three exact range parameters)
            if py_type(record[i]) is not b.int:
                raise Error("TypeError: native range parameters must be integers")
        return py_call(b.range, record[1], record[2], record[3])
    if tag == "bytearray":
        _record_size(record, 2)
        if py_type(record[1]) is not b.bytes:
            raise Error("TypeError: native bytearray record requires bytes")
        return py_call(b.bytearray, record[1])
    if tag == "Array":
        _record_size(record, 2)
        if py_type(record[1]) is not context["array_class"]:
            raise Error("TypeError: CV wire array requires the native Array owner")
        return record[1]
    if tag == "array":
        _record_size(record, 3)
        if py_type(record[1]) is not b.str or py_type(record[2]) is not b.bytes:
            raise Error("TypeError: native array record requires typecode and bytes")
        var result = native_array_empty(String(record[1]))
        _ = py_call(result.frombytes, record[2])
        return result
    if tag == "numpy_scalar" or tag == "ndarray":
        _record_size(record, 5)
        if py_type(record[3]) is not b.str or py_type(record[4]) is not b.bytes:
            raise Error("TypeError: NumPy wire dtype/order/data require exact native metadata")
        _sequence(record[2])
        for i in range(len(record[2])):  # driver-loop: exact ndarray shape dimensions
            if py_type(record[2][i]) is not b.int or Int(py=record[2][i]) < 0:
                raise Error("TypeError: NumPy wire dimensions must be nonnegative integers")
        # Required native types were connected at fixed child import/setup;
        # no task-driven imports or NumPy conversion/copy functions execute.
        var dtype = _dtype_unpack(record[1], context, depth + 1)
        if Bool(py=dtype.hasobject):
            raise Error("TypeError: native CV wire cannot restore object-reference dtype")
        var storage = py_call(b.bytearray, record[4])
        var result = py_call(native_numpy_type("ndarray", context), record[2], dtype=dtype, buffer=storage, order=record[3])
        return result[Python.tuple()] if tag == "numpy_scalar" else result
    if tag == "estimator":
        _record_size(record, 2)
        return msel_parallel_model_unpack(record[1], context, depth + 1)
    if tag == "dict":
        _record_size(record, 2)
        _sequence(record[1])
        var result = Python.dict()
        for i in range(len(record[1])):  # driver-loop: native constructor mapping restoration
            var pair = record[1][i]
            _sequence(pair)
            _record_size(pair, 2)
            py_setitem(result, msel_wire_unpack(pair[0], context, depth + 1),
                        msel_wire_unpack(pair[1], context, depth + 1))
        return result
    if tag == "list" or tag == "tuple" or tag == "set" or tag == "frozenset":
        _record_size(record, 2)
        _sequence(record[1])
        var result = Python.list()
        for i in range(len(record[1])):  # driver-loop: native constructor container restoration
            _ = py_call(result.append, msel_wire_unpack(record[1][i], context, depth + 1))
        if tag == "tuple":
            return py_call(b.tuple, result)
        if tag == "set":
            return py_call(b.set, result)
        if tag == "frozenset":
            return py_call(b.frozenset, result)
        return result
    raise Error("ValueError: unknown native CV wire value kind")


def msel_parallel_model_pack(model: PythonObject, context: PythonObject, depth: Int = 0) raises -> PythonObject:
    _depth(depth)
    var clone = msel_native_clone(model, context)
    var plan = msel_estimator_plan(clone, context)
    var fields = py_call(plan["fields_get"], clone, py_type(clone))
    var parameters = Python.dict()
    var names = plan["parameters"]
    for i in range(len(names)):  # driver-loop: only unfitted fixed constructor parameters cross a process boundary
        var name = names[i]
        var value = fields[name]
        if String(name) == "numeric_mode" and value is not None:
            if py_type(value) is not py_type(PythonObject("")) or String(value) != "identical":
                raise Error("ValueError: parallel cross-validation requires IDENTICAL estimator numeric modes")
        py_setitem(parameters, name, msel_wire_pack(value, context, depth + 1))
    return Python.tuple(MSEL_WIRE_VERSION, context["family_msel_parallel"]["class_ids"][py_type(clone)], names, parameters)


def msel_parallel_model_unpack(record: PythonObject, context: PythonObject, depth: Int = 0) raises -> PythonObject:
    _depth(depth)
    var b = native_loaded_module("builtins")
    if py_type(record) is not b.tuple or len(record) != 4:
        raise Error("ValueError: incompatible native CV estimator record")
    if py_type(record[0]) is not b.str or py_type(record[1]) is not b.str or String(record[0]) != MSEL_WIRE_VERSION:
        raise Error("ValueError: incompatible native CV estimator record")
    var cls = py_get(context["family_msel_parallel"]["classes"], record[1])
    if cls is None:
        raise Error("TypeError: worker has no connected native estimator for the requested schema")
    var plan = context["family_msel_estimators"][cls]
    var names = plan["parameters"]
    if py_type(record[2]) is not b.tuple or len(record[2]) != len(names) or py_type(record[3]) is not b.dict or len(record[3]) != len(names):
        raise Error("ValueError: native CV worker constructor schema differs from its parent")
    var keys = py_call(b.list, record[3])
    for i in range(len(names)):  # driver-loop: exact schema strings before equality/hash lookup
        if py_type(keys[i]) is not b.str or py_type(record[2][i]) is not b.str or String(record[2][i]) != String(names[i]):
            raise Error("TypeError: native CV schema requires the connected exact parameter names")
    var parameters = Python.dict()
    var arguments = Python.list()
    for i in range(len(names)):  # driver-loop: native fixed-schema constructor values, before allocation
        var value = msel_wire_unpack(record[3][names[i]], context, depth + 1)
        py_setitem(parameters, names[i], value)
        _ = py_call(arguments.append, value)
    var result = native_storage_allocate(cls)
    _ = msel_estimator_plan(result, context)
    if plan["forest"] is not None or plan["init"] is None:
        var fields = py_call(plan["fields_get"], result, cls)
        _ = py_call(fields.update, parameters)
        if plan["forest"] is not None:
            py_setitem(fields, "_has_constructor_parameters", True)
    else:
        var entry = plan["entry"]
        if entry is None or py_type(entry) is not py_type(b.len):
            raise Error("ImportError: worker native estimator constructor is unavailable")
        _ = family_call(entry, result, plan["init"], py_call(b.tuple, arguments), plan["context"])
    return result


def msel_parallel_scorer_pack(scorer: PythonObject, context: PythonObject) raises -> PythonObject:
    if scorer is None:
        return PythonObject(None)
    var fields = msel_scorer_fields(scorer, context)
    var function = fields["_score_func"]
    _ = msel_metric_plan(function, context)
    var copied = py_call(native_loaded_module("builtins").dict, fields)
    _ = py_call(copied.pop, "_score_func")
    return Python.tuple(context["family_msel_parallel"]["metric_ids"][function], msel_wire_pack(copied, context))


def msel_parallel_scorer_unpack(record: PythonObject, context: PythonObject) raises -> PythonObject:
    if record is None:
        return record
    var b = native_loaded_module("builtins")
    if py_type(record) is not b.tuple or len(record) != 2 or py_type(record[0]) is not b.str:
        raise Error("TypeError: malformed native CV scorer record")
    var function = py_get(context["family_msel_parallel"]["metrics"], record[0])
    if function is None:
        raise Error("TypeError: worker has no connected native scorer metric")
    var fields = msel_wire_unpack(record[1], context)
    if py_type(fields) is not b.dict:
        raise Error("TypeError: native CV scorer configuration must be a dictionary")
    py_setitem(fields, "_score_func", function)
    var cls = context["family_msel_scorer_class"]
    var result = native_storage_allocate(cls)
    _ = py_call(py_call(context["family_msel_scorer_fields"], result, cls).update, fields)
    _ = msel_scorer_fields(result, context)
    return result
