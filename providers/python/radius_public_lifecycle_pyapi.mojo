# SPDX-License-Identifier: Apache-2.0
# RadiusNeighbors.fit owns its complete public lifecycle here. Search remains
# the separately owned neighbor_query provider; fit only stores the admitted X.
from std.math import isfinite
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.native_loaded_module import native_loaded_module
from bindings.mode_metadata_native import _fields, _class_fields
from bindings.classical_attributes import classical_connect_numpy, classical_check_layout, classical_getattribute, _type_mro
from bindings.classical_pandas import classical_admit_value, native_pandas_exact, _pd_class, _pd_fields, _pd_get, _pd_values
from bindings.classical_warnings import classical_warn_column
from bindings.classical_arrays import classical_address
from bindings.host_inference_scalars import host_inference_scalar, host_inference_primitive_number
from bindings.host_sparse_native import host_sparse_format, _sparse_fields
from bindings.host_archive_native import host_archive_radius_refusals
from bindings.native_buffer_owner import native_buffer_view
from bindings.native_scalar_decode import native_scalar_buffer
from bindings.native_numpy_metadata import native_numpy_dtype
from bindings.native_torch_admission import native_torch_numpy
from bindings.sequential_admission import _ordinary_torch
from bindings.family_common import family_f32_snapshot, family_owned_snapshot
from providers.python.array_metadata_pyapi import array_slot_once, array_owner_once
from providers.python.neighbors_public_lifecycle_pyapi import NeighborPublicCall, neighbors_public_end, neighbors_public_finish, neighbors_public_refusal
from providers.python.estimator_lifecycle_pyapi import estimator_outside, estimator_fit_begin
from core.native_scalar_decode import native_scalar_real


def radius_settings(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var fields = _fields(model)
    # RadiusNeighbors owns the native neighbor-query attribute descriptor, not
    # the generic public-output descriptor. Use its import-connected context
    # for both layout admission and setting reads; this also admits the saved
    # CPU class registered by connect_host without invoking Python hooks.
    var attributes = py_get(context, "family_xn_radius_attributes")
    if attributes is None:
        raise Error("RuntimeError: native RadiusNeighbors attribute context was not connected")
    classical_check_layout(model, attributes)
    var ancestry = _type_mro(py_type(model))
    var setter_seen = False
    for i in range(len(ancestry)):  # driver-loop: native field writes and fit callbacks are admitted before mutation
        var names = _class_fields(ancestry[i])
        if not setter_seen and py_contains(names, "__setattr__"):
            if names["__setattr__"] is not b.object.__dict__["__setattr__"]:
                raise Error("TypeError: native RadiusNeighbors fit cannot invoke Python field setters")
            setter_seen = True
        if py_contains(names, "_mojolearn_after_fit"):
            raise Error("TypeError: native RadiusNeighbors fit cannot invoke Python after-fit callbacks")
    if py_contains(fields, "_mojolearn_after_fit"):
        raise Error("TypeError: native RadiusNeighbors fit cannot invoke Python after-fit callbacks")
    var settings = Python.dict()
    var names = Python.tuple("metric", "algorithm", "p", "radius")
    for i in range(len(names)):  # small-loop(four current radius settings)
        var value = classical_getattribute(model, names[i], attributes)
        host_inference_scalar(value, context)
        # NumPy primitives decode through retained native scalar buffers before
        # the shared archive refusal helper uses builtin float parsing. This
        # prevents NumPy numeric conversion slots or custom scalar hooks.
        py_setitem(settings, names[i], host_inference_primitive_number(value, context))
    return settings


def _radius_refusals(fields: PythonObject, settings: PythonObject, context: PythonObject) raises:
    try:
        host_archive_radius_refusals(settings, "RadiusNeighbors", context)
    finally:
        if py_contains(settings, "_metric_value"):
            py_setitem(fields, "_metric_value", settings["_metric_value"])
            py_setitem(fields, "_metric_arg", settings["_metric_arg"])


def radius_refusals(model: PythonObject, context: PythonObject) raises:
    classical_connect_numpy(context)
    var settings = radius_settings(model, context)
    _radius_refusals(_fields(model), settings, context)


def _target_length(shape: PythonObject, rows: Int) raises:
    if len(shape) > 0 and Int(py=shape[0]) != rows:
        raise Error("Found input variables with inconsistent numbers of samples: [" + String(rows) + ", " + String(Int(py=shape[0])) + "]")


def _list_finite(value: PythonObject, context: PythonObject, name: String, depth: Int = 0) raises:
    if depth > 64:
        raise Error("TypeError: native target containers exceed the supported nesting depth")
    var b = native_loaded_module("builtins")
    if py_type(value) is b.list or py_type(value) is b.tuple:
        for i in range(len(value)):  # driver-loop: ignored float targets still preserve their finite-value validation
            _list_finite(value[i], context, name, depth + 1)
        return
    host_inference_scalar(value, context)
    var number = host_inference_primitive_number(value, context)
    if not isfinite(Float64(py=py_call(b.float, number))):
        raise Error("Input y contains NaN or infinity (" + name + ").")


def _complex_scalar(value: PythonObject, context: PythonObject) raises -> Bool:
    if py_get(context, "ndarray") is None:
        return False
    var dtype = native_numpy_dtype(context)
    var names = Python.tuple("F", "D", "G")
    for i in range(len(names)):  # small-loop(three exact NumPy complex scalar identities)
        if py_type(value) is py_call(dtype, names[i]).type:
            return True
    return False


def radius_target(source: PythonObject, rows: Int, context: PythonObject, transposed: Bool,
                  columns: PythonObject, name: String = "RadiusNeighbors") raises -> PythonObject:
    # y is ignored numerically. Keep only the old public length, complex,
    # column-warning and float-finiteness semantics, without flatten/tolist or
    # a second read of exporter metadata. The returned owner survives fit.
    if source is None:
        return source
    var b = native_loaded_module("builtins")
    var sparse = host_sparse_format(source)
    if sparse.byte_length() > 0:
        var shape = _sparse_fields(source)["_shape"]
        if py_type(shape) is not b.tuple or len(shape) != 2 or py_type(shape[0]) is not b.int:
            raise Error("TypeError: sparse target shape requires native dimensions")
        _target_length(shape, rows)
        return source
    var cls = py_type(source)
    if cls is b.list or cls is b.tuple:
        _target_length(Python.tuple(len(source)), rows)
        var first = source
        var depth = 0
        while py_type(first) is b.list or py_type(first) is b.tuple:  # driver-loop: first-element target rank metadata
            if len(first) == 0:
                return source
            first = first[0]
            depth += 1
            if depth > 64:
                raise Error("TypeError: native target containers exceed the supported nesting depth")
        if py_type(first) is b.complex or _complex_scalar(first, context):
            raise Error("Complex data not supported: " + name + " got a complex y.")
        var number = host_inference_primitive_number(first, context)
        if py_type(number) is b.float:
            _list_finite(source, context, name)
        return source
    if cls is b.str or cls is b.bytes or cls is b.int or cls is b.float or cls is b.bool or cls is b.complex:
        return source  # former describe(y) has no shape for these scalar values
    if _complex_scalar(source, context):
        raise Error("Complex data not supported: " + name + " got a complex y.")
    classical_admit_value(source, context)
    var storage = source
    var array = cls is context["array_class"]
    var reshape = array or cls is py_get(context, "ndarray")
    if native_pandas_exact(source):
        var series = cls is _pd_class("pandas.core.series", "Series")
        var manager = py_get(_pd_fields(source), "_mgr")
        var blocks = PythonObject(None)
        if series:
            if py_type(manager) is not _pd_class("pandas.core.internals.managers", "SingleBlockManager"):
                raise Error("TypeError: pandas Series requires native single-block storage")
            blocks = py_get(_pd_fields(manager), "blocks")
        else:
            if py_type(manager) is not _pd_class("pandas.core.internals.managers", "BlockManager"):
                raise Error("TypeError: pandas DataFrame requires native block storage")
            blocks = _pd_get("pandas._libs.internals", "BlockManager", "blocks", manager)
        if py_type(blocks) is not b.tuple or len(blocks) == 0 or (series and len(blocks) != 1):
            raise Error("TypeError: pandas target requires ordinary nonempty block metadata")
        var owners = Python.list(source)
        var column_count = Python.list(PythonObject(0))
        for i in range(len(blocks)):  # driver-loop: native pandas target block owners, without dtype conversion
            var block = blocks[i]
            var block_type = py_type(block)
            if block_type is not _pd_class("pandas.core.internals.blocks", "NumpyBlock") and block_type is not _pd_class("pandas.core.internals.blocks", "NumericBlock") and block_type is not _pd_class("pandas.core.internals.blocks", "ObjectBlock") and block_type is not _pd_class("pandas.core.internals.blocks", "ExtensionBlock"):
                raise Error("TypeError: pandas target requires ordinary native blocks")
            var pair = _pd_values(_pd_get("pandas._libs.internals", "Block", "values", block), series, context)
            _ = py_call(owners.append, radius_target(pair[0], rows, context, not series, column_count, name))
        if not series and Int(py=column_count[0]) == 1:
            classical_warn_column(context)
        return owners
    elif _ordinary_torch(cls, context):
        storage = native_torch_numpy(source)
        reshape = True
    var owner = array_owner_once(source, context) if array else storage
    var view = PythonObject(None)
    var shape = PythonObject(None)
    var strides = PythonObject(None)
    var code = String()
    var width = 0
    var address = 0
    var little = True
    if py_type(storage) is py_get(context, "ndarray"):
        # Exact ndarray metadata also covers StringDType/object/record targets
        # without demanding a PEP3118 export they need not implement. Ignored
        # nonnumeric targets require only their native shape, never conversion.
        shape = storage.shape
        strides = storage.strides
        var dtype = storage.dtype
        code = String(dtype.char)
        width = Int(py=dtype.itemsize)
        little = String(dtype.byteorder) != ">"
        if code == "F" or code == "D" or code == "G":
            code = "Z"
        if code == "e" or code == "f" or code == "d" or code == "g":
            address = Int(py=classical_address(storage))
    else:
        view = native_buffer_view(array_slot_once(source, "_mv", context) if array else storage)
        shape = array_slot_once(source, "shape", context) if array else view.shape
        strides = array_slot_once(source, "strides", context) if array else view.strides
        code = String(view.format)
        width = Int(py=view.itemsize)
        if array:
            var dtype = String(array_slot_once(source, "dtype", context))
            width = Int(py=array_slot_once(source, "itemsize", context))
            address = Int(py=array_slot_once(source, "_addr", context))
            code = String("e") if dtype == "<f2" else String("f") if dtype == "<f4" else String("d") if dtype == "<f8" else String("integer")
    if transposed and len(shape) > 0:
        _target_length(Python.tuple(shape[len(shape) - 1]), rows)
    else:
        _target_length(shape, rows)
    if columns is not None:
        py_setitem(columns, PythonObject(0), PythonObject(Int(py=columns[0]) + (Int(py=shape[0]) if transposed and len(shape) == 2 else 1)))
    if code.startswith(">") or code.startswith("!"):
        little = False
        var suffix = String(code[byte=1:])
        code = suffix^
    elif code.startswith("<") or code.startswith("=") or code.startswith("@") or code.startswith("|"):
        var suffix = String(code[byte=1:])
        code = suffix^
    if code.startswith("Z"):
        raise Error("Complex data not supported: " + name + " got a complex y.")
    if columns is None and reshape and len(shape) == 2 and Int(py=shape[1]) == 1:
        classical_warn_column(context)
    if code != "e" and code != "f" and code != "d" and code != "g":
        return Python.tuple(source, storage, owner, view)  # integer/string/object/record targets had no finite scan
    var dimensions = List[Int]()
    var steps = List[Int]()
    var count = 1
    for axis in range(len(shape)):  # small-loop(captured target shape and strides)
        if py_type(shape[axis]) is not b.int or py_type(strides[axis]) is not b.int:
            raise Error("TypeError: target shape and strides require native integers")
        var length = Int(py=shape[axis])
        dimensions.append(length)
        steps.append(Int(py=strides[axis]))
        count *= length
    # Documented CPython Py_buffer, matching native_scalar_decode's supported
    # 64-bit ABI. Acquire from the retained memoryview, never from the exporter
    # again; word zero is buf. Release on every path.
    var buffer = List[Int](length=10, fill=0)
    var pointer = buffer.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    ref cpy = Python().cpython()
    var exported = view is not None
    if exported:
        if cpy.lib.get_function[Int32]("PyObject_GetBuffer")(view._obj_ptr, pointer, Int32(0x118)) != 0:
            raise cpy.unsafe_get_error()
        if not array:
            address = buffer[0]
    try:
        var data = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
        for i in range(count):  # driver-loop: native finite scan over the captured target buffer
            var remaining = i
            var offset = 0
            for reversed_axis in range(len(dimensions)):  # small-loop(target rank coordinates, not exporter metadata)
                var axis = len(dimensions) - reversed_axis - 1
                offset += (remaining % dimensions[axis]) * steps[axis]
                remaining //= dimensions[axis]
            if native_scalar_real(data, offset, code, width, little).special != 0:
                raise Error("Input y contains NaN or infinity (" + name + ").")
    finally:
        if exported:
            cpy.lib.get_function[NoneType]("PyBuffer_Release")(pointer)
    return Python.tuple(source, storage, owner, view)


def radius_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var settings = radius_settings(model, context)
    var fields = _fields(model)
    var depth = py_get(fields, "_mojolearn_fitting", PythonObject(0))
    if py_type(depth) is not py_type(PythonObject(0)):
        raise Error("TypeError: native radius fit depth requires an exact integer")
    if host_sparse_format(args[0]).byte_length() > 0:
        raise Error("TypeError: Sparse data was passed for X, but dense data is required. Use '.toarray()' to convert to a dense numpy array. (RadiusNeighbors takes dense X only.)")
    var x = family_f32_snapshot(args[0], 2, "X", context)
    var target = radius_target(args[1] if len(args) > 1 else PythonObject(None), x.shape[0], context,
                               False, PythonObject(None)) if Int(py=depth) == 0 else PythonObject(None)
    var call = NeighborPublicCall(args, fields, args[0], "fit", True, estimator_outside(), estimator_fit_begin(fields), x.shape[0], x.shape[1])
    try:
        _radius_refusals(fields, settings, context)
        py_setitem(fields, "_index", family_owned_snapshot(x, context))
        py_setitem(fields, "n_samples_fit_", PythonObject(x.shape[0]))
        py_setitem(fields, "n_features_in_", PythonObject(x.shape[1]))
    except error:
        var refused = neighbors_public_refusal(error, call, model)
        raise refused^
    finally:
        neighbors_public_end(model, call)
    var result = neighbors_public_finish(model, "family_xn_radius_fit", model, call, context)
    _ = x^
    _ = target^
    return result
