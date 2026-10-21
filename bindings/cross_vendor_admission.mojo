# SPDX-License-Identifier: Apache-2.0
"""Exact native types at the distributed transport boundary; no protocols."""
from std.ffi import _Global
from std.math import isfinite
from std.python import Python, PythonObject
from bindings.native_json import native_path


struct CrossVendorTypes(Defaultable, Movable):
    var bi: PythonObject
    var array_cls: PythonObject
    var c_array: PythonObject
    var ndarray: PythonObject
    var path: PythonObject
    var pure_path: PythonObject

    def __init__(out self):
        self.bi = PythonObject(None)
        self.array_cls = PythonObject(None)
        self.c_array = PythonObject(None)
        self.ndarray = PythonObject(None)
        self.path = PythonObject(None)
        self.pure_path = PythonObject(None)


comptime CROSS_VENDOR_TYPES = _Global[
    StorageType=CrossVendorTypes, name="MojoCrossVendorTypes", init_fn=CrossVendorTypes.__init__,
]


def cv_types_initialize(array_cls: PythonObject) raises:
    var c = CROSS_VENDOR_TYPES.get_or_create_ptr()
    c[].bi = Python.import_module("builtins")
    c[].array_cls = array_cls
    c[].c_array = Python.import_module("array").array
    c[].path = Python.import_module("pathlib").PosixPath
    c[].pure_path = Python.import_module("pathlib").PurePosixPath
    try:
        c[].ndarray = Python.import_module("numpy").ndarray
    except:
        c[].ndarray = PythonObject(None)


def cv_int(value: PythonObject, label: String) raises -> Int:
    if CROSS_VENDOR_TYPES.get_or_create_ptr()[].bi.type(value) is not CROSS_VENDOR_TYPES.get_or_create_ptr()[].bi.int:
        raise Error(label + " requires a native int; Python conversion hooks are not accepted")
    return Int(py=value)


def cv_float(value: PythonObject, label: String) raises -> Float64:
    var bi = CROSS_VENDOR_TYPES.get_or_create_ptr()[].bi
    if bi.type(value) is not bi.int and bi.type(value) is not bi.float:
        raise Error(label + " requires a native number; Python conversion hooks are not accepted")
    var result = Float64(py=value)
    if not isfinite(result):
        raise Error(label + " must be finite")
    return result


def cv_bool(value: PythonObject, label: String) raises -> Bool:
    var bi = CROSS_VENDOR_TYPES.get_or_create_ptr()[].bi
    if bi.type(value) is not bi.bool:
        raise Error(label + " requires bool; Python truth hooks are not accepted")
    return Bool(py=value)


def cv_text(value: PythonObject, label: String) raises -> String:
    var bi = CROSS_VENDOR_TYPES.get_or_create_ptr()[].bi
    if bi.type(value) is not bi.str:
        raise Error(label + " requires str; Python string hooks are not accepted")
    return String(value)


def cv_sequence(value: PythonObject, label: String) raises -> PythonObject:
    var bi = CROSS_VENDOR_TYPES.get_or_create_ptr()[].bi
    if bi.type(value) is not bi.list and bi.type(value) is not bi.tuple and bi.type(value) is not bi.range:
        raise Error(label + " requires list, tuple or range; Python iteration hooks are not accepted")
    return bi.list(value)


def cv_buffer(value: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_TYPES.get_or_create_ptr()
    var kind = c[].bi.type(value)
    if kind is c[].array_cls:
        var view = value._mv
        if c[].bi.type(view) is not c[].bi.memoryview:
            raise Error("cross_vendor: Array storage must be a native memoryview")
        var shape = cv_sequence(value.shape, "Array shape")
        for i in range(len(shape)):  # driver-loop: over native Array shape metadata
            _ = cv_int(shape[i], "Array dimension")
        _ = cv_text(value.dtype, "Array dtype")
        _ = cv_text(value.order, "Array order")
        _ = cv_int(value._addr, "Array address")
        # Array.from_buffer uses a native raw-address memoryview (obj=None)
        # plus a separate pin retaining the original native exporter.
        var backing = c[].bi.type(view.obj)
        if view.obj is not None and backing is not c[].ndarray and backing is not c[].c_array and backing is not c[].bi.bytes and backing is not c[].bi.bytearray:
            raise Error("cross_vendor: Array backing must be native; Python buffer hooks are not accepted")
        var pin = value._pin
        if pin is not None:
            if c[].bi.type(pin) is not c[].bi.memoryview:
                raise Error("cross_vendor: Array pin must be a native memoryview")
            var pinned = c[].bi.type(pin.obj)
            if pin.obj is not None and pinned is not c[].ndarray and pinned is not c[].c_array and pinned is not c[].bi.bytes and pinned is not c[].bi.bytearray:
                raise Error("cross_vendor: Array pin must retain a native buffer exporter")
        return view
    if kind is not c[].ndarray and kind is not c[].c_array and kind is not c[].bi.memoryview and kind is not c[].bi.bytes and kind is not c[].bi.bytearray:
        raise Error("cross_vendor: buffer must be an exact native array or memoryview; Python buffer hooks are not accepted")
    if kind is c[].bi.memoryview:
        var backing = c[].bi.type(value.obj)
        if value.obj is not None and backing is not c[].ndarray and backing is not c[].c_array and backing is not c[].bi.bytes and backing is not c[].bi.bytearray:
            raise Error("cross_vendor: memoryview backing must be native; Python buffer hooks are not accepted")
    return c[].bi.memoryview(value)


def cv_path(value: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_TYPES.get_or_create_ptr()
    var kind = c[].bi.type(value)
    if kind is not c[].bi.str and kind is not c[].bi.bytes and kind is not c[].path and kind is not c[].pure_path:
        raise Error("cross_vendor: path requires str, bytes or a standard Path; Python path hooks are not accepted")
    if kind is c[].path or kind is c[].pure_path:
        var parts = PythonObject(None)
        try:
            parts = c[].bi.object.__getattribute__(value, "_raw_paths")
        except:
            parts = c[].bi.object.__getattribute__(value, "_parts")
        var components = cv_sequence(parts, "path storage")
        for i in range(len(components)):  # driver-loop: over stored native path components
            _ = cv_text(components[i], "path component")
    return native_path(value)


def cv_json(value: PythonObject, depth: Int = 0) raises:
    if depth > 64:
        raise Error("cross_vendor: metadata nesting exceeds 64")
    var bi = CROSS_VENDOR_TYPES.get_or_create_ptr()[].bi
    var kind = bi.type(value)
    if value is None or kind is bi.str or kind is bi.int or kind is bi.bool:
        return
    if kind is bi.float:
        _ = cv_float(value, "metadata number")
        return
    if kind is bi.list or kind is bi.tuple:
        for i in range(len(value)):  # driver-loop: over native metadata container fields
            cv_json(value[i], depth + 1)
        return
    if kind is bi.dict:
        var keys = bi.list(value)
        for i in range(len(keys)):  # driver-loop: over native metadata object fields
            _ = cv_text(keys[i], "metadata key")
            cv_json(value[keys[i]], depth + 1)
        return
    raise Error("cross_vendor: metadata requires native JSON values; Python protocols are not accepted")


def cv_state(value: PythonObject) raises:
    var bi = CROSS_VENDOR_TYPES.get_or_create_ptr()[].bi
    if bi.type(value) is not bi.dict:
        raise Error("cross_vendor: state requires a native dict")
    var keys = bi.list(value)
    for i in range(len(keys)):  # driver-loop: over state metadata and four native buffers
        var key = cv_text(keys[i], "state key")
        if key == "parameters" or key == "m" or key == "v" or key == "flags":
            _ = cv_buffer(value[keys[i]])
        else:
            cv_json(value[keys[i]])
