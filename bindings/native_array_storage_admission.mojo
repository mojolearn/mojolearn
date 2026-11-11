# SPDX-License-Identifier: Apache-2.0
"""Shared native Array storage admission for construction and address reads."""
from std.python import Python, PythonObject
from bindings.cpython_table import cpython_table
from bindings.native_loaded_module import native_loaded_module, native_namespace_string_keys, native_mapping_string_keys
from bindings.native_class_admission import dict_keys_admitted, dict_keys_admit
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields
from bindings.p3_array_arena_records import (
    P3_NATIVE_ARRAY_STORAGE_LAYOUT, P3_DICT_ARRAY_STORAGE,
    p3_layout_admitted, p3_layout_admit, p3_dict_admitted, p3_dict_admit,
)


def array_storage_namespace(cls: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var fields = py_call(b.type.__dict__["__dict__"].__get__, cls)
    if not native_mapping_string_keys(fields, b.str):
        raise Error("TypeError: Array storage classes require native string keys")
    return fields


def array_storage_class(cls: PythonObject, base: PythonObject) raises -> Bool:
    var b = native_loaded_module("builtins")
    if py_type(cls) is not b.type:
        return False
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
    for i in range(len(ancestry)):  # driver-loop: native Array class ancestry
        if ancestry[i] is base:
            return True
    return False


def array_storage_admit(cls: PythonObject, base: PythonObject, slots: PythonObject,
                        who: String = "Array.from_buffer") raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(slots) is not b.dict:
        raise Error("TypeError: Array storage descriptors require a native dictionary")
    if py_type(cls) is not b.type or py_type(base) is not b.type:
        raise Error("TypeError: " + who + " builds a mojolearn Array class")
    # A class stamp alone cannot admit a changed descriptor registry. Both
    # native mutation stamps must match before reusing this exact check.
    if (p3_dict_admitted(P3_DICT_ARRAY_STORAGE, slots)
            and p3_layout_admitted(P3_NATIVE_ARRAY_STORAGE_LAYOUT, cls, slots)):
        return slots
    if not native_namespace_string_keys(slots._obj_ptr, b.str):
        raise Error("TypeError: Array storage descriptors require native string keys")
    var expected: List[String] = ["_store", "_base", "_pin", "_mv", "_addr", "_readonly",
        "shape", "dtype", "ndim", "size", "nbytes", "itemsize", "strides", "order"]
    if len(slots) != len(expected):
        raise Error("TypeError: Array native storage descriptors changed")
    for name in expected:  # small-loop: complete native Array storage schema
        if not py_contains(slots, PythonObject(name)):
            raise Error("TypeError: Array native storage descriptors changed")
    if not array_storage_class(cls, base):
        raise Error("TypeError: " + who + " builds a mojolearn Array class")
    var base_fields = array_storage_namespace(base)
    var member_type = native_loaded_module("types").MemberDescriptorType
    for name in expected:  # small-loop: retained Array member descriptors
        var key = PythonObject(name)
        var descriptor = slots[key]
        if (py_type(descriptor) is not member_type or descriptor.__objclass__ is not base
                or descriptor.__name__ != key or py_get(base_fields, key) is not descriptor):
            raise Error("TypeError: Array native storage descriptors changed")
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
    for i in range(len(ancestry)):  # driver-loop: passive subclass storage and lifetime
        var fields = array_storage_namespace(ancestry[i])
        if py_contains(fields, "__getattr__") or py_contains(fields, "__del__"):
            raise Error("TypeError: " + who + " does not invoke Python metadata or lifetime hooks")
        for name in ["__getattribute__", "__setattr__", "__new__"]:  # small-loop: native storage access protocols
            var member = py_get(fields, PythonObject(name))
            if member is not None and member is not b.object.__dict__[name]:
                raise Error("TypeError: " + who + " requires native allocation and storage access")
        if ancestry[i] is base:
            continue
        for name in expected:  # small-loop: subclass field overrides before slot access
            if py_contains(fields, PythonObject(name)):
                raise Error("TypeError: " + who + " requires its inherited native storage slots")
    p3_layout_admit(P3_NATIVE_ARRAY_STORAGE_LAYOUT, cls, slots)
    p3_dict_admit(P3_DICT_ARRAY_STORAGE, slots)
    return slots


def array_storage_address(value: PythonObject, base: PythonObject, slots: PythonObject) raises -> PythonObject:
    _ = array_storage_admit(py_type(value), base, slots, "Array native address")
    ref cpy = Python().cpython()
    var name = PythonObject("_addr")
    var raw = cpython_table()[].PyObject_GenericGetAttr(value._obj_ptr, name._obj_ptr)
    if not raw:
        raise cpy.unsafe_get_error()
    var address = PythonObject(from_owned=raw)
    if py_type(address) is not native_loaded_module("builtins").int:
        raise Error("TypeError: Array native address requires an exact integer")
    _ = value
    return address


def array_storage_context() raises -> PythonObject:
    # Only exact Array callers reach this overload. Keep the import-connected
    # full slot inventory, independent of mutable Python class attributes.
    var module = native_loaded_module("mojolearn._array")
    var context = py_get(py_fields(module), "_ARRAY_CONTEXT")
    # Module admission pins the context object, not the keys inside its
    # mutable dictionary. Admit those before a keyed read can compare a
    # colliding foreign key. The shared mutation stamp avoids a repeated walk.
    var b = native_loaded_module("builtins")
    if py_type(context) is not b.dict:
        raise Error("TypeError: Array native address context requires an exact dictionary")
    if not dict_keys_admitted(context):
        if not native_namespace_string_keys(context._obj_ptr, b.str):
            raise Error("TypeError: Array native address context requires native string keys")
        dict_keys_admit(context)
    return context


def array_storage_address(value: PythonObject) raises -> PythonObject:
    var context = array_storage_context()
    return array_storage_address(value, context["array_class"], context["from_buffer_slots"])
