# SPDX-License-Identifier: Apache-2.0
"""UNPACK import-time linear table keys and retain selected extension images."""
from std.python import PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_type
from bindings.native_class_admission import dict_keys_admitted, dict_keys_admit
from providers.python.neural_descriptor_pyapi import _neural_descriptor_dictionary
from core.linear_entry_table import LinearEntryKey


def linear_entry_key(entries: PythonObject) raises -> LinearEntryKey:
    var bi = native_loaded_module("builtins")
    if py_type(entries) is not bi.dict:
        raise Error("TypeError: linear entry metadata requires an exact dictionary")
    if not dict_keys_admitted(entries):
        _neural_descriptor_dictionary(entries, bi)
        dict_keys_admit(entries)
    var key = entries["_native_entry_key"]
    if py_type(key) is not bi.tuple or len(key) != 2:
        raise Error("TypeError: linear native entry key requires an exact pair")
    var mode = key[0]
    var vendor = key[1]
    if py_type(mode) is not bi.int or py_type(vendor) is not bi.int:
        raise Error("TypeError: linear native entry key requires exact integers")
    # Image ownership remains in entries['_native_entry_images']; callers
    # retain entries until native execution returns, including error paths.
    return LinearEntryKey(Int(py=mode), Int(py=vendor))


@fieldwise_init
struct LinearEntryConnection(Movable):
    var key: LinearEntryKey
    var images: PythonObject


def linear_entry_connection(entries: PythonObject) raises -> LinearEntryConnection:
    var key = linear_entry_key(entries)
    var pins = entries["_native_entry_images"]
    var bi = native_loaded_module("builtins")
    if py_type(pins) is not bi.tuple or len(pins) == 0:
        raise Error("TypeError: linear native entries require image pins")
    # Capture this immutable tuple before GIL release. Retaining the mutable
    # connection dict alone would not pin the old images across reconnection.
    return LinearEntryConnection(key^, pins)
