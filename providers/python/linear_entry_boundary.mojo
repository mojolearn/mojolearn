# SPDX-License-Identifier: Apache-2.0
"""Canonical import-time connections, with Python image pins only at UNPACK."""
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder, lookup_py_type_object
from std.reflection import reflect
from bindings.native_loaded_module import native_loaded_module
from bindings.native_owner_registry import borrowed_native_owner, register_native_owner
from bindings.py_balanced import py_type, py_call
from bindings.native_class_admission import dict_keys_admitted, dict_keys_admit
from bindings.py_type_registry import py_type_absent, py_type_attach
from providers.python.neural_descriptor_pyapi import (
    NeuralDescriptorProvider, _neural_descriptor_dictionary,
    _neural_descriptor_function_owner, neural_descriptor_connect_factory_once,
)
from core.entry_table import entry_table_same_call
from core.linear_entry_table import (
    LinearEntryKey, linear_table_entry, LINEAR_TABLE_ESTIMATORS,
    LINEAR_TABLE_X_LINEAR, LINEAR_TABLE_DECOMP, LINEAR_TABLE_METRICS,
)
from core.linear_fit_types import LINEAR_FIT_ABI, LINEAR_FIT_OPERATION_COUNT, RIDGE_FIT_ABI
from core.linear_metadata import linear_vendor_code
from x_decomp.descriptor_schema import XD_ABI_NAME, XD_OPERATION_COUNT
from x_metrics.descriptor_schema import METRICS_PROGRAM_ABI, MP_OPERATION_COUNT


@fieldwise_init
struct LinearEntryOwner(Movable, Writable):
    var key: LinearEntryKey
    var family: Int
    var module: PythonObject

    def write_to(self, mut writer: Some[Writer]):
        writer.write("LinearEntryOwner(family=", self.family, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def linear_entry_owner_pins(value: Int, index: Int) abi("C") -> Int:
    if not value or index < -1 or index > 0:
        return -1
    try:
        var owner = borrowed_native_owner(value)
        var native = owner.downcast_value_ptr[LinearEntryOwner]()
        var result = 1 if index == -1 else Int(native[].module._obj_ptr)
        _ = owner
        return result
    except:
        return -1


def register_linear_entry_owner(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[LinearEntryOwner]()
    except error:
        if not py_type_absent(error, String(reflect[LinearEntryOwner].name())):
            raise error
        _ = module.add_type[LinearEntryOwner]("_LinearEntryOwner")
        register_native_owner(lookup_py_type_object[LinearEntryOwner](), module.module, linear_entry_owner_pins)
        return
    py_type_attach(module.module, "_LinearEntryOwner", existing)
    register_native_owner(existing, module.module, linear_entry_owner_pins)


def linear_entry_connect(family_source: PythonObject, mode_entry: PythonObject,
    vendor_entry: PythonObject, factory: PythonObject) raises -> PythonObject:
    # Import-time only. Admit exact compiled entries and their common actual
    # extension module, then match the capsule endpoint to the published table.
    # The public operation never calls this factory or follows these functions.
    var bi = native_loaded_module("builtins")
    var function_type = py_type(bi.len)
    if (py_type(family_source) is not bi.int or py_type(mode_entry) is not function_type
            or py_type(vendor_entry) is not function_type or py_type(factory) is not function_type):
        raise Error("TypeError: linear entry connection requires compiled metadata and factory entries")
    var family = Int(py=family_source)
    var schema = String(LINEAR_FIT_ABI)
    var count = LINEAR_FIT_OPERATION_COUNT
    if family == LINEAR_TABLE_X_LINEAR:
        schema = String(RIDGE_FIT_ABI)
        count = 1
    elif family == LINEAR_TABLE_DECOMP:
        schema = String(XD_ABI_NAME)
        count = XD_OPERATION_COUNT
    elif family == LINEAR_TABLE_METRICS:
        schema = String(METRICS_PROGRAM_ABI)
        count = MP_OPERATION_COUNT
    elif family != LINEAR_TABLE_ESTIMATORS:
        raise Error("TypeError: unknown linear entry family")
    var raw_module = _neural_descriptor_function_owner(factory)
    if (not raw_module or _neural_descriptor_function_owner(mode_entry) != raw_module
            or _neural_descriptor_function_owner(vendor_entry) != raw_module):
        raise Error("TypeError: linear entry metadata must share its compiled factory image")
    var module = PythonObject(from_owned=Python().cpython().Py_NewRef(raw_module))
    if py_type(module) is not py_type(bi):
        raise Error("TypeError: linear entry factory must retain its exact extension module")
    var mode_value = py_call(mode_entry)
    var vendor_value = py_call(vendor_entry)
    if py_type(mode_value) is not bi.int or py_type(vendor_value) is not bi.str:
        raise Error("TypeError: linear entry metadata requires exact scalar values")
    var key = LinearEntryKey(Int(py=mode_value), linear_vendor_code(String(vendor_value)))
    var provider = neural_descriptor_connect_factory_once(factory, schema, count)
    var native = provider.downcast_value_ptr[NeuralDescriptorProvider]()
    var published = native[].entry
    if family == LINEAR_TABLE_ESTIMATORS:
        published = linear_table_entry[LINEAR_TABLE_ESTIMATORS](key, 0, schema, count)
    elif family == LINEAR_TABLE_X_LINEAR:
        published = linear_table_entry[LINEAR_TABLE_X_LINEAR](key, 0, schema, count)
    elif family == LINEAR_TABLE_DECOMP:
        published = linear_table_entry[LINEAR_TABLE_DECOMP](key, 0, schema, count)
    else:
        published = linear_table_entry[LINEAR_TABLE_METRICS](key, 0, schema, count)
    if native[].module is not module or not entry_table_same_call(published, native[].entry):
        raise Error("TypeError: linear entry image differs from the published native endpoint")
    var owner = PythonObject(alloc=LinearEntryOwner(key^, family, module))
    _ = provider
    return owner


def linear_entry_identity(entries: PythonObject, family: Int = LINEAR_TABLE_ESTIMATORS,
    weighted: Bool = False) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    if py_type(entries) is not bi.dict:
        raise Error("TypeError: linear entry metadata requires an exact dictionary")
    if not dict_keys_admitted(entries):
        _neural_descriptor_dictionary(entries, bi)
        dict_keys_admit(entries)
    var owner = entries["_native_weighted_connection" if weighted else "_native_entry_connection"]
    if py_type(owner) is not lookup_py_type_object[LinearEntryOwner]():
        raise Error("TypeError: linear native entries require the canonical connection owner")
    if owner.downcast_value_ptr[LinearEntryOwner]()[].family != family:
        raise Error("TypeError: linear native connection family differs")
    return owner


@fieldwise_init
struct LinearEntryConnection(Movable):
    var key: LinearEntryKey
    var images: PythonObject


def linear_entry_connection(entries: PythonObject, family: Int = LINEAR_TABLE_ESTIMATORS,
    weighted: Bool = False) raises -> LinearEntryConnection:
    var owner = linear_entry_identity(entries, family, weighted)
    var native = owner.downcast_value_ptr[LinearEntryOwner]()
    # This opaque owner has no Python-writable fields. Retaining it directly
    # pins the admitted exact image across dict mutation and GIL release.
    return LinearEntryConnection(native[].key.copy(), owner)
