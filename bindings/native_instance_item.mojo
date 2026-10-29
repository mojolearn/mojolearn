# SPDX-License-Identifier: Apache-2.0
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_type


def native_instance_item(fields: PythonObject, name: PythonObject) raises -> PythonObject:
    # 2026-10-25: suspected Mojo/Modular interoperability bug. The former
    # `return fields[name]` in params_metadata_native._value and
    # public_attributes_native._lookup crashes or raises TypeError(False) for
    # an existing instance field. A standard-library-only reproduction fails
    # with Mojo 1.0.0 (ed45d567), macOS arm64, and CPython 3.13 and 3.14;
    # class, missing-field and descriptor controls pass. The exact compiler
    # or standard-library defect is not established. Original source/failures:
    # experiments/python_object_item_lookup/MODULAR_REPORT.md and
    # ~/mojolearn-evidence/native-params-crash-20261025/standalone-stdlib/.
    # Codex's automated cybersecurity screening blocked delegated debugging
    # turns and directed us to Daybreak access. Its exact trigger is unknown;
    # these interruptions are separate from the reproduced runtime failure.
    # Andrew explicitly authorized this documented native API repair after
    # review. It changes no compiler, standard library, binary or refcount
    # internals and introduces no Python fallback or callback.
    #
    # Caller must use _fields(): it admits an exact dict and checks EVERY
    # stored key is exact str before lookup, preventing hash/equality hooks.
    # The public entry or connected parameter schema supplies an exact str.
    # Class namespaces are mapping proxies and must not use this accessor.
    if py_type(fields) is not py_type(Python.dict()) or py_type(name) is not py_type(PythonObject("")):
        raise Error("TypeError: native instance lookup requires exact dictionary and string objects")
    ref cpy = Python().cpython()
    # The binding holds the GIL throughout. This Stable ABI call borrows its
    # result; immediately acquire exactly one owned reference before returning.
    # Py_NewRef is available at our Python 3.10 floor (GetItemRef needs 3.13).
    # https://docs.python.org/3/c-api/dict.html#c.PyDict_GetItemWithError
    # https://docs.python.org/3/c-api/refcounting.html#c.Py_NewRef
    var raw = cpy.lib.get_function[PyObjectPtr]("PyDict_GetItemWithError")(fields._obj_ptr, name._obj_ptr)
    if not raw:
        if cpy.PyErr_Occurred():
            raise cpy.unsafe_get_error()
        raise Error("KeyError: missing native instance field")
    return PythonObject(from_owned=cpy.Py_NewRef(raw))


def native_instance_find_admitted(fields: PythonObject, name: PythonObject) raises -> Tuple[Bool, PythonObject]:
    # Internal form of the explicitly authorized native dictionary repair above.
    # The caller has already admitted an exact dict, EVERY stored key as exact
    # str, and an exact str name. Do not use for class mappingproxy namespaces
    # or arbitrary mappings. Keep native_instance_item's defensive interface
    # for callers that still need its redundant exact-type checks.
    #
    # One dictionary probe replaces contains+getitem. A native Bool distinguishes
    # absence from an existing None; the native Tuple creates no Python wrapper.
    ref cpy = Python().cpython()
    var raw = cpy.lib.get_function[PyObjectPtr]("PyDict_GetItemWithError")(fields._obj_ptr, name._obj_ptr)
    if not raw:
        if cpy.PyErr_Occurred():
            raise cpy.unsafe_get_error()
        return (False, PythonObject(None))
    # Same Stable ABI/GIL contract as native_instance_item. Pin the mapping
    # through the owned-reference acquisition; no borrowed pointer escapes.
    var value = PythonObject(from_owned=cpy.Py_NewRef(raw))
    _ = fields
    return (True, value^)
