# SPDX-License-Identifier: Apache-2.0
"""Linear archive class/format-metadata UNPACK and fitted-model PACK.

The typed schema, shape/refusal decisions and loaded scalar values live in
core.native_archive_linear. This module admits native class/NumPy metadata and
publishes the writable coefficient owner plus its native state pin.
"""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.cpython_table import cpython_table
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields
from bindings.native_buffer_owner_pack import native_storage_allocate
from bindings.classical_archive_output_door import archive_numeric_pack
from bindings.classical_arrays_door import classical_numpy
from bindings.classical_archive_door import _archive_class_name_unpack
from providers.python.linear_state_boundary import linear_state_publish
from core.native_archive_read import NativeArchiveDecoded
from core.native_archive_linear import NativeLinearArchivePlan, native_archive_needs_float_text


def archive_class_names_unpack(cls: PythonObject) raises -> List[String]:
    var native_type = py_type(py_type(PythonObject("")))
    var fields = native_type.__dict__
    var mro = py_call(fields["__mro__"].__get__, cls)
    var names = List[String](capacity=len(mro))
    for i in range(len(mro)):
        # Explicit immutable type slots bypass Python metaclass hooks.
        names.append(_archive_class_name_unpack(mro[i]))
    return names^


def _archive_metadata_keys(fields: PythonObject) raises:
    # Inspect keys without hash/equality lookup before reading a public
    # NumPy metadata dictionary, so foreign keys cannot invoke Python hooks.
    var strings = py_type(PythonObject(""))
    var keys = py_call(py_type(Python.list()), py_call(fields.keys))
    for i in range(len(keys)):
        if py_type(keys[i]) is not strings:
            raise Error("TypeError: archive scalar metadata requires native string keys")


def archive_print_mode_unpack(members: List[NativeArchiveDecoded], context: PythonObject) raises -> Int:
    if not classical_numpy(context) or not native_archive_needs_float_text(members):
        return 2147483647
    var native_type = py_type(py_type(PythonObject("")))
    var strings = py_type(PythonObject(""))
    var integers = py_type(PythonObject(0))
    var np = py_get(context["modules"], "numpy")
    if np is None or py_type(np) is not context["native_module_type"]:
        raise Error("TypeError: archive scalar metadata requires the connected native NumPy module")
    var numpy_fields = py_fields(np)
    _archive_metadata_keys(numpy_fields)
    var version = py_get(numpy_fields, "__version__")
    if py_type(version) is not strings:
        raise Error("TypeError: archive scalar metadata requires native NumPy version text")
    var parts = String(version).split(".")
    if len(parts) < 2:
        raise Error("TypeError: archive scalar metadata requires NumPy major.minor metadata")
    var release = Int(parts[0]) * 100 + Int(parts[1])
    var module = py_get(context["modules"], "numpy._core.arrayprint" if release >= 200 else "numpy.core.arrayprint")
    if module is None or py_type(module) is not context["native_module_type"]:
        raise Error("TypeError: archive scalar metadata requires native print configuration")
    var fields = py_fields(module)
    _archive_metadata_keys(fields)
    var options = py_get(fields, "_format_options")
    if options is None:
        var variable = py_get(fields, "format_options")
        var cls = py_type(variable)
        var type_fields = native_type.__dict__
        var flags = cpython_table()[].PyType_GetFlags(cls._obj_ptr)
        if flags & 512:
            raise Error("TypeError: archive scalar metadata requires a native print configuration ContextVar")
        var name = String(py=py_call(type_fields["__name__"].__get__, cls))
        var owner = String(py=py_call(native_type.__getattribute__, cls, "__module__"))
        if name != "ContextVar" or owner != "_contextvars":
            raise Error("TypeError: archive scalar metadata requires a native print configuration ContextVar")
        var entry = py_get(py_fields(cls), "get")
        if py_type(entry) is not py_type(context["archive_file_methods"][0]["getbuffer"]):
            raise Error("TypeError: archive scalar print metadata cannot invoke Python callbacks")
        options = py_call(entry, variable)
    if py_type(options) is not py_type(Python.dict()):
        raise Error("TypeError: archive scalar metadata requires a native print configuration dictionary")
    _archive_metadata_keys(options)
    var legacy = py_get(options, "legacy")
    if py_type(legacy) is not integers:
        raise Error("TypeError: archive scalar print configuration requires native integer legacy mode")
    var mode = Int(py=legacy)
    return min(mode, 202) if release < 203 else mode
