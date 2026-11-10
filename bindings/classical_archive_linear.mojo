# SPDX-License-Identifier: Apache-2.0
"""Linear archive class/format-metadata UNPACK and fitted-model PACK.

The typed schema, shape/refusal decisions and loaded scalar values live in
core.native_archive_linear. This module admits native class/NumPy metadata and
publishes the existing writable coefficient object plus its native state pin.
"""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.cpython_table import cpython_table
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields
from bindings.native_buffer_owner import native_storage_allocate
from bindings.classical_archive_output import archive_numeric_pack
from bindings.classical_arrays import classical_numpy, classical_is_array
from bindings.classical_archive import _archive_unpack_member, _archive_unpack_unicode
from providers.python.linear_state_boundary import linear_state_publish
from core.native_archive_span import ArchiveSpan
from core.native_archive_read import NativeArchiveDecoded, NativeArchiveText
from core.native_archive_linear import NativeLinearArchivePlan, native_archive_needs_float_text


def archive_class_names_unpack(cls: PythonObject) raises -> List[String]:
    var native_type = py_type(py_type(PythonObject("")))
    var fields = native_type.__dict__
    var mro = py_call(fields["__mro__"].__get__, cls)
    var names = List[String](capacity=len(mro))
    for i in range(len(mro)):
        # Explicit immutable type slots bypass Python metaclass hooks.
        names.append(String(py=py_call(fields["__name__"].__get__, mro[i])))
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


def archive_linear_pack(cls: PythonObject, plan: NativeLinearArchivePlan,
                         members: List[NativeArchiveDecoded], ridge: Bool, context: PythonObject,
                         var coefficients: PythonObject) raises -> PythonObject:
    if coefficients is None:
        coefficients = archive_numeric_pack(members[plan.coefficients], context)
    var model = native_storage_allocate(cls)
    ref cpy = Python().cpython()
    var raw = cpython_table()[].PyObject_GenericGetDict(model._obj_ptr, PyObjectPtr())
    if not raw:
        raise cpy.unsafe_get_error()
    var fields = PythonObject(from_owned=raw)
    py_setitem(fields, "fit_intercept", plan.fit_intercept)
    py_setitem(fields, "numeric_mode", PythonObject(plan.mode))
    py_setitem(fields, "coef_", coefficients)
    py_setitem(fields, "intercept_", plan.intercept)
    py_setitem(fields, "n_features_in_", plan.features)
    if ridge:
        py_setitem(fields, "alpha", plan.alpha)
        py_setitem(fields, "solver", "auto")
        py_setitem(fields, "normalize", False)
        py_setitem(fields, "solver_", "eig")
    linear_state_publish(model, coefficients, PythonObject(None), plan.features, plan.intercept, 0.0, False)
    return model


def archive_linear_fields_unpack(arrays: PythonObject, context: PythonObject,
                                  mut pins: List[PythonObject]) raises -> List[NativeArchiveDecoded]:
    """Compatibility UNPACK for the shared host-archive owner's result map.

    The generic host archive owner still exposes this map at its format
    boundary. Direct Ridge/OLS load bypasses it and consumes native members.
    """
    if py_type(arrays) is not py_type(Python.dict()):
        raise Error("TypeError: archive model fields require a native dictionary")
    _archive_metadata_keys(arrays)
    var names: List[String] = ["format", "estimator", "meta", "numeric_mode", "coef", "intercept", "alpha"]
    var result = List[NativeArchiveDecoded]()
    for name in names:
        var value = py_get(arrays, PythonObject(name))
        if value is None:
            continue
        pins.append(value)
        var member = NativeArchiveDecoded()
        member.name = name
        if classical_is_array(value, context) or (classical_numpy(context) and py_type(value) is context["ndarray"]):
            var admitted = _archive_unpack_member(name, value, context, pins)
            admitted.prepare()
            member.layout.dtype = admitted.dtype
            member.layout.shape = admitted.shape.copy()
            member.layout.width = admitted.width
            member.layout.count = admitted.size // admitted.width
            member.layout.payload = ArchiveSpan(admitted.source, admitted.size)
            swap(member.owned, admitted.owned)
        else:
            while py_type(value) is py_type(Python.list()):
                if len(value) == 0:
                    break
                value = value[0]
            member.layout.count = 0 if py_type(value) is py_type(Python.list()) else 1
            member.layout.kind = 1 if py_type(value) is py_type(PythonObject("")) else 2
            var text = NativeArchiveText()
            if member.layout.count:
                if member.layout.kind == 1:
                    text.points = _archive_unpack_unicode(value, context)
                elif py_type(value) is context["archive_bytes_type"]:
                    var address = cpython_table()[].PyBytes_AsString(value._obj_ptr)
                    text.bytes = ArchiveSpan(address, len(value))
                else:
                    raise Error("TypeError: archive text fields require native strings, bytes or list containers")
                member.texts.append(text^)
        result.append(member^)
    return result^
