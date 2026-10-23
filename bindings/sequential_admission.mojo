# SPDX-License-Identifier: Apache-2.0
# No Python-defined protocols are invoked by native Sequential execution.
# Admission compares exact type identities before reading attributes or
# walking containers, so subclass properties/iterators cannot run first.
from bindings.native_loaded_module import native_loaded_module
from std.python import Python, PythonObject
from bindings.native_torch_admission import native_torch_exact, native_torch_admit
from bindings.py_balanced import py_call, py_type, py_fields, py_contains, py_get


def _one_of(cls: PythonObject, types: PythonObject) raises -> Bool:
    for k in range(len(types)):  # driver-loop: over connected native ABI type identities
        if cls is types[k]:
            return True
    return False


def _ctypes_array_owner(owner_type: PythonObject, context: PythonObject) raises -> Bool:
    """A class made directly from the connected `ctypes.Array` that defines
    no Python buffer hook of its own: its buffer export is ctypes' C
    implementation. The library's uninitialized native output store
    (`_buffer._output_store`, a per-(code, length) ctypes array class) is
    one: Arrays from `astype`, the label encoders and the float64 to float32
    converters own one, and an estimator that kept such an Array as a fitted
    field could not be saved (ARIMA `llf_`, AMD verify of 7c6e0dade)."""
    var base = py_get(context, "ctypes_array_base")
    if base is None:
        return False
    var b = native_loaded_module("builtins")
    var bases = py_call(b.type.__dict__["__bases__"].__get__, owner_type)
    if len(bases) != 1 or bases[0] is not base:
        return False
    var own = py_call(b.type.__dict__["__dict__"].__get__, owner_type)
    return not py_contains(own, "__buffer__") and not py_contains(own, "__release_buffer__")


def sequential_integer(value: PythonObject, context: PythonObject) raises -> Bool:
    return _one_of(py_type(value), context["integer_types"])


def _ordinary_torch(cls: PythonObject, context: PythonObject) raises -> Bool:
    return native_torch_exact(cls)


def sequential_admit_value(value: PythonObject, context: PythonObject, depth: Int = 0) raises:
    if depth > 64:
        raise Error("TypeError: Sequential input containers exceed the supported nesting depth")
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is py_type(Python.list()) or cls is py_type(Python.tuple()):
        for k in range(len(value)):  # driver-loop: over native input container owners and metadata
            sequential_admit_value(value[k], context, depth + 1)
        return
    if _one_of(cls, context["mapping_types"]):
        if cls is context["defaultdict_class"]:
            # An exact C defaultdict can still call a Python default_factory
            # on a later missing lookup. Admit only connected C constructors.
            var factory = value.default_factory
            if factory is not None and not _one_of(factory, context["native_default_factories"]) and not _one_of(factory, context["scalar_types"]):
                raise Error("TypeError: Sequential defaultdict requires a native builtin default factory or None")
        var pairs = py_call(py_type(Python.list()), value.items())
        for k in range(len(pairs)):  # driver-loop: over native mapping keys and buffer owners
            sequential_admit_value(pairs[k][0], context, depth + 1)
            sequential_admit_value(pairs[k][1], context, depth + 1)
        return
    if cls is b.memoryview:
        # A custom Python exporter may carry release callbacks. The standard
        # buffer types connected at import expose their buffers in native C.
        var owner = value.obj
        var owner_type = py_type(owner)
        if not _one_of(owner_type, context["buffer_owner_types"]) and not _ctypes_array_owner(owner_type, context):
            raise Error("TypeError: Sequential requires a memoryview over a native buffer owner")
        return
    if _one_of(cls, context["native_value_types"]) or _one_of(cls, context["scalar_types"]):
        return
    if _ordinary_torch(cls, context):
        native_torch_admit(value)
        return
    raise Error("TypeError: Sequential requires native input types; Python-defined protocols and subclasses are not supported")


def sequential_admit(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises:
    var b = native_loaded_module("builtins")
    if operation == "load_checkpoint":
        if model is not context["sequential_class"]:
            raise Error("TypeError: Sequential checkpoint loading requires the native Sequential class")
    elif operation != "initialize":
        if py_type(model) is not context["sequential_class"]:
            raise Error("TypeError: Sequential requires its native public class, without Python subclass hooks")
        var fields = py_fields(model)
        var names = context["model_fields"]
        for k in range(len(names)):  # driver-loop: over model metadata and native buffer owners used by execution
            if py_contains(fields, names[k]):
                sequential_admit_value(fields[names[k]], context)
    sequential_admit_value(args, context)
