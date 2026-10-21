# SPDX-License-Identifier: Apache-2.0
# No Python-defined protocols are invoked by native Sequential execution.
# Admission compares exact type identities before reading attributes or
# walking containers, so subclass properties/iterators cannot run first.
from std.python import Python, PythonObject


def _one_of(cls: PythonObject, types: PythonObject) raises -> Bool:
    for k in range(len(types)):  # driver-loop: over connected native ABI type identities
        if cls is types[k]:
            return True
    return False


def sequential_integer(value: PythonObject, context: PythonObject) raises -> Bool:
    return _one_of(Python.import_module("builtins").type(value), context["integer_types"])


def _ordinary_torch(cls: PythonObject, context: PythonObject) raises -> Bool:
    # Reading an already imported module's stored dictionary neither imports
    # torch nor invokes module __getattr__. Tensor/Parameter remain supported;
    # user tensor subclasses are not admitted through their protocol hooks.
    var b = Python.import_module("builtins")
    var modules = context["modules"]
    var torch = modules.get("torch")
    if torch is not None and b.type(torch) is context["module_type"]:
        var fields = b.object.__getattribute__(torch, "__dict__")
        if cls is fields.get("Tensor"):
            return True
    var parameter = modules.get("torch.nn.parameter")
    if parameter is not None and b.type(parameter) is context["module_type"]:
        var fields = b.object.__getattribute__(parameter, "__dict__")
        if cls is fields.get("Parameter"):
            return True
    return False


def sequential_admit_value(value: PythonObject, context: PythonObject, depth: Int = 0) raises:
    if depth > 64:
        raise Error("TypeError: Sequential input containers exceed the supported nesting depth")
    var b = Python.import_module("builtins")
    var cls = b.type(value)
    if cls is b.list or cls is b.tuple:
        for k in range(len(value)):  # driver-loop: over native input container owners and metadata
            sequential_admit_value(value[k], context, depth + 1)
        return
    if _one_of(cls, context["mapping_types"]):
        var pairs = b.list(value.items())
        for k in range(len(pairs)):  # driver-loop: over native mapping keys and buffer owners
            sequential_admit_value(pairs[k][0], context, depth + 1)
            sequential_admit_value(pairs[k][1], context, depth + 1)
        return
    if cls is b.memoryview:
        # A custom Python exporter may carry release callbacks. The standard
        # buffer types connected at import expose their buffers in native C.
        var owner = value.obj
        if not _one_of(b.type(owner), context["buffer_owner_types"]):
            raise Error("TypeError: Sequential requires a memoryview over a native buffer owner")
        return
    if _one_of(cls, context["native_value_types"]) or _one_of(cls, context["scalar_types"]) or _ordinary_torch(cls, context):
        return
    raise Error("TypeError: Sequential requires native input types; Python-defined protocols and subclasses are not supported")


def sequential_admit(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises:
    var b = Python.import_module("builtins")
    if operation == "load_checkpoint":
        if model is not context["sequential_class"]:
            raise Error("TypeError: Sequential checkpoint loading requires the native Sequential class")
    elif operation != "initialize":
        if b.type(model) is not context["sequential_class"]:
            raise Error("TypeError: Sequential requires its native public class, without Python subclass hooks")
        var fields = b.object.__getattribute__(model, "__dict__")
        var names = context["model_fields"]
        for k in range(len(names)):  # driver-loop: over model metadata and native buffer owners used by execution
            if Bool(py=fields.__contains__(names[k])):
                sequential_admit_value(fields[names[k]], context)
    sequential_admit_value(args, context)
