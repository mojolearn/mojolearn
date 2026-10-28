# SPDX-License-Identifier: Apache-2.0
# Exact Program storage admission and native construction. No mutable instance
# callable or Python descriptor selects a product implementation.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_setitem, py_get, py_contains
from bindings.native_loaded_module import native_loaded_module
from bindings.native_owner_lifetime import native_release_admit, native_release_weakrefs
from bindings.classical_attributes import classical_connect_numpy
from providers.python.autograd_import_metadata_pyapi import _fields
from providers.python.autograd_program_arrays_pyapi import program_admission_context
from providers.python.autograd_program_pyapi import _integer
from providers.python.neural_descriptor_pyapi import NativeDescriptorResource, NeuralDescriptorProvider
from autograd.model_descriptor import MODEL_ABI_NAME
from sequence.neural_model_dispatch import SESSION_CLOSE


def program_owner_fields(owner: PythonObject) raises -> PythonObject:
    var context = program_admission_context()["program_save_context"]
    var b = native_loaded_module("builtins")
    if py_type(context) is not b.dict or py_type(context["layouts"]) is not b.tuple:
        raise Error("TypeError: Program requires its connected native storage layout")
    var layouts = context["layouts"]
    for i in range(len(layouts)):  # small-loop(layouts: Program and ImportedProgram native storage)
        if py_type(owner) is layouts[i]["cls"]:
            return _fields(owner, layouts[i])
    raise Error("TypeError: Program operations require an exact connected Program or ImportedProgram")


def _retained(value: PythonObject, context: PythonObject) raises -> Bool:
    var owners = context["program_save_context"]["retained"]
    for i in range(len(owners)):  # driver-loop: fixed import-time owners cannot be finalized by a Session release
        if value is owners[i]:
            return True
    return False


def _release_session(value: PythonObject, context: PythonObject) raises:
    # A valid old Session owns native registry cleanup, not a Python __del__.
    # Check its exact layout, weakrefs and every stored owner before the last
    # Program reference can disappear. Arbitrary extra attributes remain
    # admitted transitively, rather than being silently released.
    var fields = _fields(value, context["program_save_context"]["session_layout"])
    native_release_weakrefs(value, context)
    var names = py_call(native_loaded_module("builtins").list, fields)
    for i in range(len(names)):  # driver-loop: complete Session ownership before replacing Program._s
        var item = fields[names[i]]
        if _retained(item, context):
            continue
        if String(py=names[i]) == "_lifetime":
            native_release_weakrefs(item, context)
            var resource = item.downcast_value_ptr[NativeDescriptorResource]()
            var provider = resource[].provider.downcast_value_ptr[NeuralDescriptorProvider]()
            if (not _retained(resource[].provider, context) or provider[].name != MODEL_ABI_NAME
                    or resource[].close_operation != SESSION_CLOSE):
                raise Error("TypeError: Program replacement requires its connected native Session lifetime")
        else:
            native_release_admit(item, context)


def program_release_fields(fields: PythonObject, names: PythonObject) raises:
    # Complete admission before execution or the first write. A previous
    # ndarray can retain object-dtype values, buffer owners and weakrefs.
    var context = program_admission_context()
    classical_connect_numpy(context)
    for i in range(len(names)):  # driver-loop: only fields this operation will replace
        if py_contains(fields, names[i]):
            native_release_admit(fields[names[i]], context)


def program_native_sequence(value: PythonObject) raises:
    var cls = py_type(value)
    if cls is not py_type(Python.list()) and cls is not py_type(Python.tuple()):
        raise Error("TypeError: Program metadata requires a native list or tuple")


def _integer_tuple(value: PythonObject) raises -> PythonObject:
    program_native_sequence(value)
    var result = Python.list()
    for i in range(len(value)):  # driver-loop: over native graph descriptor words
        py_call(result.append, _integer(value[i], "program descriptor", -9223372036854775807))
    return py_call(native_loaded_module("builtins").tuple, result)


def program_initialize_owner(target: PythonObject, session: PythonObject, info: PythonObject) raises:
    var fields = program_owner_fields(target)
    program_native_sequence(info)
    if len(info) < 4:
        raise Error("ValueError: Program metadata requires a handle, input/output shapes and parameters")
    var pid = _integer(info[0], "program", 1)
    var shapes = Python.list()
    for k in range(2):  # small-loop(2: input/output descriptor groups)
        program_native_sequence(info[k + 1])
        var dimensions = Python.list()
        for j in range(len(info[k + 1])):  # driver-loop: over graph input/output descriptors
            py_call(dimensions.append, _integer_tuple(info[k + 1][j]))
        py_call(shapes.append, py_call(native_loaded_module("builtins").tuple, dimensions))
    var parameters = _integer_tuple(info[3])
    var kinds = PythonObject(None)
    if len(info) > 4:
        kinds = _integer_tuple(info[4])
    else:
        var zeros = Python.list()
        for i in range(len(shapes[0])):  # driver-loop: over legacy value-only input descriptors
            py_call(zeros.append, 0)
        kinds = py_call(native_loaded_module("builtins").tuple, zeros)
    var context = program_admission_context()
    classical_connect_numpy(context)
    if py_contains(fields, "_s") and fields["_s"] is not session:
        var previous = fields["_s"]
        if py_type(previous) is context["ops_session_class"]:
            _release_session(previous, context)
        else:
            native_release_admit(previous, context)
    program_release_fields(fields, Python.tuple("_id", "input_shapes", "output_shapes", "params", "input_kinds"))
    # Commit only fully admitted metadata; these exact dictionary writes cannot
    # invoke a __setattr__ hook or a user-supplied field descriptor.
    py_setitem(fields, "_s", session)
    py_setitem(fields, "_id", pid)
    py_setitem(fields, "input_shapes", shapes[0])
    py_setitem(fields, "output_shapes", shapes[1])
    py_setitem(fields, "params", parameters)
    py_setitem(fields, "input_kinds", kinds)


def program_new_owner(session: PythonObject, info: PythonObject) raises -> PythonObject:
    var cls = program_admission_context()["program_save_context"]["layouts"][0]["cls"]
    var b = native_loaded_module("builtins")
    var layout = program_admission_context()["program_save_context"]["layouts"][0]
    if py_type(cls) is not b.type:
        raise Error("TypeError: Program construction requires a native class")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    if len(mro) != len(layout["mro"]):
        raise Error("TypeError: Program class ancestry changed")
    for i in range(len(mro)):  # driver-loop: over connected class ancestry before allocation
        if mro[i] is not layout["mro"][i]:
            raise Error("TypeError: Program class ancestry changed")
    var hooks = layout["hooks"]
    for i in range(len(hooks)):  # driver-loop: reject changed finalizers before creating an owner
        var row = py_call(b.type.__dict__["__dict__"].__get__, hooks[i][0])
        if py_get(row, hooks[i][1]) is not hooks[i][2]:
            raise Error("TypeError: Program construction refuses changed Python storage hooks")
    var out = py_call(b.object.__new__, cls)
    program_initialize_owner(out, session, info)
    return out
