# SPDX-License-Identifier: Apache-2.0
# Session admission uses the common provider and retains its exact registry.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_fields, py_get, py_setitem
from bindings.native_loaded_module import native_loaded_module
from providers.python.autograd_program_arrays_pyapi import initialize_program_arrays, program_admission_context, program_numeric_scalar
from providers.python.neural_descriptor_pyapi import (
    neural_descriptor_connect, neural_descriptor_entry, neural_descriptor_matches,
    neural_descriptor_execute, neural_descriptor_adopt, NeuralDescriptorProvider,
)
from autograd.model_descriptor import MODEL_ABI_NAME
from sequence.neural_model_dispatch import MODEL_OPERATION_COUNT, SESSION_OPEN, SESSION_CLOSE


def _session_string_keys(fields: PythonObject) raises:
    var b = native_loaded_module("builtins")
    var keys = py_call(b.list, fields.keys())
    for i in range(len(keys)):  # driver-loop: over native Session metadata dictionary keys
        if py_type(keys[i]) is not py_type(PythonObject("")):
            raise Error("TypeError: Session metadata requires exact string keys")


def ops_session_fields(session: PythonObject) raises -> PythonObject:
    var context = program_admission_context()
    var cls = context["ops_session_class"]
    if py_type(session) is not cls:
        raise Error("TypeError: Session requires its connected native owner class; Python subclass hooks are unsupported")
    var b = native_loaded_module("builtins")
    var namespace = py_call(b.type.__dict__["__dict__"].__get__, cls)
    var layout = context["ops_session_layout"]
    for i in range(len(layout)):  # driver-loop: over connected Session descriptor identities
        if py_get(namespace, layout[i][0]) is not layout[i][1]:
            raise Error("TypeError: Session owner has a replaced Python attribute hook or storage descriptor")
    var fields = py_fields(session)
    if py_type(fields) is not py_type(Python.dict()):
        raise Error("TypeError: Session requires a native instance dictionary")
    _session_string_keys(fields)
    return fields


def ops_session_initialize_py(session: PythonObject, record: PythonObject, mode: PythonObject,
                              connections: PythonObject, backend: PythonObject, program_class: PythonObject) raises -> PythonObject:
    if session is None:
        # Import-time setup: the sixth slot supplies the common admission
        # context, and the fourth supplies this exact compiled module.
        initialize_program_arrays(program_class)
        if connections is None:
            return PythonObject(None)  # retained private setup ABI
        return neural_descriptor_connect(neural_descriptor_entry(connections, "model_native_api_v1"),
                                          String(MODEL_ABI_NAME), MODEL_OPERATION_COUNT)
    var context = program_admission_context()
    var fields = ops_session_fields(session)
    if program_class is not context["ops_program_class"]:
        raise Error("TypeError: Session requires its connected native Program class")
    var b = native_loaded_module("builtins")
    var requested = mode
    if requested is None:
        requested = context["backend_fields"]["_DEFAULT_MODE"]
        if requested is None:
            requested = context["backend_fields"]["_SELECTED"]
    if py_type(requested) is not py_type(PythonObject("")):
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    var normalized = py_call(py_call(requested.strip).lower)
    var key = String(normalized)
    if key != "identical" and key != "fast":
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    if py_type(connections) is not py_type(Python.dict()):
        raise Error("TypeError: Session connections require a native dictionary")
    _session_string_keys(connections)
    var entry = connections[normalized]
    if py_type(entry) is py_type(PythonObject("")):
        raise Error(String(entry))
    if py_type(entry) is not context["ops_connection_type"]:
        raise Error("TypeError: Session requires an admitted native connection record")
    var connection = py_fields(entry)
    _session_string_keys(connection)
    var module = connection["native_module"]
    var provider = connection["native_provider"]
    if not neural_descriptor_matches(provider, module):
        raise Error("TypeError: Session provider belongs to another compiled module")
    if provider.downcast_value_ptr[NeuralDescriptorProvider]()[].factory is not neural_descriptor_entry(module, "model_native_api_v1"):
        raise Error("TypeError: Session provider factory changed; reconnect the compiled binding")
    # Admit the selected public doors once, before any session is allocated.
    # These are retained entry identities; no Python/extension callable runs.
    var names = Python.tuple("call", "execute", "inspect", "control", "fit")
    var symbols = Python.tuple("ops_session_call", "ops_program_execute", "ops_program_inspect", "ops_program_control", "ops_program_fit_native")
    for i in range(len(names)):  # driver-loop: over fixed Session public entry identities
        if connection[names[i]] is not neural_descriptor_entry(module, String(symbols[i])):
            raise Error("TypeError: Session connection has a replaced compiled entry")
    if not program_numeric_scalar(record):
        raise Error("TypeError: Session record requires a native boolean or numeric scalar")
    var recording = Bool(py=record)
    # Empty shell first, matching the former allocation/error ownership rule.
    # Direct dictionary writes cannot invoke a Python __setattr__ callback.
    py_setitem(fields, "_id", PythonObject(None))
    py_setitem(fields, "_session_native", connection["call"])
    py_setitem(fields, "record", recording)
    py_setitem(fields, "numeric_mode", mode)
    py_setitem(fields, "_b", connection["binding"])
    py_setitem(fields, "_program_execute", connection["execute"])
    py_setitem(fields, "_program_inspect", connection["inspect"])
    py_setitem(fields, "_program_control", connection["control"])
    py_setitem(fields, "_program_fit", connection["fit"])
    py_setitem(fields, "_program_class", program_class)
    py_setitem(fields, "_distributed_binding", connection["distributed"])
    py_setitem(fields, "_opt_kind", Python.dict())
    var scope = List[Int](length=2, fill=0)
    var addresses: List[Int] = [Int(scope.unsafe_ptr())]
    var options: List[Int] = [1 if recording else 0, 1 if key == "identical" else 0]
    # Registry mutation retains the GIL just as the former native open entry;
    # numerical execution is unchanged and not performed by this operation.
    var sid = neural_descriptor_execute(provider, SESSION_OPEN, addresses, options, List[Float64](), False)
    var lifetime = neural_descriptor_adopt(provider, sid, SESSION_CLOSE, scope^)
    py_setitem(fields, "_lifetime", lifetime)
    py_setitem(fields, "_id", sid)
    return PythonObject(None)
