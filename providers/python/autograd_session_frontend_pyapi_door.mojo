# SPDX-License-Identifier: Apache-2.0
# Session admission uses the common provider and retains its exact registry.
from core.mode_answer import MODE_FAMILY_NEURAL, answer_mode_word
from core.linear_route import (
    LINEAR_DEVICE_CPU, LINEAR_DEVICE_INSTALL, linear_device_code, linear_device_refusal, linear_device_installed,
)
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_fields, py_get, py_setitem
from bindings.native_loaded_module_door import native_loaded_module, NativeModules, shared_dict_stamped, shared_dict_epoch, shared_dict_watch
from core.shared_slot import SharedSlot
from bindings.native_class_admission_door import class_record, class_admit, dict_keys_admitted, dict_keys_admit
from providers.python.autograd_program_arrays_pyapi_door import initialize_program_arrays, program_admission_context, program_numeric_scalar, program_builtin_types
from providers.python.neural_descriptor_pyapi_door import (
    neural_descriptor_connect, neural_descriptor_entry, neural_descriptor_matches,
    neural_descriptor_execute, neural_descriptor_adopt, NeuralDescriptorProvider,
)
from autograd.model_descriptor import MODEL_ABI_NAME
from sequence.neural_model_dispatch import MODEL_OPERATION_COUNT, SESSION_OPEN, SESSION_CLOSE
from bindings.autograd_keys_unpack import autograd_keys
from bindings.py_balanced import py_item


def _session_string_keys(fields: PythonObject) raises:
    var b = native_loaded_module("builtins")
    var keys = py_call(b.list, fields.keys())
    for i in range(len(keys)):  # driver-loop: over native Session metadata dictionary keys
        if py_type(keys[i]) is not py_type(PythonObject("")):
            raise Error("TypeError: Session metadata requires exact string keys")


struct _SessionOwnerWords(Defaultable, Movable):
    # Lane fix-program-run-per-call (2026-11-14): the connected Session class
    # and layout tuple as addresses, with the admission context dictionary
    # they were read from and its watch epoch (0 words: not resolved).
    var context: Int
    var epoch: Int
    var cls: Int
    var layout: Int

    def __init__(out self):
        self.context = 0
        self.epoch = 0
        self.cls = 0
        self.layout = 0


comptime _SESSION_OWNER_WORDS = SharedSlot[_SessionOwnerWords, "MojolearnSessionOwnerWordsV1"]


def _session_instance_fields(session: PythonObject) raises -> PythonObject:
    var fields = py_fields(session)
    # lane fix-wave2-leftovers: the dict type connected once (was a new
    # dict built per Session operation to read its type)
    if py_type(fields) is not program_builtin_types()[].dict_type:
        raise Error("TypeError: Session requires a native instance dictionary")
    if not dict_keys_admitted(fields):
        _session_string_keys(fields)
        dict_keys_admit(fields)
    return fields


def ops_session_fields(session: PythonObject) raises -> PythonObject:
    var context = program_admission_context()
    var mods = NativeModules()
    # Lane fix-program-run-per-call (2026-11-14): every Session operation
    # (about 25 per ops-session-step round, one per Program.run admission)
    # read `ops_session_class` and `ops_session_layout` from the admission
    # context by key (profile r5 ops-session-step: ops_session_fields 11%).
    # They are resolved once and kept as addresses while the context
    # dictionary is unchanged (its watch epoch; CPython 3.10 / 3.11: its
    # version tag). The class identity and its layout record are still
    # compared per call; any miss takes the full check below.
    var words = _SESSION_OWNER_WORDS.get_or_create_ptr()
    var address = Int(context._obj_ptr)
    var cls = py_type(session)
    if (words[].cls != 0 and words[].context == address and Int(cls._obj_ptr) == words[].cls
            and shared_dict_stamped() and shared_dict_epoch(address) == words[].epoch):
        var kept = class_record(mods, _SESSION_LAYOUT_SLOT, cls)
        if kept is not None and Int(py=kept) == words[].layout:
            return _session_instance_fields(session)
    var nk = autograd_keys()
    var connected = py_item(context, nk[].k_ops_session_class)
    if cls is not connected:
        raise Error("TypeError: Session requires its connected native owner class; Python subclass hooks are unsupported")
    var layout = py_item(context, nk[].k_ops_session_layout)
    # The layout check depends only on the class rows: keep it per class
    # under the type watcher (bindings/native_class_admission.mojo); the
    # instance key check per dictionary under the dict watcher. Python
    # 3.10/3.11 keep both full checks on every operation.
    var record = class_record(mods, _SESSION_LAYOUT_SLOT, cls)
    if record is None or Int(py=record) != Int(layout._obj_ptr):
        var b = native_loaded_module("builtins")
        var namespace = py_call(py_item(b.type.__dict__, nk[].k___dict__).__get__, cls)
        for i in range(len(layout)):  # driver-loop: over connected Session descriptor identities
            if py_get(namespace, layout[i][0]) is not layout[i][1]:
                raise Error("TypeError: Session owner has a replaced Python attribute hook or storage descriptor")
        class_admit(mods, _SESSION_LAYOUT_SLOT, cls, PythonObject(Int(layout._obj_ptr)))
    var fields = _session_instance_fields(session)
    if shared_dict_watch(context._obj_ptr):
        words[].context = address
        words[].epoch = shared_dict_epoch(address)
        words[].cls = Int(cls._obj_ptr)
        words[].layout = Int(layout._obj_ptr)
    return fields


# Class-watcher slot of the Session layout verdict (payload: the address of
# the connected layout tuple it was checked against; the context keeps it).
comptime _SESSION_LAYOUT_SLOT = "p3_autograd_session_layout_v1"


def _session_device_unpack(value: PythonObject, b: PythonObject) raises -> Int:
    """The device code of a non-None constructor value: exact `'gpu'` or
    `'cpu'`; anything else is refused in the pilot's words
    (core/linear_route.mojo), before a session exists."""
    if py_type(value) is b.str:
        var code = linear_device_code(String(value))
        if code >= 0:
            return code
    linear_device_refusal(String(py_call(b.repr, value)))
    return LINEAR_DEVICE_INSTALL


def ops_session_initialize_py(session: PythonObject, record: PythonObject, mode: PythonObject,
                              connections: PythonObject, device: PythonObject, program_class: PythonObject) raises -> PythonObject:
    # `device` (None, 'gpu' or 'cpu') takes the slot of the former `backend`
    # argument, which no path read. It is read ONCE, here: the session keeps
    # the connection of the selected compiled image (its registry, its public
    # provider, its program entries), and every later operation, and every
    # Program made from the session, runs through that connection. A session
    # of the CPU image on a GPU installation is the same "session owned by
    # another compiled image" the public door already forwards
    # (autograd_session_public_request_pyapi_pack.mojo); no public call tests
    # the device.
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
    var nk = autograd_keys()
    if program_class is not py_item(context, nk[].k_ops_program_class):
        raise Error("TypeError: Session requires its connected native Program class")
    var b = native_loaded_module("builtins")
    var requested = mode
    if requested is None:
        requested = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
        if requested is None:
            requested = py_item(py_item(context, nk[].k_backend_fields), nk[].k__SELECTED)
    if py_type(requested) is not py_type(PythonObject("")):
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    var normalized = py_call(py_call(requested.strip).lower)
    var key = String(normalized)
    if key != "identical" and key != "fast":
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    answer_mode_word(key, MODE_FAMILY_NEURAL)
    if py_type(connections) is not py_type(Python.dict()):
        raise Error("TypeError: Session connections require a native dictionary")
    _session_string_keys(connections)
    var entry = PythonObject(None)
    if device is None:
        # The default: this installation's device, the table of the mode.
        # Nothing about the device is read.
        entry = connections[normalized]
    else:
        # The CPU connection exists only where the installation has a GPU set
        # (python/mojolearn/ops.py connects it beside the mode tables); in a
        # CPU-only process the mode table already is the CPU one.
        var cpu = py_get(context, nk[].k_ops_session_cpu_connection)
        var resolved = linear_device_installed(_session_device_unpack(device, b), 0 if cpu is None else 1)
        if resolved == LINEAR_DEVICE_CPU and cpu is not None:
            if key != "identical":
                raise Error("ValueError: device='cpu' runs in identical mode only; numeric_mode='fast' is for Apple GPUs")
            entry = cpu
        else:
            entry = connections[normalized]
    if py_type(entry) is py_type(PythonObject("")):
        raise Error(String(entry))
    if py_type(entry) is not py_item(context, nk[].k_ops_connection_type):
        raise Error("TypeError: Session requires an admitted native connection record")
    var connection = py_fields(entry)
    _session_string_keys(connection)
    var module = py_item(connection, nk[].k_native_module)
    var provider = py_item(connection, nk[].k_native_provider)
    if not neural_descriptor_matches(provider, module):
        raise Error("TypeError: Session provider belongs to another compiled module")
    if provider.downcast_value_ptr[NeuralDescriptorProvider]()[].factory is not neural_descriptor_entry(module, "model_native_api_v1"):
        raise Error("TypeError: Session provider factory changed; reconnect the compiled binding")
    var public_provider = py_item(connection, nk[].k_public_provider)
    if not neural_descriptor_matches(public_provider, module):
        raise Error("TypeError: Session public provider belongs to another compiled module")
    var public_owner = public_provider.downcast_value_ptr[NeuralDescriptorProvider]()
    if public_owner[].record[0].name != "mojolearn.session.public.v1" or public_owner[].factory is not neural_descriptor_entry(module, "session_public_api_v1"):
        raise Error("TypeError: Session public provider requires its connected native ABI")
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
    py_setitem(fields, nk[].k__id, PythonObject(None))
    py_setitem(fields, nk[].k__session_native, py_item(connection, nk[].k_call))
    py_setitem(fields, nk[].k__public_provider, py_item(connection, nk[].k_public_provider))
    py_setitem(fields, nk[].k_record, recording)
    py_setitem(fields, nk[].k_numeric_mode, mode)
    py_setitem(fields, nk[].k_device, device)
    py_setitem(fields, nk[].k__b, py_item(connection, nk[].k_binding))
    py_setitem(fields, nk[].k__program_execute, py_item(connection, nk[].k_execute))
    py_setitem(fields, nk[].k__program_inspect, py_item(connection, nk[].k_inspect))
    py_setitem(fields, nk[].k__program_control, py_item(connection, nk[].k_control))
    py_setitem(fields, nk[].k__program_fit, py_item(connection, nk[].k_fit))
    py_setitem(fields, nk[].k__program_class, program_class)
    py_setitem(fields, nk[].k__distributed_binding, py_item(connection, nk[].k_distributed))
    var scope = List[Int](length=2, fill=0)
    var addresses: List[Int] = [Int(scope.unsafe_ptr())]
    var options: List[Int] = [1 if recording else 0, 1 if key == "identical" else 0]
    # Registry mutation retains the GIL just as the former native open entry;
    # numerical execution is unchanged and not performed by this operation.
    var sid = neural_descriptor_execute(provider, SESSION_OPEN, addresses, options, List[Float64](), False)
    var lifetime = neural_descriptor_adopt(provider, sid, SESSION_CLOSE, scope^)
    py_setitem(fields, nk[].k__lifetime, lifetime)
    py_setitem(fields, nk[].k__id, sid)
    return PythonObject(None)
