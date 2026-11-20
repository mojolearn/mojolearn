# SPDX-License-Identifier: Apache-2.0
# Program.run's admitted route, resolved once per Program and Session owner.
#
# Lane fix-program-run-per-call (2026-11-14). Profile r5 of program-run
# (py-spy --native, L40S, wheels 80a7f8aca3; in-package shares): the
# per-call Program and Session metadata admission was 22%
# (`program_owner_fields`) + 21% (`_fields`, inside it) + the Session
# owner's `ops_session_fields`: per run the admission context's dicts were
# read by key, `builtins` attributes looked up, the Program's `__dict__`
# fetched through a Python call of its connected descriptor (`fields_get`),
# and `_s`, `_id`, `_lifetime` read by key. Those values change only when
# one of the two instance dictionaries changes, so the first fully admitted
# run (the unchanged full checks in autograd_program_public_request_pyapi_pack)
# records here, in typed words, what it decided: the two dictionaries'
# addresses and watch epochs, the two classes and their admitted layout
# records, the registry resource, and the session and program handles.
# A later run reads no Python attribute and no dictionary key: one
# `PyObject_GenericGetDict` per owner (the same dictionary `fields_get`
# returns for these exact classes; proven equal at record time), the two
# dictionaries' epochs, the two class records, and the three flag words
# compared to `False`.
#
# Validity. Python >= 3.12: both dictionaries are watched (their key
# admission is recorded, bindings/native_class_admission_door.mojo), and
# every insertion, replacement, deletion, clear and deallocation bumps the
# dictionary's epoch (a freed Program's address recycled by a new one never
# matches). CPython 3.10 / 3.11 with proven version tags: the epoch is the
# dictionary's own `ma_version_tag`, unique per edit. Where neither exists
# nothing is recorded and every run takes the full admission, as before.
# A changed class (a replaced hook or storage descriptor) misses its class
# record and takes the full route, which refuses it with the same message.
# A Program.free, a Session.close, a replaced `_s`, a flag set, a session of
# another image or executor kind: all miss, and the full route decides.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_type, py_fields
from bindings.py_cpython_table import cpython_table
from bindings.native_loaded_module_door import NativeModules, shared_dict_stamped, shared_dict_epoch
from bindings.native_class_admission_door import class_record, dict_keys_admitted
from core.shared_slot import SharedSlot
from providers.python.autograd_import_metadata_pyapi_door import _IMPORT_LAYOUT_SLOT
from providers.python.autograd_session_frontend_pyapi_door import _SESSION_LAYOUT_SLOT
from providers.python.autograd_program_arrays_pyapi_door import program_admission_context
from providers.python.neural_descriptor_pyapi_door import NativeDescriptorResource
from autograd.ops_api import ops_session_registry_token
from bindings.autograd_keys_unpack import autograd_keys
from bindings.py_balanced import py_item


@fieldwise_init
struct ProgramRunRoute(Copyable, Movable):
    """One admitted Program.run route: typed words only, no Python object
    (the owners are proven alive per call through the dictionaries)."""
    var kind: Int
    var registry: Int
    var program_dict: Int
    var program_epoch: Int
    var program_class: Int
    var program_layout: Int
    var session: Int
    var session_dict: Int
    var session_epoch: Int
    var session_class: Int
    var session_layout: Int
    var resource: Int
    var sid: Int
    var pid: Int


struct ProgramRunRoutes(Defaultable, Movable):
    var entries: List[ProgramRunRoute]
    var next: Int

    def __init__(out self):
        self.entries = List[ProgramRunRoute]()
        self.next = 0


# A few live Programs per process is the common case; the oldest record is
# replaced round-robin beyond this (a miss only costs the full admission).
comptime _ROUTE_CAPACITY = 16
comptime _PROGRAM_RUN_ROUTES = SharedSlot[ProgramRunRoutes, "MojolearnProgramRunRoutesV1"]


@fieldwise_init
struct ProgramRunAdmitted(Copyable, Movable):
    """The handles of an admitted run; sid == 0 means no record matched."""
    var sid: Int
    var pid: Int


def _borrowed(address: Int) raises -> PythonObject:
    # The address is a live owner proven by an unchanged dictionary that
    # holds a strong reference to it (see program_run_route).
    var raw = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
    return PythonObject(from_borrowed=PyObjectPtr(upcast_from=Optional(raw)))


def _layout_word(mut mods: NativeModules, slot: String, cls: PythonObject) raises -> Int:
    """The admitted layout record of `cls` under `slot`, or 0."""
    var record = class_record(mods, slot, cls)
    if record is None:
        return 0
    return Int(py=record)


def program_run_route[KIND: Int](owner: PythonObject, args: PythonObject) raises -> ProgramRunAdmitted:
    """The recorded (sid, pid) when this Program.run(args) is the same
    admitted fixed-route run as before; sid 0 otherwise. `args` is the exact
    4-tuple (arrays, dynamic_shapes, optimize, profile), checked by the
    caller."""
    var missed = ProgramRunAdmitted(0, 0)
    var routes = _PROGRAM_RUN_ROUTES.get_or_create_ptr()
    if len(routes[].entries) == 0 or not shared_dict_stamped():
        return missed^
    var t = cpython_table()
    var false_word = Int(PythonObject(False)._obj_ptr)
    for i in range(1, 4):  # small-loop(3: native execution flags)
        if Int(t[].PyTuple_GetItem(args._obj_ptr, i)) != false_word:
            return missed^
    var program_class = py_type(owner)
    var program_dict = py_fields(owner)
    var pdict = Int(program_dict._obj_ptr)
    var pcls = Int(program_class._obj_ptr)
    var registry = ops_session_registry_token()
    var found = -1
    for i in range(len(routes[].entries)):  # small-loop(_ROUTE_CAPACITY: recorded Program routes): address compare, no data
        if (routes[].entries[i].program_dict == pdict and routes[].entries[i].program_class == pcls
                and routes[].entries[i].kind == KIND and routes[].entries[i].registry == registry):
            found = i
            break
    if found < 0 or shared_dict_epoch(pdict) != routes[].entries[found].program_epoch:
        return missed^
    var mods = NativeModules()
    if _layout_word(mods, _IMPORT_LAYOUT_SLOT, program_class) != routes[].entries[found].program_layout:
        return missed^
    # The unchanged Program dictionary still holds the same `_s` (alive).
    var session = _borrowed(routes[].entries[found].session)
    var session_class = py_type(session)
    if Int(session_class._obj_ptr) != routes[].entries[found].session_class:
        return missed^
    if _layout_word(mods, _SESSION_LAYOUT_SLOT, session_class) != routes[].entries[found].session_layout:
        return missed^
    var session_dict = py_fields(session)
    var sdict = Int(session_dict._obj_ptr)
    if sdict != routes[].entries[found].session_dict or shared_dict_epoch(sdict) != routes[].entries[found].session_epoch:
        return missed^
    # The unchanged Session dictionary still holds the same `_lifetime`.
    var resource = UnsafePointer[NativeDescriptorResource, MutUntrackedOrigin](
        unsafe_from_address=routes[].entries[found].resource)
    var sid = routes[].entries[found].sid
    if resource[].handle == 0 or resource[].handle != sid:
        return missed^
    _ = session_dict
    _ = program_dict
    return ProgramRunAdmitted(sid, routes[].entries[found].pid)


def program_run_route_record[KIND: Int](
    owner: PythonObject, fields: PythonObject, session: PythonObject, sid: Int, pid: Int,
) raises:
    """Record a run the full admission just accepted (`fields` is the
    Program's admitted dictionary, `session` its admitted `_s`, `sid` the
    handle `_fixed_run_session` admitted against this image's registry).
    Nothing is recorded unless every word can be proven again per call."""
    if not shared_dict_stamped():
        return
    var program_dict = py_fields(owner)
    if Int(program_dict._obj_ptr) != Int(fields._obj_ptr) or not dict_keys_admitted(program_dict):
        return
    var session_dict = py_fields(session)
    if not dict_keys_admitted(session_dict):
        return
    var t = cpython_table()
    var key = PythonObject("_lifetime")
    var lifetime_ptr = t[].PyDict_GetItem(session_dict._obj_ptr, key._obj_ptr)
    if not lifetime_ptr:
        return
    var lifetime = PythonObject(from_borrowed=lifetime_ptr)
    var resource = lifetime.downcast_value_ptr[NativeDescriptorResource]()
    if resource[].handle != sid or sid == 0:
        return
    var program_class = py_type(owner)
    var session_class = py_type(session)
    var mods = NativeModules()
    var program_layout = _layout_word(mods, _IMPORT_LAYOUT_SLOT, program_class)
    var session_layout = _layout_word(mods, _SESSION_LAYOUT_SLOT, session_class)
    if program_layout == 0 or session_layout == 0:
        return  # no class record: these classes keep the full admission per run
    var context = program_admission_context()
    var nk = autograd_keys()
    if session_layout != Int(py_item(context, nk[].k_ops_session_layout)._obj_ptr):
        return
    var route = ProgramRunRoute(
        kind=KIND, registry=ops_session_registry_token(),
        program_dict=Int(program_dict._obj_ptr), program_epoch=shared_dict_epoch(Int(program_dict._obj_ptr)),
        program_class=Int(program_class._obj_ptr), program_layout=program_layout,
        session=Int(session._obj_ptr),
        session_dict=Int(session_dict._obj_ptr), session_epoch=shared_dict_epoch(Int(session_dict._obj_ptr)),
        session_class=Int(session_class._obj_ptr), session_layout=session_layout,
        resource=Int(resource), sid=sid, pid=pid,
    )
    var routes = _PROGRAM_RUN_ROUTES.get_or_create_ptr()
    var stale = -1
    for i in range(len(routes[].entries)):  # small-loop(_ROUTE_CAPACITY: recorded Program routes): this Program's stale record
        if routes[].entries[i].program_dict == route.program_dict and routes[].entries[i].kind == KIND:
            stale = i
            break
    if stale >= 0:
        routes[].entries[stale] = route^
    elif len(routes[].entries) < _ROUTE_CAPACITY:
        routes[].entries.append(route^)
    else:
        routes[].entries[routes[].next] = route^
        routes[].next = (routes[].next + 1) % _ROUTE_CAPACITY
    _ = lifetime
