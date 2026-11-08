# SPDX-License-Identifier: Apache-2.0
"""One native record table per object: model identity, generation word, family slots.

Every per-model record of the native doors lives here (plan
docs/plans/NATIVE_ENTRY_POINTS_AFTER_0866.md, workstream E; lessons 8 and 9):
one weak side table keyed by object identity, outside the object's
`__dict__`, with one generation word per object and one slot per family. It
replaces the separate registries of `native_prepared_owner`,
`trees_side_records`, `cluster_side_records`, `gp_svm_records`,
`msel_plan_records`, `p3_array_arena_records` (class layouts) and
`p3_guard_records`; those modules are now thin users of this one.

Accessor list (the ne1 contract; other lanes code against exactly these):

  Identity and lifetime
    model_identity_key(model) raises -> Int        object address (never 0)
    records_open(model) raises -> Int              key with a live entry, or 0 when the
                                                   object takes no weak reference
    records_live(model) raises -> Bool             a live entry exists for this object
    records_release(model_key: Int)                drop every family slot (death callback),
                                                   then post the key on the dead-key ring
    records_death_seq() -> Int                     releases posted so far
    records_dead_keys_since(seq, mut keys) -> Int  keys released after `seq`; returns the
                                                   new seq, negated when the ring overflowed
  Generation word
    model_generation(model) raises -> Int          0 = no proof (never stamped, or no
                                                   dictionary watcher); else >= 1
    records_stampable(owner) raises -> Bool        opens the entry; generation != 0
    bump_model_generation(model) raises -> Int     the new word (0 = no proof)
  Named payload records (PythonObject payloads, per family)
    records_get(family, owner, name) raises -> PythonObject      None when absent
    records_put(family, owner, name, record) raises -> Bool      False: not stored
    records_drop(family, owner, name) raises                     plain and stamped
    records_stamped_get(family, owner, name) raises -> PythonObject
    records_stamped_put(family, owner, name, var content, payload) raises -> Bool
  Native content records (what a generation alone does not prove)
    ContentRecord, content_value, content_fields, content_object, content_same
    records_key_append(key, value) raises          PyList_Append, no bound method
  Family slots (native words; valid until the next records_open of ANY object
  or records_release of this one: read or write them at once, never keep the
  pointer across a call that can run Python code)
    records_slot_prepared(model_key) raises -> UnsafePointer[PreparedSlot, MutAnyOrigin]
    records_slot_trees(model_key) raises -> UnsafePointer[TreesSlot, MutAnyOrigin]
    records_slot_cluster(model_key) raises -> UnsafePointer[ClusterSlot, MutAnyOrigin]
    records_slot_gp_svm(model_key) raises -> UnsafePointer[GpSvmSlot, MutAnyOrigin]
    records_slot_msel_plan(model_key) raises -> UnsafePointer[MselPlanSlot, MutAnyOrigin]
    records_slot_array_arena(model_key) raises -> UnsafePointer[ArrayArenaSlot, MutAnyOrigin]
    records_slot_guard(model_key) raises -> UnsafePointer[GuardSlot, MutAnyOrigin]
    records_slot_resident(model_key) raises -> UnsafePointer[ResidentSlot, MutAnyOrigin]
    (each raises when `model_key` has no entry: call records_open first)
  Family codes: RECORDS_PREPARED, RECORDS_TREES, RECORDS_CLUSTER, RECORDS_GP_SVM.

Dead-key ring (2026-11-07, replaces the 2026-11-06 release hooks): a family
store keyed by model identity outside this table (the cluster/spectral
resident table, the sequence resident store) is never called back.
records_release posts the key on a fixed ring with a monotonically increasing
sequence word; each store keeps the last word it drained and, at its own
entry points and under its own lock, calls `records_dead_keys_since` and
forgets those keys (everything, when the ring overflowed since its last
poll). A store drains before it stores, so a later death on the ring always
names a model that died after the store's entry was made. This module
imports no family module and holds no function pointer (no import cycle, no
callback, no family code in every binding).

The generation word, and the `__setattr__` hook question of rows N050, N071
and N320, settled here once (2026-11-06):
- No `__setattr__` hook is installed on any class. The object's instance
  dictionary is watched with the process dictionary watcher
  (`native_loaded_module.shared_dict_watch`, CPython >= 3.12): every
  insertion, replacement, deletion, clear and deallocation of that
  dictionary bumps its epoch, so `fit`, `set_params`, `__setstate__`, every
  family setter and every plain attribute assignment (they all write the
  instance dictionary) change the generation word without any per-family
  bookkeeping. A replaced `__dict__` (a new dictionary address) and a
  reassigned `__class__` bump it too.
- `bump_model_generation` is the explicit bump for state the dictionary does
  not see (an in-place edit by our own code, the end of a fit).
- What a generation does NOT prove: the contents of exact containers held by
  a field (a list edited in place), the class of a held object, and the
  state of a nested object (a member estimator, a learner, a kernel node).
  A record that depends on those keeps a `ContentRecord`: native words
  (lengths, item addresses, class addresses, nested objects' generations)
  compared natively per call, with no Python list built.
- Python 3.10 / 3.11 (no dictionary watcher) and objects whose dictionary is
  not watchable return generation 0: every stamped record misses and the
  caller runs its full path, exactly as before.

Lifetime: an entry holds a weak reference whose native callback (the
supported module builder's C function, bound to a unique token by
PyMethod_New) releases the entry promptly when the object dies. An entry
never holds its own object. A recycled address cannot match a dead entry:
every access confirms the weak referent is the object itself.

Nothing here computes on data or changes an arithmetic or fold order.
"""
from std.ffi import _Global
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import PythonModuleBuilder, ExceptionType, raise_python_exception
from bindings.cpython_table import cpython_table
from bindings.native_loaded_module import native_loaded_module, shared_dict_watch, shared_dict_epoch
from bindings.py_balanced import py_call, py_type, py_fields, _is_exact_dict
# The dead-key ring lives in a Python-free core module so kernels can poll it;
# re-exported here for the bindings that import it from this module.
from core.records_death_ring import (
    RECORDS_DEAD_RING,
    RecordsDeathRing,
    MODEL_RECORDS_DEATHS,
    records_post_death,
    records_death_seq,
    records_dead_keys_since,
)


# Family codes of the named payload records.
comptime RECORDS_PREPARED = 0
comptime RECORDS_TREES = 1
comptime RECORDS_CLUSTER = 2
comptime RECORDS_GP_SVM = 3


# ------------------------------------------------------------------ records


struct StampedRecord(Movable):
    """A payload valid while its owner's generation equals `generation` and
    `content` still matches."""
    var generation: Int
    var content: ContentRecord
    var payload: PythonObject

    def __init__(out self, generation: Int, var content: ContentRecord, payload: PythonObject):
        self.generation = generation
        self.content = content^
        self.payload = payload


struct RecordPayloads(Defaultable, Movable):
    """Named payload records of one family for one object."""
    var records: Dict[String, PythonObject]
    var stamped: Dict[String, StampedRecord]

    def __init__(out self):
        self.records = Dict[String, PythonObject]()
        self.stamped = Dict[String, StampedRecord]()


struct PreparedSlot(Defaultable, Movable):
    """The prepared-owner records (bindings/native_prepared_owner.mojo) and
    the per-class records of bindings/native_class_admission.mojo."""
    var payloads: RecordPayloads

    def __init__(out self):
        self.payloads = RecordPayloads()


struct TreesSlot(Defaultable, Movable):
    """Tree family records (bindings/trees_side_records.mojo users)."""
    var payloads: RecordPayloads

    def __init__(out self):
        self.payloads = RecordPayloads()


struct ClusterSlot(Defaultable, Movable):
    """Cluster family records (bindings/cluster_side_records.mojo users)."""
    var payloads: RecordPayloads

    def __init__(out self):
        self.payloads = RecordPayloads()


struct GpSvmSlot(Defaultable, Movable):
    """GP, SVM and kernel-method records (bindings/gp_svm_records.mojo users)."""
    var payloads: RecordPayloads

    def __init__(out self):
        self.payloads = RecordPayloads()


struct MselPlanSlot(Defaultable, Movable):
    """`msel_estimator_plan` instance admission (bindings/msel_plan_records.mojo):
    the admitted dictionary's address and watch epoch, and the plan it was
    admitted for (held, so its address cannot be recycled)."""
    var admitted: Bool
    var fields_address: Int
    var fields_epoch: Int
    var plan: PythonObject

    def __init__(out self):
        self.admitted = False
        self.fields_address = 0
        self.fields_epoch = 0
        self.plan = PythonObject(None)


struct ArrayArenaSlot(Defaultable, Movable):
    """Class layout admissions of the Array / arena owners, kept on the
    CLASS's entry (bindings/p3_array_arena_records.mojo): triples (kind,
    index into `extras`, class epoch at admission). The `extra` context
    objects are held so their addresses cannot be recycled."""
    var words: List[Int]
    var extras: List[PythonObject]

    def __init__(out self):
        self.words = List[Int]()
        self.extras = List[PythonObject]()


struct GuardSlot(Defaultable, Movable):
    """Layout-guard row admissions, kept on the CLASS's entry
    (bindings/p3_guard_records.mojo): per `family + index` name the admitted
    row's address and the class epoch at admission."""
    var rows: Dict[String, Int]
    var epochs: Dict[String, Int]

    def __init__(out self):
        self.rows = Dict[String, Int]()
        self.epochs = Dict[String, Int]()


struct ResidentSlot(Defaultable, Movable):
    """Device-resident fitted state (workstream A lanes). `generation` is the
    owner's generation when the handles were taken; `handles` are the
    family's native handle words, `bytes` their device size. A lane that
    needs more fields adds them here (bump MODEL_RECORDS' name)."""
    var generation: Int
    var handles: List[Int]
    var bytes: Int

    def __init__(out self):
        self.generation = 0
        self.handles = List[Int]()
        self.bytes = 0


struct ModelEntry(Movable):
    var weak: PythonObject
    var token: Int
    var type_address: Int
    var dict_address: Int
    var dict_epoch: Int
    var watched: Bool
    var generation: Int
    var prepared: PreparedSlot
    var trees: TreesSlot
    var cluster: ClusterSlot
    var gp_svm: GpSvmSlot
    var msel_plan: MselPlanSlot
    var array_arena: ArrayArenaSlot
    var guard: GuardSlot
    var resident: ResidentSlot

    def __init__(out self, weak: PythonObject, token: Int, type_address: Int, dict_address: Int,
                 dict_epoch: Int, watched: Bool):
        self.weak = weak
        self.token = token
        self.type_address = type_address
        self.dict_address = dict_address
        self.dict_epoch = dict_epoch
        self.watched = watched
        self.generation = 1
        self.prepared = PreparedSlot()
        self.trees = TreesSlot()
        self.cluster = ClusterSlot()
        self.gp_svm = GpSvmSlot()
        self.msel_plan = MselPlanSlot()
        self.array_arena = ArrayArenaSlot()
        self.guard = GuardSlot()
        self.resident = ResidentSlot()


struct ModelRecords(Defaultable, Movable):
    var entries: Dict[Int, ModelEntry]
    var owners: Dict[Int, Int]      # token -> object address
    var next_token: Int
    var cleanup_module: PythonObject
    var cleanup_function: PythonObject

    def __init__(out self):
        self.entries = Dict[Int, ModelEntry]()
        self.owners = Dict[Int, Int]()
        self.next_token = 1
        self.cleanup_module = PythonObject(None)
        self.cleanup_function = PythonObject(None)


# Shared by every Mojo extension in the process: bump the name's version
# whenever a field is added, removed or reordered in any struct above.
comptime MODEL_RECORDS = _Global[
    StorageType=ModelRecords, name="MojolearnModelRecordsV1",
    init_fn=ModelRecords.__init__,
]


# ------------------------------------------------------------------ lifetime


def records_release(model_key: Int):
    """Drop the entry of `model_key` and every family slot it holds, then
    post the key on the dead-key ring (records_dead_keys_since)."""
    try:
        var g = MODEL_RECORDS.get_or_create_ptr()
        if model_key not in g[].entries:
            return
        # Move the entry OUT before any Python reference is released: its
        # payloads may free buffers or fire another object's weakref
        # callback, and that reentrant release must see updated maps and no
        # live Dict borrow.
        var released = g[].entries.pop(model_key)
        if released.token in g[].owners:
            _ = g[].owners.pop(released.token)
        _ = released^
    except:
        return
    # The families' resident stores keyed by this model poll the ring at
    # their next call (no callback into a family module).
    records_post_death(model_key)


def _records_release_token(token: Int) raises:
    var g = MODEL_RECORDS.get_or_create_ptr()
    if token not in g[].owners:
        return
    var address = g[].owners[token]
    if address in g[].entries and g[].entries[address].token == token:
        records_release(address)
    elif token in g[].owners:
        # A stale callback never erases a replacement entry (new token).
        _ = g[].owners.pop(token)


def _records_finalize_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    # Native C entry made by the supported module builder (moved from
    # native_prepared_owner). PyMethod_New binds the entry's integer token;
    # weakref supplies itself as argument two. No Python function runs.
    try:
        ref cpy = Python().cpython()
        var table = cpython_table()
        if table[].PyTuple_Size(args_ptr) != 2:
            raise Error("model record cleanup requires its native token and weakref")
        var raw_id = table[].PyTuple_GetItem(args_ptr, 0)
        var token = table[].PyLong_AsLongLong(raw_id)
        if cpy.PyErr_Occurred():
            raise cpy.unsafe_get_error()
        _records_release_token(Int(token))
        return PythonObject(None).steal_data()
    except error:
        return raise_python_exception(error, ExceptionType("PyExc_RuntimeError"))


def _records_cleanup_function() raises -> PythonObject:
    var g = MODEL_RECORDS.get_or_create_ptr()
    if g[].cleanup_function is None:
        # The standard builder owns the method definition and creates the
        # native C callable; the finalized module stays alive in this global
        # (no sys.modules write, no Python function).
        var module = PythonModuleBuilder("_mojolearn_model_records_lifetime")
        module.def_py_c_function(_records_finalize_c, "release")
        var native_module = module.finalize()
        ref cpy = Python().cpython()
        var fields = cpython_table()[].PyModule_GetDict(native_module._obj_ptr)
        var name = String("release")
        var raw = cpy.lib.get_function[PyObjectPtr]("PyDict_GetItemString")(fields, name.as_c_string_slice().unsafe_ptr())
        if not raw:
            raise Error("model record native cleanup entry is absent")
        g[].cleanup_module = native_module
        g[].cleanup_function = PythonObject(from_owned=cpy.Py_NewRef(raw))
    return g[].cleanup_function


def model_identity_key(model: PythonObject) raises -> Int:
    """The record key of `model`: its object address."""
    return Int(model._obj_ptr)


def records_live(model: PythonObject) raises -> Bool:
    """True when `model` has a live entry. A dead entry left at a recycled
    address (a callback that could not run) is released here."""
    var g = MODEL_RECORDS.get_or_create_ptr()
    var key = Int(model._obj_ptr)
    if key not in g[].entries:
        return False
    var weak = g[].entries[key].weak
    ref cpy = Python().cpython()
    # The weak reference is an exact weakref.ref made by PyWeakref_NewRef:
    # calling it with no arguments is CPython's own C referent read (a new
    # reference, or None once the referent died). Runs no Python code.
    var raw = cpython_table()[].PyObject_CallNoArgs(weak._obj_ptr)
    if not raw:
        raise cpy.unsafe_get_error()
    var held = PythonObject(from_owned=raw)
    if held._obj_ptr != model._obj_ptr:
        records_release(key)
        return False
    return True


def records_open(model: PythonObject) raises -> Int:
    """The key of `model`'s entry, made on first use (weak reference armed,
    instance dictionary watched). 0 when `model` takes no weak reference:
    nothing is stored for it and callers run their full path every call."""
    if records_live(model):
        return Int(model._obj_ptr)
    ref cpy = Python().cpython()
    var function = _records_cleanup_function()
    var g = MODEL_RECORDS.get_or_create_ptr()
    if g[].next_token == 9223372036854775807:
        raise Error("model record table exhausted native tokens")
    var token = g[].next_token
    g[].next_token += 1
    var token_object = PythonObject(token)
    # Documented public APIs: PyMethod_New binds an object without running
    # it; PyWeakref_NewRef runs the native callback promptly on owner death.
    var table = cpython_table()
    var raw_callback = table[].PyMethod_New(function._obj_ptr, token_object._obj_ptr)
    if not raw_callback:
        raise cpy.unsafe_get_error()
    var callback = PythonObject(from_owned=raw_callback)
    var raw_weak = table[].PyWeakref_NewRef(model._obj_ptr, callback._obj_ptr)
    if not raw_weak:
        cpython_table()[].PyErr_Clear()
        return 0
    var weak = PythonObject(from_owned=raw_weak)
    var type_address = Int(py_type(model)._obj_ptr)
    var dict_address = 0
    var dict_epoch = 0
    var watched = False
    try:
        var fields = py_fields(model)
        if _is_exact_dict(fields) and shared_dict_watch(fields._obj_ptr):
            watched = True
            dict_address = Int(fields._obj_ptr)
            dict_epoch = shared_dict_epoch(dict_address)
    except:
        # No instance dictionary (slots, a class's mappingproxy): the entry
        # holds records but its generation proves nothing (word 0).
        watched = False
    var key = Int(model._obj_ptr)
    g = MODEL_RECORDS.get_or_create_ptr()
    g[].entries[key] = ModelEntry(weak, token, type_address, dict_address, dict_epoch, watched)
    g[].owners[token] = key
    return key


def _records_sync(model: PythonObject, key: Int) raises:
    """Fold dictionary, dictionary-address and class changes since the last
    read into the generation word. Python facts are read before the entry is
    borrowed (no Dict borrow is live while CPython runs)."""
    var g = MODEL_RECORDS.get_or_create_ptr()
    if not g[].entries[key].watched:
        return
    var type_address = Int(py_type(model)._obj_ptr)
    var fields = py_fields(model)
    var current = Int(fields._obj_ptr)
    var rewatch = False
    if current != g[].entries[key].dict_address:
        rewatch = _is_exact_dict(fields) and shared_dict_watch(fields._obj_ptr)
    var epoch = shared_dict_epoch(current)
    g = MODEL_RECORDS.get_or_create_ptr()
    ref e = g[].entries[key]
    if current != e.dict_address:
        e.generation += 1
        if rewatch:
            e.dict_address = current
            e.dict_epoch = epoch
        else:
            e.watched = False
    elif epoch != e.dict_epoch:
        e.generation += 1
        e.dict_epoch = epoch
    if type_address != e.type_address:
        e.generation += 1
        e.type_address = type_address


def model_generation(model: PythonObject) raises -> Int:
    """`model`'s generation word: changes whenever its instance dictionary,
    its `__dict__` object, its class or an explicit bump changes. 0 when it
    proves nothing (never stamped, no dictionary watcher, no watchable
    dictionary): a caller then runs its full path."""
    if not records_live(model):
        return 0
    var key = Int(model._obj_ptr)
    _records_sync(model, key)
    var g = MODEL_RECORDS.get_or_create_ptr()
    if not g[].entries[key].watched:
        return 0
    return g[].entries[key].generation


def bump_model_generation(model: PythonObject) raises -> Int:
    """Advance `model`'s generation (every stamped record of it misses once)
    and return the new word (0 when no generation can be kept for it)."""
    var key = records_open(model)
    if key == 0:
        return 0
    _records_sync(model, key)
    var g = MODEL_RECORDS.get_or_create_ptr()
    g[].entries[key].generation += 1
    if not g[].entries[key].watched:
        return 0
    return g[].entries[key].generation


def records_stampable(owner: PythonObject) raises -> Bool:
    """True when a generation can be kept for `owner` (its entry exists or
    was just made, and its generation word is nonzero). Callers test it
    before building a content record, so Python 3.10 / 3.11 never pay for one."""
    if records_open(owner) == 0:
        return False
    return model_generation(owner) != 0


# ------------------------------------------------------------------ slots


def _need(model_key: Int) raises:
    var g = MODEL_RECORDS.get_or_create_ptr()
    if model_key not in g[].entries:
        raise Error("model record slot requested for an object without an entry (call records_open)")


def records_slot_prepared(model_key: Int) raises -> UnsafePointer[PreparedSlot, MutAnyOrigin]:
    _need(model_key)
    var g = MODEL_RECORDS.get_or_create_ptr()
    return UnsafePointer(to=g[].entries[model_key].prepared).unsafe_origin_cast[MutAnyOrigin]()


def records_slot_trees(model_key: Int) raises -> UnsafePointer[TreesSlot, MutAnyOrigin]:
    _need(model_key)
    var g = MODEL_RECORDS.get_or_create_ptr()
    return UnsafePointer(to=g[].entries[model_key].trees).unsafe_origin_cast[MutAnyOrigin]()


def records_slot_cluster(model_key: Int) raises -> UnsafePointer[ClusterSlot, MutAnyOrigin]:
    _need(model_key)
    var g = MODEL_RECORDS.get_or_create_ptr()
    return UnsafePointer(to=g[].entries[model_key].cluster).unsafe_origin_cast[MutAnyOrigin]()


def records_slot_gp_svm(model_key: Int) raises -> UnsafePointer[GpSvmSlot, MutAnyOrigin]:
    _need(model_key)
    var g = MODEL_RECORDS.get_or_create_ptr()
    return UnsafePointer(to=g[].entries[model_key].gp_svm).unsafe_origin_cast[MutAnyOrigin]()


def records_slot_msel_plan(model_key: Int) raises -> UnsafePointer[MselPlanSlot, MutAnyOrigin]:
    _need(model_key)
    var g = MODEL_RECORDS.get_or_create_ptr()
    return UnsafePointer(to=g[].entries[model_key].msel_plan).unsafe_origin_cast[MutAnyOrigin]()


def records_slot_array_arena(model_key: Int) raises -> UnsafePointer[ArrayArenaSlot, MutAnyOrigin]:
    _need(model_key)
    var g = MODEL_RECORDS.get_or_create_ptr()
    return UnsafePointer(to=g[].entries[model_key].array_arena).unsafe_origin_cast[MutAnyOrigin]()


def records_slot_guard(model_key: Int) raises -> UnsafePointer[GuardSlot, MutAnyOrigin]:
    _need(model_key)
    var g = MODEL_RECORDS.get_or_create_ptr()
    return UnsafePointer(to=g[].entries[model_key].guard).unsafe_origin_cast[MutAnyOrigin]()


def records_slot_resident(model_key: Int) raises -> UnsafePointer[ResidentSlot, MutAnyOrigin]:
    _need(model_key)
    var g = MODEL_RECORDS.get_or_create_ptr()
    return UnsafePointer(to=g[].entries[model_key].resident).unsafe_origin_cast[MutAnyOrigin]()


def _payloads(model_key: Int, family: Int) raises -> UnsafePointer[RecordPayloads, MutAnyOrigin]:
    var g = MODEL_RECORDS.get_or_create_ptr()
    if family == RECORDS_PREPARED:
        return UnsafePointer(to=g[].entries[model_key].prepared.payloads).unsafe_origin_cast[MutAnyOrigin]()
    if family == RECORDS_TREES:
        return UnsafePointer(to=g[].entries[model_key].trees.payloads).unsafe_origin_cast[MutAnyOrigin]()
    if family == RECORDS_CLUSTER:
        return UnsafePointer(to=g[].entries[model_key].cluster.payloads).unsafe_origin_cast[MutAnyOrigin]()
    if family == RECORDS_GP_SVM:
        return UnsafePointer(to=g[].entries[model_key].gp_svm.payloads).unsafe_origin_cast[MutAnyOrigin]()
    raise Error("unknown model record family")


# ------------------------------------------------------------------ payloads


def records_get(family: Int, owner: PythonObject, name: String) raises -> PythonObject:
    """The record `name` of `family` held for `owner`, or None."""
    if not records_live(owner):
        return PythonObject(None)
    var p = _payloads(Int(owner._obj_ptr), family)
    if name not in p[].records:
        return PythonObject(None)
    return p[].records[name]


def records_put(family: Int, owner: PythonObject, name: String, record: PythonObject) raises -> Bool:
    """Keep `record` for `owner` (replacing an older one). False when
    `owner` takes no weak reference: nothing is stored. `record` must not
    strongly retain `owner`."""
    var key = records_open(owner)
    if key == 0:
        return False
    var previous = PythonObject(None)
    var p = _payloads(key, family)
    if name in p[].records:
        previous = p[].records.pop(name)
    p = _payloads(key, family)
    p[].records[name] = record
    # The replaced record is released only after the map holds the new one
    # (its destruction may reenter this table).
    _ = previous^
    return True


def records_drop(family: Int, owner: PythonObject, name: String) raises:
    """Forget `owner`'s record `name` (plain and stamped) now, e.g. at a refit,
    so the arrays it holds are released at once."""
    if not records_live(owner):
        return
    var key = Int(owner._obj_ptr)
    var previous = PythonObject(None)
    var p = _payloads(key, family)
    if name in p[].records:
        previous = p[].records.pop(name)
    p = _payloads(key, family)
    if name in p[].stamped:
        # Moved out of the map first; destroyed when this branch ends.
        var stale = p[].stamped.pop(name)
        _ = stale^
    _ = previous^


def records_stamped_get(family: Int, owner: PythonObject, name: String) raises -> PythonObject:
    """The payload of the stamped record `name` when `owner`'s generation is
    the one it was stamped with and its content still matches; else None."""
    var generation = model_generation(owner)
    if generation == 0:
        return PythonObject(None)
    var p = _payloads(Int(owner._obj_ptr), family)
    if name not in p[].stamped:
        return PythonObject(None)
    if p[].stamped[name].generation != generation:
        return PythonObject(None)
    # content_same runs no Python code that can allocate (borrowed reads,
    # type reads, generation reads of already-opened entries), so no weakref
    # callback can change the table while the record is borrowed.
    if not content_same(p[].stamped[name].content):
        return PythonObject(None)
    return p[].stamped[name].payload


def records_stamped_put(family: Int, owner: PythonObject, name: String, var content: ContentRecord,
                        payload: PythonObject) raises -> Bool:
    """Stamp `payload` with `owner`'s current generation and `content`.
    False (nothing stored) when no generation can be kept for `owner` or the
    content walk met something it cannot prove natively."""
    if not content.ok:
        return False
    var key = records_open(owner)
    if key == 0:
        return False
    var generation = model_generation(owner)
    if generation == 0:
        return False
    var p = _payloads(key, family)
    if name in p[].stamped:
        # The replaced record is destroyed only after the map holds the new
        # one (its destruction may reenter this table).
        var stale = p[].stamped.pop(name)
        p = _payloads(key, family)
        p[].stamped[name] = StampedRecord(generation, content^, payload)
        _ = stale^
        return True
    p[].stamped[name] = StampedRecord(generation, content^, payload)
    return True


# ------------------------------------------------------------------ keys


def records_key_append(key: PythonObject, value: PythonObject) raises:
    """`key.append(value)` for an exact list through PyList_Append: no bound
    method object per append."""
    if cpython_table()[].PyList_Append(key._obj_ptr, value._obj_ptr) != 0:
        ref cpy = Python().cpython()
        raise cpy.unsafe_get_error()


def records_key_same_exact(held: PythonObject, key: PythonObject) raises -> Bool:
    """Two exact key lists hold the same entries: equal length and every
    entry the same object, exact ints compared by value (int's own equality,
    no hook). Borrowed native reads, no boxed indexing."""
    if held is None:
        return False
    var t = cpython_table()
    var n = t[].PyList_Size(key._obj_ptr)
    if t[].PyList_Size(held._obj_ptr) != n:
        return False
    var int_t = py_type(PythonObject(0))
    for k in range(n):  # small-loop(key: recorded identities): pointer identity compare only
        var a = t[].PyList_GetItem(held._obj_ptr, k)
        var c = t[].PyList_GetItem(key._obj_ptr, k)
        if a == c:
            continue
        var left = PythonObject(from_borrowed=a)
        var right = PythonObject(from_borrowed=c)
        if py_type(left) is int_t and py_type(right) is int_t and Bool(py=(left == right)):
            continue
        return False
    return True


# ------------------------------------------------------------------ content

# Content operations: each starts with its code, then its operands.
comptime _OP_TYPE = 1    # held index, class address
comptime _OP_LIST = 2    # held index, length, item addresses
comptime _OP_DICT = 3    # held index, size, (held key index, value address) pairs
comptime _OP_GEN = 4     # held index, generation word


struct ContentRecord(Defaultable, Movable):
    """What an owner's generation does not prove, as native words checked by
    `content_same`: exact list contents, exact dict contents, held objects'
    classes and nested objects' generations. `ok` is False when the walk met
    something it cannot check natively (a set, a dict with keys other than
    exact str / int, a nested object without a generation, too deep a
    nesting): such a record is never stored."""
    var held: List[PythonObject]
    var ops: List[Int]
    var ok: Bool

    def __init__(out self):
        self.held = List[PythonObject]()
        self.ops = List[Int]()
        self.ok = True


def _hold(mut rec: ContentRecord, value: PythonObject) -> Int:
    rec.held.append(value)
    return len(rec.held) - 1


def content_value(mut rec: ContentRecord, value: PythonObject, depth: Int, limit: Int, sets_fail: Bool) raises:
    """Record what an identity walk of `value` reads beyond `value`'s own
    identity (which the caller proves: the owner's generation, or the
    enclosing container's item words). Mirrors trees_key_shallow: exact
    lists, tuples and dicts item by item to `limit` levels, every other
    non-scalar by class. `sets_fail`: an exact set / frozenset makes the
    record unusable (the trees_key_value walk reads their items)."""
    # Iterative (ne-fix-hang2): an explicit stack in place of the former
    # self-recursion (the compile hang was the dead-key ring, not this). Children
    # are pushed in reverse so they pop in item order: the same depth-first
    # pre-order, the same words in `rec.ops`, the same `limit` cut and the
    # same stop once `rec.ok` is False.
    if not rec.ok:
        return
    var b = native_loaded_module("builtins")
    var pending = List[PythonObject]()
    var depths = List[Int]()
    pending.append(value)
    depths.append(depth)
    while len(pending) > 0 and rec.ok:  # driver-loop: over a stored value's nested containers (once per record)
        var item = pending.pop()
        var level = depths.pop()
        if item is None or level >= limit:
            continue
        var cls = py_type(item)
        if cls is b.bool or cls is b.int or cls is b.float or cls is b.str or cls is b.bytes:
            continue
        if cls is b.list:
            var n = len(item)
            var h = _hold(rec, item)
            rec.ops.append(_OP_LIST)
            rec.ops.append(h)
            rec.ops.append(n)
            var t = cpython_table()
            for i in range(len(item)):  # driver-loop: over a stored list's items (identity words, once per record)
                rec.ops.append(Int(t[].PyList_GetItem(item._obj_ptr, i)))
            for i in range(len(item) - 1, -1, -1):  # driver-loop: over a stored list's items (nested words, once per record)
                pending.append(item[i])
                depths.append(level + 1)
            continue
        if cls is b.tuple:
            for i in range(len(item) - 1, -1, -1):  # driver-loop: over a stored tuple's items (nested words, once per record)
                pending.append(item[i])
                depths.append(level + 1)
            continue
        if cls is b.dict:
            # items() of an exact dict never hashes or compares keys.
            var items = py_call(b.list, py_call(item.items))
            var h = _hold(rec, item)
            rec.ops.append(_OP_DICT)
            rec.ops.append(h)
            rec.ops.append(len(items))
            for i in range(len(items)):  # driver-loop: over a stored mapping's entries (once per record)
                var name = items[i][0]
                var kind = py_type(name)
                # The per-call lookup hashes the key: only exact str (cached
                # hash) and exact int keys, whose hashing runs no user code.
                if kind is not b.str and kind is not b.int:
                    rec.ok = False
                    return
                var held_key = _hold(rec, name)
                rec.ops.append(held_key)
                rec.ops.append(Int(items[i][1]._obj_ptr))
            for i in range(len(items) - 1, -1, -1):  # driver-loop: over a stored mapping's values (nested words, once per record)
                pending.append(items[i][1])
                depths.append(level + 1)
            continue
        if sets_fail and (cls is b.set or cls is b.frozenset):
            rec.ok = False
            return
        # Any other object: its class can be reassigned in place.
        var h = _hold(rec, item)
        rec.ops.append(_OP_TYPE)
        rec.ops.append(h)
        rec.ops.append(Int(cls._obj_ptr))


def content_fields(mut rec: ContentRecord, fields: PythonObject, limit: Int) raises:
    """`content_value` of every value of an exact field dictionary (the
    entries' identities are proven by the owner's generation)."""
    if not rec.ok:
        return
    var b = native_loaded_module("builtins")
    var items = py_call(b.list, py_call(fields.items))
    for i in range(len(items)):  # driver-loop: over a model's stored fields (once per record)
        content_value(rec, items[i][1], 0, limit, False)


def content_object(mut rec: ContentRecord, obj: PythonObject, limit: Int) raises:
    """A nested object the record depends on (a member estimator, a learner,
    a kernel node): its class, its own generation word, and the content of
    its stored fields. Unusable when `obj` has no generation."""
    if not rec.ok:
        return
    if records_open(obj) == 0:
        rec.ok = False
        return
    var generation = model_generation(obj)
    if generation == 0:
        rec.ok = False
        return
    var h = _hold(rec, obj)
    rec.ops.append(_OP_TYPE)
    rec.ops.append(h)
    rec.ops.append(Int(py_type(obj)._obj_ptr))
    rec.ops.append(_OP_GEN)
    rec.ops.append(h)
    rec.ops.append(generation)
    var fields = py_fields(obj)
    if not _is_exact_dict(fields):
        rec.ok = False
        return
    content_fields(rec, fields, limit)


def content_same(rec: ContentRecord) raises -> Bool:
    """Every recorded word still holds. Borrowed native reads only."""
    if not rec.ok:
        return False
    var t2 = cpython_table()
    var t = cpython_table()
    var i = 0
    var n = len(rec.ops)
    while i < n:  # driver-loop: over the record's content words (no Python objects built)
        var op = rec.ops[i]
        var h = rec.ops[i + 1]
        var ptr = rec.held[h]._obj_ptr
        if op == _OP_TYPE:
            if Int(py_type(rec.held[h])._obj_ptr) != rec.ops[i + 2]:
                return False
            i += 3
        elif op == _OP_LIST:
            var size = rec.ops[i + 2]
            if t2[].PyList_Size(ptr) != size:
                return False
            for k in range(size):  # small-loop(size: one recorded list's items): address compare only
                if Int(t2[].PyList_GetItem(ptr, k)) != rec.ops[i + 3 + k]:
                    return False
            i += 3 + size
        elif op == _OP_DICT:
            var size = rec.ops[i + 2]
            if t[].PyDict_Size(ptr) != size:
                return False
            for k in range(size):  # small-loop(size: one recorded mapping's entries): borrowed lookups by exact str/int keys
                var name = rec.held[rec.ops[i + 3 + 2 * k]]._obj_ptr
                if Int(t[].PyDict_GetItem(ptr, name)) != rec.ops[i + 4 + 2 * k]:
                    return False
            i += 3 + 2 * size
        elif op == _OP_GEN:
            if model_generation(rec.held[h]) != rec.ops[i + 2]:
                return False
            i += 3
        else:
            return False
    return True
