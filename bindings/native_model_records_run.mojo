# SPDX-License-Identifier: Apache-2.0
"""One native record table per object: the TYPED part (RUN; no Python).

Split from bindings/native_model_records.mojo (lane resident-pin-tables,
2026-11-12). Every entry is keyed by the object's address (`model key`) and
holds only words: the weak reference's native token, the class address, the
watched instance dictionary's address and epoch, the generation word, and the
typed family slots (msel plan admission, Array / arena class layouts, layout
guard rows). Its Python pins (the weak reference, the named payload records,
the msel plan and the arena `extra` objects) live in the door-side store of
bindings/py_model_records.mojo under the same key; that module opens and
releases an entry in both places together, under the GIL.

Family slots: valid until the next records_open of ANY object or the release
of this one: read or write them at once, never keep the pointer across a call
that can run Python code. Each raises when the key has no entry.

Nothing here computes on data or changes an arithmetic or fold order.
"""
from core.shared_slot import SharedSlot


# Family codes of the named payload records.
comptime RECORDS_PREPARED = 0
comptime RECORDS_TREES = 1
comptime RECORDS_CLUSTER = 2
comptime RECORDS_GP_SVM = 3


struct MselPlanSlot(Copyable, Defaultable, Movable):
    """`msel_estimator_plan` instance admission (bindings/msel_plan_records_unpack.mojo):
    the admitted dictionary's address and watch epoch, and the id() word of
    the plan it was admitted for (the door pins the plan, so its address
    cannot be recycled while the entry lives)."""
    var admitted: Bool
    var fields_address: Int
    var fields_epoch: Int
    var plan: Int

    def __init__(out self):
        self.admitted = False
        self.fields_address = 0
        self.fields_epoch = 0
        self.plan = 0


struct ArrayArenaSlot(Copyable, Defaultable, Movable):
    """Class layout admissions of the Array / arena owners, kept on the
    CLASS's entry (bindings/p3_array_arena_records_unpack.mojo): triples
    (kind, index into `extras`, class epoch at admission). `extras` are the
    id() words of the `extra` context objects; the door pins those objects,
    so their addresses cannot be recycled while the entry lives."""
    var words: List[Int]
    var extras: List[Int]

    def __init__(out self):
        self.words = List[Int]()
        self.extras = List[Int]()


struct GuardSlot(Copyable, Defaultable, Movable):
    """Layout-guard row admissions, kept on the CLASS's entry
    (bindings/p3_guard_records_unpack.mojo): per `family + index` name the
    admitted row's address and the class epoch at admission."""
    var rows: Dict[String, Int]
    var epochs: Dict[String, Int]

    def __init__(out self):
        self.rows = Dict[String, Int]()
        self.epochs = Dict[String, Int]()


struct ModelWords(Copyable, Movable):
    var token: Int
    var type_address: Int
    var dict_address: Int
    var dict_epoch: Int
    var watched: Bool
    var generation: Int
    var msel_plan: MselPlanSlot
    var array_arena: ArrayArenaSlot
    var guard: GuardSlot

    def __init__(out self, token: Int, type_address: Int, dict_address: Int, dict_epoch: Int, watched: Bool):
        self.token = token
        self.type_address = type_address
        self.dict_address = dict_address
        self.dict_epoch = dict_epoch
        self.watched = watched
        self.generation = 1
        self.msel_plan = MselPlanSlot()
        self.array_arena = ArrayArenaSlot()
        self.guard = GuardSlot()


struct ModelRecordsTable(Defaultable, Movable):
    var entries: Dict[Int, ModelWords]
    var owners: Dict[Int, Int]      # token -> object address
    var next_token: Int

    def __init__(out self):
        self.entries = Dict[Int, ModelWords]()
        self.owners = Dict[Int, Int]()
        self.next_token = 1


# Shared by every Mojo extension in the process: bump the name's version
# whenever a field is added, removed or reordered in any struct above.
# V3 (lane resident-pin-tables): words only; pins moved to the door store.
comptime MODEL_RECORDS = SharedSlot[ModelRecordsTable, "MojolearnModelRecordsV3", 3]


def records_run_has(model_key: Int) raises -> Bool:
    return model_key in MODEL_RECORDS.get_or_create_ptr()[].entries


def records_run_new_token() raises -> Int:
    var g = MODEL_RECORDS.get_or_create_ptr()
    if g[].next_token == 9223372036854775807:
        raise Error("model record table exhausted native tokens")
    var token = g[].next_token
    g[].next_token += 1
    return token


def records_run_insert(model_key: Int, token: Int, type_address: Int, dict_address: Int, dict_epoch: Int,
                       watched: Bool) raises:
    var g = MODEL_RECORDS.get_or_create_ptr()
    g[].entries[model_key] = ModelWords(token, type_address, dict_address, dict_epoch, watched)
    g[].owners[token] = model_key


def records_run_remove(model_key: Int) raises -> Bool:
    """Drop `model_key`'s words; False when it had no entry."""
    var g = MODEL_RECORDS.get_or_create_ptr()
    if model_key not in g[].entries:
        return False
    var released = g[].entries.pop(model_key)
    if released.token in g[].owners:
        _ = g[].owners.pop(released.token)
    return True


def records_run_token_key(token: Int) raises -> Int:
    """The key whose entry the weak-reference token `token` belongs to, or 0.
    A stale token (its address now holds a replacement entry with a new
    token) is forgotten here and never names the replacement."""
    var g = MODEL_RECORDS.get_or_create_ptr()
    if token not in g[].owners:
        return 0
    var address = g[].owners[token]
    if address in g[].entries and g[].entries[address].token == token:
        return address
    _ = g[].owners.pop(token)
    return 0


def records_run_watched(model_key: Int) raises -> Bool:
    return MODEL_RECORDS.get_or_create_ptr()[].entries[model_key].watched


def records_run_dict_address(model_key: Int) raises -> Int:
    return MODEL_RECORDS.get_or_create_ptr()[].entries[model_key].dict_address


def records_run_sync(model_key: Int, type_address: Int, current: Int, rewatch: Bool, epoch: Int) raises:
    """Fold the dictionary, dictionary-address and class facts the door read
    into the generation word."""
    var g = MODEL_RECORDS.get_or_create_ptr()
    ref e = g[].entries[model_key]
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


def records_run_generation(model_key: Int) raises -> Int:
    """The generation word; 0 when the entry's dictionary is not watched."""
    var g = MODEL_RECORDS.get_or_create_ptr()
    if not g[].entries[model_key].watched:
        return 0
    return g[].entries[model_key].generation


def records_run_bump(model_key: Int) raises -> Int:
    """Advance the generation word; the new word, 0 when unwatched."""
    var g = MODEL_RECORDS.get_or_create_ptr()
    g[].entries[model_key].generation += 1
    if not g[].entries[model_key].watched:
        return 0
    return g[].entries[model_key].generation


def _need(model_key: Int) raises:
    var g = MODEL_RECORDS.get_or_create_ptr()
    if model_key not in g[].entries:
        raise Error("model record slot requested for an object without an entry (call records_open)")


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
