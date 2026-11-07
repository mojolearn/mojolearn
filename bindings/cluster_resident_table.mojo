# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The provider side of the clustering families' resident fitted state (lane
ne1-a-cluster-spectral, plan workstream A): the x_cluster `cluster_centers_`
(owed rows N024, N021) and the SpectralClustering prediction state (N326).

One native side table, keyed by the model's identity (`model_identity_key`,
bindings/native_model_records.mojo) and a family code, holds per model the
handle a binding minted together with what it was minted from: the model's
generation word, the very fitted objects (held, so their addresses cannot
be reused by other arrays while the entry lives), the binding, the shape
words and the byte count. Never the model's `__dict__`, never a content hash.

Rules (plan section 2A):
- Callers hold only exact mojolearn `Array`s, made read-only when the copy
  is taken (`fitted_freeze`), so an in-place write raises NumPy's read-only
  error. Replacing an attribute changes the key and the next call uploads
  the new object once. Any other storage keeps the per-call upload.
- A hit needs the same generation word, the same objects (identity), the
  same binding and the same shape words; any difference releases the old
  handle and mints a new one. A fit forgets the model's entry at once.
- Bounded: at most `CRES_HELD_MAX` entries and `CRES_HELD_BYTES` bytes; the
  least recently used entry is released first, and an evicted model uploads
  again at its next call.
- Model death: `cres_forget_key(model_key)` drops every family's entry. It
  is registered once with the records module
  (`records_register_release_hook`, bindings/native_model_records.mojo) the
  first time `cres_lookup` or `cres_store` runs, and the records module's
  `records_release` calls it with the dead model's key. Held objects keep
  their addresses taken, so a new model at the same address can never be
  served the old copy.
- Nothing here computes on data: the device copy is the same bytes."""
from std.ffi import _Global
from std.python import Python, PythonObject
# Imports: std and the records module's contract only (the records module
# reaches this table through its release hook, never by import).
from bindings.native_model_records import model_identity_key, model_generation, records_register_release_hook

comptime CRES_X_CLUSTER = 1
"""Family code: x_cluster `cluster_centers_` (MiniBatchKMeans, BisectingKMeans,
MeanShift, AffinityPropagation)."""
comptime CRES_SPECTRAL = 2
"""Family code: SpectralClustering's prediction state."""
comptime _CRES_FAMILIES = 4
"""Family codes are below this; a model identity is an object address
(16-byte aligned), so `key + family` never collides."""
comptime CRES_HELD_MAX = 64
"""Entries held at once (a driver bound, not a shape rule)."""
comptime CRES_HELD_BYTES = 1 << 30
"""Bytes held at once: the sequence store's 1 GiB budget
(sequence/perf3_seq_resident.mojo) until the per-device cap of plan 2A is
measured (profile item 14)."""


struct CresHeld(Movable):
    var generation: Int
    var objects: PythonObject
    var dims: List[Int]
    var binding: PythonObject
    var release: PythonObject
    var query: PythonObject
    var handle: Int
    var nbytes: Int
    var tick: Int

    def __init__(out self, generation: Int, objects: PythonObject, var dims: List[Int], binding: PythonObject,
                 release: PythonObject, query: PythonObject, handle: Int, nbytes: Int, tick: Int):
        self.generation = generation
        self.objects = objects
        self.dims = dims^
        self.binding = binding
        self.release = release
        self.query = query
        self.handle = handle
        self.nbytes = nbytes
        self.tick = tick


struct CresTable(Defaultable, Movable):
    var entries: Dict[Int, CresHeld]
    var tick: Int
    var bytes: Int
    # The once-flag of the release-hook registration.
    var hooked: Bool

    def __init__(out self):
        self.entries = Dict[Int, CresHeld]()
        self.tick = 0
        self.bytes = 0
        self.hooked = False


comptime _CRES = _Global[StorageType=CresTable, name="MojolearnClusterResidentHeldV2", init_fn=CresTable.__init__]


def _table() raises -> UnsafePointer[CresTable, MutAnyOrigin]:
    # Only the table pointer: the release-hook registration is its own step
    # (_hook_once), taken by the public entries. Registering here made
    # _table -> cres_forget_key -> _forget_slot -> _table a recursive cycle
    # through a function-pointer reference, the suspected trigger of the
    # compiler deadlock bisected to this lane's merge (ne-fix-hang4).
    return _CRES.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _hook_once(t: UnsafePointer[CresTable, MutAnyOrigin]):
    """Register cres_forget_key with the records module the first time the
    table is used (the sequence store's _hook_once shape)."""
    if t[].hooked:
        return
    # Set first: registering may run a release that reaches this table.
    t[].hooked = True
    # Bool: False (registry full) leaves dead entries to the LRU bound.
    _ = records_register_release_hook(cres_forget_key)


def _release_quiet(var held: CresHeld):
    """The minting binding's release entry on the handle; quiet for a
    binding that cannot be reached any more."""
    try:
        _ = held.release(PythonObject(held.handle))
    except:
        pass
    _ = held^


def _forget_slot(slot: Int):
    try:
        var t = _table()
        if slot not in t[].entries:
            return
        var held = t[].entries.pop(slot)
        t[].bytes -= held.nbytes
        _release_quiet(held^)
    except:
        pass


def cres_forget_key(model_key: Int):
    """Drop every family's entry of `model_key` (the model died).
    Non-raising: it runs from death callbacks."""
    for family in range(1, _CRES_FAMILIES):  # driver-loop: over the family codes
        _forget_slot(model_key + family)


def cres_forget(model: PythonObject, family: Int):
    """Drop `family`'s entry of a live model (its fit, a replaced state)."""
    try:
        _forget_slot(model_identity_key(model) + family)
    except:
        pass


def cres_count() raises -> Int:
    """The number of held entries (tests only)."""
    return len(_table()[].entries)


def _same_objects(a: PythonObject, b: PythonObject) raises -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):  # driver-loop: over the few fitted objects of one entry
        if not (a[i] is b[i]):
            return False
    return True


def _same_dims(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):  # small-loop(dims: a handful of shape words)
        if a[i] != b[i]:
            return False
    return True


def _evict_one(t: UnsafePointer[CresTable, MutAnyOrigin]) raises -> Bool:
    """Release the least recently used entry; False when none was released."""
    var oldest_slot = 0
    var oldest_tick = -1
    # Keys only: CresHeld is Movable, not Copyable, and Dict.items() requires Copyable values.
    for key in t[].entries.keys():  # driver-loop: over at most CRES_HELD_MAX held entries
        var tick = t[].entries[key].tick
        if oldest_tick < 0 or tick < oldest_tick:
            oldest_tick = tick
            oldest_slot = key
    if oldest_tick < 0:
        return False
    var before = len(t[].entries)
    _forget_slot(oldest_slot)
    return len(t[].entries) < before


def cres_lookup(model: PythonObject, family: Int, binding: PythonObject, objects: PythonObject,
                dims: List[Int]) raises -> PythonObject:
    """`(query_entry, handle)` of the entry minted from these very `objects`
    (a tuple) by `binding` at these shape words and the model's current
    generation, else None (a stale entry is released here)."""
    var slot = model_identity_key(model) + family
    var t = _table()
    _hook_once(t)
    if slot not in t[].entries:
        return PythonObject(None)
    var generation = model_generation(model)
    var query = PythonObject(None)
    var handle = 0
    ref held = t[].entries[slot]
    var same = (held.generation == generation and held.binding is binding and _same_dims(held.dims, dims)
                and _same_objects(held.objects, objects))
    if same:
        t[].tick += 1
        held.tick = t[].tick
        query = held.query
        handle = held.handle
    if not same:
        _forget_slot(slot)
        return PythonObject(None)
    return Python.tuple(query, PythonObject(handle))


def cres_store(model: PythonObject, family: Int, binding: PythonObject, objects: PythonObject, var dims: List[Int],
               release: PythonObject, query: PythonObject, handle: Int, nbytes: Int) raises:
    """Hold a freshly minted handle (the model's older entry of this family
    is released first); the least recently used entries are released while
    the table is over its bounds."""
    var slot = model_identity_key(model) + family
    _forget_slot(slot)
    var t = _table()
    _hook_once(t)
    while len(t[].entries) > 0 and (len(t[].entries) >= CRES_HELD_MAX or t[].bytes + nbytes > CRES_HELD_BYTES):  # driver-loop: LRU eviction over held entries
        if not _evict_one(t):
            break
    t[].tick += 1
    t[].bytes += nbytes
    t[].entries[slot] = CresHeld(model_generation(model), objects, dims^, binding, release, query, handle, nbytes,
                                 t[].tick)
