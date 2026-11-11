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
- Model death: the records module's `records_release` posts the dead
  model's key on its dead-key ring (bindings/native_model_records.mojo);
  `cres_lookup` and `cres_store` drain the ring at entry
  (`records_dead_keys_since`, from the last sequence word this table saw)
  and `cres_forget_key` drops every family's entry of each dead key; when
  the ring overflowed since the last drain every entry is released. The
  drain runs before any lookup or store, so a key on the ring always names a
  model that died after its entry was made. Held objects keep their
  addresses taken, so a new model at the same address can never be served
  the old copy.
- Nothing here computes on data: the device copy is the same bytes."""
from std.python import Python, PythonObject
# Imports: std and the records module's contract only (the records module
# never reaches this table: this table polls its dead-key ring).
from bindings.native_model_records import model_identity_key, model_generation, records_dead_keys_since
from core.shared_slot import SharedSlot
from bindings.kmeans_variants_resident_boundary import VariantsResidentLease, variants_resident_retire
from bindings.spectral_native_boundary import SpectralStatePins
from bindings.spectral_resident_boundary import SpectralResidentLease, spectral_resident_retire

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


struct CresHeld(Copyable, Movable):
    # Copyable (all fields are): Dict iteration (keys() included) requires Copyable key and value types.
    var generation: Int
    var objects: PythonObject
    var dims: List[Int]
    var binding: PythonObject
    var release: PythonObject
    var query: PythonObject
    var handle: Int
    var nbytes: Int
    var tick: Int
    var native_variant: Optional[VariantsResidentLease]
    var native_spectral: Optional[SpectralResidentLease]

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
        self.native_variant = None
        self.native_spectral = None


struct CresTable(Defaultable, Movable):
    var entries: Dict[Int, CresHeld]
    var tick: Int
    var bytes: Int
    # The records module's dead-key sequence word this table drained to.
    var seen: Int

    def __init__(out self):
        self.entries = Dict[Int, CresHeld]()
        self.tick = 0
        self.bytes = 0
        self.seen = 0


comptime _CRES = SharedSlot[CresTable, "MojolearnClusterResidentHeldV6"]


def _table() raises -> UnsafePointer[CresTable, MutAnyOrigin]:
    # Only the table pointer. The release-hook registration that lived next
    # to it (a thin function pointer handed to the records module) is gone:
    # it was the trigger of the compiler deadlock bisected to this table
    # (lanes ne-fix-hang4/5); the public entries poll instead (_drain_dead).
    return _CRES.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _release_quiet(var held: CresHeld):
    """The minting binding's release entry on the handle; quiet for a
    binding that cannot be reached any more."""
    if held.native_spectral:
        spectral_resident_retire(held.native_spectral.take())
        _ = held^
        return
    if held.native_variant:
        # Dropping the table reference cannot retire an active query lease.
        # The entry was removed/accounted before this native destructor runs.
        variants_resident_retire(held.native_variant.take())
        _ = held^
        return
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
    """Drop every family's entry of `model_key` (the model died; called by
    the dead-key drain). Non-raising."""
    for family in range(1, _CRES_FAMILIES):  # driver-loop: over the family codes
        _forget_slot(model_key + family)


def _forget_all(t: UnsafePointer[CresTable, MutAnyOrigin]) raises:
    """Release every entry (the dead-key ring overflowed since the last
    drain, so which models died is not known)."""
    var slots = List[Int]()
    for key in t[].entries.keys():  # driver-loop: over at most CRES_HELD_MAX held entries
        slots.append(key)
    for j in range(len(slots)):  # driver-loop: over at most CRES_HELD_MAX held entries
        _forget_slot(slots[j])


def _drain_dead(t: UnsafePointer[CresTable, MutAnyOrigin]) raises:
    """Forget every model released since the last drain (GIL held, the
    public entries' rule). The sequence word is advanced before any release
    runs: a release entry that frees another model only posts to the ring."""
    var dead = List[Int]()
    var now = records_dead_keys_since(t[].seen, dead)
    if now < 0:
        t[].seen = -now
        _forget_all(t)
        return
    t[].seen = now
    for j in range(len(dead)):  # driver-loop: over models released since the last drain
        cres_forget_key(dead[j])


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
    _drain_dead(t)
    if slot not in t[].entries:
        return PythonObject(None)
    var generation = model_generation(model)
    var query = PythonObject(None)
    var handle = 0
    ref held = t[].entries[slot]
    var same = (not held.native_variant and not held.native_spectral and held.generation == generation and held.binding is binding and _same_dims(held.dims, dims)
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
    var t = _table()
    _drain_dead(t)
    _forget_slot(slot)
    while len(t[].entries) > 0 and (len(t[].entries) >= CRES_HELD_MAX or t[].bytes + nbytes > CRES_HELD_BYTES):  # driver-loop: LRU eviction over held entries
        if not _evict_one(t):
            break
    t[].tick += 1
    t[].bytes += nbytes
    t[].entries[slot] = CresHeld(model_generation(model), objects, dims^, binding, release, query, handle, nbytes,
                                 t[].tick)


# Variant calls retain a native Arc and its boundary pins together. Existing
# families keep their old entries; both forms share the same LRU/cap/accounting.
def cres_variant_lookup(model: PythonObject, binding: PythonObject, centers: PythonObject,
                        provider: PythonObject, dims: List[Int]) raises -> Optional[VariantsResidentLease]:
    var slot = model_identity_key(model) + CRES_X_CLUSTER
    var t = _table()
    _drain_dead(t)
    if slot not in t[].entries:
        return None
    var generation = model_generation(model)
    var matched = False
    var result = Optional[VariantsResidentLease]()
    # GIL-held metadata only. Copy the Arc and both Python pins before any
    # released-GIL work; never keep a borrowed Dict value across that work.
    ref held = t[].entries[slot]
    if held.native_variant:
        matched = (held.generation == generation and held.binding is binding
            and held.objects is centers and _same_dims(held.dims, dims)
            and held.native_variant.value().provider is provider)
    if matched:
        t[].tick += 1
        held.tick = t[].tick
        result = held.native_variant.value().copy()
    else:
        _forget_slot(slot)
    return result^


def cres_variant_store(model: PythonObject, binding: PythonObject, var dims: List[Int],
                       var lease: VariantsResidentLease, nbytes: Int) raises:
    var slot = model_identity_key(model) + CRES_X_CLUSTER
    var t = _table()
    _drain_dead(t)
    _forget_slot(slot)
    while len(t[].entries) > 0 and (len(t[].entries) >= CRES_HELD_MAX or t[].bytes + nbytes > CRES_HELD_BYTES):  # driver-loop: unchanged bounded LRU eviction
        if not _evict_one(t):
            break
    # A release in the eviction loop may let a concurrent query publish this
    # same slot. Detach that late entry with the GIL held, then finish the new
    # publication/accounting before retiring it outside the Dict operation.
    var replaced = Optional[CresHeld]()
    if slot in t[].entries:
        replaced = t[].entries.pop(slot)
        t[].bytes -= replaced.value().nbytes
    t[].tick += 1
    t[].bytes += nbytes
    var held = CresHeld(model_generation(model), lease.centers, dims^, binding,
        PythonObject(None), PythonObject(None), lease.native[].handle, nbytes, t[].tick)
    held.native_variant = lease^
    t[].entries[slot] = held^
    if replaced:
        _release_quiet(replaced.take())


# Spectral shares only the bounded LRU/accounting mechanism. Its native Arc
# retains the minting driver, and its boundary lease pins all five fitted
# objects. No private Python object tuple or callable/handle envelope is made.
def cres_spectral_lookup(model: PythonObject, binding: PythonObject, provider: PythonObject,
                         state: SpectralStatePins, training: PythonObject,
                         dims: List[Int]) raises -> Optional[SpectralResidentLease]:
    var slot = model_identity_key(model) + CRES_SPECTRAL
    var t = _table()
    _drain_dead(t)
    if slot not in t[].entries:
        return None
    var generation = model_generation(model)
    var matched = False
    var result = Optional[SpectralResidentLease]()
    ref held = t[].entries[slot]
    if held.native_spectral:
        ref lease = held.native_spectral.value()
        matched = (held.generation == generation and held.binding is binding
            and _same_dims(held.dims, dims) and lease.provider is provider
            and lease.training is training and lease.state.same(state))
    if matched:
        t[].tick += 1
        held.tick = t[].tick
        result = held.native_spectral.value().copy()
    else:
        _forget_slot(slot)
    return result^


def cres_spectral_store(model: PythonObject, binding: PythonObject, var dims: List[Int],
                        var lease: SpectralResidentLease, nbytes: Int) raises:
    var slot = model_identity_key(model) + CRES_SPECTRAL
    var t = _table()
    _drain_dead(t)
    _forget_slot(slot)
    while len(t[].entries) > 0 and (len(t[].entries) >= CRES_HELD_MAX or t[].bytes + nbytes > CRES_HELD_BYTES):  # driver-loop: unchanged bounded LRU eviction
        if not _evict_one(t):
            break
    # Retirement above may release the GIL. Detach any late replacement,
    # publish/account atomically under the GIL, then retire outside Dict work.
    var replaced = Optional[CresHeld]()
    if slot in t[].entries:
        replaced = t[].entries.pop(slot)
        t[].bytes -= replaced.value().nbytes
    t[].tick += 1
    t[].bytes += nbytes
    var held = CresHeld(model_generation(model), PythonObject(None), dims^, binding,
        PythonObject(None), PythonObject(None), lease.native[].handle, nbytes, t[].tick)
    held.native_spectral = lease^
    t[].entries[slot] = held^
    if replaced:
        _release_quiet(replaced.take())
