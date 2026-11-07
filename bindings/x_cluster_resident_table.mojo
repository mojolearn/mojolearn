# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The provider side of the resident x_cluster centers (lane
ne1-a-cluster-spectral, owed rows N024 and N021; plan workstream A).

One native side table, keyed by the model's identity
(`model_identity_key`, bindings/native_model_records.mojo), holds per model
the handle the GPU binding minted (`x_cluster_centers_prepare`,
x_cluster/resident.mojo) together with what it was minted from: the model's
generation word, the very `cluster_centers_` object (held, so its address
cannot be reused by another array while the entry lives), the binding, the
shape and the byte count. Never the model's `__dict__`, never a content hash.

Rules (plan section 2A):
- Only an exact mojolearn `Array` is held: it is made read-only when the
  copy is taken (`fitted_freeze`), so an in-place write raises NumPy's
  read-only error. Replacing `cluster_centers_` changes the key and the next
  query uploads the new object once. Any other storage keeps the per-call
  upload (bindings/cluster_native.mojo).
- A hit needs the same generation word, the same centers object, the same
  binding and the same shape; any difference releases the old handle and
  mints a new one. A fit forgets the model's entry at once (`xc_resident_forget`).
- Bounded: at most `XC_HELD_MAX` models and `XC_HELD_BYTES` bytes of centers;
  the least recently used entry is released first, and an evicted model
  uploads again at its next query.
- Model death: `xc_resident_forget_key(model_key)` drops the entry (called
  by the records module's release, OWED-SHARED to ne1-e-records); until that
  hook is wired a dead model's entry is released by the bound above.
- Nothing here computes on data: the device copy is the same bytes."""
from std.ffi import _Global
from std.python import Python, PythonObject
from bindings.family_common import family_call
from bindings.native_model_records import model_identity_key, model_generation

comptime XC_HELD_MAX = 64
"""Models with resident centers at once (a driver bound, not a shape rule)."""
comptime XC_HELD_BYTES = 1 << 30
"""Bytes of resident centers at once: the sequence store's 1 GiB budget
(sequence/perf3_seq_resident.mojo) until the per-device cap of plan 2A is
measured (profile item 14)."""


struct XcHeld(Movable):
    var generation: Int
    var centers: PythonObject
    var binding: PythonObject
    var release: PythonObject
    var query: PythonObject
    var k: Int
    var d: Int
    var handle: Int
    var tick: Int

    def __init__(out self, generation: Int, centers: PythonObject, binding: PythonObject, release: PythonObject,
                 query: PythonObject, k: Int, d: Int, handle: Int, tick: Int):
        self.generation = generation
        self.centers = centers
        self.binding = binding
        self.release = release
        self.query = query
        self.k = k
        self.d = d
        self.handle = handle
        self.tick = tick


struct XcHeldTable(Defaultable, Movable):
    var entries: Dict[Int, XcHeld]
    var tick: Int
    var bytes: Int

    def __init__(out self):
        self.entries = Dict[Int, XcHeld]()
        self.tick = 0
        self.bytes = 0


comptime _XC_HELD = _Global[StorageType=XcHeldTable, name="MojolearnXClusterResidentHeldV1",
                            init_fn=XcHeldTable.__init__]


def _table() raises -> UnsafePointer[XcHeldTable, MutAnyOrigin]:
    return _XC_HELD.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _release_quiet(var held: XcHeld):
    """The minting binding's `x_cluster_centers_release(handle)`; quiet for a
    binding that cannot be reached any more."""
    try:
        _ = family_call(held.release, PythonObject(held.handle))
    except:
        pass
    _ = held^


def xc_resident_forget_key(model_key: Int):
    """Drop the entry of `model_key` (the model died, was refitted or its
    centers changed). Non-raising: it runs from death callbacks."""
    try:
        var t = _table()
        if model_key not in t[].entries:
            return
        var held = t[].entries.pop(model_key)
        t[].bytes -= held.k * held.d * 4
        _release_quiet(held^)
    except:
        pass


def xc_resident_forget(model: PythonObject):
    """`xc_resident_forget_key` for a live model (its fit calls it)."""
    try:
        xc_resident_forget_key(model_identity_key(model))
    except:
        pass


def xc_resident_count() raises -> Int:
    """The number of models with resident centers (tests only)."""
    return len(_table()[].entries)


def _evict_one(t: UnsafePointer[XcHeldTable, MutAnyOrigin]) raises -> Bool:
    """Release the least recently used entry; False when none was released."""
    var oldest_key = 0
    var oldest_tick = -1
    for item in t[].entries.items():  # driver-loop: over at most XC_HELD_MAX held models
        if oldest_tick < 0 or item.value.tick < oldest_tick:
            oldest_tick = item.value.tick
            oldest_key = item.key
    if oldest_tick < 0:
        return False
    var before = len(t[].entries)
    xc_resident_forget_key(oldest_key)
    return len(t[].entries) < before


def xc_resident_lookup(model: PythonObject, binding: PythonObject, centers: PythonObject, k: Int,
                       d: Int) raises -> PythonObject:
    """`(query_entry, handle)` of the held copy minted from this very
    `centers` object by `binding` at this shape and the model's current
    generation, else None (a stale entry is released here)."""
    var key = model_identity_key(model)
    var t = _table()
    if key not in t[].entries:
        return PythonObject(None)
    var generation = model_generation(model)
    var same = False
    var query = PythonObject(None)
    var handle = 0
    ref held = t[].entries[key]
    same = (held.generation == generation and held.centers is centers and held.binding is binding
            and held.k == k and held.d == d)
    if same:
        t[].tick += 1
        held.tick = t[].tick
        query = held.query
        handle = held.handle
    if not same:
        xc_resident_forget_key(key)
        return PythonObject(None)
    return Python.tuple(query, PythonObject(handle))


def xc_resident_store(model: PythonObject, binding: PythonObject, centers: PythonObject, release: PythonObject,
                      query: PythonObject, k: Int, d: Int, handle: Int) raises:
    """Hold a freshly minted handle for `model` (any older entry of the model
    is released first); the least recently used models are released while
    the table is over its bounds."""
    var key = model_identity_key(model)
    xc_resident_forget_key(key)
    var t = _table()
    var nbytes = k * d * 4
    while len(t[].entries) > 0 and (len(t[].entries) >= XC_HELD_MAX or t[].bytes + nbytes > XC_HELD_BYTES):  # driver-loop: LRU eviction over held models
        if not _evict_one(t):
            break
    t[].tick += 1
    t[].bytes += nbytes
    t[].entries[key] = XcHeld(model_generation(model), centers, binding, release, query, k, d, handle, t[].tick)
