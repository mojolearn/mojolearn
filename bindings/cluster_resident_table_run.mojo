# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The clustering families' resident fitted state: the TYPED table (RUN; no
Python). Split from bindings/cluster_resident_table.mojo (lane
resident-pin-tables, 2026-11-12): the Python pins (the minting binding and the
native leases, which hold the fitted objects) live in the door-side pin store
of bindings/py_cluster_resident.mojo, keyed by the same slot word
(`model key + family`). This table keeps per slot the model's generation
word, the binding's id() word, the shape words, the byte count and the LRU
tick, and decides lookups, eviction and the dead-key drain.

Rules (plan section 2A; unchanged by the split):
- A hit needs the same generation word, the same binding (identity word; the
  door pins it while the slot is listed), the same shape words, and the
  lease's own identity checks (made by the door); any difference releases
  the slot. A fit forgets the model's slot at once.
- Bounded: at most `CRES_HELD_MAX` entries and `CRES_HELD_BYTES` bytes; the
  least recently used entry is released first.
- Model death: the dead-key ring (core/records_death_ring.mojo) is drained at
  every lookup and store, before anything else; an overflow releases all.
- Every slot that leaves the table is appended to `gone`; the door takes it
  (`cres_run_take_gone`) right after each typed call, before anything can
  release the GIL, drops those slots' pins and retires their leases.
- Nothing here computes on data."""
from core.shared_slot import SharedSlot
from core.records_death_ring import records_dead_keys_since

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


struct CresWords(Copyable, Movable):
    # Copyable (all fields are): Dict iteration (keys() included) requires Copyable key and value types.
    var generation: Int
    var dims: List[Int]
    var binding: Int
    var nbytes: Int
    var tick: Int

    def __init__(out self, generation: Int, var dims: List[Int], binding: Int, nbytes: Int, tick: Int):
        self.generation = generation
        self.dims = dims^
        self.binding = binding
        self.nbytes = nbytes
        self.tick = tick


struct CresTable(Defaultable, Movable):
    var entries: Dict[Int, CresWords]
    var tick: Int
    var bytes: Int
    # The records module's dead-key sequence word this table drained to.
    var seen: Int
    # Slots released since the door last took them (their pins go).
    var gone: List[Int]

    def __init__(out self):
        self.entries = Dict[Int, CresWords]()
        self.tick = 0
        self.bytes = 0
        self.seen = 0
        self.gone = List[Int]()


# V8 (lane resident-pin-tables): words only; pins moved to the door.
comptime _CRES = SharedSlot[CresTable, "MojolearnClusterResidentHeldV8", 8]


def _table() raises -> UnsafePointer[CresTable, MutAnyOrigin]:
    return _CRES.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()


def cres_run_forget_slot(slot: Int) raises:
    """Release `slot` (its words leave; the door retires its pins)."""
    var t = _table()
    if slot not in t[].entries:
        return
    var held = t[].entries.pop(slot)
    t[].bytes -= held.nbytes
    t[].gone.append(slot)


def _forget_key(model_key: Int) raises:
    for family in range(1, _CRES_FAMILIES):  # driver-loop: over the family codes
        cres_run_forget_slot(model_key + family)


def cres_run_drain() raises:
    """Forget every model released since the last drain; everything when the
    dead-key ring overflowed since then."""
    var t = _table()
    var dead = List[Int]()
    var now = records_dead_keys_since(t[].seen, dead)
    if now < 0:
        t[].seen = -now
        var slots = List[Int]()
        for key in t[].entries.keys():  # driver-loop: over at most CRES_HELD_MAX held entries
            slots.append(key)
        for j in range(len(slots)):  # driver-loop: over at most CRES_HELD_MAX held entries
            cres_run_forget_slot(slots[j])
        return
    t[].seen = now
    for j in range(len(dead)):  # driver-loop: over models released since the last drain
        _forget_key(dead[j])


def cres_run_has(slot: Int) raises -> Bool:
    return slot in _table()[].entries


def _same_dims(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):  # small-loop(dims: a handful of shape words)
        if a[i] != b[i]:
            return False
    return True


def cres_run_matches(slot: Int, generation: Int, binding: Int, dims: List[Int]) raises -> Bool:
    """The slot's words match (the door adds the lease's identity checks)."""
    var t = _table()
    if slot not in t[].entries:
        return False
    ref held = t[].entries[slot]
    return held.generation == generation and held.binding == binding and _same_dims(held.dims, dims)


def cres_run_touch(slot: Int) raises:
    """Mark `slot` most recently used (a hit)."""
    var t = _table()
    if slot not in t[].entries:
        return
    t[].tick += 1
    t[].entries[slot].tick = t[].tick


def _evict_one(t: UnsafePointer[CresTable, MutAnyOrigin]) raises -> Bool:
    """Release the least recently used entry; False when none was released."""
    var oldest_slot = 0
    var oldest_tick = -1
    for key in t[].entries.keys():  # driver-loop: over at most CRES_HELD_MAX held entries
        var tick = t[].entries[key].tick
        if oldest_tick < 0 or tick < oldest_tick:
            oldest_tick = tick
            oldest_slot = key
    if oldest_tick < 0:
        return False
    var before = len(t[].entries)
    cres_run_forget_slot(oldest_slot)
    return len(t[].entries) < before


def cres_run_reserve(slot: Int, nbytes: Int) raises:
    """Before a store: drop `slot`, then evict least recently used entries
    until one more entry of `nbytes` fits."""
    var t = _table()
    cres_run_forget_slot(slot)
    while len(t[].entries) > 0 and (len(t[].entries) >= CRES_HELD_MAX or t[].bytes + nbytes > CRES_HELD_BYTES):  # driver-loop: unchanged bounded LRU eviction
        if not _evict_one(t):
            break


def cres_run_publish(slot: Int, generation: Int, var dims: List[Int], binding: Int, nbytes: Int) raises -> Bool:
    """Publish `slot`'s words. True when a late entry of the same slot (a
    concurrent store while the door's retirements released the GIL) was
    replaced: the door then retires that entry's pins (not via `gone`)."""
    var t = _table()
    var replaced = False
    if slot in t[].entries:
        var late = t[].entries.pop(slot)
        t[].bytes -= late.nbytes
        replaced = True
    t[].tick += 1
    t[].bytes += nbytes
    t[].entries[slot] = CresWords(generation, dims^, binding, nbytes, t[].tick)
    return replaced


def cres_run_take_gone() raises -> List[Int]:
    """The slots released since the last take (the door drops their pins)."""
    var t = _table()
    var gone = List[Int]()
    if len(t[].gone) == 0:
        return gone^
    swap(gone, t[].gone)
    return gone^
