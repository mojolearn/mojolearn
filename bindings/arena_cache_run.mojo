# SPDX-License-Identifier: Apache-2.0
"""RUN of `DeviceCache` and the `resident()` scope (lane arena, 2026-11-21).

Typed find / put / publish / free sequences on the native cache and scope
states of `core/arena_state.mojo` (`ARENA_CACHES`, `ARENA_SCOPES`) and the
x_prep transport entry (`core/prep_entry_table.mojo`: `XP_DEV_PUT`,
`XP_DEV_FREE`, `XP_DEV_LIVE`) or, for a cache on an x_metrics image, its
program entry (`MP_DEV_PUT`, `MP_DEV_FREE`, `MP_DEV_LIVE`; lane
vdf-xm-transport). No Python import (tools/python_free_dirs.sh):
a door resolves the handle word and the entry once per public operation
(UNPACK), calls these with the interpreter lock released, and unpins the
returned pin tokens afterwards (PACK).

Order kept from the dictionary version: an entry is unlinked BEFORE its
device free, every independent free is attempted, the first failure is the
one reported, and a failed entry is linked again for a retry.

Serial by nature: the frees of one close go to one binding store on one
device context, which serializes them (each is a slot bookkeeping update,
not work); there is nothing to split across cores.

ONE CACHE PER BINDING FAMILY (lane vdf-scope-run, 2026-11-24). A device
buffer belongs to the binding whose store made it, so a `resident()` scope
keeps one cache per (binding family `prefix`, image route), as
`ArenaScopeState.cache_of` always keyed it. What was x_prep-only is the
STORE a cache frees into at the scope's close: it was re-resolved from the
route word as the x_prep transport entry. Now the scope remembers, per
(prefix, route), the transport entry the first `arena_scope_cache` call
handed it (`_ARENA_FAMILY_STORES`, process-wide), and the close frees into
that entry; x_prep keeps its resolution (no entry recorded for it is ever
needed). THE ADDITIVE API for a member family with its own device store
(trees, linear, cluster, neural):
  * its image exports a TRANSPORT ENTRY with the x_prep transport numbering
    and signature (x_prep/transport_abi.mojo: `ARENA_STORE_PUT` = XP_DEV_PUT
    a: [src] i: [n_words] -> slot id; `ARENA_STORE_FREE` i: [id];
    `ARENA_STORE_TAKE_ROWS` a: [idx] i: [src_id, row_words, n_idx] -> slot
    id; `ARENA_STORE_LIVE` -> live slots), over its own
    core/device_store.mojo store; its member route carries that EntryRef;
  * a member call with `call.scope` / `call.pin` (MEMBER_CAN_SCOPE) calls
    `arena_scope_store_put(call.scope, store, "<prefix>", x_address, words,
    call.pin)` -> `ArenaCachePut` (sets `call.pin_kept` when `.pinned`), and
    takes fold rows on its device with `arena_store_take_rows(store,
    put.store_id, row_words, idx_address, n_idx)` (a fresh slot the member
    frees with `arena_cache_free(store, id)` after its run);
  * nothing else: the scope's close frees every family's entries into the
    family's own store, and `resident.stats()` lists each family's cache.
"""
from std.memory import ArcPointer
from core.arena_state import (
    ARENA_CACHES, ARENA_SCOPES, ARENA_HANDLE_NONE, ArenaStoreEntry, ArenaScopeState,
    DeviceCacheState, arena_handle, arena_handle_word,
)
from core.entry_table import EntryRef, entry_ref_mode, entry_ref_vendor, entry_ref_word
from core.prep_entry_table import PrepEntryKey, xp_transport_ref, xp_transport_call
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from x_prep.transport_abi import XP_DEV_PUT, XP_DEV_FREE, XP_DEV_LIVE, XP_DEV_TAKE_ROWS, XP_OP_COUNT
from core.linear_entry_table import LinearEntryKey, linear_table_entry, LINEAR_TABLE_METRICS
from x_metrics.descriptor_schema import METRICS_PROGRAM_ABI, MP_OPERATION_COUNT, MP_DEV_PUT, MP_DEV_FREE, MP_DEV_LIVE

#: The transport operations every family store answers (the x_prep numbering).
comptime ARENA_STORE_PUT = XP_DEV_PUT
comptime ARENA_STORE_FREE = XP_DEV_FREE
comptime ARENA_STORE_TAKE_ROWS = XP_DEV_TAKE_ROWS
comptime ARENA_STORE_LIVE = XP_DEV_LIVE
#: The family whose store is resolved by route alone (no record needed).
comptime ARENA_PREFIX_X_PREP = "x_prep"


@fieldwise_init
struct _FamilyStore(Copyable, Movable):
    """One family's transport entry on one image route."""
    var prefix: String
    var route: Int
    var store: EntryRef


struct _FamilyStores(Defaultable, Movable):
    """The transport entries of the family stores a scope has cached into,
    process-wide (an image's entry lives for the process). A handful of
    rows: one per family and image."""
    var rows: List[_FamilyStore]
    var lock: List[Int64]

    def __init__(out self):
        self.rows = List[_FamilyStore]()
        self.lock = List[Int64](length=1, fill=Int64(0))


comptime _ARENA_FAMILY_STORES = SharedSlot[_FamilyStores, "MojolearnArenaFamilyStoresV1", 1]


def _family_store_note(prefix: String, route: Int, store: EntryRef) raises:
    """Record the transport entry of (`prefix`, `route`) once (x_prep needs
    none). Called when a scope creates that family's cache."""
    if prefix == ARENA_PREFIX_X_PREP:
        return
    var slot = _ARENA_FAMILY_STORES.get_or_create_ptr()
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](slot[].lock.unsafe_ptr())
    shared_slot_lock(word)
    try:
        for i in range(len(slot[].rows)):  # driver-loop: over the recorded family stores (a handful)
            if slot[].rows[i].route == route and slot[].rows[i].prefix == prefix:
                return
        slot[].rows.append(_FamilyStore(prefix.copy(), route, store))
    finally:
        shared_slot_unlock(word)


#: The route bit of a cache on an x_metrics image's resident store (lane
#: vdf-xm-transport, 2026-11-24): the x_metrics program entry
#: (x_metrics/descriptor_schema.mojo, `MP_DEV_*`) instead of the x_prep
#: transport entry. Two families of one tier and vendor have two stores (each
#: image owns its own, next to its own device context), so two routes.
comptime ARENA_ROUTE_METRICS = 1 << 16


def arena_store_is_metrics(store: EntryRef) -> Bool:
    """Whether a resolved store entry is an x_metrics program entry: its
    layout word carries the metrics ABI's operation count (the x_prep
    transport's differs, checked at compile time)."""
    comptime assert MP_OPERATION_COUNT != XP_OP_COUNT, "the store families are told apart by their operation counts"
    return ((store.word >> 16) & 0xFFFFFFFF) == MP_OPERATION_COUNT


def arena_metrics_store(mode: Int, vendor: Int) raises -> EntryRef:
    """The x_metrics program entry of tier `mode` on `vendor` (the image's
    published table, core/linear_entry_table.mojo LINEAR_TABLE_METRICS) as a
    resident store entry; the table refusals raise here."""
    return EntryRef(linear_table_entry[LINEAR_TABLE_METRICS](LinearEntryKey(mode, vendor), 0,
                                                              String(METRICS_PROGRAM_ABI), MP_OPERATION_COUNT),
                    entry_ref_word(mode, vendor, MP_OPERATION_COUNT))


def arena_cache_route(store: EntryRef) -> Int:
    """The route word of a resolved store entry: its mode and vendor words,
    and the family bit of an x_metrics store. Two caches with one route share
    one device store."""
    var family = ARENA_ROUTE_METRICS if arena_store_is_metrics(store) else 0
    return family | (entry_ref_mode(store) << 8) | entry_ref_vendor(store)


def arena_cache_store(route: Int) raises -> EntryRef:
    """The store entry a route word names (resolved once per scope close; a
    cache owner object holds its own)."""
    if (route & ARENA_ROUTE_METRICS) != 0:
        return arena_metrics_store((route >> 8) & 0xFF, route & 0xFF)
    return xp_transport_ref(PrepEntryKey((route >> 8) & 0xFF, route & 0xFF))


def arena_cache_store(prefix: String, route: Int) raises -> EntryRef:
    """The transport entry of the family `prefix` on `route`: the one its
    scope cache was created with (`arena_scope_cache`), x_prep's by route."""
    if prefix == ARENA_PREFIX_X_PREP:
        return arena_cache_store(route)
    var slot = _ARENA_FAMILY_STORES.get_or_create_ptr()
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](slot[].lock.unsafe_ptr())
    shared_slot_lock(word)
    try:
        for i in range(len(slot[].rows)):  # driver-loop: over the recorded family stores (a handful)
            if slot[].rows[i].route == route and slot[].rows[i].prefix == prefix:
                return slot[].rows[i].store
    finally:
        shared_slot_unlock(word)
    raise Error("RuntimeError: resident scope: no device store recorded for the binding family '" + prefix + "'")


def arena_cache_new(store: EntryRef, prefix: String) raises -> Int:
    """A fresh cache state on `store`; its handle word."""
    return arena_handle_word(ARENA_CACHES.insert(DeviceCacheState(arena_cache_route(store), prefix)))


def arena_cache_put(store: EntryRef, address: Int, words: Int) raises -> Int:
    var source: List[Int] = [address]
    var counts: List[Int] = [words]
    return xp_transport_call(store, MP_DEV_PUT if arena_store_is_metrics(store) else XP_DEV_PUT, source, counts)


def arena_cache_free(store: EntryRef, store_id: Int) raises:
    var ids: List[Int] = [store_id]
    _ = xp_transport_call(store, MP_DEV_FREE if arena_store_is_metrics(store) else XP_DEV_FREE, List[Int](), ids)


def arena_cache_live(store: EntryRef) raises -> Int:
    """Live slots of the binding store (every cache on it)."""
    return xp_transport_call(store, MP_DEV_LIVE if arena_store_is_metrics(store) else XP_DEV_LIVE,
                             List[Int](), List[Int]())


@fieldwise_init
struct ArenaCachePut(ImplicitlyCopyable, Movable):
    """The id of a buffer's resident copy after an upload, and whether the
    cache kept the caller's pin (False: another thread published the same
    buffer first; the caller drops its pin)."""
    var store_id: Int
    var pinned: Bool


def arena_cache_upload(word: Int, store: EntryRef, address: Int, words: Int, pin: Int) raises -> ArenaCachePut:
    """Put (`address`, `words`) and publish it in the cache `word`. The put
    runs outside the cache lock. A lost publication race, or a cache released
    while the put ran, frees this call's own id."""
    var store_id = arena_cache_put(store, address, words)
    var kept = store_id
    try:
        # Acquire after the put: a cache released meanwhile refuses here.
        var lease = ARENA_CACHES.acquire(arena_handle(word))
        kept = lease[].publish(address, words, store_id, pin)
    except error:
        try:
            arena_cache_free(store, store_id)
        except cleanup_error:
            raise Error("RuntimeError: DeviceCache publication failed: " + String(error) + "; cleanup of the unpublished slot "
                        + String(store_id) + " also failed: " + String(cleanup_error))
        raise error^
    if kept != store_id:
        arena_cache_free(store, store_id)
        return ArenaCachePut(kept, False)
    return ArenaCachePut(store_id, True)


def arena_cache_find_or_put(word: Int, store: EntryRef, address: Int, words: Int, pin: Int) raises -> ArenaCachePut:
    """`id_of` for a native caller: the resident id (counted as a hit), else
    an upload."""
    var lease = ARENA_CACHES.acquire(arena_handle(word))
    var found = lease[].find(address, words, True)
    if found >= 0:
        return ArenaCachePut(found, False)
    return arena_cache_upload(word, store, address, words, pin)


struct ArenaCacheReleased(Movable):
    """What a release, drop or close returns to the boundary: the pin tokens
    of the entries that left the cache (to unpin after every free and
    publication) and the first failure's text (empty: none)."""
    var pins: List[Int]
    var failure: String

    def __init__(out self):
        self.pins = List[Int]()
        self.failure = String("")

    def keep(mut self, text: String):
        if self.failure.byte_length() == 0:
            self.failure = text


def _free_or_restore(lease: ArcPointer[DeviceCacheState], store: EntryRef, entry: ArenaStoreEntry,
                     mut released: ArenaCacheReleased) raises:
    # One unlinked entry: free it; a failed free links it again for a retry
    # (its pin stays with the cache).
    try:
        arena_cache_free(store, entry.store_id)
        released.pins.append(entry.pin)
    except error:
        released.keep(String(error))
        var linked = False
        try:
            linked = lease[].restore(entry)
        except restore_error:
            released.keep(String(restore_error))
        if not linked:
            released.pins.append(entry.pin)


def arena_cache_close(word: Int, store: EntryRef) raises -> ArenaCacheReleased:
    """Free every entry of the cache `word` (`close`, context exit)."""
    var lease = ARENA_CACHES.acquire(arena_handle(word))
    # Unlink everything first: no other thread finds an entry being freed.
    var drained = lease[].drain()
    var released = ArenaCacheReleased()
    for i in range(len(drained)):  # driver-loop: attempt every independent release (one store bookkeeping free each)
        _free_or_restore(lease, store, drained[i], released)
    return released^


def arena_cache_release(word: Int, store: EntryRef, address: Int, words: Int) raises -> ArenaCacheReleased:
    """Free the entry of one buffer (`release`); nothing when it is absent."""
    var lease = ARENA_CACHES.acquire(arena_handle(word))
    var released = ArenaCacheReleased()
    var removed = lease[].remove(address, words)
    if removed.store_id >= 0:
        _free_or_restore(lease, store, removed, released)
    return released^


def arena_cache_drop(word: Int, store: EntryRef, addresses: List[Int], words: List[Int]) raises -> ArenaCacheReleased:
    """Drop the resident copies of a run's in-out inputs: a stage rewrote
    their host words, so the copies are stale. Unpublish first, then free: a
    failed free never leaves a stale entry behind (the slot is reported)."""
    var lease = ARENA_CACHES.acquire(arena_handle(word))
    var released = ArenaCacheReleased()
    for i in range(len(addresses)):  # driver-loop: over the run's in-out inputs
        var removed = lease[].remove(addresses[i], words[i])
        if removed.store_id < 0:
            continue
        released.pins.append(removed.pin)
        try:
            arena_cache_free(store, removed.store_id)
        except error:
            released.failure += (("" if released.failure.byte_length() == 0 else "; ") + "slot "
                                 + String(removed.store_id) + ": " + String(error))
    return released^


def arena_cache_finalize(word: Int, store: EntryRef) -> List[Int]:
    """Best-effort end of a cache state (an owner's destructor, a scope's
    end): free what it holds, release the state, return every pin."""
    var pins = List[Int]()
    try:
        var handle = arena_handle(word)
        var lease = ARENA_CACHES.acquire(handle)
        var drained = lease[].drain()
        for i in range(len(drained)):  # driver-loop: over the entries the cache still holds
            pins.append(drained[i].pin)
            try:
                arena_cache_free(store, drained[i].store_id)
            except error:
                pass
        ARENA_CACHES.release(handle)
    except error:
        pass
    return pins^


# ---------------------------------------------------------------- scope


def arena_scope_new(min_words: Int) raises -> Int:
    """A fresh `resident()` scope state with its size floor; its handle word."""
    return arena_handle_word(ARENA_SCOPES.insert(ArenaScopeState(min_words)))


def arena_scope_floor(scope: Int) raises -> Int:
    var lease = ARENA_SCOPES.acquire(arena_handle(scope))
    return lease[].min_words


def arena_scope_cache(scope: Int, store: EntryRef, prefix: String) raises -> Int:
    """The scope's cache on `store` for the binding family `prefix`, created
    on first use. Called once per program run that keeps an input resident."""
    var lease = ARENA_SCOPES.acquire(arena_handle(scope))
    var route = arena_cache_route(store)
    var cache = lease[].cache_of(prefix, route)
    if cache == ARENA_HANDLE_NONE:
        _family_store_note(prefix, route, store)  # the close frees into this family's own store
        cache = arena_cache_new(store, prefix)
        lease[].add_cache(prefix, route, cache)
    return cache


def arena_scope_store_put(scope: Int, store: EntryRef, prefix: String, address: Int, words: Int,
                          pin: Int) raises -> ArenaCachePut:
    """THE MEMBER FAMILIES' CALL (module note): the store id of the host
    buffer (`address`, `words` 4-byte items) resident in the family
    `prefix`'s cache of the caller's `scope` on the family's transport entry
    `store`: found (a hit), or put once and published with `pin` (`.pinned`:
    the cache kept the pin, the member sets `call.pin_kept`)."""
    return arena_cache_find_or_put(arena_scope_cache(scope, store, prefix), store, address, words, pin)


def arena_store_take_rows(store: EntryRef, src_id: Int, row_words: Int, idx_address: Int, n_idx: Int) raises -> Int:
    """Rows `idx` (int64 at `idx_address`, `n_idx` of them) of the resident
    slot `src_id` (`row_words` words a row) gathered ON THE DEVICE into a
    fresh slot of the family store `store`; its id (the caller frees it)."""
    var source: List[Int] = [idx_address]
    var counts: List[Int] = [src_id, row_words, n_idx]
    return xp_transport_call(store, ARENA_STORE_TAKE_ROWS, source, counts)


def arena_scope_cache_found(scope: Int, store: EntryRef, prefix: String) raises -> Int:
    """The scope's cache on `store`, or `ARENA_HANDLE_NONE` when no run made
    one (nothing is created)."""
    var lease = ARENA_SCOPES.acquire(arena_handle(scope))
    return lease[].cache_of(prefix, arena_cache_route(store))


def arena_scope_close(scope: Int) raises -> ArenaCacheReleased:
    """Close every cache of the scope and release the caches' states (the
    scope's own state lives until its owner object is freed). A cache with a
    failed free stays in the scope, with the failed entries, for a retry."""
    var lease = ARENA_SCOPES.acquire(arena_handle(scope))
    var taken = lease[].take_caches()
    var released = ArenaCacheReleased()
    for i in range(len(taken)):  # driver-loop: close every independent cache of the scope (one per family and mode)
        var failed = False
        try:
            var closed = arena_cache_close(taken[i].cache, arena_cache_store(taken[i].prefix, taken[i].route))
            for j in range(len(closed.pins)):  # driver-loop: over the freed entries' pins
                released.pins.append(closed.pins[j])
            if closed.failure.byte_length() > 0:
                released.keep(closed.failure)
                failed = True
            else:
                ARENA_CACHES.release(arena_handle(taken[i].cache))
        except error:
            released.keep(String(error))
            failed = True
        if failed:
            lease[].add_cache(taken[i].prefix, taken[i].route, taken[i].cache)
    return released^


def arena_scope_finalize(scope: Int) -> List[Int]:
    """Best-effort end of a scope state (its owner's destructor): finalize
    each cache it lists, release the state, return every pin."""
    var pins = List[Int]()
    try:
        var handle = arena_handle(scope)
        var lease = ARENA_SCOPES.acquire(handle)
        var taken = lease[].take_caches()
        for i in range(len(taken)):  # driver-loop: over the caches the scope still lists
            try:
                var dropped = arena_cache_finalize(taken[i].cache, arena_cache_store(taken[i].prefix, taken[i].route))
                for j in range(len(dropped)):  # driver-loop: over the freed entries' pins
                    pins.append(dropped[j])
            except error:
                pass
        ARENA_SCOPES.release(handle)
    except error:
        pass
    return pins^


@fieldwise_init
struct ArenaCacheCounts(Copyable, Movable):
    """One cache's counters for `resident.stats()`."""
    var prefix: String
    var uploads: Int
    var hits: Int


def arena_scope_stats(scope: Int) raises -> List[ArenaCacheCounts]:
    var lease = ARENA_SCOPES.acquire(arena_handle(scope))
    var counts = List[ArenaCacheCounts]()
    for i in range(len(lease[].caches)):  # driver-loop: over the scope's caches, in creation order
        var cache = ARENA_CACHES.acquire(arena_handle(lease[].caches[i].cache))
        counts.append(ArenaCacheCounts(lease[].caches[i].prefix, cache[].uploads, cache[].hits))
    return counts^
