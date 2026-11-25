# SPDX-License-Identifier: Apache-2.0
"""RUN of `DeviceCache` and the `resident()` scope (lane arena, 2026-11-21).

Typed find / put / publish / free sequences on the native cache and scope
states of `core/arena_state.mojo` (`ARENA_CACHES`, `ARENA_SCOPES`) and the
x_prep transport entry (`core/prep_entry_table.mojo`: `XP_DEV_PUT`,
`XP_DEV_FREE`, `XP_DEV_LIVE`). No Python import (tools/python_free_dirs.sh):
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

THE PER-FAMILY OP MAP (lane gd-rf-msel, 2026-11-25). A family store whose
entry does not answer the x_prep transport numbering registers its own put,
free and live operation numbers (`ArenaStoreOps`) with its FIRST cache
(`arena_scope_cache(scope, store, ops, prefix)` / `arena_scope_store_put(...,
ops, ...)`): the ops travel with the caller's call (a constant of its
family), are recorded once in `_ARENA_FAMILY_STORES` beside the entry, and
the scope's close reads them from that record. No lookup is added to a put.
Every signature without `ops` is the transport numbering
(`arena_store_ops_transport`: put 4, free 5, live 7, words), which x_prep,
the resample store (resample/resample_store.mojo, family_store_entry) and
every member family store answer; the model-selection helper table registers
MH_STORE_PUT 21 (bytes), MH_STORE_FREE 15 and no live operation
(core/msel_scope_codes.mojo `MSEL_HELPER_STORE_OPS`).
"""
from std.atomic import Atomic, Ordering
from std.memory import ArcPointer
from core.arena_state import (
    ARENA_CACHES, ARENA_SCOPES, ARENA_HANDLE_NONE, ArenaStoreEntry, ArenaScopeState,
    DeviceCacheState, arena_handle, arena_handle_word,
)
from core.entry_table import EntryRef, entry_ref_mode, entry_ref_vendor
from core.prep_entry_table import PrepEntryKey, xp_transport_ref, xp_transport_call
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from x_prep.transport_abi import XP_DEV_PUT, XP_DEV_FREE, XP_DEV_LIVE, XP_DEV_TAKE_ROWS, XP_DEV_READ

#: The transport operations every family store answers (the x_prep numbering).
comptime ARENA_STORE_PUT = XP_DEV_PUT
comptime ARENA_STORE_FREE = XP_DEV_FREE
comptime ARENA_STORE_TAKE_ROWS = XP_DEV_TAKE_ROWS
comptime ARENA_STORE_LIVE = XP_DEV_LIVE
#: Lane vdf-scope-gp (an addition): a slot's words into host memory. The x_prep store answers it; a family store
#: that does not refuses it as an unknown operation.
comptime ARENA_STORE_READ = XP_DEV_READ
#: The family whose store is resolved by route alone (no record needed).
comptime ARENA_PREFIX_X_PREP = "x_prep"


@fieldwise_init
struct ArenaStoreOps(ImplicitlyCopyable, Movable):
    """A family store's operation numbers (module note, THE PER-FAMILY OP
    MAP): put a: [src] i: [count] -> id; free i: [id]; live -> count (-1:
    the store has none); `put_bytes`: the put's count is in bytes, not
    4-byte words."""
    var put: Int
    var free: Int
    var live: Int
    var put_bytes: Bool


@always_inline
def arena_store_ops_transport() -> ArenaStoreOps:
    """The x_prep transport numbering every transport-entry family store answers."""
    return ArenaStoreOps(XP_DEV_PUT, XP_DEV_FREE, XP_DEV_LIVE, False)


@fieldwise_init
struct _FamilyStore(Copyable, Movable):
    """One family's store entry on one image route and its operation numbers."""
    var prefix: String
    var route: Int
    var store: EntryRef
    var ops: ArenaStoreOps


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


def _family_store_note(prefix: String, route: Int, store: EntryRef, ops: ArenaStoreOps) raises:
    """Record the store entry of (`prefix`, `route`) and its operation numbers
    once (x_prep needs none). Called when a scope creates that family's cache."""
    if prefix == ARENA_PREFIX_X_PREP:
        return
    var slot = _ARENA_FAMILY_STORES.get_or_create_ptr()
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](slot[].lock.unsafe_ptr())
    shared_slot_lock(word)
    try:
        for i in range(len(slot[].rows)):  # driver-loop: over the recorded family stores (a handful)
            if slot[].rows[i].route == route and slot[].rows[i].prefix == prefix:
                return
        slot[].rows.append(_FamilyStore(prefix.copy(), route, store, ops))
    finally:
        shared_slot_unlock(word)


# TOMBSTONE: ARENA_ROUTE_METRICS, arena_store_is_metrics, arena_metrics_store (a resident() cache on a
# separate x_metrics store, told apart by the program entry's operation count; lane vdf-xm-transport)
# deleted 2026-11-24 by lane vdf-xm-wire: the x_metrics programs and the _mojolearn_metrics kernel entries
# read the ONE store, the x_prep image's (x_prep/store_slot.mojo), so every cache is an x_prep transport
# cache again. Never on main: recoverable at
# lane/vd-functions 55fed275ea (no patch, AGENTS.md "Deleting an experiment" part 3).


def arena_cache_route(store: EntryRef) -> Int:
    """The route word of a resolved x_prep transport entry: its mode and
    vendor words. Two caches with one route share one device store."""
    return (entry_ref_mode(store) << 8) | entry_ref_vendor(store)


def arena_cache_store(route: Int) raises -> EntryRef:
    """The transport entry a route word names (resolved once per scope
    close; a cache owner object holds its own)."""
    return xp_transport_ref(PrepEntryKey((route >> 8) & 0xFF, route & 0xFF))


def arena_cache_store(prefix: String, route: Int) raises -> EntryRef:
    """The store entry of the family `prefix` on `route`: the one its
    scope cache was created with (`arena_scope_cache`), x_prep's by route."""
    return arena_family_store(prefix, route).store


def arena_family_store(prefix: String, route: Int) raises -> _FamilyStore:
    """The store entry of the family `prefix` on `route` and the operation
    numbers it registered (x_prep: its route's transport entry, the
    transport numbering). Read once per cache at a scope's close."""
    if prefix == ARENA_PREFIX_X_PREP:
        return _FamilyStore(prefix.copy(), route, arena_cache_store(route), arena_store_ops_transport())
    var slot = _ARENA_FAMILY_STORES.get_or_create_ptr()
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](slot[].lock.unsafe_ptr())
    shared_slot_lock(word)
    try:
        for i in range(len(slot[].rows)):  # driver-loop: over the recorded family stores (a handful)
            if slot[].rows[i].route == route and slot[].rows[i].prefix == prefix:
                return slot[].rows[i].copy()
    finally:
        shared_slot_unlock(word)
    raise Error("RuntimeError: resident scope: no device store recorded for the binding family '" + prefix + "'")


def arena_cache_new(store: EntryRef, prefix: String) raises -> Int:
    """A fresh cache state on `store`; its handle word."""
    return arena_handle_word(ARENA_CACHES.insert(DeviceCacheState(arena_cache_route(store), prefix)))


def arena_cache_put(store: EntryRef, address: Int, words: Int) raises -> Int:
    return arena_cache_put(store, arena_store_ops_transport(), address, words)


def arena_cache_put(store: EntryRef, ops: ArenaStoreOps, address: Int, words: Int) raises -> Int:
    """Put `words` 4-byte words at `address` with the family's put operation
    (its count in bytes when `ops.put_bytes`); the new slot's id."""
    var source: List[Int] = [address]
    var counts: List[Int] = [4 * words if ops.put_bytes else words]
    return xp_transport_call(store, ops.put, source, counts)


def arena_cache_free(store: EntryRef, store_id: Int) raises:
    arena_cache_free(store, arena_store_ops_transport(), store_id)


def arena_cache_free(store: EntryRef, ops: ArenaStoreOps, store_id: Int) raises:
    var ids: List[Int] = [store_id]
    _ = xp_transport_call(store, ops.free, List[Int](), ids)


def arena_cache_live(store: EntryRef) raises -> Int:
    """Live slots of the binding store (every cache on it)."""
    return arena_cache_live(store, arena_store_ops_transport())


def arena_cache_live(store: EntryRef, ops: ArenaStoreOps) raises -> Int:
    """Live slots of a family store; refused for a store with no live operation."""
    if ops.live < 0:
        raise Error("RuntimeError: resident scope: this binding family's store has no live-slot operation")
    return xp_transport_call(store, ops.live, List[Int](), List[Int]())


@fieldwise_init
struct ArenaCachePut(ImplicitlyCopyable, Movable):
    """The id of a buffer's resident copy after an upload, and whether the
    cache kept the caller's pin (False: another thread published the same
    buffer first; the caller drops its pin)."""
    var store_id: Int
    var pinned: Bool


def arena_cache_upload(word: Int, store: EntryRef, address: Int, words: Int, pin: Int) raises -> ArenaCachePut:
    return arena_cache_upload(word, store, arena_store_ops_transport(), address, words, pin)


def arena_cache_upload(word: Int, store: EntryRef, ops: ArenaStoreOps, address: Int, words: Int,
                       pin: Int) raises -> ArenaCachePut:
    """Put (`address`, `words`) and publish it in the cache `word`. The put
    runs outside the cache lock. A lost publication race, or a cache released
    while the put ran, frees this call's own id."""
    var store_id = arena_cache_put(store, ops, address, words)
    var kept = store_id
    try:
        # Acquire after the put: a cache released meanwhile refuses here.
        var lease = ARENA_CACHES.acquire(arena_handle(word))
        kept = lease[].publish(address, words, store_id, pin)
    except error:
        try:
            arena_cache_free(store, ops, store_id)
        except cleanup_error:
            raise Error("RuntimeError: DeviceCache publication failed: " + String(error) + "; cleanup of the unpublished slot "
                        + String(store_id) + " also failed: " + String(cleanup_error))
        raise error^
    if kept != store_id:
        arena_cache_free(store, ops, store_id)
        return ArenaCachePut(kept, False)
    return ArenaCachePut(store_id, True)


def arena_cache_find_or_put(word: Int, store: EntryRef, address: Int, words: Int, pin: Int) raises -> ArenaCachePut:
    return arena_cache_find_or_put(word, store, arena_store_ops_transport(), address, words, pin)


def arena_cache_find_or_put(word: Int, store: EntryRef, ops: ArenaStoreOps, address: Int, words: Int,
                            pin: Int) raises -> ArenaCachePut:
    """`id_of` for a native caller: the resident id (counted as a hit), else
    an upload."""
    var lease = ARENA_CACHES.acquire(arena_handle(word))
    var found = lease[].find(address, words, True)
    if found >= 0:
        return ArenaCachePut(found, False)
    return arena_cache_upload(word, store, ops, address, words, pin)


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


def _free_or_restore(lease: ArcPointer[DeviceCacheState], store: EntryRef, ops: ArenaStoreOps,
                     entry: ArenaStoreEntry, mut released: ArenaCacheReleased) raises:
    # One unlinked entry: free it; a failed free links it again for a retry
    # (its pin stays with the cache).
    try:
        arena_cache_free(store, ops, entry.store_id)
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
    return arena_cache_close(word, store, arena_store_ops_transport())


def arena_cache_close(word: Int, store: EntryRef, ops: ArenaStoreOps) raises -> ArenaCacheReleased:
    """Free every entry of the cache `word` (`close`, context exit)."""
    var lease = ARENA_CACHES.acquire(arena_handle(word))
    # Unlink everything first: no other thread finds an entry being freed.
    var drained = lease[].drain()
    var released = ArenaCacheReleased()
    for i in range(len(drained)):  # driver-loop: attempt every independent release (one store bookkeeping free each)
        _free_or_restore(lease, store, ops, drained[i], released)
    return released^


def arena_cache_release(word: Int, store: EntryRef, address: Int, words: Int) raises -> ArenaCacheReleased:
    """Free the entry of one buffer (`release`); nothing when it is absent."""
    var lease = ARENA_CACHES.acquire(arena_handle(word))
    var released = ArenaCacheReleased()
    var removed = lease[].remove(address, words)
    if removed.store_id >= 0:
        _free_or_restore(lease, store, arena_store_ops_transport(), removed, released)
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
    return arena_cache_finalize(word, store, arena_store_ops_transport())


def arena_cache_finalize(word: Int, store: EntryRef, ops: ArenaStoreOps) -> List[Int]:
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
                arena_cache_free(store, ops, drained[i].store_id)
            except error:
                pass
        ARENA_CACHES.release(handle)
    except error:
        pass
    return pins^


# ---------------------------------------------------------------- scope


struct _ScopesLive(Defaultable, Movable):
    """The number of `resident()` scope states alive in the process (lane
    vdf-tail-b): one Int64 word, changed only by atomic adds. Zero means no
    thread can have an active scope, so a door outside one skips its context
    and thread-local reads (`arena_scopes_live`)."""
    var count: List[Int64]

    def __init__(out self):
        self.count = List[Int64](length=1, fill=Int64(0))


comptime _ARENA_SCOPES_LIVE = SharedSlot[_ScopesLive, "MojolearnArenaScopesLiveV1", 1]


@always_inline
def _scopes_live_word() raises -> MutPointer[Int64, MutUntrackedOrigin]:
    var slot = _ARENA_SCOPES_LIVE.get_or_create_ptr()
    return MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(slot[].count.unsafe_ptr()))


def arena_scopes_live() raises -> Bool:
    """Whether any `resident()` scope state is alive in the process: one
    native slot read and one atomic load. False: no thread is inside a
    scope (a state is created before its scope is published to the thread
    and released only after it is gone); True: the caller reads its
    thread's scope as before. A state whose release failed stays counted
    (the caller then only takes the full read)."""
    return Atomic.load[ordering=Ordering.ACQUIRE](_scopes_live_word()) > 0


def arena_scope_new(min_words: Int) raises -> Int:
    """A fresh `resident()` scope state with its size floor; its handle word."""
    var word = arena_handle_word(ARENA_SCOPES.insert(ArenaScopeState(min_words)))
    _ = Atomic[Int64].fetch_add(_scopes_live_word(), Int64(1))
    return word


def arena_scope_floor(scope: Int) raises -> Int:
    var lease = ARENA_SCOPES.acquire(arena_handle(scope))
    return lease[].min_words


def arena_scope_cache(scope: Int, store: EntryRef, prefix: String) raises -> Int:
    return arena_scope_cache(scope, store, arena_store_ops_transport(), prefix)


def arena_scope_cache(scope: Int, store: EntryRef, ops: ArenaStoreOps, prefix: String) raises -> Int:
    """The scope's cache on `store` for the binding family `prefix`, created
    on first use; the family's store and operation numbers `ops` are
    recorded with it (module note). Called once per program run that keeps
    an input resident."""
    var lease = ARENA_SCOPES.acquire(arena_handle(scope))
    var route = arena_cache_route(store)
    var cache = lease[].cache_of(prefix, route)
    if cache == ARENA_HANDLE_NONE:
        _family_store_note(prefix, route, store, ops)  # the close frees into this family's own store
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
    return arena_scope_store_put(scope, store, arena_store_ops_transport(), prefix, address, words, pin)


def arena_scope_store_put(scope: Int, store: EntryRef, ops: ArenaStoreOps, prefix: String, address: Int,
                          words: Int, pin: Int) raises -> ArenaCachePut:
    """`arena_scope_store_put` on a family store with its own operation
    numbers `ops` (THE PER-FAMILY OP MAP)."""
    return arena_cache_find_or_put(arena_scope_cache(scope, store, ops, prefix), store, ops, address, words, pin)


def arena_store_take_rows(store: EntryRef, src_id: Int, row_words: Int, idx_address: Int, n_idx: Int) raises -> Int:
    """Rows `idx` (int64 at `idx_address`, `n_idx` of them) of the resident
    slot `src_id` (`row_words` words a row) gathered ON THE DEVICE into a
    fresh slot of the family store `store`; its id (the caller frees it)."""
    var source: List[Int] = [idx_address]
    var counts: List[Int] = [src_id, row_words, n_idx]
    return xp_transport_call(store, ARENA_STORE_TAKE_ROWS, source, counts)


def arena_store_read(store: EntryRef, store_id: Int, dst_address: Int, words: Int) raises:
    """The first `words` words of the slot `store_id` into the host memory at
    `dst_address` (after every launch queued on the store's context; waits).
    Lane vdf-scope-gp, an addition: the export of a fold slot a fitted state
    keeps (core/gp_member.mojo X_train_)."""
    var target: List[Int] = [dst_address]
    var counts: List[Int] = [store_id, words]
    _ = xp_transport_call(store, ARENA_STORE_READ, target, counts)


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
            var family = arena_family_store(taken[i].prefix, taken[i].route)
            var closed = arena_cache_close(taken[i].cache, family.store, family.ops)
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
                var family = arena_family_store(taken[i].prefix, taken[i].route)
                var dropped = arena_cache_finalize(taken[i].cache, family.store, family.ops)
                for j in range(len(dropped)):  # driver-loop: over the freed entries' pins
                    pins.append(dropped[j])
            except error:
                pass
        ARENA_SCOPES.release(handle)
        _ = Atomic[Int64].fetch_add(_scopes_live_word(), Int64(-1))
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
