# SPDX-License-Identifier: Apache-2.0
"""Native state of the arena owners: `DeviceRows`, `DeviceCache`, the
`resident()` scope and model selection's resident store owner.

Each state type has one process-wide registry (`core/state_registry.mojo`: a
`SharedSlot` with a layout word, a generation per index), so every binding of
the process reads the same state by handle. The Python class keeps ONE
integer, the handle word of `arena_handle_word`; a door resolves it once per
public operation with `arena_handle` and `<REGISTRY>.acquire`, then RUN reads
and writes typed fields. No Python import (tools/python_free_dirs.sh).

Bump the registry name's `V<n>` on any field change of its state struct
(core/shared_slot.mojo, THE RULE).

Lifetime. A state holds host addresses of buffers it does not own. The
`pins` words are opaque tokens of the boundary (retained CPython object
addresses taken in UNPACK and dropped in PACK when the state is released);
RUN never interprets them.

Concurrency. The registry locks only its own metadata. A cache or resident
state is mutated under its own lock word (`ArenaStateLock`); a device call is
never made while it is held. Lookups are per public operation, never per row,
batch or token, so there is nothing to split across cores here.
"""
from std.collections import Dict
from core.shared_slot import shared_slot_lock, shared_slot_unlock
from core.state_registry import StateRegistry, StateHandle

comptime ARENA_HANDLE_INDEX_BITS = 32
comptime ARENA_HANDLE_INDEX_MASK = 0xFFFFFFFF
#: The handle word of an owner that holds no native state.
comptime ARENA_HANDLE_NONE = -1


def arena_handle_word(handle: StateHandle) raises -> Int:
    """The one integer a Python owner keeps: the registry index in the low
    32 bits and the generation above it."""
    if handle.index < 0 or handle.index > ARENA_HANDLE_INDEX_MASK:
        raise Error("RuntimeError: native arena state index exceeds the handle word")
    if handle.generation >= (UInt64(1) << 31):
        raise Error("RuntimeError: native arena state generation exceeds the handle word")
    return (Int(handle.generation) << ARENA_HANDLE_INDEX_BITS) | handle.index


def arena_handle(word: Int) raises -> StateHandle:
    """The registry handle of a handle word; refuses `ARENA_HANDLE_NONE` and
    any negative word."""
    if word < 0:
        raise Error("RuntimeError: native arena state was released or never initialized")
    return StateHandle(word & ARENA_HANDLE_INDEX_MASK, UInt64(word >> ARENA_HANDLE_INDEX_BITS))


struct ArenaStateLock(Movable):
    """One lock word (0 free, 1 held) owned by a state."""
    var word: List[Int64]

    def __init__(out self):
        self.word = List[Int64](length=1, fill=0)

    def acquire(mut self):
        shared_slot_lock(rebind[MutPointer[Int64, MutUntrackedOrigin]](self.word.unsafe_ptr()))

    def release(mut self):
        shared_slot_unlock(rebind[MutPointer[Int64, MutUntrackedOrigin]](self.word.unsafe_ptr()))


#: Positions of a `DeviceRowsState`'s pin tokens: the base, the index Array,
#: the resident store owner that gathers the fold (0 without one) and the
#: materialized host rows (0 until they were published).
comptime ARENA_ROWS_PIN_BASE = 0
comptime ARENA_ROWS_PIN_INDICES = 1
comptime ARENA_ROWS_PIN_RESIDENT = 2
comptime ARENA_ROWS_PIN_HOST = 3
comptime ARENA_ROWS_PINS = 4


struct DeviceRowsState(Movable):
    """A fold view: rows `index_address[0 .. rows)` (int64) of a host base of
    `base_rows` rows, `row_words` items of `base_itemsize` bytes per row.

    The view is immutable after construction except for the one publication
    of its host rows (`publish_host`), which the boundary serializes (it is
    made with the interpreter lock held), so the state carries no lock word.
    `base_address` / `index_address` are 0 while the boundary's doors admit
    the pinned objects themselves (their admission reads the address)."""
    var base_address: Int
    var base_rows: Int
    var row_words: Int
    var base_itemsize: Int
    #: The base's dtype string ("<f4", ...), as its Array or ndarray names it.
    var base_dtype: String
    #: The base's shape after its first dimension.
    var trailing: List[Int]
    var index_address: Int
    var rows: Int
    #: Handle word of the resident owner (`ARENA_RESIDENTS`) that gathers this
    #: fold, or `ARENA_HANDLE_NONE` for a fold without one.
    var resident: Int
    #: Host address of the materialized rows, 0 until `materialize` ran.
    var host_address: Int
    var pins: List[Int]

    def __init__(out self, base_address: Int, base_rows: Int, row_words: Int, base_itemsize: Int,
                 base_dtype: String, var trailing: List[Int], index_address: Int, rows: Int,
                 resident: Int):
        self.base_address = base_address
        self.base_rows = base_rows
        self.row_words = row_words
        self.base_itemsize = base_itemsize
        self.base_dtype = base_dtype
        self.trailing = trailing^
        self.index_address = index_address
        self.rows = rows
        self.resident = resident
        self.host_address = 0
        self.pins = List[Int](length=ARENA_ROWS_PINS, fill=0)

    def ndim(self) -> Int:
        return 1 + len(self.trailing)

    def size(self) -> Int:
        """Items of the selected rows (`rows` x the trailing extents)."""
        return self.rows * self.row_words

    def row_bytes(self) -> Int:
        return self.row_words * self.base_itemsize

    def host_pin(self) -> Int:
        """The pin token of the materialized host rows, 0 before they exist."""
        return self.pins[ARENA_ROWS_PIN_HOST]

    def publish_host(mut self, address: Int, pin: Int) -> Int:
        """Record the fold's host rows (`materialize` runs at most once).
        Returns the pin kept: `pin`, or the one published first, in which
        case the caller drops its own."""
        if self.pins[ARENA_ROWS_PIN_HOST] != 0:
            return self.pins[ARENA_ROWS_PIN_HOST]
        self.pins[ARENA_ROWS_PIN_HOST] = pin
        self.host_address = address
        return pin


#: `store_id` of an entry `remove` or `drain` unlinked while a member
#: operation still held it (`find_hold`, `publish(hold=True)`): its device
#: free belongs to the last `unhold` (bindings/arena_cache_run.mojo
#: `arena_cache_unhold`); the caller of `remove` / `drain` releases the pin
#: only (the device copy no longer reads the host buffer).
comptime ARENA_STORE_ID_DEFERRED = -3


@fieldwise_init
struct ArenaStoreEntry(Copyable, Movable):
    """One resident device copy: the host buffer it was put from (`address`,
    `words` items), its id in the binding's device store, the next entry
    index of the same address (-1 at the end of the chain), its pin and the
    member operations holding its id now (`holds`)."""
    var address: Int
    var words: Int
    var store_id: Int
    var next: Int
    var pin: Int
    var holds: Int


@fieldwise_init
struct ArenaDeferredFree(Copyable, Movable):
    """A store id unlinked while held: freed by its last `unhold`."""
    var store_id: Int
    var holds: Int


struct _CacheTable(Movable):
    """A `DeviceCacheState`'s mutable state, in heap storage of its own (THE
    TABLE IS BEHIND A POINTER, `DeviceCacheState`). Every method is called
    with the state's lock held."""
    var uploads: Int
    var hits: Int
    var entries: List[ArenaStoreEntry]
    var first: Dict[Int, Int]
    var vacant: List[Int]
    var live: Int
    var deferred: List[ArenaDeferredFree]

    def __init__(out self):
        self.uploads = 0
        self.hits = 0
        self.entries = List[ArenaStoreEntry]()
        self.first = Dict[Int, Int]()
        self.vacant = List[Int]()
        self.live = 0
        self.deferred = List[ArenaDeferredFree]()

    def find_index(self, address: Int, words: Int) raises -> Int:
        if address not in self.first:
            return -1
        var index = self.first[address]
        while index >= 0:  # driver-loop: over the entries put from one address (one per distinct length)
            if self.entries[index].words == words:
                return index
            index = self.entries[index].next
        return -1

    def link(mut self, address: Int, words: Int, store_id: Int, pin: Int, holds: Int) raises:
        """A new entry at the head of its address chain; counts it live."""
        var head = -1
        if address in self.first:
            head = self.first[address]
        var entry = ArenaStoreEntry(address, words, store_id, head, pin, holds)
        var slot = len(self.entries)
        if len(self.vacant):
            slot = self.vacant.pop()
            self.entries[slot] = entry^
        else:
            self.entries.append(entry^)
        self.first[address] = slot
        self.live += 1

    def unlink(mut self, index: Int) raises -> ArenaStoreEntry:
        """Entry `index` out of its chain (its index vacated); returned as
        `retire` leaves it."""
        var removed = self.entries[index].copy()
        var head = self.first[removed.address]
        if head == index:
            if removed.next >= 0:
                self.first[removed.address] = removed.next
            else:
                _ = self.first.pop(removed.address)
        else:
            var previous = head
            while self.entries[previous].next != index:  # driver-loop: over the entries put from one address
                previous = self.entries[previous].next
            self.entries[previous].next = removed.next
        self.vacant.append(index)
        self.live -= 1
        return self.retire(removed^)

    def retire(mut self, var entry: ArenaStoreEntry) -> ArenaStoreEntry:
        """An unlinked entry still held moves its store id to `deferred`
        (the last `unhold` frees it); the caller then sees
        ARENA_STORE_ID_DEFERRED and frees nothing."""
        if entry.holds > 0:
            self.deferred.append(ArenaDeferredFree(entry.store_id, entry.holds))
            entry.store_id = ARENA_STORE_ID_DEFERRED
            entry.holds = 0
        return entry^


struct DeviceCacheState(Movable):
    """Device copies of host buffers in one binding's store, keyed by buffer
    identity (address and item count). `route` names the store entry the
    owner was constructed on (its mode and vendor words,
    `bindings/arena_cache_run.mojo` `arena_cache_route`).

    THE TABLE IS BEHIND A POINTER (lane gd-device-store, 2026-11-25; the
    pattern of core/msel_host.mojo and core/device_store.mojo). The mutable
    state lives in `_CacheTable`, the one element of the heap List `cell`,
    reached through `_table()` after the lock: as fields read through `self`
    the compiler may load a List's pointer and length before the lock and
    keep that copy. `route` and `prefix` never change after construction.

    HOLDS (same lane). A member operation that reads a found or published id
    after the lock (a fold's device gather out of the cached whole X) holds
    it (`find_hold`, `publish(hold=True)`) until `unhold`: a concurrent
    `remove` / `drain` (release, drop, close) then unlinks the entry at once
    but leaves its device free to the last `unhold`, so no held id is freed
    under its reader."""
    var route: Int
    var prefix: String
    var lock: ArenaStateLock
    #: exactly one `_CacheTable`, never grown, so its address is stable
    var cell: List[_CacheTable]

    def __init__(out self, route: Int, prefix: String):
        self.route = route
        self.prefix = prefix
        self.lock = ArenaStateLock()
        self.cell = List[_CacheTable]()
        self.cell.append(_CacheTable())

    def _table(self) -> MutPointer[_CacheTable, MutUntrackedOrigin]:
        return MutPointer[_CacheTable, MutUntrackedOrigin](unsafe_from_address=Int(self.cell.unsafe_ptr()))

    def live_count(mut self) -> Int:
        var table = self._table()
        self.lock.acquire()
        var n = table[].live
        self.lock.release()
        return n

    def hit_count(mut self) -> Int:
        var table = self._table()
        self.lock.acquire()
        var n = table[].hits
        self.lock.release()
        return n

    def upload_count(mut self) -> Int:
        var table = self._table()
        self.lock.acquire()
        var n = table[].uploads
        self.lock.release()
        return n

    def find(mut self, address: Int, words: Int, count_hit: Bool) raises -> Int:
        """The store id of the resident copy of (`address`, `words`), or -1.
        `count_hit` counts a found entry as a hit (`id_of`; `get` does not)."""
        var table = self._table()
        self.lock.acquire()
        var store_id = -1
        try:
            var index = table[].find_index(address, words)
            if index >= 0:
                store_id = table[].entries[index].store_id
                if count_hit:
                    table[].hits += 1
        finally:
            self.lock.release()
        return store_id

    def find_hold(mut self, address: Int, words: Int) raises -> Int:
        """`find` counting a hit, and the found id held until `unhold`."""
        var table = self._table()
        self.lock.acquire()
        var store_id = -1
        try:
            var index = table[].find_index(address, words)
            if index >= 0:
                store_id = table[].entries[index].store_id
                table[].entries[index].holds += 1
                table[].hits += 1
        finally:
            self.lock.release()
        return store_id

    def publish(mut self, address: Int, words: Int, store_id: Int, pin: Int, hold: Bool = False) raises -> Int:
        """Record an upload the caller made outside the lock. Returns
        `store_id`, or the id another thread published first for the same
        buffer: the caller then frees its own `store_id` and drops `pin`.
        `hold`: the returned id is held until `unhold`."""
        var table = self._table()
        self.lock.acquire()
        var kept = store_id
        try:
            var index = table[].find_index(address, words)
            if index >= 0:
                kept = table[].entries[index].store_id
                if hold:
                    table[].entries[index].holds += 1
            else:
                table[].link(address, words, store_id, pin, 1 if hold else 0)
                table[].uploads += 1
        finally:
            self.lock.release()
        return kept

    def unhold(mut self, address: Int, words: Int, store_id: Int) raises -> Int:
        """Drop one hold of `store_id` (put from `address`, `words`). Returns
        the store id the caller frees now (the last hold of an entry
        unlinked while held), else -1."""
        var table = self._table()
        self.lock.acquire()
        var freed = -1
        try:
            var index = table[].find_index(address, words)
            if index >= 0 and table[].entries[index].store_id == store_id and table[].entries[index].holds > 0:
                table[].entries[index].holds -= 1
            else:
                var at = -1
                for i in range(len(table[].deferred)):  # driver-loop: over ids unlinked while held (a few)
                    if table[].deferred[i].store_id == store_id:
                        at = i
                        break
                if at < 0:
                    raise Error("RuntimeError: DeviceCache: store id " + String(store_id) + " is not held")
                table[].deferred[at].holds -= 1
                if table[].deferred[at].holds == 0:
                    _ = table[].deferred.pop(at)
                    freed = store_id
        finally:
            self.lock.release()
        return freed

    def restore(mut self, entry: ArenaStoreEntry) raises -> Bool:
        """Link again an entry `remove` or `drain` returned whose device free
        failed, for a retry; no counter moves. False when another entry of
        the same buffer was published meanwhile (the caller keeps `entry`)."""
        var table = self._table()
        self.lock.acquire()
        var linked = False
        try:
            if table[].find_index(entry.address, entry.words) < 0:
                table[].link(entry.address, entry.words, entry.store_id, entry.pin, 0)
                linked = True
        finally:
            self.lock.release()
        return linked

    def remove(mut self, address: Int, words: Int) raises -> ArenaStoreEntry:
        """Unlink the entry of (`address`, `words`) and return it for the
        caller to free on the device and unpin; `store_id` -1 when absent,
        ARENA_STORE_ID_DEFERRED when held (unpin only)."""
        var table = self._table()
        self.lock.acquire()
        var removed = ArenaStoreEntry(address, words, -1, -1, 0, 0)
        try:
            var index = table[].find_index(address, words)
            if index >= 0:
                removed = table[].unlink(index)
        finally:
            self.lock.release()
        return removed^

    def drain(mut self) raises -> List[ArenaStoreEntry]:
        """Unlink every entry (`close`) and return them for the caller to
        free and unpin outside the lock (a held one as
        ARENA_STORE_ID_DEFERRED: unpin only)."""
        var table = self._table()
        self.lock.acquire()
        var drained = List[ArenaStoreEntry]()
        try:
            for item in table[].first.items():  # driver-loop: over the cache's distinct buffer addresses
                var index = item.value
                while index >= 0:  # driver-loop: over the entries put from one address
                    drained.append(table[].retire(table[].entries[index].copy()))
                    index = table[].entries[index].next
            table[].first = Dict[Int, Int]()
            table[].entries = List[ArenaStoreEntry]()
            table[].vacant = List[Int]()
            table[].live = 0
        finally:
            self.lock.release()
        return drained^


@fieldwise_init
struct ArenaScopeCache(Copyable, Movable):
    """One cache of a scope: the binding family (`prefix`), the store route
    it was built on (`bindings/arena_cache_run.mojo` `arena_cache_route`) and
    its handle word in `ARENA_CACHES`."""
    var prefix: String
    var route: Int
    var cache: Int


struct ArenaScopeState(Movable):
    """One `resident()` scope of a thread: the size floor and the caches the
    runner created in it, one per binding family and store route. A scope is
    reached only through its thread's thread-local, so it carries no lock
    word."""
    var min_words: Int
    var caches: List[ArenaScopeCache]

    def __init__(out self, min_words: Int):
        self.min_words = min_words
        self.caches = List[ArenaScopeCache]()

    def cache_of(self, prefix: String, route: Int) -> Int:
        """The scope's cache handle word for (`prefix`, `route`), or
        `ARENA_HANDLE_NONE`."""
        for i in range(len(self.caches)):  # driver-loop: over the scope's caches (one per binding family and mode: a handful)
            if self.caches[i].route == route and self.caches[i].prefix == prefix:
                return self.caches[i].cache
        return ARENA_HANDLE_NONE

    def add_cache(mut self, prefix: String, route: Int, cache: Int):
        self.caches.append(ArenaScopeCache(prefix, route, cache))

    def take_caches(mut self) -> List[ArenaScopeCache]:
        """Unlink every cache (`close`) and return them for the caller to
        close and release."""
        var taken = List[ArenaScopeCache]()
        for i in range(len(self.caches)):  # driver-loop: over the scope's caches
            taken.append(self.caches[i].copy())
        self.caches = List[ArenaScopeCache]()
        return taken^


@fieldwise_init
struct ArenaResidentEntry(Copyable, Movable):
    """One input model selection put into its store: the caller's object
    identity (`owner`), the prepared host buffer (`address`, `nbytes`, `rows`
    rows; int64 when `indices`), its store id (-1 for an empty buffer) and
    the boundary's pin tokens."""
    var owner: Int
    var address: Int
    var nbytes: Int
    var rows: Int
    var indices: Bool
    var store_id: Int
    var owner_pin: Int
    var prepared_pin: Int


struct _ResidentTable(Movable):
    """An `ArenaResidentState`'s entries, in heap storage of its own (THE
    TABLE IS BEHIND A POINTER, `DeviceCacheState`)."""
    var entries: List[ArenaResidentEntry]
    var by_owner: Dict[Int, Int]

    def __init__(out self):
        self.entries = List[ArenaResidentEntry]()
        self.by_owner = Dict[Int, Int]()


struct ArenaResidentState(Movable):
    """Model selection's resident store owner: inputs keyed by the caller
    object's identity, each put once and kept for the fold set. The entries
    are behind a pointer read only after the lock (lane gd-device-store,
    2026-11-25: THE TABLE IS BEHIND A POINTER, `DeviceCacheState`)."""
    var lock: ArenaStateLock
    #: exactly one `_ResidentTable`, never grown, so its address is stable
    var cell: List[_ResidentTable]

    def __init__(out self):
        self.lock = ArenaStateLock()
        self.cell = List[_ResidentTable]()
        self.cell.append(_ResidentTable())

    def _table(self) -> MutPointer[_ResidentTable, MutUntrackedOrigin]:
        return MutPointer[_ResidentTable, MutUntrackedOrigin](unsafe_from_address=Int(self.cell.unsafe_ptr()))

    def find(mut self, owner: Int) raises -> ArenaResidentEntry:
        """The entry of `owner`; `store_id` -2 when it was never put."""
        var table = self._table()
        self.lock.acquire()
        var found = ArenaResidentEntry(owner, 0, 0, 0, False, -2, 0, 0)
        try:
            if owner in table[].by_owner:
                found = table[].entries[table[].by_owner[owner]].copy()
        finally:
            self.lock.release()
        return found^

    def publish(mut self, entry: ArenaResidentEntry) raises -> ArenaResidentEntry:
        """Record a put the caller made outside the lock; returns the entry
        kept (another thread's when it published the same owner first: the
        caller then frees its own store id and drops its pins)."""
        var table = self._table()
        self.lock.acquire()
        var kept = entry.copy()
        try:
            if entry.owner in table[].by_owner:
                kept = table[].entries[table[].by_owner[entry.owner]].copy()
            else:
                table[].by_owner[entry.owner] = len(table[].entries)
                table[].entries.append(entry.copy())
        finally:
            self.lock.release()
        return kept^

    def remove(mut self, owner: Int) raises -> ArenaResidentEntry:
        """Unlink the entry of `owner` and return it for the caller to free
        and unpin outside the lock; `store_id` -2 when it was never put."""
        var table = self._table()
        self.lock.acquire()
        var removed = ArenaResidentEntry(owner, 0, 0, 0, False, -2, 0, 0)
        try:
            if owner in table[].by_owner:
                var index = table[].by_owner[owner]
                removed = table[].entries[index].copy()
                var last = len(table[].entries) - 1
                if index != last:
                    # The last entry takes the vacated index.
                    var moved = table[].entries[last].copy()
                    table[].by_owner[moved.owner] = index
                    table[].entries[index] = moved.copy()
                _ = table[].entries.pop()
                _ = table[].by_owner.pop(owner)
        finally:
            self.lock.release()
        return removed^

    def drain(mut self) raises -> List[ArenaResidentEntry]:
        """Unlink every entry and return them for the caller to free and
        unpin outside the lock."""
        var table = self._table()
        self.lock.acquire()
        var drained = List[ArenaResidentEntry]()
        try:
            for i in range(len(table[].entries)):  # driver-loop: over the owner's resident inputs
                drained.append(table[].entries[i].copy())
            table[].entries = List[ArenaResidentEntry]()
            table[].by_owner = Dict[Int, Int]()
        finally:
            self.lock.release()
        return drained^


trait ArenaStoreFree:
    """The fold store's free as the resident close and release see it
    (lane fx-leftovers item 9, 2026-11-21): `free` releases one store id
    (it may release the GIL; an id < 0, an empty buffer, frees nothing),
    `failed` keeps a failure (the first one is the one raised). The door's
    form calls the model-selection store; the native check's form fails
    chosen ids and records the order (checks/native_tests/arena_state_suite.mojo)."""

    def free(mut self, store_id: Int) raises:
        ...

    def failed(mut self, error: Error) raises:
        ...


def arena_resident_close_entries[F: ArenaStoreFree](
    mut state: ArenaResidentState, mut store: F, mut drained: List[ArenaResidentEntry],
) raises -> List[Bool]:
    """The resident close's failure order, one definition: every entry is
    unlinked first (`drain`: no other thread finds an entry being freed),
    every slot's free is attempted in entry order, each failure goes to
    `store.failed` (the first is retained), and a failed entry is published
    again for a retry. Returns which drained entries are kept; the caller
    releases the pins of the others after this returns, after every free
    and publication, so an owner's finalizer never sees a half-closed
    resident."""
    drained = state.drain()
    var kept = List[Bool](length=len(drained), fill=False)
    for i in range(len(drained)):  # driver-loop: attempt every independent release (one small device free each)
        try:
            store.free(drained[i].store_id)
        except error:
            store.failed(error)
            # A failed entry stays owned for a retry, unless another thread
            # published the same owner meanwhile: its entry is kept, so this
            # one's pins are released by the caller (lane fx-audit 28).
            try:
                var held = state.publish(drained[i])
                kept[i] = held.store_id == drained[i].store_id
            except publish_error:
                store.failed(publish_error)
    return kept^


def arena_resident_release_entry[F: ArenaStoreFree](
    mut state: ArenaResidentState, mut store: F, owner: Int, mut removed: ArenaResidentEntry,
) raises -> Bool:
    """The resident release of one owner, the same order: unlink, free,
    on a failure keep it and publish the entry again for a retry. Returns
    True when the entry is kept (its pins stay); `removed.store_id` is -2
    when the owner was never put (nothing to do)."""
    removed = state.remove(owner)
    if removed.store_id == -2:
        return False
    try:
        store.free(removed.store_id)
    except error:
        store.failed(error)
        try:
            # Kept only when this entry is the one published (another thread
            # may have published the same owner meanwhile: lane fx-audit 28).
            var held = state.publish(removed)
            return held.store_id == removed.store_id
        except publish_error:
            store.failed(publish_error)
    return False


comptime ARENA_ROWS = StateRegistry[DeviceRowsState, "MojolearnArenaRowsStatesV1"]
comptime ARENA_CACHES = StateRegistry[DeviceCacheState, "MojolearnArenaCacheStatesV2"]
comptime ARENA_SCOPES = StateRegistry[ArenaScopeState, "MojolearnArenaScopeStatesV2"]
comptime ARENA_RESIDENTS = StateRegistry[ArenaResidentState, "MojolearnArenaResidentStatesV2"]
