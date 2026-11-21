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


struct DeviceRowsState(Movable):
    """A fold view: rows `index_address[0 .. rows)` (int64) of a host base of
    `base_rows` rows, `row_words` items of `base_itemsize` bytes per row."""
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
        self.pins = List[Int]()

    def ndim(self) -> Int:
        return 1 + len(self.trailing)

    def size(self) -> Int:
        """Items of the selected rows (`rows` x the trailing extents)."""
        return self.rows * self.row_words

    def row_bytes(self) -> Int:
        return self.row_words * self.base_itemsize


@fieldwise_init
struct ArenaStoreEntry(Copyable, Movable):
    """One resident device copy: the host buffer it was put from (`address`,
    `words` items), its id in the binding's device store, and the next entry
    index of the same address (-1 at the end of the chain)."""
    var address: Int
    var words: Int
    var store_id: Int
    var next: Int
    var pin: Int


struct DeviceCacheState(Movable):
    """Device copies of host buffers in one binding's store, keyed by buffer
    identity (address and item count). `route` is the index of the connected
    put/free entry pair the owner was constructed on."""
    var route: Int
    var prefix: String
    var uploads: Int
    var hits: Int
    var entries: List[ArenaStoreEntry]
    var first: Dict[Int, Int]
    var vacant: List[Int]
    var live: Int
    var lock: ArenaStateLock

    def __init__(out self, route: Int, prefix: String):
        self.route = route
        self.prefix = prefix
        self.uploads = 0
        self.hits = 0
        self.entries = List[ArenaStoreEntry]()
        self.first = Dict[Int, Int]()
        self.vacant = List[Int]()
        self.live = 0
        self.lock = ArenaStateLock()

    def _find(self, address: Int, words: Int) raises -> Int:
        if address not in self.first:
            return -1
        var index = self.first[address]
        while index >= 0:  # driver-loop: over the entries put from one address (one per distinct length)
            if self.entries[index].words == words:
                return index
            index = self.entries[index].next
        return -1

    def find(mut self, address: Int, words: Int, count_hit: Bool) raises -> Int:
        """The store id of the resident copy of (`address`, `words`), or -1.
        `count_hit` counts a found entry as a hit (`id_of`; `get` does not)."""
        self.lock.acquire()
        var store_id = -1
        try:
            var index = self._find(address, words)
            if index >= 0:
                store_id = self.entries[index].store_id
                if count_hit:
                    self.hits += 1
        finally:
            self.lock.release()
        return store_id

    def publish(mut self, address: Int, words: Int, store_id: Int, pin: Int) raises -> Int:
        """Record an upload the caller made outside the lock. Returns
        `store_id`, or the id another thread published first for the same
        buffer: the caller then frees its own `store_id` and drops `pin`."""
        self.lock.acquire()
        var kept = store_id
        try:
            var index = self._find(address, words)
            if index >= 0:
                kept = self.entries[index].store_id
            else:
                var head = -1
                if address in self.first:
                    head = self.first[address]
                var entry = ArenaStoreEntry(address, words, store_id, head, pin)
                var slot = len(self.entries)
                if len(self.vacant):
                    slot = self.vacant.pop()
                    self.entries[slot] = entry.copy()
                else:
                    self.entries.append(entry.copy())
                self.first[address] = slot
                self.uploads += 1
                self.live += 1
        finally:
            self.lock.release()
        return kept

    def remove(mut self, address: Int, words: Int) raises -> ArenaStoreEntry:
        """Unlink the entry of (`address`, `words`) and return it for the
        caller to free on the device and unpin; `store_id` -1 when absent."""
        self.lock.acquire()
        var removed = ArenaStoreEntry(address, words, -1, -1, 0)
        try:
            var index = self._find(address, words)
            if index >= 0:
                removed = self.entries[index].copy()
                var head = self.first[address]
                if head == index:
                    if removed.next >= 0:
                        self.first[address] = removed.next
                    else:
                        _ = self.first.pop(address)
                else:
                    var previous = head
                    while self.entries[previous].next != index:  # driver-loop: over the entries put from one address
                        previous = self.entries[previous].next
                    self.entries[previous].next = removed.next
                self.vacant.append(index)
                self.live -= 1
        finally:
            self.lock.release()
        return removed^

    def drain(mut self) raises -> List[ArenaStoreEntry]:
        """Unlink every entry (`close`) and return them for the caller to
        free and unpin outside the lock."""
        self.lock.acquire()
        var drained = List[ArenaStoreEntry]()
        try:
            for item in self.first.items():  # driver-loop: over the cache's distinct buffer addresses
                var index = item.value
                while index >= 0:  # driver-loop: over the entries put from one address
                    drained.append(self.entries[index].copy())
                    index = self.entries[index].next
            self.first = Dict[Int, Int]()
            self.entries = List[ArenaStoreEntry]()
            self.vacant = List[Int]()
            self.live = 0
        finally:
            self.lock.release()
        return drained^


struct ArenaScopeState(Movable):
    """One `resident()` scope of a thread: the size floor and the per-prefix
    caches the runner created in it (handle words of `ARENA_CACHES`).
    `depth` counts the nested `with` blocks sharing it."""
    var min_words: Int
    var depth: Int
    var prefixes: List[String]
    var caches: List[Int]

    def __init__(out self, min_words: Int):
        self.min_words = min_words
        self.depth = 1
        self.prefixes = List[String]()
        self.caches = List[Int]()

    def cache_of(self, prefix: String) -> Int:
        """The scope's cache handle word for `prefix`, or `ARENA_HANDLE_NONE`."""
        for i in range(len(self.prefixes)):  # driver-loop: over the scope's binding prefixes (a handful)
            if self.prefixes[i] == prefix:
                return self.caches[i]
        return ARENA_HANDLE_NONE

    def add_cache(mut self, prefix: String, cache: Int):
        self.prefixes.append(prefix)
        self.caches.append(cache)


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


struct ArenaResidentState(Movable):
    """Model selection's resident store owner: inputs keyed by the caller
    object's identity, each put once and kept for the fold set."""
    var entries: List[ArenaResidentEntry]
    var by_owner: Dict[Int, Int]
    var lock: ArenaStateLock

    def __init__(out self):
        self.entries = List[ArenaResidentEntry]()
        self.by_owner = Dict[Int, Int]()
        self.lock = ArenaStateLock()

    def find(mut self, owner: Int) raises -> ArenaResidentEntry:
        """The entry of `owner`; `store_id` -2 when it was never put."""
        self.lock.acquire()
        var found = ArenaResidentEntry(owner, 0, 0, 0, False, -2, 0, 0)
        try:
            if owner in self.by_owner:
                found = self.entries[self.by_owner[owner]].copy()
        finally:
            self.lock.release()
        return found^

    def publish(mut self, entry: ArenaResidentEntry) raises -> ArenaResidentEntry:
        """Record a put the caller made outside the lock; returns the entry
        kept (another thread's when it published the same owner first: the
        caller then frees its own store id and drops its pins)."""
        self.lock.acquire()
        var kept = entry.copy()
        try:
            if entry.owner in self.by_owner:
                kept = self.entries[self.by_owner[entry.owner]].copy()
            else:
                self.by_owner[entry.owner] = len(self.entries)
                self.entries.append(entry.copy())
        finally:
            self.lock.release()
        return kept^

    def drain(mut self) raises -> List[ArenaResidentEntry]:
        """Unlink every entry and return them for the caller to free and
        unpin outside the lock."""
        self.lock.acquire()
        var drained = List[ArenaResidentEntry]()
        try:
            for i in range(len(self.entries)):  # driver-loop: over the owner's resident inputs
                drained.append(self.entries[i].copy())
            self.entries = List[ArenaResidentEntry]()
            self.by_owner = Dict[Int, Int]()
        finally:
            self.lock.release()
        return drained^


comptime ARENA_ROWS = StateRegistry[DeviceRowsState, "MojolearnArenaRowsStatesV1"]
comptime ARENA_CACHES = StateRegistry[DeviceCacheState, "MojolearnArenaCacheStatesV1"]
comptime ARENA_SCOPES = StateRegistry[ArenaScopeState, "MojolearnArenaScopeStatesV1"]
comptime ARENA_RESIDENTS = StateRegistry[ArenaResidentState, "MojolearnArenaResidentStatesV1"]
