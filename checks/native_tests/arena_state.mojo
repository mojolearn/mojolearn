# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Native arena owner state (core/arena_state.mojo): handle words, the
generation check of a released handle, cache identity chains, the resident
owner's entries and the scope's per-prefix caches.

Independent fixtures run across host cores; each inserts and releases its own
registry states. No device store is reached: store ids are plain integers.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from core.state_registry import StateHandle
from core.arena_state import (
    ARENA_ROWS, ARENA_CACHES, ARENA_SCOPES, ARENA_RESIDENTS, ARENA_HANDLE_NONE,
    DeviceRowsState, DeviceCacheState, ArenaScopeState, ArenaResidentState, ArenaResidentEntry, ARENA_STORE_ID_DEFERRED,
    ArenaStoreFree, arena_handle_word, arena_handle, arena_resident_close_entries, arena_resident_release_entry,
)

comptime ARENA_STATE_FIXTURES = 9


def handle_words() raises:
    var word = arena_handle_word(StateHandle(7, UInt64(3)))
    var back = arena_handle(word)
    require(back.index == 7 and back.generation == UInt64(3), "handle word round trip")
    var refused = False
    try:
        _ = arena_handle(ARENA_HANDLE_NONE)
    except error:
        refused = True
    require(refused, "the empty handle word is refused")
    refused = False
    try:
        _ = arena_handle_word(StateHandle(1 << 32, UInt64(1)))
    except error:
        refused = True
    require(refused, "an index beyond the handle word is refused")


def rows_state() raises:
    var trailing: List[Int] = [3, 2]
    var word = arena_handle_word(ARENA_ROWS.insert(DeviceRowsState(4096, 10, 6, 4, "<f4", trailing^, 8192, 4, ARENA_HANDLE_NONE)))
    var lease = ARENA_ROWS.acquire(arena_handle(word))
    require(lease[].ndim() == 3 and lease[].size() == 24 and lease[].row_bytes() == 24, "rows extents")
    require(lease[].host_address == 0 and lease[].resident == ARENA_HANDLE_NONE, "rows defaults")
    require(lease[].host_pin() == 0, "no host rows before the publication")
    require(lease[].publish_host(12288, 77) == 77 and lease[].host_address == 12288, "the first host publication is kept")
    require(lease[].publish_host(16384, 78) == 77 and lease[].host_address == 12288, "a second host publication keeps the first")
    ARENA_ROWS.release(arena_handle(word))
    var refused = False
    try:
        var stale = ARENA_ROWS.acquire(arena_handle(word))
        _ = stale
    except error:
        refused = True
    require(refused, "a released rows handle is refused")
    # The lease taken before the release still reads its state.
    require(lease[].rows == 4, "a lease outlives the release")


def cache_identity() raises:
    var word = arena_handle_word(ARENA_CACHES.insert(DeviceCacheState(0, "x_prep")))
    var lease = ARENA_CACHES.acquire(arena_handle(word))
    require(lease[].find(100, 8, True) == -1 and lease[].hit_count() == 0, "a miss counts no hit")
    require(lease[].publish(100, 8, 11, 0) == 11, "first publication keeps its id")
    require(lease[].publish(100, 8, 12, 0) == 11, "a second publication of one buffer keeps the first id")
    require(lease[].publish(100, 4, 13, 0) == 13, "another length at the same address is its own entry")
    require(lease[].publish(200, 8, 14, 0) == 14, "another address")
    require(lease[].upload_count() == 3 and lease[].live_count() == 3, "upload and live counts")
    require(lease[].find(100, 8, False) == 11 and lease[].hit_count() == 0, "get counts no hit")
    require(lease[].find(100, 4, True) == 13 and lease[].hit_count() == 1, "id_of counts a hit")
    var removed = lease[].remove(100, 8)
    require(removed.store_id == 11 and lease[].live_count() == 2, "remove returns the entry")
    require(lease[].find(100, 8, False) == -1 and lease[].find(100, 4, False) == 13, "the chain survives a removal")
    require(lease[].remove(100, 8).store_id == -1, "removing an absent buffer")
    require(lease[].publish(100, 8, 15, 0) == 15 and lease[].find(100, 8, False) == 15, "a vacated slot is reused")
    var drained = lease[].drain()
    require(len(drained) == 3 and lease[].live_count() == 0 and lease[].find(200, 8, False) == -1, "drain empties the cache")
    var uploads = lease[].upload_count()
    require(lease[].restore(drained[0]) and lease[].live_count() == 1 and lease[].upload_count() == uploads, "restore links an entry again and counts no upload")
    require(lease[].find(drained[0].address, drained[0].words, False) == drained[0].store_id, "a restored entry is found")
    require(not lease[].restore(drained[0]) and lease[].live_count() == 1, "restore keeps an entry published meanwhile")
    ARENA_CACHES.release(arena_handle(word))


def cache_chain_order() raises:
    var word = arena_handle_word(ARENA_CACHES.insert(DeviceCacheState(1, "x_metrics")))
    var lease = ARENA_CACHES.acquire(arena_handle(word))
    for words in range(1, 6):
        _ = lease[].publish(64, words, 100 + words, 0)
    # Remove from the middle, the tail and the head of the chain.
    require(lease[].remove(64, 3).store_id == 103, "middle")
    require(lease[].remove(64, 1).store_id == 101, "tail")
    require(lease[].remove(64, 5).store_id == 105, "head")
    require(lease[].find(64, 2, False) == 102 and lease[].find(64, 4, False) == 104 and lease[].live_count() == 2, "survivors")
    require(lease[].remove(64, 2).store_id == 102 and lease[].remove(64, 4).store_id == 104, "last two")
    require(lease[].live_count() == 0 and len(lease[].drain()) == 0, "empty after the last removal")
    ARENA_CACHES.release(arena_handle(word))


def resident_entries() raises:
    var word = arena_handle_word(ARENA_RESIDENTS.insert(ArenaResidentState()))
    var lease = ARENA_RESIDENTS.acquire(arena_handle(word))
    require(lease[].find(900).store_id == -2, "an owner never put")
    var kept = lease[].publish(ArenaResidentEntry(900, 4096, 80, 10, False, 5, 0, 0))
    require(kept.store_id == 5, "first put")
    kept = lease[].publish(ArenaResidentEntry(900, 4096, 80, 10, False, 6, 0, 0))
    require(kept.store_id == 5, "a second put of one owner keeps the first")
    _ = lease[].publish(ArenaResidentEntry(901, 8192, 0, 0, True, -1, 0, 0))
    require(lease[].find(901).store_id == -1 and lease[].find(901).indices, "an empty index buffer has no store id")
    _ = lease[].publish(ArenaResidentEntry(902, 12288, 8, 1, False, 7, 0, 0))
    require(lease[].remove(900).store_id == 5 and lease[].find(900).store_id == -2, "remove unlinks one owner")
    require(lease[].find(902).store_id == 7 and lease[].find(901).store_id == -1, "the others keep their entries")
    require(lease[].remove(900).store_id == -2, "a removed owner is absent")
    _ = lease[].publish(ArenaResidentEntry(900, 4096, 80, 10, False, 9, 0, 0))
    require(lease[].find(900).store_id == 9, "a removed owner can be put again")
    require(len(lease[].drain()) == 3 and lease[].find(900).store_id == -2, "drain empties the owner")
    ARENA_RESIDENTS.release(arena_handle(word))


struct _FaultStore(ArenaStoreFree):
    # A store whose free fails for chosen ids, recording every attempt and
    # every kept failure in order (the native form of the resident tests'
    # former `msel_free` fault injection; lane fx-leftovers item 9).
    var fail: List[Int]
    var attempted: List[Int]
    var failures: List[String]

    def __init__(out self, var fail: List[Int]):
        self.fail = fail^
        self.attempted = List[Int]()
        self.failures = List[String]()

    def free(mut self, store_id: Int) raises:
        if store_id < 0:
            return
        self.attempted.append(store_id)
        for i in range(len(self.fail)):  # small-loop(the fixture's failing ids)
            if self.fail[i] == store_id:
                raise Error("msel store: id " + String(store_id) + " is not live")

    def failed(mut self, error: Error) raises:
        self.failures.append(String(error))


def resident_close_order() raises:
    # Entries 900..904 with store ids 5, 6, -1 (an empty index buffer), 7, 8;
    # the frees of the first (5) and the last (8) fail.
    var word = arena_handle_word(ARENA_RESIDENTS.insert(ArenaResidentState()))
    var lease = ARENA_RESIDENTS.acquire(arena_handle(word))
    var ids: List[Int] = [5, 6, -1, 7, 8]
    for i in range(len(ids)):  # small-loop(the fixture's entries)
        _ = lease[].publish(ArenaResidentEntry(900 + i, 4096 * (i + 1), 80, 10, ids[i] < 0, ids[i], 0, 0))
    var fail_ids: List[Int] = [5, 8]
    var store = _FaultStore(fail_ids^)
    var drained = List[ArenaResidentEntry]()
    var kept = arena_resident_close_entries(lease[], store, drained)
    require(len(drained) == 5, "close drains every entry")
    require(len(store.attempted) == 4 and store.attempted[0] == 5 and store.attempted[1] == 6
            and store.attempted[2] == 7 and store.attempted[3] == 8,
            "every slot is attempted in entry order, past the first failure")
    require(len(store.failures) == 2 and store.failures[0] == "msel store: id 5 is not live"
            and store.failures[1] == "msel store: id 8 is not live", "failures kept in order, the first first")
    require(kept[0] and not kept[1] and not kept[2] and not kept[3] and kept[4], "only the failed entries are kept")
    require(lease[].find(900).store_id == 5 and lease[].find(904).store_id == 8,
            "the failed entries are published again for a retry")
    require(lease[].find(901).store_id == -2 and lease[].find(902).store_id == -2 and lease[].find(903).store_id == -2,
            "the freed entries are gone")
    # A retry once the store frees them: nothing kept, nothing left.
    var healthy = _FaultStore(List[Int]())
    var again = List[ArenaResidentEntry]()
    var kept_again = arena_resident_close_entries(lease[], healthy, again)
    require(len(again) == 2 and not kept_again[0] and not kept_again[1] and len(healthy.failures) == 0,
            "the retry frees the two retained entries")
    require(len(lease[].drain()) == 0, "empty after the retry")
    ARENA_RESIDENTS.release(arena_handle(word))


def resident_release_order() raises:
    var word = arena_handle_word(ARENA_RESIDENTS.insert(ArenaResidentState()))
    var lease = ARENA_RESIDENTS.acquire(arena_handle(word))
    _ = lease[].publish(ArenaResidentEntry(910, 4096, 80, 10, False, 11, 0, 0))
    _ = lease[].publish(ArenaResidentEntry(911, 8192, 8, 1, True, 12, 0, 0))
    var removed = ArenaResidentEntry(0, 0, 0, 0, False, -2, 0, 0)
    var fail_ids: List[Int] = [11]
    var failing = _FaultStore(fail_ids^)
    require(arena_resident_release_entry(lease[], failing, 910, removed), "a failed release keeps its entry")
    require(len(failing.failures) == 1 and removed.store_id == 11, "the failure is kept, the entry returned")
    require(lease[].find(910).store_id == 11, "the failed entry is published again for a retry")
    var healthy = _FaultStore(List[Int]())
    require(not arena_resident_release_entry(lease[], healthy, 911, removed) and removed.store_id == 12,
            "a healthy release frees and drops its entry")
    require(lease[].find(911).store_id == -2 and len(healthy.attempted) == 1, "the released entry is gone")
    require(not arena_resident_release_entry(lease[], healthy, 999, removed) and removed.store_id == -2
            and len(healthy.attempted) == 1, "an owner never put is a no-op")
    require(not arena_resident_release_entry(lease[], healthy, 910, removed) and lease[].find(910).store_id == -2,
            "the retry frees the retained entry")
    ARENA_RESIDENTS.release(arena_handle(word))


def scope_caches() raises:
    var word = arena_handle_word(ARENA_SCOPES.insert(ArenaScopeState(1 << 16)))
    var lease = ARENA_SCOPES.acquire(arena_handle(word))
    require(lease[].cache_of("x_prep", 258) == ARENA_HANDLE_NONE and lease[].min_words == 1 << 16, "a fresh scope")
    lease[].add_cache("x_prep", 258, 41)
    lease[].add_cache("x_prep", 2, 42)
    require(lease[].cache_of("x_prep", 258) == 41 and lease[].cache_of("x_prep", 2) == 42, "one cache per prefix and route")
    require(lease[].cache_of("x_metrics", 258) == ARENA_HANDLE_NONE, "another prefix has no cache")
    var taken = lease[].take_caches()
    require(len(taken) == 2 and taken[1].cache == 42 and lease[].cache_of("x_prep", 258) == ARENA_HANDLE_NONE, "take_caches empties the scope")
    ARENA_SCOPES.release(arena_handle(word))


def cache_holds() raises:
    """A held id is never freed under its reader (lane gd-device-store): `remove` / `drain` of a held entry
    unlink it at once and hand its free to the last `unhold`. Sabotage: make `_CacheTable.retire` return the
    entry unchanged -> "a held entry is not freed by remove" fails."""
    var word = arena_handle_word(ARENA_CACHES.insert(DeviceCacheState(2, "x_prep")))
    var lease = ARENA_CACHES.acquire(arena_handle(word))
    require(lease[].publish(300, 8, 21, 0, True) == 21, "a held publication")
    require(lease[].find_hold(300, 8) == 21 and lease[].hit_count() == 1, "a second holder counts a hit")
    require(lease[].publish(300, 8, 22, 0, True) == 21, "a lost race holds the kept id")
    var removed = lease[].remove(300, 8)
    require(removed.store_id == ARENA_STORE_ID_DEFERRED and lease[].live_count() == 0
            and lease[].find(300, 8, False) == -1, "a held entry is not freed by remove, only unlinked")
    require(lease[].unhold(300, 8, 21) == -1 and lease[].unhold(300, 8, 21) == -1, "two holders still hold it")
    require(lease[].unhold(300, 8, 21) == 21, "the last unhold frees the id")
    var refused = False
    try:
        _ = lease[].unhold(300, 8, 21)
    except error:
        refused = True
    require(refused, "an id no longer held is refused")
    require(lease[].publish(400, 4, 31, 0, True) == 31 and lease[].publish(500, 4, 32, 0) == 32, "one held, one not")
    var drained = lease[].drain()
    var deferred = 0
    var plain = 0
    for i in range(len(drained)):  # small-loop(two entries)
        if drained[i].store_id == ARENA_STORE_ID_DEFERRED:
            deferred += 1
        elif drained[i].store_id == 32:
            plain += 1
    require(deferred == 1 and plain == 1, "drain defers the held entry and returns the other to free")
    require(lease[].unhold(400, 4, 31) == 31, "the drained held id is freed by its unhold")
    require(lease[].find_hold(400, 4) == -1, "nothing to hold after the drain")
    ARENA_CACHES.release(arena_handle(word))


def fixture(unit: Int) raises:
    if unit == 0:
        handle_words()
    elif unit == 1:
        rows_state()
    elif unit == 2:
        cache_identity()
    elif unit == 3:
        cache_chain_order()
    elif unit == 4:
        resident_entries()
    elif unit == 5:
        scope_caches()
    elif unit == 6:
        resident_close_order()
    elif unit == 8:
        cache_holds()
    else:
        resident_release_order()


def test_arena_state() raises:
    check_cases("arena_state", ARENA_STATE_FIXTURES, fixture)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
