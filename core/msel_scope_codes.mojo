# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A splitter's label codes kept in the `resident()` scope BY THE CALLER'S
LABEL BUFFER (lane gd-rf-msel, 2026-11-25).

StratifiedKFold's program route puts the n int32 class codes of y into the
program image's x_prep store once per split call (bindings/msel_split_plan_run.mojo
`MselProgSlot`) and frees them after the plan. The codes are a fixed function
of y's words (the label UNPACK of bindings/msel_split_door.mojo), so inside a
scope they are kept in the scope's own cache for model selection's codes
(prefix `MSEL_CODES_PREFIX`, on the same x_prep store; never the "x_prep"
cache, whose entries hold the words found at their key) under the KEY of the
caller's y buffer (its address and its 4-byte word count), pinned by the
cache: the second split call on the same y in the scope finds the slot (a
hit) and uploads nothing. The scope's close frees the slot into the x_prep
store (bindings/arena_cache_run.mojo `_family_store_note`).

The same y always gives the same codes (the label form and the class count
are read from y's words), so a found entry holds what the put would upload.
A caller that rewrites y inside a scope breaks every identity-keyed resident
input in the same way; that is the scope's contract.

THE HELPER TABLE'S STORE (lane gd-rf-msel, 2026-11-25). The stratified
default folds put their codes into the model-selection helper table's own
store (`MselHeld`, MH_STORE_PUT) and run MH_FOLD_IDS_RES there. That store is
a family store under its own prefix (`MSEL_HELPER_CODES_PREFIX`) with its own
operation numbers (`msel_helper_store_ops`: MH_STORE_PUT 21, which counts
bytes; MH_STORE_FREE 15; no live operation), registered with the scope
through bindings/arena_cache_run.mojo's per-family op map, so the scope's
close frees its slots with MH_STORE_FREE.

No Python object; called with the interpreter lock released.
"""
from bindings.arena_cache_run import (
    ArenaCachePut, ArenaStoreOps, arena_store_ops_transport, arena_scope_cache, arena_cache_put, arena_cache_free,
)
from core.arena_state import ARENA_CACHES, arena_handle
from core.entry_table import EntryRef
from core.msel_entry_table import MH_STORE_PUT, MH_STORE_FREE

#: The scope cache of model selection's label codes on the program image's
#: x_prep store (StratifiedKFold's program route).
comptime MSEL_CODES_PREFIX = "msel_codes"
#: The scope cache of model selection's label codes in the helper table's own
#: store (the stratified default folds, `MselHeld`).
comptime MSEL_HELPER_CODES_PREFIX = "msel_helper_codes"


@always_inline
def msel_helper_store_ops() -> ArenaStoreOps:
    """The helper table store's operation numbers: put MH_STORE_PUT (a byte
    count), free MH_STORE_FREE, no live-slot operation."""
    return ArenaStoreOps(MH_STORE_PUT, MH_STORE_FREE, -1, True)


def msel_scope_codes_put(scope: Int, store: EntryRef, key_address: Int, key_words: Int, codes_address: Int,
                         n: Int, pin: Int) raises -> ArenaCachePut:
    """`msel_scope_codes_put` on the x_prep transport entry `store`, prefix
    `MSEL_CODES_PREFIX`."""
    return msel_scope_codes_put(scope, store, arena_store_ops_transport(), String(MSEL_CODES_PREFIX), key_address,
                                key_words, codes_address, n, pin)


def msel_scope_codes_put(scope: Int, store: EntryRef, ops: ArenaStoreOps, prefix: String, key_address: Int,
                         key_words: Int, codes_address: Int, n: Int, pin: Int) raises -> ArenaCachePut:
    """The slot of the `n` int32 codes at `codes_address` kept under the key
    (`key_address`, `key_words`) of the caller's label buffer in `scope`'s
    `prefix` cache on the family store `store` (operation numbers `ops`):
    found (a hit; nothing goes up), or put once and published with `pin`
    (`.pinned`: the cache kept the pin; else the caller drops it). A lost
    publication race frees this call's own slot."""
    var word = arena_scope_cache(scope, store, ops, prefix)
    var lookup = ARENA_CACHES.acquire(arena_handle(word))
    var found = lookup[].find(key_address, key_words, True)
    if found >= 0:
        return ArenaCachePut(found, False)
    var store_id = arena_cache_put(store, ops, codes_address, n)
    var kept = store_id
    try:
        # Acquire after the put: a cache released meanwhile refuses here.
        var lease = ARENA_CACHES.acquire(arena_handle(word))
        kept = lease[].publish(key_address, key_words, store_id, pin)
    except error:
        try:
            arena_cache_free(store, ops, store_id)
        except cleanup_error:
            raise Error("RuntimeError: label codes publication failed: " + String(error)
                        + "; cleanup of the unpublished slot " + String(store_id) + " also failed: "
                        + String(cleanup_error))
        raise error^
    if kept != store_id:
        arena_cache_free(store, ops, store_id)
        return ArenaCachePut(kept, False)
    return ArenaCachePut(store_id, True)
