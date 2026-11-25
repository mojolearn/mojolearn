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

No Python object; called with the interpreter lock released.
"""
from bindings.arena_cache_run import ArenaCachePut, arena_scope_cache, arena_cache_free
from core.arena_state import ARENA_CACHES, arena_handle
from core.entry_table import EntryRef
from core.prep_entry_table import xp_transport_call
from x_prep.transport_abi import XP_DEV_PUT

#: The scope cache of model selection's label codes (one per x_prep route).
comptime MSEL_CODES_PREFIX = "msel_codes"


def msel_scope_codes_put(scope: Int, store: EntryRef, key_address: Int, key_words: Int, codes_address: Int,
                         n: Int, pin: Int) raises -> ArenaCachePut:
    """The slot of the `n` int32 codes at `codes_address` kept under the key
    (`key_address`, `key_words`) of the caller's label buffer in `scope`'s
    codes cache on the x_prep transport entry `store`: found (a hit; nothing
    goes up), or put once and published with `pin` (`.pinned`: the cache
    kept the pin; else the caller drops it). A lost publication race frees
    this call's own slot."""
    var word = arena_scope_cache(scope, store, String(MSEL_CODES_PREFIX))
    var lookup = ARENA_CACHES.acquire(arena_handle(word))
    var found = lookup[].find(key_address, key_words, True)
    if found >= 0:
        return ArenaCachePut(found, False)
    var source: List[Int] = [codes_address]
    var counts: List[Int] = [n]
    var store_id = xp_transport_call(store, XP_DEV_PUT, source, counts)
    var kept = store_id
    try:
        # Acquire after the put: a cache released meanwhile refuses here.
        var lease = ARENA_CACHES.acquire(arena_handle(word))
        kept = lease[].publish(key_address, key_words, store_id, pin)
    except error:
        try:
            arena_cache_free(store, store_id)
        except cleanup_error:
            raise Error("RuntimeError: label codes publication failed: " + String(error)
                        + "; cleanup of the unpublished slot " + String(store_id) + " also failed: "
                        + String(cleanup_error))
        raise error^
    if kept != store_id:
        arena_cache_free(store, store_id)
        return ArenaCachePut(kept, False)
    return ArenaCachePut(store_id, True)
