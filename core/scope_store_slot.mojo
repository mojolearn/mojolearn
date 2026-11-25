# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A public function's input, resident in its GPU image's own store (lane gd-resident-2, 2026-11-25).

The door (bindings/function_scope_door.mojo) hands an image ONE optional int word per operation: the address of a
SCOPE BLOCK of three Int words: the calling thread's `resident()` scope word (-1: none), the pin token of the
caller's buffer (0: keep nothing: a converted copy, or below the scope's floor) and a "kept" word (0 in). The image
then finds or puts the input's host words in ITS OWN `DeviceStore` (the family store of that image's context,
reached through the image's own transport entry: core/family_store_transport.mojo) through the scope's cache of the
image's family prefix (bindings/arena_cache_run.mojo `arena_scope_store_put`, the call the member families make): a
hit after the first call of the scope, so the kernels read the slot in place and nothing is uploaded again. The
image writes 1 to "kept" when the cache took the pin; the door drops the pin otherwise (after the call, with the
GIL). An operation that keeps several inputs resident takes one block per input, consecutive.

The store slot belongs to the scope's cache (freed at the scope's close through the same transport entry), never
to the operation. Outside a scope nothing here runs. Serial by nature: one cache lookup or one put per input.
"""
from core.entry_table import EntryRef, entry_ref_word
from core.native_descriptor_abi import NativeDescriptorCall
from bindings.arena_cache_run import arena_scope_store_put
from x_prep.transport_abi import XP_OP_COUNT


comptime SCOPE_BLOCK_WORDS = 3


@fieldwise_init
struct ScopeInput(ImplicitlyCopyable, Movable):
    """One input's scope words as the door handed them: `scope` (-1: no scope), `pin` (0: keep nothing) and
    `kept` (the address of the block's Int word the image sets to 1 when the cache keeps the pin; 0: none)."""
    var scope: Int
    var pin: Int
    var kept: Int

    def wanted(self) -> Bool:
        return self.scope >= 0 and self.pin != 0 and self.kept != 0


def scope_input_none() -> ScopeInput:
    return ScopeInput(-1, 0, 0)


def scope_input_block(address: Int, index: Int) -> ScopeInput:
    """Input `index`'s scope words in the door's block at `address` (0: an operation called without a block, or by
    a caller that keeps nothing resident: none)."""
    if address == 0:
        return scope_input_none()
    var words = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=address)
    var at = index * SCOPE_BLOCK_WORDS
    return ScopeInput(words[at], words[at + 1], address + (at + 2) * 8)


def scope_store_ref(entry: NativeDescriptorCall, mode: Int, vendor: Int) -> EntryRef:
    """The image's own store transport entry as the scope cache's store (its route word: tier and vendor)."""
    return EntryRef(entry, entry_ref_word(mode, vendor, XP_OP_COUNT))


def scope_store_slot(store: EntryRef, prefix: String, given: ScopeInput, address: Int, words: Int) raises -> Int:
    """The slot of `store` holding the `words` 4-byte host words at `address`: found in the scope's cache of
    `prefix` (a hit) or put once and published with the door's pin. Sets the kept word when the cache took the
    pin."""
    var put = arena_scope_store_put(given.scope, store, prefix, address, words, given.pin)
    if put.pinned:
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=given.kept)[0] = 1
    return put.store_id
