# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A public FUNCTION's `resident()` scope (lane gd-resident-2, 2026-11-25).

The estimators and the x_prep / x_metrics programs read a device-resident
input in place inside `resident()`; the public functions whose GPU side
uploaded every input on every call (connected_components, kpss_test,
select_d, refine, spectral_embedding: "no store hit; stats {}" in the GPU
verify of 2026-11-25, group R) read it the same way through this module.

UNPACK (this file, with the GIL): `FunctionScope(context)` reads the calling
thread's scope once per call: one atomic load (`arena_scopes_live`) when no
scope is alive anywhere in the process, which is the whole cost outside a
scope. Inside one, `pin(source, value, words)` answers the pin token of the
buffer the scope keeps for an admitted input (the caller's own buffer:
`metric_resident_owner`, the rule the metric doors use), or 0 when the input
is a converted per-call copy or below the scope's size floor.

Two ways a door hands the input over:
- `put(store, prefix, ...)`: the door holds the family store's transport
  entry (the metrics image's, resolved from its published table): the scope's
  cache of `prefix` on that store answers the slot id here (a hit, or one
  put), and the image reads the slot (spectral_embedding).
- a `ScopeBlock` (three Int words per input on the door's stack: scope word,
  pin token, "kept") travels as ONE optional int of the image request (its
  address; 0 outside a scope) and the image puts or finds the input in ITS
  OWN store (core/scope_store_slot.mojo); the image sets "kept" when the
  cache took the pin, and the door drops every pin the cache did not keep
  (`ScopeBlock.settle`) after the call (connected_components, refine,
  kpss_test, select_d).

A context connected without the scope keys keeps nothing resident. Nothing
here is reached per row or per element.
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.py_balanced import py_get
from bindings.native_loaded_module_door import native_loaded_module
from bindings.arena_scope_door import arena_scope_active
from bindings.arena_owner_door import arena_pin, arena_unpin
from bindings.arena_cache_run import ArenaCachePut, arena_scopes_live, arena_scope_floor, arena_scope_store_put
from bindings.metric_resident_source_door import metric_resident_owner
from core.entry_table import EntryRef


struct FunctionScope(Movable):
    """The calling thread's `resident()` scope for one public function call:
    its handle word (-1 outside a scope) and size floor."""
    var scope: Int
    var floor: Int

    def __init__(out self, context: PythonObject) raises:
        self.scope = -1
        self.floor = 0
        if not arena_scopes_live():
            return
        var local = py_get(context, "arena_local")
        if local is None:
            return
        self.scope = arena_scope_active(local, native_loaded_module("_thread")._local,
                                        native_loaded_module("builtins").str, context["arena_scope_owner"],
                                        context["arena_scope_class"])
        if self.scope >= 0:
            self.floor = arena_scope_floor(self.scope)

    def active(self) -> Bool:
        return self.scope >= 0

    def pin(self, source: PythonObject, value: PythonObject, words: Int) raises -> Int:
        """The pin token of the buffer the scope keeps for `value` (admitted
        from the argument `source`, `words` 4-byte words), or 0: outside a
        scope, below the floor, or a converted copy (never kept)."""
        if self.scope < 0 or words < 1 or words < self.floor:
            return 0
        var owner = metric_resident_owner(source, value)
        if owner is None:
            return 0
        return arena_pin(owner)

    def put(self, store: EntryRef, prefix: String, source: PythonObject, value: PythonObject,
            address: Int, words: Int) raises -> Int:
        """The slot id of `value`'s words (at `address`) in the scope's cache
        of `prefix` on `store` (found: a hit; else put once), or -1 when the
        scope keeps nothing for it (`pin`). The slot belongs to the cache: the
        caller never frees it."""
        var pin = self.pin(source, value, words)
        if pin == 0:
            return -1
        var put = ArenaCachePut(-1, False)
        try:
            with GILReleased(Python()):
                put = arena_scope_store_put(self.scope, store, prefix, address, words, pin)
        except error:
            arena_unpin(pin)
            raise error^
        if not put.pinned:
            arena_unpin(pin)
        return put.store_id


comptime SCOPE_BLOCK_INPUTS = 3


struct ScopeBlock(Movable):
    """The scope words an image request carries for up to
    SCOPE_BLOCK_INPUTS inputs (core/scope_store_slot.mojo numbering: per
    input the scope word, the pin token and the "kept" word), on the
    door's stack: no allocation. `address()` is 0 when no input is kept,
    so outside a scope the request carries 0 and the image reads nothing."""
    var words: Array[Int, 3 * SCOPE_BLOCK_INPUTS]
    var pinned: Bool

    def __init__(out self):
        self.words = Array[Int, 3 * SCOPE_BLOCK_INPUTS](fill=0)
        for i in range(SCOPE_BLOCK_INPUTS):  # small-loop(the block's fixed input count): 3
            self.words[3 * i] = -1
        self.pinned = False

    def keep(mut self, index: Int, scope: FunctionScope, source: PythonObject, value: PythonObject,
             words: Int) raises:
        """Input `index` (admitted `value` from the argument `source`, `words`
        4-byte words) is kept resident when the scope keeps its buffer."""
        var pin = scope.pin(source, value, words)
        if pin == 0:
            return
        self.words[3 * index] = scope.scope
        self.words[3 * index + 1] = pin
        self.words[3 * index + 2] = 0
        self.pinned = True

    def address(self) -> Int:
        if not self.pinned:
            return 0
        return Int(self.words.unsafe_ptr())

    def settle(mut self):
        """After the image call (also when it raised), with the GIL: drop
        every pin the image's scope cache did not keep."""
        if not self.pinned:
            return
        for i in range(SCOPE_BLOCK_INPUTS):  # small-loop(the block's fixed input count): 3
            var pin = self.words[3 * i + 1]
            if pin != 0 and self.words[3 * i + 2] == 0:
                arena_unpin(pin)
            self.words[3 * i + 1] = 0
        self.pinned = False
