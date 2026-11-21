# SPDX-License-Identifier: Apache-2.0
"""UNPACK / PACK of the `resident()` scope and its caches for every door
that keeps a program input on the device (lane arena, 2026-11-21).

The thread-local `_arena_io._LOCAL` keeps ONE attribute, `scope`: the active
`_Scope` object, whose `_owner` slot holds the native owner of an
`ArenaScopeState`. A door resolves it ONCE per public operation
(`arena_scope_active`: the thread-local's dictionary, one slot read, one
downcast), asks the scope for its cache on the run's store
(`bindings/arena_cache_run.mojo` `arena_scope_cache`), takes one lease of
that cache and then looks each input up with `arena_cache_slot`: a hit is
one lock word and one dictionary read with the interpreter lock held; a miss
pins the source and uploads with the lock released. Nothing here is reached
per row, per batch or per token.
"""
from std.memory import ArcPointer
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.py_balanced import py_get, py_setitem, py_type
from bindings.native_loaded_module_door import native_namespace_string_keys
from bindings.arena_owner_door import ARENA_OWNER_SCOPE, arena_owner_held, arena_pin, arena_unpin
from bindings.arena_cache_run import ArenaCachePut, ArenaCacheReleased, arena_cache_drop, arena_cache_upload
from core.arena_state import ARENA_HANDLE_NONE, DeviceCacheState
from core.entry_table import EntryRef


def _local_fields(local: PythonObject, local_type: PythonObject, string_type: PythonObject) raises -> PythonObject:
    # The calling thread's own dictionary of the exact native thread-local
    # (a static C type: its attribute entry cannot be replaced).
    if py_type(local) is not local_type:
        raise Error("TypeError: resident scope requires the native thread-local owner")
    var fields = local.__dict__
    if not native_namespace_string_keys(fields._obj_ptr, string_type):
        raise Error("TypeError: resident thread-local keys require exact native strings")
    return fields


def arena_scope_object(local: PythonObject, local_type: PythonObject, string_type: PythonObject) raises -> PythonObject:
    """The calling thread's active scope object, or None outside a
    `resident()` block."""
    var scope = py_get(_local_fields(local, local_type, string_type), "scope")
    return scope


def arena_scope_publish(local: PythonObject, local_type: PythonObject, string_type: PythonObject,
                        scope: PythonObject) raises:
    """Make `scope` (or None) the calling thread's active scope."""
    py_setitem(_local_fields(local, local_type, string_type), "scope", scope)


def arena_scope_word(scope: PythonObject, getter: PythonObject, cls: PythonObject) raises -> Int:
    """The handle word of a scope object's native state (`getter` is the
    bound `__get__` of `_Scope`'s own `_owner` member descriptor)."""
    return arena_owner_held(getter, scope, cls, ARENA_OWNER_SCOPE, "the resident scope").word


def arena_scope_active(local: PythonObject, local_type: PythonObject, string_type: PythonObject,
                       getter: PythonObject, cls: PythonObject) raises -> Int:
    """The handle word of the calling thread's active scope, or
    `ARENA_HANDLE_NONE`. Once per public operation."""
    var scope = arena_scope_object(local, local_type, string_type)
    if scope is None:
        return ARENA_HANDLE_NONE
    return arena_scope_word(scope, getter, cls)


def arena_cache_slot(lease: ArcPointer[DeviceCacheState], word: Int, store: EntryRef, source: PythonObject,
                     address: Int, words: Int) raises -> Int:
    """The store id of the resident copy of `source`'s buffer (`address`,
    `words` 4-byte words) in the cache `word` (`lease` is the run's one lease
    of it): found, or uploaded once and published with `source` pinned."""
    var found = lease[].find(address, words, True)
    if found >= 0:
        return found
    var pin = arena_pin(source)
    var put = ArenaCachePut(-1, False)
    try:
        with GILReleased(Python()):
            put = arena_cache_upload(word, store, address, words, pin)
    except error:
        arena_unpin(pin)
        raise error^
    if not put.pinned:
        arena_unpin(pin)
    return put.store_id


def arena_unpin_all(pins: List[Int]):
    """Drop the sources of entries that left a cache: the LAST step of a
    release, after every free and publication, so a source's own finalizer
    cannot observe a half-closed cache."""
    for i in range(len(pins)):  # driver-loop: over the pins of the freed entries
        arena_unpin(pins[i])


def arena_cache_drop_inout(cache: Int, store: EntryRef, addresses: List[Int], words: List[Int]) -> String:
    """Drop the resident copies of a run's in-out inputs from the cache
    `cache` (nothing when it is `ARENA_HANDLE_NONE`). Returns failure text
    (empty on success); never raises, so the caller's own error or cleanup
    report stays in charge."""
    if cache < 0 or len(addresses) == 0:
        return String()
    try:
        var released = ArenaCacheReleased()
        with GILReleased(Python()):
            released = arena_cache_drop(cache, store, addresses, words)
        arena_unpin_all(released.pins)
        return released.failure
    except error:
        return "resident in-out drop: " + String(error)
