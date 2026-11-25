# SPDX-License-Identifier: Apache-2.0
# Native DeviceCache constructor/reinitialization, release, close and context
# exit, and the resident() scope init/enter/exit/close. One native call each.
#
# Lane arena (2026-11-21): the cache and the scope are native states
# (`core/arena_state.mojo`: `ARENA_CACHES`, `ARENA_SCOPES`) behind the one
# owner object in the Python instance's `_owner` slot
# (`bindings/arena_owner_door.mojo`). UNPACK here is the slot read and the
# argument admission; RUN is `bindings/arena_cache_run.mojo` with the
# interpreter lock released; PACK drops the freed entries' sources. A cache
# is built on the x_prep transport entry of its binding (mode and vendor
# words read once at construction): the put / free / live builtins are no
# longer held or called, and a binding without that entry is refused at
# construction.
#
# Lifetime and error order across the frees (unchanged):
#   1. every admission check runs before the first state change;
#   2. the entries are unlinked BEFORE the first free, so a thread that runs
#      while a free runs, or any finalizer, sees a closed cache, never a
#      freed slot;
#   3. every independent slot is attempted; the first failure is retained
#      and raised, later failures are dropped; failed entries are linked
#      again for a retry;
#   4. the owners (caller sources, pandas/Arrow frames) are released as the
#      LAST step, after every free and every publication, and before the
#      first failure is raised. An owner finalizer (an Arrow foreign
#      buffer's base, a user __del__) therefore runs inside the operation but
#      after its native work, and cannot observe a half-closed cache or
#      replace the retained first error. No owner graph is inspected, so
#      Arrow-backed pandas owners stay supported like every other owner.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.py_balanced import py_call, py_type
from bindings.py_typed_call import py_call_typed
from bindings.arena_cache_get_door import arena_cache_key, arena_cache_view
from bindings.arena_cache_binding_pack import cache_binding_attribute, _cache_entry_name
from bindings.arena_cache_upload_door import CachePendingError, _take_pending, _drop_pending, _restore_pending
from bindings.arena_cache_run import (
    ArenaCacheReleased, arena_cache_close, arena_cache_new, arena_cache_release, arena_scope_close, arena_scope_new,
)
from bindings.arena_owner_door import (
    ArenaOwnerView, ARENA_OWNER_CACHE, ARENA_OWNER_SCOPE, arena_owner_new, arena_owner_slot, arena_owner_view,
)
from bindings.arena_scope_door import arena_scope_object, arena_scope_publish, arena_scope_word, arena_unpin_all
from core.arena_state import ARENA_CACHES, arena_handle
from core.entry_table import EntryRef, entry_ref_unresolved
from core.prep_entry_table import PrepEntryKey, xp_transport_ref
from x_prep.transport_abi import xp_vendor_word


struct FirstFailure(Movable):
    # The first failure of a multi-release operation, owned locally while
    # later releases can release the GIL (never left in the shared pending
    # slot another thread's native call could overwrite).
    var held: Bool
    var message: String
    var pending: CachePendingError

    def __init__(out self):
        self.held = False
        self.message = String("")
        self.pending = CachePendingError()

    def keep(mut self, error: Error) raises:
        var saved = _take_pending(error)
        if self.held:
            _drop_pending(saved)
            return
        self.held = True
        self.message = String(error)
        self.pending = saved^

    def finish(mut self) raises:
        if not self.held:
            return
        self.held = False
        _restore_pending(self.pending)
        raise Error(self.message)


def _finish(released: ArenaCacheReleased) raises:
    # PACK of a release: the freed entries' sources last, then the first
    # failure (the store's own refusal text).
    arena_unpin_all(released.pins)
    if released.failure.byte_length() > 0:
        raise Error(released.failure)


def _cache_close(view: ArenaOwnerView) raises:
    var lease = ARENA_CACHES.acquire(arena_handle(view.word))
    if lease[].live_count() == 0:
        return
    var released = ArenaCacheReleased()
    with GILReleased(Python()):
        released = arena_cache_close(view.word, view.store)
    _finish(released)


def _cache_release(view: ArenaOwnerView, source: PythonObject, context: PythonObject) raises:
    var key = arena_cache_key(source, context)
    var lease = ARENA_CACHES.acquire(arena_handle(view.word))
    if lease[].find(key.address, key.words, False) < 0:
        return
    var released = ArenaCacheReleased()
    with GILReleased(Python()):
        released = arena_cache_release(view.word, view.store, key.address, key.words)
    _finish(released)


def _cache_store(binding: PythonObject, prefix: PythonObject, put: PythonObject, free: PythonObject,
                 context: PythonObject) raises -> EntryRef:
    # The native store entry of a cache: the x_prep transport of the image
    # `binding` is, named by the image's own mode and vendor words. Resolved
    # once per cache, at construction. Only compiled entries are called.
    var refusal = "TypeError: DeviceCache requires a connected native x_prep store; Python callbacks cannot execute"
    var compiled = py_type(context["arena_cast"])
    if (py_type(prefix) is not context["arena_native_types"]["str"] or String(py=prefix) != "x_prep"
            or py_type(put) is not compiled or py_type(free) is not compiled):
        raise Error(refusal)
    var missing = Python.list()
    var mode_entry = cache_binding_attribute(binding, PythonObject("x_prep_numeric_mode"), missing, False, context)
    var vendor_entry = cache_binding_attribute(binding, PythonObject("x_prep_vendor"), missing, False, context)
    if py_type(mode_entry) is not compiled or py_type(vendor_entry) is not compiled:
        raise Error(refusal)
    var mode = Int(py=py_call_typed(mode_entry))
    var vendor = xp_vendor_word(String(py=py_call_typed(vendor_entry)))
    if mode < 0 or mode > 1 or vendor <= 0:
        raise Error("RuntimeError: mojolearn: the x_prep binding publishes no native resident store")
    return xp_transport_ref(PrepEntryKey(mode, vendor))


def _cache_init(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    var binding = args[0]
    var prefix = args[1]
    # The constructor's lookup order and refusals: put, then free.
    var missing = Python.list()
    var put = cache_binding_attribute(binding,_cache_entry_name(prefix,"_dev_put",context),missing,True,context)
    var free = cache_binding_attribute(binding,_cache_entry_name(prefix,"_dev_free",context),missing,True,context)
    var store = _cache_store(binding, prefix, put, free, context)
    var previous = arena_owner_slot(context["cache_owner_get"], model, context["cache_class"],
                                    context["cache_attribute_error"])
    if previous is not None:
        # Reinitialization frees the live entries first, on the store that
        # uploaded them. A failed free keeps the old owner in the slot with
        # the failed entries linked.
        _cache_close(arena_owner_view(previous, ARENA_OWNER_CACHE))
    var word = arena_cache_new(store, String(py=prefix))
    # From here the owner object releases the state when CPython frees it.
    var owner = arena_owner_new(ARENA_OWNER_CACHE, word, store)
    _ = py_call(context["cache_owner_set"], model, owner)
    # The replaced owner (if any) is freed here: its state is empty.
    _ = previous^


def _scope_floor(value: PythonObject, context: PythonObject) raises -> Int:
    if py_type(value) is not context["arena_native_types"]["int"]:
        raise Error("TypeError: resident: min_words must be an integer")
    return Int(py=value)


def _scope_init(scope: PythonObject, floor: Int, context: PythonObject) raises:
    # A scope's native state behind the one owner in its `_owner` slot. A
    # scope resolves each cache's store from the cache's own route, so its
    # owner holds no entry.
    var owner = arena_owner_new(ARENA_OWNER_SCOPE, arena_scope_new(floor), entry_ref_unresolved())
    _ = py_call(context["cache_scope_owner_set"], scope, owner)


def _scope_close(scope: PythonObject, context: PythonObject) raises:
    var word = arena_scope_word(scope, context["cache_scope_owner_get"], context["cache_scope_class"])
    var released = ArenaCacheReleased()
    with GILReleased(Python()):
        released = arena_scope_close(word)
    _finish(released)


def _scope_enter(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    # `args`: () for `resident.__enter__` (the floor is the instance's
    # `min_words` slot), or (min_words,) from a door that opens a scope for
    # its own run and never writes the Python attribute.
    var strings = context["arena_native_types"]["str"]
    if arena_scope_object(context["cache_local"], context["cache_local_type"], strings) is None:
        var floor = 0
        if len(args):
            floor = _scope_floor(args[0], context)
        else:
            floor = _scope_floor(arena_owner_slot(context["cache_resident_floor_get"], model,
                                                  context["cache_resident_class"], context["cache_attribute_error"]),
                                 context)
        var scope = py_call(context["cache_object_new"], context["cache_scope_class"])
        _scope_init(scope, floor, context)
        # The old chained assignment's order: self._scope, then _LOCAL.scope.
        _ = py_call(context["cache_resident_scope_set"], model, scope)
        arena_scope_publish(context["cache_local"], context["cache_local_type"], strings, scope)
    return model


def _scope_exit(model: PythonObject, context: PythonObject) raises -> PythonObject:
    # A `resident` that entered inside an enclosing scope holds none (its
    # slot is None or was never set) and closes nothing.
    var scope = arena_owner_slot(context["cache_resident_scope_get"], model, context["cache_resident_class"],
                                 context["cache_attribute_error"])
    if scope is not None:
        arena_scope_publish(context["cache_local"], context["cache_local_type"],
                            context["arena_native_types"]["str"], PythonObject(None))
        # A failed close keeps self._scope (with its failed caches) for a retry.
        _scope_close(scope, context)
        _ = py_call(context["cache_resident_scope_set"], model, PythonObject(None))
    return PythonObject(False)


def arena_cache_release_operation(model: PythonObject, operation: String, args: PythonObject,
                                  context: PythonObject) raises -> PythonObject:
    if operation == "cache_init":
        _cache_init(model,args,context)
        return PythonObject(None)
    if operation == "cache_release":
        _cache_release(arena_cache_view(model,context),args[0],context)
        return PythonObject(None)
    if operation == "cache_close":
        _cache_close(arena_cache_view(model,context))
        return PythonObject(None)
    if operation == "cache_exit":
        _cache_close(arena_cache_view(model,context))
        return PythonObject(False)
    if operation == "scope_init":
        _scope_init(model,_scope_floor(args[0],context),context)
        return PythonObject(None)
    if operation == "scope_enter":
        return _scope_enter(model,args,context)
    if operation == "scope_exit":
        return _scope_exit(model,context)
    if operation == "scope_close":
        _scope_close(model,context)
        return PythonObject(None)
    raise Error("ValueError: unknown native cache release operation")
