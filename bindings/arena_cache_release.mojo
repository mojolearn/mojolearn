# SPDX-License-Identifier: Apache-2.0
# Native DeviceCache constructor/reinitialization, release, close and context
# exit, and the resident() scope enter/exit/close. One native call each.
#
# Lifetime and error order across GIL-releasing frees (the resident lifecycle
# item of docs/plans/runtime_takeover_status_20261022.json):
#   1. every admission check runs before the first state change;
#   2. the emptied state is published BEFORE the first free, so a thread that
#      runs while a free releases the GIL, or any finalizer, sees a closed
#      cache, never a freed slot;
#   3. every independent slot is attempted; the first failure is retained
#      and raised, later failures are dropped; failed entries are published
#      again for a retry;
#   4. the owners (caller sources, pandas/Arrow frames) are released as the
#      LAST step, after every free and every publication, and before the
#      first failure is restored. An owner finalizer (an Arrow foreign
#      buffer's base, a user __del__) therefore runs inside the operation but
#      after its native work, and cannot observe a half-closed cache or
#      replace the retained first error. No owner graph is inspected, so
#      Arrow-backed pandas owners stay supported like every other owner.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.arena_cache_stats import _attribute, _mapping, _string_keys
from bindings.arena_cache_get import arena_cache_key
from bindings.arena_cache_binding import cache_binding_attribute, _cache_entry_name
from bindings.arena_cache_upload import (_entries, _route, _native_call, _counter_set,
    CachePendingError, _take_pending, _drop_pending, _restore_pending)


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


def _admit_setattr(model: PythonObject, context: PythonObject) raises:
    var ancestry = py_call(context["cache_type_mro"],py_type(model))
    for i in range(len(ancestry)):  # driver-loop: fixed native attribute-write surface over the class ancestry
        var hook = py_get(_mapping(ancestry[i],context),"__setattr__")
        if hook is not None and hook is not context["cache_object_setattr"]:
            raise Error("TypeError: DeviceCache lifecycle cannot invoke a custom attribute setter")


def _cache_pairs(entries: PythonObject, context: PythonObject) raises -> PythonObject:
    # Validate the complete cleanup before the first state change.
    var b = context["arena_native_types"]
    var pairs = py_call(b["list"],py_call(entries.items))
    for i in range(len(pairs)):  # driver-loop: admit every (slot, source) entry before releasing the first slot
        var entry = pairs[i][1]
        if (py_type(entry) is not b["tuple"] and py_type(entry) is not b["list"]) or len(entry) != 2:
            raise Error("TypeError: DeviceCache entries require native (slot, source) pairs")
    return pairs


def _republish(current: PythonObject, failed: PythonObject) raises:
    for i in range(len(failed)):  # driver-loop: failed entries keep their owner for a retry
        if not py_contains(current,failed[i][0]):
            py_setitem(current,failed[i][0],failed[i][1])


def arena_cache_close(model: PythonObject, context: PythonObject) raises:
    var pairs = _cache_pairs(_entries(model,context),context)
    var route = PythonObject(None)
    if len(pairs):
        route = _route(model,context)
    _admit_setattr(model,context)
    # Publish the empty mapping first (the old method's swap order).
    _counter_set(model,"_entries",Python.dict())
    var first = FirstFailure()
    var failed = Python.list()
    for i in range(len(pairs)):  # driver-loop: attempt every independent slot release
        try:
            _ = _native_call(route[1],Python.tuple(pairs[i][1][0]))
        except error:
            first.keep(error)
            _ = py_call(failed.append,pairs[i])
    if len(failed):
        try:
            _republish(_entries(model,context),failed)
        except error:
            first.keep(error)
    # Owners are released here, after every free and publication.
    _ = failed^
    _ = pairs^
    first.finish()


def _cache_release(model: PythonObject, source: PythonObject, context: PythonObject) raises:
    var key = arena_cache_key(model,source,context)
    var entries = _entries(model,context)
    var hit = py_get(entries,key)
    if hit is None:
        return
    var b = context["arena_native_types"]
    if (py_type(hit) is not b["tuple"] and py_type(hit) is not b["list"]) or len(hit) != 2:
        raise Error("TypeError: DeviceCache entries require native (slot, source) pairs")
    var route = _route(model,context)
    # Unpublish before the GIL-releasing free: no other thread can be handed
    # the slot while it is being freed.
    _ = py_call(entries.pop,key)
    _ = entries^
    var first = FirstFailure()
    try:
        _ = _native_call(route[1],Python.tuple(hit[0]))
    except error:
        first.keep(error)
        try:
            _republish(_entries(model,context),Python.list(Python.tuple(key,hit)))
        except republish_error:
            first.keep(republish_error)
    _ = hit^
    first.finish()


def _cache_init(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    var binding = args[0]
    var prefix = args[1]
    # The old constructor's lookup order: put, free, then the optional live.
    var missing = Python.list()
    var put = cache_binding_attribute(binding,_cache_entry_name(prefix,"_dev_put",context),missing,True,context)
    var free = cache_binding_attribute(binding,_cache_entry_name(prefix,"_dev_free",context),missing,True,context)
    var live = cache_binding_attribute(binding,_cache_entry_name(prefix,"_dev_live",context),PythonObject(None),False,context)
    _admit_setattr(model,context)
    # Reinitialization frees the live entries first with the route that
    # uploaded them (the old constructor lost them). On a failed free the
    # failed entries stay published and the old fields are kept.
    var old = _attribute(model,"_entries",missing,False,context)
    if old is not missing:
        if py_type(old) is not context["arena_native_types"]["dict"]:
            raise Error("TypeError: DeviceCache entries require a native dictionary")
        if len(old):
            arena_cache_close(model,context)
    _counter_set(model,"_put",put)
    _counter_set(model,"_free",free)
    _counter_set(model,"_live",live)
    _counter_set(model,"_entries",Python.dict())
    _counter_set(model,"uploads",PythonObject(0))
    _counter_set(model,"hits",PythonObject(0))


def _local_fields(context: PythonObject) raises -> PythonObject:
    var local = context["cache_local"]
    if py_type(local) is not context["cache_local_type"]:
        raise Error("TypeError: resident scope requires the native thread-local owner")
    if py_get(_mapping(context["cache_local_type"],context),"__getattribute__") is not context["cache_local_getattribute"]:
        raise Error("TypeError: resident thread-local attribute entry has changed")
    var fields = py_call(context["cache_local_getattribute"],local,"__dict__")
    if py_type(fields) is not context["arena_native_types"]["dict"]:
        raise Error("TypeError: resident thread-local fields require a native dictionary")
    _string_keys(fields,context)
    return fields


def _scope_close(scope: PythonObject, context: PythonObject) raises:
    var b = context["arena_native_types"]
    if py_type(scope) is not context["cache_scope_class"]:
        raise Error("TypeError: resident scope close requires its ordinary scope")
    var caches = _attribute(scope,"caches",PythonObject(None),True,context)
    if py_type(caches) is not b["dict"]:
        raise Error("TypeError: resident cache registry requires a native dictionary")
    var pairs = py_call(b["list"],py_call(caches.items))
    for i in range(len(pairs)):  # driver-loop: admit every registered cache before the first release
        var entry = pairs[i][1]
        if py_type(entry) is not b["tuple"] or len(entry) != 2 or py_type(entry[0]) is not context["cache_class"]:
            raise Error("TypeError: resident cache registry requires (DeviceCache, binding) pairs")
    _admit_setattr(scope,context)
    _counter_set(scope,"caches",Python.dict())
    _ = caches^
    var first = FirstFailure()
    var failed = Python.list()
    for i in range(len(pairs)):  # driver-loop: close every independent cache of the scope
        try:
            arena_cache_close(pairs[i][1][0],context)
        except error:
            first.keep(error)
            _ = py_call(failed.append,pairs[i])
    if len(failed):
        try:
            var current = _attribute(scope,"caches",PythonObject(None),True,context)
            if py_type(current) is not b["dict"]:
                raise Error("TypeError: resident cache registry requires a native dictionary")
            _republish(current,failed)
        except error:
            first.keep(error)
    _ = failed^
    _ = pairs^
    first.finish()


def _scope_enter(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _local_fields(context)
    if py_get(fields,"scope") is None:
        var scope = py_call(context["cache_object_new"],context["cache_scope_class"])
        _admit_setattr(scope,context)
        _counter_set(scope,"min_words",_attribute(model,"min_words",PythonObject(None),True,context))
        _counter_set(scope,"caches",Python.dict())
        _admit_setattr(model,context)
        # The old chained assignment's order: self._scope, then _LOCAL.scope.
        _counter_set(model,"_scope",scope)
        py_setitem(fields,"scope",scope)
    return model


def _scope_exit(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var scope = _attribute(model,"_scope",PythonObject(None),True,context)
    if scope is not None:
        _admit_setattr(model,context)
        py_setitem(_local_fields(context),"scope",PythonObject(None))
        # A failed close keeps self._scope (with its failed caches) for a retry.
        _scope_close(scope,context)
        _counter_set(model,"_scope",PythonObject(None))
    return PythonObject(False)


def arena_cache_release_operation(model: PythonObject, operation: String, args: PythonObject,
                                  context: PythonObject) raises -> PythonObject:
    if operation == "cache_init":
        _cache_init(model,args,context)
        return PythonObject(None)
    if operation == "cache_release":
        _cache_release(model,args[0],context)
        return PythonObject(None)
    if operation == "cache_close":
        arena_cache_close(model,context)
        return PythonObject(None)
    if operation == "cache_exit":
        arena_cache_close(model,context)
        return PythonObject(False)
    if operation == "scope_enter":
        return _scope_enter(model,context)
    if operation == "scope_exit":
        return _scope_exit(model,context)
    if operation == "scope_close":
        _scope_close(model,context)
        return PythonObject(None)
    raise Error("ValueError: unknown native cache release operation")
