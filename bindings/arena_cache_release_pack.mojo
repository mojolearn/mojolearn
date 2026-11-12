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
# The PACK half of the former `bindings/arena_cache_release.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `bindings/arena_cache_release_unpack.mojo`.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.arena_cache_stats_unpack import _mapping, _string_keys
from bindings.arena_cache_stats_pack import _attribute
from bindings.arena_cache_get_door import arena_cache_key
from bindings.arena_cache_binding_pack import cache_binding_attribute, _cache_entry_name
from bindings.arena_cache_upload_door import _entries, _route, _native_call, _counter_set
from bindings.arena_cache_upload_door import CachePendingError, _take_pending, _drop_pending, _restore_pending


from bindings.arena_cache_release_unpack import (
    _admit_setattr,
    _cache_init,
    _cache_release,
    _local_fields,
    _scope_close,
    arena_cache_close,
)


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
