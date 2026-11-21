# SPDX-License-Identifier: Apache-2.0
# Native public cache lifecycle operations. Allocation, release and resident
# scope cleanup are separate callers; this entry owns only the named operation.
#
# Lane arena (2026-11-21): `live` is one call through the cache's own store
# entry (`XP_DEV_LIVE` of the x_prep transport, resolved when the cache was
# built); the `_live` builtin and its identity table are gone.
from std.python import PythonObject
from bindings.arena_owner_door import ARENA_OWNER_CACHE, arena_owner_held
from bindings.arena_cache_run import arena_cache_live


def arena_cache_lifecycle(model: PythonObject, operation: String,
                          context: PythonObject) raises -> PythonObject:
    if operation == "cache_enter":
        # Context entry has always returned self without inspecting fields.
        return model
    if operation == "cache_live":
        var view = arena_owner_held(context["cache_owner_get"], model, context["cache_class"], ARENA_OWNER_CACHE,
                                    "DeviceCache")
        # The store's live slot count: a counter read, the interpreter lock
        # stays held as it did around the builtin.
        return PythonObject(arena_cache_live(view.store))
    raise Error("ValueError: unknown native cache lifecycle operation")
