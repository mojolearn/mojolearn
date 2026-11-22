# SPDX-License-Identifier: Apache-2.0
"""The resident store ids of a DeviceRows fold at an estimator door (lane
arena, 2026-11-21; before: `door_rows_store` in bindings/device_rows_door.mojo,
which called back through Python into the "store_ids" operation of the rows
binding).

UNPACK is the fold view (`bindings/arena_owner_door.mojo` `arena_rows`, or the
one a `DoorRows` already holds). RUN is the find-or-put of the base and of the
int64 indices in the fold's resident state (`core/arena_state.mojo`
`ARENA_RESIDENTS`), the put made through the store owner's resolved helper
entry with the interpreter lock released. The result is typed
(`DoorRowsStore`); no Python object is built.

Once per estimator door call (a fold's fit or score), never per row; two
store entries per fold, so there is nothing to split across cores.

This is a module of its own because the resident entry door
(bindings/arena_resident_rows_door.mojo) imports bindings/device_rows_door.mojo
through the classical input doors.
"""
from std.python import PythonObject
from bindings.py_balanced import py_get
from bindings.arena_owner_door import arena_rows
from bindings.arena_resident_rows_door import arena_rows_store
from bindings.device_rows_door import DoorRows, door_is_device_rows
from bindings.device_rows_door_run import DoorRowsStore
from core.device_route import DEVICE_CPU


def door_rows_store_of(door: DoorRows, context: PythonObject) raises -> DoorRowsStore:
    """The resident store ids of an already resolved fold (the base and the
    indices put once per fold set by its model-selection owner)."""
    return arena_rows_store(door.view, context["door_rows_context"])


def door_rows_store(value: PythonObject, context: PythonObject) raises -> DoorRowsStore:
    """The resident store ids of a DeviceRows. `DoorRowsStore.none()` for a
    context without the rows connection (a door outside model selection),
    so the caller gathers on the host instead."""
    var rows_context = py_get(context, PythonObject("door_rows_context"))
    if rows_context is None or not door_is_device_rows(value, context):
        return DoorRowsStore.none()
    var view = arena_rows(value, rows_context["maxabs_rows_owner"], context["door_rows_class"])
    return arena_rows_store(view, rows_context)

