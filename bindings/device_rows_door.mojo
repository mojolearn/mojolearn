# SPDX-License-Identifier: Apache-2.0
# The DeviceRows entry of the shared estimator doors (lane ne1-b-door; plan
# docs/plans/NATIVE_ENTRY_POINTS_AFTER_0866.md workstream B, first release).
#
# Model selection hands a fold's X to an estimator as `_arena_io.DeviceRows`
# (base Array plus int64 row indices; the rows are gathered on the device out
# of the base model selection already holds resident). This module is the ONE
# place a door recognizes and resolves it (lesson 8: one helper per concern):
#
#   door_is_device_rows(value, context)  exact class check against the class
#                                        connected into the family context;
#                                        False for a context without it
#   DoorRows(value, context)             the resolution with no host words:
#                                        base, indices, row count, row width
#   (the door then reads the rows through `admit_x`, bindings/admitted_input_unpack.mojo:
#   the handle itself for an entry that gathers on the device (`device_x`), or
#   a parallel host gather out of the host base; never a download)
#
# A family whose kernels read a device store slot resolves a DeviceRows to a
# slot through the arena runner (`bindings/arena_prog_native.mojo` `_rows_gather`:
# the base put once through the resident scope, the rows gathered on the
# device by `<prefix>_dev_take_rows`); that stays the one device-slot path.
#
# The connected names (import-time glue, `_native_msel_dispatch.connect`, into
# every owner context model selection calls): `door_rows_class`
# (`_arena_io.DeviceRows`) and `door_rows_context` (`_arena_io._ROWS_CONTEXT`).
#
# Lane arena (2026-11-21): the fold view is the native `DeviceRowsState`
# (`bindings/arena_owner_door.mojo` `arena_rows`: one slot read and one lease
# per door call), not the instance dictionary. The fold's resident store ids
# (`door_rows_store`, `door_rows_store_of`) moved to
# `bindings/device_rows_store_door.mojo`, which reads the resident state
# natively; the call back through Python (`door_rows_call`, "store_ids") is
# gone. They live in their own module because the resident entry door imports
# this one through the classical input doors.
#
# Numerics: none. The rows a DeviceRows names are the rows model selection's
# host path gathers (the same device gather, the same order), so a fit or a
# score through this door gets the same bytes as through a host Array.
from std.python import Python, PythonObject
from bindings.py_balanced import py_type, py_get
from bindings.admitted_finite_wording import door_rows_refuse
from bindings.arena_owner_door import ArenaRows, arena_rows


def door_is_device_rows(value: PythonObject, context: PythonObject) raises -> Bool:
    """Whether `value` is exactly the connected `_arena_io.DeviceRows` class
    (a subclass is not: its methods could run Python)."""
    var cls = py_get(context, PythonObject("door_rows_class"))
    if cls is None:
        return False
    return py_type(value) is cls


struct DoorRows(Movable):
    """A DeviceRows resolved at a door without reading a host word of its
    rows: the base Array, the int64 index Array, the number of rows and the
    words per row."""
    var source: PythonObject
    var base: PythonObject
    var indices: PythonObject
    var rows: Int
    var row_words: Int
    #: The whole fold view (the resident store owner included), kept so the
    #: store ids are resolved without a second read of the instance.
    var view: ArenaRows

    def __init__(out self, value: PythonObject, context: PythonObject) raises:
        if not door_is_device_rows(value, context):
            raise Error("TypeError: a DeviceRows door requires the connected DeviceRows class")
        var view = arena_rows(value, context["door_rows_context"]["maxabs_rows_owner"], context["door_rows_class"])
        self.source = value
        self.base = view.base
        self.indices = view.indices
        self.rows = view.rows
        self.row_words = view.row_words
        self.view = view^


# TOMBSTONE: door_rows_host and door_x (the DeviceRows materialize for host-address
# kernels: device gather, download, host finite scan, kernel re-upload; REGRESSIONS
# R13) deleted 2026-11-08 by lane a5-admit-core-followup; code recoverable at f7d10d2b09.
# Restore: experiments/removed/A5_DOOR_ROWS_HOST.patch. admit_x / admit_rows_host
# (bindings/admitted_input_unpack.mojo) read a DeviceRows without the round trip.
