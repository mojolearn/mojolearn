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
#   (the door then reads the rows through `admit_x`, bindings/admitted_input.mojo:
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
# The `door_rows_call` connection went with door_rows_host (removed from the
# glue by integ/pre-0867-w2: no door reads it).
#
# Numerics: none. The rows a DeviceRows names are the rows model selection's
# host path gathers (the same device gather, the same order), so a fit or a
# score through this door gets the same bytes as through a host Array.
from std.python import PythonObject
from bindings.py_balanced import py_call, py_type, py_get


def door_is_device_rows(value: PythonObject, context: PythonObject) raises -> Bool:
    """Whether `value` is exactly the connected `_arena_io.DeviceRows` class
    (a subclass is not: its methods could run Python)."""
    var cls = py_get(context, PythonObject("door_rows_class"))
    if cls is None:
        return False
    return py_type(value) is cls


def _door_rows_fields(value: PythonObject, context: PythonObject) raises -> PythonObject:
    # The instance dictionary through the class's own `__dict__` descriptor,
    # as the DeviceRows owner reads it (bindings/arena_rows_native.mojo); the
    # exact class check above means no attribute hook can run.
    var rows_context = context["door_rows_context"]
    var fields = py_call(rows_context["maxabs_rows_dict"].__get__, value, context["door_rows_class"])
    if py_type(fields) is not rows_context["arena_native_types"]["dict"]:
        raise Error("TypeError: DeviceRows requires a native metadata dictionary")
    return fields


struct DoorRows(Movable):
    """A DeviceRows resolved at a door without reading a host word of its
    rows: the base Array, the int64 index Array, the number of rows and the
    words per row."""
    var source: PythonObject
    var base: PythonObject
    var indices: PythonObject
    var rows: Int
    var row_words: Int

    def __init__(out self, value: PythonObject, context: PythonObject) raises:
        if not door_is_device_rows(value, context):
            raise Error("TypeError: a DeviceRows door requires the connected DeviceRows class")
        var fields = _door_rows_fields(value, context)
        self.source = value
        self.base = fields["base"]
        self.indices = fields["indices"]
        self.rows = Int(py=self.indices.size)
        self.row_words = Int(py=fields["row_words"])


# TOMBSTONE: door_rows_host and door_x (the DeviceRows materialize for host-address
# kernels: device gather, download, host finite scan, kernel re-upload; REGRESSIONS
# R13) deleted 2026-11-08 by lane a5-admit-core-followup; code recoverable at f7d10d2b09.
# Restore: experiments/removed/A5_DOOR_ROWS_HOST.patch. admit_x / admit_rows_host
# (bindings/admitted_input.mojo) read a DeviceRows without the round trip.
