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
#   door_rows_host(value, context)       the host Array of the rows, for a door
#                                        whose kernels read host addresses: the
#                                        DeviceRows' own native materializer,
#                                        which gathers on the device out of the
#                                        resident base (model selection's take)
#                                        and keeps the result on the object, so
#                                        it runs at most once per object
#   door_x(value, context)               the value itself, or door_rows_host of
#                                        a DeviceRows: what the classical and
#                                        family-lifecycle X checks continue with
#
# A family whose kernels read a device store slot resolves a DeviceRows to a
# slot through the arena runner (`bindings/arena_prog_native.mojo` `_rows_gather`:
# the base put once through the resident scope, the rows gathered on the
# device by `<prefix>_dev_take_rows`); that stays the one device-slot path.
#
# The connected names (import-time glue, `_native_msel_dispatch.connect`, into
# every owner context model selection calls): `door_rows_class`
# (`_arena_io.DeviceRows`), `door_rows_call` (`_arena_io._ROWS_CALL`, the
# compiled DeviceRows owner) and `door_rows_context` (`_arena_io._ROWS_CONTEXT`).
#
# Numerics: none. The rows a DeviceRows names are the rows model selection's
# host path gathers (the same device gather, the same order), so a fit or a
# score through this door gets the same bytes as through a host Array.
from std.python import Python, PythonObject
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


def door_rows_host(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """The rows of a DeviceRows as a host Array, through its compiled owner's
    `materialize` (the device gather out of model selection's resident base;
    kept on the object, so a second call returns the same Array)."""
    var call = context["door_rows_call"]
    return py_call(call, value, PythonObject("materialize"), Python.tuple(), context["door_rows_context"])


def door_x(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`value`, or the host rows of a DeviceRows (`door_rows_host`)."""
    if door_is_device_rows(value, context):
        return door_rows_host(value, context)
    return value
