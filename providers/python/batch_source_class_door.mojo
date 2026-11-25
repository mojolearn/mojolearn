# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The connected `mojolearn.ops.BatchSource` class (lane stream-fits, 2026-11-25; moved out of
providers/python/batch_source_pyapi_door.mojo, unchanged): the one shared slot `import mojolearn` fills with the
exact class ("connect"), and the identity test of a handle against it. A module of its own, with no import beyond
the slot, so the argument admission of other public doors (bindings/sequential_admission_door.mojo: Sequential.fit
takes `source=`) can admit a handle by exact type without compiling the batch source."""
from std.python import PythonObject
from bindings.py_balanced import py_type
from core.shared_slot import SharedSlot


struct BatchSourceClassDoor(Defaultable, Movable):
    var ready: Bool
    var cls: PythonObject

    def __init__(out self):
        self.ready = False
        self.cls = PythonObject(None)


comptime BATCH_SOURCE_CLASS_DOOR = SharedSlot[BatchSourceClassDoor, "MojolearnBatchSourceDoorV1"]


def batch_source_class_connect(cls: PythonObject):
    """Record the exact connected class (import time)."""
    var door = BATCH_SOURCE_CLASS_DOOR.get_or_create_ptr()
    door[].cls = cls
    door[].ready = True


def batch_source_class_ready() -> Bool:
    return BATCH_SOURCE_CLASS_DOOR.get_or_create_ptr()[].ready


def batch_source_is_handle(value: PythonObject) raises -> Bool:
    """Whether `value` is an instance of exactly the connected BatchSource class (no subclass, no lookalike)."""
    var door = BATCH_SOURCE_CLASS_DOOR.get_or_create_ptr()
    return door[].ready and py_type(value) is door[].cls
