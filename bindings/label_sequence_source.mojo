# SPDX-License-Identifier: Apache-2.0
"""Borrowed boundary label slots, shared by public and native pin sequences."""
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from bindings.cpython_borrowed import borrowed_item


def label_source_count[native: Bool](source: PythonObject, pins: List[PythonObject]) raises -> Int:
    comptime if native:
        return len(pins)
    else:
        return len(source)


def label_source_raw[native: Bool](source: PythonObject, pins: List[PythonObject],
                                   index: Int, is_tuple: Bool = False) raises -> PyObjectPtr:
    comptime if native:
        return pins[index]._obj_ptr
    else:
        return borrowed_item(source, index, is_tuple)


def label_source_value[native: Bool](source: PythonObject, pins: List[PythonObject],
                                     index: Int) raises -> PythonObject:
    return PythonObject(from_borrowed=label_source_raw[native](source, pins, index))
