# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed lists to Python lists without a per-element loop in a shell.

Each helper copies the typed values into a fresh numpy array with one
memcpy and returns `array.tolist()`. A Python float made from a float32 or
float64 value, and a Python int made from an integer value, are the same
objects `PythonObject(value)` made one element at a time, so a caller that
built `Python.list()` by `append` gets equal lists. No arithmetic.
"""
from std.memory import memcpy
from std.python import Python, PythonObject


def _data_address(array: PythonObject) raises -> Int:
    return Int(py=array.__array_interface__["data"][0])


def py_list_f64(values: List[Float64]) raises -> PythonObject:
    var n = len(values)
    var array = Python.import_module("numpy").empty(n, dtype="float64")
    if n > 0:
        memcpy(
            dest=MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=_data_address(array)),
            src=values.unsafe_ptr(), count=n,
        )
    _ = len(values)
    return array.tolist()


def py_list_f32(values: List[Float32]) raises -> PythonObject:
    var n = len(values)
    var array = Python.import_module("numpy").empty(n, dtype="float32")
    if n > 0:
        memcpy(
            dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=_data_address(array)),
            src=values.unsafe_ptr(), count=n,
        )
    _ = len(values)
    return array.tolist()


def py_list_i32(values: List[Int32]) raises -> PythonObject:
    var n = len(values)
    var array = Python.import_module("numpy").empty(n, dtype="int32")
    if n > 0:
        memcpy(
            dest=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=_data_address(array)),
            src=values.unsafe_ptr(), count=n,
        )
    _ = len(values)
    return array.tolist()


def py_list_i64(values: List[Int64]) raises -> PythonObject:
    var n = len(values)
    var array = Python.import_module("numpy").empty(n, dtype="int64")
    if n > 0:
        memcpy(
            dest=MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=_data_address(array)),
            src=values.unsafe_ptr(), count=n,
        )
    _ = len(values)
    return array.tolist()


def py_list_u32(values: List[UInt32]) raises -> PythonObject:
    var n = len(values)
    var array = Python.import_module("numpy").empty(n, dtype="uint32")
    if n > 0:
        memcpy(
            dest=MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=_data_address(array)),
            src=values.unsafe_ptr(), count=n,
        )
    _ = len(values)
    return array.tolist()


def py_list_u64(values: List[UInt64]) raises -> PythonObject:
    var n = len(values)
    var array = Python.import_module("numpy").empty(n, dtype="uint64")
    if n > 0:
        memcpy(
            dest=MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=_data_address(array)),
            src=values.unsafe_ptr(), count=n,
        )
    _ = len(values)
    return array.tolist()


def py_list_int(values: List[Int]) raises -> PythonObject:
    """`Int` is 64 bits on every target this package builds for."""
    var n = len(values)
    var array = Python.import_module("numpy").empty(n, dtype="int64")
    if n > 0:
        memcpy(
            dest=MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=_data_address(array)),
            src=values.unsafe_ptr(), count=n,
        )
    _ = len(values)
    return array.tolist()
