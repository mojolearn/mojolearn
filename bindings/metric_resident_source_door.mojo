# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""WHICH BUFFER a `resident()` scope keeps for an admitted metric input (lane
gd-resident-functions, 2026-11-25).

The metric doors admit each array argument into the dtype and layout their
kernels read. For a 1-D input the admission makes an operation-owned copy
even when the caller's buffer already has that dtype and layout (an int32
label vector is narrowed into a fresh int32 Array, a float32 vector copied
into a fresh float32 Array: providers/python/metrics_vector_input_pyapi_unpack.mojo
`metric_i32_vector`, `metric_f32_vector`). The copy has a new address on
every call, so a scope keyed by buffer identity never found it again: the
GPU verify of 2026-11-25 (verify --all on the L40S, group R) showed
`stats {}` for 52 metric lanes (the copy was marked a temporary and never
kept) and `stats {'x_prep': (4|8, 0)}` for the union-code cluster metrics
(the copy was kept, a new entry per call, never a hit).

`metric_caller_words(source, value)` answers whether the caller's argument
`source` holds exactly the words of the admitted `value`: both exact Arrays
or exact NumPy arrays, C-contiguous, of one typestr and element count, and
the same bytes now. Then the scope keeps the CALLER'S buffer (its address,
its object pinned by the cache), as it does when admission made no copy:
put once, read in place by every later call of the scope. The byte compare
runs only inside a scope, on an input the scope would otherwise upload
(n words on the host instead of n words host to device), and makes a
mismarked transformed copy (a door that pins a normalized or remapped array
as admitted from an argument) a temporary, never a stale device read.
Outside every `resident()` scope nothing here runs.
"""
from std.python import PythonObject
from std.memory import unsafe_memcmp
from bindings.py_balanced import py_type, py_get
from bindings.native_loaded_module_door import native_loaded_module
from bindings.classical_arrays_door import classical_address, classical_same_buffer


def _exact_kind(value: PythonObject) raises -> Int:
    """1: an exact mojolearn Array; 2: an exact NumPy ndarray; 0: anything
    else (a list, a subclass, a pandas or Torch object: never substituted)."""
    var modules = native_loaded_module("sys").modules
    var arrays = py_get(modules, PythonObject("mojolearn._array"))
    if arrays is not None and py_type(value) is arrays.Array:
        return 1
    var numpy = py_get(modules, PythonObject("numpy"))
    if numpy is not None and py_type(value) is numpy.ndarray:
        return 2
    return 0


def metric_exact_buffer(value: PythonObject) raises -> Bool:
    """Whether `value` is an exact mojolearn Array or an exact NumPy ndarray:
    a caller's buffer a `resident()` scope may key by identity. The resident
    label encoders took only Arrays (`classical_is_array`), so a NumPy label
    vector, the common input, never went to the device inside a scope."""
    return _exact_kind(value) != 0


def metric_exact_ndarray(value: PythonObject) raises -> Bool:
    """Whether `value` is an exact NumPy ndarray (its `view` is NumPy's own)."""
    return _exact_kind(value) == 2


def _typestr(value: PythonObject, kind: Int) raises -> String:
    if kind == 1:
        return String(value.dtype)
    return String(value.dtype.str)


def _c_order(value: PythonObject, kind: Int) raises -> Bool:
    if kind == 1:
        return String(value.order) == "C" or Int(py=value.ndim) < 2
    return Bool(py=value.flags.c_contiguous)


def _itemsize(value: PythonObject) raises -> Int:
    return Int(py=value.itemsize)


def metric_caller_words(source: PythonObject, value: PythonObject) raises -> Bool:
    """Whether the door argument `source` holds exactly the words of the
    array `value` admitted from it (see the module docstring). Asked inside
    a `resident()` scope only."""
    var ks = _exact_kind(source)
    if ks == 0:
        return False
    var kv = _exact_kind(value)
    if kv == 0:
        return False
    var n = Int(py=value.size)
    if n < 1 or Int(py=source.size) != n:
        return False
    if _typestr(source, ks) != _typestr(value, kv):
        return False
    if not _c_order(source, ks) or not _c_order(value, kv):
        return False
    var nbytes = n * _itemsize(value)
    var a = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(source)))
    var b = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(value)))
    var same = unsafe_memcmp(a, b, nbytes) == 0
    _ = source  # read by address above
    _ = value
    return same


def metric_resident_owner(source: PythonObject, value: PythonObject) raises -> PythonObject:
    """The buffer a scope keeps for `value`, admitted from the door argument
    `source`: `value` when it reads `source`'s own buffer (no copy), `source`
    when `value` is a word-for-word copy of it (`metric_caller_words`), else
    None (a converted copy: a per-call temporary, never kept)."""
    if classical_same_buffer(source, value):
        return value
    if metric_caller_words(source, value):
        return source
    return PythonObject(None)
