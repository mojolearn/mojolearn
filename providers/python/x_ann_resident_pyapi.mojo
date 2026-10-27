# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python door of the device-resident ann indexes (`x_ann/resident.mojo`):
PythonObject arguments to typed values and back, nothing else. The address
and parameter lists are documented in `x_ann/resident.mojo`."""
from std.python import PythonObject
from providers.python.x_ann_abi_pyapi import py_int_list
from x_ann.resident import (
    x_ann_index_export_codes,
    x_ann_index_prepare,
    x_ann_index_release,
    x_ann_index_search,
)


def x_ann_index_prepare_binding(kind: PythonObject, addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    return PythonObject(x_ann_index_prepare(Int(py=kind), py_int_list(addrs), py_int_list(params)))


def x_ann_index_export_codes_binding(handle: PythonObject, address: PythonObject, count: PythonObject) raises -> PythonObject:
    """lane fg-ivf A7: write a resident IVF-PQ index's row-order codes into
    the caller's array (`x_ann_index_export_codes`)."""
    x_ann_index_export_codes(Int(py=handle), Int(py=address), Int(py=count))
    return PythonObject(0)


def x_ann_index_release_binding(handle: PythonObject) raises -> PythonObject:
    x_ann_index_release(Int(py=handle))
    return PythonObject(0)


def x_ann_index_search_binding(handle: PythonObject, addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    return PythonObject(x_ann_index_search(Int(py=handle), py_int_list(addrs), py_int_list(params)))
