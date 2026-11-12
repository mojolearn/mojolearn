# SPDX-License-Identifier: Apache-2.0
"""UNPACK of the table keys the ANN, TSNE and PCA-start owners need (lane
calltable-cn-ann, 2026-11-11): the `(mode, vendor)` an image recorded as its
module attribute when it published its table. One exact dictionary read of
the module's `__dict__` per call; the entries are then called natively
(core/neighbors_entry_table_ann.mojo, core/decomp_entry_table_pca.mojo).
Vendor 0 is the CPU column."""
from std.python import PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_type, py_get, py_fields
from core.neighbors_entry_table import NeighborsEntryKey


def _module_key(binding: PythonObject, attribute: String, what: String) raises -> NeighborsEntryKey:
    var b = native_loaded_module("builtins")
    if py_type(binding) is not py_type(b):
        raise Error("TypeError: " + what + " execution requires its connected native module")
    var key = py_get(py_fields(binding), attribute)
    if (key is None or py_type(key) is not b.tuple or len(key) != 2
            or py_type(key[0]) is not b.int or py_type(key[1]) is not b.int):
        raise Error("ImportError: mojolearn: the " + what + " binding publishes no native " + attribute
                    + " table key; rebuild it")
    return NeighborsEntryKey(Int(py=key[0]), Int(py=key[1]))


def ann_ops_key(binding: PythonObject) raises -> NeighborsEntryKey:
    """The ANN operations table key of `_mojolearn_x_ann` /
    `_mojolearn_x_ann_host` (`x_ann_native_key`)."""
    return _module_key(binding, "x_ann_native_key", "x_ann")


def decomp_pca_ops_key(binding: PythonObject) raises -> NeighborsEntryKey:
    """The PCA owned-fit table key of `_mojolearn_estimators` /
    `_mojolearn_estimators_host` (`decomp_pca_native_key`)."""
    return _module_key(binding, "decomp_pca_native_key", "estimators")
