# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""PACK of a fitted UMAP graph as a CSR tuple `(indptr, indices, data)` of
mojolearn Arrays (int64, int32, float32): `UMAP.graph_`
(bindings/mani_umap_native_door.mojo) and `mojolearn.neighbors_graph`'s
connectivities (bindings/neighbors_ann_native_door.mojo), lane
sc-ann-1-umap, 2026-11-28. The native writer (umap/graph_out.mojo
`umap_publish_graph`) fills the arrays these stores hold; the edge count is
known only after it, so the edge stores are sized by the bound
(`umap_graph_capacity`) and trimmed in place to `indptr[n]` (an array.array
slice delete: no copy of the kept words by this code). No arithmetic on
values: a size read and the slot writes of the Arrays."""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call
from bindings.py_attr import py_attr
from bindings.native_loaded_module_door import native_loaded_module
from bindings.native_array_storage_door import native_array_unfilled
from bindings.classical_arrays_door import classical_wrap_array
from core.native_array_layout import native_array_code


def umap_graph_stores(n: Int, cap: Int) raises -> PythonObject:
    """Unfilled stores `(indptr, indices, data)` of n + 1 int64 and `cap`
    int32 / float32 items (one native writer fills what is kept)."""
    return Python.tuple(native_array_unfilled(native_array_code("<i8"), n + 1),
                        native_array_unfilled(native_array_code("<i4"), cap),
                        native_array_unfilled(native_array_code("<f4"), cap))


def umap_store_address(store: PythonObject) raises -> Int:
    """The item block's address of an array.array store."""
    return Int(py=py_call(store.buffer_info)[0])


def umap_graph_pack(stores: PythonObject, n: Int, cap: Int, context: PythonObject) raises -> PythonObject:
    """The CSR tuple of the written stores (module docstring)."""
    var indptr_store = stores[0]
    var indices_store = stores[1]
    var data_store = stores[2]
    var at = umap_store_address(indptr_store)
    var nnz = Int(MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=at)[n])
    if nnz < 0 or nnz > cap:
        raise Error("RuntimeError: UMAP fitted graph exceeds its output capacity")
    if nnz < cap:
        var b = native_loaded_module("builtins")
        var tail = py_call(b.slice, PythonObject(nnz), PythonObject(None))
        _ = py_call(py_attr(indices_store, "__delitem__"), tail)
        _ = py_call(py_attr(data_store, "__delitem__"), tail)
    var indptr = classical_wrap_array(indptr_store, Python.tuple(n + 1), "<i8", "C", context)
    var indices = classical_wrap_array(indices_store, Python.tuple(nnz), "<i4", "C", context)
    var data = classical_wrap_array(data_store, Python.tuple(nnz), "<f4", "C", context)
    return Python.tuple(indptr, indices, data)
