# SPDX-License-Identifier: Apache-2.0
# Exact sparse storage to the hierarchy's existing COO edge ABI. This never
# calls SciPy tocoo or a user connectivity callback. Duplicate entries remain
# separate: summing them while densifying could erase opposite-sign edges.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased, PyObjectPtr
from bindings.cpython_table import cpython_table
from bindings.admitted_family import admit_family_meta
from bindings.py_balanced import py_call, py_type, py_get
from bindings.native_loaded_module import native_loaded_module
from bindings.classical_arrays import classical_address
from bindings.classical_pandas import _pd_type_fields
from bindings.host_sparse_native import host_sparse_format, _sparse_fields, _sparse_aliases, _SparseKinds, _sparse_scalar_word_k, _sparse_coordinate_k
from core.cast_in import cast_in_kind
from core.hierarchy_connectivity_plan import (
    ConnectivityStorage, ConnectivityEdge, HierarchyConnectivityPlan,
    connectivity_edge, connectivity_pack, connectivity_compressed, connectivity_coo, connectivity_dia,
)


struct HierarchyConnectivity(Movable):
    var owner: PythonObject
    var data: PythonObject
    var snapshot_owner: PythonObject
    var native: HierarchyConnectivityPlan
    var address: Int
    var count: Int
    var edges: Int
    var mode: Int

    def __init__(out self, owner: PythonObject, data: PythonObject, address: Int,
                 count: Int, edges: Int, mode: Int):
        self.owner = owner
        self.data = data
        self.snapshot_owner = PythonObject(None)
        self.native = HierarchyConnectivityPlan()
        self.address = address
        self.count = count
        self.edges = edges
        self.mode = mode


def _kind(dtype: PythonObject, context: PythonObject) raises -> Int:
    var b = native_loaded_module("builtins")
    var types = context["numeric_dtypes"]
    if py_type(types) is not b.dict:
        raise Error("TypeError: sparse connectivity dtype metadata requires a native dictionary")
    var known = False
    var cursor = InlineArray[Int, 1](fill=0)
    var pair = InlineArray[PyObjectPtr, 2](fill=PyObjectPtr())
    var api = cpython_table()
    while api[].PyDict_Next(types._obj_ptr, Int(cursor.unsafe_ptr()), Int(pair.unsafe_ptr()), Int(pair.unsafe_ptr() + 1)):
        # Boundary metadata: no dtype list allocation or key hashing.
        known = known or py_type(dtype) is py_type(PythonObject(from_borrowed=pair[1]))
    _ = types
    var aliases = _sparse_aliases(context)
    known = known or py_type(dtype) is py_type(aliases[0]) or py_type(dtype) is py_type(aliases[1])
    if not known:
        raise Error("TypeError: sparse connectivity requires native numeric storage")
    return cast_in_kind(Int(py=dtype.num), Int(py=dtype.itemsize), "connectivity")


struct _Storage(Movable):
    # The boundary alone owns Python storage. RUN receives only this typed
    # descriptor; each caller names keep after the released-GIL conversion.
    var keep: PythonObject
    var view: ConnectivityStorage

    def __init__(out self, value: PythonObject, rank: Int, indices: Bool, context: PythonObject) raises:
        if py_type(value) is not context["ndarray"]:
            raise Error("TypeError: sparse connectivity storage requires ordinary ndarrays")
        self.keep = value
        var shape = value.shape
        var strides = value.strides
        var sizes = List[Int]()
        var steps = List[Int]()
        for axis in range(len(shape)):  # boundary metadata: captured sparse storage rank words
            sizes.append(Int(py=shape[axis]))
            steps.append(Int(py=strides[axis]))
        if len(sizes) != rank:
            raise Error("sparse connectivity storage has the wrong rank")
        var dtype = value.dtype
        var kind = _kind(dtype, context)
        var width = Int(py=dtype.itemsize)
        var native = Bool(py=dtype.isnative)
        var address = Int(py=classical_address(value))
        if indices and (kind != 1 or (width != 4 and width != 8)):
            raise Error("TypeError: sparse connectivity indices require signed int32 or int64")
        self.view = ConnectivityStorage(address, sizes[0], sizes[1] if rank > 1 else 1,
            sizes[2] if rank > 2 else 1, steps[0], steps[1] if rank > 1 else 0,
            steps[2] if rank > 2 else 0, width, kind, native)


def _object_row(address: Int, stride: Int, row: Int) -> PythonObject:
    # Exact object ndarrays are admitted before this native storage read.
    var item = Pointer[PyObjectPtr, ImmUntrackedOrigin](unsafe_from_address=address + row * stride)[0]
    return PythonObject(from_borrowed=item) if item else PythonObject(None)


def hierarchy_connectivity(source: PythonObject, n: Int, context: PythonObject) raises -> HierarchyConnectivity:
    var b = native_loaded_module("builtins")
    var format = host_sparse_format(source)
    if format.byte_length() == 0:
        var snapshot = admit_family_meta(source, 2, "connectivity", context)
        if snapshot.shape[0] != n or snapshot.shape[1] != n:
            raise Error("Wrong shape for connectivity matrix when X has " + String(n) + " samples")
        var dense = HierarchyConnectivity(source, snapshot.keep, snapshot.source.addr, snapshot.count, snapshot.count, 1)
        dense.snapshot_owner = snapshot.owner
        return dense^
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, py_type(source))
    for i in range(len(ancestry)):  # boundary admission: refuse finalizers before storage acquisition
        if py_get(_pd_type_fields(ancestry[i]), "__del__") is not None:
            raise Error("TypeError: sparse connectivity cannot use Python finalizers")
    var fields = _sparse_fields(source)
    var shape = fields["_shape"]
    if py_type(shape) is not b.tuple or len(shape) != 2 or py_type(shape[0]) is not b.int or py_type(shape[1]) is not b.int:
        raise Error("TypeError: sparse connectivity requires two native shape dimensions")
    if Int(py=shape[0]) != n or Int(py=shape[1]) != n:
        raise Error("Wrong shape for connectivity matrix when X has " + String(n) + " samples")
    var plan = HierarchyConnectivityPlan()
    if format == "csr" or format == "csc" or format == "bsr":
        var data = _Storage(fields["data"], 3 if format == "bsr" else 1, False, context)
        var indices = _Storage(fields["indices"], 1, True, context)
        var pointers = _Storage(fields["indptr"], 1, True, context)
        with GILReleased(Python()):
            plan = connectivity_compressed(data.view, indices.view, pointers.view, n, format == "csc", format == "bsr")
        _ = data^
        _ = indices^
        _ = pointers^
    elif format == "coo":
        var coords = fields["coords"]
        if py_type(coords) is not b.tuple or len(coords) != 2:
            raise Error("TypeError: sparse connectivity coordinates require two native arrays")
        var row = _Storage(coords[0], 1, True, context)
        var col = _Storage(coords[1], 1, True, context)
        var data = _Storage(fields["data"], 1, False, context)
        with GILReleased(Python()):
            plan = connectivity_coo(row.view, col.view, data.view, n)
        _ = row^
        _ = col^
        _ = data^
    elif format == "dia":
        var offsets = _Storage(fields["offsets"], 1, True, context)
        var data = _Storage(fields["data"], 2, False, context)
        with GILReleased(Python()):
            plan = connectivity_dia(offsets.view, data.view, n)
        _ = offsets^
        _ = data^
    elif format == "dok":
        var dictionary = fields["_dict"]
        if py_type(dictionary) is not b.dict:
            raise Error("TypeError: sparse DOK connectivity requires native dictionary storage")
        var dtype = fields["dtype"]
        var kind = _kind(dtype, context)
        var width = Int(py=dtype.itemsize)
        var records = List[ConnectivityEdge]()
        var kinds = _SparseKinds(context)
        var api = cpython_table()
        var cursor = InlineArray[Int, 1](fill=0)
        var pair = InlineArray[PyObjectPtr, 2](fill=PyObjectPtr())
        # Inherently serial boundary admission under the GIL: native dictionary
        # storage order and per-item key/value refusals must remain ordered.
        # No Python items/list envelope or coordinate/value protocol executes.
        while api[].PyDict_Next(dictionary._obj_ptr, Int(cursor.unsafe_ptr()), Int(pair.unsafe_ptr()), Int(pair.unsafe_ptr() + 1)):
            var key = PythonObject(from_borrowed=pair[0])
            if py_type(key) is not b.tuple or len(key) != 2:
                raise Error("TypeError: sparse DOK coordinates require native pairs")
            var row = _sparse_coordinate_k(PythonObject(from_borrowed=api[].PyTuple_GetItem(key._obj_ptr, 0)), n, kinds)
            var col = _sparse_coordinate_k(PythonObject(from_borrowed=api[].PyTuple_GetItem(key._obj_ptr, 1)), n, kinds)
            var word = _sparse_scalar_word_k(PythonObject(from_borrowed=pair[1]), kind, width, kinds)
            records.append(connectivity_edge(row, col, word, kind, width, n))
        with GILReleased(Python()):
            plan = connectivity_pack(records)
        _ = records^
        _ = dictionary
    elif format == "lil":
        var dtype = fields["dtype"]
        var kind = _kind(dtype, context)
        var width = Int(py=dtype.itemsize)
        var indices = fields["rows"]
        var data = fields["data"]
        if py_type(indices) is not context["ndarray"] or py_type(data) is not context["ndarray"]:
            raise Error("TypeError: sparse LIL connectivity requires ordinary object ndarrays")
        if Int(py=indices.ndim) != 1 or Int(py=data.ndim) != 1 or Int(py=indices.size) != n or Int(py=data.size) != n or String(indices.dtype.kind) != "O" or String(data.dtype.kind) != "O":
            raise Error("sparse LIL connectivity requires one native object row per sample")
        var indices_address = Int(py=classical_address(indices))
        var data_address = Int(py=classical_address(data))
        var indices_stride = Int(py=indices.strides[0])
        var data_stride = Int(py=data.strides[0])
        var kinds = _SparseKinds(context)
        var api = cpython_table()
        var records = List[ConnectivityEdge]()
        # Inherently serial CPython list/object-array unpack. Preserve the
        # row/item admission order, including scalar conversion refusals;
        # native COO packing below is independent and parallel.
        for row in range(n):
            var row_indices = _object_row(indices_address, indices_stride, row)
            var row_data = _object_row(data_address, data_stride, row)
            if py_type(row_indices) is not b.list or py_type(row_data) is not b.list or len(row_indices) != len(row_data):
                raise Error("TypeError: sparse LIL rows require matching native lists")
            for at in range(len(row_data)):
                var col = _sparse_coordinate_k(PythonObject(from_borrowed=api[].PyList_GetItem(row_indices._obj_ptr, at)), n, kinds)
                var word = _sparse_scalar_word_k(PythonObject(from_borrowed=api[].PyList_GetItem(row_data._obj_ptr, at)), kind, width, kinds)
                records.append(connectivity_edge(row, col, word, kind, width, n))
        with GILReleased(Python()):
            plan = connectivity_pack(records)
        _ = records^
        _ = indices
        _ = data
    else:
        raise Error("TypeError: unsupported native sparse connectivity schema")
    var result = HierarchyConnectivity(source, PythonObject(None), plan.address(), 3 * plan.edges, plan.edges, 2)
    result.native = plan^
    return result^
