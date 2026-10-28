# SPDX-License-Identifier: Apache-2.0
# Exact sparse storage to the hierarchy's existing COO edge ABI. This never
# calls SciPy tocoo or a user connectivity callback. Duplicate entries remain
# separate: summing them while densifying could erase opposite-sign edges.
from std.memory import bitcast
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get
from bindings.native_loaded_module import native_loaded_module
from bindings.classical_arrays import classical_address
from bindings.classical_pandas import _pd_type_fields
from bindings.family_common import FAMILY_F32P, family_array, family_f32_snapshot
from bindings.host_sparse_native import host_sparse_format, _sparse_fields, _sparse_aliases, _sparse_scalar_word, _sparse_coordinate
from core.classical_pandas_impl import pd_raw_at
from core.cast_in import cast_f32_bits, cast_in_kind
from core.host_sparse_impl import host_sparse_index


struct HierarchyConnectivity(Movable):
    var owner: PythonObject
    var data: PythonObject
    var address: Int
    var count: Int
    var edges: Int
    var mode: Int

    def __init__(out self, owner: PythonObject, data: PythonObject, address: Int,
                 count: Int, edges: Int, mode: Int):
        self.owner = owner
        self.data = data
        self.address = address
        self.count = count
        self.edges = edges
        self.mode = mode


def _kind(dtype: PythonObject, context: PythonObject) raises -> Int:
    var types = py_call(native_loaded_module("builtins").list, context["numeric_dtypes"].values())
    var known = False
    for i in range(len(types)):  # small-loop(admitted native numeric dtype identities)
        known = known or py_type(dtype) is py_type(types[i])
    var aliases = _sparse_aliases(context)
    known = known or py_type(dtype) is py_type(aliases[0]) or py_type(dtype) is py_type(aliases[1])
    if not known:
        raise Error("TypeError: sparse connectivity requires native numeric storage")
    return cast_in_kind(Int(py=dtype.num), Int(py=dtype.itemsize), "connectivity")


struct _Storage(Movable):
    var keep: PythonObject
    var shape: List[Int]
    var strides: List[Int]
    var address: Int
    var width: Int
    var kind: Int
    var native: Bool

    def __init__(out self, value: PythonObject, rank: Int, indices: Bool, context: PythonObject) raises:
        if py_type(value) is not context["ndarray"]:
            raise Error("TypeError: sparse connectivity storage requires ordinary ndarrays")
        self.keep = value
        var shape = value.shape
        var strides = value.strides
        self.shape = List[Int]()
        self.strides = List[Int]()
        for axis in range(len(shape)):  # small-loop(captured sparse storage rank words)
            self.shape.append(Int(py=shape[axis]))
            self.strides.append(Int(py=strides[axis]))
        if len(self.shape) != rank:
            raise Error("sparse connectivity storage has the wrong rank")
        var dtype = value.dtype
        self.kind = _kind(dtype, context)
        self.width = Int(py=dtype.itemsize)
        self.native = Bool(py=dtype.isnative)
        self.address = Int(py=classical_address(value))
        if indices and (self.kind != 1 or (self.width != 4 and self.width != 8)):
            raise Error("TypeError: sparse connectivity indices require signed int32 or int64")

    def size(self, axis: Int = 0) raises -> Int:
        return self.shape[axis]

    def stride(self, axis: Int = 0) raises -> Int:
        return self.strides[axis]

    def index(self, at: Int) raises -> Int:
        return host_sparse_index(self.address, self.stride(), self.width, self.native, at)

    def value(self, offset: Int) -> Float32:
        return bitcast[DType.float32](cast_f32_bits(pd_raw_at(self.address, offset, self.width, self.native), self.kind, self.width))


def _edge(mut rows: List[Float32], mut cols: List[Float32], mut values: List[Float32],
          row: Int, col: Int, value: Float32, n: Int) raises:
    if row < 0 or col < 0 or row >= n or col >= n:
        raise Error("AgglomerativeClustering: a connectivity edge is outside [0, n)")
    rows.append(Float32(row))
    cols.append(Float32(col))
    values.append(value)


def hierarchy_connectivity(source: PythonObject, n: Int, context: PythonObject) raises -> HierarchyConnectivity:
    var b = native_loaded_module("builtins")
    var format = host_sparse_format(source)
    if format.byte_length() == 0:
        var snapshot = family_f32_snapshot(source, 2, "connectivity", context)
        if snapshot.shape[0] != n or snapshot.shape[1] != n:
            raise Error("Wrong shape for connectivity matrix when X has " + String(n) + " samples")
        return HierarchyConnectivity(Python.tuple(source, snapshot.owner), snapshot.keep, snapshot.source.addr, snapshot.count, snapshot.count, 1)
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, py_type(source))
    for i in range(len(ancestry)):  # driver-loop: refuse sparse Python finalizers before storage acquisition
        if py_get(_pd_type_fields(ancestry[i]), "__del__") is not None:
            raise Error("TypeError: sparse connectivity cannot use Python finalizers")
    var fields = _sparse_fields(source)
    var shape = fields["_shape"]
    if py_type(shape) is not b.tuple or len(shape) != 2 or py_type(shape[0]) is not b.int or py_type(shape[1]) is not b.int:
        raise Error("TypeError: sparse connectivity requires two native shape dimensions")
    if Int(py=shape[0]) != n or Int(py=shape[1]) != n:
        raise Error("Wrong shape for connectivity matrix when X has " + String(n) + " samples")
    var rows = List[Float32]()
    var cols = List[Float32]()
    var values = List[Float32]()
    if format == "csr" or format == "csc" or format == "bsr":
        var data = _Storage(fields["data"], 3 if format == "bsr" else 1, False, context)
        var indices = _Storage(fields["indices"], 1, True, context)
        var pointers = _Storage(fields["indptr"], 1, True, context)
        var br = data.size(1) if format == "bsr" else 1
        var bc = data.size(2) if format == "bsr" else 1
        if br < 1 or bc < 1 or n % br != 0 or n % bc != 0:
            raise Error("sparse connectivity blocks must evenly cover its shape")
        var outer = n // br
        var inner = n // bc
        if pointers.size() != outer + 1 or pointers.index(0) != 0:
            raise Error("sparse connectivity pointer count or origin is invalid")
        var count = min(data.size(), indices.size())
        var start = 0
        for major in range(outer):  # driver-loop: sparse graph's stored compressed rows or columns
            var end = pointers.index(major + 1)
            if end < start or end > count:
                raise Error("sparse connectivity pointers exceed stored data or are not ordered")
            for at in range(start, end):  # driver-loop: stored connectivity entries, preserving duplicates
                var minor = indices.index(at)
                if minor < 0 or minor >= inner:
                    raise Error("sparse connectivity index exceeds its shape")
                for r in range(br):  # driver-loop: native sparse block rows
                    for c in range(bc):  # driver-loop: native sparse block columns
                        var row = minor if format == "csc" else major * br + r
                        var col = major if format == "csc" else minor * bc + c
                        var offset = at * data.stride()
                        if format == "bsr":
                            offset += r * data.stride(1) + c * data.stride(2)
                        _edge(rows, cols, values, row, col, data.value(offset), n)
            start = end
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
        if row.size() != data.size() or col.size() != data.size():
            raise Error("sparse connectivity coordinate counts differ from stored data")
        for at in range(data.size()):  # driver-loop: stored COO entries without duplicate summation
            _edge(rows, cols, values, row.index(at), col.index(at), data.value(at * data.stride()), n)
        _ = row^
        _ = col^
        _ = data^
    elif format == "dia":
        var offsets = _Storage(fields["offsets"], 1, True, context)
        var data = _Storage(fields["data"], 2, False, context)
        if offsets.size() != data.size():
            raise Error("sparse connectivity diagonal count differs from stored data")
        for diagonal in range(offsets.size()):  # driver-loop: stored sparse diagonals
            var shift = offsets.index(diagonal)
            for col in range(data.size(1)):  # driver-loop: sparse diagonal values inside the declared square
                var row = col - shift
                if row >= 0 and row < n and col < n:
                    _edge(rows, cols, values, row, col, data.value(diagonal * data.stride() + col * data.stride(1)), n)
        _ = offsets^
        _ = data^
    elif format == "dok":
        var dictionary = fields["_dict"]
        if py_type(dictionary) is not b.dict:
            raise Error("TypeError: sparse DOK connectivity requires native dictionary storage")
        var dtype = fields["dtype"]
        var kind = _kind(dtype, context)
        var width = Int(py=dtype.itemsize)
        var pairs = py_call(b.list, dictionary.items())
        for at in range(len(pairs)):  # driver-loop: admitted native dictionary coordinate records
            var key = pairs[at][0]
            if py_type(key) is not b.tuple or len(key) != 2:
                raise Error("TypeError: sparse DOK coordinates require native pairs")
            var row = _sparse_coordinate(key[0], n, context)
            var col = _sparse_coordinate(key[1], n, context)
            var value = bitcast[DType.float32](cast_f32_bits(_sparse_scalar_word(pairs[at][1], kind, width, context), kind, width))
            _edge(rows, cols, values, row, col, value, n)
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
        for row in range(n):  # driver-loop: sparse LIL row storage
            var row_indices = indices[row]
            var row_data = data[row]
            if py_type(row_indices) is not b.list or py_type(row_data) is not b.list or len(row_indices) != len(row_data):
                raise Error("TypeError: sparse LIL rows require matching native lists")
            for at in range(len(row_data)):  # driver-loop: stored LIL entries without callback conversions
                var col = _sparse_coordinate(row_indices[at], n, context)
                var value = bitcast[DType.float32](cast_f32_bits(_sparse_scalar_word(row_data[at], kind, width, context), kind, width))
                _edge(rows, cols, values, row, col, value, n)
        _ = indices^
        _ = data^
    else:
        raise Error("TypeError: unsupported native sparse connectivity schema")
    var count = len(rows)
    var result = family_array(Python.tuple(3 * count), "<f4", context)
    var address = Int(py=classical_address(result))
    var output = FAMILY_F32P(unsafe_from_address=address)
    for at in range(count):  # driver-loop: pack the unchanged hierarchy COO ABI from native edge lists
        output[at] = rows[at]
        output[count + at] = cols[at]
        output[2 * count + at] = values[at]
    return HierarchyConnectivity(source, result, address, 3 * count, count, 2)
