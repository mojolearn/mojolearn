# SPDX-License-Identifier: Apache-2.0
"""Mixed native scalar class identity/order at the GP UNPACK/PACK boundary.

Values were admitted as exact builtin/NumPy immutable scalars by the caller.
Their existing C hash/equality and CPython list sort remain authoritative:
NumPy promotion and arbitrary-size integers cannot be replaced by Float64 keys.
Only the private dictionary/index containers become typed Mojo storage.
"""
from std.python import Python, PythonObject
from bindings.cpython_table import cpython_table
from bindings.cpython_borrowed import borrowed_item
from bindings.decomp_scalar_compare import decomp_equal_admitted


@fieldwise_init
struct GPScalarBucket(ImplicitlyCopyable, Movable):
    var first: Int
    var last: Int


struct GPScalarIndex(Movable):
    # The retained values are observable class identities, not algorithm state.
    # An insertion-only same-hash chain visits keys in insertion order, exactly
    # as the former CPython dict probe sequence does for equal hash values.
    var buckets: Dict[Int, GPScalarBucket]
    var values: List[PythonObject]
    var following: List[Int]
    var codes: List[Int]

    def __init__(out self):
        self.buckets = Dict[Int, GPScalarBucket]()
        self.values = List[PythonObject]()
        self.following = List[Int]()
        self.codes = List[Int]()

    def _find(self, value: PythonObject, hash_value: Int) raises -> Int:
        if hash_value not in self.buckets:
            return -1
        var at = self.buckets[hash_value].first
        while at >= 0:  # driver-loop: over admitted same-hash scalar identities
            if self.values[at] is value or decomp_equal_admitted(self.values[at], value):
                return at
            at = self.following[at]
        return -1

    def put(mut self, value: PythonObject, code: Int) raises:
        var hash_value = gp_scalar_hash(value)
        var at = self._find(value, hash_value)
        if at >= 0:
            # dict(zip(classes, range(...))) keeps its first key object but
            # replaces its value when a later sorted key compares equal.
            self.codes[at] = code
            return
        var row = len(self.values)
        self.values.append(value)
        self.following.append(-1)
        self.codes.append(code)
        if hash_value in self.buckets:
            var bucket = self.buckets[hash_value]
            self.following[bucket.last] = row
            bucket.last = row
            self.buckets[hash_value] = bucket
        else:
            self.buckets[hash_value] = GPScalarBucket(row, row)

    def code(self, value: PythonObject) raises -> Int:
        var at = self._find(value, gp_scalar_hash(value))
        return -1 if at < 0 else self.codes[at]


def gp_scalar_hash(value: PythonObject) raises -> Int:
    var result = cpython_table()[].PyObject_Hash(value._obj_ptr)
    _ = value
    if result == -1:
        ref cpy = Python().cpython()
        raise cpy.unsafe_get_error()
    return result


def gp_mixed_classes(labels: PythonObject) raises -> PythonObject:
    var first = GPScalarIndex()
    for i in range(len(labels)):  # driver-loop: over admitted scalar identity pins
        first.put(PythonObject(from_borrowed=borrowed_item(labels, i, False)), i)
    ref cpy = Python().cpython()
    var api = cpython_table()
    var raw = api[].PyList_New(len(first.values))
    if not raw:
        raise cpy.unsafe_get_error()
    var classes = PythonObject(from_owned=raw)
    for i in range(len(first.values)):  # driver-loop: over public class slots
        if api[].PyList_SetItem(classes._obj_ptr, i, cpy.Py_NewRef(first.values[i]._obj_ptr)) < 0:
            raise cpy.unsafe_get_error()
    # Sort the final public class list with the exact existing algorithm. No
    # temporary sorted list, Python callable lookup or replacement order exists.
    if api[].PyList_Sort(classes._obj_ptr) < 0:
        raise cpy.unsafe_get_error()
    _ = first
    _ = labels
    return classes


def gp_mixed_index(classes: PythonObject) raises -> GPScalarIndex:
    var result = GPScalarIndex()
    for i in range(len(classes)):  # driver-loop: over sorted public class identities
        result.put(PythonObject(from_borrowed=borrowed_item(classes, i, False)), i)
    _ = classes
    return result^


def gp_mixed_codes(index: GPScalarIndex, labels: PythonObject, address: Int) raises:
    var dst = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)
    # Re-query the sorted index instead of remapping the first deduplication:
    # mixed NumPy scalar equality need not be transitive under promotion.
    for i in range(len(labels)):  # driver-loop: over admitted scalar identity pins
        dst[i] = Int32(index.code(PythonObject(from_borrowed=borrowed_item(labels, i, False))))
    _ = index
    _ = labels
