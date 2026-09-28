# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""What every cluster-lane entry returns, and the Python boundary both
bindings share (lane/algos-cluster). Host code; no device."""
from std.memory import memcpy
from std.python import Python, PythonObject


struct ClusterOut(Movable):
    """A fit's answer: float arrays, int arrays and float64 scalars, in the
    order the entry documents. `to_py` is `[[f...], [i...], [s...]]`: each
    float array an `array.array('f')`, each int array an `array.array('i')`
    (one memcpy each), the scalars a list of Python floats."""

    var f: List[List[Float32]]
    var i: List[List[Int32]]
    var s: List[Float64]

    def __init__(out self):
        self.f = List[List[Float32]]()
        self.i = List[List[Int32]]()
        self.s = List[Float64]()

    def to_py(self) raises -> PythonObject:
        var arr = Python.import_module("array")
        var fl = Python.list()
        for a in self.f:
            fl.append(_py_array(arr, "f", a.unsafe_ptr().bitcast[UInt8](), len(a) * 4))
        var il = Python.list()
        for a in self.i:
            il.append(_py_array(arr, "i", a.unsafe_ptr().bitcast[UInt8](), len(a) * 4))
        var sl = Python.list()
        for v in self.s:
            sl.append(PythonObject(v))
        var out = Python.list()
        out.append(fl)
        out.append(il)
        out.append(sl)
        return out


def _py_array(arr: PythonObject, code: StaticString, src: Pointer[UInt8, _], nbytes: Int) raises -> PythonObject:
    """An `array.array(code)` holding `nbytes` bytes copied from `src` in ONE
    memcpy. The Python side reads it exactly as it read the list it replaces
    (indexing, `len`, iteration, `list()`, and `array.array(code, it)`,
    which copies an array of the same code as bytes): an 'f' item is the
    float32 itself, widened exactly on read, an 'i' item the int32. The
    element-by-element list it replaces cost seconds on an n^2 output
    (AffinityPropagation's affinity_matrix_: 25M PythonObject appends)."""
    var zero = Python.import_module("builtins").bytes(nbytes)
    var out = arr.array(PythonObject(code), zero)
    if nbytes > 0:
        var addr = Int(py=out.buffer_info()[0])
        memcpy(dest=MutPointer[UInt8, MutAnyOrigin](unsafe_from_address=addr), src=src, count=nbytes)
    return out


def py_ints(obj: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    for t in range(len(obj)):
        out.append(Int(py=obj[t]))
    return out^


def py_floats(obj: PythonObject) raises -> List[Float64]:
    var out = List[Float64]()
    for t in range(len(obj)):
        out.append(Float64(py=obj[t]))
    return out^
