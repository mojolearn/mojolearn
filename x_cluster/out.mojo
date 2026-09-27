# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""What every cluster-lane entry returns, and the Python boundary both
bindings share (lane/algos-cluster). Host code; no device."""
from std.python import Python, PythonObject


struct ClusterOut(Movable):
    """A fit's answer: float arrays, int arrays and float64 scalars, in the
    order the entry documents. `to_py` is `[[f...], [i...], [s...]]` of
    Python lists (float32 values widen exactly to Python floats)."""

    var f: List[List[Float32]]
    var i: List[List[Int32]]
    var s: List[Float64]

    def __init__(out self):
        self.f = List[List[Float32]]()
        self.i = List[List[Int32]]()
        self.s = List[Float64]()

    def to_py(self) raises -> PythonObject:
        var fl = Python.list()
        for a in self.f:
            var l = Python.list()
            for v in a:
                l.append(PythonObject(Float64(v)))
            fl.append(l)
        var il = Python.list()
        for a in self.i:
            var l = Python.list()
            for v in a:
                l.append(PythonObject(Int(v)))
            il.append(l)
        var sl = Python.list()
        for v in self.s:
            sl.append(PythonObject(v))
        var out = Python.list()
        out.append(fl)
        out.append(il)
        out.append(sl)
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
