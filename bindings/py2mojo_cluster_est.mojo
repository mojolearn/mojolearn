# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""What `density.py` computed in Python after a DBSCAN fit (lane apple-fast-py2mojo-cluster, 2026-10-03), exported by both
`_mojolearn_estimators` and `_mojolearn_estimators_host` with one contract:

  dbscan_core_arrays(x_addr, labels_addr, core_addr, [n, d])
      -> [core_sample_indices array('i'), components array('f') (n_core x d,
         row-major), core labels array('i')], ascending training index, from
         the fit's own uint8 core mask (a byte other than 0 or 1 raises).
  estimators_py2mojo_cluster() -> 1, or 0 under
         `-D MOJOLEARN_PY2MOJO_cluster_OFF` (Python takes its old path).

It acts on the host arrays the fit already returned (its
outputs), so the two columns run the same loops."""
from std.memory import memcpy
from std.python import Python, PythonObject
from std.sys.compile import is_defined

comptime EST_PY2MOJO_CLUSTER = not is_defined["MOJOLEARN_PY2MOJO_cluster_OFF"]()


def estimators_py2mojo_cluster_binding() raises -> PythonObject:
    return PythonObject(1 if EST_PY2MOJO_CLUSTER else 0)


def _new_array(arr: PythonObject, code: StaticString, nbytes: Int) raises -> PythonObject:
    """An `array.array(code)` of `nbytes` zero bytes."""
    var zero = Python.import_module("builtins").bytes(nbytes)
    return arr.array(PythonObject(code), zero)


def _array_addr(a: PythonObject) raises -> Int:
    return Int(py=a.buffer_info()[0])


def dbscan_core_arrays_binding(
    x_addr: PythonObject, labels_addr: PythonObject, core_addr: PythonObject, params: PythonObject
) raises -> PythonObject:
    if len(params) != 2:
        raise Error("dbscan_core_arrays: params must be [n, d], got " + String(len(params)) + " values")
    var n = Int(py=params[0])
    var d = Int(py=params[1])
    var xp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=x_addr))
    var lp = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=labels_addr))
    var cp = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=core_addr))
    var n_core = 0
    for i in range(n):
        var f = cp.unsafe_load(i)
        if f > 1:
            raise Error("mojolearn DBSCAN: the fit's core mask holds a value other than 0 or 1")
        n_core += Int(f)
    var arr = Python.import_module("array")
    var idx = _new_array(arr, "i", 4 * n_core)
    var comp = _new_array(arr, "f", 4 * n_core * d)
    var lab = _new_array(arr, "i", 4 * n_core)
    if n_core > 0:
        var ip = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=_array_addr(idx))
        var op = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=_array_addr(comp))
        var olp = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=_array_addr(lab))
        var o = 0
        for i in range(n):
            if cp.unsafe_load(i) != 0:
                ip.unsafe_store(o, Int32(i))
                olp.unsafe_store(o, lp.unsafe_load(i))
                memcpy(dest=op + o * d, src=xp + i * d, count=d)
                o += 1
    var out = Python.list()
    out.append(idx)
    out.append(comp)
    out.append(lab)
    return out

