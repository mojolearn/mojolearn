# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/mds_iso_dev.mojo: the PythonObject <-> typed-value
doors that `bindings/_mojolearn_x_decomp.mojo` registers under the same
names. Each door unpacks resident matrix ids and integers and calls one
typed entry in x_decomp/mds_iso_dev.mojo; no data work happens here."""
from std.python import Python, PythonObject

from x_decomp.mds_iso_dev import mds_disp_dev, mds_setup_dev
from x_decomp.resident_pyapi import _id, _n


def dev_mds_setup_py(dis: PythonObject, ids: PythonObject, p: PythonObject) raises -> PythonObject:
    """ids = [keys, idx, gid, tmp, gst, word] device matrices (keys, idx,
    gid, tmp: N2 values; gst: N2 + 1; word: 1); p = [n, N2]. Returns (m, G).
    gid always ends in ids[2] (copied from tmp when the scan ended there)."""
    var li = List[Int]()
    for i in range(6):  # driver-loop: over the 6 resident matrix ids of ids
        li.append(_id(ids[i]))
    var r = mds_setup_dev(_id(dis), li, _n(p, 0), _n(p, 1))
    return Python.tuple(r[0], r[1])


def dev_mds_disp_py(d: PythonObject, dst: PythonObject, ids: PythonObject, p: PythonObject) raises -> PythonObject:
    """One iteration's upper-triangle disparities into dst (n x n), enqueued.
    ids = [keys, idx, gid, gst, sm, wt, end, prv, last, hf, hd0, hd1, gv]
    device matrices; p = [n, m, G, first]."""
    var li = List[Int]()
    for i in range(13):  # driver-loop: over the 13 resident matrix ids of ids
        li.append(_id(ids[i]))
    var m = _n(p, 1)
    mds_disp_dev(_id(d), _id(dst), li, _n(p, 0), m, _n(p, 2), Int(py=p[3]) != 0)
    return PythonObject(m)
