# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python-callable doors of the NMF drivers (x_decomp/nmf_owner.mojo):
each unpacks one list of addresses and integers (and one of floats),
releases the GIL and calls one typed driver; no data work happens here.
The FastICA, FactorAnalysis, LatentDirichletAllocation and random projection
doors that stood here are operations of the x_decomp table since lane
dw-decomp (2026-11-21): x_decomp/descriptor_factor_ops.mojo."""
# The PACK half of the former `providers/python/x_decomp_decomp_owner_pyapi.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `providers/python/x_decomp_decomp_owner_pyapi_unpack.mojo`.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from x_decomp.cells import F32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.pca_randomized import RDev
from x_decomp.decomp_owner_kit import DecompDev
from x_decomp.nmf import nmf_args_checked
from x_decomp.nmf_owner import nmf_inverse_run, nmf_run


from providers.python.x_decomp_decomp_owner_pyapi_unpack import (
    door_dim,
    door_ptr,
)


def nmf_fit_py[E: Exec, D: RDev, RES: Bool, X: DecompDev](p: PythonObject, f: PythonObject) raises -> PythonObject:
    """NMF fit_transform (init 0..4) or transform (5). p = [x, n, d, nc,
    init, W in (read for init 4), H in (read for init 4 and 5), seed mod
    2^32, solver (0 cd, 1 mu), update_H, max_iter, shuffle, W out, H out];
    f = [beta, tol, l1W, l1H, l2W, l2H]. Returns (n_iter_, error). Lane
    a8d: an optional p[14], the finite flag: 1 asks for X's finite status
    word, returned third (0 = finite; a flagged X writes nothing)."""
    if (len(p) != 14 and len(p) != 15) or len(f) != 6:
        raise Error("x_decomp_nmf_fit: params must hold the input, the options and two outputs")
    var x = door_ptr(p, 0)
    var n = door_dim(p, 1)
    var d = door_dim(p, 2)
    var nc = door_dim(p, 3)
    var init = Int(py=p[4])
    var w_in = door_ptr(p, 5)
    var h_in = door_ptr(p, 6)
    var seed = Int(py=p[7])
    var solver = Int(py=p[8])
    var a = nmf_args_checked(Float64(py=f[0]), Int(py=p[9]) != 0, Int(py=p[10]), Float64(py=f[1]),
                             Float64(py=f[2]), Float64(py=f[3]), Float64(py=f[4]), Float64(py=f[5]),
                             Int(py=p[11]) != 0, seed)
    var w_out = door_ptr(p, 12)
    var h_out = door_ptr(p, 13)
    var test = len(p) == 15 and Int(py=p[14]) != 0
    var word = List[Int32](length=1, fill=Int32(0))
    var word_addr = Int(word.unsafe_ptr()) if test else 0
    var got = Tuple[Int, Float64](0, 0.0)
    with GILReleased(Python()):
        got = nmf_run[E, D, RES, X](x, n, d, nc, init, w_in, h_in, seed, solver, a, w_out, h_out, word_addr)
    if test:
        return Python.tuple(PythonObject(got[0]), PythonObject(got[1]), PythonObject(Int(word[0])))
    return Python.tuple(PythonObject(got[0]), PythonObject(got[1]))


def nmf_inverse_py[E: Exec, D: RDev, RES: Bool](p: PythonObject) raises -> PythonObject:
    """NMF inverse_transform. p = [W, n, nc, H, d, out]."""
    if len(p) != 6:
        raise Error("x_decomp_nmf_inverse: params must hold W, n, nc, H, d and out")
    var w = door_ptr(p, 0)
    var n = door_dim(p, 1)
    var nc = door_dim(p, 2)
    var h = door_ptr(p, 3)
    var d = door_dim(p, 4)
    var out = door_ptr(p, 5)
    with GILReleased(Python()):
        nmf_inverse_run[E, D, RES](w, n, nc, h, d, out)
    return PythonObject(n)
