# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python door of `x_prep/fastnb_csr.mojo` for `bindings/_mojolearn_x_prep.mojo`:
PythonObject addresses and sizes to Ints, the GIL released around the typed
call. No arithmetic on data."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_prep.fastnb_csr import nb_csr_fit, nb_csr_fit_int, nb_csr_jll, nb_csr_jll_chk


def nb_csr_fit_int_py(
    indptr: PythonObject, indices: PythonObject, data: PythonObject, y: PythonObject, sizes: PythonObject,
    out_fc: PythonObject, out_cnt: PythonObject, out_flag: PythonObject,
) raises -> PythonObject:
    """`x_prep_nb_csr_fit` of the IDENTICAL binding: `nb_csr_fit_py`'s
    buffers; out_flag is a level (2: a negative value, the caller refuses
    the input; 1: the caller runs the dense program, the tables are not to
    be used). Returns 1."""
    var n = Int(py=sizes[0])
    var d = Int(py=sizes[1])
    var K = Int(py=sizes[2])
    var nnz = Int(py=sizes[3])
    var pa = Int(py=indptr)
    var ia = Int(py=indices)
    var da = Int(py=data)
    var ya = Int(py=y)
    var fa = Int(py=out_fc)
    var ca = Int(py=out_cnt)
    var ga = Int(py=out_flag)
    with GILReleased(Python()):
        nb_csr_fit_int(n, d, K, nnz, pa, ia, da, ya, fa, ca, ga)
    return PythonObject(1)


def nb_csr_fit_py(
    indptr: PythonObject, indices: PythonObject, data: PythonObject, y: PythonObject, sizes: PythonObject,
    out_fc: PythonObject, out_cnt: PythonObject, out_flag: PythonObject,
) raises -> PythonObject:
    """`x_prep_nb_csr_fit`: host addresses of indptr (n + 1 int32), indices
    (nnz int32), data (nnz float32) and the class codes (n int32); sizes =
    [n, d, K, nnz]; out_fc (K * d float32), out_cnt (K float32) and out_flag
    (1 int32: 1 when a value is negative) are written. Returns 1."""
    var n = Int(py=sizes[0])
    var d = Int(py=sizes[1])
    var K = Int(py=sizes[2])
    var nnz = Int(py=sizes[3])
    var pa = Int(py=indptr)
    var ia = Int(py=indices)
    var da = Int(py=data)
    var ya = Int(py=y)
    var fa = Int(py=out_fc)
    var ca = Int(py=out_cnt)
    var ga = Int(py=out_flag)
    with GILReleased(Python()):
        nb_csr_fit(n, d, K, nnz, pa, ia, da, ya, fa, ca, ga)
    return PythonObject(1)


def nb_csr_jll_py(
    indptr: PythonObject, indices: PythonObject, data: PythonObject, flp: PythonObject, clp: PythonObject,
    sizes: PythonObject, dst: PythonObject,
) raises -> PythonObject:
    """`x_prep_nb_csr_jll`: the joint log likelihood (n x K float32 at dst)
    of a CSR matrix; flp = feature_log_prob_ (K x d), clp = class_log_prior_
    (K) or address 0 for none; sizes = [n, d, K, nnz]. Returns 1."""
    var n = Int(py=sizes[0])
    var d = Int(py=sizes[1])
    var K = Int(py=sizes[2])
    var nnz = Int(py=sizes[3])
    var pa = Int(py=indptr)
    var ia = Int(py=indices)
    var da = Int(py=data)
    var wa = Int(py=flp)
    var ba = Int(py=clp)
    var oa = Int(py=dst)
    with GILReleased(Python()):
        nb_csr_jll(n, d, K, nnz, pa, ia, da, wa, ba, oa)
    return PythonObject(1)


def nb_csr_jll_chk_py(
    indptr: PythonObject, indices: PythonObject, data: PythonObject, flp: PythonObject, clp: PythonObject,
    sizes: PythonObject, dst: PythonObject, out_flag: PythonObject,
) raises -> PythonObject:
    """`x_prep_nb_csr_jll` of the IDENTICAL binding: `nb_csr_jll_py`'s
    buffers plus out_flag (1 int32; 1: a row's columns are not strictly
    ascending, the caller runs the dense program). Returns 1."""
    var n = Int(py=sizes[0])
    var d = Int(py=sizes[1])
    var K = Int(py=sizes[2])
    var nnz = Int(py=sizes[3])
    var pa = Int(py=indptr)
    var ia = Int(py=indices)
    var da = Int(py=data)
    var wa = Int(py=flp)
    var ba = Int(py=clp)
    var oa = Int(py=dst)
    var ga = Int(py=out_flag)
    with GILReleased(Python()):
        nb_csr_jll_chk(n, d, K, nnz, pa, ia, da, wa, ba, oa, ga)
    return PythonObject(1)
