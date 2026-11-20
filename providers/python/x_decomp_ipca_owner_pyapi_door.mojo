# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The doors of IncrementalPCA's drivers (x_decomp/ipca_owner.mojo): each
unpacks one list of addresses and integers, releases the GIL and calls one
typed driver; no data work happens here. These legacy direct-call exports
remain available; the public IncrementalPCA owner uses its native descriptor."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from x_decomp.cells import F32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.pca_randomized import RDev
from x_decomp.ipca_owner import (
    IPCA_STATE_WORDS, IpcaDev, ipca_fit_run, ipca_inverse_run, ipca_partial_step, ipca_transform_run,
)


def _ptr(p: PythonObject, i: Int) raises -> F32Ptr:
    var a = Int(py=p[i])
    if a == 0:
        raise Error("x_decomp: null float32 buffer address")
    return F32Ptr(unsafe_from_address=a)


def _dim(p: PythonObject, i: Int) raises -> Int:
    var v = Int(py=p[i])
    if v < 0 or v > 2147483647:
        raise Error("x_decomp: dimension out of range")
    return v


def _state_tuple(got: Tuple[Int, Float64], kept: List[Int]) raises -> PythonObject:
    """(n_components_, noise_variance_), followed by the kept state's pool
    ids when the step kept them (the caller owns them from here)."""
    if len(kept) != IPCA_STATE_WORDS:
        return Python.tuple(PythonObject(got[0]), PythonObject(got[1]))
    return Python.tuple(PythonObject(got[0]), PythonObject(got[1]), PythonObject(kept[0]), PythonObject(kept[1]),
                        PythonObject(kept[2]), PythonObject(kept[3]), PythonObject(kept[4]))


def _state_ids(p: PythonObject, start: Int) raises -> List[Int]:
    """IPCA_STATE_WORDS borrowed pool ids from p[start:] (-1: not resident)."""
    var ids = List[Int]()
    for k in range(IPCA_STATE_WORDS):  # driver-loop: over the five state words
        var v = Int(py=p[start + k])
        if v < -1:
            raise Error("x_decomp: a resident state id must be -1 or a pool id")
        ids.append(v)
    return ids^


def ipca_fit_py[E: Exec, D: RDev, RES: Bool, X: IpcaDev](p: PythonObject) raises -> PythonObject:
    """IncrementalPCA.fit. p = [x, n, d, n_components (-1: None),
    batch_size, components (d x d room), singular, mean, var, explained
    variance, ratio (d each)], optionally followed by a keep word (1: keep
    the final state on the device; GPU bindings, lane
    ne1-a-decomp-resident). Returns (n_components_, noise_variance_), plus
    the kept state's five pool ids when they were kept."""
    if len(p) != 11 and len(p) != 12:
        raise Error("x_decomp_ipca_fit: params must hold x, n, d, n_components, batch_size and six outputs")
    var x = _ptr(p, 0)
    var n = _dim(p, 1)
    var d = _dim(p, 2)
    var requested = Int(py=p[3])
    var batch_size = Int(py=p[4])
    var pc = _ptr(p, 5)
    var ps = _ptr(p, 6)
    var pm = _ptr(p, 7)
    var pv = _ptr(p, 8)
    var pe = _ptr(p, 9)
    var pr = _ptr(p, 10)
    var keep = len(p) == 12 and Int(py=p[11]) != 0
    var kept = List[Int]()
    var got = Tuple[Int, Float64](0, 0.0)
    with GILReleased(Python()):
        got = ipca_fit_run[E, D, RES, X](x, n, d, requested, batch_size, pc, ps, pm, pv, pe, pr, keep, kept)
    return _state_tuple(got, kept)


def ipca_partial_py[E: Exec, D: RDev, RES: Bool, X: IpcaDev](p: PythonObject) raises -> PythonObject:
    """IncrementalPCA.partial_fit. p = [x, n, d, n_components (-1: None),
    seen, prev count, prev d, prev components, prev singular, prev mean,
    prev var (read when seen > 0), components (d x d room), singular, mean,
    var, explained variance, ratio (d each)], optionally followed by the
    previous state's five borrowed pool ids (-1: not resident) and a keep
    word (GPU bindings, lane ne1-a-decomp-resident). Returns
    (n_components_, noise_variance_), plus the new state's five pool ids
    when they were kept."""
    if len(p) != 17 and len(p) != 17 + IPCA_STATE_WORDS + 1:
        raise Error("x_decomp_ipca_partial: params must hold x, n, d, n_components, the state and six outputs")
    var x = _ptr(p, 0)
    var n = _dim(p, 1)
    var d = _dim(p, 2)
    var requested = Int(py=p[3])
    var seen = Int(py=p[4])
    var prev_count = Int(py=p[5])
    var prev_d = Int(py=p[6])
    var qc = _ptr(p, 7)
    var qs = _ptr(p, 8)
    var qm = _ptr(p, 9)
    var qv = _ptr(p, 10)
    var pc = _ptr(p, 11)
    var ps = _ptr(p, 12)
    var pm = _ptr(p, 13)
    var pv = _ptr(p, 14)
    var pe = _ptr(p, 15)
    var pr = _ptr(p, 16)
    var resident = len(p) == 17 + IPCA_STATE_WORDS + 1
    var prev_dev = _state_ids(p, 17) if resident else List[Int]()
    var keep = resident and Int(py=p[17 + IPCA_STATE_WORDS]) != 0
    var kept = List[Int]()
    var got = Tuple[Int, Float64](0, 0.0)
    with GILReleased(Python()):
        got = ipca_partial_step[E, D, RES, X](x, n, d, requested, seen, prev_count, prev_d, qc, qs, qm, qv,
                                              pc, ps, pm, pv, pe, pr, prev_dev, keep, kept)
    return _state_tuple(got, kept)


def ipca_project_py[E: Exec, D: RDev, RES: Bool](p: PythonObject) raises -> PythonObject:
    """IncrementalPCA.transform (p[0] 0) or inverse_transform (1). p = [op,
    x, n, d, n_components, mean, components, explained variance, whiten,
    out]; x is n x d (transform) or n x n_components (inverse). Optionally
    followed by the state's five borrowed pool ids (-1: not resident; GPU
    bindings, lane ne1-a-decomp-resident)."""
    if len(p) != 10 and len(p) != 10 + IPCA_STATE_WORDS:
        raise Error("x_decomp_ipca_project: params must hold op, x, n, d, n_components, the state, whiten and out")
    var op = Int(py=p[0])
    var x = _ptr(p, 1)
    var n = _dim(p, 2)
    var d = _dim(p, 3)
    var nc = _dim(p, 4)
    var pm = _ptr(p, 5)
    var pc = _ptr(p, 6)
    var pe = _ptr(p, 7)
    var whiten = Int(py=p[8]) != 0
    var out = _ptr(p, 9)
    var state_dev = _state_ids(p, 10) if len(p) == 10 + IPCA_STATE_WORDS else List[Int]()
    with GILReleased(Python()):
        if op == 0:
            ipca_transform_run[E, D, RES](x, n, d, nc, pm, pc, pe, whiten, out, state_dev)
        else:
            ipca_inverse_run[E, D, RES](x, n, d, nc, pm, pc, pe, whiten, out, state_dev)
    return PythonObject(n)
