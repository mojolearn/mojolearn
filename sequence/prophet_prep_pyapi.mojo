# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python entries of ProphetForecaster's input preparation
(`sequence/prophet_prep.mojo`): PythonObject arguments to typed values and
the typed result back. Nothing else."""
from std.python import PythonObject

from sequence.exec_trait import Exec
from sequence.prophet_prep import prophet_changepoints, prophet_days, prophet_features
from sequence.pyapi import py_ints


def prophet_days_py(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """addrs = [src, days (N) float64 out, info (3) float64 out]; ip = [N,
    kind]. Returns the status (0 ok, 1 a NaN, 2 not ascending)."""
    if len(addrs) != 3 or len(ip) != 2:
        raise Error("prophet_days: requires 3 addresses and 2 integer parameters")
    var N = Int(py=ip[0])
    var kind = Int(py=ip[1])
    return PythonObject(prophet_days(py_ints(addrs), N, kind))


def prophet_features_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """addrs = [days (N) float64, periods (ns) float64, t (N) float32 out,
    frac (N, max(ns, 1)) float32 out]; ip = [N, ns]; fp = [start, t_scale].
    Returns 0."""
    if len(addrs) != 4 or len(ip) != 2 or len(fp) != 2:
        raise Error("prophet_features: requires 4 addresses, 2 integer and 2 float parameters")
    var N = Int(py=ip[0])
    var ns = Int(py=ip[1])
    return PythonObject(prophet_features(ex, py_ints(addrs), N, ns, Float64(py=fp[0]), Float64(py=fp[1])))


def prophet_changepoints_py(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """addrs = [days (N) float64, user (n_user) float64 days, out (max(S, 1))
    float32]; ip = [N, n_user (-1: none given), hist, n_cp]; fp = [start,
    t_scale]. Returns S (>= 1), or -1 when a caller's changepoint falls
    outside the data (or is NaN)."""
    if len(addrs) != 3 or len(ip) != 4 or len(fp) != 2:
        raise Error("prophet_changepoints: requires 3 addresses, 4 integer and 2 float parameters")
    var N = Int(py=ip[0])
    var n_user = Int(py=ip[1])
    var hist = Int(py=ip[2])
    var n_cp = Int(py=ip[3])
    var start = Float64(py=fp[0])
    var scale = Float64(py=fp[1])
    return PythonObject(prophet_changepoints(py_ints(addrs), N, n_user, hist, n_cp, start, scale))
