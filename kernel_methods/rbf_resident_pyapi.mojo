# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python door of `kernel_methods/rbf_resident.mojo`: PythonObject
arguments to typed values, the GIL released around the typed call. No
arithmetic on data."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from kernel_methods.rbf_resident import rbf_sampler_fit_transform_into, rbf_staged


def rbf_sampler_fit_transform_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """`RBFSampler.fit_transform(X)` under KM_FAST_RBF_RESIDENT. Returns 0.

    `addrs`: 0 weights_out (d*q f32), 1 offset_out (q f32), 2 scalars_out
    (2 f64: sigma, scale), 3 x (m*d f32, read), 4 out (m*q f32).
    `params`: 0 d, 1 q, 2 gamma, 3 seed, 4 m."""
    if len(addrs) != 5:
        raise Error("rbf_sampler_fit_transform: addrs must contain 5 addresses, got " + String(len(addrs)))
    if len(params) != 5:
        raise Error("rbf_sampler_fit_transform: params must contain 5 values, got " + String(len(params)))
    var wa = Int(py=addrs[0])
    var ba = Int(py=addrs[1])
    var sa = Int(py=addrs[2])
    var xa = Int(py=addrs[3])
    var za = Int(py=addrs[4])
    var d = Int(py=params[0])
    var q = Int(py=params[1])
    var gamma = Float64(py=params[2])
    var seed_word = Int(py=params[3])
    var m = Int(py=params[4])
    with GILReleased(Python()):
        rbf_sampler_fit_transform_into(d, q, gamma, seed_word, m, wa, ba, sa, xa, za)
    return PythonObject(0)


def rbf_staged_binding() raises -> PythonObject:
    """1 when KM_FAST_RBF_STAGED is compiled in (the quality pair's reach)."""
    return PythonObject(rbf_staged())
