# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GaussianMixture sample entry of the mixture host bindings, written
once (the neighbors and density inference lane, 2026-09-15).

`bindings/_mojolearn_mixture_host.mojo` (the reference binding) and
`bindings/_mojolearn_mixture_infer_host.mojo` (the inference binding a
wheel ships) both register `gmm_sample` from here. The scoring entries that
lived here are deleted: the mixture door scores through each binding's
typed transport (mixture/host/gmm_native_host_score.mojo). The address and
params contract is the GPU binding's (`bindings/_mojolearn_mixture.mojo`).
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from core.hostptr import f32_ptr, i32_ptr, read_f32
# The sampling body (lane thin-trees-metrics, 2026-10-21) is the kernel
# module's.
from mixture.host.gmm_host_entries import gmm_sample_into


def gmm_sample_binding(
    addrs: PythonObject, params: PythonObject, sample_params: PythonObject
) raises -> PythonObject:
    """`sample(n_samples)` on the host, the GPU binding's name, arity and
    lists: `addrs[0..4]` the model, `addrs[5]` X out (n_samples * d
    float32), `addrs[6]` y out (n_samples int32); `params` k, d, n_iter,
    converged, lower_bound, 0; `sample_params` n_samples, random_state low
    32 bits, high 32 bits. The arithmetic is
    `mixture/checks/sample.mojo::gmm_sample_host`, the device kernel row for
    row. Returns n_samples."""
    if len(addrs) != 7 or len(params) != 6 or len(sample_params) != 3:
        raise Error(
            "gmm_sample: needs 7 addresses, 6 params and 3 sample_params, got "
            + String(len(addrs))
            + ", "
            + String(len(params))
            + ", "
            + String(len(sample_params))
        )
    var k = Int(py=params[0])
    var d = Int(py=params[1])
    var n = Int(py=sample_params[0])
    var seed = (UInt64(Int(py=sample_params[2])) << 32) | UInt64(
        Int(py=sample_params[1])
    )
    var weights = read_f32(Int(py=addrs[0]), max(0, k))
    var means = read_f32(Int(py=addrs[1]), max(0, k * d))
    var precisions = read_f32(Int(py=addrs[3]), max(0, k * d * d))
    var xp = f32_ptr(Int(py=addrs[5]))
    var yp = i32_ptr(Int(py=addrs[6]))
    with GILReleased(Python()):
        gmm_sample_into(weights, means, precisions, k, d, n, seed, xp, yp)
    return PythonObject(n)
