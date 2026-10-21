# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed host entries of `bindings/_mojolearn_kernel_methods_host.mojo`
(KernelRidge, Nystroem, RBFSampler), with no Python import. The binding
unpacks its arguments, calls one function here and returns; the oracle
calls and the copies into the caller's buffers live here, moved word for
word (lane thin-core, 2026-10-21: no bit change)."""
from std.math import sqrt

from kernel_methods.host.km_host_oracle import (
    kmh_kernel_ridge_fit,
    kmh_kernel_ridge_predict,
    kmh_nystroem_fit,
    kmh_nystroem_transform,
    kmh_rbf_sampler_fit,
    kmh_rbf_sampler_transform,
)

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]


def kmh_sqrt_weights(wp: _F64P, n: Int) -> List[Float32]:
    """The per-row factors sqrt(sample_weight): the binary64 square root
    of each weight rounded once to float32 (formerly Python's loop)."""
    var sw = List[Float32]()
    sw.reserve(max(0, n))
    for i in range(n):
        sw.append(Float32(sqrt(wp[i])))
    return sw^


def kmh_kernel_ridge_fit_run(
    x: List[Float32],
    y: List[Float32],
    n: Int,
    d: Int,
    t: Int,
    kernel: Int,
    degree: Int,
    gamma: Float64,
    coef0: Float64,
    alpha: Float32,
    dp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
    sw: List[Float32],
) raises -> Int:
    var dual = kmh_kernel_ridge_fit(x, y, n, d, t, kernel, degree, gamma, coef0, alpha, sw)
    for i in range(n * t):
        dp.unsafe_store(i, dual[i])
    sp.unsafe_store(0, Float64(0))
    _ = dual^
    return 0


def kmh_nystroem_fit_run(
    x: List[Float32],
    n: Int,
    d: Int,
    kernel: Int,
    degree: Int,
    gamma: Float64,
    coef0: Float64,
    q: Int,
    seed: UInt64,
    cp: MutPointer[Float32, MutUntrackedOrigin],
    ip: MutPointer[Int32, MutUntrackedOrigin],
    np_: MutPointer[Float32, MutUntrackedOrigin],
    evp: MutPointer[Float32, MutUntrackedOrigin],
    ecp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    var model = kmh_nystroem_fit(x, n, d, kernel, degree, gamma, coef0, q, seed)
    for i in range(q * d):
        cp.unsafe_store(i, model.components[i])
    for i in range(q):
        ip.unsafe_store(i, model.indices[i])
        evp.unsafe_store(i, model.eigenvalues[i])
    for i in range(q * q):
        np_.unsafe_store(i, model.normalization[i])
        ecp.unsafe_store(i, model.eigenvectors[i])
    sp.unsafe_store(0, Float64(model.sweeps))
    var sweeps = model.sweeps
    _ = model^
    return sweeps


def kmh_kernel_ridge_predict_into(
    x_fit: List[Float32], dual: List[Float32], n: Int, d: Int, t: Int,
    kernel: Int, degree: Int, gamma: Float64, coef0: Float64,
    x_new: List[Float32], q: Int, op: _F32P,
) raises:
    """`kmh_kernel_ridge_predict` into the caller's q x t output."""
    var out = kmh_kernel_ridge_predict(
        x_fit, dual, n, d, t, kernel, degree, gamma, coef0, x_new, q
    )
    for i in range(q * t):
        op.unsafe_store(i, out[i])
    _ = out^


def kmh_nystroem_transform_into(
    components: List[Float32], normalization: List[Float32], q: Int, d: Int,
    kernel: Int, degree: Int, gamma: Float64, coef0: Float64,
    x: List[Float32], m: Int, op: _F32P,
) raises:
    """`kmh_nystroem_transform` into the caller's m x q output."""
    var out = kmh_nystroem_transform(
        components, normalization, q, d, kernel, degree, gamma, coef0, x, m
    )
    for i in range(m * q):
        op.unsafe_store(i, out[i])
    _ = out^


def kmh_rbf_sampler_fit_into(
    d: Int, q: Int, gamma: Float32, seed: UInt64, wp: _F32P, bp: _F32P, sp: _F64P
) raises:
    """`kmh_rbf_sampler_fit` into the caller's weights, offset and the two
    scalars (sigma, scale)."""
    var model = kmh_rbf_sampler_fit(d, q, gamma, seed)
    for i in range(d * q):
        wp.unsafe_store(i, model.weights[i])
    for i in range(q):
        bp.unsafe_store(i, model.offset[i])
    sp.unsafe_store(0, Float64(model.sigma))
    sp.unsafe_store(1, Float64(model.scale))
    _ = model^


def kmh_rbf_sampler_transform_into(
    weights: List[Float32], offset: List[Float32], d: Int, q: Int, scale: Float32,
    x: List[Float32], m: Int, op: _F32P,
) raises:
    """`kmh_rbf_sampler_transform` into the caller's m x q output."""
    var out = kmh_rbf_sampler_transform(weights, offset, d, q, scale, x, m)
    for i in range(m * q):
        op.unsafe_store(i, out[i])
    _ = out^
