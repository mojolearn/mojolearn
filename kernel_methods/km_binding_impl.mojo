# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed entries of `bindings/_mojolearn_kernel_methods.mojo` (KernelRidge,
Nystroem, RBFSampler), with no Python import: each runs the estimator and
copies its model into the caller's buffers. The binding unpacks its
arguments and calls one of these (lane thin-core, 2026-10-21: moved word
for word, no bit change)."""
from std.math import sqrt

from bindings.hostptr import copy_f32
from core.identity_trace import IdentityTrace
from kernel_methods.estimator import (
    KernelRidgeModel,
    NystroemModel,
    RBFSamplerModel,
    kernel_ridge_fit_host,
    kernel_ridge_predict_host,
    nystroem_fit_host,
    nystroem_transform_host_into,
    nystroem_fit_ptr,
    rbf_sampler_fit_host,
    rbf_sampler_transform_host_into,
)
from svm.impl.svm_parameter import KernelParams


def km_sqrt_weights(wp: MutPointer[Float64, MutUntrackedOrigin], n: Int) -> List[Float32]:
    """The per-row factors sqrt(sample_weight): the binary64 square root
    of each weight rounded once to float32 (formerly Python's loop)."""
    var sw = List[Float32]()
    sw.reserve(max(0, n))
    for i in range(n):
        sw.append(Float32(sqrt(wp[i])))
    return sw^


def km_kernel_ridge_fit_run(
    x: List[Float32],
    y: List[Float32],
    n: Int,
    d: Int,
    t: Int,
    kp: KernelParams,
    alpha: Float32,
    dp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
    sw: List[Float32],
) raises -> Int:
    var trace = IdentityTrace()
    var model = kernel_ridge_fit_host(x, y, n, d, t, kp, alpha, trace, sw=sw)
    copy_f32(model.dual_coef.unsafe_ptr(), dp, n * t)
    sp.unsafe_store(0, Float64(model.info))
    return model.info


def km_kernel_ridge_predict_run(
    model: KernelRidgeModel,
    x_new: List[Float32],
    q: Int,
    op: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    var trace = IdentityTrace()
    var out = kernel_ridge_predict_host(model, x_new, q, trace)
    copy_f32(out.unsafe_ptr(), op, q * model.n_targets)


def km_nystroem_fit_run(
    x: List[Float32],
    n: Int,
    d: Int,
    kp: KernelParams,
    q: Int,
    seed: UInt64,
    cp: MutPointer[Float32, MutUntrackedOrigin],
    ip: MutPointer[Int32, MutUntrackedOrigin],
    np_: MutPointer[Float32, MutUntrackedOrigin],
    evp: MutPointer[Float32, MutUntrackedOrigin],
    ecp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    var trace = IdentityTrace()
    var model = nystroem_fit_host(x, n, d, kp, q, seed, trace)
    copy_f32(model.components.unsafe_ptr(), cp, q * d)
    for i in range(q):
        ip.unsafe_store(i, model.component_indices[i])
    copy_f32(model.normalization.unsafe_ptr(), np_, q * q)
    copy_f32(model.eigenvalues.unsafe_ptr(), evp, q)
    copy_f32(model.eigenvectors.unsafe_ptr(), ecp, q * q)
    sp.unsafe_store(0, Float64(model.sweeps))
    return model.sweeps


def km_nystroem_transform_run(
    model: NystroemModel,
    x: List[Float32],
    m: Int,
    op: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    var trace = IdentityTrace()
    nystroem_transform_host_into(model, x, m, op, trace)


def km_rbf_sampler_fit_run(
    d: Int,
    q: Int,
    gamma: Float32,
    seed: UInt64,
    wp: MutPointer[Float32, MutUntrackedOrigin],
    bp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises:
    var trace = IdentityTrace()
    var model = rbf_sampler_fit_host(d, q, gamma, seed, trace)
    copy_f32(model.random_weights.unsafe_ptr(), wp, d * q)
    copy_f32(model.random_offset.unsafe_ptr(), bp, q)
    sp.unsafe_store(0, Float64(model.sigma))
    sp.unsafe_store(1, Float64(model.scale))


def km_rbf_sampler_transform_run(
    model: RBFSamplerModel,
    x: List[Float32],
    m: Int,
    op: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    var trace = IdentityTrace()
    rbf_sampler_transform_host_into(model, x, m, op, trace)


def km_nystroem_fit_ptr_run(
    xaddr: Int,
    n: Int,
    d: Int,
    kp: KernelParams,
    q: Int,
    seed: UInt64,
    cp: MutPointer[Float32, MutUntrackedOrigin],
    ip: MutPointer[Int32, MutUntrackedOrigin],
    np_: MutPointer[Float32, MutUntrackedOrigin],
    evp: MutPointer[Float32, MutUntrackedOrigin],
    ecp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    """`nystroem_fit_ptr` over the caller's X, the model copied out.
    Returns the sweeps."""
    var sweeps = 0
    var trace = IdentityTrace()
    var model = nystroem_fit_ptr(xaddr, n, d, kp, q, seed, trace)
    copy_f32(model.components.unsafe_ptr(), cp, q * d)
    for i in range(q):
        ip.unsafe_store(i, model.component_indices[i])
    copy_f32(model.normalization.unsafe_ptr(), np_, q * q)
    copy_f32(model.eigenvalues.unsafe_ptr(), evp, q)
    copy_f32(model.eigenvectors.unsafe_ptr(), ecp, q * q)
    sp.unsafe_store(0, Float64(model.sweeps))
    sweeps = model.sweeps
    return sweeps
