# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN half of `bindings/_mojolearn_kernel_methods_host.mojo` (wave 8 run
split, 2026-11-12): the host copies and oracle calls each door runs with the
GIL released. The door reads its Python address and parameter lists into
ints and floats, releases the GIL around one call here exactly where it did
before, and packs the result. Typed Mojo only, no Python import
(tools/python_free_dirs.sh). Moved word for word: same copies, same calls,
same order, so the bits are unchanged."""
from core.hostptr import f64_ptr, read_f32
from kernel_methods.host.km_host_impl import (
    kmh_kernel_ridge_fit_run,
    kmh_kernel_ridge_predict_into,
    kmh_nystroem_fit_run,
    kmh_nystroem_transform_into,
    kmh_rbf_sampler_transform_into,
    kmh_sqrt_weights,
)


def kernel_ridge_fit_host_run(
    x_address: Int,
    y_address: Int,
    weighted: Bool,
    w_address: Int,
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
) raises -> Int:
    """`KernelRidge.fit(X, y)`: the host copies of X and y, the weight
    roots when weighted, and the fit; returns `info`."""
    var x = read_f32(x_address, max(0, n * d))
    var y = read_f32(y_address, max(0, n * t))
    var sw = List[Float32]()
    if weighted:
        # the per-row factors sqrt(sample_weight): the binary64 square root
        # of each weight rounded once to float32 (formerly Python's loop)
        sw = kmh_sqrt_weights(f64_ptr(w_address), n)
    var info = kmh_kernel_ridge_fit_run(
        x, y, n, d, t, kernel, degree, gamma, coef0, alpha, dp, sp, sw
    )
    _ = x^
    _ = y^
    return info


def kernel_ridge_predict_host_run(
    fit_address: Int,
    dual_address: Int,
    new_address: Int,
    n: Int,
    d: Int,
    t: Int,
    kernel: Int,
    degree: Int,
    gamma: Float64,
    coef0: Float64,
    q: Int,
    op: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """`KernelRidge.predict(X)`: the host copies and the predict."""
    var x_fit = read_f32(fit_address, max(0, n * d))
    var dual = read_f32(dual_address, max(0, n * t))
    var x_new = read_f32(new_address, max(0, q * d))
    kmh_kernel_ridge_predict_into(
        x_fit, dual, n, d, t, kernel, degree, gamma, coef0, x_new, q, op
    )
    _ = x_fit^
    _ = dual^
    _ = x_new^


def nystroem_fit_host_run(
    x_address: Int,
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
    """`Nystroem.fit(X)`: the host copy of X and the fit; returns `sweeps`."""
    var x = read_f32(x_address, max(0, n * d))
    var sweeps = kmh_nystroem_fit_run(
        x, n, d, kernel, degree, gamma, coef0, q, seed, cp, ip, np_, evp, ecp, sp
    )
    _ = x^
    return sweeps


def nystroem_transform_host_run(
    c_address: Int,
    n_address: Int,
    x_address: Int,
    q: Int,
    d: Int,
    kernel: Int,
    degree: Int,
    gamma: Float64,
    coef0: Float64,
    m: Int,
    op: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """`Nystroem.transform(X)`: the host copies and the transform."""
    var components = read_f32(c_address, max(0, q * d))
    var normalization = read_f32(n_address, max(0, q * q))
    var x = read_f32(x_address, max(0, m * d))
    kmh_nystroem_transform_into(
        components, normalization, q, d, kernel, degree, gamma, coef0, x, m, op
    )
    _ = components^
    _ = normalization^
    _ = x^


def rbf_sampler_transform_host_run(
    w_address: Int,
    o_address: Int,
    x_address: Int,
    d: Int,
    q: Int,
    scale: Float32,
    m: Int,
    op: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """`RBFSampler.transform(X)`: the host copies and the transform."""
    var weights = read_f32(w_address, max(0, d * q))
    var offset = read_f32(o_address, max(0, q))
    var x = read_f32(x_address, max(0, m * d))
    kmh_rbf_sampler_transform_into(weights, offset, d, q, scale, x, m, op)
    _ = weights^
    _ = offset^
    _ = x^
