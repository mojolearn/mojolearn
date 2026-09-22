# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_kernel_methods` family: KernelRidge,
Nystroem and RBFSampler (CPU training for the workstream D estimators,
2026-09-15).

HOST ONLY. No DeviceContext, no kernel launch. The fits, predictions and
transforms are `kernel_methods/host/km_host_oracle.mojo`, the device path of
`kernel_methods/estimator.mojo` restated on the host over the gemm profile's
normative answer (`gemm/host/gemm_oracle.mojo`), the Cholesky host profile
(`cholesky/host/chol_oracle.mojo`) and the PCA host lane's Jacobi and sign
flip (`decomposition/host/pca_oracle.mojo`). So `dual_coef_`, the Nystroem
model arrays and the random features are meant to be the GPU columns' bytes.

THE EXPORTED NAMES ARE THE GPU BINDING'S NAMES, with the GPU binding's
address and params contract word for word (`bindings/_mojolearn_kernel_methods.mojo`,
mirrored in `python/mojolearn/kernel_methods.py`), so the three classes run
unchanged on a CPU-only install through `_backend._HOST_MODULES`
(`"_mojolearn_kernel_methods": "_mojolearn_kernel_methods_host"`):
`kernel_ridge_fit` (4 addresses, 8 params), `kernel_ridge_predict` (4, 10),
`nystroem_fit` (7, 8), `nystroem_transform` (7, 9), `rbf_sampler_fit` (3, 4),
`rbf_sampler_transform` (4, 7), `kernel_methods_vendor` answering "cpu" and
`kernel_methods_numeric_mode`. ABSENT, and so refused BY NAME through
`_HostBinding`: `kernel_methods_rows_parallel_available`, the multi-GPU
driver's probe (`parallel_classical.fit_kernel_method`).

WHAT IS REFUSED BELOW THE PYTHON SURFACE, BY NAME: everything the device
entries refuse (non-finite inputs, a non-positive gamma, a negative or NaN
alpha, a kernel matrix that does not factor, an unconverged Jacobi,
n_components out of range). All five implemented kernel kinds are replayed
on the host; precomputed kernels remain refused.

The sabotage arm (`kernel_methods_host_sabotage`) is
`gemm/host/gemm_oracle.mojo::GEMM_ORACLE_HOST_SABOTAGE`
(`-D MOJOLEARN_HOST_SABOTAGE=1`): every GEMM leaf walks descending, so every
kernel matrix, factorization, normalization and feature map this binary
serves differs.
"""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from bindings.hostptr import f32_ptr, f64_ptr, i32_ptr, read_f32
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from gemm.host.identical_gemm import GEMM_ORACLE_HOST_SABOTAGE
from kernel_methods.host.km_host_oracle import (
    kmh_kernel_ridge_fit,
    kmh_kernel_ridge_predict,
    kmh_nystroem_fit,
    kmh_nystroem_transform,
    kmh_rbf_sampler_fit,
    kmh_rbf_sampler_transform,
)


def kernel_methods_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def kernel_methods_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def kernel_methods_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu".
    `bindings/build_host_family.sh` passes -D MOJOLEARN_COLUMN_CPU."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "kernel_methods host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_kernel_methods_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def kernel_methods_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary walks every GEMM leaf descending on purpose
    (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's negative control)."""
    return PythonObject(GEMM_ORACLE_HOST_SABOTAGE)


def kernel_methods_vendor_binding() raises -> PythonObject:
    """"cpu", the read-back `_backend.vendor()` expects on a CPU-only
    install."""
    return PythonObject(String("cpu"))


def kernel_methods_numeric_mode_binding() raises -> PythonObject:
    """The build's tier code; a host binding is IDENTICAL only, so 1."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


# ===========================================================================
# KernelRidge
# ===========================================================================


def _kernel_ridge_fit_run(
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
) raises -> Int:
    var dual = kmh_kernel_ridge_fit(x, y, n, d, t, kernel, degree, gamma, coef0, alpha)
    for i in range(n * t):
        dp.unsafe_store(i, dual[i])
    sp.unsafe_store(0, Float64(0))
    _ = dual^
    return 0


def kernel_ridge_fit_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`KernelRidge.fit(X, y)` on the host. `addrs`: 0 x, 1 y, 2 dual_out,
    3 scalars_out (info). `params`: 0 n, 1 d, 2 t, 3 kernel, 4 degree,
    5 gamma, 6 coef0, 7 alpha. Returns `info`, 0 on every model that comes
    back."""
    if len(addrs) != 4:
        raise Error(
            "kernel_ridge_fit: addrs must contain 4 addresses (x, y,"
            " dual_out, scalars_out), got "
            + String(len(addrs))
        )
    if len(params) != 8:
        raise Error(
            "kernel_ridge_fit: params must contain 8 values (n, d, t, kernel,"
            " degree, gamma, coef0, alpha), got "
            + String(len(params))
        )
    var dp = f32_ptr(Int(py=addrs[2]))
    var sp = f64_ptr(Int(py=addrs[3]))
    var n = Int(py=params[0])
    var d = Int(py=params[1])
    var t = Int(py=params[2])
    var kernel = Int(py=params[3])
    var degree = Int(py=params[4])
    var gamma = Float64(py=params[5])
    var coef0 = Float64(py=params[6])
    var alpha = Float32(Float64(py=params[7]))
    var x = read_f32(Int(py=addrs[0]), max(0, n * d))
    var y = read_f32(Int(py=addrs[1]), max(0, n * t))
    var info = 0
    with GILReleased(Python()):
        info = _kernel_ridge_fit_run(
            x, y, n, d, t, kernel, degree, gamma, coef0, alpha, dp, sp
        )
    _ = x^
    _ = y^
    return PythonObject(info)


def kernel_ridge_predict_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`KernelRidge.predict(X)` on the host. `addrs`: 0 x_fit, 1 dual,
    2 x_new, 3 out. `params`: 0 n, 1 d, 2 t, 3 kernel, 4 degree, 5 gamma,
    6 coef0, 7 alpha, 8 info (passed through), 9 q. Returns 0."""
    if len(addrs) != 4:
        raise Error(
            "kernel_ridge_predict: addrs must contain 4 addresses (x_fit,"
            " dual, x_new, out), got "
            + String(len(addrs))
        )
    if len(params) != 10:
        raise Error(
            "kernel_ridge_predict: params must contain 10 values (n, d, t,"
            " kernel, degree, gamma, coef0, alpha, info, q), got "
            + String(len(params))
        )
    var op = f32_ptr(Int(py=addrs[3]))
    var n = Int(py=params[0])
    var d = Int(py=params[1])
    var t = Int(py=params[2])
    var kernel = Int(py=params[3])
    var degree = Int(py=params[4])
    var gamma = Float64(py=params[5])
    var coef0 = Float64(py=params[6])
    var q = Int(py=params[9])
    var x_fit = read_f32(Int(py=addrs[0]), max(0, n * d))
    var dual = read_f32(Int(py=addrs[1]), max(0, n * t))
    var x_new = read_f32(Int(py=addrs[2]), max(0, q * d))
    with GILReleased(Python()):
        var out = kmh_kernel_ridge_predict(
            x_fit, dual, n, d, t, kernel, degree, gamma, coef0, x_new, q
        )
        for i in range(q * t):
            op.unsafe_store(i, out[i])
        _ = out^
    _ = x_fit^
    _ = dual^
    _ = x_new^
    return PythonObject(0)


# ===========================================================================
# Nystroem
# ===========================================================================


def _nystroem_fit_run(
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


def nystroem_fit_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`Nystroem.fit(X)` on the host. `addrs`: 0 x, 1 components_out,
    2 indices_out, 3 normalization_out, 4 eigenvalues_out,
    5 eigenvectors_out, 6 scalars_out (sweeps). `params`: 0 n, 1 d,
    2 kernel, 3 degree, 4 gamma, 5 coef0, 6 q, 7 seed. Returns `sweeps`."""
    if len(addrs) != 7:
        raise Error(
            "nystroem_fit: addrs must contain 7 addresses (x, components_out,"
            " indices_out, normalization_out, eigenvalues_out,"
            " eigenvectors_out, scalars_out), got "
            + String(len(addrs))
        )
    if len(params) != 8:
        raise Error(
            "nystroem_fit: params must contain 8 values (n, d, kernel, degree,"
            " gamma, coef0, q, seed), got "
            + String(len(params))
        )
    var cp = f32_ptr(Int(py=addrs[1]))
    var ip = i32_ptr(Int(py=addrs[2]))
    var np_ = f32_ptr(Int(py=addrs[3]))
    var evp = f32_ptr(Int(py=addrs[4]))
    var ecp = f32_ptr(Int(py=addrs[5]))
    var sp = f64_ptr(Int(py=addrs[6]))
    var n = Int(py=params[0])
    var d = Int(py=params[1])
    var kernel = Int(py=params[2])
    var degree = Int(py=params[3])
    var gamma = Float64(py=params[4])
    var coef0 = Float64(py=params[5])
    var q = Int(py=params[6])
    var seed = UInt64(Int(py=params[7]))
    var x = read_f32(Int(py=addrs[0]), max(0, n * d))
    var sweeps = 0
    with GILReleased(Python()):
        sweeps = _nystroem_fit_run(
            x, n, d, kernel, degree, gamma, coef0, q, seed, cp, ip, np_, evp, ecp, sp
        )
    _ = x^
    return PythonObject(sweeps)


def nystroem_transform_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`Nystroem.transform(X)` on the host. `addrs`: 0 components,
    1 indices, 2 normalization, 3 eigenvalues, 4 eigenvectors, 5 x, 6 out.
    `params`: 0 q, 1 d, 2 kernel, 3 degree, 4 gamma, 5 coef0, 6 seed,
    7 sweeps, 8 m. Returns 0."""
    if len(addrs) != 7:
        raise Error(
            "nystroem_transform: addrs must contain 7 addresses (components,"
            " indices, normalization, eigenvalues, eigenvectors, x, out),"
            " got "
            + String(len(addrs))
        )
    if len(params) != 9:
        raise Error(
            "nystroem_transform: params must contain 9 values (q, d, kernel,"
            " degree, gamma, coef0, seed, sweeps, m), got "
            + String(len(params))
        )
    var q = Int(py=params[0])
    var d = Int(py=params[1])
    var kernel = Int(py=params[2])
    var degree = Int(py=params[3])
    var gamma = Float64(py=params[4])
    var coef0 = Float64(py=params[5])
    var m = Int(py=params[8])
    var components = read_f32(Int(py=addrs[0]), max(0, q * d))
    var normalization = read_f32(Int(py=addrs[2]), max(0, q * q))
    var x = read_f32(Int(py=addrs[5]), max(0, m * d))
    var op = f32_ptr(Int(py=addrs[6]))
    with GILReleased(Python()):
        var out = kmh_nystroem_transform(
            components, normalization, q, d, kernel, degree, gamma, coef0, x, m
        )
        for i in range(m * q):
            op.unsafe_store(i, out[i])
        _ = out^
    _ = components^
    _ = normalization^
    _ = x^
    return PythonObject(0)


# ===========================================================================
# RBFSampler
# ===========================================================================


def rbf_sampler_fit_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`RBFSampler.fit(X)` on the host; no X crosses. `addrs`:
    0 weights_out, 1 offset_out, 2 scalars_out (sigma, scale). `params`:
    0 d, 1 q, 2 gamma, 3 seed. Returns 0."""
    if len(addrs) != 3:
        raise Error(
            "rbf_sampler_fit: addrs must contain 3 addresses (weights_out,"
            " offset_out, scalars_out), got "
            + String(len(addrs))
        )
    if len(params) != 4:
        raise Error(
            "rbf_sampler_fit: params must contain 4 values (d, q, gamma,"
            " seed), got "
            + String(len(params))
        )
    var wp = f32_ptr(Int(py=addrs[0]))
    var bp = f32_ptr(Int(py=addrs[1]))
    var sp = f64_ptr(Int(py=addrs[2]))
    var d = Int(py=params[0])
    var q = Int(py=params[1])
    var gamma = Float32(Float64(py=params[2]))
    var seed = UInt64(Int(py=params[3]))
    with GILReleased(Python()):
        var model = kmh_rbf_sampler_fit(d, q, gamma, seed)
        for i in range(d * q):
            wp.unsafe_store(i, model.weights[i])
        for i in range(q):
            bp.unsafe_store(i, model.offset[i])
        sp.unsafe_store(0, Float64(model.sigma))
        sp.unsafe_store(1, Float64(model.scale))
        _ = model^
    return PythonObject(0)


def rbf_sampler_transform_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`RBFSampler.transform(X)` on the host. `addrs`: 0 weights, 1 offset,
    2 x, 3 out. `params`: 0 d, 1 q, 2 gamma, 3 seed, 4 sigma, 5 scale, 6 m.
    Returns 0."""
    if len(addrs) != 4:
        raise Error(
            "rbf_sampler_transform: addrs must contain 4 addresses (weights,"
            " offset, x, out), got "
            + String(len(addrs))
        )
    if len(params) != 7:
        raise Error(
            "rbf_sampler_transform: params must contain 7 values (d, q,"
            " gamma, seed, sigma, scale, m), got "
            + String(len(params))
        )
    var d = Int(py=params[0])
    var q = Int(py=params[1])
    var scale = Float32(Float64(py=params[5]))
    var m = Int(py=params[6])
    var weights = read_f32(Int(py=addrs[0]), max(0, d * q))
    var offset = read_f32(Int(py=addrs[1]), max(0, q))
    var x = read_f32(Int(py=addrs[2]), max(0, m * d))
    var op = f32_ptr(Int(py=addrs[3]))
    with GILReleased(Python()):
        var out = kmh_rbf_sampler_transform(weights, offset, d, q, scale, x, m)
        for i in range(m * q):
            op.unsafe_store(i, out[i])
        _ = out^
    _ = weights^
    _ = offset^
    _ = x^
    return PythonObject(0)


@export
def PyInit__mojolearn_kernel_methods_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_kernel_methods_host")
        module.def_function[kernel_methods_host_numeric_mode_binding]("kernel_methods_host_numeric_mode")
        module.def_function[kernel_methods_host_vendor_binding]("kernel_methods_host_vendor")
        module.def_function[kernel_methods_host_column_binding]("kernel_methods_host_column")
        module.def_function[kernel_methods_host_sabotage_binding]("kernel_methods_host_sabotage")
        module.def_function[kernel_methods_vendor_binding]("kernel_methods_vendor")
        module.def_function[kernel_methods_numeric_mode_binding]("kernel_methods_numeric_mode")
        module.def_function[kernel_ridge_fit_binding]("kernel_ridge_fit")
        module.def_function[kernel_ridge_predict_binding]("kernel_ridge_predict")
        module.def_function[nystroem_fit_binding]("nystroem_fit")
        module.def_function[nystroem_transform_binding]("nystroem_transform")
        module.def_function[rbf_sampler_fit_binding]("rbf_sampler_fit")
        module.def_function[rbf_sampler_transform_binding]("rbf_sampler_transform")
        return module.finalize()
    except e:
        abort(String("failed to create _mojolearn_kernel_methods_host: ", e))
