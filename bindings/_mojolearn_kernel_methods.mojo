# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the kernel methods lane: KernelRidge, Nystroem,
RBFSampler (workstream D, 2026-09-14).

A separate extension module, for the reason `bindings/_mojolearn_gp.mojo`
gives: an independently changing binding must not become a merge point.
`kernel_methods/estimator.mojo`'s three host surfaces are reached here and
nothing is re-decided: every refusal lives one layer down and is raised
there by name (`km_validate_matrix`, `km_validate_kernel_params`, the
alpha, gamma, degree and n_components refusals, DEVIATION 1686), and this
file refuses only a null address and a list of the wrong length.

THE ABI IS THE GP'S. Each entry point takes exactly TWO arguments, an
address list and a params list, each length-checked, with the order written
out in the docstring and mirrored at the `python/mojolearn/kernel_methods.py`
call site. `bindings/_mojolearn_gp.mojo`'s header records why (def_function
stops inferring a signature above about nine arguments, measured
2026-09-01).

THE MODEL CROSSES AS ITS ARRAYS, NOT AS A HANDLE. `KernelRidgeModel`,
`NystroemModel` and `RBFSamplerModel` are `fieldwise_init` structs of host
lists and scalars; fit writes every field into caller-sized buffers and
predict/transform rebuild the struct from the same buffers. That is what
lets the Python side keep the model as attributes a user can read, save
and hash (the identity harness's model column), and it is what keeps
every parameter the model carries (`kernel_methods/estimator.mojo`'s
header, point 1) visible from Python.

THE GIL is released around every device call, and nothing inside a
`GILReleased` block touches a `PythonObject`.
"""

from std.os import abort
from bindings.hostptr import f32_ptr, f64_ptr, i32_ptr, copy_f32, read_f32, read_i32
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.identity_trace import IdentityTrace
from kernel_methods.estimator import (
    KernelRidgeModel,
    NystroemModel,
    RBFSamplerModel,
    kernel_ridge_fit_host,
    kernel_ridge_predict_host,
    nystroem_fit_host,
    nystroem_transform_host_into,
    rbf_sampler_fit_host,
    rbf_sampler_transform_host_into,
)
from svm.impl.svm_parameter import KernelParams


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    return f32_ptr(addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    return i32_ptr(addr)


def _f64_ptr(addr: Int) raises -> MutPointer[Float64, MutUntrackedOrigin]:
    return f64_ptr(addr)


def kernel_methods_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER, as the `NUMERIC_*` code itself: 0 FAST, 1
    IDENTICAL, 2 DETERMINISTIC (the GP's shape, for its reason)."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def kernel_methods_vendor_binding() raises -> PythonObject:
    """THE ACCELERATOR API THIS BINARY WAS COMPILED FOR: 'metal', 'cuda',
    'hip' or 'none', from `checks/vendor.mojo`."""
    return PythonObject(String(COMPILED_VENDOR))


def kernel_methods_rows_parallel_available() raises -> PythonObject:
    """1: km_kernel_matrix reaches svm kernel_op, which reads
    MOJOLEARN_SVM_DEVICE_COUNT, and potrf_lower/cho_solve read
    MOJOLEARN_CHOLESKY_DEVICE_COUNT; parallel_classical.fit_kernel_method and
    apply_kernel_method set both inside their cooperative worker."""
    return PythonObject(1)


# ===========================================================================
# KernelRidge
# ===========================================================================


def _kernel_ridge_fit_run(
    x: List[Float32],
    y: List[Float32],
    n: Int,
    d: Int,
    t: Int,
    kp: KernelParams,
    alpha: Float32,
    dp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    var trace = IdentityTrace()
    var model = kernel_ridge_fit_host(x, y, n, d, t, kp, alpha, trace)
    copy_f32(model.dual_coef.unsafe_ptr(), dp, n * t)
    sp.unsafe_store(0, Float64(model.info))
    return model.info


def kernel_ridge_fit_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`KernelRidge(alpha, kernel, ...).fit(X, y)`
    (`kernel_ridge_fit_host`). Returns `info`, always 0 on a model that
    came back (the fit REFUSES a non-zero one by name, DEVIATION 1662).

    `addrs`, in this exact order:

        0  x               n * d float32, row-major, read
        1  y               n * t float32, row-major, read
        2  dual_out        n * t float32, WRITTEN (`dual_coef_`)
        3  scalars_out     1 float64, WRITTEN: info

    `params`, in this exact order:

        0  n
        1  d
        2  t               n_targets
        3  kernel          KM_KERNEL_* code
        4  degree
        5  gamma           (float)
        6  coef0           (float)
        7  alpha           (float; the ridge, DEVIATION 1660, UNCLAMPED so
                            the negative and NaN refusals fire by name)
    """
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
    var xp = _f32_ptr(Int(py=addrs[0]))
    var yp = _f32_ptr(Int(py=addrs[1]))
    var dp = _f32_ptr(Int(py=addrs[2]))
    var sp = _f64_ptr(Int(py=addrs[3]))
    var n = Int(py=params[0])
    var d = Int(py=params[1])
    var t = Int(py=params[2])
    var kp = KernelParams(
        Int(py=params[3]),
        Int(py=params[4]),
        Float64(py=params[5]),
        Float64(py=params[6]),
    )
    var alpha = Float32(Float64(py=params[7]))
    var x = read_f32(Int(xp), max(0, n * d))
    var y = read_f32(Int(yp), max(0, n * t))
    var info = 0
    with GILReleased(Python()):
        info = _kernel_ridge_fit_run(x, y, n, d, t, kp, alpha, dp, sp)
    return PythonObject(info)


def _kernel_ridge_predict_run(
    model: KernelRidgeModel,
    x_new: List[Float32],
    q: Int,
    op: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    var trace = IdentityTrace()
    var out = kernel_ridge_predict_host(model, x_new, q, trace)
    copy_f32(out.unsafe_ptr(), op, q * model.n_targets)


def kernel_ridge_predict_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`KernelRidge.predict(X)` (`kernel_ridge_predict_host`): `K(X, X_fit)
    . dual_coef` through the identical GEMM at OP_NN (DEVIATION 1680).
    Returns 0.

    `addrs`, in this exact order:

        0  x_fit           n * d float32, read (the fit's X)
        1  dual            n * t float32, read (the fit's dual_out)
        2  x_new           q * d float32, read
        3  out             q * t float32, WRITTEN

    `params`, in this exact order:

        0  n
        1  d
        2  t
        3  kernel
        4  degree
        5  gamma
        6  coef0
        7  alpha           carried on the model, by value
        8  info            the fit's, PASSED THROUGH unjudged
        9  q               n_query
    """
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
    var xfp = _f32_ptr(Int(py=addrs[0]))
    var dp = _f32_ptr(Int(py=addrs[1]))
    var xnp = _f32_ptr(Int(py=addrs[2]))
    var op = _f32_ptr(Int(py=addrs[3]))
    var n = Int(py=params[0])
    var d = Int(py=params[1])
    var t = Int(py=params[2])
    var kernel = Int(py=params[3])
    var degree = Int(py=params[4])
    var gamma = Float64(py=params[5])
    var coef0 = Float64(py=params[6])
    var alpha = Float32(Float64(py=params[7]))
    var info = Int(py=params[8])
    var q = Int(py=params[9])
    var x_fit = read_f32(Int(xfp), max(0, n * d))
    var dual = read_f32(Int(dp), max(0, n * t))
    var x_new = read_f32(Int(xnp), max(0, q * d))
    var model = KernelRidgeModel(
        dual^, x_fit^, n, d, t, kernel, degree, gamma, coef0, alpha, info
    )
    with GILReleased(Python()):
        _kernel_ridge_predict_run(model, x_new, q, op)
    return PythonObject(0)


# ===========================================================================
# Nystroem
# ===========================================================================


def _nystroem_fit_run(
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


def nystroem_fit_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`Nystroem(kernel, n_components, random_state).fit(X)`
    (`nystroem_fit_host`). Returns `sweeps`, the Jacobi sweep count, which
    is PART OF THE MODEL (`NystroemModel.sweeps`): two fits that disagree
    on it are not comparable below that stage.

    `addrs`, in this exact order:

        0  x                  n * d float32, read
        1  components_out     q * d float32, WRITTEN
        2  indices_out        q int32, WRITTEN (row ids, in rank order)
        3  normalization_out  q * q float32, WRITTEN (NOT bitwise
                               symmetric, DEVIATION 1674)
        4  eigenvalues_out    q float32, WRITTEN (singular values |lambda|, descending, clipped)
        5  eigenvectors_out   q * q float32, WRITTEN
        6  scalars_out        1 float64, WRITTEN: sweeps

    `params`, in this exact order:

        0  n
        1  d
        2  kernel
        3  degree
        4  gamma
        5  coef0
        6  q               n_components
        7  seed
    """
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
    var xp = _f32_ptr(Int(py=addrs[0]))
    var cp = _f32_ptr(Int(py=addrs[1]))
    var ip = _i32_ptr(Int(py=addrs[2]))
    var np_ = _f32_ptr(Int(py=addrs[3]))
    var evp = _f32_ptr(Int(py=addrs[4]))
    var ecp = _f32_ptr(Int(py=addrs[5]))
    var sp = _f64_ptr(Int(py=addrs[6]))
    var n = Int(py=params[0])
    var d = Int(py=params[1])
    var kp = KernelParams(
        Int(py=params[2]),
        Int(py=params[3]),
        Float64(py=params[4]),
        Float64(py=params[5]),
    )
    var q = Int(py=params[6])
    var seed = UInt64(Int(py=params[7]))
    var x = read_f32(Int(xp), max(0, n * d))
    var sweeps = 0
    with GILReleased(Python()):
        sweeps = _nystroem_fit_run(
            x, n, d, kp, q, seed, cp, ip, np_, evp, ecp, sp
        )
    return PythonObject(sweeps)


def _nystroem_transform_run(
    model: NystroemModel,
    x: List[Float32],
    m: Int,
    op: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    var trace = IdentityTrace()
    nystroem_transform_host_into(model, x, m, op, trace)


def nystroem_transform_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`Nystroem.transform(X)` (`nystroem_transform_host`): `K(X,
    components) @ normalization.T`, the transposed arm (DEVIATION 1674).
    Returns 0.

    `addrs`, in this exact order:

        0  components      q * d float32, read
        1  indices         q int32, read
        2  normalization   q * q float32, read
        3  eigenvalues     q float32, read
        4  eigenvectors    q * q float32, read
        5  x               m * d float32, read
        6  out             m * q float32, WRITTEN

    `params`, in this exact order:

        0  q
        1  d
        2  kernel
        3  degree
        4  gamma
        5  coef0
        6  seed
        7  sweeps          the fit's, carried on the model
        8  m               n_rows
    """
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
    var seed = UInt64(Int(py=params[6]))
    var sweeps = Int(py=params[7])
    var m = Int(py=params[8])
    var components = read_f32(Int(py=addrs[0]), max(0, q * d))
    var indices = read_i32(Int(py=addrs[1]), max(0, q))
    var normalization = read_f32(Int(py=addrs[2]), max(0, q * q))
    var eigenvalues = read_f32(Int(py=addrs[3]), max(0, q))
    var eigenvectors = read_f32(Int(py=addrs[4]), max(0, q * q))
    var x = read_f32(Int(py=addrs[5]), max(0, m * d))
    var op = _f32_ptr(Int(py=addrs[6]))
    var model = NystroemModel(
        components^,
        indices^,
        normalization^,
        eigenvalues^,
        eigenvectors^,
        q,
        d,
        kernel,
        degree,
        gamma,
        coef0,
        seed,
        sweeps,
    )
    with GILReleased(Python()):
        _nystroem_transform_run(model, x, m, op)
    return PythonObject(0)


# ===========================================================================
# RBFSampler
# ===========================================================================


def _rbf_sampler_fit_run(
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


def rbf_sampler_fit_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`RBFSampler(gamma, n_components, random_state).fit(X)`
    (`rbf_sampler_fit_host`). IT DOES NOT LOOK AT X, and neither does
    scikit-learn's: the draws depend on `n_features`, `n_components` and
    the seed alone, so no X address crosses. Returns 0.

    `addrs`, in this exact order:

        0  weights_out     d * q float32, WRITTEN (`random_weights_`)
        1  offset_out      q float32, WRITTEN (`random_offset_`)
        2  scalars_out     2 float64, WRITTEN: sigma, scale (DEVIATION 1678)

    `params`, in this exact order:

        0  d               n_features
        1  q               n_components
        2  gamma           (float; UNCLAMPED, refused non-positive by name)
        3  seed
    """
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
    var wp = _f32_ptr(Int(py=addrs[0]))
    var bp = _f32_ptr(Int(py=addrs[1]))
    var sp = _f64_ptr(Int(py=addrs[2]))
    var d = Int(py=params[0])
    var q = Int(py=params[1])
    var gamma = Float32(Float64(py=params[2]))
    var seed = UInt64(Int(py=params[3]))
    with GILReleased(Python()):
        _rbf_sampler_fit_run(d, q, gamma, seed, wp, bp, sp)
    return PythonObject(0)


def _rbf_sampler_transform_run(
    model: RBFSamplerModel,
    x: List[Float32],
    m: Int,
    op: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    var trace = IdentityTrace()
    rbf_sampler_transform_host_into(model, x, m, op, trace)


def rbf_sampler_transform_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`RBFSampler.transform(X)` (`rbf_sampler_transform_host`):
    `scale * cos(X . W + b)`, the dot through the identical GEMM at OP_NN.
    Returns 0.

    `addrs`, in this exact order:

        0  weights         d * q float32, read
        1  offset          q float32, read
        2  x               m * d float32, read
        3  out             m * q float32, WRITTEN

    `params`, in this exact order:

        0  d
        1  q
        2  gamma
        3  seed
        4  sigma           the fit's, carried on the model (DEVIATION 1678)
        5  scale           the fit's, carried on the model
        6  m               n_rows
    """
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
    var gamma = Float32(Float64(py=params[2]))
    var seed = UInt64(Int(py=params[3]))
    var sigma = Float32(Float64(py=params[4]))
    var scale = Float32(Float64(py=params[5]))
    var m = Int(py=params[6])
    var weights = read_f32(Int(py=addrs[0]), max(0, d * q))
    var offset = read_f32(Int(py=addrs[1]), max(0, q))
    var x = read_f32(Int(py=addrs[2]), max(0, m * d))
    var op = _f32_ptr(Int(py=addrs[3]))
    var model = RBFSamplerModel(
        weights^, offset^, d, q, gamma, seed, sigma, scale
    )
    with GILReleased(Python()):
        _rbf_sampler_transform_run(model, x, m, op)
    return PythonObject(0)


@export
def PyInit__mojolearn_kernel_methods() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_kernel_methods")
        m.def_function[kernel_methods_rows_parallel_available](
            "kernel_methods_rows_parallel_available"
        )
        m.def_function[kernel_methods_vendor_binding]("kernel_methods_vendor")
        m.def_function[kernel_methods_numeric_mode_binding](
            "kernel_methods_numeric_mode"
        )
        m.def_function[kernel_ridge_fit_binding]("kernel_ridge_fit")
        m.def_function[kernel_ridge_predict_binding]("kernel_ridge_predict")
        m.def_function[nystroem_fit_binding]("nystroem_fit")
        m.def_function[nystroem_transform_binding]("nystroem_transform")
        m.def_function[rbf_sampler_fit_binding]("rbf_sampler_fit")
        m.def_function[rbf_sampler_transform_binding]("rbf_sampler_transform")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_kernel_methods: ", e))
