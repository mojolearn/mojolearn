# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_gp` family, exact dense Gaussian process
regression and the Cholesky door it carries (workstream E, the gp host
lane, 2026-09-14).

HOST ONLY. No DeviceContext, no kernel launch, no GPU. The fit and predict
are `gaussian_process/host/gpr_oracle.mojo::gpr_host_fit` and
`gpr_host_predict`, the device path of `gaussian_process/estimator.mojo`
restated on the host; the factorization and the solve are
`cholesky/host/chol_oracle.mojo`, the profile of `cholesky/checks/potrf.mojo`
and `trsm.mojo` restated on the host; the posterior mean is
`gemm/host/gemm_oracle.mojo::gemm_oracle`, the gemm profile's normative
answer. So `L_`, `alpha_`, `log_marginal_likelihood_value_`, the mean and
the std are meant to be the GPU columns' bytes.

THE EXPORTED NAMES ARE THE GPU BINDING'S NAMES, with the GPU binding's
address and params contract word for word (`bindings/_mojolearn_gp.mojo`,
mirrored in `python/mojolearn/_gp_impl.py` and `_cholesky_impl.py`), so
`GaussianProcessRegressor` and `Cholesky` run unchanged on a CPU-only
install through `_backend._HOST_MODULES` (`"_mojolearn_gp":
"_mojolearn_gp_host"`): `gpr_fit` (9 addresses, 5 params), `gpr_predict`
(12 addresses, 7 params), `cholesky_profile_jitter`, `cholesky_factor`
(3 addresses, 2 params), `cholesky_solve` (3 addresses, 6 params),
`gp_vendor` answering "cpu" and `gp_numeric_mode`. ABSENT, and so refused
BY NAME through `_HostBinding`: `gp_parallel_available`, the ordered
multi-GPU driver's probe (the par-gp lane's row-sharded kernel matrix is a
device schedule).

WHAT IS REFUSED BELOW THE PYTHON SURFACE, BY NAME, AS ON THE DEVICE: a
Matern nu outside {0.5, 1.5, 2.5}, a non-positive or non-finite length
scale, a negative or NaN constant or noise level, a node count or operand
stack past the pinned capacity, an unpinned alpha or jitter under
IDENTICAL, non-finite X or y, a y of the wrong length, a non-symmetric
matrix at the door, and a predict or solve against a failed factorization.
The Python surface refuses the rest before a binding is reached (the
optimizer, n_restarts_optimizer, normalize_y, copy_X_train=False,
n_targets, random_state and return_cov). `gpr_sample_y` below is the
internal CPU verifier of `sample_y` (DEVIATION 2793), not a public saved-model
surface.

The sabotage arm (`gp_host_sabotage`) is
`gaussian_process/host/gpr_oracle.mojo::GPR_ORACLE_HOST_SABOTAGE`: the scaled
squared distance walks the feature axis descending, and the gemm oracle's
leaf walks descending with it, so every fit and predict this binary serves
differs.
"""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from bindings.hostptr import f32_ptr, f64_ptr, i32_ptr, read_f32, read_i32
from gaussian_process.host.gp_theta import (
    gp_log64,
    gp_restart_uniform,
    gp_theta_param,
)
from gaussian_process.host.gpr_grad_oracle import gpr_host_lml_grad
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from cholesky.host.chol_oracle import (
    CholHostFactor,
    chol_host_jitter_pinned,
    chol_host_potrf,
    chol_host_solve,
)
from bindings.gp_host_predict import (
    _rebuild_kernel_spec,
    gpc_predict_binding,
    gpr_predict_binding,
)
from gaussian_process.host.gpr_oracle import (
    GPR_ORACLE_HOST_SABOTAGE,
    GPHostKernelSpec,
    gpr_host_fit,
)
from gaussian_process.host.sample_y_oracle import gpr_host_sample_y
# Gaussian process classification (lane/gaussian-process-classifier,
# 2026-09-15): the GPU binding's gpc_fit and gpc_predict, same contract.
from gaussian_process.host.gpc_oracle import gpc_host_fit


def gp_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def gp_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def gp_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu".
    THE COLUMN IS THE CPU COLUMN, OR THIS DOES NOT BUILD; the assert lives
    in a function body PyInit registers, so it is compiled in every build.
    `bindings/build_host_family.sh` passes -D MOJOLEARN_COLUMN_CPU."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "gp host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_gp_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


# There is no `gp_host_detected_column` read-back, for the reason 8d16ce2f
# removed it from the forest and byte LM host bindings: the detected column
# folds to the GPU of the machine that ran the build. The comptime assert
# above is the check.


def gp_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary walks the scaled distance's feature axis
    descending on purpose (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's negative
    control; `gpr_oracle.mojo::GPR_ORACLE_HOST_SABOTAGE`)."""
    return PythonObject(GPR_ORACLE_HOST_SABOTAGE)


# The GPU binding's names, same contract.


def gp_vendor_binding() raises -> PythonObject:
    """"cpu". On a CPU-only install `_backend.vendor()` is "cpu" and the
    read-back cross-check expects that string from every host binding."""
    return PythonObject(String("cpu"))


def gp_numeric_mode_binding() raises -> PythonObject:
    """The build's tier code; `_gp_impl.py::_extension` and
    `_cholesky_impl.py` refuse a binary that disagrees with the requested
    mode. A host binding is IDENTICAL only, so this answers 1."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def _gpr_fit_run(
    x: List[Float32],
    y: List[Float32],
    spec: GPHostKernelSpec,
    n_train: Int,
    n_features: Int,
    alpha: Float32,
    lp: MutPointer[Float32, MutUntrackedOrigin],
    dp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    """The GIL-free half of `gpr_fit_binding`."""
    var model = gpr_host_fit(x, n_train, n_features, y, spec, alpha)
    for i in range(n_train * n_train):
        lp.unsafe_store(i, model.l[i])
    for i in range(n_train):
        dp.unsafe_store(i, model.dual[i])
    # info, nb, logdet, ydotalpha, lml -- the GPU binding's order; each
    # float32 widens to float64 exactly.
    sp.unsafe_store(0, Float64(model.info))
    sp.unsafe_store(1, Float64(model.nb))
    sp.unsafe_store(2, Float64(model.logdet))
    sp.unsafe_store(3, Float64(model.ydotalpha))
    sp.unsafe_store(4, Float64(model.lml))
    var info = model.info
    _ = model^
    return info


def gpr_fit_binding(
    addrs: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`GaussianProcessRegressor(kernel, alpha).fit(X, y)` on the host.
    Returns LAPACK's `info` (DEVIATION 1634).

    `addrs`, in the GPU binding's order: 0 x, 1 y, 2 kinds, 3 kparams,
    4 ls_len, 5 ls, 6 l_out, 7 dual_out, 8 scalars_out (info, nb, logdet,
    ydotalpha, lml). `params`: 0 n_train, 1 n_features, 2 n_nodes, 3 n_ls,
    4 alpha (unclamped, so the pin refuses by name)."""
    if len(addrs) != 9:
        raise Error(
            "gpr_fit: addrs must contain 9 addresses (x, y, kinds, kparams,"
            " ls_len, ls, l_out, dual_out, scalars_out), got "
            + String(len(addrs))
        )
    if len(params) != 5:
        raise Error(
            "gpr_fit: params must contain 5 values (n_train, n_features,"
            " n_nodes, n_ls, alpha), got "
            + String(len(params))
        )
    var lp = f32_ptr(Int(py=addrs[6]))
    var dp = f32_ptr(Int(py=addrs[7]))
    var sp = f64_ptr(Int(py=addrs[8]))
    var n_train = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_nodes = Int(py=params[2])
    var n_ls = Int(py=params[3])
    var alpha = Float32(Float64(py=params[4]))
    var spec = _rebuild_kernel_spec(
        Int(py=addrs[2]),
        Int(py=addrs[3]),
        Int(py=addrs[4]),
        Int(py=addrs[5]),
        n_nodes,
        n_ls,
        String("gpr_fit"),
    )
    var x = read_f32(Int(py=addrs[0]), max(0, n_train * n_features))
    var y = read_f32(Int(py=addrs[1]), max(0, n_train))
    var info = 0
    with GILReleased(Python()):
        info = _gpr_fit_run(x, y, spec, n_train, n_features, alpha, lp, dp, sp)
    _ = x^
    _ = y^
    _ = spec^
    return PythonObject(info)


def gpr_sample_y_binding(
    addrs: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`sample_y(X, n_samples, random_state)` on the host, the GPU binding's
    name and contract: `addrs` 0 xtrain, 1 l, 2 dual, 3 xstar, 4 kinds,
    5 kparams, 6 ls_len, 7 ls, 8 y_out (n_star * n_samples float32);
    `params` 0 n_train, 1 n_features, 2 n_star, 3 n_nodes, 4 n_ls, 5 info
    (passed through), 6 n_samples, 7 random_state low 32 bits, 8 high 32
    bits. The arithmetic is `gaussian_process/host/sample_y_oracle.mojo::
    gpr_host_sample_y` (DEVIATION 2793). An internal verifier arm: public
    CPU sample_y from a saved model belongs to
    lane/inference-neighbors-density. Returns n_samples."""
    if len(addrs) != 9:
        raise Error(
            "gpr_sample_y: addrs must contain 9 addresses (xtrain, l, dual,"
            " xstar, kinds, kparams, ls_len, ls, y_out), got "
            + String(len(addrs))
        )
    if len(params) != 9:
        raise Error(
            "gpr_sample_y: params must contain 9 values (n_train, n_features,"
            " n_star, n_nodes, n_ls, info, n_samples, seed_lo, seed_hi), got "
            + String(len(params))
        )
    var yp = f32_ptr(Int(py=addrs[8]))
    var n_train = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_star = Int(py=params[2])
    var n_nodes = Int(py=params[3])
    var n_ls = Int(py=params[4])
    var info = Int(py=params[5])
    var n_samples = Int(py=params[6])
    var seed = (UInt64(Int(py=params[8])) << 32) | UInt64(Int(py=params[7]))
    var spec = _rebuild_kernel_spec(
        Int(py=addrs[4]),
        Int(py=addrs[5]),
        Int(py=addrs[6]),
        Int(py=addrs[7]),
        n_nodes,
        n_ls,
        String("gpr_sample_y"),
    )
    var xt = read_f32(Int(py=addrs[0]), max(0, n_train * n_features))
    var l = read_f32(Int(py=addrs[1]), max(0, n_train * n_train))
    var dual = read_f32(Int(py=addrs[2]), max(0, n_train))
    var x_star = read_f32(Int(py=addrs[3]), max(0, n_star * n_features))
    with GILReleased(Python()):
        var y = gpr_host_sample_y(
            xt, l, dual, n_train, n_features, spec, info, x_star, n_star,
            n_samples, seed,
        )
        for i in range(n_star * n_samples):
            yp.unsafe_store(i, y[i])
        _ = y^
    _ = xt^
    _ = l^
    _ = dual^
    _ = x_star^
    _ = spec^
    return PythonObject(n_samples)


# ===========================================================================
# GAUSSIAN PROCESS CLASSIFICATION, on the host
# (bindings/_mojolearn_gp.mojo::gpc_fit_binding and gpc_predict_binding,
# their address and params orders word for word).
# ===========================================================================


def _gpc_fit_run(
    x: List[Float32],
    y: List[Float32],
    spec: GPHostKernelSpec,
    n_train: Int,
    n_features: Int,
    max_iter_predict: Int,
    lp: MutPointer[Float32, MutUntrackedOrigin],
    pp: MutPointer[Float32, MutUntrackedOrigin],
    wp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    """The GIL-free half of `gpc_fit_binding`."""
    var fit = gpc_host_fit(x, n_train, n_features, y, spec, max_iter_predict)
    for i in range(n_train * n_train):
        lp.unsafe_store(i, fit.l[i])
    for i in range(n_train):
        pp.unsafe_store(i, fit.pi[i])
        wp.unsafe_store(i, fit.wsr[i])
    sp.unsafe_store(0, Float64(fit.lml))
    sp.unsafe_store(1, Float64(fit.n_iter))
    sp.unsafe_store(2, Float64(fit.nb))
    return fit.n_iter


def gpc_fit_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """One binary Laplace fit on the host. `addrs`: 0 x, 1 y, 2 kinds,
    3 kparams, 4 ls_len, 5 ls, 6 l_out, 7 pi_out, 8 wsr_out, 9 scalars_out
    (lml, n_iter, nb). `params`: 0 n_train, 1 n_features, 2 n_nodes, 3 n_ls,
    4 max_iter_predict. Returns the iteration count."""
    if len(addrs) != 10:
        raise Error(
            "gpc_fit: addrs must contain 10 addresses (x, y, kinds, kparams,"
            " ls_len, ls, l_out, pi_out, wsr_out, scalars_out), got "
            + String(len(addrs))
        )
    if len(params) != 5:
        raise Error(
            "gpc_fit: params must contain 5 values (n_train, n_features,"
            " n_nodes, n_ls, max_iter_predict), got "
            + String(len(params))
        )
    var lp = f32_ptr(Int(py=addrs[6]))
    var pp = f32_ptr(Int(py=addrs[7]))
    var wp = f32_ptr(Int(py=addrs[8]))
    var sp = f64_ptr(Int(py=addrs[9]))
    var n_train = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_nodes = Int(py=params[2])
    var n_ls = Int(py=params[3])
    var max_iter_predict = Int(py=params[4])
    var spec = _rebuild_kernel_spec(
        Int(py=addrs[2]),
        Int(py=addrs[3]),
        Int(py=addrs[4]),
        Int(py=addrs[5]),
        n_nodes,
        n_ls,
        String("gpc_fit"),
    )
    var x = read_f32(Int(py=addrs[0]), max(0, n_train * n_features))
    var y = read_f32(Int(py=addrs[1]), max(0, n_train))
    var n_iter = 0
    with GILReleased(Python()):
        n_iter = _gpc_fit_run(
            x, y, spec, n_train, n_features, max_iter_predict, lp, pp, wp, sp
        )
    _ = x^
    _ = y^
    _ = spec^
    return PythonObject(n_iter)


# ===========================================================================
# KERNEL HYPERPARAMETER OPTIMIZATION, on the host
# (bindings/_mojolearn_gp.mojo's four entries, their contracts word for word;
# lane/gp-optimizer, 2026-09-15).
# ===========================================================================


def gpr_lml_grad_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`gaussian_process/host/gpr_grad_oracle.mojo::gpr_host_lml_grad`.
    `addrs`: 0 x, 1 y, 2 kinds, 3 kparams, 4 ls_len, 5 ls, 6 free (int32 per
    node), 7 grad_out (float64), 8 scalars_out (info, lml). `params`:
    0 n_train, 1 n_features, 2 n_nodes, 3 n_ls, 4 alpha. Returns info."""
    if len(addrs) != 9:
        raise Error(
            "gpr_lml_grad: addrs must contain 9 addresses (x, y, kinds,"
            " kparams, ls_len, ls, free, grad_out, scalars_out), got "
            + String(len(addrs))
        )
    if len(params) != 5:
        raise Error(
            "gpr_lml_grad: params must contain 5 values (n_train, n_features,"
            " n_nodes, n_ls, alpha), got "
            + String(len(params))
        )
    var n_train = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_nodes = Int(py=params[2])
    var n_ls = Int(py=params[3])
    var alpha = Float32(Float64(py=params[4]))
    var spec = _rebuild_kernel_spec(
        Int(py=addrs[2]), Int(py=addrs[3]), Int(py=addrs[4]), Int(py=addrs[5]),
        n_nodes, n_ls, String("gpr_lml_grad"),
    )
    var free = read_i32(Int(py=addrs[6]), max(0, n_nodes))
    var gp = f64_ptr(Int(py=addrs[7]))
    var sp = f64_ptr(Int(py=addrs[8]))
    var x = read_f32(Int(py=addrs[0]), max(0, n_train * n_features))
    var y = read_f32(Int(py=addrs[1]), max(0, n_train))
    var info = 0
    with GILReleased(Python()):
        var r = gpr_host_lml_grad(x, n_train, n_features, y, spec, free, alpha)
        for i in range(len(r.grad)):
            gp.unsafe_store(i, Float64(r.grad[i]))
        sp.unsafe_store(0, Float64(r.info))
        sp.unsafe_store(1, Float64(r.lml))
        info = r.info
        _ = r^
    _ = x^
    _ = y^
    _ = free^
    _ = spec^
    return PythonObject(info)


def gp_log64_binding(values: PythonObject) raises -> PythonObject:
    """The GPU binding's `gp_log64`."""
    var out = Python.list()
    for i in range(len(values)):
        out.append(PythonObject(gp_log64(Float64(py=values[i]))))
    return out


def gp_theta_params_binding(values: PythonObject) raises -> PythonObject:
    """The GPU binding's `gp_theta_params`."""
    var out = Python.list()
    for i in range(len(values)):
        out.append(PythonObject(Float64(gp_theta_param(Float64(py=values[i])))))
    return out


def gp_restart_uniforms_binding(params: PythonObject) raises -> PythonObject:
    """The GPU binding's `gp_restart_uniforms`."""
    if len(params) != 4:
        raise Error(
            "gp_restart_uniforms: params must contain 4 values (n_restarts,"
            " n_dims, seed_lo, seed_hi), got " + String(len(params))
        )
    var nr = Int(py=params[0])
    var nd = Int(py=params[1])
    var seed = (UInt64(Int(py=params[3])) << 32) | UInt64(Int(py=params[2]))
    var out = Python.list()
    for r in range(nr):
        for j in range(nd):
            out.append(PythonObject(gp_restart_uniform(seed, r, j)))
    return out


# ===========================================================================
# THE CHOLESKY DOOR, on the host (the GPU binding's workstream D entries).
# ===========================================================================


def cholesky_profile_jitter_binding() raises -> PythonObject:
    """The profile's pinned ridge, 2^-20, as a Python float."""
    return PythonObject(Float64(chol_host_jitter_pinned()))


def _cholesky_factor_run(
    a: List[Float32],
    n: Int,
    jitter: Float32,
    lp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    """The GIL-free half of `cholesky_factor_binding`."""
    var f = chol_host_potrf(a, n, jitter)
    for i in range(n * n):
        lp.unsafe_store(i, f.l[i])
    # info, nb, logdet, jitter -- the GPU binding's order.
    sp.unsafe_store(0, Float64(f.info))
    sp.unsafe_store(1, Float64(f.nb))
    sp.unsafe_store(2, Float64(f.logdet))
    sp.unsafe_store(3, Float64(f.jitter))
    var info = f.info
    _ = f^
    return info


def cholesky_factor_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`cholesky_factor_host(a, n, jitter)` on the host. `addrs`: 0 a,
    1 l_out, 2 scalars_out (info, nb, logdet, jitter). `params`: 0 n,
    1 jitter (unclamped). Returns `info`."""
    if len(addrs) != 3:
        raise Error(
            "cholesky_factor: addrs must contain 3 addresses (a, l_out,"
            " scalars_out), got "
            + String(len(addrs))
        )
    if len(params) != 2:
        raise Error(
            "cholesky_factor: params must contain 2 values (n, jitter), got "
            + String(len(params))
        )
    var lp = f32_ptr(Int(py=addrs[1]))
    var sp = f64_ptr(Int(py=addrs[2]))
    var n = Int(py=params[0])
    var jitter = Float32(Float64(py=params[1]))
    var a = read_f32(Int(py=addrs[0]), max(0, n * n))
    var info = 0
    with GILReleased(Python()):
        info = _cholesky_factor_run(a, n, jitter, lp, sp)
    _ = a^
    return PythonObject(info)


def cholesky_solve_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`cholesky_solve_host(factor, b, nrhs)` on the host. `addrs`: 0 l,
    1 b, 2 x_out. `params`: 0 n, 1 nrhs, 2 info (passed through), 3 nb,
    4 logdet, 5 jitter. Returns 0."""
    if len(addrs) != 3:
        raise Error(
            "cholesky_solve: addrs must contain 3 addresses (l, b, x_out),"
            " got "
            + String(len(addrs))
        )
    if len(params) != 6:
        raise Error(
            "cholesky_solve: params must contain 6 values (n, nrhs, info,"
            " nb, logdet, jitter), got "
            + String(len(params))
        )
    var xp = f32_ptr(Int(py=addrs[2]))
    var n = Int(py=params[0])
    var nrhs = Int(py=params[1])
    var info = Int(py=params[2])
    var nb = Int(py=params[3])
    var logdet = Float32(Float64(py=params[4]))
    var jitter = Float32(Float64(py=params[5]))
    var l = read_f32(Int(py=addrs[0]), max(0, n * n))
    var b = read_f32(Int(py=addrs[1]), max(0, n * nrhs))
    var factor = CholHostFactor(l^, n, info, logdet, nb, jitter)
    with GILReleased(Python()):
        var x = chol_host_solve(factor, b, nrhs)
        for i in range(n * nrhs):
            xp.unsafe_store(i, x[i])
        _ = x^
    _ = factor^
    _ = b^
    return PythonObject(0)


@export
def PyInit__mojolearn_gp_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_gp_host")
        module.def_function[gp_host_numeric_mode_binding]("gp_host_numeric_mode")
        module.def_function[gp_host_vendor_binding]("gp_host_vendor")
        module.def_function[gp_host_column_binding]("gp_host_column")
        module.def_function[gp_host_sabotage_binding]("gp_host_sabotage")
        module.def_function[gp_vendor_binding]("gp_vendor")
        module.def_function[gp_numeric_mode_binding]("gp_numeric_mode")
        module.def_function[gpr_fit_binding]("gpr_fit")
        module.def_function[gpr_predict_binding]("gpr_predict")
        module.def_function[gpr_sample_y_binding]("gpr_sample_y")
        module.def_function[gpr_lml_grad_binding]("gpr_lml_grad")
        module.def_function[gp_log64_binding]("gp_log64")
        module.def_function[gp_theta_params_binding]("gp_theta_params")
        module.def_function[gp_restart_uniforms_binding]("gp_restart_uniforms")
        module.def_function[gpc_fit_binding]("gpc_fit")
        module.def_function[gpc_predict_binding]("gpc_predict")
        module.def_function[cholesky_profile_jitter_binding]("cholesky_profile_jitter")
        module.def_function[cholesky_factor_binding]("cholesky_factor")
        module.def_function[cholesky_solve_binding]("cholesky_solve")
        return module.finalize()
    except e:
        abort(String("failed to create _mojolearn_gp_host: ", e))
