# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_gp` family, exact dense Gaussian process
regression and the Cholesky door it carries (workstream E, the gp host
lane, 2026-09-14; brief docs/lanes/BRIEF_cpu_training_2026-09-13.md
sections 1.1 and 3.2).

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
n_targets, random_state, return_cov and sample_y).

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

from bindings.hostptr import f32_ptr, f64_ptr, i32_ptr, read_f32
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
from gaussian_process.host.gpr_oracle import (
    GPR_K_CONST,
    GPR_K_MATERN,
    GPR_K_PROD,
    GPR_K_RBF,
    GPR_K_SUM,
    GPR_K_WHITE,
    GPR_ORACLE_HOST_SABOTAGE,
    GPHostKernelSpec,
    gpr_host_fit,
    gpr_host_kernel_const,
    gpr_host_kernel_matern,
    gpr_host_kernel_prod,
    gpr_host_kernel_rbf,
    gpr_host_kernel_sum,
    gpr_host_kernel_white,
    gpr_host_predict,
)
# Gaussian process classification (lane/gaussian-process-classifier,
# 2026-09-15): the GPU binding's gpc_fit and gpc_predict, same contract.
from gaussian_process.host.gpc_oracle import gpc_host_fit, gpc_host_predict
from gaussian_process.host.gpc_steps import gpc_proba


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


def _rebuild_kernel_spec(
    kinds_addr: Int,
    kparams_addr: Int,
    ls_len_addr: Int,
    ls_addr: Int,
    n_nodes: Int,
    n_ls: Int,
    what: String,
) raises -> GPHostKernelSpec:
    """`bindings/_mojolearn_gp.mojo::_rebuild_kernel_spec`, over the host
    constructors: the postfix list walked with a stack, every value handed
    to its constructor unjudged so the constructor's refusal fires by name,
    the offsets recomputed by the combine."""
    if n_nodes < 1:
        raise Error(
            what
            + ": the kernel spec must have at least one postfix node, got "
            + String(n_nodes)
        )
    if n_ls < 0:
        raise Error(what + ": n_ls cannot be negative, got " + String(n_ls))
    var kp = i32_ptr(kinds_addr)
    var pp = f32_ptr(kparams_addr)
    var lnp = i32_ptr(ls_len_addr)
    var tp = f32_ptr(ls_addr)
    var stack = List[GPHostKernelSpec]()
    var off = 0
    for t in range(n_nodes):
        var k = Int(kp.unsafe_load(t))
        var param = pp.unsafe_load(t)
        if k == GPR_K_CONST:
            stack.append(gpr_host_kernel_const(param))
        elif k == GPR_K_WHITE:
            stack.append(gpr_host_kernel_white(param))
        elif k == GPR_K_RBF or k == GPR_K_MATERN:
            var ln = Int(lnp.unsafe_load(t))
            if ln < 1:
                raise Error(
                    what
                    + ": node "
                    + String(t)
                    + " is an RBF or Matern leaf with ls_len "
                    + String(ln)
                    + "; a leaf consumes at least one length scale, so the"
                    " two sides of this boundary disagree about the spec"
                )
            if off + ln > n_ls:
                raise Error(
                    what
                    + ": node "
                    + String(t)
                    + " consumes length scales ["
                    + String(off)
                    + ", "
                    + String(off + ln)
                    + ") of a table holding "
                    + String(n_ls)
                    + "; the two sides of this boundary disagree about the"
                    " table"
                )
            var leaf_ls = List[Float32]()
            for i in range(ln):
                leaf_ls.append(tp.unsafe_load(off + i))
            off += ln
            if k == GPR_K_RBF:
                stack.append(gpr_host_kernel_rbf(leaf_ls))
            else:
                stack.append(gpr_host_kernel_matern(leaf_ls, param))
        elif k == GPR_K_SUM or k == GPR_K_PROD:
            if len(stack) < 2:
                raise Error(
                    what
                    + ": node "
                    + String(t)
                    + " combines a stack of "
                    + String(len(stack))
                    + " operands; the postfix expression is malformed"
                )
            var b = stack.pop()
            var a = stack.pop()
            if k == GPR_K_SUM:
                stack.append(gpr_host_kernel_sum(a, b))
            else:
                stack.append(gpr_host_kernel_prod(a, b))
        else:
            raise Error(
                what
                + ": node "
                + String(t)
                + " has unknown kind "
                + String(k)
                + ". The GP_K_* codes are 0 CONST, 1 WHITE, 2 RBF,"
                " 3 MATERN, 4 SUM, 5 PROD, mirrored in _gp_impl.py"
            )
    if len(stack) != 1:
        raise Error(
            what
            + ": the postfix expression leaves "
            + String(len(stack))
            + " operands on the stack; a well-formed kernel leaves exactly"
            " one"
        )
    if off != n_ls:
        raise Error(
            what
            + ": the leaves consumed "
            + String(off)
            + " length scales of the "
            + String(n_ls)
            + " sent; the two sides of this boundary disagree about the"
            " table"
        )
    return stack.pop()


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


def _gpr_predict_run(
    xt: List[Float32],
    l: List[Float32],
    dual: List[Float32],
    spec: GPHostKernelSpec,
    x_star: List[Float32],
    n_train: Int,
    n_features: Int,
    n_star: Int,
    info: Int,
    return_std: Bool,
    mean_addr: Int,
    var_addr: Int,
    std_addr: Int,
    clamped_addr: Int,
) raises -> Int:
    """The GIL-free half of `gpr_predict_binding`. The variance, std and
    clamp addresses are resolved only in the `return_std` arm, as on the
    GPU binding."""
    var pred = gpr_host_predict(
        xt, l, dual, n_train, n_features, spec, info, x_star, n_star, return_std
    )
    var mp = f32_ptr(mean_addr)
    for i in range(n_star):
        mp.unsafe_store(i, pred.mean[i])
    if return_std:
        var vp = f32_ptr(var_addr)
        var stp = f32_ptr(std_addr)
        var cp = i32_ptr(clamped_addr)
        for i in range(n_star):
            vp.unsafe_store(i, pred.variance[i])
            stp.unsafe_store(i, pred.std[i])
            cp.unsafe_store(i, pred.clamped[i])
    var n_clamped = pred.n_clamped
    _ = pred^
    return n_clamped


def gpr_predict_binding(
    addrs: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`predict(X_star, return_std)` on the host. Returns `n_clamped`
    (DEVIATION 1760).

    `addrs`, in the GPU binding's order: 0 xtrain, 1 l, 2 dual, 3 xstar,
    4 kinds, 5 kparams, 6 ls_len, 7 ls, 8 mean_out, 9 var_out, 10 std_out,
    11 clamped_out. `params`: 0 n_train, 1 n_features, 2 n_star, 3 n_nodes,
    4 n_ls, 5 return_std, 6 info (passed through, so the refusal to predict
    from a failed fit fires by name)."""
    if len(addrs) != 12:
        raise Error(
            "gpr_predict: addrs must contain 12 addresses (xtrain, l,"
            " dual, xstar, kinds, kparams, ls_len, ls, mean_out, var_out,"
            " std_out, clamped_out), got "
            + String(len(addrs))
        )
    if len(params) != 7:
        raise Error(
            "gpr_predict: params must contain 7 values (n_train,"
            " n_features, n_star, n_nodes, n_ls, return_std, info), got "
            + String(len(params))
        )
    var mean_addr = Int(py=addrs[8])
    var var_addr = Int(py=addrs[9])
    var std_addr = Int(py=addrs[10])
    var clamped_addr = Int(py=addrs[11])
    var n_train = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_star = Int(py=params[2])
    var n_nodes = Int(py=params[3])
    var n_ls = Int(py=params[4])
    var return_std = Int(py=params[5]) != 0
    var info = Int(py=params[6])
    var spec = _rebuild_kernel_spec(
        Int(py=addrs[4]),
        Int(py=addrs[5]),
        Int(py=addrs[6]),
        Int(py=addrs[7]),
        n_nodes,
        n_ls,
        String("gpr_predict"),
    )
    var xt = read_f32(Int(py=addrs[0]), max(0, n_train * n_features))
    var l = read_f32(Int(py=addrs[1]), max(0, n_train * n_train))
    var dual = read_f32(Int(py=addrs[2]), max(0, n_train))
    var x_star = read_f32(Int(py=addrs[3]), max(0, n_star * n_features))
    var n_clamped = 0
    with GILReleased(Python()):
        n_clamped = _gpr_predict_run(
            xt,
            l,
            dual,
            spec,
            x_star,
            n_train,
            n_features,
            n_star,
            info,
            return_std,
            mean_addr,
            var_addr,
            std_addr,
            clamped_addr,
        )
    _ = xt^
    _ = l^
    _ = dual^
    _ = x_star^
    _ = spec^
    return PythonObject(n_clamped)


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


def _gpc_predict_run(
    xt: List[Float32],
    y: List[Float32],
    pi: List[Float32],
    wsr: List[Float32],
    l: List[Float32],
    spec: GPHostKernelSpec,
    x_star: List[Float32],
    n_train: Int,
    n_features: Int,
    n_star: Int,
    want_proba: Bool,
    mean_addr: Int,
    var_addr: Int,
    proba_addr: Int,
) raises -> Int:
    """The GIL-free half of `gpc_predict_binding`."""
    var lat = gpc_host_predict(
        xt, y, pi, wsr, l, n_train, n_features, spec, x_star, n_star, want_proba
    )
    var mp = f32_ptr(mean_addr)
    for t in range(n_star):
        mp.unsafe_store(t, lat.mean[t])
    if want_proba:
        var p = gpc_proba(lat.mean, lat.variance)
        var vp = f32_ptr(var_addr)
        var pr = f64_ptr(proba_addr)
        for t in range(n_star):
            vp.unsafe_store(t, lat.variance[t])
            pr.unsafe_store(t, p[t])
    return 0


def gpc_predict_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """The latent mean (and with want_proba the variance and the class-1
    probability) on the host. `addrs`: 0 xtrain, 1 y, 2 pi, 3 wsr, 4 l,
    5 xstar, 6 kinds, 7 kparams, 8 ls_len, 9 ls, 10 mean_out, 11 var_out,
    12 proba_out. `params`: 0 n_train, 1 n_features, 2 n_star, 3 n_nodes,
    4 n_ls, 5 want_proba. Returns 0."""
    if len(addrs) != 13:
        raise Error(
            "gpc_predict: addrs must contain 13 addresses (xtrain, y, pi,"
            " wsr, l, xstar, kinds, kparams, ls_len, ls, mean_out, var_out,"
            " proba_out), got "
            + String(len(addrs))
        )
    if len(params) != 6:
        raise Error(
            "gpc_predict: params must contain 6 values (n_train, n_features,"
            " n_star, n_nodes, n_ls, want_proba), got "
            + String(len(params))
        )
    var n_train = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_star = Int(py=params[2])
    var n_nodes = Int(py=params[3])
    var n_ls = Int(py=params[4])
    var want_proba = Int(py=params[5]) != 0
    var mean_addr = Int(py=addrs[10])
    var var_addr = Int(py=addrs[11])
    var proba_addr = Int(py=addrs[12])
    var spec = _rebuild_kernel_spec(
        Int(py=addrs[6]),
        Int(py=addrs[7]),
        Int(py=addrs[8]),
        Int(py=addrs[9]),
        n_nodes,
        n_ls,
        String("gpc_predict"),
    )
    var xt = read_f32(Int(py=addrs[0]), max(0, n_train * n_features))
    var y = read_f32(Int(py=addrs[1]), max(0, n_train))
    var pi = read_f32(Int(py=addrs[2]), max(0, n_train))
    var wsr = read_f32(Int(py=addrs[3]), max(0, n_train))
    var l = read_f32(Int(py=addrs[4]), max(0, n_train * n_train))
    var x_star = read_f32(Int(py=addrs[5]), max(0, n_star * n_features))
    var rc = 0
    with GILReleased(Python()):
        rc = _gpc_predict_run(
            xt, y, pi, wsr, l, spec, x_star, n_train, n_features, n_star,
            want_proba, mean_addr, var_addr, proba_addr,
        )
    _ = xt^
    _ = y^
    _ = pi^
    _ = wsr^
    _ = l^
    _ = x_star^
    _ = spec^
    return PythonObject(rc)


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
        module.def_function[gpc_fit_binding]("gpc_fit")
        module.def_function[gpc_predict_binding]("gpc_predict")
        module.def_function[cholesky_profile_jitter_binding]("cholesky_profile_jitter")
        module.def_function[cholesky_factor_binding]("cholesky_factor")
        module.def_function[cholesky_solve_binding]("cholesky_solve")
        return module.finalize()
    except e:
        abort(String("failed to create _mojolearn_gp_host: ", e))
