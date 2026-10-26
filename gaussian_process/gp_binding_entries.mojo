# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The typed GP and Cholesky bodies behind `bindings/_mojolearn_gp.mojo`'s
Python doors (lane thin-trees-metrics, 2026-10-21: the `_*_run` halves, the
kernel spec rebuild and the pointer helpers, moved word for word out of that
shell, so a Python-free caller reaches the same source; no Python here). The
address and params contracts are that file's."""
from gaussian_process.unnorm import GP_PY2MOJO, gpc_binary_out, gpc_ovr_targets
from core.hostptr import f32_ptr, f64_ptr, i32_ptr, copy_f32, read_f32, read_i32
from gaussian_process.host.gp_theta import (
    gp_theta_param,
)

from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR

from gaussian_process.checks.kernels import (
    GP_K_CONST,
    GP_K_MATERN,
    GP_K_PROD,
    GP_K_RBF,
    GP_K_SUM,
    GP_K_WHITE,
    GPKernelSpec,
    gp_kernel_const,
    gp_kernel_matern,
    gp_kernel_prod,
    gp_kernel_rbf,
    gp_kernel_sum,
    gp_kernel_white,
)
from gaussian_process.gp_optim import GPOptResult, gpr_optimize_device
from gaussian_process.estimator import (
    GPRegressor,
    gpr_fit_host,
    gpr_lml_grad_host,
    gpr_predict_host,
    gpr_predict_cov_host,
    gpr_sample_y_host,
)
# Gaussian process classification (lane/gaussian-process-classifier,
# 2026-09-15): the binary Laplace fit, its latent prediction and (since
# cpu-gpu-cleanup c-gp-kernel, 2026-10-02) the float64 probability and the
# one-vs-rest combine on the device (DEVIATIONS 2830-2833).
from gaussian_process.classifier import (
    gpc_fit_binary_host,
    gpc_ovr_combine_host,
    gpc_predict_binary_host,
)
# lane fam2-kernel-gp (2026-10-04): every class in one device session.
from gaussian_process.gpc_ovr import GPC_IDN_OVR, gpc_fit_all_device, gpc_predict_all_device
from gaussian_process.gpr_resident import GPR_IDN_PTR, gpr_fit_ptr_device, gpr_predict_ptr_device
# The Cholesky door (workstream D, 2026-09-14). `cholesky/` is already
# linked into this binary because the GP factors through it; exposing the
# one-shot host entries here adds no kernel and no second build.
from cholesky.estimator import (
    CHOL_FAST_DEVIO,
    cholesky_factor_devio,
    CholeskyFactor,
    cholesky_factor_host,
    cholesky_profile_jitter,
    cholesky_solve_host,
)


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    return f32_ptr(addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    return i32_ptr(addr)


def _f64_ptr(addr: Int) raises -> MutPointer[Float64, MutUntrackedOrigin]:
    return f64_ptr(addr)


def _rebuild_kernel_spec(
    kinds_addr: Int,
    kparams_addr: Int,
    ls_len_addr: Int,
    ls_addr: Int,
    n_nodes: Int,
    n_ls: Int,
    what: String,
) raises -> GPKernelSpec:
    """The postfix node list back into a `GPKernelSpec`, THROUGH THE
    CONSTRUCTORS (DEVIATION 1756).

    A stack machine over the four flat arrays `_gp_impl.py::_kernel_arrays`
    sends. Leaf node `t` consumes `ls_len[t]` floats from the length-scale
    table at a running offset; `gp_kernel_sum` / `gp_kernel_prod` recompute
    every offset in `_combine`, which is why the offsets are the one array
    NOT sent. Every value goes to the constructor UNJUDGED so its refusal
    fires with its own name: `gp_kernel_matern` refuses a `nu` outside the
    three closed forms BY BITS (DEVIATION 1765) and `kparams` crosses as
    float32, so the bits that arrive are the bits `np.float32(nu)` holds.

    `Int(Int32)` below SIGN-EXTENDS ([[mojo-int-widening-sign-extends]]),
    and here that is the wanted branch of the trap: a negative kind or
    ls_len must ARRIVE negative so the explicit range refusals below can
    name it, rather than be masked into a plausible small code.
    """
    if n_nodes < 1:
        raise Error(
            what
            + ": the kernel spec must have at least one postfix node, got "
            + String(n_nodes)
        )
    if n_ls < 0:
        raise Error(
            what + ": n_ls cannot be negative, got " + String(n_ls)
        )
    var kp = _i32_ptr(kinds_addr)
    var pp = _f32_ptr(kparams_addr)
    var lnp = _i32_ptr(ls_len_addr)
    var tp = _f32_ptr(ls_addr)
    var stack = List[GPKernelSpec]()
    var off = 0
    for t in range(n_nodes):
        var k = Int(kp.unsafe_load(t))
        var param = pp.unsafe_load(t)
        if k == GP_K_CONST:
            stack.append(gp_kernel_const(param))
        elif k == GP_K_WHITE:
            stack.append(gp_kernel_white(param))
        elif k == GP_K_RBF or k == GP_K_MATERN:
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
            if k == GP_K_RBF:
                stack.append(gp_kernel_rbf(leaf_ls))
            else:
                stack.append(gp_kernel_matern(leaf_ls, param))
        elif k == GP_K_SUM or k == GP_K_PROD:
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
            if k == GP_K_SUM:
                stack.append(gp_kernel_sum(a, b))
            else:
                stack.append(gp_kernel_prod(a, b))
        else:
            raise Error(
                what
                + ": node "
                + String(t)
                + " has unknown kind "
                + String(k)
                + ". The GP_K_* codes are 0 CONST, 1 WHITE, 2 RBF,"
                " 3 MATERN, 4 SUM, 5 PROD, mirrored in _gp_impl.py; a"
                " silent renumbering on either side is a WRONG KERNEL,"
                " which is why this refuses by value instead of clamping"
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
    spec: GPKernelSpec,
    n_train: Int,
    n_features: Int,
    alpha: Float32,
    lp: MutPointer[Float32, MutUntrackedOrigin],
    dp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    """The GIL-free half of `gpr_fit_binding`: everything after the
    `PythonObject`s have been read and before one is built. A separate
    function because `GPRegressor` has no default constructor to
    pre-declare across a `with` block, and an `Int` does."""
    var model = gpr_fit_host(x, n_train, n_features, y, spec, alpha)
    copy_f32(model.l.unsafe_ptr(), lp, n_train * n_train)
    copy_f32(model.dual_coef.unsafe_ptr(), dp, n_train)
    # info, nb, logdet, ydotalpha, lml -- in that order, the same five
    # words as `_gp_impl.py::fit`'s `scalars` comment. Each float32 widens
    # to float64 exactly.
    sp.unsafe_store(0, Float64(model.info))
    sp.unsafe_store(1, Float64(model.nb))
    sp.unsafe_store(2, Float64(model.logdet))
    sp.unsafe_store(3, Float64(model.ydotalpha))
    sp.unsafe_store(4, Float64(model.lml))
    return model.info


def _gpr_predict_run(
    model: GPRegressor,
    x_star: List[Float32],
    n_star: Int,
    return_std: Bool,
    mean_addr: Int,
    var_addr: Int,
    std_addr: Int,
    clamped_addr: Int,
    unnorm: Bool = False,
    y_std: Float32 = Float32(1.0),
    y_mean: Float32 = Float32(0.0),
) raises -> Int:
    """The GIL-free half of `gpr_predict_binding`, for `_gpr_fit_run`'s
    reason. The variance/std/clamp addresses are taken as `Int` and only
    resolved inside the `return_std` arm, so a caller that asked for the
    mean alone never has them dereferenced -- "safe to write or skip", and
    this side skips."""
    var pred = gpr_predict_host(
        model, x_star, n_star, return_std, unnorm=unnorm, y_std=y_std, y_mean=y_mean
    )
    var mp = _f32_ptr(mean_addr)
    for i in range(n_star):
        mp.unsafe_store(i, pred.mean[i])
    if return_std:
        var vp = _f32_ptr(var_addr)
        var stp = _f32_ptr(std_addr)
        var cp = _i32_ptr(clamped_addr)
        for i in range(n_star):
            vp.unsafe_store(i, pred.variance[i])
            stp.unsafe_store(i, pred.std[i])
            cp.unsafe_store(i, pred.clamped[i])
    return pred.n_clamped


def _gpr_sample_y_run(
    model: GPRegressor,
    x_star: List[Float32],
    n_star: Int,
    n_samples: Int,
    seed: UInt64,
    out_addr: Int,
    unnorm: Bool = False,
    y_std: Float32 = Float32(1.0),
    y_mean: Float32 = Float32(0.0),
) raises:
    """The GIL-free half of `gpr_sample_y_binding`."""
    var y = gpr_sample_y_host(
        model, x_star, n_star, n_samples, seed, unnorm=unnorm, y_std=y_std, y_mean=y_mean
    )
    copy_f32(y.unsafe_ptr(), _f32_ptr(out_addr), n_star * n_samples)


def _gpr_predict_cov_run(
    model: GPRegressor,
    x_star: List[Float32],
    n_star: Int,
    mean_addr: Int,
    cov_addr: Int,
    unnorm: Bool = False,
    y_std: Float32 = Float32(1.0),
    y_mean: Float32 = Float32(0.0),
) raises:
    """The GIL-free half of `gpr_predict_cov_binding`."""
    var r = gpr_predict_cov_host(model, x_star, n_star, unnorm=unnorm, y_std=y_std, y_mean=y_mean)
    copy_f32(r.mean.unsafe_ptr(), _f32_ptr(mean_addr), n_star)
    copy_f32(r.cov.unsafe_ptr(), _f32_ptr(cov_addr), n_star * n_star)


def _cholesky_factor_run(
    a: List[Float32],
    n: Int,
    jitter: Float32,
    lp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    """The GIL-free half of `cholesky_factor_binding`."""
    var f = cholesky_factor_host(a, n, jitter)
    copy_f32(f.l.unsafe_ptr(), lp, n * n)
    # info, nb, logdet, jitter -- in that order, mirrored in
    # `_cholesky_impl.py::Cholesky.fit`. Each widens to float64 exactly.
    sp.unsafe_store(0, Float64(f.info))
    sp.unsafe_store(1, Float64(f.nb))
    sp.unsafe_store(2, Float64(f.logdet))
    sp.unsafe_store(3, Float64(f.jitter))
    return f.info


def _cholesky_solve_run(
    factor: CholeskyFactor,
    b: List[Float32],
    nrhs: Int,
    xp: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """The GIL-free half of `cholesky_solve_binding`."""
    var x = cholesky_solve_host(factor, b, nrhs)
    copy_f32(x.unsafe_ptr(), xp, factor.n * nrhs)


def _gpc_fit_run(
    x: List[Float32],
    y: List[Float32],
    spec: GPKernelSpec,
    n_train: Int,
    n_features: Int,
    max_iter_predict: Int,
    lp: MutPointer[Float32, MutUntrackedOrigin],
    pp: MutPointer[Float32, MutUntrackedOrigin],
    wp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    """The GIL-free half of `gpc_fit_binding`."""
    var fit = gpc_fit_binary_host(x, n_train, n_features, y, spec, max_iter_predict)
    for i in range(n_train * n_train):
        lp.unsafe_store(i, fit.l[i])
    for i in range(n_train):
        pp.unsafe_store(i, fit.pi[i])
        wp.unsafe_store(i, fit.wsr[i])
    # lml, n_iter, nb, in that order; each widens to float64 exactly.
    sp.unsafe_store(0, Float64(fit.lml))
    sp.unsafe_store(1, Float64(fit.n_iter))
    sp.unsafe_store(2, Float64(fit.nb))
    return fit.n_iter


def _gpc_predict_run(
    xt: List[Float32],
    y: List[Float32],
    pi: List[Float32],
    wsr: List[Float32],
    l: List[Float32],
    spec: GPKernelSpec,
    x_star: List[Float32],
    n_train: Int,
    n_features: Int,
    n_star: Int,
    want_proba: Bool,
    mean_addr: Int,
    var_addr: Int,
    proba_addr: Int,
) raises -> Int:
    """The GIL-free half of `gpc_predict_binding`. The variance and
    probability addresses are resolved only when `want_proba`."""
    var lat = gpc_predict_binary_host(
        xt, y, pi, wsr, l, n_train, n_features, spec, x_star, n_star, want_proba
    )
    var mp = _f32_ptr(mean_addr)
    for t in range(n_star):
        mp.unsafe_store(t, lat.mean[t])
    if want_proba:
        var vp = _f32_ptr(var_addr)
        var pr = _f64_ptr(proba_addr)
        for t in range(n_star):
            vp.unsafe_store(t, lat.variance[t])
            pr.unsafe_store(t, lat.proba[t])
    return 0


def _gpr_optimize_run(
    x: List[Float32],
    y: List[Float32],
    spec: GPKernelSpec,
    free: List[Int32],
    bounds: List[Float32],
    n_train: Int,
    n_features: Int,
    alpha: Float32,
    n_restarts: Int,
    seed_lo: UInt32,
    seed_hi: UInt32,
    tp: MutPointer[Float64, MutUntrackedOrigin],
    vp: MutPointer[Float64, MutUntrackedOrigin],
    rp: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    """The GIL-free half of `gpr_optimize_binding`."""
    var r = gpr_optimize_device(
        x, n_train, n_features, y, spec, free, bounds, alpha, n_restarts, seed_lo, seed_hi
    )
    for i in range(len(r.theta)):
        tp.unsafe_store(i, Float64(r.theta[i]))
        vp.unsafe_store(i, Float64(r.values[i]))
    for i in range(len(r.runs)):
        rp.unsafe_store(i, Float64(r.runs[i]))
    return r.best


def _gpr_lml_grad_run(
    x: List[Float32],
    y: List[Float32],
    spec: GPKernelSpec,
    free: List[Int32],
    n_train: Int,
    n_features: Int,
    alpha: Float32,
    gp: MutPointer[Float64, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    """The GIL-free half of `gpr_lml_grad_binding`."""
    var r = gpr_lml_grad_host(x, n_train, n_features, y, spec, free, alpha)
    for i in range(len(r.grad)):
        gp.unsafe_store(i, Float64(r.grad[i]))
    sp.unsafe_store(0, Float64(r.info))
    sp.unsafe_store(1, Float64(r.lml))
    return r.info


# TOMBSTONE: gp_log64 (dead native entry: no live caller) deleted 2026-10-26 by @@SHA@@; code recoverable at 485ba9cc1.
# Restore: git apply experiments/removed/gp_log64.patch


def gp_theta_params_values(values: List[Float64]) -> List[Float64]:
    """`gp_theta_param` of each value, widened to float64, in order."""
    var out = List[Float64]()
    for i in range(len(values)):
        out.append(Float64(gp_theta_param(values[i])))
    return out^

