# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The typed GP host prediction bodies behind `bindings/gp_host_predict.mojo`'s
Python doors (lane thin-trees-metrics, 2026-10-21: `_rebuild_kernel_spec`,
`_gpr_predict_run` and `_gpc_predict_run` moved word for word out of that
shell; no Python here)."""
from core.hostptr import f32_ptr, f64_ptr, i32_ptr
from gaussian_process.unnorm import GP_UNNORM_MEAN, GP_UNNORM_STD, gp_unnorm_cell
from gaussian_process.host.gpc_oracle import gpc_host_predict
from gaussian_process.host.gpc_steps import gpc_proba
from gaussian_process.host.gpr_oracle import (
    GPR_K_CONST,
    GPR_K_MATERN,
    GPR_K_PROD,
    GPR_K_RBF,
    GPR_K_SUM,
    GPR_K_WHITE,
    GPHostKernelSpec,
    gpr_host_kernel_const,
    gpr_host_kernel_matern,
    gpr_host_kernel_prod,
    gpr_host_kernel_rbf,
    gpr_host_kernel_sum,
    gpr_host_kernel_white,
    gpr_host_predict,
)


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
    unnorm: Bool = False,
    y_std: Float32 = Float32(1.0),
    y_mean: Float32 = Float32(0.0),
) raises -> Int:
    """The GIL-free half of `gpr_predict_binding`. The variance, std and
    clamp addresses are resolved only in the `return_std` arm, as on the
    GPU binding. `unnorm`: normalize_y's mean and std
    (`gaussian_process/unnorm.mojo`), the variance left normalized."""
    var pred = gpr_host_predict(
        xt, l, dual, n_train, n_features, spec, info, x_star, n_star, return_std
    )
    var mp = f32_ptr(mean_addr)
    for i in range(n_star):
        mp.unsafe_store(i, gp_unnorm_cell(pred.mean[i], y_std, y_mean, GP_UNNORM_MEAN) if unnorm else pred.mean[i])
    if return_std:
        var vp = f32_ptr(var_addr)
        var stp = f32_ptr(std_addr)
        var cp = i32_ptr(clamped_addr)
        for i in range(n_star):
            vp.unsafe_store(i, pred.variance[i])
            stp.unsafe_store(
                i, gp_unnorm_cell(pred.variance[i], y_std, y_mean, GP_UNNORM_STD) if unnorm else pred.std[i]
            )
            cp.unsafe_store(i, pred.clamped[i])
    var n_clamped = pred.n_clamped
    _ = pred^
    return n_clamped


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
