# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GaussianProcessRegressor and GaussianProcessClassifier predict entries
of the gp host bindings, written once (the neighbors and density inference lane, 2026-09-15).

`bindings/_mojolearn_gp_host.mojo` (the reference binding: fit, predict and the
Cholesky door) and `bindings/_mojolearn_gp_infer_host.mojo` (the inference
binding a wheel ships: predict only) both register `gpr_predict_binding` and
`gpc_predict_binding`, and
the reference fit reuses `_rebuild_kernel_spec`. The address and params contract
is the GPU binding's (`bindings/_mojolearn_gp.mojo`), word for word.
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from bindings.hostptr import f32_ptr, f64_ptr, i32_ptr, read_f32
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
