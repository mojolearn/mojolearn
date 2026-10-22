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

from core.hostptr import read_f32
from gaussian_process.unnorm import GP_PY2MOJO, gpc_binary_out
from gaussian_process.host.gpc_steps import gpc_ovr_combine_rows
from gaussian_process.host.gpr_oracle import GPHostKernelSpec
# The typed bodies (lane thin-trees-metrics, 2026-10-21), moved word for word.
from gaussian_process.host.gp_host_entries import (
    _gpc_predict_run,
    _gpr_predict_run,
    _rebuild_kernel_spec,
)


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
    if len(params) != 7 and len(params) != 10:
        raise Error(
            "gpr_predict: params must contain 7 values (n_train,"
            " n_features, n_star, n_nodes, n_ls, return_std, info), or 10"
            " (+ normalize_y, y_std, y_mean), got "
            + String(len(params))
        )
    var unnorm = len(params) == 10 and Int(py=params[7]) != 0
    var y_std = Float32(Float64(py=params[8])) if len(params) == 10 else Float32(1.0)
    var y_mean = Float32(Float64(py=params[9])) if len(params) == 10 else Float32(0.0)
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
            unnorm,
            y_std,
            y_mean,
        )
    _ = xt^
    _ = l^
    _ = dual^
    _ = x_star^
    _ = spec^
    return PythonObject(n_clamped)


def gpc_predict_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """The latent mean (and with want_proba the variance and the class-1
    probability) on the host. `addrs`: 0 xtrain, 1 y, 2 pi, 3 wsr, 4 l,
    5 xstar, 6 kinds, 7 kparams, 8 ls_len, 9 ls, 10 mean_out, 11 var_out,
    12 proba_out. `params`: 0 n_train, 1 n_features, 2 n_star, 3 n_nodes,
    4 n_ls, 5 want_proba. Returns 0."""
    # addrs[13] + params[6] (lane apple-fast-py2mojo-cluster): an output the
    # Python side computed from these, `gpc_binary_out`'s kind
    var out_kind = Int(py=params[6]) if len(params) == 7 else 0
    if len(addrs) != 13 and not (len(addrs) == 14 and out_kind != 0):
        raise Error(
            "gpc_predict: addrs must contain 13 addresses (xtrain, y, pi,"
            " wsr, l, xstar, kinds, kparams, ls_len, ls, mean_out, var_out,"
            " proba_out), got "
            + String(len(addrs))
        )
    if len(params) != 6 and len(params) != 7:
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
    if out_kind != 0:
        gpc_binary_out(mean_addr, proba_addr, n_star, out_kind, Int(py=addrs[13]))
    return PythonObject(rc)


def gpc_ovr_combine_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """The host column of `bindings/_mojolearn_gp.mojo::gpc_ovr_combine_binding`
    (DEVIATION 2833's one-vs-rest combine), the same address contract:
    addrs 0 proba_out (n * k float64, row-major, WRITTEN), 1 codes_out
    (n int32, WRITTEN), 2..k+1 the class columns (n float64 each, read);
    params 0 n. Returns 0."""
    if len(addrs) < 3:
        raise Error(
            "gpc_ovr_combine: addrs must contain proba_out, codes_out and at"
            " least one class column, got "
            + String(len(addrs))
            + " addresses"
        )
    if len(params) != 1:
        raise Error(
            "gpc_ovr_combine: params must contain 1 value (n), got "
            + String(len(params))
        )
    var n = Int(py=params[0])
    var out_addr = Int(py=addrs[0])
    var codes_addr = Int(py=addrs[1])
    var cols = List[Int]()
    for c in range(2, len(addrs)):  # small-loop(class columns: one address word per class)
        cols.append(Int(py=addrs[c]))
    with GILReleased(Python()):
        gpc_ovr_combine_rows(cols, out_addr, codes_addr, n)
    return PythonObject(0)
