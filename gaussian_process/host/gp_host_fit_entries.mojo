# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The typed GP and Cholesky host bodies behind `bindings/_mojolearn_gp_host.mojo`'s
Python doors (lane thin-trees-metrics, 2026-10-21: moved word for word out of
that shell, so a Python-free caller reaches the same source; no Python here).
The address and params contracts are that file's."""
from core.hostptr import f32_ptr, f64_ptr
from gaussian_process.host.gp_theta import (
    gp_log64,
    gp_theta_param,
)
from gaussian_process.host.gpr_grad_oracle import gpr_host_lml_grad
from gaussian_process.gp_grad_items import gp_free_count
from gaussian_process.gp_optim_items import (
    GP_OPT_MAX_ITER,
    GP_OPT_MAX_LS,
    GP_OPT_SI_BEST,
    GP_OPT_SI_LEN,
    GP_OPT_SI_STOP,
    gp_opt_final_item,
    gp_opt_init_item,
    gp_opt_run_end_item,
    gp_opt_st_len,
    gp_opt_step_item,
    gp_opt_theta_map,
)
from cholesky.host.chol_oracle import chol_host_potrf
from gaussian_process.unnorm import GP_UNNORM_COV, GP_UNNORM_MEAN, gp_unnorm_cell
from gaussian_process.host.gpr_oracle import GPHostKernelSpec, gpr_host_fit
from gaussian_process.host.sample_y_oracle import gpr_host_predict_cov, gpr_host_sample_y
from gaussian_process.host.gpc_oracle import gpc_host_fit


comptime _HP = MutPointer[Float32, MutAnyOrigin]
comptime _HI = MutPointer[Int32, MutAnyOrigin]


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


@always_inline
def _hp(l: List[Float32]) -> _HP:
    return _HP(unsafe_from_address=Int(l.unsafe_ptr()))


@always_inline
def _hi(l: List[Int32]) -> _HI:
    return _HI(unsafe_from_address=Int(l.unsafe_ptr()))


def _gpr_host_optimize(
    x: List[Float32],
    n_train: Int,
    n_features: Int,
    y: List[Float32],
    spec: GPHostKernelSpec,
    free: List[Int32],
    bounds: List[Float32],
    alpha: Float32,
    n_restarts: Int,
    seed_lo: UInt32,
    seed_hi: UInt32,
    tp: MutPointer[Float64, MutUntrackedOrigin],
    vp: MutPointer[Float64, MutUntrackedOrigin],
    rp: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    """`gaussian_process/gp_optim.mojo::gpr_optimize_device` on the host:
    the same `gp_optim_items` statements (tid 0 of 1 thread), the same
    order, each evaluation `gpr_host_lml_grad` at the parameter table the
    items wrote."""
    var nt = gp_free_count(spec.kinds, spec.ls_len, free)
    if nt < 1:
        raise Error("gpr_optimize: the kernel has no free hyperparameter")
    if len(bounds) != 2 * nt:
        raise Error(
            "gpr_optimize: bounds holds " + String(len(bounds)) + " values, the kernel's "
            + String(nt) + " free hyperparameters need " + String(2 * nt)
        )
    if n_restarts < 0:
        raise Error("gpr_optimize: n_restarts cannot be negative")
    var tmap = gp_opt_theta_map(spec.kinds, spec.ls_off, spec.ls_len, free)
    var n_runs = 1 + n_restarts
    var hpar = spec.params.copy()
    var hls = spec.length_scales.copy()
    if len(hls) == 0:
        hls.append(Float32(1.0))
    var n_ls = len(spec.length_scales)
    var st = List[Float32](length=gp_opt_st_len(nt), fill=Float32(0.0))
    var si = List[Int32](length=GP_OPT_SI_LEN, fill=Int32(0))
    var rec = List[Float32](length=n_runs * 4, fill=Float32(0.0))
    var res = List[Float32](length=2 * nt, fill=Float32(0.0))
    var lml = List[Float32](length=1, fill=Float32(0.0))
    var graw = List[Float32](length=nt, fill=Float32(0.0))
    var eval_cap = GP_OPT_MAX_ITER * (GP_OPT_MAX_LS + 1) + 2
    for run in range(n_runs):
        gp_opt_init_item(0, 1, nt, _hp(st), _hi(si), _hi(tmap), _hp(bounds), _hp(hpar), _hp(hls), run, seed_lo, seed_hi)
        var evals = 0
        while True:
            evals += 1
            if evals > eval_cap:
                raise Error("gpr_optimize: the optimizer passed its evaluation cap without a stop word")
            var ls_now = List[Float32]()
            for i in range(n_ls):
                ls_now.append(hls[i])
            var at = GPHostKernelSpec(
                spec.kinds.copy(), hpar.copy(), spec.ls_off.copy(), spec.ls_len.copy(), ls_now^
            )
            var r = gpr_host_lml_grad(x, n_train, n_features, y, at, free, alpha)
            lml[0] = r.lml
            for i in range(nt):
                graw[i] = r.grad[i] if i < len(r.grad) else Float32(0.0)
            gp_opt_step_item(0, 1, nt, _hp(st), _hi(si), _hi(tmap), _hp(hpar), _hp(hls), _hp(lml), _hp(graw), r.info)
            _ = r^
            _ = at^
            if Int(si[GP_OPT_SI_STOP]) != 0:
                break
        gp_opt_run_end_item(0, 1, nt, _hp(st), _hi(si), _hp(rec), run)
    gp_opt_final_item(0, 1, nt, _hp(st), _hi(tmap), _hp(hpar), _hp(hls), _hp(res))
    for i in range(nt):
        tp.unsafe_store(i, Float64(res[i]))
        vp.unsafe_store(i, Float64(res[nt + i]))
    for i in range(n_runs * 4):
        rp.unsafe_store(i, Float64(rec[i]))
    var best = Int(si[GP_OPT_SI_BEST])
    _ = st^
    _ = si^
    _ = rec^
    _ = res^
    _ = lml^
    _ = graw^
    _ = tmap^
    _ = hpar^
    _ = hls^
    return best


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


def _gpr_sample_y_run(
    xt: List[Float32],
    l: List[Float32],
    dual: List[Float32],
    n_train: Int,
    n_features: Int,
    spec: GPHostKernelSpec,
    info: Int,
    x_star: List[Float32],
    n_star: Int,
    n_samples: Int,
    seed: UInt64,
    yp: MutPointer[Float32, MutUntrackedOrigin],
    unnorm: Bool,
    y_std: Float32,
    y_mean: Float32,
) raises:
    """The GIL-free half of `gpr_sample_y_binding`."""
    var y = gpr_host_sample_y(
        xt, l, dual, n_train, n_features, spec, info, x_star, n_star,
        n_samples, seed,
    )
    for i in range(n_star * n_samples):
        yp.unsafe_store(i, gp_unnorm_cell(y[i], y_std, y_mean, GP_UNNORM_MEAN) if unnorm else y[i])
    _ = y^


def _gpr_predict_cov_run(
    xt: List[Float32],
    l: List[Float32],
    dual: List[Float32],
    n_train: Int,
    n_features: Int,
    spec: GPHostKernelSpec,
    info: Int,
    x_star: List[Float32],
    n_star: Int,
    mp: MutPointer[Float32, MutUntrackedOrigin],
    cp: MutPointer[Float32, MutUntrackedOrigin],
    unnorm: Bool,
    y_std: Float32,
    y_mean: Float32,
) raises:
    """The GIL-free half of `gpr_predict_cov_binding`."""
    var mean = List[Float32]()
    var cov = gpr_host_predict_cov(
        xt, l, dual, n_train, n_features, spec, info, x_star, n_star, mean
    )
    for i in range(n_star):
        mp.unsafe_store(i, gp_unnorm_cell(mean[i], y_std, y_mean, GP_UNNORM_MEAN) if unnorm else mean[i])
    for i in range(n_star * n_star):
        cp.unsafe_store(i, gp_unnorm_cell(cov[i], y_std, y_mean, GP_UNNORM_COV) if unnorm else cov[i])
    _ = mean^
    _ = cov^


def _gpr_lml_grad_run(
    x: List[Float32],
    n_train: Int,
    n_features: Int,
    y: List[Float32],
    spec: GPHostKernelSpec,
    free: List[Int32],
    alpha: Float32,
    gp: MutPointer[Float64, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    """The GIL-free half of `gpr_lml_grad_binding`; returns `info`."""
    var r = gpr_host_lml_grad(x, n_train, n_features, y, spec, free, alpha)
    for i in range(len(r.grad)):
        gp.unsafe_store(i, Float64(r.grad[i]))
    sp.unsafe_store(0, Float64(r.info))
    sp.unsafe_store(1, Float64(r.lml))
    var info = r.info
    _ = r^
    return info


def gp_log64_list(values: List[Float64]) -> List[Float64]:
    """`gp_log64` of each value, in order."""
    var out = List[Float64]()
    for i in range(len(values)):
        out.append(gp_log64(values[i]))
    return out^


def gp_theta_params_list(values: List[Float64]) -> List[Float64]:
    """`gp_theta_param` of each value, widened to float64, in order."""
    var out = List[Float64]()
    for i in range(len(values)):
        out.append(Float64(gp_theta_param(values[i])))
    return out^

