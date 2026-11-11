# SPDX-License-Identifier: Apache-2.0
"""Native GP request execution through the existing numerical entry bodies.

The former Python address/option envelopes are decoded by gp_native_api.
These methods retain the existing copies, narrowing, calls and route switches.
"""
from core.gp_native_api import (
    GPNativeKernels, GPKernelView, GPQueryResult, GPRFitRequest, GPRQueryRequest,
    GPRCovarianceRequest, GPRSampleRequest, GPRGradientRequest, GPROptimizeRequest,
    GPCFitRequest, GPCFitAllRequest, GPCQueryRequest, GPCQueryAllRequest,
    GPCCombineRequest, GPThetaRequest, GP_CAP_FIT, GP_CAP_COVARIANCE,
    GP_CAP_SAMPLE, GP_CAP_GRADIENT, GP_CAP_OPTIMIZE, GP_CAP_THETA,
    GP_CAP_POINTER, GP_CAP_OVR, GP_CAP_RESIDENT,
    GP_ROUTE_LIST, GP_ROUTE_POINTER, GP_ROUTE_RESIDENT,
)
from core.pca_native_api import pca_vendor_code
from gaussian_process.unnorm import gpc_binary_out, gpc_ovr_targets
from core.hostptr import f32_ptr, f64_ptr, i32_ptr, read_f32, read_i32
from std.memory import memcpy
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.gp_native_storage import gp_native_ints, gp_native_f64
from gaussian_process.host.gp_host_entries import _rebuild_kernel_spec, _gpr_predict_run, _gpc_predict_run
from core.gp_native_host_infer import GPHostInferenceKernels
from gaussian_process.host.gp_host_fit_entries import (
    _gpr_fit_run, _gpc_fit_run, _gpr_lml_grad_run, _gpr_host_optimize,
    _gpr_predict_cov_run, _gpr_sample_y_run, gp_theta_params_list,
)
from gaussian_process.gp_grad_items import gp_free_count


struct GPHostKernels(GPNativeKernels):
    @staticmethod
    def mode() raises -> Int:
        return GLOBAL_NUMERIC_MODE

    @staticmethod
    def vendor() raises -> Int:
        return 1

    @staticmethod
    def capabilities() raises -> Int:
        return GP_CAP_FIT | GP_CAP_COVARIANCE | GP_CAP_SAMPLE | GP_CAP_GRADIENT | GP_CAP_OPTIMIZE | GP_CAP_THETA

    @staticmethod
    def lease_open() raises -> Int:
        return GPHostInferenceKernels.lease_open()

    @staticmethod
    def lease_close(lease: Int) raises:
        GPHostInferenceKernels.lease_close(lease)

    @staticmethod
    def gpr_fit(r: GPRFitRequest) raises -> Int:
        if r.pointer_route:
            raise Error("GP native host: pointer fit is unavailable")
        var lp = f32_ptr(r.factor)
        var dp = f32_ptr(r.dual)
        var sp = f64_ptr(r.scalars)
        var n_train = r.rows
        var n_features = r.features
        var n_nodes = r.kernel.nodes
        var n_ls = r.kernel.scale_count
        var alpha = Float32(r.alpha)
        var spec = _rebuild_kernel_spec(
            r.kernel.kinds,
            r.kernel.parameters,
            r.kernel.lengths,
            r.kernel.scales,
            n_nodes,
            n_ls,
            String("gpr_fit"),
        )
        var x = read_f32(r.source, max(0, n_train * n_features))
        var y = read_f32(r.targets, max(0, n_train))
        var info = 0
        info = _gpr_fit_run(x, y, spec, n_train, n_features, alpha, lp, dp, sp)
        _ = x^
        _ = y^
        _ = spec^
        return info

    @staticmethod
    def gpr_covariance(r: GPRCovarianceRequest) raises -> Int:
        var unnorm = r.normalize
        var y_std = Float32(r.y_std)
        var y_mean = Float32(r.y_mean)
        var mp = f32_ptr(r.mean)
        var cp = f32_ptr(r.covariance)
        var n_train = r.training_rows
        var n_features = r.features
        var n_star = r.rows
        var n_nodes = r.kernel.nodes
        var n_ls = r.kernel.scale_count
        var info = r.info
        var spec = _rebuild_kernel_spec(
            r.kernel.kinds, r.kernel.parameters, r.kernel.lengths, r.kernel.scales,
            n_nodes, n_ls, String("gpr_predict_cov"),
        )
        var xt = read_f32(r.training, max(0, n_train * n_features))
        var l = read_f32(r.factor, max(0, n_train * n_train))
        var dual = read_f32(r.dual, max(0, n_train))
        var x_star = read_f32(r.source, max(0, n_star * n_features))
        _gpr_predict_cov_run(
            xt, l, dual, n_train, n_features, spec, info, x_star, n_star,
            mp, cp, unnorm, y_std, y_mean,
        )
        _ = xt^
        _ = l^
        _ = dual^
        _ = x_star^
        _ = spec^
        return n_star

    @staticmethod
    def gpr_sample(r: GPRSampleRequest) raises -> Int:
        var unnorm = r.normalize
        var y_std = Float32(r.y_std)
        var y_mean = Float32(r.y_mean)
        var yp = f32_ptr(r.dst)
        var n_train = r.training_rows
        var n_features = r.features
        var n_star = r.rows
        var n_nodes = r.kernel.nodes
        var n_ls = r.kernel.scale_count
        var info = r.info
        var n_samples = r.samples
        var seed = (UInt64(r.seed_high) << 32) | UInt64(r.seed_low)
        var spec = _rebuild_kernel_spec(
            r.kernel.kinds,
            r.kernel.parameters,
            r.kernel.lengths,
            r.kernel.scales,
            n_nodes,
            n_ls,
            String("gpr_sample_y"),
        )
        var xt = read_f32(r.training, max(0, n_train * n_features))
        var l = read_f32(r.factor, max(0, n_train * n_train))
        var dual = read_f32(r.dual, max(0, n_train))
        var x_star = read_f32(r.source, max(0, n_star * n_features))
        _gpr_sample_y_run(
            xt, l, dual, n_train, n_features, spec, info, x_star, n_star,
            n_samples, seed, yp, unnorm, y_std, y_mean,
        )
        _ = xt^
        _ = l^
        _ = dual^
        _ = x_star^
        _ = spec^
        return n_samples

    @staticmethod
    def gpr_gradient(r: GPRGradientRequest) raises -> Int:
        var n_train = r.rows
        var n_features = r.features
        var n_nodes = r.kernel.nodes
        var n_ls = r.kernel.scale_count
        var alpha = Float32(r.alpha)
        var spec = _rebuild_kernel_spec(
            r.kernel.kinds, r.kernel.parameters, r.kernel.lengths, r.kernel.scales,
            n_nodes, n_ls, String("gpr_lml_grad"),
        )
        var free = read_i32(r.free_flags, max(0, n_nodes))
        var gp = f64_ptr(r.gradient)
        var sp = f64_ptr(r.scalars)
        var x = read_f32(r.source, max(0, n_train * n_features))
        var y = read_f32(r.targets, max(0, n_train))
        var info = 0
        info = _gpr_lml_grad_run(x, n_train, n_features, y, spec, free, alpha, gp, sp)
        _ = x^
        _ = y^
        _ = free^
        _ = spec^
        return info

    @staticmethod
    def gpr_optimize(r: GPROptimizeRequest) raises -> Int:
        var n_train = r.rows
        var n_features = r.features
        var n_nodes = r.kernel.nodes
        var n_ls = r.kernel.scale_count
        var alpha = Float32(r.alpha)
        var n_restarts = r.restarts
        var seed_lo = UInt32(r.seed_low & 0xFFFFFFFF)
        var seed_hi = UInt32(r.seed_high & 0xFFFFFFFF)
        var spec = _rebuild_kernel_spec(
            r.kernel.kinds, r.kernel.parameters, r.kernel.lengths, r.kernel.scales,
            n_nodes, n_ls, String("gpr_optimize"),
        )
        var free = read_i32(r.free_flags, max(0, n_nodes))
        var n_theta = gp_free_count(spec.kinds, spec.ls_len, free)
        var bounds = read_f32(r.bounds, 2 * n_theta)
        var tp = f64_ptr(r.theta)
        var vp = f64_ptr(r.values)
        var rp = f64_ptr(r.runs)
        var x = read_f32(r.source, max(0, n_train * n_features))
        var y = read_f32(r.targets, max(0, n_train))
        var best = 0
        best = _gpr_host_optimize(
            x, n_train, n_features, y, spec, free, bounds, alpha, n_restarts,
            seed_lo, seed_hi, tp, vp, rp,
        )
        _ = x^
        _ = y^
        _ = free^
        _ = spec^
        _ = bounds^
        return best

    @staticmethod
    def gpc_fit(r: GPCFitRequest) raises -> Int:
        var lp = f32_ptr(r.factor)
        var pp = f32_ptr(r.pi)
        var wp = f32_ptr(r.wsr)
        var sp = f64_ptr(r.scalars)
        var n_train = r.rows
        var n_features = r.features
        var n_nodes = r.kernel.nodes
        var n_ls = r.kernel.scale_count
        var max_iter_predict = r.max_iterations
        var spec = _rebuild_kernel_spec(
            r.kernel.kinds,
            r.kernel.parameters,
            r.kernel.lengths,
            r.kernel.scales,
            n_nodes,
            n_ls,
            String("gpc_fit"),
        )
        var x = read_f32(r.source, max(0, n_train * n_features))
        var y: List[Float32]
        if r.coded_targets:
            y = gpc_ovr_targets(read_i32(r.targets, max(0, n_train)), n_train, r.column)
            if r.target_output != 0:
                var yo = f32_ptr(r.target_output)
                memcpy(dest=yo, src=y.unsafe_ptr(), count=max(0, n_train))
                _ = len(y)
        else:
            y = read_f32(r.targets, max(0, n_train))
        var n_iter = 0
        n_iter = _gpc_fit_run(
            x, y, spec, n_train, n_features, max_iter_predict, lp, pp, wp, sp
        )
        _ = x^
        _ = y^
        _ = spec^
        return n_iter

    @staticmethod
    def gpr_query(r: GPRQueryRequest) raises -> GPQueryResult:
        return GPHostInferenceKernels.gpr_query(r)

    @staticmethod
    def gpc_query(r: GPCQueryRequest) raises -> Int:
        return GPHostInferenceKernels.gpc_query(r)

    @staticmethod
    def gpc_combine(r: GPCCombineRequest) raises -> Int:
        return GPHostInferenceKernels.gpc_combine(r)

    @staticmethod
    def gpc_fit_all(r: GPCFitAllRequest) raises -> Int:
        return GPHostInferenceKernels.gpc_fit_all(r)

    @staticmethod
    def gpc_query_all(r: GPCQueryAllRequest) raises -> GPQueryResult:
        return GPHostInferenceKernels.gpc_query_all(r)

    @staticmethod
    def theta(r: GPThetaRequest) raises -> Int:
        var values = gp_native_f64(r.source, r.count)
        var result = gp_theta_params_list(values)
        if r.count > 0:
            memcpy(f64_ptr(r.dst), result.unsafe_ptr(), r.count)
        return r.count
