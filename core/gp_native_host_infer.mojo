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
from gaussian_process.host.gpc_steps import gpc_ovr_combine_rows


struct GPHostInferenceKernels(GPNativeKernels):
    @staticmethod
    def mode() raises -> Int:
        return GLOBAL_NUMERIC_MODE

    @staticmethod
    def vendor() raises -> Int:
        return 1

    @staticmethod
    def capabilities() raises -> Int:
        return 0

    @staticmethod
    def lease_open() raises -> Int:
        raise Error("GP native host: resident leases are unavailable")

    @staticmethod
    def lease_close(lease: Int) raises:
        raise Error("GP native host: resident leases are unavailable")

    @staticmethod
    def gpr_query(r: GPRQueryRequest) raises -> GPQueryResult:
        if r.route != GP_ROUTE_LIST:
            raise Error("GP native host: only the host query route is available")
        var unnorm = r.normalize
        var y_std = Float32(r.y_std)
        var y_mean = Float32(r.y_mean)
        var mean_addr = r.mean
        var var_addr = r.variance
        var std_addr = r.stddev
        var clamped_addr = r.clamped
        var n_train = r.training_rows
        var n_features = r.features
        var n_star = r.rows
        var n_nodes = r.kernel.nodes
        var n_ls = r.kernel.scale_count
        var return_std = r.want_std
        var info = r.info
        var spec = _rebuild_kernel_spec(
            r.kernel.kinds,
            r.kernel.parameters,
            r.kernel.lengths,
            r.kernel.scales,
            n_nodes,
            n_ls,
            String("gpr_predict"),
        )
        var xt = read_f32(r.training, max(0, n_train * n_features))
        var l = read_f32(r.factor, max(0, n_train * n_train)) if return_std else List[Float32]()
        var dual = read_f32(r.dual, max(0, n_train))
        var x_star = read_f32(r.source, max(0, n_star * n_features))
        var n_clamped = 0
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
        return GPQueryResult(n_clamped, 0, 0)

    @staticmethod
    def gpc_query(r: GPCQueryRequest) raises -> Int:
        var out_kind = r.output_kind
        var n_train = r.training_rows
        var n_features = r.features
        var n_star = r.rows
        var n_nodes = r.kernel.nodes
        var n_ls = r.kernel.scale_count
        var want_proba = r.want_probability
        var mean_addr = r.mean
        var var_addr = r.variance
        var proba_addr = r.probability
        var spec = _rebuild_kernel_spec(
            r.kernel.kinds,
            r.kernel.parameters,
            r.kernel.lengths,
            r.kernel.scales,
            n_nodes,
            n_ls,
            String("gpc_predict"),
        )
        var xt = read_f32(r.training, max(0, n_train * n_features))
        var y = read_f32(r.targets, max(0, n_train))
        var pi = read_f32(r.pi, max(0, n_train))
        var wsr = read_f32(r.wsr, max(0, n_train))
        var l = read_f32(r.factor, max(0, n_train * n_train))
        var x_star = read_f32(r.source, max(0, n_star * n_features))
        var rc = 0
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
            gpc_binary_out(mean_addr, proba_addr, n_star, out_kind, r.dst)
        return rc

    @staticmethod
    def gpc_combine(r: GPCCombineRequest) raises -> Int:
        var columns = gp_native_ints(r.columns, r.members)
        gpc_ovr_combine_rows(columns, r.probabilities, r.codes, r.rows)
        return 0

    @staticmethod
    def gpr_fit(r: GPRFitRequest) raises -> Int:
        raise Error("GP native inference image: gpr_fit is unavailable")

    @staticmethod
    def gpr_covariance(r: GPRCovarianceRequest) raises -> Int:
        raise Error("GP native inference image: gpr_covariance is unavailable")

    @staticmethod
    def gpr_sample(r: GPRSampleRequest) raises -> Int:
        raise Error("GP native inference image: gpr_sample is unavailable")

    @staticmethod
    def gpr_gradient(r: GPRGradientRequest) raises -> Int:
        raise Error("GP native inference image: gpr_gradient is unavailable")

    @staticmethod
    def gpr_optimize(r: GPROptimizeRequest) raises -> Int:
        raise Error("GP native inference image: gpr_optimize is unavailable")

    @staticmethod
    def gpc_fit(r: GPCFitRequest) raises -> Int:
        raise Error("GP native inference image: gpc_fit is unavailable")

    @staticmethod
    def gpc_fit_all(r: GPCFitAllRequest) raises -> Int:
        raise Error("GP native inference image: gpc_fit_all is unavailable")

    @staticmethod
    def gpc_query_all(r: GPCQueryAllRequest) raises -> GPQueryResult:
        raise Error("GP native inference image: gpc_query_all is unavailable")

    @staticmethod
    def theta(r: GPThetaRequest) raises -> Int:
        raise Error("GP native inference image: theta is unavailable")
