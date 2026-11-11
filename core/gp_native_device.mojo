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
from std.memory import unsafe_memcpy
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.gp_native_storage import gp_native_ints, gp_native_f64
from checks.vendor import COMPILED_VENDOR
from core.device_query_x import QueryFold
from gaussian_process.gp_binding_entries import (
    _f32_ptr, _f64_ptr, _i32_ptr, _rebuild_kernel_spec,
    _gpr_fit_run, _gpr_predict_run, _gpr_predict_cov_run, _gpr_sample_y_run,
    _gpr_lml_grad_run, _gpr_optimize_run, _gpc_fit_run, _gpc_predict_run,
    gp_theta_params_values,
)
from gaussian_process.estimator import GPRegressor
from gaussian_process.checks.kernels import GP_K_RBF, GP_K_MATERN
from gaussian_process.gpr_resident import GPR_IDN_PTR, gpr_fit_ptr_device, gpr_predict_ptr_device
from gaussian_process.gpc_ovr import GPC_IDN_OVR, gpc_fit_all_device, gpc_predict_all_device
from gaussian_process.classifier import gpc_ovr_combine_host
from gaussian_process.gp_resident_state import gp_resident_lease_open, gpr_predict_resident_device
from gaussian_process.gp_resident_store import gp_resident_store
from core.fitted_resident_store import fitted_lease_close


struct GPDeviceKernels(GPNativeKernels):
    @staticmethod
    def mode() raises -> Int:
        return GLOBAL_NUMERIC_MODE

    @staticmethod
    def vendor() raises -> Int:
        return pca_vendor_code(String(COMPILED_VENDOR))

    @staticmethod
    def capabilities() raises -> Int:
        return (GP_CAP_OVR if GPC_IDN_OVR else 0) | (GP_CAP_POINTER if GPR_IDN_PTR else 0) | GP_CAP_RESIDENT | GP_CAP_FIT | GP_CAP_COVARIANCE | GP_CAP_SAMPLE | GP_CAP_GRADIENT | GP_CAP_OPTIMIZE | GP_CAP_THETA

    @staticmethod
    def lease_open() raises -> Int:
        return gp_resident_lease_open()

    @staticmethod
    def lease_close(lease: Int) raises:
        fitted_lease_close(gp_resident_store(), lease)

    @staticmethod
    def gpr_fit(r: GPRFitRequest) raises -> Int:
        if r.pointer_route:
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
                String("gpr_fit_ptr"),
            )
            var x_addr = r.source
            var y_addr = r.targets
            var l_addr = r.factor
            var dual_addr = r.dual
            var scalars_addr = r.scalars
            var info = 0
            info = gpr_fit_ptr_device(
                x_addr, y_addr, n_train, n_features, spec, alpha, l_addr, dual_addr, scalars_addr
            )
            return info
        var xp = _f32_ptr(r.source)
        var yp = _f32_ptr(r.targets)
        var lp = _f32_ptr(r.factor)
        var dp = _f32_ptr(r.dual)
        var sp = _f64_ptr(r.scalars)
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
        var x = read_f32(Int(xp), n_train * n_features)
        var y = read_f32(Int(yp), n_train)
        var info = 0
        info = _gpr_fit_run(
            x, y, spec, n_train, n_features, alpha, lp, dp, sp
        )
        return info

    @staticmethod
    def gpr_query(r: GPRQueryRequest) raises -> GPQueryResult:
        if r.route == GP_ROUTE_RESIDENT:
            var n_train = r.training_rows
            var n_features = r.features
            var n_star = r.rows
            var n_nodes = r.kernel.nodes
            var n_ls = r.kernel.scale_count
            var return_std = r.want_std
            var info = r.info
            var unnorm = r.normalize
            var y_std = Float32(r.y_std)
            var y_mean = Float32(r.y_mean)
            var lease = r.lease
            var spec = _rebuild_kernel_spec(
                r.kernel.kinds,
                r.kernel.parameters,
                r.kernel.lengths,
                r.kernel.scales,
                n_nodes,
                n_ls,
                String("gpr_predict_resident"),
            )
            var xt_addr = r.training
            var l_addr = r.factor
            var dual_addr = r.dual
            var xs_addr = r.source
            var mean_addr = r.mean
            var var_addr = r.variance
            var std_addr = r.stddev
            var clamped_addr = r.clamped
            var fold = QueryFold(r.fold_base, r.fold_rows, r.fold_index)
            var result: Tuple[Int, Int32, Int32]
            result = gpr_predict_resident_device(
                lease, xt_addr, l_addr, dual_addr, xs_addr, n_train, n_features, n_star, spec,
                return_std, info, unnorm, y_std, y_mean,
                mean_addr, var_addr, std_addr, clamped_addr, fold,
            )
            return GPQueryResult(result[0], Int(result[1]), Int(result[2]))
        if r.route == GP_ROUTE_POINTER:
            var n_train = r.training_rows
            var n_features = r.features
            var n_star = r.rows
            var n_nodes = r.kernel.nodes
            var n_ls = r.kernel.scale_count
            var return_std = r.want_std
            var info = r.info
            var unnorm = r.normalize
            var y_std = Float32(r.y_std)
            var y_mean = Float32(r.y_mean)
            var spec = _rebuild_kernel_spec(
                r.kernel.kinds,
                r.kernel.parameters,
                r.kernel.lengths,
                r.kernel.scales,
                n_nodes,
                n_ls,
                String("gpr_predict_ptr"),
            )
            var xt_addr = r.training
            var l_addr = r.factor
            var dual_addr = r.dual
            var xs_addr = r.source
            var mean_addr = r.mean
            var var_addr = r.variance
            var std_addr = r.stddev
            var clamped_addr = r.clamped
            var n_clamped = 0
            n_clamped = gpr_predict_ptr_device(
                xt_addr, l_addr, dual_addr, xs_addr, n_train, n_features, n_star, spec,
                return_std, info, unnorm, y_std, y_mean,
                mean_addr, var_addr, std_addr, clamped_addr,
            )
            return GPQueryResult(n_clamped, 0, 0)
        if r.route != GP_ROUTE_LIST:
            raise Error("GP native query: unknown route")
        var xtp = _f32_ptr(r.training)
        var lp = _f32_ptr(r.factor)
        var dp = _f32_ptr(r.dual)
        var xsp = _f32_ptr(r.source)
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
        var unnorm = r.normalize
        var y_std = Float32(r.y_std)
        var y_mean = Float32(r.y_mean)
        var spec = _rebuild_kernel_spec(
            r.kernel.kinds,
            r.kernel.parameters,
            r.kernel.lengths,
            r.kernel.scales,
            n_nodes,
            n_ls,
            String("gpr_predict"),
        )
        var xt = read_f32(Int(xtp), n_train * n_features)
        var l = read_f32(Int(lp), n_train * n_train)
        var dual = read_f32(Int(dp), n_train)
        var x_star = read_f32(Int(xsp), max(0, n_star * n_features))
        var yzero = List[Float32]()
        var model = GPRegressor(
            xt^,
            yzero^,
            n_train,
            n_features,
            spec^,
            Float32(0.0),
            l^,
            dual^,
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
            info,
            0,
        )
        var n_clamped = 0
        n_clamped = _gpr_predict_run(
            model,
            x_star,
            n_star,
            return_std,
            mean_addr,
            var_addr,
            std_addr,
            clamped_addr,
            unnorm,
            y_std,
            y_mean,
        )
        return GPQueryResult(n_clamped, 0, 0)

    @staticmethod
    def gpr_covariance(r: GPRCovarianceRequest) raises -> Int:
        var unnorm = r.normalize
        var y_std = Float32(r.y_std)
        var y_mean = Float32(r.y_mean)
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
        var mean_addr = r.mean
        var cov_addr = r.covariance
        var yzero = List[Float32]()
        var model = GPRegressor(
            xt^, yzero^, n_train, n_features, spec^, Float32(0.0), l^, dual^,
            Float32(0.0), Float32(0.0), Float32(0.0), info, 0,
        )
        _gpr_predict_cov_run(model, x_star, n_star, mean_addr, cov_addr, unnorm, y_std, y_mean)
        return n_star

    @staticmethod
    def gpr_sample(r: GPRSampleRequest) raises -> Int:
        var unnorm = r.normalize
        var y_std = Float32(r.y_std)
        var y_mean = Float32(r.y_mean)
        var out_addr = r.dst
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
        var yzero = List[Float32]()
        var model = GPRegressor(
            xt^,
            yzero^,
            n_train,
            n_features,
            spec^,
            Float32(0.0),
            l^,
            dual^,
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
            info,
            0,
        )
        _gpr_sample_y_run(model, x_star, n_star, n_samples, seed, out_addr, unnorm, y_std, y_mean)
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
        var gp = _f64_ptr(r.gradient)
        var sp = _f64_ptr(r.scalars)
        var x = read_f32(r.source, max(0, n_train * n_features))
        var y = read_f32(r.targets, max(0, n_train))
        var info = 0
        info = _gpr_lml_grad_run(x, y, spec, free, n_train, n_features, alpha, gp, sp)
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
        var n_theta = 0
        for t in range(n_nodes):  # small-loop(n_nodes: nodes of the kernel expression tree): hyperparameter count, not data
            if Int(free[t]) != 0:
                var k = Int(spec.kinds[t])
                n_theta += Int(spec.ls_len[t]) if (k == GP_K_RBF or k == GP_K_MATERN) else 1
        var bounds = read_f32(r.bounds, 2 * n_theta)
        var tp = _f64_ptr(r.theta)
        var vp = _f64_ptr(r.values)
        var rp = _f64_ptr(r.runs)
        var x = read_f32(r.source, max(0, n_train * n_features))
        var y = read_f32(r.targets, max(0, n_train))
        var best = 0
        best = _gpr_optimize_run(
            x, y, spec, free, bounds, n_train, n_features, alpha, n_restarts,
            seed_lo, seed_hi, tp, vp, rp,
        )
        return best

    @staticmethod
    def gpc_fit(r: GPCFitRequest) raises -> Int:
        var lp = _f32_ptr(r.factor)
        var pp = _f32_ptr(r.pi)
        var wp = _f32_ptr(r.wsr)
        var sp = _f64_ptr(r.scalars)
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
                unsafe_memcpy(dest=yo, src=y.unsafe_ptr(), count=max(0, n_train))
                _ = len(y)
        else:
            y = read_f32(r.targets, max(0, n_train))
        var n_iter = 0
        n_iter = _gpc_fit_run(
            x, y, spec, n_train, n_features, max_iter_predict, lp, pp, wp, sp
        )
        return n_iter

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
        if out_kind != 0:
            gpc_binary_out(mean_addr, proba_addr, n_star, out_kind, r.dst)
        return rc

    @staticmethod
    def gpc_fit_all(r: GPCFitAllRequest) raises -> Int:
        if r.members < 1:
            raise Error("gpc_fit_all: n_fits must be positive")
        var spec = _rebuild_kernel_spec(r.kernel.kinds, r.kernel.parameters,
            r.kernel.lengths, r.kernel.scales, r.kernel.nodes, r.kernel.scale_count, "gpc_fit_all")
        var columns = gp_native_ints(r.columns, r.members)
        var targets = gp_native_ints(r.targets, r.members)
        var factors = gp_native_ints(r.factors, r.members)
        var pis = gp_native_ints(r.pis, r.members)
        var wsrs = gp_native_ints(r.wsrs, r.members)
        var scalars = gp_native_ints(r.scalars, r.members)
        gpc_fit_all_device(r.source, r.codes, r.rows, r.features, spec, r.max_iterations,
            columns, targets, factors, pis, wsrs, scalars)
        return 0

    @staticmethod
    def gpc_query_all(r: GPCQueryAllRequest) raises -> GPQueryResult:
        if r.members < 1:
            raise Error("gpc_predict_all: k must be positive")
        var spec = _rebuild_kernel_spec(r.kernel.kinds, r.kernel.parameters,
            r.kernel.lengths, r.kernel.scales, r.kernel.nodes, r.kernel.scale_count, "gpc_predict_all")
        var targets = gp_native_ints(r.targets, r.members)
        var pis = gp_native_ints(r.pis, r.members)
        var wsrs = gp_native_ints(r.wsrs, r.members)
        var factors = gp_native_ints(r.factors, r.members)
        if r.lease < -1:
            raise Error("gpc_predict_all_resident: the lease must be 0 or a lease, got " + String(r.lease))
        var fold = QueryFold(r.fold_base, r.fold_rows, r.fold_index)
        var words = gpc_predict_all_device(r.training, r.training_rows, r.features, spec,
            r.source, r.rows, targets, pis, wsrs, factors, r.output_kind, r.dst, r.lease, fold)
        return GPQueryResult(0, Int(words[0]), Int(words[1]))

    @staticmethod
    def gpc_combine(r: GPCCombineRequest) raises -> Int:
        var columns = gp_native_ints(r.columns, r.members)
        gpc_ovr_combine_host(columns, r.probabilities, r.codes, r.rows)
        return 0

    @staticmethod
    def theta(r: GPThetaRequest) raises -> Int:
        var values = gp_native_f64(r.source, r.count)
        var result = gp_theta_params_values(values)
        if r.count > 0:
            unsafe_memcpy(dest=f64_ptr(r.dst), src=result.unsafe_ptr(), count=r.count)
        return r.count
