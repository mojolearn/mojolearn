# SPDX-License-Identifier: Apache-2.0
"""Original GP analytic-gradient, finite-difference and optimizer workloads.

CPU cases and gradient perturbations are independent host work. GPU wrappers
order the graphs and perturbations on one physical GPU; all product kernels
retain their parallel implementation. No reference optimizer is reproduced.
"""
from std.math import isfinite, log
from core.host_parallel import host_parallelize
from core.gp_native_api import GPDriver, GPRFitRequest, GPRGradientRequest, GPROptimizeRequest, GPThetaRequest
from core.gp_native_storage import GPKernelStorage
from checks.native_tests.gp_state_suite import require
from checks.native_tests.gp_native_data import input_values, reference_values, kernel_storage, guards, guards64, failures_clear, GP_GUARD, GP_GUARD64

comptime GP_LEARNING_FIXTURES = 6


def values_for(kind: Int) -> List[Float64]:
    if kind == 2:
        return [1.5, 0.8, 1.2, 2.0, 0.2]
    if kind == 3:
        return [1.0, 1.1, 0.1]
    if kind == 4:
        return [0.7, 1.0, 0.6, 1.4, 0.1]
    if kind == 5:
        return [0.9, 0.3, 0.2]
    if kind == 6:
        return [1.0, 1.0, 1.0, 1.0, 0.1]
    return [1.0, 1.0, 0.1]


def apply_values(mut kernel: GPKernelStorage, values: List[Float64], offset: Int, kind: Int):
    if kind == 5:
        kernel.scales[0] = Float32(values[offset])
        kernel.parameters[1] = Float32(values[offset + 1])
        kernel.parameters[3] = Float32(values[offset + 2])
    else:
        kernel.parameters[0] = Float32(values[offset])
        for leaf in range(len(kernel.scales)):  # driver-loop: one supplied parameter per ARD kernel leaf
            kernel.scales[leaf] = Float32(values[offset + 1 + leaf])
        kernel.parameters[3] = Float32(values[offset + len(kernel.scales) + 1])
    _ = values


def perturbed_likelihood(driver: GPDriver, source: Int, target: Int,
                        kind: Int, parameter: Int, step: Float64) raises -> Float64:
    var kernel = kernel_storage(kind)
    var values = values_for(kind)
    var count = len(values)
    var theta = List[Float64](length=count, fill=0)
    var tp = theta.unsafe_ptr()
    var vp = values.unsafe_ptr()
    def coordinate(index: Int) {imm tp, imm vp, imm parameter, imm step}:
        tp.unsafe_store(index, log(vp[index]) + (step if index == parameter else Float64(0)))
    host_parallelize(coordinate, count)
    var converted = List[Float64](length=count + 2, fill=GP_GUARD64)
    var flags: List[Int32] = [1, 1, 0, 1, 0]
    var gradient = List[Float64](length=count + 2, fill=GP_GUARD64)
    var scalars = List[Float64](length=4, fill=GP_GUARD64)
    try:
        require(driver.theta(GPThetaRequest(Int(theta.unsafe_ptr()), Int(converted.unsafe_ptr() + 1), count)) == count,
            "GP finite-difference native theta conversion count")
        apply_values(kernel, converted, 1, kind)
        var status = driver.gpr_gradient(GPRGradientRequest(source, target, Int(flags.unsafe_ptr()),
            Int(gradient.unsafe_ptr() + 1), Int(scalars.unsafe_ptr() + 1), 40, 3, kernel.view(), 0.00000095367431640625))
        require(status == 0 and scalars[1] == 0 and isfinite(scalars[2]), "GP finite-difference perturbation succeeds")
        guards64(converted, "GP theta conversion guards")
        guards64(gradient, "GP perturbed gradient guards")
        guards64(scalars, "GP perturbed likelihood guards")
        return scalars[2]
    finally:
        _ = kernel^
        _ = values^
        _ = theta^
        _ = converted^
        _ = flags^
        _ = gradient^
        _ = scalars^


def gradient_case(driver: GPDriver, unit: Int, host: Bool) raises:
    var names: List[String] = ["gradient_rbf40", "gradient_matern05_40", "gradient_matern15_40", "gradient_matern25_40"]
    var name = names[unit].copy()
    var kind = unit + 2
    var source = input_values(name, "source", 120)
    var target = input_values(name, "target", 40)
    var kernel = kernel_storage(kind)
    var values = values_for(kind)
    var count = len(values)
    var flags: List[Int32] = [1, 1, 0, 1, 0]
    var gradient = List[Float64](length=count + 2, fill=GP_GUARD64)
    var scalars = List[Float64](length=4, fill=GP_GUARD64)
    var expected = reference_values(name, "gradient", count)
    var statistics = reference_values(name, "statistics", 5)
    var differences = List[Float64](length=count, fill=0)
    var failures = List[Int](length=count, fill=0)
    try:
        var status = driver.gpr_gradient(GPRGradientRequest(Int(source.unsafe_ptr()), Int(target.unsafe_ptr()),
            Int(flags.unsafe_ptr()), Int(gradient.unsafe_ptr() + 1), Int(scalars.unsafe_ptr() + 1),
            40, 3, kernel.view(), 0.00000095367431640625))
        require(status == 0 and scalars[1] == 0, "GP original analytic-gradient fit succeeds")
        require(abs(scalars[2] - statistics[2]) < Float64(1e-3) * (abs(statistics[2]) + 1),
            "GP original independent likelihood tolerance")
        var entry = driver.entry
        var source_address = Int(source.unsafe_ptr())
        var target_address = Int(target.unsafe_ptr())
        var dp = differences.unsafe_ptr()
        var fp = failures.unsafe_ptr()
        def difference(parameter: Int) {imm entry, imm source_address, imm target_address, imm kind, imm dp, imm fp}:
            try:
                var high = perturbed_likelihood(GPDriver(entry), source_address, target_address, kind, parameter, 0.01)
                var low = perturbed_likelihood(GPDriver(entry), source_address, target_address, kind, parameter, -0.01)
                dp.unsafe_store(parameter, (high - low) / Float64(0.02))
            except error:
                fp.unsafe_store(parameter, 1)
                print("FAIL gp_native_gradient perturbation", kind, parameter, String(error))
        if host:
            host_parallelize(difference, count)
        else:
            for parameter in range(count):  # One GPU graph at a time; original parameter perturbations unchanged.
                difference(parameter)
        failures_clear(failures, "GP gradient perturbation operations")
        var gp = gradient.unsafe_ptr() + 1
        var ep = expected.unsafe_ptr()
        def oracle(parameter: Int) {imm gp, imm ep, imm dp, imm fp}:
            var value = gp[parameter]
            var okay = isfinite(value) and abs(value - ep[parameter]) / (abs(ep[parameter]) + 1) < Float64(1e-3)
            okay = okay and abs(value - dp[parameter]) / (abs(dp[parameter]) + 1) < Float64(5e-3)
            fp.unsafe_store(parameter, Int(not okay))
        host_parallelize(oracle, count)
        failures_clear(failures, "GP original analytic-gradient and central-difference bounds")
        guards64(gradient, "GP original gradient guards")
        guards64(scalars, "GP original gradient scalar guards")
    finally:
        _ = source^
        _ = target^
        _ = kernel^
        _ = values^
        _ = flags^
        _ = gradient^
        _ = scalars^
        _ = expected^
        _ = statistics^
        _ = differences^


def optimizer_case(driver: GPDriver, restarts: Bool) raises:
    var name = "restart48" if restarts else "optimize48"
    var kind = 6 if restarts else 1
    var source = input_values(name, "source", 144)
    var target = input_values(name, "target", 48)
    var kernel = kernel_storage(kind)
    var count = 5 if restarts else 3
    var starts = 3 if restarts else 1
    var flags: List[Int32] = [1, 1, 0, 1, 0]
    var bounds = List[Float32](length=2 * count, fill=0)
    var bp = bounds.unsafe_ptr()
    def bound(index: Int) {imm bp}:
        bp.unsafe_store(2 * index, Float32(1e-5))
        bp.unsafe_store(2 * index + 1, Float32(1e5))
    host_parallelize(bound, count)
    var theta = List[Float64](length=count + 2, fill=GP_GUARD64)
    var values = List[Float64](length=count + 2, fill=GP_GUARD64)
    var runs = List[Float64](length=4 * starts + 2, fill=GP_GUARD64)
    var gradient = List[Float64](length=count + 2, fill=GP_GUARD64)
    var initial = List[Float64](length=4, fill=GP_GUARD64)
    var final_scalars = List[Float64](length=4, fill=GP_GUARD64)
    var factor = List[Float32](length=48 * 48 + 2, fill=GP_GUARD)
    var dual = List[Float32](length=50, fill=GP_GUARD)
    var fitted = List[Float64](length=7, fill=GP_GUARD64)
    try:
        require(driver.gpr_gradient(GPRGradientRequest(Int(source.unsafe_ptr()), Int(target.unsafe_ptr()),
            Int(flags.unsafe_ptr()), Int(gradient.unsafe_ptr() + 1), Int(initial.unsafe_ptr() + 1),
            48, 3, kernel.view(), 0.00000095367431640625)) == 0, "GP optimizer starting likelihood")
        var best = driver.gpr_optimize(GPROptimizeRequest(Int(source.unsafe_ptr()), Int(target.unsafe_ptr()),
            Int(flags.unsafe_ptr()), Int(bounds.unsafe_ptr()), Int(theta.unsafe_ptr() + 1),
            Int(values.unsafe_ptr() + 1), Int(runs.unsafe_ptr() + 1), 48, 3, kernel.view(),
            0.00000095367431640625, 2 if restarts else 0, 7 if restarts else 0, 0))
        require(best >= 0 and best < starts, "GP optimizer selected run index")
        var first = 0
        for run in range(starts):  # driver-loop: metadata for original optimizer starts
            if runs[1 + 4 * run + 3] < runs[1 + 4 * first + 3]:
                first = run
        require(best == first, "GP optimizer smallest negative likelihood wins; earliest tie")
        require(isfinite(runs[1 + 4 * best + 3]), "GP selected optimizer run has finite objective")
        if not restarts:
            require(runs[1] > 0 and -runs[4] > initial[2], "GP original optimizer takes steps and improves likelihood")
        var failures = List[Int](length=count, fill=0)
        var fp = failures.unsafe_ptr()
        var vp = values.unsafe_ptr() + 1
        var tp = theta.unsafe_ptr() + 1
        def parameter(index: Int) {imm fp, imm vp, imm tp}:
            fp.unsafe_store(index, Int(not (isfinite(vp[index]) and vp[index] > 0
                and Float64(Float32(vp[index])) == vp[index] and isfinite(tp[index]))))
        host_parallelize(parameter, count)
        failures_clear(failures, "GP optimized parameters retain exact admitted Float32 values")
        apply_values(kernel, values, 1, kind)
        require(driver.gpr_fit(GPRFitRequest(Int(source.unsafe_ptr()), Int(target.unsafe_ptr()),
            Int(factor.unsafe_ptr() + 1), Int(dual.unsafe_ptr() + 1), Int(fitted.unsafe_ptr() + 1),
            48, 3, kernel.view(), 0.00000095367431640625, False)) == 0, "GP fit of selected optimized kernel")
        require(driver.gpr_gradient(GPRGradientRequest(Int(source.unsafe_ptr()), Int(target.unsafe_ptr()),
            Int(flags.unsafe_ptr()), Int(gradient.unsafe_ptr() + 1), Int(final_scalars.unsafe_ptr() + 1),
            48, 3, kernel.view(), 0.00000095367431640625)) == 0, "GP selected kernel likelihood reevaluation")
        require(fitted[5] == -runs[1 + 4 * best + 3] and final_scalars[2] == fitted[5],
            "GP original fit, selected run and evaluated likelihood consistency")
        guards64(theta, "GP optimizer theta guards")
        guards64(values, "GP optimizer values guards")
        guards64(runs, "GP optimizer run guards")
        guards64(gradient, "GP optimizer gradient guards")
        guards64(initial, "GP optimizer starting scalar guards")
        guards64(final_scalars, "GP optimizer final scalar guards")
        guards64(fitted, "GP optimizer final fit scalar guards")
        guards(factor, "GP optimizer final factor guards")
        guards(dual, "GP optimizer final dual guards")
    finally:
        _ = source^
        _ = target^
        _ = kernel^
        _ = flags^
        _ = bounds^
        _ = theta^
        _ = values^
        _ = runs^
        _ = gradient^
        _ = initial^
        _ = final_scalars^
        _ = factor^
        _ = dual^
        _ = fitted^


def learning_fixture(driver: GPDriver, unit: Int, host: Bool) raises:
    if unit < 4:
        gradient_case(driver, unit, host)
    else:
        optimizer_case(driver, unit == 5)
