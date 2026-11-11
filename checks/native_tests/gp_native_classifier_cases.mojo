# SPDX-License-Identifier: Apache-2.0
"""Original 48x3 binary/OVR GP workload through actual typed providers.

Original labels were ranked from the Python random.gauss Float64 observations
before Float32 admission. No classifier optimizer or Newton oracle is copied.
Probability normalization, range, latent sign and variance are independent
predicates with the original tolerances. GPU graphs are ordered by the caller.
"""
from std.math import isfinite
from core.host_parallel import host_parallelize
from core.gp_native_api import GPDriver, GPCFitRequest, GPCFitAllRequest, GPCQueryRequest, GPCQueryAllRequest, GPCCombineRequest, GP_CAPABILITIES, GP_CAP_OVR, GP_CAP_RESIDENT
from core.gp_native_storage import GPKernelStorage, GPCMemberTables
from checks.native_tests.gp_state_suite import require
from checks.native_tests.gp_native_data import input_values, input_labels, failures_clear, GP_GUARD, GP_GUARD64
from checks.native_tests.gp_native_cases import unchanged

comptime GP_CLASSIFIER_CPU_FIXTURES = 4
comptime GP_CLASSIFIER_GPU_FIXTURES = 6


def classifier_kernel(multiclass: Bool) -> GPKernelStorage:
    var kernel = GPKernelStorage()
    kernel.kinds = [0, Int32(3 if multiclass else 2), 5]
    kernel.parameters = [Float32(2.0 if multiclass else 1.0), Float32(1.5 if multiclass else 0.0), 0.0]
    kernel.lengths = [0, 1, 0]
    kernel.scales = [1.0]
    kernel.finish()
    return kernel^


def member_guards(values: List[Float32], members: Int, extent: Int, name: String) raises:
    for member in range(members):  # driver-loop: guard metadata for fitted binary members
        var at = member * (extent + 2)
        require(values[at] == GP_GUARD and values[at + extent + 1] == GP_GUARD, name)
    _ = values


def classifier_fixture(driver: GPDriver, query_driver: GPDriver, unit: Int, host: Bool) raises:
    var multiclass = (unit & 1) != 0
    var classes = 3 if multiclass else 2
    var members = 3 if multiclass else 1
    var all_members = not host and unit >= 2
    var resident = not host and unit >= 4
    var source = input_values("classifier48", "source", 144)
    var query = input_values("classifier48", "query", 30)
    var codes = input_labels("classifier48", "labels3" if multiclass else "labels2", 48)
    var source_before = source.copy()
    var query_before = query.copy()
    var kernel = classifier_kernel(multiclass)
    var tables = GPCMemberTables(members)
    var targets = List[Float32](length=50 * members, fill=GP_GUARD)
    var factors = List[Float32](length=2306 * members, fill=GP_GUARD)
    var pis = List[Float32](length=50 * members, fill=GP_GUARD)
    var wsrs = List[Float32](length=50 * members, fill=GP_GUARD)
    var means = List[Float32](length=12 * members, fill=GP_GUARD)
    var variances = List[Float32](length=12 * members, fill=GP_GUARD)
    var columns = List[Float64](length=12 * members, fill=GP_GUARD64)
    var probabilities = List[Float64](length=10 * classes + 2, fill=GP_GUARD64)
    var predicted = List[Int64](length=12, fill=-31)
    var failures = List[Int](length=48 * members, fill=0)
    var lease = -1
    tables.prepare_scalars()
    for member in range(members):  # driver-loop: address metadata for original one-versus-rest members
        tables.columns[member] = member if multiclass else 1
        tables.targets[member] = Int(targets.unsafe_ptr() + member * 50 + 1)
        tables.factors[member] = Int(factors.unsafe_ptr() + member * 2306 + 1)
        tables.pis[member] = Int(pis.unsafe_ptr() + member * 50 + 1)
        tables.wsrs[member] = Int(wsrs.unsafe_ptr() + member * 50 + 1)
    try:
        if host and unit >= 2:
            require(query_driver.metadata(GP_CAPABILITIES) == 0, "GPC inference-only image capabilities")
        if all_members:
            require((driver.metadata(GP_CAPABILITIES) & GP_CAP_OVR) != 0, "GPC selected fused all-member capability")
            require(driver.gpc_fit_all(GPCFitAllRequest(Int(source.unsafe_ptr()), Int(codes.unsafe_ptr()),
                Int(tables.columns.unsafe_ptr()), Int(tables.targets.unsafe_ptr()), Int(tables.factors.unsafe_ptr()),
                Int(tables.pis.unsafe_ptr()), Int(tables.wsrs.unsafe_ptr()), Int(tables.scalars.unsafe_ptr()),
                48, 3, kernel.view(), 100, members)) == 0, "GPC fused fit status")
        else:
            var entry = driver.entry
            var source_address = Int(source.unsafe_ptr())
            var code_address = Int(codes.unsafe_ptr())
            var target_address = Int(targets.unsafe_ptr())
            var factor_address = Int(factors.unsafe_ptr())
            var pi_address = Int(pis.unsafe_ptr())
            var wsr_address = Int(wsrs.unsafe_ptr())
            var scalar_address = Int(tables.scalar_storage.unsafe_ptr())
            var fp = failures.unsafe_ptr()
            def fit_member(member: Int) {imm entry, imm source_address, imm code_address, imm target_address, imm factor_address, imm pi_address, imm wsr_address, imm scalar_address, imm multiclass, imm fp}:
                try:
                    var member_kernel = classifier_kernel(multiclass)
                    var iterations = GPDriver(entry).gpc_fit(GPCFitRequest(source_address, code_address,
                        factor_address + 4 * (2306 * member + 1), pi_address + 4 * (50 * member + 1),
                        wsr_address + 4 * (50 * member + 1), scalar_address + 8 * 3 * member,
                        48, 3, member_kernel.view(), 100, True, member if multiclass else 1,
                        target_address + 4 * (50 * member + 1)))
                    require(iterations >= 1 and iterations <= 100, "GPC original iteration range")
                    _ = member_kernel^
                except error:
                    fp.unsafe_store(member, 1)
                    print("FAIL gp_native_classifier fit member", member, String(error))
            if host:
                host_parallelize(fit_member, members)
            else:
                for member in range(members):  # One GPU graph at a time; kernels remain parallel.
                    fit_member(member)
            failures_clear(failures, "GPC independent binary member fits")
        for member in range(members):  # driver-loop: published member scalar metadata
            require(isfinite(tables.scalar_storage[3 * member])
                and tables.scalar_storage[3 * member + 1] >= 1
                and tables.scalar_storage[3 * member + 1] <= 100
                and tables.scalar_storage[3 * member + 2] > 0, "GPC original finite likelihood and iteration range")
        var fp = failures.unsafe_ptr()
        var cp = codes.unsafe_ptr()
        var tp = targets.unsafe_ptr()
        var pp = pis.unsafe_ptr()
        var wp = wsrs.unsafe_ptr()
        var lp = factors.unsafe_ptr()
        def fitted_cell(index: Int) {imm fp, imm cp, imm tp, imm pp, imm wp, imm lp, imm multiclass}:
            var member = index // 48
            var row = index % 48
            var column = member if multiclass else 1
            var at = member * 50 + row + 1
            var expected = Float32(1.0) if Int(cp[row]) == column else Float32(0.0)
            var okay = tp[at] == expected and isfinite(pp[at]) and pp[at] >= 0 and pp[at] <= 1
            okay = okay and isfinite(wp[at]) and wp[at] >= 0
            var diagonal = lp[member * 2306 + row * 48 + row + 1]
            okay = okay and isfinite(diagonal) and diagonal > 0
            fp.unsafe_store(index, Int(not okay))
        host_parallelize(fitted_cell, 48 * members)
        failures_clear(failures, "GPC native coded-target projection and fitted matrix predicates")
        if all_members:
            if resident:
                require((query_driver.metadata(GP_CAPABILITIES) & GP_CAP_RESIDENT) != 0, "GPC resident capability")
                lease = query_driver.lease_open()
            var request = GPCQueryAllRequest(Int(source.unsafe_ptr()), Int(query.unsafe_ptr()),
                Int(probabilities.unsafe_ptr() + 1), Int(tables.targets.unsafe_ptr()), Int(tables.pis.unsafe_ptr()),
                Int(tables.wsrs.unsafe_ptr()), Int(tables.factors.unsafe_ptr()), 48, 3, 10, kernel.view(),
                2, members, lease, -1, 0, -1)
            var result = query_driver.gpc_query_all(request)
            require(result.value == 0 and result.finite == 0 and result.range_status == 0, "GPC all-member probability status")
            request.output_kind = 1
            request.dst = Int(predicted.unsafe_ptr() + 1)
            result = query_driver.gpc_query_all(request)
            require(result.value == 0 and result.finite == 0 and result.range_status == 0, "GPC all-member prediction status")
        else:
            var entry = query_driver.entry
            var source_address = Int(source.unsafe_ptr())
            var query_address = Int(query.unsafe_ptr())
            var target_address = Int(targets.unsafe_ptr())
            var factor_address = Int(factors.unsafe_ptr())
            var pi_address = Int(pis.unsafe_ptr())
            var wsr_address = Int(wsrs.unsafe_ptr())
            var mean_address = Int(means.unsafe_ptr())
            var variance_address = Int(variances.unsafe_ptr())
            var column_address = Int(columns.unsafe_ptr())
            var probability_address = Int(probabilities.unsafe_ptr() + 1)
            var predicted_address = Int(predicted.unsafe_ptr() + 1)
            def query_member(member: Int) {imm entry, imm source_address, imm query_address, imm target_address, imm factor_address, imm pi_address, imm wsr_address, imm mean_address, imm variance_address, imm column_address, imm probability_address, imm predicted_address, imm multiclass, imm fp}:
                try:
                    var member_kernel = classifier_kernel(multiclass)
                    var request = GPCQueryRequest(source_address, target_address + 4 * (50 * member + 1),
                        pi_address + 4 * (50 * member + 1), wsr_address + 4 * (50 * member + 1),
                        factor_address + 4 * (2306 * member + 1), query_address,
                        mean_address + 4 * (12 * member + 1), variance_address + 4 * (12 * member + 1),
                        column_address + 8 * (12 * member + 1), 48, 3, 10, member_kernel.view(), True,
                        0 if multiclass else 2, 0 if multiclass else probability_address)
                    _ = GPDriver(entry).gpc_query(request)
                    if not multiclass:
                        request.want_probability = False
                        request.output_kind = 1
                        request.dst = predicted_address
                        _ = GPDriver(entry).gpc_query(request)
                    _ = member_kernel^
                except error:
                    fp.unsafe_store(member, 1)
                    print("FAIL gp_native_classifier query member", member, String(error))
            if host:
                host_parallelize(query_member, members)
            else:
                for member in range(members):  # One GPU graph at a time; kernels remain parallel.
                    query_member(member)
            failures_clear(failures, "GPC independent binary member queries")
            for member in range(members):  # driver-loop: per-member probability pointer metadata
                tables.columns[member] = Int(columns.unsafe_ptr() + 12 * member + 1)
            if multiclass:
                require(query_driver.gpc_combine(GPCCombineRequest(Int(tables.columns.unsafe_ptr()),
                    Int(probabilities.unsafe_ptr() + 1), Int(predicted.unsafe_ptr() + 1), 10, members)) == 0,
                    "GPC original per-member native probability combine")
        if all_members and not multiclass:
            # Public binary latent_mean_and_variance remains a separate native
            # operation even when probability/predict use the fused all route.
            _ = query_driver.gpc_query(GPCQueryRequest(Int(source.unsafe_ptr()), tables.targets[0], tables.pis[0],
                tables.wsrs[0], tables.factors[0], Int(query.unsafe_ptr()), Int(means.unsafe_ptr() + 1),
                Int(variances.unsafe_ptr() + 1), 0, 48, 3, 10, kernel.view(), False, 0, 0))
        var prob = probabilities.unsafe_ptr() + 1
        var pred = predicted.unsafe_ptr() + 1
        var mp = means.unsafe_ptr() + 1
        var vp = variances.unsafe_ptr() + 1
        def query_cell(row: Int) {imm fp, imm prob, imm pred, imm mp, imm vp, imm classes, imm multiclass}:
            var total = Float64(0)
            var best = 0
            var okay = True
            for label in range(classes):  # Per-row class-axis oracle; rows execute in parallel.
                var value = prob[row * classes + label]
                okay = okay and isfinite(value) and value >= 0 and value <= 1
                total += value
                if value > prob[row * classes + best]:
                    best = label
            okay = okay and abs(total - Float64(1)) < Float64(1e-12)
            if multiclass:
                okay = okay and pred[row] == Int64(best)
            else:
                okay = okay and isfinite(mp[row]) and isfinite(vp[row]) and vp[row] > 0
                okay = okay and pred[row] == Int64(1 if mp[row] > 0 else 0)
            fp.unsafe_store(row, Int(not okay))
        host_parallelize(query_cell, 10)
        failures_clear(failures, "GPC original1e-12 row normalization, ranges, binary latent sign/positive variance")
        member_guards(targets, members, 48, "GPC target guards")
        member_guards(factors, members, 2304, "GPC factor guards")
        member_guards(pis, members, 48, "GPC pi guards")
        member_guards(wsrs, members, 48, "GPC weight guards")
        member_guards(means, members, 10, "GPC mean guards")
        member_guards(variances, members, 10, "GPC variance guards")
        for member in range(members):  # driver-loop: per-member probability guard metadata
            require(columns[12 * member] == GP_GUARD64 and columns[12 * member + 11] == GP_GUARD64,
                "GPC probability-column guards")
        require(probabilities[0] == GP_GUARD64 and probabilities[len(probabilities) - 1] == GP_GUARD64,
            "GPC combined probability guards")
        require(predicted[0] == Int64(-31) and predicted[11] == Int64(-31), "GPC predicted-code guards")
        unchanged(source, source_before, "GPC borrowed training source unchanged")
        unchanged(query, query_before, "GPC borrowed query unchanged")
    finally:
        if lease >= 0:
            query_driver.lease_close(lease)
        _ = source^
        _ = query^
        _ = codes^
        _ = kernel^
        _ = tables^
        _ = targets^
        _ = factors^
        _ = pis^
        _ = wsrs^
        _ = means^
        _ = variances^
        _ = columns^
        _ = probabilities^
        _ = predicted^
