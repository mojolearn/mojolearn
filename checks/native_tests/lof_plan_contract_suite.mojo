# SPDX-License-Identifier: Apache-2.0
"""LOF plan ownership, wire, refusal and interpolation contracts.

The native entry below is a prescribed boundary witness, not a numerical
backend: it checks ABI fields and writes fixed markers/known sorted scores.
Actual neighbor/sort/density numerical oracles belong to lof_native_cpu.
Independent fixtures run concurrently without shared mutable test state.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.neighbors_native_driver import XnDriver
from core.lof_plan import LofPlan, LofLabelsPlan, lof_f32_address, lof_i32_address, lof_matrix_count, lof_percentile, lof_offset

comptime LOF_PLAN_CONTRACT_FIXTURES = 28
comptime _F32 = MutPointer[Float32, MutUntrackedOrigin]
comptime _I32 = MutPointer[Int32, MutUntrackedOrigin]
comptime _F64 = Pointer[Float64, ImmUntrackedOrigin]


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def witness(operation: Int, a: List[Int], p: List[Int], ff: Int, nf: Int) raises:
    if operation == 0:  # sqdist; a real matrix owner, no source-data reads
        require(len(a) == 3 and len(p) == 4 and nf == 0, "sqdist envelope lengths")
        require(a[0] == 101 and a[1] == 103, "query/reference order")
        require(p[0] == 3 and p[1] == 5 and p[2] == 7 and p[3] != 0, "sqdist shape/status")
        var matrix = _F32(unsafe_from_address=a[2])
        require(matrix[0] == 0 and matrix[14] == 0, "fresh matrix must be zero initialized")
        matrix[0] = 111
        matrix[14] = 113
        var status = _I32(unsafe_from_address=p[3])
        status[0] = 23
    elif operation == 7:  # knn_select
        require(len(a) == 3 and len(p) == 4 and nf == 0, "select envelope lengths")
        require(p[0] == 3 and p[1] == 5 and p[2] == 2 and (p[3] == 0 or p[3] == 1), "select shape/exclude")
        var matrix = _F32(unsafe_from_address=a[0])
        require(matrix[0] == 111 and matrix[14] == 113, "unfused matrix owner survives selection")
        var square = _F32(unsafe_from_address=a[1])
        var indices = _I32(unsafe_from_address=a[2])
        require(square[0] == 0 and square[5] == 0 and indices[0] == 0 and indices[5] == 0, "fresh selection owners")
        square[0] = 4
        square[5] = 9
        indices[0] = 1
        indices[5] = 4
    elif operation == 8 or operation == 63:  # legacy fused / tiled
        require(len(a) == 4 and len(p) == 6 and nf == 0, "fused envelope lengths")
        require(a[0] == 101 and a[1] == 103, "fused query/reference order")
        require(p[0] == 3 and p[1] == 5 and p[2] == 7 and p[3] == 2 and p[5] != 0, "fused dimensions/status")
        require(p[4] == 0 or p[4] == 1, "fused exclusion flag")
        var square = _F32(unsafe_from_address=a[2])
        var indices = _I32(unsafe_from_address=a[3])
        require(square[0] == 0 and square[5] == 0 and indices[0] == 0 and indices[5] == 0, "fresh fused owners")
        square[0] = 4
        square[5] = 9
        indices[0] = 1
        indices[5] = 4
        var status = _I32(unsafe_from_address=p[5])
        status[0] = 23
    elif operation == 6:  # unary
        require(len(a) == 2 and len(p) == 2 and nf == 2 and ff != 0, "unary envelope lengths")
        var f = _F64(unsafe_from_address=ff)
        require(f[0] == 1.0 and a[0] != a[1], "unary scale and detached destination")
        var source = _F32(unsafe_from_address=a[0])
        var dst = _F32(unsafe_from_address=a[1])
        if p[1] == 2:
            require(p[0] == 6 and f[1] == 0.0, "sqrt shape/constants")
            require(source[0] == 4 and source[5] == 9, "square owner survives unary")
            require(dst[0] == 0 and dst[5] == 0, "root destination zeros")
            dst[0] = 2
            dst[5] = 3
        else:
            require(p[0] == 3 and p[1] == 6 and f[1] == 1.5, "decision shape/constants")
            require(source[0] == -2 and source[2] == -4, "score owner survives decision")
            require(dst[0] == 0 and dst[2] == 0, "decision destination zeros")
            dst[0] = -0.5
            dst[2] = -2.5
    elif operation == 13:  # lof_lrd
        require(len(a) == 4 and len(p) == 3 and nf == 0, "density envelope lengths")
        require(p[0] == 2 and p[1] == 3 and p[2] == 11, "density k/query/fitted metadata order")
        require(a[2] == 107, "fit distance source slot")
        var distances = _F32(unsafe_from_address=a[0])
        var indices = _I32(unsafe_from_address=a[1])
        require(distances[0] == 2 and distances[5] == 3 and indices[0] == 1 and indices[5] == 4, "density sources remain owned")
        var dst = _F32(unsafe_from_address=a[3])
        require(dst[0] == 0 and dst[2] == 0, "density destination zeros")
        dst[0] = 13
        dst[2] = 17
    elif operation == 14:  # lof_score
        require(len(a) == 4 and len(p) == 3 and nf == 0, "score envelope lengths")
        require(p[0] == 2 and p[1] == 3 and p[2] == 11 and a[1] == 109, "score k/query/fitted source order")
        var indices = _I32(unsafe_from_address=a[0])
        var densities = _F32(unsafe_from_address=a[2])
        require(indices[0] == 1 and indices[5] == 4 and densities[0] == 13 and densities[2] == 17, "score sources remain owned")
        var dst = _F32(unsafe_from_address=a[3])
        require(dst[0] == 0 and dst[2] == 0, "score destination zeros")
        dst[0] = -2
        dst[2] = -4
    elif operation == 47:  # p2m_sign_label
        require(len(a) == 2 and len(p) == 2 and nf == 1 and ff != 0, "label envelope lengths")
        require(a[0] == 101 and p[0] == 3 and (p[1] == 0 or p[1] == 1), "label input/count/mode")
        require(_F64(unsafe_from_address=ff)[0] == (-0.5 if p[1] == 0 else 0.0), "label threshold")
        var dst = _I32(unsafe_from_address=a[1])
        require(dst[0] == 0 and dst[1] == 0 and dst[2] == 0, "label destination zeros")
        dst[0] = -1
        dst[1] = 1
        dst[2] = -1 if p[1] == 0 else 1
    elif operation == 56:  # prescribed row-sort result for interpolation
        require(len(a) == 5 and len(p) == 4 and nf == 0, "sort envelope lengths")
        require(p[0] == 1 and p[1] == 4 and p[2] == 4 and p[3] == 3, "sort network shape")
        require(a[2] == 101, "sort score input slot")
        var offsets = _I32(unsafe_from_address=a[0])
        var columns = _I32(unsafe_from_address=a[1])
        var output_columns = _I32(unsafe_from_address=a[3])
        var dst = _F32(unsafe_from_address=a[4])
        require(offsets[0] == 0 and offsets[1] == 4, "sort offsets")
        require(columns[0] == 0 and columns[3] == 0 and output_columns[0] == 0 and output_columns[3] == 0,
            "sort index buffers are independent zero owners")
        require(dst[0] == 0 and dst[3] == 0 and a[1] != a[3], "sort output starts detached and zeroed")
        dst[0] = -9
        dst[1] = -3
        dst[2] = 1
        dst[3] = 5
    else:
        raise Error("unexpected native operation " + String(operation))


def witness_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        var a = native_descriptor_words(aa, na) if na else List[Int]()
        var p = native_descriptor_words(pp, np) if np else List[Int]()
        witness(operation, a, p, ff, nf)
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return -1


def refusal_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    native_descriptor_report(Error("prescribed native refusal"), context, sink)
    return -1


def route_entry[EXPECTED: Int, EXCLUDE: Int](operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if operation == 0 or operation == 7 or operation == 8 or operation == 63:
            require(operation == EXPECTED or (EXPECTED == 0 and operation == 7), "selected neighbor route changed")
            var p = native_descriptor_words(pp, np)
            if operation == 7:
                require(np == 4 and p[3] == EXCLUDE, "unfused exclusion flag changed")
            elif operation == 8 or operation == 63:
                require(np == 6 and p[4] == EXCLUDE, "fused exclusion flag changed")
        return witness_entry(operation, aa, na, pp, np, ff, nf, result, context, sink)
    except error:
        native_descriptor_report(error, context, sink)
        return -1


def fixture(unit: Int) raises:
    var entry: NativeDescriptorCall = witness_entry
    var fail_entry: NativeDescriptorCall = refusal_entry
    var driver = XnDriver(entry)
    var failing = XnDriver(fail_entry)
    if unit < 6:
        var route = unit // 2
        var selected: NativeDescriptorCall = route_entry[63, 0]
        if unit == 1:
            selected = route_entry[63, 1]
        elif unit == 2:
            selected = route_entry[8, 0]
        elif unit == 3:
            selected = route_entry[8, 1]
        elif unit == 4:
            selected = route_entry[0, 0]
        elif unit == 5:
            selected = route_entry[0, 1]
        driver = XnDriver(selected)
        var plan = LofPlan(3, 5, 7, 2, unit % 2 == 1, route == 2, route == 1 or unit == 5)
        require(len(plan.squared) == 0 and len(plan.indices) == 0, "construction must defer scratch")
        require(plan.select(driver, 101, 103) == 23, "selection status survives native return")
        var destination = List[Float32](length=6, fill=0)
        plan.roots(driver, Int(destination.unsafe_ptr()))
        require(destination[0] == 2 and destination[5] == 3 and len(plan.squared) == 0, "root stage releases only squared scratch")
        require(plan.indices[0] == 1 and plan.indices[5] == 4, "indices survive square release")
        _ = destination^
    elif unit == 6 or unit == 7:
        var plan = LofPlan(3, 5, 7, 2, False, unit == 7)
        require(plan.query_neighbors(driver, 101, 103) == 23, "query finite result")
        require(len(plan.densities) == 0 and len(plan.scores) == 0, "late query stages remain unallocated")
        plan.query_density(driver, 107, 11)
        require(plan.densities[0] == 13 and plan.densities[2] == 17 and len(plan.scores) == 0, "density stage")
        plan.query_score(driver, 109, 11)
        require(plan.scores[0] == -2 and plan.scores[2] == -4 and len(plan.decisions) == 0, "score stage")
        plan.query_decision(driver, 1.5)
        require(plan.decisions[0] == -0.5 and plan.decisions[2] == -2.5, "private decision stage")
    elif unit == 8 or unit == 9:
        var labels = LofLabelsPlan(3)
        labels.classify(driver, 101, unit - 8, -0.5 if unit == 8 else 0.0)
        var destination = List[Int64](length=3, fill=0)
        labels.write_i64(Int(destination.unsafe_ptr()))
        require(destination[0] == -1 and destination[1] == 1 and destination[2] == (-1 if unit == 8 else 1), "native label widening")
        _ = labels^
    elif unit == 10:
        var labels = LofLabelsPlan(0)
        labels.classify(failing, 0, 0, 0.0)
        labels.write_i64(0)
    elif unit == 11:
        var f = List[Float32]()
        var i = List[Int32]()
        require(lof_f32_address(f) == 0 and lof_i32_address(i) == 0, "empty scratch preserves null ABI")
        require(lof_matrix_count(0, 5) == 0, "zero rows preserve empty shape")
    elif unit < 18:
        var quantiles: List[Float64] = [0, 12.5, 25, 50, 75, 100]
        var expected: List[Float64] = [-9, -6.75, -4.5, -1, 2, 5]
        require(lof_percentile(driver, 101, 4, quantiles[unit - 12]) == expected[unit - 12], "independent four-point linear interpolation")
    elif unit == 18:
        var offset = lof_offset(driver, 101, 4, 0.1)
        require(bitcast[DType.uint32](Float32(offset)) == UInt32(0xc0e66666), "contamination rounds once to Float32 -7.2")
        require(offset == Float64(Float32(-7.2)), "public offset widens the rounded Float32")
    elif unit < 22:
        var message = String("<no refusal>")
        var expected = String("ValueError: negative dimensions are not allowed")
        try:
            if unit == 19:
                _ = lof_matrix_count(3, -1)
            elif unit == 20:
                expected = "ValueError: array element count exceeds native storage"
                _ = lof_matrix_count(2, 9223372036854775807)
            else:
                expected = "ValueError: array byte count exceeds native storage"
                _ = lof_matrix_count(1, 2305843009213693952)
        except error:
            message = String(error)
        require(message == expected, "native shape refusal: " + message)
    elif unit < 26:
        var plan = LofPlan(3, 5, 7, 2, False, unit == 25)
        var destination = List[Float32](length=6, fill=0)
        var message = String("<no refusal>")
        try:
            if unit == 22:
                _ = plan.select(driver, 101, 103)
                plan.roots(failing, Int(destination.unsafe_ptr()))
            elif unit == 23:
                _ = plan.query_neighbors(driver, 101, 103)
                plan.query_density(failing, 107, 11)
            elif unit == 24:
                _ = lof_percentile(failing, 101, 4, 25)
            else:
                _ = plan.select(failing, 101, 103)
        except error:
            message = String(error)
        _ = destination^
        require(message == "prescribed native refusal", "native refusal must propagate before consuming unfilled output")
        require(len(plan.scores) == 0 and len(plan.decisions) == 0, "refusal must not advance later stages")
        if unit == 22:
            require(len(plan.squared) == 6 and len(plan.indices) == 6 and len(plan.distances) == 0,
                "root refusal retains the unconsumed selection owners")
        elif unit == 23:
            require(len(plan.distances) == 6 and len(plan.densities) == 3, "density refusal preserves earlier stage owners")
        if unit == 25:
            require(len(plan.indices) == 0 and len(plan.squared) == 0, "unfused sqdist refusal precedes selection allocation")
    elif unit == 26:
        var plan = LofPlan(3, 5, 7, 2, False)
        _ = plan.select(driver, 101, 103)
        _ = plan.select(driver, 101, 103)
        require(plan.indices[0] == 1 and plan.indices[5] == 4, "repeated selection owns fresh initialized scratch")
    else:
        var plan = LofPlan(3, 5, 7, 2, False)
        _ = plan.select(driver, 101, 103)
        var retained = plan^
        var destination = List[Float32](length=6, fill=0)
        retained.roots(driver, Int(destination.unsafe_ptr()))
        require(destination[0] == 2 and destination[5] == 3 and retained.indices[5] == 4, "moving the plan retains both buffer owners")
        _ = destination^


def run_checks() raises:
    var failures = List[Int](length=LOF_PLAN_CONTRACT_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def run(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp[unit] = 1
            print("FAIL lof_plan_contract fixture", unit, String(error))
    host_parallelize(run, LOF_PLAN_CONTRACT_FIXTURES)
    var failed = 0
    for value in failures:  # metadata: fixture summaries
        failed += value
    require(failed == 0, "native LOF plan contracts failed " + String(failed) + " fixtures")
    print("PASS lof_plan_contract:", LOF_PLAN_CONTRACT_FIXTURES, "native state/wire/interpolation/refusal fixtures")
