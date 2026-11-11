# SPDX-License-Identifier: Apache-2.0
"""Original DBSCAN fits/queries against independent mathematical graph refs.

Cluster numbers may be permuted. Distinct core components must remain
distinct, borders must attach to an allowed core component, and core/noise
membership is exact. Float32 source bytes and original settings are retained.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.dbscan_native_api import DBSCANNativeDriver, DBSCANFitRequest, DBSCANCoreRequest, DBSCANGatherRequest, DBSCANReferenceRequest, DBSCANQueryRequest, DBSCANResidentQuery, DBSCANFitWorkspace, dbscan_empty_labels, DBSCAN_MODE, DBSCAN_VENDOR, DBSCAN_CAPABILITIES, DBSCAN_CAP_RESIDENT
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.native_tests.dbscan_native_profiles import DBSCANCase, profile, query_count, query_name, query_rows
from checks.native_tests.pca_native_cases import guards, unchanged, PCA_TEST_GUARD

comptime DBSCAN_NATIVE_FIXTURES = 17
comptime DBSCAN_I32_GUARD = Int32(987654)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def input_bytes(name: String, count: Int) raises -> List[UInt8]:
    with open("checks/native_tests/fixtures/dbscan/" + name, "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == count, "DBSCAN independent artifact extent " + name)
        return bytes^


def input_f32(name: String, count: Int) raises -> List[Float32]:
    var bytes = input_bytes(name + ".f32", 4 * count)
    var values = List[Float32](unsafe_uninit_length=count)
    var bp = bytes.unsafe_ptr()
    var vp = values.unsafe_ptr()
    def cell(index: Int) {imm bp, imm vp}:
        var at = 4 * index
        var bits = UInt32(bp[at]) | (UInt32(bp[at + 1]) << 8) | (UInt32(bp[at + 2]) << 16) | (UInt32(bp[at + 3]) << 24)
        vp.unsafe_store(index, bitcast[DType.float32](bits))
    host_parallelize(cell, count)
    _ = bytes^
    return values^


def input_i32(name: String, count: Int) raises -> List[Int32]:
    var bytes = input_bytes(name + ".i32", 4 * count)
    var values = List[Int32](unsafe_uninit_length=count)
    var bp = bytes.unsafe_ptr()
    var vp = values.unsafe_ptr()
    def cell(index: Int) {imm bp, imm vp}:
        var at = 4 * index
        var bits = UInt32(bp[at]) | (UInt32(bp[at + 1]) << 8) | (UInt32(bp[at + 2]) << 16) | (UInt32(bp[at + 3]) << 24)
        vp.unsafe_store(index, bitcast[DType.int32](bits))
    host_parallelize(cell, count)
    _ = bytes^
    return values^


def integer_guards(values: List[Int32]) raises:
    require(values[0] == DBSCAN_I32_GUARD and values[len(values) - 1] == DBSCAN_I32_GUARD,
        "DBSCAN int32 destination canaries")


def fit_oracle(p: DBSCANCase, labels: List[Int32], core: List[UInt8]) raises:
    var expected_core = input_bytes(p.name + "-core.u8", p.rows)
    var noise = input_bytes(p.name + "-noise.u8", p.rows)
    var anchors = input_i32(p.name + "-anchors", p.clusters)
    var allowed = input_bytes(p.name + "-allowed.u8", p.rows * p.clusters)
    var failures = List[Int](length=p.rows + p.clusters * p.clusters, fill=0)
    var fp = failures.unsafe_ptr()
    var lp = labels.unsafe_ptr() + 1
    var cp = core.unsafe_ptr()
    var ep = expected_core.unsafe_ptr()
    var np = noise.unsafe_ptr()
    var ap = anchors.unsafe_ptr()
    var bp = allowed.unsafe_ptr()
    var rows = p.rows
    var clusters = p.clusters
    def cell(index: Int) {imm fp, imm lp, imm cp, imm ep, imm np, imm ap, imm bp, imm rows, imm clusters}:
        var okay = True
        if index < rows:
            okay = cp[index] == ep[index]
            if np[index] != 0:
                okay = okay and lp[index] == Int32(-1)
            else:
                var admitted = False
                for group in range(clusters):  # Independent oracle membership, not a product clustering algorithm.
                    if bp[index * clusters + group] != 0 and lp[index] == lp[Int(ap[group])]:
                        admitted = True
                okay = okay and lp[index] >= Int32(0) and admitted
        else:
            var pair = index - rows
            var first = pair // clusters
            var second = pair % clusters
            okay = lp[Int(ap[first])] >= Int32(0)
            if first != second:
                okay = okay and lp[Int(ap[first])] != lp[Int(ap[second])]
        fp.unsafe_store(index, Int(not okay))
    host_parallelize(cell, len(failures))
    for failure in failures:
        require(failure == 0, "DBSCAN independent core/noise/partition oracle " + p.name)
    integer_guards(labels)
    _ = labels
    _ = core
    _ = expected_core^
    _ = noise^
    _ = anchors^
    _ = allowed^


def gather_oracle(p: DBSCANCase, source: List[Float32], labels: List[Int32], indices: List[Int32],
                  components: List[Float32], core_labels: List[Int32]) raises:
    var expected = input_i32(p.name + "-core-indices", p.core_count)
    var failures = List[Int](length=p.core_count * (p.features + 2), fill=0)
    var fp = failures.unsafe_ptr()
    var ep = expected.unsafe_ptr()
    var xp = source.unsafe_ptr()
    var lp = labels.unsafe_ptr() + 1
    var ip = indices.unsafe_ptr() + 1
    var cp = components.unsafe_ptr() + 1
    var kp = core_labels.unsafe_ptr() + 1
    var features = p.features
    def cell(index: Int) {imm fp, imm ep, imm xp, imm lp, imm ip, imm cp, imm kp, imm features}:
        var row = index // (features + 2)
        var col = index % (features + 2)
        var original = Int(ep[row])
        var okay = True
        if col == features:
            okay = ip[row] == ep[row]
        elif col == features + 1:
            okay = kp[row] == lp[original]
        else:
            okay = cp[row * features + col] == xp[original * features + col]
        fp.unsafe_store(index, Int(not okay))
    host_parallelize(cell, len(failures))
    for failure in failures:
        require(failure == 0, "DBSCAN core gather order/values/labels " + p.name)
    integer_guards(indices)
    integer_guards(core_labels)
    guards(components, "DBSCAN core components")
    _ = source
    _ = labels
    _ = indices
    _ = components
    _ = core_labels
    _ = expected^


def query_oracle(labels: List[Int32], indices: List[Int32], expected: List[Int32],
                 dst: List[Int32], chosen: List[Int32], borrowed: Bool) raises:
    var count = len(expected)
    var core_count = len(indices) - 2
    var failures = List[Int](length=count, fill=0)
    var fp = failures.unsafe_ptr()
    var ep = expected.unsafe_ptr()
    var lp = labels.unsafe_ptr() + 1
    var ip = indices.unsafe_ptr() + 1
    var dp = dst.unsafe_ptr() + 1
    var cp = chosen.unsafe_ptr() + 1
    def cell(row: Int) {imm fp, imm ep, imm lp, imm ip, imm dp, imm cp, imm borrowed, imm core_count}:
        var original = Int(ep[row])
        var wanted = Int32(-1) if original < 0 else lp[original]
        var okay = dp[row] == wanted
        if borrowed:
            var selected = Int(cp[row])
            if original < 0:
                okay = okay and selected == -1
            elif selected < 0 or selected >= core_count:
                okay = False
            else:
                okay = okay and ip[selected] == ep[row]
        fp.unsafe_store(row, Int(not okay))
    host_parallelize(cell, count)
    for failure in failures:
        require(failure == 0, "DBSCAN independent nearest-core radius/lowest-original-index tie oracle")
    integer_guards(dst)
    integer_guards(chosen)
    _ = labels
    _ = indices
    _ = expected
    _ = dst
    _ = chosen


def single_query(driver: DBSCANNativeDriver, request: DBSCANQueryRequest, labels: Int,
                 keys: Int, expected: Int, core_count: Int) raises:
    var dst = List[Int32](length=3, fill=DBSCAN_I32_GUARD)
    var chosen = List[Int32](length=3, fill=DBSCAN_I32_GUARD)
    var actual = request.copy()
    actual.dst = Int(dst.unsafe_ptr()) + 4
    actual.chosen = Int(chosen.unsafe_ptr()) + 4
    require(driver.query(actual) == 0, "DBSCAN single-query return")
    var lp = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=labels)
    var kp = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=keys)
    var wanted = Int32(-1) if expected < 0 else lp[expected]
    require(dst[1] == wanted, "DBSCAN original single-row query label")
    if expected < 0:
        require(chosen[1] == Int32(-1), "DBSCAN original single-row query noise")
    else:
        require(chosen[1] >= Int32(0) and Int(chosen[1]) < core_count, "DBSCAN single-row chosen extent")
        require(kp[Int(chosen[1])] == Int32(expected), "DBSCAN single-row nearest original index")
    integer_guards(dst)
    integer_guards(chosen)
    _ = dst^
    _ = chosen^


def singles(driver: DBSCANNativeDriver, p: DBSCANCase, source: List[Float32], labels: List[Int32],
            indices: List[Int32], components: List[Float32], core_labels: List[Int32], host: Bool) raises:
    var expected = input_i32(p.name + "-query-train-core", p.rows)
    var failures = List[Int](length=p.rows, fill=0)
    var fp = failures.unsafe_ptr()
    var ep = expected.unsafe_ptr()
    var xa = Int(source.unsafe_ptr())
    var la = Int(labels.unsafe_ptr()) + 4
    var ia = Int(indices.unsafe_ptr()) + 4
    var ca = Int(components.unsafe_ptr()) + 4
    var ka = Int(core_labels.unsafe_ptr()) + 4
    var count = p.core_count
    var features = p.features
    var metric = p.metric
    var eps = p.eps
    def task(row: Int) {imm fp, imm ep, imm xa, imm la, imm ia, imm ca, imm ka, imm count, imm features, imm metric, imm eps, imm driver}:
        try:
            single_query(driver, DBSCANQueryRequest(ca, ia, ka, xa + 4 * row * features, 0, 0,
                count, 1, features, metric, True, eps), la, ia, Int(ep[row]), count)
        except error:
            fp.unsafe_store(row, 1)
            print("FAIL DBSCAN single-row query", row, String(error))
    if host:
        host_parallelize(task, p.rows)
    else:
        # One graph at a time per physical GPU/shared context; kernels stay parallel.
        for row in range(p.rows):
            task(row)
    for failure in failures:
        require(failure == 0, "DBSCAN original twenty single-row query graphs")
    _ = source
    _ = labels
    _ = indices
    _ = components
    _ = core_labels
    _ = expected^


def fixture(driver: DBSCANNativeDriver, unit: Int, host: Bool) raises:
    var p = profile(unit)
    require(driver.metadata(DBSCAN_MODE) == Int(GLOBAL_NUMERIC_MODE), "DBSCAN actual mode")
    require(driver.metadata(DBSCAN_VENDOR) == 1 if host else driver.metadata(DBSCAN_VENDOR) > 1,
        "DBSCAN actual host/device vendor")
    require(driver.metadata(DBSCAN_CAPABILITIES) == (0 if host else DBSCAN_CAP_RESIDENT), "DBSCAN resident capability")
    var source = input_f32(p.name + "-source", p.rows * p.features)
    var source_before = source.copy()
    var weights = input_f32(p.name + "-weights", p.rows if p.weighted else 0)
    var weights_before = weights.copy()
    var labels = List[Int32](length=p.rows + 2, fill=DBSCAN_I32_GUARD)
    var workspace = DBSCANFitWorkspace(p.rows)
    var indices = List[Int32](length=p.core_count + 2, fill=DBSCAN_I32_GUARD)
    var components = List[Float32](length=p.core_count * p.features + 2, fill=PCA_TEST_GUARD)
    var core_labels = List[Int32](length=p.core_count + 2, fill=DBSCAN_I32_GUARD)
    var handle = -1
    try:
        var xa = Int(source.unsafe_ptr())
        var la = Int(labels.unsafe_ptr()) + 4
        var wa = Int(weights.unsafe_ptr()) if p.weighted else 0
        var ca = workspace.address()
        var passes = driver.fit(DBSCANFitRequest(xa, la, wa, ca, p.rows, p.features, p.min_samples,
            0, 0, p.method, p.metric, p.eps))
        require(passes >= (1 if unit == 7 else 0), "DBSCAN native fit pass count; original host512 minimum one")
        fit_oracle(p, labels, workspace.core)
        require(driver.core_count(DBSCANCoreRequest(xa, la, ca, p.rows, p.features)) == p.core_count,
            "DBSCAN exact independent core count")
        require(driver.core_gather(DBSCANGatherRequest(xa, la, ca, Int(indices.unsafe_ptr()) + 4,
            Int(components.unsafe_ptr()) + 4, Int(core_labels.unsafe_ptr()) + 4, p.rows, p.features, p.core_count)) == 0,
            "DBSCAN native gather result")
        gather_oracle(p, source, labels, indices, components, core_labels)
        if not host and p.core_count > 0 and query_count(unit) > 0:
            handle = driver.prepare(DBSCANReferenceRequest(Int(components.unsafe_ptr()) + 4,
                Int(indices.unsafe_ptr()) + 4, Int(core_labels.unsafe_ptr()) + 4, p.core_count, p.features, p.metric))
            require(handle >= 0, "DBSCAN actual resident reference handle")
        # Original query batches form one fitted-state graph; GPU submissions are ordered.
        for batch in range(query_count(unit)):
            var rows = query_rows(unit, batch)
            var stem = p.name + "-query-" + query_name(unit, batch)
            var queries = input_f32(stem, rows * p.features)
            var before = queries.copy()
            var expected = input_i32(stem + "-core", rows)
            var dst = List[Int32](length=rows + 2, fill=DBSCAN_I32_GUARD)
            var chosen = List[Int32](length=rows + 2, fill=DBSCAN_I32_GUARD)
            try:
                if p.core_count == 0:
                    dbscan_empty_labels(Int(dst.unsafe_ptr()) + 4, rows)
                else:
                    require(driver.query(DBSCANQueryRequest(Int(components.unsafe_ptr()) + 4,
                        Int(indices.unsafe_ptr()) + 4, Int(core_labels.unsafe_ptr()) + 4, Int(queries.unsafe_ptr()),
                        Int(dst.unsafe_ptr()) + 4, Int(chosen.unsafe_ptr()) + 4, p.core_count, rows, p.features,
                        p.metric, True, p.eps)) == 0, "DBSCAN borrowed query return")
                query_oracle(labels, indices, expected, dst, chosen, p.core_count > 0)
                if handle >= 0:
                    var retained = List[Int32](length=rows + 2, fill=DBSCAN_I32_GUARD)
                    try:
                        require(driver.resident(DBSCANResidentQuery(Int(queries.unsafe_ptr()), Int(retained.unsafe_ptr()) + 4,
                            handle, rows, p.features, p.metric, True, p.eps)) == 0, "DBSCAN resident query return")
                        query_oracle(labels, indices, expected, retained, chosen, False)
                    finally:
                        _ = retained^
                unchanged(queries, before, "DBSCAN query input")
            finally:
                _ = queries^
                _ = before^
                _ = expected^
                _ = dst^
                _ = chosen^
        if unit == 0:
            singles(driver, p, source, labels, indices, components, core_labels, host)
        unchanged(source, source_before, "DBSCAN original fit input")
        unchanged(weights, weights_before, "DBSCAN original sample weights")
    finally:
        try:
            if handle >= 0:
                var owned = handle
                handle = -1
                require(driver.release(owned) == 0, "DBSCAN resident reference release")
        finally:
            _ = source^
            _ = source_before^
            _ = weights^
            _ = weights_before^
            _ = labels^
            _ = workspace^
            _ = indices^
            _ = components^
            _ = core_labels^
