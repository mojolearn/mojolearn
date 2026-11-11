# SPDX-License-Identifier: Apache-2.0
"""Actual typed Agglomerative graphs and independent mathematical references.

Single linkage uses a saved Float64 minimax closure, independent of an MST.
Ward references recompute centroids; average/complete use original pairwise
distances. Trees are checked by leaf-pair cophenetic relations and partitions,
allowing valid tied merge orientation/order and label-number permutations.
"""
from std.math import isfinite
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.dbscan_native_api import DBSCANNativeDriver, DBSCANQueryRequest
from core.agglo_native_api import AggloSolverDriver, AggloClusterDriver, AggloZeroRowsDriver, AggloSolverRequest, AggloClusterRequest, AggloZeroRowsRequest, AggloSolverScratch, AggloClusterScratch, AggloZeroRowsScratch, agglo_reference_request, agglo_query_request, agglo_resident_query
from checks.native_tests.agglo_native_profiles import AggloCase, profile, query_count, query_name, query_rows
from checks.native_tests.pca_native_cases import unchanged

comptime AGGLO_NATIVE_FIXTURES = 9
comptime AGGLO_I32_GUARD = Int32(987654)
comptime AGGLO_F32_GUARD = Float32(27182.5)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def bytes(name: String, count: Int) raises -> List[UInt8]:
    with open("checks/native_tests/fixtures/agglo/" + name, "r") as file:
        var value = file.read_bytes()
        require(len(value) == count, "Agglomerative independent artifact extent " + name)
        return value^


def source_values(name: String, count: Int) raises -> List[Float32]:
    var data = bytes(name + ".f32", 4 * count)
    var values = List[Float32](unsafe_uninit_length=count)
    var bp = data.unsafe_ptr()
    var vp = values.unsafe_ptr()
    def cell(index: Int) {imm bp, imm vp}:
        var at = 4 * index
        var bits = UInt32(bp[at]) | (UInt32(bp[at + 1]) << 8) | (UInt32(bp[at + 2]) << 16) | (UInt32(bp[at + 3]) << 24)
        vp.unsafe_store(index, bitcast[DType.float32](bits))
    host_parallelize(cell, count)
    _ = data^
    return values^


def reference_values(name: String, count: Int) raises -> List[Float64]:
    var data = bytes(name + ".f64", 8 * count)
    var values = List[Float64](unsafe_uninit_length=count)
    var bp = data.unsafe_ptr()
    var vp = values.unsafe_ptr()
    def cell(index: Int) {imm bp, imm vp}:
        var bits = UInt64(0)
        for byte in range(8):  # Fixed-width independent reference decoding.
            bits |= UInt64(bp[index * 8 + byte]) << UInt64(byte * 8)
        vp.unsafe_store(index, bitcast[DType.float64](bits))
    host_parallelize(cell, count)
    _ = data^
    return values^


def nearest_values(name: String, count: Int) raises -> List[Int32]:
    var data = bytes(name + ".i32", 4 * count)
    var values = List[Int32](unsafe_uninit_length=count)
    var bp = data.unsafe_ptr()
    var vp = values.unsafe_ptr()
    def cell(index: Int) {imm bp, imm vp}:
        var at = 4 * index
        var bits = UInt32(bp[at]) | (UInt32(bp[at + 1]) << 8) | (UInt32(bp[at + 2]) << 16) | (UInt32(bp[at + 3]) << 24)
        vp.unsafe_store(index, bitcast[DType.int32](bits))
    host_parallelize(cell, count)
    _ = data^
    return values^


def integer_guards(value: List[Int32]) raises:
    require(value[0] == AGGLO_I32_GUARD and value[len(value) - 1] == AGGLO_I32_GUARD,
        "Agglomerative int32 destination guards")


def tree_oracle(p: AggloCase, children: List[Int32], labels: List[Int32], distances: List[Float32]) raises:
    var n = p.rows
    var nodes = 2 * n - 1
    var reference = reference_values(p.name + "-cophenetic", n * n)
    var membership = List[UInt8](length=nodes * n, fill=0)
    var parents = List[Int](length=nodes, fill=-1)
    var representative = List[Int](length=nodes, fill=-1)
    var failures = List[Int](length=n * n + nodes, fill=0)
    var mp = membership.unsafe_ptr()
    var pp = parents.unsafe_ptr()
    var rp = representative.unsafe_ptr()
    var fp = failures.unsafe_ptr()
    var expected = reference.unsafe_ptr()
    def leaf(index: Int) {imm mp, imm rp, imm n}:
        mp.unsafe_store(index * n + index, UInt8(1))
        rp.unsafe_store(index, index)
    host_parallelize(leaf, n)
    var previous_height = Float64(0)
    for step in range(n - 1):  # driver-loop: over dependent hierarchical merges
        var left = Int(children[1 + 2 * step])
        var right = Int(children[2 + 2 * step])
        var merged = n + step
        require(left >= 0 and right >= 0 and left < merged and right < merged and left != right,
            "Agglomerative children refer to distinct earlier nodes")
        require(parents[left] == -1 and parents[right] == -1, "Agglomerative child is consumed by exactly one parent")
        parents[left] = merged
        parents[right] = merged
        representative[merged] = representative[left]
        var height = Float64(distances[step + 1]) if p.tree else reference[representative[left] * n + representative[right]]
        require(isfinite(height) and height >= 0 and height + Float64(2e-5) + Float64(2e-4) * abs(previous_height) >= previous_height,
            "Agglomerative finite nonnegative monotone merge heights")
        previous_height = height
        def combine(index: Int) {imm mp, imm fp, imm n, imm left, imm right, imm merged}:
            var a = mp[left * n + index]
            var b = mp[right * n + index]
            if a != 0 and b != 0:
                fp.unsafe_store(n * n + index, 1)
            mp.unsafe_store(merged * n + index, a | b)
        host_parallelize(combine, n)
        def relation(index: Int) {imm mp, imm fp, imm expected, imm n, imm left, imm right, imm height}:
            var row = index // n
            var col = index % n
            var cross = (mp[left * n + row] != 0 and mp[right * n + col] != 0)
            if cross:
                var wanted = expected[index]
                if not (abs(height - wanted) <= Float64(2e-5) + Float64(2e-4) * abs(wanted)):
                    fp.unsafe_store(index, 1)
        host_parallelize(relation, n * n)
    var roots = List[Int](length=n, fill=-1)
    var roots_pointer = roots.unsafe_ptr()
    var cut_stop = n + n - p.clusters
    var lp = labels.unsafe_ptr() + 1
    var clusters = p.clusters
    def cut(index: Int) {imm pp, imm roots_pointer, imm fp, imm lp, imm n, imm cut_stop, imm clusters}:
        var current = index
        while pp[current] >= 0 and pp[current] < cut_stop:
            current = pp[current]
        roots_pointer.unsafe_store(index, current)
        if lp[index] < Int32(0) or Int(lp[index]) >= clusters:
            fp.unsafe_store(n * n + index, 1)
    host_parallelize(cut, n)
    def partition(index: Int) {imm lp, imm roots_pointer, imm fp, imm n}:
        var row = index // n
        var col = index % n
        if (lp[row] == lp[col]) != (roots_pointer[row] == roots_pointer[col]):
            fp.unsafe_store(index, 1)
    host_parallelize(partition, n * n)
    def parent_check(index: Int) {imm pp, imm mp, imm fp, imm nodes, imm n}:
        var okay = pp[index] == -1 if index == nodes - 1 else pp[index] >= n
        if index < n:
            okay = okay and mp[(nodes - 1) * n + index] == UInt8(1)
        if not okay:
            fp.unsafe_store(n * n + index, 1)
    host_parallelize(parent_check, nodes)
    for failure in failures:
        require(failure == 0, "Agglomerative independent cophenetic/partition/tree oracle " + p.name)
    integer_guards(children)
    integer_guards(labels)
    require(distances[0] == AGGLO_F32_GUARD and distances[len(distances) - 1] == AGGLO_F32_GUARD,
        "Agglomerative distance destination guards")
    _ = reference^
    _ = membership^
    _ = parents^
    _ = representative^
    _ = roots^
    _ = children
    _ = labels
    _ = distances


def expected_index(labels: Pointer[Int32, ImmUntrackedOrigin], first: Int, second: Int) -> Int:
    if second >= 0 and (labels[second] < labels[first] or (labels[second] == labels[first] and second < first)):
        return second
    return first


def query_oracle(labels: List[Int32], nearest: List[Int32], dst: List[Int32], chosen: List[Int32], borrowed: Bool) raises:
    var rows = len(nearest) // 2
    var count = len(labels) - 2
    var failures = List[Int](length=rows, fill=0)
    var fp = failures.unsafe_ptr()
    var lp = labels.unsafe_ptr() + 1
    var np = nearest.unsafe_ptr()
    var dp = dst.unsafe_ptr() + 1
    var cp = chosen.unsafe_ptr() + 1
    def cell(row: Int) {imm fp, imm lp, imm np, imm dp, imm cp, imm borrowed, imm count}:
        var first = Int(np[2 * row])
        var second = Int(np[2 * row + 1])
        if first < 0 or first >= count or second < -1 or second >= count:
            fp.unsafe_store(row, 1)
            return
        var wanted = first
        if second >= 0 and (lp[second] < lp[first] or (lp[second] == lp[first] and second < first)):
            wanted = second
        var okay = dp[row] == lp[wanted]
        if borrowed:
            okay = okay and cp[row] == Int32(wanted)
        fp.unsafe_store(row, Int(not okay))
    host_parallelize(cell, rows)
    for failure in failures:
        require(failure == 0, "Agglomerative independent nearest-row/lowest-cluster-label tie oracle")
    integer_guards(dst)
    integer_guards(chosen)
    _ = labels
    _ = nearest
    _ = dst
    _ = chosen


def single_query(driver: DBSCANNativeDriver, request: DBSCANQueryRequest, labels: Int, first: Int, second: Int) raises:
    var dst = List[Int32](length=3, fill=AGGLO_I32_GUARD)
    var chosen = List[Int32](length=3, fill=AGGLO_I32_GUARD)
    try:
        var call = request.copy()
        call.dst = Int(dst.unsafe_ptr()) + 4
        call.chosen = Int(chosen.unsafe_ptr()) + 4
        require(driver.query(call) == 0, "Agglomerative single-query return")
        var lp = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=labels)
        var wanted = expected_index(lp, first, second)
        require(dst[1] == lp[wanted] and chosen[1] == Int32(wanted), "Agglomerative original individual-row query")
        integer_guards(dst)
        integer_guards(chosen)
    finally:
        _ = dst^
        _ = chosen^


def singles(driver: DBSCANNativeDriver, p: AggloCase, source: List[Float32], labels: List[Int32], host: Bool) raises:
    var nearest = nearest_values(p.name + "-query-train-nearest", 2 * p.rows)
    var failures = List[Int](length=p.rows, fill=0)
    var fp = failures.unsafe_ptr()
    var np = nearest.unsafe_ptr()
    var xa = Int(source.unsafe_ptr())
    var la = Int(labels.unsafe_ptr()) + 4
    var rows = p.rows
    var features = p.features
    def task(row: Int) {imm fp, imm np, imm xa, imm la, imm rows, imm features, imm driver}:
        try:
            single_query(driver, agglo_query_request(xa, la, xa + 4 * row * features, 0, 0,
                rows, 1, features), la, Int(np[2 * row]), Int(np[2 * row + 1]))
        except error:
            fp.unsafe_store(row, 1)
            print("FAIL Agglomerative individual query", row, String(error))
    if host:
        host_parallelize(task, rows)
    else:
        # One query graph at a time on one physical GPU/shared context.
        for row in range(rows):
            task(row)
    for failure in failures:
        require(failure == 0, "Agglomerative original eighteen individual queries")
    _ = nearest^
    _ = source
    _ = labels


def zero_rows(driver: AggloZeroRowsDriver) raises:
    var negative_zero = bitcast[DType.float32](UInt32(0x80000000))
    var source: List[Float32] = [0, negative_zero, 0, 1, 0, 0, 0, 0, 3]
    var before = source.copy()
    var flags = AggloZeroRowsScratch(3)
    try:
        driver.run(AggloZeroRowsRequest(Int(source.unsafe_ptr()), flags.address(), 3, 3))
        require(flags.flags[0] == Int32(1) and flags.flags[1] == Int32(0) and flags.flags[2] == Int32(0)
            and flags.any_zero(), "Agglomerative actual zero-row admission flags include signed zero")
        unchanged(source, before, "Agglomerative zero-row source")
    finally:
        _ = source^
        _ = before^
        _ = flags^


def fixture(solver: AggloSolverDriver, cluster: AggloClusterDriver, zero: AggloZeroRowsDriver,
            near: DBSCANNativeDriver, unit: Int, host: Bool) raises:
    if unit == 8:
        zero_rows(zero)
        return
    var p = profile(unit)
    var source = source_values(p.name + "-source", p.rows * p.features)
    var before = source.copy()
    var children = List[Int32](length=2 * (p.rows - 1) + 2, fill=AGGLO_I32_GUARD)
    var labels = List[Int32](length=p.rows + 2, fill=AGGLO_I32_GUARD)
    var distances = List[Float32](length=p.rows + 1, fill=AGGLO_F32_GUARD)
    var info = AggloSolverScratch()
    var scalars = AggloClusterScratch()
    var handle = -1
    try:
        var xa = Int(source.unsafe_ptr())
        var ca = Int(children.unsafe_ptr()) + 4
        var la = Int(labels.unsafe_ptr()) + 4
        if p.tree:
            cluster.fit(AggloClusterRequest(xa, 0, ca, la, Int(distances.unsafe_ptr()) + 4,
                scalars.address(), p.rows * p.features, 0, p.rows, p.features, p.linkage, -1,
                -1, p.rows - 1, 0, p.clusters, False, 2.0, 0.0))
            require(scalars.components() == 1 and scalars.clusters() == p.clusters,
                "Agglomerative actual dense tree connected-component/cluster counts")
        else:
            var rounds = solver.fit(AggloSolverRequest(xa, ca, la, info.address(), p.rows, p.features,
                p.clusters, 1, False))
            require(rounds == info.rounds() and info.components() == 1,
                "Agglomerative actual solver result and dense connectivity metadata")
            require(rounds == -1 if host else rounds >= 1, "Agglomerative existing CPU sentinel/GPU Boruvka rounds")
        tree_oracle(p, children, labels, distances)
        if not host and query_count(unit) > 0:
            handle = near.prepare(agglo_reference_request(xa, la, p.rows, p.features))
            require(handle >= 0, "Agglomerative actual native reference handle")
        # Original query batches depend on this fitted state; GPU graphs ordered.
        for batch in range(query_count(unit)):
            var rows = query_rows(unit, batch)
            var stem = p.name + "-query-" + query_name(batch)
            var queries = source_values(stem, rows * p.features)
            var query_before = queries.copy()
            var nearest = nearest_values(stem + "-nearest", 2 * rows)
            var dst = List[Int32](length=rows + 2, fill=AGGLO_I32_GUARD)
            var chosen = List[Int32](length=rows + 2, fill=AGGLO_I32_GUARD)
            try:
                require(near.query(agglo_query_request(xa, la, Int(queries.unsafe_ptr()), Int(dst.unsafe_ptr()) + 4,
                    Int(chosen.unsafe_ptr()) + 4, p.rows, rows, p.features)) == 0, "Agglomerative borrowed query return")
                query_oracle(labels, nearest, dst, chosen, True)
                if handle >= 0:
                    var resident = List[Int32](length=rows + 2, fill=AGGLO_I32_GUARD)
                    try:
                        require(near.resident(agglo_resident_query(Int(queries.unsafe_ptr()), Int(resident.unsafe_ptr()) + 4,
                            handle, rows, p.features)) == 0, "Agglomerative retained query return")
                        query_oracle(labels, nearest, resident, chosen, False)
                    finally:
                        _ = resident^
                unchanged(queries, query_before, "Agglomerative query input")
            finally:
                _ = queries^
                _ = query_before^
                _ = nearest^
                _ = dst^
                _ = chosen^
        if unit == 0:
            singles(near, p, source, labels, host)
        unchanged(source, before, "Agglomerative original fit input")
    finally:
        try:
            if handle >= 0:
                var owned = handle
                handle = -1
                require(near.release(owned) == 0, "Agglomerative retained reference released once")
        finally:
            _ = source^
            _ = before^
            _ = children^
            _ = labels^
            _ = distances^
            _ = info^
            _ = scalars^
