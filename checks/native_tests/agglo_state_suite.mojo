# SPDX-License-Identifier: Apache-2.0
"""Prescribed Agglomerative options/tree plans and native ownership metadata.

Independent fixtures run across host cores. Fake handles only exercise pure
matching/take-once state; they never reach a device registry or destructor.
"""
from core.host_parallel import host_parallelize
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.dbscan_native_api import DBSCANNativeDriver, DBSCAN_MODE
from core.agglo_native_api import AggloSolverScratch, AggloClusterScratch, AggloZeroRowsScratch
from core.agglo_native_state import AggloOptions, AggloReferenceKey, AggloReferenceState, agglo_linkage_code, agglo_metric_code, agglo_connectivity_code, agglo_predicts, agglo_tree_plan, AGGLO_PAIRWISE_MAX_ROWS, AGGLO_FULL_AUTO, AGGLO_FULL_FALSE, AGGLO_FULL_TRUE

comptime AGGLO_STATE_FIXTURES = 103


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def state_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    if operation != DBSCAN_MODE or result == 0 or na != 0 or np != 0 or nf != 0:
        native_descriptor_report(Error("Agglomerative state fixture permits only mode metadata"), context, sink)
        return 1
    MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = 1
    return 0


def key() -> AggloReferenceKey:
    return AggloReferenceKey(11, 13, 17, 3)


def state(handle: Int) -> AggloReferenceState:
    return AggloReferenceState(19, 23, key(), handle, DBSCANNativeDriver(state_entry))


def transfer(var owner: AggloReferenceState) -> AggloReferenceState:
    return owner^


def move_flags(var room: AggloZeroRowsScratch) -> AggloZeroRowsScratch:
    return room^


def plan_fixture(unit: Int) raises:
    # Prescribed original full-tree rule, including strict threshold neighbors.
    var rows: List[Int] = [200, 200, 200, 200, 5000, 5000, 5001, 5500, 5500, 10000, 10000, 5000, 5000, 5000, 5000, 5000]
    var clusters: List[Int] = [3, 3, 3, 3, 99, 100, 100, 109, 110, 199, 200, 200, 200, 200, 200, 200]
    var modes: List[Int] = [AGGLO_FULL_FALSE, AGGLO_FULL_TRUE, AGGLO_FULL_FALSE, AGGLO_FULL_AUTO,
        AGGLO_FULL_AUTO, AGGLO_FULL_AUTO, AGGLO_FULL_AUTO, AGGLO_FULL_AUTO, AGGLO_FULL_AUTO, AGGLO_FULL_AUTO,
        AGGLO_FULL_AUTO, AGGLO_FULL_AUTO, AGGLO_FULL_TRUE, AGGLO_FULL_FALSE, AGGLO_FULL_FALSE, AGGLO_FULL_AUTO]
    var full: List[Bool] = [True, True, False, True, True, False, True, True, False, True, False, True, True, False, True, True]
    var merges: List[Int] = [199, 199, 197, 199, 4999, 4900, 5000, 5499, 5390, 9999, 9800, 4999, 4999, 4800, 4999, 4999]
    var cuts: List[Int] = [3, 3, 0, 3, 99, 0, 100, 109, 0, 199, 0, 1, 1, 0, 1, 200]
    var threshold = unit == 11 or unit == 12 or unit == 14
    var constrained = unit != 0 and unit != 14 and unit != 15
    var options = AggloOptions(0, -1, constrained, threshold, True, False, modes[unit])
    var result = agglo_tree_plan(options, rows[unit], clusters[unit])
    require(result.full == full[unit] and result.merges == merges[unit] and result.cut == cuts[unit]
        and result.use_threshold == threshold, "Agglomerative prescribed full/partial/threshold tree plan")


def fixture(unit: Int) raises:
    comptime assert AGGLO_PAIRWISE_MAX_ROWS == 46340
    if unit < 5:
        var names: List[String] = ["ward", "complete", "average", "single", "Single"]
        var codes: List[Int] = [0, 1, 2, 3, -1]
        require(agglo_linkage_code(names[unit]) == codes[unit], "Agglomerative exact linkage codes")
    elif unit < 14:
        var names: List[String] = ["euclidean", "l2", "l1", "manhattan", "cityblock", "cosine", "precomputed", "", "Euclidean"]
        var codes: List[Int] = [-1, -1, 1, 1, 1, 4, 5, -2, -2]
        require(agglo_metric_code(names[unit - 5]) == codes[unit - 5], "Agglomerative tree metric aliases and unknown names")
    elif unit < 17:
        var names: List[String] = ["pairwise", "knn", "KNN"]
        var codes: List[Int] = [0, 1, -1]
        require(agglo_connectivity_code(names[unit - 14]) == codes[unit - 14], "Agglomerative connectivity names")
    elif unit < 29:
        var variant = unit - 17
        var metrics: List[Int] = [-1, 1, 4]
        require(agglo_predicts(variant // 3, metrics[variant % 3]) == (variant == 9),
            "Agglomerative prediction admission matrix: single Euclidean only")
        var options = AggloOptions(variant // 3, metrics[variant % 3], False, False, False, False, AGGLO_FULL_AUTO)
        require(options.solver_route() == (variant == 9), "Agglomerative solver linkage/metric admission matrix")
    elif unit < 61:
        var variant = unit - 29
        var options = AggloOptions(3, -1, (variant & 1) != 0, (variant & 2) != 0, (variant & 4) != 0,
            (variant & 8) != 0, AGGLO_FULL_FALSE if (variant & 16) != 0 else AGGLO_FULL_TRUE)
        require(options.solver_route() == ((variant & 7) == 0),
            "Agglomerative solver selection; prediction data/full-tree metadata do not change the route")
    elif unit < 77:
        plan_fixture(unit - 61)
    elif unit < 83:
        var first = key()
        var other = first.copy()
        if unit == 82:
            first.refs_owner = 9007199254740992
            other = first.copy()
            other.refs_owner += 1
        elif unit == 78:
            other.refs_owner += 1
        elif unit == 79:
            other.labels_owner += 1
        elif unit == 80:
            other.rows += 1
        elif unit == 81:
            other.features += 1
        require(first.matches(other) == (unit == 77) and other.matches(first) == (unit == 77),
            "Agglomerative stored reference owner/dimension identity")
        require(first.refs_owner == (9007199254740992 if unit == 82 else 11)
            and first.labels_owner == 13 and first.rows == 17 and first.features == 3,
            "Agglomerative copied key independence")
    elif unit < 89:
        var handle = 0 if unit == 83 else (9007199254740993 if unit == 84 else (-1 if unit == 85 else 37))
        var value = state(handle)
        var candidate = key()
        if unit == 88:
            candidate.labels_owner += 1
        require(value.matches(29 if unit == 86 else 19, 31 if unit == 87 else 23, candidate)
            == (unit == 83 or unit == 84), "Agglomerative native reference matching and exact integer handles")
        require(value.handle == handle, "Agglomerative matching is nonconsuming")
    elif unit < 92:
        var handle = 0 if unit == 89 else (9007199254740993 if unit == 90 else 37)
        var value = state(handle)
        var moved = transfer(value^)
        require(moved.driver.metadata(DBSCAN_MODE) == 1 and moved.matches(19, 23, key()),
            "Agglomerative moved native identity/driver")
        require(moved.take_handle() == handle and moved.take_handle() == -1 and not moved.matches(19, 23, key()),
            "Agglomerative handle taken once, including zero and above2^53")
    elif unit < 94:
        var room = AggloSolverScratch()
        require(len(room.info) == 2 and room.address() == Int(room.info.unsafe_ptr()), "Agglomerative solver two native Int32 words")
        if unit == 93:
            room.info[0] = -1
            room.info[1] = 37
        require(room.rounds() == (-1 if unit == 93 else 0) and room.components() == (37 if unit == 93 else 0),
            "Agglomerative solver scratch preserves CPU rounds sentinel and component count")
    elif unit < 96:
        var room = AggloClusterScratch()
        require(len(room.scalars) == 2 and room.address() == Int(room.scalars.unsafe_ptr()), "Agglomerative tree two native Float64 words")
        if unit == 95:
            room.scalars[0] = 7
            room.scalars[1] = 11
        require(room.components() == (7 if unit == 95 else 0) and room.clusters() == (11 if unit == 95 else 0),
            "Agglomerative tree native count publication")
    else:
        var rows = 0 if unit == 96 else 33
        var room = AggloZeroRowsScratch(rows)
        var before = room.address()
        require(len(room.flags) == rows and before == Int(room.flags.unsafe_ptr()), "Agglomerative native zero-row flag extent")
        if unit == 98:
            room.flags[0] = 1
        elif unit == 99:
            room.flags[16] = 7
        elif unit == 100:
            room.flags[32] = 1
        elif unit == 101:
            room.flags[16] = -1
        if unit == 102:
            room.flags[32] = 1
            var moved = move_flags(room^)
            require(moved.address() == before and len(moved.flags) == 33 and moved.any_zero(),
                "Agglomerative moved flag storage retains identity/value")
        else:
            require(room.any_zero() == (unit >= 98 and unit <= 100),
                "Agglomerative native flag reduction: empty/zero/positive edges/negative flag")


def run_checks() raises:
    var failures = List[Int](length=AGGLO_STATE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL agglo_state fixture", unit, String(error))
    host_parallelize(task, AGGLO_STATE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "agglo_state failed " + String(failed) + " fixtures")
    print("PASS agglo_state:", AGGLO_STATE_FIXTURES, "native options/tree/ownership/scratch fixtures")
