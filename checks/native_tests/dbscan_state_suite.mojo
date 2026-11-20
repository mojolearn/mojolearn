# SPDX-License-Identifier: Apache-2.0
"""DBSCAN typed scratch/reference identity and take-once state contracts.

Independent fixtures and scratch words run in parallel. Fake reference
handles never enter real device prepare/query/release; pure state has no
destructor callback. Storage is flat native List ownership.
"""
from core.host_parallel import host_parallelize
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.dbscan_native_api import DBSCANNativeDriver, DBSCANFitWorkspace, DBSCAN_MODE
from core.dbscan_native_state import DBSCANReferenceKey, DBSCANReferenceState

comptime DBSCAN_STATE_FIXTURES = 28


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def state_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    if operation != DBSCAN_MODE or result == 0 or na != 0 or np != 0 or nf != 0:
        native_descriptor_report(Error("DBSCAN state fixture permits only mode metadata"), context, sink)
        return 1
    MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = 1
    return 0


def key() -> DBSCANReferenceKey:
    return DBSCANReferenceKey(11, 13, 17, 19, 23, 29, 31, 3, 0)


def state(handle: Int = 37) -> DBSCANReferenceState:
    return DBSCANReferenceState(41, 43, key(), handle, DBSCANNativeDriver(state_entry))


def move_state(var value: DBSCANReferenceState) -> DBSCANReferenceState:
    return value^


def move_workspace(var value: DBSCANFitWorkspace) -> DBSCANFitWorkspace:
    return value^


def check_bytes(value: List[UInt8], written: Bool) raises:
    var failures = List[Int](length=len(value), fill=0)
    var fp = failures.unsafe_ptr()
    var vp = value.unsafe_ptr()
    def cell(index: Int) {imm fp, imm vp, imm written}:
        var expected = UInt8(index % 251 + 1) if written else UInt8(0)
        fp.unsafe_store(index, Int(vp[index] != expected))
    host_parallelize(cell, len(value))
    for failure in failures:
        require(failure == 0, "DBSCAN native scratch bytes")
    _ = value


def fill_workspace(mut workspace: DBSCANFitWorkspace):
    # This mutable borrow owns the captured pointer's entire parallel lifetime.
    # Returning ends the borrow before the caller transfers the workspace.
    var pointer = workspace.core.unsafe_ptr()
    def cell(index: Int) {imm pointer}:
        pointer.unsafe_store(index, UInt8(index % 251 + 1))
    host_parallelize(cell, 33)
    _ = workspace


def fixture(unit: Int) raises:
    if unit <= 3:
        var rows = 0 if unit == 0 else (1 if unit == 1 else (17 if unit == 2 else 1025))
        var workspace = DBSCANFitWorkspace(rows)
        require(len(workspace.core) == rows and workspace.address() == Int(workspace.core.unsafe_ptr()),
            "DBSCAN native core-mask capacity/address")
        check_bytes(workspace.core, False)
        _ = workspace
    elif unit == 4:
        var workspace = DBSCANFitWorkspace(33)
        var before = workspace.address()
        fill_workspace(workspace)
        var moved = move_workspace(workspace^)
        require(moved.address() == before and len(moved.core) == 33, "DBSCAN move retains scratch allocation")
        check_bytes(moved.core, True)
        _ = moved
    elif unit == 5:
        var first = DBSCANFitWorkspace(17)
        var second = DBSCANFitWorkspace(17)
        require(first.address() != second.address(), "DBSCAN independent native scratch owners never alias")
        first.core[0] = 7
        check_bytes(second.core, False)
        _ = first
        _ = second
    elif unit <= 17:
        var first = key()
        var second = first.copy()
        if unit == 7:
            second.refs_owner += 1
        elif unit == 8:
            second.keys_owner += 1
        elif unit == 9:
            second.labels_owner += 1
        elif unit == 10:
            second.refs += 1
        elif unit == 11:
            second.keys += 1
        elif unit == 12:
            second.labels += 1
        elif unit == 13:
            second.rows += 1
        elif unit == 14:
            second.features += 1
        elif unit == 15:
            second.metric = 2
        elif unit == 16:
            second.refs_owner = 101
            second.metric = 1
            require(first.refs_owner == 11 and first.metric == 0, "DBSCAN copied native key is independent")
        elif unit == 17:
            first.rows = 0
            first.refs = 0
            first.keys = 0
            first.labels = 0
            second = first.copy()
        var expected = unit == 6 or unit == 17
        require(first.matches(second) == expected and second.matches(first) == expected,
            "DBSCAN complete owner/address/dimension/metric key")
    elif unit <= 23:
        var handle = 0 if unit == 18 else (9007199254740993 if unit == 19 else (-1 if unit == 20 else 37))
        var value = state(handle)
        var candidate = key()
        if unit == 23:
            candidate.refs_owner += 1
        require(value.matches(42 if unit == 21 else 41, 44 if unit == 22 else 43, candidate)
            == (unit == 18 or unit == 19), "DBSCAN valid handle/owner/provider/key matching")
        require(value.handle == handle, "DBSCAN matching does not consume the reference owner")
    else:
        var handle = 0 if unit == 24 else (9007199254740993 if unit == 25 else (-1 if unit == 26 else 37))
        var value = state(handle)
        if unit == 27:
            var moved = move_state(value^)
            require(moved.matches(41, 43, key()) and moved.driver.metadata(DBSCAN_MODE) == 1,
                "DBSCAN moved reference state retains native identity/driver")
            require(moved.take_handle() == 37 and moved.take_handle() == -1, "DBSCAN moved handle taken once")
        else:
            require(value.take_handle() == handle and value.handle == -1 and value.take_handle() == -1,
                "DBSCAN native handle transfer preserves exact integer and invalidates once")
            require(not value.matches(41, 43, key()) and value.owner == 41 and value.provider == 43
                and value.key.matches(key()) and value.driver.metadata(DBSCAN_MODE) == 1,
                "DBSCAN take changes only handle state")


def run_checks() raises:
    var failures = List[Int](length=DBSCAN_STATE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL dbscan_state fixture", unit, String(error))
    host_parallelize(task, DBSCAN_STATE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "dbscan_state failed " + String(failed) + " fixtures")
    print("PASS dbscan_state:", DBSCAN_STATE_FIXTURES, "native scratch/reference ownership fixtures")
