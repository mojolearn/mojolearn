# SPDX-License-Identifier: Apache-2.0
"""Native table/handle contracts; independent fixtures use the host pool."""
from core.entry_table import entry_table_publish, entry_table_read
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_call
from core.state_registry import StateRegistry, StateHandle
from core.host_parallel import host_parallelize
from core.linear_model_state import LinearModelState, linear_state_predict_request
from core.linear_boundary_run import linear_predict_run
from glm.host.linear_predict_native import linear_predict_host_v1

comptime ENTRY_TABLE_CONTRACT_CASES = 20


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def table_probe(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    if operation != 0 or np != 2 or not pp or not result:
        return 1
    var words = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
    MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = words[0] * 100 + words[1]
    return 0


def other_probe(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    if not result:
        return 1
    MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = 999
    return 0


def table_case[unit: Int]() raises:
    comptime NAME = (
        "MojolearnTestEntriesMissingIdenticalCPUV1" if unit == 0 else
        "MojolearnTestEntriesInvokeIdenticalCPUV1" if unit == 1 else
        "MojolearnTestEntriesModeIdenticalCPUV1" if unit == 2 else
        "MojolearnTestEntriesVendorIdenticalCPUV1" if unit == 3 else
        "MojolearnTestEntriesSchemaIdenticalCPUV1" if unit == 4 else
        "MojolearnTestEntriesCountIdenticalCPUV1" if unit == 5 else
        "MojolearnTestEntriesIndexIdenticalCPUV1" if unit == 6 else
        "MojolearnTestEntriesPublisherIdenticalCPUV1" if unit == 7 else
        "MojolearnTestEntriesEndpointIdenticalCPUV1" if unit == 8 else
        "MojolearnTestEntriesRepeatIdenticalCPUV1" if unit == 9 else
        "MojolearnTestEntriesParallelIdenticalCPUV1"
    )
    if unit != 0 and unit != 10:
        entry_table_publish[NAME](1, 0, 17, [String("test.entry.v1")], [1], [table_probe])
    if unit == 10:
        var failures = List[Int](length=16, fill=0)
        var flags = failures.unsafe_ptr()
        def task(index: Int) {imm flags}:
            try:
                entry_table_publish[NAME](1, 0, 17, [String("test.entry.v1")], [1], [table_probe])
                var entry = entry_table_read[NAME](1, 0, 0, String("test.entry.v1"), 1)
                require(native_descriptor_call(entry, 0, [], [index, 11], []) == index * 100 + 11,
                    "concurrent table publication or call crossed")
            except:
                flags[index] = 1
        host_parallelize(task, 16)
        for flag in failures:  # metadata: collect independent fixture status
            require(flag == 0, "concurrent first publication failed")
        return
    if unit == 1 or unit == 9:
        if unit == 9:
            entry_table_publish[NAME](1, 0, 17, [String("test.entry.v1")], [1], [table_probe])
        var entry = entry_table_read[NAME](1, 0, 0, String("test.entry.v1"), 1)
        require(native_descriptor_call(entry, 0, [], [7, 3], []) == 703, "published C pointer was not invoked")
        return
    var refused = False
    try:
        if unit == 7:
            entry_table_publish[NAME](1, 0, 18, [String("test.entry.v1")], [1], [table_probe])
        elif unit == 8:
            entry_table_publish[NAME](1, 0, 17, [String("test.entry.v1")], [1], [other_probe])
        else:
            _ = entry_table_read[NAME](0 if unit == 2 else 1, 1 if unit == 3 else 0,
                1 if unit == 6 else 0, String("test.entry.v2" if unit == 4 else "test.entry.v1"),
                2 if unit == 5 else 1)
    except:
        refused = True
    require(refused, "missing/mismatched native entry was accepted")


struct RegistryValue(Movable):
    var values: List[Int]

    def __init__(out self, value: Int):
        self.values = List[Int](length=3, fill=value)


def registry_case[unit: Int]() raises:
    comptime NAME = (
        "MojolearnTestStateMoveV1" if unit == 0 else
        "MojolearnTestStateReleaseV1" if unit == 1 else
        "MojolearnTestStateGenerationV1" if unit == 2 else
        "MojolearnTestStateReplaceV1" if unit == 3 else
        "MojolearnTestStateMissingV1" if unit == 4 else
        "MojolearnTestStateWrongGenerationV1" if unit == 5 else
        "MojolearnTestStateDoubleReleaseV1" if unit == 6 else
        "MojolearnTestStateParallelV1"
    )
    comptime STORE = StateRegistry[RegistryValue, NAME]
    if unit == 7:
        var failures = List[Int](length=16, fill=0)
        var flags = failures.unsafe_ptr()
        def task(index: Int) {imm flags}:
            try:
                var handle = STORE.insert(RegistryValue(index + 31))
                var lease = STORE.acquire(handle)
                STORE.release(handle)
                require(lease[].values[0] == index + 31, "parallel native owner value crossed")
                _ = lease
            except:
                flags[index] = 1
        host_parallelize(task, 16)
        for flag in failures:  # metadata: collect independent fixture status
            require(flag == 0, "parallel native registry operation failed")
        return
    var source = RegistryValue(37)
    var address = Int(source.values.unsafe_ptr())
    var handle = STORE.insert(source^)
    var lease = STORE.acquire(handle)
    require(Int(lease[].values.unsafe_ptr()) == address, "registry insertion copied native buffer")
    var released = False
    try:
        if unit == 0:
            require(lease[].values[2] == 37, "moved native owner lost values")
        elif unit == 1 or unit == 2 or unit == 6:
            STORE.release(handle)
            released = True
            require(lease[].values[1] == 37, "release invalidated active native lease")
            var refused = False
            try:
                if unit == 6:
                    STORE.release(handle)
                else:
                    _ = STORE.acquire(handle)
            except:
                refused = True
            require(refused, "stale native handle was accepted")
            if unit == 2:
                var replacement = STORE.insert(RegistryValue(91))
                try:
                    require(replacement.index == handle.index, "isolated registry did not reuse its free slot")
                    require(replacement.generation != handle.generation, "reused state generation did not change")
                    var newer = STORE.acquire(replacement)
                    require(newer[].values[0] == 91 and lease[].values[0] == 37, "old/new leases aliased after reuse")
                    _ = newer
                finally:
                    STORE.release(replacement)
        elif unit == 3:
            STORE.replace(handle, RegistryValue(52))
            var newer = STORE.acquire(handle)
            require(newer[].values[0] == 52 and lease[].values[0] == 37, "replacement rewrote active snapshot")
            _ = newer
        else:
            var wrong = StateHandle(-1 if unit == 4 else handle.index, handle.generation + 1)
            var refused = False
            try:
                _ = STORE.acquire(wrong)
            except:
                refused = True
            require(refused, "invalid native handle was accepted")
    finally:
        if not released:
            STORE.release(handle)
        _ = lease


def borrowed_linear_case() raises:
    # Explicit native buffer pins model the actual UNPACK ownership contract.
    # The registry does not own these lists through their integer descriptors.
    comptime STORE = StateRegistry[LinearModelState, "MojolearnTestBorrowedLinearStateV1"]
    var coefficients: List[Float32] = [2, -1, 3]
    var newer_coefficients: List[Float32] = [4, 5, 6]
    var source: List[Float32] = [1, 2, 1, 2, 0, -1]
    var destination = List[Float32](length=2, fill=0)
    var state = LinearModelState()
    state.coef = Int(coefficients.unsafe_ptr())
    state.coef_count = 3
    state.features = 3
    state.intercept = 0.25
    state.has_coef = True
    state.fitted = True
    var handle = STORE.insert(state.copy())
    var lease = STORE.acquire(handle)
    state.coef = Int(newer_coefficients.unsafe_ptr())
    STORE.replace(handle, state.copy())
    STORE.release(handle)
    try:
        var request = linear_state_predict_request(lease[], Int(source.unsafe_ptr()),
            Int(destination.unsafe_ptr()), 2, 3, False, String("Ridge"))
        linear_predict_run(linear_predict_host_v1, request)
        require(destination[0] == Float32(3.25) and destination[1] == Float32(1.25),
            "active linear lease mixed a refit's coefficients")
        var refused = False
        try:
            _ = STORE.acquire(handle)
        except:
            refused = True
        require(refused, "deleted linear owner left a usable stale handle")
    finally:
        _ = coefficients
        _ = newer_coefficients
        _ = source
        _ = destination
        _ = lease


def fixture(unit: Int) raises:
    if unit == 0:
        table_case[0]()
    elif unit == 1:
        table_case[1]()
    elif unit == 2:
        table_case[2]()
    elif unit == 3:
        table_case[3]()
    elif unit == 4:
        table_case[4]()
    elif unit == 5:
        table_case[5]()
    elif unit == 6:
        table_case[6]()
    elif unit == 7:
        table_case[7]()
    elif unit == 8:
        table_case[8]()
    elif unit == 9:
        table_case[9]()
    elif unit == 10:
        table_case[10]()
    elif unit == 11:
        registry_case[0]()
    elif unit == 12:
        registry_case[1]()
    elif unit == 13:
        registry_case[2]()
    elif unit == 14:
        registry_case[3]()
    elif unit == 15:
        registry_case[4]()
    elif unit == 16:
        registry_case[5]()
    elif unit == 17:
        registry_case[6]()
    elif unit == 18:
        registry_case[7]()
    elif unit == 19:
        borrowed_linear_case()


def run_checks() raises:
    var failures = List[Int](length=ENTRY_TABLE_CONTRACT_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            fixture(unit)
        except problem:
            flags[unit] = 1
            print("FAIL entry_table_contract", unit, String(problem))
    host_parallelize(task, ENTRY_TABLE_CONTRACT_CASES)
    var failed = 0
    for flag in failures:  # metadata: collect independent fixture status
        failed += flag
    require(failed == 0, "native entry/state contracts failed: " + String(failed))
    print("PASS entry_table_contract:", ENTRY_TABLE_CONTRACT_CASES)
