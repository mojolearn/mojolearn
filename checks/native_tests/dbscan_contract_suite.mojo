# SPDX-License-Identifier: Apache-2.0
"""DBSCAN prescribed request fields, exact integer results and ABI refusals.

Independent contracts run in parallel. Mock tensor addresses are never
dereferenced or passed to real kernels; heap-owned error storage stays live.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.native_call_output import NativeCallOutput
from core.native_descriptor_call import native_descriptor_call, native_descriptor_error
from core.dbscan_native_api import DBSCANNativeKernels, DBSCANNativeDriver, DBSCANFitRequest, DBSCANCoreRequest, DBSCANGatherRequest, DBSCANReferenceRequest, DBSCANQueryRequest, DBSCANResidentQuery, dbscan_native_v1, DBSCAN_NATIVE_COUNT, DBSCAN_MODE, DBSCAN_VENDOR, DBSCAN_CAPABILITIES, DBSCAN_FIT, DBSCAN_CORE_COUNT, DBSCAN_CORE_GATHER, DBSCAN_PREPARE, DBSCAN_RELEASE, DBSCAN_QUERY, DBSCAN_RESIDENT_QUERY

comptime DBSCAN_CONTRACT_FIXTURES = 45
comptime EXACT_RESULT = 9007199254740993


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


struct DBSCANWireOracle(DBSCANNativeKernels):
    @staticmethod
    def mode() -> Int:
        return 1

    @staticmethod
    def vendor() -> Int:
        return 4

    @staticmethod
    def capabilities() -> Int:
        return 1

    @staticmethod
    def fit(r: DBSCANFitRequest) raises -> Int:
        require(r.x == 11 and r.labels == 13 and (r.weights == 0 or r.weights == 17) and r.core == 19,
            "DBSCAN fit tensor slots")
        require(r.rows == 23 and r.features == 29 and r.min_samples == 31 and r.budget_mb == 37
            and r.max_iter == 41 and r.method == 1 and r.metric == 3, "DBSCAN fit integer options")
        if r.eps == 0:
            require(bitcast[DType.uint64](r.eps) == UInt64(0x8000000000000000), "DBSCAN signed-zero radius")
        else:
            require(r.eps == Float64(0.10000000000000002), "DBSCAN binary64 radius is not narrowed")
        return EXACT_RESULT + Int(r.weights != 0)

    @staticmethod
    def core_count(r: DBSCANCoreRequest) raises -> Int:
        require(r.x == 43 and r.labels == 47 and r.core == 53 and r.rows == 59 and r.features == 61,
            "DBSCAN core count addresses and dimensions")
        return EXACT_RESULT + 2

    @staticmethod
    def core_gather(r: DBSCANGatherRequest) raises -> Int:
        require(r.x == 67 and r.labels == 71 and r.core == 73 and r.indices == 79
            and r.components == 83 and r.core_labels == 89, "DBSCAN core gather addresses")
        require(r.rows == 97 and r.features == 101 and r.count == 103, "DBSCAN core gather dimensions")
        return EXACT_RESULT + 3

    @staticmethod
    def prepare(r: DBSCANReferenceRequest) raises -> Int:
        require(r.refs == 107 and r.keys == 109 and r.labels == 113 and r.features == 127 and r.metric == 2,
            "DBSCAN reference addresses/dimensions/metric")
        require(r.rows == 131 or r.rows == 0, "DBSCAN reference row count")
        return EXACT_RESULT + 4 if r.rows != 0 else 0

    @staticmethod
    def release(handle: Int) raises -> Int:
        require(handle == EXACT_RESULT or handle == 0, "DBSCAN release handle is exact Int")
        return EXACT_RESULT + 5 if handle != 0 else 0

    @staticmethod
    def query(r: DBSCANQueryRequest) raises -> Int:
        require(r.refs == 137 and r.keys == 139 and r.labels == 149 and r.query == 151
            and r.dst == 157 and r.chosen == 163, "DBSCAN borrowed query addresses")
        require(r.rows == 167 and r.queries == 173 and r.features == 179 and r.metric == 1
            and r.eps == Float64(0.10000000000000002), "DBSCAN borrowed query options")
        return EXACT_RESULT + 6 + Int(r.has_threshold)

    @staticmethod
    def resident(r: DBSCANResidentQuery) raises -> Int:
        require(r.query == 181 and r.dst == 191 and r.handle == EXACT_RESULT and r.queries == 193
            and r.features == 197 and r.metric == 0 and r.eps == Float64(0.10000000000000002),
            "DBSCAN resident exact handle/options")
        return EXACT_RESULT + 8 + Int(r.has_threshold)


def fit_request(weighted: Bool) -> DBSCANFitRequest:
    return DBSCANFitRequest(11, 13, 17 if weighted else 0, 19, 23, 29, 31, 37, 41, 1, 3, 0.10000000000000002)


def missing_span(mut output: NativeCallOutput, unit: Int) raises:
    var expected = String("DBSCAN native ABI: missing argument span")
    var result = Int(MutPointer(to=output.result))
    if unit == 13:
        expected = "DBSCAN native ABI: missing result span"
        result = 0
    var status = dbscan_native_v1[DBSCANWireOracle](99, 0, Int(unit == 14), 0, Int(unit == 15),
        0, Int(unit == 16), result, Int(MutPointer(to=output.error)), native_descriptor_error)
    output.error.append(0)
    require(status != 0 and String(unsafe_from_utf8_ptr=output.error.unsafe_ptr()) == expected,
        "DBSCAN null span refusal precedes operation dispatch")
    _ = output


def malformed(unit: Int) raises:
    var operation = 99
    var addresses = List[Int]()
    var integers = List[Int]()
    var scalars = List[Float64]()
    var expected = String("DBSCAN native ABI: unknown operation")
    if unit >= 18 and unit <= 20:
        operation = unit - 18
        expected = "DBSCAN native ABI: metadata takes no arguments"
        if unit == 18:
            addresses.append(1)
        elif unit == 19:
            integers.append(1)
        else:
            scalars.append(1)
    elif unit >= 21:
        var part = (unit - 21) % 3
        operation = DBSCAN_FIT + (unit - 21) // 3
        var na = 0
        var np = 0
        var nf = 0
        var name = String("")
        if operation == DBSCAN_FIT:
            na = 4
            np = 7
            nf = 1
            name = "fit"
        elif operation == DBSCAN_CORE_COUNT:
            na = 3
            np = 2
            name = "core_count"
        elif operation == DBSCAN_CORE_GATHER:
            na = 6
            np = 3
            name = "core_gather"
        elif operation == DBSCAN_PREPARE:
            na = 3
            np = 3
            name = "prepare"
        elif operation == DBSCAN_RELEASE:
            np = 1
            name = "release"
        elif operation == DBSCAN_QUERY:
            na = 6
            np = 5
            nf = 1
            name = "query"
        else:
            na = 2
            np = 5
            nf = 1
            name = "resident"
        if part == 0:
            na = na - 1 if na > 0 else 1
        elif part == 1:
            np -= 1
        else:
            nf = 0 if nf > 0 else 1
        addresses = List[Int](length=na, fill=0)
        integers = List[Int](length=np, fill=0)
        scalars = List[Float64](length=nf, fill=0)
        expected = "DBSCAN native ABI: " + name + " span length mismatch"
    var refused = False
    try:
        _ = native_descriptor_call(dbscan_native_v1[DBSCANWireOracle], operation, addresses, integers, scalars)
    except error:
        refused = String(error) == expected
    finally:
        _ = addresses^
        _ = integers^
        _ = scalars^
    require(refused, "DBSCAN malformed request preserves exact error")


def fixture(unit: Int) raises:
    comptime assert DBSCAN_NATIVE_COUNT == 10 and DBSCAN_MODE == 0 and DBSCAN_VENDOR == 1 and DBSCAN_CAPABILITIES == 2
    comptime assert DBSCAN_FIT == 3 and DBSCAN_CORE_COUNT == 4 and DBSCAN_CORE_GATHER == 5 and DBSCAN_PREPARE == 6
    comptime assert DBSCAN_RELEASE == 7 and DBSCAN_QUERY == 8 and DBSCAN_RESIDENT_QUERY == 9
    var driver = DBSCANNativeDriver(dbscan_native_v1[DBSCANWireOracle])
    if unit == 0:
        require(driver.metadata(DBSCAN_MODE) == 1 and driver.metadata(DBSCAN_VENDOR) == 4
            and driver.metadata(DBSCAN_CAPABILITIES) == 1, "DBSCAN fixed metadata slots")
    elif unit <= 2:
        require(driver.fit(fit_request(unit == 2)) == EXACT_RESULT + unit - 1, "DBSCAN optional weights and integer fit result")
    elif unit == 3:
        require(driver.core_count(DBSCANCoreRequest(43, 47, 53, 59, 61)) == EXACT_RESULT + 2, "DBSCAN integer core count")
    elif unit == 4:
        require(driver.core_gather(DBSCANGatherRequest(67, 71, 73, 79, 83, 89, 97, 101, 103)) == EXACT_RESULT + 3,
            "DBSCAN core gather result")
    elif unit == 5 or unit == 43:
        require(driver.prepare(DBSCANReferenceRequest(107, 109, 113, 131 if unit == 5 else 0, 127, 2))
            == (EXACT_RESULT + 4 if unit == 5 else 0), "DBSCAN exact reference handle, including zero")
    elif unit == 6 or unit == 42:
        require(driver.release(EXACT_RESULT if unit == 6 else 0) == (EXACT_RESULT + 5 if unit == 6 else 0),
            "DBSCAN exact release result and zero handle")
    elif unit <= 8:
        require(driver.query(DBSCANQueryRequest(137, 139, 149, 151, 157, 163, 167, 173, 179, 1, unit == 8,
            0.10000000000000002)) == EXACT_RESULT + unit - 1, "DBSCAN query threshold flag and integer result")
    elif unit <= 10:
        require(driver.resident(DBSCANResidentQuery(181, 191, EXACT_RESULT, 193, 197, 0, unit == 10,
            0.10000000000000002)) == EXACT_RESULT + unit - 1, "DBSCAN resident threshold flag and integer result")
    elif unit == 11:
        var original = fit_request(False)
        var copied = original.copy()
        copied.weights = 17
        copied.method = 0
        copied.eps = 2.0
        require(original.weights == 0 and original.method == 1 and original.eps == Float64(0.10000000000000002),
            "DBSCAN copied native request independence")
        require(driver.fit(original) == EXACT_RESULT, "DBSCAN copied request retains original descriptor fields")
    elif unit == 12:
        var request = fit_request(False)
        request.x = -1
        var refused = False
        try:
            _ = driver.fit(request)
        except error:
            refused = String(error) == "DBSCAN fit tensor slots"
        require(refused, "DBSCAN provider error crosses descriptor unchanged")
    elif unit <= 16:
        var owners = List[NativeCallOutput]()
        owners.append(NativeCallOutput())
        try:
            missing_span(owners[0], unit)
        finally:
            _ = owners^
    elif unit == 44:
        var request = fit_request(False)
        request.eps = bitcast[DType.float64](UInt64(0x8000000000000000))
        require(driver.fit(request) == EXACT_RESULT, "DBSCAN signed-zero radius wire preservation")
    else:
        malformed(unit)


def run_checks() raises:
    var failures = List[Int](length=DBSCAN_CONTRACT_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL dbscan_contract fixture", unit, String(error))
    host_parallelize(task, DBSCAN_CONTRACT_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "dbscan_contract failed " + String(failed) + " fixtures")
    print("PASS dbscan_contract:", DBSCAN_CONTRACT_FIXTURES, "native request/result/error fixtures")
