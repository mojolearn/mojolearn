# SPDX-License-Identifier: Apache-2.0
"""KDE length-delimited names, exact handles/totals and native wire refusals.

The mock checks prescribed metadata only; fake tensor addresses are never
dereferenced. All requests and error/name storage stay owned through calls.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.native_call_output import NativeCallOutput
from core.native_descriptor_call import native_descriptor_call, native_descriptor_error
from core.kde_native_api import KDENativeKernels, KDENativeDriver, KDEPrepareRequest, KDEQueryRequest, KDEResidentQuery, kde_native_v1, KDE_NATIVE_COUNT, KDE_MODE, KDE_VENDOR, KDE_CAPABILITIES, KDE_PREPARE, KDE_RELEASE, KDE_QUERY, KDE_RESIDENT_QUERY

comptime KDE_CONTRACT_FIXTURES = 32


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


struct KDEWireOracle(KDENativeKernels):
    @staticmethod
    def mode() -> Int:
        return 1

    @staticmethod
    def vendor() -> Int:
        return 4

    @staticmethod
    def capabilities() -> Int:
        return 7

    @staticmethod
    def prepare(r: KDEPrepareRequest) raises -> Int:
        require(r.train == 11 and r.weights == 13 and r.features == 19 and r.bandwidth == Float64(0.10000000000000002),
            "KDE prepare fields")
        if r.rows == 99:
            require(r.kernel == "" and r.metric == "", "KDE empty names retain zero length")
            return 17
        require(r.rows == 17 and r.kernel == "kernel\0λ🙂" and r.metric == "metric\0世界",
            "KDE prepare name spans and dimensions")
        return 9007199254740993 + Int(r.has_weights) + 2 * Int(r.finite_tested)

    @staticmethod
    def release(handle: Int) raises:
        require(handle == 9007199254740993, "KDE release integer handle is not rounded through Float64")

    @staticmethod
    def query(r: KDEQueryRequest) raises -> Float64:
        require(r.train == 23 and r.query == 29 and r.weights == 31 and r.dst == 37,
            "KDE query addresses")
        require(r.train_rows == 41 and r.query_rows == 43 and r.features == 47
            and r.bandwidth == Float64(0.10000000000000002), "KDE query dimensions/scalar")
        require(r.kernel == ("weighted\0🙂" if r.has_weights else "uniform\0λ") and r.metric == "metric\0世界",
            "KDE query Boolean and length-delimited names")
        var bits = UInt64(0x8000000000000000)
        if r.want_total:
            bits = 0x3FF0000000000001
        return bitcast[DType.float64](bits)

    @staticmethod
    def resident(r: KDEResidentQuery) raises -> Float64:
        require(r.handle == 9007199254740993 and r.query == 53 and r.dst == 59 and r.status == 61
            and r.rows == 67 and r.features == 71 and r.bandwidth == Float64(0.10000000000000002),
            "KDE resident handle/addresses/dimensions/scalar")
        require(r.kernel == "kernel\0λ🙂" and r.metric == "metric\0世界", "KDE resident name spans")
        return -123.125 if r.want_total else 0.0


def prepare(variant: Int) -> KDEPrepareRequest:
    return KDEPrepareRequest(11, 13, 17, 19, 0.10000000000000002, (variant & 1) != 0,
        (variant & 2) != 0, String("kernel\0λ🙂"), String("metric\0世界"))


def missing_result(mut output: NativeCallOutput) raises:
    var status = kde_native_v1[KDEWireOracle](KDE_MODE, 0, 0, 0, 0, 0, 0, 0,
        Int(MutPointer(to=output.error)), native_descriptor_error)
    output.error.append(0)
    require(status != 0 and String(unsafe_from_utf8_ptr=output.error.unsafe_ptr())
        == "KDE native ABI: missing result span", "KDE missing-result refusal")
    _ = output


def fixture(unit: Int) raises:
    comptime assert KDE_NATIVE_COUNT == 7 and KDE_MODE == 0 and KDE_VENDOR == 1 and KDE_CAPABILITIES == 2
    comptime assert KDE_PREPARE == 3 and KDE_RELEASE == 4 and KDE_QUERY == 5 and KDE_RESIDENT_QUERY == 6
    var driver = KDENativeDriver(kde_native_v1[KDEWireOracle])
    if unit == 0:
        require(driver.metadata(KDE_MODE) == 1 and driver.metadata(KDE_VENDOR) == 4
            and driver.metadata(KDE_CAPABILITIES) == 7, "KDE metadata operation slots")
    elif unit <= 4:
        require(driver.prepare(prepare(unit - 1)) == 9007199254740993 + unit - 1,
            "KDE prepare flags and exact integer result")
    elif unit <= 8:
        var variant = unit - 5
        var weighted = (variant & 1) != 0
        var total = driver.query(KDEQueryRequest(23, 29, 31, 37, 41, 43, 47, 0.10000000000000002,
            weighted, (variant & 2) != 0, String("weighted\0🙂" if weighted else "uniform\0λ"), String("metric\0世界")))
        var expected_bits = UInt64(0x8000000000000000)
        if (variant & 2) != 0:
            expected_bits = 0x3FF0000000000001
        require(bitcast[DType.uint64](total) == expected_bits,
            "KDE total keeps binary64 precision and signed zero")
    elif unit <= 10:
        var total = driver.resident(KDEResidentQuery(9007199254740993, 53, 59, 61, 67, 71,
            0.10000000000000002, unit == 10, String("kernel\0λ🙂"), String("metric\0世界")))
        require(total == (-123.125 if unit == 10 else 0.0), "KDE resident total result")
    elif unit == 11:
        driver.release(9007199254740993)
    elif unit == 12:
        var original = prepare(0)
        var copied = original.copy()
        copied.train = 97
        copied.kernel = "changed"
        copied.has_weights = True
        require(original.train == 11 and original.kernel == "kernel\0λ🙂" and not original.has_weights,
            "KDE copied request has independent native fields")
        require(driver.prepare(original) == 9007199254740993, "KDE request survives copied-name replacement")
    elif unit == 13:
        var request = prepare(0)
        request.train = -1
        var refused = False
        try:
            _ = driver.prepare(request)
        except error:
            refused = String(error) == "KDE prepare fields"
        require(refused, "KDE provider error crosses native descriptor unchanged")
    elif unit == 30:
        var owners = List[NativeCallOutput]()
        owners.append(NativeCallOutput())
        try:
            missing_result(owners[0])
        finally:
            _ = owners^
    elif unit == 31:
        var request = prepare(0)
        request.rows = 99
        request.kernel = ""
        request.metric = ""
        require(driver.prepare(request) == 17, "KDE empty length-delimited names")
    else:
        var operation = 99
        var addresses = List[Int]()
        var integers = List[Int]()
        var scalars = List[Float64]()
        var expected = String("KDE native ABI: unknown operation")
        if unit == 15:
            operation = KDE_MODE
            addresses.append(1)
            expected = "KDE native ABI: metadata takes no arguments"
        elif unit == 16:
            operation = KDE_RELEASE
            expected = "KDE native ABI: release requires one integer"
        elif (unit >= 17 and unit <= 19) or unit >= 28:
            operation = KDE_PREPARE
            addresses = List[Int](length=3 if unit == 17 else 4, fill=0)
            integers = List[Int](length=5 if unit == 18 else 6, fill=0)
            scalars = List[Float64](length=0 if unit == 19 else 1, fill=0)
            expected = "KDE native ABI: prepare requires 4 addresses, 6 integers and one scalar"
            if unit >= 28:
                integers[4] = -1 if unit == 28 else 1
                expected = "KDE native ABI: invalid name span"
        elif unit >= 20 and unit <= 23:
            operation = KDE_QUERY
            addresses = List[Int](length=6 if unit == 20 else 7, fill=0)
            integers = List[Int](length=6 if unit == 21 else 7, fill=0)
            scalars = List[Float64](length=2 if unit == 22 else 1, fill=0)
            expected = "KDE native ABI: missing total output" if unit == 23 else "KDE native ABI: query requires 7 addresses, 7 integers and one scalar"
        elif unit >= 24 and unit <= 27:
            operation = KDE_RESIDENT_QUERY
            addresses = List[Int](length=5 if unit == 24 else 6, fill=0)
            integers = List[Int](length=5 if unit == 25 else 6, fill=0)
            scalars = List[Float64](length=0 if unit == 26 else 1, fill=0)
            expected = "KDE native ABI: missing total output" if unit == 27 else "KDE native ABI: resident query requires 6 addresses, 6 integers and one scalar"
        var refused = False
        try:
            _ = native_descriptor_call(kde_native_v1[KDEWireOracle], operation, addresses, integers, scalars)
        except error:
            refused = String(error) == expected
        finally:
            _ = addresses^
            _ = integers^
            _ = scalars^
        require(refused, "KDE malformed request exact refusal")


def run_checks() raises:
    var failures = List[Int](length=KDE_CONTRACT_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL kde_contract fixture", unit, String(error))
    host_parallelize(task, KDE_CONTRACT_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "kde_contract failed " + String(failed) + " fixtures")
    print("PASS kde_contract:", KDE_CONTRACT_FIXTURES, "native request/name/handle/error fixtures")
