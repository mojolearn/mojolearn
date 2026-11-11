# SPDX-License-Identifier: Apache-2.0
"""Typed TSVD wire fields, status words, leases and malformed-call contracts.

Distinct literal slot values expose address/option swaps. The wire oracle
implements no decomposition algorithm. Independent fixtures run in parallel.
"""
from core.host_parallel import host_parallelize
from core.native_call_output import NativeCallOutput
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_call, native_descriptor_error
from core.pca_native_api import PCAFoldStore
from core.tsvd_native_api import (
    TSVD_NATIVE_COUNT, TSVD_MODE, TSVD_VENDOR, TSVD_CAPABILITIES, TSVD_QFIX,
    TSVD_FIT, TSVD_QUERY, TSVD_OPEN, TSVD_CLOSE, TSVDNativeKernels,
    TSVDNativeDriver, TSVDFitRequest, TSVDFitResult, TSVDQueryRequest, tsvd_native_dispatch,
)

comptime TSVD_CONTRACT_FIXTURES = 32


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


struct TSVDWireOracle(TSVDNativeKernels):
    @staticmethod
    def mode() -> Int:
        return 1

    @staticmethod
    def vendor() -> Int:
        return 4

    @staticmethod
    def capabilities() -> Int:
        return 63

    @staticmethod
    def qfix() -> Int:
        return 2

    @staticmethod
    def fit(r: TSVDFitRequest) raises -> TSVDFitResult:
        require(r.source == 11 and r.components == 13 and r.singular == 17
                and r.variance == 19 and r.ratio == 23, "TSVD fit address slots")
        require(r.rows == 29 and r.features == 31 and r.count == 37
                and r.oversamples == 41 and r.iterations == 43 and r.seed == 47,
                "TSVD fit dimension/options slots")
        require(r.route >= 0 and r.route <= 5, "TSVD fit route slot")
        if r.fold.active():
            require(r.fold.base == 53 and r.fold.rows == 59 and r.fold.indices == 61,
                    "TSVD resident fold slots")
        return TSVDFitResult(-13, r.route + 16 * Int(r.auto_iterations)
            + 32 * Int(r.test_finite) + 64 * Int(r.fold.active()))

    @staticmethod
    def query(r: TSVDQueryRequest) raises -> Int:
        require(r.source == 67 and r.components == 71 and r.dst == 73,
                "TSVD query address slots")
        require(r.rows == 79 and r.features == 83 and r.count == 89 and r.lease == 97,
                "TSVD query dimension/lease slots")
        return -10 - Int(r.inverse) - 2 * Int(r.test_finite) - 4 * Int(r.resident)

    @staticmethod
    def open() raises -> Int:
        return 7331

    @staticmethod
    def close(lease: Int) raises:
        require(lease == 7331, "TSVD native lease close preserves exact token")


def tsvd_wire_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return tsvd_native_dispatch[TSVDWireOracle](operation, aa, na, pp, np, ff, nf, result, context, sink)


def fit_request(variant: Int) -> TSVDFitRequest:
    return TSVDFitRequest(11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
        (variant & 1) != 0, 47, (variant & 2) != 0, variant % 6,
        PCAFoldStore(53, 59, 61) if (variant & 4) != 0 else PCAFoldStore())


def query_request(variant: Int) -> TSVDQueryRequest:
    return TSVDQueryRequest(67, 71, 73, 79, 83, 89,
        (variant & 1) != 0, (variant & 2) != 0, (variant & 4) != 0, 97)


def missing_result(mut output: NativeCallOutput) raises:
    var status = tsvd_wire_entry(TSVD_MODE, 0, 0, 0, 0, 0, 0, 0,
        Int(MutPointer(to=output.error)), native_descriptor_error)
    output.error.append(0)
    require(status != 0 and String(unsafe_from_utf8_ptr=output.error.unsafe_ptr())
            == "TruncatedSVD native ABI: missing result span", "TSVD missing-result refusal")


def fixture(unit: Int) raises:
    comptime assert TSVD_NATIVE_COUNT == 8 and TSVD_MODE == 0 and TSVD_VENDOR == 1
    comptime assert TSVD_CAPABILITIES == 2 and TSVD_QFIX == 3 and TSVD_FIT == 4
    comptime assert TSVD_QUERY == 5 and TSVD_OPEN == 6 and TSVD_CLOSE == 7
    var driver = TSVDNativeDriver(tsvd_wire_entry)
    if unit == 0:
        require(driver.metadata(TSVD_MODE) == 1 and driver.metadata(TSVD_VENDOR) == 4
                and driver.metadata(TSVD_CAPABILITIES) == 63 and driver.metadata(TSVD_QFIX) == 2,
                "TSVD metadata operation/result mapping")
    elif unit < 9:
        var variant = unit - 1
        var answer = driver.fit(fit_request(variant))
        require(answer.finite == -13 and answer.in_range == variant % 6 + 16 * variant,
                "TSVD fit status slots and independent Boolean combinations")
    elif unit < 17:
        var variant = unit - 9
        require(driver.query(query_request(variant)) == -10 - variant,
                "TSVD query status and independent Boolean combinations")
    elif unit == 17:
        var lease = driver.open()
        require(lease == 7331, "TSVD opened lease result")
        driver.close(lease)
    elif unit == 18:
        var refused = False
        try:
            driver.close(7332)
        except error:
            refused = String(error) == "TSVD native lease close preserves exact token"
        require(refused, "TSVD lease-close native errors propagate")
    elif unit == 19:
        var original = fit_request(7)
        var copied = original.copy()
        copied.source = 1001
        copied.fold.base = -1
        copied.auto_iterations = False
        require(original.source == 11 and original.fold.base == 53 and original.auto_iterations,
                "TSVD fit copy owns native nested settings")
    elif unit == 20:
        var original = query_request(7)
        var copied = original.copy()
        copied.components = 1003
        copied.lease = 1007
        copied.resident = False
        require(original.components == 71 and original.lease == 97 and original.resident,
                "TSVD query copy preserves original lease/address settings")
    elif unit == 31:
        var owners = List[NativeCallOutput]()
        owners.append(NativeCallOutput())
        missing_result(owners[0])
        _ = owners^
    else:
        var operation = 8
        var addresses = List[Int]()
        var integers = List[Int]()
        var scalars = List[Float64]()
        var expected = String("TruncatedSVD native ABI: unknown operation")
        if unit == 22:
            operation = TSVD_FIT
            expected = "TruncatedSVD native ABI: fit requires 6 addresses and 12 integers"
        elif unit == 23:
            operation = TSVD_QUERY
            expected = "TruncatedSVD native ABI: query requires 3 addresses and 7 integers"
        elif unit == 24:
            operation = TSVD_QFIX
            addresses.append(11)
            expected = "TruncatedSVD native ABI: metadata takes no arguments"
        elif unit == 25:
            operation = TSVD_OPEN
            integers.append(11)
            expected = "TruncatedSVD native ABI: lease open takes no arguments"
        elif unit == 26 or unit == 27:
            operation = TSVD_CLOSE
            expected = "TruncatedSVD native ABI: lease close requires one integer"
            if unit == 27:
                integers.append(7331)
                scalars.append(0.0)
        elif unit == 28 or unit == 29:
            operation = TSVD_FIT
            addresses = List[Int](length=6, fill=0)
            integers = List[Int](length=12, fill=0)
            expected = "TruncatedSVD native ABI: missing fit output span"
            if unit == 29:
                scalars.append(0.0)
                expected = "TruncatedSVD native ABI: fit requires 6 addresses and 12 integers"
        elif unit == 30:
            operation = TSVD_QUERY
            addresses = List[Int](length=3, fill=0)
            integers = List[Int](length=7, fill=0)
            scalars.append(0.0)
            expected = "TruncatedSVD native ABI: query requires 3 addresses and 7 integers"
        var refused = False
        try:
            _ = native_descriptor_call(tsvd_wire_entry, operation, addresses, integers, scalars)
        except error:
            refused = String(error) == expected
        require(refused, "TSVD malformed wire request refusal")


def run_checks() raises:
    var failures = List[Int](length=TSVD_CONTRACT_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL tsvd_contract fixture", unit, String(error))
    host_parallelize(task, TSVD_CONTRACT_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "tsvd_contract failed " + String(failed) + " fixtures")
    print("PASS tsvd_contract:", TSVD_CONTRACT_FIXTURES, "typed request/result/lease contracts")
