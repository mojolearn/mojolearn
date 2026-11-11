# SPDX-License-Identifier: Apache-2.0
"""NMF native wire/state contracts and analytic regularization values.

The test-only entry validates distinct request slots and implements no NMF
solver. Independent fixtures run in parallel, with heap-owned ABI outputs.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.native_call_output import NativeCallOutput
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_call, native_descriptor_error
from x_decomp.nmf import NmfArgs
from core.nmf_native_api import (
    NMF_NATIVE_COUNT, NMF_MODE, NMF_VENDOR, NMF_CAPABILITIES, NMF_FIT, NMF_INVERSE,
    NMFNativeKernels, NMFNativeDriver, NMFFitRequest, NMFFitResult, NMFInverseRequest,
    nmf_native_dispatch, nmf_regularization,
)

comptime NMF_CONTRACT_FIXTURES = 28


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


struct NMFWireOracle(NMFNativeKernels):
    @staticmethod
    def mode() -> Int:
        return 1

    @staticmethod
    def vendor() -> Int:
        return 3

    @staticmethod
    def capabilities() -> Int:
        return 3

    @staticmethod
    def fit(r: NMFFitRequest) raises -> NMFFitResult:
        require(r.source == 11 and r.w_initial == 13 and r.h_initial == 17
                and r.w_output == 19 and r.h_output == 23, "NMF fit address slots")
        require(r.rows == 29 and r.features == 31 and r.components == 37
                and r.initialization == 41 and r.solver == 43, "NMF fit dimension/route slots")
        require(r.args.beta == 1.5 and r.args.max_iter == 47 and r.args.tol == 0.0625
                and r.args.l1w == 0.125 and r.args.l1h == 0.25 and r.args.l2w == 0.5
                and r.args.l2h == 0.75 and r.args.seed == 53, "NMF native options and Float64 slots")
        return NMFFitResult(59, -0.0, Int(r.args.update_h) + 2 * Int(r.args.shuffle) + 4 * Int(r.test_finite))

    @staticmethod
    def inverse(r: NMFInverseRequest) raises:
        require(r.source == 61 and r.components == 67 and r.dst == 71
                and r.rows == 73 and r.features == 79 and r.count == 83,
                "NMF inverse address/dimension slots")


def nmf_wire_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return nmf_native_dispatch[NMFWireOracle](operation, aa, na, pp, np, ff, nf, result, context, sink)


def fit_request(variant: Int) -> NMFFitRequest:
    var args = NmfArgs(1.5, (variant & 1) != 0, 47, 0.0625,
        0.125, 0.25, 0.5, 0.75, (variant & 2) != 0, 53)
    return NMFFitRequest(11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
        args^, (variant & 4) != 0)


def regularization_fixture(unit: Int) raises:
    var rows = 7
    var features = 11
    var alpha_w = Float64(0.5)
    var alpha_h = Float64(0.25)
    var ratio = Float64(0.25)
    var expected: List[Float64] = [1.375, 0.4375, 4.125, 1.3125]
    if unit == 12:
        ratio = 0
        expected = [0, 0, 5.5, 1.75]
    elif unit == 13:
        ratio = 1
        expected = [5.5, 1.75, 0, 0]
    elif unit == 15:
        alpha_w = 0
        expected = [0, 0.4375, 0, 1.3125]
    elif unit == 16:
        rows = 1
        features = 1
        ratio = 0.5
        expected = [0.25, 0.125, 0.25, 0.125]
    elif unit == 17:
        rows = 28
        features = 44
        expected = [5.5, 1.75, 16.5, 5.25]
    var actual = nmf_regularization(rows, features, alpha_w, alpha_h, ratio)
    require(actual.l1w == expected[0] and actual.l1h == expected[1]
            and actual.l2w == expected[2] and actual.l2h == expected[3],
            "NMF analytic row/feature-scaled L1/L2 regularization")


def missing_result(mut output: NativeCallOutput) raises:
    var status = nmf_wire_entry(NMF_MODE, 0, 0, 0, 0, 0, 0, 0,
        Int(MutPointer(to=output.error)), native_descriptor_error)
    output.error.append(0)
    require(status != 0 and String(unsafe_from_utf8_ptr=output.error.unsafe_ptr())
            == "NMF native ABI: missing result span", "NMF missing-result refusal")
    _ = output


def fixture(unit: Int) raises:
    comptime assert NMF_NATIVE_COUNT == 5 and NMF_MODE == 0 and NMF_VENDOR == 1
    comptime assert NMF_CAPABILITIES == 2 and NMF_FIT == 3 and NMF_INVERSE == 4
    var driver = NMFNativeDriver(nmf_wire_entry)
    if unit == 0:
        require(driver.metadata(NMF_MODE) == 1 and driver.metadata(NMF_VENDOR) == 3
                and driver.metadata(NMF_CAPABILITIES) == 3, "NMF metadata operation results")
    elif unit < 9:
        var variant = unit - 1
        var answer = driver.fit(fit_request(variant))
        require(answer.iterations == 59 and answer.finite == variant, "NMF fit integer result slots")
        require(bitcast[DType.uint64](answer.error) == UInt64(0x8000000000000000),
                "NMF independent Float64 result preserves signed zero")
    elif unit == 9:
        driver.inverse(NMFInverseRequest(61, 67, 71, 73, 79, 83))
    elif unit == 10:
        var original = fit_request(7)
        var copied = original.copy()
        copied.args.tol = 0.5
        copied.args.update_h = False
        copied.w_output = 101
        require(original.args.tol == 0.0625 and original.args.update_h and original.w_output == 19,
                "NMF native request copy does not alias nested options")
    elif unit == 11:
        var original = NMFInverseRequest(61, 67, 71, 73, 79, 83)
        var copied = original.copy()
        copied.components = 107
        copied.count = 109
        require(original.components == 67 and original.count == 83, "NMF inverse request copy independence")
    elif unit < 18:
        regularization_fixture(unit)
    elif unit == 18:
        var original = nmf_regularization(7, 11, 0.5, 0.25, 0.25)
        var copied = original.copy()
        copied.l1w = 99
        require(original.l1w == 1.375 and copied.l1w == 99 and copied.l2h == 1.3125,
                "NMF regularization record copy owns typed values")
    elif unit == 27:
        var owners = List[NativeCallOutput]()
        owners.append(NativeCallOutput())
        try:
            missing_result(owners[0])
        finally:
            _ = owners^
    else:
        var operation = 5
        var addresses = List[Int]()
        var integers = List[Int]()
        var scalars = List[Float64]()
        var words = List[Int](length=2, fill=-1)
        var error_value = List[Float64](length=1, fill=-1)
        var expected = String("NMF native ABI: unknown operation")
        if unit == 20:
            operation = NMF_FIT
            expected = "NMF native ABI: fit requires 7 addresses, 10 integers and 6 floats"
        elif unit == 21:
            operation = NMF_INVERSE
            expected = "NMF native ABI: inverse requires 3 addresses and 3 integers"
        elif unit == 22:
            operation = NMF_MODE
            addresses.append(1)
            expected = "NMF native ABI: metadata takes no arguments"
        elif unit >= 23 and unit <= 25:
            operation = NMF_FIT
            addresses = List[Int](length=7, fill=0)
            integers = List[Int](length=10, fill=0)
            scalars = List[Float64](length=5 if unit == 25 else 6, fill=0)
            addresses[5] = 0 if unit == 23 else Int(words.unsafe_ptr())
            addresses[6] = 0 if unit == 24 else Int(error_value.unsafe_ptr())
            expected = "NMF native ABI: fit requires 7 addresses, 10 integers and 6 floats" if unit == 25 else "NMF native ABI: missing fit output spans"
        elif unit == 26:
            operation = NMF_INVERSE
            addresses = List[Int](length=3, fill=0)
            integers = List[Int](length=3, fill=0)
            scalars.append(0.0)
            expected = "NMF native ABI: inverse requires 3 addresses and 3 integers"
        var refused = False
        try:
            _ = native_descriptor_call(nmf_wire_entry, operation, addresses, integers, scalars)
        except error:
            refused = String(error) == expected
        finally:
            # Both output spans stay owned even on the malformed/refusal path.
            _ = words^
            _ = error_value^
            _ = addresses^
            _ = integers^
            _ = scalars^
        require(refused, "NMF malformed wire refusal")


def run_checks() raises:
    var failures = List[Int](length=NMF_CONTRACT_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL nmf_contract fixture", unit, String(error))
    host_parallelize(task, NMF_CONTRACT_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "nmf_contract failed " + String(failed) + " fixtures")
    print("PASS nmf_contract:", NMF_CONTRACT_FIXTURES, "native wire/regularization/state fixtures")
