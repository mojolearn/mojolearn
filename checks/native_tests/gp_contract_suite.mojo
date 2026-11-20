# SPDX-License-Identifier: Apache-2.0
"""Prescribed GP request transport and provider-error contracts.

All57 independent cases run across host cores. The provider below reads only
the descriptor envelopes and the driver's own two-word status storage. Fake
tensor addresses are never dereferenced. Numerical providers have separate
actual execution fixtures; this suite proves no numerical behavior.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.gp_native_api import GPDriver, GPKernelView, GPRFitRequest, GPRQueryRequest, GPRCovarianceRequest, GPRSampleRequest, GPRGradientRequest, GPROptimizeRequest, GPCFitRequest, GPCFitAllRequest, GPCQueryRequest, GPCQueryAllRequest, GPCCombineRequest, GPThetaRequest
from checks.native_tests.gp_state_suite import require

comptime GP_CONTRACT_FIXTURES = 57
comptime GP_WIRE_RESULT = 9007199254740993


def check_words(address: Int, count: Int, expected: List[Int], label: String) raises:
    require(count == len(expected), "GP " + label + " extent")
    if count > 0:
        require(address != 0, "GP " + label + " address")
        var words = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=address)
        for slot in range(count):  # driver-loop: fixed request-envelope metadata
            if expected[slot] == -999:
                require(words[slot] != 0, "GP owned query status storage")
            else:
                require(words[slot] == expected[slot], "GP " + label + " slot " + String(slot))


def prescribed_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                     result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        require(operation >= 0 and operation < 17 and result != 0, "GP operation/result envelope")
        var a = List[Int]()
        var p = List[Int]()
        var floats = List[Float64]()
        var variant = 0
        if operation >= 3 and operation <= 14:
            require(pp != 0 and np > 0, "GP prescribed request integer span")
            variant = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)[0] - 101
            require(variant >= 0 and variant < 12, "GP fixture variant")
        if operation == 3:
            a = [31, 37, 41, 43, 47, 11, 13, 17, 19]
            p = [101 + variant, 103, 23, 29, variant & 1]
            floats = [0.10000000000000002]
        elif operation == 4:
            a = [31, 41, 43, 53, 59, 61, 67, 71, 11, 13, 17, 19, -999]
            p = [101 + variant, 103, 107, 23, 29, 109, variant & 1, (variant >> 1) & 1,
                 variant // 4, GP_WIRE_RESULT, 113, 127, 131]
            floats = [1.0000000000000002, bitcast[DType.float64](UInt64(0x8000000000000000))]
        elif operation == 5:
            a = [31, 41, 43, 53, 59, 73, 11, 13, 17, 19]
            p = [101 + variant, 103, 107, 23, 29, 109, variant & 1]
            floats = [1.0000000000000002, bitcast[DType.float64](UInt64(0x8000000000000000))]
        elif operation == 6:
            a = [31, 41, 43, 53, 79, 11, 13, 17, 19]
            p = [101 + variant, 103, 107, 23, 29, 109, 137, 4294967295, 2147483649, variant & 1]
            floats = [1.0000000000000002, bitcast[DType.float64](UInt64(0x8000000000000000))]
        elif operation == 7:
            a = [31, 37, 83, 89, 47, 11, 13, 17, 19]
            p = [101 + variant, 103, 23, 29]
            floats = [0.10000000000000002]
        elif operation == 8:
            a = [31, 37, 83, 97, 139, 149, 151, 11, 13, 17, 19]
            p = [101 + variant, 103, 23, 29, 137 + variant, 4294967295, 2147483649]
            floats = [0.10000000000000002]
        elif operation == 9:
            a = [31, 37, 41, 163, 167, 47, 11, 13, 17, 19, 151]
            p = [101 + variant, 103, 23, 29, 139, variant & 1, 149]
        elif operation == 10:
            a = [31, 181, 157, 37, 41, 163, 167, 47, 11, 13, 17, 19]
            p = [101 + variant, 103, 23, 29, 139, 173]
        elif operation == 11:
            a = [31, 37, 163, 167, 41, 53, 59, 61, 179, 11, 13, 17, 19, 181]
            p = [101 + variant, 103, 107, 23, 29, variant & 1, variant // 2]
        elif operation == 12:
            a = [31, 53, 181, 37, 163, 167, 41, 11, 13, 17, 19, -999]
            p = [101 + variant, 103, 107, 23, 29, 1 + variant, 173, GP_WIRE_RESULT, 113, 127, 131]
        elif operation == 13:
            a = [157, 179, 181]
            p = [101 + variant, 173]
        elif operation == 14:
            a = [31, 181]
            p = [101 + variant]
        elif operation == 16:
            require(pp != 0 and np == 1, "GP lease-close span")
            var lease = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)[0]
            require(lease == 0 or lease == GP_WIRE_RESULT, "GP exact zero/wide lease handle")
            p = [lease]
        check_words(aa, na, a, "address")
        check_words(pp, np, p, "integer")
        require(nf == len(floats), "GP binary64 span extent")
        if nf > 0:
            require(ff != 0, "GP binary64 span address")
            var words = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
            for slot in range(nf):  # driver-loop: fixed scalar request metadata
                require(bitcast[DType.uint64](words[slot]) == bitcast[DType.uint64](floats[slot]),
                    "GP binary64 value/signed-zero transport")
        if operation == 4 or operation == 12:
            var addresses = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var status = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=addresses[12 if operation == 4 else 11])
            status[0] = 17
            status[1] = -19
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = GP_WIRE_RESULT + operation
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def rejected_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                   result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    native_descriptor_report(Error("GP prescribed native provider refusal"), context, sink)
    return 1


def invoke(driver: GPDriver, operation: Int, variant: Int) raises -> Int:
    var kernel = GPKernelView(11, 13, 17, 19, 23, 29)
    var rows = 101 + variant
    var negative_zero = bitcast[DType.float64](UInt64(0x8000000000000000))
    if operation < 3:
        return driver.metadata(operation)
    if operation == 3:
        return driver.gpr_fit(GPRFitRequest(31, 37, 41, 43, 47, rows, 103, kernel.copy(), 0.10000000000000002, (variant & 1) != 0))
    if operation == 4:
        var result = driver.gpr_query(GPRQueryRequest(31, 41, 43, 53, 59, 61, 67, 71, rows, 103, 107,
            kernel.copy(), 109, (variant & 1) != 0, (variant & 2) != 0, 1.0000000000000002, negative_zero,
            variant // 4, GP_WIRE_RESULT, 113, 127, 131))
        require(result.finite == 17 and result.range_status == -19, "GP query returns both owned status words")
        return result.value
    if operation == 5:
        return driver.gpr_covariance(GPRCovarianceRequest(31, 41, 43, 53, 59, 73, rows, 103, 107,
            kernel.copy(), 109, (variant & 1) != 0, 1.0000000000000002, negative_zero))
    if operation == 6:
        return driver.gpr_sample(GPRSampleRequest(31, 41, 43, 53, 79, rows, 103, 107, kernel.copy(),
            109, 137, 4294967295, 2147483649, (variant & 1) != 0, 1.0000000000000002, negative_zero))
    if operation == 7:
        return driver.gpr_gradient(GPRGradientRequest(31, 37, 83, 89, 47, rows, 103, kernel.copy(), 0.10000000000000002))
    if operation == 8:
        return driver.gpr_optimize(GPROptimizeRequest(31, 37, 83, 97, 139, 149, 151, rows, 103,
            kernel.copy(), 0.10000000000000002, 137 + variant, 4294967295, 2147483649))
    if operation == 9:
        return driver.gpc_fit(GPCFitRequest(31, 37, 41, 163, 167, 47, rows, 103,
            kernel.copy(), 139, (variant & 1) != 0, 149, 151))
    if operation == 10:
        return driver.gpc_fit_all(GPCFitAllRequest(31, 181, 157, 37, 41, 163, 167, 47,
            rows, 103, kernel.copy(), 139, 173))
    if operation == 11:
        return driver.gpc_query(GPCQueryRequest(31, 37, 163, 167, 41, 53, 59, 61, 179,
            rows, 103, 107, kernel.copy(), (variant & 1) != 0, variant // 2, 181))
    if operation == 12:
        var result = driver.gpc_query_all(GPCQueryAllRequest(31, 53, 181, 37, 163, 167, 41,
            rows, 103, 107, kernel.copy(), 1 + variant, 173, GP_WIRE_RESULT, 113, 127, 131))
        require(result.finite == 17 and result.range_status == -19, "GPC query returns both owned status words")
        return result.value
    if operation == 13:
        return driver.gpc_combine(GPCCombineRequest(157, 179, 181, rows, 173))
    if operation == 14:
        return driver.theta(GPThetaRequest(31, 181, rows))
    if operation == 15:
        return driver.lease_open()
    driver.lease_close(0 if variant == 1 else GP_WIRE_RESULT)
    return GP_WIRE_RESULT + 16


def fixture(unit: Int) raises:
    var operation = unit
    var variant = 0
    if unit >= 40:
        operation = unit - 40
        var refused = False
        try:
            _ = invoke(GPDriver(rejected_entry), operation, 0)
        except error:
            require(String(error) == "GP prescribed native provider refusal", "GP original provider error propagated")
            refused = True
        require(refused, "GP every operation propagates native refusal")
        return
    if unit == 17:
        operation = 3
        variant = 1
    elif unit >= 18 and unit <= 28:
        operation = 4
        variant = unit - 17
    elif unit == 29:
        operation = 5
        variant = 1
    elif unit == 30:
        operation = 6
        variant = 1
    elif unit == 31:
        operation = 8
        variant = 1
    elif unit == 32:
        operation = 9
        variant = 1
    elif unit >= 33 and unit <= 37:
        operation = 11
        variant = unit - 32
    elif unit == 38:
        operation = 12
        variant = 1
    elif unit == 39:
        operation = 16
        variant = 1
    require(invoke(GPDriver(prescribed_entry), operation, variant) == GP_WIRE_RESULT + operation,
        "GP all result words cross without Float64 rounding")


def run_checks() raises:
    var failures = List[Int](length=GP_CONTRACT_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL gp_contract fixture", unit, String(error))
    host_parallelize(task, GP_CONTRACT_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "gp_contract failed " + String(failed) + " fixtures")
    print("PASS gp_contract:", GP_CONTRACT_FIXTURES, "native request transport/error fixtures")
