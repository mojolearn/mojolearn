# SPDX-License-Identifier: Apache-2.0
"""IncrementalPCA typed request/state/result wire contracts, no numerical solver.

Distinct field values and every Boolean combination identify slot confusion.
Test-only handles never reach a real resident pool. ABI output spans and error
owners use stable heap storage held through success and refusal paths.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.native_call_output import NativeCallOutput
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_call, native_descriptor_error
from core.ipca_native_api import (
    IPCANativeKernels, IPCANativeDriver, IPCAStateIds, IPCAOutputs, IPCAPrevious,
    IPCAFitRequest, IPCAPartialRequest, IPCAQueryRequest, IPCAFitResult,
    ipca_native_dispatch, IPCA_NATIVE_COUNT, IPCA_MODE, IPCA_VENDOR,
    IPCA_CAPABILITIES, IPCA_FIT, IPCA_PARTIAL, IPCA_QUERY, IPCA_PREPARE, IPCA_FREE,
)

comptime IPCA_CONTRACT_FIXTURES = 34


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def outputs() -> IPCAOutputs:
    return IPCAOutputs(13, 17, 19, 23, 29, 31)


def check_outputs(value: IPCAOutputs) raises:
    require(value.components == 13 and value.singular == 17 and value.mean == 19
        and value.population_variance == 23 and value.explained_variance == 29 and value.ratio == 31,
        "IPCA six typed output address slots")


def input_ids() -> IPCAStateIds:
    var ids = IPCAStateIds()
    ids.values = [107, 109, 113, 127, 131]
    return ids^


def result_ids() -> IPCAStateIds:
    var ids = IPCAStateIds()
    ids.values = [0, 61, 67, 71, 73]
    return ids^


def check_ids(value: IPCAStateIds, expected: IPCAStateIds) raises:
    for index in range(5):  # Fixed five native identity fields.
        require(value.values[index] == expected.values[index], "IPCA native state identity slots")


struct IPCAWireOracle(IPCANativeKernels):
    @staticmethod
    def mode() -> Int:
        return 1

    @staticmethod
    def vendor() -> Int:
        return 4

    @staticmethod
    def capabilities() -> Int:
        return 15

    @staticmethod
    def fit(r: IPCAFitRequest) raises -> IPCAFitResult:
        require(r.source == (12 if r.keep else 11) and r.rows == 37 and r.features == 41
            and r.requested == 43 and r.batch_size == 47, "IPCA fit source/dimensions/keep slots")
        check_outputs(r.outputs)
        return IPCAFitResult(53, -0.0, r.keep, result_ids())

    @staticmethod
    def partial(r: IPCAPartialRequest) raises -> IPCAFitResult:
        var variant = r.source - 79
        require(variant >= 0 and variant < 4 and r.keep == ((variant & 1) != 0)
            and r.resident == ((variant & 2) != 0), "IPCA partial resident/keep flags")
        require(r.rows == 137 and r.features == 139 and r.requested == 149 and r.seen == 151
            and r.previous_count == 157 and r.previous_features == 163,
            "IPCA partial shape/history slots")
        require(r.previous.components == 83 and r.previous.singular == 89 and r.previous.mean == 97
            and r.previous.population_variance == 101, "IPCA previous native array slots")
        check_outputs(r.outputs)
        check_ids(r.state, input_ids())
        return IPCAFitResult(167, 0.25, r.keep, result_ids())

    @staticmethod
    def query(r: IPCAQueryRequest) raises:
        var variant = r.source - 211
        require(variant >= 0 and variant < 8 and r.inverse == ((variant & 1) != 0)
            and r.whiten == ((variant & 2) != 0) and r.resident == ((variant & 4) != 0),
            "IPCA query inverse/whitening/resident flags")
        require(r.rows == 223 and r.features == 227 and r.count == 229 and r.mean == 233
            and r.components == 239 and r.variance == 241 and r.dst == 251,
            "IPCA query dimensions and address slots")
        check_ids(r.state, input_ids())

    @staticmethod
    def prepare(address: Int, count: Int) raises -> Int:
        require((address == 263 or address == 269) and count == 271, "IPCA prepare address/count slots")
        return 0 if address == 263 else 277

    @staticmethod
    def free(handle: Int) raises:
        require(handle == 0 or handle == 277, "IPCA free preserves valid zero or positive handle")


def ipca_wire_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return ipca_native_dispatch[IPCAWireOracle](operation, aa, na, pp, np, ff, nf, result, context, sink)


def fit_request(keep: Bool) -> IPCAFitRequest:
    return IPCAFitRequest(12 if keep else 11, 37, 41, 43, 47, outputs(), keep)


def partial_request(variant: Int) -> IPCAPartialRequest:
    return IPCAPartialRequest(79 + variant, 137, 139, 149, 151, 157, 163,
        IPCAPrevious(83, 89, 97, 101), outputs(), (variant & 2) != 0, input_ids(), (variant & 1) != 0)


def query_request(variant: Int) -> IPCAQueryRequest:
    return IPCAQueryRequest(211 + variant, 223, 227, 229, 233, 239, 241, 251,
        (variant & 1) != 0, (variant & 2) != 0, (variant & 4) != 0, input_ids())


def missing_result(mut output: NativeCallOutput) raises:
    var status = ipca_wire_entry(IPCA_MODE, 0, 0, 0, 0, 0, 0, 0,
        Int(MutPointer(to=output.error)), native_descriptor_error)
    output.error.append(0)
    require(status != 0 and String(unsafe_from_utf8_ptr=output.error.unsafe_ptr())
        == "IncrementalPCA native ABI: missing result span", "IPCA missing-result refusal")
    _ = output


def fixture(unit: Int) raises:
    comptime assert IPCA_NATIVE_COUNT == 8 and IPCA_MODE == 0 and IPCA_VENDOR == 1
    comptime assert IPCA_CAPABILITIES == 2 and IPCA_FIT == 3 and IPCA_PARTIAL == 4
    comptime assert IPCA_QUERY == 5 and IPCA_PREPARE == 6 and IPCA_FREE == 7
    var driver = IPCANativeDriver(ipca_wire_entry)
    if unit == 0:
        require(driver.metadata(0) == 1 and driver.metadata(1) == 4 and driver.metadata(2) == 15,
                "IPCA native metadata operation/result slots")
    elif unit < 3:
        var result = driver.fit(fit_request(unit == 2))
        require(result.count == 53 and result.kept == (unit == 2)
            and bitcast[DType.uint64](result.noise) == UInt64(0x8000000000000000),
            "IPCA fit result preserves count, kept and Float64 signed zero")
        check_ids(result.state, result_ids())
    elif unit < 7:
        var result = driver.partial(partial_request(unit - 3))
        require(result.count == 167 and result.noise == 0.25 and result.kept == (((unit - 3) & 1) != 0),
                "IPCA partial result preserves all scalar slots")
        check_ids(result.state, result_ids())
    elif unit < 15:
        driver.query(query_request(unit - 7))
    elif unit == 15 or unit == 16:
        require(driver.prepare(263 if unit == 15 else 269, 271) == (0 if unit == 15 else 277),
                "IPCA prepare transports valid-zero/positive native handles")
    elif unit == 17 or unit == 18:
        driver.free(0 if unit == 17 else 277)
    elif unit == 19:
        var original = IPCAStateIds()
        var copied = original.copy()
        copied.values[0] = 0
        for index in range(5):
            require(original.values[index] == -1, "IPCA empty identities default to minus one and copies do not alias")
        require(copied.values[0] == 0, "IPCA zero identity is preserved")
    elif unit == 20:
        var original = fit_request(False)
        var copied = original.copy()
        copied.outputs.components = 501
        copied.keep = True
        require(original.outputs.components == 13 and not original.keep,
                "IPCA copied nested output/request metadata does not alias")
    elif unit == 21:
        var original = query_request(0)
        var copied = original.copy()
        copied.state.values[0] = 0
        copied.inverse = True
        require(original.state.values[0] == 107 and not original.inverse,
                "IPCA copied nested state/query metadata does not alias")
    elif unit == 33:
        var owners = List[NativeCallOutput]()
        owners.append(NativeCallOutput())
        try:
            missing_result(owners[0])
        finally:
            _ = owners^
    else:
        var operation = 8
        var addresses = List[Int]()
        var integers = List[Int]()
        var scalars = List[Float64]()
        var words = List[Int](length=7, fill=-1)
        var noise = List[Float64](length=1, fill=-1)
        var expected = String("IncrementalPCA native ABI: unknown operation")
        if unit == 23:
            operation = IPCA_MODE
            addresses.append(1)
            expected = "IncrementalPCA native ABI: metadata takes no arguments"
        elif unit == 24:
            operation = IPCA_FIT
            expected = "IncrementalPCA native ABI: fit requires 9 addresses and 5 integers"
        elif unit == 25:
            operation = IPCA_PARTIAL
            expected = "IncrementalPCA native ABI: partial requires 13 addresses and 13 integers"
        elif unit == 26:
            operation = IPCA_QUERY
            expected = "IncrementalPCA native ABI: query requires 5 addresses and 11 integers"
        elif unit == 27:
            operation = IPCA_PREPARE
            expected = "IncrementalPCA native ABI: prepare requires an address and count"
        elif unit == 28:
            operation = IPCA_FREE
            expected = "IncrementalPCA native ABI: free requires one handle"
        elif unit >= 29:
            var partial = unit >= 31
            operation = IPCA_PARTIAL if partial else IPCA_FIT
            addresses = List[Int](length=13 if partial else 9, fill=0)
            integers = List[Int](length=13 if partial else 5, fill=0)
            var word_slot = 11 if partial else 7
            addresses[word_slot] = 0 if unit == 29 or unit == 31 else Int(words.unsafe_ptr())
            addresses[word_slot + 1] = 0 if unit == 30 or unit == 32 else Int(noise.unsafe_ptr())
            expected = "IncrementalPCA native ABI: missing partial output spans" if partial else "IncrementalPCA native ABI: missing fit output spans"
        var refused = False
        try:
            _ = native_descriptor_call(ipca_wire_entry, operation, addresses, integers, scalars)
        except error:
            refused = String(error) == expected
        finally:
            _ = words^
            _ = noise^
            _ = addresses^
            _ = integers^
            _ = scalars^
        require(refused, "IPCA malformed native wire refusal")


def run_checks() raises:
    var failures = List[Int](length=IPCA_CONTRACT_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL ipca_contract fixture", unit, String(error))
    host_parallelize(task, IPCA_CONTRACT_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "ipca_contract failed " + String(failed) + " fixtures")
    print("PASS ipca_contract:", IPCA_CONTRACT_FIXTURES, "native typed request/result/wire fixtures")
