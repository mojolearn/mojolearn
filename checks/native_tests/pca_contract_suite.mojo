# SPDX-License-Identifier: Apache-2.0
"""PCA typed request/result wire, copy, fold state and refusal contracts.

Fixed distinct field values expose slot swaps. Numerical kernels are not
reimplemented here; their execution lives in the PCA CPU/GPU suites.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.native_call_output import NativeCallOutput
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_call, native_descriptor_error
from core.pca_native_checks import pca_whiten_dimensions, pca_whiten_device_pointer
from core.pca_native_api import (
    PCA_MODE, PCA_VENDOR, PCA_CAPABILITIES, PCA_FIT, PCA_QUERY, PCANativeKernels,
    PCANativeDriver, PCAFoldStore, PCAFitRequest, PCAFitResult, PCAQueryRequest,
    PCAQueryResult, pca_native_dispatch, pca_vendor_name, pca_vendor_code,
)

comptime PCA_CONTRACT_FIXTURES = 32


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


struct PCAWireOracle(PCANativeKernels):
    @staticmethod
    def mode() -> Int:
        return 1

    @staticmethod
    def vendor() -> Int:
        return 3

    @staticmethod
    def capabilities() -> Int:
        return 47

    @staticmethod
    def fit(r: PCAFitRequest) raises -> PCAFitResult:
        require(r.source == 11 and r.components == 13 and r.mean == 17 and r.variance == 19
                and r.ratio == 23 and r.singular == 29, "PCA fit address slots")
        require(r.rows == 37 and r.features == 41 and r.count == 43 and r.keep_mode == 2,
                "PCA fit dimensions/selection mode")
        require(r.oversamples == 47 and r.iterations == 53 and r.seed == 59 and r.fraction == 0.625,
                "PCA fit options/fraction")
        require(r.auto_iterations == (not r.dense) and r.test_finite == (not r.whiten),
                "PCA independent Boolean slots")
        require(r.projection == (31 if r.whiten else 0), "PCA optional projection")
        require(r.fold.active() == r.randomized, "PCA fit fold activity")
        if r.fold.active():
            require(r.fold.base == 61 and r.fold.rows == 67 and r.fold.indices == 71, "PCA fit fold slots")
        return PCAFitResult(73, -0.0, -7, Int(r.dense) + 2 * Int(r.whiten) + 4 * Int(r.randomized))

    @staticmethod
    def query(r: PCAQueryRequest) raises -> PCAQueryResult:
        require(r.source == 11 and r.mean == 13 and r.components == 17 and r.singular == 19
                and r.dst == 23, "PCA query address slots")
        require(r.rows == 29 and r.features == 31 and r.count == 37 and r.fit_rows == 41,
                "PCA query shape/fitted-row slots")
        if r.fold.active():
            require(r.fold.base == 43 and r.fold.rows == 47 and r.fold.indices == 53, "PCA query fold slots")
        return PCAQueryResult(-5, Int(r.inverse) + 2 * Int(r.whiten) + 4 * Int(r.fold.active()))


def pca_wire_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return pca_native_dispatch[PCAWireOracle](operation, aa, na, pp, np, ff, nf, result, context, sink)


def fit_request(variant: Int) -> PCAFitRequest:
    var dense = (variant & 1) != 0
    var whiten = (variant & 2) != 0
    var randomized = (variant & 4) != 0
    return PCAFitRequest(11, 13, 17, 19, 23, 29, 31 if whiten else 0,
        37, 41, 43, dense, 2, whiten, randomized, 47, 53, not dense, 59,
        not whiten, 0.625, PCAFoldStore(61, 67, 71) if randomized else PCAFoldStore())


def shape_request() -> PCAQueryRequest:
    return PCAQueryRequest(4096, 8192, 12288, 16384, 20480, 3, 5, 2, 7, False, True, PCAFoldStore())


def missing_result(mut output: NativeCallOutput) raises:
    var status = pca_wire_entry(PCA_MODE, 0, 0, 0, 0, 0, 0, 0,
        Int(MutPointer(to=output.error)), native_descriptor_error)
    output.error.append(0)
    require(status != 0 and String(unsafe_from_utf8_ptr=output.error.unsafe_ptr())
            == "PCA native ABI: missing result span", "PCA missing-result refusal")


def fixture(unit: Int) raises:
    comptime assert PCA_MODE == 0 and PCA_VENDOR == 1 and PCA_CAPABILITIES == 2 and PCA_FIT == 3 and PCA_QUERY == 4
    var driver = PCANativeDriver(pca_wire_entry)
    if unit == 0:
        require(driver.metadata(PCA_MODE) == 1 and driver.metadata(PCA_VENDOR) == 3
                and driver.metadata(PCA_CAPABILITIES) == 47, "PCA independent metadata results")
        var names: List[String] = ["none", "cpu", "metal", "cuda", "hip"]
        for index in range(5):  # Fixed ABI names, not a dataset loop.
            require(pca_vendor_name(index) == names[index] and pca_vendor_code(names[index]) == index,
                    "PCA vendor code/name mapping")
        require(pca_vendor_name(-1) == "none" and pca_vendor_name(5) == "none"
                and pca_vendor_code("other") == 0, "PCA unknown vendor sentinel")
    elif unit < 9:
        var variant = unit - 1
        var answer = driver.fit(fit_request(variant))
        require(answer.keep == 73 and answer.finite == -7 and answer.in_range == variant,
                "PCA fit result slots")
        require(bitcast[DType.uint64](answer.noise) == UInt64(0x8000000000000000),
                "PCA Float64 noise result preserves signed zero")
    elif unit < 17:
        var variant = unit - 9
        var fold = PCAFoldStore(43, 47, 53) if (variant & 4) != 0 else PCAFoldStore()
        var answer = driver.query(PCAQueryRequest(11, 13, 17, 19, 23, 29, 31, 37, 41,
            (variant & 1) != 0, (variant & 2) != 0, fold))
        require(answer.finite == -5 and answer.in_range == variant, "PCA query result slots")
    elif unit < 23:
        var operation = 5
        var addresses = List[Int]()
        var integers = List[Int]()
        var scalars = List[Float64]()
        var expected = String("PCA native ABI: unknown operation")
        if unit == 18:
            operation = PCA_FIT
            expected = "PCA native ABI: fit requires 8 addresses, 15 integers and 1 float"
        elif unit == 19:
            operation = PCA_QUERY
            expected = "PCA native ABI: query requires 6 addresses and 9 integers"
        elif unit == 20:
            operation = PCA_MODE
            addresses.append(11)
            expected = "PCA native ABI: metadata takes no arguments"
        elif unit == 21:
            operation = PCA_FIT
            addresses = List[Int](length=8, fill=0)
            integers = List[Int](length=15, fill=0)
            scalars.append(0.0)
            expected = "PCA native ABI: missing fit output span"
        elif unit == 22:
            operation = PCA_QUERY
            addresses = List[Int](length=6, fill=0)
            integers = List[Int](length=9, fill=0)
            expected = "PCA native ABI: missing query output span"
        var refused = False
        try:
            _ = native_descriptor_call(pca_wire_entry, operation, addresses, integers, scalars)
        except error:
            refused = String(error) == expected
        require(refused, "PCA malformed wire request refusal")
    elif unit == 23:
        var owners = List[NativeCallOutput]()
        owners.append(NativeCallOutput())
        missing_result(owners[0])
        _ = owners^
    elif unit < 30:
        var request = shape_request()
        var expected = String("PCA whitening requires positive rows/features/components and fit_rows>=2")
        if unit == 24:
            request.rows = 0
        elif unit == 25:
            request.fit_rows = 1
        elif unit == 26:
            request.rows = 2147483648
            expected = "PCA whitening dimensions exceed Int32"
        elif unit == 27:
            request.rows = 2147483647
            expected = "PCA whitening matrix cell count exceeds Int32"
        elif unit == 28:
            expected = "PCA whitening requires positive aligned FP32 pointers"
        else:
            expected = "PCA whitening pointer span overflows signed Int"
        var refused = False
        try:
            if unit < 28:
                pca_whiten_dimensions(request)
            else:
                _ = pca_whiten_device_pointer(5 if unit == 28 else 9223372036854775804, 2)
        except error:
            refused = String(error) == expected
        require(refused, "PCA native whitening admission refusal")
    elif unit == 30:
        require(not PCAFoldStore().active() and not PCAFoldStore(-1, 17, 3).active()
                and not PCAFoldStore(2, 17, -1).active() and PCAFoldStore(2, 17, 3).active(),
                "PCA fold activation requires both native identities")
        pca_whiten_dimensions(shape_request())
        require(Int(pca_whiten_device_pointer(4096, 15)) == 4096, "PCA admitted pointer address")
    else:
        var original = fit_request(7)
        var changed = original.copy()
        changed.source = 101
        changed.fold.indices = 103
        changed.whiten = False
        require(original.source == 11 and original.fold.indices == 71 and original.whiten,
                "PCA typed request copy must not share mutable metadata")
        var answer = driver.fit(original)
        require(answer.in_range == 7, "PCA original request survives copied-record mutation")


def run_checks() raises:
    var failures = List[Int](length=PCA_CONTRACT_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL pca_contract fixture", unit, String(error))
    host_parallelize(task, PCA_CONTRACT_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "pca_contract failed " + String(failed) + " fixtures")
    print("PASS pca_contract:", PCA_CONTRACT_FIXTURES, "native wire/state fixtures")
