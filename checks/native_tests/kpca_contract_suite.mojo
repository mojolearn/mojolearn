# SPDX-License-Identifier: Apache-2.0
"""KernelPCA typed eigen envelopes and owning-image metadata contracts.

The prescribed wire oracle performs no kernel centering or eigendecomposition.
Its fake addresses never reach numerical code. Error outputs are heap-owned.
"""
from core.host_parallel import host_parallelize
from core.native_call_output import NativeCallOutput
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_call, native_descriptor_error
from core.kpca_native_api import (
    KPCAEigenRequest, KPCANativeKernels, KPCANativeDriver, kpca_native_v1,
    KPCA_NATIVE_COUNT, KPCA_MODE, KPCA_VENDOR, KPCA_CAPABILITIES, KPCA_EIGEN,
)
from core.kpca_native_info import KPCANeighborInfo

comptime KPCA_CONTRACT_FIXTURES = 20


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


struct KPCAWireOracle(KPCANativeKernels):
    @staticmethod
    def mode() -> Int:
        return 1

    @staticmethod
    def vendor() -> Int:
        return 3

    @staticmethod
    def capabilities() -> Int:
        return 1

    @staticmethod
    def eigen(r: KPCAEigenRequest) raises -> Int:
        var variant = r.source - 11
        require(variant >= 0 and variant < 8 and r.resident == ((variant & 1) != 0)
            and r.lanczos == ((variant & 2) != 0) and r.remove_zero == ((variant & 4) != 0),
            "KPCA eigen source/flag slots")
        require(r.centered == 23 and r.values == 29 and r.vectors == 31 and r.columns == 37
            and r.total == 41 and r.rows == 43 and r.features == 47 and r.requested == 53
            and r.gamma == Float64(0.0625), "KPCA eigen field slots")
        return 59 + variant


def request(variant: Int) -> KPCAEigenRequest:
    return KPCAEigenRequest(23, 11 + variant, 29, 31, 37, 41, (variant & 1) != 0,
        43, 47, 53, (variant & 2) != 0, (variant & 4) != 0, 0.0625)


def missing_result(mut output: NativeCallOutput) raises:
    var status = kpca_native_v1[KPCAWireOracle](KPCA_MODE, 0, 0, 0, 0, 0, 0, 0,
        Int(MutPointer(to=output.error)), native_descriptor_error)
    output.error.append(0)
    require(status != 0 and String(unsafe_from_utf8_ptr=output.error.unsafe_ptr())
        == "KernelPCA native ABI: missing result span", "KPCA missing-result refusal")
    _ = output


def fixture(unit: Int) raises:
    comptime assert KPCA_NATIVE_COUNT == 4 and KPCA_MODE == 0 and KPCA_VENDOR == 1
    comptime assert KPCA_CAPABILITIES == 2 and KPCA_EIGEN == 3
    var driver = KPCANativeDriver(kpca_native_v1[KPCAWireOracle])
    if unit == 0:
        require(driver.metadata(0) == 1 and driver.metadata(1) == 3 and driver.metadata(2) == 1,
                "KPCA native metadata slots")
    elif unit <= 8:
        require(driver.eigen(request(unit - 1)) == 59 + unit - 1,
                "KPCA distinct addresses/dimensions/scalar and Boolean combinations")
    elif unit == 9:
        var original = request(0)
        var copied = original.copy()
        copied.centered = 101
        copied.gamma = 0.5
        copied.resident = True
        require(original.centered == 23 and original.gamma == 0.0625 and not original.resident,
                "KPCA typed request copies do not alias")
    elif unit == 17:
        var invalid = request(0)
        invalid.source = 999
        var refused = False
        try:
            _ = driver.eigen(invalid)
        except error:
            refused = String(error) == "KPCA eigen source/flag slots"
        require(refused, "KPCA native provider exception crosses descriptor unchanged")
    elif unit == 18:
        var neighbor = KPCANativeDriver(kpca_native_v1[KPCANeighborInfo[False, False]])
        require(neighbor.metadata(0) == 1 and neighbor.metadata(1) == 1 and neighbor.metadata(2) == 2,
                "KPCA host neighbor image advertises only its owned capability")
        var refused = False
        try:
            _ = neighbor.eigen(request(0))
        except error:
            refused = String(error) == "KernelPCA neighbor provider does not own the eigensolver"
        require(refused, "KPCA metadata image cannot execute another image's eigen operation")
    elif unit == 19:
        var owners = List[NativeCallOutput]()
        owners.append(NativeCallOutput())
        try:
            missing_result(owners[0])
        finally:
            _ = owners^
    else:
        var operation = 4
        var addresses = List[Int]()
        var integers = List[Int]()
        var scalars = List[Float64]()
        var expected = String("KernelPCA native ABI: unknown operation")
        if unit == 11:
            operation = KPCA_MODE
            addresses.append(1)
            expected = "KernelPCA native ABI: metadata takes no arguments"
        elif unit >= 12:
            operation = KPCA_EIGEN
            expected = "KernelPCA native ABI: eigen requires 6 addresses, 6 integers and one scalar"
            if unit >= 13:
                addresses = List[Int](length=5 if unit == 13 else 6, fill=0)
                integers = List[Int](length=5 if unit == 14 else 6, fill=0)
                scalars = List[Float64](length=0 if unit == 15 else (2 if unit == 16 else 1), fill=0)
        var refused = False
        try:
            _ = native_descriptor_call(kpca_native_v1[KPCAWireOracle], operation, addresses, integers, scalars)
        except error:
            refused = String(error) == expected
        finally:
            _ = addresses^
            _ = integers^
            _ = scalars^
        require(refused, "KPCA malformed native wire refusal")


def run_checks() raises:
    var failures = List[Int](length=KPCA_CONTRACT_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL kpca_contract fixture", unit, String(error))
    host_parallelize(task, KPCA_CONTRACT_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "kpca_contract failed " + String(failed) + " fixtures")
    print("PASS kpca_contract:", KPCA_CONTRACT_FIXTURES, "native eigen wire/owning-image fixtures")
