# SPDX-License-Identifier: Apache-2.0
"""Actual optimizer device descriptor and native storage-plan contracts."""
from core.native_descriptor_abi import NativeDescriptorCall
from training.optimizer_native_api import OptimizerNativeDriver, OPT_NATIVE_COUNT
from training.optimizer_device_storage_plan import (
    optimizer_device_count, optimizer_device_reshape, optimizer_device_output,
    optimizer_device_upload, optimizer_device_transfer_span,
)
from bindings.tensor_view import TensorView
from checks.native_tests.optimizer_descriptor_cases import ErrorProbe, capture_error, require
from core.host_parallel import host_parallelize

comptime DEVICE_MALFORMED = 20
comptime DEVICE_PLAN = 28
comptime DEVICE_COMMON = DEVICE_MALFORMED + DEVICE_PLAN + 1
comptime DEVICE_UNAVAILABLE = "RuntimeError: this build has no training device arrays (identical GPU builds only)"
comptime WORD_GUARD = UInt32(0xA19B73C5)


def malformed(entry: NativeDescriptorCall, fixture_id: Int) raises:
    var probes = List[ErrorProbe]()
    probes.append(ErrorProbe())
    var addresses: List[Int] = [0]
    var integers: List[Int] = [0, 0, 0, 0]
    var floats: List[Float64] = [0]
    var operation = 11 + fixture_id // 5
    var variant = fixture_id % 5
    var na = 0
    var np = 0 if operation == 11 else (1 if operation == 12 else 3)
    var nf = 0
    var pp = Int(integers.unsafe_ptr())
    var result = Int(MutPointer(to=probes[0].result))
    var expected = String("native optimizer: request span counts differ from the operation")
    if variant == 0:
        na = 1
    elif variant == 1:
        np += 1
    elif variant == 2:
        nf = 1
    elif variant == 3:
        result = 0
        expected = "native optimizer: unknown operation or missing result storage"
    elif operation == 11:
        operation = OPT_NATIVE_COUNT
        expected = "native optimizer: unknown operation or missing result storage"
    else:
        pp = 0
        expected = "native optimizer: missing request span"
    var status = entry(operation, Int(addresses.unsafe_ptr()), na, pp, np,
        Int(floats.unsafe_ptr()), nf, result, Int(probes.unsafe_ptr()), capture_error)
    require(status != 0, "malformed device descriptor succeeded")
    require(probes[0].result == 72391, "malformed device descriptor wrote result")
    require(probes[0].calls == 1, "device descriptor error sink count differs")
    require(len(probes[0].bytes) == len(expected.as_bytes()), "device descriptor error was truncated")
    probes[0].bytes.append(0)
    var copied = String(unsafe_from_utf8_ptr=probes[0].bytes.unsafe_ptr())
    require(copied == expected, "device descriptor error changed: " + copied)
    _ = probes
    _ = addresses
    _ = integers
    _ = floats


def plan(fixture_id: Int) raises:
    var storage: List[UInt32] = [WORD_GUARD, 0x80000000, 0x7FC01234, 0x3F800000, 0, WORD_GUARD]
    var source = TensorView()
    source.addr = Int(storage.unsafe_ptr()) + 4
    source.nbytes = 16
    source.itemsize = 4
    source.fmt = "f"
    source.shape = [4]
    source.strides = [4]
    source.has_strides = True
    var expected = String("")
    var wanted = 4
    if fixture_id == 3 or fixture_id == 17:
        expected = "ValueError: training device shape dimensions must be nonnegative"
    elif fixture_id == 4:
        expected = "OverflowError: training device shape exceeds native indexing"
    elif fixture_id == 6:
        expected = "ValueError: training device reshape does not hold the array's floats"
    elif fixture_id == 8:
        expected = "ValueError: training device download exceeds its output shape"
    elif fixture_id == 11 or fixture_id == 12:
        expected = "TypeError: mojolearn.train_to_device: a must be float32"
    elif fixture_id == 13:
        expected = "ValueError: mojolearn.train_to_device: a must be nonempty"
    elif fixture_id >= 14 and fixture_id <= 16:
        expected = "ValueError: training device upload requires complete float32 storage"
    elif fixture_id == 22 or fixture_id == 23:
        expected = "ValueError: training device transfer count is out of range"
    elif fixture_id >= 24 and fixture_id <= 26:
        expected = "ValueError: training device transfer requires complete native storage"
    var refused = False
    try:
        if fixture_id == 0:
            require(optimizer_device_count([]) == 1, "device scalar count")
        elif fixture_id == 1:
            require(optimizer_device_count([2, 3]) == 6, "device matrix count")
        elif fixture_id == 2:
            require(optimizer_device_count([3, 0, 4]) == 0, "device empty-axis count")
        elif fixture_id == 3:
            _ = optimizer_device_count([2, -1])
        elif fixture_id == 4:
            _ = optimizer_device_count([9223372036854775807, 2])
        elif fixture_id == 5 or fixture_id == 6:
            optimizer_device_reshape([2, 3], 6 if fixture_id == 5 else 5)
        elif fixture_id == 7 or fixture_id == 8:
            optimizer_device_output([2, 3], 3 if fixture_id == 7 else 7)
        elif fixture_id < 20:
            if fixture_id == 10:
                source.addr += 12
                source.strides = [-4]
                source.c_contig = False
                source.readonly = True  # Gather preparation accepts a borrowed readonly source.
            elif fixture_id == 11:
                source.fmt = "i"
            elif fixture_id == 12:
                source.itemsize = 8
            elif fixture_id == 13:
                source.shape = [0]
                source.nbytes = 0
            elif fixture_id == 14:
                source.nbytes = 12
            elif fixture_id == 15:
                source.addr = 0
            elif fixture_id == 16:
                source.shape = [2305843009213693952]
                source.nbytes = 0
            elif fixture_id == 17:
                source.shape = [-1]
            elif fixture_id == 18:
                source.fmt = "<f"
            elif fixture_id == 19:
                source.shape = []
                source.strides = []
                source.nbytes = 4
                wanted = 1
            require(optimizer_device_upload(source) == wanted, "device upload plan count")
        else:
            var address = source.addr
            var count = 4
            if fixture_id == 20:
                address = 0
                count = 0
            elif fixture_id == 21:
                address = 1
                count = 1  # Pure span admission does not dereference this address.
            elif fixture_id == 22:
                count = -1
            elif fixture_id == 23:
                count = 2305843009213693952
            elif fixture_id == 24:
                address = 0
            elif fixture_id == 25:
                address = -1
            elif fixture_id == 26:
                address = 9223372036854775807
            else:
                address = 9223372036854775804
                count = 1  # Last byte is exactly representable.
            optimizer_device_transfer_span(address, count)
    except error:
        refused = True
        require(expected != "" and String(error) == expected, "device plan refusal differs: " + String(error))
    require(refused == (expected != ""), "device plan refusal missing")
    require(storage[0] == WORD_GUARD and storage[5] == WORD_GUARD and storage[1] == 0x80000000
        and storage[2] == 0x7FC01234 and storage[3] == 0x3F800000 and storage[4] == 0,
        "metadata plan changed source storage")
    _ = storage


def common(driver: OptimizerNativeDriver, fixture_id: Int) raises:
    if fixture_id < DEVICE_MALFORMED:
        malformed(driver.entry, fixture_id)
    elif fixture_id < DEVICE_MALFORMED + DEVICE_PLAN:
        plan(fixture_id - DEVICE_MALFORMED)
    else:
        require(driver.device_available() == (not driver.host), "device capability disagrees with target")


def pattern(index: Int) -> UInt32:
    var slot = index % 8
    if slot == 0:
        return 0
    if slot == 1:
        return 0x80000000
    if slot == 2:
        return 0x7FC00042
    if slot == 3:
        return 0xFFC04321
    if slot == 4:
        return 0x7F800000
    if slot == 5:
        return 0xFF800000
    if slot == 6:
        return UInt32(0x3F000000) | UInt32(index & 0x007FFFFF)
    return 0x80000001


struct RawWords(Movable):
    var words: List[UInt32]
    var count: Int

    def __init__(out self, count: Int, values: Bool = False):
        self.count = count
        self.words = List[UInt32](length=count + 2, fill=WORD_GUARD)
        if values:
            var p = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(self.words.unsafe_ptr()))
            def cell(index: Int) {imm p}:
                p[index + 1] = pattern(index)
            host_parallelize(cell, count)
            _ = self.words

    def address(self) -> Int:
        return Int(self.words.unsafe_ptr()) + 4

    def expect(self, prefix: Int) raises:
        var bad = List[Int](length=self.count + 2, fill=0)
        var bp = bad.unsafe_ptr()
        var p = self.words.unsafe_ptr()
        var count = self.count
        def cell(index: Int) {imm bp, imm p, imm count, imm prefix}:
            var expected = pattern(index - 1) if index > 0 and index <= prefix else WORD_GUARD
            if p[index] != expected:
                bp.unsafe_store(index, 1)
        host_parallelize(cell, count + 2)
        var failed = 0
        for value in bad:
            failed += value
        require(failed == 0, "device copy/guards changed " + String(failed) + " raw words")
        _ = self


def transfer_refusal(driver: OptimizerNativeDriver, upload: Bool, handle: Int, address: Int,
    count: Int, expected: String) raises:
    var refused = False
    try:
        _ = driver.upload_device(handle, address, count) if upload else driver.download_device(handle, address, count)
    except error:
        refused = True
        require(String(error) == expected, "native device transfer refusal differs: " + String(error))
    require(refused, "invalid native device transfer succeeded")
