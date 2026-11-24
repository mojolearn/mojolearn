# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""THE ORDERED FOLD OF k SHARDS IN ONE CALL (lane vdf-linalg, 2026-11-24): cross_vendor's `fold_pair` /
`ordered_fold` and the coordinator's fold, on the base image's `PB_ORDERED_FOLD_F32`
(core/parallel_entry_table.mojo): the device kernel `device_ordered_fold_f32` (core/hotpath_device.mojo:
element-parallel, the shard order fixed per element, dst resident) and the host column `ordered_fold_f32`
(core/hotpath_impl_rows.mojo: the elements in tasks across the host's cores).

Every case compares the table call's bytes with an independent serial reference written here: for each element,
acc = dst, then for every shard in order acc = canonical_nan(ftz(ftz(acc) + ftz(shard))). It is the recurrence the
deleted per-shard pair fold computed, so the bits are unchanged. On a GPU build the device column's bytes are also
compared with the host column's `ordered_fold_f32` run in the same process: device bytes == host bytes == reference.

Cases:
  0  one shard of 3 words (the pair form): 1 + 0.5, a subnormal + 0 (FTZ to a signed zero), -0.25 + 0
  1  1 + 2^-24 + 2^-24 is 1.0 (left fold, one rounding per add); 2^-24 + 2^-24 + 1 is 1 + 2^-23 (order matters)
  2  a NaN shard word becomes the canonical word 0x7fc00000; -0 + -0 stays -0; infinities keep their sign
  3  70,001 elements x 9 shards of SplitMix64 words (several host tasks, many GPU blocks, an odd tail)
  4  5 shards in the reversed order differ from case 3's order on at least one element (the check can fail)

FAIL-PROOF EDITS (each must make a named case fail): fold the shards in reverse inside the kernel or the host task
(cases 1, 3); drop the inner `ftz` of the accumulator (case 0); drop `canonicalize_nan` (case 2); fold only the
first group of a split launch (case 3, if HPD_FOLD_STAGE is lowered below 9 x 70,001).
"""
from std.memory import bitcast
from core.entry_table import EntryRef
from core.parallel_entry_table import PARALLEL_TABLE_BASE, ParallelEntryKey, parallel_ref, parallel_entry_vendor, parallel_ordered_fold_f32
from core.hotpath_impl_rows import ordered_fold_f32
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from bindings.parallel_base_entry_host import publish_parallel_base_host
from bindings.parallel_base_entry_device import publish_parallel_base_device
from sequence.schedule import splitmix64
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, check_cases_serial, require, CPU_COLUMN_BUILD

comptime ORDERED_FOLD_CASES = 5


def _word(v: Float32) -> UInt32:
    return bitcast[DType.uint32](v)


def _f(bits: UInt32) -> Float32:
    return bitcast[DType.float32](bits)


def _ftz(v: Float32) -> Float32:
    var b = _word(v)
    if (b & UInt32(0x7F800000)) == 0 and (b & UInt32(0x007FFFFF)) != 0:
        return _f(b & UInt32(0x80000000))
    return v


def _canonical(v: Float32) -> Float32:
    if v != v:
        return _f(UInt32(0x7FC00000))
    return v


def _reference(dst: List[Float32], shards: List[List[Float32]]) -> List[Float32]:
    """The serial reference recurrence (independent of the code under test)."""
    var out = dst.copy()
    for i in range(len(out)):  # driver-loop: the reference's elements (a check, not product code)
        var acc = out[i]
        for j in range(len(shards)):  # driver-loop: the shards in their logical order
            acc = _canonical(_ftz(_ftz(acc) + _ftz(shards[j][i])))
        out[i] = acc
    return out^


def _fold(entry: EntryRef, dst: List[Float32], shards: List[List[Float32]]) raises -> List[Float32]:
    """One `PB_ORDERED_FOLD_F32` call of the table under test."""
    var out = dst.copy()
    var table = List[Int64](capacity=len(shards))
    for j in range(len(shards)):  # driver-loop: the shard addresses in logical order
        table.append(Int64(Int(shards[j].unsafe_ptr())))
    _ = parallel_ordered_fold_f32(entry, Int(out.unsafe_ptr()), Int(table.unsafe_ptr()), len(shards), len(out))
    _ = table^
    _ = shards
    return out^


def _host(dst: List[Float32], shards: List[List[Float32]]) raises -> List[Float32]:
    """The host column called directly (the CPU column of a GPU build)."""
    var out = dst.copy()
    var table = List[Int64](capacity=len(shards))
    for j in range(len(shards)):  # driver-loop: the shard addresses in logical order
        table.append(Int64(Int(shards[j].unsafe_ptr())))
    _ = ordered_fold_f32(Int(out.unsafe_ptr()), Int(table.unsafe_ptr()), len(shards), len(out))
    _ = table^
    _ = shards
    return out^


def _same(a: List[Float32], b: List[Float32], what: String) raises:
    require(len(a) == len(b), what + ": lengths differ")
    for i in range(len(a)):  # driver-loop: byte comparison of the check's outputs
        if _word(a[i]) != _word(b[i]):
            raise Error(what + ": element " + String(i) + " is " + String(_word(a[i])) + ", want " + String(_word(b[i])))


def _draws(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32](capacity=n)
    var state = seed
    for _ in range(n):  # driver-loop: fixture values
        state = splitmix64(state)
        # finite values of moderate magnitude, signs mixed, some subnormal-range sums
        var bits = UInt32(state & UInt64(0x807FFFFF)) | UInt32(0x3C000000 + ((state >> 40) & UInt64(0x03000000)))
        out.append(_f(bits))
    return out^


def _case_inputs(case: Int, mut dst: List[Float32], mut shards: List[List[Float32]]):
    if case == 0:
        dst = [Float32(1.0), _f(UInt32(0x80000001)), Float32(-0.25)]
        shards = [[Float32(0.5), Float32(0.0), Float32(0.0)]]
    elif case == 1:
        var t = _f(UInt32(0x33800000))  # 2^-24
        dst = [Float32(1.0), t]
        shards = [[t, t], [t, Float32(1.0)]]
    elif case == 2:
        dst = [Float32(1.0), _f(UInt32(0x80000000)), _f(UInt32(0x7F800000))]
        shards = [[_f(UInt32(0x7FA00001)), _f(UInt32(0x80000000)), Float32(1.0)]]
    else:
        var n = 70001
        dst = _draws(n, UInt64(11))
        shards = List[List[Float32]]()
        var k = 9 if case == 3 else 5
        for j in range(k):  # driver-loop: fixture shards
            shards.append(_draws(n, UInt64(101 + j)))


def run_case(entry: EntryRef, case: Int, host_too: Bool) raises:
    var dst = List[Float32]()
    var shards = List[List[Float32]]()
    _case_inputs(case, dst, shards)
    var want = _reference(dst, shards)
    var got = _fold(entry, dst, shards)
    _same(got, want, "ordered fold case " + String(case))
    if host_too:
        _same(_host(dst, shards), got, "host column vs device column, case " + String(case))
    if case == 0:
        require(_word(got[1]) == UInt32(0x80000000), "a subnormal accumulator is flushed to -0")
    if case == 1:
        require(_word(got[0]) == UInt32(0x3F800000), "1 + 2^-24 + 2^-24 is not 1.0")
        require(_word(got[1]) == UInt32(0x3F800001), "2^-24 + 2^-24 + 1 is not 1 + 2^-23")
    if case == 2:
        require(_word(got[0]) == UInt32(0x7FC00000), "a NaN sum is the canonical word")
        require(_word(got[1]) == UInt32(0x80000000), "-0 + -0 stays -0")
    if case == 4:
        var reversed = List[List[Float32]]()
        for j in range(len(shards) - 1, -1, -1):  # driver-loop: the reversed order
            reversed.append(shards[j].copy())
        var other = _fold(entry, dst, reversed)
        var differs = False
        for i in range(len(other)):  # driver-loop: byte comparison of the check's outputs
            if _word(other[i]) != _word(got[i]):
                differs = True
                break
        require(differs, "the reversed shard order gave the same bytes on every element")


def _run_cpu() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    publish_parallel_base_host()
    var entry = parallel_ref(PARALLEL_TABLE_BASE, ParallelEntryKey(Int(GLOBAL_NUMERIC_MODE), 0))

    def task(unit: Int) raises {imm entry}:
        run_case(entry, unit, False)

    check_cases("cross_vendor_ordered_fold", ORDERED_FOLD_CASES, task)


def _run_gpu() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var vendor = parallel_entry_vendor(COMPILED_VENDOR)
    require(vendor >= 1, "cross_vendor_ordered_fold_gpu: this build has no GPU vendor")
    publish_parallel_base_device()
    var entry = parallel_ref(PARALLEL_TABLE_BASE, ParallelEntryKey(Int(GLOBAL_NUMERIC_MODE), vendor))

    # ordered launches on the one shared device context
    def ordered_case(unit: Int) raises {mut entry}:
        run_case(entry, unit, True)

    check_cases_serial("cross_vendor_ordered_fold", ORDERED_FOLD_CASES, ordered_case)


def test_cross_vendor_ordered_fold() raises:
    comptime if CPU_COLUMN_BUILD:
        _run_cpu()
    else:
        _run_gpu()


def main() raises:
    # The tests by name, not discover_tests: the device column's functions do not compile in the host column build.
    var suite = TestSuite()
    suite.test[test_cross_vendor_ordered_fold]()
    suite^.run()
