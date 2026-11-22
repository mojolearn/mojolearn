# SPDX-License-Identifier: Apache-2.0
"""What the standard library's `TestSuite` does not give the native checks, written once.

A native check file is a TestSuite file (docs/plans/TESTSUITE_CONVERSION.md): `test_*` functions and

    def main() raises:
        TestSuite.discover_tests[__functions_in_module()]().run()

The runner (tools/ops/native_checks_steps.sh build and shard) compiles each such file to its own binary and runs it
as one process, one check per file. This module adds:

- `check_cases(label, n, body)`: a case table looped inside ONE `test_*`. The `n` cases are independent, so they run
  across the host cores (`host_parallelize`); every failing row is named in the one raised error, so TestSuite's
  FAIL line says which rows failed and why.
- `check_rows(label, rows, body)`: the same over a `List` of parameter structs; a failing row is named by its
  index and its fields (the struct's `Writable` form).
- `assert_same_words` and `assert_close`: the host-versus-device (or result-versus-reference) comparisons the
  suites repeated, one parallel pass over the cells, the count and the first differing cell in the message.
- `column_name()` and `CPU_COLUMN_BUILD`: the column this binary was built for (`-D MOJOLEARN_COLUMN_CPU=1` builds
  the host column); `CPU_COLUMN_BUILD` is for the `comptime if` that keeps device-only code out of the host build
  (discovery elaborates every function of the module, docs/plans/TESTSUITE_CONVERSION.md).
- `check_cases_serial(label, n, body)`: the same table in order on the calling thread, for cases that share the one
  device context and its pool (a check that had a `serial` reason).
- `require(ok, message)`: raise `message` when `ok` is false (the plain assertion every suite used).
- `fixture_words`, `fixture_f32`, `fixture_i32` (a stored little-endian 32-bit vector of `count` words, the extent
  checked) and `read_f32_file`, `read_i32_file` (the same, its length taken from the file).

No `main()` here: shared helpers live in a module that is not a test file, so no file's discovery picks them up.
"""
from std.memory import bitcast
from std.sys import size_of
from std.sys.compile import is_defined

from core.host_parallel import host_parallelize

# How many failing rows one error message lists (the count is always exact).
comptime _LISTED_FAILURES = 20

# True in the host column build (`-D MOJOLEARN_COLUMN_CPU=1`).
comptime CPU_COLUMN_BUILD = is_defined["MOJOLEARN_COLUMN_CPU"]()


def require(ok: Bool, message: String) raises:
    """Raise `message` when `ok` is false."""
    if not ok:
        raise Error(message)


def column_name() -> String:
    """`cpu` for the host column build (`-D MOJOLEARN_COLUMN_CPU=1`), else `gpu`."""
    comptime if is_defined["MOJOLEARN_COLUMN_CPU"]():
        return "cpu"
    return "gpu"


def _raise_failures(label: String, n: Int, failed: List[Int], messages: List[String], names: List[String]) raises:
    var count = 0
    var text = String("")
    for i in range(n):  # driver-loop: over the case slots, after the parallel run
        if failed[i] != 0:
            count += 1
            if count <= _LISTED_FAILURES:
                text += "\n      " + label + " row " + String(i)
                if len(names) > i and names[i].byte_length() > 0:
                    text += " (" + names[i] + ")"
                text += ": " + messages[i]
    if count > 0:
        raise Error(label + ": " + String(count) + " of " + String(n) + " cases failed" + text)


def check_cases[F: def(Int) raises -> None](label: String, n: Int, ref body: F) raises:
    """Run `body(0) ... body(n - 1)` across the host cores; raise one error naming every failing row."""
    var failed = List[Int](length=n, fill=0)
    var messages = List[String](length=n, fill=String(""))
    var fp = failed.unsafe_ptr()
    var mp = messages.unsafe_ptr()
    def task(unit: Int) {imm fp, imm mp, imm body}:
        try:
            body(unit)
        except problem:
            mp[unsafe_offset=unit] = String(problem)
            fp.unsafe_store(unit, 1)
    host_parallelize(task, n)
    _raise_failures(label, n, failed, messages, List[String]())


def check_cases_serial[F: def(Int) raises -> None](label: String, n: Int, ref body: F) raises:
    """Run `body(0) ... body(n - 1)` in order on the calling thread (the cases share the one device context and its
    pool, so they may not run concurrently); raise one error naming every failing row."""
    var failed = List[Int](length=n, fill=0)
    var messages = List[String](length=n, fill=String(""))
    for unit in range(n):  # driver-loop: over the cases, in order on the one device
        try:
            body(unit)
        except problem:
            messages[unit] = String(problem)
            failed[unit] = 1
    _raise_failures(label, n, failed, messages, List[String]())


def check_rows[T: Copyable & Writable, F: def(T) raises -> None](label: String, rows: List[T], ref body: F) raises:
    """Run `body(row)` for every row of the table across the host cores; a failing row is named by its index and
    its fields."""
    var n = len(rows)
    var failed = List[Int](length=n, fill=0)
    var messages = List[String](length=n, fill=String(""))
    var names = List[String](length=n, fill=String(""))
    var fp = failed.unsafe_ptr()
    var mp = messages.unsafe_ptr()
    var np = names.unsafe_ptr()
    var rp = rows.unsafe_ptr()
    def task(unit: Int) {imm fp, imm mp, imm np, imm rp, imm body}:
        try:
            body(rp[unsafe_offset=unit])
        except problem:
            mp[unsafe_offset=unit] = String(problem)
            np[unsafe_offset=unit] = String(rp[unsafe_offset=unit])
            fp.unsafe_store(unit, 1)
    host_parallelize(task, n)
    _raise_failures(label, n, failed, messages, names)


def assert_same_words[dt: DType](got: List[Scalar[dt]], want: List[Scalar[dt]], what: String) raises:
    """Equal bit for bit, cell by cell (a device result against the host column, or a result against its stored
    reference). The message gives the number of differing cells and the first one with both values."""
    if len(got) != len(want):
        raise Error(what + ": extent " + String(len(got)) + " vs " + String(len(want)))
    var n = len(got)
    var differ = List[Int](length=n, fill=0)
    var dp = differ.unsafe_ptr()
    var gp = got.unsafe_ptr()
    var wp = want.unsafe_ptr()
    comptime width = 8 * size_of[dt]()
    comptime U = DType.uint8 if width == 8 else (DType.uint16 if width == 16 else (
        DType.uint32 if width == 32 else DType.uint64))
    def cell(i: Int) {imm dp, imm gp, imm wp}:
        dp.unsafe_store(i, Int(bitcast[U](gp[unsafe_offset=i]) != bitcast[U](wp[unsafe_offset=i])))
    host_parallelize(cell, n)
    _report_cells(what, "words differ", n, differ, got, want)


def assert_close[dt: DType](got: List[Scalar[dt]], want: List[Scalar[dt]], rtol: Float64, atol: Float64,
                            what: String) raises:
    """`|got - want| <= atol + rtol * |want|` in every cell, NaN only against NaN."""
    comptime assert dt.is_floating_point(), "assert_close compares floating-point cells"
    if len(got) != len(want):
        raise Error(what + ": extent " + String(len(got)) + " vs " + String(len(want)))
    var n = len(got)
    var differ = List[Int](length=n, fill=0)
    var dp = differ.unsafe_ptr()
    var gp = got.unsafe_ptr()
    var wp = want.unsafe_ptr()
    def cell(i: Int) {imm dp, imm gp, imm wp, imm rtol, imm atol}:
        var g = Float64(gp[unsafe_offset=i])
        var w = Float64(wp[unsafe_offset=i])
        var ok: Bool
        if g != g or w != w:
            ok = g != g and w != w
        else:
            ok = abs(g - w) <= atol + rtol * abs(w)
        dp.unsafe_store(i, Int(not ok))
    host_parallelize(cell, n)
    _report_cells(what, "cells outside rtol " + String(rtol) + " atol " + String(atol), n, differ, got, want)


def _report_cells[dt: DType](what: String, kind: String, n: Int, differ: List[Int], got: List[Scalar[dt]],
                             want: List[Scalar[dt]]) raises:
    var count = 0
    var first = -1
    for i in range(n):  # driver-loop: over the per-cell flags, after the parallel compare
        if differ[i] != 0:
            count += 1
            if first < 0:
                first = i
    if count > 0:
        raise Error(what + ": " + String(count) + " of " + String(n) + " " + kind + "; first at " + String(first)
                    + " (got " + String(got[first]) + ", want " + String(want[first]) + ")")


def fixture_words(path: String, count: Int) raises -> List[UInt32]:
    """The little-endian 32-bit words of a stored fixture (paths relative to the repository root). `count` is the
    extent the file must hold; a negative `count` takes it from the file (its length must be a multiple of 4). One
    independent cell per word across the host cores."""
    var f = open(path, "r")
    var bytes = f.read_bytes()
    f.close()
    var n = count
    if n < 0:
        if len(bytes) % 4 != 0:
            raise Error("fixture extent: " + path + " holds " + String(len(bytes)) + " bytes, not 32-bit words")
        n = len(bytes) // 4
    if len(bytes) != 4 * n:
        raise Error("fixture extent: " + path + " holds " + String(len(bytes)) + " bytes, expected " + String(4 * n))
    var values = List[UInt32](length=n, fill=UInt32(0))
    var bp = bytes.unsafe_ptr()
    var vp = values.unsafe_ptr()
    def word(index: Int) {imm bp, imm vp}:
        vp.unsafe_store(index, UInt32(bp[4 * index]) | (UInt32(bp[4 * index + 1]) << 8)
                        | (UInt32(bp[4 * index + 2]) << 16) | (UInt32(bp[4 * index + 3]) << 24))
    host_parallelize(word, n)
    _ = bytes^
    return values^


def fixture_f32(path: String, count: Int) raises -> List[Float32]:
    """A stored little-endian float32 vector of `count` values (negative: the file's length)."""
    var words = fixture_words(path, count)
    var n = len(words)
    var values = List[Float32](length=n, fill=Float32(0))
    var wp = words.unsafe_ptr()
    var vp = values.unsafe_ptr()
    def cell(index: Int) {imm wp, imm vp}:
        vp.unsafe_store(index, bitcast[DType.float32](wp[index]))
    host_parallelize(cell, n)
    _ = words^
    return values^


def fixture_i32(path: String, count: Int) raises -> List[Int32]:
    """A stored little-endian int32 vector of `count` values (negative: the file's length)."""
    var words = fixture_words(path, count)
    var n = len(words)
    var values = List[Int32](length=n, fill=Int32(0))
    var wp = words.unsafe_ptr()
    var vp = values.unsafe_ptr()
    def cell(index: Int) {imm wp, imm vp}:
        vp.unsafe_store(index, bitcast[DType.int32](wp[index]))
    host_parallelize(cell, n)
    _ = words^
    return values^


def read_f32_file(path: String) raises -> List[Float32]:
    """A stored little-endian float32 vector, its length taken from the file."""
    return fixture_f32(path, -1)


def read_i32_file(path: String) raises -> List[Int32]:
    """A stored little-endian int32 vector, its length taken from the file."""
    return fixture_i32(path, -1)
