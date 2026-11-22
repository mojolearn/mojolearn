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
- `column_name()`: the column this binary was built for (`-D MOJOLEARN_COLUMN_CPU=1` builds the host column).

No `main()` here: shared helpers live in a module that is not a test file, so no file's discovery picks them up.
"""
from std.memory import bitcast
from std.sys import size_of
from std.sys.compile import is_defined

from core.host_parallel import host_parallelize

# How many failing rows one error message lists (the count is always exact).
comptime _LISTED_FAILURES = 20


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
