# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""`mexp_gather_i32` split over the host pool (core/metrics_exp_impl.mojo,
lane vdf-xm-labels 2026-11-24) against the one serial loop it replaced,
written out here as the reference.

  0 200,000 rows (four 65,536-row runs), misses in runs 0 and 3: the first
    miss is run 0's
  1 200,000 rows, the only misses in runs 2 and 3 (a code mapped to -1 and
    a code past the table): the first miss is run 2's
  2 200,000 rows, no miss: -1
  3 3 rows (one run, the calling thread), a negative code
Every row of every case is compared, so a run that stops writing at its
first miss, a run boundary off by one, or a first miss taken from the last
run instead of the first fails here.
Host column: the suite is host work and does not branch on the target."""
from std.testing import TestSuite

from core.metrics_exp_impl import mexp_gather_i32
from checks.native_tests.testsuite_cases import require

comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]


def _reference(table: List[Int32], codes: List[Int32], mut dst: List[Int32]) -> Int:
    """The serial loop as it stood before the split."""
    var first = -1
    for i in range(len(codes)):  # reference loop: the serial form under check
        var c = Int(codes[i])
        var v = Int32(-1)
        if c >= 0 and c < len(table):
            v = table[c]
        dst[i] = v
        if v < 0 and first < 0:
            first = i
    return first


def _check(table: List[Int32], codes: List[Int32], want: Int, name: String) raises:
    var n = len(codes)
    var ref_dst = List[Int32](length=n, fill=Int32(-9))
    var rf = _reference(table, codes, ref_dst)
    require(rf == want, name + ": reference first miss " + String(rf) + ", expected " + String(want))
    var dst = List[Int32](length=n, fill=Int32(-9))
    var f = mexp_gather_i32(_I32P(unsafe_from_address=Int(table.unsafe_ptr())), len(table),
                            _I32P(unsafe_from_address=Int(codes.unsafe_ptr())), n,
                            _I32P(unsafe_from_address=Int(dst.unsafe_ptr())))
    require(f == rf, name + ": split first miss " + String(f) + ", serial " + String(rf))
    for i in range(n):  # check loop: every row
        require(dst[i] == ref_dst[i], name + ": row " + String(i) + " split " + String(dst[i])
                + ", serial " + String(ref_dst[i]))


def test_metric_gather_host() raises:
    var table: List[Int32] = [5, -1, 0, 2, 1]
    var n = 200000
    var codes = List[Int32](length=n, fill=Int32(0))
    for i in range(n):  # fixture: codes 0, 2, 3, 4 (never the -1 class)
        codes[i] = Int32(0 if i % 4 == 0 else (i % 4) + 1)
    codes[70] = 1
    codes[199990] = 1
    _check(table, codes, 70, "misses in runs 0 and 3")
    codes[70] = 0
    codes[140000] = 7  # past the table, run 2
    codes[150001] = 1
    codes[199990] = 1
    _check(table, codes, 140000, "misses in runs 2 and 3 only")
    codes[140000] = 0
    codes[150001] = 0
    codes[199990] = 0
    _check(table, codes, -1, "no miss")
    var small: List[Int32] = [2, -3, 4]
    _check(table, small, 1, "one run, a negative code")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
