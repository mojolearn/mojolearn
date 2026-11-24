# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Lane mf-del8 (2026-11-23): THE UNIT CLASSES (core/member_unit_classes.mojo), the native pass that replaced the
member fit doors for a classifier unit whose rows lack a class (a bagging draw, a stacking cv fold). Exact
oracles: the codes are written here and the held classes and re-coded codes follow by hand. Written, not run
(WRITE CODE ONLY); IN_FILTER=mf_del8_unit_classes. The meta fits over the re-coded members are the Python test
python/mojolearn/tests/test_mf_del8_refusals.py and the digests tools/mf_del8_digests.py.

    test_every_class_held          codes 0 1 2 2 1 0 with k = 3: empty (every class held), every row and a draw
    test_lacking_rows              codes 0 2 2 0 (k = 3): held [0, 2]; re-coded 0 1 1 0
    test_lacking_draw              codes 0 1 2 1 0 2, int32 draw rows [0, 4, 2] (k = 3): held [0, 2]; the
                                   int64 draw of the same rows: the same; the re-code of every row maps 1 to 0
                                   (no draw of the unit reads it) and 2 to 1
    test_one_class                 codes 1 1 1 (k = 3): held [1]; re-coded 0 0 0
    test_outside_raises            a code 3 with k = 3 raises MEMBER_UNIT_OUTSIDE (an invariant)
    test_split_rows                20000 rows, codes i % 5 except class 3 absent (k = 5), split across the host
                                   cores: held [0, 1, 2, 4]; the re-code of row i equals its index in held

FAIL-PROOF (each edit makes the named case fail): return `held` even when it holds every class ->
test_every_class_held; drop the `_unit_row` call (read row i, not the draw's) -> test_lacking_draw; write
`table[code] = Int32(code)` in member_unit_recode -> test_lacking_rows; drop the out-of-range flag ->
test_outside_raises; merge only task 0's marks -> test_split_rows.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import require
from core.member_unit_classes import member_unit_classes, member_unit_recode, MEMBER_UNIT_OUTSIDE


def _same(got: List[Int32], want: List[Int32]) -> Bool:
    if len(got) != len(want):
        return False
    for i in range(len(got)):
        if got[i] != want[i]:
            return False
    return True


def test_every_class_held() raises:
    var codes: List[Int32] = [0, 1, 2, 2, 1, 0]
    var held = member_unit_classes(Int(codes.unsafe_ptr()), 0, False, len(codes), 3)
    require(len(held) == 0, "every class held: empty")
    var draw: List[Int32] = [0, 1, 2]
    var drawn = member_unit_classes(Int(codes.unsafe_ptr()), Int(draw.unsafe_ptr()), False, len(draw), 3)
    require(len(drawn) == 0, "a draw holding every class: empty")
    _ = len(codes)
    _ = len(draw)


def test_lacking_rows() raises:
    var codes: List[Int32] = [0, 2, 2, 0]
    var held = member_unit_classes(Int(codes.unsafe_ptr()), 0, False, len(codes), 3)
    var want_held: List[Int32] = [0, 2]
    require(_same(held, want_held), "held [0, 2]")
    var recoded = member_unit_recode(Int(codes.unsafe_ptr()), len(codes), 3, held)
    var want: List[Int32] = [0, 1, 1, 0]
    require(_same(recoded, want), "re-coded 0 1 1 0")
    _ = len(codes)


def test_lacking_draw() raises:
    var codes: List[Int32] = [0, 1, 2, 1, 0, 2]
    var draw: List[Int32] = [0, 4, 2]
    var held = member_unit_classes(Int(codes.unsafe_ptr()), Int(draw.unsafe_ptr()), False, len(draw), 3)
    var want_held: List[Int32] = [0, 2]
    require(_same(held, want_held), "int32 draw: held [0, 2]")
    var wide: List[Int64] = [0, 4, 2]
    var held_wide = member_unit_classes(Int(codes.unsafe_ptr()), Int(wide.unsafe_ptr()), True, len(wide), 3)
    require(_same(held_wide, want_held), "int64 draw: held [0, 2]")
    var recoded = member_unit_recode(Int(codes.unsafe_ptr()), len(codes), 3, held)
    var want: List[Int32] = [0, 0, 1, 0, 0, 1]
    require(_same(recoded, want), "every row re-coded: 1 -> 0 (unread), 2 -> 1")
    _ = len(codes)
    _ = len(draw)
    _ = len(wide)


def test_one_class() raises:
    var codes: List[Int32] = [1, 1, 1]
    var held = member_unit_classes(Int(codes.unsafe_ptr()), 0, False, len(codes), 3)
    var want_held: List[Int32] = [1]
    require(_same(held, want_held), "held [1]")
    var recoded = member_unit_recode(Int(codes.unsafe_ptr()), len(codes), 3, held)
    var want: List[Int32] = [0, 0, 0]
    require(_same(recoded, want), "re-coded 0 0 0")
    _ = len(codes)


def test_outside_raises() raises:
    var codes: List[Int32] = [0, 3, 1]
    var raised = String("")
    try:
        _ = member_unit_classes(Int(codes.unsafe_ptr()), 0, False, len(codes), 3)
    except error:
        raised = String(error)
    _ = len(codes)
    require(raised == String(MEMBER_UNIT_OUTSIDE), "a code outside 0 .. k-1 raises: '" + raised + "'")


def test_split_rows() raises:
    var n = 20000
    var codes = List[Int32](length=n, fill=Int32(0))
    for i in range(n):
        var c = i % 5
        codes[i] = Int32(4 if c == 3 else c)
    var held = member_unit_classes(Int(codes.unsafe_ptr()), 0, False, n, 5)
    var want_held: List[Int32] = [0, 1, 2, 4]
    require(_same(held, want_held), "split rows: held [0, 1, 2, 4]")
    var recoded = member_unit_recode(Int(codes.unsafe_ptr()), n, 5, held)
    var ok = len(recoded) == n
    for i in range(min(n, len(recoded))):
        var c = Int(codes[i])
        var want = 3 if c == 4 else c
        if Int(recoded[i]) != want:
            ok = False
            break
    require(ok, "split rows: each row's code is its index in held")
    _ = len(codes)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
