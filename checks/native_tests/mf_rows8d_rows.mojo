# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""THE LAST SEARCH ROWS (lane mf-rows8d, 2026-11-23), in a process with no Python, over the host column's helper
entries. StratifiedShuffleSplit over a MULTI-OUTPUT target stratifies on each row's joined words (scikit-learn 1.9
model_selection/_split.py lines 2341-2348: `" ".join(row.astype("str"))`, then `np.unique(return_inverse=True)`);
the search RUN builds those strings (core/ensemble_member.mojo `_joined_rows`) and codes them
(bindings/msel_split_spec_run.mojo `msel_joined_row_codes`), then splits as the one-output path does. Expected values
worked by hand from the same inputs:

0. rows "1 0", "0 1", "1 0", "10 1", "1 10": the sorted distinct strings are "0 1" < "1 0" < "1 10" < "10 1"
   (space 0x20 sorts before '0'), so the codes are [1, 0, 1, 3, 2] and there are 4 classes.
1. labels holding a space join alike: rows "a b c" (labels "a b" and "c"), "a b c" (labels "a" and "b c") and "x":
   one class for the first two, as NumPy's unique of the strings makes it: codes [0, 0, 1], 2 classes.
2. NumPy's scalar spelling the joined words use (core/native_archive_scalar_text.mojo `archive_float_text`, print
   mode sys.maxsize): float64 1.0 "1.0", 1e16 "1e+16", 1e-05 "1e-05", 2.5 "2.5"; float32 0.1 "0.1".
3. StratifiedShuffleSplit (n_splits 1, test_size 2, seed 3) over joined rows whose class "0 1" has one member: its
   refusal "The least populated classes in y have only 1 member, ...".
4. The same splitter over rows "0 1" x 4 and "1 0" x 4 (two classes of four): one split, two test rows (one of each
   class, `_approximate_mode`), six train rows, train and test disjoint.

FAIL-PROOF EDITS (each must make a case fail): in `msel_joined_row_codes` drop `sort(ordered)` (case 0 gets the
first-seen order [0, 1, 0, 2, 3]); key the dictionary by row index instead of the string (case 1 gets 3 classes);
in `_joined_rows` join with "," (the strings no longer match case 0's oracle when built by the RUN; covered by the
digest units); in `msel_plan_strat_shuffle` drop the `least < 2` refusal (case 3 splits).
"""
from std.testing import TestSuite
from std.memory import bitcast
from checks.native_tests.testsuite_cases import check_cases, require
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.msel_entry_table import MselEntryKey
from core.msel_helpers_host import publish_msel_helpers_host
from core.msel_helpers_calls import msel_helpers_ref
from core.base_entry_table import BaseEntryKey
from core.base_helpers_host import publish_base_helpers_host
from core.base_helpers_calls import base_helpers_ref
from core.native_archive_scalar_text import archive_float_text
from x_metrics.descriptor_host import metrics_program_host_v1
from bindings.msel_split_plan_run import MselSeed, MselSplitRoute, MselSizeWord, MSEL_SIZE_INT
from bindings.msel_split_spec_run import (
    MselSplitSpec, MselGroupCodes, msel_spec_sides, msel_joined_row_codes, MSEL_SPEC_STRATIFIED_SHUFFLE,
)

comptime CASES = 5
comptime PRINT_MODE = 2147483647


def _route() raises -> MselSplitRoute:
    var mode = Int(GLOBAL_NUMERIC_MODE)
    return MselSplitRoute(msel_helpers_ref(MselEntryKey(mode, 0)), base_helpers_ref(BaseEntryKey(mode, 0)),
                          metrics_program_host_v1, False, False)


def _sss() raises -> MselSplitSpec:
    var spec = MselSplitSpec()
    spec.kind = MSEL_SPEC_STRATIFIED_SHUFFLE
    spec.route = _route()
    spec.rows_helpers = base_helpers_ref(BaseEntryKey(Int(GLOBAL_NUMERIC_MODE), 0))
    spec.k = 1
    spec.test = MselSizeWord(MSEL_SIZE_INT, 2, 0.0, 64, String(2))
    spec.seed = MselSeed(3, False)
    return spec^


def check_sorted_codes() raises:
    var rows: List[String] = ["1 0", "0 1", "1 0", "10 1", "1 10"]
    var got = msel_joined_row_codes(rows)
    var want: List[Int] = [1, 0, 1, 3, 2]
    require(got.m == 4, "classes: " + String(got.m))
    for i in range(len(want)):  # small-loop(oracle)
        require(Int(got.codes[i]) == want[i], "row " + String(i) + " code " + String(Int(got.codes[i])))


def check_alike_strings() raises:
    var rows: List[String] = [String("a b") + " " + "c", String("a") + " " + "b c", "x"]
    var got = msel_joined_row_codes(rows)
    require(got.m == 2, "classes: " + String(got.m))
    require(Int(got.codes[0]) == 0 and Int(got.codes[1]) == 0 and Int(got.codes[2]) == 1, "codes [0, 0, 1]")


def check_numpy_words() raises:
    require(archive_float_text(bitcast[DType.uint64](Float64(1.0)), 8, PRINT_MODE) == "1.0", "1.0")
    require(archive_float_text(bitcast[DType.uint64](Float64(1e16)), 8, PRINT_MODE) == "1e+16", "1e+16")
    require(archive_float_text(bitcast[DType.uint64](Float64(1e-05)), 8, PRINT_MODE) == "1e-05", "1e-05")
    require(archive_float_text(bitcast[DType.uint64](Float64(2.5)), 8, PRINT_MODE) == "2.5", "2.5")
    require(archive_float_text(UInt64(bitcast[DType.uint32](Float32(0.1))), 4, PRINT_MODE) == "0.1", "float32 0.1")


def check_least_refusal() raises:
    var rows: List[String] = ["1 0", "1 0", "0 1", "1 0"]
    var coded = msel_joined_row_codes(rows)
    var words = String("")
    try:
        _ = msel_spec_sides(_sss(), 4, Int(coded.codes.unsafe_ptr()), coded.m, MselGroupCodes())
    except e:
        words = String(e)
    _ = len(coded.codes)
    require(words.startswith("The least populated classes in y have only 1 member"), "refusal: " + words)


def check_two_classes() raises:
    var rows: List[String] = ["0 1", "1 0", "0 1", "1 0", "0 1", "1 0", "0 1", "1 0"]
    var coded = msel_joined_row_codes(rows)
    var sides = msel_spec_sides(_sss(), 8, Int(coded.codes.unsafe_ptr()), coded.m, MselGroupCodes())
    _ = len(coded.codes)
    require(sides.splits() == 1, "one split")
    require(len(sides.test[0]) == 2 and len(sides.train[0]) == 6, "sizes 6 / 2")
    var classes = List[Int](length=2, fill=0)
    for i in range(2):  # small-loop(the test rows)
        classes[Int(coded.codes[Int(sides.test[0][i])])] += 1
        for j in range(6):  # small-loop(oracle: disjoint sides)
            require(sides.train[0][j] != sides.test[0][i], "train and test disjoint")
    require(classes[0] == 1 and classes[1] == 1, "one test row of each class")


def test_mf_rows8d_rows() raises:
    publish_msel_helpers_host()
    publish_base_helpers_host()

    def run_case(which: Int) raises:
        if which == 0:
            check_sorted_codes()
        elif which == 1:
            check_alike_strings()
        elif which == 2:
            check_numpy_words()
        elif which == 3:
            check_least_refusal()
        else:
            check_two_classes()
    check_cases("mf_rows8d_rows", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
