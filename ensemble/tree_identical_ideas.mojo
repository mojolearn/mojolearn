# SPDX-License-Identifier: Apache-2.0
"""2026-10-06 tree source experiments. A opts in; absence preserves B.

NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
No symbol in this module is evidence of runtime reach or qualification.
The source integration ledger is experiments/trees_identical_20261006/forest/.
"""
from std.sys.compile import is_defined, get_defined_int
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

# Every switch below is default OFF and IDENTICAL-only. The shared status above
# applies individually to every switch and its parameter/sub-arm.
comptime T01 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T01"]()
comptime T01_REPLICAS = get_defined_int["MOJOLEARN_TREES_T01_REPLICAS", 4]()
comptime T01_ROWS = get_defined_int["MOJOLEARN_TREES_T01_ROWS", 256]()
comptime T02 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T02"]()
comptime T03 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T03"]()
comptime T04 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T04"]()
comptime T05 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T05"]()
comptime T06 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T06"]()
comptime T07 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T07"]()
comptime T08 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T08"]()
comptime T08_LAYOUT = get_defined_int["MOJOLEARN_TREES_T08_LAYOUT", 1]()
comptime T08_BITS = get_defined_int["MOJOLEARN_TREES_T08_BITS", 8]()
comptime T09 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T09"]()
comptime T10 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T10"]()
comptime T11 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T11"]()
comptime T11_LEVELS = get_defined_int["MOJOLEARN_TREES_T11_LEVELS", 8]()
comptime T12 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T12"]()
comptime T12_BYTES = get_defined_int["MOJOLEARN_TREES_T12_BYTES", 256*1024*1024]()
comptime T13 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T13"]()
comptime T13_BYTES = get_defined_int["MOJOLEARN_TREES_T13_BYTES", 16*1024*1024]()
comptime T14 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T14"]()
comptime T14_EXACT = T14 and is_defined["MOJOLEARN_TREES_T14_EXACT"]()
comptime T15 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T15"]()
comptime C48 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_C48"]()

@always_inline
def histogram_task_rows(histogram_bytes: Int, row_bytes: Int, reference: Int) -> Int:
    """T02/C47: amortize one histogram clear/flush over comparable input bytes.

    Keep at least one 128-thread block, at most eight row visits per lane.
    The bound follows descriptor overhead/row work and is unrelated to datasets.
    Partition descriptors remain on their unchanged 128-row geometry.
    """
    comptime if T02:
        var rows = max(128, min(1024, (2 * histogram_bytes) // max(1, row_bytes)))
        return ((rows + 127) // 128) * 128
    comptime if T01:
        return max(128, min(1024, ((T01_ROWS + 127) // 128) * 128))
    return reference
