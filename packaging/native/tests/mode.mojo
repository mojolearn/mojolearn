# SPDX-License-Identifier: Apache-2.0
from std.sys.compile import is_defined
from mojolearn import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, NUMERIC_FAST
from mojolearn.checks.numerics import NumericMode


def main() raises:
    comptime expected = NUMERIC_FAST if is_defined["MOJOLEARN_NUMERIC_FAST"]() else NUMERIC_IDENTICAL
    comptime assert GLOBAL_NUMERIC_MODE == expected, "incorrect package default"
    if NumericMode.default().mode != expected:
        raise Error("NumericMode.default disagrees with the compiled mode")
    print("PASS native mode", GLOBAL_NUMERIC_MODE)
