"""LU result encoding preserves every non-NaN word, pins computed NaNs."""
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_decomp.cells import lu_result_word


def main() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var inputs: List[UInt32] = [
        0x00000000, 0x80000000, 0x00000001, 0x80000001,
        0x3f800000, 0xbf800000, 0x7f7fffff, 0xff7fffff,
        0x7f800000, 0xff800000, 0x7f800001, 0xff800001,
        0x7fc00000, 0xffc00000, 0x7fffffff, 0xffffffff,
    ]
    for bits in inputs:
        var expected = UInt32(0x7fc00000) if (bits & UInt32(0x7fffffff)) > UInt32(0x7f800000) else bits
        var got = bitcast[DType.uint32](lu_result_word(bitcast[DType.float32](bits)))
        if got != expected:
            raise Error("LU result encoding changed a non-NaN word or left a vendor NaN payload")
    print("LU_RESULT_WORD PASS finite/infinite/zero/subnormal preserved; NaNs pinned")
