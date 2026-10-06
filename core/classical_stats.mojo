# SPDX-License-Identifier: Apache-2.0
"""C01 classical mean, sharing logical leaves with regression metrics.
NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
No GPU geometry or global/neural numeric mode enters this profile.
"""
from x_metrics.common import PairSum
from checks.numerics import ftz, identical_div


def classical_column_mean(
    values: MutPointer[Float32, MutAnyOrigin], rows: Int, cols: Int, column: Int,
) -> Float32:
    var sum = PairSum()
    for row in range(rows):
        sum.add(ftz(values.unsafe_load(row*cols+column)))
    return ftz(identical_div(sum.result(), Float32(rows)))
