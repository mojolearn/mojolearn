# SPDX-License-Identifier: Apache-2.0
"""Independent, unexecuted arithmetic references for NEURAL experiments.

NOT TESTED — NOT COMPILED — NOT MEASURED. These are source definitions,
not validation results or a test runner. Model-quality and all-column gates
remain pending. No reference is called from the production GPU runtime.
"""

from checks.numerics import ftz


def m2_gradient_tree_reference(
    part: MutPointer[Float32, MutAnyOrigin],
    first_leaf: Int, leaf_count: Int, cols: Int, column: Int,
) -> Float32:
    """M10 adjacent-pair tree, expressed independently as recursive spans.

    Split at the largest power of two strictly below the span length. This
    describes the same promotion of odd nodes as iterative adjacent-pair
    compaction, without mutating scratch or calling its implementation.
    A leaf returns verbatim; internal nodes FTZ each operand and result.
    Caller supplies leaf_count >= 1 and the original 256-row leaf values.
    """
    if leaf_count == 1:
        return part.unsafe_load(first_leaf * cols + column)
    var left_count = 1
    while left_count * 2 < leaf_count:
        left_count *= 2
    var left = m2_gradient_tree_reference(part, first_leaf, left_count, cols, column)
    var right = m2_gradient_tree_reference(part, first_leaf + left_count,
                                          leaf_count - left_count, cols, column)
    return ftz(ftz(left) + ftz(right))
