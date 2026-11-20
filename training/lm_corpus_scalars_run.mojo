# SPDX-License-Identifier: Apache-2.0
"""RUN piece of the corpus step scalar preparation
(bindings/lm_corpus_scalars.mojo): NumPy floating scalar multiplication
`step * batch * length` reproduced on native typed words, with the overflow
refusals. Typed Mojo values only, no Python import
(tools/python_free_dirs.sh). The door reads the step into a Float64 through
an exact builtin float and converts the result back through an exact builtin
int (lane rs-embedding-bytelm, 2026-11-12: the body moved unchanged)."""
from std.math import isfinite


def corpus_floating_product(value: Float64, batch: Int, length: Int, width: Int) raises -> Float64:
    """`value * batch * length` in the NumPy floating width `width` (2, 4 or
    8 bytes), each product rounded to that width; refuses a non-finite step,
    dimension or product."""
    if not isfinite(value):
        raise Error("ValueError: corpus step multiplication requires a finite native scalar")
    var result: Float64
    if width == 2:
        # NumPy half scalar multiplication rounds each Float32 product back
        # to Float16 before using that scalar in the following multiplication.
        var left = Float16(value)
        var row_count = Float16(batch)
        var row_length = Float16(length)
        if not isfinite(Float32(row_count)) or not isfinite(Float32(row_length)):
            raise Error("ValueError: corpus dimensions overflow NumPy float16 promotion")
        var first = Float16(Float32(left) * Float32(row_count))
        var product = Float16(Float32(first) * Float32(row_length))
        if not isfinite(Float32(first)) or not isfinite(Float32(product)):
            raise Error("ValueError: corpus step multiplication overflows NumPy float16")
        result = Float64(product)
    elif width == 4:
        var first = Float32(value) * Float32(batch)
        var product = first * Float32(length)
        if not isfinite(first) or not isfinite(product):
            raise Error("ValueError: corpus step multiplication overflows NumPy float32")
        result = Float64(product)
    else:
        var first = value * Float64(batch)
        var product = first * Float64(length)
        if not isfinite(first) or not isfinite(product):
            raise Error("ValueError: corpus step multiplication overflows NumPy float64")
        result = product
    return result
