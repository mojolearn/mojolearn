# SPDX-License-Identifier: Apache-2.0
# Typed argument words for the autograd kernel entries. The Python doors
# (autograd/*_pyapi.mojo) convert PythonObject arguments into Ints and
# Float64s once; these checks then run here, in the order the entries
# used to run them, with the same messages. No data is touched.
from sequence.ops import FP
from core.cast_in import CastIn, cast_in_kind


def fptr_word(a: Int, what: String) raises -> FP:
    """`sequence.pyapi.fptr` of an address already converted to an Int."""
    if a == 0:
        raise Error("sequence: null float32 buffer for " + what)
    return FP(unsafe_from_address=a)


def fval_word(p: List[Float64], i: Int) -> Float32:
    """`sequence.pyapi.fval` of a float already converted to a Float64."""
    return Float32(p[i])


def cast_in_words(desc: List[Int], what: String) raises -> CastIn:
    """`autograd.pyapi.cast_in_of` of a descriptor already converted to Ints:
    [address, NumPy dtype number, item size, rows, cols, row stride, column
    stride] as a checked CastIn."""
    if len(desc) != 7:
        raise Error(what + ": the input descriptor takes 7 integers")
    var c = CastIn(
        desc[0], cast_in_kind(desc[1], desc[2], what), desc[2],
        desc[3], desc[4], desc[5], desc[6],
    )
    c.check(what)
    return c
