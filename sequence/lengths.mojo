# SPDX-License-Identifier: Apache-2.0
"""Admission of the float32 sequence lengths used by recurrent models."""

from sequence.ops import FP


@always_inline
def recurrent_length_valid(value: Float32, steps: Int) -> Bool:
    # The rounded float32 encoding of steps can exceed the integer buffer length.
    if not (value >= Float32(1) and value < Float32(Int.MAX)):
        return False
    var n = Int(value)
    return n <= steps and Float32(n) == value


def check_recurrent_lengths(lengths: FP, rows: Int, steps: Int, operation: String, padded: Bool = False) raises:
    for r in range(rows):  # admission: one comparison per row before a model uses the lengths
        var v = lengths.unsafe_load(r)
        if not recurrent_length_valid(v, steps):
            var which = String("padded sequence length ") if padded else String("sequence length ")
            raise Error("ValueError: " + operation + ": every length must be a whole number from 1 to the "
                        + which + String(steps))
