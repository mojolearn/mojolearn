# SPDX-License-Identifier: Apache-2.0
"""Host admission of the float32 sequence lengths used by recurrent models."""

from sequence.ops import FP


def check_recurrent_lengths(lengths: FP, rows: Int, steps: Int, operation: String, padded: Bool = False) raises:
    for r in range(rows):  # admission: one comparison per row before a model uses the lengths
        var v = lengths.unsafe_load(r)
        if v != v or v < Float32(1.0) or v > Float32(steps) or Float32(Int(v)) != v:
            var which = String("padded sequence length ") if padded else String("sequence length ")
            raise Error("ValueError: " + operation + ": every length must be a whole number from 1 to the "
                        + which + String(steps))
