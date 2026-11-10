# SPDX-License-Identifier: Apache-2.0
"""Typed numeric promotion and preparation plans for admitted sequence inputs."""

def sequence_promote(mut kind: String, mut size: Int, incoming: String, width: Int) raises:
    if (incoming != "b" and incoming != "i" and incoming != "u" and incoming != "f") or (
        incoming == "f" and width != 2 and width != 4 and width != 8
    ) or (incoming != "f" and width != 1 and width != 2 and width != 4 and width != 8):
        raise Error("TypeError: inputs must contain supported numeric scalars")
    if size == 0 or kind == "b":
        kind = incoming
        size = width
    elif incoming == "b":
        return
    elif kind == incoming:
        size = max(size, width)
    elif kind == "f" or incoming == "f":
        var integer_width = width if kind == "f" else size
        var float_width = size if kind == "f" else width
        # Numeric dtype promotion: half holds 8-bit integers, single holds
        # 16-bit integers; 32/64-bit integer mixtures use double precision.
        size = max(float_width, 2 if integer_width <= 1 else (4 if integer_width <= 2 else 8))
        kind = "f"
    else:
        var signed_width = size if kind == "i" else width
        var unsigned_width = size if kind == "u" else width
        if signed_width > unsigned_width:
            kind = "i"
            size = signed_width
        elif unsigned_width < 8:
            kind = "i"
            size = unsigned_width * 2
        else:
            kind = "f"
            size = 8

