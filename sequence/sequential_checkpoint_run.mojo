# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN body of `bindings/sequential_checkpoint.mojo` (wave 8 run split): the
JSON string quoting of the verification report, moved unchanged."""

from std.collections.string import chr


def _quote(value: String) -> String:
    var result = value.replace("\\", "\\\\").replace("\"", "\\\"")
    var digits = "0123456789abcdef"
    for k in range(32):  # small-loop(32: JSON control-character alphabet): report text encoding
        result = result.replace(chr(k), "\\u00" + String(digits[byte=k // 16]) + String(digits[byte=k % 16]))
    return "\"" + result + "\""
