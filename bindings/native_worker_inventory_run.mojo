# SPDX-License-Identifier: Apache-2.0
"""Typed checks of the worker inventory (RUN): the driver status refusal and the hexadecimal field test."""
from std.ffi import c_int


def _inventory_status(status: c_int, operation: String) raises:
    if status != 0:
        raise Error("worker inventory: " + operation + " failed with driver status " + String(status))


def _inventory_hex(text: String, count_min: Int, count_max: Int) -> Bool:
    if text.byte_length() < count_min or text.byte_length() > count_max:
        return False
    var bytes = text.as_bytes()
    for i in range(len(bytes)):  # small-loop(the bounded UUID or PCI hexadecimal field)
        var byte = bytes[i]
        if not (UInt8(48) <= byte <= UInt8(57)) and not (UInt8(97) <= byte <= UInt8(102)):
            return False
    return True
