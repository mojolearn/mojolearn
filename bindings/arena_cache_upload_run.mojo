# SPDX-License-Identifier: Apache-2.0
"""RUN of the arena cache upload owner (bindings/arena_cache_upload.mojo): the
arbitrary-precision binary increment of a DeviceCache counter that does not fit
a machine word. No Python import (tools/python_free_dirs.sh). Lane
rs-shared-infra-b (2026-11-12): moved word for word; one counter per call,
nothing to split across cores."""


def _counter_binary_increment(text: String) raises -> String:
    """`text` is `bin(n)` of an exact integer `n` ("0b..." or "-0b..."); the
    binary spelling (no prefix) of `n + 1`."""
    var negative = text.startswith("-")
    var offset = 3 if negative else 2
    var digits = List[UInt8]()
    for i in range(offset,text.byte_length()):  # driver-loop: exact arbitrary-precision binary digits
        digits.append(text.as_bytes()[i])
    var carry = True
    for i in range(len(digits)-1,-1,-1):  # driver-loop: increment a positive magnitude or decrement a negative one
        if not carry:
            break
        if negative:
            if digits[i] == 49:
                digits[i] = 48
                carry = False
            else:
                digits[i] = 49
        elif digits[i] == 48:
            digits[i] = 49
            carry = False
        else:
            digits[i] = 48
    var result = String("-") if negative else String("")
    if carry and not negative:
        result += "1"
    for i in range(len(digits)):  # driver-loop: publish the native integer's computed binary spelling
        result += "1" if digits[i] == 49 else "0"
    return result
