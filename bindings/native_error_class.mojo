# SPDX-License-Identifier: Apache-2.0
"""The Python exception class a native error names in its text prefix.

Native code raises `Error("TypeError: ...")`, `Error("ValueError: ...")` and
so on; the C boundary picks the Python class from that prefix. Python prints
the class itself, so the boundary drops the `<Class>: ` prefix exactly once
before raising, and the user sees `TypeError: export target has wrong dtype`
instead of `TypeError: TypeError: export target has wrong dtype`.
"""


def error_without_class(error: Error, kind: String) -> Error:
    # Strips one leading `<kind>: ` (or `<kind>:`); any other text, and a
    # message that does not start with `kind`, is returned unchanged.
    var message = String(error)
    var spaced = kind + ": "
    if message.startswith(spaced):
        return Error(String(message[byte=spaced.byte_length():]))
    var bare = kind + ":"
    if message.startswith(bare):
        return Error(String(message[byte=bare.byte_length():]))
    return Error(message)
