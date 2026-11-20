# SPDX-License-Identifier: Apache-2.0
# Exact Unicode input-error packing; no runtime codec or name lookup remains.
from std.python import PythonObject
from bindings.py_balanced import py_call


def _archive_unicode_failure(raw: PythonObject, encoding: String, start: Int,
                             end: Int, reason: String, context: PythonObject,
                             encode: Bool = False) raises:
    var category = context["archive_unicode_encode_error" if encode else "archive_unicode_decode_error"]
    var error = py_call(category, encoding, raw, start, end, reason)
    # The incumbent Mojo codec call surfaced the Unicode error text as a
    # ValueError at the public door. Preserve that established error contract.
    raise Error(String(error))

