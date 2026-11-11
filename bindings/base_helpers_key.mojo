# SPDX-License-Identifier: Apache-2.0
"""UNPACK of the base helper table key (lane calltable-base-helpers,
2026-11-11): the `(mode, vendor)` pair a bridge recorded at import under
`base_native_key` in the helper table it hands its door
(`python/mojolearn/_buffer.py::base_native_key`), or the refusal recorded
then. The door passes the key to `core/base_helpers_calls.mojo`; nothing
after this read touches Python."""
from std.python import PythonObject
from bindings.py_balanced import py_get, py_type
from core.base_entry_table import BaseEntryKey


def base_helpers_key(helpers: PythonObject) raises -> BaseEntryKey:
    """The base helper table key recorded in `helpers` (a bridge's helper
    dict), refused with the recorded "<Class>: <message>" when the base
    binding had none."""
    var key = py_get(helpers, PythonObject("base_native_key"))
    if key is None:
        raise Error("ImportError: mojolearn: the base binding has no native helper table key")
    if py_type(key) is py_type(PythonObject("")):
        raise Error(String(py=key))
    return BaseEntryKey(Int(py=key[0]), Int(py=key[1]))
