# SPDX-License-Identifier: Apache-2.0
"""UNPACK of the base helper table key (lane calltable-base-helpers,
2026-11-11): the `(mode, vendor)` pair a bridge recorded at import under
`base_native_key` in the helper table it hands its door
(`python/mojolearn/_buffer.py::base_native_key`), or the refusal recorded
then, resolved once into the table's `EntryRef` (lane
entry-table-zero-alloc): the door holds that ref and passes it to every
`core/base_helpers_calls.mojo` call of the operation; nothing after this read
touches Python or the table."""
from std.python import PythonObject
from bindings.py_balanced import py_get, py_type
from core.base_entry_table import BaseEntryKey
from core.base_helpers_calls import base_helpers_ref
from core.entry_table import EntryRef


def base_helpers_key(helpers: PythonObject) raises -> EntryRef:
    """The base helper entry of the table key recorded in `helpers` (a
    bridge's helper dict), resolved once; refused with the recorded "<Class>: <message>" when the base
    binding had none."""
    var key = py_get(helpers, PythonObject("base_native_key"))
    if key is None:
        raise Error("ImportError: mojolearn: the base binding has no native helper table key")
    if py_type(key) is py_type(PythonObject("")):
        raise Error(String(py=key))
    return base_helpers_ref(BaseEntryKey(Int(py=key[0]), Int(py=key[1])))
