# SPDX-License-Identifier: Apache-2.0
"""UNPACK of the model-selection helper table key (lane calltable-msel,
2026-11-11): the `(mode, vendor)` pair the model-selection bridge recorded
at import under `msel_native_key` in `family_msel_entries`
(`python/mojolearn/_native_msel_inputs.py`), or the refusal recorded then.
The key is resolved here, once, into the table's `EntryRef` (lane
entry-table-zero-alloc); the door holds that ref and passes it to every
`core/msel_helpers_calls.mojo` call; nothing after this read touches Python
or the table."""
from std.python import PythonObject
from bindings.py_balanced import py_get, py_type
from core.msel_entry_table import MselEntryKey
from core.msel_helpers_calls import msel_helpers_ref
from core.entry_table import EntryRef


def msel_helpers_key(context: PythonObject) raises -> EntryRef:
    """The helper table entry of the key recorded in the operation context,
    resolved once; refused with
    the recorded "<Class>: <message>" when the base binding had none."""
    var key = py_get(context["family_msel_entries"], PythonObject("msel_native_key"))
    if key is None:
        raise Error("ImportError: mojolearn: the base binding has no model-selection helper table key")
    if py_type(key) is py_type(PythonObject("")):
        raise Error(String(py=key))
    return msel_helpers_ref(MselEntryKey(Int(py=key[0]), Int(py=key[1])))
