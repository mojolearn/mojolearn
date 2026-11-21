# SPDX-License-Identifier: Apache-2.0
"""UNPACK of the model-selection helper table key (lane calltable-msel,
2026-11-11): the `(mode, vendor)` pair the model-selection bridge recorded
at import under `msel_native_key` in `family_msel_entries`
(`python/mojolearn/_native_msel_inputs.py`), or the refusal recorded then.

The key is resolved ONCE per connected context into the table's `EntryRef`
and held in a process slot (device wave round 2, 2026-11-21). Every later
`msel_helpers_key(context)` is one pointer compare of the context and a
two-word copy: no dictionary read, no tuple read, no table slot lookup and
no schema String. Before, each of the about twenty helper call sites of the
doors (one per fold, per splitter step, per scatter) read
`context["family_msel_entries"]["msel_native_key"]`, converted its two
words and looked the table slot up again with a freshly built schema
String.

What is held: the context object the entry was resolved for (the family
owners' shared context, which lives for the process) and the resolved ref.
The key is written once, at import, by the bridge; another context object
(a test's substituted connection) misses and resolves for itself. A
recorded refusal, or a missing key, is never held: it raises at every call,
as before, in the same words."""
from std.python import PythonObject
from bindings.py_balanced import py_get, py_type
from core.msel_entry_table import MselEntryKey
from core.msel_helpers_calls import msel_helpers_ref
from core.entry_table import EntryRef, entry_ref_unresolved
from core.shared_slot import SharedSlot


struct MselHelpersHeld(Defaultable, Movable):
    """The helper entry resolved for one context. `context` is written last:
    a reader that sees its context sees the entry resolved for it."""
    var context: PythonObject
    var entry: EntryRef

    def __init__(out self):
        self.context = PythonObject(None)
        self.entry = entry_ref_unresolved()


comptime MSEL_HELPERS_HELD = SharedSlot[MselHelpersHeld, "MojolearnMselHelpersHeldV1"]


def _msel_helpers_resolve(context: PythonObject) raises -> EntryRef:
    """The miss: read the recorded key and resolve the table entry; refused
    with the recorded "<Class>: <message>" when the base binding had none."""
    var key = py_get(context["family_msel_entries"], PythonObject("msel_native_key"))
    if key is None:
        raise Error("ImportError: mojolearn: the base binding has no model-selection helper table key")
    if py_type(key) is py_type(PythonObject("")):
        raise Error(String(py=key))
    return msel_helpers_ref(MselEntryKey(Int(py=key[0]), Int(py=key[1])))


def msel_helpers_key(context: PythonObject) raises -> EntryRef:
    """The helper table entry of the key recorded in the operation context.
    A hit (every call after the first for this context) is one pointer
    compare; the door passes the ref to every
    `core/msel_helpers_calls.mojo` call."""
    var held = MSEL_HELPERS_HELD.get_or_create_ptr()
    if held[].context is context:
        return held[].entry
    var entry = _msel_helpers_resolve(context)
    held[].entry = entry
    held[].context = context
    return entry
