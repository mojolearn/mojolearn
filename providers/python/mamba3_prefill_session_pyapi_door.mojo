# SPDX-License-Identifier: Apache-2.0
# Python type of the Mamba-3 prefill session handle owner and the boundary
# helpers every caller uses (lane mamba3-prefill-handle, 2026-11-12; the shape
# of providers/python/transformer_block_session_pyapi.mojo). Registration is
# module setup with the GIL held; open and word are one allocation and one
# native downcast at a public operation's UNPACK. Every session operation is
# one call through the training image's Mamba entry (NEURAL_TABLE_MAMBA_TRAIN).
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder, lookup_py_type_object
from bindings.py_type_registry import py_type_absent, py_type_attach
from bindings.py_balanced import py_get, py_setitem
from core.entry_table import EntryRef, entry_ref_call
from core.neural_entry_table import (
    NEURAL_MAMBA_RESERVE, NEURAL_MAMBA_RELEASE, NEURAL_MAMBA3_SESSION_INFO, NEURAL_MAMBA3_SESSION_CAPS,
)
from core.mamba3_prefill_session import Mamba3PrefillHandle


def mamba3_prefill_session_type(mut module: PythonModuleBuilder) raises:
    """Register `_Mamba3PrefillHandle` on `module` (the image that owns the
    neural doors' Python types, `_mojolearn_tokenizer_host`). A second load in
    the same interpreter attaches the class the first one built."""
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[Mamba3PrefillHandle]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[Mamba3PrefillHandle]("_Mamba3PrefillHandle")
        return
    py_type_attach(module.module, "_Mamba3PrefillHandle", existing)


def mamba3_prefill_caps(entry: EntryRef) raises -> Int:
    """The training image's fixed capability bits (MAMBA3_CAP_*): one table
    call, no Python. Replaces the former hasattr probes and the
    `mamba3_prefill_session_reports_optional` / `mamba3_owned_weights_enabled`
    Python entries."""
    return entry_ref_call(entry, NEURAL_MAMBA3_SESSION_CAPS, List[Int](), List[Int](), List[Float64]())


def mamba3_prefill_session_open(entry: EntryRef) raises -> PythonObject:
    """A new prefill session in the image `entry` names, held by its native
    owner (one allocation; no device work)."""
    return PythonObject(alloc=Mamba3PrefillHandle(entry))


def mamba3_prefill_session_serves(owner: PythonObject, entry: EntryRef) raises -> Bool:
    """Whether a block's session owner (or None) is open in the image `entry`
    names (the former `_prefill_binding is entries.canonical` check)."""
    if owner is None:
        return False
    return owner.downcast_value_ptr[Mamba3PrefillHandle]()[].serves(entry)


def mamba3_prefill_session_handle(owner: PythonObject) raises -> Int:
    """The handle word of a block's session owner: one native downcast (the
    former `mamba_prefill_native_handle_v1` Python call). The caller keeps
    `owner` alive through every operation that uses the word."""
    return owner.downcast_value_ptr[Mamba3PrefillHandle]()[].word()


def mamba3_prefill_session_ensure(fields: PythonObject, entry: EntryRef) raises -> PythonObject:
    """The block's open session in the image `entry` names: the one in
    `fields["_prefill_session"]` when it serves `entry`, else a new one stored
    there (the former create-if-binding-differs dance)."""
    var session = py_get(fields, "_prefill_session")
    if not mamba3_prefill_session_serves(session, entry):
        session = mamba3_prefill_session_open(entry)
        py_setitem(fields, "_prefill_session", session)
    return session


def mamba3_prefill_session_caps(owner: PythonObject) raises -> Int:
    """The capability bits the session's image answered when it opened (one
    native downcast, no table call)."""
    return owner.downcast_value_ptr[Mamba3PrefillHandle]()[].caps


def mamba3_prefill_session_close(owner: PythonObject) raises:
    """Close a block's session owner (None is a no-op). A busy session
    refuses and keeps its handle."""
    if owner is None:
        return
    owner.downcast_value_ptr[Mamba3PrefillHandle]()[].close()


def mamba3_prefill_session_run(owner: PythonObject, operation: Int, addresses: List[Int],
    scalars: List[Int]) raises -> Int:
    """One session operation through the owner's Mamba entry: reserve with
    the GIL held (the image's busy/usable admission), run `operation` with the
    GIL released, release. The handle word leads the operation's words. The
    caller keeps `owner` and every addressed buffer alive through the call."""
    var held = owner.downcast_value_ptr[Mamba3PrefillHandle]()
    var entry = held[].entry
    var handle = held[].word()
    var slot: List[Int] = [handle]
    var words = List[Int](capacity=1 + len(scalars))
    words.append(handle)
    for i in range(len(scalars)):  # driver-loop: over argument words (fixed per operation, not data); small-loop(session operation scalar words)
        words.append(scalars[i])
    _ = entry_ref_call(entry, NEURAL_MAMBA_RESERVE, List[Int](), slot, List[Float64]())
    var result = 0
    try:
        with GILReleased(Python()):
            result = entry_ref_call(entry, operation, addresses, words, List[Float64]())
    except problem:
        _ = entry_ref_call(entry, NEURAL_MAMBA_RELEASE, List[Int](), slot, List[Float64]())
        raise problem
    _ = entry_ref_call(entry, NEURAL_MAMBA_RELEASE, List[Int](), slot, List[Float64]())
    _ = words^
    return result


def mamba3_prefill_session_run_held(owner: PythonObject, operation: Int, addresses: List[Int],
    scalars: List[Int]) raises -> Int:
    """Lane fix-mamba-forward-doors (core/neural_entry_table.mojo
    IDN_MAMBA3_FORWARD_DOOR_OLD): a session forward (op 6 / 8) as ONE call
    through the owner's Mamba entry with the GIL released: the trailing word
    1 asks the image to reserve, run and release in that call
    (MAMBA3_CAP_HELD_FORWARD; the caller checked the owner's caps). The
    caller keeps `owner` and every addressed buffer alive through the call."""
    var held = owner.downcast_value_ptr[Mamba3PrefillHandle]()
    var entry = held[].entry
    var words = List[Int](capacity=2 + len(scalars))
    words.append(held[].word())
    for i in range(len(scalars)):  # driver-loop: over argument words (fixed per operation, not data); small-loop(session operation scalar words)
        words.append(scalars[i])
    words.append(1)
    var result = 0
    with GILReleased(Python()):
        result = entry_ref_call(entry, operation, addresses, words, List[Float64]())
    _ = words^
    return result


def mamba3_prefill_session_info(owner: PythonObject) raises -> List[Int]:
    """The session's six counters [weight_uploads, weight_recopies,
    weight_reuses, backward_reuses, backward_recomputes, stages_held]: one
    table call writing a native span (the former Python entry
    `mamba3_prefill_session_info`)."""
    var held = owner.downcast_value_ptr[Mamba3PrefillHandle]()
    var counters = List[Int](length=6, fill=0)
    var addresses: List[Int] = [Int(counters.unsafe_ptr())]
    var words: List[Int] = [held[].word()]
    _ = entry_ref_call(held[].entry, NEURAL_MAMBA3_SESSION_INFO, addresses, words, List[Float64]())
    return counters^
