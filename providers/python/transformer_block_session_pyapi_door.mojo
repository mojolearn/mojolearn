# SPDX-License-Identifier: Apache-2.0
# Python type of the Transformer block session handle owner and the two
# boundary helpers every caller uses (lane transformer-session-native,
# 2026-11-12). Registration is module setup with the GIL held; open and word
# are one allocation and one native downcast at a public operation's UNPACK.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder, lookup_py_type_object
from bindings.py_type_registry import py_type_absent, py_type_attach
from core.entry_table import EntryRef, entry_ref_call
from core.neural_entry_table import NEURAL_BLOCK_RESERVE, NEURAL_BLOCK_RELEASE
from core.transformer_block_session import TransformerBlockSession


def transformer_block_session_type(mut module: PythonModuleBuilder) raises:
    """Register `_TransformerBlockSession` on `module` (the image that owns
    the neural doors' Python types, `_mojolearn_tokenizer_host`). A second
    load in the same interpreter attaches the class the first one built (the
    causal_session_handles_type pattern)."""
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[TransformerBlockSession]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[TransformerBlockSession]("_TransformerBlockSession")
        return
    py_type_attach(module.module, "_TransformerBlockSession", existing)


def transformer_block_session_open(entry: EntryRef) raises -> PythonObject:
    """A new block session in the image `entry` names, held by its native
    owner (one allocation; no device work)."""
    return PythonObject(alloc=TransformerBlockSession(entry))


def transformer_block_session_handle(owner: PythonObject) raises -> Int:
    """The handle word of a block's session owner: one native downcast (the
    former `transformer_native_session_handle_v1` Python call). The caller
    keeps `owner` alive through every operation that uses the word."""
    return owner.downcast_value_ptr[TransformerBlockSession]()[].word()


def transformer_block_session_serves(owner: PythonObject, entry: EntryRef) raises -> Bool:
    """Whether a block's session owner (or None) is open in the image `entry`
    names (the former `_session_binding is <binding module>` check)."""
    if owner is None:
        return False
    return owner.downcast_value_ptr[TransformerBlockSession]()[].serves(entry)


def transformer_block_session_close(owner: PythonObject) raises:
    """Close a block's session owner (None is a no-op). A busy session
    refuses and keeps its handle."""
    if owner is None:
        return
    owner.downcast_value_ptr[TransformerBlockSession]()[].close()


def transformer_block_session_entry(owner: PythonObject) raises -> EntryRef:
    """The block entry a session owner was opened in: one native downcast at
    a layer call's UNPACK, beside `transformer_block_session_handle`."""
    return owner.downcast_value_ptr[TransformerBlockSession]()[].entry


def transformer_block_session_run_words(entry: EntryRef, handle: Int, operation: Int, addresses: List[Int],
    scalars: List[Int]) raises -> Int:
    """One session operation through the session's block entry: reserve with
    the GIL held (the image's busy/closed admission), run `operation` with the
    GIL released, release. The handle word leads the operation's words.
    `entry` and `handle` are the session owner's (`transformer_block_session_entry`,
    `transformer_block_session_handle`), unpacked by the caller, which keeps
    the owner and every addressed buffer alive through the call.
    Lane dw-neural-models (2026-11-21): the owner form
    (`transformer_block_session_run(owner, ...)`) is gone; its one caller,
    the CausalLM layer run, passes a typed request."""
    var slot: List[Int] = [handle]
    var words = List[Int](capacity=1 + len(scalars))
    words.append(handle)
    for i in range(len(scalars)):  # driver-loop: over argument words (fixed per operation, not data); small-loop(session operation scalar words)
        words.append(scalars[i])
    _ = entry_ref_call(entry, NEURAL_BLOCK_RESERVE, List[Int](), slot, List[Float64]())
    var result = 0
    try:
        with GILReleased(Python()):
            result = entry_ref_call(entry, operation, addresses, words, List[Float64]())
    except problem:
        _ = entry_ref_call(entry, NEURAL_BLOCK_RELEASE, List[Int](), slot, List[Float64]())
        raise problem
    _ = entry_ref_call(entry, NEURAL_BLOCK_RELEASE, List[Int](), slot, List[Float64]())
    _ = words^
    return result
