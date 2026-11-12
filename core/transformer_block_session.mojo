# SPDX-License-Identifier: Apache-2.0
"""The Transformer block session as a native registry handle (lane
transformer-session-native, 2026-11-12).

A Transformer block's retained session (its device context, packed weights,
attention step arm and backward stages: `TransformerSession` in
`transformer/transformer_binding_impl.mojo`) lives in the GPU Transformer
image's own state registry (`core/state_registry.mojo`, generation-checked
handles, Arc leases; `transformer/native_block_dispatch.mojo`). A caller holds
one handle word and reaches every session operation through the block entry
of `core/neural_entry_table.mojo` (NEURAL_TABLE_BLOCK_FORWARD / _TRAIN, which
the GPU image publishes with the same pointer), keyed by the block's (tier,
vendor). The handle word is the first integer word of every session
operation; it is never a memory address.

`TransformerBlockSession` is the one native value a block keeps for its
session (`_native_session`): the resolved entry and the handle word. It
creates the session at construction (NEURAL_BLOCK_SESSION_CREATE), closes it on
`close()` or when Python drops it (NEURAL_BLOCK_SESSION_CLOSE), as the former
Python-allocated `_TransformerSession` released its device state on
collection. It holds no Python object and no device state of its own. This
replaces the Python entries `transformer_session_create`,
`transformer_session_close`, `transformer_native_session_handle_v1`,
`transformer_session_forward`, `transformer_session_forward_int15`,
`transformer_native_api_v1` and the Python session type.

A handle word is `generation << 32 | index` (never 0: generations start at 1);
0 means "no handle". Creation and release are small metadata operations with
no independent numerical work to split.
"""
from core.entry_table import EntryRef, entry_ref_call, entry_ref_same
from core.neural_entry_table import NEURAL_BLOCK_SESSION_CREATE, NEURAL_BLOCK_SESSION_CLOSE
from core.state_registry import StateHandle


def transformer_block_session_word(handle: StateHandle) raises -> Int:
    """One plain word for a registry handle (index low, generation high)."""
    if handle.index < 0 or handle.index > 0x7FFFFFFF or handle.generation == 0 or handle.generation > UInt64(0x7FFFFFFF):
        raise Error("native Transformer block sessions: handle out of range")
    return (Int(handle.generation) << 32) | handle.index


def transformer_block_session_of(word: Int) raises -> StateHandle:
    if word <= 0:
        raise Error("transformer block native ABI: null session")
    return StateHandle(word & 0xFFFFFFFF, UInt64(word >> 32))


struct TransformerBlockSession(Movable, Writable):
    var entry: EntryRef
    """The block entry of the session's image, resolved once by the caller."""
    var handle: Int
    """The session's handle word; 0 once closed."""

    def __init__(out self, entry: EntryRef) raises:
        """Create one session in the image `entry` names (no device work)."""
        self.entry = entry
        self.handle = 0
        var none = List[Int]()
        var created = entry_ref_call(entry, NEURAL_BLOCK_SESSION_CREATE, none, List[Int](), List[Float64]())
        if created <= 0:
            raise Error("native provider: state admission returned a null handle")
        self.handle = created

    def serves(self, entry: EntryRef) -> Bool:
        """Whether this open session belongs to the image `entry` was
        resolved for: the same published pointer and layout word (one GPU
        Transformer image per tier and vendor publishes the block entry, in
        the forward and the training family alike).
        Replaces the former `_session_binding is <binding module>` check."""
        return self.handle != 0 and entry_ref_same(self.entry, entry)

    def word(self) raises -> Int:
        """The handle word a session operation leads with."""
        if self.handle == 0:
            raise Error("transformer: session is busy or closed")
        return self.handle

    def close(mut self) raises:
        """Close the session and release its handle. A busy session refuses
        and keeps its handle, as the former Python close did."""
        if self.handle == 0:
            return
        var words: List[Int] = [self.handle]
        _ = entry_ref_call(self.entry, NEURAL_BLOCK_SESSION_CLOSE, List[Int](), words, List[Float64]())
        self.handle = 0

    def write_to(self, mut writer: Some[Writer]):
        writer.write("TransformerBlockSession(native)")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def __deinit__(deinit self):
        if self.handle == 0:
            return
        try:
            var words: List[Int] = [self.handle]
            _ = entry_ref_call(self.entry, NEURAL_BLOCK_SESSION_CLOSE, List[Int](), words, List[Float64]())
        except:
            # Collection cannot report; a busy session is freed with its
            # registry entry's last lease at process end.
            pass
