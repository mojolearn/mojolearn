# SPDX-License-Identifier: Apache-2.0
"""The Mamba-3 prefill session as a native registry handle (lane
mamba3-prefill-handle, 2026-11-12; the shape of
core/transformer_block_session.mojo).

A Mamba-3 block's prefill session (its device context, owned weight snapshot
and retained forward stages: `Mamba3PrefillSession` in
`mamba/impl/mamba_shell_impl.mojo`; on the CPU training image the owned host
snapshot `M3OwnedHostSession` in `mamba/host/mamba_host_shell_impl.mojo`) lives
in its training image's own state registry (`core/state_registry.mojo`,
generation-checked handles, Arc leases; `mamba/native_block_dispatch.mojo`,
`mamba/host/native_block_dispatch.mojo`). A caller holds one handle word and
reaches every session operation through the image's Mamba entry of
`core/neural_entry_table.mojo` (NEURAL_TABLE_MAMBA_TRAIN, the same pointer the
image's capsule carries), keyed by the block's training (tier, vendor). The
handle word is the first integer word of every session operation; it is never
a memory address.

`Mamba3PrefillHandle` is the one native value a block keeps for its session
(`_prefill_session`): the resolved entry and the handle word. It creates the
session at construction (NEURAL_MAMBA3_SESSION_CREATE) and closes it on
`close()` or when Python drops it (NEURAL_MAMBA3_SESSION_CLOSE). It holds no
Python object and no device state of its own. This replaces the Python entries
`mamba3_prefill_session_create`, `_close`, `_forward`, `_forward_owned`,
`_info`, `_export_weights`, `_reports_optional`, `mamba3_state_owned`,
`mamba3_owned_weights_enabled`, `mamba_prefill_native_handle_v1`, the Python
session types `_Mamba3PrefillSession` / `_M3OwnedHostSession` and the block's
`_prefill_binding` identity field (`serves` compares the entry instead).

A handle word is `generation << 32 | index` (never 0: generations start at 1).
Creation and release are small metadata operations with no independent
numerical work to split.
"""
from core.entry_table import EntryRef, entry_ref_call, entry_ref_same
from core.neural_entry_table import NEURAL_MAMBA3_SESSION_CREATE, NEURAL_MAMBA3_SESSION_CLOSE
from core.state_registry import StateHandle


def mamba3_prefill_session_word(handle: StateHandle) raises -> Int:
    """One plain word for a registry handle (index low, generation high)."""
    if handle.index < 0 or handle.index > 0x7FFFFFFF or handle.generation == 0 or handle.generation > UInt64(0x7FFFFFFF):
        raise Error("native Mamba3 prefill sessions: handle out of range")
    return (Int(handle.generation) << 32) | handle.index


def mamba3_prefill_session_of(word: Int) raises -> StateHandle:
    if word <= 0:
        raise Error("Mamba block native ABI: null session")
    return StateHandle(word & 0xFFFFFFFF, UInt64(word >> 32))


struct Mamba3PrefillHandle(Movable, Writable):
    var entry: EntryRef
    """The Mamba entry of the session's training image, resolved once by the caller."""
    var handle: Int
    """The session's handle word; 0 once closed."""

    def __init__(out self, entry: EntryRef) raises:
        """Create one session in the image `entry` names (no device work)."""
        self.entry = entry
        self.handle = 0
        var created = entry_ref_call(entry, NEURAL_MAMBA3_SESSION_CREATE, List[Int](), List[Int](), List[Float64]())
        if created <= 0:
            raise Error("native provider: state admission returned a null handle")
        self.handle = created

    def serves(self, entry: EntryRef) -> Bool:
        """Whether this open session belongs to the image `entry` was resolved
        for (the same published pointer and layout word). Replaces the former
        `_prefill_binding is entries.canonical` check."""
        return self.handle != 0 and entry_ref_same(self.entry, entry)

    def word(self) raises -> Int:
        """The handle word a session operation leads with."""
        if self.handle == 0:
            raise Error("mamba3 prefill session: closed")
        return self.handle

    def close(mut self) raises:
        """Close the session and release its handle. A busy session refuses
        and keeps its handle, as the former Python close did."""
        if self.handle == 0:
            return
        var words: List[Int] = [self.handle]
        _ = entry_ref_call(self.entry, NEURAL_MAMBA3_SESSION_CLOSE, List[Int](), words, List[Float64]())
        self.handle = 0

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Mamba3PrefillHandle(native)")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def __deinit__(deinit self):
        if self.handle == 0:
            return
        try:
            var words: List[Int] = [self.handle]
            _ = entry_ref_call(self.entry, NEURAL_MAMBA3_SESSION_CLOSE, List[Int](), words, List[Float64]())
        except:
            # Collection cannot report; a busy session is freed with its
            # registry entry's last lease at process end.
            pass
