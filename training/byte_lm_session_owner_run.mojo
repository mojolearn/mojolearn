# SPDX-License-Identifier: Apache-2.0
"""RUN half of the resident byte LM session owner (lane rs-neural, wave 8
run split, from bindings/byte_lm_session_owner.mojo): the typed handle
owner whose finalizer releases the session through the image's session
entry. Typed Mojo only, no Python import (tools/python_free_dirs.sh). It is a
field of the trainer's native state (`ByteLmTrainerState`,
bindings/byte_lm_trainer_native_run.mojo), which also holds the step request
words; it is no longer a Python object (lane byte-state, 2026-11-21)."""
from core.state_registry import StateHandle
from core.byte_lm_entry_table import ByteLmEntryKey
from core.entry_table import EntryRef
from core.byte_lm_entry_table_sessions import byte_lm_sessions_ref, byte_lm_resident_open, byte_lm_resident_close


struct ByteLmSessionOwner(Movable, Writable):
    var mode: Int
    var vendor: Int
    var index: Int
    var generation: UInt64
    # The sessions entry, resolved once at construction (lane
    # entry-table-zero-alloc): every step and the close call through it.
    var entry: EntryRef

    def __init__(out self, key: ByteLmEntryKey) raises:
        var entry = byte_lm_sessions_ref(key)
        var handle = byte_lm_resident_open(entry)
        self.entry = entry
        self.mode = key.mode
        self.vendor = key.vendor
        self.index = handle.index
        self.generation = handle.generation

    def key(self) -> ByteLmEntryKey:
        return ByteLmEntryKey(self.mode, self.vendor)

    def handle(self) -> StateHandle:
        return StateHandle(self.index, self.generation)

    def __deinit__(deinit self):
        try:
            byte_lm_resident_close(self.entry, StateHandle(self.index, self.generation))
        except:
            pass  # Quiet finalization: an unpublished table or a stale handle has nothing to release.

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ByteLmSessionOwner(handle=", self.index, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)
