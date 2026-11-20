# SPDX-License-Identifier: Apache-2.0
"""The thin owner of one Samba stack program handle (lane samba-sessions-native,
2026-11-12).

The Python object a `SambaStack` keeps for its per-call programs
(`_samba_program`, `_samba_state_program`) and a `SambaDecodeSession` keeps
for its program (`_handle`) is this one native value: the stack's (tier,
vendor) key, the program's batch and capacity, the handle word and the
program entry of `core/samba_entry_table_programs.mojo`, resolved once at
open. It releases the handle on `close()` or when Python drops it, as the
former Python-owned `_SambaStackProgram` object freed its device state on
collection. It holds no Python object and no device state of its own.
"""
from core.entry_table import EntryRef
from core.samba_entry_table import SambaEntryKey
from core.samba_entry_table_programs import samba_programs_call, SAMBA_PROGRAM_RELEASE


struct SambaProgramHandle(Movable, Writable):
    var mode: Int
    var vendor: Int
    var handle: Int
    """The program's handle word; 0 once released."""
    var b: Int
    """The program's batch rows (fixed at open)."""
    var smax: Int
    """The program's attention capacity (fixed at open)."""
    var entry: EntryRef
    """The program entry, resolved once at open: every call goes through it."""

    def __init__(out self, key: SambaEntryKey, entry: EntryRef, handle: Int, b: Int, smax: Int):
        self.entry = entry
        self.mode = key.mode
        self.vendor = key.vendor
        self.handle = handle
        self.b = b
        self.smax = smax

    def key(self) -> SambaEntryKey:
        return SambaEntryKey(self.mode, self.vendor)

    def call(self, operation: Int, addresses: List[Int], words: List[Int]) raises -> Int:
        """One program operation on this handle (`words` after the handle
        word). The caller releases the GIL around it, as the former Python
        entries did, and keeps every addressed buffer alive."""
        if self.handle == 0:
            raise Error("Samba program: the program was released")
        var full = List[Int](capacity=len(words) + 1)
        full.append(self.handle)
        for i in range(len(words)):  # small-loop(scalar words of one program call): argument words, not data
            full.append(words[i])
        return samba_programs_call(self.entry, operation, addresses, full, List[Float64]())

    def close(mut self) raises:
        """Release the handle; the program is freed with its last lease."""
        if self.handle == 0:
            return
        var word = self.handle
        self.handle = 0
        _ = samba_programs_call(self.entry, SAMBA_PROGRAM_RELEASE, List[Int](), [word], List[Float64]())

    def write_to(self, mut writer: Some[Writer]):
        writer.write("SambaProgramHandle(native)")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def __deinit__(deinit self):
        if self.handle == 0:
            return
        try:
            _ = samba_programs_call(self.entry, SAMBA_PROGRAM_RELEASE, List[Int](), [self.handle], List[Float64]())
        except:
            # Collection cannot report; a released or busy program is freed
            # with its registry entry's last lease.
            pass

