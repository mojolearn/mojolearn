# SPDX-License-Identifier: Apache-2.0
"""The thin Python owner of one resident byte LM session handle (lane
byte-lm-sessions-native, 2026-11-11).

The trainer door (`bindings/byte_lm_trainer_native.mojo`) creates it when a
resident call first needs a session and stores it as the trainer's
`_native_session`; dropping it (close, load_state_dict, a binding change or
the trainer's own destruction) is the native finalizer: it releases the
handle through the image's session entry
(`core/byte_lm_entry_table_sessions.mojo`), which tears the session down
once no call holds its lease. It holds no Python object and no session
pointer: only the table key and the generation-checked handle words.
Registered once, by `_mojolearn_runtime_byte_trainer_host`.
"""
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder, lookup_py_type_object
from bindings.py_type_registry import py_type_absent, py_type_attach
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
    # The step request words (handle, operation, plan), refilled in place by
    # every step (`byte_lm_resident_run`): allocated once per session, not
    # once per training step.
    var words: List[Int]

    def __init__(out self, key: ByteLmEntryKey) raises:
        var entry = byte_lm_sessions_ref(key)
        var handle = byte_lm_resident_open(entry)
        self.entry = entry
        self.words = List[Int]()
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


def byte_lm_session_owner_new(key: ByteLmEntryKey) raises -> PythonObject:
    return PythonObject(alloc=ByteLmSessionOwner(key))


def register_bytelmsessionowner(mut module: PythonModuleBuilder) raises:
    """Lookup-guarded like the classical host's registrars
    (bindings/py_type_registry.mojo): a second initialization in one process
    attaches the existing type object."""
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[ByteLmSessionOwner]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[ByteLmSessionOwner]("_ByteLmSessionOwner")
        return
    py_type_attach(module.module, "_ByteLmSessionOwner", existing)
