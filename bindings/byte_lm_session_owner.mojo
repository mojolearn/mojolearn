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
from core.byte_lm_entry_table import ByteLmEntryKey
# RUN (lane rs-neural): the typed owner, no Python import.
from training.byte_lm_session_owner_run import ByteLmSessionOwner


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
