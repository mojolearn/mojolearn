# SPDX-License-Identifier: Apache-2.0
"""PACK: the Python holder of one byte LM trainer's native state handle.

The trainer door (`bindings/byte_lm_trainer_native_door.mojo`) allocates it at
construction and at each checkpoint restore and stores it as the trainer's
`_native`; dropping it (the trainer's destruction) is the native finalizer,
which releases the state in the registry and, with it, the resident session.
It holds no Python object and no pointer: only the generation-checked handle
words (`ByteLmTrainerOwner`, bindings/byte_lm_trainer_native_run.mojo).
Registered once, by `_mojolearn_runtime_byte_trainer_host`.
"""
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder, lookup_py_type_object
from bindings.py_type_registry import py_type_absent, py_type_attach
from bindings.byte_lm_trainer_native_run import ByteLmTrainerOwner, ByteLmTrainerState


def byte_lm_trainer_owner_new(var state: ByteLmTrainerState) raises -> PythonObject:
    return PythonObject(alloc=ByteLmTrainerOwner(state^))


def register_bytelmtrainerowner(mut module: PythonModuleBuilder) raises:
    """Lookup-guarded like the classical host's registrars
    (bindings/py_type_registry.mojo): a second initialization in one process
    attaches the existing type object."""
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[ByteLmTrainerOwner]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[ByteLmTrainerOwner]("_ByteLmTrainerOwner")
        return
    py_type_attach(module.module, "_ByteLmTrainerOwner", existing)
