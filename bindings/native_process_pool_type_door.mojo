# SPDX-License-Identifier: Apache-2.0
"""Registration of the process pool's native record type, `ProcessPoolOwner`
(bindings/native_process_pool_door.mojo).

The pool door is compiled into several bindings, and a pool is created by
whichever of them a caller reaches first (the process entry of the tokenizer
host, the fold workers of model selection, the sharded drivers of the
parallel host, the causal model). Mojo keeps one Python type object per Mojo
type per interpreter, so every such binding calls this registrar when it is
built: the first creates the type, the others attach the existing one
(bindings/py_type_registry.mojo). Module setup only; never in an execution
loop."""
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder, lookup_py_type_object
from bindings.py_type_registry import py_type_absent, py_type_attach
from bindings.native_process_pool_door import ProcessPoolOwner


def register_process_pool_owner(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[ProcessPoolOwner]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[ProcessPoolOwner]("_ProcessPoolOwner")
        return
    py_type_attach(module.module, "_ProcessPoolOwner", existing)
