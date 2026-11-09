# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the compiled Sequential orchestration owner."""


def connect(context):
    """Retain the direct native entry for public calls and the legacy ABI."""
    from types import BuiltinFunctionType
    from . import _backend

    module = _backend.load_host_module("_mojolearn_runtime_sequential_host")  # cpu-route: import-time connection of existing native Sequential orchestration
    entry = module.sequential_native_call
    if not isinstance(entry, BuiltinFunctionType):
        raise TypeError("Sequential requires the compiled sequential_native_call entry")
    context.update(sequential_owner_entry=entry, sequential_builtin_type=BuiltinFunctionType)
    return entry
