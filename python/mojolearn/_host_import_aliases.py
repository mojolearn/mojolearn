# SPDX-License-Identifier: Apache-2.0
"""Connect fixed native module aliases once, before public APIs retain entries."""
import os
from types import BuiltinFunctionType, ModuleType

from . import host_surface


def connect(basename, module):
    # Import setup only. Successful aliases are the native objects themselves,
    # so no Python function chooses or forwards a public product operation.
    from . import _backend

    if type(module) is not ModuleType:
        raise ImportError("native import aliases require an exact extension module")
    fields = vars(module)
    pending, missing = {}, []
    for spec in host_surface.HOST_IMPORT_ALIASES.get(basename, ()):  # glue: fixed import-time native ABI metadata
        marker = fields.get(spec["marker"])
        if type(marker) is not BuiltinFunctionType or marker() != spec["abi"]:
            # An older tokenizer registers TreesResident itself. Refuse before
            # loading its new owner, which would register the same type twice.
            raise ImportError("the tokenizer native ownership ABI is outdated; rebuild the complete host layout")
        owner_name = spec["binding"]
        if not os.path.isfile(_backend.host_module_path(owner_name)):
            # The legacy module's native refusal entries keep metadata import
            # available. Connected entries are fixed for this process; adding
            # a previously missing binary requires a fresh process.
            missing.append(owner_name)
            continue
        owner = _backend.load_host_module(owner_name)  # cpu-route: import-time connection of the canonical native alias owner
        if type(owner) is not ModuleType:
            raise ImportError(f"native alias owner {owner_name} is not an extension module")
        source = vars(owner)
        for name in spec["functions"]:  # glue: retain verified compiled entry objects once, without wrappers
            entry = source.get(name)
            if type(entry) is not BuiltinFunctionType:
                raise ImportError(f"native alias owner {owner_name} requires compiled {name}")
            pending[name] = entry
        for name in spec["types"]:  # glue: alias the existing registered type object; never register it again
            owner_type = source.get(name)
            if type(owner_type) is not type:
                raise ImportError(f"native alias owner {owner_name} has no registered type {name}")
            pending[name] = owner_type
    # Publish only after every present owner has passed the complete inventory.
    # Exact module dictionaries avoid module-level __getattr__ callbacks.
    fields.update(pending)
    fields["_mojolearn_missing_import_owners"] = tuple(missing)
