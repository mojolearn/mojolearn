# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the compiled model-selection owner."""

_ENTRY = None


def connect():
    """Return the native entry and the shared family admission context."""
    global _ENTRY
    from . import _backend, _native_family_bridge
    from types import BuiltinFunctionType

    if _ENTRY is None:
        # This import-time connection replaces tokenizer._binding() for the
        # same native orchestration. Estimator/device dispatch stays in the
        # unchanged Mojo model-selection owner and its admitted context.
        module = _backend.load_host_module("_mojolearn_runtime_model_selection_host")  # cpu-route: import-time connection of existing native model-selection orchestration
        entry = module.model_selection_native_call
        if not isinstance(entry, BuiltinFunctionType):
            raise TypeError("model selection requires the compiled model_selection_native_call entry")
        _ENTRY = entry
    return _ENTRY, _native_family_bridge.shared_context()
