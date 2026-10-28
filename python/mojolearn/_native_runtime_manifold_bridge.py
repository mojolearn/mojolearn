# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the compiled manifold orchestration owner."""

_ENTRY = None


def connect():
    """Return the retained native entry and shared family context."""
    global _ENTRY
    from . import _backend, _native_family_bridge
    from types import BuiltinFunctionType

    if _ENTRY is None:
        _native_family_bridge.connect_bindings(
            "_mojolearn_x_decomp", "_mojolearn_x_ann", "_mojolearn_estimators")
        try:
            # Replaces the same orchestration inside tokenizer._binding().
            # Numerical bindings and their resident stores keep their owners.
            module = _backend.load_host_module("_mojolearn_runtime_manifold_host")  # cpu-route: import-time connection of existing native manifold orchestration
            entry = module.manifold_native_call
            if not isinstance(entry, BuiltinFunctionType):
                raise TypeError("manifold requires the compiled manifold_native_call entry")
            _ENTRY = entry
        except ImportError as error:
            # Missing-owner refusal only; never execute through another owner.
            _ENTRY = _native_family_bridge._unavailable(error)
    context = _native_family_bridge.shared_context()
    context["family_mani_door"] = _ENTRY
    return _ENTRY, context
