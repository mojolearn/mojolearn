# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the compiled manifold orchestration owner."""
from types import FunctionType

_ENTRY = None

_LIFECYCLE_METHODS = {
    ("mojolearn._expansion_decomp", "Isomap"): ("fit", "fit_transform", "transform"),
    ("mojolearn._expansion_decomp", "ClassicalMDS"): ("fit", "fit_transform"),
    ("mojolearn._expansion_decomp", "MDS"): ("fit", "fit_transform"),
    ("mojolearn._expansion_decomp", "LocallyLinearEmbedding"): ("fit", "fit_transform", "transform"),
    ("mojolearn._expansion_ann", "TSNE"): ("fit", "fit_transform"),
}


def native_lifecycle(method):
    """Declare an original native method before _mode sees its class."""
    if type(method) is not FunctionType:
        raise TypeError("manifold lifecycle requires the package's original public method")
    owner, separator, name = method.__qualname__.rpartition(".")
    names = _LIFECYCLE_METHODS.get((method.__module__, owner), ())
    if not separator or name not in names:
        raise TypeError("manifold lifecycle requires the package's original public method")
    # These flags already described the finished native methods. Publish them
    # before class creation, avoiding output/fit/CPU wrappers and later unwraps.
    # Constructors keep their independent numeric_mode injection unchanged.
    method._mojolearn_native_public_operation = True
    method._mojolearn_public_out = True
    method._mojolearn_validates = True
    if name in ("fit", "fit_transform"):
        method._mojolearn_tracks_fit = True
    else:
        method._mojolearn_needs_fit = True
    return method


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
