# SPDX-License-Identifier: Apache-2.0
"""Import-time connection for the native owner of the decomposition
expansion's estimators (bindings/decomp_native.mojo)."""

#: The compiled decomp drivers the owners call (a build without one refuses
#: that operation by name).
ENTRY_NAMES = ("x_decomp_ica_fit", "x_decomp_ica_project")


def connect(classes):
    # glue: connect the compiled decomp drivers once, before any operation;
    # `classes` maps each owned class's name to the class
    from types import BuiltinFunctionType
    from . import _backend
    from .linear_model import LinearRegression, Ridge
    from ._native_classical_bridge import connect as connect_classical, _numpy_free_context
    from .tokenizer import _binding

    _, shared = connect_classical(Ridge, LinearRegression)
    context = dict(shared)
    decomp = {}
    selected = _backend.default_mode()
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: fixed numeric-mode ABI connection
        try:
            binding = _backend.binding("_mojolearn_x_decomp", mode)
            entries = {}
            for name in ENTRY_NAMES:  # glue: retain checked compiled decomp symbols
                try:
                    entry = getattr(binding, name)
                except (AttributeError, ImportError):
                    continue
                if isinstance(entry, BuiltinFunctionType):
                    entries[name] = entry
            decomp[mode] = entries
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            decomp[mode] = type(error).__name__ + ": " + str(error)
    # Fitted arrays and results are mojolearn Arrays, as `_M.out` made them.
    context["output_context"] = _numpy_free_context()
    context.update(decomp_classes=dict(classes), decomp_entries=decomp)
    entry = _binding()._module.decomp_native_call
    if not isinstance(entry, BuiltinFunctionType):
        raise TypeError("the decomposition owner requires a compiled decomp_native_call")
    return entry, context


def connect_public(classes):
    """Connect the compiled owner once, at import; returns its entry and
    context. An install without it keeps the package importable and the
    estimators explain what is missing."""
    try:
        return connect(classes)
    except (ImportError, AttributeError) as error:
        message = f"mojolearn: the decomposition estimators are unavailable: {error}"

        def unavailable(*args):
            raise ModuleNotFoundError(message, name=getattr(error, "name", None))
        return unavailable, {"decomp_unavailable": message}
