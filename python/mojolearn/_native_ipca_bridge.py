# SPDX-License-Identifier: Apache-2.0
"""Import-time connection for the native IncrementalPCA owner."""


def connect(ipca_class):
    # glue: connect the compiled decomp drivers once, before any operation
    from types import BuiltinFunctionType
    from . import _backend
    from .linear_model import LinearRegression, Ridge
    from ._native_classical_bridge import connect as connect_classical, _numpy_free_context
    from .tokenizer import _binding

    _, shared = connect_classical(Ridge, LinearRegression)
    context = dict(shared)
    names = ("x_decomp_ipca_fit", "x_decomp_ipca_partial", "x_decomp_ipca_project")
    decomp = {}
    selected = _backend.default_mode()
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: fixed numeric-mode ABI connection
        try:
            binding = _backend.binding("_mojolearn_x_decomp", mode)
            entries = {}
            for name in names:  # glue: retain checked compiled decomp symbols
                try:
                    entry = getattr(binding, name)
                except (AttributeError, ImportError):
                    continue
                if isinstance(entry, BuiltinFunctionType):
                    entries[name] = entry
            decomp[mode] = entries
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            decomp[mode] = type(error).__name__ + ": " + str(error)
    # Fitted arrays and projections are mojolearn Arrays, as `_M.out` made them.
    context["output_context"] = _numpy_free_context()
    context.update(ipca_class=ipca_class, ipca_decomp=decomp)
    entry = _binding()._module.ipca_native_call
    if not isinstance(entry, BuiltinFunctionType):
        raise TypeError("IncrementalPCA owner requires a compiled ipca_native_call")
    return entry, context


def connect_public(ipca_class):
    """Connect the compiled owner of the public IncrementalPCA class once, at
    import; returns its entry and context. An install without the compiled
    owner keeps the package importable and IncrementalPCA explains what is
    missing."""
    try:
        return connect(ipca_class)
    except (ImportError, AttributeError) as error:
        message = f"mojolearn: IncrementalPCA is unavailable: {error}"

        def unavailable(*args):
            raise ModuleNotFoundError(message, name=getattr(error, "name", None))
        return unavailable, {"ipca_unavailable": message}
