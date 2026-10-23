# SPDX-License-Identifier: Apache-2.0
"""Import-time connection for the native TruncatedSVD owner."""


def connect(tsvd_class):
    # glue: connect the compiled estimators and decomp entries once, before any operation
    from types import BuiltinFunctionType
    from . import _backend
    from .linear_model import LinearRegression, Ridge
    from ._native_classical_bridge import connect as connect_classical, _numpy_free_context
    from .tokenizer import _binding

    _, shared = connect_classical(Ridge, LinearRegression)
    context = dict(shared)
    required = ("estimators_vendor", "estimators_numeric_mode", "tsvd_fit", "tsvd_explained",
                "tsvd_transform", "inverse_transform")
    # A build without tsvd_fit_explained takes the tsvd_fit / tsvd_explained pair.
    optional = ("tsvd_fit_explained",)
    # The decomp kit's entries: the TSVD_QFIX flag and the two route drivers.
    # A build without a driver refuses that route by name (bindings/tsvd_native.mojo).
    decomp_names = ("x_decomp_qfix_flags", "x_decomp_tsvd_randomized", "x_decomp_tsvd_tsqr")
    connected = {}
    decomp = {}
    selected = _backend.default_mode()
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: fixed numeric-mode ABI connection
        try:
            binding = _backend.binding("_mojolearn_estimators", mode)
            entries = {}
            for name in required + optional:  # glue: retain checked compiled estimator symbols
                try:
                    entry = getattr(binding, name)
                except (AttributeError, ImportError):
                    if name in required:
                        raise
                    continue
                if not isinstance(entry, BuiltinFunctionType):
                    raise TypeError(f"TruncatedSVD owner requires compiled {name}")
                entries[name] = entry
            connected[mode] = entries
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            connected[mode] = type(error).__name__ + ": " + str(error)
        try:
            binding = _backend.binding("_mojolearn_x_decomp", mode)
            entries = {}
            for name in decomp_names:  # glue: retain checked compiled decomp symbols
                try:
                    entry = getattr(binding, name)
                except (AttributeError, ImportError):
                    continue
                if isinstance(entry, BuiltinFunctionType):
                    entries[name] = entry
            decomp[mode] = entries
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            decomp[mode] = type(error).__name__ + ": " + str(error)
    # Fitted arrays are mojolearn Arrays, as `_buffer.empty` made them.
    context["output_context"] = _numpy_free_context()
    context.update(tsvd_class=tsvd_class, tsvd_bindings=connected, tsvd_decomp=decomp)
    entry = _binding()._module.tsvd_native_call
    if not isinstance(entry, BuiltinFunctionType):
        raise TypeError("TruncatedSVD owner requires a compiled tsvd_native_call")
    return entry, context


def connect_public(tsvd_class):
    """Connect the compiled owner of the public TruncatedSVD class once, at
    import; returns its entry and context. An install without the compiled
    owner keeps the package importable and TruncatedSVD explains what is
    missing."""
    try:
        return connect(tsvd_class)
    except (ImportError, AttributeError) as error:
        message = f"mojolearn: TruncatedSVD is unavailable: {error}"

        def unavailable(*args):
            raise ModuleNotFoundError(message, name=getattr(error, "name", None))
        return unavailable, {"tsvd_unavailable": message}
