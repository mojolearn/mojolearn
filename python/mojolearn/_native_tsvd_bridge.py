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
    connected = {}
    decomp = {}
    selected = _backend.default_mode()
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: fixed native provider images
        for basename, name, table in (("_mojolearn_estimators", "tsvd_native_api_v1", connected),
                                      ("_mojolearn_x_decomp", "tsvd_decomp_native_api_v1", decomp)):  # glue: fixed descriptor factories
            try:
                factory = getattr(_backend.binding(basename, mode), name)
                if not isinstance(factory, BuiltinFunctionType):
                    raise TypeError(f"TruncatedSVD owner requires compiled {name}")
                table[mode] = factory
            except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
                table[mode] = type(error).__name__ + ": " + str(error)
    # Fitted arrays are mojolearn Arrays, as `_buffer.empty` made them.
    context["output_context"] = _numpy_free_context()
    # glue (lane a8d): the Gram fit of a cross-validation fold resident on
    # the device, exported by the identical GPU base binding next to the
    # fold store it reads (bindings/msel_device.mojo); a build without it
    # (the CPU column) leaves None and the door gathers host rows.
    from ._buffer import _native
    try:
        fold = _native('tsvd_fold_native_api_v1')
    except (ImportError, AttributeError, TypeError, RuntimeError):
        fold = None
    context["tsvd_fold_factory"] = fold if isinstance(fold, BuiltinFunctionType) else None
    context.update(tsvd_class=tsvd_class, tsvd_bindings=connected, tsvd_decomp=decomp,
                   tsvd_host_factory="ImportError: Mojolearn host TruncatedSVD provider is unavailable")
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
