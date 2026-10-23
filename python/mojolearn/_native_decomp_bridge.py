# SPDX-License-Identifier: Apache-2.0
"""Import-time connection for the native owner of the decomposition
expansion's estimators (bindings/decomp_native.mojo)."""

#: The compiled decomp drivers the owners call (a build without one refuses
#: that operation by name).
ENTRY_NAMES = ("x_decomp_ica_fit", "x_decomp_ica_project", "x_decomp_nmf_fit", "x_decomp_nmf_inverse",
               "x_decomp_fa_fit", "x_decomp_fa_query", "x_decomp_lda_owner", "x_decomp_kpca_eig",
               "x_decomp_rp_fit", "x_decomp_rp_project")

#: The x_neighbors entries KernelPCA's kernel matrix steps call (connected
#: only for a caller that asks, `neighbors=True`).
NEIGHBOR_NAMES = ("x_neighbors_vendor", "x_neighbors_numeric_mode", "x_neighbors_kpca_resident", "xn_kernel",
                  "xn_colsum", "xn_rowsum", "xn_scale_div", "xn_kpca_center", "xn_kpca_alpha_scale",
                  "xn_kpca_transform", "xn_matmul")


def connect(classes, neighbors=False, metadata=None):
    # glue: connect the compiled decomp drivers once, before any operation;
    # `classes` maps each owned class's name to the class, `metadata` is
    # plain constant data the caller's module defines (merged as is)
    from types import BuiltinFunctionType
    from . import _backend
    from .linear_model import LinearRegression, Ridge
    from ._native_classical_bridge import connect as connect_classical, _numpy_free_context
    from .tokenizer import _binding

    _, shared = connect_classical(Ridge, LinearRegression)
    context = dict(shared)
    decomp = {}
    near = {}
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
        if neighbors:
            try:
                binding = _backend.binding("_mojolearn_x_neighbors", mode)
                entries = {}
                for name in NEIGHBOR_NAMES:  # glue: retain checked compiled neighbors symbols
                    try:
                        entry = getattr(binding, name)
                    except (AttributeError, ImportError):
                        continue
                    if isinstance(entry, BuiltinFunctionType):
                        entries[name] = entry
                near[mode] = entries
            except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
                near[mode] = type(error).__name__ + ": " + str(error)
    # Fitted arrays and results are mojolearn Arrays, as `_M.out` made them.
    context["output_context"] = _numpy_free_context()
    context.update(decomp_classes=dict(classes), decomp_entries=decomp, decomp_neighbors=near)
    context.update(metadata or {})
    entry = _binding()._module.decomp_native_call
    if not isinstance(entry, BuiltinFunctionType):
        raise TypeError("the decomposition owner requires a compiled decomp_native_call")
    return entry, context


def connect_public(classes, neighbors=False, metadata=None):
    """Connect the compiled owner once, at import; returns its entry and
    context. An install without it keeps the package importable and the
    estimators explain what is missing."""
    try:
        return connect(classes, neighbors, metadata)
    except (ImportError, AttributeError) as error:
        message = f"mojolearn: the decomposition estimators are unavailable: {error}"

        def unavailable(*args):
            raise ModuleNotFoundError(message, name=getattr(error, "name", None))
        return unavailable, {"decomp_unavailable": message}
